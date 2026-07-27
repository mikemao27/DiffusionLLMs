"""
Quantization Test for Fast-dLLM v2.

Measures INT8 weight quantization's speedup in isolation, and stacked with block eviction, as a confirmatory test of the profiling-derived diagnosis 
that decode is bottlenecked on reading model weight bytes (GEMM-dominated), not on attention/KV cache, rather than as a test of the eviction kernel itself. 
See kernel/test_eviction.py and kernel/profile_eviction.py for the eviction-only characterization this builds on.

Three configs are run back-to-back in one process so the comparison is self-contained and uses the same warm-up-corrected methodology as the rest of 
this project's benchmarks: bf16 with no eviction as the baseline, INT8 with no eviction to isolate quantization's own effect, and INT8 + eviction (K = 16) 
to test whether the two stack multiplicatively, as predicted (~1.3x attention x ~1.6x GEMM).

Requires bitsandbytes (pip install --break-system-packages bitsandbytes) and a recent enough transformers/accelerate for BitsAndBytesConfig support with a 
custom PreTrainedModel subclass: this is the first time this codebase has gone through the HF quantization loading path, so treat the very first run as a 
smoke test of that integration, not just of the speedup number.

Run from DiffusionLLMs/:
    CUDA_VISIBLE_DEVICES=0 python -m kernel.test_quantization \\
        --n_questions 80 --batch_size 32 --context_len 2048 --k 16
"""

import argparse

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

from kernel.modeling import Fast_dLLM_QwenForCausalLM
from kernel.generation_functions import setup_model
from kernel.block_eviction_scheduler import BlockEvictionScheduler
from kernel.test_eviction import (
    BLOCK_SIZE, DEFAULT_MODEL, load_gsm8k_batched, run_generation, warm_up_model,
    _check_answer,
)

def load_model_variant(model_path, device, quantize):
    """
    Loads Fast-dLLM v2 either in plain bf16 or with INT8 weight quantization. model_path is the HuggingFace repo ID or local path, 
    and device is the target CUDA device string. quantize is "none" for bf16, or "int8" for 8-bit weight quantization via bitsandbytes. 
    Returns the loaded model, in eval mode, with .mdm_sample bound.
    """
    if quantize == "int8":
        # llm_int8_skip_modules leaves lm_head in bf16: quantizing the final projection to the vocabulary is a common source of avoidable
        # accuracy loss for comparatively little memory/speed benefit, since it's a single matmul rather than one per layer.
        bnb_config = BitsAndBytesConfig(
            load_in_8bit = True,
            llm_int8_skip_modules = ["lm_head"],
        )
        model = Fast_dLLM_QwenForCausalLM.from_pretrained(
            model_path,
            quantization_config = bnb_config,
            device_map = device,
        )
    elif quantize == "none":
        model = Fast_dLLM_QwenForCausalLM.from_pretrained(
            model_path,
            torch_dtype = torch.bfloat16,
            device_map = device,
        )
    else:
        raise ValueError(f"Unknown quantize mode: {quantize!r}.")

    model.eval()
    setup_model(model)
    return model

def evaluate_config(model, tokenizer, device, batches, k, union_mode, max_new_tokens, threshold, temperature):
    """
    Runs one full GSM8K batch sweep against an already-loaded model, optionally with block eviction armed at the given K, and returns accuracy + throughput.

    model is a model instance with mdm_sample bound, tokenizer is the model tokenizer, and device is the CUDA device string. 
    batches is a list of (encoded_batch, answers) tuples from load_gsm8k_batched. k is the number of prefix blocks to retain, or None to run the full, 
    unpruned prefix, and union_mode is the union strategy passed to BlockEvictionScheduler when k is set. max_new_tokens is the maximum tokens to 
    generate per question, threshold is the unmasking confidence threshold, and temperature is the sampling temperature (0 = greedy).

    Returns an (accuracy, tok_per_sec) tuple.
    """
    correct = 0
    total = 0
    total_new_tokens = 0
    total_time = 0.0

    scheduler = None
    if k is not None:
        scheduler = BlockEvictionScheduler(model, k = k, block_size = BLOCK_SIZE, union_mode = union_mode)
        scheduler.install()

    try:
        for encoded, answers in batches:
            input_ids = encoded.input_ids
            plen = input_ids.shape[1]
            out, elapsed = run_generation(
                model, tokenizer, input_ids,
                max_new_tokens = max_new_tokens, threshold = threshold, temperature = temperature,
            )
            total_time += elapsed
            for sample_tensor, gt in zip(out, answers):
                n_new = len(sample_tensor) - plen
                total_new_tokens += max(n_new, 0)
                text = tokenizer.decode(sample_tensor[plen:], skip_special_tokens = True)
                result = _check_answer(text, gt)
                if result is True:
                    correct += 1
                if result is not None:
                    total += 1
    finally:
        if scheduler is not None:
            scheduler.uninstall()

    tok_per_sec = total_new_tokens / total_time if total_time > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, tok_per_sec

def run_one_config(label, model_path, device, quantize, k, args):
    """
    Loads a fresh model for one config, runs the eval, and tears the model down before returning. 
    Each config gets its own process-level state rather than reusing a loaded model, since switching quantization on a live model isn't supported, 
    and this keeps every config's warm-up cost handled identically and independently.

    Returns an (accuracy, tok_per_sec) tuple.
    """
    print(f"\n=== {label} ===")
    print(f"Loading model (quantize = {quantize})...")
    model = load_model_variant(model_path, device, quantize)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code = True)

    print("Warming up...")
    warm_up_model(model, tokenizer, device)

    batches = load_gsm8k_batched(args.n_questions, tokenizer, device, args.batch_size, args.context_len)

    acc, tok_s = evaluate_config(
        model, tokenizer, device, batches, k, args.union_mode,
        args.max_new_tokens, args.threshold, args.temperature,
    )
    print(f"acc = {acc:.3f} tok/s = {tok_s:.1f}")

    del model
    torch.cuda.empty_cache()
    return acc, tok_s

def main():
    """
    Runs the three-config comparison and prints a summary table.
    """
    parser = argparse.ArgumentParser(description = "Quantization test for Fast-dLLM v2")
    parser.add_argument("--model", default = DEFAULT_MODEL)
    parser.add_argument("--device", default = "cuda:0")
    parser.add_argument("--k", type = int, default = 16,
                        help = "K value for the eviction+quantization stacking config")
    parser.add_argument("--n_questions", type = int, default = 80)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--context_len", type = int, default = 2048)
    parser.add_argument("--union_mode", choices = ["matmul", "naive"], default = "matmul")
    parser.add_argument("--threshold", type = float, default = 0.95)
    parser.add_argument("--temperature", type = float, default = 0.0)
    parser.add_argument("--max_new_tokens", type = int, default = 512)
    args = parser.parse_args()

    results = {}
    results["bf16 baseline"] = run_one_config(
        "Config 1/3: bf16, no eviction (baseline)", args.model, args.device, "none", None, args,
    )
    results["INT8 only"] = run_one_config(
        "Config 2/3: INT8, no eviction", args.model, args.device, "int8", None, args,
    )
    results[f"INT8 + eviction K = {args.k}"] = run_one_config(
        f"Config 3/3: INT8 + eviction (K = {args.k})", args.model, args.device, "int8", args.k, args,
    )

    base_acc, base_tok_s = results["bf16 baseline"]

    print("\n" + "=" * 60)
    print("Results Summary:")
    print(f"{'Config':<24} {'Accuracy':<12} {'tok/s':<10} {'Speedup'}")
    for label, (acc, tok_s) in results.items():
        speedup = tok_s / base_tok_s if base_tok_s > 0 else 0.0
        print(f"{label:<24} {acc:<12.3f} {tok_s:<10.1f} {speedup:.2f}x")

    # Illustrative comparison against the naive multiplicative prediction (eviction-only speedup, measured separately in test_eviction.py, times
    # INT8-only speedup measured just above). Not a claim that the two effects are perfectly independent: just a sanity check on whether
    # the stacked result is in the right neighborhood of that estimate.
    int8_only_speedup = results["INT8 only"][1] / base_tok_s if base_tok_s > 0 else 0.0
    stacked_speedup = results[f"INT8 + eviction K = {args.k}"][1] / base_tok_s if base_tok_s > 0 else 0.0
    print(f"\nINT8-only speedup: {int8_only_speedup:.2f}x")
    print(f"Measured stacked speedup (INT8 + eviction): {stacked_speedup:.2f}x")
    print(
        "(Compare against your separately-measured eviction-only speedup "
        "multiplied by the INT8-only speedup above, as a rough check on "
        "whether the two effects compound as expected.)"
    )

if __name__ == "__main__":
    main()