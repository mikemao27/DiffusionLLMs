"""
Test and Evaluation Harness for KV Block Eviction.

Job: verify the eviction hook is a correctness no-op at K=N, then measure
accuracy and throughput across a range of K values on GSM8K.

Two entry points:

  python test_eviction.py --mode noop
      No-op correctness gate. Runs baseline and K=N+10 side-by-side on GSM8K
      examples. Output must be token-identical. If this fails, no K-sweep result
      can be trusted.

  python test_eviction.py --mode eval [--k_values 4 8 16 32] [--n_questions 80]
      Accuracy + throughput sweep over multiple K values on GSM8K.

Run from DiffusionLLMs/:
    CUDA_VISIBLE_DEVICES=0 python -m kernel.test_eviction --mode noop
    CUDA_VISIBLE_DEVICES=0 python -m kernel.test_eviction --mode eval \\
        --k_values 8 16 32 --n_questions 80 --batch_size 32 --union_mode matmul
"""

import re
import sys
import time
import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from kernel.modeling import Fast_dLLM_QwenForCausalLM
from kernel.generation_functions import setup_model
from kernel.block_eviction_scheduler import BlockEvictionScheduler

BLOCK_SIZE = 32
DEFAULT_MODEL = "Efficient-Large-Model/Fast_dLLM_v2_7B"
FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "

def load_model(model_path, device = "cuda:0"):
    """
    Load Fast-dLLM v2 from kernel.modeling and inject the batch_sample generation
    method. Using kernel.modeling gives us the eviction hooks directly in the
    attention forward without trust_remote_code.

    Args:
        - model_path: HuggingFace repo ID or local path.
        - device: target CUDA device string.

    Returns:
        - the loaded model, in eval mode, with .mdm_sample bound.
    """
    model = Fast_dLLM_QwenForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16,
        device_map = device,
    )
    model.eval()
    setup_model(model)
    return model

def _build_filler_ids(tokenizer, min_tokens):
    """
    Builds a filler token-id tensor with at least min_tokens tokens by
    repeating FILLER_SENTENCE, doubling the repeat count until long enough.

    Why this exists: a fixed-size filler constant silently caps out at
    whatever it tokenizes to -- past that length, _make_prompt pads with
    everything available and stops, with no error, producing a prompt far
    shorter than the requested target_ctx_len. Building the filler on demand
    means --context_len works correctly at any length, including ones larger
    than previously tested.

    Args:
        - tokenizer: the model tokenizer.
        - min_tokens: minimum number of filler tokens required.

    Returns:
        - a 1-D LongTensor of at least min_tokens token ids.
    """
    reps = max(200, (min_tokens // 8) + 1)  # ~8 tokens/sentence, rough starting guess
    while True:
        ids = tokenizer(FILLER_SENTENCE * reps, return_tensors = "pt").input_ids[0]
        if len(ids) >= min_tokens:
            return ids
        reps *= 2

def _make_prompt(tokenizer, question, target_ctx_len = None):
    """
    Build a chat-templated prompt for a GSM8K question. If target_ctx_len is given,
    prepend filler so the prompt reaches approximately that many tokens, putting the
    model in a long-context regime where block eviction has real blocks to prune.

    Args:
        - tokenizer: the model tokenizer.
        - question: raw question string.
        - target_ctx_len: optional token count to pad the prompt to.

    Returns:
        - the chat-templated prompt string.
    """
    filler_prefix = ""
    if target_ctx_len is not None:
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize = False,
            add_generation_prompt = True,
        )
        base_len = len(tokenizer(base, return_tensors = "pt").input_ids[0])
        need = max(0, target_ctx_len - base_len)
        if need > 0:
            fids = _build_filler_ids(tokenizer, need)
            filler_prefix = tokenizer.decode(fids[:need], skip_special_tokens = True) + " "

    content = filler_prefix + question
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize = False,
        add_generation_prompt = True,
    )

def _extract_gsm8k_answer(text):
    """
    Pulls the final "#### N" answer out of a GSM8K reference solution string.
    """
    m = re.search(r"####\s*([\d,]+)", text)
    return m.group(1).replace(",", "") if m else None

def _check_answer(generated, ground_truth):
    """
    Checks a generated answer against ground truth. Prefers an explicit
    "#### N" marker in the generation; falls back to the last number
    mentioned if no marker is present.

    Args:
        - generated: the model's generated text.
        - ground_truth: the reference answer string, or None (unscored sample).

    Returns:
        - True/False if scored, or None if ground_truth was None.
    """
    if ground_truth is None:
        return None
    m = re.search(r"####\s*([\d,]+)", generated)
    if m:
        return m.group(1).replace(",", "") == ground_truth
    nums = re.findall(r"\b\d+(?:,\d+)*\b", generated)
    return bool(nums) and nums[-1].replace(",", "") == ground_truth


def load_gsm8k_batched(n, tokenizer, device, batch_size, target_ctx_len = None):
    """
    Load n GSM8K test questions and batch them with left-padding.

    Args:
        - n: number of questions to load.
        - tokenizer: the model tokenizer.
        - device: CUDA device string.
        - batch_size: number of questions per batch.
        - target_ctx_len: optional token count to pad prompts to.

    Returns:
        - a list of (encoded_batch, answers) tuples.
    """
    ds = load_dataset("gsm8k", "main", split = "test")
    questions = ds[:n]["question"]
    answers = [_extract_gsm8k_answer(ds[i]["answer"]) for i in range(n)]
    texts = [_make_prompt(tokenizer, q, target_ctx_len) for q in questions]

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    batches = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        encoded = tokenizer(chunk, return_tensors = "pt", padding = True).to(device)
        batches.append((encoded, answers[i:i + batch_size]))

    tokenizer.padding_side = orig_side

    if target_ctx_len is not None:
        actual_len = batches[0][0].input_ids.shape[1]
        print(
            f"  [ctx] target = {target_ctx_len}  actual prompt_len = {actual_len}  "
            f"n_blocks = {actual_len // BLOCK_SIZE}"
        )
    return batches

def warm_up_model(model, tokenizer, device):
    """
    Runs one small, untimed generation call to absorb one-time costs (CUDA
    context growth, cuDNN algorithm selection, memory allocator warm-up) that
    would otherwise land inside whichever measurement happens to run first.

    Why this matters: across repeated eval-sweep runs, the K-value throughput
    numbers were consistently stable (within ~0.3% run to run) while the
    baseline (full-prefix, always measured first) varied by up to ~26%. That
    pattern -- stable everywhere except the first timed call in the process --
    is the signature of warm-up cost, not GPU contention or thermal state.
    Running a throwaway generation before any timed measurement keeps that
    cost out of the results entirely.

    Args:
        - model: model instance with mdm_sample bound.
        - tokenizer: the model tokenizer.
        - device: CUDA device string.

    Returns:
        - None.
    """
    dummy_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2?"}],
        tokenize = False,
        add_generation_prompt = True,
    )
    input_ids = tokenizer(dummy_text, return_tensors = "pt").input_ids.to(device)
    run_generation(model, tokenizer, input_ids, max_new_tokens = 32)

def run_generation(
    model,
    tokenizer,
    input_ids,
    max_new_tokens = 512,
    threshold = 0.95,
    temperature = 0.0,
):
    """
    Run batch_sample on a single input_ids tensor.

    Args:
        - model: model instance with mdm_sample bound.
        - tokenizer: the model tokenizer.
        - input_ids: [B, L] token tensor on the correct device.
        - max_new_tokens: maximum tokens to generate.
        - threshold: unmasking confidence threshold.
        - temperature: sampling temperature (0 = greedy).

    Returns:
        - ordered: list of output token tensors, one per input row, each
            containing the full sequence including the prompt.
        - elapsed: wall-clock seconds for the generation call.
    """
    plen = input_ids.shape[1]
    seq_len = torch.full((input_ids.shape[0],), plen, dtype = torch.long, device = input_ids.device)

    t0 = time.perf_counter()
    finished = model.mdm_sample(
        input_ids = input_ids,
        tokenizer = tokenizer,
        block_size = BLOCK_SIZE,
        max_new_tokens = max_new_tokens,
        small_block_size = 8,
        min_len = plen,
        seq_len = seq_len,
        threshold = threshold,
        temperature = temperature,
    )
    elapsed = time.perf_counter() - t0

    # finished is a dict {sample_idx: tensor}. Reassemble in order.
    ordered = [finished[i] for i in range(len(finished))]
    return ordered, elapsed

def run_noop_test(
    model,
    tokenizer,
    device,
    n_examples = 3,
    context_len = None,
    union_mode = "matmul",
    use_sparse_kernel = False,
):
    """
    Verify K = N+10 produces token-identical output to the baseline.

    Args:
        - model: model instance with mdm_sample bound.
        - tokenizer: the model tokenizer.
        - device: CUDA device string.
        - n_examples: number of GSM8K questions to test.
        - context_len: optional token count to pad prompts to.
        - union_mode: union strategy passed to BlockEvictionScheduler.
        - use_sparse_kernel: whether to also exercise the Triton sparse kernel
            path during the no-op check.

    Returns:
        - True if every example was token-identical to baseline, else False.
    """
    print(f"\nNo-op correctness test (union_mode = {union_mode})")
    ds = load_dataset("gsm8k", "main", split = "test")
    tokenizer.pad_token = tokenizer.eos_token

    all_passed = True
    for i in range(n_examples):
        q = ds[i]["question"]
        text = _make_prompt(tokenizer, q, context_len)
        input_ids = tokenizer(text, return_tensors = "pt").input_ids.to(device)
        plen = input_ids.shape[1]
        n_blocks = plen // BLOCK_SIZE
        k_noop = n_blocks + 10

        print(f"\nExample {i} (n_blocks = {n_blocks}, K_noop = {k_noop}):")

        base_out, _ = run_generation(model, tokenizer, input_ids)
        base_tokens = base_out[0]

        scheduler = BlockEvictionScheduler(
            model, k = k_noop, block_size = BLOCK_SIZE, union_mode = union_mode,
            use_sparse_kernel = use_sparse_kernel,
        )
        with scheduler:
            topk_out, _ = run_generation(model, tokenizer, input_ids)
        topk_tokens = topk_out[0]

        base_text = tokenizer.decode(base_tokens[plen:], skip_special_tokens = True)
        topk_text = tokenizer.decode(topk_tokens[plen:], skip_special_tokens = True)
        print(f"  [baseline] {base_text[:120]!r}")
        print(f"  [K={k_noop}] {topk_text[:120]!r}")

        min_len = min(len(base_tokens), len(topk_tokens))
        match = torch.equal(base_tokens[:min_len], topk_tokens[:min_len])
        length_match = len(base_tokens) == len(topk_tokens)

        if match and length_match:
            print(f"  PASS — token-identical ({len(base_tokens)} tokens)")
        else:
            all_passed = False
            if not length_match:
                print(
                    f"  FAIL — length mismatch: "
                    f"baseline = {len(base_tokens)}, topk = {len(topk_tokens)}"
                )
            if not match:
                first_diff = next(
                    (j for j in range(min_len) if base_tokens[j] != topk_tokens[j]),
                    min_len,
                )
                print(
                    f"  FAIL — first token divergence at position {first_diff}  "
                    f"baseline[{first_diff}] = {base_tokens[first_diff].item()}  "
                    f"topk[{first_diff}] = {topk_tokens[first_diff].item()}"
                )

    verdict = "ALL PASS" if all_passed else "SOME FAILED"
    detail = "hook is a correct no-op" if all_passed else "DO NOT proceed with K-sweep"
    print(f"\n{verdict} — {detail}")
    return all_passed

def run_eval_sweep(
    model,
    tokenizer,
    device,
    k_values,
    n_questions,
    batch_size,
    context_len = None,
    union_mode = "matmul",
    threshold = 0.95,
    temperature = 0.0,
    max_new_tokens = 512,
    use_sparse_kernel = False,
):
    """
    Sweep over k_values and compare accuracy + throughput against the dense baseline.

    Args:
        - model: model instance with mdm_sample bound.
        - tokenizer: the model tokenizer.
        - device: CUDA device string.
        - k_values: list of K values to evaluate.
        - n_questions: number of GSM8K questions to evaluate on.
        - batch_size: questions per batch.
        - context_len: optional token count to pad prompts to.
        - union_mode: union strategy passed to BlockEvictionScheduler.
        - threshold: unmasking confidence threshold.
        - temperature: sampling temperature (0 = greedy).
        - max_new_tokens: maximum tokens to generate per question.
        - use_sparse_kernel: whether to use the Triton sparse kernel path.

    Returns:
        - a list of (k, accuracy, tok_per_sec, speedup, frac_kv_loaded) tuples,
            one per value in k_values.
    """
    print(
        f"\nLoading {n_questions} GSM8K questions "
        f"(batch_size = {batch_size}, context_len = {context_len}) ..."
    )
    batches = load_gsm8k_batched(n_questions, tokenizer, device, batch_size, context_len)

    def evaluate_k(k):
        """
        Runs the full batch set once, either at the dense baseline (k=None)
        or with a BlockEvictionScheduler installed at the given k.

        Returns:
            - (accuracy, tok_per_sec, avg_frac_loaded, sparse_diag) where
                sparse_diag is (hits, misses, hit_rate, rebuild_misses).
        """
        correct = 0
        total = 0
        total_new_tokens = 0
        total_time = 0.0

        scheduler = None
        if k is not None:
            scheduler = BlockEvictionScheduler(
                model, k = k, block_size = BLOCK_SIZE, union_mode = union_mode,
                use_sparse_kernel = use_sparse_kernel,
            )
            scheduler.install()

        try:
            for encoded, answers in batches:
                input_ids = encoded.input_ids
                plen = input_ids.shape[1]
                out, elapsed = run_generation(
                    model, tokenizer, input_ids,
                    max_new_tokens = max_new_tokens,
                    threshold = threshold,
                    temperature = temperature,
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
                sparse_stats = scheduler.sparse_kernel_stats
                scheduler.uninstall()

        tok_per_sec = total_new_tokens / total_time if total_time > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        avg_n, avg_u, avg_frac = scheduler.avg_stats if scheduler is not None else (0, 0, 0.0)
        sparse_diag = sparse_stats if scheduler is not None else (0, 0, 0.0, 0)
        return accuracy, tok_per_sec, avg_frac, sparse_diag

    print("\nEvaluating baseline (full prefix) ...")
    base_acc, base_tok_s, _, _ = evaluate_k(None)
    print(f"  baseline: acc = {base_acc:.3f}  tok/s = {base_tok_s:.1f}")

    results = []
    for k in k_values:
        print(f"\nEvaluating K = {k} (union_mode = {union_mode}) ...")
        acc, tok_s, frac, sparse_diag = evaluate_k(k)
        speedup = tok_s / base_tok_s if base_tok_s > 0 else 0.0
        print(
            f"  K = {k}: acc = {acc:.3f}  tok/s = {tok_s:.1f}  "
            f"speedup = {speedup:.2f}x  frac_loaded = {frac:.2%}"
        )
        if use_sparse_kernel:
            hits, misses, hit_rate, rebuild_misses = sparse_diag
            other_misses = misses - rebuild_misses
            print(
                f"    [sparse kernel] hits = {hits}  misses = {misses}  "
                f"hit_rate = {hit_rate:.1%}  "
                f"(rebuild_misses = {rebuild_misses}, other_misses = {other_misses})"
            )
        results.append((k, acc, tok_s, speedup, frac))

    print("\nResults Summary:")
    print(f"  {'K':<8} {'KV loaded':<12} {'Accuracy':<12} {'tok/s':<10} {'Speedup'}")
    print(f"  {'full':<8} {'100%':<12} {base_acc:<12.3f} {base_tok_s:<10.1f} 1.00x")
    for k, acc, tok_s, speedup, frac in results:
        print(f"  {k:<8} {frac:<12.1%} {acc:<12.3f} {tok_s:<10.1f} {speedup:.2f}x")
    return results

def main():
    """
    Parses CLI args and dispatches to run_noop_test or run_eval_sweep.
    """
    parser = argparse.ArgumentParser(description = "Block eviction kernel tests and evaluation")
    parser.add_argument("--mode", choices = ["noop", "eval"], default = "noop",
                        help = "noop: correctness gate; eval: accuracy+throughput sweep")
    parser.add_argument("--model", default = DEFAULT_MODEL,
                        help = "HuggingFace model path or local path")
    parser.add_argument("--device", default = "cuda:0")
    parser.add_argument("--k_values", type = int, nargs = "+", default = [4, 8, 16, 32])
    parser.add_argument("--n_questions", type = int, default = 80)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--context_len", type = int, default = None,
                        help = "Pad prompts to approximately this many tokens (e.g. 2048)")
    parser.add_argument("--n_examples", type = int, default = 3,
                        help = "Number of examples for noop test")
    parser.add_argument("--union_mode", choices = ["matmul", "naive"], default = "matmul",
                        help = "matmul: global top-K (smaller union); "
                               "naive: per-sequence union (baseline behavior)")
    parser.add_argument("--use_sparse_kernel", action = "store_true",
                        help = "Use the Triton sparse_attn_merge kernel instead of "
                               "gather+cat+SDPA for the small-step attention path.")
    parser.add_argument("--threshold", type = float, default = 0.95)
    parser.add_argument("--temperature", type = float, default = 0.0)
    parser.add_argument("--max_new_tokens", type = int, default = 512)
    args = parser.parse_args()

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model, device = args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code = True)

    print("Warming up (absorbing one-time CUDA/cuDNN/allocator costs) ...")
    warm_up_model(model, tokenizer, args.device)

    if args.mode == "noop":
        passed = run_noop_test(
            model, tokenizer, args.device,
            n_examples = args.n_examples,
            context_len = args.context_len,
            union_mode = args.union_mode,
            use_sparse_kernel = args.use_sparse_kernel,
        )
        sys.exit(0 if passed else 1)
    else:
        run_eval_sweep(
            model, tokenizer, args.device,
            k_values = args.k_values,
            n_questions = args.n_questions,
            batch_size = args.batch_size,
            context_len = args.context_len,
            union_mode = args.union_mode,
            threshold = args.threshold,
            temperature = args.temperature,
            max_new_tokens = args.max_new_tokens,
            use_sparse_kernel = args.use_sparse_kernel,
        )

if __name__ == "__main__":
    main()