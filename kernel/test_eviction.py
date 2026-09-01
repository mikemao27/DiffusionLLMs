"""
Test and Evaluation Harness for KV Block Eviction.

Verifies the eviction hook is a correctness no-op at K = N, then measures accuracy and throughput across a range of K values on GSM8K. There are two
entry points.

python test_eviction.py --mode noop

No-op correctness gate. Runs baseline and K = N+10 side-by-side on GSM8K examples. Output must be token-identical. If this fails, no K-sweep result can 
be trusted.

python test_eviction.py --mode eval [--k_values 4 8 16 32] [--n_questions 80]

Accuracy + throughput sweep over multiple K values on GSM8K.

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

from kernel.modeling import Fast_dLLM_QwenForCausalLM, fuse_all_projections
from kernel.generation_functions import setup_model
from kernel.block_eviction_scheduler import BlockEvictionScheduler

BLOCK_SIZE = 32
DEFAULT_MODEL = "Efficient-Large-Model/Fast_dLLM_v2_7B"
FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "

def load_model(model_path, device = "cuda:0", fuse_projections = False, compile_model = False):
    """
    Load Fast-dLLM v2 from kernel.modeling and inject the batch_sample generation method. Using kernel.modeling gives us the eviction hooks directly in 
    the attention forward without trust_remote_code.

    model_path is the HuggingFace repo ID or local path, and device is the target CUDA device string. fuse_projections, if True, fuses each layer's QKV 
    and gate + up projections into single matmuls after loading (see kernel.modeling.fuse_all_projections). Mathematically exact, verified independently
    in /home/claude/verify_fusion_math.py, but this is its first time running against the real checkpoint and the full generation loop, so treat the 
    first eval sweep with this flag as a smoke test too, same as with the quantization loading path. compile_model, if True, wraps model.forward with 
    torch.compile(dynamic = True). Experimental: the generation loop's Python-level branching (use_block_cache, training vs eval, mask-shape-dependent 
    paths) and the shrinking batch dimension as samples finish mid-generation are exactly the kind of dynamic control flow that can cause graph breaks 
    or frequent recompilation. Watch stderr for recompilation warnings on the first run: if compilation time dominates, most of any expected speedup 
    will be hidden in that one-time (or repeated) cost rather than showing up in the reported tok/s.

    Returns the loaded model, in eval mode, with .mdm_sample bound.
    """
    model = Fast_dLLM_QwenForCausalLM.from_pretrained(
        model_path,
        torch_dtype = torch.bfloat16,
        device_map = device,
    )
    model.eval()
    setup_model(model)
    if fuse_projections:
        fuse_all_projections(model)
    if compile_model:
        model.forward = torch.compile(model.forward, dynamic = True)
    return model

def _build_filler_ids(tokenizer, min_tokens):
    """
    Builds a filler token-id tensor with at least min_tokens tokens by repeating FILLER_SENTENCE, doubling the repeat count until long enough.

    This exists because a fixed-size filler constant silently caps out at whatever it tokenizes to: past that length, _make_prompt pads with everything 
    available and stops, with no error, producing a prompt far shorter than the requested target_ctx_len. Building the filler on demand means --context_len 
    works correctly at any length, including ones larger than previously tested.

    tokenizer is the model tokenizer, and min_tokens is the minimum number of filler tokens required. Returns a 1-D LongTensor of at least min_tokens 
    token ids.
    """
    reps = max(200, (min_tokens // 8) + 1) # ~8 tokens/sentence, rough starting guess.
    while True:
        ids = tokenizer(FILLER_SENTENCE * reps, return_tensors = "pt").input_ids[0]
        if len(ids) >= min_tokens:
            return ids
        reps *= 2

def _make_prompt(tokenizer, question, target_ctx_len = None):
    """
    Build a chat-templated prompt for a GSM8K question. If target_ctx_len is given, prepend filler so the prompt reaches approximately that many tokens, 
    putting the model in a long-context regime where block eviction has real blocks to prune. tokenizer is the model tokenizer, question is the raw 
    question string, and target_ctx_len is an optional token count to pad the prompt to. Returns the chat-templated prompt string.
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
    Checks a generated answer against ground truth. Prefers an explicit "#### N" marker in the generation; falls back to the last number mentioned if 
    no marker is present. generated is the model's generated text, and ground_truth is the reference answer string, or None (unscored sample). Returns 
    True/False if scored, or None if ground_truth was None.
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
    Load n GSM8K test questions and batch them with left-padding. n is the number of questions to load, tokenizer is the model tokenizer, and device is 
    the CUDA device string. batch_size is the number of questions per batch, and target_ctx_len is an optional token count to pad prompts to. Returns a 
    list of (encoded_batch, answers) tuples.
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
            f"[ctx] target = {target_ctx_len} actual prompt_len = {actual_len} "
            f"n_blocks = {actual_len // BLOCK_SIZE}"
        )
    return batches

def warm_up_model(model, tokenizer, device):
    """
    Runs one small, untimed generation call to absorb one-time costs (CUDA context growth, cuDNN algorithm selection, memory allocator warm-up) that 
    would otherwise land inside whichever measurement happens to run first.

    This matters because, across repeated eval-sweep runs, the K-value throughput numbers were consistently stable (within ~0.3% run to run) while the 
    baseline (full-prefix, always measured first) varied by up to ~26%. That pattern, stable everywhere except the first timed call in the process, is 
    the signature of warm-up cost, not GPU contention or thermal state. Running a throwaway generation before any timed measurement keeps that cost out 
    of the results entirely.

    model is a model instance with mdm_sample bound, tokenizer is the model tokenizer, and device is the CUDA device string. Returns nothing.
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
    small_block_size = 8,
):
    """
    Run batch_sample on a single input_ids tensor.

    model is a model instance with mdm_sample bound, tokenizer is the model tokenizer, and input_ids is a [B, L] token tensor on the correct device.
    max_new_tokens is the maximum tokens to generate, threshold is the unmasking confidence threshold, and temperature is the sampling temperature 
    (0 = greedy). small_block_size is the tokens unmasked together within a block, and must evenly divide BLOCK_SIZE (32): valid values are 1, 2, 4, 8, 
    16, 32. Smaller values mean more, smaller sub-block forward calls per block; larger values mean fewer, bigger ones. num_small_blocks = BLOCK_SIZE // 
    small_block_size is the minimum number of forward calls per block, regardless of how confidently the model unmasks: this is the floor that threshold 
    tuning alone cannot go below.

    Returns ordered, a list of output token tensors, one per input row, each containing the full sequence including the prompt, and elapsed, the 
    wall-clock seconds for the generation call.
    """
    plen = input_ids.shape[1]
    seq_len = torch.full((input_ids.shape[0],), plen, dtype = torch.long, device = input_ids.device)

    t0 = time.perf_counter()
    finished = model.mdm_sample(
        input_ids = input_ids,
        tokenizer = tokenizer,
        block_size = BLOCK_SIZE,
        max_new_tokens = max_new_tokens,
        small_block_size = small_block_size,
        min_len = plen,
        seq_len = seq_len,
        threshold = threshold,
        temperature = temperature,
        # Without this, batch_sample's use_block_cache defaults to False, and every small-step forward call takes generation_functions.py's non-cache 
        # branch, which always queries the FULL block_size = 32 width regardless of small_block_size. That defeats modeling.py's is_subblock_call gate
        # (query_states.shape[-2] < cur_block_kv_width) unconditionally, so the sparse kernel path -- sparse_attn_merge before, 
        # block_sparse_attention_fused now, can never fire: hit_rate is 0% for every K and batch size, regardless of which kernel is wired in. 
        # small_block_size only has any effect (on call count, or on which kernel path gets exercised) when this is True.
        use_block_cache = True,
    )
    elapsed = time.perf_counter() - t0

    # Finished is a dict {sample_idx: tensor}. Reassemble in order.
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
    Verify K = N+10 produces token-identical output to the baseline. model is a model instance with mdm_sample bound, tokenizer is the model tokenizer, 
    and device is the CUDA device string. n_examples is the number of GSM8K questions to test, and context_len is an optional token count to pad 
    prompts to. union_mode is the union strategy passed to BlockEvictionScheduler, and use_sparse_kernel controls whether to also exercise the Triton 
    sparse kernel path during the no-op check.

    Returns True if every example was token-identical to baseline, else False.
    """
    print(f"\nNo-op correctness test (union_mode = {union_mode}).")
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
        print(f"[baseline] {base_text[:120]!r}")
        print(f"[K = {k_noop}] {topk_text[:120]!r}")

        min_len = min(len(base_tokens), len(topk_tokens))
        match = torch.equal(base_tokens[:min_len], topk_tokens[:min_len])
        length_match = len(base_tokens) == len(topk_tokens)

        if match and length_match:
            print(f"PASS: token-identical ({len(base_tokens)} tokens).")
        else:
            all_passed = False
            if not length_match:
                print(
                    f"FAIL: length mismatch: "
                    f"Baseline = {len(base_tokens)}, topk = {len(topk_tokens)}."
                )
            if not match:
                first_diff = next(
                    (j for j in range(min_len) if base_tokens[j] != topk_tokens[j]),
                    min_len,
                )
                print(
                    f"FAIL: first token divergence at position {first_diff} "
                    f"Baseline[{first_diff}] = {base_tokens[first_diff].item()} "
                    f"topk[{first_diff}] = {topk_tokens[first_diff].item()}."
                )

    verdict = "ALL PASS" if all_passed else "SOME FAILED"
    detail = "hook is a correct no-op" if all_passed else "DO NOT proceed with K-sweep"
    print(f"\n{verdict}: {detail}.")
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
    small_block_size = 8,
):
    """
    Sweep over k_values and compare accuracy + throughput against the dense baseline.

    model is a model instance with mdm_sample bound, tokenizer is the model tokenizer, and device is the CUDA device string. k_values is the list of 
    K values to evaluate, n_questions is the number of GSM8K questions to evaluate on, and batch_size is the questions per batch. context_len is an 
    optional token count to pad prompts to, and union_mode is the union strategy passed to BlockEvictionScheduler. threshold is the unmasking 
    confidence threshold, temperature is the sampling temperature (0 = greedy), and max_new_tokens is the maximum tokens to generate per question. 
    use_sparse_kernel controls whether to use the Triton sparse kernel path. small_block_size is the tokens unmasked together within a block and must 
    evenly divide BLOCK_SIZE (32); see run_generation for why this, not threshold, is the lever that reduces the minimum forward-call count per block.

    Returns a list of (k, accuracy, tok_per_sec, speedup, frac_kv_loaded) tuples, one per value in k_values.
    """
    print(
        f"\nLoading {n_questions} GSM8K questions "
        f"(batch_size = {batch_size}, context_len = {context_len})..."
    )
    batches = load_gsm8k_batched(n_questions, tokenizer, device, batch_size, context_len)

    def evaluate_k(k):
        """
        Runs the full batch set once, either at the dense baseline (k = None) or with a BlockEvictionScheduler installed at the given k. Returns 
        (accuracy, tok_per_sec, avg_frac_loaded, sparse_diag) where sparse_diag is (hits, misses, hit_rate, rebuild_misses).
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
                    small_block_size = small_block_size,
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

    print("\nEvaluating baseline (full prefix)...")
    base_acc, base_tok_s, _, _ = evaluate_k(None)
    print(f"Baseline: acc = {base_acc:.3f}  tok/s = {base_tok_s:.1f}")

    results = []
    for k in k_values:
        print(f"\nEvaluating K = {k} (union_mode = {union_mode})...")
        acc, tok_s, frac, sparse_diag = evaluate_k(k)
        speedup = tok_s / base_tok_s if base_tok_s > 0 else 0.0
        print(
            f"K = {k}: acc = {acc:.3f} tok/s = {tok_s:.1f}  "
            f"Speedup = {speedup:.2f}x frac_loaded = {frac:.2%}"
        )
        if use_sparse_kernel:
            hits, misses, hit_rate, rebuild_misses = sparse_diag
            other_misses = misses - rebuild_misses
            print(
                f"[sparse kernel] hits = {hits} misses = {misses}  "
                f"hit_rate = {hit_rate:.1%} "
                f"(rebuild_misses = {rebuild_misses}, other_misses = {other_misses})"
            )
        results.append((k, acc, tok_s, speedup, frac))

    print("\nResults Summary:")
    print(f"{'K':<8} {'KV loaded':<12} {'Accuracy':<12} {'tok/s':<10} {'Speedup'}")
    print(f"{'full':<8} {'100%':<12} {base_acc:<12.3f} {base_tok_s:<10.1f} 1.00x")
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
    parser.add_argument("--small_block_size", type = int, default = 8,
                        help = "Tokens unmasked together within a block. Must evenly divide "
                               "BLOCK_SIZE (32): valid values: 1, 2, 4, 8, 16, 32. Lower values "
                               "= more, smaller forward calls per block; higher values = fewer, "
                               "bigger ones. This is the lever that actually reduces forward-call "
                               "count: threshold tuning alone cannot go below "
                               "BLOCK_SIZE // small_block_size calls per block.")
    parser.add_argument("--fuse_projections", action = "store_true",
                        help = "Fuse each layer's QKV and gate+up projections into single "
                               "matmuls. Orthogonal to eviction: stacks with --use_sparse_kernel "
                               "and any --k_values.")
    parser.add_argument("--compile", action = "store_true", dest = "compile_model",
                        help = "Wrap model.forward with torch.compile(dynamic = True). Experimental: "
                               "watch stderr for recompilation warnings on first use.")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = load_model(
        args.model, device = args.device,
        fuse_projections = args.fuse_projections, compile_model = args.compile_model,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code = True)

    print("Warming up (absorbing one-time CUDA/cuDNN/allocator costs)...")
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
        if BLOCK_SIZE % args.small_block_size != 0:
            parser.error(f"--small_block_size ({args.small_block_size}) must evenly divide BLOCK_SIZE ({BLOCK_SIZE})")
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
            small_block_size = args.small_block_size,
        )

if __name__ == "__main__":
    main()