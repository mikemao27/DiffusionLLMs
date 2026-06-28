"""
Benchmark and accuracy comparison: Fast-dLLM v2 baseline vs. our paged
block-eviction kernel.

What this script does:

1. Loads the Fast-dLLM v2 model (Qwen2.5-7B or 1.5B) and tokenizer.
2. Defines a small set of GSM8K-style arithmetic prompts.
3. Runs each prompt twice:
    (A) Baseline — generation_functions.py with the original DynamicCache.
    (B) Paged — generation_functions.py with PagedKVCache + eviction.
4. Records wall-clock time and token throughput for both paths.
5. Prints a side-by-side accuracy/speed table.

Usage:

    python test_kernel.py \
        --model_path Efficient-Large-Model/Fast_dLLM_v2_7B \
        --block_size 32 \
        --top_k 4 \
        --max_new_tokens 256 \
        --num_prompts 10

    # Smaller model for quick iteration:
    python test_kernel.py --model_path Efficient-Large-Model/Fast_dLLM_v2_1.5B

Measuring "accuracy":

For each prompt we extract the final integer answer (####  <number> pattern
from GSM8K) and compare baseline vs. paged outputs.  We also record whether
the paged output matches the baseline (agreement rate) as a proxy for
quality preservation.  Because we run both on the same prompt with the same
random seed, divergence directly quantifies quality degradation from eviction.

Speedup:

We measure:
    tokens_per_second = max_new_tokens / wall_clock_time

The primary bottleneck in Fast-dLLM v2 is the attention over the growing
KV cache as more blocks are committed.  Block eviction bounds this cost at
O(top_k * block_size) rather than O(num_generated_tokens), producing the
expected 2–3× speedup for sequences longer than top_k blocks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Path setup — allow running from repo root or from kernel/ directly.
REPO_ROOT = Path(__file__).parent.parent # DiffusionLLMs/
FAST_DLLM_PATH = REPO_ROOT / "Fast-dLLM" / "v2"
sys.path.insert(0, str(FAST_DLLM_PATH))
sys.path.insert(0, str(Path(__file__).parent)) # kernel/ itself

# Fast-dLLM v2 generation helpers.
try:
    from generation_functions import (
        Fast_dLLM_QwenForCausalLM,
        setup_model_with_custom_generation,
    )
except ImportError as e:
    raise ImportError(
        f"Could not import generation_functions from {FAST_DLLM_PATH}.\n"
        f"Ensure Fast-dLLM/v2/ is at {FAST_DLLM_PATH}.\n"
        f"Original error: {e}"
    )

# Our kernel files.
from paged_kv_cache import PagedKVCache, BLOCK_SIZE, DEFAULT_MAX_SLOTS
from eviction_scheduler import EvictionScheduler
from block_sparse_attention import install_paged_attention_hook

# Sample GSM8K-style prompts:
GSM8K_PROMPTS: List[str] = [
    "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?",
    "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She currently has 20 chickens. How many cups of feed does she use every week?",
    "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
    "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. How many golf balls did he have at the end of Wednesday?",
    "There were nine computers in the server room. Five more computers were installed each day, from Monday to Thursday. How many computers are now in the server room?",
    "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
    "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
]

SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Write your final answer as ####  <number>.\n\n"
)

# Helper: extract numeric answer from model output
def extract_answer(text: str) -> Optional[str]:
    """Extract the #### <number> answer pattern from GSM8K output."""
    match = re.search(r"####\s*([\d,\.\-]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    return None


# Baseline generation (original DynamicCache path)
@torch.no_grad()
def run_baseline(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    block_size: int,
    max_new_tokens: int,
    threshold: float,
) -> Tuple[str, float]:
    """
    Run the unmodified Fast-dLLM v2 batch_sample with the default
    DynamicCache.  Returns (decoded_text, tokens_per_second).
    """
    seq_len = torch.tensor([input_ids.shape[1]], device=model.device)
    min_len = seq_len.min().item()

    t0 = time.perf_counter()
    finished = model.batch_sample(
        input_ids=input_ids.clone(),
        tokenizer=tokenizer,
        block_size=block_size,
        max_new_tokens=max_new_tokens,
        small_block_size=block_size,
        min_len=min_len,
        seq_len=seq_len.clone(),
        threshold=threshold,
        use_block_cache=False,
    )
    elapsed = time.perf_counter() - t0

    output_ids = finished[0]
    prompt_len = input_ids.shape[1]
    new_tokens = output_ids[prompt_len:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    tps = new_tokens.shape[0] / elapsed if elapsed > 0 else 0.0
    return decoded, tps


# Paged generation (our kernel path):
@torch.no_grad()
def run_paged(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    block_size: int,
    max_new_tokens: int,
    threshold: float,
    top_k: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[str, float]:
    """
    Run Fast-dLLM v2 batch_sample with PagedKVCache and the eviction scheduler.

    Integration strategy:

    We create a PagedKVCache, record the prompt length, install the paged
    attention hook on the model, run batch_sample (which calls forward()
    normally — the hook intercepts), then uninstall the hook.

    Because we install/uninstall per call, the baseline and paged runs
    never interfere with each other.
    """
    # Maximum number of blocks we can expect.
    max_slots = (max_new_tokens // block_size + 32) * 2 # headroom factor 2

    paged_cache = PagedKVCache(
        num_layers = num_layers,
        num_kv_heads = num_kv_heads,
        head_dim = head_dim,
        max_slots = max_slots,
        block_size = block_size,
        device = device,
        dtype = dtype,
    )
    paged_cache.record_prompt_length(input_ids.shape[1])

    scheduler = EvictionScheduler(
        cache = paged_cache,
        top_k = top_k,
        ema_alpha = 0.9,
    )

    # Install the paged attention hook on the model.
    install_paged_attention_hook(model, paged_cache)

    # We need to hook into batch_sample to run the eviction scheduler after
    # each block is committed. We do this by wrapping model.forward to
    # detect block boundaries and call scheduler.step().
    original_forward = model.forward
    last_committed = [0] # mutable closure variable

    def patched_forward(*args, update_past_key_values=False, **kwargs):
        result = original_forward(*args, update_past_key_values = update_past_key_values, **kwargs)
        if update_past_key_values:

            # A block boundary was just committed.
            n_now = paged_cache.num_committed_blocks
            if n_now > last_committed[0]:
                last_committed[0] = n_now
                scheduler.step()
        return result

    model.forward = patched_forward

    seq_len = torch.tensor([input_ids.shape[1]], device=device)
    min_len = seq_len.min().item()

    t0 = time.perf_counter()
    finished = model.batch_sample(
        input_ids = input_ids.clone(),
        tokenizer = tokenizer,
        block_size = block_size,
        max_new_tokens = max_new_tokens,
        small_block_size = block_size,
        min_len = min_len,
        seq_len = seq_len.clone(),
        threshold = threshold,
        use_block_cache = False,
    )
    elapsed = time.perf_counter() - t0

    # Restore original forward.
    model.forward = original_forward

    # Uninstall the paged attention hook.
    # (Re-installing on the next call recreates the hooks cleanly.)
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "_original_forward"):
            layer.self_attn.forward = layer.self_attn._original_forward

    output_ids = finished[0]
    prompt_len = input_ids.shape[1]
    new_tokens = output_ids[prompt_len:]
    decoded    = tokenizer.decode(new_tokens, skip_special_tokens=True)

    tps = new_tokens.shape[0] / elapsed if elapsed > 0 else 0.0
    return decoded, tps

# Main:
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark paged KV eviction kernel")
    p.add_argument("--model_path", default="Efficient-Large-Model/Fast_dLLM_v2_1.5B")
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--top_k", type=int, default=4,
                   help="Number of KV blocks to retain after eviction")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--threshold", type=float, default=0.9,
                   help="Fast-dLLM v2 confidence threshold for token unmasking")
    p.add_argument("--num_prompts", type=int, default=5,
                   help="How many GSM8K prompts to benchmark (max 10)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_json", default=None,
                   help="If set, write results to this JSON file")
    return p.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # Load model and tokenizer:
    print(f"Loading model: {args.model_path} …")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    model = setup_model_with_custom_generation(model)
    print("Model loaded.")

    # Extract model dimensions for PagedKVCache.
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(
        f"Architecture: {num_layers} layers, "
        f"{num_kv_heads} KV heads, head_dim={head_dim}"
    )

    # Run benchmarks:
    prompts = GSM8K_PROMPTS[: args.num_prompts]
    results = []

    print(
        f"\n{'='*70}\n"
        f"Benchmark: {len(prompts)} prompts | "
        f"block_size = {args.block_size} | top_k = {args.top_k} | "
        f"max_new_tokens = {args.max_new_tokens}\n"
        f"{'='*70}\n"
    )
    header = f"{'#':<4} {'Baseline TPS':>14} {'Paged TPS':>12} {'Speedup':>9} {'Match':>7}"
    print(header)
    print("-" * len(header))

    total_baseline_tps = 0.0
    total_paged_tps = 0.0
    num_matches = 0

    for i, prompt in enumerate(prompts):
        full_prompt = SYSTEM_PROMPT + prompt
        enc = tokenizer(full_prompt, return_tensors="pt").to(device)
        input_ids = enc.input_ids # [1, seq_len]

        # Warm up GPU on first iteration.
        if i == 0 and device.type == "cuda":
            with torch.no_grad():
                _ = model(input_ids[:, :4])
            torch.cuda.synchronize()

        # Baseline run.
        baseline_text, baseline_tps = run_baseline(
            model, tokenizer, input_ids,
            block_size = args.block_size,
            max_new_tokens = args.max_new_tokens,
            threshold = args.threshold,
        )

        # Paged run.
        paged_text, paged_tps = run_paged(
            model, tokenizer, input_ids,
            block_size = args.block_size,
            max_new_tokens = args.max_new_tokens,
            threshold = args.threshold,
            top_k = args.top_k,
            num_layers = num_layers,
            num_kv_heads = num_kv_heads,
            head_dim = head_dim,
            device = device,
            dtype = dtype,
        )

        baseline_ans = extract_answer(baseline_text)
        paged_ans = extract_answer(paged_text)
        match = (baseline_ans == paged_ans) and (baseline_ans is not None)
        speedup = paged_tps / baseline_tps if baseline_tps > 0 else 0.0

        total_baseline_tps += baseline_tps
        total_paged_tps += paged_tps
        num_matches += int(match)

        row = (
            f"{i:<4} {baseline_tps:>14.1f} {paged_tps:>12.1f} "
            f"{speedup:>9.2f}x {str(match):>7}"
        )
        print(row)

        results.append({
            "prompt_idx": i,
            "prompt": prompt[:80] + "…",
            "baseline_tps": baseline_tps,
            "paged_tps": paged_tps,
            "speedup": speedup,
            "baseline_ans": baseline_ans,
            "paged_ans": paged_ans,
            "match": match,
            "baseline_text": baseline_text[:300],
            "paged_text": paged_text[:300],
        })

    # Summary:
    n = len(prompts)
    avg_baseline_tps = total_baseline_tps / n
    avg_paged_tps = total_paged_tps / n
    avg_speedup = avg_paged_tps / avg_baseline_tps if avg_baseline_tps > 0 else 0.0
    agreement = num_matches / n * 100.0

    print(f"\n{'='*70}")
    print(f"{'Summary':}")
    print(f"  Avg baseline TPS : {avg_baseline_tps:.1f}")
    print(f"  Avg paged TPS : {avg_paged_tps:.1f}")
    print(f"  Avg speedup : {avg_speedup:.2f}x")
    print(f"  Answer agreement : {agreement:.0f}% ({num_matches}/{n})")
    print(f"  top_k kept : {args.top_k} blocks  "
          f"(= {args.top_k * args.block_size} tokens of context)")
    print(f"{'='*70}\n")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "summary": {
                        "avg_baseline_tps": avg_baseline_tps,
                        "avg_paged_tps": avg_paged_tps,
                        "avg_speedup": avg_speedup,
                        "agreement_pct": agreement,
                    },
                    "results": results,
                },
                f, indent=2,
            )
        print(f"Results written to {args.output_json}")


if __name__ == "__main__":
    main()