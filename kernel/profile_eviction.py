"""
Profiling harness for one K-value generation run.

Answers "where is the remaining gap between measured speedup and the theoretical eviction ceiling actually going?" with a real torch.profiler trace, instead of
guessing between the commit-step scoring cost and the S_q = 32 fallback path.

Run from KINETIC/: CUDA_VISIBLE_DEVICES=0 python -m kernel.profile_eviction --k 16 --n_questions 16

Interpreting the output: look at the printed table's top rows by self CUDA time. Any op named like "aten::scaled_dot_product_attention" attributable to score-capture
(paired with a matmul/softmax done in eager mode rather than fused) is the commit-step double-attention cost. Time in "aten::index" / "aten::cat" /
"aten::repeat_interleave" ahead of "aten::scaled_dot_product_attention" is the gather + cat + repeat overhead in the S_q = 32 fallback path. The exported chrome trace
(profile_trace.json) can be loaded at chrome://tracing or https://ui.perfetto.dev for a visual timeline if the text table isn't enough to localize the cost.
"""

import argparse

import torch
from torch.profiler import profile, ProfilerActivity, record_function

from kernel.modeling import Fast_dLLM_QwenForCausalLM
from kernel.generation_functions import setup_model
from kernel.block_eviction_scheduler import BlockEvictionScheduler
from kernel.test_eviction import load_model, load_gsm8k_batched, run_generation, BLOCK_SIZE, warm_up_model
from transformers import AutoTokenizer

def main():
    """
    Loads the model, warms up, then profiles one batch of generation with the scheduler installed at the given K.
    """
    parser = argparse.ArgumentParser(description = "Profile one K-value generation run")
    parser.add_argument("--model", default = "Efficient-Large-Model/Fast_dLLM_v2_7B")
    parser.add_argument("--device", default = "cuda:0")
    parser.add_argument("--k", type = int, default = 16)
    parser.add_argument("--n_questions", type = int, default = 16,
                        help = "Keep this small: profiling adds overhead per op, so a full 80-question sweep is unnecessarily slow to trace.")
    parser.add_argument("--batch_size", type = int, default = 16)
    parser.add_argument("--context_len", type = int, default = 2048)
    parser.add_argument("--union_mode", choices = ["matmul", "naive"], default = "matmul")
    parser.add_argument("--use_sparse_kernel", action = "store_true")
    parser.add_argument("--baseline", action = "store_true",
                        help = "Profile with no scheduler installed at all (dense, full-prefix attention). "
                               "Use this to directly measure attention's real cost without eviction, rather "
                               "than extrapolating it from a K-value profile.")
    parser.add_argument("--max_new_tokens", type = int, default = 128,
                        help = "Keep this small: we only need enough blocks to see steady-state commit + small-step cost.")
    parser.add_argument("--trace_out", default = "profile_trace.json")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = load_model(args.model, device = args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code = True)

    print("Warming up...")
    warm_up_model(model, tokenizer, args.device)

    batches = load_gsm8k_batched(args.n_questions, tokenizer, args.device, args.batch_size, args.context_len)
    input_ids = batches[0][0].input_ids

    scheduler = None
    if not args.baseline:
        scheduler = BlockEvictionScheduler(
            model, k = args.k, block_size = BLOCK_SIZE, union_mode = args.union_mode,
            use_sparse_kernel = args.use_sparse_kernel,
        )
        scheduler.install()

    label = "baseline (no eviction, dense attention)" if args.baseline else f"K = {args.k}, use_sparse_kernel = {args.use_sparse_kernel}"
    print(f"Profiling one batch ({label})...")
    try:
        with profile(
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes = True,
            with_stack = False,
        ) as prof:
            with record_function("full_generation"):
                run_generation(model, tokenizer, input_ids, max_new_tokens = args.max_new_tokens)
    finally:
        if scheduler is not None:
            scheduler.uninstall()

    print("\nTop ops by self CUDA time:")
    print(prof.key_averages().table(sort_by = "self_cuda_time_total", row_limit = 25))

    # commit_step / small_step are record_function wrapper scopes (added in BlockEvictionScheduler._patched_forward), not real ops: almost all of their own
    # time is spent inside child ops, so they show near-zero "self" time and won't surface in the self-time table above no matter how expensive they are. Their
    # cuda_time_total (which does include descendants) is the number that actually answers "which of these two dominates": printed here explicitly since the table
    # above can't show it.
    print("\nCommit-step vs small-step breakdown (total CUDA time, including all nested ops):")
    tag_totals = {}
    for evt in prof.key_averages():
        if evt.key in ("commit_step", "small_step"):
            # Attribute name varies by torch version: newer versions generalized
            # cuda_time_total to device_time_total as part of supporting non-CUDA
            # accelerators. Try both rather than assume.
            time_us = getattr(evt, "device_time_total", None)
            if time_us is None:
                time_us = getattr(evt, "cuda_time_total", None)
            if time_us is None:
                # Last-resort fallback, not ideal but keeps this from crashing.
                time_us = evt.cpu_time_total
            tag_totals[evt.key] = (time_us / 1000.0, evt.count) # us -> ms.
    if not tag_totals:
        print("(No tagged commit_step/small_step events found: is this an up-to-date "
              "block_eviction_scheduler.py with the record_function tagging?)")
    else:
        total_ms = sum(v[0] for v in tag_totals.values())
        for label in ("commit_step", "small_step"):
            if label in tag_totals:
                ms, count = tag_totals[label]
                share = (ms / total_ms * 100) if total_ms > 0 else 0.0
                print(f"  {label:<12} {ms:>10.1f} ms   {share:>5.1f}%   ({count} calls, {ms / count:.3f} ms/call)")

    prof.export_chrome_trace(args.trace_out)
    print(f"\nChrome trace written to {args.trace_out}: load at chrome://tracing or https://ui.perfetto.dev for a visual timeline.")

if __name__ == "__main__":
    main()