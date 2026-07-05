"""
Microbenchmark: sparse Triton kernel path vs. gather+cat+SDPA fallback path,
in isolation from the model, dataset, and generation loop.

Job: answer "why does the sparse kernel fire 99.9% of the time but the run
gets slower?" with a direct, controlled timing comparison of the two attention
paths at the real call shapes seen during generation -- without GSM8K, without
loading the 7B checkpoint, without any batching/dataset noise in the way.

Run from DiffusionLLMs/ with the kernel/ package on the path:
    CUDA_VISIBLE_DEVICES = 0 python -m kernel.benchmark_sparse_kernel

Interpreting the output:
    - If the sparse path's per-call latency is higher than the fallback's at
      S_q = 8 (the dominant call shape), that directly confirms the kernel's
      fixed overhead (launch cost + wasted 32-wide tile for an 8-wide query)
      outweighs its HBM savings at this problem size -- matching the K = 16
      GSM8K result where hit_rate = 99.9% but throughput dropped.
    - If S_q = 32 (block-rebuild calls) shows the sparse path winning while
      S_q = 8 shows it losing, that pinpoints the BLOCK_M = 32 tile-sizing waste
      specifically, since that call shape uses the full tile with no waste.
"""

import time
import torch

from kernel.block_sparse_attn import block_sparse_attention_with_stats, sparse_attn_merge

# Real shapes from the Fast-dLLM v2 7B config and the harness's own settings.
B = 32 # batch_size used in test_eviction.py
H_Q = 28
H_KV = 4
NUM_KV_GROUPS = H_Q // H_KV
D = 128
BLOCK_SIZE = 32
N_PREFIX_BLOCKS = 64 # matches context_len=2048 / block_size=32 seen in the transcript
K_SELECTED = 16 # the K value being evaluated
N_WARMUP = 10
N_ITERS = 200

device = "cuda"
dtype = torch.bfloat16

def make_inputs(s_q):
    """
    Builds random tensors matching one attention forward call at the given query width.
    """
    query_states = torch.randn(B, H_Q, s_q, D, device = device, dtype = dtype)
    prefix_k = torch.randn(B, H_KV, N_PREFIX_BLOCKS * BLOCK_SIZE, D, device = device, dtype = dtype)
    prefix_v = torch.randn(B, H_KV, N_PREFIX_BLOCKS * BLOCK_SIZE, D, device = device, dtype = dtype)
    cur_k = torch.randn(B, H_KV, BLOCK_SIZE, D, device = device, dtype = dtype)
    cur_v = torch.randn(B, H_KV, BLOCK_SIZE, D, device = device, dtype = dtype)
    sel = sorted(torch.randperm(N_PREFIX_BLOCKS)[:K_SELECTED].tolist())
    block_indices = torch.tensor([sel] * B, dtype = torch.int32, device = device)
    return query_states, prefix_k, prefix_v, cur_k, cur_v, block_indices


def time_sparse_path(s_q, n_iters = N_ITERS):
    """
    Times block_sparse_attention_with_stats + sparse_attn_merge (the Triton path).
    """
    q, prefix_k, prefix_v, cur_k, cur_v, block_indices = make_inputs(s_q)
    scaling = D ** -0.5

    for _ in range(N_WARMUP):
        sparse_attn_merge(
            query_states = q, prefix_k = prefix_k, prefix_v = prefix_v,
            block_indices = block_indices, cur_k = cur_k[:, :, :s_q, :], cur_v = cur_v[:, :, :s_q, :],
            attention_mask = None, scaling = scaling, num_kv_groups = NUM_KV_GROUPS,
            block_size = BLOCK_SIZE,
        )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        sparse_attn_merge(
            query_states = q, prefix_k = prefix_k, prefix_v = prefix_v,
            block_indices = block_indices, cur_k = cur_k[:, :, :s_q, :], cur_v = cur_v[:, :, :s_q, :],
            attention_mask = None, scaling = scaling, num_kv_groups = NUM_KV_GROUPS,
            block_size = BLOCK_SIZE,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / n_iters

def time_fallback_path(s_q, n_iters = N_ITERS):
    """
    Times the gather+cat+SDPA fallback: gather K selected blocks, cat with current block, SDPA.
    """
    q, prefix_k, prefix_v, cur_k, cur_v, block_indices = make_inputs(s_q)
    scaling = D ** -0.5
    sel = block_indices[0].tolist()
    tok_idx = torch.tensor(
        [t for blk in sel for t in range(blk * BLOCK_SIZE, (blk + 1) * BLOCK_SIZE)],
        device = device, dtype = torch.long,
    )

    def run_once():
        k_sel = prefix_k[:, :, tok_idx, :]
        v_sel = prefix_v[:, :, tok_idx, :]
        k_full = torch.cat([k_sel, cur_k[:, :, :s_q, :]], dim = 2)
        v_full = torch.cat([v_sel, cur_v[:, :, :s_q, :]], dim = 2)
        k_exp = k_full.repeat_interleave(NUM_KV_GROUPS, dim = 1)
        v_exp = v_full.repeat_interleave(NUM_KV_GROUPS, dim = 1)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k_exp, v_exp, attn_mask = None, scale = scaling,
        )

    for _ in range(N_WARMUP):
        run_once()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        run_once()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / n_iters

def main():
    """
    Runs both paths at both dominant call shapes (S_q = 8 reuse, S_q = 32 rebuild) and prints a comparison.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    print(f"Config: B={B} H_q={H_Q} H_kv={H_KV} D={D} block_size={BLOCK_SIZE} "
          f"n_prefix_blocks={N_PREFIX_BLOCKS} K={K_SELECTED} dtype={dtype}\n")

    for s_q, label in [(8, "S_q=8  (sub-block reuse -- dominant call shape)"),
                        (32, "S_q=32 (block-cache rebuild)")]:
        sparse_ms = time_sparse_path(s_q) * 1000
        fallback_ms = time_fallback_path(s_q) * 1000
        winner = "sparse kernel" if sparse_ms < fallback_ms else "SDPA fallback"
        ratio = fallback_ms / sparse_ms if sparse_ms > 0 else float("inf")

        print(f"{label}")
        print(f"  sparse kernel path:   {sparse_ms:.4f} ms/call")
        print(f"  SDPA fallback path:   {fallback_ms:.4f} ms/call")
        print(f"  faster path: {winner}  (fallback/sparse ratio = {ratio:.2f}x)\n")

if __name__ == "__main__":
    main()