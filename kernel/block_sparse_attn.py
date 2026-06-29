"""
Block-Sparse Flash Attention Kernel.

Implements Flash Attention 2 with a block-sparse KV mask: instead of iterating
over all N prefix blocks in the outer KV loop, we iterate only over the K selected
blocks. Evicted blocks are never loaded from HBM.

This is the primary throughput lever. The speedup over dense attention is
approximately:

    speedup ≈ N_prefix / (2 × K_selected + N_current_block)

At K=16, N=64, block_size=32 (the settings from the slides showing 1.35x with
naive union), the theoretical ceiling with a perfect sparse kernel is:

    64 / (2 × 16 + 1) ≈ 1.94x

The naive union was expanding K_effective to ~25 blocks (25.6% KV loaded).
The matmul union keeps it at exactly K=16 blocks, pushing toward the ceiling.

Fast-dLLM v2 specifics:
    - Q shape (small step): [B, H_q=28, block_size=32, D=128]
    - KV prefix shape: [B, H_kv=4, K*block_size, D=128]  (after eviction selection)
    - KV current shape: [B, H_kv=4, block_size=32, D=128]
    - GQA: each KV head serves H_q/H_kv = 7 query heads
    - No causal mask within the current block (Fast-dLLM uses bidirectional
      attention within the generation block, not causal)
    - The prefix is attended to causally at the block level

Kernel design (following Flash Attention 2 §3.1):
    Outer loop: selected KV blocks (K iterations, not N)
    Inner tiling: BLOCK_M query tokens × BLOCK_N KV tokens per SRAM tile

    Each SM handles one (batch, query_head) pair. The KV loop loads one tile of
    K and V from HBM per selected block, accumulates the online softmax running
    statistics (m, l) and the output accumulator O in registers, then writes O
    back to HBM once after all K blocks are processed.

    The block_indices tensor [K] tells the kernel which KV blocks to load, in
    sorted order. Within each selected block, the kernel loads block_size=32 KV
    vectors, so each KV SRAM tile covers exactly one generation block.

Memory traffic analysis:
    Dense:    reads N × block_size × D × 2 (K+V) × H_kv bytes per forward pass
    Sparse:   reads K × block_size × D × 2 (K+V) × H_kv bytes per forward pass
    Ratio:    K/N — matches the "KV loaded" column in the benchmark slides

GQA handling:
    The query head index h_q maps to KV head h_kv = h_q // num_kv_groups.
    We load K[h_kv] and V[h_kv] but Q[h_q], so there is no expand/repeat needed.
"""

import torch
import triton
import triton.language as tl

# Tile sizes. BLOCK_M covers the query sequence dimension (32 tokens per
# generation block, so BLOCK_M=32 processes the entire Q in one tile).
# BLOCK_N covers one KV block at a time (block_size=32).
BLOCK_M: int = 32
BLOCK_N: int = 32

@triton.jit
def _block_sparse_attn_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    block_indices_ptr,
    n_kv_blocks,
    scale,
    B,
    H_q,
    H_kv,
    S_q: tl.constexpr,
    S_kv,
    D: tl.constexpr,
    num_kv_groups: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_ib,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Forward kernel body. One program per (batch, query_head) pair.

    Iterates over only the K selected KV blocks (loaded from block_indices),
    accumulating a Flash Attention 2 online softmax in registers, and writes
    the final output once all blocks are processed.

    Args:
    - Q_ptr: [B, H_q, S_q, D] query tensor.
    - K_ptr: [B, H_kv, S_kv_full, D] full prefix key tensor (we index into it
      using block_indices to load only selected blocks).
    - V_ptr: [B, H_kv, S_kv_full, D] full prefix value tensor.
    - Out_ptr: [B, H_q, S_q, D] output tensor (pre-zeroed).
    - block_indices_ptr: [B, n_kv_blocks] int32 tensor of selected block indices.
    - n_kv_blocks: number of selected KV blocks (K, not N).
    - scale: softmax scaling factor (1/sqrt(D)).
    - B, H_q, H_kv: batch, query heads, KV heads.
    - S_q: query sequence length (constexpr = block_size = 32).
    - S_kv: full prefix sequence length (N*block_size).
    - D: head dimension (constexpr = 128).
    - num_kv_groups: H_q // H_kv (constexpr = 7 for Qwen2.5-7B).
    - stride_*: tensor strides.
    - BLOCK_M, BLOCK_N: tile sizes (constexpr, both = 32).
    """
    b = tl.program_id(0)
    h_q = tl.program_id(1)
    h_kv = h_q // num_kv_groups

    # Query tile: load all S_q=32 query tokens for this head.
    q_offset = b * stride_qb + h_q * stride_qh
    q_offs_m = tl.arange(0, BLOCK_M)
    q_offs_d = tl.arange(0, D)
    Q_ptrs = Q_ptr + q_offset + q_offs_m[:, None] * stride_qs + q_offs_d[None, :] * stride_qd
    Q = tl.load(Q_ptrs, mask = q_offs_m[:, None] < S_q, other = 0.0).to(tl.float32)
    Q = Q * scale

    # Online softmax running state.
    m_i = tl.full([BLOCK_M], float("-inf"), dtype = tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype = tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype = tl.float32)

    # Base pointers for K and V for this (b, h_kv).
    k_base = K_ptr + b * stride_kb + h_kv * stride_kh
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh
    idx_base = block_indices_ptr + b * stride_ib

    kv_offs_d = tl.arange(0, D)
    kv_offs_n = tl.arange(0, BLOCK_N)

    # Outer loop: iterate over selected KV blocks only.
    for blk_i in range(n_kv_blocks):
        # Load the block index and compute the token start position.
        blk_idx = tl.load(idx_base + blk_i)
        kv_start = blk_idx * BLOCK_N

        # Load K tile: [BLOCK_N, D]
        K_ptrs = k_base + (kv_start + kv_offs_n[:, None]) * stride_ks + kv_offs_d[None, :] * stride_kd
        kv_mask = (kv_start + kv_offs_n) < S_kv
        K_tile = tl.load(K_ptrs, mask = kv_mask[:, None], other = 0.0).to(tl.float32)

        # Attention scores: [BLOCK_M, BLOCK_N] = Q [BLOCK_M, D] @ K^T [D, BLOCK_N]
        scores = tl.dot(Q, tl.trans(K_tile))

        # Mask out-of-bounds KV positions with -inf.
        scores = tl.where(kv_mask[None, :], scores, float("-inf"))

        # Online softmax update (Flash Attention 2 algorithm):
        m_new = tl.maximum(m_i, tl.max(scores, axis = 1))
        exp_scores = tl.exp(scores - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_scores, axis = 1)

        # Load V tile: [BLOCK_N, D]
        V_ptrs = v_base + (kv_start + kv_offs_n[:, None]) * stride_vs + kv_offs_d[None, :] * stride_vd
        V_tile = tl.load(V_ptrs, mask = kv_mask[:, None], other = 0.0).to(tl.float32)

        # Rescale running accumulator and add new contribution.
        acc = acc * (tl.exp(m_i - m_new) / l_new)[:, None] + tl.dot(exp_scores / l_new[:, None], V_tile)

        m_i = m_new
        l_i = l_new

    # Normalize the accumulator (already normalized online above, but
    # l_i encodes the final softmax denominator so we don't divide again).
    # Write output: [BLOCK_M, D]
    out_offset = b * stride_ob + h_q * stride_oh
    Out_ptrs = Out_ptr + out_offset + q_offs_m[:, None] * stride_os + q_offs_d[None, :] * stride_od
    tl.store(Out_ptrs, acc.to(tl.float16), mask = q_offs_m[:, None] < S_q)

def block_sparse_attention(
    Q: torch.Tensor,
    K_prefix: torch.Tensor,
    V_prefix: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Compute attention over a sparse subset of prefix KV blocks.

    Q attends to only the blocks listed in block_indices, never loading the
    evicted blocks from HBM. This is the primary throughput lever: if K blocks
    are selected out of N total, HBM reads for K and V are reduced by factor N/K.

    Args:
    - Q: [B, H_q, S_q, D] query tensor for the current generation block.
      S_q = block_size = 32 (one generation block at a time).
    - K_prefix: [B, H_kv, S_kv, D] full prefix key tensor (not pre-sliced).
    - V_prefix: [B, H_kv, S_kv, D] full prefix value tensor.
    - block_indices: [B, K] int32 tensor of selected block indices, sorted.
      Each value is a block index into the prefix (0-indexed, 0 = first 32 tokens).
    - block_size: tokens per block (must match generation block size, default 32).

    Returns [B, H_q, S_q, D] float16 attention output.

    Notes:
    - Q, K, V should be on the same CUDA device.
    - K_prefix and V_prefix are the *full* cache tensors. The kernel uses
      block_indices to load only the selected tiles — no Python-level gather needed.
    - The output does not include attention over the current generation block
      itself (handled separately by the model's standard attention).
    """
    B, H_q, S_q, D = Q.shape
    _, H_kv, S_kv, _ = K_prefix.shape
    K_selected = block_indices.shape[1]
    num_kv_groups = H_q // H_kv
    scale = D ** -0.5

    assert S_q <= BLOCK_M, (
        f"block_sparse_attention: S_q={S_q} > BLOCK_M={BLOCK_M}. "
        "Increase BLOCK_M or process in sub-blocks."
    )
    assert block_size == BLOCK_N, (
        f"block_sparse_attention: block_size={block_size} != BLOCK_N={BLOCK_N}. "
        "Recompile the kernel with matching BLOCK_N."
    )
    assert D in (64, 128, 256), f"block_sparse_attention: unexpected head_dim {D}"
    assert block_indices.dtype == torch.int32, "block_indices must be int32"

    # Output buffer.
    Out = torch.empty(B, H_q, S_q, D, device = Q.device, dtype = torch.float16)

    # Ensure contiguous float16 inputs (Triton needs contiguous memory).
    Q_c = Q.contiguous().to(torch.float16)
    K_c = K_prefix.contiguous().to(torch.float16)
    V_c = V_prefix.contiguous().to(torch.float16)
    idx_c = block_indices.contiguous()

    grid = (B, H_q)

    _block_sparse_attn_fwd_kernel[grid](
        Q_c, K_c, V_c, Out,
        idx_c,
        K_selected,
        scale,
        B, H_q, H_kv,
        S_q, S_kv, D,
        num_kv_groups,
        Q_c.stride(0), Q_c.stride(1), Q_c.stride(2), Q_c.stride(3),
        K_c.stride(0), K_c.stride(1), K_c.stride(2), K_c.stride(3),
        V_c.stride(0), V_c.stride(1), V_c.stride(2), V_c.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        idx_c.stride(0),
        BLOCK_M = BLOCK_M,
        BLOCK_N = BLOCK_N,
    )

    return Out