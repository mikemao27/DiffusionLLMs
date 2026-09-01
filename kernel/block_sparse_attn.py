"""
Block-Sparse Flash Attention Kernel.

Computes attention over only the K selected prefix KV blocks, without ever loading the evicted blocks from HBM. Implements Flash Attention 2 with a 
block-sparse KV mask: instead of iterating over all N prefix blocks in the outer KV loop, we iterate only over the K selected blocks. Evicted blocks 
are never loaded from HBM. This is the primary throughput lever. The speedup over dense attention is approximately: speedup ~= N_prefix / 
(2 * K_selected + N_current_block)

At K = 16, N = 64, block_size = 32 (the settings from the slides showing 1.35x with naive union), the theoretical ceiling with a perfect sparse kernel 
is: 64 / (2 * 16 + 1) ~= 1.94x

The naive union was expanding K_effective to ~25 blocks (25.6% KV loaded). The matmul union keeps it at exactly K = 16 blocks, pushing toward the 
ceiling.

Fast-dLLM v2 specifics: the Q shape (small step) is [B, H_q = 28, block_size = 32, D = 128], the KV prefix shape is [B, H_kv = 4, K * block_size, 
D = 128] (after eviction selection), and the KV current shape is [B, H_kv = 4, block_size = 32, D = 128]. This is GQA: each KV head serves H_q / 
H_kv = 7 query heads. There is no causal mask within the current block (Fast-dLLM uses bidirectional attention within the generation block, not causal); 
the prefix is attended to causally at the block level.

Kernel design follows Flash Attention 2 section 3.1: the outer loop runs over selected KV blocks (K iterations, not N), with inner tiling of BLOCK_M 
query tokens by BLOCK_N KV tokens per SRAM tile. Each SM handles one (batch, query_head) pair. The KV loop loads one tile of K and V from HBM per 
selected block, accumulates the online softmax running statistics (m, l) and the output accumulator O in registers, then writes O back to HBM once 
after all K blocks are processed. The block_indices tensor [K] tells the kernel which KV blocks to load, in sorted order. Within each selected block, 
the kernel loads block_size = 32 KV vectors, so each KV SRAM tile covers exactly one generation block.

Memory traffic analysis: dense reads N * block_size * D * 2 (K + V) * H_kv bytes per forward pass, sparse reads K * block_size * D * 2 (K + V) * H_kv 
bytes per forward pass, and the ratio K/N matches the "KV loaded" column in the benchmark slides.

GQA handling: the query head index h_q maps to KV head h_kv = h_q // num_kv_groups. We load K[h_kv] and V[h_kv] but Q[h_q], so there is no 
expand/repeat needed.

Interface contract (never changes): block_sparse_attention() and block_sparse_attention_with_stats() take the full prefix K/V cache tensors plus a list
of selected block indices, never a pre-gathered subset. All HBM-traffic savings come from the kernel indexing into the full tensor itself, not from a 
Python-level slice before the call. block_sparse_attention_with_stats() and sparse_attn_merge() exist purely to support merging this partial 
(prefix-only) attention result with a second partial result (the current generation block) via the Flash Decoding combine. Callers who only need prefix 
attention in isolation should use block_sparse_attention() instead.
"""

import torch
import triton
import triton.language as tl

# Tile sizes. BLOCK_M covers the query sequence dimension (32 tokens per generation block, so BLOCK_M = 32 processes the entire Q in one tile). 
# BLOCK_N covers one KV block at a time (block_size = 32).
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
    Forward kernel body. One program per (batch, query_head) pair. Iterates over only the K selected KV blocks (loaded from block_indices), accumulating 
    a Flash Attention 2 online softmax in registers, and writes the final output once all blocks are processed.

    Q_ptr is the [B, H_q, S_q, D] query tensor. K_ptr and V_ptr are the [B, H_kv, S_kv_full, D] full prefix key and value tensors (indexed via 
    block_indices to load only selected blocks). Out_ptr is the [B, H_q, S_q, D] output tensor (pre-zeroed). block_indices_ptr is a [B, n_kv_blocks] 
    int32 tensor of selected block indices, and n_kv_blocks is the number of selected KV blocks (K, not N). scale is the softmax scaling factor 
    (1 / sqrt(D)). B, H_q, H_kv are the batch, query head, and KV head counts. S_q is the query sequence length (constexpr = block_size = 32) and S_kv 
    is the full prefix sequence length (N * block_size). D is the head dimension (constexpr = 128), and num_kv_groups is H_q // H_kv (constexpr = 7 for 
    Qwen2.5-7B). The stride_* arguments are tensor strides, and BLOCK_M / BLOCK_N are the tile sizes (constexpr, both = 32). Writes directly into 
    Out_ptr and returns nothing.
    """
    b = tl.program_id(0)
    h_q = tl.program_id(1)
    h_kv = h_q // num_kv_groups

    # Query tile: load all S_q = 32 query tokens for this head.
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

    # l_i already encodes the final softmax denominator from the online updates above, so acc is already normalized: no final division needed. Write
    # output: [BLOCK_M, D]
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
    Compute attention over a sparse subset of prefix KV blocks. Q attends to only the blocks listed in block_indices, never loading the evicted blocks 
    from HBM. This is the primary throughput lever: if K blocks are selected out of N total, HBM reads for K and V are reduced by factor N/K.

    Q is the [B, H_q, S_q, D] query tensor for the current generation block, where S_q = block_size = 32 (one generation block at a time). K_prefix and 
    V_prefix are the [B, H_kv, S_kv, D] full prefix key and value tensors (not pre-sliced). block_indices is a [B, K] int32 tensor of selected block 
    indices, sorted, where each value is a block index into the prefix (0-indexed, 0 = first 32 tokens). block_size is the tokens per block, which must 
    match the generation block size (default 32).

    Returns a [B, H_q, S_q, D] float16 attention output.

    Q, K, and V should be on the same CUDA device. K_prefix and V_prefix are the full cache tensors: the kernel uses block_indices to load only the 
    selected tiles, so no Python-level gather is needed. The output does not include attention over the current generation block itself (handled 
    separately by the model's standard attention).
    """
    B, H_q, S_q, D = Q.shape
    _, H_kv, S_kv, _ = K_prefix.shape
    K_selected = block_indices.shape[1]
    num_kv_groups = H_q // H_kv
    scale = D ** -0.5

    assert S_q <= BLOCK_M, (
        f"block_sparse_attention: S_q = {S_q} > BLOCK_M = {BLOCK_M}. "
        "Increase BLOCK_M or process in sub-blocks."
    )
    assert block_size == BLOCK_N, (
        f"block_sparse_attention: block_size = {block_size} != BLOCK_N = {BLOCK_N}. "
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

@triton.jit
def _block_sparse_attn_stats_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Out_ptr,
    Acc_ptr,
    M_ptr,
    L_ptr,
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
    Same as _block_sparse_attn_fwd_kernel, but additionally writes the raw (un-normalized) accumulator and the online softmax statistics (m, l), so a 
    caller can correctly merge this partial attention result with another partial result (e.g. the current generation block's self-attention) using the 
    Flash Decoding log-sum-exp combine.

    A separate raw accumulator is required because the online softmax update at each step rescales acc by 1/l_new before adding the new contribution, 
    so acc is already the normalized output by the time the loop ends (Out = acc). If a caller took that normalized Out and multiplied it by l again 
    during a merge, the l term would be double-counted, corrupting the result. Acc_ptr instead stores acc * l_i, which recovers the un-normalized weighted 
    sum of values, exactly what the merge needs.

    Q_ptr, K_ptr, and V_ptr are as in _block_sparse_attn_fwd_kernel. Out_ptr is the [B, H_q, S_q, D] normalized output (same as the plain kernel). 
    Acc_ptr is the [B, H_q, S_q, D] raw un-normalized accumulator (acc * l_i). M_ptr is the [B, H_q, S_q] float32 final running max logit per query, 
    and L_ptr is the [B, H_q, S_q] float32 final running softmax denominator per query. block_indices_ptr, n_kv_blocks, scale, B, H_q, H_kv, S_q, S_kv, 
    D, num_kv_groups, stride_*, BLOCK_M, and BLOCK_N are as in the plain kernel. Writes directly into Out_ptr, Acc_ptr, M_ptr, and L_ptr, and returns 
    nothing.
    """
    b = tl.program_id(0)
    h_q = tl.program_id(1)
    h_kv = h_q // num_kv_groups

    q_offset = b * stride_qb + h_q * stride_qh
    q_offs_m = tl.arange(0, BLOCK_M)
    q_offs_d = tl.arange(0, D)
    Q_ptrs = Q_ptr + q_offset + q_offs_m[:, None] * stride_qs + q_offs_d[None, :] * stride_qd
    Q = tl.load(Q_ptrs, mask = q_offs_m[:, None] < S_q, other = 0.0).to(tl.float32)
    Q = Q * scale

    m_i = tl.full([BLOCK_M], float("-inf"), dtype = tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype = tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype = tl.float32) # Normalized (for Out).
    raw_acc = tl.zeros([BLOCK_M, D], dtype = tl.float32) # Un-normalized (for Acc).

    k_base = K_ptr + b * stride_kb + h_kv * stride_kh
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh
    idx_base = block_indices_ptr + b * stride_ib

    kv_offs_d = tl.arange(0, D)
    kv_offs_n = tl.arange(0, BLOCK_N)

    for blk_i in range(n_kv_blocks):
        blk_idx = tl.load(idx_base + blk_i)
        kv_start = blk_idx * BLOCK_N

        K_ptrs = k_base + (kv_start + kv_offs_n[:, None]) * stride_ks + kv_offs_d[None, :] * stride_kd
        kv_mask = (kv_start + kv_offs_n) < S_kv
        K_tile = tl.load(K_ptrs, mask = kv_mask[:, None], other = 0.0).to(tl.float32)

        scores = tl.dot(Q, tl.trans(K_tile))
        scores = tl.where(kv_mask[None, :], scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis = 1))
        exp_scores = tl.exp(scores - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_scores, axis = 1)

        V_ptrs = v_base + (kv_start + kv_offs_n[:, None]) * stride_vs + kv_offs_d[None, :] * stride_vd
        V_tile = tl.load(V_ptrs, mask = kv_mask[:, None], other = 0.0).to(tl.float32)

        # Normalized accumulator (for Out): divided by running l at each step.
        acc = acc * (tl.exp(m_i - m_new) / l_new)[:, None] + tl.dot(exp_scores / l_new[:, None], V_tile)

        # Un-normalized accumulator (for Acc): rescale previous by exp(m_old-m_new), add new contribution WITHOUT dividing by l. This gives 
        # sum(exp(s - m) * V).
        raw_acc = raw_acc * tl.exp(m_i - m_new)[:, None] + tl.dot(exp_scores, V_tile)

        m_i = m_new
        l_i = l_new

    # Out: normalized output (matches the plain kernel exactly).
    out_offset = b * stride_ob + h_q * stride_oh
    Out_ptrs = Out_ptr + out_offset + q_offs_m[:, None] * stride_os + q_offs_d[None, :] * stride_od
    tl.store(Out_ptrs, acc.to(tl.float16), mask = q_offs_m[:, None] < S_q)

    # Acc: raw un-normalized accumulator = sum(exp(score - m_final) * V). Maintained separately as raw_acc throughout the loop: never divided by l.
    Acc_ptrs = Acc_ptr + out_offset + q_offs_m[:, None] * stride_os + q_offs_d[None, :] * stride_od
    tl.store(Acc_ptrs, raw_acc, mask = q_offs_m[:, None] < S_q)

    # M, L: final running statistics per query, needed for the merge rescale.
    ml_offset = b * (H_q * S_q) + h_q * S_q
    ml_offs = tl.arange(0, BLOCK_M)
    tl.store(M_ptr + ml_offset + ml_offs, m_i, mask = ml_offs < S_q)
    tl.store(L_ptr + ml_offset + ml_offs, l_i, mask = ml_offs < S_q)

def block_sparse_attention_with_stats(
    Q: torch.Tensor,
    K_prefix: torch.Tensor,
    V_prefix: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Same as block_sparse_attention, but also returns the raw accumulator and softmax statistics needed to merge this partial result with another partial 
    attention computation (Flash Decoding combine). Takes the same Q, K_prefix, V_prefix, block_indices, and block_size arguments as 
    block_sparse_attention.

    Returns Out, the [B, H_q, S_q, D] float16 normalized attention output over the K selected prefix blocks only; Acc, the [B, H_q, S_q, D] float32 raw 
    (un-normalized) weighted sum of values; M, the [B, H_q, S_q] float32 running max logit; and L, the [B, H_q, S_q] float32 running softmax denominator.
    """
    B, H_q, S_q, D = Q.shape
    _, H_kv, S_kv, _ = K_prefix.shape
    K_selected = block_indices.shape[1]
    num_kv_groups = H_q // H_kv
    scale = D ** -0.5

    assert S_q <= BLOCK_M, (
        f"block_sparse_attention_with_stats: S_q = {S_q} > BLOCK_M = {BLOCK_M}."
    )
    assert block_size == BLOCK_N, (
        f"block_sparse_attention_with_stats: block_size = {block_size} != BLOCK_N = {BLOCK_N}."
    )
    assert D in (64, 128, 256), f"block_sparse_attention_with_stats: unexpected head_dim {D}"
    assert block_indices.dtype == torch.int32, "block_indices must be int32"

    Out = torch.empty(B, H_q, S_q, D, device = Q.device, dtype = torch.float16)
    Acc = torch.empty(B, H_q, S_q, D, device = Q.device, dtype = torch.float32)
    M = torch.empty(B, H_q, S_q, device = Q.device, dtype = torch.float32)
    L = torch.empty(B, H_q, S_q, device = Q.device, dtype = torch.float32)

    Q_c = Q.contiguous().to(torch.float16)
    K_c = K_prefix.contiguous().to(torch.float16)
    V_c = V_prefix.contiguous().to(torch.float16)
    idx_c = block_indices.contiguous()

    grid = (B, H_q)

    _block_sparse_attn_stats_kernel[grid](
        Q_c, K_c, V_c, Out, Acc, M, L,
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

    return Out, Acc, M, L

@triton.jit
def _block_sparse_attn_fused_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    CurK_ptr,
    CurV_ptr,
    Mask_ptr,
    Out_ptr,
    block_indices_ptr,
    n_kv_blocks,
    scale,
    B,
    H_q,
    H_kv,
    S_q: tl.constexpr,
    S_kv,
    S_cur: tl.constexpr,
    D: tl.constexpr,
    num_kv_groups: tl.constexpr,
    HAS_MASK: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ckb, stride_ckh, stride_cks, stride_ckd,
    stride_cvb, stride_cvh, stride_cvs, stride_cvd,
    stride_mb, stride_mh, stride_ms, stride_mn,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_ib,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused variant of _block_sparse_attn_stats_kernel that also folds in the current generation block's (dense) attention, in the same kernel launch and 
    the same online-softmax accumulator, instead of leaving that to a second PyTorch pass plus a Python-level Flash Decoding merge (see 
    sparse_attn_merge below).

    This mirrors the collaborator's block_sparse_attention_kernel fusion strategy (gather + scatter + attention in one launch) applied on top of KINETIC's
    existing top-K prefix block selection, rather than replacing it: the prefix loop below is unchanged from _block_sparse_attn_stats_kernel (only the 
    top-K selected blocks are read from HBM), and the current block's KV (always exactly block_size = BLOCK_N tokens, dense, no eviction) is folded in 
    as one more online-softmax tile immediately after, using the same running (m_i, l_i, raw_acc) state. There is no separate Acc/M/L output and no 
    merge step: this kernel writes the final normalized output directly. Unlike _block_sparse_attn_fwd_kernel's single-purpose "acc" (only valid for 
    exactly one tile: renormalizing at every step silently corrupts the result from the second tile onward, verified numerically), this kernel tracks 
    raw_acc unnormalized through every tile of both passes and divides by l_i exactly once at the end, matching the correct pattern 
    _block_sparse_attn_stats_kernel already uses for its own raw_acc / Acc output.

    Q_ptr, K_ptr, V_ptr, block_indices_ptr, n_kv_blocks, scale, B, H_q, H_kv, S_q, S_kv, D, num_kv_groups, and the stride_q/k/v/o/i arguments are 
    exactly as in _block_sparse_attn_stats_kernel. CurK_ptr and CurV_ptr are the [B, H_kv, S_cur, D] current-block key/value tensors (S_cur is a 
    constexpr, always block_size). Mask_ptr is an optional [B, H_q, S_q, S_cur] additive attention mask for the current-block columns only (pass a 
    dummy 1-element tensor and HAS_MASK = False when there is no mask, e.g. the use_block_cache cache-hit path). HAS_MASK is a constexpr flag gating 
    whether Mask_ptr is read at all. Out_ptr is the [B, H_q, S_q, D] float16 output, matching query_states' layout after the caller's own transpose.
    """
    b = tl.program_id(0)
    h_q = tl.program_id(1)
    h_kv = h_q // num_kv_groups

    q_offset = b * stride_qb + h_q * stride_qh
    q_offs_m = tl.arange(0, BLOCK_M)
    q_offs_d = tl.arange(0, D)
    Q_ptrs = Q_ptr + q_offset + q_offs_m[:, None] * stride_qs + q_offs_d[None, :] * stride_qd
    Q = tl.load(Q_ptrs, mask = q_offs_m[:, None] < S_q, other = 0.0).to(tl.float32)
    Q = Q * scale

    m_i = tl.full([BLOCK_M], float("-inf"), dtype = tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype = tl.float32)

    # raw_acc is the UNNORMALIZED weighted sum sum(exp(s - m_i) * V), never divided by l_i mid-loop. This is the correction for the bug described above:
    # renormalizing (dividing by l) at every tile, as _block_sparse_attn_fwd_kernel's single-purpose "acc" does, is only valid for a single tile: from
    # the second tile onward it divides by the OLD l an extra, spurious time, silently corrupting the output. Keeping raw_acc unnormalized through every
    # tile of both passes and dividing by the final l_i exactly once (after the current-block pass below) is the standard, correct flash-attention
    # online-softmax recurrence, matching what _block_sparse_attn_stats_kernel's raw_acc / Acc output already does correctly for the prefix alone.
    raw_acc = tl.zeros([BLOCK_M, D], dtype = tl.float32)

    k_base = K_ptr + b * stride_kb + h_kv * stride_kh
    v_base = V_ptr + b * stride_vb + h_kv * stride_vh
    idx_base = block_indices_ptr + b * stride_ib

    kv_offs_d = tl.arange(0, D)
    kv_offs_n = tl.arange(0, BLOCK_N)

    # Sparse prefix, K selected blocks only, read via block_indices.
    for blk_i in range(n_kv_blocks):
        blk_idx = tl.load(idx_base + blk_i)
        kv_start = blk_idx * BLOCK_N

        K_ptrs = k_base + (kv_start + kv_offs_n[:, None]) * stride_ks + kv_offs_d[None, :] * stride_kd
        kv_mask = (kv_start + kv_offs_n) < S_kv
        K_tile = tl.load(K_ptrs, mask = kv_mask[:, None], other = 0.0).to(tl.float32)

        scores = tl.dot(Q, tl.trans(K_tile))
        scores = tl.where(kv_mask[None, :], scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis = 1))
        exp_scores = tl.exp(scores - m_new[:, None])
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_scores, axis = 1)

        V_ptrs = v_base + (kv_start + kv_offs_n[:, None]) * stride_vs + kv_offs_d[None, :] * stride_vd
        V_tile = tl.load(V_ptrs, mask = kv_mask[:, None], other = 0.0).to(tl.float32)

        raw_acc = raw_acc * tl.exp(m_i - m_new)[:, None] + tl.dot(exp_scores, V_tile)

        m_i = m_new
        l_i = l_new

    # Current generation block, dense (no eviction, always S_cur = BLOCK_N tokens), folded into the same online-softmax state instead of a second kernel 
    # launch / PyTorch pass. cur_k/cur_v use the same h_kv mapping as the prefix (both are KV-head tensors).
    ck_base = CurK_ptr + b * stride_ckb + h_kv * stride_ckh
    cv_base = CurV_ptr + b * stride_cvb + h_kv * stride_cvh
    cur_offs_n = tl.arange(0, BLOCK_N)

    K_cur_ptrs = ck_base + cur_offs_n[:, None] * stride_cks + kv_offs_d[None, :] * stride_ckd
    cur_valid = cur_offs_n < S_cur
    K_cur = tl.load(K_cur_ptrs, mask = cur_valid[:, None], other = 0.0).to(tl.float32)

    scores_cur = tl.dot(Q, tl.trans(K_cur))

    if HAS_MASK:
        mask_offset = b * stride_mb + h_q * stride_mh
        Mask_ptrs = Mask_ptr + mask_offset + q_offs_m[:, None] * stride_ms + cur_offs_n[None, :] * stride_mn
        mask_vals = tl.load(Mask_ptrs, mask = (q_offs_m[:, None] < S_q) & cur_valid[None, :], other = 0.0).to(tl.float32)
        scores_cur = scores_cur + mask_vals

    scores_cur = tl.where(cur_valid[None, :], scores_cur, float("-inf"))

    m_new = tl.maximum(m_i, tl.max(scores_cur, axis = 1))
    exp_scores = tl.exp(scores_cur - m_new[:, None])
    l_new = tl.exp(m_i - m_new) * l_i + tl.sum(exp_scores, axis = 1)

    V_cur_ptrs = cv_base + cur_offs_n[:, None] * stride_cvs + kv_offs_d[None, :] * stride_cvd
    V_cur = tl.load(V_cur_ptrs, mask = cur_valid[:, None], other = 0.0).to(tl.float32)

    raw_acc = raw_acc * tl.exp(m_i - m_new)[:, None] + tl.dot(exp_scores, V_cur)

    m_i = m_new
    l_i = l_new

    # Single normalization, after both passes are done -- not once per tile.
    acc = raw_acc / l_i[:, None]

    out_offset = b * stride_ob + h_q * stride_oh
    Out_ptrs = Out_ptr + out_offset + q_offs_m[:, None] * stride_os + q_offs_d[None, :] * stride_od
    tl.store(Out_ptrs, acc.to(tl.float16), mask = q_offs_m[:, None] < S_q)

def block_sparse_attention_fused(
    query_states: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    block_indices: torch.Tensor,
    cur_k: torch.Tensor,
    cur_v: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    num_kv_groups: int,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Drop-in replacement for sparse_attn_merge with the same signature and the same (Flash-Decoding-equivalent) output, but computed as a single fused 
    Triton kernel launch instead of (Triton prefix pass) + (PyTorch current-block pass) + (Python-level merge). See _block_sparse_attn_fused_fwd_kernel's 
    docstring for the mechanism. This is the change discussed for closing the constant-factor gap with the collaborator's single-launch kernel, applied 
    on top of KINETIC's own top-K prefix eviction rather than replacing it.

    Arguments are identical to sparse_attn_merge: query_states is [B, H_q, S_q, D] (post-RoPE), prefix_k/prefix_v are the full [B, H_kv, S_prefix, D] 
    prefix cache (not pre-sliced), block_indices is [B, K] int32 selected block indices, cur_k/cur_v are the [B, H_kv, S_q, D] current-block keys/values 
    (post-RoPE), attention_mask is the optional [..., S_q, S_prefix + S_q] additive mask from eval_mask() (only its current-block columns are used here; the prefix
    eviction gather upstream already handles the prefix columns by construction, same as sparse_attn_merge), scaling is 1/sqrt(D), num_kv_groups is
    H_q // H_kv, and block_size must match BLOCK_N (default 32).

    Returns a [B, H_q, S_q, D] attention output, same dtype as query_states.

    Unverified on hardware: this kernel has not yet been run through kernel/test_eviction.py --mode noop (no GPU was available while writing it). Run 
    that gate before trusting any accuracy/speedup numbers from it, exactly as the repo's correctness-first convention requires for any change here.
    """
    B, H_q, S_q, D = query_states.shape
    _, H_kv, S_prefix, _ = prefix_k.shape
    _, _, S_cur, _ = cur_k.shape
    K_selected = block_indices.shape[1]

    assert S_q <= BLOCK_M, f"block_sparse_attention_fused: S_q = {S_q} > BLOCK_M = {BLOCK_M}."
    assert S_cur == BLOCK_N, f"block_sparse_attention_fused: current block width {S_cur} != BLOCK_N = {BLOCK_N}."
    assert block_size == BLOCK_N, f"block_sparse_attention_fused: block_size = {block_size} != BLOCK_N = {BLOCK_N}."
    assert D in (64, 128, 256), f"block_sparse_attention_fused: unexpected head_dim {D}"
    assert block_indices.dtype == torch.int32, "block_indices must be int32"

    Out = torch.empty(B, H_q, S_q, D, device = query_states.device, dtype = torch.float16)

    Q_c = query_states.contiguous().to(torch.float16)
    K_c = prefix_k.contiguous().to(torch.float16)
    V_c = prefix_v.contiguous().to(torch.float16)
    CurK_c = cur_k.contiguous().to(torch.float16)
    CurV_c = cur_v.contiguous().to(torch.float16)
    idx_c = block_indices.contiguous()

    has_mask = attention_mask is not None
    if has_mask:
        # Same slicing as sparse_attn_merge's Pass 2: attention_mask covers [..., S_prefix_in_mask + S_cur]; only the current-block columns are needed 
        # here, since the prefix eviction gather upstream already accounts for the prefix columns. Expanded to a full [B, H_q, S_q, S_cur] contiguous 
        # fp32 tensor so the kernel can index it with plain strides (mirrors the implicit broadcast the original scores_cur + cur_mask did in PyTorch).
        S_prefix_in_mask = attention_mask.shape[-1] - cur_k.shape[-2]
        cur_mask = attention_mask[..., S_prefix_in_mask:]
        Mask_c = cur_mask.expand(B, H_q, S_q, S_cur).contiguous().to(torch.float32)
    else:
        Mask_c = torch.empty(1, device = query_states.device, dtype = torch.float32)

    grid = (B, H_q)
    scale = D ** -0.5

    _block_sparse_attn_fused_fwd_kernel[grid](
        Q_c, K_c, V_c, CurK_c, CurV_c, Mask_c, Out,
        idx_c,
        K_selected,
        scale,
        B, H_q, H_kv,
        S_q, S_prefix, S_cur, D,
        num_kv_groups,
        has_mask,
        Q_c.stride(0), Q_c.stride(1), Q_c.stride(2), Q_c.stride(3),
        K_c.stride(0), K_c.stride(1), K_c.stride(2), K_c.stride(3),
        V_c.stride(0), V_c.stride(1), V_c.stride(2), V_c.stride(3),
        CurK_c.stride(0), CurK_c.stride(1), CurK_c.stride(2), CurK_c.stride(3),
        CurV_c.stride(0), CurV_c.stride(1), CurV_c.stride(2), CurV_c.stride(3),
        Mask_c.stride(0) if has_mask else 0,
        Mask_c.stride(1) if has_mask else 0,
        Mask_c.stride(2) if has_mask else 0,
        Mask_c.stride(3) if has_mask else 0,
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        idx_c.stride(0),
        BLOCK_M = BLOCK_M,
        BLOCK_N = BLOCK_N,
    )

    return Out.view(B, H_q, S_q, D).to(query_states.dtype)

def sparse_attn_merge(
    query_states: torch.Tensor,
    prefix_k: torch.Tensor,
    prefix_v: torch.Tensor,
    block_indices: torch.Tensor,
    cur_k: torch.Tensor,
    cur_v: torch.Tensor,
    attention_mask: torch.Tensor,
    scaling: float,
    num_kv_groups: int,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Compute attention over (sparse prefix blocks via Triton) + (current generation block via manual softmax), then merge with the Flash Decoding 
    log-sum-exp combine. No gather or concatenation of the prefix cache occurs.

    This function exists to fix a correctness bug: block_sparse_attention_with_stats returns Acc as the raw (un-normalized) weighted sum of values, i.e. 
    Acc = sum(exp(score - m) * V), not divided by l. The merge below uses Acc directly, not Out (which is already normalized). Using Out here would 
    double-count the l_pre denominator and silently corrupt every output, which is exactly what produced the near-zero accuracy in the earlier attempt.

    The merge formula (Flash Decoding) is m = max(m_pre, m_cur), l = exp(m_pre - m) * l_pre + exp(m_cur - m) * l_cur, and o = (exp(m_pre - m) * Acc_pre 
    + exp(m_cur - m) * Acc_cur) / l.

    query_states is the [B, H_q, S_q, D] current block queries (post-RoPE). prefix_k and prefix_v are the [B, H_kv, S_prefix, D] full prefix cache, not 
    pre-sliced. block_indices is a [B, K] int32 tensor of selected block indices (sorted), one set shared across the batch (matmul union) or 
    per-sequence. cur_k and cur_v are the [B, H_kv, S_q, D] current block keys/values (post-RoPE). attention_mask is a float additive mask [..., S_q, 
    S_prefix + S_q] from eval_mask(), used only for the current-block columns (last S_q columns). scaling is 1 / sqrt(D), num_kv_groups is H_q // H_kv 
    for GQA expansion, and block_size is the tokens per block, which must match BLOCK_N (default 32).

    Returns a [B, H_q, S_q, D] attention output, same dtype as query_states.
    """
    B, H_q, S_q, D = query_states.shape
    _, H_kv, S_prefix, _ = prefix_k.shape

    # Sparse prefix attention via Triton, with raw stats for merging.
    Out_pre, Acc_pre, M_pre, L_pre = block_sparse_attention_with_stats(
        query_states, prefix_k, prefix_v, block_indices, block_size = block_size
    )

    # The Triton kernel allocates output buffers of size BLOCK_M = 32 regardless of the actual query length S_q. When S_q < BLOCK_M (e.g. S_q = 8 for 
    # use_block_cache sub-block calls), positions S_q...31 contain garbage. Slice to the actual query length before the merge.
    if S_q < Out_pre.shape[2]:
        Out_pre = Out_pre[:, :, :S_q, :]
        Acc_pre = Acc_pre[:, :, :S_q, :]
        M_pre   = M_pre[:, :, :S_q]
        L_pre   = L_pre[:, :, :S_q]

    # Current-block attention via manual softmax (need m, l explicitly).
    cur_k_exp = cur_k.repeat_interleave(num_kv_groups, dim = 1)
    cur_v_exp = cur_v.repeat_interleave(num_kv_groups, dim = 1)

    scores_cur = torch.matmul(
        query_states.float() * scaling, cur_k_exp.float().transpose(-1, -2)
    )

    if attention_mask is not None:
        # attention_mask was already sliced by the eviction gather path to match the selected prefix tokens. Its shape is [..., sliced_prefix + 
        # S_cur_kv]. We need only the current-block columns.
        S_prefix_in_mask = attention_mask.shape[-1] - cur_k.shape[-2]
        if S_prefix_in_mask >= 0 and S_prefix_in_mask < attention_mask.shape[-1]:
            cur_mask = attention_mask[..., S_prefix_in_mask:]
            scores_cur = scores_cur + cur_mask.to(scores_cur.dtype)

    m_cur = scores_cur.max(dim = -1, keepdim = True).values.squeeze(-1) # [B, H_q, S_q].
    exp_cur = torch.exp(scores_cur - m_cur.unsqueeze(-1))
    l_cur = exp_cur.sum(dim = -1) # [B, H_q, S_q].
    Acc_cur = torch.matmul(exp_cur, cur_v_exp.float()) # [B, H_q, S_q, D], raw (un-normalized).

    # Flash Decoding merge using raw accumulators (Acc_pre, Acc_cur), not the normalized Out_pre.
    m_global = torch.maximum(M_pre, m_cur) # [B, H_q, S_q].
    scale_pre = torch.exp(M_pre - m_global).unsqueeze(-1) # [B, H_q, S_q, 1].
    scale_cur = torch.exp(m_cur - m_global).unsqueeze(-1)
    l_global = (torch.exp(M_pre - m_global) * L_pre + torch.exp(m_cur - m_global) * l_cur).unsqueeze(-1)

    o_merged = (scale_pre * Acc_pre + scale_cur * Acc_cur) / l_global

    return o_merged.to(query_states.dtype)