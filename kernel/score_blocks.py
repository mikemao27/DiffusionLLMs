"""
Block Importance Scoring Kernel (key-norm proxy).

Job: score N prefix blocks by an attention-free proxy signal, for use as a
block-selection method when true attention weights are not available.

STATUS: NOT CURRENTLY WIRED IN. BlockEvictionScheduler (block_eviction_scheduler.py)
scores blocks directly from captured attention weights (_last_attn_weights),
which is a strictly more accurate signal than the key-norm proxy implemented
here, since it doesn't need a proxy at all. Nothing in this codebase currently
imports score_blocks(). This file is kept as an alternative/fallback scoring
strategy for cases where attention-weight capture isn't available (e.g. before
the _capture_attn_scores hook existed). If you don't need that fallback, this
file and its Triton kernel can be deleted with no effect on the eviction path.

Scores N prefix blocks by computing the mean squared L2 norm of key vectors
within each block, averaged across all KV heads and all transformer layers.
The result is a [B, N] importance tensor, in the same shape BlockEvictionScheduler
expects if it were wired in to use this scoring method instead.

Why key norms as the scoring signal:
    Ideally we would use softmax attention weights, but those require a query
    tensor which is not available at score time (the commit step has already
    returned). Key vector norms are a well-established proxy: tokens with large
    key magnitudes attract more attention mass, and the correlation with true
    attention weights is strong enough for block selection purposes. MInference
    and H2O both use variants of this proxy.

Why a Triton kernel instead of PyTorch:
    The naive Python loop over layers does:
        [L layers] x [B x H_kv x S x D -> B x S -> B x N]
    Each step materializes an intermediate tensor. For a 28-layer Qwen2.5-7B
    model with a 2048-token prefix this is 28 separate .norm() and .mean()
    calls, each launching their own CUDA kernels. The Triton kernel fuses all
    layers and the block-reduction into a single pass over HBM, reading each
    key tensor once and writing only the final [B, N] score matrix.

Kernel design:
    Grid: [B, N_blocks, H_kv]
    Each program handles one (batch, block, head) triple. It loops over the
    D=128 dimensions and block_size=32 tokens in that block, accumulating the
    sum of squared values as a scalar float32. It then atomically adds that
    scalar into out[b, block_idx].

    Key correctness point: block_sum must be a scalar (f32), not a tensor,
    because tl.atomic_add expects ptr<f32> vs f32, not ptr<f32> vs tensor<1xf32>.
    The accumulation uses a plain Python float variable pattern via a 1-D
    tensor reduced to scalar with tl.sum before the atomic.

Interface contract (never changes):
    - score_blocks() takes the *full* per-layer key caches and returns one
      score per complete block. It never mutates its inputs.
    - Higher score = more important = more likely to be kept if this scorer
      were wired into BlockEvictionScheduler's selection logic.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def _score_blocks_kernel(
    keys_ptr,
    out_ptr,
    B,
    H_kv,
    S,
    D: tl.constexpr,
    block_size: tl.constexpr,
    n_blocks,
    stride_b,
    stride_h,
    stride_s,
    stride_ob,
    stride_on,
):
    """
    One program per (batch, block_idx, head) triple.

    Computes the sum of squared values of all key vectors in this block
    (block_size tokens x D dimensions), then atomically adds the scalar
    result into out[b, block_idx].

    Args:
        - keys_ptr: [B, H_kv, S, D] float32 key tensor (row-major).
        - out_ptr: [B, N_blocks] float32 output, pre-zeroed.
        - B: batch size.
        - H_kv: number of KV heads.
        - S: full prefix sequence length.
        - D: head dimension (constexpr, must be power of 2, e.g. 128).
        - block_size: tokens per generation block (constexpr, e.g. 32).
        - n_blocks: number of complete blocks (S // block_size).
        - stride_b, stride_h, stride_s: key tensor strides (in elements).
        - stride_ob, stride_on: output tensor strides (in elements).

    Returns:
        - None. Atomically accumulates into out_ptr.
    """
    b = tl.program_id(0)
    blk = tl.program_id(1)
    h = tl.program_id(2)

    tok_start = blk * block_size

    # Base pointer to keys[b, h, tok_start, 0].
    base = (
        keys_ptr
        + b * stride_b
        + h * stride_h
        + tok_start * stride_s
    )

    d_offsets = tl.arange(0, D)
    block_sum = tl.zeros([1], dtype = tl.float32)

    for tok_off in tl.static_range(block_size):
        row_ptr = base + tok_off * stride_s
        mask = (tok_start + tok_off) < S
        vals = tl.load(row_ptr + d_offsets, mask = mask, other = 0.0).to(tl.float32)
        block_sum += tl.sum(vals * vals, axis = 0, keep_dims = True)

    # Reduce the 1-element tensor to a scalar for atomic_add.
    scalar_sum = tl.sum(block_sum, axis = 0)

    out_ptr_elem = out_ptr + b * stride_ob + blk * stride_on
    tl.atomic_add(out_ptr_elem, scalar_sum)


def score_blocks(key_cache_layers: list, block_size: int = 32) -> torch.Tensor:
    """
    Compute per-block importance scores from all layers of the KV key cache.

    Args:
        - key_cache_layers: list of [B, H_kv, S, D] key tensors, one per layer.
        None entries are skipped.
        - block_size: number of tokens per block.

    Returns:
        - a [B, N_blocks] float32 tensor of block importance scores, where
            N_blocks = S // block_size.
    """
    layers = [k for k in key_cache_layers if k is not None]
    if not layers:
        raise ValueError("score_blocks: all key cache layers are None.")

    B, H_kv, S, D = layers[0].shape
    n_blocks = S // block_size
    if n_blocks == 0:
        return torch.zeros(B, 0, device = layers[0].device, dtype = torch.float32)

    assert D in (32, 64, 128, 256), f"score_blocks: unexpected head_dim {D}"

    out = torch.zeros(B, n_blocks, device = layers[0].device, dtype = torch.float32)
    grid = (B, n_blocks, H_kv)

    for keys in layers:
        keys_c = keys.contiguous()
        if keys_c.dtype != torch.float32:
            keys_c = keys_c.float()

        _score_blocks_kernel[grid](
            keys_c,
            out,
            B, H_kv, S, D,
            block_size,
            n_blocks,
            keys_c.stride(0),
            keys_c.stride(1),
            keys_c.stride(2),
            out.stride(0),
            out.stride(1),
        )

    # Normalize: mean squared norm per block.
    n_layers = len(layers)
    out /= (n_layers * H_kv * block_size * D)
    return out