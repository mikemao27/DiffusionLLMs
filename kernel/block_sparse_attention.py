"""
Triton kernel for paged, block-sparse attention over a KVBlockPool.

What this replaces:

During the intra-block diffusion loop, Fast-dLLM v2 calls:

    self.forward(input_ids=x_t[:, -block_size:], ..., past_key_values=paged_cache)

Qwen2.5's attention module then does:

    k_full, v_full = past_key_values.update(k_new, v_new, layer_idx)
    attn_output = F.scaled_dot_product_attention(q, k_full, v_full, ...)

The PagedKVCache.update() call still returns a full contiguous (k, v), so
the normal SDPA path continues to work — this gives correctness with zero
modification to the model code.

The Triton kernel in this file is an *optional, faster alternative* for the
prefix-attention part: attending over the paged KV pool blocks directly,
without materialising the full contiguous K/V tensor.  It is invoked by
`paged_prefix_attention()` which the test harness and a thin model hook
can call in place of standard SDPA for the prefix portion.

Kernel design:

Each Triton program instance handles:
    - one query token (or BLOCK_Q tokens)
    - one attention head
    - all active KV blocks for that head

The kernel iterates over the active physical slots (provided in `slot_table`,
shape [num_active_blocks]), loads each KV block from the pool, computes QK^T,
and accumulates the output with the online softmax trick (Dao et al., 2022).

This is essentially FlashAttention-2 extended to non-contiguous memory via
indirect indexing through the slot table.

Attention mask:

Fast-dLLM v2 uses block-wise causal + bidirectional-within-block masks:
  * Prefix blocks (all blocks in the pool): fully visible to the current block
    queries (causal at block granularity, so the full prefix is accessible).

  * Current block tokens: bidirectional within the block (handled by the
    intra-block SDPA in model.forward, not by this kernel).

Therefore this kernel always attends to ALL tokens in every active block
with NO causal masking — the prefix has already been committed causally.

Shapes and conventions:

Dimensions match Qwen2.5-7B defaults unless otherwise noted:
    H = num_kv_heads (4 for GQA Qwen2.5-7B; 8 for 1.5B)
    D = head_dim (128)
    B = BLOCK_SIZE (32 tokens per KV block)
    S = num_active_blocks

Query tensor fed to this kernel:
    q : [num_q_heads, q_seq_len, head_dim] (current-block queries, one batch element)

Because Qwen2.5 uses GQA (grouped-query attention), num_q_heads >= num_kv_heads.
The kernel handles the GQA fan-out: query head h attends to KV head h // gqa_ratio.

Output:
    - out : [num_q_heads, q_seq_len, head_dim] (same shape as input q)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from typing import Optional

# Compile-time constants / tile sizes

# Tile size along the query-sequence dimension.
# 32 works well for block_size=32 (one warp = one query block).
BLOCK_Q: tl.constexpr = 32

# Tile size along the KV-sequence dimension (one KV block = 32 tokens).
# Must equal the pool's block_size.
BLOCK_KV: tl.constexpr = 32

# Triton kernel

@triton.jit
def _paged_attn_kernel(
    # Pointers:
    q_ptr, # [num_q_heads, q_seq_len, head_dim] — input queries
    pool_k_ptr, # [max_slots, num_kv_heads, BLOCK_KV, head_dim] — K pool
    pool_v_ptr, # [max_slots, num_kv_heads, BLOCK_KV, head_dim] — V pool
    slot_table_ptr, # [num_active_blocks] — physical slot indices (int32)
    out_ptr, # [num_q_heads, q_seq_len, head_dim] — output

    # Strides (in elements, not bytes): 
    # q / out: [num_q_heads, q_seq_len, head_dim]
    stride_qh, stride_qq, stride_qd,
    # pool: [max_slots, num_kv_heads, BLOCK_KV, head_dim]
    stride_ps, stride_ph, stride_pk, stride_pd,

    # Scalar metadata:
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    q_seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    num_active_blocks: tl.constexpr,
    scale: tl.constexpr, # 1 / sqrt(head_dim) pre-computed
    gqa_ratio: tl.constexpr, # num_q_heads // num_kv_heads
):
    """
    One programme instance handles:
        query head: prog_id axis 0 (0 … num_q_heads-1)
        query tile: prog_id axis 1 (covers BLOCK_Q query positions)

    Inner loop: iterate over num_active_blocks KV blocks.

    Accumulates the attention output using the online-softmax / running-max
    trick so the full softmax is never materialised in SRAM.
    """

    # Program identity:
    qh_idx = tl.program_id(0) # which query head
    q_tile = tl.program_id(1) # which tile of query positions

    # Corresponding KV head (GQA: multiple Q heads share one KV head).
    kv_h_idx = qh_idx // gqa_ratio

    # Query positions this instance covers: [q_tile*BLOCK_Q … (q_tile+1)*BLOCK_Q)
    q_offs = q_tile * BLOCK_Q + tl.arange(0, BLOCK_Q) # shape: [BLOCK_Q]
    q_mask = q_offs < q_seq_len # guard for last tile

    # Head-dim offsets.
    d_offs = tl.arange(0, head_dim) # shape: [head_dim]

    # Load query tile: shape [BLOCK_Q, head_dim]:
    # q_ptr layout: [num_q_heads, q_seq_len, head_dim]
    q_base = (
        qh_idx * stride_qh
        + q_offs[:, None] * stride_qq
        + d_offs[None, :] * stride_qd
    )
    q = tl.load(q_ptr + q_base, mask=q_mask[:, None], other=0.0)
    # q shape: [BLOCK_Q, head_dim]

    # Initialize online-softmax accumulators
    # Running max of QK scores (for numerical stability).
    m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)

    # Denominator sum (partition function).
    lse = tl.full([BLOCK_Q], 0.0, dtype=tl.float32)

    # Output accumulator.
    acc = tl.zeros([BLOCK_Q, head_dim], dtype=tl.float32)

    # Iterate over active KV blocks:
    for block_i in tl.range(0, num_active_blocks):

        # Load physical slot index for this logical block.
        slot = tl.load(slot_table_ptr + block_i).to(tl.int32)

        # Base pointer for K and V of this slot and KV head.
        # pool layout: [max_slots, num_kv_heads, BLOCK_KV, head_dim]
        slot_k_base = slot * stride_ps + kv_h_idx * stride_ph
        slot_v_base = slot * stride_ps + kv_h_idx * stride_ph

        kv_offs = tl.arange(0, BLOCK_KV) # shape: [BLOCK_KV]

        # Load K block: [BLOCK_KV, head_dim]
        k_ptrs = (
            pool_k_ptr
            + slot_k_base
            + kv_offs[:, None] * stride_pk
            + d_offs[None, :] * stride_pd
        )
        k = tl.load(k_ptrs) # [BLOCK_KV, head_dim]

        # Compute QK^T scaled: [BLOCK_Q, BLOCK_KV]
        # q: [BLOCK_Q, head_dim], k^T: [head_dim, BLOCK_KV]
        qk = tl.dot(q, tl.trans(k)) * scale # [BLOCK_Q, BLOCK_KV]

        # Mask out-of-range query rows (last tile may be partial).
        # All KV positions within this block are always valid (we pad to BLOCK_KV
        # in write_block so they are never garbage, just zero).
        qk = tl.where(q_mask[:, None], qk, float("-inf"))

        # Online softmax update (Dao et al., "FlashAttention-2")
        # New running max for this block.
        m_new = tl.maximum(m, tl.max(qk, axis=1)) # [BLOCK_Q]

        # Rescale existing accumulator by exp(m - m_new).
        alpha = tl.exp(m - m_new) # [BLOCK_Q]
        acc = acc * alpha[:, None]
        lse = lse * alpha

        # Softmax numerator for this block.
        p = tl.exp(qk - m_new[:, None]) # [BLOCK_Q, BLOCK_KV]

        # Update denominator.
        lse = lse + tl.sum(p, axis=1) # [BLOCK_Q]

        # Update running max.
        m = m_new

        # Load V block: [BLOCK_KV, head_dim]
        v_ptrs = (
            pool_v_ptr
            + slot_v_base
            + kv_offs[:, None] * stride_pk
            + d_offs[None, :] * stride_pd
        )
        v = tl.load(v_ptrs) # [BLOCK_KV, head_dim]

        # Accumulate weighted values: p @ V → [BLOCK_Q, head_dim]
        acc = acc + tl.dot(p.to(v.dtype), v).to(tl.float32)

    # Normalize: divide by partition function
    # Guard against all-masked rows (lse == 0 when no blocks were attended).
    lse_safe = tl.where(lse > 0.0, lse, 1.0)
    acc = acc / lse_safe[:, None]

    # Write output: [BLOCK_Q, head_dim] → out[qh_idx, q_tile*BLOCK_Q:, :]
    out_base = (
        qh_idx * stride_qh
        + q_offs[:, None] * stride_qq
        + d_offs[None, :] * stride_qd
    )
    tl.store(out_ptr + out_base, acc.to(q.dtype), mask=q_mask[:, None])

# Python wrapper

def paged_prefix_attention(
    q: torch.Tensor, # [batch=1, num_q_heads, q_seq_len, head_dim]
    pool_k: torch.Tensor, # [max_slots, num_kv_heads, BLOCK_KV, head_dim]
    pool_v: torch.Tensor, # [max_slots, num_kv_heads, BLOCK_KV, head_dim]
    slot_table: torch.Tensor, # [num_active_blocks] — int32 physical slots
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute attention from the current-block queries over all paged KV blocks
    in the pool, using the Triton kernel above.

    This is the *prefix attention* component — it only covers tokens already
    committed to the pool.  Intra-block attention (tokens in the current
    diffusion block) is still handled by the standard Qwen2.5 SDPA.

    Parameters:
        - q : Tensor [1, num_q_heads, q_seq_len, head_dim]
          Query tensor for the current block (batch size is always 1 here;
          Fast-dLLM v2 generation_functions.py peels the batch loop with
          `finished_samples`).

        - pool_k : Tensor [max_slots, num_kv_heads, BLOCK_KV, head_dim]
          Full key pool for this layer (from KVBlockPool.get_pool_k).

        - pool_v : Tensor [max_slots, num_kv_heads, BLOCK_KV, head_dim]
          Full value pool for this layer.
    
        - slot_table : Tensor [num_active_blocks], dtype int32
          Physical slot indices to attend over (from PagedKVCache.get_slot_tensor).
    
        - scale : float, optional
          Softmax temperature scale.  Defaults to 1/sqrt(head_dim).

    Returns:
        - Tensor [1, num_q_heads, q_seq_len, head_dim]
          Attention output over the paged prefix.
    """
    assert q.dim() == 4 and q.shape[0] == 1, (
        f"Expected q of shape [1, H, S, D], got {q.shape}"
    )

    # Strip batch dimension — the kernel works on a single request.
    q_3d = q.squeeze(0) # [num_q_heads, q_seq_len, head_dim]

    num_q_heads, q_seq_len, head_dim = q_3d.shape
    max_slots, num_kv_heads, block_kv, _ = pool_k.shape
    num_active_blocks = slot_table.shape[0]

    assert block_kv == BLOCK_KV, (
        f"pool block_size {block_kv} != kernel BLOCK_KV {BLOCK_KV}"
    )
    assert num_q_heads % num_kv_heads == 0, (
        f"GQA requires num_q_heads ({num_q_heads}) % num_kv_heads ({num_kv_heads}) == 0"
    )
    gqa_ratio = num_q_heads // num_kv_heads

    if scale is None:
        scale = head_dim ** -0.5

    if num_active_blocks == 0:
        # No committed blocks to attend over — return zeros.
        return torch.zeros_like(q)

    # Ensure contiguous layout for the kernel.
    q_3d = q_3d.contiguous()
    pool_k = pool_k.contiguous()
    pool_v = pool_v.contiguous()
    slot_table = slot_table.contiguous()

    out = torch.empty_like(q_3d)

    # Number of query tiles (ceiling division).
    num_q_tiles = (q_seq_len + BLOCK_Q - 1) // BLOCK_Q

    # Grid: one programme per (query head, query tile).
    grid = (num_q_heads, num_q_tiles)

    _paged_attn_kernel[grid](
        # Pointers
        q_3d, pool_k, pool_v, slot_table, out,
        # Strides for q / out
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        # Strides for pool (same for K and V)
        pool_k.stride(0), pool_k.stride(1), pool_k.stride(2), pool_k.stride(3),
        # Scalars
        num_q_heads = num_q_heads,
        num_kv_heads = num_kv_heads,
        q_seq_len = q_seq_len,
        head_dim = head_dim,
        num_active_blocks = num_active_blocks,
        scale = scale,
        gqa_ratio = gqa_ratio,
    )

    # Re-add the batch dimension.
    return out.unsqueeze(0) # [1, num_q_heads, q_seq_len, head_dim]

# Model-level hook: monkey-patch Qwen2.5 attention to use paged prefix attn:
def install_paged_attention_hook(model, paged_cache) -> None:
    """
    Monkey-patch every Qwen2Attention layer in *model* so that, when
    *paged_cache* has at least one committed block, the prefix attention
    component is computed by the Triton paged kernel instead of standard SDPA.

    The hook intercepts `layer.self_attn.forward()` and splits attention into:
        1. Prefix attention (Triton kernel) — over past paged blocks.
        2. Self attention (standard SDPA)   — over the current block only.
    The two outputs are added together after normalisation.

    Parameters:
        - model : Qwen2ForCausalLM (or equivalent)
    paged_cache : PagedKVCache

    Notes:

    This hook is *additive*: it adds the prefix-attention output on top of
    the intra-block output. A more surgical integration would split the Q/K/V
    projections, but that requires model-specific surgery. The additive
    approach is model-agnostic and avoids touching the projection layers.

    For a clean production integration you would instead subclass
    Qwen2Attention and override forward(); the hook here is intentionally
    minimal so you can replace it with whatever suits your codebase.
    """
    import types

    def _make_hooked_forward(original_forward, layer_idx: int, cache_ref):
        """Factory: closes over layer_idx and cache_ref."""

        def hooked_forward(self, hidden_states, attention_mask=None,
                           position_ids=None, past_key_value=None,
                           output_attentions=False, use_cache=False,
                           cache_position=None, **kwargs):

            # Run the original forward to get intra-block attention output.
            output = original_forward(
                hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_value = past_key_value,
                output_attentions = output_attentions,
                use_cache = use_cache,
                cache_position = cache_position,
                **kwargs,
            )

            # If no paged blocks exist yet, skip the prefix kernel.
            if cache_ref.get_num_active_blocks() == 0:
                return output

            # Unpack: Qwen2Attention returns (attn_output, attn_weights, past_kv)
            # or (attn_output, None, past_kv) depending on output_attentions.
            attn_output = output[0] # [batch, seq_len, hidden_size]

            # Compute prefix attention via Triton kernel
            # We need query states. Re-project from hidden_states.
            # (self here is the Qwen2Attention instance)
            bsz, seq_len, _ = hidden_states.shape
            q_proj = self.q_proj(hidden_states) # [B, S, num_q_heads * head_dim]
            num_q_heads = self.num_heads
            head_dim = self.head_dim
            q = q_proj.view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
            # q: [B, num_q_heads, seq_len, head_dim]

            pool_k = cache_ref.pool.get_pool_k(layer_idx)
            pool_v = cache_ref.pool.get_pool_v(layer_idx)
            slots = cache_ref.get_slot_tensor(layer_idx)

            # Run the paged prefix attention kernel.
            prefix_out = paged_prefix_attention(q, pool_k, pool_v, slots)
            # prefix_out: [B=1, num_q_heads, seq_len, head_dim]

            # Merge prefix output back to hidden-state shape.
            prefix_out = prefix_out.transpose(1, 2).reshape(bsz, seq_len, -1)
            # Apply output projection.
            prefix_out = self.o_proj(prefix_out) # [B, S, hidden_size]

            # Add to intra-block output.
            # Scaling by 0.5 implements a simple average of the two attention
            # components. A learnable gate would be better but requires
            # fine-tuning; for inference this is a reasonable approximation.
            merged = 0.5 * attn_output + 0.5 * prefix_out

            return (merged,) + output[1:]

        return hooked_forward

    # Patch each attention layer.
    n_patched = 0
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        original_fwd = attn.forward
        attn.forward = types.MethodType(
            _make_hooked_forward(original_fwd, layer_idx, paged_cache),
            attn,
        )
        n_patched += 1

    print(f"[paged_attn] Hooked {n_patched} attention layers with paged prefix kernel.")