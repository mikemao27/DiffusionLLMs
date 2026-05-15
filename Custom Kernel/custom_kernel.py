import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

# triton kernel to compute cosine-similarities between tensors
@triton.jit
def compute_cosine_similarity(
    previous_hidden_ptr, # pointer to the previous hidden states
    current_hidden_ptr, # pointer to the current hidden states
    active_mask_ptr, # pointer to the mask that tracks token active/frozen state
    freeze_counter_ptr,
    block_length: tl.constexpr, # length of one decoding block (e.g. 32, 64)
    dimension_size: tl.constexpr, # hidden state dimension (power of two)
    BLOCK_SIZE: tl.constexpr,
    threshold_front: tl.constexpr,
    threshold_mid: tl.constexpr,
    threshold_end: tl.constexpr,
    freeze_steps_front: tl.constexpr,
    freeze_steps_mid: tl.constexpr,
    freeze_steps_end: tl.constexpr,
):
    token_idx = tl.program_id(0)

    # token relative position within the decoding block, then integer-divide into thirds
    relative_pos = token_idx % block_length
    position_idx = (relative_pos * 3) // block_length # 0 = front, 1 = mid, 2 = end

    # select position specific parameters
    threshold = tl.where(position_idx == 0, threshold_front,
                         tl.where(position_idx == 1, threshold_mid, threshold_end))
    freeze_steps = tl.where(position_idx == 0, freeze_steps_front,
                            tl.where(position_idx == 1, freeze_steps_mid, freeze_steps_end))

    token_start = token_idx * dimension_size
    token_offsets = tl.arange(0, BLOCK_SIZE)

    dot_product = tl.zeros([], dtype=tl.float32)
    norm_a_squared = tl.zeros([], dtype=tl.float32)
    norm_b_squared = tl.zeros([], dtype=tl.float32)

    for i in range(0, dimension_size, BLOCK_SIZE):
        offsets = token_start + i + token_offsets

        a = tl.load(previous_hidden_ptr + offsets).to(tl.float32)
        b = tl.load(current_hidden_ptr + offsets).to(tl.float32)

        dot_product += tl.sum(a * b, axis=0)
        norm_a_squared += tl.sum(a * a, axis=0)
        norm_b_squared += tl.sum(b * b, axis=0)

    cosine_similarity = dot_product / tl.sqrt(norm_a_squared + norm_b_squared + 1e-8) # small offset to avoid dividing by 0

    freeze_counter = tl.load(freeze_counter_ptr + token_idx)

    updated_active_mask = tl.where(freeze_counter > 0, 0,
                                   tl.where(cosine_similarity > threshold, 0, 1))
    updated_freeze_counter = tl.where(freeze_counter > 0, freeze_counter - 1,
                                      tl.where(cosine_similarity > threshold, freeze_steps, freeze_counter))

    tl.store(active_mask_ptr + token_idx, updated_active_mask.to(tl.int32))
    tl.store(freeze_counter_ptr + token_idx, updated_freeze_counter)

def check_cosine_similarity(
    previous_hidden: torch.Tensor,
    current_hidden: torch.Tensor,
    active_mask: torch.Tensor,
    freeze_counter: torch.Tensor,
    block_length: int,
    threshold_front: float = 0.99,
    threshold_mid: float = 0.99,
    threshold_end: float = 0.995,
    freeze_steps_front: int = 3,
    freeze_steps_mid: int = 2,
    freeze_steps_end: int = 1,
):  
    assert previous_hidden.shape == current_hidden.shape, \
        f"shape mismatch: {previous_hidden.shape} vs. {current_hidden.shape}"
    assert previous_hidden.is_contiguous() and current_hidden.is_contiguous(), \
        f"hidden state tensors must be contiguous"
    
    seq_length, dimension_size = previous_hidden.shape
    BLOCK_SIZE = min(dimension_size, 256)

    assert dimension_size % BLOCK_SIZE == 0, \
        f"dimension_size {dimension_size} must be divisible by BLOCK_SIZE {BLOCK_SIZE}"
    
    grid = (seq_length,)

    compute_cosine_similarity[grid](
        previous_hidden, current_hidden, active_mask, freeze_counter,
        block_length=block_length, dimension_size=dimension_size, BLOCK_SIZE=BLOCK_SIZE,
        threshold_front=threshold_front, threshold_mid=threshold_mid, threshold_end=threshold_end,
        freeze_steps_front=freeze_steps_front, freeze_steps_mid=freeze_steps_mid, freeze_steps_end=freeze_steps_end
    )

@triton.jit
def update_hidden_cache(
    new_hidden_ptr, # pointer to newly computed hidden states
    cached_hidden_ptr, # pointer to existing cache
    active_indices_ptr, # pointer to active token indices
    dimension_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    token_idx = tl.program_id(0)

    active_indices = tl.load(active_indices_ptr + token_idx)

    token_start = active_indices * dimension_size
    token_offsets = tl.arange(0, BLOCK_SIZE)

    for i in range(0, dimension_size, BLOCK_SIZE):
        offsets = token_start + i + token_offsets

        new_hidden_states = tl.load(new_hidden_ptr + offsets)

        tl.store(cached_hidden_ptr + offsets, new_hidden_states)

def apply_update_hidden_cache(
    new_hidden: torch.Tensor,
    cached_hidden: torch.Tensor,
    active_indices: torch.Tensor,
):
    assert new_hidden.shape == cached_hidden.shape, \
        f"shape mismatch: {new_hidden.shape} vs. {cached_hidden.shape}"
    
    _, dimension_size = new_hidden.shape
    BLOCK_SIZE = min(dimension_size, 256)

    assert dimension_size % BLOCK_SIZE == 0, \
        f"dimension_size {dimension_size} must be divisible by BLOCK_SIZE {BLOCK_SIZE}"

    assert active_indices.is_contiguous(), \
        f"active_indices tensor must be contiguous"

    grid = (active_indices.shape[0],)

    update_hidden_cache[grid](
        new_hidden, cached_hidden, active_indices,
        dimension_size=dimension_size, BLOCK_SIZE=BLOCK_SIZE
    )

def kernel_integration(
        current_hidden: torch.Tensor, 
        persistent_cache: torch.Tensor,
        active_mask: torch.Tensor,
        freeze_counter: torch.Tensor,
        block_length: int = 64,
        threshold_front: float = 0.99,
        threshold_mid: float = 0.99,
        threshold_end: float = 0.995,
        freeze_steps_front: int = 3,
        freeze_steps_mid: int = 2,
        freeze_steps_end: int = 1,
):
    check_cosine_similarity(
        persistent_cache, current_hidden, 
        active_mask, freeze_counter,
        block_length=block_length,
        threshold_front=threshold_front, threshold_mid=threshold_mid, threshold_end=threshold_end,
        freeze_steps_front=freeze_steps_front, freeze_steps_mid=freeze_steps_mid, freeze_steps_end=freeze_steps_end
    )

    active_indices = torch.nonzero(active_mask).squeeze(1).to(torch.int32).contiguous()

    apply_update_hidden_cache(
        current_hidden, persistent_cache, active_indices
    )

    return active_indices, active_mask

@triton.jit
def compute_sparse_attention(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    active_indices_ptr,
    n_active,
    seq_length,
    num_heads,
    head_dimension, # unused, but still good to have 
    softmax_scale,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    program_id = tl.program_id(0)
    offset_hb = tl.program_id(1) # offset for the head and the batches
    offset_head = offset_hb % num_heads
    offset_batch = offset_hb // num_heads

    tile_offsets = program_id * BLOCK_M + tl.arange(0, BLOCK_M)
    tile_mask = tile_offsets < n_active
    token_positions = tl.load(active_indices_ptr + tile_offsets, mask=tile_mask, other=0.0)

    new_q_ptr = q_ptr + offset_batch * stride_qb + token_positions[:, None] * stride_qs + offset_head * stride_qh + tl.arange(0, D_HEAD)[None, :] * stride_qd

    q = tl.load(new_q_ptr, mask=tile_mask[:, None], other=0.0).to(tl.float32)
    q = q * softmax_scale * 1.44269504

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    for i in range(0, seq_length, BLOCK_N):
        kv_offsets = i + tl.arange(0, BLOCK_N)
        kv_mask = kv_offsets < seq_length

        new_k_ptr = k_ptr + offset_batch * stride_kb + kv_offsets[:, None] * stride_ks + offset_head * stride_kh + tl.arange(0, D_HEAD)[None, :] * stride_kd
        keys = tl.load(new_k_ptr, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        new_v_ptr = v_ptr + offset_batch * stride_vb + kv_offsets[:, None] * stride_vs + offset_head * stride_vh + tl.arange(0, D_HEAD)[None, :] * stride_vd
        values = tl.load(new_v_ptr, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        qk = tl.dot(q, tl.trans(keys))
        qk = tl.where(kv_mask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + tl.dot(p, values)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij
    
    acc = acc / l_i[:, None]
    output_ptr += offset_batch * stride_ob + token_positions[:, None] * stride_os + offset_head * stride_oh + tl.arange(0, D_HEAD)[None, :] * stride_od

    tl.store(output_ptr, acc.to(tl.float16), mask=tile_mask[:, None])

def apply_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    active_indices: torch.Tensor,
    output: torch.Tensor = None,
):
    assert q.shape == k.shape == v.shape, \
        f"Q, K, V shape mismatch: {q.shape}, {k.shape}, {v.shape}"

    batch_size, seq_length, num_heads, head_dimension = q.shape

    assert head_dimension in {64, 128}, \
        f"head_dimension must be 64 of 128, got {head_dimension}"
    assert active_indices.dtype == torch.int32, \
        f"active_indices must be int32, got {active_indices.dtype}"
    assert active_indices.is_contiguous(), \
        f"active_indices must be a contiguous tensor"
    
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    n_active = active_indices.shape[0]

    if output is None:
        output = torch.zeros_like(q)
    
    if n_active == 0:
        return output
    
    softmax_scale = 1.0 / (head_dimension ** 0.5)
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(n_active, BLOCK_M), batch_size * num_heads)

    compute_sparse_attention[grid](
        q, k, v, output,
        active_indices,
        n_active,
        seq_length,
        num_heads,
        head_dimension,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        D_HEAD=head_dimension
    )

    return output