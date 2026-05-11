import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def token_similarity_check(
    current_hidden_ptr, # pointer to current hidden states [batch, seq_len, d_model]
    previous_hidden_ptr, # pointer to previous hidden states [batch, seq_len, d_model]
    freeze_counter_ptr, # pointer to integer freeze counters [batch, seq_len]
    active_mask_ptr, # pointer to output boolean mask [batch, seq_len]
    threshold, # cosine similarity threshold (e.g. 0.99)
    freeze_steps, # how many steps to freeze when triggered (e.g. 3)
    seq_len,
    d_model,
    BLOCK_D: tl.constexpr # block size over the d_model dimension
):
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    token_offset = (batch_id * seq_len + seq_id)

    current_hidden_ptr += token_offset * d_model
    previous_hidden_ptr += token_offset * d_model
    freeze_counter_ptr += token_offset
    active_mask_ptr += token_offset

    dot_product = 0.0
    norm_a_squared = 0.0
    norm_b_squared = 0.0

    offsets = tl.arange(0, BLOCK_D)
    for i in range(0, d_model, BLOCK_D):
        mask = (i + offsets) < d_model

        a = tl.load(current_hidden_ptr + i + offsets, mask=mask)
        b = tl.load(previous_hidden_ptr + i + offsets, mask=mask)

        dot_product += tl.sum(a * b)
        norm_a_squared += tl.sum(a * a)
        norm_b_squared += tl.sum(b * b)
    
    cosine_similarity = dot_product / (tl.sqrt(norm_a_squared * norm_b_squared) + 1e-8) # add a small epsilon for safety

    freeze_counter = tl.load(freeze_counter_ptr)

    active_mask = tl.where(freeze_counter > 0, 0,
                           tl.where(cosine_similarity > threshold, 0, 1))
    freeze_counter = tl.where(freeze_counter > 0, freeze_counter - 1,
                              tl.where(cosine_similarity > threshold, freeze_steps, freeze_counter))

    tl.store(active_mask_ptr, active_mask.to(tl.int1))
    tl.store(freeze_counter_ptr, freeze_counter)

@triton.jit
def sparse_attention_forward(
    q_ptr, k_ptr, v_ptr, output_ptr,
    active_indices_ptr,
    n_active,
    seq_len,
    n_heads,
    sm_scale,
    stride_qb, stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_ob, stride_os, stride_oh, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr
):
    program_id = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_h = off_hz % n_heads
    off_z = off_hz // n_heads

    active_idx_offsets = program_id * BLOCK_M + tl.arange(0, BLOCK_M)
    active_mask = active_idx_offsets < n_active
    actual_seq_positions = tl.load(active_indices_ptr + active_idx_offsets, mask=active_mask, other=0)

    q_ptrs = (q_ptr 
              + off_z * stride_qb
              + actual_seq_positions[:, None] * stride_qs
              + off_h * stride_qh
              + tl.arange(0, D_HEAD)[None, :] * stride_qd)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    q = tl.load(q_ptrs, mask=active_mask[:, None], other=0.0)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    for start_n in range(0, seq_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        k_ptrs = (k_ptr + off_z * stride_kb
                  + (start_n + offs_n[:, None]) * stride_ks
                  + off_h * stride_kh
                  + offs_d[None, :] * stride_kd)
        k = tl.load(k_ptrs, mask=(start_n + offs_n[:, None]) < seq_len, other=0.0)

        v_ptrs = (v_ptr + off_z * stride_vb
                  + (start_n + offs_n[:, None]) * stride_vs
                  + off_h * stride_vh
                  + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs, mask=(start_n + offs_n[:, None]) < seq_len, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * qk_scale
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    acc = acc / l_i[:, None]

    output_ptrs = (output_ptr 
              + off_z * stride_ob
              + actual_seq_positions[:, None] * stride_os
              + off_h * stride_oh
              + tl.arange(0, D_HEAD)[None, :] * stride_od)
    tl.store(output_ptrs, acc, mask=active_mask[:, None])

def sparse_attention(
        q: torch.Tensor, # [batch, seq_len, n_heads, d_head]
        k: torch.Tensor,
        v: torch.Tensor,
        active_mask: torch.Tensor,
        output: torch.Tensor
) -> torch.Tensor:
    # NOTE: currently assumes batch_size = 1
    active_indices = torch.nonzero(active_mask.squeeze(0), as_tuple=False).squeeze(1).to(torch.int32).contiguous()
    n_active = active_indices.shape[0]

    batch_size, seq_len, n_heads, d_head = q.shape
    sm_scale = 1.0 / (d_head ** 0.5)
    
    BLOCK_M = 16
    BLOCK_N = 32
    grid = (triton.cdiv(n_active, BLOCK_M), batch_size * n_heads)

    sparse_attention_forward[grid](
        q, k, v, output,
        active_indices,
        n_active,
        seq_len,
        n_heads,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_M,
        BLOCK_N,
        d_head
    )

    return output

@triton.jit
def update_hidden_cache(
    new_hidden_ptr, # newly computed hidden states [batch, seq_len, d_model]
    cached_hidden_ptr, # the cache to update in-place [batch, seq_len, d_model]
    active_indices_ptr, # which positions were active [n_active]
    n_active,
    d_model,
    stride_b, # batch stride (how many elements to skip per batch item)
    stride_s, # seq stride (how many elements to skip per token = d_model)
    BLOCK_D: tl.constexpr
):
    program_id = tl.program_id(0)
    off_hz = tl.program_id(1)

    if program_id >= n_active:
        return

    actual_seq_position = tl.load(active_indices_ptr + program_id)
    
    token_offset = off_hz * stride_b + actual_seq_position * stride_s
    new_base = new_hidden_ptr + token_offset
    cache_base = cached_hidden_ptr + token_offset

    for i in range(0, d_model, BLOCK_D):
        offsets = i + tl.arange(0, BLOCK_D)
        mask = offsets < d_model

        vals = tl.load(new_base + offsets, mask=mask)

        tl.store(cache_base + offsets, vals, mask=mask)

def update_hidden_cache_launcher(new_hidden, cached_hidden, active_indices, batch_size):
    n_active = active_indices.shape[0]
    d_model = new_hidden.shape[-1]
    BLOCK_D = min(triton.next_power_of_2(d_model), 256)

    grid = (n_active, batch_size)
    update_hidden_cache[grid](
        new_hidden, cached_hidden, active_indices,
        n_active, d_model,
        new_hidden.stride(0), new_hidden.stride(1),
        BLOCK_D=BLOCK_D
    )
    
def check_token_similarity(
        current_hidden: torch.Tensor, # [batch, seq_len, d_model]
        previous_hidden: torch.Tensor, # [batch, seq_len, d_model]
        freeze_counters: torch.Tensor, # [batch, seq_len] int32
        active_mask: torch.Tensor, # [batch, seq_len] int8, updated in-place
        threshold: float,
        freeze_steps: int,
):
    batch_size, seq_len, d_model = current_hidden.shape
    BLOCK_D = min(triton.next_power_of_2(d_model), 256)
    grid = (batch_size, seq_len)

    token_similarity_check[grid](
        current_hidden, previous_hidden,
        freeze_counters, active_mask,
        threshold, freeze_steps,
        seq_len, d_model,
        BLOCK_D=BLOCK_D
    )

def sparse_decoding_step(
        q: torch.Tensor, # [batch, seq_len, n_heads, d_head]
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor, # [batch, seq_len, n_heads, d_head] pre-allocated
        current_hidden: torch.Tensor, # [batch, seq_len, d_model] from this step's forward pass
        cached_hidden: torch.Tensor, # [batch, seq_len, d_model] persistent across steps
        freeze_counters: torch.Tensor, # [batch, seq_len] int32, persistent across steps
        active_mask: torch.Tensor, # [batch, seq_len] int8, updated each step
        threshold: float = 0.99,
        freeze_steps: int = 3,
) -> torch.Tensor:
    check_token_similarity(
        current_hidden, cached_hidden,
        freeze_counters, active_mask,
        threshold, freeze_steps
    )

    sparse_attention(q, k, v, active_mask, output)

    active_indices = (torch.nonzero(active_mask.squeeze(0), as_tuple=False)
                      .squeeze(1)
                      .to(torch.int32)
                      .contiguous())
    update_hidden_cache_launcher(
        current_hidden, cached_hidden, active_indices, back_size=q.shape[0]
    )

    return output