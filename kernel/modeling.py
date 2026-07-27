"""
Fast-dLLM v2 / Qwen2.5 Model Definition.

Defines the transformer architecture (attention, MLP, decoder layer, full model, causal LM head) used by Fast-dLLM v2, 
including the KV block eviction and sparse-attention hooks that BlockEvictionScheduler drives. This is a HuggingFace-style modeling file: it follows the same structure as transformers' own Qwen2 implementation, 
with two additions specific to this project. First, block-diffusion training and generation support (block_past_key_values, replace_position, the noising logic in Fast_dLLM_QwenForCausalLM.forward, and the .generate() sampling loop). 
Second, KV block eviction hooks inside Fast_dLLM_QwenAttention.forward, which BlockEvictionScheduler (block_eviction_scheduler.py) arms and disarms to keep only the top-K prefix blocks in view for later generation steps.

Interface contract for the eviction hooks (never changes without updating block_eviction_scheduler.py in lockstep): 
self._capture_attn_scores is a bool that, when True, makes forward() also compute and store explicit softmax attention weights for block-scoring purposes, without changing the actual output (still computed via SDPA). 
self._topk_token_indices is a LongTensor or None that, when set, makes forward() slice the prefix K/V (and the attention mask) down to only these token positions before concatenating with the current block. 
self._use_sparse_kernel is a bool that, when True (and a block-aligned selection is available), makes forward() call the Triton sparse_attn_merge kernel instead of the gather + cat + SDPA fallback path for the small-step attention computation.
"""

from typing import Callable, Optional, Union
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from transformers.generation.utils import GenerateDecoderOnlyOutput
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging
from kernel.configuration import Fast_dLLM_QwenConfig
from kernel.block_sparse_attn import sparse_attn_merge
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

logger = logging.get_logger(__name__)

@dataclass
class CausalLMOutputWithPastAndBlockCache(CausalLMOutputWithPast):
    """
    CausalLMOutputWithPast, extended with the block-diffusion sub-block KV cache.
    """
    block_past_key_values: Optional[Cache] = None

@dataclass
class BaseModelOutputWithPastAndBlockCache(BaseModelOutputWithPast):
    """
    BaseModelOutputWithPast, extended with the block-diffusion sub-block KV cache.
    """
    block_past_key_values: Optional[Cache] = None

@torch.compile(fullgraph = True, mode = "max-autotune-no-cudagraphs")
def fused_flex_attention(q, k, v, mask = None):
    """
    Compiled wrapper around flex_attention, used for the training-time block-diffusion attention pattern (block_diff_mask). q, k, and v are [B, H, S, D] query/key/value tensors, 
    and mask is a block mask built by create_block_mask(), or None. Returns a [B, H, S, D] attention output.
    """
    return flex_attention(q, k, v, block_mask = mask, enable_gqa = True)

def block_diff_mask(b, h, q_idx, kv_idx, block_size = None, n = None):
    """
    Constructs the specialized block diffusion attention mask for training, composed of three masks: the Block Diagonal Mask (M_BD) for self-attention within noised blocks, 
    the Offset Block-Causal Mask (M_OBC) for cross-attention to conditional context, and the Block-Causal Mask (M_BC) for attention that updates x0.

    b and h are batch and head indices (ignored: the mask is the same for every batch element and head). q_idx and kv_idx are query and key position indices. 
    block_size is the tokens per block, defining the block structure, and n is the length of one half of the doubled training sequence (the sequence is [noisy_half; clean_half], each of length n).

    Returns a boolean attention mask (True = attend, False = block).
    """
    # Indicate whether token belongs to xt (noisy half) or x0 (clean half).
    x0_flag_q = (q_idx >= n)
    x0_flag_kv = (kv_idx >= n)

    # Compute block indices, offsetting into the correct half first.
    block_q = torch.where(
        x0_flag_q == 1,
        (q_idx - n) // block_size,
        q_idx // block_size,
    )
    block_kv = torch.where(
        x0_flag_kv == 1,
        (kv_idx - n) // block_size,
        kv_idx // block_size,
    )

    # 1. Block Diagonal Mask (M_BD): same block, same half.
    block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

    # 2. Offset Block-Causal Mask (M_OBC): a noisy-half query attends to earlier clean-half blocks (its conditioning context).
    offset_block_causal = (
        (block_q > block_kv)
        & (x0_flag_kv == 1)
        & (x0_flag_q == 0)
    )

    # 3. Block-Causal Mask (M_BC): a clean-half query attends causally to clean-half blocks (standard next-token prediction over x0).
    block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

    # 4. Combine: a position may attend if any of the three masks permit it.
    return block_diagonal | offset_block_causal | block_causal

def eval_block_diff_mask(q_idx, kv_idx, block_size = None):
    """
    Simplified block-causal mask used at evaluation time (no noisy/clean halves: generation only ever sees "clean" committed tokens plus the current block). 
    q_idx and kv_idx are query and key position indices, and block_size is the tokens per block.

    Returns a boolean attention mask (True = attend, False = block): a query may attend to any key whose block index is not later than its own.
    """
    block_q = q_idx // block_size
    block_kv = kv_idx // block_size
    return block_q >= block_kv

class Fast_dLLM_QwenMLP(nn.Module):
    """
    Standard SwiGLU MLP block: down_proj(act(gate_proj(x)) * up_proj(x)).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias = False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias = False)
        self.act_fn = ACT2FN[config.hidden_act]
        self._fused = False

    def fuse_projections(self):
        """
        Concatenates gate_proj and up_proj's weights into a single matrix, so forward() can issue one larger matmul instead of two smaller ones. 
        gate_proj and up_proj are left in place, untouched, purely so state_dict loading (from_pretrained) keeps working exactly as before: this method is meant to be called once, 
        after the checkpoint is already loaded, not at __init__ time. Mathematically exact: each output element is a sum over the same input row against the same weight row it always was, 
        so cat-then-matmul only regroups which output rows are computed in the same kernel launch, changing nothing about the arithmetic. Idempotent: safe to call more than once.

        Returns nothing; sets self._gate_up_weight and self._fused as a side effect.
        """
        if self._fused:
            return
        # dim = 0 concatenates along the output-feature axis: nn.Linear weight shape is [out_features, in_features], so this stacks gate_proj's
        # intermediate_size output rows directly above up_proj's, with no effect on how any individual output row's dot product is computed.
        self.register_buffer(
            "_gate_up_weight",
            torch.cat([self.gate_proj.weight, self.up_proj.weight], dim = 0),
            persistent = False,
        )
        # forward() never reads gate_proj/up_proj again once _fused = True: the original weights were only being kept alive for the cat()
        # above. Left as-is, that's a permanent doubling of this layer's gate + up weight memory (fused copy AND originals, both resident).
        # Replacing the underlying storage with an empty tensor releases it back to the CUDA allocator; the submodules stay in place
        # structurally (so anything that still checks for their existence doesn't break), just holding nothing.
        self.gate_proj.weight.data = torch.empty(0, device = self.gate_proj.weight.device, dtype = self.gate_proj.weight.dtype)
        self.up_proj.weight.data = torch.empty(0, device = self.up_proj.weight.device, dtype = self.up_proj.weight.dtype)
        self._fused = True

    def forward(self, x):
        if self._fused:
            gate_up = F.linear(x, self._gate_up_weight)
            gate, up = gate_up.split(self.intermediate_size, dim = -1)
            down_proj = self.down_proj(self.act_fn(gate) * up)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def rotate_half(x):
    """
    Rotates half the hidden dims of the input (the standard RoPE rotation helper: swaps and negates the two halves of the last dimension). 
    x is a [..., D] tensor, D even. Returns a [..., D] tensor: concat(-x[..., D/2:], x[..., :D/2]).
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids = None, unsqueeze_dim = 1):
    """
    Applies Rotary Position Embedding to the query and key tensors. q and k are the query and key tensors, and cos and sin are the cosine and sine parts of the rotary embedding. 
    position_ids is deprecated and unused. unsqueeze_dim is the dimension along which to unsqueeze cos[position_ids] and sin[position_ids] so they broadcast against q and k. 
    cos/sin have shape [batch_size, seq_len, head_dim]; if q and k have shape [batch_size, heads, seq_len, head_dim], use unsqueeze_dim = 1 (the default), 
    and if q and k instead have shape [batch_size, seq_len, heads, head_dim], use unsqueeze_dim = 2.

    Returns (q_embed, k_embed): the query and key tensors after rotation.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to torch.repeat_interleave(x, dim = 1, repeats = n_rep). Expands GQA key/value heads so each is repeated n_rep times to match the number of query heads: 
    (batch, num_key_value_heads, seqlen, head_dim) becomes (batch, num_attention_heads, seqlen, head_dim). hidden_states is the [B, H_kv, S, D] key or value tensor, 
    and n_rep is the number of query heads served per KV head (H_q // H_kv). Returns a [B, H_kv * n_rep, S, D] tensor.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Fast_dLLM_QwenAttention(nn.Module):
    """
    Multi-headed GQA attention (Qwen2.5-style), extended with block-diffusion cache handling (block_past_key_values, replace_position), KV block eviction hooks 
    (_topk_token_indices, _use_sparse_kernel), and attention-weight capture for block scoring (_capture_attn_scores). 
    See the module-level docstring for the eviction-hook interface contract.
    """

    def __init__(self, config: Fast_dLLM_QwenConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias = True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias = True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias = True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias = False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        self._fused = False

    def fuse_projections(self):
        """
        Concatenates q_proj, k_proj, and v_proj's weights and biases into a single matrix/vector, so forward() can issue one larger matmul instead of three smaller ones. 
        The original q_proj/k_proj/v_proj submodules are left in place, untouched, purely so state_dict loading (from_pretrained) keeps working exactly as before: 
        this method is meant to be called once, after the checkpoint is already loaded, not at __init__ time. Mathematically exact for the same reason as Fast_dLLM_QwenMLP.fuse_projections: 
        concatenating output rows and splitting them back apart afterward doesn't change how any individual output element's dot product is computed. Idempotent: safe to call more than once.

        Returns nothing; sets self._qkv_weight, self._qkv_bias, self._q_size, self._kv_size, and self._fused as a side effect.
        """
        if self._fused:
            return
        self.register_buffer(
            "_qkv_weight",
            torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim = 0),
            persistent = False,
        )
        self.register_buffer(
            "_qkv_bias",
            torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], dim = 0),
            persistent = False,
        )
        self._q_size = self.q_proj.out_features
        self._kv_size = self.k_proj.out_features
        # Same reasoning as Fast_dLLM_QwenMLP.fuse_projections: free the original q/k/v weight and bias storage now that forward() will
        # never read it again, rather than leaving both copies resident.
        for layer in (self.q_proj, self.k_proj, self.v_proj):
            layer.weight.data = torch.empty(0, device = layer.weight.device, dtype = layer.weight.dtype)
            layer.bias.data = torch.empty(0, device = layer.bias.device, dtype = layer.bias.dtype)
        self._fused = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        """
        Computes attention for one decoder layer, handling the persistent prefix cache, the block-diffusion sub-block cache, and the KV eviction/sparse-kernel hooks. 
        hidden_states is the [B, S, hidden_size] layer input, and position_embeddings is the (cos, sin) tuple from the rotary embedding module. 
        attention_mask is an additive float mask, or None (no masking needed, e.g. a use_block_cache cache-hit call). past_key_value is the persistent (committed) prefix KV cache, 
        and cache_position is the absolute token positions being written this call. update_past_key_values is True on a "commit" forward (writes into past_key_value) 
        and False on a "small step" forward (reads from it without committing). block_past_key_values is the block-diffusion sub-block KV cache, used only when use_block_cache = True upstream, 
        and replace_position is the sub-block start offset to overwrite within block_past_key_values, when reusing rather than rebuilding it. **kwargs is forwarded to the SDPA attention interface.

        Returns attn_output, the [B, S, hidden_size] attention output after o_proj.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self._fused:
            qkv = F.linear(hidden_states, self._qkv_weight, self._qkv_bias)
            q_flat, k_flat, v_flat = qkv.split([self._q_size, self._kv_size, self._kv_size], dim = -1)
            query_states = q_flat.view(hidden_shape).transpose(1, 2)
            key_states = k_flat.view(hidden_shape).transpose(1, 2)
            value_states = v_flat.view(hidden_shape).transpose(1, 2)
        else:
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if self.training:
            # Training input is [noisy_half; clean_half] concatenated along the sequence dimension. Each half gets its own RoPE application
            # since they represent independent absolute positions.
            q_1 = query_states[:, :, :query_states.shape[2] // 2]
            q_2 = query_states[:, :, query_states.shape[2] // 2:]
            k_1 = key_states[:, :, :key_states.shape[2] // 2]
            k_2 = key_states[:, :, key_states.shape[2] // 2:]
            q_1, k_1 = apply_rotary_pos_emb(q_1, k_1, cos, sin)
            q_2, k_2 = apply_rotary_pos_emb(q_2, k_2, cos, sin)
            query_states = torch.cat((q_1, q_2), dim = -2)
            key_states = torch.cat((k_1, k_2), dim = -2)
        else:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if block_past_key_values is not None:
            if len(block_past_key_values) <= self.layer_idx:
                # First touch this layer's block cache: initialize it via the standard Cache.update() path.
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = block_past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            else:
                # Reuse the existing block cache, overwriting only the sub-block at replace_position with freshly computed KV.
                block_cache_key_states = block_past_key_values[self.layer_idx][0]
                block_cache_value_states = block_past_key_values[self.layer_idx][1]

                block_cache_key_states[:, :, replace_position:replace_position + key_states.shape[2]] = key_states
                block_cache_value_states[:, :, replace_position:replace_position + value_states.shape[2]] = value_states
                key_states = block_cache_key_states
                value_states = block_cache_value_states

        # Track the current-block KV width at this point, before the prefix cat. This is 32 in all relevant small-step paths (both full-block and sub-block
        # calls, since sub-block calls expand into the full block cache above). Used by the sparse kernel gate below to avoid checking query width.
        self._cur_block_kv_width = key_states.shape[-2]

        if past_key_value is not None:
            if update_past_key_values:
                # Commit step: write this block's KV into the persistent cache.
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            elif len(past_key_value) > self.layer_idx:
                # Small step: read the persistent prefix cache without committing, optionally pruned to the eviction selection.
                prefix_k = past_key_value[self.layer_idx][0]
                prefix_v = past_key_value[self.layer_idx][1]

                # Top-K block loading: if armed, slice prefix to selected token positions only.
                _idx = getattr(self, '_topk_token_indices', None)
                _armed_for_len = getattr(self, '_topk_armed_for_len', None)
                self._sparse_block_indices = None
                self._sparse_prefix_k_full = None
                self._sparse_prefix_v_full = None

                # Validity check: idx is only trustworthy if it was armed for EXACTLY this call's prefix length. This used to be checked
                # with `_idx[_idx < prefix_k.shape[-2]]`, a boolean-mask index whose output size is data-dependent: PyTorch must sync
                # device-to-host to learn the count of valid entries before allocating the result, every layer, every small-step call.
                # Confirmed via profiling: aten::index accounted for 43% of total CPU time in one trace despite ~1.5% of GPU compute,
                # the signature of a sync stall rather than real work.

                # _armed_for_len is a plain Python int recorded by BlockEvictionScheduler at the moment idx was built, from
                # the same prefix_kv_len idx's positions are bounded by. If it matches this call's prefix_k.shape[-2] (also a plain
                # int, no sync), every value in idx is guaranteed in-bounds by construction: no per-call data-dependent check needed.
                # A cross-call state leak (root-caused via the [eviction debug] print this check replaces: a previous generation's
                # armed idx bled into a new, shorter-prefix generation) is now prevented at the source in BlockEvictionScheduler, but
                # this comparison is kept as a cheap, sync-free safety net: on any future mismatch we skip eviction for this one call
                # (full, unpruned prefix) rather than trusting stale data.
                idx_is_valid = _idx is not None and _armed_for_len == prefix_k.shape[-2]
                if _idx is not None and not idx_is_valid and not getattr(self, '_stale_idx_warned', False):
                    self._stale_idx_warned = True
                    print(
                        f"[Eviction debug] Layer {self.layer_idx}: idx armed for "
                        f"prefix length {_armed_for_len}, but this call's prefix_k.shape[-2] "
                        f"= {prefix_k.shape[-2]}. Skipping eviction for this call.",
                        flush = True,
                    )
                if idx_is_valid:
                    if len(_idx) < prefix_k.shape[-2]:
                        # Sparse kernel path (only actually taken when use_sparse_kernel = True downstream): stash the full
                        # prefix tensors and the block-aligned token indices so the attention computation can call the Triton
                        # kernel directly, with zero gather/cat of the prefix cache. Token indices are block-aligned (multiples
                        # of BLOCK_N = 32) by construction in BlockEvictionScheduler, with the tail appended
                        # separately: block_indices below derives the unique block ids from those token positions.
                        self._sparse_prefix_k_full = prefix_k
                        self._sparse_prefix_v_full = prefix_v
                        self._sparse_block_indices = (_idx // 32).unique(sorted = True).to(torch.int32)

                        # Fallback path (used whenever the sparse kernel path is not taken below): actually gather the selected
                        # tokens only.
                        prefix_len = prefix_k.shape[-2]
                        prefix_k = prefix_k[:, :, _idx, :]
                        prefix_v = prefix_v[:, :, _idx, :]

                        # Slice the attention mask to match the pruned prefix length. Mask shape: [..., prefix_len + block_cache_len].
                        # Keep prefix columns at _idx positions plus all block-cache columns.
                        if attention_mask is not None:
                            prefix_mask = attention_mask[..., :prefix_len][..., _idx]
                            block_mask = attention_mask[..., prefix_len:]
                            attention_mask = torch.cat([prefix_mask, block_mask], dim = -1)

                    # else: all tokens selected (K >= N): skip the gather to preserve memory layout and keep output bit-identical to
                    # baseline.
                key_states = torch.cat((prefix_k, key_states), dim = -2)
                value_states = torch.cat((prefix_v, value_states), dim = -2)

        if self.training:
            attn_output = fused_flex_attention(query_states, key_states, value_states, mask = attention_mask)
            attn_output = attn_output.transpose(1, 2).contiguous()

        elif getattr(self, '_capture_attn_scores', False):
            # Score-capture path: use SDPA for the actual output (numerically identical to baseline so committed KV is not perturbed), then
            # compute attention weights separately in a no_grad block purely for block-selection bookkeeping.
            attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout = 0.0,
                scaling = self.scaling,
            )

            # GQA expansion for manual weight computation (capture only, no grad).
            with torch.no_grad():
                k_exp = key_states.repeat_interleave(self.num_key_value_groups, dim = 1)
                v_exp = value_states.repeat_interleave(self.num_key_value_groups, dim = 1)
                scores = torch.matmul(query_states, k_exp.transpose(-1, -2)) * self.scaling
                if attention_mask is not None:
                    scores = scores + attention_mask
                attn_weights = torch.softmax(scores.float(), dim = -1).to(query_states.dtype)

            self._last_attn_weights = attn_weights.detach() # [B, H, Q, KV]
            self._last_token_scores = attn_weights.mean(dim = (0, 1, 2)).detach() # [kv_len]

        else:
            # The sparse kernel path assumes the current query block is a full generation block (block_size tokens), with an eval_mask built for
            # exactly that width. When use_block_cache = True, Fast-dLLM issues additional sub-block forward calls of width small_block_size
            # (e.g. 8 tokens) with replace_position set, and on cache-hit calls attention_mask is set to None entirely. Neither of these
            # match the sparse kernel's assumptions, so we only take the sparse path when the query width matches the configured block
            # size: otherwise we fall back to the proven gather + cat + SDPA path, which handles all of Fast-dLLM's call shapes correctly
            # already.
            
            # Sparse kernel gate. We fire when:
            # - use_sparse_kernel is armed
            # - we have a sparse block index selection
            # - the current block's KV is 32 tokens wide (true for all small-step forward calls, including 8-token sub-block
            #   use_block_cache calls, since those expand into the full block cache before this point)
            # - attention_mask is either None (use_block_cache cache-hit, no mask needed for current block) or shaped correctly for prefix+32
            cur_block_kv_width = getattr(self, '_cur_block_kv_width', key_states.shape[-2])
            prefix_full_k = self._sparse_prefix_k_full if hasattr(self, '_sparse_prefix_k_full') else None

            # The eviction branch above already sliced attention_mask to match the gathered (sliced) prefix. So mask_ok checks whether
            # the mask is either absent or sized for the sliced prefix plus the current block.
            sliced_prefix_len = key_states.shape[-2] - cur_block_kv_width
            mask_ok = (
                attention_mask is None
                or attention_mask.shape[-1] == sliced_prefix_len + cur_block_kv_width
            )
            # Microbenchmark finding (bench: kernel/benchmark_sparse_kernel.py): the sparse kernel beats the SDPA fallback by ~1.23x at S_q = 8
            # (sub-block reuse calls, query narrower than the block), but loses to it by ~0.94x at S_q = 32 (full-block rebuild calls,
            # query width == cur_block_kv_width). Restricting the sparse path to genuine sub-block calls is a strict improvement regardless
            # of the true reuse/rebuild call mix in a given generation run: rebuild calls fall back to the already-correct, already-faster
            # SDPA path instead of paying the sparse kernel's fixed overhead for no benefit.
            is_subblock_call = query_states.shape[-2] < cur_block_kv_width
            use_sparse = (
                getattr(self, '_use_sparse_kernel', False)
                and getattr(self, '_sparse_block_indices', None) is not None
                and cur_block_kv_width == 32
                and prefix_full_k is not None
                and mask_ok
                and is_subblock_call
            )

            if getattr(self, '_use_sparse_kernel', False):
                if not hasattr(self, '_sparse_path_hits'):
                    self._sparse_path_hits = 0
                    self._sparse_path_misses = 0
                    self._sparse_path_rebuild_misses = 0
                if use_sparse:
                    self._sparse_path_hits += 1
                else:
                    self._sparse_path_misses += 1
                    # Specifically count misses caused by a full-block rebuild call (query width == block width) that were otherwise
                    # eligible, to directly measure how often the reuse/rebuild gate above is the deciding factor, versus other gate
                    # conditions (armed state, mask shape, etc).
                    if (
                        getattr(self, '_sparse_block_indices', None) is not None
                        and cur_block_kv_width == 32
                        and prefix_full_k is not None
                        and mask_ok
                        and not is_subblock_call
                    ):
                        self._sparse_path_rebuild_misses += 1

            if use_sparse:
                # Sparse kernel path: prefix attention via Triton (reads only the K selected blocks from HBM, no gather/cat), current
                # block via manual softmax, combined with a Flash Decoding merge. cur_k/cur_v are the last cur_block_kv_width = 32
                # tokens of key_states/value_states (after the prefix cat above).
                cur_k = key_states[:, :, -cur_block_kv_width:, :]
                cur_v = value_states[:, :, -cur_block_kv_width:, :]
                attn_output = sparse_attn_merge(
                    query_states = query_states,
                    prefix_k = self._sparse_prefix_k_full,
                    prefix_v = self._sparse_prefix_v_full,
                    block_indices = self._sparse_block_indices.unsqueeze(0).expand(
                        query_states.shape[0], -1
                    ).contiguous(),
                    cur_k = cur_k,
                    cur_v = cur_v,
                    attention_mask = attention_mask,
                    scaling = self.scaling,
                    num_kv_groups = self.num_key_value_groups,
                ).transpose(1, 2).contiguous()
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS["sdpa"]

                attn_output, attn_weights = attention_interface(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    is_causal = False,
                    dropout = 0.0 if not self.training else self.attention_dropout,
                    scaling = self.scaling,
                    sliding_window = self.sliding_window, # Main diff with Llama
                    **kwargs,
                )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

@use_kernel_forward_from_hub("RMSNorm")
class Fast_dLLM_QwenRMSNorm(nn.Module):
    """
    RMSNorm (equivalent to T5LayerNorm): normalizes by root-mean-square, no mean subtraction.
    """

    def __init__(self, hidden_size, eps = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps = {self.variance_epsilon}"

class Fast_dLLM_QwenDecoderLayer(GradientCheckpointingLayer):
    """
    One transformer block: self-attention followed by an MLP, each with a residual connection and a preceding RMSNorm.
    """

    def __init__(self, config: Fast_dLLM_QwenConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Fast_dLLM_QwenAttention(config = config, layer_idx = layer_idx)

        self.mlp = Fast_dLLM_QwenMLP(config)
        self.input_layernorm = Fast_dLLM_QwenRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = Fast_dLLM_QwenRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None, # Necessary, but kept here for BC.
        update_past_key_values: Optional[bool] = False,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs
    ) -> tuple[torch.Tensor]:
        """
        Runs one decoder layer's self-attention and MLP. See Fast_dLLM_QwenAttention.forward for the eviction- and block-cache-related arguments; 
        the rest are standard transformer decoder layer inputs. Returns hidden_states, the [B, S, hidden_size] layer output.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states = hidden_states,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_value = past_key_value,
            use_cache = use_cache,
            cache_position = cache_position,
            position_embeddings = position_embeddings,
            update_past_key_values = update_past_key_values,
            use_block_cache = use_block_cache,
            block_past_key_values = block_past_key_values,
            replace_position = replace_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class Fast_dLLM_QwenPreTrainedModel(PreTrainedModel):
    """
    Shared HuggingFace PreTrainedModel boilerplate: config class, weight init, supported backends/caches.
    """

    config_class = Fast_dLLM_QwenConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Fast_dLLM_QwenDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Fast_dLLM_QwenDecoderLayer,
        "attentions": Fast_dLLM_QwenAttention,
    }

    def _init_weights(self, module):
        """
        Standard HF weight initialization: normal for Linear/Embedding, ones for RMSNorm.
        """
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean = 0.0, std = std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean = 0.0, std = std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Fast_dLLM_QwenRMSNorm):
            module.weight.data.fill_(1.0)

class Fast_dLLM_QwenRotaryEmbedding(nn.Module):
    """
    Computes RoPE cos/sin tables for a given set of position ids.
    """

    def __init__(self, config: Fast_dLLM_QwenConfig, device = None):
        super().__init__()
        # BC: "rope_type" was originally "type".
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type not in ROPE_INIT_FUNCTIONS:
            # This transformers build omits "default"; register standard RoPE inline.
            def _default_rope_init(config, device = None, **kwargs):
                head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
                theta = getattr(config, "rope_theta", 10000.0)
                inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype = torch.float32) / head_dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent = False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update # Power user: used with advanced RoPE types (e.g. dynamic rope).
    def forward(self, x, position_ids):
        """
        x is any tensor on the target device/dtype (used only for its device and dtype, not its values), and position_ids is a [B, S] tensor of absolute token positions. 
        Returns (cos, sin), each [B, S, head_dim], cast to x's dtype.
        """
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type = device_type, enabled = False): # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim = -1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype = x.dtype), sin.to(dtype = x.dtype)

class Fast_dLLM_QwenModel(Fast_dLLM_QwenPreTrainedModel):
    """
    The decoder stack: embeddings, N decoder layers, final norm. No LM head (see Fast_dLLM_QwenForCausalLM).
    """

    def __init__(self, config: Fast_dLLM_QwenConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.bd_size = config.bd_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Fast_dLLM_QwenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Fast_dLLM_QwenRMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.rotary_emb = Fast_dLLM_QwenRotaryEmbedding(config = config)
        self.gradient_checkpointing = True

        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def eval_mask(self, seqlen, block_size, cache_seq_len):
        """
        Builds the evaluation-time block-causal additive attention mask for the current forward call. seqlen is the number of new query tokens this call, 
        block_size is the tokens per generation block, and cache_seq_len is the number of already-committed prefix tokens.

        Returns a [seqlen, seqlen + cache_seq_len] float additive mask (0.0 = attend, -inf = blocked), suitable for the SDPA interface.
        """
        q_indices = torch.arange(seqlen) + cache_seq_len
        k_indices = torch.arange(seqlen + cache_seq_len)
        bool_mask = eval_block_diff_mask(
            q_idx = q_indices[:, None],
            kv_idx = k_indices[None, :],
            block_size = block_size,
        )
        # sdpa_attention_forward requires a float additive mask (0.0 = attend, -inf = block).
        float_mask = torch.zeros_like(bool_mask, dtype = torch.float32)
        float_mask = float_mask.masked_fill(~bool_mask, float('-inf'))
        return float_mask

    def gen_mask(self, seqlen, block_size, B, H):
        """
        Builds the training-time block-diffusion attention mask via flex_attention's block-mask compiler. seqlen is the length of one half (noisy or clean) of the training sequence, 
        block_size is the tokens per generation block, and B and H are the batch size and number of attention heads. Returns a compiled BlockMask covering the doubled sequence length (2 * seqlen).
        """
        mask = create_block_mask(
            partial(block_diff_mask, block_size = block_size, n = seqlen),
            B = B, H = H, Q_LEN = seqlen * 2, KV_LEN = seqlen * 2,
        )
        return mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        **kwargs
    ) -> BaseModelOutputWithPast:
        """
        Runs the decoder stack. Exactly one of input_ids / inputs_embeds must be given. labels is required when self.training 
        (drives the training-time attention mask construction) and is unused at inference time. attention_mask, position_ids, and cache_position are standard transformer inputs, 
        auto-computed when None (see body). past_key_values is the persistent prefix KV cache, and use_cache controls whether to create/use past_key_values. update_past_key_values, 
        block_size, use_block_cache, block_past_key_values, and replace_position are block-diffusion generation controls; see Fast_dLLM_QwenAttention.forward.

        Returns a BaseModelOutputWithPastAndBlockCache with last_hidden_state, past_key_values (if use_cache), and block_past_key_values (if use_block_cache).
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if use_block_cache and block_past_key_values is None:
            block_past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            if self.training:
                # Training input is doubled (noisy half + clean half); only
                # the first half's length worth of new positions is written.
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1] // 2, device = inputs_embeds.device
                )
            else:
                if use_block_cache:
                    block_start_position = past_seen_tokens + replace_position if replace_position is not None else past_seen_tokens
                    cache_position = torch.arange(
                        block_start_position, block_start_position + inputs_embeds.shape[1], device = inputs_embeds.device
                    )
                else:
                    cache_position = torch.arange(
                        past_seen_tokens,
                        past_seen_tokens + inputs_embeds.shape[1] if not self.training else inputs_embeds.shape[1] // 2,
                        device = inputs_embeds.device,
                    )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if self.training:
            attention_mask = self.gen_mask(labels.shape[1], self.bd_size, labels.shape[0], self.config.num_attention_heads).to(device = inputs_embeds.device)
        else:
            if use_block_cache and block_past_key_values.get_seq_length() != 0:
                # Cache-hit sub-block call: no mask needed, the block cache
                # already only contains valid positions.
                attention_mask = None
            else:
                _kv_len = (past_key_values.get_kv_seq_length() if hasattr(past_key_values, 'get_kv_seq_length') else past_key_values.get_seq_length()) if past_key_values is not None else 0
                attention_mask = self.eval_mask(input_ids.shape[1], block_size, _kv_len).to(device = inputs_embeds.device, dtype = inputs_embeds.dtype)

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across the decoder layers.
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask = attention_mask,
                position_ids = position_ids,
                past_key_value = past_key_values,
                use_cache = use_cache,
                cache_position = cache_position,
                position_embeddings = position_embeddings,
                update_past_key_values = update_past_key_values,
                use_block_cache = use_block_cache,
                block_past_key_values = block_past_key_values,
                replace_position = replace_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPastAndBlockCache(
            last_hidden_state = hidden_states,
            past_key_values = past_key_values if use_cache else None,
            block_past_key_values = block_past_key_values if use_block_cache else None,
        )

def fuse_all_projections(model):
    """
    Calls fuse_projections() on every layer's attention (QKV) and MLP (gate+up) modules. Meant to be called once, right after loading a checkpoint via from_pretrained: 
    fusing before the checkpoint is loaded would fuse zero-initialized weights instead of the real ones. model is a Fast_dLLM_QwenForCausalLM instance (or anything exposing 
    .model.layers with the standard decoder layer structure). Returns the same model instance, with every layer's projections fused in place.
    """
    for layer in model.model.layers:
        layer.self_attn.fuse_projections()
        layer.mlp.fuse_projections()
    # Each layer's fuse_projections() frees the original (now-unused) weight storage, but PyTorch's caching allocator doesn't necessarily hand that
    # memory back to CUDA immediately: it's available for reuse within this process either way, but empty_cache() here makes the actual freed
    # state visible (e.g. in nvidia-smi) rather than left ambiguous. One-time cost, right after model load, not on any hot path.
    torch.cuda.empty_cache()
    return model

class Fast_dLLM_QwenForCausalLM(Fast_dLLM_QwenPreTrainedModel, GenerationMixin):
    """
    Fast_dLLM_QwenModel plus an LM head, block-diffusion training noise injection, and the block-diffusion .generate() loop.
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Fast_dLLM_QwenModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)

        # Initialize weights and apply final processing.
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        update_past_key_values: Optional[bool] = False,
        block_size: Optional[int] = 32,
        use_block_cache: Optional[bool] = False,
        block_past_key_values: Optional[Cache] = None,
        replace_position: Optional[int] = None,
        mask_id: Optional[int] = 151665,
        **kwargs
    ) -> CausalLMOutputWithPastAndBlockCache:
        """
        Runs the full causal LM forward pass. See Fast_dLLM_QwenModel.forward for shared arguments. labels, when self.training, 
        drives the block-diffusion noising procedure below (masking + complementary masking, doubling the effective batch) and the loss computation. 
        logits_to_keep is how many trailing positions to compute logits for (0 or an int slice count, or an explicit index tensor). mask_id is the token id used to represent 
        an unfilled position during training-time noising.

        Returns a CausalLMOutputWithPastAndBlockCache with loss (if labels given), logits, past_key_values, hidden_states, attentions, and block_past_key_values.
        """
        if self.training:
            # Block-diffusion training noise injection: for each block, mask a random subset of positions (per a per-sample noise level t)
            # to build x_t, and build a complementary example that masks exactly the other positions: so every position is trained
            # as both "visible context" and "masked target" across the two halves of the doubled batch.
            original_labels = labels.clone()
            original_input_ids = input_ids.clone()

            noisy_input_ids = input_ids.clone()

            input_ids = input_ids.reshape(input_ids.shape[0] * input_ids.shape[1] // self.model.bd_size, self.model.bd_size)
            b, l = input_ids.shape
            t = torch.rand((b,), device = input_ids.device)
            eps = 1e-3
            p_mask = (1 - eps) * t + eps
            p_mask = p_mask[:, None].repeat(1, l)

            mask_indices = torch.rand((b, l), device = input_ids.device) < p_mask
            x_t = torch.where(mask_indices, mask_id, input_ids).reshape(labels.shape)
            noisy_input_ids[labels != -100] = x_t[labels != -100]
            mask = (noisy_input_ids != mask_id)
            labels[mask] = -100
            input_ids = torch.cat([noisy_input_ids, input_ids.reshape(labels.shape)], dim = 1)

            complementary_noisy_input_ids = original_input_ids.clone()
            complementary_labels = original_labels.clone()

            complementary_input_ids = original_input_ids.reshape(original_input_ids.shape[0] * original_input_ids.shape[1] // self.model.bd_size, self.model.bd_size)

            complementary_mask_indices = ~mask_indices
            complementary_x_t = torch.where(complementary_mask_indices, mask_id, complementary_input_ids).reshape(labels.shape)
            complementary_noisy_input_ids[complementary_labels != -100] = complementary_x_t[complementary_labels != -100]
            complementary_mask = (complementary_noisy_input_ids != mask_id)
            complementary_labels[complementary_mask] = -100
            complementary_input_ids = torch.cat([complementary_noisy_input_ids, complementary_input_ids.reshape(complementary_labels.shape)], dim = 1)

            input_ids = torch.cat([input_ids, complementary_input_ids], dim = 0)
            labels = torch.cat([labels, complementary_labels], dim = 0)

        outputs: BaseModelOutputWithPastAndBlockCache = self.model(
            input_ids = input_ids,
            labels = labels,
            attention_mask = attention_mask,
            position_ids = position_ids,
            past_key_values = past_key_values,
            inputs_embeds = inputs_embeds,
            use_cache = use_cache,
            cache_position = cache_position,
            update_past_key_values = update_past_key_values,
            block_size = block_size,
            use_block_cache = use_block_cache,
            block_past_key_values = block_past_key_values,
            replace_position = replace_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        if self.training:
            # Only the first half (the noisy/x_t half) has real targets; the second half of the doubled batch exists only to have supplied
            # context during flex_attention, not to produce its own logits here.
            hidden_states = hidden_states[:, :hidden_states.shape[1] // 2, :]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss.
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits = logits, labels = labels, vocab_size = self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPastAndBlockCache(
            loss = loss,
            logits = logits,
            past_key_values = outputs.past_key_values,
            hidden_states = hidden_states,
            attentions = outputs.attentions,
            block_past_key_values = outputs.block_past_key_values,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens = None,
        max_length = None,
        tokenizer = None,
        mask_id = 151665,
        threshold = 1,
        small_block_size = 8,
        block_size = 32,
        stop_token = 151645,
        stopping_criteria = None,
        top_p = 0.95,
        temperature = 0,
        use_block_cache = False,
        return_dict_in_generate = False,
        output_scores = False,
        output_hidden_states = False,
        **kwargs
    ):
        """
        Single-sequence-batch block-diffusion generation loop (no per-sample early-exit bookkeeping: 
        see kernel.generation_functions.batch_sample for the batched, early-exit-aware version used by the eval harness).

        input_ids is the [B, L] prompt token tensor. Exactly one of max_new_tokens / max_length should be given as the length bound; 
        if max_new_tokens is None it is derived from max_length. tokenizer is unused directly here (kept for interface parity with batch_sample) and present for future stopping-criteria support. 
        mask_id is the token id used to represent an unfilled position, and threshold is the confidence above which a position is unmasked early. 
        small_block_size is the tokens unmasked together within a block, and block_size is the tokens committed to the persistent KV cache per block. stop_token is the token id that ends generation. 
        stopping_criteria is an unused placeholder, kept for interface parity with HF's GenerationMixin. top_p and temperature are passed through to sample_with_top_p, 
        and use_block_cache controls whether to use the sub-block KV cache path. return_dict_in_generate, output_scores, and output_hidden_states are standard HF generation output controls.

        If return_dict_in_generate, returns a GenerateDecoderOnlyOutput with sequences (and scores / hidden_states if requested); otherwise returns the [B, S] output token tensor directly.
        """
        if max_new_tokens is None and max_length is None:
            raise ValueError("Either max_new_tokens or max_length must be specified")
        if max_new_tokens is None:
            max_new_tokens = max_length - input_ids.shape[1]

        scores_list = [] if output_scores else None
        decoder_hidden_states = [] if output_hidden_states else None

        num_blocks = max_new_tokens // block_size
        original_input_length = input_ids.shape[1]

        if input_ids.shape[1] > block_size:
            # Prime the KV cache with every full block of the prompt.
            output = self.forward(
                input_ids = input_ids[:, :(input_ids.shape[1] // block_size * block_size)],
                use_cache = True,
                update_past_key_values = True,
                block_size = block_size,
            )
            logits, past_key_values = output.logits, output.past_key_values

            if output_scores:
                scores_list.append(logits)
            if output_hidden_states and hasattr(output, 'hidden_states'):
                decoder_hidden_states.append(output.hidden_states)

            if input_ids.shape[1] % block_size == 0:
                next_token = logits[:, -1:, :].argmax(dim = -1)
                input_ids = torch.cat([input_ids, next_token], dim = 1)
        else:
            past_key_values = None

        num_small_blocks = block_size // small_block_size

        for block_idx in range(num_blocks):
            if stop_token in input_ids[:, original_input_length:]:
                break
            prompt_length = input_ids.shape[1]

            # Initialize x_init with mask_id.
            x_init = mask_id * torch.ones(
                (input_ids.shape[0], block_size - prompt_length % block_size),
                device = self.device,
                dtype = torch.long,
            )
            x_init = torch.cat([input_ids, x_init], dim = 1)

            x_t = x_init.clone()
            block_past_key_values = None

            while True:
                if stop_token in x_t[:, prompt_length:]:
                    stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                    if (x_t[:, prompt_length:prompt_length + stop_token_idx] == mask_id).sum() == 0:
                        break
                mask_idx = (x_t[:, -block_size:] == mask_id)

                # Decode a complete block, update cache, and generate the next token.
                if mask_idx.sum() == 0:
                    output = self.forward(
                        input_ids = x_t[:, -block_size:],
                        use_cache = True,
                        past_key_values = past_key_values,
                        update_past_key_values = True,
                        block_size = block_size,
                    )
                    logits, past_key_values = output.logits, output.past_key_values

                    # Collect output info.
                    if output_scores:
                        scores_list.append(logits)
                    if output_hidden_states and hasattr(output, 'hidden_states'):
                        decoder_hidden_states.append(output.hidden_states)

                    next_token = logits[:, -1:, :].argmax(dim = -1)
                    x_t = torch.cat([x_t, next_token], dim = 1)
                    break

                for small_block_idx in range(num_small_blocks):
                    small_block_start_idx = small_block_idx * small_block_size
                    small_block_end_idx = small_block_start_idx + small_block_size

                    start = -block_size + small_block_start_idx
                    end = None if block_size == small_block_end_idx else -block_size + small_block_end_idx

                    while True:
                        mask_idx = (x_t[:, -block_size:] == mask_id)
                        if mask_idx[:, start:end].sum() == 0:
                            break
                        if stop_token in x_t[:, prompt_length:]:
                            stop_token_idx = (x_t[:, prompt_length:] == stop_token).nonzero()[0][1]
                            if (x_t[:, prompt_length:prompt_length + stop_token_idx] == mask_id).sum() == 0:
                                break

                        if use_block_cache:
                            if block_past_key_values is None or (x_t[:, -block_size + small_block_start_idx] == mask_id).any():
                                output = self.forward(
                                    input_ids = x_t[:, -block_size:],
                                    use_cache = True,
                                    past_key_values = past_key_values,
                                    update_past_key_values = False,
                                    use_block_cache = True,
                                )
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim = 1)
                                logits = logits[:, start:end]
                            else:
                                output = self.forward(
                                    input_ids = x_t[:, start:end],
                                    use_cache = True,
                                    past_key_values = past_key_values,
                                    update_past_key_values = False,
                                    use_block_cache = True,
                                    block_past_key_values = block_past_key_values,
                                    replace_position = small_block_start_idx,
                                )
                                logits = output.logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim = 1)
                        else:
                            output = self.forward(
                                input_ids = x_t[:, -block_size:],
                                use_cache = True,
                                past_key_values = past_key_values,
                                update_past_key_values = False,
                            )
                            logits = output.logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim = 1)
                            logits = logits[:, start:end]

                        if output_scores:
                            scores_list.append(logits)
                        if output_hidden_states and hasattr(output, 'hidden_states'):
                            decoder_hidden_states.append(output.hidden_states)

                        x_1, p_1t = self.sample_with_top_p(logits, top_p = top_p, temperature = temperature)
                        # Select tokens with probability greater than threshold from p_1t.
                        x1_p = torch.squeeze(torch.gather(p_1t, dim = -1, index = torch.unsqueeze(x_1, -1)), -1)
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim = -1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

            input_ids = x_t

        # Truncate stop_token.
        if stop_token in input_ids[:, original_input_length:]:
            stop_token_idx = (input_ids[:, original_input_length:] == stop_token).nonzero()[0][1]
            input_ids = input_ids[:, :stop_token_idx + original_input_length + 1]

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences = input_ids,
                scores = tuple(scores_list) if output_scores and scores_list else None,
                hidden_states = tuple(decoder_hidden_states) if output_hidden_states and decoder_hidden_states else None,
            )
        else:
            return input_ids

    def sample_with_top_p(self, logits, top_p = 0.95, temperature = 1.0):
        """
        Sample one token per position from logits, using nucleus (top-p) sampling when temperature > 0, or greedy argmax when temperature == 0. 
        logits is a [..., vocab_size] tensor of unnormalized logits. top_p is the cumulative probability mass to keep before sampling, and temperature is the sampling temperature, 
        where 0 means greedy (ignores top_p).

        Returns x_1, the [...] sampled token ids, and p_1t, the [..., vocab_size] (possibly top-p-truncated and renormalized) probability distribution actually sampled from.
        """
        if temperature > 0:
            scaled_logits = logits / temperature
        else:
            p_1t = torch.softmax(logits, dim = -1)
            x_1 = p_1t.argmax(dim = -1)
            return x_1, p_1t

        probs = F.softmax(scaled_logits, dim = -1)

        sorted_probs, sorted_indices = torch.sort(probs, descending = True)
        cumulative_probs = torch.cumsum(sorted_probs, dim = -1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(probs, dtype = torch.bool).scatter_(
            dim = -1, index = sorted_indices, src = sorted_indices_to_remove
        )

        probs[indices_to_remove] = 0

        # Renormalize so that the probabilities of remaining tokens sum to 1. Add a small epsilon value to prevent division by zero.
        probs_sum = torch.sum(probs, dim = -1, keepdim = True)
        normalized_probs = probs / probs_sum

        p_1t = normalized_probs
        x_1 = torch.multinomial(p_1t[0], num_samples = 1).unsqueeze(0).squeeze(-1)

        return x_1, p_1t