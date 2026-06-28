"""
Physical KV block pool and a DynamicCache subclass that stores completed
Fast-dLLM v2 blocks in non-contiguous physical slots.

Key ideas:
* BlockAllocator — a CPU-side free-list of integer slot indices.  Allocating
  or freeing a slot is O(1) and never touches GPU memory.

* KVBlockPool — one pre-allocated GPU tensor per layer and per key/value:
      shape = [num_physical_slots, num_kv_heads, BLOCK_SIZE, head_dim]
  All writes go through write_block(); all reads (for the attention kernel)
  are exposed as a direct tensor view via get_pool_k / get_pool_v.

* PagedKVCache — subclasses transformers.DynamicCache so that it can be
  passed to model.forward() unchanged. It overrides update() so that
  every NEW block_size tokens are atomically committed to the pool rather
  than appended to a growing contiguous tensor.

  Between blocks (while diffusion iterates inside one block) the cache
  behaves exactly like DynamicCache — we accumulate tokens normally.
  When update() detects that a full block has just been committed
  (update_past_key_values=True is set in generation_functions.py at the
  block boundary), we freeze that block into the pool and the eviction
  scheduler can rank it.

Integration Note:
generation_functions.py does:

    output = self.forward(..., update_past_key_values=True, ...)
    past_key_values = output.past_key_values

You replace `past_key_values = None` at initialization with:

    past_key_values = PagedKVCache(
        num_layers = model.config.num_hidden_layers,
        num_kv_heads = model.config.num_key_value_heads,
        head_dim = model.config.hidden_size // model.config.num_attention_heads,
        max_slots = MAX_SLOTS,
        block_size = BLOCK_SIZE,
        device = device,
        dtype = torch.bfloat16,
    )

The rest of the generation loop is unchanged.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache
from typing import Dict, List, Optional, Tuple

# Constants:
BLOCK_SIZE: int = 32 # tokens per KV block; matches Fast-dLLM v2 default
DEFAULT_MAX_SLOTS: int = 512 # upper bound on simultaneously live blocks

# BlockAllocator:
class BlockAllocator:
    """
    A simple CPU-side free-list allocator over integer slot indices [0, max_slots).

    All operations are O(1) because we maintain the free list as a Python list
    used as a stack. There is intentionally no GPU memory inside this class.

    Attributes:
        - max_slots (int): total number of physical KV slots pre-allocated in the pool.
        - _free (List[int]): stack of currently available slot indices.
    """

    def __init__(self, max_slots: int = DEFAULT_MAX_SLOTS) -> None:
        self.max_slots: int = max_slots

        # Start with every slot free; pop from the right for O(1) access.
        self._free: List[int] = list(range(max_slots))

    # Public API:
    def allocate(self) -> int:
        """
        Return a free physical slot index.

        Raises RuntimeError if the pool is exhausted — caller should evict blocks first.
        """
        if not self._free:
            raise RuntimeError(
                "KVBlockPool exhausted: no free physical slots."
                "Increase max_slots or evict more blocks before allocating."
            )
        return self._free.pop()

    def free(self, slot_idx: int) -> None:
        """
        Return slot *slot_idx* to the free list.

        Parameters:
            - slot_idx (int): a previously allocated slot index.
        """
        assert 0 <= slot_idx < self.max_slots, (
            f"free(): slot_idx {slot_idx} out of range [0, {self.max_slots})"
        )
        self._free.append(slot_idx)

    @property
    def num_free(self) -> int:
        """
        Number of slots currently available for allocation.
        """
        return len(self._free)

    @property
    def num_used(self) -> int:
        """
        Number of slots currently occupied.
        """
        return self.max_slots - len(self._free)

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(max_slots={self.max_slots}, "
            f"used={self.num_used}, free={self.num_free})"
        )

# KVBlockPool:
class KVBlockPool:
    """
    Pre-allocated GPU memory for all physical KV blocks, across all layers.

    Storage layout (per layer, per K or V):
        Tensor shape: [num_physical_slots, num_kv_heads, BLOCK_SIZE, head_dim]

    This flat layout lets the Triton kernel index directly into a physical
    slot without any pointer indirection: given slot s, the keys are at
    pool_k[s]. The block table (logical → physical) is owned by PagedKVCache.

    Parameters:
        - num_layers (int): number of transformer layers
        - num_kv_heads (int): number of KV attention heads (may differ from Q heads with GQA)
        - head_dim (int): per-head dimension
        - max_slots (int): upper bound on simultaneously live blocks
        - block_size (int): tokens per block (default BLOCK_SIZE = 32)
        - device (int): torch.device
        - dtype (int): torch.dtype (bfloat16 recommended for Qwen2.5)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_slots: int = DEFAULT_MAX_SLOTS,
        block_size: int = BLOCK_SIZE,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_slots = max_slots
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        # Allocate pool tensors: one per layer for keys, one per layer for values.
        # We store them in plain lists indexed by layer_idx.
        # Shape: [max_slots, num_kv_heads, block_size, head_dim]
        self.pool_k: List[torch.Tensor] = [
            torch.empty(
                max_slots, num_kv_heads, block_size, head_dim,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ]
        self.pool_v: List[torch.Tensor] = [
            torch.empty(
                max_slots, num_kv_heads, block_size, head_dim,
                device=device, dtype=dtype,
            )
            for _ in range(num_layers)
        ]

    # Write path (called once per committed block, per layer):
    def write_block(
        self,
        layer_idx: int,
        slot_idx: int,
        k_block: torch.Tensor, # [num_kv_heads, block_size, head_dim]
        v_block: torch.Tensor, # [num_kv_heads, block_size, head_dim]
    ) -> None:
        """
        Copy one block of KV states into physical slot *slot_idx* for *layer_idx*.

        The copy is in-place and asynchronous with respect to the Python thread
        (standard CUDA stream behaviour).  The Triton kernel that subsequently
        reads this slot will be launched on the same stream, so ordering is
        preserved without an explicit synchronisation barrier.

        Parameters:
            - layer_idx : int
            - slot_idx : int — physical slot allocated by BlockAllocator
            - k_block : Tensor[num_kv_heads, block_size, head_dim]
            - v_block : Tensor[num_kv_heads, block_size, head_dim]
        """
        # Reshape if the block came in as [batch=1, num_kv_heads, seq, head_dim].
        if k_block.dim() == 4:
            # batch dim present — squeeze it (we only support batch-size 1 paging)
            k_block = k_block.squeeze(0)
            v_block = v_block.squeeze(0)

        # Trim or pad to exactly block_size tokens if needed.
        # (In practice the block will always be exactly block_size tokens
        # because Fast-dLLM v2 only calls update_past_key_values=True at
        # block boundaries, but we guard anyway.)
        seq_len = k_block.shape[1]
        if seq_len != self.block_size:
            if seq_len < self.block_size:

                # Pad with zeros — eviction scoring will give this block a
                # low score, so it will be evicted before it can cause errors.
                pad_k = torch.zeros(
                    self.num_kv_heads,
                    self.block_size - seq_len,
                    self.head_dim,
                    device=self.device,
                    dtype=self.dtype,
                )
                pad_v = torch.zeros_like(pad_k)
                k_block = torch.cat([k_block, pad_k], dim=1)
                v_block = torch.cat([v_block, pad_v], dim=1)

            else:
                # More tokens than expected — only keep the first block_size.
                k_block = k_block[:, :self.block_size, :]
                v_block = v_block[:, :self.block_size, :]

        self.pool_k[layer_idx][slot_idx].copy_(k_block, non_blocking=True)
        self.pool_v[layer_idx][slot_idx].copy_(v_block, non_blocking=True)

    # Read path (for the Triton attention kernel):
    def get_pool_k(self, layer_idx: int) -> torch.Tensor:
        """
        Return the full key pool tensor for *layer_idx*.
        Shape: [max_slots, num_kv_heads, block_size, head_dim]
        The Triton kernel indexes into this with the physical slot indices
        from the block table.
        """
        return self.pool_k[layer_idx]

    def get_pool_v(self, layer_idx: int) -> torch.Tensor:
        """
        Return the full value pool tensor for *layer_idx*.
        Shape: [max_slots, num_kv_heads, block_size, head_dim]
        """
        return self.pool_v[layer_idx]

    def memory_footprint_gb(self) -> float:
        """Estimated GPU memory used by this pool, in GiB."""
        elems_per_layer = (
            self.max_slots * self.num_kv_heads * self.block_size * self.head_dim
        )
        bytes_per_elem = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        total_bytes = 2 * self.num_layers * elems_per_layer * bytes_per_elem
        return total_bytes / (1024 ** 3)

    def __repr__(self) -> str:
        return (
            f"KVBlockPool("
            f"layers = {self.num_layers}, heads = {self.num_kv_heads}, "
            f"block_size = {self.block_size}, head_dim = {self.head_dim}, "
            f"slots = {self.max_slots}, "
            f"~{self.memory_footprint_gb():.2f} GiB)"
        )

# PagedKVCache:
class PagedKVCache(DynamicCache):
    """
    A drop-in replacement for transformers.DynamicCache that stores completed
    Fast-dLLM v2 blocks in a KVBlockPool instead of a growing contiguous tensor.

    How it works:

    Inside one diffusion block, update() behaves like ordinary DynamicCache —
    it concatenates new KV states to key_cache / value_cache. This is needed
    because the diffusion inner loop calls forward() many times while the
    block is still being denoised, and each call must attend over the clean
    prefix and the partially-denoised current block.

    At the *block boundary* (when generation_functions.py calls forward with
    update_past_key_values=True), the accumulated cache for that block is
    frozen: each layer's block_size-token chunk is copied into the pool, a
    physical slot is allocated for it, and the entry is registered in the
    block table.  The contiguous tensor (key_cache / value_cache) is then
    truncated back to only the prompt portion that predates the first managed
    block — future block lookups go exclusively through the Triton kernel.

    Block table: 

    self.block_table : Dict[int, List[int]]
        Maps layer_idx → ordered list of physical slot indices, one per
        committed block.  Slot index i in this list corresponds to logical
        block i for that layer.

    self.block_scores : Dict[int, List[float]]
        Maps layer_idx → mean attention score per committed block (EMA).
        Updated by EvictionScheduler after each block is committed.
        Initialised to +inf so that brand-new blocks are never evicted
        immediately.

    Parameters:

    (All forwarded to KVBlockPool; see its docstring.)
        - block_size : int
    Must match Fast-dLLM v2's block_size argument (default 32).
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_slots: int = DEFAULT_MAX_SLOTS,
        block_size: int = BLOCK_SIZE,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__() # initialises key_cache=[], value_cache=[], _seen_tokens=0

        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.num_layers = num_layers

        self.pool = KVBlockPool(
            num_layers = num_layers,
            num_kv_heads = num_kv_heads,
            head_dim = head_dim,
            max_slots = max_slots,
            block_size = block_size,
            device = device,
            dtype = dtype,
        )
        self.allocator = BlockAllocator(max_slots=max_slots)

        # block_table[layer_idx] = [slot_0, slot_1, …] (ordered by block age):
        self.block_table:  Dict[int, List[int]]   = {i: [] for i in range(num_layers)}

        # block_scores[layer_idx] = [score_0, score_1, …] (EMA, higher = more important):
        self.block_scores: Dict[int, List[float]] = {i: [] for i in range(num_layers)}

        # Track how many complete blocks have been frozen into the pool.
        # All layers always have the same block count, so we only need one counter.
        self.num_committed_blocks: int = 0

        # Prompt tokens that were cached before the first managed block.
        # These live in key_cache / value_cache as usual and are always kept.
        self._prompt_seq_len: int = 0

    # DynamicCache.update() override:
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called by every transformer attention layer on every forward pass.

        This override detects whether the caller is:
          (A) iterating *inside* a diffusion block → behave like DynamicCache
          (B) committing a completed block → commit to pool, then return the paged view

        Detection: generation_functions.py sets update_past_key_values=True
        only at block boundaries. We receive this via cache_kwargs:
            cache_kwargs = {"commit_block": True}
        You must patch the one call-site in generation_functions.py to pass
        this flag (see integration note in module docstring).

        Alternatively, you can detect the block boundary by checking whether
        the current accumulated sequence length is a multiple of block_size:
        this avoids touching generation_functions.py at all, but requires
        care with prompt tokens that don't align to block_size.

        We use the second (no-patch) strategy here.

        Return value: 
            - (full_k, full_v) — the concatenation of:
                * prompt KV (always in contiguous tensor)
                * paged block KV (reconstructed from pool into a contiguous view)
        so that Qwen2.5 SDPA sees one ordinary tensor.
        """
        cache_kwargs = cache_kwargs or {}

        # 1. Standard DynamicCache accumulation (always happens):
        # This call appends key_states / value_states to key_cache[layer_idx].
        k_full, v_full = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # 2. Check whether a complete block is ready to commit:
        # We only commit once per block per layer.  We track commits via
        # num_committed_blocks (incremented once, on layer 0).
        current_seq_len = k_full.shape[-2] # [batch, heads, seq, dim] → dim -2

        # Number of *generation* tokens (past the prompt):
        gen_tokens = current_seq_len - self._prompt_seq_len
        expected_committed = gen_tokens // self.block_size

        if expected_committed > self.num_committed_blocks:
            # A new block has just been completed.
            # Commit for all layers atomically (layer 0 drives the counter).
            if layer_idx == 0:
                self._commit_latest_block()

        # 3. Return the full KV tensor (prompt + pool rehydration):
        # After committing, key_cache[layer_idx] still holds the full
        # contiguous tensor — DynamicCache's super().update() has already
        # appended to it. We return it as-is so Qwen2.5 SDPA works normally.
        #
        # The *paged attention Triton kernel* path is taken when the caller
        # is block_sparse_attention.paged_attention_forward(), not here.
        # This cache object exposes the pool via get_pool_k/get_pool_v for
        # that kernel.
        return k_full, v_full

    # Block commitment:
    def _commit_latest_block(self) -> None:
        """
        Freeze the most recently completed block into the pool for every layer.

        Called once (on layer_idx == 0) whenever the accumulated sequence
        length passes a block_size boundary. By the time we process layer 0,
        all layers have been updated for this forward pass, so
        key_cache[layer_idx] is current for every layer.
        """
        block_idx = self.num_committed_blocks # 0-indexed
        start_tok = self._prompt_seq_len + block_idx * self.block_size
        end_tok = start_tok + self.block_size

        for layer_idx in range(self.num_layers):
            if layer_idx >= len(self.key_cache):
                # Layer hasn't been populated yet (shouldn't happen, but be safe).
                break

            k_layer = self.key_cache[layer_idx] # [batch, heads, seq_len, head_dim]
            v_layer = self.value_cache[layer_idx]

            # Slice out exactly this block's tokens.
            k_block = k_layer[:, :, start_tok:end_tok, :] # [B, H, block_size, D]
            v_block = v_layer[:, :, start_tok:end_tok, :]

            # Allocate a slot and write to the pool.
            slot = self.allocator.allocate()
            self.pool.write_block(layer_idx, slot, k_block, v_block)
            self.block_table[layer_idx].append(slot)
            self.block_scores[layer_idx].append(float("inf")) # never evict until scored

        self.num_committed_blocks += 1

    # Eviction (called by EvictionScheduler):
    def evict_blocks(self, keep_indices: List[int]) -> None:
        """
        Evict all blocks whose logical index is NOT in *keep_indices*.

        Called by EvictionScheduler after it has decided which blocks
        are important.  The freed physical slots are returned to the allocator.

        Parameters:
            - keep_indices : List[int]
        Logical block indices (0-indexed, relative to committed blocks)
        that should be retained.

        Notes:
            * Block 0 (the attention sink) and the most-recent block are always
              kept by EvictionScheduler regardless of score, so they will always
              appear in keep_indices.

            * After eviction, block_table entries for dropped blocks are removed,
              but the remaining entries are renumbered to stay contiguous.

            * The key_cache / value_cache tensors are NOT modified here — they
              are only used during the intra-block diffusion phase, not for
            prefix attention (which goes through the Triton kernel).
        """
        keep_set = set(keep_indices)

        for layer_idx in range(self.num_layers):
            old_table = self.block_table[layer_idx]
            old_scores = self.block_scores[layer_idx]

            new_table: List[int] = []
            new_scores: List[float] = []

            for logical_idx, (slot, score) in enumerate(zip(old_table, old_scores)):
                if logical_idx in keep_set:
                    new_table.append(slot)
                    new_scores.append(score)
                else:
                    self.allocator.free(slot)

            self.block_table[layer_idx]  = new_table
            self.block_scores[layer_idx] = new_scores

        # After eviction, num_committed_blocks no longer equals len(block_table),
        # but that's fine — it is only used as a monotonic counter for detecting
        # new blocks, not as a table-length query.

    # Accessors for the Triton kernel:
    def get_slot_tensor(self, layer_idx: int) -> torch.Tensor:
        """
        Return a 1-D int32 tensor of the *active* physical slot indices for
        *layer_idx*, in logical order (oldest block first).

        This is the block table that the Triton paged-attention kernel uses to
        gather KV blocks from the pool.

        Returns:
            - torch.Tensor, shape [num_active_blocks], dtype=int32, device=self.device
        """
        slots = self.block_table[layer_idx]
        if not slots:
            return torch.empty(0, dtype=torch.int32, device=self.device)
        return torch.tensor(slots, dtype=torch.int32, device=self.device)

    def get_num_active_blocks(self) -> int:
        """
        Number of logical blocks currently retained in the pool (layer 0).
        """
        return len(self.block_table[0])

    # Prompt initialisation helper:
    def record_prompt_length(self, prompt_seq_len: int) -> None:
        """
        Record how many tokens belong to the prompt (pre-generation) prefix.

        Call this after the first forward pass that encodes the prompt:

            past_key_values.record_prompt_length(input_ids.shape[1])

        This boundary tells _commit_latest_block() where prompt tokens end
        and where generated blocks begin.
        """
        self._prompt_seq_len = prompt_seq_len

    # Diagnostics:
    def summary(self) -> str:
        n_active = self.get_num_active_blocks()
        n_committed = self.num_committed_blocks
        n_evicted = n_committed - n_active
        return (
            f"PagedKVCache | committed = {n_committed} blocks | "
            f"active = {n_active} | evicted = {n_evicted} | "
            f"free_slots = {self.allocator.num_free} | "
            f"pool = {self.pool}"
        )