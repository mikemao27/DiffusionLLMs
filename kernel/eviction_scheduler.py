"""
Ranks past KV blocks by importance and evicts all but the top-k.

Design:

* We never compute attention weights inside this file — that would require
  a full forward pass just for scoring. Instead we use a proxy: the EMA
  of the L2 norm of each block's key vectors, averaged over heads and
  layers. This is computed once per committed block from the data already
  sitting in the pool.

  Why key norms? Empirically, blocks that are heavily attended tend to have
  larger-magnitude key vectors (they are more "distinctive"). This is a
  well-known proxy used in StreamingLLM and H2O; it is fast to compute and
  needs no extra forward pass.

  If you later want to switch to true attention-weight scoring you only
  need to replace `_compute_block_scores()` — everything else stays the same.

* Two blocks are always protected from eviction regardless of score:
    - Block 0 ("attention sink" — the first block almost always receives
      a disproportionate share of attention in transformer models).

    - The most recently committed block (we never evict what we just wrote,
      because it is likely the most contextually relevant).

* top_k is a fixed integer. For short GSM8K sequences this is typically
  ≥ the total number of blocks, so eviction is a no-op for most examples.
  You should set it to a value smaller than the maximum number of blocks
  you expect (e.g. top_k=4 for 128-token generations with block_size=32).

Usage:

    scheduler = EvictionScheduler(
        cache = paged_kv_cache, # PagedKVCache instance
        top_k = 4, # keep this many blocks (incl. sink + latest)
        ema_alpha = 0.9, # EMA decay; higher = more weight on recent scores
    )

    # Call this once after each block is committed:
        scheduler.step()
"""

from __future__ import annotations

import torch
from typing import List

# PagedKVCache is imported lazily inside the class to avoid a circular import.

class EvictionScheduler:
    """
    Scores committed KV blocks and evicts the lowest-scoring ones, keeping
    only the top-k plus the always-protected attention-sink and latest block.

    Parameters:
        - cache : PagedKVCache
          The paged cache whose blocks are to be managed.
    
        - top_k : int
          Total number of blocks to retain after each eviction step, including
          the forced sink (block 0) and the forced latest block.
          Must be >= 2 (you always keep at least sink + latest).

        - ema_alpha : float
          EMA decay factor in [0, 1].  New score = alpha * old + (1-alpha) * raw.
          Set to 0.0 to disable EMA and always use the raw score.
    
        - layer_sample_stride : int
          To keep scoring fast, we average over every *stride*-th layer
          rather than all layers.  E.g. stride=4 on a 28-layer model means
          we sample 7 layers.  Set to 1 to use all layers.
    """

    def __init__(
        self,
        cache, # PagedKVCache
        top_k: int = 4,
        ema_alpha: float = 0.9,
        layer_sample_stride: int = 4,
    ) -> None:
        if top_k < 2:
            raise ValueError("top_k must be >= 2 (need at least sink + latest block).")

        self.cache = cache
        self.top_k = top_k
        self.ema_alpha = ema_alpha
        self.layer_sample_stride = layer_sample_stride

        # Total number of eviction steps taken so far (for diagnostics).
        self._steps: int = 0

    # Main entry point:
    def step(self) -> List[int]:
        """
        Score all currently committed blocks, apply EMA, and evict the
        lowest-scoring ones so that at most top_k blocks remain.

        Returns:
            - List[int]: The logical block indices that were *kept* after this step.
              Useful for logging or debugging.

        Notes: If the number of committed blocks <= top_k, no eviction occurs and
        all blocks are returned.
        """
        num_blocks = self.cache.get_num_active_blocks()

        if num_blocks <= self.top_k:
            # Nothing to evict yet.
            self._steps += 1
            return list(range(num_blocks))

        # 1. Compute raw importance scores for every active block.
        raw_scores = self._compute_block_scores(num_blocks)

        # 2. Apply EMA to the stored scores.
        for layer_idx in range(self.cache.num_layers):
            stored = self.cache.block_scores[layer_idx]
            for logical_idx in range(num_blocks):
                old = stored[logical_idx]
                raw = raw_scores[logical_idx]
                if old == float("inf"):
                    # First time this block is scored — initialise directly.
                    stored[logical_idx] = raw

                else:
                    stored[logical_idx] = (
                        self.ema_alpha * old + (1.0 - self.ema_alpha) * raw
                    )

        # 3. Compute the per-block mean score across all layers for ranking.
        mean_scores: List[float] = []
        for logical_idx in range(num_blocks):
            layer_scores = [
                self.cache.block_scores[l][logical_idx]
                for l in range(self.cache.num_layers)
                if logical_idx < len(self.cache.block_scores[l])
            ]
            mean_scores.append(
                sum(layer_scores) / len(layer_scores) if layer_scores else 0.0
            )

        # 4. Determine which blocks to keep.
        keep_indices = self._select_top_k(mean_scores, num_blocks)

        # 5. Evict the rest.
        evict_set = set(range(num_blocks)) - set(keep_indices)
        if evict_set:
            self.cache.evict_blocks(keep_indices)

        self._steps += 1
        return keep_indices

    # Scoring:
    def _compute_block_scores(self, num_blocks: int) -> List[float]:
        """
        Estimate the importance of each active block as the mean L2 norm of
        its key vectors, averaged over sampled layers and all KV heads.

        Shape accessed per layer:
            pool_k[layer_idx] : [max_slots, num_kv_heads, block_size, head_dim]

        For each logical block b with physical slot s = block_table[0][b]:
            key_block = pool_k[layer_idx][s] → [num_kv_heads, block_size, head_dim]
            score contribution = mean(||key_block||_2) over heads and tokens

        Returns:
            - List[float], length = num_blocks
              Raw (pre-EMA) importance score for each logical block.
              Higher = more important = keep.
        """
        pool = self.cache.pool
        n_layers = self.cache.num_layers

        # Which layers to sample.
        sampled_layers = list(range(0, n_layers, self.layer_sample_stride))
        if not sampled_layers:
            sampled_layers = [0]

        # Accumulate scores: shape [num_blocks]
        scores = [0.0] * num_blocks

        with torch.no_grad():
            for layer_idx in sampled_layers:
                pool_k = pool.get_pool_k(layer_idx) # [max_slots, heads, block_size, dim]
                slot_list = self.cache.block_table[layer_idx] # list of physical slots

                for logical_idx, slot in enumerate(slot_list):
                    if logical_idx >= num_blocks:
                        break
                    # key_block: [num_kv_heads, block_size, head_dim]
                    key_block = pool_k[slot]
                    # L2 norm per position per head, then mean over both.
                    # norm shape: [num_kv_heads, block_size]
                    norm = torch.norm(key_block.float(), dim=-1)
                    scores[logical_idx] += norm.mean().item()

        # Normalise by the number of sampled layers.
        n = len(sampled_layers)
        scores = [s / n for s in scores]
        return scores

    # Selection:
    def _select_top_k(
        self,
        mean_scores: List[float],
        num_blocks: int,
    ) -> List[int]:
        """
        Return the logical block indices to keep, respecting:
            - Always keep block 0 (attention sink).
            - Always keep block num_blocks-1 (most recently committed).
            - Fill the remaining top_k - 2 slots with the highest-scoring
              blocks from the rest.

        Parameters:
            - mean_scores : List[float]
              Score for each logical block index 0 … num_blocks - 1.

            - num_blocks : int

        Returns:
            - List[int]
              Sorted list of logical indices to retain.
        """
        protected = {0, num_blocks - 1} # sink + latest
        free_budget = max(0, self.top_k - len(protected)) # slots left for scoring

        # Collect candidates (everything except protected blocks).
        candidates = [
            (score, idx)
            for idx, score in enumerate(mean_scores)
            if idx not in protected
        ]

        # Sort descending by score; take the top free_budget candidates.
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected_candidates = [idx for _, idx in candidates[:free_budget]]

        # Combine protected + selected, sort to preserve block order.
        keep = sorted(protected | set(selected_candidates))
        return keep

    # Diagnostics:
    def __repr__(self) -> str:
        return (
            f"EvictionScheduler("
            f"top_k = {self.top_k}, "
            f"ema_alpha = {self.ema_alpha}, "
            f"steps = {self._steps}, "
            f"active_blocks = {self.cache.get_num_active_blocks()})"
        )