"""
Block Eviction Scheduler for Fast-dLLM v2 KV Cache.

Uses the attention capture hooks already wired into kernel/modeling.py by the
collaborator. The two hooks are:

- _capture_attn_scores (bool on each attention module): when True during a commit
  step forward, the attention module computes explicit softmax weights with GQA
  expansion and stores them as _last_attn_weights [B, H, Q, KV] and
  _last_token_scores [kv_len]. These are far more accurate than key norm proxies.

- _topk_token_indices (LongTensor or None on each attention module): when set,
  the attention module slices prefix_k and prefix_v to only the selected token
  positions before the cat, and also slices the attention mask to match. This is
  the correct hook point — post-RoPE, pre-cat, so cache_position and position_ids
  are completely unaffected.

Block scoring:
    We read _last_attn_weights [B, H, Q, KV] from every layer after a commit step,
    average across layers, heads, and query positions to get per-token importance
    [B, KV], then sum within each 32-token block to get [B, N_blocks].

    Matmul union: average across batch → topk(global_scores, K) → exactly K blocks.
    Naive union: per-sequence topk unioned across batch → up to B*K blocks.
"""

import torch

BLOCK_SIZE = 32

class BlockEvictionScheduler:
    """
    Args:
    - model: Fast_dLLM_QwenForCausalLM from kernel.modeling.
    - k: number of prefix blocks to retain per commit step.
    - block_size: tokens per block (default 32).
    - union_mode: "matmul" (default) or "naive".
    """

    def __init__(self, model, k: int, block_size: int = BLOCK_SIZE, union_mode: str = "matmul"):
        self.model = model
        self.k = k
        self.block_size = block_size
        self.union_mode = union_mode
        self._original_forward = None
        self._n_commits = 0
        self._sum_n_blocks = 0
        self._sum_selected = 0
        self.last_n_blocks = 0
        self.last_selected = 0

    def _read_attn_weights(self):
        """
        Collect _last_attn_weights [B, H, Q, KV] from every layer that has them.
        Returns a stacked mean [B, H, Q, KV] or None.
        """
        all_weights = []
        for layer in self.model.model.layers:
            w = getattr(layer.self_attn, "_last_attn_weights", None)
            if w is not None:
                all_weights.append(w)
        if not all_weights:
            return None
        return torch.stack(all_weights, dim = 0).mean(dim = 0)

    def _select_and_arm(self, prefix_kv_len: int):
        """
        Read attention weights captured during the commit step, compute per-block
        importance scores, select top-K, and write _topk_token_indices to every
        attention module.

        Args:
        - prefix_kv_len: number of prefix tokens scored (S after commit).
        """
        weights = self._read_attn_weights()
        if weights is None:
            self._disarm()
            return

        B = weights.shape[0]
        KV = weights.shape[-1]

        score_len = min(prefix_kv_len, KV)
        n_blocks = score_len // self.block_size
        if n_blocks == 0:
            self._disarm()
            return

        # Average over heads and query positions: [B, score_len]
        per_seq = weights[:, :, :, :score_len].mean(dim = (1, 2))

        # Sum within each 32-token block: [B, n_blocks]
        trimmed = per_seq[:, :n_blocks * self.block_size]
        block_scores = trimmed.reshape(B, n_blocks, self.block_size).sum(dim = -1)

        k_actual = min(self.k, n_blocks)

        if self.union_mode == "matmul":
            global_scores = block_scores.mean(dim = 0)
            selected_blocks = torch.topk(global_scores, k_actual).indices.sort().values.tolist()
        else:
            selected_set = set()
            for b in range(B):
                selected_set.update(torch.topk(block_scores[b], k_actual).indices.tolist())
            selected_blocks = sorted(selected_set)

        self.last_n_blocks = n_blocks
        self.last_selected = len(selected_blocks)
        self._n_commits += 1
        self._sum_n_blocks += n_blocks
        self._sum_selected += len(selected_blocks)

        # Expand block indices to token positions.
        token_indices = []
        for blk in selected_blocks:
            start = blk * self.block_size
            token_indices.extend(range(start, start + self.block_size))

        # Keep tail tokens beyond the last complete block.
        tail_start = n_blocks * self.block_size
        if tail_start < score_len:
            token_indices.extend(range(tail_start, score_len))

        idx = torch.tensor(
            sorted(set(token_indices)), dtype = torch.long,
            device = weights.device,
        )

        for layer in self.model.model.layers:
            layer.self_attn._topk_token_indices = idx

    def _disarm(self):
        for layer in self.model.model.layers:
            layer.self_attn._topk_token_indices = None

    def _enable_capture(self):
        for layer in self.model.model.layers:
            layer.self_attn._capture_attn_scores = True

    def _disable_capture(self):
        for layer in self.model.model.layers:
            layer.self_attn._capture_attn_scores = False

    @property
    def avg_stats(self):
        if self._n_commits == 0:
            return 0, 0, 0.0
        n = self._sum_n_blocks / self._n_commits
        u = self._sum_selected / self._n_commits
        return n, u, (u / n) if n > 0 else 0.0

    def reset_stats(self):
        self._n_commits = 0
        self._sum_n_blocks = 0
        self._sum_selected = 0

    def install(self):
        """Patch model.forward. Idempotent."""
        if self._original_forward is not None:
            return self
        self.reset_stats()
        self._disarm()
        self._disable_capture()

        scheduler = self
        original_forward = self.model.forward
        self._original_forward = original_forward

        def _patched_forward(*args, **kwargs):
            update_pkv = kwargs.get("update_past_key_values", False)
            past_key_values = kwargs.get("past_key_values", None)
            use_block_cache = kwargs.get("use_block_cache", False)

            cache_ready = (
                past_key_values is not None
                and len(past_key_values) > 0
                and past_key_values[0][0] is not None
            )

            is_commit = update_pkv and not use_block_cache and cache_ready

            if is_commit:
                # Disarm so the commit sees the full prefix — unbiased scores.
                scheduler._disarm()
                # Enable weight capture for this commit forward.
                scheduler._enable_capture()

            output = original_forward(*args, **kwargs)

            if is_commit:
                scheduler._disable_capture()
                # Read the captured weights and arm layers for subsequent small steps.
                if output.past_key_values is not None:
                    pkv = output.past_key_values
                    if len(pkv) > 0 and pkv[0][0] is not None:
                        prefix_kv_len = pkv[0][0].shape[-2]
                        if prefix_kv_len >= scheduler.block_size:
                            scheduler._select_and_arm(prefix_kv_len)

            return output

        self.model.forward = _patched_forward
        return self

    def uninstall(self):
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None
        self._disarm()
        self._disable_capture()

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.uninstall()