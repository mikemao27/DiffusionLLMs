"""
Block Eviction Scheduler for Fast-dLLM v2 KV Cache.

Decides which prefix KV blocks to keep for the next few generation steps, and arms the model's attention layers to only look at those blocks. 
Uses the attention capture hooks already wired into kernel/modeling.py by the collaborator. The two hooks are _capture_attn_scores (bool on each attention module), 
which when True during a commit step forward makes the attention module compute explicit softmax weights with GQA expansion and store them as 
_last_attn_weights [B, H, Q, KV] and _last_token_scores [kv_len] (these are far more accurate than key norm proxies, see score_blocks.py, which implements that proxy 
but is not used here), and _topk_token_indices (LongTensor or None on each attention module), which when set makes the attention module slice prefix_k and prefix_v to 
only the selected token positions before the cat, and also slice the attention mask to match. This is the correct hook point: post-RoPE, pre-cat, so cache_position and 
position_ids are completely unaffected.

Block scoring reads _last_attn_weights [B, H, Q, KV] from every layer after a commit step, averages across layers, heads, and query positions to get per-token importance [B, KV], 
then sums within each 32-token block to get [B, N_blocks]. The matmul union averages across batch, then takes topk(global_scores, K) for exactly K blocks. The naive union takes 
a per-sequence topk unioned across batch, for up to B * K blocks.

Interface contract (never changes): BlockEvictionScheduler only ever mutates its own bookkeeping attributes plus a fixed set of underscore-prefixed attributes on each attention module 
(_topk_token_indices, _use_sparse_kernel, _capture_attn_scores, _sparse_path_hits, _sparse_path_misses). It never touches model weights.

install() patches model.forward; uninstall() restores the original. Both are idempotent, and the scheduler is also usable as a context manager (`with scheduler: ...`) 
which calls install/uninstall for you.
"""

import torch

BLOCK_SIZE = 32

class BlockEvictionScheduler:
    """
    Wraps a Fast_dLLM_QwenForCausalLM model (from kernel.modeling) and manages block eviction. k is the number of prefix blocks to retain per commit step, 
    block_size is the tokens per block (default 32), and union_mode is "matmul" (default) or "naive". use_sparse_kernel, when True, arms _use_sparse_kernel on every attention module alongside 
    _topk_token_indices, switching the small-step attention computation from gather+cat+SDPA to the Triton sparse_attn_merge path. It defaults to False so existing callers are unaffected, 
    since this is the new, not-yet-fully-validated speedup path.
    """

    def __init__(
        self,
        model,
        k: int,
        block_size: int = BLOCK_SIZE,
        union_mode: str = "matmul",
        use_sparse_kernel: bool = False,
    ):
        self.model = model
        self.k = k
        self.block_size = block_size
        self.union_mode = union_mode
        self.use_sparse_kernel = use_sparse_kernel
        self._original_forward = None
        self._n_commits = 0
        self._sum_n_blocks = 0
        self._sum_selected = 0
        self.last_n_blocks = 0
        self.last_selected = 0

    def _read_attn_weights(self):
        """
        Collect _last_attn_weights [B, H, Q, KV] from every layer that has them. Returns a stacked mean [B, H, Q, KV] tensor, or None if no layer captured weights on the most recent commit step.
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
        Read attention weights captured during the commit step, compute per-block importance scores, select top-K, and write _topk_token_indices to every attention module. 
        prefix_kv_len is the number of prefix tokens scored (S after commit). Returns nothing; arms (or disarms) every attention module's eviction hooks as a side effect.
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

        # Safety guard: the sparse kernel path assumes every selected token index is block-aligned (a multiple of block_size), because it recovers block
        # ids via (idx // block_size).unique(). If a non-block-aligned tail was appended above (prefix_kv_len not a multiple of block_size), arming the
        # sparse kernel would cause it to read a misaligned/partial block. In that case we keep _topk_token_indices armed (gather path still works on
        # arbitrary indices) but leave _use_sparse_kernel off for this commit.
        tail_is_aligned = (tail_start >= score_len) # No tail tokens were appended.
        arm_sparse = self.use_sparse_kernel and tail_is_aligned

        for layer in self.model.model.layers:
            layer.self_attn._topk_token_indices = idx
            layer.self_attn._use_sparse_kernel = arm_sparse
            # Recorded so the attention forward can verify idx is still valid for the CURRENT call with a plain Python int comparison
            # (prefix_k.shape[-2] == this value) instead of a GPU boolean-mask bounds check, which forces a device-to-host sync every layer,
            # every small-step call (confirmed via profiling: aten::index accounted for 43% of total CPU time despite ~1.5% of GPU
            # compute). idx is built entirely from block/tail positions bounded by score_len <= prefix_kv_len above, so as long as this
            # recorded length still matches the live cache length, every value in idx is guaranteed in-bounds: no data-dependent check
            # needed. If it doesn't match (e.g. stale state from an unrelated generation call), the attention forward falls back to the full,
            # unpruned prefix for that call rather than trusting stale data.
            layer.self_attn._topk_armed_for_len = prefix_kv_len

    def _disarm(self):
        """
        Clear the eviction hooks on every attention module (full prefix, no sparse kernel).
        """
        for layer in self.model.model.layers:
            layer.self_attn._topk_token_indices = None
            layer.self_attn._use_sparse_kernel = False
            layer.self_attn._topk_armed_for_len = None

    def _enable_capture(self):
        """
        Turn on attention-weight capture on every attention module.
        """
        for layer in self.model.model.layers:
            layer.self_attn._capture_attn_scores = True

    def _disable_capture(self):
        """
        Turn off attention-weight capture on every attention module.
        """
        for layer in self.model.model.layers:
            layer.self_attn._capture_attn_scores = False

    @property
    def avg_stats(self):
        """
        Returns a (avg_n_blocks, avg_selected, avg_fraction_selected) tuple, averaged across every commit step seen since the last reset_stats() call. 
        All three are 0 / 0 / 0.0 if no commit has happened yet.
        """
        if self._n_commits == 0:
            return 0, 0, 0.0
        n = self._sum_n_blocks / self._n_commits
        u = self._sum_selected / self._n_commits
        return n, u, (u / n) if n > 0 else 0.0

    @property
    def sparse_kernel_stats(self):
        """
        Aggregated across all attention layers. hits is the number of forward calls where the Triton sparse_attn_merge path was actually taken; 
        misses is the number of calls that fell back to gather+cat+SDPA despite use_sparse_kernel being enabled; rebuild_misses is the subset of misses 
        specifically caused by a full-block rebuild call (query width == block width) rather than by some other gate condition (armed state, mask shape, etc). 
        Comparing rebuild_misses to misses tells you how much of the miss count is the expected, intentional rebuild-call fallback versus something else unexpectedly failing the gate. 
        Only meaningful when use_sparse_kernel = True was set on this scheduler.

        Returns a (hits, misses, hit_rate, rebuild_misses) tuple.
        """
        hits = sum(
            getattr(layer.self_attn, '_sparse_path_hits', 0)
            for layer in self.model.model.layers
        )
        misses = sum(
            getattr(layer.self_attn, '_sparse_path_misses', 0)
            for layer in self.model.model.layers
        )
        rebuild_misses = sum(
            getattr(layer.self_attn, '_sparse_path_rebuild_misses', 0)
            for layer in self.model.model.layers
        )
        total = hits + misses
        return hits, misses, (hits / total if total > 0 else 0.0), rebuild_misses

    def reset_stats(self):
        """
        Zero out commit-count and sparse-kernel hit/miss counters.
        """
        self._n_commits = 0
        self._sum_n_blocks = 0
        self._sum_selected = 0
        for layer in self.model.model.layers:
            layer.self_attn._sparse_path_hits = 0
            layer.self_attn._sparse_path_misses = 0
            layer.self_attn._sparse_path_rebuild_misses = 0

    def install(self):
        """
        Patch model.forward to intercept commit steps, capture attention weights, and arm eviction on every attention module. Idempotent: a second call while already installed is a no-op. Returns self, so this can be chained (e.g. `scheduler = BlockEvictionScheduler(...).install()`).
        """
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

            # A None past_key_values means this call is the start of a brand-new, independent generation sequence (e.g. a new batch
            # within an eval sweep's batch loop). Any _topk_token_indices still armed from a PREVIOUS generation's final commit must not
            # be allowed to leak into this new one: it was computed for that previous sequence's prefix length, which is unrelated to
            # (and typically much longer than) this fresh sequence's. Root-caused via the [eviction debug] print: idx.max() matched
            # exactly (prompt_len + max_new_tokens) from the prior batch, while prefix_k.shape[-2] matched this new batch's freshly
            # reset prompt length: the signature of cross-call state leakage, not a real out-of-bounds selection.
            if past_key_values is None:
                scheduler._disarm()

            cache_ready = (
                past_key_values is not None
                and len(past_key_values) > 0
                and past_key_values[0][0] is not None
            )

            is_commit = update_pkv and not use_block_cache and cache_ready

            if is_commit:
                # Disarm so the commit sees the full prefix: unbiased scores.
                scheduler._disarm()
                # Enable weight capture for this commit forward.
                scheduler._enable_capture()

            # Tagged so a torch.profiler trace can report commit-step and small-step cost separately, rather than lumped into one
            # aggregate: the two have very different per-call cost and frequency (commits: infrequent, full unpruned attention plus a
            # second manual scoring pass; small-steps: frequent, K-pruned attention only), and telling them apart from an unlabeled
            # profile requires guessing from call counts.
            from torch.profiler import record_function
            with record_function("commit_step" if is_commit else "small_step"):
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
        """
        Restore the original model.forward and clear all eviction hooks.
        """
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None
        self._disarm()
        self._disable_capture()

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.uninstall()