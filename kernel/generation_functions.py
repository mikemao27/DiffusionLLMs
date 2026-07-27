"""
Fast-dLLM v2 Generation Functions.

Implements the block-diffusion sampling loop (batch_sample) that drives generation, and binds it onto a loaded model instance. 
This is a direct copy of Fast-dLLM/v2/generation_functions.py from the official NVlabs repository, placed in kernel/ so our eviction harness can import 
it without relying on sys.path manipulation.

The only class here, Fast_dLLM_QwenForCausalLM, is not a real nn.Module. It is a namespace of static methods that get bound onto a model instance loaded 
via AutoModelForCausalLM.from_pretrained(..., trust_remote_code = True). The actual model architecture lives inside the HuggingFace checkpoint's own modeling file.

Interface contract (never changes): batch_sample() takes a left-padded batch of prompts plus per-sample seq_len (the prompt length of each sample before padding), 
and returns a dict {original_batch_index: full_token_sequence_tensor}. Finished samples (those that produced stop_token) are dropped from the active batch mid-loop 
but are still present in the returned dict. setup_model() is the only supported way to attach these methods; it must be called once per model instance before generation.

Usage (mirroring eval.py from the official repo):
    import types
    from kernel.generation_functions import Fast_dLLM_QwenForCausalLM
    model.mdm_sample = types.MethodType(Fast_dLLM_QwenForCausalLM.batch_sample, model)
"""

import torch
import types

MASK_ID = 151665
STOP_TOKEN = 151645

class Fast_dLLM_QwenForCausalLM:

    @torch.no_grad()
    def batch_sample(
        self,
        input_ids,
        tokenizer,
        block_size,
        max_new_tokens,
        small_block_size,
        min_len,
        seq_len,
        mask_id = 151665,
        threshold = 0.95,
        stop_token = 151645,
        use_block_cache = False,
        top_p = 0.95,
        temperature = 0.0,
    ):
        """
        Generate a batch of sequences using block-diffusion sampling. Each sample advances one block_size-token block at a time. 
        Within a block, tokens are unmasked in small_block_size-token sub-blocks: the model repeatedly scores all still-masked positions 
        in the current sub-block and commits any token whose confidence exceeds threshold (always committing at least the single highest-confidence 
        token per step, so every sub-block eventually completes). Once a full block has no masked tokens left, its KV is committed to the persistent 
        cache and the next block begins. Samples that emit stop_token are removed from the active batch (and from past_key_values) as soon as they finish, 
        so remaining compute is not wasted on already-finished sequences.

        self is a model instance with .forward() and .sample_with_top_p() (the latter bound via setup_model()). input_ids is the [B, L] left-padded prompt 
        token tensor, and tokenizer is the model tokenizer, used for tokenizer.pad_token_id. block_size is the tokens committed to the persistent KV cache 
        per block, max_new_tokens is the upper bound on new tokens generated per sample, and small_block_size is the tokens unmasked together within a block. 
        min_len is the shortest prompt length across the batch (post left-padding, this is the length of the initial full-block prefix all samples share). 
        seq_len is a [B] tensor of each sample's true prompt length (before left-padding), used to track per-sample block progress. mask_id is the token 
        id used to represent an unfilled position. threshold is the confidence above which a position is unmasked early, rather than waiting for its sub-block 
        to be forced to converge. stop_token is the token id that ends generation for a sample. use_block_cache, if True, uses the sub-block KV cache path 
        (faster, re-computes only the current block) instead of full re-forward. top_p is the nucleus sampling threshold, passed through to sample_with_top_p, 
        and temperature is the sampling temperature (0 = greedy), also passed through to sample_with_top_p.

        Returns a dict {original_batch_index: token_tensor}, one entry per input sample, each containing the full sequence (prompt + generated tokens).
        """
        num_blocks = max_new_tokens // block_size + seq_len.max().item() // block_size
        batch_size = input_ids.shape[0]

        # Prime the KV cache with every full block that every sample already shares (min_len // block_size blocks), so per-sample generation
        # below only has to handle the remaining partial block onward.
        if min_len > block_size:
            output = self.forward(
                input_ids = input_ids[:, :(min_len // block_size * block_size)],
                use_cache = True,
                update_past_key_values = True,
                block_size = block_size,
                # This call's logits are only ever read as [..., -1:, :] below: computing logits_to_keep = 0 (the default: every
                # position) means lm_head projects the entire prompt through the vocab (hidden_size x 151,936 x prompt_len x
                # batch), almost all of it immediately discarded. At batch = 64/2048-token context that one tensor alone is
                # ~37 GB in bf16: confirmed via OOM traceback pointing directly at this lm_head call. logits_to_keep = 1 returns
                # only the position actually used, with no change in behavior (verified by inspection: nothing downstream in
                # this branch reads any other position).
                logits_to_keep = 1,
            )
            logits, past_key_values = output.logits, output.past_key_values
            if min_len % block_size == 0:
                # min_len landed exactly on a block boundary: predict the first new token now for samples whose prompt ends here.
                predict_sample_idx = (seq_len == min_len)
                predict_logits = logits[predict_sample_idx, -1:, :]
                next_token = predict_logits.argmax(dim = -1)
                if input_ids.shape[1] <= min_len:
                    input_ids = torch.cat([input_ids, next_token], dim = 1)
                else:
                    input_ids[predict_sample_idx, min_len] = next_token.squeeze(dim = -1)
        else:
            past_key_values = None

        seq_block_idx = seq_len // block_size
        finished_flag = torch.zeros((batch_size), device = self.device, dtype = torch.bool)

        start_block_idx = min_len // block_size
        num_small_blocks = block_size // small_block_size

        # Track which row of the (shrinking) active batch corresponds to which original input sample, so finished_samples can be indexed correctly
        # even after finished rows are dropped mid-loop.
        sample_indices = torch.arange(batch_size, device = self.device)
        finished_samples = {}

        for block_idx in range(start_block_idx, num_blocks):
            if finished_flag.all():
                break
            if (seq_block_idx == block_idx).all():
                # Every remaining sample needs a fresh masked block appended.
                x_init = mask_id * torch.ones(
                    (input_ids.shape[0], block_size - input_ids.shape[1] % block_size),
                    device = self.device,
                    dtype = torch.long,
                )
                x_init = torch.cat([input_ids, x_init], dim = 1)
                input_ids = x_init
            else:
                # Some samples' prompts extend further into this block than
                # others: their existing tokens are reused as-is.
                x_init = input_ids[:, :(block_idx + 1) * block_size]

            # Samples that already finished don't need a fresh masked block; pad their tail with pad_token_id instead so they're inert.
            x_init[finished_flag, -block_size:] = tokenizer.pad_token_id
            x_t = x_init.clone()
            step = 0
            block_past_key_values = None

            while True:
                mask_idx = (x_t[:, -block_size:] == mask_id)

                # Fast path: the whole block is already unmasked (every position was committed during the sub-block loop below on
                # a previous pass through this while-loop). Commit the block to the persistent KV cache and predict the next token.
                if mask_idx.sum() == 0:
                    for sample_idx in range(x_t.shape[0]):
                        if finished_flag[sample_idx] and seq_len[sample_idx] < (block_idx + 1) * block_size:
                            stop_token_idx = (x_t[sample_idx, seq_len[sample_idx]:] == stop_token).nonzero()[0][0]
                            x_t[sample_idx, seq_len[sample_idx] + stop_token_idx + 1:] = tokenizer.pad_token_id
                    if finished_flag.all():
                        break
                    output = self.forward(
                        input_ids = x_t[:, -block_size:],
                        use_cache = True,
                        past_key_values = past_key_values,
                        update_past_key_values = True,
                        block_size = block_size,
                        # Same reasoning as the priming call above: only logits[:, -1:, :] is ever read below, so there's
                        # no need to project all block_size positions through lm_head. Smaller win per call than the
                        # priming fix (32x here vs. up to 2048x there), but this call happens once per generated block:
                        # over a full generation the cumulative saving adds up across many more calls than the one-time
                        # priming call gets.
                        logits_to_keep = 1,
                    )
                    logits, past_key_values = output.logits, output.past_key_values
                    next_token = logits[:, -1:, :].argmax(dim = -1)
                    next_token[finished_flag] = tokenizer.pad_token_id
                    x_t = torch.cat([x_t, next_token], dim = 1)
                    step += 1
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

                        if use_block_cache:
                            # First pass through this sub-block, or a mask changed upstream: rebuild the block cache from
                            # scratch. Otherwise, reuse it and only recompute the sub-block at replace_position.
                            if block_past_key_values is None or (x_t[:, -block_size + small_block_start_idx] == mask_id).any():
                                output = self.forward(
                                    input_ids = x_t[:, -block_size:],
                                    use_cache = True,
                                    past_key_values = past_key_values,
                                    update_past_key_values = False,
                                    use_block_cache = True,
                                )
                                logits, block_past_key_values = output.logits, output.block_past_key_values
                                # Shift logits right by one: logits[i] should predict token i (not i+1), matching the
                                # block-diffusion training objective.
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim = 1)
                                logits = logits[:, start:end]
                            else:
                                logits = self.forward(
                                    input_ids = x_t[:, start:end],
                                    use_cache = True,
                                    past_key_values = past_key_values,
                                    update_past_key_values = False,
                                    use_block_cache = True,
                                    block_past_key_values = block_past_key_values,
                                    replace_position = small_block_start_idx,
                                ).logits
                                logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim = 1)
                        else:
                            logits = self.forward(
                                input_ids = x_t[:, -block_size:],
                                use_cache = True,
                                past_key_values = past_key_values,
                                update_past_key_values = False,
                            ).logits
                            logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim = 1)
                            logits = logits[:, start:end]

                        x_1, p_1t = self.sample_with_top_p(logits, top_p = top_p, temperature = temperature)
                        # Confidence of the sampled token at each position.
                        x1_p = torch.squeeze(torch.gather(p_1t, dim = -1, index = torch.unsqueeze(x_1, -1)), -1)

                        # Only masked positions are eligible for unmasking.
                        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)

                        # Unmask every position above threshold, but always unmask at least the single most confident position
                        # so the sub-block is guaranteed to make progress.
                        unmask_idx = (x1_p > threshold)
                        max_prob_idx = x1_p.argmax(dim = -1)
                        unmask_idx[torch.arange(x_1.shape[0]), max_prob_idx] = True
                        unmask_idx = unmask_idx & mask_idx[:, start:end]

                        x_t[:, start:end][unmask_idx] = x_1[unmask_idx]

                        # A sample finishes the moment it unmasks stop_token.
                        finished_row_flags = ((x_1 == stop_token) & unmask_idx).any(dim = 1)
                        finished_flag = finished_flag | finished_row_flags

                        step += 1

            if input_ids.shape[1] == x_t.shape[1]:
                input_ids = x_t
            else:
                input_ids[:, :(block_idx + 1) * block_size] = x_t[:, :-1]
                if (seq_block_idx == block_idx).all():
                    input_ids = torch.cat([input_ids, x_t[:, -1:]], dim = 1)
                else:
                    if input_ids.shape[1] <= (block_idx + 1) * block_size:
                        input_ids = x_t
                    else:
                        input_ids[seq_block_idx == block_idx, (block_idx + 1) * block_size] = x_t[seq_block_idx == block_idx, (block_idx + 1) * block_size]

            seq_block_idx[seq_block_idx == block_idx] = block_idx + 1

            # Drop any samples that finished this block from the active batch: this includes dropping them from past_key_values, so subsequent
            # blocks don't waste compute on sequences that are already done.
            if finished_flag.any():
                for sample_idx in range(x_t.shape[0]):
                    if finished_flag[sample_idx]:
                        original_idx = sample_indices[sample_idx].item()
                        finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim = 0)

                sample_indices = sample_indices[~finished_flag]
                input_ids = input_ids[~finished_flag]
                seq_block_idx = seq_block_idx[~finished_flag]
                seq_len = seq_len[~finished_flag]
                x_t = x_t[~finished_flag]

                for layer_id in range(len(past_key_values)):
                    past_key_values.key_cache[layer_id] = past_key_values.key_cache[layer_id][~finished_flag]
                    past_key_values.value_cache[layer_id] = past_key_values.value_cache[layer_id][~finished_flag]

                finished_flag = finished_flag[~finished_flag]

        # Any samples still active when the block loop ends (hit num_blocks without emitting stop_token) are recorded here as "finished" too.
        if len(finished_samples) < batch_size:
            for sample_idx in range(x_t.shape[0]):
                original_idx = sample_indices[sample_idx].item()
                finished_samples[original_idx] = x_t[sample_idx:sample_idx + 1].clone().squeeze(dim = 0)

        assert len(finished_samples) == batch_size
        return finished_samples

    @torch.no_grad()
    def sample_with_top_p(self, logits, top_p = 0.95, temperature = 1.0):
        """
        Sample one token per position from logits, using nucleus (top-p) sampling when temperature > 0, or greedy argmax when temperature == 0. 
        self is a model instance (only used for tensor device/dtype context). logits is a [..., vocab_size] tensor of unnormalized logits. 
        top_p is the cumulative probability mass to keep before sampling, and temperature is the sampling temperature, where 0 means greedy (ignores top_p).

        Returns x_1, the [...] sampled token ids, and p_1t, the [..., vocab_size] (possibly top-p-truncated and renormalized) probability distribution actually sampled from.
        """
        if temperature > 0:
            scaled_logits = logits / temperature
        else:
            p_1t = torch.softmax(logits, dim = -1)
            x_1 = p_1t.argmax(dim = -1)
            return x_1, p_1t

        import torch.nn.functional as F
        probs = F.softmax(scaled_logits, dim = -1)
        sorted_probs, sorted_indices = torch.sort(probs, descending = True)
        cumulative_probs = torch.cumsum(sorted_probs, dim = -1)

        # Keep the smallest prefix of sorted probabilities whose cumulative mass exceeds top_p, always keeping at least the top-1 token (the
        # shift below un-marks index 0 for removal regardless of its own mass).
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = torch.zeros_like(probs, dtype = torch.bool).scatter_(
            dim = -1, index = sorted_indices, src = sorted_indices_to_remove
        )
        probs[indices_to_remove] = 0

        # Renormalize so the remaining (kept) probabilities sum to 1.
        probs_sum = torch.sum(probs, dim = -1, keepdim = True)
        normalized_probs = probs / probs_sum

        p_1t = normalized_probs
        x_1 = torch.multinomial(p_1t[0], num_samples = 1).unsqueeze(0).squeeze(-1)
        return x_1, p_1t

def setup_model(model):
    """
    Bind batch_sample and sample_with_top_p onto a model instance loaded via AutoModelForCausalLM.from_pretrained(..., trust_remote_code = True). 
    This is the correct way to use the official Fast-dLLM v2 generation loop. model is a loaded Fast-dLLM v2 model instance. Returns the same model instance, 
    with .mdm_sample and .sample_with_top_p bound.
    """
    model.mdm_sample = types.MethodType(Fast_dLLM_QwenForCausalLM.batch_sample, model)
    model.sample_with_top_p = types.MethodType(Fast_dLLM_QwenForCausalLM.sample_with_top_p, model)
    return model