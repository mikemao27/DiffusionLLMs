# DiffusionLLMs
A custom Triton kernel implementing **paged KV cache block eviction** for
[Fast-dLLM v2](https://github.com/NVlabs/Fast-dLLM), targeting 2–3× inference
speedup on reasoning tasks (GSM8K) with minimal accuracy degradation.

## Motivation
Fast-dLLM v2 is a masked diffusion language model that generates text
block-by-block rather than token-by-token.  Each completed block is committed
to a growing KV cache that every subsequent block attends over.  As generation
progresses, this prefix attention cost grows linearly with sequence length —
the primary bottleneck for longer outputs.

Empirically, transformer attention is highly sparse at the block level: a
small subset of past blocks receives the vast majority of attention weight
(attention sinks, recent context, a few high-salience spans).  This means we
can **evict low-importance blocks from the KV cache** and attend only over the
retained top-k blocks, bounding attention cost to `O(top_k × block_size)`
regardless of total sequence length.

## Architecture

```
DiffusionLLMs/
├── kernel/
│   ├── paged_kv_cache.py         # Block allocator + paged DynamicCache subclass
│   ├── eviction_scheduler.py     # EMA block scoring + top-k eviction policy
│   ├── block_sparse_attention.py # Triton paged attention kernel + model hook
│   └── test_kernel.py            # Benchmark: baseline vs. paged on GSM8K prompts
└── Fast-dLLM/
    └── v2/
        └── generation_functions.py   # Unmodified Fast-dLLM v2 generation loop
```

### `paged_kv_cache.py`
Three classes with clean separation of concerns:

- **`BlockAllocator`** — CPU-side free-list over integer physical slot indices.
  O(1) allocate/free, no GPU memory.
- **`KVBlockPool`** — pre-allocated GPU tensors shaped
  `[max_slots, num_kv_heads, block_size, head_dim]` per layer per K/V.
  Flat slot-indexed layout lets the Triton kernel do `pool[slot]` without
  pointer chasing.
- **`PagedKVCache`** — subclasses `transformers.DynamicCache`, making it a
  drop-in replacement.  Inside a diffusion block it behaves like `DynamicCache`
  (normal concatenation) so the model's intra-block SDPA is untouched.  At
  block boundaries it commits the completed block to the pool atomically across
  all layers.  No changes to `generation_functions.py` are required.

### `eviction_scheduler.py`
- **Scoring proxy:** mean L2 norm of key vectors per block, averaged over
  sampled layers (every 4th by default).  High-norm blocks are more
  "distinctive" and typically receive more attention — a proxy used by H2O
  and StreamingLLM.  No extra forward pass required.
- **EMA smoothing:** `new = α × old + (1−α) × raw` (default α=0.9) to
  prevent transient low scores from evicting important blocks.
- **Protected blocks:** block 0 (attention sink) and the most-recently
  committed block are always retained regardless of score.
- **Top-k selection:** remaining budget filled by highest-scoring candidates.

### `block_sparse_attention.py`
Triton kernel (`_paged_attn_kernel`) with a 2D programme grid:
- Axis 0: query head
- Axis 1: query tile (BLOCK_Q = 32 positions)

Inner loop iterates over active physical slots from the block table, loading
each KV block and accumulating with the **FlashAttention-2 online softmax**
(running max + rescaled accumulator), so the full attention matrix is never
materialised.  GQA fan-out (e.g. Qwen2.5-7B: 28 Q heads, 4 KV heads) is
handled inside the kernel via `kv_h_idx = qh_idx // gqa_ratio`.

A Python hook (`install_paged_attention_hook`) patches each attention layer to
blend paged prefix attention with standard intra-block SDPA — no weight
changes or fine-tuning needed.

## Usage

```bash
cd DiffusionLLMs

# 7B model, full benchmark
python kernel/test_kernel.py \
    --model_path Efficient-Large-Model/Fast_dLLM_v2_7B \
    --block_size 32 \
    --top_k 4 \
    --max_new_tokens 256 \
    --num_prompts 10 \
    --output_json results.json

# 1.5B model, quick iteration
python kernel/test_kernel.py \
    --model_path Efficient-Large-Model/Fast_dLLM_v2_1.5B \
    --block_size 32 \
    --top_k 4 \
    --max_new_tokens 256 \
    --num_prompts 5
```

### Key Arguments
| Argument | Default | Description |
|---|---|---|
| `--model_path` | `Fast_dLLM_v2_1.5B` | HuggingFace model ID |
| `--block_size` | `32` | Tokens per KV block (must match Fast-dLLM v2) |
| `--top_k` | `4` | Blocks to retain after eviction (incl. sink + latest) |
| `--max_new_tokens` | `256` | Generation length |
| `--threshold` | `0.9` | Fast-dLLM v2 confidence threshold for token unmasking |
| `--num_prompts` | `5` | Number of GSM8K prompts to benchmark (max 10) |
| `--output_json` | `None` | Optional path to write full results as JSON |

### Tuning `top_k`
`top_k` directly controls the accuracy/speed tradeoff:

- Higher `top_k` → more context retained → closer to baseline accuracy, less speedup
- Lower `top_k` → faster attention → more aggressive eviction, potential quality loss

For short GSM8K examples (< 8 blocks total) `top_k=4` retains half the cache.
Start there and increase if answer agreement drops below ~90%.

## Expected Results
On GSM8K-style prompts with `max_new_tokens=256`, `block_size=32`, `top_k=4`:

| Metric | Expected |
|---|---|
| Speedup over baseline | 2–3× |
| Answer agreement with baseline | ≥ 90% |

The speedup comes from reducing prefix attention from O(n\_blocks) to O(top\_k)
per forward pass inside the diffusion loop, where n\_blocks grows with sequence
length and top\_k is fixed.

## Dependencies
- Python ≥ 3.10
- PyTorch ≥ 2.1
- Triton ≥ 2.2
- `transformers` ≥ 4.40
- Fast-dLLM v2 (at `DiffusionLLMs/Fast-dLLM/v2/`)