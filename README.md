<h1 align = "center">Custom Kernels for Fast-dLLM Wall-clock Runtime Improvements</h1>

**DiffusionLLMs** is a project developing a custom Triton kernel for **KV cache block eviction** and **block-sparse attention**, targeting inference speedups for [Fast-dLLM v2](https://github.com/NVlabs/Fast-dLLM), a masked diffusion language model that generates text block-by-block rather than token-by-token. The aim is to bound prefix attention cost to a fixed top-K block budget instead of letting it grow linearly with sequence length, without sacrificing the accuracy of the underlying model.

*We're currently at a 1.41x wall-clock speedup with no accuracy loss relative to baseline, working toward a 2x target. The approach pairs an eviction scheduler that scores and retains only the most important prefix KV blocks with a Triton kernel that reads only those selected blocks from HBM, and experiments with INT8 weight quantization to attack the GEMM-bound cost that remains once attention is no longer the bottleneck.*

> [!IMPORTANT]
> This kernel modifies the model's attention computation and KV cache handling. Any change to the eviction scheduler, block scoring, or sparse kernel path should be checked against the no-op correctness gate (`kernel/test_eviction.py --mode noop`) before being trusted on a real accuracy sweep.

## Further Exploration
This project builds directly on [Fast-dLLM v2](https://github.com/NVlabs/Fast-dLLM) and draws on established KV-cache eviction techniques from H2O and StreamingLLM-style attention-sink retention. The `kernel/` directory contains the eviction scheduler, the Triton sparse-attention kernel, and the benchmarking/test harnesses used to validate both correctness and speedup.

## Contacts
Feel free to reach out with questions or collaboration ideas.

## Citation
If you find this project useful, please give it a star and cite it via [**GitHub**](https://github.com/mikemao27/DiffusionLLMs). See `LICENSE` (Apache 2.0) for terms of use and attribution.
