<div align="center">

## KINETIC: KV-Informed Neural Inference and Token-Skipping Execution Core

<p align="center">
  <img src="https://img.shields.io/badge/Kernel-Triton-blue?style=flat-square" alt="Kernel">
  <img src="https://img.shields.io/badge/Core-KV%20Block%20Eviction-green?style=flat-square" alt="Core">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square" alt="License">
</p>

*A Triton-native inference kernel that bounds prefix KV attention to a fixed top-K block budget for masked diffusion language models, without touching model weights or accuracy.*

</div>

**KINETIC** is a project developing a custom Triton kernel for **KV cache block eviction** and **block-sparse attention**, targeting inference speedups for [Fast-dLLM v2](https://github.com/NVlabs/Fast-dLLM), a masked diffusion language model that generates text block-by-block rather than token-by-token. The aim is to bound prefix attention cost to a fixed top-K block budget instead of letting it grow linearly with sequence length, without sacrificing the accuracy of the underlying model. Runtime wall-clock speedups on Fast-dLLM v2 and GSM8K should lie around the 1.3-1.4x range. Accuracy levels should also remain similar to the baseline accuracy at 70-80% (with 80% being slightly higher than the baseline).

> [!IMPORTANT]
> This kernel modifies the model's attention computation and KV cache handling. Any change to the eviction scheduler, block scoring, or sparse kernel path should be checked against the no-op correctness gate (`kernel/test_eviction.py --mode noop`) before being trusted on a real accuracy sweep.

## Further Exploration
This project builds directly on [Fast-dLLM v2](https://github.com/NVlabs/Fast-dLLM) and draws on established KV-cache eviction techniques from H2O and StreamingLLM-style attention-sink retention. The `kernel/` directory contains the eviction scheduler, the Triton sparse-attention kernel, and the benchmarking/test harnesses used to validate both correctness and speedup. The `model/` directory vendors the unmodified upstream Fast-dLLM v2 release that the kernel targets.

## Contacts
Feel free to reach out with questions or collaboration ideas.

## Citation
If you find this project useful, please give it a star and cite it via [**GitHub**](https://github.com/mikemao27/KINETIC). See `LICENSE.txt` (Apache 2.0) for terms of use and attribution.

```bibtex
@software{KINETIC,
  author = {Mao, Mike},
  title = {KINETIC: KV-Informed Neural Inference and Token-Skipping Execution Core},
  year = {2026},
  url = {https://github.com/mikemao27/KINETIC},
  version = {1.0.0}
}
```
