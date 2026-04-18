## Fast-dLLM, Diffusion and Parallel Decoding
The paper we are interested in here is titled "Fast-dLLM: Trainign-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding" (**http://nvlabs.github.io/Fast-dLLM/**).

Diffusion-based LLMs have shown promise for non-autoregressive text generation but, their inference speed often lags behind autoregressive models because of a lack of KV Caching (Key-Value Cache). They propose a confidence-aware parallel decoding strategy that only decodes tokens that exceed a certain confidence threshold (this solves the issue of quality degradation).
* Demonstrated a 27.6x throughput improvement.
* Minimum accuracy loss.