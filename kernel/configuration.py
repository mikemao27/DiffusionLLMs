"""
Fast-dLLM Qwen Model Configuration.

Defines the HuggingFace-compatible configuration class for the Fast-dLLM v2 / Qwen2.5 architecture used throughout kernel/modeling.py. 
This is a standalone version of the configuration class, placed in kernel/ so that kernel/modeling.py can import it without a relative package import, 
and so the model registers correctly with HuggingFace's from_pretrained() machinery.

Interface contract (never changes): Fast_dLLM_QwenConfig is a plain PretrainedConfig subclass, and all fields are stored as plain attributes on the instance, 
exactly as HuggingFace's from_pretrained() / from_dict() / to_dict() machinery expects. The constructor never raises for malformed rope_scaling or layer_types; 
validation failures are swallowed so that a partially-specified config (e.g. one built by hand, rather than loaded from a checkpoint) still loads. 
This mirrors the upstream Qwen2 config behavior.
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)

def _layer_type_validation(layer_types):
    """
    Raises ValueError if any entry in layer_types is not a recognized attention type string. 
    layer_types is a list of per-layer attention type strings.
    """
    valid = {"full_attention", "sliding_attention"}
    for t in layer_types:
        if t not in valid:
            raise ValueError(f"layer_type must be one of {valid}, got {t!r}")

class Fast_dLLM_QwenConfig(PretrainedConfig):
    """
    Configuration for Fast-dLLM v2 / Qwen2.5. Inherits from PretrainedConfig so that from_pretrained() can load and populate all fields from config.json.

    vocab_size is the vocabulary size, hidden_size is the transformer hidden dimension, and intermediate_size is the MLP intermediate dimension. 
    num_hidden_layers is the number of decoder layers and num_attention_heads is the number of query heads. num_key_value_heads is the number of KV heads (GQA), 
    defaulting to num_attention_heads (no GQA) if left as None. hidden_act is the activation function name (e.g. "silu"). max_position_embeddings is the maximum sequence length for RoPE. 
    initializer_range is the standard deviation used for weight initialization, and rms_norm_eps is the epsilon added inside RMSNorm for numerical stability. use_cache controls whether 
    to use the KV cache during inference, and tie_word_embeddings controls whether to tie the input and output embeddings. rope_theta is the RoPE base frequency, and rope_scaling is an 
    optional dict configuring dynamic RoPE variants. use_sliding_window controls whether any layers use sliding-window attention, and sliding_window is the sliding window size, only used 
    if use_sliding_window. max_window_layers is the index at or above which layers use sliding attention, with layers below it using full attention. layer_types is a list of attention type strings, 
    one per layer, derived automatically from use_sliding_window / max_window_layers if left None. attention_dropout is the dropout probability applied inside attention, and bd_size is the 
    block-diffusion block size (tokens per generation block).
    """

    model_type = "Fast_dLLM_Qwen"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size = 151936,
        hidden_size = 4096,
        intermediate_size = 22016,
        num_hidden_layers = 32,
        num_attention_heads = 32,
        num_key_value_heads = 32,
        hidden_act = "silu",
        max_position_embeddings = 32768,
        initializer_range = 0.02,
        rms_norm_eps = 1e-6,
        use_cache = True,
        tie_word_embeddings = False,
        rope_theta = 10000.0,
        rope_scaling = None,
        use_sliding_window = False,
        sliding_window = 4096,
        max_window_layers = 28,
        layer_types = None,
        attention_dropout = 0.0,
        bd_size = 32,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        # GQA: if num_key_value_heads is left unset, fall back to full multi-head attention (one KV head per query head).
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.bd_size = bd_size

        # BC: some configs use the older key "type" instead of "rope_type".
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        # Validation is best-effort: a hand-built or partially-specified config should still load rather than crash at construction time.
        try:
            rope_config_validation(self)
        except Exception:
            pass

        # Auto-derive layer_types from use_sliding_window / max_window_layers when the caller hasn't specified them explicitly.
        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        try:
            _layer_type_validation(self.layer_types)
        except Exception:
            pass

        super().__init__(tie_word_embeddings = tie_word_embeddings, **kwargs)