"""Tiny Qwen3_5 test model (mirrors the 0.8B architecture at reduced size).

Used by A/B-gate tests: real 3mod4 hybrid architecture (GDN + full-attn),
o_proj present on full-attn layers, D=64 so a full [D,D] Jacobian is cheap
while the matrix-free code path is exercised exactly as in production.
"""

from transformers.models.qwen3_5 import (  # type: ignore[import-not-found]
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
)


def make_tiny_qwen(
    n_layers: int = 8,
    hidden: int = 64,
    n_heads: int = 2,
    head_dim: int = 32,
    vocab: int = 64,
    intermediate: int = 128,
    interval: int = 4,
    attn_implementation: str | None = None,
) -> Qwen3_5ForConditionalGeneration:
    text = {
        "model_type": "qwen3_5",
        "hidden_size": hidden,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": max(1, n_heads // 2),
        "head_dim": head_dim,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "full_attention_interval": interval,
        "layer_types": [
            "full_attention" if i % interval == interval - 1 else "linear_attention"
            for i in range(n_layers)
        ],
        "rms_norm_eps": 1e-6,
        "use_cache": True,
        "tie_word_embeddings": True,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
    }
    config = Qwen3_5Config(
        text_config=text,
        image_token_id=vocab - 1,
        video_token_id=vocab - 2,
        vision_start_token_id=vocab - 3,
        vision_end_token_id=vocab - 4,
        tie_word_embeddings=True,
    )
    if attn_implementation is not None:
        config.text_config._attn_implementation = attn_implementation
        config._attn_implementation = attn_implementation
    return Qwen3_5ForConditionalGeneration(config)


def full_attn_layers(model) -> list[int]:
    return [
        i
        for i, t in enumerate(model.config.text_config.layer_types)
        if t == "full_attention"
    ]
