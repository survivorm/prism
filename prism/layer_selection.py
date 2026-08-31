"""Layer re-selection rule (docs/exp17_change.md 0.1/0.2, user 2026-08-25).

l_S = full_attention small-model layers + l_S=0 (0_0) + last layer.
l_L = nearest preimage under uniform layer mapping (tie -> smaller l_L).
Iterate per selected l_S (mapping is many-to-one; per-l_L iteration dupes).
Classify per l_S layer type: full_attention -> "bridge" (o_proj exists),
else "kl_only" (GDN: no o_proj; edge pairs train lm_head LoRA only).

STANDALONE ON PURPOSE: no `prism.*` imports. The Kaggle kernel build inlines
each prism module as a notebook cell and strips prism imports only from the
EXPERIMENT cell, not from inlined modules — a bare `from prism.X import ...`
here would survive into the notebook and crash on Kaggle (no prism package)
with ModuleNotFoundError. The two small helpers are inlined here instead.
"""

from __future__ import annotations


def _uniform_layer_mapping(n_L: int, n_S: int) -> dict[int, int]:
    """Uniform large->small layer mapping: l_L -> round(l_L*(n_S-1)/(n_L-1)).

    Inlined copy of prism.calibration.uniform_layer_mapping (identical formula)
    so this module stays standalone for kernel inlining.
    """
    return {i: round(i * (n_S - 1) / (n_L - 1)) for i in range(n_L)}


def get_layer_types(model) -> list[str]:
    """Per-layer attention type from model.config.layer_types.

    Qwen3.5 style hybrid: entries 'full_attention' or 'linear_attention'.
    `model.config` (post-AutoModel load) exposes `layer_types` at top level
    (Qwen3_5TextConfig). Returns lowercase list.
    """
    types = getattr(model.config, "layer_types", None)
    if types is None:
        # Fallback: nested text_config (defensive; Qwen3_5 exposes top-level).
        tc = getattr(model.config, "text_config", None)
        types = getattr(tc, "layer_types", None)
    if types is None:
        raise RuntimeError(
            "model.config has no layer_types (and no text_config.layer_types)"
        )
    return [str(t) for t in types]


def select_pairs(n_L: int, n_S: int, layer_types: list[str]) -> list[tuple[int, int]]:
    """Return (l_L, l_S) pairs: full-attn l_S + 0_0 + last, sorted by l_S."""
    if n_L < 1 or n_S < 1 or len(layer_types) != n_S:
        raise ValueError(
            f"invalid: n_L={n_L}, n_S={n_S}, layer_types={len(layer_types)}"
        )
    mapping = _uniform_layer_mapping(n_L, n_S)
    full_S = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    selected = {0, n_S - 1} | full_S
    pairs = [_nearest_preimage(mapping, j) for j in sorted(selected)]
    if (n_L - 1, n_S - 1) not in pairs:
        pairs.append((n_L - 1, n_S - 1))
    return pairs


def _nearest_preimage(mapping: dict[int, int], j: int) -> tuple[int, int]:
    """l_L = argmin |mapping[l] - j|, tie -> smaller l (the selected l_S is j)."""
    best_l, best_d = 0, float("inf")
    for li in range(len(mapping)):
        d = abs(mapping[li] - j)
        if d < best_d:
            best_l, best_d = li, d
    return (best_l, j)


def classify_pair(l_S: int, layer_types: list[str]) -> str:
    """'bridge' if small-model layer l_S has o_proj (full_attention), else 'kl_only'."""
    return (
        "bridge"
        if l_S < len(layer_types) and layer_types[l_S] == "full_attention"
        else "kl_only"
    )
