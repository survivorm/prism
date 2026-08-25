"""Layer re-selection rule (docs/exp17_change.md 0.1/0.2, user 2026-08-25).

l_S = full_attention small-model layers + l_S=0 (0_0) + last layer.
l_L = nearest preimage under uniform_layer_mapping (tie -> smaller l_L).
Iterate per selected l_S (mapping is many-to-one; per-l_L iteration dupes).
Classify per l_S layer type: full_attention -> "bridge" (o_proj exists),
else "kl_only" (GDN: no o_proj; edge pairs train lm_head LoRA only).
"""

from prism.calibration import uniform_layer_mapping


def select_pairs(n_L: int, n_S: int, layer_types: list[str]) -> list[tuple[int, int]]:
    """Return (l_L, l_S) pairs: full-attn l_S + 0_0 + last, sorted by l_S."""
    if n_L < 1 or n_S < 1 or len(layer_types) != n_S:
        raise ValueError(
            f"invalid: n_L={n_L}, n_S={n_S}, layer_types={len(layer_types)}"
        )
    try:
        mapping = uniform_layer_mapping(n_L, n_S)
    except Exception as e:
        raise ValueError(f"uniform_layer_mapping failed: {e!r}") from e
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
