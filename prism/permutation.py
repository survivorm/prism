"""Permutation max-statistic (FWER) and cluster-based alternative
(TZ part 3, §6).

No forward passes: works on already-computed scalars (from MetricCache)
and a paired permutation null generator. BH-FDR is deliberately NOT
implemented: Westfall–Young max-statistic controls FWER under arbitrary
correlation between layers (L18–L23 are strongly correlated), while
BH-FDR requires independence/PRDS. This is a choice, not a compromise.

paper.md §8.9, §13 — per-layer profiles and FWER control.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

__all__ = [
    "permutation_max_statistic",
    "generate_null",
    "cluster_based_permutation",
]


def _check_finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = int(np.argmax(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains NaN/Inf (first at flat index {bad}); "
            f"a bug in the null generator or metric is being hidden"
        )


def _check_common(
    observed: np.ndarray, null_samples: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    observed = np.asarray(observed, dtype=float)
    null_samples = np.asarray(null_samples, dtype=float)
    if observed.ndim != 1:
        raise ValueError(
            f"observed must be a flat [n_layers] profile, got shape {observed.shape}"
        )
    if null_samples.ndim != 2:
        raise ValueError(
            f"null_samples must be [B, n_layers], got shape {null_samples.shape}"
        )
    if null_samples.shape[1] != len(observed):
        raise ValueError(
            f"null_samples columns ({null_samples.shape[1]}) != len(observed) "
            f"({len(observed)})"
        )
    if null_samples.shape[0] < 100:
        raise ValueError(
            f"B = {null_samples.shape[0]} < 100: too few permutations for FWER"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    _check_finite("observed", observed)
    _check_finite("null_samples", null_samples)
    # Return the CONVERTED arrays: rebinding parameters alone would not
    # propagate to the callers, and cluster_based_permutation needs array
    # semantics (fancy indexing) — a list input must not TypeError there.
    return observed, null_samples


def permutation_max_statistic(
    observed: np.ndarray,
    null_samples: np.ndarray,
    alpha: float = 0.05,
) -> dict:
    """Westfall–Young max-statistic FWER over an [n_layers] profile.

    p = (1 + Σ_b [max_l |null[b, l]| ≥ max_l |observed[l]|]) / (B + 1) —
    the exact discrete permutation formula (Wright et al., 2015). The
    max over layers is what absorbs inter-layer correlation; no
    independence assumption.

    TZ selfcheck §6.6.3 is self-contradictory (asserts "all null maxes
    below max_obs" AND "k = 100, p = 1.0"); the formula above gives
    k = 0, p = 1/(B+1) in that case. The formula is followed.

    ``observed`` should be the same statistic that ``metric_fn`` returns
    in generate_null (e.g. per-layer median(edit − base) across docs).

    Returns:
        dict with ``p_value``, ``max_observed``, ``null_max_distribution``
        ([B]), ``significant`` (p < alpha), ``alpha``, ``n_layers``, ``B``.

    Errors:
        ValueError: shape mismatch, B < 100, alpha outside (0, 1),
            NaN/Inf in inputs.
    """
    observed, null_samples = _check_common(observed, null_samples, alpha)

    max_obs = float(np.max(np.abs(observed)))
    max_null = np.max(np.abs(null_samples), axis=1)  # [B]
    if max_obs == 0.0:
        p_value = 1.0
    else:
        k = int(np.sum(max_null >= max_obs))
        p_value = (1 + k) / (len(max_null) + 1)

    return {
        "p_value": float(p_value),
        "max_observed": max_obs,
        "null_max_distribution": max_null,
        "significant": p_value < alpha,
        "alpha": alpha,
        "n_layers": len(observed),
        "B": int(len(max_null)),
    }


def _paired_permutation_indices(n_docs: int, rng: np.random.Generator) -> np.ndarray:
    """Per-doc coin flip: True keeps (base, edit) as is, False swaps."""
    return rng.random(n_docs) < 0.5


def generate_null(
    base_values: np.ndarray,
    edit_values: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_permutations: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Paired permutation null [B, n_layers] (TZ §6.3).

    For each permutation, each document independently swaps its
    base/edit labels (coin flip) — the standard paired permutation test
    under H0: base and edit are exchangeable within pairs. ``metric_fn``
    is called B times with (base_perm, edit_perm) and must return the
    SAME statistic used for ``observed`` in permutation_max_statistic.

    Args:
        base_values, edit_values: [n_docs] or [n_docs, n_layers], same shape.
        metric_fn: (base_perm, edit_perm) -> [n_layers]; pure.
        n_permutations: B ≥ 100.
        seed: np.random.default_rng(seed) — deterministic, no global seed.

    Errors:
        ValueError: shape mismatch, n_docs < 2, B < 100, NaN/Inf,
            metric_fn output not a 1-D array.
    """
    base_values = np.asarray(base_values, dtype=float)
    edit_values = np.asarray(edit_values, dtype=float)
    if base_values.shape != edit_values.shape:
        raise ValueError(
            f"base_values shape {base_values.shape} != "
            f"edit_values shape {edit_values.shape}"
        )
    n_docs = base_values.shape[0]
    if n_docs < 2:
        raise ValueError(f"paired test requires n_docs >= 2, got {n_docs}")
    if n_permutations < 100:
        raise ValueError(f"n_permutations must be >= 100, got {n_permutations}")
    _check_finite("base_values", base_values)
    _check_finite("edit_values", edit_values)

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    for b in range(n_permutations):
        signs = _paired_permutation_indices(n_docs, rng)
        # signs True: keep (base, edit); False: swap within the pair.
        # 2-D ([n_docs, n_layers]): broadcast along ROWS — the per-doc coin
        # flip must swap the whole document row, not per column. (The bare
        # np.where(signs, ...) broadcasts signs left-padded to (1, n_docs),
        # which errors when n_docs != n_layers and silently swaps COLUMNS —
        # breaking the paired structure — when they are equal.)
        if base_values.ndim == 1:
            base_perm = np.where(signs, base_values, edit_values)
            edit_perm = np.where(signs, edit_values, base_values)
        else:
            s = signs[:, None]  # [n_docs, 1]
            base_perm = np.where(s, base_values, edit_values)
            edit_perm = np.where(s, edit_values, base_values)
        stat = np.asarray(metric_fn(base_perm, edit_perm), dtype=float)
        if stat.ndim != 1 or stat.size == 0:
            raise ValueError(
                f"metric_fn must return a non-empty 1-D [n_layers] array, "
                f"got shape {stat.shape} at permutation {b}"
            )
        if not np.all(np.isfinite(stat)):
            raise ValueError(
                f"metric_fn returned NaN/Inf at permutation {b}"
            )
        if rows and stat.shape != rows[0].shape:
            raise ValueError(
                f"metric_fn returned inconsistent shapes: {rows[0].shape} "
                f"vs {stat.shape} at permutation {b}"
            )
        rows.append(stat)
    return np.stack(rows, axis=0)


def _find_clusters(values: np.ndarray, threshold: float) -> list[list[int]]:
    """Contiguous runs of indices with |values[l]| > threshold."""
    mask = np.abs(values) > threshold
    clusters: list[list[int]] = []
    cur: list[int] = []
    for l in range(len(values)):
        if mask[l]:
            cur.append(l)
        elif cur:
            clusters.append(cur)
            cur = []
    if cur:
        clusters.append(cur)
    return clusters


def cluster_based_permutation(
    observed: np.ndarray,
    null_samples: np.ndarray,
    threshold: float = 2.0,
    alpha: float = 0.05,
) -> dict:
    """Cluster-based permutation FWER (Maris & Oostenveld, 2007).

    Groups adjacent super-threshold layers and tests the cluster sum
    Σ |values[l]| against the max cluster sum over the null. The caller
    must pass a NORMALIZED statistic (e.g. paired t or Cohen's d per
    layer); this function uses |values| as the statistic.

    Secondary to permutation_max_statistic: use it when the max-statistic
    p is slightly above alpha but 2–3 adjacent layers clearly show the
    effect (grouping raises power).

    Returns:
        dict with ``clusters`` (list of {"layers", "statistic", "p_value",
        "significant"}), ``null_max_cluster_distribution`` ([B]),
        ``alpha``, ``n_layers``, ``B``.

    Errors:
        ValueError: threshold ≤ 0, alpha outside (0, 1), shape issues,
            B < 100, NaN/Inf.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    observed, null_samples = _check_common(observed, null_samples, alpha)

    obs_clusters = _find_clusters(observed, threshold)

    # Null: for each b, threshold |null[b]|, sum |.| per cluster, take max
    # over clusters (0.0 if none).
    null_max_cluster = np.zeros(len(null_samples))
    for b in range(len(null_samples)):
        clusters = _find_clusters(null_samples[b], threshold)
        if clusters:
            null_max_cluster[b] = max(
                float(np.sum(np.abs(null_samples[b][c]))) for c in clusters
            )

    clusters_out = []
    for c in obs_clusters:
        stat = float(np.sum(np.abs(observed[c])))
        k = int(np.sum(null_max_cluster >= stat))
        p = (1 + k) / (len(null_max_cluster) + 1)
        clusters_out.append({
            "layers": c,
            "statistic": stat,
            "p_value": float(p),
            "significant": p < alpha,
        })

    return {
        "clusters": clusters_out,
        "null_max_cluster_distribution": null_max_cluster,
        "alpha": alpha,
        "n_layers": len(observed),
        "B": int(len(null_max_cluster)),
    }
