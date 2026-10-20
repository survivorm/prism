"""Zigzag persistent homology across point clouds + generic projections
(TZ part 3, §7).

Optional backends: giotto-tda and/or dionysus. This module imports
WITHOUT them; only ``zigzag_persistence`` requires a backend (lazy
ImportError with pip instructions). The projections are pure numpy.

Zigzag vs ordinary PH: ordinary persistence requires a NESTED filtration
(K_0 ⊂ K_1 ⊂ ...); zigzag allows BACKWARD steps (K_i → K_{i+1} → K_i),
so features can be tracked as they are born and die when moving between
layers (the clouds are not nested).

paper.md §8.10 — topology across layers.

NOT covered here (TZ §7.8): compare_tail_vs_pca (experiments/exp_T1.py),
total_persistence (caller), reducing differing d_l to a common d
(caller: use project_to_pca_subspace before the zigzag call — the
function does NOT project implicitly), loading tail_basis, plotting.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "zigzag_persistence",
    "project_to_subspace",
    "project_to_pca_subspace",
]


def _import_giotto():
    try:
        from gtda.diagrams import ZigzagPersistence
    except ImportError as e:
        raise ImportError(
            "giotto-tda is required for backend='giotto'. "
            "Install: pip install giotto-tda"
        ) from e
    return ZigzagPersistence


def _import_dionysus():
    try:
        import dionysus
    except ImportError as e:
        raise ImportError(
            "dionysus is required for backend='dionysus'. "
            "Install: pip install dionysus"
        ) from e
    return dionysus


def _validate_point_clouds(point_clouds: list[np.ndarray]) -> int:
    """Checks: len ≥ 2, each 2-D, same n_points, d ≥ 1, n_points ≥ 5,
    finite. Returns n_points."""
    if len(point_clouds) < 2:
        raise ValueError(
            f"need >= 2 point clouds for zigzag, got {len(point_clouds)}"
        )
    n_points = None
    for i, x in enumerate(point_clouds):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"point cloud {i} is {x.ndim}-D, need 2-D [n, d]")
        n, d = x.shape
        if d < 1:
            raise ValueError(f"point cloud {i} has d = 0")
        if n_points is None:
            n_points = n
        elif n != n_points:
            raise ValueError(
                f"point cloud {i} has n = {n} != {n_points} "
                f"(the same positions/docs across all clouds)"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError(f"point cloud {i} contains NaN/Inf")
    if n_points is None or n_points < 5:
        raise ValueError(
            f"n_points = {n_points} < 5: PH on < 5 points is trivial"
        )
    return n_points


def zigzag_persistence(
    point_clouds: list[np.ndarray],
    maxdim: int = 1,
    metric: str = "euclidean",
    backend: str = "giotto",
) -> dict:
    """Zigzag PH over a sequence of point clouds (layers).

    Maps between clouds are implicitly identity — the caller must reduce
    all clouds to a COMMON dimension first (e.g. project_to_pca_subspace
    to d=15, or tail modes with the same r_tail). Differing d_l is NOT
    handled here.

    Args:
        point_clouds: list of [n_points, d], same n_points (≥ 5), len ≥ 2.
        maxdim: 0 or 1 (H2 at N ~ 128 is noise).
        metric: "euclidean" | "cosine" | "precomputed" (distance matrices).
        backend: "giotto" (default) | "dionysus"; lazy import.

    Returns:
        dict with ``dgms`` (list of [n_features, 2] birth/death arrays,
        np.inf for essential features), ``n_layers``, ``maxdim``,
        ``backend``, ``metric``, ``n_points``. Feature order may differ
        between backends/versions — do not rely on it.

    Errors:
        ValueError: maxdim > 1, bad metric, cloud validation failures.
        ImportError: backend not installed (with pip instruction).
    """
    if maxdim not in (0, 1):
        raise ValueError(f"maxdim must be 0 or 1, got {maxdim}")
    if metric not in ("euclidean", "cosine", "precomputed"):
        raise ValueError(f"metric must be euclidean/cosine/precomputed, got {metric!r}")
    n_points = _validate_point_clouds(point_clouds)

    if backend == "giotto":
        ZigzagPersistence = _import_giotto()
        # gtda: list of point clouds; for "precomputed" the caller passes
        # distance matrices (accepted as-is).
        zz = ZigzagPersistence(metric=metric if metric != "precomputed" else "precomputed")
        # gtda ZigzagPersistence.fit_transform returns [n_features, 2]
        dgm = zz.fit_transform(point_clouds)
        dgms = [np.asarray(dgm, dtype=float)]
        if maxdim == 1:
            # gtda returns H0+H1 combined when maxdim allows; split by
            # persistence convention: H0 features are those born at 0.
            # HEURISTIC split (docs/changed_review.md, 2026-10): for zigzag
            # the first column may be a time index rather than a
            # filtration radius, so some H0 features can be born != 0 (and
            # some H1 at 0). The gtda API gives no explicit H0/H1
            # separation; treat dgms[0]/dgms[1] as "born-at-0" /
            # "born-at-other", NOT as guaranteed H0/H1.
            h0 = dgm[np.asarray(dgm, dtype=float)[:, 0] == 0.0]
            h1 = dgm[np.asarray(dgm, dtype=float)[:, 0] != 0.0]
            dgms = [np.asarray(h0, dtype=float), np.asarray(h1, dtype=float)]
    elif backend == "dionysus":
        dionysus = _import_dionysus()
        # dionysus API differs across versions; use the zigzag helper when
        # available. This path is exercised only if dionysus is installed.
        if not hasattr(dionysus, "zigzag_persistence"):
            raise ImportError(
                "dionysus.zigzag_persistence not available in this version; "
                "use backend='giotto'"
            )
        res = dionysus.zigzag_persistence(point_clouds, maxdim=maxdim)
        dgms = [np.asarray(r, dtype=float) for r in res]
    else:
        raise ValueError(f"unknown backend {backend!r} (giotto | dionysus)")

    return {
        "dgms": dgms,
        "n_layers": len(point_clouds),
        "maxdim": maxdim,
        "backend": backend,
        "metric": metric,
        "n_points": n_points,
    }


def project_to_subspace(h: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project h onto the subspace spanned by ``basis``.

    ``basis`` is [r, D] (rows = basis vectors, default) or [D, r]
    (columns). Disambiguation: if basis.shape[1] == h.shape[-1] → rows;
    if basis.shape[0] == h.shape[-1] → cols; if BOTH match (square) →
    ValueError. Orthonormality is NOT checked (caller's responsibility;
    V_r from top_r_svd is already orthonormal).

    Returns [N, r] (or [r] for 1-D h), dtype float32.

    Errors:
        ValueError: incompatible shapes, ambiguous square basis,
            basis.dim() != 2.
    """
    h = np.asarray(h, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    if basis.ndim != 2:
        raise ValueError(f"basis must be 2-D, got {basis.ndim}-D")
    d = h.shape[-1]
    rows_ok = basis.shape[1] == d
    cols_ok = basis.shape[0] == d
    if rows_ok and cols_ok:  # square basis: compatible both ways
        raise ValueError(
            f"basis shape {basis.shape} is square and compatible both ways; "
            f"pass an explicit [r, D] or [D, r] basis"
        )
    if rows_ok:
        return h @ basis.T
    if cols_ok:
        return h @ basis
    raise ValueError(
        f"incompatible shapes: h last dim = {d}, basis = {basis.shape} "
        f"(need basis [r, {d}] or [{d}, r])"
    )


def project_to_pca_subspace(
    h_batch: np.ndarray,
    n_components: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """PCA to n_components (centered; numpy SVD, float32).

    Returns (h_proj, components): h_proj [N, n_components] (centered —
    add the mean back manually for uncentered projections), components
    [n_components, D] (rows, orthonormal). Signs of components may differ
    between numpy versions — do not rely on them.

    Errors:
        ValueError: N < 2, n_components > min(N, D).
    """
    h = np.asarray(h_batch, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError(f"h_batch must be 2-D [N, D], got {h.ndim}-D")
    n, d = h.shape
    if n < 2:
        raise ValueError(f"N = {n} < 2: PCA undefined")
    if n_components > min(n, d):
        raise ValueError(
            f"n_components = {n_components} > min(N, D) = {min(n, d)}"
        )
    centered = h - h.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].astype(np.float32)
    h_proj = (centered @ components.T).astype(np.float32)
    return h_proj, components
