"""Nulls and statistics (ТЗ E40_math_core §II.9, M8/M9; paper §6 items 22/23).

Consolidation target for the ad-hoc code in ``noise.py`` / ``permutation.py``
and the per-experiment bootstrap copies. The rule (item 23): any
``‖OX‖²/‖X‖²``-type claim needs BOTH nulls — the operator null (isotropic X:
how much ``O`` eats from random input) and a content-matched null (preserved
norms/marginals, randomized directions: separates data structure from
norm/aggregation artifacts).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "operator_null", "mp_upper_edge", "mp_lambda_max_mc", "reduction_control",
    "paired_bootstrap", "permutation_test", "resolution_floor",
]


def operator_null(O: np.ndarray, X: np.ndarray, kind: str = "iid",
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Random ``X`` of the same shape as ``X`` for the two-null protocol.

    ``kind="iid"`` — Gaussian, same total Frobenius norm (mechanical operator
    null: the returned X makes ``‖O X‖²/‖X‖²`` equal the isotropic reference).
    ``kind="content_matched"`` — per-row (token-axis) norms of ``X`` preserved,
    directions re-randomized: the dir-matched null of E-39 supplement / E44
    §5.1.1. Rows = axis ``-2`` (the token axis the operator acts on).
    """
    O = np.asarray(O, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    rng = rng or np.random.default_rng(0)
    if X.shape[-2] != O.shape[1]:
        raise ValueError(f"X token axis {X.shape[-2]} != O dim {O.shape[1]}")
    if kind == "iid":
        Z = rng.standard_normal(X.shape)
        return Z * float(np.sqrt((X ** 2).sum() / (Z ** 2).sum()))
    if kind == "content_matched":
        Z = rng.standard_normal(X.shape)
        Z = Z / np.linalg.norm(Z, axis=-1, keepdims=True)
        norms = np.linalg.norm(X, axis=-1, keepdims=True)
        return Z * norms
    raise ValueError(f"kind must be iid/content_matched, got {kind!r}")


def mp_upper_edge(p: int, n: int) -> float:
    """Marchenko–Pastur upper edge ``(1 + sqrt(p/n))²`` of ``XᵀX/n``, ``X``
    ``[n, p]`` — the normalization under which the ТЗ's «MP-скобки» reproduce:
    p=6, n=32 → 2.0535 ≈ 2.05; p=8, n=32 → 2.2500 = 2.25. The second numbers
    of the early ТЗ pairs are n=192 edges (1.3848 for p=6; p=8 gives 1.4425,
    NOT the quoted 1.45) — патч v2 §2.1 retracts both second numbers as
    n-unlabelled; finite-n MC means sit below the edge and are the plaintiff
    (user ruling 2026-11-03)."""
    if p <= 0 or n <= 0:
        raise ValueError("p and n must be positive")
    return float((1.0 + np.sqrt(p / n)) ** 2)


def mp_lambda_max_mc(p: int, n: int, B: int = 400,
                     rng: np.random.Generator | None = None) -> dict:
    """Monte-Carlo calibration of ``λ_max`` of ``XᵀX/n``, ``X`` ``[n, p]`` iid
    N(0,1): mean/median/quantiles vs the MP edge. Test ТЗ §IV.6."""
    rng = rng or np.random.default_rng(0)
    lams = np.empty(B)
    for b in range(B):
        X = rng.standard_normal((n, p))
        lams[b] = np.linalg.eigvalsh(X.T @ X / n)[-1]
    return {"p": p, "n": n, "B": B,
            "mean": float(lams.mean()), "median": float(np.median(lams)),
            "q05": float(np.quantile(lams, 0.05)), "q95": float(np.quantile(lams, 0.95)),
            "mp_edge": mp_upper_edge(p, n)}


def reduction_control(statistic: np.ndarray, trivial_predictor: np.ndarray,
                      partial_vars: np.ndarray | None = None) -> dict:
    """Standard pattern for derived metrics: regress the statistic on the
    trivial predictor (+ optional covariates), return the residual signal and
    how much the trivial model already explains. Guards «correlates with X»
    claims where X is mechanical (e.g. energy vs b_mag, E44 §5.1.1)."""
    y = np.asarray(statistic, dtype=np.float64).ravel()
    cols = [np.ones_like(y), np.asarray(trivial_predictor, dtype=np.float64).ravel()]
    if partial_vars is not None:
        P = np.atleast_2d(np.asarray(partial_vars, dtype=np.float64))
        if P.shape[0] < P.shape[1]:
            P = P.T
        cols += [P[:, j] for j in range(P.shape[1])]
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else 1.0
    return {"residual": resid, "r2_trivial": float(min(r2, 1.0)),
            "residual_std": float(resid.std(ddof=A.shape[1]))}


def paired_bootstrap(docs: np.ndarray, stat_fn, B: int = 1000,
                     seed: int = 0, ci: float = 95.0) -> dict:
    """Percentile bootstrap over document index (resampling the DOCS, not
    rows) — the paired CI convention of E-39f/E-39m artifacts."""
    docs = np.asarray(docs)
    rng = np.random.default_rng(seed)
    n = docs.shape[0]
    vals = np.array([stat_fn(docs[rng.integers(0, n, n)]) for _ in range(B)])
    a = (100.0 - ci) / 2.0
    return {"point": float(stat_fn(docs)),
            "ci_low": float(np.quantile(vals, a / 100.0)),
            "ci_high": float(np.quantile(vals, 1 - a / 100.0)),
            "B": B}


def permutation_test(observed: float, null_values: np.ndarray,
                     alternative: str = "greater") -> float:
    """One-sided p-value ``(1 + #{null ≥ obs}) / (B + 1)`` (``"less"`` mirrors).
    ``null_values`` come from ``prism.permutation.generate_null`` or
    :func:`operator_null` — this function only scores them."""
    nv = np.asarray(null_values, dtype=np.float64)
    B = nv.size
    if B == 0:
        raise ValueError("empty null distribution")
    hits = int((nv >= observed).sum()) if alternative == "greater" else int((nv <= observed).sum())
    return float((1 + hits) / (B + 1))


def resolution_floor(metric_fn, docs: np.ndarray,
                     subsamples: tuple[int, ...] = (8, 16, 32),
                     reps: int = 40, seed: int = 0) -> dict:
    """Per-test resolution floor: spread of ``metric_fn`` over subsamples of
    docs at each size (within the test, never a global constant — ТЗ §II.9).
    Compare an effect against ``floor_full`` BEFORE interpreting absence:
    effect below the floor ⇒ status ``not_testable``, not «null»."""
    docs = np.asarray(docs)
    rng = np.random.default_rng(seed)
    n = docs.shape[0]
    sizes = sorted(set(subsamples) | {n})
    floors = {}
    for m in sizes:
        if m > n:
            continue
        vals = [float(metric_fn(docs[rng.choice(n, m, replace=False)])) for _ in range(reps)]
        floors[m] = float(np.std(vals))
    return {"floor_by_size": floors, "floor_full": floors.get(n),
            "n_docs": n}
