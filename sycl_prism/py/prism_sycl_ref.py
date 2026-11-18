"""Эталон прибора на numpy — семантика 1:1 с `prism/field.py` (канон AT0).

Этот файл — ОРАКЕЛ для SYCL-теста: `test_parity.py` сравнивает вывод SYCL-библиотеки
с этим эталоном, а не с «правдой на глаз».

Точность: эталон считает в float64 (как field.py). SYCL-ядра принимают float32/float16
на входе и аккумулируют в float32 ⇒ допуски в тесте зависят от входного dtype.

Проверено: numpy-эталон этого файла совпадает с `prism.field` на синтетике
(см. `_validate_against_field.py`).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "vpath",
    "dcurve",
    "acf_moments",
    "f_dc",
    "ra_split",
    "rho_t",
    "permutation_indices",
    "perm_stat",
]


def _unit(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)


def vpath(u: np.ndarray, hn: np.ndarray) -> tuple[np.ndarray, int]:
    """Ортогонализованный тангенс (канон dip_V). Идентична field.vpath.

    ВНИМАНИЕ: умножение на hn математически сокращается (U = u*hn, затем
    нормировка) ⇒ hn на результат не влияет (до fp). Оставлено ради паритета.
    """
    U = np.asarray(u, dtype=np.float64) * np.asarray(hn, dtype=np.float64)[:, None]
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    T = U.shape[0]
    Vo = np.empty((T - 1, U.shape[1]), dtype=np.float64)
    for t in range(T - 1):
        Vo[t] = U[t + 1] - float(U[t + 1] @ U[t]) * U[t]
    ok = np.linalg.norm(Vo, axis=1) > 1e-6
    V = Vo[ok] / np.maximum(np.linalg.norm(Vo[ok], axis=1, keepdims=True), 1e-12)
    return V, int((~ok).sum())


def rho_t(P: np.ndarray, k: int) -> np.ndarray:
    """Per-t косинусы ⟨P̂_t, P̂_{t+k}⟩ (len T−k)."""
    Pn = _unit(P)
    T = Pn.shape[0]
    if k == 0:
        return np.ones(T, dtype=np.float64)
    return np.einsum("ij,ij->i", Pn[:-k], Pn[k:])


def dcurve(P: np.ndarray, ks, stat: str = "median") -> np.ndarray:
    """Кривая: для каждого k — mean|median per-t косинусов (нормированных строк)."""
    ks = list(ks)
    out = np.empty(len(ks), dtype=np.float64)
    for i, k in enumerate(ks):
        out[i] = float(getattr(np, stat)(rho_t(P, k)))
    return out


def acf_moments(x: np.ndarray, K: int) -> dict:
    """Моментный ACF: c0, c[1..K], rho, rho_Tk, r, sum_rho_full, gram_resid."""
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    if not (0 < K <= N - 1):
        raise ValueError(f"acf_moments: K={K} вне диапазона [1, {N - 1}] для N={N}")
    c0 = float(np.einsum("ij,ij->i", x, x).sum())
    if c0 <= 0:
        raise ValueError("c(0) = 0: вырожденный объект")
    Kfull = N - 1
    cfull = np.empty(Kfull, dtype=np.float64)
    for k in range(1, Kfull + 1):
        cfull[k - 1] = float(np.einsum("ij,ij->i", x[:N - k], x[k:]).sum())
    e = np.einsum("ij,ij->i", x, x)
    S = np.cumsum(e)
    d = S[N - 2 - np.arange(K)]
    r = float(np.linalg.norm(x.sum(axis=0)) ** 2 / c0)
    gram_resid = abs(1.0 + 2.0 * cfull.sum() / c0 - r) / max(1.0, r)
    return {
        "c0": c0,
        "c": cfull[:K],
        "rho": cfull[:K] / c0,
        "rho_Tk": cfull[:K] / d,
        "r": r,
        "sum_rho_full": float(cfull.sum() / c0),
        "gram_resid": float(gram_resid),
    }


def f_dc(x: np.ndarray) -> float:
    """(G3) f_DC = ‖mean(x)‖² / mean(‖x‖²)."""
    x = np.asarray(x, dtype=np.float64)
    m2 = float(np.linalg.norm(x.mean(axis=0)) ** 2)
    mean_n2 = float(np.einsum("ij,ij->i", x, x).mean())
    if mean_n2 <= 0:
        raise ValueError("f_dc: нулевой объект (mean ‖x‖² = 0)")
    return m2 / mean_n2


def ra_split(u: np.ndarray, hn: np.ndarray) -> dict:
    """(G6) R/A-split Δh = r̄·Δu + Δr·ū + closure/cross санити."""
    U = _unit(u)
    hn = np.asarray(hn, dtype=np.float64)
    h = U * hn[:, None]
    dh = h[1:] - h[:-1]
    du = U[1:] - U[:-1]
    rbar = 0.5 * (hn[1:] + hn[:-1])
    u_mid = 0.5 * (U[1:] + U[:-1])
    dr = hn[1:] - hn[:-1]
    radial = dr[:, None] * u_mid
    angular = rbar[:, None] * du
    closure_max = float(
        np.max(
            np.abs(
                np.einsum("ij,ij->i", dh, dh)
                - np.einsum("ij,ij->i", radial, radial)
                - np.einsum("ij,ij->i", angular, angular)
            )
        )
    )
    cross_max = float(np.max(np.abs(np.einsum("ij,ij->i", du, u_mid))))
    return {
        "dh": dh,
        "du": du,
        "radial": radial,
        "angular": angular,
        "rbar": rbar,
        "dr": dr,
        "u_mid": u_mid,
        "closure_max": closure_max,
        "du_u_cross_max": cross_max,
    }


# ------------------------- перестановочный null-слой -------------------------
# Протокол AT0_1_accounting._perm_stat (experiments/AT0_1_accounting.py:396-419).
# Индексы перестановки генерируются numpy default_rng — тот же поток, что в серии,
# поэтому null-распределение остаётся ПОВТОРЯЕМЫМ при переносе арифметики на GPU.


def permutation_indices(n: int, seed: int, blk: int = 2) -> np.ndarray:
    """Индексы block-blk перестановки: 1:1 с AT0_1_accounting._perm_stat."""
    nb, rem = divmod(n, blk)
    rng = np.random.default_rng(seed)
    bp = rng.permutation(nb)
    idx = np.empty(n, dtype=np.int64)
    core = np.empty(nb * blk, dtype=np.int64)
    for j in range(blk):
        core[j::blk] = bp * blk + j
    idx[:nb * blk] = core
    if rem:
        idx[nb * blk:] = np.arange(nb * blk, n)
    return idx


def perm_stat(x: np.ndarray, f: float, seed: int, blk: int = 2, K: int = 96):
    """Статистика одной block-blk перестановки: (band, e3, e4).

    1:1 с AT0_1_accounting._perm_stat: S_band = mean_{k=9..K}(ρ̄_Tk(k) − f),
    e3 = ρ̄_Tk(3) − f, e4 = ρ̄_Tk(4) − f; f = f_dc(x) инвариантна к перестановке.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    idx = permutation_indices(n, seed, blk)
    xp = x[idx]
    e = np.einsum("ij,ij->i", xp, xp)
    S = np.cumsum(e)
    e3 = float(np.einsum("ij,ij->i", xp[: n - 3], xp[3:]).sum()) / float(S[n - 4]) - f
    e4 = float(np.einsum("ij,ij->i", xp[: n - 4], xp[4:]).sum()) / float(S[n - 5]) - f
    K = min(K, n - 1)
    vals = []
    for k in range(9, K + 1):
        ck = float(np.einsum("ij,ij->i", xp[: n - k], xp[k:]).sum())
        vals.append(ck / float(S[n - 1 - k]))
    return float(np.mean(vals)) - f, e3, e4


# --------------------- AT0_3 (TZ_sycl_new_ops v5): numpy-эталон f64 ---------------------

def gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C = A @ B (f64-эталон, row-major)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return a @ b


def eig(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Полный спектр симметричного батча (f64): (vals по возрастанию [b,n],
    vecs [b,n,n] — столбцы). Эталон eigjac."""
    a = np.asarray(a, dtype=np.float64)
    vals, vecs = np.linalg.eigh(a)   # vals уже по возрастанию (порядок torch)
    return vals, vecs


def _eig_subspace_resid(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    """‖AV − V(VᵀAV)‖_F/‖A‖_F по батчу (a [b,n,n], v [b,n,r])."""
    w = a @ v
    p = v.transpose(0, 2, 1) @ w
    r = w - v @ p
    return np.linalg.norm(r, axis=(1, 2)) / np.maximum(np.linalg.norm(a, axis=(1, 2)), 1e-300)


def eig_top(a: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """r верхних собственных пар (f64): (vals [b,r], vecs [b,n,r] — столбцы,
    resid [b] — подпространственный остаток). Эталон eigtop."""
    a = np.asarray(a, dtype=np.float64)
    vals, vecs = np.linalg.eigh(a)
    r = int(r)
    return vals[:, -r:], vecs[:, :, -r:], _eig_subspace_resid(a, vecs[:, :, -r:])


def eig_bottom(a: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """r нижних собственных пар (f64): (vals [b,r], vecs [b,n,r], resid [b]).
    Эталон eigbot."""
    a = np.asarray(a, dtype=np.float64)
    vals, vecs = np.linalg.eigh(a)
    r = int(r)
    return vals[:, :r], vecs[:, :, :r], _eig_subspace_resid(a, vecs[:, :, :r])


def svd_top(a: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """top-r SVD (f64, конвенция torch: S descending): (U [m,r], S [r], Vh [r,n]).
    Эталон ps.svd_top."""
    a = np.asarray(a, dtype=np.float64)
    U, S, Vh = np.linalg.svd(a, full_matrices=False)
    r = int(r)
    return U[:, :r], S[:r], Vh[:r, :]
