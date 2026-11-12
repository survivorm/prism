#!/usr/bin/env python3
"""prism/field.py — канонический оценщик серии AT (ТЗ v1.1 §0, гейт G1′).

Один прибор на все клетки T0–T3. Чистая переписывающая реализация (AST-сплайсинг
запрещён): поведению обязан совпадать с сериийной батареей (curve_u/Vpath, dcurve) —
поведенческий якорь проверяется в AT0_1 (LADDER_V2 dip_V,med = −0.0722).

Ядро (все формулы — ТЗ §1.3, индексы T-нормировка):
  c(0) = Σ_t ‖x_t‖²,  c(k) = Σ_{t=0}^{N−1−k} ⟨x_t, x_{t+k}⟩,  ρ̂(k) = c(k)/c(0)
  (G1) r ≡ ‖Σ_t x_t‖² / c(0) = 1 + 2 Σ_{k≥1} ρ̂(k)          — точно, для любого объекта
  (G2) ρ̂(k) = (1 − k/N)·ρ̄(k)  — ТОЛЬКО для ‖x_t‖ ≡ 1
  (G3) r = N·f_DC,  f_DC ≡ ‖mean(x)‖² / mean(‖x‖²)            — тождество
  (G6) Δh = r̄·Δu + Δr·ū,  r̄=(r_t+r_{t+1})/2, ū=(u_t+u_{t+1})/2,
       ⟨Δu,ū⟩ = 0 ТОЧНО,  ‖Δh‖² = r̄²‖Δu‖² + (Δr)²‖ū‖²
  (G7) r и Σρ̂ инвариантны к любой перестановке приращений
"""
import hashlib


import numpy as np

__all__ = [
    "vpath", "dcurve", "acf_moments", "rho_t", "f_dc",
    "ra_split", "cross_acf", "degen", "jclean", "sha256_file",
]


def vpath(u, hn):
    """Ортогонализованный тангенс (канон dip_V). Тождественен curve_u/Vpath батареей.

    u (T,D) — unit-состояния, hn (T,) — их нормы. Возвращает (V, dropped):
    V (T−1−m, D) — unit-векторы Vo[t] = U[t+1] − ⟨U[t+1],U[t]⟩·U[t], нормированные;
    rows с ‖Vo‖ ≤ 1e-6 выбрасываются (m = dropped).
    """
    U = np.asarray(u, dtype=np.float64) * np.asarray(hn, dtype=np.float64)[:, None]
    U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    T = U.shape[0]
    Vo = np.empty((T - 1, U.shape[1]))
    for t in range(T - 1):
        Vo[t] = U[t + 1] - float(U[t + 1] @ U[t]) * U[t]
    ok = np.linalg.norm(Vo, axis=1) > 1e-6
    V = Vo[ok] / np.maximum(np.linalg.norm(Vo[ok], axis=1, keepdims=True), 1e-12)
    return V, int((~ok).sum())


def _unit(P):
    return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)


def rho_t(P, k):
    """Per-t косинусы ⟨P̂_t, P̂_{t+k}⟩, массив длины T−k (P нормируется по строкам).

    k = 0 → ones(T) (косинус с самим собой); k < 0 или k ≥ T — ValueError (fail-loud).
    """
    Pn = _unit(np.asarray(P, dtype=np.float64))
    T = Pn.shape[0]
    if k < 0 or k >= T:
        raise ValueError(f"rho_t: k={k} вне диапазона [0, {T - 1}]")
    if k == 0:
        return np.ones(T, dtype=np.float64)
    return np.einsum("ij,ij->i", Pn[:-k], Pn[k:])


def dcurve(P, kk, stat="median"):
    """Кривая dip: для каждого k — stat (mean|median) per-t косинусов. (b4-батарея)."""
    return np.array([float(getattr(np, stat)(rho_t(P, k))) for k in kk])


def acf_moments(x, K):
    """Моментный ACF (T-нормировка, точный G1-объект). x (N,D) — приращения, K ∈ 1..N−1.

    gram-аудит (G8) — по ВСЕМ лагам k=1..N−1 (тождество G1 срезается только на
    полном суммировании); отчитываемый rho — k=1..K.
    Возвращает dict: c0, c (K,), rho (K,) = c(k)/c0,
    rho_Tk (K,) = c(k)/Σ_{t<N−k}‖x_t‖² — точная T−k-нормировка (для unit-объекта
    это ровно ρ̄(k) = mean_t cos, для переменных норм — взвешенный знаменатель, ТЗ §1.4-1),
    r = ‖Σx‖²/c0,
    gram_resid = |1 + 2Σ_{k<N} ρ̂(k) − r| / max(1, r).
    """
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
    d = S[N - 2 - np.arange(K)]          # d[k−1] = S[N−2−(k−1)] = Σ_{t=0}^{N−1−k} ‖x_t‖²
    r = float(np.linalg.norm(x.sum(axis=0)) ** 2 / c0)
    gram_resid = abs(1.0 + 2.0 * cfull.sum() / c0 - r) / max(1.0, r)
    return {"c0": c0, "c": cfull[:K], "rho": cfull[:K] / c0, "rho_Tk": cfull[:K] / d,
            "r": r,
            "sum_rho_full": float(cfull.sum() / c0), "gram_resid": float(gram_resid)}


def f_dc(x):
    """(G3) f_DC = ‖mean(x)‖² / mean(‖x‖²); r = N·f_DC точно. Нулевой объект — ValueError."""
    x = np.asarray(x, dtype=np.float64)
    m2 = float(np.linalg.norm(x.mean(axis=0)) ** 2)
    mean_n2 = float(np.einsum("ij,ij->i", x, x).mean())
    if mean_n2 <= 0:
        raise ValueError("f_dc: нулевой объект (mean ‖x‖² = 0)")
    return m2 / mean_n2


def ra_split(u, hn):
    """(G6) точный R/A-split Δh = r̄·Δu + Δr·ū.

    u (T,D) — unit-состояния (перенормировываются), hn (T,) — их нормы.
    Возвращает dict с рядами (длиной T−1): dh, du, radial, angular,
    plus sanity: closure_max = max_t |‖Δh_t‖² − ‖radial_t‖² − ‖angular_t‖²|,
    du_u_cross_max = max_t |⟨Δu_t, ū_t⟩| (должно быть 0 до fp).
    """
    U = _unit(np.asarray(u, dtype=np.float64))
    h = U * np.asarray(hn, dtype=np.float64)[:, None]
    dh = h[1:] - h[:-1]
    du = U[1:] - U[:-1]
    rbar = 0.5 * (np.asarray(hn, dtype=np.float64)[1:] + np.asarray(hn, dtype=np.float64)[:-1])
    u_mid = 0.5 * (U[1:] + U[:-1])
    dr = np.asarray(hn, dtype=np.float64)[1:] - np.asarray(hn, dtype=np.float64)[:-1]
    radial = dr[:, None] * u_mid
    angular = rbar[:, None] * du
    closure_max = float(np.max(np.abs(
        np.einsum("ij,ij->i", dh, dh)
        - np.einsum("ij,ij->i", radial, radial)
        - np.einsum("ij,ij->i", angular, angular))))
    cross_max = float(np.max(np.abs(np.einsum("ij,ij->i", du, u_mid))))
    return {"dh": dh, "du": du, "radial": radial, "angular": angular,
            "rbar": rbar, "dr": dr, "u_mid": u_mid,
            "closure_max": closure_max, "du_u_cross_max": cross_max}


def cross_acf(a, b, K):
    """Кросс-ACF c(k) = Σ_t a_t·b_{t+k} (для T0.2b: radial×angular кросс-члены)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    N = min(a.shape[0], b.shape[0])
    out = np.empty(K, dtype=np.float64)
    for k in range(1, K + 1):
        out[k - 1] = float(np.einsum("ij,ij->i", a[:N - k], b[k:]).sum())
    return out


def degen(vcurve, toks):
    """Фильтр вырожденности (ТЗ §0.3: `>= 0.95` OR `>= 0.90` — порог ТЗ, НЕ LADDER_V2;
    LADDER_V2 использовал strict `> 0.95`, на границе ровно 0.95 наборы могут различаться).

    degen := max_{k≥2} median_t cos(V_t, V_{t+k}) ≥ 0.95  OR  max_token_share ≥ 0.90.
    vcurve (Kmax,) — кривая dip_V (stat=median) по k=1..Kmax.
    """
    share = float(np.max(np.unique(toks, return_counts=True)[1])) / len(toks) if len(toks) else 0.0
    m = float(np.max(vcurve[1:])) if len(vcurve) > 1 else 0.0
    return bool(m >= 0.95 or share >= 0.90), m, share


def jclean(o):
    """numpy → python для json.dump (каст float, списки; NaN/inf → None)."""
    if isinstance(o, dict):
        return {str(k): jclean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jclean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jclean(o.tolist())
    return o


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- самопроверка

def _selfcheck():
    rng = np.random.default_rng(0)
    D, N = 64, 384
    # G1 + G3 + G7 на случайных приращениях
    x = rng.standard_normal((N, D))
    a = acf_moments(x, 96)
    assert a["gram_resid"] < 1e-10, "G8"
    af = acf_moments(x, N - 1)   # полный размах: тождество G1 без усечения
    assert abs(1.0 + 2.0 * af["rho"].sum() - af["r"]) < 1e-12, "G1"
    assert abs(a["r"] - N * f_dc(x)) < 1e-12, "G3"
    p = rng.permutation(N)
    bf = acf_moments(x[p], N - 1)   # инвариантность G7 — только по ПОЛНОМУ размаху (все пары)
    assert abs(a["r"] - bf["r"]) < 1e-9 and abs(a["sum_rho_full"] - bf["sum_rho_full"]) < 1e-12, "G7"
    # G6 на случайном пути по сфере
    u = rng.standard_normal((N, D))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    hn = np.cumsum(rng.standard_normal(N) * 0.01) + 10.0
    s = ra_split(u, hn)
    assert s["closure_max"] < 1e-12, f"G6 closure {s['closure_max']}"
    assert s["du_u_cross_max"] < 1e-14, f"G6 cross {s['du_u_cross_max']}"
    # G2: unit-объект — ρ̂(k) = (1−k/N)·mean_t cos
    P = rng.standard_normal((N, D))
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    am = acf_moments(P, 24)
    for k in (1, 5, 24):
        assert abs(am["rho"][k - 1] - (1 - k / N) * float(np.mean(rho_t(P, k)))) < 1e-12, f"G2 k={k}"
    # period-2: S(0) = 0 точно (положительный контроль равенства), чётное N
    x2 = np.zeros((N, D))
    x2[:, 0] = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    a2 = acf_moments(x2, 96)
    assert abs(a2["r"]) < 1e-14, f"period-2 S(0)={a2['r']}"
    # iid: ρ̂(1) мал
    xi = rng.standard_normal((383, 64))
    ai = acf_moments(xi, 96)
    assert abs(ai["rho"][0]) < 0.1, f"iid rho1={ai['rho'][0]}"
    # vpath: длина и unit-нормы
    V, m = vpath(u, hn)
    assert V.shape[0] == N - 1 - m
    assert float(np.max(np.abs(np.linalg.norm(V, axis=1) - 1.0))) < 1e-12
    # rho_t: k=0 — ones, k вне диапазона — fail-loud
    assert np.all(rho_t(P, 0) == 1.0) and len(rho_t(P, 0)) == N
    for bad in (-1, N):
        try:
            rho_t(P, bad)
            raise AssertionError(f"rho_t(k={bad}) не упал")
        except ValueError:
            pass
    # acf_moments: T−k-нормировка; для unit-объекта rho_Tk = (1−k/N)·mean_t cos (G2)
    for k in (1, 5, 24):
        assert abs(am["rho_Tk"][k - 1] - float(np.mean(rho_t(P, k)))) < 1e-12, f"T-k k={k}"
    try:
        acf_moments(x, N)   # K = N — вне диапазона
        raise AssertionError("acf_moments(K=N) не упал")
    except ValueError:
        pass
    # f_dc: нулевой объект — fail-loud
    try:
        f_dc(np.zeros((4, 3)))
        raise AssertionError("f_dc(0) не упал")
    except ValueError:
        pass
    # jclean: bool → true/false (не 0/1), numpy-bool тоже
    import json as _json
    assert _json.loads(_json.dumps(jclean({"a": True, "b": np.bool_(False), "c": 3.0}))) == {"a": True, "b": False, "c": 3.0}
    # degen: граница 0.95 (семантика ТЗ `>=`)
    assert degen(np.array([-0.07, 0.95, 0.5]), [1, 2, 3])[0] is True
    assert degen(np.array([-0.07, 0.9499, 0.5]), [1, 2, 3])[0] is False
    # cross_acf: совпадает с суммой попарных скалярных произведений на k=1
    aa = rng.standard_normal((50, 8))
    bb = rng.standard_normal((50, 8))
    assert abs(cross_acf(aa, bb, 3)[0] - float(np.einsum("ij,ij->i", aa[:-1], bb[1:]).sum())) < 1e-12
    # __all__: все имена существуют (мёртвые имена не проходят)
    import sys as _sys
    mod = _sys.modules[__name__]
    assert all(hasattr(mod, n) for n in mod.__all__), [n for n in mod.__all__ if not hasattr(mod, n)]
    print("field selfcheck: OK (G1 G2 G3 G6 G7 T-k period-2 iid vpath jclean bounds)")


if __name__ == "__main__":
    _selfcheck()
