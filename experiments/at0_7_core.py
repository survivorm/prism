#!/usr/bin/env python3
"""AT0_7 core — токено-повтор эстиманды (A) + h-уровневый (токен,фаза)-lookup (C) + Spearman
+ синтетические строители. ТЗ: docs/TZ_add_0-3.md T0.7 (v3). Импорт: experiments/AT0_7_loop.py.

Формулы (ТЗ T0.7 ESTIMAND):
 (A) PPF(p) = #{t∈[0,T−p−1]: tok[t]==tok[t+p]}/(T−p); rep_frac(p) = PPF(p) (ключевая: доля дока
     в p-repeat); C = Σ_v f(v)² (эмпирич. вероятность совпадения, «шанс», не зависит от p);
     RATIO(p) = PPF/C; maxrun(p) = число последовательных t с tok[t]=tok[t+p] в длиннейшем беге
     (конвенция: чистый p-repeat → T−p, НЕ размах m+p−1). Guard: p ≥ T → None.
 (C) x^sur_{t,d} = mean_{t': tok[t']=tok[t] ∧ t'≡t (mod p)} x_{t',d} — ТОЧНО сохраняет p-бина
     (group-mean, взвешенный count'ом = group-sum) ⇒ S_x^sur(p) = S_x(p) строго ⇒
     R(p) = Σ_d|S_x−S_sur|²/Σ_d|S_x|² ≡ 0 (АЛГЕБРАИЧЕСКОЕ ТОЖДЕСТВО, не измерение; fp64-уровень).
     Rtok(p) — per-token mean БЕЗ фазы: словарезависим, ВЫРОЖДЁН на V≫T (каждый токен ~T/V раз).
"""
import math

import numpy as np

# p ∈ 3..48 — идентично AT0_6 P_GRID (T0.3 K=96 → p ≤ 48; p=2 — T0.2, закрыт)
P_GRID = list(range(3, 49))

# ------------------------------------------------------------------- константы ТЗ T0.7 (v3, пре-рег)
REPMIN = 0.8           # порог стабильного repeat (пре-рег)
H1_FLOOR = 28          # f1 ≥ 28 → PASS (пре-рег)
R0_MIN = 0.888         # r0-якорь: min rep_frac (code_14) — воспроизведение в 1e-12
B = 199                # per-doc перестановки (конвенция AT0_6); p_floor = 1/200
SEED_BASE = 2026       # null: 2026 + doc_idx·1009 + s (doc_idx — индекс в sorted(96))
SEED_STRIDE = 1009
ALPHA = 0.05
D_DIM = 1024
T_PRIME = 383          # gen/own T' = T−1
G7_N = 20
G7_SEED = 777
G7_RATIO_THR = 3.0     # count-гейт: ≤1/20 с max_p RATIO ≥ 3 (r2 MAJOR-A6)
G7_K_MAX = 3           # Q-null: k ≤ 3 (как T0.6 G7)
G7_TOK_T = 383         # длина токенов синтетиков G7/G1d (ТЗ: T=383)
V_SYNTH = 4096         # словарь синтетиков (ЗАДАН, r1)
G1D_SEED = 777         # = G7 k=0 (тот же поток токенов)
G1A_T, G1A_P = 383, 13
G1B_SEED = 77700       # = T0.6 G1b — воспроизводит замер 66.5
G1B_D = 32
POS_SEED = 7770        # 2 синтетика: 7770+k (h: 7770/7771; токены: 7771/7772)
POS_P0 = 13
POS_A = 1.0
POS_D = 32
POS_PPF_THR = 0.2      # POS2: max_p PPF < 0.2
POS_QD = 50.0
H5_TOL = 1e-20         # R(p) ≡ 0 на fp64-уровне (r0: 2.5e-32..3.5e-30)

def unique_root(nm, pairing):
    """Корень уникального текста (минимальное имя в паре — детерминированно).
    pairing: {doc: partner or None} (дубли по токенам)."""
    root = nm
    while pairing.get(root) is not None and pairing[root] < root:
        root = pairing[root]
    return root


_log = []


def logline(s=""):
    print(s, flush=True)
    _log.append(s)


# ------------------------------------------------------------------- (A) токено-повтор

def ppf(toks, p):
    """PPF(p) = #{t∈[0,T−p−1]: tok[t]==tok[t+p]}/(T−p) = rep_frac(p). None: p ≥ T или T = 0."""
    toks = np.asarray(toks)
    T = int(len(toks))
    if T == 0 or p >= T:
        return None
    eq = toks[:T - p] == toks[p:]
    return float(eq.sum()) / (T - p)


def chance(toks):
    """C = Σ_v f(v)² — эмпирич. вероятность совпадения (не зависит от p). None: T = 0."""
    toks = np.asarray(toks)
    T = int(len(toks))
    if T == 0:
        return None
    _, c = np.unique(toks, return_counts=True)
    f = c / T
    return float(np.sum(f * f))


def ratio(toks, p):
    """RATIO(p) = PPF(p)/C. None: p ≥ T или T = 0."""
    r = ppf(toks, p)
    if r is None:
        return None
    return r / chance(toks)


def maxrun(toks, p):
    """Длиннейший бег последовательных t с tok[t]=tok[t+p]. None: p ≥ T или T = 0.
    Конвенция: чистый p-repeat → T−p (НЕ размах m+p−1)."""
    toks = np.asarray(toks)
    T = int(len(toks))
    if T == 0 or p >= T:
        return None
    eq = toks[:T - p] == toks[p:]
    if not eq.any():
        return 0
    d = np.diff(np.concatenate(([0], eq.view(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max())


def pstar(toks):
    """p* = argmax_{p∈3..48} RATIO(p) (descriptive, без Q). (None, None): T < 3."""
    toks = np.asarray(toks)
    T = int(len(toks))
    if T == 0:
        return None, None
    c = chance(toks)
    best, best_r = None, -1.0
    for p in P_GRID:
        if p >= T:
            break
        r = ppf(toks, p)
        q = r / c
        if q > best_r:
            best_r, best = q, p
    return best, (best_r if best is not None else None)


def token_row(toks, p_rep):
    """Per-ток doc-строка: p_rep (p_T0.3, либо p* если p_rep is None), rep_frac, C, RATIO,
    maxrun, p*, max_ratio, V (словарь). Все None-возможны (корпус: без токенов)."""
    toks = np.asarray(toks)
    T = int(len(toks))
    ps, mr = pstar(toks)
    row = {"T": T, "V": int(len(np.unique(toks))) if T else 0,
           "p_star": ps, "max_ratio": mr}
    p = int(p_rep) if p_rep is not None else ps
    if p is None:
        row.update({"p_rep": None, "rep_frac": None, "C": None, "RATIO": None, "maxrun": None})
        return row
    row["p_rep"] = p
    row["rep_frac"] = ppf(toks, p)
    if row["rep_frac"] is None:   # M2 (r1-exec): p ≥ T → guard (latent TypeError None/C)
        row.update({"C": None, "RATIO": None, "maxrun": None})
        return row
    row["C"] = chance(toks)
    row["RATIO"] = row["rep_frac"] / row["C"]
    row["maxrun"] = maxrun(toks, p)
    return row


# ------------------------------------------------------------------- (C) h-lookup

def lookup_resid(x, tk, p):
    """R(p) и Rtok(p) на центрированном x (T', D) с токенами tk (T',), выровненными по t.
    R(p) ≡ 0 — алгебраическое тождество ((токен,фаза)-lookup сохраняет p-бина строго);
    возвращается численно (fp64). Rtok — per-token mean без фазы (вырожден при V≫T).
    (None, None): degenerate den (не ожидается на реальных данных)."""
    T, D = x.shape
    b = np.exp(-2j * math.pi * np.arange(T) / p)
    Sx = x.T @ b                                  # (D,) — S_d(p) = Σ_t x_{t,d} e^{−2πit/p}
    den = float(np.sum(np.abs(Sx) ** 2))
    if den <= 0.0:
        return None, None
    # (токен, фаза)-группы: x^sur = group-mean; sum по группе = group-sum ⇒ p-бина сохранены
    key = tk.astype(np.int64) * int(p) + (np.arange(T) % p)
    R, Rtok = _group_resid(x, b, key, den), None
    key2 = tk.astype(np.int64)                    # per-token, БЕЗ фазы
    Rtok = _group_resid(x, b, key2, den)
    return R, Rtok


def _group_resid(x, b, key, den):
    """Σ_d|S_x − S_groupmean|²/Σ_d|S_x|² для группировки key (создатель lookup/Rtok)."""
    T = x.shape[0]
    order = np.argsort(key, kind="stable")
    ks = key[order]
    groups = np.split(order, np.flatnonzero(ks[1:] != ks[:-1]) + 1)
    out = np.zeros_like(x)
    for g in groups:
        out[g] = x[g].mean(axis=0, keepdims=True)
    Sg = out.T @ b
    return float(np.sum(np.abs(Sx_of(x, b) - Sg) ** 2) / den)


def Sx_of(x, b):
    """S_d(p) = Σ_t x_{t,d} e^{−2πit/p} — общий базисный сумматор (x: (T', D))."""
    return x.T @ b


# ------------------------------------------------------------------- статистика

def spearman(a, b_):
    """Spearman ρ_s (mean-ranks, без scipy). None: n < 3, разный len или вырожденный ранг."""
    a = np.asarray(a, dtype=float)
    b_ = np.asarray(b_, dtype=float)
    n = len(a)
    if n < 3 or n != len(b_) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b_)):
        return None

    def ranks(v):
        o = np.argsort(v, kind="stable")
        r = np.empty(n)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[o[j + 1]] == v[o[i]]:
                j += 1
            r[o[i:j + 1]] = np.mean(np.arange(i, j + 1)) + 1.0
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b_)
    if ra.std() == 0.0 or rb.std() == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ------------------------------------------------------------------- синтетика

def repeat_tokens(vals, T):
    """Чистый p-repeat: tok[t] = vals[t mod p], p = len(vals), длина T (хвост — часть цикла)."""
    v = np.asarray(vals)
    return np.tile(v, (T // len(v) + 1))[:T]


# ------------------------------------------------------------------- self-check

def _selfcheck():
    """Минимальный check core (fail-loud): идентичности (A)/(C) на точных синтетиках."""
    # G1a-ядро: 13 различных, T=383 → PPF(13) = 1.0 ТОЧНО, maxrun = 370
    t = repeat_tokens(np.arange(13), 383)
    assert ppf(t, 13) == 1.0
    assert maxrun(t, 13) == 370
    assert abs(ratio(t, 13) - 1.0 / chance(t)) < 1e-12
    # guard: p ≥ T и пустые токены
    assert ppf(np.arange(10), 13) is None and maxrun(np.arange(10), 13) is None
    assert ppf(np.array([], dtype=np.int64), 3) is None and chance(np.array([])) is None
    # (C): R(p) ≡ 0 на p-repeat токенах с произвольным x (fp64-уровень)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((383, 64))
    x = x - x.mean(axis=0, keepdims=True)
    tk = repeat_tokens(np.arange(13), 383)
    R, Rtok = lookup_resid(x, tk, 13)
    assert R < 1e-20, f"R(p) не на fp64-уровне: {R}"
    # Rtok: iid-токены из словаря V — вырождение при V≫T (ТЗ: V=13→~0.9, V=4096→~0 на iid-x)
    x2 = rng.standard_normal((383, 64))
    x2 = x2 - x2.mean(axis=0, keepdims=True)
    _, rt13 = lookup_resid(x2, rng.integers(0, 13, 383), 13)
    _, rt4096 = lookup_resid(x2, rng.integers(0, 4096, 383), 13)
    assert rt13 > 0.5, f"Rtok V=13 обязан быть ~1 (сглаживание): {rt13}"
    assert rt4096 < 0.5, f"Rtok V=4096 обязан вырождаться (каждый токен ~T/V раз): {rt4096}"
    # Spearman: монотонно-ненелинейная связь → 1.0; константа → None
    assert abs(spearman(np.arange(10.0), np.arange(10.0) ** 3) - 1.0) < 1e-12
    assert spearman(np.ones(5), np.arange(5.0)) is None
    # maxrun: разрыв бег — точная длина (t2 = 11111 222 1111, p=2: бега t0-2, t5, t8-9 → 3)
    t2 = np.array([1] * 5 + [2] * 3 + [1] * 4)
    assert maxrun(t2, 2) == 3


if __name__ == "__main__":
    _selfcheck()
    print("at0_7_core selfcheck OK")
