#!/usr/bin/env python3
"""AT0_6 — T0.6 «Спектральная линия на петлевом периоде» (matched-filter, v3 после внешнего
ревью AT0_6_REVIEW: C1 null-mismatch интегрированного h, C2/C3 H2′+стратификация, C4/C5,
M1–M4).

ТЗ: docs/TZ_add_0-3.md §1.5 T0.6 (v3). Ревью: docs/AT0_6_REVIEW.md (C1–C5, M1–M4) +
docs/dynamic/AT0_6_TZ_REVIEW_r1.md (r1: М1 YW-матрица, М2 AR(2)-null — закрыты v2.2).

Вопрос: есть ли в спектре h|L23 статистически значимая УЗКАЯ период-p компонента на частоте
1/p (p = p_T0.3), и loopy-специфична ли она? (Дискриминация «линия vs AR(2) с тем же ω₀» при
T′=383 невозможна — dwell-time аргумент; этот вопрос — у AT0_5. Объявлено в ТЗ-ОБЛАСТИ.)

КЛЮЧЕВОЕ ИЗМЕНЕНИЕ v3 (C1): h — интегрированный ряд (tilt спектра ~1/ω²); плоский
перестановочный нуль мерит «мощность на частоте» против «нет вообще временной структуры» —
анти-консервативно на автокоррелированных данных (v2.2 full-run: H2 g = 66/66 при ожидаемых
~3). Первичная статистика — УЗОСТЬ R(p) = Q(p)/mean(Q(p−1), Q(p+1)): гладкий tilt ⇒ R ≈ 1
(замер: AR(1) φ=0.95 → R(48) = 1.04 < null p99 1.052 при ложно-«значимом» Q(48)/D = 6.3),
узкая линия ⇒ R ≫ 1 (a/σ=0.5: R ∈ [1.59 (p=32), 24.8 (p=7)]; off-grid 26.7 → R(27) = 2.25).
Q — вторичная (power + effect size a/σ = sqrt(4(Q/D−1)/T′)).

Estimand (клетка × p): matched filter на ТОЧНОМ базисе периода p (базис — не DFT-бин;
T′-простота в утечку не входит (M4); relevant: Dirichlet-side-lobe T′·|1/p − 1/p′|):
S_d(p) = Σ_t x_{t,d}·e^{−2πit/p}, Q(p) = Σ_d |S_d(p)|²/(T′·σ²_d), σ²_d = var_t(x_{t,d}).
R(p) = Q(p)/((Q(p−1)+Q(p+1))/2) (края сетки: R(3)=Q(3)/Q(4), R(48)=Q(48)/Q(47); max-
статистики — только внутренние p ∈ 4..47, объявлено).
x = h_raw|L, T′ = T−1 per-doc (без начального; gen/own 383, corpus 82..254), центрир. по t.
Δh-клетка (G9): x = h[1:]−h[:-1] по полному h, центрир.

Null (exact, parametric-free): per-doc перестановка t, B = 199; seed = 2026 + doc_idx·1009 + s
(doc_idx = индекс в sorted(96) — конвенция T0.1/AT0_2; ТЕ ЖЕ сиды, что v2.2 — перестановки
идентичны версиям). Перестановка сохраняет per-dim multiset (var/mean точно — assert),
разрушает временную структуру (включая tilt) ⇒ нуль для R: «нет узкой линии на ω_p».
p = (#{perm: stat_perm ≥ stat_obs}+1)/(B+1), p_floor = 1/200. Одна и та же формула на обеих
сторонах (Q-сетка перестановок — один batched-gemm; R/max — редукции по тем же массивам).

Замороженный вход: results/AT0/AT0_3_AREA.json (sha256[:16] = 8fe61affb13b443f):
period_structure.per_doc — 30 detected с p (loopy), 66 без p (11 loopy-undetected + 53 clean +
2 remainder {own_w0, own_w4}). T0.6 НЕ переоткрывает периоды: p_T0.3 вербатим.
P_set = distinct p_T0.3 = {7,8,9,11,13,14,15,16,17,18,19,20,24,27,31,32} (16 значений).

Тесты (пре-рег, ТЗ v3):
H1 (primary): h_raw|L23, 30 detected, per-doc R(p_T0.3) на 3-p null (p−1,p,p+1).
  f = #{p_R ≤ 0.05}/30; f ≥ 15 → CONFIRMED (P(Bin(30,.05)≥15) = 2.3e-12); f ≥ 10 → значимо
  (P = 1.16e-6), INCONCLUSIVE; f < 10 → нет. C4: НЕ независимое переоткрытие (p выбран на той
  же траектории через du = Δ(h/‖h‖)).
H2′ (специфичность, primary; real-background калибровка C2/C5): h_raw|L23, 53 clean,
  max_{p ∈ P_set(doc)} R(p), P_set(doc) = P_set ∩ {4..⌊T′/4⌋−1}; null = max по тем же p в тех
  же перестановках. g2 = #{p ≤ 0.05}/53; g2 ≥ 7 → CONFIRMED (P(Bin(53,.05)≥7) = 0.0160 —
  консервативнее 5%, объявлено).
H2 (descriptive, НЕ в вердикт; стратификация C3): h_raw|L23, 66 без p, max Q по per-doc сетке
  p ≤ min(48, ⌊T′/4⌋) (M1), null = max по той же сетке. g per stratum (clean/loopy-und/remainder).
G9 (гейт; C1-fix 1): Δh|L23, 30, тот же R(p_T0.3)-тест. f_dh = #{p_R ≤ 0.05}/30.
  H1c ∧ f_dh ≤ 3 → TILT_ARTIFACT (H1 не интерпретируется); 4 ≤ f_dh ≤ 14 → флаг dh_weak;
  f_dh ≥ 15 → линейность подтверждена в дифференциальном объекте (du-носитель T0.3).
H3 (robustness, descriptive): Q-тест (1-p null) на h_raw|L{5,11,17} (30×3) — f per layer.
DESCRIPTIVE: (a) Q-сетка (spectral_map) + R-сетка (r_map) по 96 докам; (b) background/null
  ratio per-doc Q_obs(p)/median_perm(Q(p)); (c) AR(2)-фит (описательно) с нормировкой
  r_k = c_k/(N−k) (M3), матрица [r0 r1; r1 r0] rhs [r1 r2] (r1-M1); (d) эмпирический SD
  перестановочного нуля Q(p_T0.3) per H1-док (M2); (e) per-doc p_at_max/R_max.

Вердикт (primary family = {H1, H2′}; FWER не применяется — объявлено):
 Предварительно G9 (H1c ∧ f_dh ≤ 3 → TILT_ARTIFACT).
 H1 ∧ ¬H2′ → CONFIRMED; H1 ∧ H2′ → NON-SPECIFIC (T0.3-detected = порогой срез ρ≥0.9 более
   широкой популяции линий); ¬H1 ∧ H2′ → STRANGE; ¬H1 ∧ ¬H2′ → REFUTED; 10 ≤ f < 15 →
   INCONCLUSIVE. H1-пул = 30 degen-доков — вывод на degen/loopy популяцию.

Контроли (гейты, ДО интерпретации):
 G0: sha256 gen/own/cor npz + AT0_3-артефакт + gen_meta sha_ok 64/64 + декод-пробы (corpus
   not_testable) — как AT0_2.
 G1a (identity): Q(p) ТРЁМЯ независимыми формулами — production gemm vs einsum vs cos/sin —
   max rel dev < 1e-12 на 2 доках (sorted[0], sorted[n//2]), все 46 p (R — отношение
   валидированных Q).
 G1b (unit): cos a=1, σ=1, p₀=13, D=32, T′=383: Q(p₀)/D ≥ 50 ∧ off/on ≤ 0.2 ∧ R(p₀) ≥ 5;
   AR(2) φ=(0.5,−0.3), N=20000: YW |φ̂−φ|∞ < 0.05 с нормировкой (N−k) (M3).
 G2: 96 доков; T′ = T−1 (per-doc); per-dim var > 0; per-doc p-сетка p ≤ min(48, ⌊T′/4⌋) (M1:
   cor_w8 → p ≤ 20, cor_w11 → p ≤ 23); L1 (absent own_w0/own_w4) не используется.
 G7 (калибровка реализации): 20 iid (D=1024, T′=383, сид 777+k): (a) max-Q: k ≤ 3
   (FPR ≤ 0.15 — слово исправлено C5; P(k≥4|Bin(20,.05)) = 0.016); (b) G7b: max-R по
   внутренним p ∈ 4..47: k_r ≤ 3.
 POSITIVE: 4 синтетика a=0.5σ, p₀ ∈ {7,13,17,27}, независимые фазы per dim, D=1024, T′=383,
   сид 7770+k; R(p₀)-тест (3-p null): 4/4 p ≤ 0.05 (замеры R: 24.8/20.3/~11/2.57 ≫ null p95).
 G8: NaN/Inf — jclean во всех артефактах; field.py sha-пин + _selfcheck().
Детерминизм: все rng засеяны (2026+doc_idx·1009+s; G7 777+k; POS 7770+k; G1b 77700/77701);
нет global np.random; sorted(docs); store sort_keys; json — jclean.
Цена: 150 doc-тестов (96 grid|L23 + 30 Δh + 20 G7 + 4 POS + 90 H3 single-p) ≈ 17–22 мин CPU,
память пик < 1 GB per-doc. НЕ долгий — local.
Границы: read-only данные + field.py (пин) + AT0_1_accounting + AT0_3-артефакт; без форвардов;
writes только results/AT0/AT0_6_* (+ archive/).
"""
import hashlib
import json
import math
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402

_fspec = importlib.util.spec_from_file_location(
    "prism_field_pinned", os.path.join(ROOT, "prism/field.py"))
pf = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(pf)
sys.modules["prism_field_pinned"] = pf
jclean, sha256_file = pf.jclean, pf.sha256_file

import AT0_1_accounting as a1  # noqa: E402  (load_source, decode_probe, LAYERS, пути)

LAYERS = a1.LAYERS
GEN, OWN, COR = a1.GEN, a1.OWN, a1.COR
AT03_PATH = os.path.join(ROOT, "results/AT0/AT0_3_AREA.json")
AT03_SHA16 = "8fe61affb13b443f"   # замероз. вход (ТЗ T0.6)

B = 199                            # per-doc перестановки; p_floor = 1/(B+1) = 1/200
SEED_BASE = 2026
SEED_STRIDE = 1009
ALPHA = 0.05
P_GRID = list(range(3, 49))        # p ∈ 3..48 (T0.3 K=96 → p ≤ 48; p=2 — T0.2, закрыт)
T_PRIME = 383
H1_FLOOR = 15                      # f ≥ 15/30 → H1 CONFIRMED (P = 2.3e-12)
H1_SIG = 10                        # 10 ≤ f ≤ 14 → значимо (P = 1.16e-6), INCONCLUSIVE
H2P_FLOOR = 7                      # g2 ≥ 7/53 → H2′ CONFIRMED (P(Bin(53,.05)≥7) = 0.0160)
G9_TILT_MAX = 3                    # H1c ∧ f_dh ≤ 3 → TILT_ARTIFACT
G9_WEAK_MAX = 14                   # 4 ≤ f_dh ≤ 14 → флаг dh_weak
G7_K_MAX = 3                       # 20 iid: k ≤ 3 (FPR ≤ 0.15; P(k≥4|Bin(20,.05)) = 0.016)
POS_A = 0.5                        # a = 0.5σ (измеренный эффект 0.55, v2.2 full: Q/D 22..75)
POS_P0 = [7, 13, 17, 27]
POS_SEED = 7770
G7_SEED = 777
G7_N = 20
G1A_TOL = 1e-12
G1B_COS_QD = 50.0                  # Q(p₀)/D ≥ 50
G1B_COS_OFF = 0.2                  # Q(p₀±1)/Q(p₀) ≤ 0.2
G1B_COS_R = 5.0                    # R(p₀) ≥ 5 (v3: явный narrowness-чек)
G1B_YW_TOL = 0.05
D_DIM = 1024
OUT_DIR = os.path.join(ROOT, "results", "AT0")
STORE_DIR = os.path.join(OUT_DIR, "store")
SMOKE = "--smoke" in sys.argv

_log = []
_E_CACHE = {}


def logline(s=""):
    print(s, flush=True)
    _log.append(s)


def binom_sf(k, n, p):
    """P(X ≥ k), X ~ Bin(n, p) — точный (math.comb)."""
    return math.fsum(math.comb(n, j) * p ** j * (1.0 - p) ** (n - j) for j in range(k, n + 1))


def pmax_grid(T):
    """M1: per-doc ограничение сетки: p ≤ min(48, floor(T′/4)). T′=383 → 48;
    cor_w8 (82) → 20; cor_w11 (94) → 23; T′=254 → 48."""
    return min(P_GRID[-1], T // 4)


def pmax_idx(T):
    """Индекс срезa P_GRID для per-doc сетки (p ≤ pmax_grid(T))."""
    pm = pmax_grid(T)
    return sum(1 for p in P_GRID if p <= pm)


# ------------------------------------------------------------------- замероз. вход

def load_frozen_t03():
    """AT0_3: period_structure.per_doc → p_map (30 detected), h2_pool (66 без p),
    strata (clean/loopy/remainder per doc)."""
    d = json.load(open(AT03_PATH))
    per_doc = d["results"]["period_structure"]["per_doc"]
    p_map, h2_pool, strata = {}, [], {}
    for st, dd in per_doc.items():
        for doc, info in dd.items():
            strata[doc] = st
            p = info.get("p")
            if p is not None:
                p_map[doc] = int(p)
            else:
                h2_pool.append(doc)
    return p_map, h2_pool, strata


# ------------------------------------------------------------------- данные

def load_all_docs():
    docs = {}
    for path, kind in ((GEN, "gen"), (OWN, "state"), (COR, "state")):
        docs.update(a1.load_source(path, kind))
    return docs


def _h_full(d, L):
    """Полный h|L: (T, D) float64 (НЕ центрированный). gen: h = u·hn; own/cor: u·hn ≡ raw."""
    u, hn = d["layers"][L]
    U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    return (U * hn[:, None]).astype(np.float64)


def h_cell(d, L):
    """h_raw|L: (T′, D) float64, центрированный по t per dim. T′ = T−1 (без начального)."""
    x = _h_full(d, L)[1:]
    return x - x.mean(axis=0, keepdims=True)


def dh_cell(d, L):
    """Δh|L: (T′, D) float64: h[1:]−h[:-1] по полному h, центрированный (G9-клетка)."""
    h = _h_full(d, L)
    x = h[1:] - h[:-1]
    return x - x.mean(axis=0, keepdims=True)


# ------------------------------------------------------------------- ядро: Q(p), R(p), null

def basis_grid(T, n_p=None):
    """E: (n_p, T) complex — e^{+2πit/p} на сетке P_GRID[:n_p] для длины T (per-doc T′)."""
    n_p = n_p if n_p is not None else len(P_GRID)
    key = (T, n_p)
    if key not in _E_CACHE:
        t = np.arange(T)
        E = np.empty((n_p, T), dtype=np.complex128)
        for i, p in enumerate(P_GRID[:n_p]):
            E[i] = np.exp(2j * math.pi * t / p)
        _E_CACHE[key] = E
    return _E_CACHE[key]


def basis_set(ps, T):
    """E: (n, T) complex для произвольного набора p (3-p тест: p−1, p, p+1)."""
    t = np.arange(T)
    E = np.empty((len(ps), T), dtype=np.complex128)
    for i, p in enumerate(ps):
        E[i] = np.exp(2j * math.pi * t / p)
    return E


def q_grid(x, n_p=None):
    """Q(p) = Σ_d |S_d(p)|²/(T′·σ²_d) по P_GRID[:n_p]: (n_p,). x: (T′, D) центрир.
    Production path: один gemm (D,T)@(T,n_p)."""
    T, D = x.shape
    n_p = n_p if n_p is not None else len(P_GRID)
    E = basis_grid(T, n_p)
    var_d = np.var(x, axis=0)
    S = x.T @ E.T.conj()                    # (D, n_p)
    return np.sum(np.abs(S) ** 2 / (T * var_d[:, None]), axis=0)


def q_set(x, ps):
    """Q для набора p: (n,). Один gemm (D,T)@(T,n)."""
    T, D = x.shape
    E = basis_set(ps, T)
    var_d = np.var(x, axis=0)
    S = x.T @ E.T.conj()
    return np.sum(np.abs(S) ** 2 / (T * var_d[:, None]), axis=0)


def q_single(x, p):
    T, D = x.shape
    b = np.exp(2j * math.pi * np.arange(T) / p)
    var_d = np.var(x, axis=0)
    S = x.T @ b.conj()[:, None]
    return float(np.sum(np.abs(S) ** 2 / (T * var_d[:, None])))


def r_from_q(q, pset):
    """R(p) = Q(p)/mean(Q(p−1), Q(p+1)) по набору pset (сортир. p, соседями по pset).
    Край: односторонне (R(first) = Q/Q_next, R(last) = Q/Q_prev). q: (n,), pset: (n,)."""
    q = np.asarray(q, dtype=np.float64)
    n = len(pset)
    r = np.empty(n)
    for i in range(n):
        if n == 1:
            r[i] = np.nan          # один p — R не определён (не используется: H1 всегда 3-p)
        elif i == 0:
            r[i] = q[0] / q[1]
        elif i == n - 1:
            r[i] = q[-1] / q[-2]
        else:
            r[i] = q[i] / ((q[i - 1] + q[i + 1]) / 2.0)
    return r


def _perm_batch(x, Bn, doc_idx):
    """Все Bn перестановок одним gather: (B, T, D) + assert точности per-dim var
    (инвариант exact-null: перестановка сохраняет per-dim multiset)."""
    T, D = x.shape
    perm_idx = np.empty((Bn, T), dtype=np.intp)
    for s in range(Bn):
        rng = np.random.default_rng(SEED_BASE + doc_idx * SEED_STRIDE + s)
        perm_idx[s] = rng.permutation(T)
    X = x[perm_idx]                          # (B, T, D) — один gather
    var_d = np.var(x, axis=0)
    assert np.allclose(np.var(X[0], axis=0), var_d, rtol=1e-10, atol=0), \
        "exact-null инвариант: per-dim var перестановки != real"
    return X


def perm_q_grid(x, doc_idx, Bn=B, n_p=None):
    """Q-сетка перестановок: (B, n_p). Один batched-gemm (B,D,T)@(T,n_p)."""
    T, D = x.shape
    n_p = n_p if n_p is not None else len(P_GRID)
    E = basis_grid(T, n_p)
    var_d = np.var(x, axis=0)
    X = _perm_batch(x, Bn, doc_idx)          # (B, T, D)
    S = X.transpose(0, 2, 1) @ E.T.conj()[None]    # (B, D, n_p)
    return np.sum(np.abs(S) ** 2 / (T * var_d[None, :, None]), axis=1)


def perm_q_set(x, ps, doc_idx, Bn=B):
    """Q перестановок для набора p: (B, n). 3-p null (H1/G9/POS)."""
    T, D = x.shape
    E = basis_set(ps, T)
    var_d = np.var(x, axis=0)
    X = _perm_batch(x, Bn, doc_idx)          # (B, T, D)
    S = X.transpose(0, 2, 1) @ E.T.conj()[None]    # (B, D, n)
    return np.sum(np.abs(S) ** 2 / (T * var_d[None, :, None]), axis=1)


def null_pvalue(stat_obs, stat_perm):
    """p = (#{perm: stat ≥ obs}+1)/(B+1); p_floor = 1/(B+1). Одна формула на обеих сторонах."""
    n_ge = int(np.sum(np.asarray(stat_perm) >= stat_obs))
    Bn = len(stat_perm)
    return (n_ge + 1) / (Bn + 1)


# ------------------------------------------------------------------- гейты

def g0_data(docs):
    """G0: sha npz + AT0_3 + gen_meta sha_ok + декод-пробы (как AT0_2)."""
    out = {}
    out["data_sha256"] = {"gen": sha256_file(GEN), "own": sha256_file(OWN), "cor": sha256_file(COR)}
    at03_sha = sha256_file(AT03_PATH)
    out["at03_sha256"] = at03_sha
    ok_sha = at03_sha[:16] == AT03_SHA16
    gen_meta = json.load(open(GEN.replace(".npz", "_meta.json")))
    gen_sha_ok = sum(1 for n in gen_meta if gen_meta[n].get("sha_ok"))
    dec = a1.decode_probe(docs[sorted(docs)[0]]["toks"], "gen")
    own_docs = [nm for nm in docs if nm.startswith("own_")]
    own_dec = (a1.decode_probe(docs[sorted(own_docs)[0]]["toks"], "own")
               if any(len(docs[nm]["toks"]) for nm in own_docs) else {"n_toks": 0})
    cor_toks = sum(len(docs[nm]["toks"]) for nm in docs if nm.startswith("cor_"))
    out["gen_sha_ok"] = f"{gen_sha_ok}/64"
    out["decode_probe"] = {"gen": dec, "own": own_dec, "corpus_toks": int(cor_toks),
                           "corpus_note": "not_testable G16"}
    out["ok"] = bool(ok_sha and gen_sha_ok == 64)
    logline(f"G0: gen {out['data_sha256']['gen'][:16]} own {out['data_sha256']['own'][:16]} "
            f"cor {out['data_sha256']['cor'][:16]} at03 {at03_sha[:16]} "
            f"(sha16-match={ok_sha}); gen_sha_ok {gen_sha_ok}/64; corpus toks {cor_toks}/16 (G16)")
    return out


def g1a_identity(docs):
    """G1a: Q(p) — ТРИ независимые формулы: (1) production gemm q_grid (x.T @ E.T.conj),
    (2) прямой комплексный einsum, (3) (Σxcos)²+(Σxsin)². max rel dev < 1e-12. per-doc T′,
    пер-doc p-сетка (M1). Geyt валидирует ПУТ, используемый в тестах (gemm).
    R — отношение валидированных Q (отдельный identity не нужен — ТЗ v3)."""
    names = sorted(docs)
    picked = [names[0], names[len(names) // 2]]
    per_doc = {}
    worst = 0.0
    for nm in picked:
        x = h_cell(docs[nm], 23)
        T = x.shape[0]
        n_p = pmax_idx(T)
        E = basis_grid(T, n_p)
        var_d = np.var(x, axis=0)
        q_gemm = q_grid(x, n_p)              # production path
        max_dev = 0.0
        for i, p in enumerate(P_GRID[:n_p]):
            S = np.einsum("t,td->d", E[i], x)
            q1 = np.abs(S) ** 2 / (T * var_d)     # прямой комплексный
            c = np.einsum("t,td->d", E[i].real, x)
            sn = np.einsum("t,td->d", E[i].imag, x)
            q2 = (c ** 2 + sn ** 2) / (T * var_d)  # cos/sin
            dev_einsum = float(np.max(np.abs(q1 - q2) / np.maximum(q1, 1e-30)))
            dev_gemm = float(np.abs(q_gemm[i] - float(np.sum(q1))) / max(float(np.sum(q1)), 1e-30))
            max_dev = max(max_dev, dev_einsum, dev_gemm)
        per_doc[nm] = max_dev
        worst = max(worst, max_dev)
    logline(f"G1a: identity max rel dev = {worst:.3e} (tol {G1A_TOL}) на {picked} (gemm vs einsum vs cos/sin)")
    return {"ok": bool(worst < G1A_TOL), "max_rel_dev": float(worst), "per_doc": per_doc}


def g1b_unit():
    """G1b: (i) cos-синтетика: узкий пик + R-чек (v3); (ii) AR(2)-фит с (N−k)-нормировкой (M3)."""
    # (i) чистый cos a=1, σ=1, p0=13, D=32
    rng = np.random.default_rng(77700)
    p0 = 13
    t = np.arange(T_PRIME)
    ph = rng.uniform(0, 2 * math.pi, size=32)
    x = np.zeros((T_PRIME, 32))
    for d in range(32):
        x[:, d] = np.cos(2 * math.pi * t / p0 + ph[d]) + rng.normal(0, 1.0, T_PRIME)
    x = x - x.mean(axis=0, keepdims=True)
    q0 = q_single(x, p0)
    off_m = q_single(x, p0 - 1)
    off_p = q_single(x, p0 + 1)
    r0 = q0 / ((off_m + off_p) / 2.0)
    ok_cos = (q0 / 32 >= G1B_COS_QD and off_m / q0 <= G1B_COS_OFF
              and off_p / q0 <= G1B_COS_OFF and r0 >= G1B_COS_R)
    # (ii) AR(2)-синтетика φ=(0.5, −0.3), N=20000: YW с нормировкой r_k = c_k/(N−k) (M3)
    phi_true = np.array([0.5, -0.3])
    rng2 = np.random.default_rng(77701)
    eps = rng2.standard_normal(20000)
    z = np.zeros(20000)
    for tt in range(2, 20000):
        z[tt] = 0.5 * z[tt - 1] - 0.3 * z[tt - 2] + eps[tt]
    z = z - z.mean()
    N = len(z)
    ac0 = float(np.sum(z * z)) / N
    ac1 = float(np.sum(z[:-1] * z[1:])) / (N - 1)
    ac2 = float(np.sum(z[:-2] * z[2:])) / (N - 2)
    M = np.array([[ac0, ac1], [ac1, ac0]])
    phi_hat = np.linalg.solve(M, np.array([ac1, ac2]))
    yw_err = float(np.max(np.abs(phi_hat - phi_true)))
    ok_yw = yw_err < G1B_YW_TOL
    logline(f"G1b: cos a=1 σ=1 p0=13 D=32: Q(p0)/D = {q0 / 32:.1f} (≥{G1B_COS_QD}), "
            f"off/on = {off_m / q0:.3f}/{off_p / q0:.3f} (≤{G1B_COS_OFF}), R = {r0:.1f} (≥{G1B_COS_R}); "
            f"AR2-fit φ̂ = ({phi_hat[0]:.4f}, {phi_hat[1]:.4f}) vs (0.5, −0.3), |Δ|∞ = {yw_err:.4f} "
            f"(tol {G1B_YW_TOL})")
    return {"ok": bool(ok_cos and ok_yw),
            "cos": {"q0_over_D": q0 / 32, "off_over_on": [off_m / q0, off_p / q0], "r": float(r0)},
            "ar2_fit": {"phi_true": [0.5, -0.3], "phi_hat": [float(phi_hat[0], ), float(phi_hat[1])],
                        "abs_err_inf": yw_err, "norm": "r_k = c_k/(N-k) (M3)"}}


def g2_cells(docs):
    """G2: 96 доков; T′ = T−1 (per-doc: gen/own 383, corpus 82..254); per-dim var > 0;
    per-doc p-сетка p ≤ min(48, ⌊T′/4⌋) (M1). L1 не используется (absent own_w0/own_w4)."""
    n = len(docs)
    bad = []
    tpmin, tpmax = 10 ** 9, 0
    pmaxs = {}
    for nm in sorted(docs):
        u, hn = docs[nm]["layers"][23]
        T = u.shape[0]
        x = h_cell(docs[nm], 23)
        if x.shape[0] != T - 1:
            bad.append((nm, "T'"))
        if not np.all(np.var(x, axis=0) > 0):
            bad.append((nm, "var"))
        tpmin, tpmax = min(tpmin, x.shape[0]), max(tpmax, x.shape[0])
        pmaxs[nm] = pmax_grid(x.shape[0])
    logline(f"G2: {n} доков, T'∈[{tpmin},{tpmax}] (per-doc T−1), per-dim var>0; "
            f"per-doc p-сетка p ≤ min(48, ⌊T′/4⌋) (M1): cor_w8→{pmaxs.get('cor_w8')}, "
            f"cor_w11→{pmaxs.get('cor_w11')}, gen→{pmaxs.get('chat_0', 48)}; "
            f"{'OK' if not bad else bad[:6]}")
    return {"ok": bool(n == 96 and not bad), "n_docs": int(n),
            "t_prime_range": [int(tpmin), int(tpmax)],
            "pmax_rule": "p ≤ min(48, floor(T'/4)) (M1)",
            "pmax_examples": {"cor_w8": pmaxs.get("cor_w8"), "cor_w11": pmaxs.get("cor_w11"),
                              "gen_own": pmaxs.get("chat_0")},
            "bad": bad}


def run_g7_calibration(Bn=B, N=None):
    """G7: 20 iid-синтетик (D=1024, T′=383, сид 777+k, per-dim N(0,1)): полный пайплайн.
    (a) max-Q по сетке p∈3..48: k = #{p ≤ 0.05}/20, гейт k ≤ 3 (FPR ≤ 0.15 — C5-слово;
    ложный фейл при истинном FPR 0.05: P(k≥4|Bin(20,.05)) = 0.016).
    (b) G7b (v3): max-R по Внутренним p ∈ 4..47: k_r ≤ 3 (R-пайплайн; iid обменен ⇒ exact)."""
    N = N or G7_N
    interior = [p for p in P_GRID if 4 <= p <= 47]
    hits_q = 0
    hits_r = 0
    rows = []
    for k in range(N):
        x = np.random.default_rng(G7_SEED + k).standard_normal((T_PRIME, D_DIM))
        x = x - x.mean(axis=0, keepdims=True)
        q_obs = q_grid(x)
        Qp = perm_q_grid(x, 9000 + k, Bn=Bn)            # (B, 46)
        # (a) max-Q
        max_obs = float(q_obs.max())
        p_q = null_pvalue(max_obs, Qp.max(axis=1))
        hit_q = bool(p_q <= ALPHA)
        hits_q += hit_q
        # (b) max-R по внутренним p (индексы interior в P_GRID: 1..44)
        idx = [P_GRID.index(p) for p in interior]
        r_obs_arr = r_from_q(q_obs, P_GRID)
        r_obs = float(np.max(r_obs_arr[idx]))
        Rperm = np.array([r_from_q(Qp[b], P_GRID)[idx] for b in range(Bn)])
        r_perm_max = Rperm.max(axis=1)   # max по p на КАЖДУЮ перестановку (r4 MAJOR-1: axis=0 был транспонирован)
        p_r = null_pvalue(r_obs, r_perm_max)
        hit_r = bool(p_r <= ALPHA)
        hits_r += hit_r
        rows.append({"k": k, "q_max_obs": max_obs, "p_q": float(p_q), "hit_q": hit_q,
                     "r_max_obs": float(r_obs), "p_r": float(p_r), "hit_r": hit_r})
        logline(f"G7 iid k={k}: Q_max = {max_obs:.1f} (p={p_q:.3f} {'HIT' if hit_q else '—'}), "
                f"R_max(int) = {r_obs:.3f} (p={p_r:.3f} {'HIT' if hit_r else '—'})")
    ok = hits_q <= G7_K_MAX and hits_r <= G7_K_MAX
    logline(f"G7: k(Q) = {hits_q}/{N}, k(R) = {hits_r}/{N} (гейт ≤ {G7_K_MAX} each; "
            f"P(k≥4|Bin(20,.05)) = {binom_sf(G7_K_MAX + 1, N, ALPHA):.4f})")
    return {"gate": f"k ≤ {G7_K_MAX} из {G7_N} (Q и R)", "ok": bool(ok),
            "k_q": int(hits_q), "k_r": int(hits_r), "rows": rows,
            "smoke_note": (f"smoke: B={Bn}, N={N}" if (Bn != B or N != G7_N) else None)}


def synth_line(p0, a, seed, T=T_PRIME, D=D_DIM):
    """x_t = a·cos(2πt/p₀ + φ_d) + ε_t; φ_d — независимые per dim; ε ~ N(0,1)."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    ph = rng.uniform(0, 2 * math.pi, size=D)
    x = np.zeros((T, D))
    for d in range(D):
        x[:, d] = a * np.cos(2 * math.pi * t / p0 + ph[d]) + rng.normal(0, 1.0, T)
    return x - x.mean(axis=0, keepdims=True)


def run_positive(p0s=None, Bn=B):
    """POSITIVE (v3): 4 синтетика a=0.5σ, p₀ ∈ {7,13,17,27}; тест = R(p₀) на 3-p null
    (p₀−1, p₀, p₀+1): гейт 4/4 p ≤ 0.05 (замеры R: 24.8/20.3/~11/2.57 ≫ null p95 ≈ 1.04)."""
    rows = []
    p0s = p0s or POS_P0
    for k, p0 in enumerate(p0s):
        x = synth_line(p0, POS_A, POS_SEED + k)
        ps = [p0 - 1, p0, p0 + 1]
        q_obs = q_set(x, ps)
        Qp = perm_q_set(x, ps, 900 + k, Bn=Bn)          # (B, 3)
        r_obs = float(r_from_q(q_obs, ps)[1])           # средний индекс = p0
        Rperm = np.array([r_from_q(Qp[b], ps)[1] for b in range(Bn)])
        pval = null_pvalue(r_obs, Rperm)
        rows.append({"p0": p0, "q_obs_p0": float(q_obs[1]), "r_obs": float(r_obs),
                     "p": float(pval), "ok": bool(pval <= ALPHA)})
        logline(f"POSITIVE p0={p0}: Q(p0) = {q_obs[1]:.1f}, R(p0) = {r_obs:.2f}, "
                f"p = {pval:.4f} ({'OK' if pval <= ALPHA else 'FAIL'})")
    ok = sum(1 for r in rows if r["ok"])
    return {"gate": f"{ok}/{len(p0s)} ≤ {ALPHA}", "ok": bool(ok == len(p0s)), "rows": rows}


# ------------------------------------------------------------------- тесты

def run_h1(docs, p_map, doc_names, Bn=B):
    """H1 (primary): 30 detected, h_raw|L23, R(p_T0.3) на 3-p null (p−1,p,p+1 ⊂ 3..48
    для всех p_T0.3 ∈ 7..32). C4: локализация на уже найденном T0.3 периоде (не
    независимое переоткрытие). Вторично: Q(p_T0.3) + a/σ + эмпирич. SD перестановок (M2)."""
    rows = []
    names = sorted(p_map)
    for nm in names:
        di = doc_names[nm]
        x = h_cell(docs[nm], 23)
        p = p_map[nm]
        ps = [p - 1, p, p + 1]
        q_obs = q_set(x, ps)
        Qp = perm_q_set(x, ps, di, Bn=Bn)               # (B, 3)
        r_obs = float(r_from_q(q_obs, ps)[1])
        Rperm = np.array([r_from_q(Qp[b], ps)[1] for b in range(Bn)])
        pval = null_pvalue(r_obs, Rperm)
        T = x.shape[0]
        a_sigma = math.sqrt(max(4.0 * (float(q_obs[1]) / D_DIM - 1.0), 0.0) / T)
        rows.append({"doc": nm, "p": p, "r_obs": r_obs, "q_obs_p": float(q_obs[1]),
                     "a_sigma": float(a_sigma), "p": float(pval),
                     "perm_q_sd": float(np.std(Qp[:, 1]))})   # M2: эмпирич. SD нуля
        logline(f"H1 {nm} p={p}: R = {r_obs:.2f}, Q = {q_obs[1]:.1f} (a/σ = {a_sigma:.3f}), "
                f"p = {pval:.4f}")
    f = sum(1 for r in rows if r["p"] <= ALPHA)
    return {"rows": rows, "n": len(rows), "f": int(f),
            "note": "C4: p_T0.3 выбран на той же траектории через du — НЕ независимое переоткрытие"}


def run_g9(docs, p_map, doc_names, Bn=B):
    """G9 (гейт, C1-fix 1): Δh|L23, те же 30, тот же R(p_T0.3) 3-p null. f_dh."""
    rows = []
    for nm in sorted(p_map):
        di = doc_names[nm]
        x = dh_cell(docs[nm], 23)
        p = p_map[nm]
        ps = [p - 1, p, p + 1]
        q_obs = q_set(x, ps)
        Qp = perm_q_set(x, ps, di, Bn=Bn)
        r_obs = float(r_from_q(q_obs, ps)[1])
        Rperm = np.array([r_from_q(Qp[b], ps)[1] for b in range(Bn)])
        pval = null_pvalue(r_obs, Rperm)
        rows.append({"doc": nm, "p": p, "r_obs": float(r_obs), "p": float(pval)})
        logline(f"G9 dh_L23 {nm} p={p}: R = {r_obs:.2f}, p = {pval:.4f}")
    f = sum(1 for r in rows if r["p"] <= ALPHA)
    return {"rows": rows, "n": len(rows), "f_dh": int(f)}


def run_h2p(docs, h2p_pool, doc_names, pset, grid, Bn=B):
    """H2′ (специфичность, primary; real-background калибровка C2/C5): 53 clean,
    max_{p ∈ P_set(doc)} R(p), P_set(doc) = P_set ∩ {4..⌊T′/4⌋−1} (только внутренние p,
    M1); null = max по тем же p в тех же B-перестановках. grid: per-doc (q_obs, medians)."""
    rows = []
    for nm in sorted(h2p_pool):
        di = doc_names[nm]
        T = grid["T"][nm]
        n_p = pmax_idx(T)
        pmax = pmax_grid(T)
        ps = [p for p in pset if 4 <= p <= pmax - 1]     # P_set ∩ {4..⌊T′/4⌋−1}
        assert len(ps) >= 5, f"H2′ {nm}: слишком мало matched-p (T′={T})"
        idx = [P_GRID.index(p) for p in ps]
        q_obs = grid["q_obs"][nm][:n_p]
        Qp = grid["Qp"][nm]                              # (B, n_p)
        r_obs_arr = r_from_q(q_obs, P_GRID[:n_p])
        r_obs = float(np.max(r_obs_arr[idx]))
        p_at = ps[int(np.argmax(r_obs_arr[idx]))]
        Rperm_max = np.array([np.max(r_from_q(Qp[b], P_GRID[:n_p])[idx]) for b in range(Bn)])
        pval = null_pvalue(r_obs, Rperm_max)
        rows.append({"doc": nm, "t_prime": T, "n_pset": len(ps), "r_max_obs": r_obs,
                     "p_at_max": int(p_at), "p": float(pval)})
        logline(f"H2p {nm}: R_max(P_set∩{4}..{pmax - 1}) = {r_obs:.2f} при p={p_at}, "
                f"p = {pval:.4f}")
    g2 = sum(1 for r in rows if r["p"] <= ALPHA)
    return {"rows": rows, "n": len(rows), "g2": int(g2),
            "pset": sorted(pset),
            "note": "matched-p на detected-наборе (C2: одна статистика, один нуль; C5: real-"
                    "background калибровка — автокоррелированные РЕАЛЬНЫЕ clean-доки)"}


def run_h2_desc(docs, h2_pool, strata, doc_names, grid, Bn=B):
    """H2 (descriptive, НЕ в вердикт; стратификация C3): 66 без T0.3-p, max Q по per-doc
    сетке p ≤ min(48, ⌊T′/4⌋) (M1), null = max по той же per-doc сетке. g per stratum.
    Интерпретация: где линии есть без T0.3-detected (в т.ч. пороговые промахи: math_14 ρ=0.895)."""
    rows = []
    for nm in sorted(h2_pool):
        di = doc_names[nm]
        n_p = pmax_idx(grid["T"][nm])
        q_obs = grid["q_obs"][nm][:n_p]
        Qp = grid["Qp"][nm]
        max_obs = float(q_obs.max())
        p_at = int(P_GRID[int(np.argmax(q_obs))])
        n_ge = int(np.sum(np.max(Qp, axis=1) >= max_obs))
        pval = (n_ge + 1) / (Bn + 1)
        rows.append({"doc": nm, "stratum": strata[nm], "q_max_obs": max_obs,
                     "p_at_max": p_at, "p": float(pval)})
        logline(f"H2 {nm} ({strata[nm]}): Q_max = {max_obs:.1f} при p={p_at}, p = {pval:.4f}")
    by_st = {}
    for r in rows:
        by_st.setdefault(r["stratum"], []).append(r)
    g_by = {st: sum(1 for r in rs if r["p"] <= ALPHA) for st, rs in by_st.items()}
    g_tot = sum(1 for r in rows if r["p"] <= ALPHA)
    return {"rows": rows, "n": len(rows), "g_total": int(g_tot), "g_by_stratum": g_by,
            "n_by_stratum": {st: len(rs) for st, rs in by_st.items()},
            "note": "descriptive (C3): стратумная структура линий без T0.3-detected; "
                    "НЕ в первичный вердикт (H2′ — primary)"}


def run_h3(docs, p_map, doc_names, Bn=B):
    """H3 (robustness, descriptive): Q-тест (1-p null, как v2.2) на h_raw|L{5,11,17} (30×3)."""
    strata = {"h_L5": [], "h_L11": [], "h_L17": []}
    for nm in sorted(p_map):
        di = doc_names[nm]
        p = p_map[nm]
        for key, L in (("h_L5", 5), ("h_L11", 11), ("h_L17", 17)):
            x = h_cell(docs[nm], L)
            b = np.exp(2j * math.pi * np.arange(x.shape[0]) / p)
            var_d = np.var(x, axis=0)
            q_obs = float(np.sum(np.abs(x.T @ b.conj()) ** 2 / (x.shape[0] * var_d)))
            X = _perm_batch(x, Bn, di)
            S = X.transpose(0, 2, 1) @ b.conj()[:, None]
            Qp = np.sum(np.abs(S) ** 2 / (x.shape[0] * var_d[None, :, None]), axis=1).ravel()
            pval = null_pvalue(q_obs, Qp)
            strata[key].append({"doc": nm, "p": int(p), "q_obs": q_obs, "p": float(pval)})
            logline(f"H3 {key} {nm}: Q = {q_obs:.1f}, p = {pval:.4f}")
    out = {}
    for key, rows in strata.items():
        out[key] = {"n": len(rows), "f": int(sum(1 for r in rows if r["p"] <= ALPHA)),
                    "rows": rows}
    return out


def run_descriptive(docs, p_map, grid, h1_rows, Bn=B):
    """DESCRIPTIVE (reportable only):
    (a) spectral_map: Q-сетка + r_map (R-сетка) per-doc (per-doc p-сетка, M1);
    (b) bnn_ratio: Q_obs(p)/median_perm(Q(p)) per p (знак смещения на реальных данных, C5);
    (c) AR(2)-фит (описательно) на 30 H1-доках с нормировкой r_k = c_k/(N−k) (M3),
        матрица [r0 r1; r1 r0] rhs [r1 r2] (r1-M1);
    (d) эмпирич. SD перестановочного нуля Q(p_T0.3) per H1-док (M2, из h1_rows);
    (e) per-doc: p_at_max Q, R_max (внутренние)."""
    spec_map, r_map, bnn = {}, {}, {}
    for nm in sorted(docs):
        n_p = pmax_idx(grid["T"][nm])
        q = grid["q_obs"][nm][:n_p]
        ps_grid = P_GRID[:n_p]
        spec_map[nm] = {str(p): float(v) for p, v in zip(ps_grid, q)}
        r_map[nm] = {str(p): float(v) for p, v in zip(ps_grid, r_from_q(q, ps_grid))}
        bnn[nm] = {str(p): float(v) for p, v in
                   zip(ps_grid, q / np.maximum(grid["medians"][nm], 1e-12))}
    ar2 = {}
    for nm in sorted(p_map):
        x = h_cell(docs[nm], 23)
        N = x.shape[0]
        r0 = float(np.sum(x * x)) / N
        r1 = float(np.sum(x[:, :-1] * x[:, 1:])) / (N - 1)
        r2 = float(np.sum(x[:, :-2] * x[:, 2:])) / (N - 2)
        M = np.array([[r0, r1], [r1, r0]])
        try:
            phi = np.linalg.solve(M, np.array([r1, r2]))
        except np.linalg.LinAlgError:
            ar2[nm] = {"status": "singular"}
            continue
        roots = np.roots([1.0, -phi[0], -phi[1]])
        r = float(np.max(np.abs(roots)))
        ar2[nm] = {"phi": [float(phi[0]), float(phi[1])], "r": r,
                   "nonstationary": bool(r > 1 + 1e-9)}
    rs = [v["r"] for v in ar2.values() if "r" in v]
    logline(f"DESCRIPTIVE: Q/R-карты {len(spec_map)} доков; AR2 {len(ar2)} доков "
            f"(r: min={min(rs):.3f} max={max(rs):.3f})" if rs else
            "DESCRIPTIVE: Q/R-карты; AR2: все сингулярные")
    # (e) per-doc: p_at_max Q, R_max (внутренние) — helper (чистота, не inline)
    per_doc_summary = {}
    for nm in sorted(docs):
        n_p = pmax_idx(grid["T"][nm])
        q = grid["q_obs"][nm][:n_p]
        ps_grid = P_GRID[:n_p]
        r_arr = r_from_q(q, ps_grid)
        interior_i = [i for i, p in enumerate(ps_grid) if 4 <= p <= pmax_grid(grid["T"][nm]) - 1]
        per_doc_summary[nm] = {
            "p_at_max_q": int(ps_grid[int(np.argmax(q))]),
            "r_max_interior": float(np.max(r_arr[interior_i])) if interior_i else None,
        }
    return {
        "spectral_map": spec_map,
        "r_map": r_map,
        "bnn_ratio": bnn,
        "bnn_note": "Q_obs(p)/median_perm(Q(p)) — real-background/null по p (C5)",
        "ar2_fit": ar2,
        "ar2_norm": "r_k = c_k/(N-k) (M3)",
        "perm_q_sd_per_doc": {r["doc"]: r["perm_q_sd"] for r in h1_rows},   # (d) M2
        "per_doc_summary": per_doc_summary,
    }


# ------------------------------------------------------------------- main

def build_grid(docs, doc_names, subset, Bn):
    """Per-doc grid-null: Q-сетка (per-doc p-сетка, M1) real + (B, n_p) перестановок
    одним batched-gemm. Возвращает dict {T, q_obs, Qp, medians} по subset-докам."""
    grid = {"T": {}, "q_obs": {}, "Qp": {}, "medians": {}}
    for nm in subset:
        x = h_cell(docs[nm], 23)
        T = x.shape[0]
        n_p = pmax_idx(T)
        q_obs = q_grid(x, n_p)
        Qp = perm_q_grid(x, doc_names[nm], Bn=Bn, n_p=n_p)
        grid["T"][nm] = T
        grid["q_obs"][nm] = q_obs
        grid["Qp"][nm] = Qp
        grid["medians"][nm] = np.median(Qp, axis=0)
        logline(f"GRID {nm}: T' = {T}, n_p = {n_p} (p ≤ {pmax_grid(T)}), "
                f"Q_max = {q_obs.max():.1f}")
    return grid


def main():
    os.makedirs(STORE_DIR, exist_ok=True)
    pf._selfcheck()
    pf_sha = sha256_file(os.path.join(ROOT, "prism/field.py"))
    logline(f"G1′ field.py sha {pf_sha[:16]} selfcheck OK")

    docs = load_all_docs()
    g0 = g0_data(docs)
    p_map, h2_pool, strata = load_frozen_t03()
    logline(f"замероз. вход T0.3: {len(p_map)} доков с p (p∈{min(p_map.values())}..{max(p_map.values())}); "
            f"H2-пул {len(h2_pool)} доков (loopy-und "
            f"{sum(1 for d in h2_pool if strata[d] == 'loopy')}, clean "
            f"{sum(1 for d in h2_pool if strata[d] == 'clean')}, remainder "
            f"{sum(1 for d in h2_pool if strata[d] == 'remainder')})")
    assert len(p_map) == 30, f"H1-пул = {len(p_map)} (ожид. 30)"
    assert len(h2_pool) == 66, f"H2-пул = {len(h2_pool)} (ожид. 66)"
    assert not (set(p_map) & set(h2_pool)) and len(p_map) + len(h2_pool) == 96 == len(docs)
    pset = sorted(set(p_map.values()))
    h2p_pool = [d for d in h2_pool if strata[d] == "clean"]
    assert len(h2p_pool) == 53, f"H2′-пул (clean) = {len(h2p_pool)} (ожид. 53)"
    doc_names = {nm: i for i, nm in enumerate(sorted(docs))}

    # гейты (ДО интерпретации)
    g1a = g1a_identity(docs)
    g1b = g1b_unit()
    g2 = g2_cells(docs)
    g7 = run_g7_calibration(Bn=49, N=4) if SMOKE else run_g7_calibration()
    pos = run_positive(p0s=POS_P0[:1], Bn=49) if SMOKE else run_positive()
    gates_ok = bool(g0["ok"] and g1a["ok"] and g1b["ok"] and g2["ok"] and g7["ok"] and pos["ok"])
    logline(f"GEYTY: G0={g0['ok']} G1a={g1a['ok']} G1b={g1b['ok']} G2={g2['ok']} "
            f"G7={g7['ok']} POS={pos['ok']} → {'PASS' if gates_ok else 'FAIL'}")

    # per-doc grid-null (96 доков|L23 — один заход; H2/H2′/DESC — редукции поверх)
    Bn = B
    if SMOKE:
        sm_loopy, sm_clean = "wiki_1", "chat_1"
        grid_docs = [sm_loopy, sm_clean]
        Bn = 49
    else:
        grid_docs = sorted(docs)
    grid = build_grid(docs, doc_names, grid_docs, Bn)

    # тесты
    if SMOKE:
        h1 = run_h1(docs, {sm_loopy: p_map[sm_loopy]}, doc_names, Bn=Bn)
        g9 = run_g9(docs, {sm_loopy: p_map[sm_loopy]}, doc_names, Bn=Bn)
        h2p = run_h2p(docs, [sm_clean], doc_names, pset, grid, Bn=Bn)
        h2 = run_h2_desc(docs, [sm_clean], strata, doc_names, grid, Bn=Bn)
        h3 = {"smoke_only": True}
        desc = {"smoke_only": True}
    else:
        h1 = run_h1(docs, p_map, doc_names, Bn=Bn)
        g9 = run_g9(docs, p_map, doc_names, Bn=Bn)
        h2p = run_h2p(docs, h2p_pool, doc_names, pset, grid, Bn=Bn)
        h2 = run_h2_desc(docs, h2_pool, strata, doc_names, grid, Bn=Bn)
        h3 = run_h3(docs, p_map, doc_names, Bn=Bn)
        desc = run_descriptive(docs, p_map, grid, h1["rows"], Bn=Bn)

    # вердикт (G9-предварительный → H1/H2′)
    if SMOKE:
        verdict = {"status": "SMOKE (без вердикта)"}
    else:
        H1c = h1["f"] >= H1_FLOOR
        H2pc = h2p["g2"] >= H2P_FLOOR
        f_dh = g9["f_dh"]
        if H1c and f_dh <= G9_TILT_MAX:
            verdict = {"status": "TILT_ARTIFACT",
                       "detail": f"H1 (f={h1['f']}/30) срабатывает, но Δh|L23 (f_dh={f_dh}/30 ≤ "
                                 f"{G9_TILT_MAX}) — «линия» h = integration-amplified структура "
                                 "(tilt интегрированного ряда), не компонента динамики; H1 НЕ "
                                 f"интерпретируется (G9, C1)"}
        elif h1["f"] >= H1_SIG and not H1c:
            v4 = {"status": "INCONCLUSIVE",
                  "detail": f"H1 значима (f={h1['f']}/30, P={binom_sf(H1_SIG, 30, ALPHA):.3e}), "
                            f"но порог пре-рега {H1_FLOOR}/30 не достигнут; f_dh = {f_dh}"}
            if f_dh <= G9_TILT_MAX:
                v4["dh_silent"] = True   # r4 MINOR-4: tilt-like при суб-пороговом H1
            verdict = v4
        elif H1c and not H2pc:
            verdict = {"status": "CONFIRMED",
                       "detail": f"узкая спектральная линия на петлевом периоде в h|L23 (f={h1['f']}/30), "
                                 f"loopy-специфична (H2′: g2={h2p['g2']}/53 < {H2P_FLOOR} на matched-p), "
                                 f"короборована Δh (f_dh={f_dh}/30). C4: локализация на уже найденном "
                                 f"T0.3 периоде; НЕ «сверх AR(2)» — см. ОБЛАСТЬ"}
        elif H1c and H2pc:
            verdict = {"status": "NON-SPECIFIC",
                       "detail": f"узкие period-p линии не loopy-специфичны: есть и в clean на "
                                 f"detected-периодах (H2′: g2={h2p['g2']}/53 ≥ {H2P_FLOOR}, matched-p); "
                                 f"T0.3-detected = порогой срез (ρ ≥ 0.9) более широкой популяции линий "
                                 f"(H2-стратумы: loopy-undetected тоже имеют линии)"}
        elif (not H1c) and H2pc:
            verdict = {"status": "STRANGE",
                       "detail": f"узкие линии в clean на detected-p (g2={h2p['g2']}/53), но не у loopy "
                                 f"на p_T0.3 (f={h1['f']}/30) — противоречие T0.3↔T0.6, ручной разбор"}
        else:
            verdict = {"status": "REFUTED",
                       "detail": f"период-p структура T0.3 не видна как узкая спектральная линия в h "
                                 f"(f={h1['f']}/30) и не специфична (g2={h2p['g2']}/53); f_dh = {f_dh}"}
        if not (H1c and f_dh <= G9_TILT_MAX) and H1c and G9_TILT_MAX < f_dh <= G9_WEAK_MAX:
            verdict["dh_weak"] = True
        logline(f"VERDICT: {verdict['status']} (f={h1['f']}/30, f_dh={g9['f_dh']}/30, "
                f"g2={h2p['g2']}/53, g_desc={h2['g_total']}/66)")

    # архив v2.2 (если полный артефакт прежней версии на месте)
    old_claim = os.path.join(OUT_DIR, "AT0_6_SPECTRUM.json")
    if os.path.exists(old_claim):
        try:
            old = json.load(open(old_claim))
            if old.get("round") != "v3":
                arch = os.path.join(OUT_DIR, "archive", "v2.2")
                os.makedirs(arch, exist_ok=True)
                for fn in ("AT0_6_SPECTRUM.json", "AT0_6_manifest.json", "AT0_6_run.log"):
                    pth = os.path.join(OUT_DIR, fn)
                    if os.path.exists(pth):
                        os.replace(pth, os.path.join(arch, fn))
                logline(f"архив v2.2 артефактов → {os.path.relpath(arch, OUT_DIR)}/")
        except (json.JSONDecodeError, AttributeError):
            pass

    # артефакты
    store = {
        "meta": {
            "round": "v3",
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "field_sha256": pf_sha,
            "data_sha256": g0["data_sha256"],
            "at03_sha256": g0["at03_sha256"],
            "gen_sha_ok": g0["gen_sha_ok"],
            "decode_probe": g0["decode_probe"],
            "n_docs": len(docs),
            "B": B, "t_prime_range": g2["t_prime_range"], "p_grid": [P_GRID[0], P_GRID[-1]],
            "pmax_rule": "p ≤ min(48, floor(T'/4)) (M1)",
            "pset": pset,
            "seed_convention": "2026 + doc_idx*1009 + s (sorted(96)); synth 777+k / 7770+k "
                               "(те же перестановки, что v2.2 — сопоставимость версий)",
            "smoke": bool(SMOKE),
        },
        "prereg": {
            "tz": "docs/TZ_add_0-3.md §1.5 T0.6 (v3; ревью docs/AT0_6_REVIEW.md C1–C5, M1–M4)",
            "h1": {"pool": "30 T0.3-detected, h_raw|L23, R(p_T0.3) на 3-p null (p−1,p,p+1)",
                   "floor": H1_FLOOR, "sig": H1_SIG, "alpha": ALPHA,
                   "binom_p_floor": binom_sf(H1_FLOOR, 30, ALPHA),
                   "binom_p_sig": binom_sf(H1_SIG, 30, ALPHA),
                   "c4": "локализация на уже найденном T0.3 периоде — не независимое переоткрытие"},
            "h2p": {"pool": "53 clean (T0.3-стратум), h_raw|L23, max R(p) по P_set(doc) = "
                           "P_set ∩ {4..floor(T'/4)−1}, null = max по тем же p",
                    "floor": H2P_FLOOR, "alpha": ALPHA,
                    "binom_p_floor": binom_sf(H2P_FLOOR, 53, ALPHA),
                    "note": "P(Bin(53,.05)≥7) = 0.0160 — консервативнее 5%, объявлено (C2/C3)"},
            "h2": {"pool": "66 без T0.3-p, max Q по per-doc сетке (M1), null max по той же",
                   "status": "descriptive (НЕ в вердикт; C3 стратификация)"},
            "g9": {"rule": "H1c ∧ f_dh ≤ 3 → TILT_ARTIFACT; 4 ≤ f_dh ≤ 14 → dh_weak",
                   "tilt_max": G9_TILT_MAX, "weak_max": G9_WEAK_MAX,
                   "object": "Δh|L23, те же 30, R(p_T0.3) 3-p null (C1-fix 1)"},
            "g7": {"n": G7_N, "k_max": G7_K_MAX,
                   "false_fail_p": binom_sf(G7_K_MAX + 1, G7_N, ALPHA),
                   "note": "(a) max-Q k ≤ 3 (FPR ≤ 0.15 — C5-слово); (b) G7b: max-R по "
                           "внутренним p ∈ 4..47: k_r ≤ 3"},
            "positive": {"a": POS_A, "p0": POS_P0, "seed": POS_SEED,
                         "gate": "4/4 p ≤ 0.05 (R-тест)",
                         "effect_size_source": "v2.2 full-run: Q/D = 22..75 по 30 detected "
                                               "(a/σ ≈ 0.55)"},
        },
        "gates": {
            "G0": {"ok": g0["ok"], "gen_sha_ok": g0["gen_sha_ok"]},
            "G1_field": {"field_sha256": pf_sha, "selfcheck": "OK"},
            "G1a_identity": g1a,
            "G1b_unit": g1b,
            "G2": g2,
            "G7_calibration": {"ok": g7["ok"], "k_q": g7["k_q"], "k_r": g7["k_r"], "gate": g7["gate"]},
            "POSITIVE": {"ok": pos["ok"], "gate": pos["gate"], "rows": pos["rows"]},
            "all_ok": gates_ok,
        },
        "tests": {
            "H1": {"n": h1["n"], "f": h1["f"], "per_doc": h1["rows"], "note": h1["note"]},
            "H2p": {"n": h2p["n"], "g2": h2p["g2"], "per_doc": h2p["rows"], "pset": h2p["pset"],
                    "note": h2p["note"]},
            "H2": {"n": h2["n"], "g_total": h2["g_total"], "g_by_stratum": h2["g_by_stratum"],
                   "n_by_stratum": h2["n_by_stratum"], "per_doc": h2["rows"], "note": h2["note"]},
            "G9": g9,
            "H3": h3,
            "verdict": verdict,
        },
        "descriptive": desc if not SMOKE else {"smoke_only": True},
    }
    store_j = jclean(store)
    blob = json.dumps(store_j, ensure_ascii=False, sort_keys=True)
    assert "NaN" not in blob and "Infinity" not in blob, "store: NaN/Inf в json"
    store_sha = hashlib.sha256(blob.encode()).hexdigest()
    store_path = os.path.join(STORE_DIR, f"{store_sha[:16]}.json")
    with open(store_path, "w") as f:
        f.write(blob)
    logline(f"store: {os.path.relpath(store_path, ROOT)} ({os.path.getsize(store_path) / 1e6:.1f} MB)")

    claim = {
        "experiment": "AT0_6 T0.6 (спектральная линия на петлевом периоде, matched-filter + R-статистика v3)",
        "round": "v3",
        "date": "2026-11-15 (UTC+3)",
        "store": os.path.relpath(store_path, OUT_DIR),
        "store_sha256": store_sha,
        "meta": store["meta"],
        "prereg": store["prereg"],
        "gates": store["gates"],
        "tests": {k: store["tests"][k] for k in ("H1", "H2p", "H2", "G9", "verdict")},
        "h3_summary": ({k: {"n": v["n"], "f": v["f"]} for k, v in store["tests"]["H3"].items()
                        if isinstance(v, dict) and "n" in v} if not SMOKE else store["tests"]["H3"]),
    }
    claim_j = jclean(claim)
    claim_str = json.dumps(claim_j, ensure_ascii=False, indent=1, sort_keys=True)
    assert "NaN" not in claim_str and "Infinity" not in claim_str, "claim: NaN/Inf в json"
    claim_sha = hashlib.sha256(claim_str.encode()).hexdigest()
    claim_path = os.path.join(OUT_DIR, "AT0_6_SPECTRUM.json")
    with open(claim_path, "w") as f:
        f.write(claim_str)
    logline(f"claim: {claim_path} sha {claim_sha[:16]}")

    man = {
        "experiment": "AT0_6 T0.6 (v3)",
        "date": "2026-11-15 (UTC+3)",
        "claim_path": os.path.relpath(claim_path, ROOT),
        "claim_sha256": claim_sha,
        "store_path": os.path.relpath(store_path, ROOT),
        "store_sha256": store_sha,
        "script_sha256": store["meta"]["script_sha256"],
        "field_sha256": pf_sha,
        "data_sha256": store["meta"]["data_sha256"],
        "at03_sha256": store["meta"]["at03_sha256"],
    }
    man_str0 = json.dumps(man, ensure_ascii=False, indent=1, sort_keys=True)
    man["manifest_sha256"] = hashlib.sha256(man_str0.encode()).hexdigest()
    man_str = json.dumps(man, ensure_ascii=False, indent=1, sort_keys=True)
    with open(os.path.join(OUT_DIR, "AT0_6_manifest.json"), "w") as f:
        f.write(man_str)
    logline("manifest: OK (self-hash по blob до вставки)")

    with open(os.path.join(OUT_DIR, "AT0_6_run.log"), "w") as f:
        f.write("\n".join(_log) + "\n")
    logline("AT0_6 DONE" + (" (SMOKE)" if SMOKE else ""))


if __name__ == "__main__":
    main()
