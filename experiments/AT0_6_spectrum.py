#!/usr/bin/env python3
"""AT0_6 — T0.6 «Спектральная линия на петлевом периоде» (matched-filter, v2.2 после r1–r3 ревью).

ТЗ: docs/TZ_add_0-3.md §1.5 T0.6 (v2.2). Ревью: docs/dynamic/AT0_6_TZ_REVIEW_r1.md (r1: М1 YW-
матрица, М2 AR(2)-null недискриминативен — оба закрыты редизайном: AR(2) теперь ТОЛЬКО
описательно; тест — matched filter + per-doc permutation null).

Вопрос: есть ли в спектре h|L23 статистически значимая узкая период-p компонента на частоте
1/p (p = p_T0.3), и loopy-специфична ли она? (Дискриминация «линия vs AR(2) с тем же ω₀» при
T′=383 невозможна — dwell-time аргумент; этот вопрос — у AT0_5. Объявлено в ТЗ-ОБЛАСТИ.)

Estimand (клетка × p): matched filter на ТОЧНОМ базисе периода p (не DFT-бин; T′=383 простое —
утечка DFT-бинов не проблема): S_d(p) = Σ_{t=0}^{T′−1} x_{t,d}·e^{−2πit/p},
Q(p) = Σ_d |S_d(p)|²/(T′·σ²_d), σ²_d = var_t(x_{t,d}). x = h_raw|L, T′ = T−1 per-doc (без начального;
gen/own 383, corpus 82..254; конвенция T0.3).
E[Q] = D = 1024, SD ≈ √D ≈ 32 (сумма D·Exp(1): Var = D; замер r2: 31.3).

Null (exact, parametric-free): per-doc перестановка t, B = 199; seed = 2026 + doc_idx·1009 + s
(doc_idx = индекс в sorted(96) — конвенция T0.1/AT0_2). Перестановка сохраняет per-dim
multiset (var точно, mean точно ⇒ центрирование одинаково на обеих сторонах), разрушает ВСЮ
периодичность (AR/Лоренц-фон, drift) ⇒ точный нуль «нет линии на ω_p». p =
(#{perm: Q_perm ≥ Q_obs}+1)/(B+1), p_floor = 1/200. Одна и та же формула Q на обеих сторонах.

Замороженный вход: results/AT0/AT0_3_AREA.json (sha256[:16] = 8fe61affb13b443f):
period_structure.per_doc — 30 detected с p (loopy), 66 без p (11 loopy-undetected + 53 clean +
2 remainder {own_w0, own_w4}). T0.6 НЕ переоткрывает периоды: p_T0.3 вербатим.

Тесты (пре-рег):
H1 (primary): h_raw|L23, 30 detected-доков, per-doc на p = p_T0.3[doc]; null = перестановки на
  том же p (НЕ max-сетка). f = #{docs: p ≤ 0.05}/30; f ≥ 15 → H1 CONFIRMED
  (P(Bin(30,0.05)≥15) = 2.3e-12); 10 ≤ f ≤ 14 → значимо (P = 1.16e-6), уровень отчёта,
  вердикт INCONCLUSIVE (порог пре-рега не достигнут); f < 10 → не подтверждена.
H2 (специфичность): h_raw|L23, 66 доков без T0.3-detection; статистика = max_{p∈3..48} Q(p);
  null = max по той же сетке в тех же B-перестановках (мультиселектность учтена). g =
  #{docs: p_max ≤ 0.05}/66; g ≥ 7 → H2 CONFIRMED (P(Bin(66,0.05)≥7) = 0.0464); g ≤ 6 →
  не подтверждена (P(≥6) = 0.112 — НЕ тест 5%, порог пре-рега).
H3 (robustness, descriptive — без вердикта): H1-тест (per-doc p_T0.3, перестановочный null на
  том же p) на Δh|L23 (30 доков) и h_raw|L{5,11,17} (30×3 клетки) — f per stratum, отчёт.
DESCRIPTIVE (reportable only): (a) сетка Q(p), p∈3..48, по 96 докам|L23 («спектральная карта»);
  (b) AR(2)-фит (ТОЛЬКО описательно): Yule-Walker на суммарных по dim'ам моментах центрир. x:
  [c0 c1; c1 c0][φ₁;φ₂] = [c1; c2] (верная матрица, r1-M1); отчёт r = max|корень λ²−φ₁λ−φ₂=0|,
  флаг нестационарности (|λ| > 1+1e-9).

Вердикт (primary family = {H1, H2} пуловые binomial; FWER не применяется — объявлено):
 H1 ∧ ¬H2 → CONFIRMED (loopy-специфичная спектральная линия на петлевом периоде в h|L23).
 H1 ∧ H2 → NON-SPECIFIC (спектральный пик не loopy-специфичен; T0.3-структура — общее свойство
   h-динамики; НЕ «петля неспецифична» — T0.3 standalone стоит).
 ¬H1 ∧ H2 → STRANGE (flag для ручного разбора: противоречие T0.3↔T0.6).
 ¬H1 ∧ ¬H2 → REFUTED (структура T0.3 не видна как спектральная линия в h — живёт в
   Δh/остаточной структуре).
 (10 ≤ f < 15 → INCONCLUSIVE: значимо, но порог пре-рега не достигнут.)

Контроли (гейты, ДО интерпретации):
 G0: sha256 gen/own/cor npz + AT0_3-артефакт sha + gen_meta sha_ok 64/64 + декод-пробы
   (gen/own; corpus not_testable G16) — как AT0_2.
 G1a (identity): Q(p) двумя независимыми формулами — комплексный einsum vs
   (Σ x·cos(2πt/p))² + (Σ x·sin(2πt/p))² — max rel dev < 1e-12 на 2 random-доках, все 46 p.
 G1b (unit): (i) чистый cos a=1, σ=1, p₀=13, D=32, T′=383: Q(p₀)/D ≥ 50 ∧
   Q(p₀±1)/Q(p₀) ≤ 0.2 (узкий пик); (ii) AR(2)-синтетика φ=(0.5,−0.3), N=20000: YW-фит
   |φ̂−φ|∞ < 0.05 (валидация описательного фита; r1-M1).
 G2: 96 доков загружены; T′ = 383 per doc (L23); per-dim var > 0. (L1 absent own_w0/own_w4 —
   только L1; H1/H2/H3 не используют L1.)
 G7 (калибровка реализации): 20 iid-синтетик (D=1024, T′=383, seed 777+k, k=0..19, per-dim
   N(0,1)): полный пайплайн (grid-max Q, permutation null B=199) → k = #{p ≤ 0.05}/20;
   гейт: k ≤ 3 (FPR ∈ [0.00, 0.10]; P(Bin(20,0.05) ≥ 4) = 0.016).
 POSITIVE: 4 синтетика a = 0.5σ (измеренный размер эффекта: matched-filter замер дал
   mean a/σ ≈ 0.55 по 30 detected-докам), p₀ ∈ {7, 13, 17, 27} по одному, независимые фазы
   per dim, D=1024, T′=383, seed 7770+k; H1-тест на p₀: гейт 4/4 с p ≤ 0.05.
 G8: NaN/Inf — jclean во всех артефактах; field.py sha-пин + _selfcheck().
Детерминизм: все rng засеяны (2026+doc_idx·1009+s; синтетика 777+k / 7770+k); нет global
np.random; sorted(docs); store sort_keys; json — jclean.
Цена: ~0.7 TFLOP ≈ 1–5 мин CPU; память per-doc < 10 MB.
Границы: read-only данные + field.py (пин) + AT0_1_accounting + AT0_3-артефакт; без форвардов;
writes только results/AT0/AT0_6_*.
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
SF420 = sum(math.comb(20, j) * 0.05 ** j * 0.95 ** (20 - j) for j in range(4, 21))  # P(k≥4|Bin(20,.05)) = 0.016
P_GRID = list(range(3, 49))        # p ∈ 3..48 (T0.3 K=96 → p ≤ 48; p=2 — T0.2, закрыт)
T_PRIME = 383
H1_FLOOR = 15                      # f ≥ 15/30 → H1 CONFIRMED (P = 2.3e-12)
H1_SIG = 10                        # 10 ≤ f ≤ 14 → значимо (P = 1.16e-6), INCONCLUSIVE
H2_FLOOR = 7                       # g ≥ 7/66 → H2 CONFIRMED (P = 0.0464)
G7_K_MAX = 3                       # 20 iid: k ≤ 3 (FPR ≤ 0.15; ложный фейл P(k≥4|Bin(20,0.05)) = 0.016)
POS_A = 0.5                        # a = 0.5σ (измеренный эффект 0.55)
POS_P0 = [7, 13, 17, 27]
POS_SEED = 7770
G7_SEED = 777
G7_N = 20
G1A_TOL = 1e-12
G1B_COS_QD = 50.0                  # Q(p₀)/D ≥ 50
G1B_COS_OFF = 0.2                  # Q(p₀±1)/Q(p₀) ≤ 0.2
G1B_YW_TOL = 0.05
D_DIM = 1024
OUT_DIR = os.path.join(ROOT, "results", "AT0")
STORE_DIR = os.path.join(OUT_DIR, "store")
SMOKE = "--smoke" in sys.argv

_log = []


def logline(s=""):
    print(s, flush=True)
    _log.append(s)


def binom_sf(k, n, p):
    """P(X ≥ k), X ~ Bin(n, p) — точный (math.comb)."""
    return math.fsum(math.comb(n, j) * p ** j * (1.0 - p) ** (n - j) for j in range(k, n + 1))


# ------------------------------------------------------------------- замероз. вход

def load_frozen_t03():
    """AT0_3: period_structure.per_doc → {doc: p} (30 detected) + пул H2 (66 без p)."""
    d = json.load(open(AT03_PATH))
    per_doc = d["results"]["period_structure"]["per_doc"]
    p_map, h2_pool = {}, []
    for stratum in ("loopy", "clean", "remainder"):
        for doc, info in per_doc.get(stratum, {}).items():
            p = info.get("p")
            if p is not None:
                p_map[doc] = int(p)
            else:
                h2_pool.append(doc)
    return p_map, h2_pool


# ------------------------------------------------------------------- данные

def load_all_docs():
    docs = {}
    for path, kind in ((GEN, "gen"), (OWN, "state"), (COR, "state")):
        docs.update(a1.load_source(path, kind))
    return docs


def h_cell(d, L):
    """h_raw|L: (T′, D) float64, центрированный по t per dim. T′ = T−1 (без начального):
    gen/own: 383; corpus: 82..254 (per-doc, T0.3-конвенция)."""
    u, hn = d["layers"][L]
    U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    h = (U * hn[:, None]).astype(np.float64)
    x = h[1:]
    return x - x.mean(axis=0, keepdims=True)


def dh_cell(d, L):
    """Δh|L: (T′, D) float64: h[1:]−h[:-1] по полному h, центрированный."""
    u, hn = d["layers"][L]
    U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    h = (U * hn[:, None]).astype(np.float64)
    x = h[1:] - h[:-1]
    return x - x.mean(axis=0, keepdims=True)


# ------------------------------------------------------------------- ядро: Q(p) + null

def basis_grid(T):
    """E: (n_p, T) complex — e^{+2πi·t/p} на сетке P_GRID для длины T."""
    if T not in _E_CACHE:
        t = np.arange(T)
        E = np.empty((len(P_GRID), T), dtype=np.complex128)
        for i, p in enumerate(P_GRID):
            E[i] = np.exp(2j * math.pi * t / p)
        _E_CACHE[T] = E
    return _E_CACHE[T]


def basis_single(p, T):
    t = np.arange(T)
    return np.exp(2j * math.pi * t / p)


_E_CACHE = {}


def q_grid(x):
    """Q(p) = Σ_d |S_d(p)|²/(T′·σ²_d) по сетке P_GRID: (n_p,). x: (T′, D) центрир.
    (базис из T′ = x.shape[0] — per-doc; один gemm (D,T)@(T,n_p))."""
    T, D = x.shape
    E = basis_grid(T)
    var_d = np.var(x, axis=0)
    S = x.T @ E.T.conj()                    # (D, n_p)
    q = np.sum(np.abs(S) ** 2 / (T * var_d[:, None]), axis=0)
    return q


def q_single(x, p):
    """Q(p) для одного p: один gemm (D,T)@(T,1)."""
    T, D = x.shape
    b = basis_single(p, T)
    var_d = np.var(x, axis=0)
    S = x.T @ b.conj()[:, None]             # (D, 1)
    return float(np.sum(np.abs(S) ** 2 / (T * var_d[:, None])))


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


def perm_null_single(x, p, doc_idx, Bn=B):
    """Per-doc permutation-null на ОДНОМ p: p-value (p_floor = 1/(B+1)).
    Одна формула на обеих сторонах: Q = Σ_d|S_d|²/(T′σ²_d), σ²_d — var real'а
    (перестановка сохраняет его точно — assert выше)."""
    T, D = x.shape
    b = basis_single(p, T)
    var_d = np.var(x, axis=0)
    q_obs = q_single(x, p)
    X = _perm_batch(x, Bn, doc_idx)          # (B, T, D)
    S = X.transpose(0, 2, 1) @ b.conj()[:, None]     # (B, D, 1)
    Q = np.sum(np.abs(S) ** 2 / (T * var_d[None, :, None]), axis=1).ravel()
    n_ge = int(np.sum(Q >= q_obs))
    return q_obs, (n_ge + 1) / (Bn + 1)


def perm_null_grid(x, doc_idx, Bn=B):
    """Per-doc permutation-null на СЕТКЕ (max_{p} Q): p-value + Q_obs-сетка.
    Мультиселектность: max по той же сетке в тех же перестановках."""
    T, D = x.shape
    E = basis_grid(T)
    var_d = np.var(x, axis=0)
    q_obs = q_grid(x)
    max_obs = float(q_obs.max())
    X = _perm_batch(x, Bn, doc_idx)          # (B, T, D)
    S = X.transpose(0, 2, 1) @ E.T.conj()[None]   # (B, D, n_p)
    Q = np.sum(np.abs(S) ** 2 / (T * var_d[None, :, None]), axis=1)   # (B, n_p)
    n_ge = int(np.sum(Q.max(axis=1) >= max_obs))
    return q_obs, max_obs, (n_ge + 1) / (Bn + 1)


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
    (2) прямой комплексный einsum, (3) (Σxcos)²+(Σxsin)². max rel dev < 1e-12. per-doc T′.
    Гейт валидирует ПУТ, используемый в тестах (gemm), а не только альтернативную формулу."""
    names = sorted(docs)
    picked = [names[0], names[len(names) // 2]]
    per_doc = {}
    worst = 0.0
    for nm in picked:
        x = h_cell(docs[nm], 23)
        T = x.shape[0]
        var_d = np.var(x, axis=0)
        q_gemm = q_grid(x)              # production path (D,T)@(T,n_p)
        max_dev = 0.0
        for i, p in enumerate(P_GRID):
            b = basis_single(p, T)
            S = np.einsum("t,td->d", b, x)
            q1 = np.abs(S) ** 2 / (T * var_d)     # прямой комплексный
            c = np.einsum("t,td->d", b.real, x)
            sn = np.einsum("t,td->d", b.imag, x)
            q2 = (c ** 2 + sn ** 2) / (T * var_d)  # cos/sin
            dev_einsum = float(np.max(np.abs(q1 - q2) / np.maximum(q1, 1e-30)))
            dev_gemm = float(np.abs(q_gemm[i] - float(np.sum(q1))) / max(float(np.sum(q1)), 1e-30))
            max_dev = max(max_dev, dev_einsum, dev_gemm)
        per_doc[nm] = max_dev
        worst = max(worst, max_dev)
    logline(f"G1a: identity max rel dev = {worst:.3e} (tol {G1A_TOL}) на {picked} (gemm vs einsum vs cos/sin)")
    return {"ok": bool(worst < G1A_TOL), "max_rel_dev": float(worst), "per_doc": per_doc}


def g1b_unit():
    """G1b: (i) cos-синтетика: узкий пик; (ii) AR(2)-фит восстанавливает φ."""
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
    ok_cos = (q0 / 32 >= G1B_COS_QD) and (off_m / q0 <= G1B_COS_OFF) and (off_p / q0 <= G1B_COS_OFF)
    # (ii) AR(2)-синтетика φ=(0.5, −0.3), N=20000: YW-фит (верная матрица [c0 c1; c1 c0])
    phi_true = np.array([0.5, -0.3])
    rng2 = np.random.default_rng(77701)
    eps = rng2.standard_normal(20000)
    z = np.zeros(20000)
    for tt in range(2, 20000):
        z[tt] = 0.5 * z[tt - 1] - 0.3 * z[tt - 2] + eps[tt]
    z = z - z.mean()
    c0 = float(np.sum(z * z))
    c1 = float(np.sum(z[:-1] * z[1:]))
    c2 = float(np.sum(z[:-2] * z[2:]))
    M = np.array([[c0, c1], [c1, c0]])
    phi_hat = np.linalg.solve(M, np.array([c1, c2]))
    yw_err = float(np.max(np.abs(phi_hat - phi_true)))
    ok_yw = yw_err < G1B_YW_TOL
    logline(f"G1b: cos a=1 σ=1 p0=13 D=32: Q(p0)/D = {q0 / 32:.1f} (≥{G1B_COS_QD}), "
            f"off/on = {off_m / q0:.3f}/{off_p / q0:.3f} (≤{G1B_COS_OFF}); "
            f"AR2-fit φ̂ = ({phi_hat[0]:.4f}, {phi_hat[1]:.4f}) vs (0.5, −0.3), |Δ|∞ = {yw_err:.4f} "
            f"(tol {G1B_YW_TOL})")
    return {"ok": bool(ok_cos and ok_yw), "cos": {"q0_over_D": q0 / 32, "off_over_on": [off_m / q0, off_p / q0]},
            "ar2_fit": {"phi_true": [0.5, -0.3], "phi_hat": [float(phi_hat[0]), float(phi_hat[1])],
                        "abs_err_inf": yw_err}}


def g2_cells(docs):
    """G2: 96 доков; T′ = T−1 (per-doc: gen/own 383, corpus 82..254); per-dim var > 0.
    Короткие клетки (T′ < 96: cor_w8 82, cor_w11 94): перестановочный нуль exact при любом
    T′ (одна формула на обеих сторонах) ⇒ FPR-смещения нет; только power-потеря при p > T′/2."""
    n = len(docs)
    bad = []
    tpmin, tpmax = 10 ** 9, 0
    for nm in sorted(docs):
        u, hn = docs[nm]["layers"][23]
        T = u.shape[0]
        x = h_cell(docs[nm], 23)
        if x.shape[0] != T - 1:
            bad.append((nm, "T'"))
        if not np.all(np.var(x, axis=0) > 0):
            bad.append((nm, "var"))
        tpmin, tpmax = min(tpmin, x.shape[0]), max(tpmax, x.shape[0])
    logline(f"G2: {n} доков, T'∈[{tpmin},{tpmax}] (per-doc T−1), per-dim var>0: "
            f"{'OK' if not bad else bad[:6]}")
    return {"ok": bool(n == 96 and not bad), "n_docs": int(n), "t_prime_range": [int(tpmin), int(tpmax)],
            "short_cells_note": "T′<96 (cor_w8, cor_w11): exact-null, только power-потеря при p > T′/2",
            "bad": bad}


# ------------------------------------------------------------------- тесты

def run_h1(docs, p_map, doc_names, smoke=None):
    """H1: 30 detected, h_raw|L23, per-doc на p_T0.3, permutation-null на том же p."""
    rows = []
    names = sorted(p_map) if smoke is None else [nm for nm in smoke if nm in p_map]
    for nm in names:
        di = doc_names[nm]
        x = h_cell(docs[nm], 23)
        q_obs, pval = perm_null_single(x, p_map[nm], di)
        rows.append({"doc": nm, "p": int(p_map[nm]), "q_obs": float(q_obs), "p": float(pval)})
        logline(f"H1 {nm} p={p_map[nm]}: Q = {q_obs:.1f} (null mean ~1024), p = {pval:.4f}")
    f = sum(1 for r in rows if r["p"] <= ALPHA)
    return {"rows": rows, "n": len(rows), "f": int(f)}


def run_h2(docs, h2_pool, doc_names, smoke=None):
    """H2: 66 доков без T0.3-p, grid-max Q, permutation-null max по той же сетке."""
    rows = []
    names = sorted(h2_pool) if smoke is None else [nm for nm in smoke if nm in h2_pool]
    for nm in names:
        di = doc_names[nm]
        x = h_cell(docs[nm], 23)
        q_obs, max_obs, pval = perm_null_grid(x, di)
        p_at_max = int(P_GRID[int(np.argmax(q_obs))])
        rows.append({"doc": nm, "q_max_obs": float(max_obs), "p_at_max": p_at_max,
                     "p": float(pval), "q_grid": {str(p): float(v) for p, v in zip(P_GRID, q_obs)}})
        logline(f"H2 {nm}: Q_max = {max_obs:.1f} при p={p_at_max}, p = {pval:.4f}")
    g = sum(1 for r in rows if r["p"] <= ALPHA)
    return {"rows": rows, "n": len(rows), "g": int(g)}


def run_h3(docs, p_map, doc_names, smoke=None):
    """H3 (descriptive): H1-тест на Δh|L23 (30) и h_raw|L{5,11,17} (30×3)."""
    strata = {"dh_L23": [], "h_L5": [], "h_L11": [], "h_L17": []}
    names = sorted(p_map) if smoke is None else [nm for nm in smoke if nm in p_map]
    for nm in names:
        di = doc_names[nm]
        p = p_map[nm]
        for key, x in (("dh_L23", dh_cell(docs[nm], 23)),
                       ("h_L5", h_cell(docs[nm], 5)),
                       ("h_L11", h_cell(docs[nm], 11)),
                       ("h_L17", h_cell(docs[nm], 17))):
            q_obs, pval = perm_null_single(x, p, di)
            strata[key].append({"doc": nm, "p": int(p), "q_obs": float(q_obs), "p": float(pval)})
            logline(f"H3 {key} {nm}: Q = {q_obs:.1f}, p = {pval:.4f}")
    out = {}
    for key, rows in strata.items():
        f = sum(1 for r in rows if r["p"] <= ALPHA)
        out[key] = {"n": len(rows), "f": int(f), "rows": rows}
    return out


def run_descriptive(docs, p_map):
    """DESCRIPTIVE: (a) Q-сетка по 96 докам|L23 (per-doc T′); (b) AR(2)-фит (описательно) на 30 H1-доках."""
    spec_map = {}
    for nm in sorted(docs):
        x = h_cell(docs[nm], 23)
        q = q_grid(x)
        spec_map[nm] = {str(p): float(v) for p, v in zip(P_GRID, q)}
    ar2 = {}
    for nm in sorted(p_map):
        x = h_cell(docs[nm], 23)
        T, D = x.shape
        c0 = float(np.sum(x * x))
        c1 = float(np.sum(x[:, :-1] * x[:, 1:]))
        c2 = float(np.sum(x[:, :-2] * x[:, 2:]))
        M = np.array([[c0, c1], [c1, c0]])
        try:
            phi = np.linalg.solve(M, np.array([c1, c2]))
        except np.linalg.LinAlgError:
            ar2[nm] = {"status": "singular"}
            continue
        roots = np.roots([1.0, -phi[0], -phi[1]])
        r = float(np.max(np.abs(roots)))
        ar2[nm] = {"phi": [float(phi[0]), float(phi[1])], "r": r,
                   "nonstationary": bool(r > 1 + 1e-9)}
    rs = [v["r"] for v in ar2.values() if "r" in v]
    logline(f"DESCRIPTIVE: Q-сетка {len(spec_map)} доков; AR2-фит {len(ar2)} доков, "
            f"r: min={min(rs):.3f} max={max(rs):.3f}" if rs else
            f"DESCRIPTIVE: Q-сетка {len(spec_map)} доков; AR2-фит: все сингулярные")
    return {"spectral_map": spec_map, "ar2_fit": ar2}


# ------------------------------------------------------------------- контроли (синтетика)

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
    """POSITIVE: 4 синтетика a=0.5σ, p₀ ∈ {7,13,17,27}; H1-тест на p₀: 4/4 p ≤ 0.05."""
    rows = []
    p0s = p0s or POS_P0
    for k, p0 in enumerate(p0s):
        x = synth_line(p0, POS_A, POS_SEED + k)
        q_obs, pval = perm_null_single(x, p0, 900 + k, Bn=Bn)
        rows.append({"p0": p0, "q_obs": float(q_obs), "p": float(pval), "ok": bool(pval <= ALPHA)})
        logline(f"POSITIVE p0={p0}: Q = {q_obs:.1f}, p = {pval:.4f} ({'OK' if pval <= ALPHA else 'FAIL'})")
    ok = sum(1 for r in rows if r["ok"])
    return {"gate": f"{ok}/{len(p0s)} ≤ {ALPHA}", "ok": bool(ok == len(p0s)), "rows": rows}


def run_g7_calibration(Bn=B, N=None):
    """G7: 20 iid-синтетик, grid-max + permutation-null; гейт k ≤ 3 (FPR ≤ 0.15)."""
    N = N or G7_N
    hits = 0
    rows = []
    for k in range(N):
        x = np.random.default_rng(G7_SEED + k).standard_normal((T_PRIME, D_DIM))
        x = x - x.mean(axis=0, keepdims=True)
        _, max_obs, pval = perm_null_grid(x, 9000 + k, Bn=Bn)
        hit = bool(pval <= ALPHA)
        hits += hit
        rows.append({"k": k, "q_max_obs": float(max_obs), "p": float(pval), "hit": hit})
        logline(f"G7 iid k={k}: Q_max = {max_obs:.1f}, p = {pval:.4f} ({'HIT' if hit else '—'})")
    ok = hits <= G7_K_MAX
    logline(f"G7: k = {hits}/{N} (гейт ≤ {G7_K_MAX}; P(k≥4|Bin(20,0.05)) = {SF420:.4f})")
    return {"gate": f"k ≤ {G7_K_MAX} из {G7_N}", "ok": bool(ok), "k": int(hits), "rows": rows,
            "smoke_note": (f"smoke: B={Bn}, N={N}" if (Bn != B or N != G7_N) else None)}


# ------------------------------------------------------------------- main

def main():
    os.makedirs(STORE_DIR, exist_ok=True)
    pf._selfcheck()
    pf_sha = sha256_file(os.path.join(ROOT, "prism/field.py"))
    logline(f"G1′ field.py sha {pf_sha[:16]} selfcheck OK")

    docs = load_all_docs()
    g0 = g0_data(docs)
    p_map, h2_pool = load_frozen_t03()
    logline(f"замероз. вход T0.3: {len(p_map)} доков с p (p∈{min(p_map.values())}..{max(p_map.values())}); "
            f"H2-пул {len(h2_pool)} доков")
    assert len(p_map) == 30, f"H1-пул = {len(p_map)} (ожид. 30)"
    assert len(h2_pool) == 66, f"H2-пул = {len(h2_pool)} (ожид. 66)"
    assert not (set(p_map) & set(h2_pool)) and len(p_map) + len(h2_pool) == 96 == len(docs)

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

    # тесты
    smoke_docs = ["wiki_1"] if SMOKE else None
    h2_smoke = sorted(h2_pool)[:1] if SMOKE else None
    h1 = run_h1(docs, p_map, doc_names, smoke=smoke_docs)
    h2 = run_h2(docs, h2_pool, doc_names, smoke=h2_smoke)
    if SMOKE:
        h3 = {"smoke_only": True}
        desc = {"smoke_only": True}
    else:
        h3 = run_h3(docs, p_map, doc_names)
        desc = run_descriptive(docs, p_map)

    # вердикт
    if SMOKE:
        verdict = {"status": "SMOKE (без вердикта)"}
    else:
        H1C = h1["f"] >= H1_FLOOR
        H2C = h2["g"] >= H2_FLOOR
        if h1["f"] >= H1_SIG and not H1C:
            verdict = {"status": "INCONCLUSIVE",
                       "detail": f"H1 значима (f={h1['f']}/30, P={binom_sf(H1_SIG, 30, ALPHA):.3e}), "
                                 f"но порог пре-рега {H1_FLOOR}/30 не достигнут"}
        elif H1C and not H2C:
            verdict = {"status": "CONFIRMED",
                       "detail": "loopy-специфичная спектральная линия на петлевом периоде в h|L23 "
                                 "(R9-6 spectral carrier; НЕ «сверх AR(2)» — см. ОБЛАСТЬ ТЗ)"}
        elif H1C and H2C:
            verdict = {"status": "NON-SPECIFIC",
                       "detail": "спектральный пик не loopy-специфичен: период-p линии есть и у "
                                 "non-loopy доков (T0.3-структура — общее свойство h-динамики)"}
        elif (not H1C) and H2C:
            verdict = {"status": "STRANGE",
                       "detail": "флаг для ручного разбора: линии в non-loopy, но не на p_T0.3 loopy "
                                 "(противоречие T0.3↔T0.6)"}
        else:
            verdict = {"status": "REFUTED",
                       "detail": "период-p структура T0.3 не видна как спектральная линия в h "
                                 "(живёт в Δh/остаточной структуре)"}
        logline(f"VERDICT: {verdict['status']} (f={h1['f']}/30, g={h2['g']}/66)")

    # артефакты
    store = {
        "meta": {
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "field_sha256": pf_sha,
            "data_sha256": g0["data_sha256"],
            "at03_sha256": g0["at03_sha256"],
            "gen_sha_ok": g0["gen_sha_ok"],
            "decode_probe": g0["decode_probe"],
            "n_docs": len(docs),
            "B": B, "t_prime_range": g2["t_prime_range"], "p_grid": [P_GRID[0], P_GRID[-1]],
            "seed_convention": "2026 + doc_idx*1009 + s (sorted(96)); synth 777+k / 7770+k",
            "smoke": bool(SMOKE),
        },
        "prereg": {
            "tz": "docs/TZ_add_0-3.md §1.5 T0.6 (v2)",
            "h1": {"pool": "30 T0.3-detected, h_raw|L23, per-doc p_T0.3, perm-null на том же p",
                   "floor": H1_FLOOR, "sig": H1_SIG, "alpha": ALPHA,
                   "binom_p_floor": binom_sf(H1_FLOOR, 30, ALPHA),
                   "binom_p_sig": binom_sf(H1_SIG, 30, ALPHA)},
            "h2": {"pool": "66 без T0.3-p, grid-max Q, p∈3..48, perm-null max по сетке",
                   "floor": H2_FLOOR, "alpha": ALPHA,
                   "binom_p_floor": binom_sf(H2_FLOOR, 66, ALPHA),
                   "note": "P(Bin(66,0.05)≥6) = 0.112 — НЕ тест 5%, только порог пре-рега"},
            "h3": "descriptive: dh|L23 + h|L{5,11,17}, per-doc p_T0.3, без вердикта",
            "g7": {"n": G7_N, "k_max": G7_K_MAX, "false_fail_p": SF420},
            "positive": {"a": POS_A, "p0": POS_P0, "seed": POS_SEED, "gate": "4/4 p ≤ 0.05",
                         "effect_size_source": "matched-filter замер: mean a/σ ≈ 0.55 по 30 detected (провенанс)"},
        },
        "gates": {
            "G0": {"ok": g0["ok"], "gen_sha_ok": g0["gen_sha_ok"]},
            "G1_field": {"field_sha256": pf_sha, "selfcheck": "OK"},
            "G1a_identity": g1a,
            "G1b_unit": g1b,
            "G2": g2,
            "G7_calibration": {"ok": g7["ok"], "k": g7["k"], "gate": g7["gate"]},
            "POSITIVE": {"ok": pos["ok"], "gate": pos["gate"], "rows": pos["rows"]},
            "all_ok": gates_ok,
        },
        "tests": {
            "H1": {"n": h1["n"], "f": h1["f"], "per_doc": h1["rows"]},
            "H2": {"n": h2["n"], "g": h2["g"], "per_doc": h2["rows"]},
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
        "experiment": "AT0_6 T0.6 (спектральная линия на петлевом периоде, matched-filter v2)",
        "round": "v2.2",
        "date": "2026-11-15 (UTC+3)",
        "store": os.path.relpath(store_path, OUT_DIR),
        "store_sha256": store_sha,
        "meta": store["meta"],
        "prereg": store["prereg"],
        "gates": store["gates"],
        "tests": {k: store["tests"][k] for k in ("H1", "H2", "verdict")},
        "h3_summary": {k: {"n": v["n"], "f": v["f"]} for k, v in store["tests"]["H3"].items()
                       if isinstance(v, dict) and "n" in v} if not SMOKE else store["tests"]["H3"],
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
        "experiment": "AT0_6 T0.6 (v2)",
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
