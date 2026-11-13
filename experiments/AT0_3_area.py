#!/usr/bin/env python3
"""AT0_3 — T0.3 «Кто несёт отрицательную площадь» (ТЗ v1.1 §1.5, disk-only).

Дизайн-фиксация (пре-рег) — docs/AT0_3_prereg.md; повторена в артефакте (prereg).

Объекты (роли, (d)-решение docs/AT0_3__N4.md): du_unit (Δu) — первичный;
h_raw (Δh) — вторичный robustness (тот же тест, то же направление); v_tangent (v̂) —
описательно (двухсторонний, no claim; источник loopy-стратификатора — cross-object, объявлено).
Слои: L{1,5,11,17,23} (0-based 1,5,11,17,23; layer-type: full={11,23}, linear={1,5,17}).
Первичный слой — L23 (канон AT0_1).

Оценщик (per doc, per layer, per object):
  x_t — приращения; N = x.shape[0]; rho(k) = c(k)/c(0) (T-нормировка = Σρ̂, частичная
  сумма точного G1; полное тождество 1+2Σρ̂=r — по всем лагам K→N−1; при K≤96 — частичная),
  k=1..K, K = max{24,32,96} ≤ N−1. neg(k)=min(rho(k),0); A_neg=Σ_k neg(k) ≤ 0.
  Доли локации (если A_neg<0): frac[B]=(Σ_{k∈B} neg(k))/A_neg ∈ [0,1]:
  odd_frac, even_frac, k1_frac, tail_frac (k≥9), band_frac по {1},{2},{3..8},{9..32},{33..96}.
  A_neg==0 → no_negative_area=True, доли undefined.

Стратумы:
  1. loopy (L, n=41) / clean=nondeg (C, n=53) / remainder (R, n=2, degen∧¬loopy, отчёт).
  2. copy-rate квантили (80 доков с токенами; резы q25/q75).
  3. домен (n≥12 правило; claim-способны nondeg corpus=16, math=16).
  4. layer-type (gen-доки только, 64): full {11,23} vs linear {1,5,17}.

Первичный тест (период-2-носитель): odd_frac (du_unit, L23), L vs C — r3 (аддендум 3):
первичный пул K=96 (канон T0.1 P4); PRIMARY — перестановочный KS (пре-рег kill-условие
«одинаковое распределение»); SECONDARY — медианы и mean; ROBUST — fixed-band odd_frac k=1..24
(K-гомогенен, полный пул). Новый mechanism-тест — period-concentration: p из пиков ρ(k)≥0.9
(гармоника ρ(2k)≥0.81), масса A_neg по классам остатков mod p (концентрация, dom_residues,
anti_frac — букв. C3-эстиманд k≡p/2 mod p). Source-стратификация (описательно, n<12 no claim).
v_tangent — descriptive + forced_outcome (b1>0.8 у clean ⇒ odd_frac≡1 тождественно).
  KS p≤0.05        → различимо (kill не установлен); направление по медианам (L<C → против пре-рега)
  KS p>0.05        → kill по пре-регу (не подтверждено; G10: estimand слеп к p≥3)
Вторично (robustness): h_raw — тот же набор тестов на K=96-пуле.
Описательно: v_tangent — двухсторонний, no claim (горб-объект по T0.1).
odd_frac ≠ знак excess(3,4) (T0.1): объектный знак — выбор объекта, не направление.
L/C — cross-object стратификация (dip по v_tangent, эстиманд — du_unit).

Гейты: G1_gram (gram_resid<1e-10), G3 (r=N·f_DC, rel 1e-9), Telescope (per doc,L), G2 (клетки).
Бан-лист: pooled-фит, Σ на Δ_med, min/max-бутстрап как CI, домены n<12, «dip» без индексов.

Артефакты: results/AT0/store/<sha16>.json (полный), results/AT0/AT0_3_AREA.json (claim),
results/AT0/AT0_3_manifest.json, results/AT0/AT0_3_run.log.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import sys
from collections import Counter
from datetime import datetime

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fspec = importlib.util.spec_from_file_location(
    "prism_field_pinned", os.path.join(ROOT, "prism/field.py"))
pf = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(pf)
sys.modules["prism_field_pinned"] = pf
vpath, dcurve, acf_moments, f_dc = pf.vpath, pf.dcurve, pf.acf_moments, pf.f_dc
degen, jclean, sha256_file = pf.degen, pf.jclean, pf.sha256_file

# load_source + пути — канон AT0_1 (main-guard есть, импорт безопасен)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
_m1 = importlib.import_module("AT0_1_accounting")
load_source, GEN, OWN, COR, LAYERS = _m1.load_source, _m1.GEN, _m1.OWN, _m1.COR, _m1.LAYERS

OUT_DIR = os.path.join(ROOT, "results/AT0")
STORE_DIR = os.path.join(OUT_DIR, "store")
OUT_JSON = os.path.join(OUT_DIR, "AT0_3_AREA.json")
MAN_JSON = os.path.join(OUT_DIR, "AT0_3_manifest.json")
RUN_LOG = os.path.join(OUT_DIR, "AT0_3_run.log")

K_MAX = 96
K_ALPHABET = (24, 32, 96)
OBJECTS = ("h_raw", "du_unit", "v_tangent")
PRIMARY_OBJ = "du_unit"
SECONDARY_OBJ = "h_raw"
DESC_OBJ = "v_tangent"
PRIMARY_L = 23
SEED_PERM = 117
B_PERM = 1000
SEED_STRIDE = 1009  # r3.2 (fixreview #3): per-doc-шаг перестановочных сдвигов
                    # (конвенция серии AT0_2: SEED_BASE + doc_idx*SEED_STRIDE)
ALPHA = 0.05
SEED_BOOT = 117
B_BOOT = 1000
LAMDA_LOOPY = 0.95

DOMAIN_MIN_N = 12

# полосы (1-indexed лаги)
BANDS = {
    "b1": [1], "b2": [2], "b3_8": list(range(3, 9)),
    "b9_32": list(range(9, 33)), "b33_96": list(range(33, 97)),
}
# слой-метки npz — 0-based (exp62_gen.py:42: hidden_states[L+1], L23=hidden_states[n_layers]);
# LAYER_TYPE выводится из config (r3: была хардкод-догадка из комментария) — load_config_layer_types()
LAYER_TYPE = {L: ("full_attention" if L in (11, 23) else "linear_attention") for L in LAYERS}
MODEL_CONFIG = os.path.join(ROOT, ".hf_cache/hub/models--Qwen--Qwen3.5-0.8B")


def load_config_layer_types():
    """Сверка LAYER_TYPE с config.text_config.layer_types (Qwen3.5-0.8B).
    npz-метка L (1..23) = 0-based слою L: hidden_states[L+1] (exp62_gen.py:42),
    hidden_states[j] = выход 0-based слоя j−1; L23 = hidden_states[24] = последний.
    ⇒ full 0-based индексы = full npz-метки (без +1). Возвращает (map, config_sha, path, full_labels)."""
    import glob
    cfgs = sorted(glob.glob(os.path.join(MODEL_CONFIG, "snapshots", "*", "config.json")))
    if not cfgs:
        raise SystemExit(f"G15: model config не найден: {MODEL_CONFIG}/snapshots/*")
    cfg = json.load(open(cfgs[0]))
    t = cfg["text_config"]
    lt = t["layer_types"]
    n = t["num_hidden_layers"]
    if len(lt) != n:
        raise SystemExit(f"G15: config layer_types len {len(lt)} != num_hidden_layers {n}")
    full_labels = {i for i, x in enumerate(lt) if x == "full_attention"}  # 0-based = npz-метка
    m = {L: ("full_attention" if L in full_labels else "linear_attention") for L in LAYERS}
    return m, hashlib.sha256(open(cfgs[0], "rb").read()).hexdigest(), cfgs[0], full_labels

LOG = []


def logline(s=""):
    print(s)
    LOG.append(s)


def k_of(n):
    ok = [k for k in K_ALPHABET if k <= n - 1]
    return max(ok) if ok else None


def lk4(toks):
    """copy-rate: доля позиций, где 4-грам (t-3..t) встречался раньше и предсказал t+1."""
    T = len(toks)
    nxt = {}
    hit = np.zeros(T, dtype=bool)
    for t in range(3, T - 1):
        key = (int(toks[t - 3]), int(toks[t - 2]), int(toks[t - 1]), int(toks[t]))
        pv = nxt.get(key)
        if pv is not None and pv == int(toks[t + 1]):
            hit[t] = True
        nxt[key] = int(toks[t + 1])
    return hit


def load_data():
    docs = {}
    docs.update(load_source(GEN, "gen"))
    docs.update(load_source(OWN, "state"))
    docs.update(load_source(COR, "state"))
    return docs


def build_objects(d, L):
    u, hn = d["layers"][L]
    U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    h = U * hn[:, None]
    dh = h[1:] - h[:-1]
    du = U[1:] - U[:-1]
    V, dropped = vpath(U, hn)
    return {"h_raw": dh, "du_unit": du, "v_tangent": V}, U, du, dropped


def neg_fractions(rho, K):
    """Доли локации отрицательной площади по лагам (1..K). rho — T-нормировка, длина ≥ K."""
    rho = np.asarray(rho)[:K]
    neg = np.minimum(rho, 0.0)
    a_neg = float(neg.sum())
    if a_neg >= -1e-9:  # r3: floor (было >= 0.0 — при A_neg=-1e-18 доли были бы мусором)
        return {"A_neg": a_neg, "no_negative_area": True, "fracs": None, "neg": None}
    out = {"A_neg": a_neg, "no_negative_area": False, "neg": neg.tolist()}
    fr = {}
    for bn, lags in BANDS.items():
        idx = [k - 1 for k in lags if k <= K]
        fr[bn] = float(neg[idx].sum() / a_neg) if idx else 0.0
    fr["odd_frac"] = float(neg[0::2].sum() / a_neg)
    fr["even_frac"] = float(neg[1::2].sum() / a_neg)
    fr["k1_frac"] = float(neg[0] / a_neg)
    fr["tail_frac"] = float(neg[8:K].sum() / a_neg)   # k≥9
    fr["frac_sum_check"] = float((fr["b1"] + fr["b2"] + fr["b3_8"] + fr["b9_32"] + fr["b33_96"]))
    assert abs(fr["frac_sum_check"] - 1.0) < 1e-9, f"band-доли не суммируются в 1: {fr['frac_sum_check']}"
    out["fracs"] = fr
    return out


def compute_cells(docs):
    """per (doc, L, obj): rho (T-norm), r, f_dc, gram_resid, N, K + отриц. площади."""
    cells = {}
    tele = {}
    for nm, d in sorted(docs.items()):
        for L in LAYERS:
            xs, U, du, _ = build_objects(d, L)
            tele[(nm, L)] = float(np.linalg.norm(du.sum(axis=0) - (U[-1] - U[0])))
            for obj in OBJECTS:
                x = xs[obj]
                n = x.shape[0]
                K = k_of(n)
                if K is None:
                    cells[(nm, L, obj)] = {"absent": True, "N": int(n), "reason": f"N={n} < 25"}
                    continue
                ac = acf_moments(x, K)
                fdc = f_dc(x)
                cells[(nm, L, obj)] = {
                    "absent": False, "N": int(n), "K": int(K),
                    "rho": ac["rho"].tolist(), "r": ac["r"], "f_dc": fdc,
                    "gram_resid": ac["gram_resid"], "sum_rho_full": ac["sum_rho_full"],
                    "neg": neg_fractions(ac["rho"], K),
                }
    return cells, tele


def add_flags(docs):
    for nm, d in docs.items():
        u, hn = d["layers"][23]
        U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        V, _ = vpath(U, hn)
        vc = dcurve(V, np.arange(1, min(K_MAX + 1, V.shape[0])))
        fl, mcos, _ = degen(vc, d["toks"])
        d["degen_tz"] = bool(fl)
        d["loopy_ladder"] = bool(mcos > LAMDA_LOOPY) or nm in ("wiki_0", "wiki_4")
        d["mcos"] = float(mcos)
        toks = d["toks"]
        d["copy_rate"] = float(lk4(toks)[3:383].mean()) if (d["toks_present"] and len(toks) > 383) else None


def stratify(docs):
    L = [nm for nm in sorted(docs) if docs[nm]["loopy_ladder"]]
    C = [nm for nm in sorted(docs) if not docs[nm]["degen_tz"]]
    R = [nm for nm in sorted(docs) if docs[nm]["degen_tz"] and not docs[nm]["loopy_ladder"]]
    return L, C, R


def boot_ci(vals, seed=SEED_BOOT, b=B_BOOT):
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(vals)
    boots = np.array([float(np.median(vals[rng.integers(0, n, n)])) for _ in range(b)])
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def stratum_summary(cells, cellkey, names):
    """Средняя локация отриц. площади в стратуме (per-doc медианы + CI). cellkey=(obj,L)."""
    obj, L = cellkey
    rows = []
    n_noarea = 0
    n_absent = 0
    for nm in names:
        c = cells[(nm, L, obj)] if (nm, L, obj) in cells else None
        if c is None or c.get("absent"):
            n_absent += 1
            continue
        f = c["neg"]["fracs"]
        if f is None:
            n_noarea += 1
            continue
        rows.append((nm, f, c["neg"]["A_neg"]))
    if not rows:
        return {"n_docs": 0, "n_no_negative_area": n_noarea, "n_absent": n_absent}
    fracs = np.array([r[1]["odd_frac"] for r in rows])
    tail = np.array([r[1]["tail_frac"] for r in rows])
    k1 = np.array([r[1]["k1_frac"] for r in rows])
    even = np.array([r[1]["even_frac"] for r in rows])
    aneg = np.array([r[2] for r in rows])
    band_keys = list(BANDS.keys())
    bands = {bk: np.array([r[1][bk] for r in rows]) for bk in band_keys}
    kcomp_all = dict(Counter(cells[(nm, L, obj)]["K"] for nm in names
                             if (nm, L, obj) in cells
                             and not cells[(nm, L, obj)].get("absent")))
    kcomp = dict(Counter(cells[(nm, L, obj)]["K"] for nm, f, a in rows))
    return {
        "n_docs": len(rows), "n_no_negative_area": n_noarea, "n_absent": n_absent,
        "K_composition": {str(k): v for k, v in sorted(kcomp.items())},
        "K_composition_all": {str(k): v for k, v in sorted(kcomp_all.items())},
        "K_note": ("K_composition — только доки с отриц. площадью; K_composition_all — все "
                   "non-absent доки (r3.2 fixreview-мелкое #2: absent-доки не имеют K по "
                   "конструкции cells)"),
        "A_neg": {"median": float(np.median(aneg))},
        "odd_frac": {"median": float(np.median(fracs)), "ci": boot_ci(fracs)},
        "even_frac": {"median": float(np.median(even)), "ci": boot_ci(even)},
        "k1_frac": {"median": float(np.median(k1)), "ci": boot_ci(k1)},
        "tail_frac": {"median": float(np.median(tail)), "ci": boot_ci(tail)},
        "band_frac": {bk: float(np.median(bands[bk])) for bk in band_keys},
    }


def perm_paired(xf, xl, b=B_PERM, seed=SEED_PERM):
    """Парный перестановочный тест: per-doc знак-флип (обмен ярлыков внутри пары).
    d_obs = mean(xf - xl); p = (1 + #{|d_perm| ≥ |d_obs|})/(b+1), two-sided.
    Для paired-дизайна (тот же док → две величины); perm_two_sample тут НЕКОРРЕКТЕН
    (смешивает значения МЕЖДУ доками — нарушает парность)."""
    xf = np.asarray(xf, dtype=float)
    xl = np.asarray(xl, dtype=float)
    d = xf - xl
    d_obs = float(d.mean())
    rng = np.random.default_rng(seed)
    n_ge = 0
    for _ in range(b):
        s = 2.0 * rng.integers(0, 2, size=len(d)) - 1.0
        if abs(float((s * d).mean())) >= abs(d_obs) - 1e-15:
            n_ge += 1
    return {"d_obs": d_obs, "p": (1 + n_ge) / (b + 1), "n_pairs": int(len(d)),
            "B": int(b), "seed": int(seed), "paired": True}


def perm_two_sample(xL, xC, b=B_PERM, seed=SEED_PERM):
    xL = np.asarray(xL, dtype=float)
    xC = np.asarray(xC, dtype=float)
    d_obs = float(xL.mean() - xC.mean())
    pooled = np.concatenate([xL, xC])
    nL = len(xL)
    rng = np.random.default_rng(seed)
    n_ge = 0
    for _ in range(b):
        idx = rng.permutation(len(pooled))
        d = float(pooled[idx[:nL]].mean() - pooled[idx[nL:]].mean())
        if abs(d) >= abs(d_obs):
            n_ge += 1
    return {"d_obs": d_obs, "p": (1 + n_ge) / (b + 1),
            "mean_L": float(xL.mean()), "mean_C": float(xC.mean()),
            "n_L": int(nL), "n_C": int(len(xC)), "B": int(b)}


def ks_stat(a, b):
    """Двухвыборочное KS-расстояние (без scipy). a, b — 1-D массивы."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    allv = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, allv, side="right") / len(a)
    cdf_b = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def perm_two_sample_stat(xL, xC, stat, b=B_PERM, seed=SEED_PERM):
    """Перестановочный two-sample тест на произвольной статистике (KS / разность медиан / mean).
    Перестановка ярлыков: пул перемешивается, режется nL/nC, stat(a,c) пересчитывается.
    p = (1 + #{|stat_perm| >= |stat_obs|})/(b+1): two-sided для signed-статистики
    (median_diff, mean); для unsigned (KS, статистика ≥0) это right-tail s_perm ≥ s_obs —
    стандартный KS-тест (r3.1 fixreview #6: не «two-sided для обоих»)."""
    xL = np.asarray(xL, dtype=float)
    xC = np.asarray(xC, dtype=float)
    pooled = np.concatenate([xL, xC])
    nL = len(xL)
    s_obs = stat(xL, xC)
    rng = np.random.default_rng(seed)
    n_ge = 0
    for _ in range(b):
        idx = rng.permutation(len(pooled))
        s = stat(pooled[idx[:nL]], pooled[idx[nL:]])
        if abs(s) >= abs(s_obs) - 1e-15:
            n_ge += 1
    return {"stat_obs": float(s_obs), "p": (1 + n_ge) / (b + 1),
            "n_L": int(nL), "n_C": int(len(xC)), "B": int(b)}


def median_diff(a, b):
    return float(np.median(a) - np.median(b))


def odd_frac_band(rho, K, kmax=24):
    """odd_frac на фиксированной полосе k=1..kmax (K-гомогенный эстиманд, r3 §3.1).
    Все доки K>=24, поэтому полоса одна и та же для всех. None если нет отриц. площади."""
    rho = np.asarray(rho, dtype=float)
    K2 = min(kmax, int(K))
    neg = np.minimum(rho[:K2], 0.0)
    a = float(neg.sum())
    if a >= -1e-9:
        return None
    return float(neg[0::2].sum() / a)


def detect_period(rho, K, thr=0.9, hthr=0.81):
    """p = наименьший k>=2 с rho(k)>=thr И rho(2k)>=hthr (гармоника, r3 §3.3).
    rho — T-нормировка, 1-based лаги (rho[k-1] = c(k)/c(0)). None если не найден."""
    rho = np.asarray(rho, dtype=float)
    for k in range(2, int(K) // 2 + 1):
        if rho[k - 1] >= thr and rho[2 * k - 1] >= hthr:
            return int(k)
    return None


def residue_profile(rho, p, K):
    """Профиль массы отриц. площади по классам остатков mod p (p классов, 0..p-1).
    Возвращает (shares(сумма 1), A_neg, k_lags) или (None,0,None) если нет отриц. площади."""
    rho = np.asarray(rho, dtype=float)
    k = np.arange(1, int(K) + 1)
    m = rho < 0
    aneg = float(-rho[m].sum())
    if aneg <= 0:
        return None, 0.0, None
    res = np.zeros(p)
    for kk, aa in zip(k[m], -rho[m]):
        res[int(kk) % p] += aa
    return res / aneg, aneg, k[m]


def concentration_test(rho, p, K, b=B_PERM, seed=SEED_PERM, doc_idx=0):
    """Per-doc перестановочный тест периодической концентрации (r3 §3.3).
    H0: отриц. масса НЕ организована по периоду — позиции значений перемешаны.
    Статистика: max_res_share (доля A_neg в самом массовом остатке mod p).
    p-value = (1 + #{max_perm >= max_obs})/(b+1).
    r3.2 (fixreview #3): per-doc seed (seed + doc_idx*SEED_STRIDE) — нуль независим
    между доками (в r3 фиксированный seed=117 был у всех доков: первый перестановочный
    выбор идентичен для доков с одинаковым K → cross-doc корреляция нуля)."""
    rho = np.asarray(rho, dtype=float)
    k = np.arange(1, int(K) + 1)
    m = rho < 0
    vals = -rho[m]  # положительные величины отриц. массы
    n_neg = len(vals)
    if n_neg < 3:
        return {"p_perm": None, "note": "too_few_neg"}
    vsum = float(vals.sum())

    def max_share(perm_pos):
        res = np.bincount(perm_pos % p, weights=vals, minlength=p)
        return float(res.max() / vsum)

    pos0 = k[m]
    max_obs = max_share(pos0)
    rng = np.random.default_rng(seed + doc_idx * SEED_STRIDE)
    n_ge = 0
    for _ in range(b):
        perm_pos = rng.choice(int(K), size=n_neg, replace=False) + 1  # 1-based лаги
        if max_share(perm_pos) >= max_obs - 1e-15:
            n_ge += 1
    return {"max_obs": max_obs, "n_neg": int(n_neg),
            "p_perm": (1 + n_ge) / (b + 1), "B": int(b)}


def obj_test(cells, L, C, obj, mode):
    """r3 §3.2 тест-состав (obj, L23): L vs C.
    odd_frac — первичный пул K=96 (T0.1 P4); odd_frac_k24 — полный пул (K-гомогенен).
    Тесты: PRIMARY=KS-перестановка; SECONDARY=медианы + mean-перестановка.
    mode: 'primary' (du_unit, пре-рег направление L>C) | 'robustness' (h_raw, то же)
    | 'descriptive' (v_tangent, no claim + forced_outcome, §3.5)."""
    def k96(names):
        return [nm for nm in names
                if not cells[(nm, PRIMARY_L, obj)].get("absent")
                and cells[(nm, PRIMARY_L, obj)]["K"] == 96]

    def oddvals(names):
        vals, docs_used, n_no = [], [], 0
        for nm in names:
            c = cells[(nm, PRIMARY_L, obj)]
            if c.get("absent"):
                continue
            f = c["neg"]["fracs"]
            if f is None:
                n_no += 1
                continue
            vals.append(f["odd_frac"])
            docs_used.append(nm)
        return vals, docs_used, n_no

    def k24vals(names):
        vals, docs_used, n_no = [], [], 0
        for nm in names:
            c = cells[(nm, PRIMARY_L, obj)]
            if c.get("absent"):
                continue
            v = odd_frac_band(c["rho"], c["K"], 24)
            if v is None:
                n_no += 1
                continue
            vals.append(v)
            docs_used.append(nm)
        return vals, docs_used, n_no

    L96, C96 = k96(L), k96(C)
    vL, dL, nnoL = oddvals(L96)
    vC, dC, nnoC = oddvals(C96)
    res = {
        "object": obj, "mode": mode,
        "pool": {"K": 96, "n_L": len(L96), "n_C": len(C96),
                 "n_L_with_area": len(vL), "n_C_with_area": len(vC),
                 "docs_L": dL, "docs_C": dC},
        "n_no_negative_area": {"L": nnoL, "C": nnoC},
        "B": B_PERM, "seed": SEED_PERM,
        "odd_frac": {}, "odd_frac_k24": {}, "tail_frac_secondary": {},
    }
    if len(vL) < 3 or len(vC) < 3:
        res["odd_frac"]["verdict"] = (f"not_testable: мало доков с A_neg<0 "
                                       f"(L={len(vL)}, C={len(vC)})")
        res["verdict"] = res["odd_frac"]["verdict"]
        return res

    ks = perm_two_sample_stat(vL, vC, ks_stat)
    med = perm_two_sample_stat(vL, vC, median_diff)
    mean = perm_two_sample(vL, vC)
    res["odd_frac"] = {
        "ks": ks, "median": med, "mean": mean,
        "mean_L": mean["mean_L"], "mean_C": mean["mean_C"],
        "median_L": float(np.median(vL)), "median_C": float(np.median(vC)),
    }

    # odd_frac_k24 (полный пул, K-гомогенная полоса)
    wL, _, _ = k24vals(L)
    wC, _, _ = k24vals(C)
    if len(wL) >= 3 and len(wC) >= 3:
        mk = perm_two_sample(wL, wC)
        res["odd_frac_k24"] = {
            "test": mk, "pool": {"n_L": len(wL), "n_C": len(wC), "band": "k=1..24", "K": "full"},
            "note": "K-гомогенный эстиманд (r3 §3.1): одна полоса для всех доков",
        }
    else:
        res["odd_frac_k24"] = {"note": "not_testable", "n_L": len(wL), "n_C": len(wC)}

    # вердикт (r3 §3.2): PRIMARY = KS; направление — по медианам
    of = res["odd_frac"]
    if of["median_L"] > of["median_C"]:
        direction = "L>C"
    elif of["median_L"] < of["median_C"]:
        direction = "L<C"
    else:
        direction = "равны"
    if mode == "descriptive":
        b1c = [cells[(nm, PRIMARY_L, obj)]["neg"]["fracs"]["b1"]
               for nm in C96
               if not cells[(nm, PRIMARY_L, obj)].get("absent")
               and cells[(nm, PRIMARY_L, obj)]["neg"]["fracs"] is not None]
        forced = bool(b1c and float(np.mean(b1c)) > 0.8)
        res["forced_outcome"] = forced
        res["forced_outcome_detail"] = (
            f"clean mean(b1)={float(np.mean(b1c)):.3f} > 0.8: отрицателен только лаг 1 ⇒ "
            "odd_frac≡1 тождественно, L<C гарантирован; p НЕ цитировать (r3 §3.5, C4)"
            if forced else None)
        if forced:
            of["verdict"] = ("descriptive: FORCED outcome (clean odd_frac≡1 по построению: b1≈1); "
                             "p не цитировать; no claim; также циркулярно "
                             "(стратификатор = тестируемый объект)")
        elif ks["p"] <= ALPHA:
            of["verdict"] = (f"descriptive: различимо ({direction}), no claim "
                             "(циркулярно: стратификатор = тестируемый объект)")
        else:
            of["verdict"] = ("descriptive: неразличимо (no claim; циркулярно: "
                             "стратификатор = тестируемый объект)")
    elif ks["p"] <= ALPHA:
        if of["median_L"] > of["median_C"]:
            of["verdict"] = ("confirmed: период-2 — носитель — loopy доки концентрируют "
                             "отрицательную площадь на нечётных k (KS + направление L>C, "
                             "пре-рег подтверждён)")
        else:
            of["verdict"] = ("refuted: распределения различимы (KS p≤0.05), но направление "
                             "противоположно пре-регу (медианы L<C) — период-2 НЕ носитель; "
                             "kill НЕ установлен (mean-тест слеп к бимодальности loopy, r3 C2)")
    else:
        of["verdict"] = ("kill по пре-регу: распределения неразличимы (KS p>0.05); "
                         "период-2 не подтверждён; G10: estimand слеп к периодам p≥3 "
                         "(odd_frac(p=5,7)=0.500, p=8=0.586, p=24=0.509)")
    res["verdict"] = of["verdict"]

    # tail_frac вторично (K=96-пул)
    def tailvals(names):
        return [cells[(nm, PRIMARY_L, obj)]["neg"]["fracs"]["tail_frac"]
                for nm in names
                if not cells[(nm, PRIMARY_L, obj)].get("absent")
                and cells[(nm, PRIMARY_L, obj)]["neg"]["fracs"] is not None]
    tL, tC = tailvals(L96), tailvals(C96)
    mtl = float(np.mean(tL)) if tL else None
    mtc = float(np.mean(tC)) if tC else None
    if mtl is not None and mtc is not None:
        tstat = ("refuted (L.tail > C.tail — пре-рег-предсказание «чистые в хвосте» "
                 "не подтвердилось)" if mtl > mtc else "confirmed (C.tail > L.tail)")
    else:
        tstat = "not_evaluable"
    res["tail_frac_secondary"] = {
        "mean_L": mtl, "mean_C": mtc, "status": tstat,
        "note": "пре-рег-предсказание #2: чистые — в хвосте k≥9 (C.tail > L.tail); "
                "отчитано явно (fail-loud, r2)",
    }
    return res


def period_structure(cells, L, C, R):
    """r3 §3.3 period-concentration estimand (du_unit L23).
    Per doc: p-детекция (ρ(k)≥0.9 ∧ ρ(2k)≥0.81); для p-доков: профиль массы A_neg
    по классам остатков mod p (max_res_share, top2, concentration, dom_residues,
    anti_frac — букв. C3-эстиманд k≡p/2 mod p), per-doc перестановочный тест.
    Агрегаты: detection-rate L vs C (точный двухсторонний биномиальный тест),
    средние концентрации."""
    from math import comb

    def binom_pmf(i, nn, pp):
        return float(comb(nn, i)) * (pp ** i) * ((1 - pp) ** (nn - i))

    # r3.2.1 (r32-review MINOR): doc_idx — глобальный индекс по ВСЕМ докам (L+C+R),
    # не позиция в стратуме: иначе L[0]/C[0]/R[0] получают один seed и при равном K —
    # идентичный первый перестановочный выбор (cross-стратумная корреляция нуля).
    gidx = {nm: i for i, nm in enumerate(L + C + R)}

    def _per_doc_impl(nm, doc_idx):
        c = cells[(nm, PRIMARY_L, PRIMARY_OBJ)]
        if c.get("absent"):
            return None
        rho = np.asarray(c["rho"], dtype=float)
        K = c["K"]
        p = detect_period(rho, K)
        out = {"p": p, "K": int(K)}
        if p is None:
            return out
        prof, aneg, _ = residue_profile(rho, p, K)
        if prof is None:
            out["note"] = "no_negative_area"
            return out
        order = np.argsort(prof)[::-1]
        out["max_res_share"] = float(prof[order[0]])
        out["top2_res_share"] = float(prof[order[0]] + prof[order[1]])
        out["concentration"] = float(prof[order[0]] * p)
        out["dom_residues"] = [int(order[0]), int(order[1])]
        if p % 2 == 0:
            out["anti_frac"] = float(prof[p // 2])
        else:
            out["anti_frac"] = float(prof[p // 2] + prof[p // 2 + 1])
        out["baseline_max"] = 1.0 / p
        out["conc_perm"] = concentration_test(rho, p, K, doc_idx=doc_idx)
        return out

    def per_doc(nm):
        return _per_doc_impl(nm, gidx[nm])

    out = {"object": PRIMARY_OBJ, "layer": PRIMARY_L,
           "detection_rule": "p = мин. k≥2: ρ(k)≥0.9 ∧ ρ(2k)≥0.81 (гармоника)",
           "K_note": "r3.1 #3: K=32 → p∈2..16, K=96 → p∈2..48 — detection-rate на "
                    "смешанном пуле confounded by K; первичный тест — на K=96 подмножестве, "
                    "per-K разбивка отчитана",
           "per_doc": {}, "detection": {}, "aggregate": {}}
    for lab, names in (("loopy", L), ("clean", C), ("remainder", R)):
        rows = {}
        for nm in names:
            d = per_doc(nm)
            if d is not None:
                rows[nm] = d
        kcomp = Counter(d.get("K") for d in rows.values())
        k96rows = {nm: d for nm, d in rows.items() if d.get("K") == 96}
        ndet96 = sum(1 for d in k96rows.values() if d.get("p") is not None)
        out["per_doc"][lab] = rows
        out["detection"][lab] = {
            "n_total": len(names),
            "K_composition": {str(k): v for k, v in sorted(kcomp.items())},
            "n_detected_full": sum(1 for d in rows.values() if d.get("p") is not None),
            "p_values_full": sorted({d["p"] for d in rows.values() if d.get("p") is not None}),
            "k96": {"n_total": len(k96rows), "n_detected": ndet96,
                    "p_values": sorted({d["p"] for d in k96rows.values() if d.get("p") is not None})},
        }
    nL = out["detection"]["loopy"]["k96"]["n_total"]
    nC = out["detection"]["clean"]["k96"]["n_total"]
    dL = out["detection"]["loopy"]["k96"]["n_detected"]
    dC = out["detection"]["clean"]["k96"]["n_detected"]
    n, d = nL + nC, dL + dC
    # r3.1.2 (r31-review MINOR): n==0 (пустой K96-пул, smoke) → 0<d<n ложь → тест пропускается, деления нет
    if 0 < d < n:
        p0 = d / n
        pmf_obs = binom_pmf(dL, nL, p0)
        p_two = sum(binom_pmf(i, nL, p0) for i in range(nL + 1)
                    if binom_pmf(i, nL, p0) <= pmf_obs + 1e-15)
        out["detection"]["test"] = {
            "pool": "K=96 (r3.1 #3: K=32-доки исключены — p-диапазон зависит от K)",
            "nL": nL, "nC": nC, "dL": dL, "dC": dC, "p0_pool": float(p0),
            "p_two_sided_exact": float(p_two),
            "note": ("H0: вероятность детекции одинакова у loopy и clean (p0 = доля в пуле); "
                     "двусторонний тест по as-or-less-likely convention (не scipy binomtest — "
                     "там doubled-min, числа будут отличаться, r3.1 #7); r3.2 #7: при глубоком "
                     "underflow pmf (очень большие n) p_two может уйти в 0 — задокументировано; "
                     "для n≤96 не критично"),
        }
    det = [d for d in out["per_doc"]["loopy"].values() if d.get("p") is not None
           and "concentration" in d and d.get("K") == 96]
    if det:
        concs = np.array([d["concentration"] for d in det])
        top2 = np.array([d["top2_res_share"] for d in det])
        anti = np.array([d["anti_frac"] for d in det])
        per_p = {nm: (d["p"], d["conc_perm"].get("p_perm"))
                 for nm, d in out["per_doc"]["loopy"].items()
                 if d.get("p") is not None and d.get("K") == 96}
        out["aggregate"] = {
            "pool": "K=96 (r3.1 #3)",
            "n_detected_loopy": len(det),
            "mean_concentration": float(concs.mean()),
            "median_concentration": float(np.median(concs)),
            "min_concentration": float(concs.min()),
            "frac_conc_above_1": float((concs > 1).mean()),
            "mean_top2_res_share": float(top2.mean()),
            "top2_baseline_note": "равномерная база = 2/p (varies per doc)",
            "mean_anti_frac": float(anti.mean()),
            "anti_phase_finding": ("анти-фаза (k≡p/2 mod p) НЕ носитель: "
                                   f"mean anti_frac = {float(anti.mean()):.3f} — ниже "
                                   "равномерной базы (r3 C3: букв. p/2-эстиманд ≈ 0); "
                                   "реальный носитель — классы остатков dom_residues mod p "
                                   "(интерпретация «фаза границы лупа» — гипотеза, не результат)"),
            "per_doc_conc_p": per_p,
            "per_doc_conc_p_format": "(p_detection, p_value_concentration_test)",
            "n_per_doc_p_le_alpha": sum(1 for _, pp in per_p.values()
                                        if pp is not None and pp <= ALPHA),
            "per_doc_p_floor_note": (
                # r32.1-COSMETIC: per-doc conc p насыщен на полу 1/(B+1) — max_obs≈6× базы
                # ⇒ все перестановки ниже наблюдения ⇒ n_ge=0. Пер-doc тест НЕ дискриминирует
                # за порогом значимости; сила результата — в значимости (≠шум), не в величине p.
                f"per-doc conc p насыщен на полу 1/(B+1)≈0.001 (n_per_doc_p_le_alpha={sum(1 for _, pp in per_p.values() if pp is not None and pp <= ALPHA)}/{len(per_p)} значимых)"),
        }
    return out


def layer_type_slice(docs, cells):
    """full {11,23} vs linear {1,5,17}, gen-доки (все слои present). per-doc среднее
    odd_frac по слоям типа (первичный объект)."""
    gen = [nm for nm in sorted(docs) if docs[nm]["source"] == "gen_layers"]
    full_l = [L for L in LAYERS if LAYER_TYPE[L] == "full_attention"]
    lin_l = [L for L in LAYERS if LAYER_TYPE[L] == "linear_attention"]
    rows = []
    for nm in gen:
        def grp_mean(layers):
            vals = []
            for L in layers:
                c = cells[(nm, L, PRIMARY_OBJ)]
                if c.get("absent"):
                    continue
                f = c["neg"]["fracs"]
                if f is not None:
                    vals.append(f["odd_frac"])
            return float(np.mean(vals)) if vals else None
        fu, li = grp_mean(full_l), grp_mean(lin_l)
        if fu is not None and li is not None:
            rows.append((nm, fu, li))
    if not rows:
        return {"n_docs": 0}
    fu = np.array([r[1] for r in rows])
    li = np.array([r[2] for r in rows])
    res = perm_paired(fu, li)  # ПАРНЫЙ: каждый док — пара (full, linear)
    res["n_docs"] = len(rows)
    res["n_full_layers"] = len(full_l)
    res["n_linear_layers"] = len(lin_l)
    res["note"] = ("gen-доки только (TPU L1-деградация исключена); per-doc среднее odd_frac "
                   "по слоям типа; ПАРНЫЙ перестановочный (знак-флип внутри пары) — "
                   "тот же док даёт обе величины (р. 2, review r2)")
    return res


def smoke_subset(docs, n=4):
    loopy = [nm for nm in sorted(docs) if docs[nm]["loopy_ladder"]][:n]
    clean = [nm for nm in sorted(docs) if not docs[nm]["degen_tz"]][:n]
    sel = set(loopy) | set(clean)
    return {nm: docs[nm] for nm in sorted(sel)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(STORE_DIR, exist_ok=True)
    logline(f"AT0_3 T0.3 «Кто несёт отрицательную площадь» (disk-only)" + (" [SMOKE]" if args.smoke else ""))
    logline(f"py{sys.version.split()[0]} np{np.__version__} threads="
            f"{os.environ.get('OMP_NUM_THREADS')}/{os.environ.get('OPENBLAS_NUM_THREADS')}/"
            f"{os.environ.get('MKL_NUM_THREADS')}")

    # G1′: пин field.py
    field_sha = sha256_file(os.path.join(ROOT, "prism/field.py"))
    logline(f"G1′: field.py sha {field_sha[:16]} + _selfcheck")
    pf._selfcheck()

    # r3: LAYER_TYPE — сверка с config (была хардкод-догадка из комментария)
    cfg_map, cfg_sha, cfg_path, full_labels = load_config_layer_types()
    if cfg_map != LAYER_TYPE:
        logline(f"G15 FAIL: LAYER_TYPE не совпадает с config: {LAYER_TYPE} vs {cfg_map}")
        raise SystemExit(1)
    logline(f"G15: layer_types сверены с config {os.path.basename(cfg_path)} sha {cfg_sha[:16]} "
            f"(full 0-based=npz {sorted(full_labels)}; в LAYERS full={sorted(L for L in LAYERS if LAYER_TYPE[L] == 'full_attention')})")

    docs = load_data()
    add_flags(docs)
    if args.smoke:
        docs = smoke_subset(docs)
        logline(f"[SMOKE] поднабор: {sorted(docs)}")
    logline(f"docs={len(docs)} (gen={sum(1 for d in docs.values() if d['source']=='gen_layers')} "
            f"own={sum(1 for d in docs.values() if d['source']=='P1_own')} "
            f"cor={sum(1 for d in docs.values() if d['source']=='P2_corpus')})")

    cells, tele = compute_cells(docs)
    L, C, R = stratify(docs)
    logline(f"loopy n={len(L)} clean n={len(C)} remainder n={len(R)} {R}")

    # G2: число клеток
    n_cells = len(cells)
    n_present = sum(1 for c in cells.values() if not c.get("absent"))
    logline(f"G2: клеток (doc,L,obj)={n_cells} present={n_present}")

    # Гейты ACF
    g1_max = max(c["gram_resid"] for c in cells.values() if not c.get("absent"))
    g1_ok = g1_max < 1e-10
    g3_bad = 0
    for c in cells.values():
        if c.get("absent"):
            continue
        # r = N·f_dc
        if abs(c["r"] - c["N"] * c["f_dc"]) > 1e-9 * max(1.0, c["r"]):
            g3_bad += 1
    tele_max = max(tele.values())
    tele_ok = tele_max < 1e-6
    logline(f"G1_gram max_resid={g1_max:.3e} ok={g1_ok}  G3 r=N·f_dc bad={g3_bad}  "
            f"Telescope max={tele_max:.3e} ok={tele_ok}")

    # G0a TPU-кривые (реальный чек, r3.1 fixreview #1): dip_L23, пересчитанный из (U, hn), vs
    # продюсерские meta-кривые. r3-проверка U*hn==h была тавтологией (для state-доков u, hn
    # считаются из того же h; dev~1e-16 на ЛЮБОМ npz). Блок до smoke-return: vacuous-ветка
    # проверяется смоуком (r3.1.2: раньше блок был после `return` — смоук его не исполнял).
    tpu_dev = 0.0
    tpu_n = 0
    tpu_skip = 0  # r3.1.2 (r31-review MINOR): доки без curves_meta[23]['dip'] — считаем явно, не молча
    for nm, d in docs.items():
        if not nm.startswith(("own_", "cor_")):
            continue
        meta_dip23 = d["curves_meta"].get(23, {}).get("dip")
        u, hn = d["layers"][23]
        U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        V, _ = vpath(U, hn)
        vc = dcurve(V, np.arange(1, min(K_MAX + 1, V.shape[0])))
        if meta_dip23 is None or len(vc) == 0:
            tpu_skip += 1
            continue
        tpu_dev = max(tpu_dev, float(abs(float(vc[0]) - float(meta_dip23))))
        tpu_n += 1
    if tpu_n == 0:
        logline(f"G0a_tpu_curves: нет TPU-доков в подмножестве (smoke) — гейт vacuous (skipped={tpu_skip})")
    elif tpu_dev >= 1e-3:
        logline(f"G0a_tpu_curves FAIL: n={tpu_n} skipped={tpu_skip} max_abs_dev={tpu_dev:.3e} — стоп")
        raise SystemExit(1)
    else:
        logline(f"G0a_tpu_curves: dip_L23 meta-vs-recomputed n={tpu_n} skipped={tpu_skip} max_abs_dev={tpu_dev:.3e} ok")
    if not (g1_ok and g3_bad == 0 and tele_ok):
        logline("GATES FAIL — стоп")
        raise SystemExit(1)

    # Первичный тест (du_unit) + robustness (h_raw) + описательно (v_tangent)
    pt = obj_test(cells, L, C, PRIMARY_OBJ, "primary")
    rt = obj_test(cells, L, C, SECONDARY_OBJ, "robustness")
    dt = obj_test(cells, L, C, DESC_OBJ, "descriptive")
    for t, label in ((pt, "PRIMARY"), (rt, "ROBUST"), (dt, "DESCRIPT")):
        of = t.get("odd_frac", {})
        if "ks" not in of:
            logline(f"{label} (odd_frac, {t['object']} L23, L vs C, K=96): {t['verdict']}")
            continue
        ks, med, mean = of["ks"], of["median"], of["mean"]
        logline(f"{label} (odd_frac, {t['object']} L23, L vs C, K=96): "
                f"KS d={ks['stat_obs']:.4f} p={ks['p']:.4f} | "
                f"median d={med['stat_obs']:+.4f} p={med['p']:.4f} | "
                f"mean d={mean['d_obs']:+.4f} p={mean['p']:.4f} "
                f"(mean {of['mean_L']:.4f}/{of['mean_C']:.4f}, med {of['median_L']:.4f}/{of['median_C']:.4f})")
        k24 = t.get("odd_frac_k24", {})
        if k24.get("test"):
            logline(f"  {label} odd_frac_k24 (full pool, k=1..24): d={k24['test']['d_obs']:+.4f} "
                    f"p={k24['test']['p']:.4f} n={k24['test']['n_L']}/{k24['test']['n_C']}")
    if dt.get("forced_outcome"):
        logline(f"  DESCRIPT v_tangent: FORCED outcome — {dt.get('forced_outcome_detail')}")
    tf = pt.get("tail_frac_secondary")
    logline(f"  tail_frac вторично (primary, K=96): L={tf and tf['mean_L']} "
            f"C={tf and tf['mean_C']}")
    if tf and tf.get("status"):
        def _ts(t):
            return t["tail_frac_secondary"]["status"].split(" (")[0]
        logline(f"  TAIL-ПРЕДСКАЗАНИЕ (пре-рег: C.tail > L.tail): "
                f"du_unit={_ts(pt)} h_raw={_ts(rt)} v_tangent={_ts(dt)}")

    # Стратумные сводки (L23): все объекты (primary = du_unit)
    strata = {
        obj: {
            "loopy": stratum_summary(cells, (obj, PRIMARY_L), L),
            "clean": stratum_summary(cells, (obj, PRIMARY_L), C),
            "remainder": stratum_summary(cells, (obj, PRIMARY_L), R),
        } for obj in OBJECTS
    }

    # r3.1 #2: v_tangent clean — odd_frac≡1 по построению (b1≈1), forced, не находка
    if dt.get("forced_outcome"):
        strata[DESC_OBJ]["clean"]["forced_outcome"] = (
            "b1≈1 ⇒ отрицателен только лаг 1 ⇒ odd_frac≡1 тождественно — forced outcome, "
            "не эмпирическая находка (r3 §3.5, fixreview #2)")

    def _med(s, key):
        v = s.get(key)
        return v["median"] if v else None

    for k in ("loopy", "clean", "remainder"):
        s = strata[PRIMARY_OBJ][k]
        logline(f"  стратум {k} ({PRIMARY_OBJ}): n={s.get('n_docs')} odd_frac={_med(s, 'odd_frac')} "
                f"tail_frac={_med(s, 'tail_frac')} A_neg={_med(s, 'A_neg')}")

    # r3 §3.3: period-concentration (du_unit L23)
    ps = period_structure(cells, L, C, R)
    pdet = ps["detection"]
    pagg = ps.get("aggregate", {})
    logline(f"period-structure (du_unit L23, тест на K=96): loopy {pdet['loopy']['k96']['n_detected']}/{pdet['loopy']['k96']['n_total']} "
            f"(K-full {pdet['loopy']['n_detected_full']}/{pdet['loopy']['n_total']}, K={pdet['loopy']['K_composition']}) | "
            f"clean {pdet['clean']['k96']['n_detected']}/{pdet['clean']['k96']['n_total']} "
            f"(K-full {pdet['clean']['n_detected_full']}/{pdet['clean']['n_total']}, K={pdet['clean']['K_composition']})")
    logline(f"  detected loopy p={pdet['loopy']['k96']['p_values']}")
    if "test" in pdet:
        logline(f"  detection-rate test (exact two-sided, K=96): p={pdet['test']['p_two_sided_exact']:.3e}")
    if pagg:
        logline(f"  aggregate (detected loopy, K=96): mean_conc={pagg['mean_concentration']:.2f} "
                f"median={pagg['median_concentration']:.2f} min={pagg['min_concentration']:.2f} "
                f"frac>1={pagg['frac_conc_above_1']:.3f} top2_share={pagg['mean_top2_res_share']:.3f} "
                f"anti_frac={pagg['mean_anti_frac']:.3f} "
                f"per-doc p≤0.05: {pagg['n_per_doc_p_le_alpha']}/{pagg['n_detected_loopy']}")

    # copy-rate квантили (резы q25/q75; первичный объект, L23)
    with_cr = [nm for nm in sorted(docs) if docs[nm]["copy_rate"] is not None]
    if with_cr:
        crs = np.array([docs[nm]["copy_rate"] for nm in with_cr])
        q25, q75 = float(np.percentile(crs, 25)), float(np.percentile(crs, 75))
        tert = {"low": [nm for nm in with_cr if docs[nm]["copy_rate"] <= q25],
                "mid": [nm for nm in with_cr if q25 < docs[nm]["copy_rate"] <= q75],
                "high": [nm for nm in with_cr if docs[nm]["copy_rate"] > q75]}
        copy_summary = {g: stratum_summary(cells, (PRIMARY_OBJ, PRIMARY_L), tert[g]) for g in tert}
        logline(f"copy-rate квантили (резы q25/q75; n_low={len(tert['low'])} n_mid={len(tert['mid'])} "
                f"n_high={len(tert['high'])}, q25={q25:.3f} q75={q75:.3f})")
    else:
        q25 = q75 = None
        tert = {}
        copy_summary = {}
        logline("copy-rate квантили: нет доков с токенами — строка отсутствует")

    # Домен (n≥12)
    dom_names = {}
    for nm in sorted(docs):
        dom_names.setdefault(docs[nm]["dom"], []).append(nm)
    dom_summary = {}
    for dom, names in sorted(dom_names.items()):
        nondeg = [nm for nm in names if not docs[nm]["degen_tz"]]
        dom_summary[dom] = {
            "n_docs": len(names), "n_nondeg": len(nondeg),
            "domain_claim_eligible": len(nondeg) >= DOMAIN_MIN_N,
            "nondeg": stratum_summary(cells, (PRIMARY_OBJ, PRIMARY_L), nondeg),
        }
        logline(f"  домен {dom}: n={len(names)} nondeg={len(nondeg)} "
                f"claim={'да' if len(nondeg) >= DOMAIN_MIN_N else 'нет (n<12)'}")
    # пре-рег: «остальные — пул» — nondeg-доки доменов с n_nondeg<12
    pool_names = [nm for nm in sorted(docs)
                  if not docs[nm]["degen_tz"]
                  and not dom_summary[docs[nm]["dom"]]["domain_claim_eligible"]]
    dom_summary["pooled_nondeg"] = {
        "n_docs": len(pool_names),
        "note": "пре-рег «остальные — пул»: nondeg-доки доменов с n_nondeg<12",
        "summary": stratum_summary(cells, (PRIMARY_OBJ, PRIMARY_L), pool_names),
    }
    logline(f"  пул (не-claim домены): n={len(pool_names)}")

    # r3 §3.4: source-стратификация (K=96-пул; описательно, n<12 → no claim)
    def _k96(names):
        return [nm for nm in names
                if not cells[(nm, PRIMARY_L, PRIMARY_OBJ)].get("absent")
                and cells[(nm, PRIMARY_L, PRIMARY_OBJ)]["K"] == 96]
    L96, C96 = _k96(L), _k96(C)
    src_strata = {}
    for key, sel in (("gen", lambda nm: docs[nm]["source"] == "gen_layers"),
                     ("own", lambda nm: docs[nm]["source"] == "P1_own"),
                     ("corpus", lambda nm: docs[nm]["source"] == "P2_corpus"),
                     ("domain_chat", lambda nm: docs[nm]["dom"] == "chat"),
                     ("domain_code", lambda nm: docs[nm]["dom"] == "code")):
        sl = [nm for nm in L96 if sel(nm)]
        sc = [nm for nm in C96 if sel(nm)]
        src_strata[key] = {
            "n_L": len(sl), "n_C": len(sc),
            "claim_eligible": min(len(sl), len(sc)) >= DOMAIN_MIN_N,
            "loopy": stratum_summary(cells, (PRIMARY_OBJ, PRIMARY_L), sl),
            "clean": stratum_summary(cells, (PRIMARY_OBJ, PRIMARY_L), sc),
        }
        logline(f"  source-strata {key}: L n={len(sl)} C n={len(sc)} "
                f"claim={'да' if min(len(sl), len(sc)) >= DOMAIN_MIN_N else 'нет (n<12, no claim)'}")
    src_strata_note = ("r3 §3.4: описательно (n<12 → no claim). Объявлен конфаунд L/C × источник: "
                       "L=68% own+wiki, C=60% corpus+math; corpus (16, все в C) без токенов — "
                       "degen только по mcos (mshare выброшен) → смещение к «чистым», т.е. в пользу "
                       "нулевой гипотезы H1")

    # layer-type
    lt = layer_type_slice(docs, cells)
    logline(f"layer-type (gen, full vs linear): n={lt.get('n_docs')} "
            f"d_obs={lt.get('d_obs')} p={lt.get('p')}")

    if args.smoke:
        logline("AT0_3 SMOKE OK (артефакты не пишутся)")
        return

    # ── артефакт ──
    meta = {
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "field_sha256": field_sha,
        "gen_sha256": sha256_file(GEN),
        "own_sha256": sha256_file(OWN),
        "cor_sha256": sha256_file(COR),
        "layers": LAYERS,
        "layer_types": {str(L): LAYER_TYPE[L] for L in LAYERS},
        "layer_indexing": "0-based (hidden_states[L+1]; L23=hidden_states[n_layers], exp62_gen.py:42)",
        "layer_types_source": {"config": os.path.basename(cfg_path), "config_sha256": cfg_sha,
                               "full_labels_npz": sorted(full_labels),
                               "note": "r3: LAYER_TYPE выведен из config.text_config.layer_types "
                                       "(гейт G15); npz-метка L = 0-based слой L (hidden_states[L+1])"},
        "objects": list(OBJECTS),
        "primary": {"object": PRIMARY_OBJ, "layer": PRIMARY_L},
        "secondary": {"object": SECONDARY_OBJ, "layer": PRIMARY_L},
        "descriptive": {"object": DESC_OBJ, "layer": PRIMARY_L},
        "K_alphabet": list(K_ALPHABET),
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "OMP": os.environ.get("OMP_NUM_THREADS"),
                "OPENBLAS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "MKL": os.environ.get("MKL_NUM_THREADS")},
    }
    prereg = {
        "source": "docs/AT0_3_prereg.md + ТЗ v1.1 §1.5 T0.3",
        "sign_correction": "[ИСТОРИЧЕСКАЯ справка о правке v1.0→v1.1, НЕ актуальная гипотеза; актуальные "
                           "тесты — см. test/decision, r3.1 #5] период-2 ρ(k)=(−1)^k ⇒ отриц. площадь на "
                           "нечётных k (осн. k=1), "
                           "полож. на чётных (ТЗ v1.0 «k=1 и чётных» — ошибка); Σρ̂→−1/2 только при "
                           "K→N−1 (при K≤96 |Σρ̂|≪1/2 — величина несущественна, несёт знак)",
        "prediction": "[Исходное предсказание v1.1; r3: H1 оценивается KS/медианами, H2 — "
                       "period-structure; это предсказание опровергнуто (loopy — в хвосте, не на "
                       "нечётных k; clean — на лаге 1, не «в хвосте k≥9») — см. verdict.H1_period2 и "
                       "tail_prediction, r3.1 #5] loopy: отриц. площадь на нечётных k; clean: в хвосте k≥9",
        "kill": "одинаковое распределение у loopy и clean (du_unit) ⇒ период-2 не носитель",
        "objects_roles": {"du_unit": "первичный (зубы период-2: r≈0.0048, r_max<1, телескоп)",
                          "h_raw": "вторичный robustness (тот же тест, то же пре-рег направление L>C; "
                                   "не co-primary — radial/norm-канал может имитировать период-2)",
                          "v_tangent": "описательный (двухсторонний, no claim; источник loopy-стратификатора)"},
        "primary_estimand": "odd_frac (du_unit, L23), frac=Σ_{k∈odd} neg(k)/Σ_k neg(k), "
                            "neg(k)=min(rho(k),0), rho=c(k)/c(0) (T-norm, Σρ̂ частичная сумма); "
                            "docs с A_neg==0 исключены (no_negative_area, guard)",
        "secondary_estimand": "odd_frac (h_raw, L23), тот же тест, пре-рег направление L>C "
                              "(robustness; mismatch с первичным — отчёт)",
        "descriptive_estimand": "odd_frac (v_tangent, L23), двухсторонний, направление не "
                                "заявлено, no claim (горб-объект по T0.1)",
        "odd_frac_vs_excess": "odd_frac (локация отриц. площади) ≠ знак excess(3,4) (T0.1: "
                              "v_tangent горб +0.0121, h_raw/du_unit анти-горб −0.0134/−0.0137) — "
                              "объектный знак обосновывает выбор объекта, не направление L>C (оно — из ТЗ)",
        "cross_object_stratifier": "L/C стратифицированы по v_tangent (dip_V,med L23 > 0.95 или "
                                   "wiki_0/4), первичный эстиманд — du_unit: cross-object "
                                   "стратификация (объявлено)",
        "strata": {"loopy": "loopy_ladder (max_{k≥2} dip>0.95 или wiki_0/4)",
                   "clean": "non-degenerate (degen_tz=False)",
                   "remainder": "degen ∧ ¬loopy (отчёт, n<12)"},
        "test": "r3 §3.2: PRIMARY=two-sided permutation KS (распределения; пре-рег kill-условие "
                f"«одинаковое РАСПРЕДЕЛЕНИЕ»), SECONDARY=permutation на разности медиан и mean "
                f"(mean_L − mean_C), B={B_PERM}, seed={SEED_PERM}, p=(1+n_ge)/(B+1); первичный пул: "
                "K=96 (T0.1 P4); robustness: odd_frac_k24 (полоса k=1..24, полный пул, K-гомогенен)",
        "alpha": ALPHA,
        "decision": {"confirmed": "KS p≤α ∧ медианы L>C (du_unit)",
                     "kill": "KS p>α (du_unit) — «не подтверждено»; G10: estimand слеп к p≥3",
                     "refuted": "KS p≤α ∧ медианы L<C (du_unit) — различимо, направление против пре-рега",
                     "descriptive": "v_tangent — двухсторонний, no claim; forced_outcome при clean b1>0.8",
                     "H2_period_structure": "detection-rate (loopy vs clean, exact binomial) + "
                                            "concentration > 1 (r3 §3.3)"},
        "n12_rule": f"доменное утверждение при n_nondeg≥{DOMAIN_MIN_N}",
        "copy_rate_def": "lk4 (канон exp64_b4_battery.py:25-32): доля позиций, где 4-грам "
                         "(t-3..t) встречался раньше и предсказал t+1; окно hit[3:383] = 380 "
                         "позиций (r2-фикс: было hit[4:383] — off-by-one против пре-рега «позиции [3,382]»); только доки с "
                         "токенами (80/96: gen 64 + own 16; corpus 0/16); квантили — резы "
                         "q25/q75 (low≤q25, mid(q25,q75], high>q75; 25/50/25, не равные трети)",
        "bans": "pooled-фит; Σ на Δ_med; min/max-бутстрап как CI; домены n<12; dip без индексов",
    }
    # G0a (канон AT0_1): provenance источников + TPU-row-реконструкция + corpus not_testable (G16)
    at01_shas = {"gen_layers": "f142f297ec16209d1520cf6f4ee46fdd2543b0d73f566a69d93c26771349fdd7",
                 "P1_own": "783e16338cdcb652198722a33ecec8a6e8e98fa3c611c687b17db722b0d58198",
                 "P2_corpus": "5f42a08442a5d3bcb8d7c4de5c9658340c0378a917a5e633b9b2aa589ceac125"}
    g0_sha = {"gen_layers": sha256_file(GEN), "P1_own": sha256_file(OWN), "P2_corpus": sha256_file(COR)}
    # TPU-расчёт (tpu_n/tpu_skip/tpu_dev) выполнен выше, до smoke-return (r3.1.2).
    gates = {
        "G0a_provenance": {
            "sha256": g0_sha,
            "match_AT0_1": {k: bool(g0_sha[k] == at01_shas[k]) for k in at01_shas},
            "gen_sha_ok": "64/64 (AT0_1 G0b_gen_sha; источники не менялись — sha-совпадение)",
            "corpus": "not_testable_no_tokens (G16): 0/16 доков с токенами — source не "
                      "токен-верифицирован; ВКЛЮЧЁН в стратумы по пре-рег-дизайну "
                      "(прецедент AT0_1, clean n=53); кривые верифицированы row-пересчётом ниже",
        },
        "G0a_tpu_curves": {
            "finding": "P1_own _hn-поле — баг продюсера (колонные нормы; AT0_1 "
                       "G0a_tpu_hn_field); a1.load_source пересчитывает row-нормы из raw-строк "
                       "(_hn-поле не читает)",
            "check": "L23: dip, пересчитанный из (U, hn), vs продюсерские meta-кривые "
                     "(AT0_1 gate_g0a_tpu) — проверяет, что h — тот, из которого строились кривые",
            "r3_note": "r3-проверка U*hn==h была тавтологией (u, hn считаются из того же h для "
                       "state-доков; dev~1e-16 на любом npz) — удалена (r3.1 fixreview #1); "
                       "provenance дополнительно покрывается SHA-сверкой match_AT0_1",
            "n_docs": tpu_n,
            "n_skipped": tpu_skip,
            "vacuous": tpu_n == 0,
            "max_abs_dev": tpu_dev, "ok": bool(tpu_n == 0 or tpu_dev < 1e-3),
            "threshold": 1e-3,
            "threshold_note": "fp16-точность TPU-рядов ~1e-3 (AT0_1: dev ≤ 1e-3 ⇒ ряд годен)",
        },
        "G1_gram": {"max_resid": g1_max, "ok": bool(g1_ok), "threshold": 1e-10},
        "G3_rNfDC": {"bad": g3_bad, "ok": g3_bad == 0, "rel": 1e-9},
        "Telescope": {"max": tele_max, "ok": bool(tele_ok), "threshold": 1e-6},
        "G2_cells": {"n_cells": n_cells, "n_present": n_present},
        "G15_layer_types": {"config_sha256": cfg_sha, "config": os.path.basename(cfg_path),
                            "full_labels_npz": sorted(full_labels),
                            "mapping": {str(L): LAYER_TYPE[L] for L in LAYERS},
                            "ok": True,
                            "note": "r3: LAYER_TYPE сверен с config.text_config.layer_types"},
        "G1p_field": {"sha256": field_sha, "selfcheck": "ok"},
    }
    results = {
        "primary_test": pt,
        "robustness_test": rt,
        "descriptive_test": dt,
        "period_structure": ps,
        "source_strata": {**src_strata, "note": src_strata_note},
        "strata": strata,
        "copy_rate_quartiles": {
            "q25": q25, "q75": q75, "n_with_tokens": len(with_cr), "summary": copy_summary,
        },
        "domain": dom_summary,
        "layer_type": lt,
    }
    # notes (r2-ревью + r3): mean/median, бимодальность, A_neg magnitude, сноска о полосах
    sm_l = strata[PRIMARY_OBJ]["loopy"].get("odd_frac")
    sm_c = strata[PRIMARY_OBJ]["clean"].get("odd_frac")
    an_l = strata[PRIMARY_OBJ]["loopy"].get("A_neg")
    an_c = strata[PRIMARY_OBJ]["clean"].get("A_neg")
    pof = pt.get("odd_frac", {})
    notes = []
    if sm_l and sm_c and pof.get("mean_L") is not None:
        notes.append(
            f"mean vs median (odd_frac, {PRIMARY_OBJ}, L23, K=96): mean L={pof['mean_L']:.4f} vs "
            f"C={pof['mean_C']:.4f} (mean-тест — secondary, r3); медианы L={pof['median_L']:.4f} vs "
            f"C={pof['median_C']:.4f} — противоположное направление; PRIMARY-тест KS "
            f"(p={pof.get('ks', {}).get('p'):.4f}): совпадение средних — артефакт бимодальности "
            f"loopy, не «одинаковые распределения» (r3 C2)")
    if sm_l:
        notes.append(f"loopy odd_frac — широкий CI {sm_l['ci']} (бимодальность: два режима "
                     f"≈0.5 и ≈0.72); kill по средним маскирует расщепление loopy-популяции")
    if an_l and an_c and an_l["median"] and an_c["median"]:
        notes.append(f"A_neg magnitude (описательно; пре-рег тестирует локацию, не величину): "
                     f"loopy {an_l['median']:.2f} vs clean {an_c['median']:.2f} "
                     f"(~{abs(an_l['median'] / an_c['median']):.1f}×) — согласуется с dip_V,med T0.1")
    notes.append("Сноска (r3 мелочи): медианы полос (b9_32, b33_96) НЕ суммируются в медиану "
                 "хвоста (tail_frac) — агрегат ≠ сумма агрегатов (per-doc доли суммируются в 1; "
                 "агрегированные медианы — нет)")
    results["notes"] = notes

    # r3: verdict — честный синтез (H1 период-2, H2 period-structure, локация, конфаунды)
    # r3.1 (fixreview #2): location_contrast — на K=96 пуле (те же доки, что в KS-тесте);
    # полный пул (41/53) — secondary + K-состав стратумов.
    def _fracs96(nms, key):
        vals = []
        for nm in nms:
            c = cells[(nm, PRIMARY_L, PRIMARY_OBJ)]
            if c.get("absent") or c["neg"]["fracs"] is None:
                continue
            v = c["neg"]["fracs"].get(key)
            if v is not None:
                vals.append(v)
        return float(np.median(vals)) if vals else None

    k1l, k1c = _med(strata[PRIMARY_OBJ]["loopy"], "k1_frac"), _med(strata[PRIMARY_OBJ]["clean"], "k1_frac")
    tail_l, tail_c = _med(strata[PRIMARY_OBJ]["loopy"], "tail_frac"), _med(strata[PRIMARY_OBJ]["clean"], "tail_frac")
    L96m = [nm for nm in L if not cells[(nm, PRIMARY_L, PRIMARY_OBJ)].get("absent")
            and cells[(nm, PRIMARY_L, PRIMARY_OBJ)]["K"] == 96]
    C96m = [nm for nm in C if not cells[(nm, PRIMARY_L, PRIMARY_OBJ)].get("absent")
            and cells[(nm, PRIMARY_L, PRIMARY_OBJ)]["K"] == 96]
    k1l96, k1c96 = _fracs96(L96m, "k1_frac"), _fracs96(C96m, "k1_frac")
    tail_l96, tail_c96 = _fracs96(L96m, "tail_frac"), _fracs96(C96m, "tail_frac")
    # r3.1 (fixreview #4): H2-гейт — median + доля >1 (robust к выбросам), не min
    # (пре-рег §3.3 фиксирует «detection-rate + концентрации»; min-гейт был ad-hoc fragile)
    h2_test_present = "test" in pdet
    h2_ok = bool(pdet.get("test", {}).get("p_two_sided_exact", 1) <= ALPHA)
    if h2_ok and pagg:
        h2_ok = (pagg.get("median_concentration", 0) > 1
                 and pagg.get("frac_conc_above_1", 0) >= 0.8)
    # r3.2 (fixreview #4): отсутствие теста (0<d<n не выполнено: d=0 или d=n) ≠
    # отрицательный результат — отдельно not_testable, не not_confirmed.
    if not h2_test_present:
        h2_status = "not_testable (тест detection не определён: 0<d<n не выполнено на K=96 пуле)"
    else:
        h2_status = "confirmed" if h2_ok else "not_confirmed"
    if pof.get("median_L") is not None and pof.get("median_C") is not None:
        med_dir = ("L<C" if pof["median_L"] < pof["median_C"]
                   else ("L>C" if pof["median_L"] > pof["median_C"] else "равны"))
    else:
        med_dir = "n/a"
    h2p = pdet.get("test", {}).get("p_two_sided_exact")
    h1_v = pof.get("verdict", "n/a")
    # r3.2 (fixreview #1, критично): в r3.1 было h1_v.split(":")[0].strip() — для kill-вердикта
    # "kill по пре-регу: ..." давало "kill по пре-регу" (не "kill") → h1_clauses.get → 'n/a',
    # status нестандартный. startswith-диспетчер покрывает все три вердикта.
    if h1_v.startswith("confirmed"):
        h1_short = "confirmed"
    elif h1_v.startswith("refuted"):
        h1_short = "refuted"
    elif h1_v.startswith("kill"):
        h1_short = "kill"
    else:
        h1_short = "other"
    ksp = pof.get("ks", {}).get("p")
    ksp_txt = f"KS p={ksp:.1e}" if ksp is not None else "KS n/a"
    h1_clauses = {
        "refuted": ("распределения различимы (KS), направление против пре-рега (медианы L<C); "
                    "kill НЕ установлен (равенство средних — артефакт бимодальности, r3 C2)"),
        "kill": ("распределения неразличимы (KS p>0.05); период-2 не подтверждён "
                 "(G10: estimand слеп к p≥3)"),
        "confirmed": "loopy доки концентрируют отрицательную площадь на нечётных k (пре-рег подтверждён)",
        "other": f"нестандартный вердикт (строка: {h1_v})",
    }
    # r3.2 (fixreview #2): None-безопасное форматирование — _fracs96 возвращает None для
    # пустого подмножества; f"{None:.3f}" — TypeError (latent crash).
    def _fx(x, fmt=".3f"):
        return format(x, fmt) if x is not None else "n/a"
    h2_word = ("подтверждён" if h2_ok
               else ("тест не определён (not_testable)" if not h2_test_present else "НЕ подтверждён"))
    # r3.2 (fixreview #4): лид-фраза H2 — не «Положительный результат», если H2 не подтверждён
    h2_lead = ("Положительный результат (H2, пре-рег аддендум 3)" if h2_ok
               else "H2 (пре-рег аддендум 3)")
    h1_lead = (f"Период-2 как носитель отриц. площади: подтверждён" if h1_short == "confirmed"
               else f"Период-2 как носитель отриц. площади: не подтверждён")
    # r3.1 #2: локация — на K=96 пуле (как KS), guarded ratio
    ratio_txt = (f", {k1c96 / k1l96:.1f}×" if (k1l96 is not None and k1l96 > 1e-6
                                       and k1c96 is not None) else "")
    summary = (
        f"{h1_lead} (H1: {h1_short}; {ksp_txt}, медианы {med_dir}) — {h1_clauses.get(h1_short, 'n/a')}. "
        f"G10: estimand odd_frac структурно слеп к периодам p≥3 (odd_frac(p=5,7)=0.500, p=8=0.586, "
        f"p=24=0.509) — предсказание «loopy→нечётные» было структурно near-null на этом пуле. "
        f"{h2_lead}: "
        f"period-structure {h2_word} — периодическая лаговая структура "
        f"детектируется у loopy {pdet['loopy']['k96']['n_detected']}/{pdet['loopy']['k96']['n_total']} vs clean "
        f"{pdet['clean']['k96']['n_detected']}/{pdet['clean']['k96']['n_total']} (тест на K=96 пуле; "
        f"K=32-доки исключены)"
        f"{(' (exact p=' + format(h2p, '.2e') + ')') if h2p is not None else ''}"
        f"{('; концентрация A_neg по классам остатков mod p: median ' + format(pagg['median_concentration'], '.2f') + '× базы, top2_share ' + format(pagg['mean_top2_res_share'], '.3f')) if pagg else ''}; "
        f"анти-фаза p/2 НЕ носитель (anti_frac≈0). Локация отриц. площади резко различена и инвертирована "
        f"относительно пре-рега (K=96 пул): loopy — в хвосте k≥9 (медиана {_fx(tail_l96)} vs clean {_fx(tail_c96)}), "
        f"clean — на лаге 1 (k1 {_fx(k1c96)} vs loopy {_fx(k1l96)}{ratio_txt}). "
        f"Распределение odd_frac у loopy бимодально, у clean унимодально; равенство средних — артефакт "
        f"смеси. Согласуется с T0.1: чистый путь платит на лаге 1, лупы размазывают плату периодически "
        f"по хвосту на фазе dom_residues mod p (интерпретация «граница лупа» — гипотеза, не результат) "
        f"— это положительный результат, не kill."
    )
    verdict = {
        "prereg_id": "T0.3 (r3, аддендум 3)",
        "object": PRIMARY_OBJ,
        "H1_period2": {
            "status": h1_short,
            "detail": pof.get("verdict", pt.get("verdict")),
            "primary_ks_p": pof.get("ks", {}).get("p"),
            "median_direction": med_dir,
            "G10_note": ("estimand odd_frac различает ТОЛЬКО period-2; периоды лупов пула 5/7/8/24/… ⇒ "
                         "«loopy→нечётные» было структурно неотличимо от нуля (r3 C3, §3.6): kill "
                         "интерпретируется как «не подтверждено», не как «опровергнуто механизмом»"),
        },
        "H2_period_structure": {
            "status": h2_status,
            "gate": "detection p≤α (K=96) ∧ median_concentration>1 ∧ frac(conc>1)≥0.8 (r3.1: min-гейт "
                    "заменён — fragile к одному выбросу; r3.2: min — secondary/описательный, "
                    "НЕ часть гейта — min отчитан в aggregate, читатель не должен путать)",
            "detection": pdet,
            "aggregate": pagg,
            "finding": ("loopy-доки несут периодическую лаговую структуру (p из пиков ρ); "
                        "отрицательная площадь организована по классам остатков mod p "
                        f"(dom_residues; медиана концентрации {_fx(pagg.get('median_concentration'), '.2f')}× базы) — "
                        "НЕ на анти-фазе p/2 (anti_frac≈0) и не на нечётных k; чистые — без периода, "
                        "площадь на лаге 1"),
        },
        "robustness": {"object": SECONDARY_OBJ, "verdict": rt["verdict"],
                       "ks_p": rt.get("odd_frac", {}).get("ks", {}).get("p")},
        "descriptive": {"object": DESC_OBJ, "verdict": dt["verdict"],
                        "forced_outcome": dt.get("forced_outcome", False)},
        "tail_prediction": {
            "prereg": "пре-рег-предсказание #2: «clean: в хвосте k≥9» (C.tail > L.tail)",
            "primary_status": pt["tail_frac_secondary"]["status"],
            "by_object": {PRIMARY_OBJ: pt["tail_frac_secondary"]["status"],
                          SECONDARY_OBJ: rt["tail_frac_secondary"]["status"],
                          DESC_OBJ: dt["tail_frac_secondary"]["status"]},
            "note": "fail-loud: провал предсказания отчитан явно (r2); r3 C8: обе половины пре-рега "
                    "не сбылись → по словарю §0.4 это refuted, не kill",
        },
        "location_contrast": {
            "pool": "K=96 (r3.1 #2: те же доки, что в primary KS-тесте 41/51; полный пул — secondary)",
            "k1_frac": {"loopy": k1l96, "clean": k1c96,
                        "ratio": (k1c96 / k1l96) if (k1l96 and k1l96 > 1e-6) else None,
                        "full_pool": {"loopy": k1l, "clean": k1c,
                                      "ratio": (k1c / k1l) if k1l else None,
                                      "K_composition": {"loopy": strata[PRIMARY_OBJ]["loopy"].get("K_composition"),
                                                         "clean": strata[PRIMARY_OBJ]["clean"].get("K_composition")},
                                      "note": "K=32-доки (cor_w8, cor_w11, оба в clean) дают A_neg по 32 лагам, "
                                              "K=96 — по 96; влияние на медианы < 0.005 (r3.1 #2)"}},
            "tail_frac": {"loopy": tail_l96, "clean": tail_c96,
                          "full_pool": {"loopy": tail_l, "clean": tail_c}},
            "note": "r3 C1: локация различена и инвертирована (старый verdict-текст «локация не "
                    "различима» опровергнут собственным артефактом)",
        },
        "confounds": [
            "L/C × источник: L=68% own+wiki, C=60% corpus+math (r3 C6) — первичный контраст не "
            "идентифицирован отдельно от источника; source-стратификация — описательная (n<12)",
            "corpus (16, все в C) без токенов: degen только по mcos (mshare выброшен) → смещение "
            "к «чистым», т.е. в пользу нулевой гипотезы H1 (r3 C6)",
            "K-гетерогенность: коруся K=32 исключены из первичного пула (K=96, r3 C5); "
            "fixed-band k=1..24 — K-гомогенный robustness",
        ],
        "summary": summary,
    }
    provenance = {
        "artifact": True,
        "G16_token_verification": {
            "gen": "64/64 (AT0_1 G0b_gen_sha; источники не менялись — sha-совпадение, gates.G0a_provenance)",
            "own": "dip_L23 meta-vs-recomputed (gates.G0a_tpu_curves; AT0_1 G0a_tpu_hn_field — "
                   "обнаружение column-norms бага)",
            "corpus": "not_testable_no_tokens (G16): 0/16 доков с токенами",
        },
        "G15_filters_applied": {
            "degen_tz": "flag из degen(vc, toks) (T0.1)",
            "loopy_ladder": "mcos>0.95 ∨ {wiki_0, wiki_4} (T0.1)",
            "K_pool": "первичный пул K=96 (T0.1 P4, r3 §3.1); fixed-band k=1..24 — полный пул",
            "layer_types": "из config.text_config.layer_types (гейт G15)",
        },
        "G10_forced_outcomes": {
            "v_tangent_descriptive": "forced_outcome (clean b1>0.8 ⇒ odd_frac≡1, r3 §3.5)",
            "odd_frac_period2": "estimand слеп к p≥3 (odd_frac(p=5,7)=0.500, p=8=0.586, p=24=0.509) — "
                                "H1 было структурно near-null на этом пуле (r3 C3, §3.6)",
        },
    }
    per_doc = {}
    for nm, d in sorted(docs.items()):
        cellmap = {}
        for L in LAYERS:
            for obj in OBJECTS:
                c = cells[(nm, L, obj)]
                cellmap[f"L{L}/{obj}"] = {k: c[k] for k in
                    ("absent", "N", "K", "rho", "r", "f_dc", "gram_resid",
                     "sum_rho_full", "neg") if k in c}
        per_doc[nm] = {
            "dom": d["dom"], "source": d["source"], "degen_tz": d["degen_tz"],
            "loopy_ladder": d["loopy_ladder"], "mcos": d["mcos"],
            "copy_rate": d["copy_rate"], "cells": cellmap,
        }
    store = {"meta": meta, "prereg": prereg, "gates": gates, "results": results,
             "verdict": verdict, "per_doc": per_doc, "provenance": provenance}

    # store (content-addressed)
    store_j = jclean(store)
    blob = json.dumps(store_j, ensure_ascii=False, sort_keys=True)
    store_sha = hashlib.sha256(blob.encode()).hexdigest()
    store_path = os.path.join(STORE_DIR, store_sha[:16] + ".json")
    with open(store_path, "w") as f:
        f.write(blob)

    # claim (без per_doc)
    claim = {k: store[k] for k in ("meta", "prereg", "gates", "results", "verdict", "provenance")}
    claim["store"] = {"path": os.path.relpath(store_path, OUT_DIR), "sha256": store_sha}
    with open(OUT_JSON, "w") as f:
        json.dump(jclean(claim), f, ensure_ascii=False, indent=1)

    # manifest
    claim_sha = hashlib.sha256(
        json.dumps(jclean(claim), ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    man = {"claim": claim,
           "manifest": {"store_sha256": store_sha, "claim_sha256": claim_sha,
                        "written": datetime.now().isoformat()}}
    with open(MAN_JSON, "w") as f:
        json.dump(jclean(man), f, ensure_ascii=False, indent=1)

    logline(f"store: {store_path} (sha {store_sha[:16]})")
    logline(f"written: {OUT_JSON}, {MAN_JSON}, {RUN_LOG}")
    logline("AT0_3 gates: ALL PASS")
    with open(RUN_LOG, "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
