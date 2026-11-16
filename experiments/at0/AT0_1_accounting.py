#!/usr/bin/env python3
"""AT0_1 — T0.1 «Пересчёт учёта на трёх объектах» (ТЗ v1.1 §1.5, disk-only).

Rev. 5 (архитектурное ревью — docs/AT0_arch_review.md, sha f2decaf1, формального VERDICT нет;
обработано по явной команде оператора):
- D1: переименование block-2 null: «DC+MA(1) null» → «pair-block null (чётный лаг-1 + DC)».
  Старый label ЛОЖЕН: block-2 сохраняет чётный лаг-1 (x_{2i}, x_{2i+1}) и рандомизирует
  нечётный лаг-1 (x_{2i+1}, x_{t+2}) — измерено в ревью (MA(1), N=20000): лаг-1 −0.396 → −0.199.
  Второй столбец: аналитический MA(1)-null, matched по лаг-0/лаг-1 (гауссов, первый порядок;
  описательно — вердикт по первичному каналу). Вердикты не меняются (ревьюер: «не отзыв»).
- D2: NNULL = 60 → 199 (min p = 1/200 = 0.005 ≤ Бонферрони α = 0.05/6 = 0.00833 на 6 пре-регов);
  σ/эффект — первичный канал, p_perm — вторичный (объявлено явно).
- D3: P5 level-канал только по ē (|σ(ē)| ≥ 3); e3/e4 — описательно (было max по 3 статистикам —
  та же инфляция, что sign-OR). Вердикты не меняются (σ_ee = −10.2/−10.9/+14.1).
- D4: source-компаньон-таблица (gen-only / +corpus / +own); строка source_P2_corpus —
  provenance_limited (16/16 доков без токенов, G0a), не домен-подобная; pooled_nondeg —
  примечание о несопоставимости с якорем (16 из 53 — корпус; фильтр ≠ LADDER-якоря).
- D5: aggregates.t0_headline — заголовочный вопрос T0: Σ(ρ̂(k) − f_dc) per object × {T, T−k} ×
  {none, bartlett} (T-none — полный сум по всем лагам, точно; T-k/bartlett — частичный k≤96, объявлено).
- D6: следствие двух нормировок заявлено явно: S(0) = 1 + 2Σρ̄ — только для unit-variance
  нормировки; для энерго-взвешенной (norm=T) замыкается Gram-тождество 1 + 2Σρ̂ = r (G8), не
  unit-ACF; для T-k unit-variance-тождество не выполняется вовсе (знаменатель переменный).
- G10 C/D: wording синхронизирован с p_perm-процедурой (r4-code review) + DC+MA(1) → pair-block;
  «локализовать по k, AT0_2» → AT0_3 (T0.3).
- Продюсер-баг (TPU _hn, e64_gpu.py:125) — вынесен в реестр docs/PRODUCER_BUGS.md.

Rev. 4 (после числового ре-ревью round 2 — docs/AT0_1_review_r2_numbers.md):
- CRIT-1/MAJOR-1: вердикты П4/П5 — против СОБСТВЕННОЙ null статистики, не против нуля.
  Matched null = block-2-перестановка ряда приращений (pair-block: сохраняет чётный лаг-1,
  уничтожает нечётный лаг-1 и лаги ≥ 2 — r5, D1; ТЗ §1.5: null для формы = перестановка),
  199 сидов (r5, D2: family-wise; ТЗ §0.4: ≥5), отдельный сид на (док, s). Аналитическая
  закрытая форма (full permutation точна; у blk=2 при k≥2 пары тоже случайные различные элементы):
  E[ρ̄(k) − f] = −(1−f)/(N−1). В артефакт: null-распределение, p_perm, σ-канал (П5 по
  уровню медиан — sign-счёт не информативен при одностороннем сдвиге, общем для доков).
- MAJOR-2: пре-рег в артефакте: полоса k=9..96 (буква ТЗ §1.5-4), агрегатор per-doc mean,
  DELTA_MIN = 2e-4 (≈ sd null / пол-ширина CI, замер 1e-4…6e-4); K=96-обрезка полосы
  объявлена явно (полного плато k>96 в артефакте нет).
- MAJOR-4: G10 — исход A недостижим на данных (r_max = 4/Σ‖Δu‖² ≤ 4/19.3 ≈ 0.208 < 1;
  замер Σ‖Δu‖² = 19.3…518.3 по 478 du_unit-клеткам) ⇒ исход B реализован.
- MINOR-4: G3 гейтит алфавиты norm/window (T/none). MINOR-2: у П5 guard N > 32 (как у П4;
  инертен: мин N пула = 82). MINOR-7: gram_resid — заметка «только абсолют, путь суммирования».

Rev. 3 (после ре-ревью round 2 — docs/AT0_1_review_r2_code.md; числовое — AT0_1_review_r2_numbers.md):
- CRIT-1/MAJOR-1/2: П4 перепроектирован — полоса k≥9 по букве ТЗ (не 9..32), агрегатор =
  per-doc СРЕДНЕЕ по полосе (медиана по лагам — только чувствительность; рев.2: медиана по
  скошенному лаг-профилю ≠ уровень плато, и нуль старой статистики не ноль — тождество G1
  Σρ̂ = (r−1)/2 ≈ −1/2), DELTA_MIN = 2e-4 зафиксирован ДО прогона (§0.4: ниже разрешения
  ⇒ not_testable, не refuted), вердикты из словаря §0.4.
- MINOR-12: П5 — один комбинированный sign-тест по ē = mean(excess(3), excess(4)), α=0.05
  (два теста через or давали α≈0.098).
- MAJOR-5: dip_mean_1/dip_med_1 только в своих stat-клетках (agg читает из правильной).
- MINOR-5: K-алфавит §0.2 {24,32,96} (K = max {24,32,96} ≤ N−1; K=0 — moment absent);
  поле rho_Tk_norm="T-k" (клетка norm=T — идентичности G3/G8; rho_Tk — T-k, §1.4-1).
- MINOR-10: G8 — счёт и по клеткам, и по независимым (doc,L,obj) (моментный блок shared
  по stat-оси, 2868 = 2×1434).
- MAJOR-2: G10 — без захардкоженного ok-самоаттестата; reachability описана.
- MINOR-6: §1.7 — телескоп на ВСЕХ (doc,L) (не только L23), field._selfcheck() в гейте,
  рамочная чувствительность (θ, ‖Δu‖ vs 2·sin(θ/2), dip_V − dip_du).
- MINOR-7: G0-a — L23 meta-vs-recomputed dip_V,med (оба TPU-источника, per-doc max dev).
- MINOR-8: env (python/numpy/треки) в meta. MINOR-9: одна схема агрегатных строк.
- MINOR-3: П2 n_checked. MINOR-4: _bad_names ловит Δ_med/Δmed. MINOR-11: kind="state".
- MINOR-1: dead placeholder удалён. MAJOR-6: main() → функции (один файл; правила
  «90/400 строк» в репозитории нет — см. ответ в ревью, split на уровне функций).
- MAJOR-4: round-1 .gz/hash-реестр/осиротевшие store-блобы удалены из results/AT0.
- MINOR-14: stdout-лог прогона пишется в results/AT0/AT0_1_run.log (tracked).

Объекты: h_raw (Δh), du_unit (Δu), v_tangent (ортогонализованный v̂, tangent_frame = u_t).
Источники: gen_layers_qwen.npz (64 дока), P1_own.npz (16), P2_corpus.npz (16),
t25_ekurve.json (только provenance — скрипт утрачен, P14).

Пре-рег (ТЗ §1.5 T0.1):
  1. gram_resid < 1e-10 на всех клетках (G8) — гейт
  2. r ≤ r_max на du_unit — теорема, sanity
  3. r_max измерен; доля доков с r_max < 1 → статус арбитра T0 (per-doc)
  4. плато ρ̄(k≥9) == f_DC в пределах парного CI (DC+MA(1))
  5. горб: excess ρ̄(3,4) − f_DC ≠ 0 — единственная не-DC структура
  6. Δ_med: S(0)/Σ НЕ ВЫЧИСЛЯЮТСЯ (П3)
G-гейты: G0-a (sha npz + tok + декод-проба; отклонение: pool_fp текстов недоступен),
G0-b (gen_sha 64/64), G1′ (field.py + якорь LADDER_V2), G2 (hardcoded 64/16/16 + 2880),
G3 (реестр + r = N·f_DC на клетке), G6 (R/A closure, relative), G7 (hash-store),
G8 (грам), G10 (исходы), §1.7 (телескоп, перестановки, selfcheck, рамка).
Артефакты: results/AT0/store/<sha16>.json (полный), results/AT0/T0_ACCOUNTING.json (claim),
results/AT0/AT0_1_manifest.json, results/AT0/AT0_1_run.log.
"""
import hashlib
import inspect
import json
import os
import shutil
import sys
from math import comb

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# G1′ фиксирует ФАЙЛ prism/field.py, не пакет prism: импорт — напрямую из файла
# (importlib), без prism/__init__ (тот требует datasets/transformers — P3, ревью field_r1).
import importlib.util  # noqa: E402

_fspec = importlib.util.spec_from_file_location(
    "prism_field_pinned", os.path.join(ROOT, "prism/field.py"))
pf = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(pf)
sys.modules["prism_field_pinned"] = pf
vpath, dcurve, acf_moments, f_dc = pf.vpath, pf.dcurve, pf.acf_moments, pf.f_dc
ra_split, rho_t, degen, jclean, sha256_file = (
    pf.ra_split, pf.rho_t, pf.degen, pf.jclean, pf.sha256_file)  # noqa: E402

LAYERS = [1, 5, 11, 17, 23]
STATS = ("mean", "median")
K_MAX = 96
K_ALPHABET = (24, 32, 96)      # §0.2: K ∈ {24, 32, 96}
SEED_BOOT = 117
B = 1000
# П4, зафиксировано ДО прогона (§0.4: DELTA_MIN сравнивается с шириной парного CI до
# прогона). Оправдание: width mean-оценщика П4 на round-2 данных 2.1e-4–2.3e-4 (Δh/Δu,
# полоса k≥9); эффекты меньше ~2e-4 этим тестом неразличимы ⇒ not_testable.
DELTA_MIN = 2e-4
ALPHA_P4 = 0.05
ALPHA_P5 = 0.05
# r4 (r2-numbers CRIT-1): matched null П4/П5 — block-2-перестановка ряда приращений.
# r5 (arch-review D1): blk=2 сохраняет ЧЁТНЫЙ лаг-1 (x_{2i}, x_{2i+1}) и рандомизирует
# НЕЧЁТНЫЙ лаг-1 (x_{2i+1}, x_{2i+2}) — «сохраняет ровно лаг-1 (DC+MA(1))» было ложно
# (ревью измерило: MA(1)-лаг-1 вдвое). Label: pair-block null (чётный лаг-1 + DC);
# аналитический MA(1)-null (matched по лаг-0/лаг-1) — второй столбец (P4.null_analytic_ma1).
# NNULL = 199 (r5, D2: family-wise на 6 пре-регов: min p = 1/(NNULL+1) ≤ α/6 = 0.00833;
# ТЗ §0.4: ≥5), отдельный сид на (док, s).
BLK_NULL = 2
NNULL = 199
ANCHOR_LADDER = -0.0722      # LADDER_V2 real_pooled (nonwiki nonloopy, L23, dip_V,med)
ANCHOR_TOL = 0.0006
L1_DEGRADE_TOL = 1e-3        # per-doc гейт: |recomputed dip_V,med L1 − meta| ≤ 1e-3 → валидны

GEN = "/data/prism_kernels/exp62/gen_layers_qwen.npz"
OWN = os.path.join(ROOT, "results/e64/tpu_v8/P1_own.npz")
COR = os.path.join(ROOT, "results/e64/tpu_v8/P2_corpus.npz")
T25_SRC = "/tmp/e63_rerun/t25_ekurve.json"
MODEL_DIR = os.path.join(ROOT, ".hf_cache/hub/models--Qwen--Qwen3.5-0.8B")
OUT_DIR = os.path.join(ROOT, "results/AT0")
STORE_DIR = os.path.join(OUT_DIR, "store")
OUT_JSON = os.path.join(OUT_DIR, "T0_ACCOUNTING.json")   # claim-JSON (без per_doc)
MAN_JSON = os.path.join(OUT_DIR, "AT0_1_manifest.json")
RUN_LOG = os.path.join(OUT_DIR, "AT0_1_run.log")

LOG = []


def logline(s=""):
    print(s)
    LOG.append(s)


def k_of(n):
    """K-алфавит §0.2: max {24,32,96} ≤ N−1 (n = число приращений). None — vacuous."""
    ok = [k for k in K_ALPHABET if k <= n - 1]
    return max(ok) if ok else None


def load_source(path, kind):
    """kind: 'gen' (u+hn+tok), 'state' (raw h; hn-поле P1_own — баг продюсера, см. гейт
    G0a_tpu_hn_field; строки — raw fp16)."""
    z = np.load(path, allow_pickle=True)
    docs = {}
    if kind == "gen":
        meta = json.load(open(path.replace(".npz", "_meta.json")))
        for n in meta:
            d = {}
            for L in LAYERS:
                d[L] = (np.asarray(z[f"{n}_L{L}_u"], dtype=np.float64),
                        np.asarray(z[f"{n}_L{L}_hn"], dtype=np.float64))
            docs[n] = {"layers": d, "toks": np.asarray(z[f"{n}_tok"]),
                       "dom": meta[n]["dom"], "gen_sha": meta[n].get("gen"),
                       "sha_ok": meta[n].get("sha_ok"),
                       "source": "gen_layers", "curves_meta": {}, "toks_present": True}
    else:
        prefix = "own_w" if os.path.basename(path).startswith("P1") else "cor_w"
        m = json.load(open(path.replace(".npz", "_meta.json")))
        for i in range(16):
            n = f"{prefix}{i}"
            d = {}
            for L in LAYERS:
                h = np.asarray(z[f"{n}_L{L}"], dtype=np.float64)
                hn = np.linalg.norm(h, axis=1)   # ‖row‖ — истинная норма (fp16-точность)
                u = h / np.maximum(hn, 1e-12)[:, None]
                d[L] = (u, hn)
            toks = np.asarray(z[f"{n}_tok"]) if f"{n}_tok" in z.files else np.array([], dtype=np.int64)
            cm = m.get(n, {}).get("curves", {})
            docs[n] = {"layers": d, "toks": toks,
                       "dom": "own" if prefix == "own_w" else "corpus",
                       "gen_sha": None, "sha_ok": None,
                       "source": "P1_own" if prefix == "own_w" else "P2_corpus",
                       "curves_meta": {L: cm.get(str(L), {}) for L in LAYERS},
                       "toks_present": bool(len(toks))}
    return docs


def decode_probe(tok, tok_name):
    """G0-a декод-проба: 3 строки по 40 токенов глазами (ТЗ §0.1)."""
    from transformers import AutoTokenizer
    snap = sorted(os.listdir(os.path.join(MODEL_DIR, "snapshots")))
    t = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "snapshots", snap[-1]))
    lines = []
    for i in range(3):
        seg = tok[i * 40:(i + 1) * 40]
        if len(seg) == 0:
            break
        lines.append({"chunk": f"tok[{40*i}:{40*(i+1)}]", "text": t.decode(seg, skip_special_tokens=False)})
    return {"doc": tok_name, "n_toks": int(len(tok)), "lines": lines}


def per_t_cos(x, Kmax):
    """mean_t и median_t cos(P̂_t, P̂_{t+k}) для k=1..Kmax (dip-статистики)."""
    n = x.shape[0]
    Kmax = min(Kmax, n - 1)
    means = np.empty(Kmax)
    meds = np.empty(Kmax)
    for k in range(1, Kmax + 1):
        c = rho_t(x, k)
        means[k - 1] = float(np.mean(c))
        meds[k - 1] = float(np.median(c))
    return means, meds


def boot_diffs(diffs, agg, seed=SEED_BOOT, b=B):
    """Парный бутстрап по докум: точка = agg(diffs), CI = percentile[2.5, 97.5]
    бутстрап-распределения agg (одна статистика — §0.4). agg: 'median' | 'mean'."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    f = np.median if agg == "median" else np.mean
    boots = np.array([f(diffs[rng.integers(0, n, n)]) for _ in range(b)])
    return float(f(diffs)), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def p_two_sided(n, pos, neg):
    """Двусторонний sign-тест: X = число плюсов ~ Bin(n, 0.5), p = 2·P(X ≤ min(pos,neg))."""
    k = min(pos, neg)
    return min(1.0, 2.0 * float(sum(comb(n, i) for i in range(0, k + 1))) / 2.0 ** n)


def _bad_names(o):
    """П6: в артефакте не должно быть полей Δ_med/S0 (тождества к медиане не применяются)."""
    bad = ("dmed", "delta_med", "s0", "S(0)", "Δ_med", "Δmed")   # «d_med» убрано: ловило pooled_median (rev.3)
    if isinstance(o, dict):
        return any(any(b in str(k).lower() for b in bad) or _bad_names(v) for k, v in o.items())
    if isinstance(o, (list, tuple)):
        return any(_bad_names(v) for v in o)
    return False


def compute_cells(docs):
    """2880 клеток (doc, L, obj, stat). Возвращает cells, ra, vcurves, l1_status."""
    objects = ("h_raw", "du_unit", "v_tangent")
    cells = {}
    ra = {}
    vcurves = {}
    l1_status = {}
    for nm, d in sorted(docs.items()):
        tpu = nm.startswith(("own_", "cor_"))
        if tpu:
            u1, hn1 = d["layers"][1]
            U1 = u1 / np.maximum(np.linalg.norm(u1, axis=1, keepdims=True), 1e-12)
            V1, _ = vpath(U1, hn1)
            vc1 = dcurve(V1, np.arange(1, min(K_MAX + 1, V1.shape[0])))
            meta_dip1 = d["curves_meta"].get(1, {}).get("dip")
            l1_dev = None if (len(vc1) == 0 or meta_dip1 is None) else float(abs(float(vc1[0]) - meta_dip1))
            l1_status[nm] = {"recomputed_dip1": float(vc1[0]) if len(vc1) else None,
                             "meta_dip1": meta_dip1, "dev": l1_dev,
                             "degraded": bool(l1_dev is not None and l1_dev > L1_DEGRADE_TOL)}
        for L in LAYERS:
            u, hn = d["layers"][L]
            U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
            h = U * hn[:, None]
            dh = h[1:] - h[:-1]
            du = U[1:] - U[:-1]
            V, dropped = vpath(U, hn)
            vc = dcurve(V, np.arange(1, min(K_MAX + 1, V.shape[0])))
            vcurves[(nm, L)] = vc

            # телескоп §1.7: Σ_t Δu_t ≡ u_T − u_0 — на ВСЕХ (doc, L) (rev.3, MINOR-6:
            # round 2 — только L23, 96 из 480)
            assert float(np.linalg.norm(du.sum(axis=0) - (U[-1] - U[0]))) < 1e-6, \
                f"§1.7 телескоп: {nm} L{L}"

            s = ra_split(U, hn)
            ra[(nm, L)] = {"closure_max": s["closure_max"], "du_u_cross_max": s["du_u_cross_max"],
                           "closure_scale": float(np.max(np.einsum("ij,ij->i", s["dh"], s["dh"])))}

            xs = {"h_raw": dh, "du_unit": du, "v_tangent": V}
            degraded_l1 = tpu and L == 1 and l1_status[nm]["degraded"]
            for obj in objects:
                x = xs[obj]
                n = x.shape[0]
                if degraded_l1:
                    # L1-TPU per-doc: fp16-шаг L1 ниже квантового шума → моментные величины
                    # absent (K=0); meta-кривая (dip_V,med) — только в median-клетке v_tangent.
                    cm = d["curves_meta"].get(1, {})
                    curv = {int(k): v for k, v in cm.get("curve", {}).items()}
                    prov = (f"fp16_l1_degraded[{nm}]: recomputed dip_V,med={l1_status[nm]['recomputed_dip1']:.4f} "
                            f"vs meta {l1_status[nm]['meta_dip1']:.4f} (dev {l1_status[nm]['dev']:.4f} > {L1_DEGRADE_TOL}); "
                            "L1-шаг ниже fp16-квантового шума; moment values absent")
                    for st in STATS:
                        rec = {
                            "space": obj, "stat": st, "agg": "per_doc",
                            "norm": "T", "K": 0, "window": "none", "N": None, "N_absent": True,
                            "rho": None, "rho_Tk": None, "rho_Tk_norm": "T-k",
                            "r": None, "sum_rho_full": None, "gram_resid": None, "f_dc": None,
                            "dip_mean_1": None, "dip_med_1": None,
                            "moment_stat_independent": True, "provenance": prov,
                        }
                        if obj == "v_tangent":
                            rec["tangent_frame"] = "u_t"
                            if st == "median" and curv:
                                rec["dip_med_1"] = [curv.get(k) for k in range(1, max(curv) + 1)]
                        cells[(nm, L, obj, st)] = rec
                    continue
                K = k_of(n)
                if K is None:
                    raise ValueError(f"vacuous: n={n} < 25 ({nm}, L{L}, {obj})")
                means, meds = per_t_cos(x, K)
                ac = acf_moments(x, K)
                fdc = f_dc(x)
                # r = N·f_DC — тождество (G3), проверка на каждой клетке
                assert abs(ac["r"] - n * fdc) <= 1e-9 * max(1.0, ac["r"]), \
                    f"G3: r != N·f_DC ({nm}, L{L}, {obj})"
                common = {
                    "space": obj, "agg": "per_doc",
                    "norm": "T", "K": int(K), "window": "none", "N": int(n),
                    "D": int(x.shape[1]),
                    "rho": ac["rho"].tolist(), "rho_Tk": ac["rho_Tk"].tolist(),
                    "rho_Tk_norm": "T-k",
                    "r": ac["r"], "sum_rho_full": ac["sum_rho_full"],
                    "gram_resid": ac["gram_resid"], "f_dc": fdc,
                    "moment_stat_independent": True,
                }
                if obj == "du_unit":
                    common["r_max"] = 4.0 / float(np.einsum("ij,ij->i", du, du).sum())
                if obj == "v_tangent":
                    common["tangent_frame"] = "u_t"
                    common["R_norm_over_sqrtN"] = float(np.linalg.norm(V.sum(axis=0)) / np.sqrt(n))
                    common["v_dropped"] = dropped
                for st in STATS:
                    rec = dict(common)
                    rec["stat"] = st
                    # MAJOR-5 (rev.3): dip-статистика — только в СВОЕЙ stat-клетке
                    rec["dip_mean_1"] = float(means[0]) if st == "mean" else None
                    rec["dip_med_1"] = float(meds[0]) if st == "median" else None
                    cells[(nm, L, obj, st)] = rec
    return cells, ra, vcurves, l1_status


def flags_and_anchor(docs, vcurves):
    """degen-флаги (TЗ §0.3 по L23) + loopy_ladder (LADDER_V2-канон) + якорь G1′."""
    for nm, d in docs.items():
        fl23, mcos23, mshare23 = degen(vcurves[(nm, 23)], d["toks"])
        d["degen_tz"] = bool(fl23)
        d["loopy_ladder"] = bool(mcos23 > 0.95) or nm in ("wiki_0", "wiki_4")
        d["mcos"] = mcos23
        d["mshare"] = mshare23 if d["toks_present"] else None
        d["share_unavailable"] = not d["toks_present"]

    # LADDER_V2-совместимый набор: gen-доки (без own/cor), nonwiki, nonloopy.
    # loopy: max_{k≥2} dip > 0.95 + исторические wiki_0/wiki_4 — в точности exp90_ladder_v2.py:50-51.
    anchor_docs = []
    for nm, d in sorted(docs.items()):
        if not nm.startswith(("own_", "cor_")) and d["dom"] != "wiki" and not d["loopy_ladder"]:
            anchor_docs.append((nm, float(vcurves[(nm, 23)][0])))
    anchor_val = float(np.median([v for _, v in anchor_docs])) if anchor_docs else float("nan")
    anchor_ok = abs(anchor_val - ANCHOR_LADDER) <= ANCHOR_TOL
    nong = sorted(nm for nm in docs if not docs[nm]["degen_tz"])
    nong_by_dom = {}
    for nm in nong:
        nong_by_dom.setdefault(docs[nm]["dom"], []).append(nm)
    return nong, nong_by_dom, anchor_val, anchor_ok, len(anchor_docs), anchor_docs


def gate_g0a_tpu(docs):
    """G0-a TPU: hn-поле — баг продюсера; L23 meta-vs-recomputed (rev.3, MINOR-7)."""
    z_own = np.load(OWN, allow_pickle=True)
    _hn0 = np.asarray(z_own["own_w0_hn"], dtype=np.float64)
    _col = np.linalg.norm(np.asarray(z_own["own_w0_L23"], dtype=np.float64)[:, :len(_hn0)], axis=0)
    # L23: recomputed dip_V,med vs meta curves.23.dip (оба TPU-источника)
    dev_l23 = {}
    for nm, d in sorted(docs.items()):
        if not nm.startswith(("own_", "cor_")):
            continue
        meta_dip23 = d["curves_meta"].get(23, {}).get("dip")
        u, hn = d["layers"][23]
        U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        V, _ = vpath(U, hn)
        vc = dcurve(V, np.arange(1, min(K_MAX + 1, V.shape[0])))
        if meta_dip23 is None or len(vc) == 0:
            continue
        dev_l23[nm] = {"meta_dip23": float(meta_dip23), "recomputed_dip23": float(vc[0]),
                       "abs_dev": float(abs(float(vc[0]) - float(meta_dip23)))}
    by_src = {}
    for nm, r in dev_l23.items():
        by_src.setdefault("P1_own" if nm.startswith("own_") else "P2_corpus", []).append(r["abs_dev"])
    return {
        "finding": "_hn в P1_own.npz — баг продюсера (experiments/kaggle_e64gpu/e64_gpu.py:125): "
                   "norms первых 384 СТОЛБЦОВ матрицы L23, а не строк; строки пишутся raw "
                   "(e64_gpu.py:218: vec[L].astype(float16), без масштабирования)",
        "evidence_own_w0": {
            "row_norm_median": float(np.median(docs["own_w0"]["layers"][23][1])),
            "hn_median": float(np.median(_hn0)),
            "hn_vs_column_norms_max_rel": float(np.max(np.abs(_hn0 / _col - 1.0))),
            "row_norm_vs_hn_corr": float(np.corrcoef(docs["own_w0"]["layers"][23][1], _hn0)[0, 1]),
        },
        "l23_meta_vs_recomputed": {
            "max_abs_dev_by_source": {k: float(max(v)) for k, v in by_src.items()},
            "n_docs": {k: len(v) for k, v in by_src.items()},
            "worst_doc": max(dev_l23, key=lambda k: dev_l23[k]["abs_dev"]) if dev_l23 else None,
            "note": "TPU-ряды vs продюсерский meta L23 (fp16-точность ~1e-3; dev ≤ 1e-3 ⇒ ряды годны)",
        },
        "conclusion": "строки TPU-npz — сырые fp16-состояния; ‖row‖ — истинная норма (fp16-точность ~1e-3); "
                      "моментные величины h_raw и R/A-split для TPU вычисляемы и вычислены",
    }


def _perm_stat(x, f, seed, blk=BLK_NULL):
    """r4: статистика одной block-blk перестановки ряда x (N,D): S_band = mean_{k=9..96}
    (rho_Tk(k) − f) и e3/e4 = rho_Tk(3,4) − f. f = f_dc(x) инвариантна к перестановке
    (среднее векторов) — не пересчитывается. Знаменатель rho_Tk(k) = Σ_{t<N−k}‖x_t‖² —
    как в acf_moments (field.py)."""
    n = x.shape[0]
    nb, rem = divmod(n, blk)
    rng = np.random.default_rng(seed)
    bp = rng.permutation(nb)
    idx = np.empty(n, dtype=np.int64)
    core = np.empty(nb * blk, dtype=np.int64)
    for j in range(blk):
        core[j::blk] = bp * blk + j
    idx[:nb * blk] = core
    if rem:   # хвост (n не кратно blk) остаётся в конце — blk=2: одна лишняя строка
        idx[nb * blk:] = np.arange(nb * blk, n)
    xp = x[idx]
    e = np.einsum("ij,ij->i", xp, xp)
    S = np.cumsum(e)
    e3 = float(np.einsum("ij,ij->i", xp[:n - 3], xp[3:]).sum()) / float(S[n - 4]) - f
    e4 = float(np.einsum("ij,ij->i", xp[:n - 4], xp[4:]).sum()) / float(S[n - 5]) - f
    K = min(K_MAX, n - 1)
    vals = []
    for k in range(9, K + 1):
        ck = float(np.einsum("ij,ij->i", xp[:n - k], xp[k:]).sum())
        vals.append(ck / float(S[n - 1 - k]))
    return float(np.mean(vals)) - f, e3, e4


def null_permutations(seqs, fdc, pool, objects):
    """r4 (CRIT-1/MAJOR-1): matched null по (док, объект): матрицы (n_docs, NNULL)
    S_band / e3 / e4 по block-2 перестановкам. Детерминировано: сид (s, i) фиксирован,
    пул отсортирован, einsum детерминирован."""
    R = {}
    for obj in objects:
        n_pool = len(pool)
        bands = np.empty((n_pool, NNULL))
        e3s = np.empty((n_pool, NNULL))
        e4s = np.empty((n_pool, NNULL))
        for i, nm in enumerate(pool):
            x = seqs[nm][obj]
            f = fdc[nm][obj]
            for s in range(NNULL):
                bands[i, s], e3s[i, s], e4s[i, s] = _perm_stat(x, f, (s + 1) * 1009 + i)
        R[obj] = {"bands": bands, "e3": e3s, "e4": e4s}
    R["_pool"] = list(pool)   # порядок строк матриц
    return R


def _null_sel(nulls, obj, key, pool):
    """Строки null-матрицы (порядок nulls['_pool']) для подпула pool."""
    sel = [i for i, nm in enumerate(nulls["_pool"]) if nm in set(pool)]
    assert len(sel) == len(pool), "null-подпул: несовпадение документов"
    return nulls[obj][key][sel]


def _p_perm_two_sided(null_vals, obs):
    """p = (1 + #{s: |null_s − E_null| ≥ |obs − E_null|})/(NNULL+1) — двухсторонний
    перестановочный p-value (коррекция +1: наблюдаемое входит в пространство выводов)."""
    e = float(np.mean(null_vals))
    n_ge = int(np.sum(np.abs(null_vals - e) >= abs(obs - e)))
    return (1 + n_ge) / (len(null_vals) + 1), e


def p4_plateau(cells, pool, objects, nulls):
    """П4: плато ρ̄(k≥9) == f_DC (TZ §1.5-4). Rev.4 (CRIT-1): вердикт — против СОБСТВЕННОЙ
    null статистики (block-2 перестановка, r5: 199 сидов) + закрытая форма −(1−f)/(N−1),
    не против нуля (null ρ̄(k)−f ≈ −1/N по модулю больше «значимых» оценок).
    r5 (D1): null — pair-block (чётный лаг-1 + DC), НЕ DC+MA(1); второй столбец —
    аналитический MA(1)-null, matched по лаг-0/лаг-1 (D1b). r5 (D2): σ/эффект —
    первичный канал, p_perm — вторичный (family-wise: NNULL=199 ⇒ min p = 0.005 ≤ α/6).
    Статистика: per-doc mean по полосе k=9..96 (медиана по лагам — чувствительность),
    точка = медиана по доку, CI — парный бутстрап медиан (§0.4). DELTA_MIN = 2e-4 до прогона:
    |est − null| < DELTA_MIN ⇒ not_testable (ниже разрешения); p_perm ≤ 0.05 ⇒ refuted;
    иначе confirmed (плато на уровне matched null)."""
    out = {}
    z_ma1 = {}
    for obj in objects:
        diffs, lags_med = [], []
        for nm in pool:
            c = cells[(nm, 23, obj, "median")]
            if c["K"] != K_MAX:
                raise ValueError(f"П4: {nm} {obj} K={c['K']} != {K_MAX} (пул обязан быть K=96)")
            rt = np.array(c["rho_Tk"])
            band = rt[8:K_MAX]                     # k=9..96 (T-k-нормировка, ρ̄)
            diffs.append(float(np.mean(band)) - c["f_dc"])
            lags_med.append(float(np.median(band)) - c["f_dc"])
        diffs = np.array(diffs)
        est, ci = boot_diffs(diffs, "median")
        # matched null (r4): pool-медиана S_band по сидам — по строкам П4-пула (подмножество
        # null-пула); закрытая форма — аналитика
        bn = np.median(_null_sel(nulls, obj, "bands", pool), axis=0)   # (NNULL,)
        p_perm, e_null = _p_perm_two_sided(bn, est)
        cf = float(np.median([-(1.0 - cells[(nm, 23, obj, "median")]["f_dc"])
                              / (cells[(nm, 23, obj, "median")]["N"] - 1) for nm in pool]))
        gap = est - e_null
        if abs(gap) < DELTA_MIN:
            verdict = ("not_testable: |est − null| < DELTA_MIN — неотличимо от pair-block null "
                       "(чётный лаг-1 + DC) при разрешении теста (против собственной null, r4; r5, D1)")
        elif p_perm <= ALPHA_P4:
            verdict = ("refuted: структура сверх pair-block null (чётный лаг-1 + DC) "
                       "(p_perm против pair-block null; MA(1)-null — второй столбец; "
                       "локализовать по k — AT0_3, T0.3)")
        else:
            verdict = ("confirmed: структура сверх pair-block null не обнаружена — плато на "
                       "уровне matched null (p_perm=" + f"{p_perm:.3f}" + ", выше разрешения)")
        # r5 (arch-review D1b): аналитический MA(1)-null, matched по лаг-0/лаг-1 — второй
        # столбец. Модель: per-dim гауссов MA(1), θ matched к наблюдаемому ρ(1) (invertible-
        # корень; |ρ1| > 0.5 → no_match). k ≥ 2: ρ_true = 0 ⇒ E[mean_band] = 0 (per-doc
        # статистика S = mean_band − f_dc ⇒ E = −f_dc). Первый порядок (dim-независимые,
        # Bartlett-поправка нулевая для k > p+1 = 2): SE = sqrt(Σ_{k=9..96} 1/((N−k)·D))/98.
        # Описательно: вердикт — по первичному каналу (σ/эффект, D2).
        z_docs = {}
        thetas = []
        for nm in pool:
            c = cells[(nm, 23, obj, "median")]
            rho1 = float(np.asarray(c["rho"])[0])
            if abs(rho1) > 0.5:
                z_docs[nm] = None
                continue
            n, D = c["N"], c["D"]
            rt = np.asarray(c["rho_Tk"])
            s_doc = float(np.mean(rt[8:K_MAX])) - c["f_dc"]
            th = 0.0 if rho1 == 0.0 else (1.0 - float(np.sqrt(1.0 - 4.0 * rho1 ** 2))) / (2.0 * rho1)
            se = float(np.sqrt(np.sum(1.0 / ((n - np.arange(9, K_MAX + 1)) * D))) / (K_MAX - 8))
            z_docs[nm] = {"z": float((s_doc + c["f_dc"]) / se), "theta": float(th)}
            thetas.append(float(th))
        matched = [v for v in z_docs.values() if v is not None]
        z_vals = np.array([v["z"] for v in matched])
        out_ma1 = {
            "median_z": float(np.median(z_vals)) if z_vals.size else None,
            "mean_z": float(np.mean(z_vals)) if z_vals.size else None,
            "n_matched": int(len(z_vals)), "n_no_match": int(len(pool) - len(z_vals)),
            "n_abs_z_ge_3": int(np.sum(np.abs(z_vals) >= 3.0)) if z_vals.size else 0,
            "theta_median_abs": float(np.median(np.abs(np.array(thetas)))) if thetas else None,
            "E_doc": "−f_dc (matched MA(1): ρ_true(k≥2) = 0 ⇒ E[mean_band] = 0)",
            "SE_formula": "sqrt(Σ_{k=9..96} 1/((N−k)·D)) / 98 — первый порядок, per-dim гауссов "
                          "MA(1), D = скрытое измерение; Bartlett-поправка нулевая для k > p+1",
            "note": "r5 (arch-review D1b): второй столбец — MA(1)-null, реально сохраняющий "
                    "лаг-1 (θ matched к ρ(1), invertible-корень); описательно, вердикт — по "
                    "первичному каналу (σ/эффект, D2)",
        }
        z_ma1[obj] = z_docs
        out[obj] = {
            "n_docs": len(diffs),
            "est_median": est, "ci_median": ci, "ci_width": ci[1] - ci[0],
            "null_block2": {
                "E": float(np.mean(bn)), "sd": float(np.std(bn)),
                "min": float(np.min(bn)), "max": float(np.max(bn)),
                "p_perm": p_perm,
            },
            "null_analytic_pool_median": cf,
            "null_analytic_ma1": out_ma1,
            "null_note": "closed form (full permutation точна; blk=2 при k≥2 — те же пары различных "
                         "элементов): E[ρ̄(k) − f] = −(1−f)/(N−1) ≈ −1/N (N=383: −0.0026). r5 (D1): "
                         "block-2 — pair-block null: сохраняет ЧЁТНЫЙ лаг-1 (x_{2i}, x_{2i+1}), "
                         "рандомизирует НЕЧЁТНЫЙ лаг-1 и лаги ≥ 2; «DC+MA(1) null» было ложным "
                         "label (ревью: MA(1)-лаг-1 вдвое). MA(1)-null, matched по лаг-0/лаг-1 — "
                         "null_analytic_ma1 (второй столбец)",
            "delta_min": DELTA_MIN, "alpha": ALPHA_P4,
            "prereg": "полоса k=9..96 (буква ТЗ §1.5-4 «плато ρ̄(k≥9)»), агрегатор = per-doc mean "
                      "по лагам, DELTA_MIN = 2e-4 (≈ sd null / пол-ширина CI, замер 1e-4…6e-4), "
                      "вердикт против matched null — всё зафиксировано ДО прогона (r4, MAJOR-2); "
                      "r5 (D2): σ/эффект — первичный канал, p_perm — вторичный (family-wise, "
                      "NNULL=199 ⇒ min p = 0.005 ≤ α/6 = 0.00833)",
            "truncation": "полоса ограничена k_alphabet (K=96), не N (N = 82…383): «полного» "
                          "плато при k>96 в артефакте нет — обрезка объявлена явно (r4, MAJOR-2)",
            "sensitivity_median_over_lags": {
                "est": float(np.median(np.array(lags_med))),
                "note": "медиана по лагам (round-2 статистика) — НЕ вердикт: нуль этой статистики "
                        "не ноль (тождество G1 Σρ̂=(r−1)/2≈−1/2; ревью round 2, CRIT-1)",
            },
            "pool": "nondegen ∩ {K=96, все объекты} (одинаковая полоса k=9..96)",
            "estimator": "per-doc: mean_{k=9..96} rho_Tk(k) − f_dc; точка = медиана по доку; "
                         "CI — парный бутстрап медиан (seed 117, B=1000)",
            "verdict": verdict,
        }
    return out, z_ma1


def _sigma_of(obs, med_null):
    """σ = (obs − E_null)/sd_null; null без дисперсии — баг машины, fail-loud."""
    sd = float(med_null.std())
    assert sd > 0, "null-медиана не диспергирует (sd=0) — баг null-машины"
    return float((obs - float(med_null.mean())) / sd)


def p5_hump(cells, pool, objects, nulls):
    """П5: горб excess ρ̄(3,4) − f_DC ≠ 0 (ТЗ §1.5-5). Rev.4 (MAJOR-1): sign-тест —
    против matched null (pair-block, r5: 199 сидов), НЕ против Binomial(n, 0.5): null e3 ≈ −1/N,
    P(e3>0) ≈ 0.2, не 0.5. Два канала: (a) sign-счёт ē против null-распределения счёта;
    (b) уровень: pool-медиана ē против null-медианы (σ). r5 (D3): level-канал ТОЛЬКО по ē
    (|σ(ē)| ≥ 3); e3/e4 — описательно (было max по 3 статистикам — та же инфляция, что sign-OR;
    вердикты не меняются: σ_ee = −10.2/−10.9/+14.1). Confirmed — если любой канал значим
    (α = 0.05 / |σ| ≥ 3). Один комбинированный тест (rev.3, MINOR-12)."""
    out = {}
    for obj in objects:
        es, e3s, e4s = [], [], []
        for nm in pool:
            c = cells[(nm, 23, obj, "median")]
            n = c["N"]
            if n is None or n <= 32 or c["f_dc"] is None or c["rho_Tk"] is None:
                continue   # r4, MINOR-2: guard N > 32, как у П4 (инертен: мин N пула = 82)
            e3 = c["rho_Tk"][2] - c["f_dc"]
            e4 = c["rho_Tk"][3] - c["f_dc"]
            es.append(0.5 * (e3 + e4))
            e3s.append(e3)
            e4s.append(e4)
        es = np.array(es)
        e3s = np.array(e3s)
        e4s = np.array(e4s)
        pos = int((es > 0).sum())
        neg = int((es < 0).sum())
        p_binom = p_two_sided(len(es), pos, neg)
        # matched null (r4): count-канал и level-канал (строки null-матрицы для П5-пула)
        e3_null = _null_sel(nulls, obj, "e3", pool)
        e4_null = _null_sel(nulls, obj, "e4", pool)
        cnt_null = np.sum(e3_null > 0, axis=0)        # (NNULL,) — счёт плюсов e3 по доку
        med_null3 = np.median(e3_null, axis=0)       # (NNULL,) — pool-медиана e3
        med_null4 = np.median(e4_null, axis=0)
        ee_null = 0.5 * (e3_null + e4_null)
        cnt_null_ee = np.sum(ee_null > 0, axis=0)
        med_null_ee = np.median(ee_null, axis=0)
        p_sign_ee = (1 + int(np.sum(np.abs(cnt_null_ee - cnt_null_ee.mean())
                                    >= abs(pos - cnt_null_ee.mean())))) / (NNULL + 1)
        p_sign_e3 = (1 + int(np.sum(np.abs(cnt_null - cnt_null.mean())
                                    >= abs(int((e3s > 0).sum()) - cnt_null.mean())))) / (NNULL + 1)
        sigma = {
            k: _sigma_of(v, mn) for k, v, mn in (
                ("e3", float(np.median(e3s)), med_null3),
                ("e4", float(np.median(e4s)), med_null4),
                ("ee", float(np.median(es)), med_null_ee))
        }
        sig_sign = p_sign_ee <= ALPHA_P5
        sig_level = abs(sigma["ee"]) >= 3.0   # r5 (D3): level ТОЛЬКО по ē; e3/e4 — описательно
        if sig_sign or sig_level:
            sign = ("positive (excess > 0 — горб)" if np.median(es) > 0
                    else "negative (excess < 0 — анти-горб)")
            verdict = (f"confirmed: excess(3,4) ≠ 0, знак {sign} (каналы: sign={'да' if sig_sign else 'нет'}"
                       f", level(ē)={'да' if sig_level else 'нет'})")
        else:
            verdict = "not_resolved: excess(3,4) == 0 в пределах теста (оба канала)"
        out[obj] = {
            "n": len(es), "pos": pos, "neg": neg,
            "e3_median": float(np.median(e3s)), "e4_median": float(np.median(e4s)),
            "ee_median": float(np.median(es)),
            "p_two_sided_binom05": p_binom,
            "p_two_sided_binom05_note": "Binomial(n, 0.5) — неверный null (null e3 ≈ −1/N, P(+)≈0.2); "
                                        "оставлено для сравнения, вердикт — по matched null (r4)",
            "null_block2": {
                "pos_frac_e3": [float(cnt_null.mean() / len(pool)),
                                float(cnt_null.std() / len(pool))],
                "pos_frac_ee": [float(cnt_null_ee.mean() / len(pool)),
                                float(cnt_null_ee.std() / len(pool))],
                "p_perm_sign_ee": p_sign_ee,
                "p_perm_sign_e3": p_sign_e3,
                "sigma": {"ee": sigma["ee"]},
                "sigma_descriptive": {"e3": sigma["e3"], "e4": sigma["e4"]},
                "med_null_e3": [float(med_null3.mean()), float(med_null3.std())],
                "med_null_e4": [float(med_null4.mean()), float(med_null4.std())],
                "med_null_ee": [float(med_null_ee.mean()), float(med_null_ee.std())],
                "level_channel_note": "r5 (arch-review D3): level-канал — ТОЛЬКО по ē "
                                      "(было max(|σ(e3)|,|σ(e4)|,|σ(ē)|) — та же инфляция, что "
                                      "sign-OR); e3/e4 — описательно (sigma_descriptive)",
                "pair_block_note": "r5 (arch-review D1): block-2 — pair-block null (чётный "
                                   "лаг-1 + DC), НЕ DC+MA(1) (нечётный лаг-1 тоже рандомизируется; "
                                   "ревью измерило: MA(1)-лаг-1 вдвое)",
            },
            "alpha": ALPHA_P5,
            "statistic": "per-doc ē = mean(ρ̄(3)−f_dc, ρ̄(4)−f_dc); r4: sign-счёт и уровень — "
                         "против matched null (pair-block перестановка; r5: 199 сидов)",
            "verdict": verdict,
        }
    return out


def frame_sensitivity(docs, cells, nong):
    """§1.7 (rev.3, MINOR-6): рамочная чувствительность — θ_t (u_t, u_{t+1}),
    ‖Δu_t‖ vs 2·sin(θ/2) (тождество для unit), dip_V,med − dip_du,med per doc (L23)."""
    rows = []
    for nm in nong:
        d = docs[nm]
        u, hn = d["layers"][23]
        U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        c = np.clip((U[:-1] * U[1:]).sum(axis=1), -1.0, 1.0)
        theta = np.arccos(c)
        du = U[1:] - U[:-1]
        dun = np.linalg.norm(du, axis=1)
        # тождество хорды: ‖Δu‖ = 2·sin(θ/2) точно для unit-векторов
        assert float(np.max(np.abs(dun - 2.0 * np.sin(theta / 2.0)))) < 1e-12, f"chord: {nm}"
        dv = cells[(nm, 23, "v_tangent", "median")]["dip_med_1"]
        dd = cells[(nm, 23, "du_unit", "median")]["dip_med_1"]
        if dv is None or dd is None:
            continue
        rows.append({"doc": nm, "med_theta_deg": float(np.degrees(np.median(theta))),
                     "med_du_norm": float(np.median(dun)),
                     "dip_V_minus_dip_du": float(dv - dd)})
    if not rows:
        return {"note": "нет доков"}
    return {
        "n_docs": len(rows),
        "pooled_median": {
            "med_theta_deg": float(np.median([r["med_theta_deg"] for r in rows])),
            "med_du_norm": float(np.median([r["med_du_norm"] for r in rows])),
            "dip_V_minus_dip_du": float(np.median([r["dip_V_minus_dip_du"] for r in rows])),
        },
        "chord_identity_max_dev": "< 1e-12 (asserted per doc)",
        "note": "ТЗ §1.7: «расхождение v_t vs Δu ≈ sin(θ/2) — отчитать чувствительность дипа "
                "к рамке» — отчёт, не тест (вердикта в ТЗ нет)",
    }


def srcs_of(nms, docs):
    s = {}
    for nm in nms:
        s[docs[nm]["source"]] = s.get(docs[nm]["source"], 0) + 1
    return s


def t0_headline(cells, nong, objects):
    """r5 (arch-review D5): заголовочный вопрос T0 — компенсируется ли отрицательная площадь
    лага-1 положительной при k≥2 (Σρ̄ ≈ 0?). Отчёт per object × {T, T−k} × {none, bartlett}:
    excess = Σ(ρ̂(k) − f_dc). T-none-full — ПОЛНАЯ сумма по всем лагам (точно: sum_rho_full =
    (r−1)/2, тождество G1/Gram); T-k и bartlett — частичные по k=1..K (K=96, объявлено).
    D6: тождество S(0) = 1 + 2Σρ̄ выполняется только для unit-variance нормировки ρ̂; для
    энерго-взвешенной (norm=T: c(k)/c(0)) замыкается Gram-тождество 1 + 2Σρ̂ = r (G8), не
    unit-ACF; для T-k unit-variance-тождество не выполняется вовсе (знаменатель переменный).
    Per-doc значения выводимы из store (cells: sum_rho_full, f_dc, rho, rho_Tk, N, r)."""
    out = {}
    for obj in objects:
        recs = []
        for nm in nong:
            c = cells[(nm, 23, obj, "median")]
            if c.get("sum_rho_full") is None or c.get("f_dc") is None or c.get("N") is None:
                continue
            N, K, fdc = c["N"], c["K"], c["f_dc"]
            rho = np.asarray(c["rho"], dtype=float)
            rho_Tk = np.asarray(c["rho_Tk"], dtype=float)
            w = 1.0 - np.arange(1, K + 1, dtype=float) / (K + 1)   # Bartlett по k=1..K
            recs.append({
                "T_none_full": c["sum_rho_full"] - (N - 1) * fdc,
                "T_none_K96": float(rho.sum()) - K * fdc,
                "Tk_none_K96": float(rho_Tk.sum()) - K * fdc,
                "T_bart_K96": float((w * rho).sum()) - float(w.sum()) * fdc,
                "Tk_bart_K96": float((w * rho_Tk).sum()) - float(w.sum()) * fdc,
                "r": c["r"], "f_dc": fdc,
            })
        if not recs:
            out[obj] = {"n_docs": 0}
            continue
        keys = ("T_none_full", "T_none_K96", "Tk_none_K96", "T_bart_K96", "Tk_bart_K96")
        out[obj] = {
            "n_docs": len(recs),
            "excess_median": {k: float(np.median([r[k] for r in recs])) for k in keys},
            "excess_mean": {k: float(np.mean([r[k] for r in recs])) for k in keys},
            "r_median": float(np.median([r["r"] for r in recs])),
            "f_dc_median": float(np.median([r["f_dc"] for r in recs])),
        }
    return out


def aggregates(docs, cells, nong, nong_by_dom, objects):
    """Агрегаты L23: доменные строки (n≥12), source-строки (r5, D4: +gen + provenance_limited
    у corpus), source-компаньон (r5, D4), pooled, t0_headline (r5, D5/D6). Одна схема
    (rev.3, MINOR-9). dip_mean_1 — из mean-клетки, dip_med_1 — из median-клетки (MAJOR-5)."""
    K_NOTE = "K = max {24,32,96} ≤ N−1 per doc (реестр §0.2)"

    def row(nms, allowed, extra=None):
        def med(obj, field, st):
            vals = [cells[(nm, 23, obj, st)][field] for nm in nms
                    if (nm, 23, obj, st) in cells and cells[(nm, 23, obj, st)][field] is not None]
            if not vals:
                return None
            return float(np.median(vals))

        r = {
            "L": 23, "n_docs": len(nms), "sources": srcs_of(nms, docs),
            "row_allowed": allowed, "window": "none", "K_note": K_NOTE,
            "dip_V_med": med("v_tangent", "dip_med_1", "median") if allowed else None,
            "dip_V_mean": med("v_tangent", "dip_mean_1", "mean") if allowed else None,
            "r_v": med("v_tangent", "r", "median") if allowed else None,
            "r_du": med("du_unit", "r", "median") if allowed else None,
            "f_dc_du": med("du_unit", "f_dc", "median") if allowed else None,
            "rmax_du": med("du_unit", "r_max", "median") if allowed else None,
        }
        if extra:
            r.update(extra)
        return r

    agg = {}
    for dm in ("wiki", "math", "code", "chat"):
        nms = nong_by_dom.get(dm, [])
        agg[dm] = row(nms, len(nms) >= 12)   # §0.3: доменное утверждение только при n ≥ 12
    # r5 (arch-review D4): gen-строка (у gen-доков dom — реальный домен, не источник) +
    # corpus-строка — provenance_limited (16/16 доков без токенов, G0a)
    gen_nms = [nm for nm in nong if docs[nm]["source"] == "gen_layers"]
    agg["source_gen_layers"] = row(gen_nms, len(gen_nms) >= 12)
    for src, tag, extra in (
            ("own", "P1_own", None),
            ("corpus", "P2_corpus", {
                "provenance_limited": True, "domain_like": False,
                "note": "r5 (arch-review D4): 16/16 доков без токенов (G0a corpus_toks absent) — "
                        "«те ли токены» невыполнимо в принципе; строка НЕ домен-подобная; "
                        "корпус-зависимые числа — provenance_limited до восстановления токенов"})):
        nms = nong_by_dom.get(src, [])
        agg[f"source_{tag}"] = row(nms, len(nms) >= 12, extra=extra)
    vals = [cells[(nm, 23, "h_raw", "mean")]["dip_mean_1"] for nm in nong
            if cells[(nm, 23, "h_raw", "mean")]["dip_mean_1"] is not None]
    pooled = row(nong, True, extra={
        "filter_note": "degen-фильтр (§0.3) — НЕ loopy-фильтр якоря G1′ (тот: nonwiki nonloopy); "
                       "различие фильтров — в имени/полях",
        "anchor_comparability": "r5 (arch-review D4): pooled dip_V_med НЕ сопоставим с LADDER-якорем "
                                "−0.0722 (фильтр degen_tz n=53 vs nonwiki-nonloopy n=35; 16 из 53 — "
                                "корпус без токенов — число частично corpus-обусловлено)",
        "dip_dh_mean": float(np.median(vals)) if vals else None,
        "dip_dh_mean_tz_name": "dip_raw,mean (ТЗ §0.2)",
    })
    agg["pooled_nondeg"] = pooled
    # r5 (arch-review D4): source-компаньон-таблица (gen-only / +corpus / +own / все)
    comp = {}
    for tag, nms in (
            ("gen_only", [nm for nm in nong if docs[nm]["source"] == "gen_layers"]),
            ("gen_plus_corpus", [nm for nm in nong
                                 if docs[nm]["source"] in ("gen_layers", "P2_corpus")]),
            ("gen_plus_own", [nm for nm in nong
                              if docs[nm]["source"] in ("gen_layers", "P1_own")]),
            ("all_nondeg", list(nong))):
        comp[tag] = row(nms, True)
    agg["source_companion"] = comp
    agg["source_companion_note"] = "r5 (arch-review D4): компаньон-таблица по источникам " \
        "(gen-only / +corpus / +own / все nondegen) — для отсечения corpus-условленности"
    # r5 (arch-review D5/D6): заголовочный вопрос T0 (Σρ̄) — агрегирован, без перепрогона
    agg["t0_headline"] = t0_headline(cells, nong, objects)
    agg["t0_headline_note"] = ("r5 (arch-review D5/D6): excess = Σ(ρ̂(k) − f_dc) per object × norm × "
        "window; T_none_full — полная сумма по всем лагам (точно, тождество G1/Gram: "
        "sum_rho_full = (r−1)/2); T_none_K96 / Tk_none_K96 / T_bart_K96 / Tk_bart_K96 — "
        "частичные по k=1..96 (объявлено). Тождество S(0) = 1 + 2Σρ̄ — только для unit-variance "
        "нормировки; для norm=T замыкается Gram-тождество 1 + 2Σρ̂ = r (G8), не unit-ACF; для "
        "T-k unit-variance-тождество не выполняется вовсе (D6). Per-doc — из store (cells)")
    return agg


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STORE_DIR, exist_ok=True)
    if os.path.exists(T25_SRC):
        shutil.copy(T25_SRC, os.path.join(OUT_DIR, "t25_ekurve.json"))
        t25_sha = sha256_file(os.path.join(OUT_DIR, "t25_ekurve.json"))
    else:
        t25_sha = None

    docs = {}
    docs.update(load_source(GEN, "gen"))
    docs.update(load_source(OWN, "state"))
    docs.update(load_source(COR, "state"))

    # G2 (предзапись): жёсткая разбивка источников
    src_counts = {}
    for d in docs.values():
        src_counts[d["source"]] = src_counts.get(d["source"], 0) + 1
    assert src_counts.get("gen_layers") == 64, f"G2: gen_layers != 64 ({src_counts})"
    assert src_counts.get("P1_own") == 16, f"G2: P1_own != 16 ({src_counts})"
    assert src_counts.get("P2_corpus") == 16, f"G2: P2_corpus != 16 ({src_counts})"

    objects = ("h_raw", "du_unit", "v_tangent")
    cells, ra, vcurves, l1_status = compute_cells(docs)
    nong, nong_by_dom, anchor_val, anchor_ok, n_anchor, anchor_docs = flags_and_anchor(docs, vcurves)

    # пулы пре-регов (зафиксированы): П4 — K=96 во всех объектах (одинаковая полоса k=9..96);
    # П5 — все nondegen (k=3,4 доступны при любом K ≥ 5)
    pool4 = [nm for nm in nong
             if all(cells[(nm, 23, obj, "median")]["K"] == K_MAX for obj in objects)]
    pool5 = list(nong)

    # r4 (CRIT-1/MAJOR-1): matched null П4/П5 — block-2-перестановка рядов приращений (L23)
    seqs_pool = {}
    for nm in pool5:
        u, hn = docs[nm]["layers"][23]
        U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        h = U * hn[:, None]
        V, _ = vpath(U, hn)
        seqs_pool[nm] = {"h_raw": h[1:] - h[:-1], "du_unit": U[1:] - U[:-1], "v_tangent": V}
    fdc_pool = {nm: {obj: cells[(nm, 23, obj, "median")]["f_dc"] for obj in objects} for nm in pool5}
    logline(f"null (r4): block-{BLK_NULL}-перестановка приращений, {NNULL} сидов × {len(pool5)} доков × 3 объекта")
    nulls = null_permutations(seqs_pool, fdc_pool, pool5, objects)

    # ---------------- G-гейты ----------------
    gates = {}
    gates["G1prime_field_sha256"] = sha256_file(os.path.join(ROOT, "prism/field.py"))
    gates["G1prime_fn_sha256"] = {
        fn: hashlib.sha256(inspect.getsource(getattr(pf, fn)).encode()).hexdigest()
        for fn in ("vpath", "dcurve", "acf_moments", "f_dc", "ra_split", "degen")}
    gates["G1prime_anchor_LADDER_V2"] = {
        "expected": ANCHOR_LADDER, "computed": anchor_val, "n_docs": n_anchor,
        "docs": [nm for nm, _ in anchor_docs], "ok": anchor_ok}
    gates["G0a_provenance"] = {
        "deviation": "pool_fp = sha256(тексты) недоступен (тексты не в npz); замена: sha256(npz) + "
                     "sha256(tok-массивов, внутри npz) + декод-проба 3 строк/источник — отклонение зафиксировано",
        "tokenizer_id": "Qwen/Qwen3.5-0.8B (local .hf_cache/hub/models--Qwen--Qwen3.5-0.8B)",
        "gen_layers_sha256": sha256_file(GEN),
        "P1_own_sha256": sha256_file(OWN), "P2_corpus_sha256": sha256_file(COR),
        "decode_probe": {
            "gen_wiki_3": decode_probe(docs["wiki_3"]["toks"], "wiki_3"),
            "own_w0": decode_probe(docs["own_w0"]["toks"], "own_w0"),
            "cor_w0": "unavailable: P2_corpus.npz не содержит tok (0/16 доков) — проверено по файлу",
        },
        "corpus_toks": "absent (0/16) — «те ли токены» для P2_corpus невыполнимо в принципе (зафиксировано)",
    }
    gates["G0a_tpu_hn_field"] = gate_g0a_tpu(docs)
    gates["G0b_gen_sha"] = {
        "gen_layers_sha_ok": sum(1 for d in docs.values() if d["source"] == "gen_layers"
                                 and d.get("gen_sha") and d.get("sha_ok")),
        "total_gen": 64,
        "note": "TPU-источники: per-doc gen_sha отсутствует в метаданных (artifact без sha)",
    }
    assert gates["G0b_gen_sha"]["gen_layers_sha_ok"] == 64, "G0-b: gen_sha < 64/64"
    expected = {(nm, L, obj, st) for nm in docs for L in LAYERS for obj in objects for st in STATS}
    gates["G2_completeness"] = {"expected": len(expected), "computed": len(cells),
                                "sources": src_counts,
                                "missing": sorted(str(k) for k in expected - set(cells))[:10]}
    gates["G2_ok"] = len(cells) == len(expected) and not gates["G2_completeness"]["missing"]
    # G3: реестр (§0.2): поля + алфавиты stat/K + dip-клеточность (MAJOR-5) + r = N·f_DC (в цикле)
    g3_bad = []
    for k, c in cells.items():
        req = {"space", "stat", "agg", "norm", "K", "window", "N", "dip_mean_1", "dip_med_1",
               "r", "f_dc", "gram_resid", "rho", "rho_Tk", "rho_Tk_norm"}
        if not req <= set(c.keys()):
            g3_bad.append((str(k), sorted(req - set(c.keys()))))
            continue
        if c["K"] not in (0, 24, 32, 96):
            g3_bad.append((str(k), f"K={c['K']} вне алфавита {{0,24,32,96}}"))
        if c["norm"] != "T" or c["window"] != "none":
            g3_bad.append((str(k), f"norm/window = {c['norm']}/{c['window']} вне алфавита (T/none)"))
        if c["stat"] not in STATS:
            g3_bad.append((str(k), "stat-alphabet"))
        if c["space"] == "v_tangent" and c.get("tangent_frame") != "u_t":
            g3_bad.append((str(k), "tangent_frame"))
        if c["stat"] == "mean" and c["dip_med_1"] is not None:
            g3_bad.append((str(k), "dip_med_1 в mean-клетке"))
        if c["stat"] == "median" and c["dip_mean_1"] is not None:
            g3_bad.append((str(k), "dip_mean_1 в median-клетке"))
    gates["G3_registry"] = {"ok": not g3_bad, "bad": g3_bad[:5],
                            "r_equals_N_fdc": "asserted per cell in loop (1e-9 rel)",
                            "note": "клетка norm=T (идентичности G3/G8 — T-норм ρ̂); поле rho_Tk — "
                                    "T-k-нормировка (ρ̄, §1.4-1) — зафиксировано полем rho_Tk_norm; "
                                    "K/norm/window гейтятся (r4, MINOR-4); "
                                    "K=0 — moment absent (L1-fp16-degraded, provenance в клетке)"}
    # G8: и по клеткам, и по независимым (doc,L,obj) (rev.3, MINOR-10: моментный блок shared
    # по stat-оси — 2868 клеток = 2×1434 независимых)
    g8_cells = {k: v for k, v in cells.items() if v["gram_resid"] is not None}
    g8_bad = {str(k): v["gram_resid"] for k, v in g8_cells.items() if v["gram_resid"] >= 1e-10}
    gates["G8_gram"] = {"max_resid": max(v["gram_resid"] for v in g8_cells.values()),
                        "fail": len(g8_bad),
                        "cells": len(g8_cells),
                        "independent_doc_layer_obj": len({(k[0], k[1], k[2]) for k in g8_cells}),
                        "excluded_L1_tpu_docs": [nm for nm, s in sorted(l1_status.items()) if s["degraded"]],
                        "path_note": "gram_resid — метрика пути суммирования (порядок лагов/BLAS): "
                                     "интерпретировать только по абсолютной величине (все ≪ 1e-15); "
                                     "rel-сравнение между путями бессмысленно; закрытая форма "
                                     "Σ_{k≥1}c(k) = (‖Σx‖² − Σ‖x‖²)/2 точна (r2-code M3 / r2-numbers MINOR-7)"}
    gates["G4_filter_symmetry"] = "n/a (нет контрольной стороны в T0.1; null'ы — в AT0_2); " \
                                  "П4/П5: одна общая выборка доков на всех объектах (поля pool в prereg)"
    gates["G5_amplitude"] = "n/a (нет возмущений)"
    gates["G9_arch_null"] = "n/a (нет JVP/патч-машины)"
    # G6 sanity (R/A closure; все 96 доков — строки TPU raw, см. G0a_tpu_hn_field).
    # Идентичность точная (float128-резид ~2e-18, проверено ревью r2); fp64-шум масштаба
    # ‖Δh‖² (~3.6e4 на L23) → гейт по ОТНОСИТЕЛЬНОМУ closure (отклонение от абсолютного 1e-12
    # буквы ТЗ зафиксировано; абсолют в артефакте).
    ra_vals = list(ra.values())
    g6_rel = max(v["closure_max"] / v["closure_scale"] for v in ra_vals)
    g6_max = max(v["closure_max"] for v in ra_vals)
    g6_cross = max(v["du_u_cross_max"] for v in ra_vals)
    gates["G6_ra_sanity"] = {"closure_rel_max": g6_rel, "closure_abs_max": g6_max,
                             "du_u_cross_max": g6_cross, "n_cells": len(ra_vals),
                             "deviation": "relative closure (scale = max ‖Δh‖²); абсолют ТЗ 1e-12 "
                                          "недостижим в fp64 при ‖Δh‖²~3.6e4; float128-верификация в ревью r2",
                             "ok": g6_rel < 1e-12 and g6_cross < 1e-14}
    # G10 (rev.3, MAJOR-2): без захардкоженного ok-самоаттестата
    gates["G10"] = {
        "outcomes": [
            "A: большинство доков r_max >= 1 (frac_rmax_below_1 ≤ 0.5) ⇒ T0 = учётный гейт, не арбитр (T1-T3 — с оговоркой)",
            "B: большинство доков r_max < 1 (frac_rmax_below_1 > 0.5) ⇒ T0 = арбитр, T1-T3 идут с оговоркой",
            "C: |est − E_null| ≥ DELTA_MIN и p_perm > α ⇒ структуры сверх pair-block null не обнаружено (confirmed)",
            "D: p_perm ≤ α ⇒ структура сверх pair-block null, локализуется по k (AT0_3, T0.3)",
        ],
        "reachability": "A: НЕДОСТИЖИМ на текущих данных — r_max = 4/Σ‖Δu‖² ≤ 4/19.3 ≈ 0.208 < 1 "
                        "(замер Σ‖Δu‖² = 19.3…518.3 по 478 du_unit-клеткам) ⇒ реализован исход B "
                        "(frac = 1.0); B/C/D — достижимы данными (r4, r2-numbers MAJOR-4: round-2 "
                        "«все исходы достижимы» ложно для A; assert ≥2 исходов проходит через B/C/D)",
        "note": "round 2: 'ok: True' + self-assert и post-hoc DELTA_MIN := |оценка| (исход C "
                "недостижим) — удалены; DELTA_MIN зафиксирован ДО прогона (§0.4)",
    }
    assert len(gates["G10"]["outcomes"]) >= 2

    # ---------------- §1.7 ----------------
    # перестановочная инвариантность (5 сидов): probe — вырожденный док (wiki_3: degen_tz)
    probe = ("wiki_3", 23, "h_raw")
    u, hn = docs[probe[0]]["layers"][probe[1]]
    U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    h = U * hn[:, None]
    x0 = h[1:] - h[:-1]
    sums = [acf_moments(x0, min(K_MAX, x0.shape[0] - 1))["sum_rho_full"]]
    for sd in (1, 2, 3, 4, 5):
        p = np.random.default_rng(sd).permutation(x0.shape[0])
        sums.append(acf_moments(x0[p], min(K_MAX, x0.shape[0] - 1))["sum_rho_full"])
    perm_spread = float(np.max(sums) - np.min(sums))
    try:
        pf._selfcheck()
        selfcheck = "ok (G1/G3/G7/G8/G6/G2 на синтетике — prism/field.py)"
    except AssertionError as e:
        selfcheck = f"FAIL: {e}"
    gates["sec17"] = {
        "telescope_all_doc_L": "asserted per (doc, L) in compute_cells (rev.3: round 2 — только L23)",
        "field_selfcheck": selfcheck,
        "permutation_invariance": {"probe": str(probe),
                                   "probe_note": "probe — degen-док (безразлично для Σ-инвариантности)",
                                   "sums": sums, "spread": perm_spread, "ok": perm_spread < 1e-12},
        "frame_sensitivity": frame_sensitivity(docs, cells, nong),
        "r_equals_N_fdc": "asserted per cell (G3)",
    }

    # ---------------- пре-рег ----------------
    prereg = {}
    # П1: gram_resid
    prereg["1_gram"] = {"max_resid": gates["G8_gram"]["max_resid"], "ok": not g8_bad,
                        "verdict": "gate_pass" if not g8_bad else "gate_fail",
                        "note": "гейт по ТЗ («гейт G8, не гипотеза») — словарь §0.4 к гейтам неприменим"}
    # П2: r <= r_max (du_unit, теорема: ‖ΣΔu‖ = ‖u_T − u_0‖ ≤ 2)
    rmax_bad = []
    n_checked = 0
    for nm in docs:
        for L in LAYERS:
            c = cells[(nm, L, "du_unit", "median")]
            if c.get("r_max") is None or c.get("r") is None:
                continue
            n_checked += 1
            if c["r"] > c["r_max"] + 1e-9:
                rmax_bad.append((nm, L, c["r"], c["r_max"]))
    prereg["2_r_le_rmax"] = {"n_checked": n_checked, "bad": len(rmax_bad),
                             "verdict": "confirmed" if not rmax_bad else "refuted (теорема сломана — баг)",
                             "examples": rmax_bad[:5]}
    # П3: r_max статус (per-doc; формула зафиксирована; индикатор в CI не входит)
    fracs = {nm: cells[(nm, 23, "du_unit", "median")]["r_max"] for nm in nong}
    below = np.array([v < 1.0 for v in fracs.values()], dtype=float)
    frac_below = float(below.mean())
    rng = np.random.default_rng(SEED_BOOT)
    boots = np.array([below[rng.integers(0, len(below), len(below))].mean() for _ in range(B)])
    status = ("арбитр: r_max < 1 у большинства доков (исход B)" if frac_below > 0.5
              else "учётный гейт: r_max >= 1 у большинства доков (исход A; T1-T3 — с оговоркой)")
    prereg["3_rmax_status"] = {
        "pool": "nondegen (§0.3 degen по L23), n = " + str(len(fracs)),
        "n_docs": len(fracs), "frac_rmax_below_1": frac_below,
        "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "rmax_median": float(np.median(list(fracs.values()))),
        "formula": "frac = mean(1[r_max < 1]) по доку; CI = percentile-бутстрап доли (seed 117, B=1000)",
        "status": status, "verdict": status,
        "note": "пре-рег 3 — декларация статуса, не гипотеза; словарь §0.4 неприменим",
    }
    # П4: плато ρ̄(k≥9) == f_DC (rev.4, CRIT-1: вердикт против matched null; полоса/агрегатор/
    # DELTA_MIN — в артефактном поле prereg; r5: pair-block relabel + MA(1)-столбец)
    p4_res, z_ma1 = p4_plateau(cells, pool4, objects, nulls)
    prereg["4_plateau_vs_fDC"] = p4_res
    prereg["4_plateau_vs_fDC"]["pool4_n"] = len(pool4)
    prereg["4_plateau_vs_fDC"]["pool5_note"] = f"П4-пул (K=96) n={len(pool4)} из {len(nong)} nondegen; " \
        f"исключены: {sorted(set(nong) - set(pool4))}"
    # П5: горб (rev.4, MAJOR-1: matched null; rev.3, MINOR-12: один комбинированный test, α=0.05;
    # r5, D3: level-канал только по ē)
    prereg["5_hump"] = p5_hump(cells, pool5, objects, nulls)
    # r5 (arch-review D6): следствие двух нормировок — заявлено явно
    prereg["normalization_consequence"] = (
        "r5 (arch-review D6): тождество S(0) = 1 + 2Σρ̄ выполняется ТОЛЬКО для unit-variance "
        "нормировки ρ̂; для энерго-взвешенной (norm=T: c(k)/c(0)) замыкается Gram-тождество "
        "1 + 2Σ_{k<N}ρ̂(k) = r (G8), не unit-ACF; для T-k нормировки (знаменатель "
        "Σ_{t<N−k}‖x_t‖² переменный) unit-variance-тождество не выполняется вовсе. "
        "Заголовочный T0 (Σρ̄) — в aggregates.t0_headline с явными нормировками")
    # П6: Δ_med не вычисляется (П3) — assert _bad_names(out) до записи
    prereg["6_median_sum_absent"] = {"verdict": "pending (assert до записи)",
                                     "note": "compliance-проверка (поля Δ_med/S0 отсутствуют); "
                                             "словарь §0.4 неприменим"}

    # ---------------- агрегаты / вывод ----------------
    agg = aggregates(docs, cells, nong, nong_by_dom, objects)
    per_doc = {}
    for nm, d in sorted(docs.items()):
        pd = {}
        for L in LAYERS:
            pd[L] = {obj: {st: cells[(nm, L, obj, st)] for st in STATS} for obj in objects}
        per_doc[nm] = {
            "dom": d["dom"], "source": d["source"],
            "degen_tz": d["degen_tz"], "loopy_ladder": d["loopy_ladder"],
            "mcos": d["mcos"], "mshare": d["mshare"], "share_unavailable": d["share_unavailable"],
            "gen_sha": d.get("gen_sha"), "T": int(d["layers"][23][0].shape[0]),
            "l1_tpu": l1_status.get(nm),
            "null_ma1_z": {obj: z_ma1[obj].get(nm) for obj in objects},
            "layers": pd,
        }
    out = {
        "meta": {
            "cell": "T0.1", "tz": "docs/TZ_add_0-3.md",
            "tz_sha256": sha256_file(os.path.join(ROOT, "docs/TZ_add_0-3.md")),
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "layers": LAYERS, "K": K_MAX, "stats": list(STATS),
            "k_alphabet": list(K_ALPHABET),
            "boot_seed": SEED_BOOT, "boot_B": B,
            "n_docs": len(docs), "n_cells": len(cells),
            "objects": list(objects),
            "registry_note": "клетка = (doc, L, obj, stat), stat ∈ {mean, median} (§0.2); "
                             "моментные величины (rho, rho_Tk, r, sum_rho_full, gram_resid, f_dc) — "
                             "shared across stat (поле moment_stat_independent); dip_mean_1 — только в "
                             "mean-клетке, dip_med_1 — только в median-клетке (MAJOR-5, rev.3); "
                             "v_tangent: tangent_frame=u_t",
            "t25_ekurve_sha256": t25_sha,
            "t25_provenance": "artifact; скрипт утрачен (P14) — в вердикты AT0_1 не входит",
            "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                              "MKL_NUM_THREADS")},
                    "note": "setdefault — значения обёртки имеют приоритет (rev.3, MINOR-8); "
                            "отчитаны фактические"},
            "provenance_limited": [
                "P14 (TPU, L1, per-doc): fp16 деградирует мелкую L1-дрейф-структуру ТОЛЬКО у доков, где "
                "L1-шаг ниже квантового шума (recomputed vs meta dev > 1e-3) — список в G8.excluded_L1_tpu_docs; "
                "у них L1-моментные величины absent, dip_V,med-кривая из meta (artifact).",
                "P14 (TPU, hn-поле): _hn — баг продюсера (нормы первых 384 столбцов L23, e64_gpu.py:125), "
                "неприменим к строкам; строки — raw fp16-состояния (e64_gpu.py:218), ‖row‖ — истинная норма; "
                "годность TPU-рядов — замером (G0a_tpu_hn_field.l23_meta_vs_recomputed).",
                "P14 (t25): скрипт утрачен — t25_ekurve.json только как artifact, в вердикты AT0_1 не входит.",
                "P10: T0.4 (T1.0-режим) не вычисляется (нет данных в TPU-npz).",
                "G0-a: P2_corpus.npz без tok (0/16) — декод-проба и «те ли токены» для него невыполнимы (зафиксировано).",
            ],
        },
        "gates": gates,
        "prereg": prereg,
        "aggregates": agg,
        "per_doc": per_doc,
    }
    out["prereg"]["6_median_sum_absent"]["verdict"] = "ok (S(0)/Σ для Δ_med не вычисляются, П3)"

    # ---------------- жёсткие гейты ДО записи (провал = стоп, на диске нет артефакта) ----------------
    assert anchor_ok, "G1′: якорь LADDER_V2 не воспроизведён — баг оценщика"
    assert gates["G2_ok"], f"G2: неполный набор клеток: {gates['G2_completeness']}"
    assert gates["G3_registry"]["ok"], f"G3: реестр неполный: {g3_bad[:5]}"
    assert not g8_bad, f"G8: gram_resid >= 1e-10: {list(g8_bad.items())[:3]}"
    assert gates["G6_ra_sanity"]["ok"], "G6: R/A closure"
    assert prereg["2_r_le_rmax"]["verdict"] == "confirmed", "П2: r > r_max — баг"
    assert gates["sec17"]["permutation_invariance"]["ok"], \
        f"§1.7: перестановочная инвариантность: spread={perm_spread}"
    assert "FAIL" not in gates["sec17"]["field_selfcheck"], f"§1.7: field._selfcheck: {gates['sec17']['field_selfcheck']}"
    assert not _bad_names(out), "П6: в артефакте появились поля Δ_med/S0"

    # G7: hash-store (полный артефакт) + claim-JSON (без per_doc) + манифест + run-лог
    full_bytes = json.dumps(jclean(out), ensure_ascii=False).encode("utf-8")
    store_sha = hashlib.sha256(full_bytes).hexdigest()
    store_path = os.path.join(STORE_DIR, store_sha[:16] + ".json")
    with open(store_path, "wb") as f:
        f.write(full_bytes)
    claim = {k: out[k] for k in ("meta", "gates", "prereg", "aggregates")}
    claim["store"] = {"path": os.path.relpath(store_path, OUT_DIR), "sha256": store_sha,
                      "n_docs": len(out["per_doc"]), "n_cells": out["meta"]["n_cells"]}
    with open(OUT_JSON, "w") as f:
        json.dump(jclean(claim), f, ensure_ascii=False)
    man = dict(claim)
    man["manifest"] = {"store_sha256": store_sha, "claim_sha256": None, "written": True}
    with open(MAN_JSON, "w") as f:
        json.dump(jclean(man), f, ensure_ascii=False, indent=1)
    man["manifest"]["claim_sha256"] = sha256_file(OUT_JSON)
    with open(MAN_JSON, "w") as f:
        json.dump(jclean(man), f, ensure_ascii=False, indent=1)

    logline(f"docs={len(docs)} cells={len(cells)} anchor={anchor_val:.4f} (want {ANCHOR_LADDER}, "
            f"n={n_anchor}, ok={anchor_ok})")
    logline(f"G8 max_resid={gates['G8_gram']['max_resid']:.3e} (cells={gates['G8_gram']['cells']}, "
            f"independent={gates['G8_gram']['independent_doc_layer_obj']}) "
            f"G6 closure_rel={g6_rel:.3e} perm_spread={perm_spread:.3e} "
            f"excluded_L1={gates['G8_gram']['excluded_L1_tpu_docs']}")
    logline(f"П3: {prereg['3_rmax_status']['status']} (frac_below_1={frac_below:.3f} "
            f"ci={prereg['3_rmax_status']['ci']})")
    for obj in objects:
        p4 = prereg["4_plateau_vs_fDC"][obj]
        ma1 = p4["null_analytic_ma1"]
        logline(f"П4 {obj}: est={p4['est_median']:+.4f} null_pairblock(E)={p4['null_block2']['E']:+.4f} "
                f"p_perm={p4['null_block2']['p_perm']:.3f} ma1(z_med={ma1['median_z']:+.2f}, "
                f"n_nm={ma1['n_no_match']}) ci={p4['ci_median']} -> {p4['verdict']}")
        p5 = prereg["5_hump"][obj]
        logline(f"П5 {obj}: e3={p5['e3_median']:+.4f} e4={p5['e4_median']:+.4f} "
                f"pos_ee={p5['pos']}/{p5['n']} p_sign={p5['null_block2']['p_perm_sign_ee']:.3f} "
                f"sigma_ee={p5['null_block2']['sigma']['ee']:+.2f} -> {p5['verdict']}")
    logline("aggregates pooled: " + str({k: v for k, v in agg["pooled_nondeg"].items() if v is not None}))
    logline(f"store: {store_path} (sha {store_sha[:16]})")
    logline(f"written: {OUT_JSON}, {MAN_JSON}, {RUN_LOG}")
    logline("AT0_1 gates: ALL PASS")
    with open(RUN_LOG, "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
