#!/usr/bin/env python
"""AT8_2-R2 (T8.2 redesign, 17.11): локальный отклик и релаксация поля после
инъекции повторяющегося блока.

ТЗ (пре-рег): docs/AT8_2_R2_TZ.md v1.0. Черновик: docs/AT8_2_review.md.
Машина: r1 (форварды) + r2 (анализ, ревизия по ревью-раунду 1:
docs/AT8_2_review_r1_answers.md, F1–F7/M1–M7): primary dose n=16 (F2),
δ_near/δ_far через ту же near/far-статистику (F4), CI-правила вердиктов (F3),
parный G_inside (F5), слой-теги без LAYER_INCONSISTENT (F6), DOMAINS_MIN=4 (F7),
phrase-token-check (F1), finite-гейт (M5), token-sha (M4), npz из store (M7),
filler_source в манифесте (M3), шовный инкремент — диагностика, option A (M1).

Моды:
  (нет флага)     полный прогон (14 доков, n=2/4/8/16; 98 форвардов)
  --smoke         wiki × 2 дока × n=2 (8 форвардов) + якоря (TZ §14)
  --reanalysis    disk-only: store → verdict + JSON (машина r2, TZ §12)
  --test-synth    синтетические тесты вердикт-машины (без модели)

Артефакты: results/AT8/AT8_2_R2{,_store,_manifest}.json, AT8_2_R2_series.npz
(>10 MB — gitignored, бэкап /data/prism_backups/at8_2_r2/, sha256 в manifest),
logs/AT8/AT8_2_R2_run.log (trackio at8_2_r2_r1).

Привилегия 16.11: prism/explog.py + prism/nulls_and_stats.py (по файлу —
prism/__init__ тянет matplotlib, которого нет в runtime-окружении).
"""

import hashlib
import json
import math
import os
import sys
import time
import importlib.util
from collections import Counter
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_explog = _load("prism_explog", os.path.join(ROOT, "prism", "explog.py"))
_nulls = _load("prism_nulls_and_stats", os.path.join(ROOT, "prism", "nulls_and_stats.py"))
ExpLog = _explog.ExpLog
paired_bootstrap = _nulls.paired_bootstrap

# ----------------------------- константы (TZ §17 + машина r2) -----------------------------
MODEL = "/data/Qwen/Qwen3.5-0.8B"
LAYERS = [6, 12, 18, 19, 23]            # TZ §4: 3 GDN + full L19 + readout L23
PRIMARY_LAYER = 23
PRIMARY_DOSE = 16                       # F2 (ревью r1): первичная доза пре-рег
PHRASE = " The class will be held at the park."        # 9 токенов (E-71B, р1)
PHRASE_IDS = [561, 523, 668, 381, 5467, 506, 279, 5926, 13]
N_DOSE = [2, 4, 8, 16]
H = 64
NEAR_END = 15                              # near: d ∈ [0, 15] (16 точек)
FAR_START = H - 16                         # far:  d ∈ [48, 63] (16 точек)
DOMAIN_ORDER = ["wiki", "math", "code", "chat"]
PINS = {
    d: os.path.join(ROOT, "data", "calib_pins",
                    f"{d}___data_Qwen_Qwen3.5_0.8B__n12__min192__max512__plain.json")
    for d in DOMAIN_ORDER
}
T_MIN, T_MAX = 192, 512
GRAM = 9
GRAM_DUP_MAX = 1                           # S_no_baseline_loop: 9-грамма ≥2 → out
N_PER_DOMAIN = 4
DOCS_MIN = 12
DOMAINS_MIN = 4                            # F7 (ревью r1): все 4 домена обязательны,
                                           # по ≥2 дока (старое 3 — мёртвая логика)
CE_LOOP_MAX = 1.0                          # вторичный диагностический абс-гейт (TZ §8)
CE_FILLER_MIN = 2.0
INSIDE_DOSE = 16
INSIDE_PASS_FRAC = 0.75                    # F5 (ревью r1): парный гейт, ≥75% доков
SEED = 0
B_BOOT = 2000
RUN_REVISION = "r1"                        # форварды (store fingerprint)
MACHINE_REVISION = "r2"                    # анализ по ревью-раунду 1

OUT_DIR = os.path.join(ROOT, "results", "AT8")
CLAIM = os.path.join(OUT_DIR, "AT8_2_R2.json")
STORE = os.path.join(OUT_DIR, "AT8_2_R2_store.json")
MANIFEST = os.path.join(OUT_DIR, "AT8_2_R2_manifest.json")
SERIES = os.path.join(OUT_DIR, "AT8_2_R2_series.npz")
RUNLOG = os.path.join(ROOT, "logs", "AT8", "AT8_2_R2_run.log")

# ----------------------------- лог + trackio (установка r1) -----------------------------
_log = ExpLog()
logline = _log


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_ids(ids):
    # M4 (ревью r1): fingerprint фактического входа (токены), не сырого текста
    return hashlib.sha256(np.asarray(ids, dtype="int64").tobytes()).hexdigest()[:16]


def init_trackio(t0, script_path):
    # AGENTS п.6: TRACKIO_DIR задаёт run_one.sh; прямой запуск — fail-loud.
    if not os.environ.get("TRACKIO_DIR"):
        raise SystemExit(
            "trackio: TRACKIO_DIR не задан — запускать через ./run_one.sh "
            "(он экспортирует TRACKIO_DIR=$PRISM_DIR/.trackio); для "
            "--test-synth/--reanalysis задайте TRACKIO_DIR вручную.")
    import trackio
    # API как в AT0_8_vtanp: init() сразу создаёт run.
    tr = trackio.init(project="prism", name=f"at8_2_r2_{RUN_REVISION}", group="AT8_2",
                      config={"device": "cpu", "model": MODEL,
                              "log": os.path.abspath(RUNLOG),
                              "run_revision": RUN_REVISION,
                              "machine_revision": MACHINE_REVISION,
                              "script_sha": sha256_file(script_path),
                              "seed": SEED})
    tr.log({"started_at": datetime.now(timezone.utc).isoformat()}, step=0)
    state = {"done": False}

    def finish(status="ok"):
        if state["done"]:
            return
        state["done"] = True
        tr.log({"finished_at": datetime.now(timezone.utc).isoformat(),
                "status": status, "duration": round(time.time() - t0, 1)},
               step=int(time.time() - t0))
        tr.finish()
    return tr, finish


def tstage(tr, t0, name):
    tr.log({"stage": name, "stage_s": round(time.time() - t0, 1)},
           step=int(time.time() - t0))
    logline(f"stage: {name} ({time.time() - t0:.1f}s)")

# ----------------------------- пул (TZ §6) -----------------------------


def max_gram_count(ids):
    g = [tuple(ids[j:j + GRAM]) for j in range(len(ids) - GRAM + 1)]
    return max(Counter(g).values()) if g else 0


def load_pool(tk):
    """12 кандидатов/домен из пинов → 9g-фильтр (max count ≥2 → out) →
    первые 4/домен (fallback: все прошедшие; нужно ≥12 доков и 4 домена)."""
    cand, pool_fps = {}, {}
    for d in DOMAIN_ORDER:
        pin = json.load(open(PINS[d]))
        pool_fps[d] = pin["fp"]
        lst = []
        for t in pin["texts"]:
            ids = tk(t, max_length=T_MAX, truncation=True,
                     add_special_tokens=False)["input_ids"]
            if len(ids) < T_MIN:
                continue
            lst.append((t, ids, max_gram_count(ids)))
        cand[d] = lst
    selected = {}
    for d in DOMAIN_ORDER:
        passed = [(t, i, g) for (t, i, g) in cand[d] if g <= GRAM_DUP_MAX]
        if len(passed) < 2:
            # F7 (ревью r1): домен выпадает → прогон стоп (DOMAINS_MIN=4)
            raise SystemExit(f"INSUFFICIENT_POOL: домен {d} < 2 доков "
                             f"(DOMAINS_MIN=4 — все 4 домена обязательны)")
        selected[d] = passed[:N_PER_DOMAIN] if len(passed) >= N_PER_DOMAIN else passed
    docs = []
    for d in DOMAIN_ORDER:
        for j, (t, ids, g) in enumerate(selected[d]):
            docs.append(dict(name=f"{d}_{j}", domain=d, text=t, ids=ids, gram=g))
    n_dom = sum(1 for d in DOMAIN_ORDER if selected[d])
    if len(docs) < DOCS_MIN or n_dom < DOMAINS_MIN:
        raise SystemExit(f"INSUFFICIENT_POOL: {len(docs)} доков / {n_dom} доменов "
                         f"(нужно ≥{DOCS_MIN}/≥{DOMAINS_MIN})")
    # filler-карта (TZ §7): round-robin +1/+2 внутри домена (k≥3); при k=2 —
    # следующий домен в DOMAIN_ORDER (циклически): fillerA = его док [0],
    # fillerB = док [1]. M3 (ревью r1): филлеры — из того же пула (cross-doc).
    filler_map = {}
    for d in DOMAIN_ORDER:
        k = len(selected[d])
        next_d = DOMAIN_ORDER[(DOMAIN_ORDER.index(d) + 1) % len(DOMAIN_ORDER)]
        for i in range(k):
            if k >= 3:
                src = [f"{d}_{(i + 1) % k}", f"{d}_{(i + 2) % k}"]
            else:
                src = [f"{next_d}_0", f"{next_d}_1"]
            filler_map[f"{d}_{i}"] = src
    meta = dict(pool_fps=pool_fps,
                n_candidates={d: len(cand[d]) for d in DOMAIN_ORDER},
                n_passed={d: sum(1 for (_, _, g) in cand[d] if g <= GRAM_DUP_MAX)
                          for d in DOMAIN_ORDER},
                baseline_loop_metrics={doc["name"]: doc["gram"] for doc in docs},
                filler_map=filler_map,
                filler_source="same_pool_cross_doc")   # M3 (ревью r1)
    return docs, meta

# ----------------------------- форвард + ряды (TZ §3) -----------------------------


def fwd(m, ids, torch):
    """Один forward: per-position CE + hidden states 5 слоёв. use_cache=False —
    GDN-путь р1 (r1-якорь: CE ctrl-doc wiki_0 = 3.2320)."""
    caps = {}
    hooks = []
    for L in LAYERS:
        hooks.append(m.model.layers[L].register_forward_hook(
            lambda mod, i, o, l=L: caps.__setitem__(
                l, (o[0] if isinstance(o, (tuple, list)) else o).detach())))
    t = torch.tensor([ids])
    try:
        with torch.no_grad():
            o = m(input_ids=t, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    ce = torch.nn.functional.cross_entropy(
        o.logits[0, :-1].float(), t[0, 1:], reduction="none")
    T = len(ids)
    return dict(
        ce=ce.numpy().astype("float32"),        # ce[i] = CE(token i+1 | prefix)
        h={L: caps[L][0, :T].to(torch.float32).numpy() for L in LAYERS},
        T=T,
    )


def cond_series(res, T, cut, L_n):
    """Позиционные ряды условия.
    cos_raw[p]    = cos(Δh_p, Δh_{p+1}),  Δh_p = h_{p+1} − h_p   (primary stat)
    cos_state[p]  = cos(u_p, u_{p+1}),  u = unit(h)  (вторичный)
    norm_delta[p] = ‖Δh_p‖
    p = 0..T'-2; post-индекс d: cut + L_n + d (loop/filler) или cut + d (ctrl).
    M1 (ревью r1): option A — шовный инкремент cos_raw[cut+L_n-1] в post-ряд
    НЕ входит (near[0] = первая внутренняя пара суффикса); значение шва —
    диагностика в анализе."""
    T_ = res["T"]
    out = {"ce": [round(float(x), 6) for x in res["ce"]]}
    for L in LAYERS:
        h = res["h"][L]
        dh = h[1:] - h[:-1]                                   # [T'-1, D]
        nrm_dh = np.linalg.norm(dh, axis=1)
        nrm_h = np.linalg.norm(h, axis=1)
        cos_raw = np.zeros(T_ - 2, dtype="float32")
        for p in range(T_ - 2):
            cos_raw[p] = float(dh[p] @ dh[p + 1]) / max(float(nrm_dh[p] * nrm_dh[p + 1]), 1e-12)
        u = h / np.maximum(nrm_h[:, None], 1e-12)
        cos_state = np.array([float(u[p] @ u[p + 1]) for p in range(T_ - 2)],
                             dtype="float32")
        out[f"L{L}"] = dict(
            cos_raw=[round(float(x), 6) for x in cos_raw],
            cos_state=[round(float(x), 6) for x in cos_state],
            norm_delta=[round(float(x), 6) for x in nrm_dh],
        )
    return out


def post_series(cell, cut, L_n):
    """post-серии (длина H) из cos_raw: loop/filler — сдвиг L_n, ctrl — 0."""
    out = {}
    for L in LAYERS:
        base = cell[f"L{L}"]["cos_raw"]
        start = cut + (L_n if L_n > 0 else 0)
        out[str(L)] = [base[start + d] for d in range(H)]
    return out


def filler_ids(tk, text, n):
    return tk(text, max_length=T_MAX, truncation=True,
              add_special_tokens=False)["input_ids"][:9 * n]

# ----------------------------- статистики (машина r2) -----------------------------
N_W = NEAR_END + 1            # 16 точек near
F_W = H - FAR_START           # 16 точек far


def _near(vals):
    """90th pct |vals| по первым N_W позициям (TZ §8)."""
    return float(np.percentile(np.abs(np.asarray(vals)[:N_W]), 90))


def _far(vals):
    """mean |vals| по последним F_W позициям (TZ §8)."""
    return float(np.mean(np.abs(np.asarray(vals)[FAR_START:])))


def _dose_tag(i_loop):
    """TZ §9 (док-медианные I(n)): FLAT I16/I2≤2; SATURATING >2 и I16/I8≤2;
    MONOTONIC I16/I8>2 и строго растёт; иначе NOT_RESOLVED."""
    vals = [i_loop.get(str(n)) for n in N_DOSE]
    if any(v is None for v in vals):
        return "DOSE_NOT_RESOLVED"
    if any(b <= a for a, b in zip(vals, vals[1:])):
        return "DOSE_NOT_RESOLVED"
    r2, r8 = vals[3] / vals[0], vals[3] / vals[2]
    if r2 <= 2.0:
        return "DOSE_FLAT"
    if r8 <= 2.0:
        return "DOSE_SATURATING"
    return "DOSE_MONOTONIC"


def _boot(arr2col):
    """doc-bootstrap CI95 пары колонок (near, far, near-far, near_generic)."""
    a = np.asarray(arr2col, dtype="float64")
    res = {}
    rng = np.random.default_rng(seed=SEED)
    n = a.shape[0]
    for name, col in (("near", 0), ("far", 1), ("near_minus_far", 2),
                      ("near_generic", 3)):
        boots = []
        for _ in range(B_BOOT):
            idx = rng.integers(0, n, size=n)
            boots.append(float(a[idx, col].mean()))
        res[name] = dict(point=float(a[:, col].mean()),
                         ci_low=float(np.percentile(boots, 2.5)),
                         ci_high=float(np.percentile(boots, 97.5)),
                         B=B_BOOT)
    return res


def compute_verdict(ci, dmin_near, dmin_far, near, far, near_generic,
                    near_by_layer, dmin_near_layer, inside_pass, n_docs,
                    i_loop, n_domains, gates_ok=True):
    """Чистая функция; дерево машины r2 (ревью r1, F2–F4, F6):
    первичная доза = PRIMARY_DOSE; значимость — по CI (CI_low > δ той же
    статистики); слой-тег без LAYER_INCONSISTENT.
    ci[n] = {near, far, near_minus_far, near_generic} — doc-bootstrap CI95."""
    diag = dict(n_docs=n_docs, n_domains=n_domains,
                inside_pass=f"{inside_pass}/{n_docs}",
                inside_pass_min=f"{int(math.ceil(INSIDE_PASS_FRAC * n_docs))}/{n_docs}")
    dose_tag = _dose_tag(i_loop)
    pn = str(PRIMARY_DOSE)
    nci = ci[pn]["near"]
    fci = ci[pn]["far"]
    dci = ci[pn]["near_minus_far"]
    gci = {n: ci[n]["near_generic"] for n in ci}

    if not gates_ok:
        return dict(value="NOT_RESOLVED", match_pre_reg=False, n_star=pn,
                    tags=dict(layer=None, dose=dose_tag),
                    reasons=["gates failed"], diagnostics=diag)
    if inside_pass < int(math.ceil(INSIDE_PASS_FRAC * n_docs)):
        return dict(value="INSENSITIVE_CELL", match_pre_reg=False, n_star=pn,
                    tags=dict(layer=None, dose=dose_tag),
                    reasons=[f"G_inside (парный) {inside_pass}/{n_docs} < "
                             f"ceil(0.75×{n_docs}) — клетка не видит стимул"],
                    diagnostics=diag)
    if nci["ci_low"] > dmin_near:
        reasons = [f"primary n={pn}: CI95(near)=[{nci['ci_low']:.4f}, "
                   f"{nci['ci_high']:.4f}] > δ_near={dmin_near:.4f}"]
        if fci["ci_low"] > dmin_far:
            v = "LOOP_RESPONSE_PERSISTS"
            reasons.append(f"CI95(far)=[{fci['ci_low']:.4f}, {fci['ci_high']:.4f}] "
                           f"> δ_far={dmin_far:.4f} — специфический отклик не "
                           f"затухает до конца горизонта H={H}")
        elif dci["ci_low"] > 0.0 and fci["ci_high"] <= dmin_far:
            # F3 (ревью r1): DECAYS = CI_low(near−far) > 0 И CI_high(far) ≤ δ_far
            v = "LOOP_RESPONSE_DECAYS"
            reasons.append(f"CI95(near−far)=[{dci['ci_low']:.4f}, "
                           f"{dci['ci_high']:.4f}] > 0 И CI95(far)="
                           f"[{fci['ci_low']:.4f}, {fci['ci_high']:.4f}] "
                           f"≤ δ_far={dmin_far:.4f} — релаксация к базовой линии "
                           f"(НЕ упругость)")
        else:
            v = "NOT_RESOLVED"
            reasons.append(f"near значимо, но far/релаксация не резолвятся: "
                           f"CI95(near−far)=[{dci['ci_low']:.4f}, "
                           f"{dci['ci_high']:.4f}], CI95(far)="
                           f"[{fci['ci_low']:.4f}, {fci['ci_high']:.4f}], "
                           f"δ_far={dmin_far:.4f}")
    else:
        mg = max(c["ci_low"] for c in gci.values())
        if mg > dmin_near:
            v = "GENERIC_INSERTION_EFFECT"
            reasons = [f"CI95(near) ≤ δ_near={dmin_near:.4f} на всех дозах; "
                       f"e_generic значим (max CI_low={mg:.4f} > δ_near) — эффект "
                       f"от вставки как таковой, не специфичный луп"]
        else:
            v = "NO_LOOP_SPECIFIC_EFFECT"
            reasons = [f"CI95(near) ≤ δ_near={dmin_near:.4f} на всех дозах "
                       f"(max CI_low={max(ci[n]['near']['ci_low'] for n in ci):.4f}) "
                       f"И e_generic ≤ δ_near (max CI_low={mg:.4f}) — "
                       f"специфического локального отклика на луп нет (ниже "
                       f"разрешения)"]
    # слой-тег (F6): значимость — near_layer > δ_near_layer (та же статистика)
    sig = [L for L in LAYERS
           if near_by_layer[pn][str(L)] > dmin_near_layer[str(L)]]
    if PRIMARY_LAYER in sig and len(sig) >= 2:
        lt = "FIELD_RESPONSE"
    elif sig == [PRIMARY_LAYER]:
        lt = "READOUT_ONLY"
    elif PRIMARY_LAYER not in sig and len(sig) >= 1:
        lt = "INTERNAL_ONLY"
    else:
        lt = None
    return dict(value=v, match_pre_reg=False, n_star=pn,
                tags=dict(layer=lt, dose=dose_tag), reasons=reasons,
                diagnostics=diag)

# ----------------------------- анализ (TZ §8, §10 + машина r2) -----------------------------


def _check_series(cell, dn, label):
    ce = np.asarray(cell["ce"], dtype="float64")
    if not np.isfinite(ce).all():
        raise SystemExit(f"M5 finite-gate: {dn} {label} ce содержит NaN/Inf")
    for L in LAYERS:
        for key in ("cos_raw", "cos_state", "norm_delta"):
            a = np.asarray(cell[f"L{L}"][key], dtype="float64")
            if not np.isfinite(a).all():
                raise SystemExit(f"M5 finite-gate: {dn} {label} L{L} "
                                 f"{key} содержит NaN/Inf")


def _finite_gate(store, docs, doses):
    """M5 (ревью r1): fail-loud finite-гейт до вердикта."""
    for dn in docs:
        _check_series(store[dn]["ctrl"], dn, "ctrl")
        for n in doses:
            cell = store[dn][f"n{n}"]
            for cond in ("loop", "fillerA", "fillerB"):
                _check_series(cell[cond], dn, f"n{n} {cond}")
            for key in ("e_spec", "e_generic", "e_null"):
                for L in LAYERS:
                    a = np.asarray(cell[key][str(L)], dtype="float64")
                    if not np.isfinite(a).all():
                        raise SystemExit(f"M5 finite-gate: {dn} n{n} {key} "
                                         f"L{L} содержит NaN/Inf")
    logline("M5 finite-gate: PASS (все ряды конечны)")


def analyze(store, doses=None):
    """store → сводки + δ (та же статистика, F4) + гейты + bootstrap + вердикт."""
    doses = list(doses or N_DOSE)
    docs = sorted(k for k in store.keys() if k != "__config__")
    n_docs = len(docs)
    n_domains = len({store[k]["domain"] for k in docs})
    _finite_gate(store, docs, doses)
    # δ (F4, ревью r1): null прогоняется через тот же near/far-пайплайн;
    # 95th pct по докам×дозам (per-doc), не по сырым позициям
    dmin_near_layer, dmin_far_layer = {}, {}
    for L in LAYERS:
        nn, fn = [], []
        for dn in docs:
            for n in doses:
                en = store[dn][f"n{n}"]["e_null"][str(L)]
                nn.append(_near(en))
                fn.append(_far(en))
        dmin_near_layer[str(L)] = float(np.percentile(nn, 95))
        dmin_far_layer[str(L)] = float(np.percentile(fn, 95))
    dmin_near = dmin_near_layer[str(PRIMARY_LAYER)]
    dmin_far = dmin_far_layer[str(PRIMARY_LAYER)]
    # пер-доковые near/far/generic + bootstrap по докам (4 колонки)
    near, far, near_generic = {}, {}, {}
    near_by_layer = {}
    boot = {}
    for n in doses:
        ns, fs, ng, nbl = [], [], [], {}
        for dn in docs:
            c = store[dn][f"n{n}"]
            ns.append(_near(c["e_spec"][str(PRIMARY_LAYER)]))
            fs.append(_far(c["e_spec"][str(PRIMARY_LAYER)]))
            ng.append(_near(c["e_generic"][str(PRIMARY_LAYER)]))
            for L in LAYERS:
                nbl.setdefault(str(L), []).append(_near(c["e_spec"][str(L)]))
        near[str(n)] = float(np.mean(ns))
        far[str(n)] = float(np.mean(fs))
        near_generic[str(n)] = float(np.mean(ng))
        near_by_layer[str(n)] = {str(L): float(np.mean(v)) for L, v in nbl.items()}
        boot[str(n)] = _boot(list(zip(ns, fs, [a - b for a, b in zip(ns, fs)], ng)))
    # G_inside (F5, ревью r1): парный гейт — loop < fillerA И loop < fillerB,
    # ≥75% доков на PRIMARY_DOSE; абсолютные пороги — вторичная диагностика
    inside_dose = INSIDE_DOSE if INSIDE_DOSE in doses else max(doses)
    inside_pass = 0
    abs_pass = 0
    for dn in docs:
        c = store[dn][f"n{inside_dose}"]
        lb, fa, fb = (c["loop"]["ce_block"], c["fillerA"]["ce_block"],
                      c["fillerB"]["ce_block"])
        if lb < fa and lb < fb:
            inside_pass += 1
        if lb < CE_LOOP_MAX and fa >= CE_FILLER_MIN:
            abs_pass += 1
    # I_loop: doc-медиана Σ CE по луп-блоку (TZ §9)
    i_loop = {str(n): float(np.median([store[dn][f"n{n}"]["loop"]["i_loop"]
                                       for dn in docs])) for n in doses}
    # шовный инкремент (M1, диагностика): cos_raw[cut+L_n-1], loop − fillerA
    seam = {}
    for n in doses:
        L_n = 9 * n
        vals = []
        for dn in docs:
            cut = store[dn]["cut"]
            c = store[dn][f"n{n}"]
            vals.append(c["loop"][f"L{PRIMARY_LAYER}"]["cos_raw"][cut + L_n - 1]
                        - c["fillerA"][f"L{PRIMARY_LAYER}"]["cos_raw"][cut + L_n - 1])
        seam[str(n)] = float(np.mean(vals))
    v = compute_verdict(boot, dmin_near, dmin_far, near, far, near_generic,
                        near_by_layer, dmin_near_layer, inside_pass, n_docs,
                        i_loop, n_domains)
    return dict(dmin_near=dmin_near, dmin_far=dmin_far,
                dmin_near_layer=dmin_near_layer, dmin_far_layer=dmin_far_layer,
                near=near, far=far, near_generic=near_generic,
                near_by_layer=near_by_layer, bootstrap=boot,
                inside_pass=inside_pass, inside_abs_pass=abs_pass,
                seam=seam, i_loop=i_loop, verdict=v,
                n_docs=n_docs, n_domains=n_domains, docs=docs)
# ----------------------------- артефакты (TZ §12) -----------------------------


def _round_ci(c):
    return {k: (round(x, 6) if isinstance(x, float) else x) for k, x in c.items()}


def _write_artifacts(an, manifest, mode):
    v = an["verdict"]
    claim = dict(
        experiment="AT8_2-R2", tz="docs/AT8_2_R2_TZ.md",
        run_revision=RUN_REVISION, machine=MACHINE_REVISION, mode=mode,
        verdict=v,
        delta_min=dict(
            near_L23=round(an["dmin_near"], 6), far_L23=round(an["dmin_far"], 6),
            near_by_layer={k: round(x, 6) for k, x in an["dmin_near_layer"].items()},
            far_by_layer={k: round(x, 6) for k, x in an["dmin_far_layer"].items()},
            note="95th pct per-doc null near/far (F4: та же статистика, что и "
                 "сравниваемые near/far e_spec)"),
        near={k: round(x, 6) for k, x in an["near"].items()},
        far={k: round(x, 6) for k, x in an["far"].items()},
        near_generic={k: round(x, 6) for k, x in an["near_generic"].items()},
        near_by_layer={n: {k: round(x, 6) for k, x in m.items()}
                       for n, m in an["near_by_layer"].items()},
        bootstrap={n: {k: _round_ci(c) for k, c in m.items()}
                   for n, m in an["bootstrap"].items()},
        seam={k: round(x, 6) for k, x in an["seam"].items()},
        i_loop={k: round(x, 6) for k, x in an["i_loop"].items()},
        inside_pass=an["inside_pass"],
        inside_abs_pass=an["inside_abs_pass"],
        gate=dict(primary_dose=PRIMARY_DOSE,
                  inside_rule="pair: loop_ce < fillerA_ce AND loop_ce < fillerB_ce, "
                              f"pass >= {int(INSIDE_PASS_FRAC * 100)}% docs "
                              f"(F5); abs diag: loop<{CE_LOOP_MAX} & fillerA>="
                              f"{CE_FILLER_MIN}",
                  inside_dose=INSIDE_DOSE,
                  H=H, near=[0, NEAR_END], far=[FAR_START, H - 1],
                  seam="option A: шовный инкремент (d=-1) вне post-ряда (M1), "
                       "значение шва — диагностика 'seam'",
                  gram_filter=f"9g max_count ≤ {GRAM_DUP_MAX}"),
        layers=LAYERS, primary_layer=PRIMARY_LAYER,
        primary_dose=PRIMARY_DOSE,
        n_docs=an["n_docs"], n_domains=an["n_domains"],
    )
    json.dump(claim, open(CLAIM, "w"), indent=1)
    manifest["claim_sha256"] = sha256_file(CLAIM)
    json.dump(manifest, open(MANIFEST, "w"), indent=1)
    logline(f"claim sha256={manifest['claim_sha256'][:16]}")


def _series_from_store(store, doses):
    """M7 (ревью r1): npz строится ИЗ STORE (полный набор при любом resume)."""
    out = {}
    for dn in sorted(k for k in store.keys() if k != "__config__"):
        cell_doc = store[dn]
        out[f"{dn}__ctrl__n0"] = cell_doc["ctrl"]
        for n in doses:
            if f"n{n}" not in cell_doc:
                continue
            for cond in ("loop", "fillerA", "fillerB"):
                out[f"{dn}__{cond}__n{n}"] = cell_doc[f"n{n}"][cond]
    return out

# ----------------------------- run -----------------------------


def run_main(mode="full"):
    import torch
    sys.modules["fla"] = None
    sys.modules["fla.ops"] = None
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RUNLOG), exist_ok=True)
    _logf = open(RUNLOG, "w")

    class _Tee:
        def __init__(self, a, b): self.a, self.b = a, b
        def write(self, s): self.a.write(s); self.b.write(s); self.b.flush()
        def flush(self): self.a.flush(); self.b.flush()

    sys.stdout = _Tee(sys.__stdout__, _logf)
    logline(f"AT8_2-R2 start {datetime.now(timezone.utc).isoformat()} "
             f"py={sys.version.split()[0]} torch={torch.__version__} "
             f"(CPU, position contrasts, run={RUN_REVISION}, "
             f"machine={MACHINE_REVISION}, mode={mode})")
    tr, tr_done = init_trackio(t0, os.path.abspath(__file__))

    tk = AutoTokenizer.from_pretrained(MODEL)
    # F1 (ревью r1): phrase-token-check — стимул обязан совпадать с токенизатором
    pcheck = tk(PHRASE, add_special_tokens=False)["input_ids"]
    assert pcheck == PHRASE_IDS, f"phrase mismatch: {pcheck} != {PHRASE_IDS}"
    logline(f"F1 phrase-token-check: PASS {pcheck}")
    docs_all, pool_meta = load_pool(tk)
    # тексты филлеров всегда из пула (filler_map ссылается только на доки пула)
    texts_by_name = {d["name"]: d["text"] for d in docs_all}
    docs = docs_all
    if mode == "smoke":
        docs = [d for d in docs_all if d["domain"] == "wiki"][:2]
        pool_meta = dict(pool_meta, smoke=True,
                         filler_map={d["name"]: pool_meta["filler_map"][d["name"]]
                                     for d in docs})
    n_docs, n_domains = len(docs), len({d["domain"] for d in docs})
    logline(f"pool: {n_docs} docs / {n_domains} domains "
             f"lens={[len(d['ids']) for d in docs]}")
    logline(f"pool_fps={pool_meta['pool_fps']} passed={pool_meta['n_passed']} "
             f"filler_map={pool_meta['filler_map']} "
             f"filler_source={pool_meta['filler_source']}")
    tstage(tr, t0, "pool")

    # store + resume (config-fingerprint)
    cfg_fp = dict(model=MODEL, layers=LAYERS, phrase_ids=PHRASE_IDS,
                  n_dose=N_DOSE if mode == "full" else [2], H=H,
                  pool_fps=pool_meta["pool_fps"], run_revision=RUN_REVISION,
                  filler_map=pool_meta["filler_map"], mode=mode,
                  docs_sha256={d["name"]: sha256_ids(d["ids"]) for d in docs})
    store = {}
    if os.path.exists(STORE):
        store = json.load(open(STORE))
        if store.get("__config__") != cfg_fp:
            raise SystemExit("C4: store config-fingerprint не совпадает — "
                             "resume чужого store запрещён (TZ §12); удалите "
                             "AT8_2_R2_store.json для нового режима")
        logline("resume: store найден, config fp совпадает")
    store["__config__"] = cfg_fp

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    attn_impl = getattr(m.config, "_attn_implementation", "unknown")
    logline(f"C8: attn_impl={attn_impl} (fla заглушен None — CPU-fallback)")
    tstage(tr, t0, "model_loaded")

    doses = N_DOSE if mode == "full" else [2]
    n_fwd = 0
    for doc in docs:
        nm, ids = doc["name"], doc["ids"]
        T = len(ids)
        cut = T // 2
        if T - cut < H + 3:
            raise SystemExit(f"{nm}: short_horizon (T={T}, T−T//2 < H+3) — "
                             f"пул должен был отфильтровать (TZ §5)")
        cell_doc = store.get(nm) or {}
        if cell_doc.get("T") not in (None, T) or cell_doc.get("cut") not in (None, cut):
            raise SystemExit(f"{nm}: resume-конфликт (T/cut не совпадают с store)")
        store[nm] = dict(cell_doc, T=T, cut=cut, domain=doc["domain"])
        # ctrl (один форвард на док)
        if "ctrl" not in store[nm]:
            r = fwd(m, ids, torch)
            n_fwd += 1
            c = cond_series(r, T, cut, 0)
            store[nm]["ctrl"] = c
            logline(f"  {nm} ctrl: T={T} cut={cut} CE={float(r['ce'].mean()):.4f} "
                    f"(якорь р1 wiki-шкала ~3.2)")
            json.dump(store, open(STORE, "w"), default=float)
        for n in doses:
            if f"n{n}" in store[nm]:
                continue
            L_n = len(PHRASE_IDS) * n
            fa_txt = texts_by_name[pool_meta["filler_map"][nm][0]]
            fb_txt = texts_by_name[pool_meta["filler_map"][nm][1]]
            seqs = {
                "loop": ids[:cut] + PHRASE_IDS * n + ids[cut:],
                "fillerA": ids[:cut] + filler_ids(tk, fa_txt, n) + ids[cut:],
                "fillerB": ids[:cut] + filler_ids(tk, fb_txt, n) + ids[cut:],
            }
            # S_stimulus (TZ §7)
            assert len(seqs["loop"]) == T + L_n
            assert len(seqs["fillerA"]) == T + L_n
            assert len(seqs["fillerB"]) == T + L_n
            if seqs["fillerA"] == seqs["fillerB"]:
                raise SystemExit(f"{nm} n={n}: fillerA == fillerB (S_stimulus fail)")
            if seqs["loop"] == seqs["fillerA"] or seqs["loop"] == seqs["fillerB"]:
                raise SystemExit(f"{nm} n={n}: filler == loop (S_stimulus fail)")
            phrase_g = tuple(PHRASE_IDS)
            for cond in ("fillerA", "fillerB"):
                blk = seqs[cond][cut:cut + L_n]
                if any(tuple(blk[j:j + 9]) == phrase_g
                       for j in range(len(blk) - 8)):
                    raise SystemExit(f"{nm} n={n} {cond}: луп-9-грамма в филлере "
                                     f"(S_stimulus fail)")
            cell = {"L_n": L_n}
            for cond, seq in seqs.items():
                r = fwd(m, seq, torch)
                n_fwd += 1
                s = cond_series(r, T, cut, L_n)
                ce = r["ce"]
                s["ce_block"] = round(float(np.mean(ce[cut - 1:cut + L_n - 1])), 6)
                s["i_loop"] = round(float(np.sum(ce[cut - 1:cut + L_n - 1])), 6)
                cell[cond] = s
                logline(f"  {nm} n={n} {cond}: len={len(seq)} "
                        f"CE={float(ce.mean()):.4f} block_CE={s['ce_block']:.3f}")
                json.dump(store, open(STORE, "w"), default=float)
            # контрасты (TZ §2): из cos_raw одной истины
            pc = post_series(store[nm]["ctrl"], cut, 0)
            for L in LAYERS:
                pl = post_series(cell["loop"], cut, L_n)[str(L)]
                pa = post_series(cell["fillerA"], cut, L_n)[str(L)]
                pb = post_series(cell["fillerB"], cut, L_n)[str(L)]
                cell.setdefault("e_spec", {})[str(L)] = \
                    [round(a - b, 6) for a, b in zip(pl, pa)]
                cell.setdefault("e_generic", {})[str(L)] = \
                    [round(a - b, 6) for a, b in zip(pa, pc[str(L)])]
                cell.setdefault("e_null", {})[str(L)] = \
                    [round(a - b, 6) for a, b in zip(pa, pb)]
            store[nm][f"n{n}"] = cell
            tr.log({"doc": nm, "n": n, "n_fwd": n_fwd,
                    "block_ce_loop": cell["loop"]["ce_block"]},
                   step=int(time.time() - t0))
            json.dump(store, open(STORE, "w"), default=float)
        tstage(tr, t0, f"doc_{nm}")
    tr.log({"n_fwd": n_fwd, "mode": mode}, step=int(time.time() - t0))
    tstage(tr, t0, "forwards")

    if mode == "smoke":
        # смоук-якоря (TZ §14.1)
        ok = True
        for dn in sorted(k for k in store if k != "__config__"):
            ce_ctrl = float(np.mean(store[dn]["ctrl"]["ce"]))
            if not (2.0 <= ce_ctrl <= 4.0):
                ok = False
                logline(f"  SMOKE FAIL: {dn} ctrl CE={ce_ctrl:.3f} вне [2,4]")
            c2 = store[dn]["n2"]
            lb, fb = c2["loop"]["ce_block"], c2["fillerA"]["ce_block"]
            logline(f"  SMOKE {dn}: ctrl CE={ce_ctrl:.3f}, loop_block CE={lb:.3f} "
                    f"(гейт 1.0 — на n=16), filler_block CE={fb:.3f}")
            if not (fb >= 2.0):
                ok = False
                logline(f"  SMOKE FAIL: {dn} filler block CE < 2.0")
            en = c2["e_null"]["23"]
            if not np.isfinite(en).all():
                ok = False
                logline(f"  SMOKE FAIL: {dn} e_null содержит NaN/Inf")
        logline("SMOKE_PASS" if ok else "SMOKE_FAIL")
        tr_done("ok" if ok else "crash")
        _log.write_to(RUNLOG)
        return 0 if ok else 1

    # полный прогон: npz (из store, M7) → анализ → артефакты
    npz = _series_from_store(store, doses)
    np.savez(SERIES, **npz)
    tstage(tr, t0, "series_saved")
    an = analyze(store)
    manifest = dict(
        generated_by="AT8_2_R2.py", run_revision=RUN_REVISION,
        machine=MACHINE_REVISION,
        script_sha256=sha256_file(os.path.abspath(__file__)),
        store_file_sha256=sha256_file(STORE),
        series_npz_sha256=sha256_file(SERIES),
        model=MODEL, attn_impl=attn_impl,
        threads=os.environ.get("OMP_NUM_THREADS", "?"),
        seed=SEED, B_boot=B_BOOT, mode=mode,
        phrase_token_check=True,
        tokenizer=MODEL,
        pool_fps=pool_meta["pool_fps"],
        n_candidates=pool_meta["n_candidates"],
        n_passed=pool_meta["n_passed"],
        baseline_loop_metrics=pool_meta["baseline_loop_metrics"],
        filler_source=pool_meta["filler_source"],
        docs_sha256=cfg_fp["docs_sha256"],
        doc_lens={d["name"]: len(d["ids"]) for d in docs},
        filler_map=pool_meta["filler_map"],
        n_fwd=n_fwd,
        claim_sha256=None,
        wall_s=round(time.time() - t0, 1),
    )
    tr.log({"stage": "analysis", "stage_s": round(time.time() - t0, 1)},
           step=int(time.time() - t0))
    _write_artifacts(an, manifest, mode)
    v = an["verdict"]
    logline(f"VERDICT: {v['value']} (n*={v['n_star']}, tags={v['tags']})")
    for r in v["reasons"]:
        logline(f"  {r}")
    logline(f"δ: near_L23={an['dmin_near']:.4f} far_L23={an['dmin_far']:.4f} "
             f"near_by_layer={json.dumps({k: round(x, 4) for k, x in an['dmin_near_layer'].items()})}")
    logline(f"inside pair={an['inside_pass']}/{an['n_docs']} "
             f"abs={an['inside_abs_pass']} | i_loop={an['i_loop']} | "
             f"seam={an['seam']}")
    logline(f"AT8_2_R2_DONE verdict={v['value']} mode={mode} "
             f"(wall {time.time() - t0:.0f}s)")
    tr_done("ok")
    return 0

# ----------------------------- reanalysis (disk-only) -----------------------------


def reanalysis_main():
    t0 = time.time()
    store = json.load(open(STORE))
    if "__config__" not in store:
        raise SystemExit("store без __config__ — не R2-store")
    an = analyze(store, doses=store["__config__"]["n_dose"])
    manifest = dict(
        generated_by="AT8_2_R2.py --reanalysis", run_revision=RUN_REVISION,
        machine=MACHINE_REVISION,
        script_sha256=sha256_file(os.path.abspath(__file__)),
        store_file_sha256=sha256_file(STORE),
        series_npz_sha256=sha256_file(SERIES) if os.path.exists(SERIES) else None,
        mode="reanalysis", claim_sha256=None,
        wall_s=round(time.time() - t0, 1),
    )
    v = an["verdict"]
    _write_artifacts(an, manifest, "reanalysis")
    logline(f"VERDICT: {v['value']} (n*={v['n_star']}, tags={v['tags']})")
    for r in v["reasons"]:
        logline(f"  {r}")
    logline(f"AT8_2_R2_REANALYSIS_DONE verdict={v['value']} "
             f"(wall {time.time() - t0:.1f}s)")
    return 0

# ----------------------------- синтетические тесты (TZ §14.2, машина r2) -----------------------------


def _synth_store(kind, n_docs=14):
    """Синтетический store (TZ §14.2): e_spec/e_generic/e_null по (док, n, слой).
    Профили (машина r2: near/far-статистики, δ через тот же пайплайн):
    null    — e_spec ~ N(0, 0.01), e_generic ~ N(0, 0.005) → NO_LOOP
    decays  — e_spec = exp(−d/8) + N(0, 0.005) → DECAYS
    persists— e_spec = 0.3 + N(0, 0.005) → PERSISTS
    generic — e_spec ~ N(0, 0.005), e_generic = 0.6 → GENERIC"""
    rng = np.random.default_rng(0)
    store = {"__config__": {"n_dose": N_DOSE}}
    for i in range(n_docs):
        dn = f"syn_{i}"
        store[dn] = {"T": 256, "cut": 128, "domain": "syn"}
        store[dn]["ctrl"] = {f"L{L}": dict(cos_raw=[0.9] * 500, cos_state=[0.9] * 500,
                                            norm_delta=[1.0] * 500) for L in LAYERS}
        store[dn]["ctrl"]["ce"] = [3.0] * 256
        for n in N_DOSE:
            L_n = 9 * n
            cell = {"L_n": L_n}
            for cond in ("loop", "fillerA", "fillerB"):
                cell[cond] = {f"L{L}": dict(cos_raw=[0.9] * (256 + L_n),
                                             cos_state=[0.9] * (256 + L_n),
                                             norm_delta=[1.0] * (256 + L_n))
                              for L in LAYERS}
                cell[cond]["ce"] = [3.0] * (256 + L_n)
                cell[cond]["ce_block"] = 0.5 if cond == "loop" else 4.0
                cell[cond]["i_loop"] = float(2.06 + 0.38 * (L_n - 9))
            es = {str(L): [] for L in LAYERS}
            eg = {str(L): [] for L in LAYERS}
            en = {str(L): [] for L in LAYERS}
            for d in range(H):
                for L in LAYERS:
                    if kind == "null":
                        s = rng.normal(0, 0.01)
                    elif kind == "decays":
                        s = math.exp(-d / 8.0) + rng.normal(0, 0.005)
                    elif kind == "persists":
                        s = 0.3 + rng.normal(0, 0.005)
                    elif kind == "generic":
                        s = rng.normal(0, 0.005)
                    else:
                        s = 0.0
                    es[str(L)].append(float(s))
                    eg[str(L)].append(0.6 if kind == "generic"
                                      else float(rng.normal(0, 0.005)))
                    en[str(L)].append(float(rng.normal(0, 0.01)))
            cell["e_spec"] = es
            cell["e_generic"] = eg
            cell["e_null"] = en
            store[dn][f"n{n}"] = cell
    return store


def test_synth_main():
    cases = [
        ("null", "NO_LOOP_SPECIFIC_EFFECT"),
        ("decays", "LOOP_RESPONSE_DECAYS"),
        ("persists", "LOOP_RESPONSE_PERSISTS"),
        ("generic", "GENERIC_INSERTION_EFFECT"),
    ]
    ok = 0
    for kind, expected in cases:
        store = _synth_store(kind)
        an = analyze(store)
        v = an["verdict"]
        good = v["value"] == expected
        ok += good
        logline(f"  synth {kind}: {v['value']} (ожид. {expected}) "
                f"{'PASS' if good else 'FAIL'} | δ_near={an['dmin_near']:.4f} "
                f"near_ci16=[{an['bootstrap']['16']['near']['ci_low']:.4f},"
                f"{an['bootstrap']['16']['near']['ci_high']:.4f}]")
    store = _synth_store("null")
    an = analyze(store)
    v = compute_verdict(an["bootstrap"], an["dmin_near"], an["dmin_far"],
                        an["near"], an["far"], an["near_generic"],
                        an["near_by_layer"], an["dmin_near_layer"],
                        3, 14, an["i_loop"], 14)
    good = v["value"] == "INSENSITIVE_CELL"
    ok += good
    logline(f"  synth insensitive: {v['value']} {'PASS' if good else 'FAIL'}")
    v = compute_verdict(an["bootstrap"], an["dmin_near"], an["dmin_far"],
                        an["near"], an["far"], an["near_generic"],
                        an["near_by_layer"], an["dmin_near_layer"],
                        14, 14, an["i_loop"], 14, gates_ok=False)
    good = v["value"] == "NOT_RESOLVED"
    ok += good
    logline(f"  synth gates-fail: {v['value']} {'PASS' if good else 'FAIL'}")
    logline(f"SYNTH_TEST: {ok}/6 PASS")
    return 0 if ok == 6 else 1

# ----------------------------- main -----------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--test-synth" in args:
        sys.exit(test_synth_main())
    if "--reanalysis" in args:
        sys.exit(reanalysis_main())
    if "--smoke" in args:
        sys.exit(run_main(mode="smoke"))
    sys.exit(run_main())
