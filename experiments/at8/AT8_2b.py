#!/usr/bin/env python
"""AT8_2b (TZ v1.6, пре-рег 19.11): второй стимул (loop2) — «повторение» vs
«фраза», факторный дизайн 2×2.

ТЗ (пре-рег): docs/AT8_2b_TZ.md v1.6 (ревью r1–r6 закрыты). Машина: 2b-r1.

Дизайн (TZ §1–§2):
  H_phr (фраза-канал):  E_phr = L1 − L2 (простой эффект на периодических
      строках) против N_PHR = S1' − S2' (терм-непересекающийся нуль);
      наблюдаемая norm_delta, первичная доза n=2 (TZ §7.3, C1).
  H_rep (повторение-канал): REP = (L1+L2−S1−S2)/2 против N_perm =
      (S1+S2−S1'−S2')/2 (сматчен, Var = σ_S²); наблюдаемая cos_raw,
      первичная доза n=16.
  INT = D_L − D_S — report-only + тег INTERACT_SIGNIFICANT (эвристика без
      калибровки, F5). PHR_main/N_content/(S1−S1')/e_spec1/e_spec2/e_null/
      e_rep_legacy — legacy-диагностика.
  R2-store read-only (sha 78419a69…): ctrl/L1/fillerA/fillerB — с диска.
  S_cross_store (TZ §10.2): свежие форварды (wiki_0, n=2 И n=16, loop) vs
      стор; свежий ряд = cond_series R2 verbatim (round 6); гейт =
      bit-identity округлённых значений на всём ряду (5 слоёв,
      cos_raw/cos_state/norm_delta); rel 1e-6 — диагностика с полом 5e-7;
      провал → fallback (перегон 4 R2-условий свежими, НЕ стоп; смешение
      численных путей запрещено).
  Новые форварды: 5 условий (L2, S1, S2, S1', S2') × 14 доков × 2 дозы = 140
      (fallback: + 4 R2-условия → 238).

Моды:
  (нет флага)     полный прогон (140 форвардов; resume)
  --smoke         wiki × 2 дока × 2 дозы (20) + S_cross_store (2) = 22
                  форварда + все смоук-якоря TZ §15 → решение ДО main
                  (S_cross_store + маржа G_inside_rep, M2-r5) в
                  results/AT8/AT8_2b_decision.json
  --reanalysis    disk-only: store → verdict + JSON (машина 2b-r1)
  --test-synth    10 синтетических сценариев вердикт-машины (без модели, TZ §16)

Артефакты: results/AT8/AT8_2b{,_store,_manifest}.json, AT8_2b_series.npz,
AT8_2b_decision.json (smoke→main), logs/AT8/AT8_2b_run.log (trackio
at8_2b_r1). Store/npz >10 MB — gitignored + бэкап /data/prism_backups/at8_2b/.

Привилегия 16.11: prism/explog.py + prism/nulls_and_stats.py (по файлу —
prism/__init__ тянет matplotlib, которого нет в runtime-окружении).
"""

import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import zlib
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
ExpLog = _explog.ExpLog  # привилегия 16.11 (общий логгер)
# TZ §12 (r2d-2.5): CI95 всех колонок ОДНИМИ индексами доков — многоколонный
# парный бутстреп (_boot_multi ниже); одиноколонный shared paired_bootstrap
# здесь не применим (он ресэмплирует колонки независимо).

# ----------------------------- константы (TZ v1.6) -----------------------------
MODEL = "/data/Qwen/Qwen3.5-0.8B"
LAYERS = [6, 12, 18, 19, 23]            # TZ §6: тот же набор R2
PRIMARY_LAYER = 23                      # TZ §6: primary по преемственности
PHRASE1 = " The class will be held at the park."        # loop1 (R2, frozen)
PHRASE1_IDS = [561, 523, 668, 381, 5467, 506, 279, 5926, 13]
PHRASE2 = " Three new buses wait near a quiet station."  # loop2 (пре-рег v1.0)
PHRASE2_IDS = [14095, 491, 32275, 3655, 3043, 264, 11012, 7803, 13]
DOSES = [2, 16]                         # TZ §2
PHR_DOSE = 2                            # TZ §9: E_phr primary n=2
REP_DOSE = 16                           # TZ §9: REP primary n=16
PHR_OBS = "norm_delta"                  # TZ §7.3 (C1): E_phr → norm_delta
REP_OBS = "cos_raw"                     # TZ §7.3 (C1): REP → cos_raw
H = 64
NEAR_END = 15                           # near: d ∈ [0, 15]
FAR_START = H - 16                      # far:  d ∈ [48, 63]
N_W = NEAR_END + 1                      # 16 точек near
F_W = H - FAR_START                     # 16 точек far
GRAM = 9
DOMAIN_ORDER = ["wiki", "math", "code", "chat"]
PINS = {
    d: os.path.join(ROOT, "data", "calib_pins",
                    f"{d}___data_Qwen_Qwen3.5_0.8B__n12__min192__max512__plain.json")
    for d in DOMAIN_ORDER
}
T_MIN, T_MAX = 192, 512
GRAM_DUP_MAX = 1
N_PER_DOMAIN = 4
DOCS_MIN = 12
DOMAINS_MIN = 4
DOCS14 = ([f"wiki_{i}" for i in range(4)] + ["math_0", "math_1"]
          + [f"code_{i}" for i in range(4)] + [f"chat_{i}" for i in range(4)])

# R2-store (TZ §10.1: read-only, sha-пин)
R2_STORE = os.path.join(ROOT, "results", "AT8", "AT8_2_R2_store.json")
R2_STORE_SHA = ("78419a69a2edbc767248b3a8f9512e1961ea82c154970772c66c35e2a42b16c0")
# TZ §8: legacy-δ anchor «0.1480 / 0.0332» (4-значные литералы ТЗ) — 6-значная
# форма фактических выводов R2-пайплайна (claim delta_min.near_L23/far_L23);
# чек — против 6-значных значений (иначе 4-значное округление не проходит 1e-6)
LEGACY_NEAR_ANCHOR = 0.148001
LEGACY_FAR_ANCHOR = 0.033238
ANCHOR_TOL = 1e-6
# TZ §15: e_spec1 cross-check «0.1020 ± 1e-6» — 6-значная форма R2-claim near['16']
ESPEC1_NEAR_N16 = 0.10203

# Сиды (TZ §3, единый литерал; m1-r5: 32 бита crc32, БЕЗ % 99991)
def seed_str(i, doc, n):
    return f"S{i}|{doc}|{n}"

SEED_STRINGS = [seed_str(i, doc, n) for i in (1, 2, 3, 4)
                for doc in DOCS14 for n in (2, 16)]
SEEDS_112 = [zlib.crc32(s.encode()) for s in SEED_STRINGS]
SEED_SHA_PIN = ("3a98ef6f950a44134ceab378798fe0deeab7bbd06f1b7776ebb5960cb3e20692")
# i ∈ {1,2,3,4} = (S1, S2, S1', S2'); S1'/S2' — независимый draw от L1/L2
SHUFFLE_DEF = {"S1": (1, PHRASE1_IDS), "S2": (2, PHRASE2_IDS),
               "S1p": (3, PHRASE1_IDS), "S2p": (4, PHRASE2_IDS)}
COND_NEW = ["L2", "S1", "S2", "S1p", "S2p"]      # новые условия (5)
COND_R2KEY = {"L1": "loop", "fillerA": "fillerA", "fillerB": "fillerB"}
# 12 контрастов finite-гейта (M5/M3-r5: 2 вердиктных + 2 нуля + INT + 7 legacy)
CONTRASTS = ["E_phr", "REP", "INT", "PHR_main", "N_PHR", "N_perm",
             "N_content", "S1mS1p", "e_spec1", "e_spec2", "e_null",
             "e_rep_legacy"]

SEED = 0
B_BOOT = 2000
INSIDE_PASS_FRAC = 0.75                 # TZ §10.6: ceil(0.75×14) = 11/14
MARGIN_N2 = 0.3                         # TZ §10.6: медиана min(m₁,m₂) на n=2
RUN_REVISION = "r1"
MACHINE_REVISION = "r1"
MACHINE_SUBREVISION = "r1"
# TZ §10.1: численный путь R2 (провенанс-пин; защита — гейт S_cross_store)
R2_ENV_PIN = {"attn_impl": "sdpa", "py": "3.12.9", "torch": "2.14.1",
              "omp": "6", "use_cache": "False", "grad_enabled": "False"}
BACKUP_DIR = "/data/prism_backups/at8_2b"

OUT_DIR = os.path.join(ROOT, "results", "AT8")
CLAIM = os.path.join(OUT_DIR, "AT8_2b.json")
STORE = os.path.join(OUT_DIR, "AT8_2b_store.json")
MANIFEST = os.path.join(OUT_DIR, "AT8_2b_manifest.json")
SERIES = os.path.join(OUT_DIR, "AT8_2b_series.npz")
RUNLOG = os.path.join(ROOT, "logs", "AT8", "AT8_2b_run.log")
DECISION = os.path.join(OUT_DIR, "AT8_2b_decision.json")

# ----------------------------- лог + trackio -----------------------------
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
    return hashlib.sha256(np.asarray(ids, dtype="int64").tobytes()).hexdigest()[:16]


def init_trackio(t0, script_path, mode):
    # AGENTS: TRACKIO_DIR задаёт run_one.sh; прямой запуск — fail-loud.
    if not os.environ.get("TRACKIO_DIR"):
        raise SystemExit(
            "trackio: TRACKIO_DIR не задан — запускать через ./run_one.sh "
            "(он экспортирует TRACKIO_DIR=$PRISM_DIR/.trackio); для "
            "--test-synth/--reanalysis задайте TRACKIO_DIR вручную.")

    class _NoopTr:
        """trackio в .venv сломан (huggingface_hub 1.28, ImportError httpx —
        зафиксировано в docstring trackio_run.py): тайм-метки ведёт wrapper
        trackio_run.py (системный python3), внутренний — no-op."""
        def log(self, *a, **k): pass
        def finish(self, *a, **k): pass

    try:
        import trackio
        tr = trackio.init(project="prism", name="at8_2b_r1", group="AT8_2b",
                          config={"device": "cpu", "model": MODEL, "mode": mode,
                                  "log": os.path.abspath(RUNLOG),
                                  "revision": RUN_REVISION,
                                  "machine_revision": MACHINE_REVISION,
                                  "script_sha": sha256_file(script_path),
                                  "seed": SEED})
    except Exception as e:  # noqa: BLE001 — трекинг не должен убивать прогон
        logline(f"TRACKIO_UNAVAILABLE (внутренний): {type(e).__name__}: {e} — "
                "тайм-метки ведёт wrapper trackio_run.py")
        tr = _NoopTr()
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

# ----------------------------- пул (TZ §4: заморожен, из R2) -----------------------------


def max_gram_count(ids):
    g = [tuple(ids[j:j + GRAM]) for j in range(len(ids) - GRAM + 1)]
    return max(Counter(g).values()) if g else 0


def load_pool(tk):
    """Детерминированная пересборка пула из замороженных пинов (то же, что R2:
    12 кандидатов/домен → 9g-фильтр → первые 4/домен). Пул НЕ пересобирается
    по существу — пересборка только для токенов; идентичность со R2-store
    проверяет check_pool (T/cut/docs_sha256/pool_fps/filler_map)."""
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
                filler_map=filler_map,
                filler_source="same_pool_cross_doc (из R2-store)")
    return docs, meta


def check_pool(pool, pool_meta, r2store):
    """TZ §4: пул заморожен — идентичность со R2-store (read-only-проверка).
    D1 (20.11): docs_sha256 (id-уровень) — WARNING, не стоп: тексты уже
    верифицированы pool_fps (байт-fp), длины — T/cut; id-хаши — прокси
    идентичности ОКРУЖЕНИЯ, а прямой гейт на окружение — S_cross_store
    (свежий форвард vs store, bit-identity). В текущем .venv (py3.13/
    torch 2.13 vs R2 py3.12/torch 2.14) id-последовательности пула не
    воспроизводятся, хотя тексты/длины/фраза-иды идентичны."""
    cfg = r2store["__config__"]
    docs = {d["name"]: d for d in pool}
    if sorted(docs) != sorted(DOCS14):
        raise SystemExit(f"pool: имена 14 доков != TZ §4: {sorted(docs)}")
    if pool_meta["pool_fps"] != cfg["pool_fps"]:
        raise SystemExit("pool: pool_fps != R2-store — стоп (TZ §4: пул заморожен)")
    if pool_meta["filler_map"] != cfg["filler_map"]:
        raise SystemExit("pool: filler_map != R2-store — стоп (TZ §4)")
    n_sha_diff = 0
    for name in DOCS14:
        d = docs[name]
        if len(d["ids"]) != r2store[name]["T"]:
            raise SystemExit(f"pool: {name} T={len(d['ids'])} != store "
                             f"T={r2store[name]['T']} — стоп")
        if len(d["ids"]) // 2 != r2store[name]["cut"]:
            raise SystemExit(f"pool: {name} cut != R2-store — стоп")
        if sha256_ids(d["ids"]) != cfg["docs_sha256"][name]:
            n_sha_diff += 1
            logline(f"pool: {name} docs_sha256 != R2-store (D1: окружение "
                    f"drift — id-уровень; решение примет S_cross_store)")
    logline("pool check: PASS (14 доков: T/cut/pool_fps/filler_map == R2-store"
            + (f"; docs_sha256: {n_sha_diff}/14 DIFF — D1 warning, не стоп)"
               if n_sha_diff else "; docs_sha256: 14/14 match"))
    return n_sha_diff

# ----------------------------- R2-store (TZ §10.1, read-only) -----------------------------


def check_r2_store():
    """Integrity-гейт (TZ §10.1): sha + fingerprint + пул-гейт. Провал → стоп."""
    if sha256_file(R2_STORE) != R2_STORE_SHA:
        raise SystemExit(f"R2-store integrity: sha256 {sha256_file(R2_STORE)[:16]}… "
                         f"!= {R2_STORE_SHA[:16]}… (TZ §10.1) — стоп")
    s = json.load(open(R2_STORE))
    cfg = s.get("__config__", {})
    for key, val in (("model", MODEL), ("layers", LAYERS),
                     ("phrase_ids", PHRASE1_IDS), ("H", H)):
        if cfg.get(key) != val:
            raise SystemExit(f"R2-store fingerprint: {key}={cfg.get(key)!r} != "
                             f"{val!r} — стоп")
    if not set(DOSES) <= set(cfg.get("n_dose", [])):
        raise SystemExit("R2-store fingerprint: n_dose не покрывает {2,16} — стоп")
    docs = [k for k in s if k != "__config__"]
    if sorted(docs) != sorted(DOCS14):
        raise SystemExit(f"R2-store: 14 имён доков != TZ §4 — стоп")
    per_dom = Counter(s[k]["domain"] for k in docs)
    if len(per_dom) < DOMAINS_MIN or any(v < 2 for v in per_dom.values()):
        raise SystemExit(f"R2-store: пул-гейт (4 домена, ≥2/домен) провален: "
                         f"{dict(per_dom)} — стоп")
    logline(f"R2-store integrity: PASS (sha {R2_STORE_SHA[:16]}…, "
            f"14 доков, 4 домена, {dict(per_dom)})")
    return s


def get_cell(store2b, r2store, dn, n, cond, fallback):
    """Объединённый доступ к условию: новые — из 2b-store; R2-условия — из
    R2-store (норма) или из 2b-store (fallback: свежие). L1 ↔ store['loop'].
    ctrl — doc-level (без дозы)."""
    if cond in COND_NEW:
        return store2b[dn][f"n{n}"][cond]
    if cond == "ctrl":
        return (store2b[dn]["ctrl"] if fallback
                else r2store[dn]["ctrl"])
    key = COND_R2KEY[cond]
    return ((store2b[dn][f"n{n}"][key] if fallback
             else r2store[dn][f"n{n}"][key]))


def legacy_delta(r2store, doses):
    """TZ §8: legacy-δ (e_null, cos_raw, L23) — R2-пайплайн verbatim (near =
    p90|d∈[0,15]|, far = mean|d∈[48,63]|; δ = 95th pct по 14 докам × дозам)."""
    docs = sorted(k for k in r2store if k != "__config__")
    Ls = str(PRIMARY_LAYER)
    nn, fn = [], []
    for dn in docs:
        for n in doses:
            en = r2store[dn][f"n{n}"]["e_null"][Ls]
            nn.append(_near(en))
            fn.append(_far(en))
    return float(np.percentile(nn, 95)), float(np.percentile(fn, 95))


def check_legacy_anchor(r2store):
    """TZ §8: integrity-якорь (fail-loud): 4 дозы verbatim, независимо от
    дозового списка 2b [2,16] (m2-r2: по 2 дозам = 0.1533/0.0306 — ложный
    STOP-ловушка; дегенерация не глосуется)."""
    d_near, d_far = legacy_delta(r2store, [2, 4, 8, 16])
    ok = (abs(d_near - LEGACY_NEAR_ANCHOR) <= ANCHOR_TOL
          and abs(d_far - LEGACY_FAR_ANCHOR) <= ANCHOR_TOL)
    if not ok:
        raise SystemExit(f"legacy-δ anchor: near={d_near:.6f}/"
                         f"{LEGACY_NEAR_ANCHOR}, far={d_far:.6f}/"
                         f"{LEGACY_FAR_ANCHOR} (±{ANCHOR_TOL}, 4 дозы verbatim) — "
                         f"store повреждён/подменён — стоп (TZ §8)")
    logline(f"legacy-δ anchor: PASS (near={d_near:.6f}, far={d_far:.6f}; "
            f"TZ-литералы 0.1480/0.0332; 4 дозы × 14 доков = 56 значений)")
    return d_near, d_far


def espec1_crosscheck(r2store):
    """TZ §15: e_spec1 воспроизводит R2 near (L23, n16: 0.1020 ± 1e-6) —
    ТОЛЬКО дисковая переагрегация из стора (расхождение численного пути не
    ловит — для этого S_cross_store)."""
    docs = sorted(k for k in r2store if k != "__config__")
    Ls = str(PRIMARY_LAYER)
    val = float(np.mean([_near(r2store[dn]["n16"]["e_spec"][Ls]) for dn in docs]))
    if abs(val - ESPEC1_NEAR_N16) > ANCHOR_TOL:
        raise SystemExit(f"e_spec1 cross-check: {val:.6f} != {ESPEC1_NEAR_N16} "
                         f"(±{ANCHOR_TOL}) — стоп")
    logline(f"e_spec1 cross-check: PASS ({val:.6f} = R2 near n16)")
    return val

# ----------------------------- форвард + ряды (R2 verbatim, TZ §5) -----------------------------


def fwd(m, ids, torch):
    """Один forward: per-position CE + hidden states 5 слоёв. use_cache=False —
    R2-путь (запинен, TZ §10.1)."""
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
    """Позиционные ряды условия. R2 VERBATIM (вкл. round 6) — S_cross_store
    зависит от bit-identity.
    cos_raw[p]    = cos(Δh_p, Δh_{p+1}),  Δh_p = h_{p+1} − h_p
    cos_state[p]  = cos(u_p, u_{p+1}),  u = unit(h)
    norm_delta[p] = ‖Δh_p‖
    p = 0..T'-2; post-индекс d: cut + L_n + d (loop/filler) или cut + d (ctrl)."""
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


def post_series(cell, cut, L_n, obs="cos_raw"):
    """post-серии (длина H) из ряда obs: loop/filler — сдвиг L_n, ctrl — 0.
    obs="cos_raw" — R2 verbatim; "norm_delta" — та же геометрия (TZ §7.3:
    наблюдаемые per контраст)."""
    out = {}
    for L in LAYERS:
        base = cell[f"L{L}"][obs]
        start = cut + (L_n if L_n > 0 else 0)
        out[str(L)] = [base[start + d] for d in range(H)]
    return out


def filler_ids(tk, text, n):
    return tk(text, max_length=T_MAX, truncation=True,
              add_special_tokens=False)["input_ids"][:9 * n]

# ----------------------------- стимулы (TZ §3) -----------------------------


def sha256_tokens(ids):
    """m2-r5: sha256 токен-массива (полный hex)."""
    return hashlib.sha256(np.asarray(ids, dtype="int64").tobytes()).hexdigest()


def _shuffle_block(phrase_ids, i, doc, n):
    """L_i(n)[π_{i,doc,n}], π = default_rng(crc32(f'S{i}|{doc}|{n}'))
    .permutation(9n) — TZ §3 (per-doc сид; полные 32 бита, без % 99991)."""
    L = phrase_ids * n
    pi = np.random.default_rng(zlib.crc32(seed_str(i, doc, n).encode())).permutation(len(L))
    return [L[int(k)] for k in pi]


def new_seqs(doc, cut, n):
    """Новые условия (док, n): L2 + S1/S2/S1p/S2p. {cond: seq}."""
    out = {"L2": doc["ids"][:cut] + PHRASE2_IDS * n + doc["ids"][cut:]}
    for cond, (i, ph) in SHUFFLE_DEF.items():
        out[cond] = (doc["ids"][:cut] + _shuffle_block(ph, i, doc["name"], n)
                     + doc["ids"][cut:])
    return out


def stimulus_checks(doc, n, out):
    """S_stimulus + S_shuffle bug-guard (TZ §3/§10.3-4): длина, multiset,
    период-9-чек (fail-loud assert), max 9-gram, sha256 токен-массива.
    Возвращает {cond: info} для артефакта."""
    T = len(doc["ids"])
    cut = T // 2
    L_n = 9 * n
    info = {}
    for cond, seq in out.items():
        if len(seq) != T + L_n:
            raise SystemExit(f"{doc['name']} n={n} {cond}: len={len(seq)} != "
                             f"T+L_n={T + L_n} (S_stimulus fail)")
        blk = seq[cut:cut + L_n]
        ref = (PHRASE2_IDS if cond == "L2" else SHUFFLE_DEF[cond][1]) * n
        if sorted(blk) != sorted(ref):
            raise SystemExit(f"{doc['name']} n={n} {cond}: multiset != L_i — "
                             f"S_stimulus fail")
        period9 = all(blk[k] == blk[k + 9] for k in range(len(blk) - 9))
        if cond != "L2" and period9:
            raise SystemExit(f"{doc['name']} n={n} {cond}: период-9-чек FAIL — "
                             f"баг перестановки (S_shuffle bug-guard, TZ §3)")
        info[cond] = dict(seq_sha256=sha256_tokens(blk),
                          max_gram9=max_gram_count(blk),
                          period9=period9)
    return info

# ----------------------------- контрасты (TZ §7.2) -----------------------------


OBS_LIST = ("cos_raw", "norm_delta")


def build_contrasts(cell_view, cut, L_n):
    """12 контрастных массивов per наблюдаемая (TZ §7.2; m4-r2: массивы
    сохраняются в store; m4/G14: формат {C: {L: {obs: arr(64)}}})."""
    post = {}
    for c in ("L1", "L2", "S1", "S2", "S1p", "S2p", "fillerA", "fillerB"):
        for obs in OBS_LIST:
            post[(c, obs)] = post_series(cell_view[c], cut, L_n, obs)

    coef = {
        "E_phr": {"L1": 1.0, "L2": -1.0},
        "REP": {"L1": 0.5, "L2": 0.5, "S1": -0.5, "S2": -0.5},
        "INT": {"L1": 0.5, "S1": -0.5, "L2": -0.5, "S2": 0.5},
        "PHR_main": {"L1": 0.5, "L2": -0.5, "S1": 0.5, "S2": -0.5},
        "N_PHR": {"S1p": 1.0, "S2p": -1.0},
        "N_perm": {"S1": 0.5, "S2": 0.5, "S1p": -0.5, "S2p": -0.5},
        "N_content": {"S1": 1.0, "S2": -1.0},
        "S1mS1p": {"S1": 1.0, "S1p": -1.0},
        "e_spec1": {"L1": 1.0, "fillerA": -1.0},
        "e_spec2": {"L2": 1.0, "fillerB": -1.0},
        "e_null": {"fillerA": 1.0, "fillerB": -1.0},
        "e_rep_legacy": {"L1": 0.5, "fillerA": -0.5,
                         "L2": 0.5, "fillerB": -0.5},
    }
    out = {}
    for C, cs in coef.items():
        out[C] = {}
        for obs in OBS_LIST:
            out[C][obs] = {
                str(L): [round(float(sum(w * post[(c, obs)][str(L)][d]
                                         for c, w in cs.items())), 6)
                         for d in range(H)]
                for L in LAYERS}
    return out

# ----------------------------- статистики -----------------------------


def _near(vals):
    """90th pct |vals| по первым N_W позициям (TZ §8)."""
    return float(np.percentile(np.abs(np.asarray(vals)[:N_W]), 90))


def _far(vals):
    """mean |vals| по последним F_W позициям (TZ §8)."""
    return float(np.mean(np.abs(np.asarray(vals)[FAR_START:])))


def _dose_tag_2b(i_loop):
    """TZ §9: 2-дозное правило (переписано под {2,16}, F2): I16/I2 ≤ 2 →
    DOSE_FLAT; > 2 → DOSE_MONOTONIC; не-finite или ≤ 0 → DOSE_NOT_RESOLVED
    (DOSE_SATURATING удалён — недостижим без средней дозы)."""
    v2 = i_loop.get(str(PHR_DOSE))
    v16 = i_loop.get(str(REP_DOSE))
    if v2 is None or v16 is None:
        return "DOSE_NOT_RESOLVED"
    if (not np.isfinite(v2)) or (not np.isfinite(v16)) or v2 <= 0 or v16 <= 0:
        return "DOSE_NOT_RESOLVED"
    r = v16 / v2
    return "DOSE_FLAT" if r <= 2.0 else "DOSE_MONOTONIC"


def _boot_multi(a2col):
    """doc-bootstrap CI95 ВСЕХ колонок ОДНИМИ доками (TZ §12, пара́тность
    r2d-2.5): один rng-поток (seed 0), B=2000; на каждом шаге один набор
    индексов для всех колонок. Эквивалентен R2-_boot (по-колонки, тот же
    поток)."""
    a = np.asarray(a2col, dtype="float64")
    n, nc = a.shape
    rng = np.random.default_rng(seed=SEED)
    boots = np.empty((B_BOOT, nc))
    for b in range(B_BOOT):
        idx = rng.integers(0, n, size=n)
        boots[b] = a[idx].mean(axis=0)
    out = {}
    for j in range(nc):
        out[j] = dict(point=float(a[:, j].mean()),
                      ci_low=float(np.percentile(boots[:, j], 2.5)),
                      ci_high=float(np.percentile(boots[:, j], 97.5)),
                      B=B_BOOT)
    return out

# ----------------------------- вердикт-машина (TZ §11, порядок жёсткий) -----------------------------


def compute_verdict_2b(an):
    """Машина 2b-r1: чистая функция от выводов analyze_2b (и синтетики).
    Дерево TZ §11: гейты → G_inside_rep (по дозам) → вердиктная клетка
    (ровно 2 теста) → substructure → INT (отчёт) → теги."""
    tags = {}
    diag = dict(n_docs=an["n_docs"], n_domains=an["n_domains"],
                inside2=f'{an["inside"]["2"]["n_pass"]}/{an["n_docs"]}',
                inside16=f'{an["inside"]["16"]["n_pass"]}/{an["n_docs"]}',
                margin_n2=an["inside"]["2"]["med_min"],
                margin_decision=an["margin_decision"])

    # слойный тег (TZ §11.6: точечная оценка, без CI; union — только если оба
    # значимы на L23 на своей первичной дозе, m7-r5)
    sig_phr = [L for L in LAYERS
               if an["near_layer"]["E_phr"][str(L)] > an["dmin_PHR_layer"][str(L)]]
    sig_rep = [L for L in LAYERS
               if an["near_layer"]["REP"][str(L)] > an["dmin_REP_layer"][str(L)]]
    union = (sorted(set(sig_phr) | set(sig_rep))
             if (PRIMARY_LAYER in sig_phr and PRIMARY_LAYER in sig_rep) else None)
    tags["layer"] = {"E_phr": sig_phr, "REP": sig_rep, "union": union}
    tags["dose"] = an["dose_tag"]
    # INT (отчёт, не вердикт; TZ §11.5: тег — эвристика без калибровки, F5)
    if an["boot"]["INT_near_n2"]["ci_low"] > an["dmin_PHR"]:
        tags["INTERACT_SIGNIFICANT"] = (
            "эвристика без калибровки (F5): CI_low(near_INT n2, norm_delta) > "
            "δ_PHR; INT Var-сматчен с N_perm (≈σ²), порог δ_PHR на √2 шире по "
            "дисперсии — консервативно, но НЕ тест с α=0.025; INT читается с "
            "непрерывным значением + signed near + CI")
    # LEARN_ASYMM (TZ §9, F1: тег на обеих первичных дозах, порог (1/3, 3))
    for n, tagname in ((PHR_DOSE, "LEARN_ASYMM_N2"), (REP_DOSE, "LEARN_ASYMM_N16")):
        r = an["ratio"].get(str(n))
        if r is not None and (not np.isfinite(r) or not (1.0 / 3.0 < r < 3.0)):
            cal = ("калиброванный порог (F1)" if n == REP_DOSE
                   else "некалиброванная эвристика (F1)")
            tags[tagname] = f"ratio_n{n}={r:.3f} ∉ (1/3, 3) — {cal}"
    # CALIBRATION_WARN (m2': пост-прогоновый прямой σ_L/σ_S > 1.5)
    if an["calib_warn"]:
        tags["CALIBRATION_WARN"] = (
            f"σ_L/σ_S > 1.5: E_phr={an['sd_ratio_phr']:.2f} (norm_delta@n2), "
            f"REP={an['sd_ratio_rep']:.2f} (cos_raw@n16) — порог "
            "анти-консервативен (m2'); конфаунд (F4): при fires отношение "
            "включает вариацию реального эффекта — чистая интерпретация только "
            f"при нулевом вердикте; незагрязнённый якорь (N_PHR, N_perm) = "
            f"{an['sd_anchor_nullpair']:.2f}")
    # LOW_MARGIN (M2-r5: решение ДО main)
    if an["margin_decision"] == "LOW_MARGIN":
        tags["LOW_MARGIN"] = (
            f"смок: 2-доковый median(min(m₁,m₂)) n2 = "
            f"{an.get('margin_smoke', float('nan')):.3f} < {MARGIN_N2} — "
            "решение ДО main (M2-r5): порог маржи не гейт; реализованная "
            f"14-доковая медиана = {an['inside']['2']['med_min']:.3f}; "
            "бинарный 11/14 — гейт")

    def base(value, reasons, extra=None):
        t = dict(tags)
        if extra:
            t.update(extra)
        return dict(value=value,
                    n_star={"E_phr": PHR_DOSE, "REP": REP_DOSE},
                    tags=t, reasons=reasons, diagnostics=diag)

    if not an["gates_ok"]:
        return base("NOT_RESOLVED", [an["gate_reason"] or "gates failed (TZ §10)"])

    in2 = an["inside"]["2"]["gate_ok"]
    in16 = an["inside"]["16"]["gate_ok"]
    if not in2 and not in16:
        return base("INSENSITIVE_CELL", [
            f"G_inside_rep провален на обеих дозах (n2: "
            f"{an['inside']['2']['n_pass']}/{an['n_docs']}, n16: "
            f"{an['inside']['16']['n_pass']}/{an['n_docs']}) — периодическая "
            "строка не представлена как фраза: каналы на обеих дозах "
            "не тестируемы (логика TZ §10.6)"])

    def substruct(name, dose_k):
        b = an["boot"]
        nk, fk, dk = (f"{name}_near_{dose_k}", f"{name}_far_{dose_k}",
                      f"{name}_diff_{dose_k}")
        d_far = an["dmin_far_PHR"] if name == "E_phr" else an["dmin_far_REP"]
        if b[dk]["ci_low"] > 0.0 and b[fk]["ci_high"] <= d_far:
            return "DECAYS", (f"CI95(near−far)=[{b[dk]['ci_low']:.4f}, "
                              f"{b[dk]['ci_high']:.4f}] > 0 И CI95(far)="
                              f"[{b[fk]['ci_low']:.4f}, {b[fk]['ci_high']:.4f}] ≤ "
                              f"δ_far={d_far:.4f} — релаксация к базовой линии")
        if b[fk]["ci_low"] > d_far:
            return "PERSISTS", (f"CI95(far)=[{b[fk]['ci_low']:.4f}, "
                                f"{b[fk]['ci_high']:.4f}] > δ_far={d_far:.4f} — "
                                f"не затухает до конца горизонта H={H}")
        return "AMBIGUOUS", (f"far/релаксация не резолвятся: CI95(near−far)="
                             f"[{b[dk]['ci_low']:.4f}, {b[dk]['ci_high']:.4f}], "
                             f"CI95(far)=[{b[fk]['ci_low']:.4f}, "
                             f"{b[fk]['ci_high']:.4f}], δ_far={d_far:.4f}")

    bp = an["boot"]["E_phr_near_n2"]
    br = an["boot"]["REP_near_n16"]
    S_phr = in2 and bp["ci_low"] > an["dmin_PHR"]
    S_rep = in16 and br["ci_low"] > an["dmin_REP"]
    rp = (f"E_phr n2 (norm_delta): CI95(near)=[{bp['ci_low']:.4f}, "
          f"{bp['ci_high']:.4f}] vs δ_PHR={an['dmin_PHR']:.4f}")
    rr = (f"REP n16 (cos_raw): CI95(near)=[{br['ci_low']:.4f}, "
          f"{br['ci_high']:.4f}] vs δ_REP={an['dmin_REP']:.4f}")

    if in2 and in16:
        if S_phr and S_rep:
            sub_p, rs_p = substruct("E_phr", "n2")
            sub_r, rs_r = substruct("REP", "n16")
            return base("BOTH_SPECIFIC",
                        ["оба канала значимы (ровно 2 пре-рег-теста, семейный "
                         f"уровень ≈5%, TZ §11): {rp}; {rr}", rs_p, rs_r],
                        {"substructure": {"E_phr": sub_p, "REP": sub_r}})
        if S_phr:
            sub_p, rs_p = substruct("E_phr", "n2")
            if sub_p == "AMBIGUOUS":
                return base("NOT_RESOLVED",
                            [rp + " — E_phr значим, но substructure единственного "
                             "значимого канала не решается (TZ §11.4)"],
                            {"substructure": {"E_phr": "AMBIGUOUS"}})
            return base(f"PHRASE_SPECIFIC_{sub_p}", [rp, rs_p],
                        {"substructure": {"E_phr": sub_p}})
        if S_rep:
            sub_r, rs_r = substruct("REP", "n16")
            if sub_r == "AMBIGUOUS":
                return base("NOT_RESOLVED",
                            [rr + " — REP значим, но substructure единственного "
                             "значимого канала не решается (TZ §11.4)"],
                            {"substructure": {"REP": "AMBIGUOUS"}})
            return base(f"REPETITION_SPECIFIC_{sub_r}", [rr, rs_r],
                        {"substructure": {"REP": sub_r}})
        return base("NO_2B_SPECIFIC_EFFECT",
                    [f"ни один канал не превышает шкалу shuffled-нулей: {rp}; "
                     f"{rr} — ОБА канала протестированы (M2); отсутствие не "
                     "доказано (конвенция r2d-1.5; задекларированный исход при "
                     "умеренных эффектах — TZ §11)"])
    # ОДНА доза с провалом гейта (M2: отсутствие измерения ≠ свидетельство
    # отсутствия)
    if not in2:
        gate_tag, name, chan, dose_k, fired, rtxt = (
            "GATE_N2", "REP", "REPETITION", "n16", S_rep, rr)
        untested = "E_phr"
    else:
        gate_tag, name, chan, dose_k, fired, rtxt = (
            "GATE_N16", "E_phr", "PHRASE", "n2", S_phr, rp)
        untested = "REP"
    note = (f"{gate_tag}: G_inside_rep провален на n={'2' if gate_tag == 'GATE_N2' else '16'} "
            f"— канал {untested} не протестирован (чтение не покрывает его; M2)")
    if fired:
        sub_r, rs = substruct(name, dose_k)
        if sub_r == "AMBIGUOUS":
            return base("NOT_RESOLVED",
                        [rtxt, "substructure единственного значимого канала не "
                         "решается", note],
                        {"substructure": {name: "AMBIGUOUS"}})
        return base(f"{chan}_SPECIFIC_{sub_r}", [rtxt, rs, note],
                    {"substructure": {name: sub_r}})
    return base("NOT_RESOLVED",
                [f"{rtxt} — решённый канал нулевой",
                 "отсутствие измерения ≠ свидетельство отсутствия (M2)", note])

# ----------------------------- finite-гейт (M5, M3-r5: 12 контрастов) -----------------------------


def _finite_cell(cell, dn, label):
    ce = np.asarray(cell["ce"], dtype="float64")
    if not np.isfinite(ce).all():
        raise SystemExit(f"M5 finite-gate: {dn} {label} ce содержит NaN/Inf")
    for L in LAYERS:
        for key in ("cos_raw", "cos_state", "norm_delta"):
            a = np.asarray(cell[f"L{L}"][key], dtype="float64")
            if not np.isfinite(a).all():
                raise SystemExit(f"M5 finite-gate: {dn} {label} L{L} {key} "
                                 f"содержит NaN/Inf")


def _finite_gate_2b(store2b, r2store, docs, doses, fallback):
    """TZ §10.5: ce + cos_raw/cos_state/norm_delta по всем условиям × слоям
    (новые — полный; R2-условия — spot-проверка) + все 12 контрастов (INT
    включён — NaN в нём уедет в claim, m3-r5). Провал → NOT_RESOLVED."""
    for dn in docs:
        _finite_cell(get_cell(store2b, r2store, dn, None, "ctrl", fallback),
                     dn, "ctrl" + (" (fresh)" if fallback else " (R2-store, spot)"))
        for n in doses:
            cell = store2b[dn][f"n{n}"]
            for c in COND_NEW:
                _finite_cell(cell[c], dn, f"n{n} {c}")
            for c in ("L1", "fillerA", "fillerB"):
                _finite_cell(get_cell(store2b, r2store, dn, n, c, fallback),
                             dn, f"n{n} {c}"
                             + (" (fresh)" if fallback else " (R2-store, spot)"))
            for key in CONTRASTS:
                for L in LAYERS:
                    for obs in OBS_LIST:
                        a = np.asarray(cell[key][obs][str(L)], dtype="float64")
                        if not np.isfinite(a).all():
                            raise SystemExit(f"M5 finite-gate: {dn} n{n} {key} "
                                             f"L{L} {obs} содержит NaN/Inf")
    logline("M5 finite-gate: PASS (все ряды конечны, 12 контрастов)")

# ----------------------------- анализ (TZ §8, §11 + машина 2b) -----------------------------


def analyze_2b(store2b, r2store, doses=None, margin_decision="THRESHOLD",
               margin_smoke=None):
    """store → δ (C1: на своей наблюдаемой) + сводки + гейты + bootstrap +
    вердикт. r2store=None — синтетика (legacy-якорь и L1 с диска не нужны —
    условия в самом store)."""
    doses = list(doses or DOSES)
    cfg = store2b.get("__config__", {})
    fallback = bool(cfg.get("fallback", False))
    docs = sorted(k for k in store2b if k != "__config__")
    n_docs = len(docs)
    n_domains = len({store2b[k]["domain"] for k in docs})
    L23 = str(PRIMARY_LAYER)
    _finite_gate_2b(store2b, r2store, docs, doses, fallback)

    # δ per-layer (TZ §8, F3): δ_PHR — N_PHR на norm_delta; δ_REP — N_perm на
    # cos_raw; 95th pct 28 per-doc значений (14 док × 2 дозы)
    dmin_PHR_layer, dmin_far_PHR_layer = {}, {}
    dmin_REP_layer, dmin_far_REP_layer = {}, {}
    for L in LAYERS:
        Ls = str(L)
        nn_p, fn_p, nn_r, fn_r = [], [], [], []
        for dn in docs:
            for n in doses:
                c = store2b[dn][f"n{n}"]
                nn_p.append(_near(c["N_PHR"][PHR_OBS][Ls]))
                fn_p.append(_far(c["N_PHR"][PHR_OBS][Ls]))
                nn_r.append(_near(c["N_perm"][REP_OBS][Ls]))
                fn_r.append(_far(c["N_perm"][REP_OBS][Ls]))
        dmin_PHR_layer[Ls] = float(np.percentile(nn_p, 95))
        dmin_far_PHR_layer[Ls] = float(np.percentile(fn_p, 95))
        dmin_REP_layer[Ls] = float(np.percentile(nn_r, 95))
        dmin_far_REP_layer[Ls] = float(np.percentile(fn_r, 95))
    dmin_PHR = dmin_PHR_layer[L23]
    dmin_far_PHR = dmin_far_PHR_layer[L23]
    dmin_REP = dmin_REP_layer[L23]
    dmin_far_REP = dmin_far_REP_layer[L23]
    # S_null (TZ §10.8)
    gates_ok = all(np.isfinite(x) and x > 0 for x in
                   (dmin_PHR, dmin_far_PHR, dmin_REP, dmin_far_REP))
    gate_reason = None
    if not gates_ok:
        gate_reason = (f"S_null: δ не конечен/нулевой (δ_PHR={dmin_PHR:.6g}, "
                       f"δ_far_PHR={dmin_far_PHR:.6g}, δ_REP={dmin_REP:.6g}, "
                       f"δ_far_REP={dmin_far_REP:.6g}) — NOT_RESOLVED (TZ §10.8)")
        logline(f"  S_null: FAIL — {gate_reason}")
    else:
        logline(f"δ: PHR(norm_delta) near={dmin_PHR:.4f} far={dmin_far_PHR:.4f} | "
                f"REP(cos_raw) near={dmin_REP:.4f} far={dmin_far_REP:.4f} "
                f"(28 per-doc значений)")

    # bootstrap (TZ §12): ОДНИ доки на все контрасты (пара́тность r2d-2.5)
    rows = []
    for dn in docs:
        c2, c16 = store2b[dn]["n2"], store2b[dn]["n16"]
        e_n = _near(c2["E_phr"][PHR_OBS][L23])
        e_f = _far(c2["E_phr"][PHR_OBS][L23])
        r_n = _near(c16["REP"][REP_OBS][L23])
        r_f = _far(c16["REP"][REP_OBS][L23])
        i_n = _near(c2["INT"][PHR_OBS][L23])
        p_n = _near(c2["N_PHR"][PHR_OBS][L23])
        p_f = _far(c2["N_PHR"][PHR_OBS][L23])
        q_n = _near(c16["N_perm"][REP_OBS][L23])
        q_f = _far(c16["N_perm"][REP_OBS][L23])
        rows.append([e_n, e_f, e_n - e_f, r_n, r_f, r_n - r_f,
                     i_n, p_n, p_f, q_n, q_f])
    COLNAMES = ["E_phr_near_n2", "E_phr_far_n2", "E_phr_diff_n2",
                "REP_near_n16", "REP_far_n16", "REP_diff_n16",
                "INT_near_n2", "N_PHR_near_n2", "N_PHR_far_n2",
                "N_perm_near_n16", "N_perm_far_n16"]
    boot = {name: c for name, c in zip(COLNAMES,
                                       _boot_multi(np.array(rows)).values())}
    # координаты (m4/G14) — в claim: BOOT_COORDS
    assert set(boot) == set(COLNAMES)

    # G_inside_rep (TZ §10.6): per doc CE(L1)<CE(S1) И CE(L2)<CE(S2);
    # ≥11/14 на дозу; + медиана min(m₁,m₂) ≥ 0.3 на n=2 (по решению)
    inside = {}
    for n in doses:
        p = 0
        margins = []
        for dn in docs:
            c = store2b[dn][f"n{n}"]
            L1b = get_cell(store2b, r2store, dn, n, "L1", fallback)["ce_block"]
            L2b, S1b, S2b = (c["L2"]["ce_block"], c["S1"]["ce_block"],
                             c["S2"]["ce_block"])
            p += 1 if (L1b < S1b and L2b < S2b) else 0
            margins.append(min(S1b - L1b, S2b - L2b))
        pass_min = int(math.ceil(INSIDE_PASS_FRAC * n_docs))
        med = float(np.median(margins))
        margin_ok = (med >= MARGIN_N2) if margin_decision == "THRESHOLD" else True
        gate_ok = (p >= pass_min) and (margin_ok if n == PHR_DOSE else True)
        inside[str(n)] = dict(n_pass=p, pass_min=pass_min, med_min=med,
                              margins=[round(float(x), 6) for x in margins],
                              margin_decision=margin_decision,
                              gate_ok=bool(gate_ok))
    logline(f"G_inside_rep: n2={inside['2']['n_pass']}/{n_docs} "
            f"(margin_med={inside['2']['med_min']:.3f}, "
            f"decision={margin_decision}) | n16={inside['16']['n_pass']}/{n_docs}")

    # I_loop2 + дозовый тег (TZ §9)
    i2 = {str(n): float(np.median([store2b[dn][f"n{n}"]["L2"]["i_loop"]
                                   for dn in docs])) for n in doses}
    dose_tag = _dose_tag_2b(i2)
    i1 = {}
    if r2store is not None:
        for n in doses:
            i1[str(n)] = float(np.median([r2store[dn][f"n{n}"]["loop"]["i_loop"]
                                          for dn in docs]))

    # learnability parity (TZ §9, F1): ratio_n = док-медиана block-CE_L2/L1
    ratio = {}
    for n in doses:
        rr = [store2b[dn][f"n{n}"]["L2"]["ce_block"]
              / get_cell(store2b, r2store, dn, n, "L1", fallback)["ce_block"]
              for dn in docs]
        ratio[str(n)] = float(np.median(rr))

    # CALIBRATION (m2'): прямой σ_L/σ_S на форвардах 2b (SD — выборочная, ddof=1)
    def _sd(vals):
        a = np.asarray(vals, dtype="float64")
        return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
    e_near = [r[0] for r in rows]
    p_near = [r[7] for r in rows]
    r_near = [r[3] for r in rows]
    q_near = [r[9] for r in rows]
    sd_ratio_phr = _sd(e_near) / _sd(p_near)
    sd_ratio_rep = _sd(r_near) / _sd(q_near)
    sd_anchor_nullpair = _sd(p_near) / _sd(q_near)
    calib_warn = (np.isfinite(sd_ratio_phr) and sd_ratio_phr > 1.5
                  or np.isfinite(sd_ratio_rep) and sd_ratio_rep > 1.5)

    # слойные точечные оценки (для слойного тега, TZ §11.6)
    near_layer = {}
    for name, n, obs in (("E_phr", PHR_DOSE, PHR_OBS), ("REP", REP_DOSE, REP_OBS)):
        near_layer[name] = {}
        for L in LAYERS:
            near_layer[name][str(L)] = float(np.mean(
                [_near(store2b[dn][f"n{n}"][name][obs][str(L)]) for dn in docs]))

    # G_inside_legacy (TZ §10.7, диагностика): n=16, L1/L2 < fillerA/fillerB
    legacy_pass = 0
    for dn in docs:
        c = store2b[dn]["n16"]
        L1b = get_cell(store2b, r2store, dn, 16, "L1", fallback)["ce_block"]
        L2b = c["L2"]["ce_block"]
        fa = get_cell(store2b, r2store, dn, 16, "fillerA", fallback)["ce_block"]
        fb = get_cell(store2b, r2store, dn, 16, "fillerB", fallback)["ce_block"]
        legacy_pass += 1 if (L1b < fa and L1b < fb and L2b < fa and L2b < fb) else 0

    # шов (TZ §7.2, диагностика для L2): cos_raw_L2[cut+L_n−1] −
    # cos_raw_fillerB[cut+L_n−1]
    seam2 = {}
    for n in doses:
        L_n = 9 * n
        L23s = f"L{PRIMARY_LAYER}"
        vals = []
        for dn in docs:
            cut = store2b[dn]["cut"]
            l2 = store2b[dn][f"n{n}"]["L2"][L23s]["cos_raw"]
            fb = get_cell(store2b, r2store, dn, n, "fillerB", fallback)
            vals.append(l2[cut + L_n - 1] - fb[L23s]["cos_raw"][cut + L_n - 1])
        seam2[str(n)] = float(np.mean(vals))

    # сводки (TZ §11): 12 контрастов × дозы × наблюдаемые; координаты (m4/G14)
    summary = {}
    for C in CONTRASTS:
        obs_list = ([PHR_OBS] if C == "E_phr" else [REP_OBS] if C == "REP"
                    else ["cos_raw", "norm_delta"])
        summary[C] = {}
        for n in doses:
            summary[C][str(n)] = {}
            for obs in obs_list:
                ns = [_near(store2b[dn][f"n{n}"][C][obs][L23]) for dn in docs]
                fs = [_far(store2b[dn][f"n{n}"][C][obs][L23]) for dn in docs]
                entry = dict(near=round(float(np.mean(ns)), 6),
                             far=round(float(np.mean(fs)), 6))
                if C == "INT" and n == PHR_DOSE and obs == PHR_OBS:
                    # signed near (направление чтения, TZ §11.5)
                    entry["signed_near"] = round(float(np.mean(
                        [float(np.mean(store2b[dn][f"n{n}"][C][obs][L23][:N_W]))
                         for dn in docs])), 6)
                summary[C][str(n)][obs] = entry

    # legacy-δ (TZ §8: только дисковая проверка из R2-store; в fallback —
    # остаётся store-integrity-проверкой, метка «store»)
    legacy = None
    if r2store is not None:
        ln, lf = legacy_delta(r2store, [2, 4, 8, 16])
        legacy = dict(near=round(ln, 6), far=round(lf, 6),
                      anchor_near=LEGACY_NEAR_ANCHOR, anchor_far=LEGACY_FAR_ANCHOR,
                      tol=ANCHOR_TOL, tz_literal="0.1480/0.0332",
                      doses=[2, 4, 8, 16],
                      ok=bool(abs(ln - LEGACY_NEAR_ANCHOR) <= ANCHOR_TOL
                              and abs(lf - LEGACY_FAR_ANCHOR) <= ANCHOR_TOL),
                      source="R2-store (disk, read-only; fallback: метка 'store')")
        if not legacy["ok"]:
            raise SystemExit(f"legacy-δ anchor: {legacy['near']}/{legacy['far']} "
                             f"!= {LEGACY_NEAR_ANCHOR}/{LEGACY_FAR_ANCHOR} (±"
                             f"{ANCHOR_TOL}) — store повреждён — стоп")

    an = dict(
        n_docs=n_docs, n_domains=n_domains, docs=docs,
        dmin_PHR=dmin_PHR, dmin_far_PHR=dmin_far_PHR,
        dmin_REP=dmin_REP, dmin_far_REP=dmin_far_REP,
        dmin_PHR_layer=dmin_PHR_layer, dmin_far_PHR_layer=dmin_far_PHR_layer,
        dmin_REP_layer=dmin_REP_layer, dmin_far_REP_layer=dmin_far_REP_layer,
        gates_ok=gates_ok, gate_reason=gate_reason,
        boot=boot, inside=inside, dose_tag=dose_tag,
        i_loop2=i2, i_loop1=i1, ratio=ratio,
        sd_ratio_phr=sd_ratio_phr, sd_ratio_rep=sd_ratio_rep,
        sd_anchor_nullpair=sd_anchor_nullpair, calib_warn=bool(calib_warn),
        near_layer=near_layer, inside_legacy_pass=legacy_pass,
        seam2=seam2, summary=summary, legacy=legacy,
        margin_decision=margin_decision, margin_smoke=margin_smoke,
        store_config=cfg, fallback=fallback,
    )
    an["verdict"] = compute_verdict_2b(an)
    return an

# ----------------------------- S_cross_store (TZ §10.2) -----------------------------

BOOT_COORDS = {
    "E_phr_near_n2": ("E_phr", PHR_OBS, "n2", "near"),
    "E_phr_far_n2": ("E_phr", PHR_OBS, "n2", "far"),
    "E_phr_diff_n2": ("E_phr", PHR_OBS, "n2", "near-far"),
    "REP_near_n16": ("REP", REP_OBS, "n16", "near"),
    "REP_far_n16": ("REP", REP_OBS, "n16", "far"),
    "REP_diff_n16": ("REP", REP_OBS, "n16", "near-far"),
    "INT_near_n2": ("INT", PHR_OBS, "n2", "near"),
    "N_PHR_near_n2": ("N_PHR", PHR_OBS, "n2", "near"),
    "N_PHR_far_n2": ("N_PHR", PHR_OBS, "n2", "far"),
    "N_perm_near_n16": ("N_perm", REP_OBS, "n16", "near"),
    "N_perm_far_n16": ("N_perm", REP_OBS, "n16", "far"),
}
CROSS_DOC = "wiki_0"          # TZ §10.2: свежие форварды двух R2-условий
CROSS_DOSES = [2, 16]         # F2-r6: обе первичные дозы (путь dose-зависим)
REL_DIAG_TOL = 1e-6          # диагностика (не гейт)
REL_ABS_FLOOR = 5e-7         # абсолютный пол (полпоследнего разряда)


def cross_store_gate(m, doc, r2store, torch):
    """TZ §10.2 (F1-r6): свежие форварды (wiki_0, n=2 И n=16, loop) vs стор.
    Свежий ряд = cond_series R2 verbatim (round 6); гейт = bit-identity
    округлённых значений на всём сохранённом ряду (5 слоёв, cos_raw/
    cos_state/norm_delta). rel 1e-6 — только диагностика с полом 5e-7.
    Возвращает (ok, report)."""
    ids, T = doc["ids"], len(doc["ids"])
    cut = T // 2
    report = {}
    ok = True
    for n in CROSS_DOSES:
        L_n = len(PHRASE1_IDS) * n
        seq = ids[:cut] + PHRASE1_IDS * n + ids[cut:]
        r = fwd(m, seq, torch)
        fresh = cond_series(r, T, cut, L_n)
        stored = r2store[CROSS_DOC][f"n{n}"]["loop"]
        n_mismatch = 0
        max_rel = 0.0
        for L in LAYERS:
            for key in ("cos_raw", "cos_state", "norm_delta"):
                a, b = fresh[f"L{L}"][key], stored[f"L{L}"][key]
                if len(a) != len(b):
                    raise SystemExit(f"S_cross_store: {CROSS_DOC} n={n} L{L} "
                                     f"{key}: длина {len(a)} != {len(b)}")
                for x, y in zip(a, b):
                    if x != y:
                        n_mismatch += 1
                    rel = abs(x - y) / max(abs(x), abs(y), REL_ABS_FLOOR)
                    max_rel = max(max_rel, rel)
        ok_n = n_mismatch == 0
        ok = ok and ok_n
        report[str(n)] = dict(ok=ok_n,
                              bit_mismatch=n_mismatch,
                              max_rel_diag=max_rel,
                              rel_diag_tol=REL_DIAG_TOL,
                              rel_abs_floor=REL_ABS_FLOOR)
        logline(f"S_cross_store {CROSS_DOC} n={n}: bit_mismatch={n_mismatch}, "
                f"max_rel_diag={max_rel:.2e} {'PASS' if ok_n else 'FAIL'}")
    return ok, report

# ----------------------------- артефакты (TZ §13) -----------------------------


def _round_ci(c):
    return {k: (round(x, 6) if isinstance(x, float) else x) for k, x in c.items()}


def _write_artifacts_2b(an, manifest, mode):
    v = an["verdict"]
    cfg = an.get("store_config", {})
    claim = dict(
        experiment="AT8_2b", tz="docs/AT8_2b_TZ.md",
        run_revision=RUN_REVISION, machine=MACHINE_REVISION,
        machine_subrevision=MACHINE_SUBREVISION, mode=mode,
        verdict=v,
        r2_store_sha256=sha256_file(R2_STORE),
        pool_rule=("R2-store (read-only, sha-пин TZ §10.1): 14 доков, 4 домена, "
                   ">=2 дока на домен; пул не пересобирается (TZ §4)"),
        pool_per_domain=dict(Counter(an["store_config"]["domain_of"][d]
                                     for d in an["docs"])),
        n_docs=an["n_docs"], n_domains=an["n_domains"],
        docs_table={d: dict(domain=an["store_config"]["domain_of"][d],
                            T=an["store_config"]["T_of"][d],
                            cut=an["store_config"]["cut_of"][d])
                    for d in an["docs"]},
        doses=DOSES,
        obs_bind=dict(E_phr=PHR_OBS, REP=REP_OBS,
                      note="C1: δ и сводки — только на наблюдаемой своего "
                           "контраста; INT/диагностика — обе"),
        delta_min=dict(
            PHR=dict(near_L23=round(an["dmin_PHR"], 6),
                     far_L23=round(an["dmin_far_PHR"], 6),
                     obs=PHR_OBS,
                     near_by_layer={k: round(x, 6)
                                   for k, x in an["dmin_PHR_layer"].items()},
                     far_by_layer={k: round(x, 6)
                                   for k, x in an["dmin_far_PHR_layer"].items()}),
            REP=dict(near_L23=round(an["dmin_REP"], 6),
                     far_L23=round(an["dmin_far_REP"], 6),
                     obs=REP_OBS,
                     near_by_layer={k: round(x, 6)
                                   for k, x in an["dmin_REP_layer"].items()},
                     far_by_layer={k: round(x, 6)
                                   for k, x in an["dmin_far_REP_layer"].items()}),
            note="95th pct 28 per-doc значений (14 док × 2 дозы); δ_PHR — на "
                 "norm_delta (N_PHR), δ_REP — на cos_raw (N_perm) (TZ §8, C1)"),
        legacy_delta=an["legacy"],
        summary=an["summary"],
        bootstrap={k: _round_ci(c) for k, c in an["boot"].items()},
        bootstrap_coordinates={k: {"contrast": c[0], "obs": c[1], "dose": c[2],
                                   "stat": c[3], "layer": str(PRIMARY_LAYER)}
                               for k, c in BOOT_COORDS.items()},
        inside_rep=an["inside"],
        inside_legacy_pass=an["inside_legacy_pass"],
        i_loop2={k: round(x, 6) for k, x in an["i_loop2"].items()},
        i_loop1={k: round(x, 6) for k, x in an["i_loop1"].items()},
        dose_tag=an["dose_tag"],
        learnability=dict(ratio_n2=round(an["ratio"]["2"], 6),
                          ratio_n16=round(an["ratio"]["16"], 6),
                          note="док-медианы block-CE_L2/block-CE_L1 (TZ §9); "
                               "LEARN_ASYMM_N2 — некалиброванная эвристика (F1)"),
        sigma_calib=dict(
            sd_ratio_phr=round(an["sd_ratio_phr"], 6),
            sd_ratio_phr_spec="SD_docs(near_E_phr n2, norm_delta)/"
                              "SD_docs(near_N_PHR n2, norm_delta) — m2'",
            sd_ratio_rep=round(an["sd_ratio_rep"], 6),
            sd_ratio_rep_spec="SD_docs(near_REP n16, cos_raw)/"
                              "SD_docs(near_N_perm n16, cos_raw) — m2'",
            sd_anchor_nullpair=round(an["sd_anchor_nullpair"], 6),
            sd_anchor_spec="SD(N_PHR)/SD(N_perm) — незагрязнённый якорь "
                           "(обе стороны — нули, F4-r6)",
            warn=an["calib_warn"],
            confound=("F4-r6: при fires отношение включает вариацию реального "
                      "эффекта (гетерогенность по докам); чистая "
                      "интерпретация CALIBRATION_WARN — только при нулевом "
                      "вердикте")),
        int_summary=an["summary"]["INT"],
        seam2={k: round(x, 6) for k, x in an["seam2"].items()},
        shuffle_table=cfg.get("shuffle_table", {}),
        seeds=dict(formula="zlib.crc32(f'S{i}|{doc}|{n}') — единый литерал TZ §3",
                   n_seeds=len(SEEDS_112),
                   n_unique=len(set(SEEDS_112)),
                   concat_sha256=sha256_text("\n".join(SEED_STRINGS)),
                   concat_sha256_pin=SEED_SHA_PIN,
                   pin_ok=sha256_text("\n".join(SEED_STRINGS)) == SEED_SHA_PIN),
        fwer_block=dict(
            n_tests=2,
            alpha_per_test=0.025,
            fwer_independent=round(1 - 0.975 ** 2, 4),
            fwer_bound="[0.025, 0.049] (тесты не независимы — общие доки)",
            family_level="≈ 5% — поправка не нужна (TZ §11, поверхность = "
                         "ровно 2 зафиксированных теста)"),
        cross_store=cfg.get("cross_store", {}),
        fallback=bool(an["fallback"]),
        layers=LAYERS, primary_layer=PRIMARY_LAYER,
        gates=an.get("gates_report", {}),
    )
    json.dump(claim, open(CLAIM, "w"), indent=1)
    manifest["claim_sha256"] = sha256_file(CLAIM)
    json.dump(manifest, open(MANIFEST, "w"), indent=1)
    logline(f"claim sha256={manifest['claim_sha256'][:16]}")


def _series_from_stores(store2b, r2store, fallback, doses):
    """M7: npz из обоих сторов (R2 + 2b; в fallback — свежий путь, метка
    source). Полные ряды (не только post-окно)."""
    out = {}
    docs = sorted(k for k in store2b if k != "__config__")
    for dn in docs:
        cell_doc = store2b[dn]
        out[f"{dn}__ctrl__n0"] = get_cell(store2b, r2store, dn, None, "ctrl",
                                          fallback)
        for n in doses:
            for cond in COND_NEW:
                out[f"{dn}__{cond}__n{n}"] = cell_doc[f"n{n}"][cond]
            for cond in ("L1", "fillerA", "fillerB"):
                out[f"{dn}__{cond}__n{n}"] = get_cell(store2b, r2store, dn, n,
                                                      cond, fallback)
            for C in CONTRASTS:
                for obs in OBS_LIST:
                    out[f"{dn}__{C}__{obs}__n{n}"] = cell_doc[f"n{n}"][C][obs]
    return out


def backup_artifacts():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    for p in (STORE, SERIES):
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(BACKUP_DIR, os.path.basename(p)))
    logline(f"backup: {STORE}, {SERIES} → {BACKUP_DIR}/")

# ----------------------------- run -----------------------------


def run_main(mode="full"):
    import torch
    sys.modules["fla"] = None
    sys.modules["fla.ops"] = None
    torch.set_grad_enabled(False)          # TZ §10.1: запиненный путь R2
    import transformers
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
    logline(f"AT8_2b start {datetime.now(timezone.utc).isoformat()} "
             f"py={sys.version.split()[0]} torch={torch.__version__} "
             f"(CPU, 2x2 factor design, run={RUN_REVISION}, "
             f"machine={MACHINE_REVISION}, mode={mode})")
    tr, tr_done = init_trackio(t0, os.path.abspath(__file__), mode)

    # гейты DO модели (TZ §10.1, §8, §15)
    r2store = check_r2_store()
    check_legacy_anchor(r2store)
    espec1_crosscheck(r2store)

    tk = AutoTokenizer.from_pretrained(MODEL)
    # S_stimulus: phrase-token-check (оба)
    for ph, ph_ids in ((PHRASE1, PHRASE1_IDS), (PHRASE2, PHRASE2_IDS)):
        pcheck = tk(ph, add_special_tokens=False)["input_ids"]
        assert pcheck == list(ph_ids), f"phrase mismatch: {pcheck} != {ph_ids}"
        logline(f"phrase-token-check: PASS {ph_ids}")
    # сиды: полная проверка 112 (m1') + sha-пин (m1-r5)
    assert len(set(SEEDS_112)) == 112, "коллизия сидов (TZ §3)"
    assert sha256_text("\n".join(SEED_STRINGS)) == SEED_SHA_PIN, (
        "sha256 конкатенации 112 сид-строк != пин TZ §15 (m1-r5)")
    logline(f"seeds: 112 уникальных, concat sha256={SEED_SHA_PIN[:16]}… (пин)")

    docs_all, pool_meta = load_pool(tk)
    pool_sha_diff = check_pool(docs_all, pool_meta, r2store)
    texts_by_name = {d["name"]: d["text"] for d in docs_all}
    docs = docs_all
    if mode == "smoke":
        docs = [d for d in docs_all if d["domain"] == "wiki"][:2]
    n_docs, n_domains = len(docs), len({d["domain"] for d in docs})
    logline(f"pool: {n_docs} docs / {n_domains} domains "
             f"(2b: {DOCS14 if mode == 'full' else 'smoke 2 wiki'})")
    tstage(tr, t0, "pool_check")

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    attn_impl = getattr(m.config, "_attn_implementation", "unknown")
    logline(f"C8: attn_impl={attn_impl} (fla заглушен None — CPU-fallback)")
    tstage(tr, t0, "model_loaded")

    # S_cross_store (TZ §10.2): исполняется СМОУКОМ (+2 фд); main наследует
    # решение из decision-файла ДО форвардов (M2-r5-конвенция: решение — до
    # main; n_fwd main = 140 без гейта)
    if mode == "smoke":
        xok, xreport = cross_store_gate(m, docs_all[0], r2store, torch)
        fallback = not xok
        if fallback:
            logline("S_cross_store: FAIL — fallback (TZ §10.2, F3-r6): перегон "
                    "4 R2-условий свежими; смешение численных путей запрещено")
    else:
        if not os.path.exists(DECISION):
            raise SystemExit("no decision: прогоните --smoke ДО main "
                             "(S_cross_store + маржа, TZ §15/M2-r5)")
        decision_pre = json.load(open(DECISION))
        fallback = decision_pre.get("s_cross_store") == "FAIL_FALLBACK"
        xreport = decision_pre.get("cross_store_report", {})
        xok = not fallback
        logline(f"S_cross_store: решение из смока: "
                f"{'PASS' if xok else 'FAIL → fallback'}, report={xreport}")
    tstage(tr, t0, "cross_store")

    # store + resume (config-fingerprint)
    cfg_fp = dict(model=MODEL, layers=LAYERS,
                  phrase1_ids=PHRASE1_IDS, phrase2_ids=PHRASE2_IDS,
                  n_dose=DOSES, H=H, run_revision=RUN_REVISION,
                  fallback=fallback, mode=mode,
                  pool_fps=pool_meta["pool_fps"],
                  filler_map=r2store["__config__"]["filler_map"],
                  r2_store_sha256=R2_STORE_SHA,
                  docs_sha256={d["name"]: sha256_ids(d["ids"]) for d in docs_all},
                  domain_of={d["name"]: d["domain"] for d in docs_all},
                  T_of={d["name"]: len(d["ids"]) for d in docs_all},
                  cut_of={d["name"]: len(d["ids"]) // 2 for d in docs_all},
                  cross_store=xreport)
    store = {}
    if os.path.exists(STORE):
        store = json.load(open(STORE))
        if store.get("__config__") != cfg_fp:
            raise SystemExit("C4: store config-fingerprint не совпадает — "
                             "resume чужого store запрещён (TZ §12); удалите "
                             "AT8_2b_store.json для нового режима")
        logline("resume: store найден, config fp совпадает")
    store["__config__"] = cfg_fp

    shuffle_table = store.get("shuffle_table", {})
    n_fwd = 0
    for doc in docs:
        nm, ids = doc["name"], doc["ids"]
        T = len(ids)
        cut = T // 2
        if T - cut < H + 3:
            raise SystemExit(f"{nm}: short_horizon (T={T}) — пул должен был "
                             f"отфильтровать (TZ §5)")
        cell_doc = store.get(nm) or {}
        if cell_doc.get("T") not in (None, T) or cell_doc.get("cut") not in (None, cut):
            raise SystemExit(f"{nm}: resume-конфликт (T/cut не совпадают с store)")
        store[nm] = dict(cell_doc, T=T, cut=cut, domain=doc["domain"])
        # ctrl: только в fallback (свежий путь; в норме — R2-store, read-only)
        if fallback and "ctrl" not in store[nm]:
            r = fwd(m, ids, torch)
            n_fwd += 1
            store[nm]["ctrl"] = cond_series(r, T, cut, 0)
            logline(f"  {nm} ctrl: T={T} cut={cut} CE={float(r['ce'].mean()):.4f} "
                    f"(fallback: fresh)")
            json.dump(store, open(STORE, "w"), default=float)
        for n in DOSES:
            if f"n{n}" in store[nm]:
                continue
            L_n = len(PHRASE2_IDS) * n
            out = new_seqs(doc, cut, n)
            info = stimulus_checks(doc, n, out)
            for cond, sinfo in info.items():
                shuffle_table[f"{cond}|{nm}|n{n}"] = sinfo
            fa_txt = texts_by_name[r2store["__config__"]["filler_map"][nm][0]]
            fb_txt = texts_by_name[r2store["__config__"]["filler_map"][nm][1]]
            # условия, которые гоняем (TZ §18: только 5 новых; fallback — все 8)
            conds = list(COND_NEW) if not fallback else \
                [f"loop", "fillerA", "fillerB"] + list(COND_NEW)
            cell = {"L_n": L_n}
            for cond in conds:
                if cond == "loop":
                    seq = ids[:cut] + PHRASE1_IDS * n + ids[cut:]
                elif cond in ("fillerA", "fillerB"):
                    txt = fa_txt if cond == "fillerA" else fb_txt
                    seq = ids[:cut] + filler_ids(tk, txt, n) + ids[cut:]
                else:
                    seq = out[cond]
                if len(seq) != T + L_n:
                    raise SystemExit(f"{nm} n={n} {cond}: len mismatch")
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
            # post-вид для контрастов: новые + (R2-store или свежие)
            cell_view = {c: cell[c] for c in COND_NEW}
            if fallback:
                cell_view.update({c: cell[c] for c in ("loop", "fillerA", "fillerB")})
                cell_view = {"L1": cell_view["loop"],
                             "fillerA": cell_view["fillerA"],
                             "fillerB": cell_view["fillerB"],
                             **cell_view}
            else:
                cell_view = {**cell_view,
                             "L1": r2store[nm][f"n{n}"]["loop"],
                             "fillerA": r2store[nm][f"n{n}"]["fillerA"],
                             "fillerB": r2store[nm][f"n{n}"]["fillerB"]}
            cell.update(build_contrasts(cell_view, cut, L_n))
            store[nm][f"n{n}"] = cell
            tr.log({"doc": nm, "n": n, "n_fwd": n_fwd},
                   step=int(time.time() - t0))
            json.dump(store, open(STORE, "w"), default=float)
        tstage(tr, t0, f"doc_{nm}")
    store["shuffle_table"] = shuffle_table
    json.dump(store, open(STORE, "w"), default=float)
    tr.log({"n_fwd": n_fwd, "mode": mode, "fallback": fallback},
           step=int(time.time() - t0))
    tstage(tr, t0, "forwards")

    if mode == "smoke":
        # смоук-якоря TZ §15
        ok = True
        # per-doc сиды: S_i(doc_a) != S_i(doc_b)
        a, b = docs[0]["name"], docs[1]["name"]
        for cond, (i, ph) in SHUFFLE_DEF.items():
            sa = _shuffle_block(ph, i, a, 2)
            sb = _shuffle_block(ph, i, b, 2)
            if sa == sb:
                ok = False
                logline(f"  SMOKE FAIL: {cond}({a}) == {cond}({b}) — per-doc "
                        f"сиды не различаются (F1)")
        logline("  seeds: S_i(doc_a) != S_i(doc_b) OK; 112 уникальных; "
                "concat sha = пин")
        # S_stimulus + S_shuffle (пройдены fail-loud в run-цикле) — таблица
        logline(f"  S_shuffle: 16 shas записаны (smoke: 2 дока × 4 i × 2 дозы); "
                f"период-9-чек PASS (fail-loud); max-9g-таблица в store")
        # block-CE L2(n=2) ∈ [0.5, 2 × block-CE_L1(n=2)]
        for doc in docs:
            dn = doc["name"]
            L2b = store[dn]["n2"]["L2"]["ce_block"]
            L1b = r2store[dn]["n2"]["loop"]["ce_block"]
            if not (0.5 <= L2b <= 2.0 * L1b):
                ok = False
                logline(f"  SMOKE FAIL: {dn} block-CE L2(n2)={L2b:.3f} вне "
                        f"[0.5, 2×{L1b:.3f}]")
        # G_inside_rep(n=2) вычислима + маржи → решение ДО main (M2-r5)
        margins = []
        margins_ml = []
        in_pass = 0
        for doc in docs:
            dn = doc["name"]
            c = store[dn]["n2"]
            L1b = r2store[dn]["n2"]["loop"]["ce_block"]
            m1, m2 = c["S1"]["ce_block"] - L1b, c["S2"]["ce_block"] - c["L2"]["ce_block"]
            in_pass += 1 if (L1b < c["S1"]["ce_block"] and c["L2"]["ce_block"] < c["S2"]["ce_block"]) else 0
            margins.append(min(m1, m2))
            margins_ml.append((m1, m2))
        med2 = float(np.median(margins))
        margin_decision = "LOW_MARGIN" if med2 < MARGIN_N2 else "THRESHOLD"
        logline(f"  G_inside_rep(n2): {in_pass}/{len(docs)}, margins m1/m2: "
                + ", ".join(f"{doc['name']}: ({m[0]:.3f},{m[1]:.3f})"
                            for doc, m in zip(docs, margins_ml))
                + f" → median(min)={med2:.3f} → decision={margin_decision} "
                  "(ДО main, M2-r5)")
        # finite-гейт + δ на своих наблюдаемых (C1)
        _finite_gate_2b(store, r2store, [d["name"] for d in docs], DOSES, fallback)
        names = [d["name"] for d in docs]
        dphr = drep = float("nan")
        try:
            nn_p = [_near(store[dn][f"n{n}"]["N_PHR"][PHR_OBS][str(PRIMARY_LAYER)])
                    for dn in names for n in DOSES]
            nn_r = [_near(store[dn][f"n{n}"]["N_perm"][REP_OBS][str(PRIMARY_LAYER)])
                    for dn in names for n in DOSES]
            dphr, drep = (float(np.percentile(nn_p, 95)),
                          float(np.percentile(nn_r, 95)))
            if not (dphr > 0 and drep > 0):
                ok = False
                logline(f"  SMOKE FAIL: δ не > 0 на своих наблюдаемых "
                        f"(δ_PHR={dphr:.4f} norm_delta, δ_REP={drep:.4f} cos_raw)")
        except Exception as e:  # noqa: BLE001
            ok = False
            logline(f"  SMOKE FAIL: δ-расчёт: {e}")
        logline(f"  δ (smoke, 2 дока × 2 дозы): δ_PHR(norm_delta)={dphr:.4f}, "
                f"δ_REP(cos_raw)={drep:.4f} — C1 привязка OK")
        # решение записывается ДО main (TZ §15, M2-r5)
        decision = dict(
            decided_at=datetime.now(timezone.utc).isoformat(),
            s_cross_store="PASS" if xok else "FAIL_FALLBACK",
            cross_store_report=xreport,
            margin_smoke_n2=med2,
            margin_decision=margin_decision,
            inside_smoke=f"{in_pass}/{len(docs)}",
            legacy_delta=dict(near=LEGACY_NEAR_ANCHOR, far=LEGACY_FAR_ANCHOR,
                              note="проверено fail-loud до смока (TZ §8)"),
            espec1_near_n16=ESPEC1_NEAR_N16,
            n_fwd=n_fwd, mode="smoke",
        )
        json.dump(decision, open(DECISION, "w"), indent=1)
        logline(f"  decision written: {DECISION} (fallback={fallback}, "
                f"decision={margin_decision}, margin_smoke={med2:.3f})")
        logline("SMOKE_PASS" if ok else "SMOKE_FAIL")
        tr_done("ok" if ok else "crash")
        _log.write_to(RUNLOG)
        return 0 if ok else 1

    # полный прогон: решение из decision-файла (M2-r5) — уже прочитано выше
    decision = decision_pre
    margin_decision = decision.get("margin_decision", "THRESHOLD")
    margin_smoke = decision.get("margin_smoke_n2")
    logline(f"decision (из смока): s_cross_store="
            f"{decision.get('s_cross_store')}, margin_decision="
            f"{margin_decision}, margin_smoke={margin_smoke}")

    npz = _series_from_stores(store, r2store, fallback, DOSES)
    np.savez(SERIES, **npz)
    tstage(tr, t0, "series_saved")
    an = analyze_2b(store, r2store, margin_decision=margin_decision,
                    margin_smoke=margin_smoke)
    tstage(tr, t0, "analysis")
    an["gates_report"] = dict(
        r2_store_integrity="pass (sha + fingerprint + пул-гейт, TZ §10.1)",
        legacy_delta_anchor=f"pass (4 дозы verbatim: {LEGACY_NEAR_ANCHOR}/"
                            f"{LEGACY_FAR_ANCHOR} ± {ANCHOR_TOL}; TZ-литералы "
                            "0.1480/0.0332)",
        espec1_crosscheck=f"pass ({ESPEC1_NEAR_N16} ± {ANCHOR_TOL})",
        s_cross_store=("PASS" if xok else "FAIL → fallback (fresh path)"),
        s_stimulus="pass (phrase_token_check ×2, multiset-check, L_n=9n, "
                   "finitem — fail-loud)",
        s_shuffle="pass (период-9-чек fail-loud + max-9g-таблица в артефакте)",
        pool_docs_sha256=(f"D1: {pool_sha_diff}/14 DIFF (окружение-drift, "
                          "id-уровень; тексты pool_fps-верифицированы, "
                          "решение — S_cross_store)"
                          if pool_sha_diff else "14/14 match"),
        finite_gate="pass (analyze: _finite_gate_2b — 12 контрастов)",
        s_null="pass" if an["gates_ok"] else f"FAIL: {an['gate_reason']}")
    manifest = dict(
        generated_by="AT8_2b.py", run_revision=RUN_REVISION,
        machine=MACHINE_REVISION, machine_subrevision=MACHINE_SUBREVISION,
        script_sha256=sha256_file(os.path.abspath(__file__)),
        store_file_sha256=None, series_npz_sha256=None,
        model=MODEL,
        env=dict(attn_impl=attn_impl, dtype="float32",
                 py=sys.version.split()[0], torch=torch.__version__,
                 transformers=transformers.__version__,
                 threads=os.environ.get("OMP_NUM_THREADS", "?"),
                 use_cache="False", grad_enabled="False",
                 r2_env_pin=R2_ENV_PIN),
        seed=SEED, B_boot=B_BOOT, mode=mode,
        cross_store=xreport, fallback=fallback,
        margin=dict(decision=margin_decision,
                    margin_smoke_n2=margin_smoke,
                    main_median_n2=None,  # заполняется после analyze
                    note="M2-r5/F6-r6: манифест пишет и смок-значение, и "
                         "реализованную 14-доковую медиану (какая ветка "
                         "взята и была ли она права, F6-r6)"),
        pool_fps=pool_meta["pool_fps"],
        pool_docs_sha256_diff=pool_sha_diff,
        docs_table={d["name"]: dict(domain=d["domain"], T=len(d["ids"]),
                                    cut=len(d["ids"]) // 2,
                                    sha256=sha256_ids(d["ids"]))
                    for d in docs_all},
        filler_map=r2store["__config__"]["filler_map"],
        n_fwd=n_fwd, claim_sha256=None,
        wall_s=round(time.time() - t0, 1),
    )
    manifest["margin"]["main_median_n2"] = an["inside"]["2"]["med_min"]
    _write_artifacts_2b(an, manifest, mode)
    # sha артефактов (после записи claim)
    manifest["store_file_sha256"] = sha256_file(STORE)
    manifest["series_npz_sha256"] = sha256_file(SERIES)
    json.dump(manifest, open(MANIFEST, "w"), indent=1)
    backup_artifacts()
    v = an["verdict"]
    logline(f"VERDICT: {v['value']} (tags={v['tags']})")
    for r in v["reasons"]:
        logline(f"  {r}")
    logline(f"δ: PHR(norm_delta) near={an['dmin_PHR']:.4f} "
             f"far={an['dmin_far_PHR']:.4f} | REP(cos_raw) "
             f"near={an['dmin_REP']:.4f} far={an['dmin_far_REP']:.4f}")
    logline(f"inside rep n2={an['inside']['2']['n_pass']}/{an['n_docs']} "
             f"(med_min={an['inside']['2']['med_min']:.3f}) "
             f"n16={an['inside']['16']['n_pass']}/{an['n_docs']} | "
             f"i_loop2={an['i_loop2']} | seam2={an['seam2']}")
    logline(f"AT8_2b_DONE verdict={v['value']} mode={mode} "
             f"(wall {time.time() - t0:.0f}s)")
    tr_done("ok")
    return 0

# ----------------------------- reanalysis (disk-only) -----------------------------


def reanalysis_main():
    t0 = time.time()
    r2store = check_r2_store()
    check_legacy_anchor(r2store)
    store = json.load(open(STORE))
    if "__config__" not in store:
        raise SystemExit("store без __config__ — не 2b-store")
    decision = json.load(open(DECISION)) if os.path.exists(DECISION) else {}
    an = analyze_2b(store, r2store,
                    margin_decision=decision.get("margin_decision", "THRESHOLD"),
                    margin_smoke=decision.get("margin_smoke_n2"))
    # full_run (r2c/r2d-конвенция): лог — source of truth
    log_text = open(RUNLOG).read() if os.path.exists(RUNLOG) else None
    n_fwd_log = len(re.findall(r"\bCE=", log_text)) if log_text is not None else None
    fb = an["fallback"]
    n_fwd_store = sum(
        sum(1 for k in store[dn][f"n{n}"] if k in COND_NEW)
        + (sum(1 for k in store[dn][f"n{n}"]
               if k in ("loop", "fillerA", "fillerB")) if fb else 0)
        for dn in [k for k in store if k != "__config__"] for n in DOSES)
    if fb:
        n_fwd_store += sum(1 for dn in [k for k in store if k != "__config__"]
                           if "ctrl" in store[dn])  # ctrl: 1 на док (fresh)
    assert n_fwd_log is not None and n_fwd_log == n_fwd_store, (
        f"n_fwd mismatch: log={n_fwd_log} store={n_fwd_store} "
        "(факт, а не ожидание)")
    full_run = dict(log=RUNLOG,
                    log_sha256=sha256_file(RUNLOG) if log_text else None,
                    n_fwd=n_fwd_store, n_fwd_log=n_fwd_log,
                    n_fwd_store=n_fwd_store, wall_s=None, wall_source="absent")
    if log_text:
        m = re.search(r"AT8_2b_DONE verdict=\S+ mode=\S+ \(wall (\d+)s\)",
                      log_text)
        if m:
            full_run["wall_s"] = int(m.group(1))
            full_run["wall_source"] = "log"
    manifest = dict(
        generated_by="AT8_2b.py --reanalysis", run_revision=RUN_REVISION,
        machine=MACHINE_REVISION, machine_subrevision=MACHINE_SUBREVISION,
        script_sha256=sha256_file(os.path.abspath(__file__)),
        store_file_sha256=sha256_file(STORE),
        series_npz_sha256=sha256_file(SERIES) if os.path.exists(SERIES) else None,
        mode="reanalysis", claim_sha256=None,
        full_run=full_run,
        r2_store_sha256=sha256_file(R2_STORE),
        wall_s=round(time.time() - t0, 1))
    an["gates_report"] = dict(
        r2_store_integrity="pass (reanalysis: sha + fingerprint)",
        legacy_delta_anchor="pass (reanalysis)",
        s_cross_store=("PASS" if not an["fallback"]
                       else "FAIL → fallback (см. store __config__)"))
    _write_artifacts_2b(an, manifest, "reanalysis")
    v = an["verdict"]
    logline(f"VERDICT: {v['value']} (tags={v['tags']})")
    for r in v["reasons"]:
        logline(f"  {r}")
    logline(f"AT8_2b_REANALYSIS_DONE verdict={v['value']} "
             f"(wall {time.time() - t0:.1f}s)")
    return 0

# ----------------------------- синтетические тесты (TZ §16, 10 сценариев) -----------------------------


def _mk_null_arr(rng, near_base=0.5, mid_base=0.3, far_base=0.2, s=0.05):
    """64-значный массив: near [0:16], mid [16:48], far [48:64]."""
    return ([float(rng.normal(near_base, s)) for _ in range(16)]
            + [float(rng.normal(mid_base, s)) for _ in range(32)]
            + [float(rng.normal(far_base, s)) for _ in range(16)])


def _scale(arr, near_f, far_f):
    return [x * near_f for x in arr[:16]] + arr[16:48] + [x * far_f for x in arr[48:64]]


def _synth_store_2b(kind, n_docs=14):
    """Синтетический 2b-store (TZ §16). Профили — на primary-наблюдаемых
    (E_phr: norm_delta@n2, REP: cos_raw@n16); null-шкала — N_PHR/N_perm
    (δ через тот же пайплайн). Нормальные сценарии: primary-контраст =
    точная копия null-массива (детерминированное НЕ-срабатывание);
    сильные — ×4 (near 2.0 ≫ δ ≈ 0.58).
    Scenario 7 (bag): E_phr ≡ N_PHR (одна шкала — тест не срабатывает, M1);
    10 (adversarial_bag): E_phr = N_PHR × 12 (≫ шкалы, INT ≈ 0);
    8 (interact_only): INT = N_PHR × 4, E_phr ≡ N_PHR."""
    rng = np.random.default_rng(0)
    store = {"__config__": dict(
        n_dose=DOSES, fallback=True,
        domain_of={f"syn_{i}": "syn" for i in range(n_docs)},
        T_of={f"syn_{i}": 256 for i in range(n_docs)},
        cut_of={f"syn_{i}": 128 for i in range(n_docs)})}
    for i in range(n_docs):
        dn = f"syn_{i}"
        store[dn] = {"T": 256, "cut": 128, "domain": "syn"}
        store[dn]["ctrl"] = {"ce": [3.0] * 256,
                             **{f"L{L}": dict(cos_raw=[0.9] * 254,
                                              cos_state=[0.9] * 254,
                                              norm_delta=[1.0] * 254)
                                for L in LAYERS}}
        fail_inside = (kind == "insensitive" and i < n_docs - 3)  # 3/14 pass
        for n in DOSES:
            L_n = 9 * n
            len_seq = 256 + L_n
            cell = {"L_n": L_n}
            for cond in ("loop", "fillerA", "fillerB", "L2", "S1", "S2",
                         "S1p", "S2p"):
                cell[cond] = {"ce": [3.0] * len_seq,
                              **{f"L{L}": dict(cos_raw=[0.9] * (len_seq - 2),
                                               cos_state=[0.9] * (len_seq - 2),
                                               norm_delta=[1.0] * (len_seq - 2))
                                  for L in LAYERS}}
                cell[cond]["ce_block"] = 5.0 if fail_inside else \
                    (0.5 if cond in ("loop", "L2") else 4.0)
                cell[cond]["i_loop"] = 2.0
            # null-массивы (оба слоя/наблюдаемые — одна копия)
            nphr_n2 = _mk_null_arr(rng)     # N_PHR @ n2 (norm_delta)
            nphr_n16 = _mk_null_arr(rng)    # N_PHR @ n16
            nperm_n2 = _mk_null_arr(rng)    # N_perm @ n2 (cos_raw)
            nperm_n16 = _mk_null_arr(rng)   # N_perm @ n16
            null = {(2, "N_PHR"): nphr_n2, (16, "N_PHR"): nphr_n16,
                    (2, "N_perm"): nperm_n2, (16, "N_perm"): nperm_n16}
            # primary-контрасты по сценарию: (dose, C) → преобразование null
            strong = lambda arr: _scale(arr, 4.0, 4.0)      # persists: far ×4
            decays = lambda arr: _scale(arr, 4.0, 0.75)     # decays: far ×0.75
            null2, null16 = null[(2, "N_PHR")], null[(16, "N_PHR")]
            prof = {
                "no_effect": {(2, "E_phr"): list(null2),
                              (16, "REP"): list(null16)},
                "bag_mode": {(2, "E_phr"): list(null2)},
                "adversarial_bag": {(2, "E_phr"): [x * 12.0 for x in null2]},
                "phr_decays": {(2, "E_phr"): decays(null2)},
                "phr_persists": {(2, "E_phr"): strong(null2)},
                "rep_decays": {(16, "REP"): decays(null16)},
                "rep_persists": {(16, "REP"): strong(null16)},
                "both": {(2, "E_phr"): decays(null2),
                         (16, "REP"): decays(null16)},
                "interact_only": {(2, "E_phr"): list(null2),
                                  (2, "INT"): strong(null2)},
                "insensitive": {},
            }[kind]
            for C in CONTRASTS:
                null_arr = (null2 if C in ("E_phr", "INT", "N_PHR")
                            else null16)
                arr = prof.get((n, C), list(null_arr))
                cell[C] = {obs: {str(L): list(arr) for L in LAYERS}
                           for obs in OBS_LIST}
            store[dn][f"n{n}"] = cell
    return store


def test_synth_main():
    cases = [
        ("no_effect", "NO_2B_SPECIFIC_EFFECT", {}),
        ("phr_decays", "PHRASE_SPECIFIC_DECAYS", {}),
        ("phr_persists", "PHRASE_SPECIFIC_PERSISTS", {}),
        ("rep_decays", "REPETITION_SPECIFIC_DECAYS", {}),
        ("rep_persists", "REPETITION_SPECIFIC_PERSISTS", {}),
        ("both", "BOTH_SPECIFIC", {}),
        ("bag_mode", "NO_2B_SPECIFIC_EFFECT",
         {"not_value": "PHRASE_SPECIFIC"}),   # M1-регрессия: НЕ PHRASE
        ("interact_only", "NO_2B_SPECIFIC_EFFECT",
         {"tag": "INTERACT_SIGNIFICANT"}),
        ("insensitive", "INSENSITIVE_CELL", {}),
        ("adversarial_bag", "PHRASE_SPECIFIC",
         {"no_tag": "INTERACT_SIGNIFICANT"}),  # fires без INT (m4')
    ]
    ok = 0
    for kind, expected, extra in cases:
        store = _synth_store_2b(kind)
        an = analyze_2b(store, None, margin_decision="THRESHOLD")
        v = an["verdict"]
        good = (v["value"] == expected if expected != "PHRASE_SPECIFIC"
                else v["value"].startswith("PHRASE_SPECIFIC"))
        if "not_value" in extra and v["value"] == extra["not_value"]:
            good = False
        if "tag" in extra and extra["tag"] not in v["tags"]:
            good = False
        if "no_tag" in extra and extra["no_tag"] in v["tags"]:
            good = False
        ok += good
        logline(f"  synth {kind}: {v['value']} (ожид. {expected}) "
                f"{'PASS' if good else 'FAIL'} | δ_PHR={an['dmin_PHR']:.4f} "
                f"δ_REP={an['dmin_REP']:.4f}")
    logline(f"SYNTH_TEST: {ok}/10 PASS")
    return 0 if ok == 10 else 1

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
