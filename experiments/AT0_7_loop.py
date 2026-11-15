#!/usr/bin/env python3
"""AT0_7 — T0.7 «Петля = стабильный copy-paste: repeat как necessary признак петли (disk-only)».

ТЗ: docs/TZ_add_0-3.md T0.7 (v3; закрыто внешним ревью r3: PASS, 0 MAJOR). Модули:
- experiments/at0_7_core.py — константы ТЗ, logline, токено-эстиманды (A): PPF/rep_frac, C,
  RATIO, maxrun, p*; (C): (токен,фаза)-lookup R(p) ≡ 0 / Rtok; Spearman; синтетика; selfcheck.
- experiments/at0_7_tests.py — гейты (G1a/b/d, G2, smoke-edges) + G7 + POSITIVE + H1–H5 + вердикт
  (полная пре-рег-документация — в docstring'ах; Q/null-формулы — AT0_6 v2.2, PIN-импорт).
- этот файл — пулы (контракт T0.3), оркестрация, trackio, артефакты.
AT0_6 v2.2 — ПИН (решение супервизора, 15.11): experiments/AT0_6_spectrum.py в репо активно
переписывают (v2.2→v3, WIP) — ТЗ T0.7 замораживает «формулы идентичны AT0_6_spectrum.py v2.2»
⇒ импортируется byte-идентичная копия blob-а коммита 7bef68b (v2.2 + docstring k≤3):
   experiments/at06_v22_pinned.py
(sha256 c24cb668…; /tmp/at06_v22_pinned.py — та же копия). WIP-файл из репо НЕ импортируется.
Провенанс: в репо есть файлы параллельной сессии — AT0_7_loop_mine.py (external, not canonical),
AT0_8_threshold.py (external) — не трогаются (решение супервизора).

Вопрос: период-p структура T0.3 — динамический режим h или артефакт повторения токенов?
ЧЕСТНАЯ ОБЛАСТЬ: feedforward ⇒ p-периодичные токены ⇒ h p-периодичен ⇒ период-p структура h —
ПОСЛЕДСТВИЕ repeat'а (не «сверх»). (a) петля ⇒ стабильный repeat (necessary, ретроспективно r0);
(b) repeat не loopy-специфичен (H2/H3 descriptive); (c) h-ГРАНИЦА H5: R(p) ≡ 0 — per-doc
disk-only НЕ разделяет copy-paste и динамику (нужен контрфактуальный forward — вне scope).

ESTIMAND: (A) PPF(p) = #{t∈[0,T−p−1]: tok[t]==tok[t+p]}/(T−p) = rep_frac; C = Σ_v f(v)²;
RATIO = PPF/C; maxrun(p) = длиннейший бег (чистый p-repeat → T−p); p* = argmax_{p∈3..48} RATIO.
(B) Q(p) = Σ_d|S_d(p)|²/(T'·σ²_d), S_d(p)=Σ_t x_{t,d}e^{−2πit/p} (формулы AT0_6 v2.2); per-doc
permutation-null B=199, сид 2026+doc_idx·1009+s (doc_idx — индекс в sorted(96), как T0.6).
(C) R(p) = Σ_d|S_x−S_sur|²/Σ_d|S_x|² ≡ 0 ((токен,фаза)-lookup; x_t = h_{t+1}−mean ⇒ x_t ↔ tok[t+1]);
Rtok — per-token без фазы (вырожден при V≫T).

ПУЛЫ (контракт T0.3, AT0_3_AREA.json sha256[:16] = 8fe61affb13b443f): 96 = 30 loopy-detected
(p_T0.3 verbatim) + 11 loopy-undetected + 53 clean (37 non-corpus = 36 gen + own_w13; 16 corpus —
без токенов, not_testable G0-a) + 2 remainder (own_w0, own_w4). Токенизируемые: 80. own ≡ wiki —
дубли по ТОКЕНАМ (np.array_equal 16/16) ⇒ per-unique-text отчёт (H2-пул 39 = 38 уникальных:
own_w13≡wiki_13). x = h_raw|L23 (T0.3/T0.6-конвенция), T' = T−1 (gen/own 383).

ТЕСТЫ (пре-рег; детали — at0_7_tests.py): H1 (primary, ретросп. r0 + валидация):
f1 = #{rep_frac(p_T0.3) ≥ 0.8}/30, PASS ≥ 28 (r0: 30/30, min 0.888 (code_14) — 1e-12;
f1 ПРЕОПРЕДЕЛЕНО данными — честно; prospective = H3/H4 + гейты + первый committed Q+p_perm).
H2 (descriptive): per-subpool #{rep_frac ≥ 0.8}: LU 6/11 = 5/8 unique, clean37 0/37, rem 2/2.
H3 (descriptive): кросстаб 80 (стабильный repeat × T0.3-detected) → доля loopy среди стабильных
(necessary, не sufficient). H4 (descriptive): Spearman(Q, maxrun) по 30 (ρ_s ≥ 0.4 → h-линия
следует за repeat'ом) + Spearman(Q, rep_frac). H5 (граница, НЕ тест): R(p) ≡ 0 (30 loopy:
[2.5e-32, 3.5e-30] r0; clean max ~1e-28); пороги 0.1/0.3 УБРАНЫ.
ГЕЙТЫ: G0 (sha npz + AT0_3 + gen_sha_ok 64/64 + декод-пробы, как T0.6); G1 (a) rep_frac-identity
(13×T=383: PPF=1.0 ТОЧНО, |RATIO−1/C|<1e-12, C=11287/146689, RATIO=12.996279≠13, maxrun=370);
(b) Q на чистой p-синусоиде (a=1, σ=1, p₀=13, D=32) ≥ 50·D (T0.6 G1b, замер 66.5, seed 77700);
(c) УБРАН (r2); (d) iid-токены (V=4096, seed 777, T=383): max_p RATIO ≤ 1.1 (замер 1.0118 @p37).
G2 (96 доков; n_tok 30+39+11=80, corpus 0/16 — явный assert; T'=T−1; per-dim var>0). G7 (20 iid,
seed 777+k): (a) ≤1/20 с max_p RATIO ≥ 3 (r2: «≤1.1» падало на 784=1.98/785=1.96); (b) Q-null
k ≤ 3 (как T0.6 G7). POSITIVE (2 синтетика, 7770+k, a=1, σ=1, D=32): (1) чистый p-repeat токенов
(p=13, v_i = rng(7771), 13 различных) ⇒ rep_frac(13) = 1.0 точно (p=13, не кратные); (2) чистая
h-осцилляция (iid-токены rng(7772)) ⇒ max_p PPF < 0.2 ∧ Q(13)/32 ≥ 50.
ВЕРДИКТ (primary = H1; FWER не применяется — объявлено): H1 + H3 + H4 → петля = стабильный
copy-paste ∧ сильная Δu-структура (T0.3/T0.6-структуры — feedforward-последствия token-repeat).
K4: T0.7 НЕ влечёт T0.6 (линия в h ≠ линия в du; носители разные — r1); T0.6 (smoke) не
переписывается. ¬H1 (f1 < 28) → расхождение прогона с r0 — разбор.
Детерминизм: null 2026+doc_idx·1009+s; G7/G1d токены rng(777+k).integers(0,4096,383) (G1d = k=0);
POS h 7770/7771, токены 7771/7772; G1b 77700; G1a v_i = i. Нет global np.random; sorted(docs);
store sort_keys; json — jclean. Цена ≈ 15–20 мин CPU (G7-grid 46p×20×200 ≈ 16 мин доминирует);
память < 1 GB. Артефакты: results/AT0/{store/<sha16>.json, AT0_7_LOOP.json, AT0_7_manifest.json,
AT0_7_run.log} — writes только results/AT0/AT0_7_* + store. trackio: at07_loop_r1.
"""
import atexit
import hashlib
import json
import os
import sys
import time
from datetime import datetime

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402

# AT0_6 v2.2 — ПИН (решение супервизора): репо-файл experiments/AT0_6_spectrum.py переписывают
# (v2.2→v3, WIP, нестабилен) — импортируем byte-идентичную копию blob-а коммита 7bef68b
# (v2.2 + 1 строка docstring k≤3; формулы/сиды = v2.2). Регистрация в
# sys.modules под именем AT0_6_spectrum: at0_7_tests импортирует это имя и получает пин.
A6_PIN = os.path.join(ROOT, "experiments", "at06_v22_pinned.py")
A6_PIN_SHA = "c24cb6681e5eb26f6511fa177cbbc7a2b793333f7c5665357b70ac11ab1132bb"
assert os.path.exists(A6_PIN), f"AT0_7: pinned AT0_6 v2.2 не найден: {A6_PIN}"
_spec = importlib.util.spec_from_file_location("AT0_6_spectrum", A6_PIN)
a6 = importlib.util.module_from_spec(_spec)
sys.modules["AT0_6_spectrum"] = a6
_spec.loader.exec_module(a6)
import at0_7_core as core  # noqa: E402
import at0_7_tests as tests  # noqa: E402

pf = sys.modules["prism_field_pinned"]   # тот же pinned-модуль, что загрузил AT0_6 v2.2
jclean, sha256_file = pf.jclean, pf.sha256_file
assert sha256_file(A6_PIN) == A6_PIN_SHA, "AT0_7: sha pinned AT0_6 v2.2 не совпадает с пином"

OUT_DIR = os.path.join(ROOT, "results", "AT0")
STORE_DIR = os.path.join(OUT_DIR, "store")
OUT_JSON = os.path.join(OUT_DIR, "AT0_7_LOOP.json")
MAN_JSON = os.path.join(OUT_DIR, "AT0_7_manifest.json")
RUN_LOG = os.path.join(OUT_DIR, "AT0_7_run.log")
REVISION = "r1"
SMOKE = "--smoke" in sys.argv

logline = core.logline


# ------------------------------------------------------------------- пулы (контракт T0.3)

def build_pools(docs, t03):
    """96 = 30 loopy-detected (p_T0.3) + 11 loopy-undetected + 53 clean (37 non-corpus + 16
    corpus) + 2 remainder (own_w0, own_w4). Пулы — verbatim из period_structure T0.3."""
    ps = t03["results"]["period_structure"]["per_doc"]
    p_map = {nm: int(r["p"]) for nm, r in ps["loopy"].items() if r.get("p") is not None}
    loopy = sorted(ps["loopy"])
    clean = sorted(ps["clean"])
    rem = sorted(ps["remainder"])
    lu11 = [nm for nm in loopy if nm not in p_map]
    clean37 = [nm for nm in clean if not nm.startswith("cor_")]
    cor16 = [nm for nm in clean if nm.startswith("cor_")]
    assert len(p_map) == 30 and len(lu11) == 11, f"loopy: 30+11, got {len(p_map)}+{len(lu11)}"
    assert len(clean37) == 37 and len(cor16) == 16, "clean: 37 non-corpus + 16 corpus"
    assert rem == ["own_w0", "own_w4"], f"remainder: {rem}"
    assert "own_w13" in clean37 and len(docs) == 96
    assert set(docs) == set(loopy + clean + rem)
    return {"p_map": p_map, "loopy30": sorted(p_map), "lu11": lu11, "clean37": clean37,
            "cor16": cor16, "rem2": rem, "h2": sorted(clean37 + rem),
            "tokenizable80": sorted(set(p_map) | set(lu11) | set(clean37) | set(rem))}


def token_pairing(docs, names):
    """Дубли по ТОКЕНАМ (np.array_equal; ТЗ: 16 own = TPU-дубли 16 wiki gen-доков):
    {doc: partner or None} внутри переданных 80 токенизируемых."""
    toks = {nm: docs[nm]["toks"] for nm in names if len(docs[nm]["toks"])}
    nms = sorted(toks)
    out = {nm: None for nm in nms}
    for i, a in enumerate(nms):
        for b in nms[i + 1:]:
            if len(toks[a]) == len(toks[b]) and np.array_equal(toks[a], toks[b]):
                out[a], out[b] = b, a
                break   # K1 (r1-exec): первая пара для a (тройной дубликат недостижим на данных)
    return out


unique_root = core.unique_root   # shared (core) — использует at0_7_tests.run_h2


def all_token_rows(docs, pools):
    """Per-doc токено-строки по 96 (80 с токенами): p_rep = p_T0.3 (detected) или p* (остальные);
    corpus — T=0, все None (not_testable G0-a)."""
    return {nm: core.token_row(docs[nm]["toks"], pools["p_map"].get(nm))
            for nm in sorted(docs)}


# ------------------------------------------------------------------- trackio

def _init_trackio(t0):
    """AGENTS: trackio обязателен для ВСЕХ новых локальных (CPU) экспериментов.
    Возвращает (run, done_flag); краш — status=crash через atexit."""
    try:
        import trackio
    except ImportError as e:
        raise SystemExit(f"trackio: модуль не установлен — прогон остановлен "
                         f"(AGENTS: trackio обязателен для CPU-прогонов): {e}")
    tr_cfg = {
        "started_at": datetime.now().isoformat(),
        "revision": REVISION,
        "script_sha": sha256_file(os.path.abspath(__file__)),
        "seed": ("multi: null 2026+doc_idx*1009+s; G7/G1d 777+k; POS 7770+k (токены "
                 "7771/7772); G1b 77700; G1a deterministic"),
        "device": "CPU", "model": "AT0_7", "log": os.path.abspath(RUN_LOG),
        "B": core.B, "T_prime": core.T_PRIME, "p_grid": [core.P_GRID[0], core.P_GRID[-1]],
    }
    tr = trackio.init(project="prism", name=f"at07_loop_{REVISION}", group="AT0", config=tr_cfg)
    tr.log({"started": 1.0}, step=0)
    logline(f"trackio: run at07_loop_{REVISION} init ok")
    done = [False]

    def _on_exit():
        if not done[0]:
            try:
                tr.log({"finished_at": datetime.now().isoformat(), "status": "crash",
                        "error": "unhandled exception / abort (fin-точка не записана)"},
                       step=int(time.time() - t0))
                tr.finish()
            except Exception:
                pass
    atexit.register(_on_exit)
    return tr, done


def _tstage(tr, t0, name, extra=None):
    m = {"stage": name, "stage_s": round(float(time.time() - t0), 1)}
    if extra:
        m.update(extra)
    if tr is not None:
        tr.log(m, step=int(time.time() - t0))


# ------------------------------------------------------------------- гейты (оркестрация)

def _run_gates(docs, pools, t0, tr):
    """G0 (как T0.6) + G1a/b/d + G2 (+ явные assert ТЗ) + G7 + POSITIVE. ДО интерпретации."""
    g0 = a6.g0_data(docs)
    _tstage(tr, t0, "g0")
    toks777 = np.random.default_rng(core.G1D_SEED).integers(0, core.V_SYNTH, core.G7_TOK_T)
    g1a = tests.g1a_identity()
    g1b = tests.g1b_q_identity()
    g1d = tests.g1d_iid(toks777)
    g2 = tests.g2_cells(docs, pools)
    # явные assert ТЗ G2: 30 (loopy) + 39 (H2-пул) + 11 (LU) = 80, corpus 0/16
    assert g2["n_tok_loopy"] == 30 and g2["n_tok_h2"] == 39 and g2["n_tok_lu11"] == 11, \
        f"G2: n_tok 30+39+11, got {g2['n_tok_loopy']}+{g2['n_tok_h2']}+{g2['n_tok_lu11']}"
    assert g2["n_tok_total"] == 80 and g2["n_corpus_no_tok"] == 16, \
        f"G2: 80 токенизируемых + corpus 0/16, got {g2['n_tok_total']}+{g2['n_corpus_no_tok']}"
    _tstage(tr, t0, "g1_g2")
    g7 = tests.run_g7(N=2 if SMOKE else core.G7_N, Bn=49 if SMOKE else core.B)
    _tstage(tr, t0, "g7")
    pos = tests.run_positive()
    _tstage(tr, t0, "positive")
    gates_ok = bool(g0["ok"] and g1a["ok"] and g1b["ok"] and g1d["ok"] and g2["ok"]
                    and g7["ok"] and pos["ok"])
    logline(f"GATES: G0={g0['ok']} G1a={g1a['ok']} G1b={g1b['ok']} G1d={g1d['ok']} "
            f"G2={g2['ok']} G7={g7['ok']} POS={pos['ok']} → {'PASS' if gates_ok else 'FAIL'}")
    return {"G0": g0, "G1a_identity": g1a, "G1b_unit": g1b, "G1d_iid": g1d, "G2": g2,
            "G7_calibration": g7, "POSITIVE": pos, "all_ok": gates_ok}


# ------------------------------------------------------------------- артефакты

def _write_store(store):
    """store blob → results/AT0/store/<sha16>.json (sort_keys, jclean, no NaN)."""
    blob = json.dumps(jclean(store), ensure_ascii=False, sort_keys=True)
    assert "NaN" not in blob and "Infinity" not in blob, "store: NaN/Inf в json"
    sha = hashlib.sha256(blob.encode()).hexdigest()
    path = os.path.join(STORE_DIR, f"{sha[:16]}.json")
    with open(path, "w") as f:
        f.write(blob)
    logline(f"store: {os.path.relpath(path, ROOT)} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path, sha


def _write_claim(store, store_path, store_sha):
    """claim → results/AT0/AT0_7_LOOP.json (пер-док — в store; claim — сводка + пер-док H1/H4)."""
    g, t = store["gates"], store["tests"]
    claim = {
        "experiment": "AT0_7 T0.7 (петля = стабильный copy-paste; repeat как necessary признак, v3)",
        "round": f"v3-{REVISION}", "date": store["meta"]["date"],
        "store": os.path.relpath(store_path, OUT_DIR), "store_sha256": store_sha,
        "meta": store["meta"], "pools": store["pools"], "p_map_T0_3": store["p_map_T0_3"],
        "token_pairing": store["token_pairing"], "prereg": store["prereg"],
        "gates": {k: g[k] for k in ("G0", "G1_field", "G1a_identity", "G1b_unit",
                                    "G1d_iid", "G2", "all_ok")},
        "gates_G7_POS": {"G7_calibration": {k: g["G7_calibration"][k]
                                            for k in ("ok", "n", "B", "a", "b")},
                         "POSITIVE": g["POSITIVE"]},
        "tests": {
            "H1": {"n": t["H1"]["n"], "f1": t["H1"]["f1"], "gate": t["H1"]["gate"],
                   "ok": t["H1"]["ok"], "min_rep_frac": t["H1"]["min_rep_frac"],
                   "median_rep_frac": t["H1"]["median_rep_frac"],
                   "r0_anchor": t["H1"]["r0_anchor"], "r0_reproduced": t["H1"]["r0_reproduced"],
                   "per_doc": t["H1"]["rows"]},
            "H2": {k: {kk: vv for kk, vv in v.items() if kk != "docs"}
                   for k, v in t["H2"].items() if isinstance(v, dict)},
            "H3": t["H3"],
            "H4": {"n": t["H4"]["n"], "spearman_q_maxrun": t["H4"]["spearman_q_maxrun"],
                   "spearman_q_repfrac": t["H4"]["spearman_q_repfrac"],
                   "gate": t["H4"]["gate"], "per_doc": t["H4"]["rows"]},
            "H5": {k: v for k, v in t["H5"].items() if k != "per_doc"},
            "verdict": t["verdict"],
        },
    }
    s = json.dumps(jclean(claim), ensure_ascii=False, indent=1, sort_keys=True)
    assert "NaN" not in s and "Infinity" not in s, "claim: NaN/Inf в json"
    sha = hashlib.sha256(s.encode()).hexdigest()
    with open(OUT_JSON, "w") as f:
        f.write(s)
    logline(f"claim: {OUT_JSON} sha {sha[:16]}")
    return sha


def _write_manifest(store, store_path, store_sha, claim_sha):
    """manifest → results/AT0/AT0_7_manifest.json (self-hash по blob до вставки)."""
    man = {
        "experiment": "AT0_7 T0.7 (v3)", "date": store["meta"]["date"],
        "claim_path": os.path.relpath(OUT_JSON, ROOT), "claim_sha256": claim_sha,
        "store_path": os.path.relpath(store_path, ROOT), "store_sha256": store_sha,
        "script_sha256": store["meta"]["script_sha256"],
        "core_sha256": store["meta"]["core_sha256"],
        "at06_sha256": store["meta"]["at06_sha256"],
        "field_sha256": store["meta"]["field_sha256"],
        "data_sha256": store["meta"]["data_sha256"],
        "at03_sha256": store["meta"]["at03_sha256"],
        "at06_v22_pin": store["meta"]["at06_v22_pin"],
        "parallel_session_files": store["meta"]["parallel_session_files"],
    }
    s0 = json.dumps(man, ensure_ascii=False, indent=1, sort_keys=True)
    man["manifest_sha256"] = hashlib.sha256(s0.encode()).hexdigest()
    with open(MAN_JSON, "w") as f:
        f.write(json.dumps(man, ensure_ascii=False, indent=1, sort_keys=True))
    logline("manifest: OK (self-hash по blob до вставки)")


# ------------------------------------------------------------------- main

def main():
    t0 = time.time()
    os.makedirs(STORE_DIR, exist_ok=True)
    logline(f"AT0_7 start {datetime.now().isoformat()} py={sys.version.split()[0]} "
            f"np={np.__version__} smoke={SMOKE}")
    tr, tr_done = (None, None) if SMOKE else _init_trackio(t0)

    pf._selfcheck()
    pf_sha = sha256_file(os.path.join(ROOT, "prism/field.py"))
    logline(f"G1′ field.py sha {pf_sha[:16]} selfcheck OK")

    docs = a6.load_all_docs()
    t03 = json.load(open(a6.AT03_PATH))
    pools = build_pools(docs, t03)
    pairing = token_pairing(docs, pools["tokenizable80"])
    doc_idx = {nm: i for i, nm in enumerate(sorted(docs))}
    logline(f"пулы (контракт T0.3): loopy30 = {len(pools['loopy30'])}, "
            f"lu11 = {len(pools['lu11'])}, clean37 = {len(pools['clean37'])}, "
            f"corpus16 = {len(pools['cor16'])}, rem2 = {len(pools['rem2'])}; "
            f"дубли по токенам: {sum(1 for v in pairing.values() if v)} docs")

    gates = _run_gates(docs, pools, t0, tr)
    rows = all_token_rows(docs, pools)
    _tstage(tr, t0, "token_rows")

    if SMOKE:
        tests.smoke_edges(docs, pools)
        tests.run_h1_h4(docs, pools, doc_idx, rows, pools["loopy30"][:3], Bn=49)
        tests.run_h2(rows, pools, pairing)
        tests.run_h3(rows, pools)
        tests.run_h5(docs, rows, pools)
        logline("AT0_7 SMOKE DONE (артефакты не пишутся)")
        return

    h1, h4, r0_ok = tests.run_h1_h4(docs, pools, doc_idx, rows, pools["loopy30"])
    _tstage(tr, t0, "h1_h4")
    h2 = tests.run_h2(rows, pools, pairing)
    h3 = tests.run_h3(rows, pools)
    h5 = tests.run_h5(docs, rows, pools)
    _tstage(tr, t0, "h2_h3_h5")
    verdict = tests.make_verdict(gates["all_ok"], h1, r0_ok, h4, h3, h5)
    logline(f"VERDICT: {verdict['status']} (f1 = {h1['f1']}/30, r0_ok = {r0_ok})")
    _tstage(tr, t0, "verdict", {"status": verdict["status"]})

    store = {
        "meta": {
            "script_sha256": sha256_file(os.path.abspath(__file__)),
            "core_sha256": sha256_file(os.path.abspath(core.__file__)),
            "at06_sha256": sha256_file(os.path.abspath(a6.__file__)),
            "field_sha256": pf_sha, "data_sha256": gates["G0"]["data_sha256"],
            "at03_sha256": gates["G0"]["at03_sha256"], "gen_sha_ok": gates["G0"]["gen_sha_ok"],
            "decode_probe": gates["G0"]["decode_probe"],
            "n_docs": len(docs), "B": core.B, "p_grid": [core.P_GRID[0], core.P_GRID[-1]],
            "t_prime": "T−1 per-doc (gen/own 383)",
            "at06_v22_pin": {"path": os.path.relpath(A6_PIN, ROOT), "sha256": A6_PIN_SHA,
                             "note": ("byte-идентична blob-у коммита 7bef68b:experiments/AT0_6_spectrum.py "
                                      "(v2.2 + 1 строка docstring k≤3; формулы/сиды = v2.2); репо-HEAD — v3, "
                                      "не импортируется (решение супервизора); пин верифицируем в git")},
            "parallel_session_files": ("AT0_7_loop_mine.py (external, not canonical), "
                                        "AT0_8_threshold.py (external) — не трогаются"),
            "seed_convention": ("null 2026+doc_idx*1009+s (sorted(96)); G7/G1d токены "
                                "rng(777+k).integers(0,4096,383) (G1d = k=0); POS h 7770+k, "
                                "токены 7771/7772; G1b 77700 (= T0.6 G1b); G1a v_i = i"),
            "smoke": False, "date": datetime.now().isoformat(),
            "python": sys.version.split()[0], "numpy": np.__version__,
            "trackio_run": f"at07_loop_{REVISION}",
        },
        "pools": {k: v for k, v in pools.items() if k != "p_map"},
        "p_map_T0_3": pools["p_map"],
        "token_pairing": {nm: p for nm, p in sorted(pairing.items()) if p is not None},
        "prereg": {
            "tz": "docs/TZ_add_0-3.md T0.7 (v3, closed r3: PASS, 0 MAJOR)",
            "h1": {"pool": "30 loopy-detected, p = p_T0.3 verbatim", "floor": core.H1_FLOOR,
                   "threshold": core.REPMIN, "r0_anchor": h1["r0_anchor"],
                   "note": ("ретроспективный r0 (30/30, min 0.888) + валидация пайплайна; "
                            "f1 ПРЕОПРЕДЕЛЕНО данными — прогон воспроизводит в 1e-12; "
                            "prospective = H3/H4 + гейты + первый committed Q(p_T0.3)+p_perm")},
            "h2": {"pools": "LU11 / clean37 / remainder2 (+ unique-text)",
                   "threshold": core.REPMIN,
                   "note": "descriptive, НЕ бинарный; бинарный «f2 ≥ 1» УБРАН (r1)"},
            "h3": {"pool": "80 токенизируемых", "threshold": core.REPMIN,
                   "note": "descriptive, граница necessary (corpus-16 not_testable)"},
            "h4": {"pool": "30 loopy", "gate": "ρ_s(Q, maxrun) ≥ 0.4",
                   "extra": "Spearman(Q, rep_frac)"},
            "h5": {"note": ("h-уровневая ГРАНИЦА: R(p) ≡ 0 (тождество); пороги 0.1/0.3 УБРАНЫ "
                            "(недостижимы); Rtok вырожден при V≫T")},
            "g1": {"a": "rep_frac-identity (13×T=383: PPF=1.0, |RATIO−1/C|<1e-12, maxrun=370)",
                   "b": "Q(p₀)/D ≥ 50 на чистой p-синусоиде (a=1, σ=1, p₀=13, D=32)",
                   "c": "УБРАН (r2: R≡0 тождество, тривиален)",
                   "d": "iid-токены V=4096, seed 777, T=383: max_p RATIO ≤ 1.1"},
            "g7": {"n": core.G7_N,
                   "a": f"count-гейт: ≤1/{core.G7_N} с max_p RATIO ≥ {core.G7_RATIO_THR}",
                   "b": f"k ≤ {core.G7_K_MAX}"},
            "positive": {"seeds": "7770+k (h); токены 7771 (v_i) / 7772 (iid)",
                         "p0": core.POS_P0, "a": core.POS_A, "D": core.POS_D,
                         "gates": ["rep_frac(13) = 1.0 точно (p=13, не кратные)",
                                   "max_p PPF < 0.2 ∧ Q(13)/32 ≥ 50"]},
        },
        "gates": {"G1_field": {"field_sha256": pf_sha, "selfcheck": "OK"}, **gates},
        "tests": {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "verdict": verdict},
    }
    store_path, store_sha = _write_store(store)
    claim_sha = _write_claim(store, store_path, store_sha)
    _write_manifest(store, store_path, store_sha, claim_sha)
    logline(f"written: {OUT_JSON}, {MAN_JSON}, {RUN_LOG}")
    logline("AT0_7 DONE")
    logline("RUN_RC=0")
    with open(RUN_LOG, "w") as f:
        f.write("\n".join(core._log) + "\n")

    if tr is not None:
        tr_done[0] = True
        tr.log({"finished_at": datetime.now().isoformat(), "status": "ok",
                "verdict": verdict["status"],
                "log_md5": hashlib.md5(open(RUN_LOG, "rb").read()).hexdigest(),
                "wall_sec": round(time.time() - t0, 1), "store_sha": store_sha[:16]},
               step=int(time.time() - t0))
        tr.finish()
        logline(f"trackio: finish ok (status=ok, verdict={verdict['status']})")


if __name__ == "__main__":
    main()
