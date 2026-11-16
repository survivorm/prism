# -*- coding: utf-8 -*-
"""AT8_2 (T8.2) — «Оконное восстановление + доза» (CPU). TZ docs/AT8_2_TZ.md v1.2.

В здоровую модель (λ=1, base Qwen3.5-0.8B) вставляется блок «луп» (фраза × n), меряется
dip в окнах ПОСЛЕ вставки против того же окна контроля; измеряется длина восстановления
ℓ_rec(n) (ось A по W=64) и цель восстановления (контроль vs глобальное разбавление,
to_global на одной шкале — dip_control_whole). Обобщение E-71B
(experiments/e71_80/exp71_soliton.py часть B): там фикс. доза 8, окно W=64; здесь
n∈{2,4,8,16}, W∈{8,16,32,64}, 8 доков, + ℓ_rec(n) + dilution-check.

Р1-фиксы (TZ-ревью):
- MAJOR-1: dip_control_whole + to_global на одной документной шкале;
- MAJOR-2: S8 in-process (те же h23-массивы, одна функция) — tol 1e-12;
- MINOR-1: ось A — W=64 основная, combined вторичный;
- MINOR-2: floor-ограничение — в claim.prereg.note;
- MINOR-3: S2b control_sha256 + json-sanitize (float-касты, NaN→None);
- MINOR-4: W=8 — W8_noisy_flag в ячейках, не в оси A.

Р2-фиксы (results-ревью r1, docs/dynamic/AT8_2_results_review.md; v2-реанализ уже
пересчитал агрегаты из r1-store — этот код для будущих прогонов):
- MAJOR-1: ось A — полная популяция (все измеримые (doc,W) с ℓ_rec, TZ §5);
  censored_excluded — вторичный, явно помечен (at8_2_analysis.axis_A);
- MINOR-1: NOT_RESOLVED-гейты по явным популяциям (at8_2_analysis.compute_verdict);
- NIT-3: hook в measure() снимается в try/finally;
- MINOR-2 (trackio, AGENTS п.3–5): seed в config (n/a — детерминированный forward),
  stage-точки на крупных этапах, log_md5 в finish;
- resume: решение «что пересчитывать» — функция at8_2_analysis.resume_plan
  (designed-сценарий протестирован exec-тестом, AGENTS п.2).

Механика dip — пин e64_gpu.py (гейт S7). Запуск: ./run_one.sh at8_2 ...
(OMP=6, nice-10; TRACKIO_DIR экспортирует run_one.sh — AGENTS п.6).
trackio: run at8_2_recovery_{REV} (AGENTS: обязателен для CPU-прогонов).
"""
import ast, hashlib, json, os, sys, time
from datetime import datetime, timezone
import numpy as np
import torch

# torch-fallback (без fla/triton) — референсный путь серии (AT8_1): на CPU triton-драйвер
# поднимает "0 active drivers".
sys.modules["fla"] = False
sys.modules["fla.ops"] = False

torch.set_grad_enabled(False)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # experiments/at8/ → repo
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from at8_2_analysis import (N_DOSE, W_SET, W_AXIS, EPS_MIN, THR, D_MIN_MEAS,
                            build_cells_from_store, axis_A, axis_B,
                            compute_verdict, resume_plan)

# ---- константы (зафиксированы в TZ) ----
MODEL = "/data/Qwen/Qwen3.5-0.8B"
LAYER = 23
PHRASE = " The class will be held at the park."          # 9 токенов (E-71B)
PHRASE_IDS = [561, 523, 668, 381, 5467, 506, 279, 5926, 13]   # TZ §3 (add_special_tokens=False)
POOL_PIN = "data/calib_pins/wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json"
POOL_FP = "144c722df3046f2d"
E64_PATH = "experiments/kaggle_e64gpu/e64_gpu.py"
E64_SHA = "8c517e18f801fa877dfde9265ed5ce18df1e7d3af705194fc01ff6b150cf7633"

OUT_DIR = os.path.join(_ROOT, "results", "AT8")
CLAIM = os.path.join(OUT_DIR, "AT8_2_RECOVERY.json")
STORE = os.path.join(OUT_DIR, "AT8_2_RECOVERY_store.json")
MANIFEST = os.path.join(OUT_DIR, "AT8_2_RECOVERY_manifest.json")
RUNLOG = os.path.join(_ROOT, "logs", "AT8", "AT8_2_run.log")
REVISION = "r2"


def logline(s):
    print(s, flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_np(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- dip-механика из e64_gpu.py (гейт S7) ----
def load_dip_machinery():
    src = open(os.path.join(_ROOT, E64_PATH)).read()
    sha = hashlib.sha256(src.encode()).hexdigest()
    assert sha == E64_SHA, f"S7 fail: e64_gpu.py sha {sha[:16]} != {E64_SHA[:16]}"
    tree = ast.parse(src)
    fns = {n.name: ast.get_source_segment(src, n)
           for n in tree.body if isinstance(n, ast.FunctionDef)}
    ns = {"np": np}
    for w in ["vpath_units", "dcurve", "curves_series"]:
        exec(fns[w], ns)
    return ns["curves_series"], sha


# ---- документы из пина (plain-путь load_calibration_data, как E-71B) ----
def load_docs():
    meta = json.load(open(os.path.join(_ROOT, POOL_PIN)))
    assert meta["fp"] == POOL_FP, f"pool_fp mismatch: {meta['fp']} != {POOL_FP}"
    tk = AutoTokenizer.from_pretrained(MODEL)
    docs, names = [], []
    for i, t in enumerate(meta["texts"]):
        tok = tk(t, max_length=256, truncation=True, return_tensors="pt")
        if tok["input_ids"].shape[1] >= 64:
            docs.append(tok["input_ids"][0].tolist()[:256])
            names.append(f"wiki_{i}")
        if len(docs) >= 8:
            break
    return docs, names, tk


def docs_sha256(docs, names):
    h = hashlib.sha256()
    for nm, d in zip(names, docs):
        h.update(nm.encode())
        h.update(np.array(d, dtype=np.int64).tobytes())
    return h.hexdigest()


# ---- forward: h23 (layer 23) + nll ----
def measure(m, ids):
    caps = {}
    hook = m.model.layers[LAYER].register_forward_hook(
        lambda mod, i, o, l=LAYER: caps.__setitem__(l, o.detach()))
    t = torch.tensor([ids])
    try:
        with torch.no_grad():
            o = m(input_ids=t, use_cache=False)
    finally:
        hook.remove()   # NIT-3 (r2): снятие в try/finally — при краше forward хук не утекает
    nll = float(torch.nn.functional.cross_entropy(o.logits[0, :-1].float(), t[0, 1:]))
    h23 = caps[LAYER][0, :len(ids)].to(torch.float32).numpy()
    return nll, h23


def win_dips(curves_series, h23, start, W):
    """dip окна h23[start:start+W] (None, если не влезает)."""
    if start + W > len(h23):
        return None
    return float(curves_series(h23[start:start + W])["dip"])


# ---- trackio (AGENTS: обязателен для CPU-прогонов) ----
def init_trackio(t0):
    # AGENTS п.6: TRACKIO_DIR задаёт run_one.sh — скрипт его не переопределяет.
    # Fail-loud (r2 NIT-5): прямой запуск без run_one.sh — не писать live-точки в
    # дефолтный HF-cache (сценарий AT2/AT3) — стоп с явным сообщением.
    if not os.environ.get("TRACKIO_DIR"):
        raise SystemExit(
            "trackio: TRACKIO_DIR не задан — запускать через ./run_one.sh "
            "(он экспортирует TRACKIO_DIR=$PRISM_DIR/.trackio, AGENTS п.6); "
            "прямой запуск остановлен (иначе live-точки ушли бы в HF-cache)")
    try:
        import trackio
    except ImportError as e:
        raise SystemExit(f"trackio: модуль не установлен — прогон остановлен "
                         f"(AGENTS: trackio обязателен для CPU-прогонов): {e}")
    tr_cfg = {
        "started_at": datetime.now().isoformat(),
        "revision": REVISION,
        "script_sha": sha256_file(os.path.abspath(__file__)),
        "device": "CPU", "model": MODEL, "layer": LAYER,
        "n_dose": N_DOSE, "W_set": W_SET, "W_axis": W_AXIS,
        "use_cache": False, "log": os.path.abspath(RUNLOG),
        "seed": "n/a (deterministic forward, no sampling)",   # AGENTS п.5 (r2 MINOR-2)
        "e64_gpu_sha": E64_SHA[:16], "pool_fp": POOL_FP,
    }
    tr = trackio.init(project="prism", name=f"at8_2_recovery_{REVISION}", group="AT8", config=tr_cfg)
    tr.log({"started": 1.0}, step=0)
    logline(f"trackio: run at8_2_recovery_{REVISION} init ok")
    done = [False]
    import atexit
    def _on_exit():
        if not done[0]:
            try:
                tr.log({"finished_at": datetime.now().isoformat(), "status": "crash"},
                       step=int(time.time() - t0))
                tr.finish()
            except Exception:
                pass
    atexit.register(_on_exit)
    return tr, done


def tstage(tr, t0, name):
    """AGENTS п.4: крупные этапы — точка {"stage": ..., "stage_s": wall-t}."""
    tr.log({"stage": name, "stage_s": round(time.time() - t0, 1)})
    logline(f"stage: {name} ({time.time() - t0:.1f}s)")


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(RUNLOG), exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    # run-log (трекаемый артефакт, TZ §9): tee stdout → logs/AT8/AT8_2_run.log
    _logf = open(RUNLOG, "w")
    class _Tee:
        def __init__(self, a, b): self.a, self.b = a, b
        def write(self, s):
            self.a.write(s); self.b.write(s); self.b.flush()
        def flush(self):
            self.a.flush(); self.b.flush()
    sys.stdout = _Tee(sys.__stdout__, _logf)
    logline(f"AT8_2 (T8.2) start {datetime.now(timezone.utc).isoformat()} "
            f"py={sys.version.split()[0]} torch={torch.__version__} (CPU, windowed recovery, rev={REVISION})")

    tr, tr_done = init_trackio(t0)

    curves_series, e64_sha = load_dip_machinery()
    logline(f"S7: e64_gpu.py sha {e64_sha[:16]} PASS")

    # ---- S1: config ----
    cfg = AutoConfig.from_pretrained(MODEL)
    n_layers = getattr(cfg, "num_hidden_layers", None)
    hidden = getattr(cfg, "hidden_size", None)
    vocab = getattr(cfg, "vocab_size", None)
    sub = getattr(cfg, "text_config", None)
    if n_layers is None and sub is not None:   # qwen3_5: вложенный конфиг
        n_layers = getattr(sub, "num_hidden_layers", n_layers)
        hidden = getattr(sub, "hidden_size", hidden)
        vocab = getattr(sub, "vocab_size", vocab)
    s1 = dict(num_hidden_layers=n_layers, hidden_size=hidden,
              vocab_size=vocab, model_type=getattr(cfg, "model_type", None),
              dtype="fp32", eval=True)
    assert n_layers == 24 and hidden == 1024, f"S1 fail: {s1}"
    logline(f"S1: config {s1} PASS")

    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    tstage(tr, t0, "model_loaded")

    # ---- S3: доки ----
    docs, names, tk = load_docs()
    dsha = docs_sha256(docs, names)
    lens = [len(d) for d in docs]
    logline(f"S3: {len(docs)} docs lens={lens} docs_sha256={dsha[:16]} pool_fp={POOL_FP}")
    phrase_ids = PHRASE_IDS
    pcheck = tk(PHRASE, add_special_tokens=False)["input_ids"]
    assert pcheck == phrase_ids, f"phrase-токены не совпадают: {pcheck} != {phrase_ids}"
    logline(f"phrase: {len(phrase_ids)} токенов {phrase_ids} (сверено с токенизатором)")
    tstage(tr, t0, "docs_loaded")

    # ---- два прохода: forward → dip-кривые по d (скаляры) → store ----
    res = {}
    if os.path.exists(STORE):
        res = json.load(open(STORE))
    # r2 (AGENTS п.2): решение «что пересчитывать» — чистая функция resume_plan
    # (протестирована exec-тестом experiments/at8/test_at8_2_resume.py)
    plan = resume_plan(names, N_DOSE, res)
    if res:
        logline(f"resume: в store {len(res)} записей, план resume: {plan['n_resume']} "
                f"(ctrl={sum(plan['ctrl_resume'].values())}, "
                f"dose={sum(plan['dose_resume'].values())})")
    for di, ids in enumerate(docs):
        nm = names[di]
        T = len(ids)
        cut = T // 2
        # контроль: один forward на док (одна статистика на обеих сторонах)
        if not plan["ctrl_resume"][nm]:
            nll_c, h0 = measure(m, ids)
            csha = sha256_np(h0)
            dip_ctrl_curve = {}
            dip_global_c = float(curves_series(h0)["dip"])
            for W in W_SET:
                D_max = T - cut - W
                if D_max < 0:
                    dip_ctrl_curve[str(W)] = None
                    continue
                dip_ctrl_curve[str(W)] = [
                    win_dips(curves_series, h0, cut + d, W) for d in range(D_max + 1)]
            res[f"{nm}__ctrl"] = dict(T=T, cut=cut, nll=round(nll_c, 6),
                                      dip_ctrl=dip_ctrl_curve,
                                      dip_control_whole=round(dip_global_c, 6),
                                      h23_sha=csha)
            logline(f"  {nm} ctrl: T={T} cut={cut} nll={nll_c:.4f} "
                    f"dip_control_whole={dip_global_c:+.4f}")
            json.dump(res, open(STORE, "w"), default=float)
            del h0  # h23 не храним (двухпроходность) — dip-кривые уже в store

        for n in N_DOSE:
            if plan["dose_resume"][(nm, n)]:
                continue
            L_n = len(phrase_ids) * n
            inj = ids[:cut] + phrase_ids * n + ids[cut:]
            nll_i, h1 = measure(m, inj)
            dip_global = float(curves_series(h1)["dip"])
            dip_inj_curve = {}
            for W in W_SET:
                D_max = T - cut - W
                if D_max < 0:
                    dip_inj_curve[str(W)] = None
                    continue
                dip_inj_curve[str(W)] = [
                    win_dips(curves_series, h1, cut + L_n + d, W) for d in range(D_max + 1)]
            res[f"{nm}__n{n}"] = dict(T=T, cut=cut, L_n=L_n, nll=round(nll_i, 6),
                                      dip_inj=dip_inj_curve,
                                      dip_global=round(dip_global, 6))
            logline(f"  {nm} n={n}: L_n={L_n} len(inj)={len(inj)} nll={nll_i:.4f} "
                    f"dip_global={dip_global:+.4f}")
            tr.log({"doc": di + 1, "n": n, "nll": nll_i,
                    "dip_global": dip_global}, step=int(time.time() - t0))
            json.dump(res, open(STORE, "w"), default=float)
            del h1
    tstage(tr, t0, "forwards_done")

    # ---- S2: полнота ----
    expected = set()
    for nm in names:
        expected.add(f"{nm}__ctrl")
        for n in N_DOSE:
            expected.add(f"{nm}__n{n}")
    have = set(res.keys())
    assert have == expected, f"S2 fail: missing={expected - have} extra={have - expected}"
    logline(f"S2: полнота {len(have)}/{len(expected)} PASS")

    # ---- S2b: одна статистика контроля на обеих сторонах (р1 MINOR-3) ----
    # контроль считается один раз на док и переиспользуется по всем дозам:
    # все записи доз ссылаются на тот же h0 (в анализе — одна dip_ctrl-кривая
    # на (doc,W) для всех n). control_sha256 — h23_sha по doc из store.
    s2b = {nm: res[f"{nm}__ctrl"]["h23_sha"] for nm in names}
    n_dose_reuse = len(names) * len(N_DOSE)
    logline(f"S2b: control_sha256 (1 на док, reuse={n_dose_reuse} доз): "
            + " ".join(f"{k}={v[:8]}" for k, v in s2b.items()))

    # ---- S4: nll finite ----
    s4_ok = all(np.isfinite(res[k]["nll"]) for k in have)
    logline(f"S4: nll finite в {sum(1 for k in have if np.isfinite(res[k]['nll']))}/{len(have)} "
            f"{'PASS' if s4_ok else 'FAIL'}")
    assert s4_ok, "S4 fail: nll не finite"

    # ---- анализ: ℓ_rec, dilution, оси, вердикт — ОБЩИЙ фрагмент (at8_2_analysis) ----
    tstage(tr, t0, "analysis")
    cells = build_cells_from_store(res, names)
    meas = [c for c in cells.values() if "dip_inj_d0" in c and c.get("status") is None]
    logline(f"окна: {len(cells)} пар (doc,n,W); measurable={len(meas)}")

    # ---- S5: dip finite ----
    s5_ok = all(np.isfinite(c["dip_inj_d0"]) and np.isfinite(c["dip_ctrl_d0"]) and
                np.isfinite(c["dip_global"]) and np.isfinite(c["dip_control_whole"])
                for c in meas)
    logline(f"S5: dip finite в {len(meas)} measurable окнах {'PASS' if s5_ok else 'FAIL'}")
    assert s5_ok, "S5 fail"

    axA = axis_A(cells)
    axB = axis_B(cells)
    verdict, vdetail, diag = compute_verdict(axA, axB, cells)
    logline(f"ось A (W={W_AXIS}, полная популяция): ratio={axA['W64']['ratio']} "
            f"b={axA['W64']['b']} r2={axA['W64']['r2']} lrec={axA['W64']['lrec']}")
    logline(f"  censored_excluded (вторичный): lrec={axA['W64']['lrec_censored_excluded']}")
    logline(f"ось B: frac_to_global={axB['frac_to_global']} (n={axB['n_to_global_pairs']})")

    # ---- вердикт (приоритет DIFFUSION > ELASTIC > DILUTION > NOT_RESOLVED, TZ §5) ----
    match = (verdict == "ELASTIC")
    logline(f"VERDICT: {verdict} (match_pre_reg={match}): {vdetail}")
    logline(f"diag: {diag}")

    # ---- S8: in-process самопроверка механики E-71B (р1 MAJOR-2) ----
    # wiki_1, n=8, W=64, d=0: E-71B-спецкейс (срез h1[cut+L8:cut+L8+64]) vs общий
    # путь (dip_inj-кривая при d=0) — ОДИН процесс, ОДИН forward, ОДИНАКОВЫЕ срезы →
    # bit-идентично (tol 1e-12; от числа нитей не зависит — пересчёт не вне процесса).
    assert "wiki_1" in names, "S8: wiki_1 отсутствует в пуле (фиксированный пул, ТЗ §2)"
    di1 = names.index("wiki_1")
    ids1 = docs[di1]
    T1 = len(ids1); cut1 = T1 // 2
    L8 = len(phrase_ids) * 8
    c8 = res[f"wiki_1__ctrl"]
    i8 = res[f"wiki_1__n8"]
    dip_post_e71b = i8["dip_inj"]["64"][0]        # общий путь: win_dips(h1, cut1+L8, 64)
    dip_ctrl_e71b = c8["dip_ctrl"]["64"][0]       # общий путь: win_dips(h0, cut1, 64)
    # независимый пересчёт по определению E-71B на тех же h23 (один forward):
    n0chk, h0chk = measure(m, ids1)
    n1chk, h1chk = measure(m, ids1[:cut1] + phrase_ids * 8 + ids1[cut1:])
    cw_e71 = float(curves_series(h1chk[cut1 + L8:cut1 + L8 + 64])["dip"])
    kw_e71 = float(curves_series(h0chk[cut1:cut1 + 64])["dip"])
    dev_post = abs(cw_e71 - dip_post_e71b)
    dev_ctrl = abs(kw_e71 - dip_ctrl_e71b)
    s8 = dict(doc="wiki_1", n=8, W=64, d=0,
              dip_post_window_store=round(dip_post_e71b, 6),
              dip_control_window_store=round(dip_ctrl_e71b, 6),
              dip_post_window_recompute=round(cw_e71, 6),
              dip_control_window_recompute=round(kw_e71, 6),
              dev_post=dev_post, dev_ctrl=dev_ctrl,
              nll_recompute=(round(n0chk, 6), round(n1chk, 6)),
              tol=1e-12,
              ok=bool(dev_post <= 1e-12 and dev_ctrl <= 1e-12))
    logline(f"S8: E-71B in-process wiki_1 n=8 W=64: dev_post={dev_post:.2e} "
            f"dev_ctrl={dev_ctrl:.2e} pass={s8['ok']}")
    assert s8["ok"], "S8 fail: обобщение сломало механику E-71B"

    # ---- артефакты ----
    tstage(tr, t0, "artifact")
    claim = dict(
        experiment="AT8_2", cell="T8.2 windowed recovery + dose",
        tz_version="v1.2 (16.11, по results-review r1: MAJOR-1 + MINOR-1)",
        date=datetime.now(timezone.utc).isoformat(),
        setup=dict(model=MODEL, layer=LAYER, phrase=PHRASE, phrase_ids=phrase_ids,
                   L_n={f"n{n}": len(phrase_ids) * n for n in N_DOSE},
                   W_set=W_SET, W_axis=W_AXIS, n_dose=N_DOSE, use_cache=False,
                   threads="run_one.sh OMP/MKL/OPENBLAS=6", device="cpu",
                   e64_gpu_sha=e64_sha, trackio_run=f"at8_2_recovery_{REVISION}"),
        docs=dict(names=names, lens=lens, docs_sha256=dsha, pool_fp=POOL_FP,
                  control_sha256=s2b),
        gates=dict(S1=s1, S2="pass", S2b=dict(control_sha256=s2b,
                   n_dose_reuse=n_dose_reuse, note="контроль — 1 forward на док, "
                   "переиспользуется по всем дозам (одна статистика на обеих сторонах)"),
                   S3=dict(docs_sha256=dsha, pool_fp=POOL_FP),
                   S4="pass" if s4_ok else "fail",
                   S5="pass" if s5_ok else "fail",
                   S7=dict(e64_gpu_sha=e64_sha, ok=True), S8=s8),
        prereg=dict(eps_min=EPS_MIN, thr=THR, D_max_rule="T - cut - W",
                    d_min_meas=D_MIN_MEAS,
                    verdict_thresholds=dict(
                        diffusion="ratio>=4 & b>0 & r2>=0.6 (W=64, полная популяция)",
                        elastic="ratio<=2 & frac_to_global<=0.4 & n_pos>=2 (W=64; "
                                "n_pos>=2 — доп. консервативный гейт, TZ v1.2)",
                        dilution="frac_to_global>0.6",
                        not_resolved="<6 пар (doc,n) с l_rec>0 на W=64 ИЛИ "
                                     "большинство not_recovered по полной измеримой "
                                     "популяции (TZ v1.2 §5)"),
                    note=("р1 MINOR-2 (ограничение мощности): при малом возмущении "
                          "(0.01<=pert0<~0.04) порог 0.25*pert0 может быть ниже шумового "
                          "пола dip => l_rec->D_max (not_recovered) даже при восстановлении; "
                          "ось A неинформативна в маловозмущённом режиме — вердикт сходится "
                          "через ось B")),
        lrec_by_n=dict(W64=axA["W64"]["lrec"],
                       W64_censored_excluded=axA["W64"]["lrec_censored_excluded"],
                       combined_secondary=axA["combined_secondary"]["lrec"],
                       combined_secondary_censored_excluded=
                       axA["combined_secondary"]["lrec_censored_excluded"]),
        axis_A=axA, axis_B=axB,
        diagnostics=diag,
        cells=cells,
        verdict=dict(value=verdict, match_pre_reg=match, detail=vdetail),
        summary=(f"T8.2: {verdict}. {vdetail}. 40 форвардов + 2 S8-check, "
                 f"wall {time.time()-t0:.0f}s."),
        wall_s=round(time.time() - t0, 1),
    )

    def _san(o):
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_san(v) for v in o]
        if isinstance(o, (np.floating,)):
            o = float(o)
        if isinstance(o, (np.integer,)):
            o = int(o)
        if isinstance(o, (np.bool_,)):
            o = bool(o)
        if isinstance(o, float) and not np.isfinite(o):
            o = "inf" if o == float("inf") else "-inf"
        return o
    claim = _san(claim)
    json.dump(claim, open(CLAIM, "w"), indent=1, ensure_ascii=False)
    csha = sha256_file(CLAIM)
    man = dict(claim_file_sha256=csha, store_file_sha256=sha256_file(STORE),
               e64_gpu_sha=e64_sha, docs_sha256=dsha, pool_fp=POOL_FP,
               model=MODEL, date=datetime.now(timezone.utc).isoformat(),
               verdict=verdict, wall_s=claim["wall_s"])
    json.dump(man, open(MANIFEST, "w"), indent=1, ensure_ascii=False)
    logline(f"written: {CLAIM} (sha {csha[:16]}), {MANIFEST}")

    tr.log({"finished_at": datetime.now(timezone.utc).isoformat(), "status": "ok",
            "verdict": verdict, "match_pre_reg": match,
            "claim_sha": csha[:16], "wall_sec": round(time.time() - t0, 1),
            "log_md5": md5_file(RUNLOG)},          # AGENTS п.5 (r2 MINOR-2)
           step=int(time.time() - t0))
    tr.finish()
    tr_done[0] = True
    logline(f"trackio: finish ok (status=ok, verdict={verdict})")
    logline(f"AT8_2_DONE verdict={verdict} match_pre_reg={match} (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
