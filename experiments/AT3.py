#!/usr/bin/env python
"""AT3 (T3.1) — Margin-conditional dip (disk-only).

Пре-рег: docs/AT3_prereg.md (v1 + правка r2; ревью r1). Scope: docs/dynamic/orch/AT3_scope.md.
Канон dip: prism/field.py (vpath, dcurve) — spec-лоад (prism/__init__ тянет matplotlib,
которого нет в /opt/miniconda3 base — паттерн AT2.py).
Общие модули (указание user 16.11): prism/explog.py (ExpLog), prism/nulls_and_stats.py
(paired_bootstrap, spearman) — spec-лоад, без копий.
Стороны: TF (P2_corpus + токены из локального пина, margin из z-формулы) и
генерация (P1_own, saved mg/p1/tok). Layer L23 primary, L17 secondary.
"""
import atexit, importlib.util, json, os, sys, time
from datetime import datetime
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
ART = os.path.join(RESULTS, "AT3", "AT3.json")
LOG_PATH = os.path.join(RESULTS, "AT3_run.log")
SMOKE = os.environ.get("AT3_SMOKE", "0") == "1"
REVISION = os.environ.get("AT3_REVISION", "r1")
SEED = 0
EPS = 1e-6
B_BOOT = 100 if SMOKE else 1000
THRESH = 0.02
LAYER = 23
LAYER_SEC = 17
PIN = os.path.join(
    ROOT, "data/calib_pins/wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json")
MODEL_ST = "/data/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
P1 = os.path.join(RESULTS, "e64/tpu_v8/P1_own.npz")
P2 = os.path.join(RESULTS, "e64/tpu_v8/P2_corpus.npz")

SHA_FILES = {
    "P1_own.npz": "783e16338cdcb652198722a33ecec8a6e8e98fa3c611c687b17db722b0d58198",
    "P2_corpus.npz": "5f42a08442a5d3bcb8d7c4de5c9658340c0378a917a5e633b9b2aa589ceac125",
    "P1_own_meta.json": "3cebe6f328083c39a7c0130555dde84a62c1ab4c745d48b1f28b5f5eb3538042",
    "P2_corpus_meta.json": "5fb47a32f9402074cdc47284a5e26bc7996572563abfc7b46e2c8eb661581a2e",
    "REPLAY.json": "5580d8f2c8341e9a149d18b20ba16bbe8ad43729dacf3441fd1a7905111e5e8f",
    "pin_plain.json": "083c732ff67c04f8585a5c942eadc9fea7fa2134f78cf869826f11ab3d35ce2f",
    "model.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
}
SHA_PATHS = {
    "P1_own.npz": P1, "P2_corpus.npz": P2,
    "P1_own_meta.json": os.path.join(RESULTS, "e64/tpu_v8/P1_own_meta.json"),
    "P2_corpus_meta.json": os.path.join(RESULTS, "e64/tpu_v8/P2_corpus_meta.json"),
    "REPLAY.json": os.path.join(RESULTS, "e81/REPLAY.json"),
    "pin_plain.json": PIN, "model.safetensors": MODEL_ST,
}
DOCS = list(range(16)) if not SMOKE else [0, 8]


# ------------------------------------------------------------------- spec-лоады
def _sl(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


_explog = _sl("prism_explog", os.path.join(ROOT, "prism/explog.py"))
_nulls = _sl("prism_nulls_and_stats", os.path.join(ROOT, "prism/nulls_and_stats.py"))
_pf = _sl("prism_field", os.path.join(ROOT, "prism/field.py"))
vpath, dcurve = _pf.vpath, _pf.dcurve
paired_bootstrap, spearman = _nulls.paired_bootstrap, _nulls.spearman

_log = _explog.ExpLog()
logline = _log


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trackio_start(name, cfg):
    try:
        import trackio
    except Exception as e:
        raise SystemExit(
            f"trackio: модуль не установлен — прогон остановлен "
            f"(AGENTS: trackio обязателен для CPU-прогонов): {e}")
    os.environ.setdefault("TRACKIO_DIR", os.path.join(ROOT, ".trackio"))
    return trackio.init(project="prism", name=name, group="AT3", config=cfg)


def tlog(tr, t, d):
    if tr is not None:
        tr.log(d, step=round(t, 1))


def tstage(tr, t, name):
    if tr is not None:
        tr.log({"stage": name, "stage_s": round(t, 1)}, step=round(t, 1))


# ------------------------------------------------------------------- данные
def load_e_w():
    from safetensors import safe_open
    with safe_open(MODEL_ST, framework="pt") as sf:
        E = sf.get_tensor("model.language_model.embed_tokens.weight").float().numpy()
        wn = sf.get_tensor("model.language_model.norm.weight").float().numpy()
    return E, wn


def tf_tokens():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("/data/Qwen/Qwen3.5-0.8B")
    texts = json.load(open(PIN))["texts"]
    return [np.array(tk.encode(t), dtype=np.int64)[:256] for t in texts]


def logits_of(H, E, wn):
    """z = (rmsnorm(h) * w) @ E.T  (scope §2: cos 0.999 vs real logits, fp16-шум ~4)."""
    H = H.astype(np.float32)
    r = np.sqrt((H.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + EPS)
    Hn = ((H / r.astype(np.float32)) * wn[None, :]).astype(np.float32)
    return Hn @ E.T


def top1_top2(z):
    top2 = np.sort(z, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0], np.exp(top2[:, 1] - np.logaddexp.reduce(z, axis=1))


def dip_positions(H):
    """Per-position dip(1) с t-mapping (пре-рег §1):
    Vo[t] = u[t+1] - <u[t+1], u[t]> u[t], ok = ‖Vo‖ > 1e-6,
    dip[t] = cos(Vo[t], Vo[t+1]) где ok[t] & ok[t+1]; иначе NaN.
    Возвращает (dip (T-2,), ok (T-1,)).
    """
    H = H.astype(np.float64)
    T = H.shape[0]
    u = H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)
    c = np.einsum("td,td->t", H[1:], u[:-1])
    Vo = H[1:] - c[:, None] * u[:-1]
    nv = np.linalg.norm(Vo, axis=1)
    ok = nv > 1e-6
    Vn = np.zeros_like(Vo)
    Vn[ok] = Vo[ok] / nv[ok, None]
    cos1 = np.einsum("td,td->t", Vn[:-1], Vn[1:])
    dip = np.full(T - 2, np.nan)
    m = ok[:-1] & ok[1:]
    dip[m] = cos1[m]
    return dip, ok


def dip_canon_doc(H):
    """Канон dip_V,med per-doc (vpath + dcurve k=1 median) — для S2-якорей."""
    H = H.astype(np.float64)
    hn = np.linalg.norm(H, axis=1)
    u = H / np.maximum(hn[:, None], 1e-12)
    V, dropped = vpath(u, hn)
    return float(dcurve(V, [1], "median")[0]), int(dropped)


def _bin_of(vals, edges):
    return np.clip(np.searchsorted(edges, vals, side="right"), 0, 3)


def side_block(states, margins, p1s, layer_tag, verdict):
    """Блок стороны+слоя: бинация margin (глобальные квартили), per-doc per-bin
    медианы dip, Δ = dip|Q1 − dip|Q4, CI (paired bootstrap по докум), вердикт.
    margin[j] — margin предсказания, сделанного из state[j] (Vo[j] ↔ margin[j]).
    L17: те же margin (финальный readout — L23), другой слой dip; вердиктов нет.
    """
    dips, margs = {}, {}
    for d in DOCS:
        dip, _ok = dip_positions(states[d])
        dips[d] = dip
        margs[d] = margins[d][:dip.shape[0]]
    allm = np.concatenate([margs[d][~np.isnan(dips[d])] for d in DOCS])
    q = [float(np.quantile(allm, x)) for x in (0.25, 0.50, 0.75)]
    per_doc_bin, sizes, bnd_num, n_valid = {}, [0, 0, 0, 0], 0, 0
    for d in DOCS:
        dip, m = dips[d], margs[d]
        v = ~np.isnan(dip)
        b = _bin_of(m[v], q)
        n_valid += int(v.sum())
        per_doc_bin[d] = {}
        for bi in range(4):
            sel = b == bi
            sizes[bi] += int(sel.sum())
            per_doc_bin[d][bi] = (float(np.median(dip[v][sel]))
                                  if sel.sum() >= 3 else None)
            if layer_tag == "L23" and sel.sum():
                dmin = np.abs(m[v][sel][None, :] - np.array(q)[:, None]).min(axis=1)
                bnd_num += int((dmin < 0.05).sum())
    deltas = []
    for d in DOCS:
        a, bb = per_doc_bin[d][0], per_doc_bin[d][3]
        if a is not None and bb is not None:
            deltas.append(a - bb)
    deltas = np.array(deltas)
    if not SMOKE and deltas.size < 4:
        raise SystemExit(f"S5 FAIL: n_docs_delta={deltas.size} < 4 (per-doc Δ не строится)")
    boot = paired_bootstrap(deltas, np.median, B=B_BOOT, seed=SEED)
    dl, dh = boot["ci_low"], boot["ci_high"]
    vd, sub = None, None
    if verdict:
        if dh < -THRESH:
            vd = "a_boundary"
        elif dl > THRESH:
            vd = "c_beyond_map"
        else:
            vd = "b_suffix_map"
            sub = "weak" if abs(boot["point"]) < THRESH else "insufficient_power"
    pooled = {}
    for bi in range(4):
        vals = np.concatenate(
            [dips[d][~np.isnan(dips[d])][_bin_of(margs[d][~np.isnan(dips[d])], q) == bi]
            for d in DOCS])
        pooled[bi] = float(np.median(vals))
    p1_block = None
    if p1s is not None:
        allp = np.concatenate(
            [p1s[d][:dips[d].shape[0]][~np.isnan(dips[d])] for d in DOCS])
        qp = [float(np.quantile(allp, x)) for x in (0.25, 0.50, 0.75)]
        p1per = {}
        for d in DOCS:
            dip = dips[d]
            p = p1s[d][:dip.shape[0]]
            v = ~np.isnan(dip)
            bb = _bin_of(p[v], qp)
            p1per[d] = {}
            for bi in range(4):
                sel = bb == bi
                p1per[d][bi] = (float(np.median(dip[v][sel]))
                                if sel.sum() >= 3 else None)
        p1_block = {
            "quartiles": qp,
            "per_doc_median_docs": {
                bi: float(np.median([p1per[d][bi] for d in DOCS
                                     if p1per[d][bi] is not None]))
                for bi in range(4)},
        }
    return dict(
        layer=layer_tag,
        margin_quartiles=q,
        bin_sizes=sizes,
        n_positions_valid=n_valid,
        n_docs_delta=int(len(deltas)),
        per_doc_bin_median=per_doc_bin,
        delta_per_doc=[float(x) for x in deltas],
        delta_median_docs=float(np.median(deltas)),
        ci95=[float(dl), float(dh)],
        pooled_median=pooled,
        bin_boundary_frac=(bnd_num / max(1, sum(sizes))) if layer_tag == "L23" else None,
        p1=p1_block,
        verdict=vd,
        verdict_sub=sub,
        monotonic=[float(np.median([per_doc_bin[d][bi] for d in DOCS
                                    if per_doc_bin[d][bi] is not None]))
                   for bi in range(4)],
    )


def verdict_from(delta_md, ci):
    """S6: вердикт восстанавливается из (delta, ci) — та же логика, что в блоке."""
    if ci[1] < -THRESH:
        return "a_boundary"
    if ci[0] > THRESH:
        return "c_beyond_map"
    return "b_suffix_map"


# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    logline(f"AT3 (T3.1) start {datetime.now().isoformat()} py={sys.version.split()[0]} "
            f"smoke={SMOKE} rev={REVISION}")

    tr, tr_done = None, [False]
    if not SMOKE:
        tr_cfg = {
            "started_at": datetime.now().isoformat(),
            "revision": REVISION,
            "script_sha": sha256_file(os.path.abspath(__file__)),
            "seed": SEED, "device": "cpu",
            "model": "Qwen/Qwen3.5-0.8B (tied E; z=(rmsnorm(h)*w)@E.T)",
            "log": os.path.abspath(LOG_PATH),
            "layer": LAYER, "layer_sec": LAYER_SEC,
            "boot_b": B_BOOT, "n_docs": len(DOCS), "threshold": THRESH,
        }
        tr = trackio_start(f"at3_margin_{REVISION}", tr_cfg)
        tlog(tr, 0, {"started": 1.0})
        logline(f"trackio: run at3_margin_{REVISION} init ok")

        def _trackio_on_exit():
            if not tr_done[0]:
                try:
                    tlog(tr, time.time() - t0, {
                        "finished_at": datetime.now().isoformat(), "status": "crash",
                        "error": "unhandled exception / abort (fin-точка не записана)"})
                    tr.finish()
                except Exception:
                    pass
        atexit.register(_trackio_on_exit)

    gates = {}

    # --- S1: sha ------------------------------------------------------------
    tstage(tr, time.time() - t0, "gates")
    if not SMOKE:
        for fn, expect in SHA_FILES.items():
            got = sha256_file(SHA_PATHS[fn])
            if got != expect:
                raise SystemExit(f"S1 FAIL: sha {fn} = {got} != {expect}")
        gates["S1_sha"] = True
        logline("S1 sha-гейт: 7/7 OK")

    tstage(tr, time.time() - t0, "load")
    E, wn = load_e_w()
    z1 = np.load(P1)
    z2 = np.load(P2)
    tf_ids = tf_tokens()
    logline(f"load: E {E.shape}, wn min {wn.min():.3f}, "
            f"tf_ids lens {[len(a) for a in tf_ids][:4]}...")

    # --- states --------------------------------------------------------------
    gen_states = {d: z1[f"own_w{d}_L{LAYER}"].astype(np.float32) for d in DOCS}
    gen_states17 = {d: z1[f"own_w{d}_L{LAYER_SEC}"].astype(np.float32) for d in DOCS}
    tf_states = {d: z2[f"cor_w{d}_L{LAYER}"].astype(np.float32) for d in DOCS}
    tf_states17 = {d: z2[f"cor_w{d}_L{LAYER_SEC}"].astype(np.float32) for d in DOCS}
    gen_mg = {d: z1[f"own_w{d}_mg"].astype(np.float64) for d in DOCS}
    gen_p1 = {d: z1[f"own_w{d}_p1"].astype(np.float64) for d in DOCS}

    # --- S2: dip-якоря (canonical, L23) ---------------------------------------
    tstage(tr, time.time() - t0, "anchors")
    anchors = {}
    for side, st in (("tf", tf_states), ("gen", gen_states)):
        vals = [dip_canon_doc(st[d])[0] for d in DOCS]
        anchors[side] = float(np.median(vals))
        logline(f"S2 {side} L23 dip_V,med median-of-docs = {anchors[side]:.6f}")
    if not SMOKE:
        if not (-0.16714 - 0.001 <= anchors["tf"] <= -0.16714 + 0.001):
            raise SystemExit(f"S2 FAIL: tf anchor {anchors['tf']} вне [-0.16814, -0.16614]")
        if not (-0.1111 - 0.002 <= anchors["gen"] <= -0.1111 + 0.002):
            raise SystemExit(f"S2 FAIL: gen anchor {anchors['gen']} вне [-0.1131, -0.1091]")
        gates["S2_anchors"] = True
        logline("S2 anchor-гейт: OK (tf -0.16714±0.001, gen -0.1111±0.002)")

    # --- TF: z → margin, p1, gate ----------------------------------------------
    tstage(tr, time.time() - t0, "tf_margin")
    tf_mg, tf_p1, tf_gate = {}, {}, {}
    for d in DOCS:
        n = tf_states[d].shape[0]
        z = logits_of(tf_states[d], E, wn)
        mg, p1 = top1_top2(z)
        tf_mg[d] = mg
        tf_p1[d] = p1
        tf_gate[d] = float((z.argmax(axis=1) == tf_ids[d][1:n + 1]).mean())
    if not SMOKE:
        bad = [d for d in DOCS if tf_gate[d] < 0.25]
        if bad:
            raise SystemExit(
                f"S3 FAIL: TF gate < 0.25 на докум {bad}: "
                f"{[round(tf_gate[d], 3) for d in bad]}")
        gates["S3_tf_gate"] = True
        logline(f"S3 TF-gate: OK (min {min(tf_gate.values()):.4f}, "
                f"max {max(tf_gate.values()):.4f})")

    # --- S4: alignment-гейт (gen: Spearman recomputed vs saved) -----------------
    tstage(tr, time.time() - t0, "gen_s4")
    s4 = {}
    if not SMOKE:
        for d in DOCS:
            n = gen_states[d].shape[0]
            z = logits_of(gen_states[d], E, wn)
            mgz, _ = top1_top2(z)
            r = spearman(mgz[:n - 2], gen_mg[d][:n - 2])  # float | None (prism, не scipy)
            s4[d] = float(r)
            if r is None or r < 0.95:
                raise SystemExit(f"S4 FAIL: doc w{d} spearman {r} < 0.95 (None = вырожденный)")
        gates["S4_alignment"] = True
        logline(f"S4 alignment: OK (min {min(s4.values()):.4f})")

    # --- L23: обе стороны --------------------------------------------------------
    tstage(tr, time.time() - t0, "dip_l23")
    gen_block = side_block(gen_states, gen_mg, gen_p1, "L23", verdict=True)
    tf_block = side_block(tf_states, tf_mg, tf_p1, "L23", verdict=True)
    logline(f"L23 gen: Δ {gen_block['delta_median_docs']:+.4f} CI {gen_block['ci95']} "
            f"verdict={gen_block['verdict']}")
    logline(f"L23 tf : Δ {tf_block['delta_median_docs']:+.4f} CI {tf_block['ci95']} "
            f"verdict={tf_block['verdict']}")

    # --- S5: bin-size -------------------------------------------------------------
    tstage(tr, time.time() - t0, "bin_size")
    if not SMOKE:
        for nm, blk in (("gen", gen_block), ("tf", tf_block)):
            small = [b for b, s in enumerate(blk["bin_sizes"]) if s < 50]
            if small:
                raise SystemExit(f"S5 FAIL: {nm} L23 bin-size < 50: {blk['bin_sizes']}")
        gates["S5_bin_size"] = True
        logline(f"S5 bin-size: OK (gen {gen_block['bin_sizes']}, "
                f"tf {tf_block['bin_sizes']})")

    tstage(tr, time.time() - t0, "dip_l17")
    gen_block17 = side_block(gen_states17, gen_mg, None, "L17", verdict=False)
    tf_block17 = side_block(tf_states17, tf_mg, None, "L17", verdict=False)

    # --- итог ----------------------------------------------------------------------
    tstage(tr, time.time() - t0, "finalize")
    result = {
        "run": f"at3_margin_{REVISION}",
        "smoke": SMOKE,
        "seed": SEED,
        "threshold": THRESH,
        "gates": gates,
        "anchors": anchors,
        "tf_gate": {f"w{d}": v for d, v in tf_gate.items()},
        "s4_spearman": {f"w{d}": v for d, v in s4.items()},
        "L23_gen": gen_block,
        "L23_tf": tf_block,
        "L17_gen": gen_block17,
        "L17_tf": tf_block17,
        "verdict_tf": tf_block["verdict"],
        "verdict_gen": gen_block["verdict"],
        "wall_sec": round(time.time() - t0, 1),
    }

    def _cast(o):
        if isinstance(o, dict):
            return {k: _cast(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_cast(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    with open(ART, "w") as f:
        json.dump(_cast(result), f, indent=1, allow_nan=False)
    logline(f"AT3 (T3.1) DONE wall={time.time() - t0:.1f}s "
            f"verdicts: tf={tf_block['verdict']} gen={gen_block['verdict']}")
    _log.write_to(LOG_PATH)
    if tr is not None:
        tlog(tr, time.time() - t0, {
            "finished_at": datetime.now().isoformat(), "status": "ok",
            "verdict_tf": tf_block["verdict"], "verdict_gen": gen_block["verdict"],
            "delta_tf": tf_block["delta_median_docs"],
            "delta_gen": gen_block["delta_median_docs"]})
        tr.finish()
        tr_done[0] = True


if __name__ == "__main__":
    if "--recheck" in sys.argv:
        # S6: вердикты восстанавливаемы из полей артефакта
        r = json.load(open(sys.argv[sys.argv.index("--recheck") + 1]))
        for side in ("L23_tf", "L23_gen"):
            blk = r[side]
            got = verdict_from(blk["delta_median_docs"], blk["ci95"])
            if got != blk["verdict"]:
                raise SystemExit(f"S6 FAIL: {side} verdict {got} != {blk['verdict']}")
        print("S6 recheck OK:", r["verdict_tf"], r["verdict_gen"])
    else:
        main()
