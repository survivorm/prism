#!/usr/bin/env python
"""AT3 (T3.1) — Margin-conditional dip (disk-only), r2 (правка по ревью r1).

Пре-рег: docs/AT3_prereg.md (r3). Ревью: docs/AT3_review_r1.md (user) +
docs/AT3_review_answers_r1.md (ответы). Scope: docs/dynamic/orch/AT3_scope.md.

r2 (все — disk-only, без пересчёта dip):
- F1: DEGENERATE_DOCS={0,4} (AT0_4 token_provenance: период-1 лупы) исключены
  из ВСЕХ научных агрегатов (S2-якоря остаются по 16 докам — это воспроизведение).
- F2: точный хэд для pre-norm состояний e64 = (h/rms)·w·Eᵀ (S4 0.9868,
  fidelity 0.9958 ≈ ТЗ «gate 1.000»); no-w — sensitivity (ТЗ-конвенция
  относится к post-norm L23 jcache; на этом кэхе no-w хуже: S4 0.8974).
- F3/F4: порог verdict = max(0.02, разброс агрегаций) — записано в артефакт;
  sign-test в отчёте.
- F5: per_doc_bin_n + min-n sensitivity {3,20,50}.
- F6: tf_p1 в блоках; dip по p1-квантилям на обеих сторонах (L23 и L17).
- F7: ±1 margin-shift sensitivity (dip[j] ↔ mg[j] vs mg[j+1]).
- Мелкое: bin_boundary_frac из эмпирического окна; b_suffix_map → b_non_boundary.
"""
import atexit, importlib.util, json, os, sys, time
from datetime import datetime
from math import comb
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
ART = os.path.join(RESULTS, "AT3", "AT3.json")
LOG_PATH = os.path.join(RESULTS, "AT3_run.log")
SMOKE = os.environ.get("AT3_SMOKE", "0") == "1"
REVISION = os.environ.get("AT3_REVISION", "r2")
SEED = 0
EPS = 1e-6
B_BOOT = 100 if SMOKE else 1000
THRESH_BASE = 0.02
LAYER = 23
LAYER_SEC = 17
PIN = os.path.join(
    ROOT, "data/calib_pins/wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json")
MODEL_ST = "/data/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors"
P1 = os.path.join(RESULTS, "e64/tpu_v8/P1_own.npz")
P2 = os.path.join(RESULTS, "e64/tpu_v8/P2_corpus.npz")
AT04 = os.path.join(RESULTS, "AT0/AT0_4_MIX.json")

# F1: период-1 repeat-лупы (AT0_4 token_provenance: max_share 0.9297/0.9401,
# min_period_90pct=1). Исключаются из всех научных агрегатов.
DEGENERATE_DOCS = {0, 4}

SHA_FILES = {
    "P1_own.npz": "783e16338cdcb652198722a33ecec8a6e8e98fa3c611c687b17db722b0d58198",
    "P2_corpus.npz": "5f42a08442a5d3bcb8d7c4de5c9658340c0378a917a5e633b9b2aa589ceac125",
    "P1_own_meta.json": "3cebe6f328083c39a7c0130555dde84a62c1ab4c745d48b1f28b5f5eb3538042",
    "P2_corpus_meta.json": "5fb47a32f9402074cdc47284a5e26bc7996572563abfc7b46e2c8eb661581a2e",
    "REPLAY.json": "5580d8f2c8341e9a149d18b20ba16bbe8ad43729dacf3441fd1a7905111e5e8f",
    "pin_plain.json": "083c732ff67c04f8585a5c942eadc9fea7fa2134f78cf869826f11ab3d35ce2f",
    "model.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
    "AT0_4_MIX.json": "d571a4217f32d67c5b45b7921bbd2c3a8116b7449ee4b01f13eb1deab599ecc9",
}
SHA_PATHS = {
    "P1_own.npz": P1, "P2_corpus.npz": P2,
    "P1_own_meta.json": os.path.join(RESULTS, "e64/tpu_v8/P1_own_meta.json"),
    "P2_corpus_meta.json": os.path.join(RESULTS, "e64/tpu_v8/P2_corpus_meta.json"),
    "REPLAY.json": os.path.join(RESULTS, "e81/REPLAY.json"),
    "pin_plain.json": PIN, "model.safetensors": MODEL_ST,
    "AT0_4_MIX.json": AT04,
}
DOCS = list(range(16)) if not SMOKE else [0, 8]
DOCS_EFF = [d for d in DOCS if d not in DEGENERATE_DOCS]


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


def logits_of(H, E, wn, use_w=True):
    """Точный хэд для pre-norm состояний e64: (h/rms)·w·Eᵀ (S4 0.9868,
    fidelity 0.9958 ≈ ТЗ «gate 1.000»). use_w=False — ТЗ-конвенция
    (rms(h)@Eᵀ), верна для post-norm L23 jcache; на этом кэхе — sensitivity."""
    H = H.astype(np.float32)
    r = np.sqrt((H.astype(np.float64) ** 2).mean(axis=1, keepdims=True) + EPS)
    Hn = (H / r.astype(np.float32)).astype(np.float32)
    if use_w:
        Hn = (Hn * wn[None, :]).astype(np.float32)
    return Hn @ E.T


def top1_top2(z):
    top2 = np.sort(z, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0], np.exp(top2[:, 1] - np.logaddexp.reduce(z, axis=1))


def dip_positions(H):
    """Per-position dip(1): Vo[t] = u[t+1] - <u[t+1], u[t]> u[t], ok = ‖Vo‖ > 1e-6,
    dip[t] = cos(Vo[t], Vo[t+1]) где ok[t] & ok[t+1]; иначе NaN.
    (ТЗ-тождествен каноническому V: Vo = ‖H[t+1]‖·Vo_canon, Vn нормирует.)
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


def _deltas_from_bins(per_doc_bin, per_doc_n, min_n):
    out = []
    for d in per_doc_bin:
        if (per_doc_n[d][0] >= min_n and per_doc_n[d][3] >= min_n
                and per_doc_bin[d][0] is not None and per_doc_bin[d][3] is not None):
            out.append(per_doc_bin[d][0] - per_doc_bin[d][3])
    return out


def _p1_block(dips, p1s, docs):
    """Dip по p1-квантилям (F6: p1 — единственная ограниченная сопоставимая ось)."""
    allp = np.concatenate([p1s[d][:dips[d].shape[0]][~np.isnan(dips[d])] for d in docs])
    qp = [float(np.quantile(allp, x)) for x in (0.25, 0.50, 0.75)]
    per = {}
    for d in docs:
        dip = dips[d]
        p = p1s[d][:dip.shape[0]]
        v = ~np.isnan(dip)
        bb = _bin_of(p[v], qp)
        per[d] = {}
        for bi in range(4):
            sel = bb == bi
            per[d][bi] = (float(np.median(dip[v][sel])) if sel.sum() >= 3 else None)
    return {
        "quartiles": qp,
        "per_doc_median_docs": {
            bi: float(np.median([per[d][bi] for d in docs if per[d][bi] is not None]))
            for bi in range(4)},
    }


def side_block(states, margins, p1s, layer_tag, window):
    """Блок стороны+слоя, агрегаты по DOCS_EFF (без вердикта; вердикт — в main
    с threshold_used = max(THRESH_BASE, разброс агрегаций) — F3/F4).
    margin[j] — margin токена, потребляемого переходом state[j]→state[j+1]
    (Vo[j] ↔ margin[j]; scope §2, ТЗ-несмещённое row_t→tok[t]).
    L17: те же margin (финальный readout — L23), другой слой dip; вердиктов нет.
    """
    docs = DOCS_EFF
    dips, margs = {}, {}
    for d in docs:
        dip, _ok = dip_positions(states[d])
        dips[d] = dip
        margs[d] = margins[d][:dip.shape[0]]
    allm = np.concatenate([margs[d][~np.isnan(dips[d])] for d in docs])
    q = [float(np.quantile(allm, x)) for x in (0.25, 0.50, 0.75)]
    per_doc_bin, per_doc_n, sizes, bnd_num, n_valid = {}, {}, [0, 0, 0, 0], 0, 0
    for d in docs:
        dip, m = dips[d], margs[d]
        v = ~np.isnan(dip)
        b = _bin_of(m[v], q)
        n_valid += int(v.sum())
        per_doc_bin[d], per_doc_n[d] = {}, {}
        for bi in range(4):
            sel = b == bi
            sizes[bi] += int(sel.sum())
            per_doc_n[d][bi] = int(sel.sum())
            per_doc_bin[d][bi] = (float(np.median(dip[v][sel]))
                                  if sel.sum() >= 3 else None)
            if layer_tag == "L23" and sel.sum() and window is not None:
                dmin = np.abs(m[v][sel][None, :] - np.array(q)[:, None]).min(axis=1)
                bnd_num += int((dmin < window).sum())
    # первичный Δ (min-n=3, по пре-регу)
    deltas = np.array(_deltas_from_bins(per_doc_bin, per_doc_n, 3), dtype=float)
    if not SMOKE and deltas.size < 4:
        raise SystemExit(f"FAIL: n_docs_delta={deltas.size} < 4 (per-doc Δ не строится)")
    boot = paired_bootstrap(deltas, np.median, B=B_BOOT, seed=SEED)
    dl, dh = boot["ci_low"], boot["ci_high"]
    pooled = {}
    for bi in range(4):
        vals = np.concatenate(
            [dips[d][~np.isnan(dips[d])][_bin_of(margs[d][~np.isnan(dips[d])], q) == bi]
             for d in docs])
        pooled[bi] = float(np.median(vals))
    # F5: min-n sensitivity
    min_n_sens = {}
    for mn in (3, 20, 50):
        dd = _deltas_from_bins(per_doc_bin, per_doc_n, mn)
        min_n_sens[mn] = (float(np.median(dd)) if len(dd) >= 3 else None)
    # F7: ±1 margin-shift (dip[j] ↔ mg[j+1]: условие на решении в точке поворота)
    mshift = {d: margins[d][1:dips[d].shape[0] + 1] for d in docs}
    allms = np.concatenate([mshift[d][~np.isnan(dips[d])] for d in docs])
    qs = [float(np.quantile(allms, x)) for x in (0.25, 0.50, 0.75)]
    pbin_s, pn_s = {}, {}
    for d in docs:
        dip, m = dips[d], mshift[d]
        v = ~np.isnan(dip)
        b = _bin_of(m[v], qs)
        pbin_s[d], pn_s[d] = {}, {}
        for bi in range(4):
            sel = b == bi
            pn_s[d][bi] = int(sel.sum())
            pbin_s[d][bi] = (float(np.median(dip[v][sel])) if sel.sum() >= 3 else None)
    dshift = np.array(_deltas_from_bins(pbin_s, pn_s, 3), dtype=float)
    boot_s = (paired_bootstrap(dshift, np.median, B=B_BOOT, seed=SEED)
              if len(dshift) >= 4 else None)
    out = dict(
        layer=layer_tag,
        docs=docs,
        margin_quartiles=q,
        bin_sizes=sizes,
        n_positions_valid=n_valid,
        n_docs_delta=int(len(deltas)),
        per_doc_bin_median={str(k): v for k, v in per_doc_bin.items()},
        per_doc_bin_n={str(k): v for k, v in per_doc_n.items()},
        delta_per_doc=[float(x) for x in deltas],
        delta_median_docs=float(np.median(deltas)),
        ci95=[float(dl), float(dh)],
        pooled_median=pooled,
        monotonic=[float(np.median([per_doc_bin[d][bi] for d in docs
                                    if per_doc_bin[d][bi] is not None]))
                   for bi in range(4)],
        min_n_sensitivity={str(k): v for k, v in min_n_sens.items()},
        shift_sensitivity={
            "margin_quartiles": qs,
            "delta_median_docs": (float(np.median(dshift)) if len(dshift) else None),
            "ci95": ([float(boot_s["ci_low"]), float(boot_s["ci_high"])]
                     if boot_s is not None else None),
        },
        bin_boundary_frac=(bnd_num / max(1, sum(sizes))) if (layer_tag == "L23"
                                                              and window is not None)
        else None,
        p1=_p1_block(dips, p1s, docs) if p1s is not None else None,
    )
    return out


def make_verdict(delta_md, ci, thresh):
    """Партиция (r2, без щели: (b) = остаток). b_non_boundary — НЕ утверждение
    о суффикс-карте (ревью §9): отказ подтвердить (a) — это всё, что мы знаем."""
    if ci[1] < -thresh:
        return "a_boundary", None
    if ci[0] > thresh:
        return "c_beyond_map", None
    sub = "weak" if abs(delta_md) < thresh else "insufficient_power"
    return "b_non_boundary", sub


def sign_test_twosided(k_neg, n):
    """Двусторонний sign-test: 2·Σ_{i≤min(k,n-k)} C(n,i)/2^n."""
    if n < 4:
        return None
    return 2.0 * sum(comb(n, i) for i in range(0, min(k_neg, n - k_neg) + 1)) / 2.0 ** n


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
            "model": "Qwen/Qwen3.5-0.8B (tied E; точный хэд pre-norm: (h/rms)*w@E.T)",
            "log": os.path.abspath(LOG_PATH),
            "layer": LAYER, "layer_sec": LAYER_SEC,
            "boot_b": B_BOOT, "n_docs": len(DOCS),
            "degenerate_docs": sorted(DEGENERATE_DOCS),
            "threshold_base": THRESH_BASE,
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

    # --- S1: sha ---------------------------------------------------------------
    tstage(tr, time.time() - t0, "gates")
    if not SMOKE:
        for fn, expect in SHA_FILES.items():
            got = sha256_file(SHA_PATHS[fn])
            if got != expect:
                raise SystemExit(f"S1 FAIL: sha {fn} = {got} != {expect}")
        gates["S1_sha"] = True
        logline(f"S1 sha-гейт: {len(SHA_FILES)}/{len(SHA_FILES)} OK "
                f"(+ AT0_4_MIX.json — источник дегенеративного фильтра)")

    tstage(tr, time.time() - t0, "load")
    E, wn = load_e_w()
    z1 = np.load(P1)
    z2 = np.load(P2)
    tf_ids = tf_tokens()
    logline(f"load: E {E.shape}, wn min {wn.min():.3f}, "
            f"tf_ids lens {[len(a) for a in tf_ids][:4]}...")

    # --- states ------------------------------------------------------------------
    gen_states = {d: z1[f"own_w{d}_L{LAYER}"].astype(np.float32) for d in DOCS}
    gen_states17 = {d: z1[f"own_w{d}_L{LAYER_SEC}"].astype(np.float32) for d in DOCS}
    tf_states = {d: z2[f"cor_w{d}_L{LAYER}"].astype(np.float32) for d in DOCS}
    tf_states17 = {d: z2[f"cor_w{d}_L{LAYER_SEC}"].astype(np.float32) for d in DOCS}
    gen_mg = {d: z1[f"own_w{d}_mg"].astype(np.float64) for d in DOCS}
    gen_p1 = {d: z1[f"own_w{d}_p1"].astype(np.float64) for d in DOCS}
    gen_tok = {d: z1[f"own_w{d}_tok"].astype(np.int64) for d in DOCS}

    # --- S2: dip-якоря (канон, L23, 16 доков — воспроизведение ТЗ, не наука) ------
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

    # --- TF: margin/p1 (оба хэда), gate --------------------------------------------
    tstage(tr, time.time() - t0, "tf_margin")
    tf_mg, tf_mg_nw, tf_p1, tf_gate, tf_gate_nw = {}, {}, {}, {}, {}
    for d in DOCS:
        n = tf_states[d].shape[0]
        zw = logits_of(tf_states[d], E, wn, use_w=True)
        zn = logits_of(tf_states[d], E, wn, use_w=False)
        tf_mg[d], tf_p1[d] = top1_top2(zw)
        tf_mg_nw[d] = top1_top2(zn)[0]
        # TЗ-несмещённое row_t→tok[t] в (state, предсказанный токен) индексе:
        # state[j] предсказывает ids[j+1] (проверено: несмещённое vs ids[j] = 0.003 ≈ chance)
        tf_gate[d] = float((zw.argmax(axis=1) == tf_ids[d][1:n + 1]).mean())
        tf_gate_nw[d] = float((zn.argmax(axis=1) == tf_ids[d][1:n + 1]).mean())
    if not SMOKE:
        bad = [d for d in DOCS if tf_gate[d] < 0.25]
        if bad:
            raise SystemExit(
                f"S3 FAIL: TF next-token gate < 0.25 на докум {bad}: "
                f"{[round(tf_gate[d], 3) for d in bad]}")
        gates["S3_tf_gate"] = True
        logline(f"S3 TF-next-token-gate: OK (w: min {min(tf_gate.values()):.4f}, "
                f"max {max(tf_gate.values()):.4f})")

    # --- S4: alignment + S3b: head fidelity (gen) -----------------------------------
    tstage(tr, time.time() - t0, "gen_s4")
    s4, s4_nw, fid, fid_nw, flip = {}, {}, {}, {}, {}
    mgw_all, mgn_all, mg_diff = [], [], []
    for d in DOCS:
        n = gen_states[d].shape[0]
        zw = logits_of(gen_states[d], E, wn, use_w=True)
        zn = logits_of(gen_states[d], E, wn, use_w=False)
        mgw, _ = top1_top2(zw)
        mgn, _ = top1_top2(zn)
        a = n - 2
        s4[d] = spearman(mgw[:a], gen_mg[d][:a])
        s4_nw[d] = spearman(mgn[:a], gen_mg[d][:a])
        fid[d] = float((zw.argmax(axis=1)[:a] == gen_tok[d][:a]).mean())
        fid_nw[d] = float((zn.argmax(axis=1)[:a] == gen_tok[d][:a]).mean())
        flip[d] = float((zw.argmax(axis=1)[:a] != zn.argmax(axis=1)[:a]).mean())
        mgw_all.append(mgw[:a])
        mgn_all.append(mgn[:a])
        mg_diff.append(np.abs(mgw[:a] - gen_mg[d][:a]))  # остаток S4: эмпирическая ошибка margin (gen)
        if not SMOKE:
            if s4[d] is None or s4[d] < 0.95:
                raise SystemExit(f"S4 FAIL: doc w{d} spearman {s4[d]} < 0.95")
            if fid[d] < 0.98:
                raise SystemExit(
                    f"S3b FAIL: head fidelity doc w{d} = {fid[d]:.4f} < 0.98 "
                    f"(ТЗ «gate 1.000» не воспроизведён)")
    if not SMOKE:
        gates["S4_alignment"] = True
        gates["S3b_head_fidelity"] = True
        logline(f"S4 alignment: OK (min {min(s4.values()):.4f})")
        logline(f"S3b head-fidelity: OK (min {min(fid.values()):.4f}, "
                f"mean {np.mean(list(fid.values())):.4f}) — ТЗ «gate 1.000» на этом кэхе")
    # окна bin_boundary_frac (мелкое): gen — п95 остатка S4 (эмпирическая ошибка margin);
    # tf — п95 |margin_w − margin_nw| (верхняя граница: конвенция + fp16-шум)
    win_gen = float(np.quantile(np.concatenate(mg_diff), 0.95))
    win_tf = float(np.quantile(
        np.concatenate([np.abs(tf_mg[d] - tf_mg_nw[d]) for d in DOCS]), 0.95))
    mgw_cat = np.concatenate(mgw_all)
    mgn_cat = np.concatenate(mgn_all)

    # --- L23: обе стороны -------------------------------------------------------------
    tstage(tr, time.time() - t0, "dip_l23")
    gen_block = side_block(gen_states, gen_mg, gen_p1, "L23", window=win_gen)
    tf_block = side_block(tf_states, tf_mg, tf_p1, "L23", window=win_tf)
    # F4: порог = max(базовый, разброс агрегаций) — до вердикта, из данных блока
    spread_g = max(abs(a - b) for a, b in zip(gen_block["pooled_median"].values(), gen_block["monotonic"]))
    spread_t = max(abs(a - b) for a, b in zip(tf_block["pooled_median"].values(), tf_block["monotonic"]))
    thresh_g = max(THRESH_BASE, spread_g)
    thresh_t = max(THRESH_BASE, spread_t)
    gen_block["threshold_used"], gen_block["agg_spread"] = thresh_g, spread_g
    tf_block["threshold_used"], tf_block["agg_spread"] = thresh_t, spread_t
    vg, vg_sub = make_verdict(gen_block["delta_median_docs"], gen_block["ci95"], thresh_g)
    vt, vt_sub = make_verdict(tf_block["delta_median_docs"], tf_block["ci95"], thresh_t)
    gen_block["verdict"], gen_block["verdict_sub"] = vg, vg_sub
    tf_block["verdict"], tf_block["verdict_sub"] = vt, vt_sub
    logline(f"L23 gen: Δ {gen_block['delta_median_docs']:+.4f} CI {gen_block['ci95']} "
            f"(T={thresh_g:.4f}) verdict={vg}/{vg_sub}")
    logline(f"L23 tf : Δ {tf_block['delta_median_docs']:+.4f} CI {tf_block['ci95']} "
            f"(T={thresh_t:.4f}) verdict={vt}/{vt_sub}")

    # --- S5: bin-size ------------------------------------------------------------------
    tstage(tr, time.time() - t0, "bin_size")
    if not SMOKE:
        for nm, blk in (("gen", gen_block), ("tf", tf_block)):
            small = [b for b, s in enumerate(blk["bin_sizes"]) if s < 50]
            if small:
                raise SystemExit(f"S5 FAIL: {nm} L23 bin-size < 50: {blk['bin_sizes']}")
        gates["S5_bin_size"] = True
        logline(f"S5 bin-size: OK (gen {gen_block['bin_sizes']}, "
                f"tf {tf_block['bin_sizes']})")

    # --- F1: отчёт о дегенеративном фильтре ---------------------------------------------
    tstage(tr, time.time() - t0, "deg_filter")
    n_dropped = {}
    for side, st in (("gen", gen_states), ("tf", tf_states)):
        nd = 0
        for d in sorted(DEGENERATE_DOCS):
            if d in st:
                dip, _ = dip_positions(st[d])
                nd += int((~np.isnan(dip)).sum())
        n_dropped[side] = nd
    logline(f"F1 deg-filter: docs {sorted(DEGENERATE_DOCS)} исключены из агрегатов; "
            f"отброшено позиций (L23): gen {n_dropped['gen']}, tf {n_dropped['tf']}")

    tstage(tr, time.time() - t0, "dip_l17")
    gen_block17 = side_block(gen_states17, gen_mg, gen_p1, "L17", window=None)
    tf_block17 = side_block(tf_states17, tf_mg, tf_p1, "L17", window=None)

    # --- F3: sign-test (gen L23) ----------------------------------------------------------
    tstage(tr, time.time() - t0, "sign_test")
    k_neg = sum(1 for x in gen_block["delta_per_doc"] if x < 0)
    st_p = sign_test_twosided(k_neg, len(gen_block["delta_per_doc"]))

    # --- итог ------------------------------------------------------------------------------
    tstage(tr, time.time() - t0, "finalize")
    result = {
        "run": f"at3_margin_{REVISION}",
        "smoke": SMOKE,
        "seed": SEED,
        "threshold_base": THRESH_BASE,
        "degenerate_docs": sorted(DEGENERATE_DOCS),
        "n_positions_dropped_L23": n_dropped,
        "gates": gates,
        "anchors": anchors,
        "tf_gate": {f"w{d}": round(v, 4) for d, v in tf_gate.items()},
        "tf_gate_nw": {f"w{d}": round(v, 4) for d, v in tf_gate_nw.items()},
        "s4_spearman": {f"w{d}": v for d, v in s4.items()},
        "head_sensitivity": {
            "convention": "exact(pre-norm)=(h/rms)*w@E.T; nw=ТЗ-post-norm sensitivity",
            "s4_w_min": min(v for v in s4.values() if v is not None),
            "s4_nw_min": min(v for v in s4_nw.values() if v is not None),
            "s4_nw": {f"w{d}": v for d, v in s4_nw.items()},
            "fidelity_w_min": min(fid.values()),
            "fidelity_w_mean": float(np.mean(list(fid.values()))),
            "fidelity_nw_min": min(fid_nw.values()),
            "fidelity_nw_mean": float(np.mean(list(fid_nw.values()))),
            "argmax_flip_w_nw_mean": float(np.mean(list(flip.values()))),
            "tf_gate_nw_min": min(tf_gate_nw.values()),
            "tf_gate_nw_max": max(tf_gate_nw.values()),
            "margin_median_w": float(np.median(mgw_cat)),
            "margin_median_nw": float(np.median(mgn_cat)),
            "margin_diff_w_saved_gen": {"median": float(np.median(np.concatenate(mg_diff))),
                                        "p95": win_gen},
        },
        "boundary_window": {"gen": win_gen, "tf": win_tf},
        "L23_gen": gen_block,
        "L23_tf": tf_block,
        "L17_gen": gen_block17,
        "L17_tf": tf_block17,
        "sign_test_gen_L23": {
            "k_neg": k_neg, "n": len(gen_block["delta_per_doc"]),
            "p_twosided": (float(st_p) if st_p is not None else None)},
        "verdict_tf": vt,
        "verdict_gen": vg,
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
            f"verdicts: tf={vt}/{vt_sub} gen={vg}/{vg_sub}")
    _log.write_to(LOG_PATH)
    if tr is not None:
        tlog(tr, time.time() - t0, {
            "finished_at": datetime.now().isoformat(), "status": "ok",
            "verdict_tf": vt, "verdict_gen": vg,
            "delta_tf": tf_block["delta_median_docs"],
            "delta_gen": gen_block["delta_median_docs"],
            "threshold_used_gen": thresh_g, "threshold_used_tf": thresh_t})
        tr.finish()
        tr_done[0] = True


if __name__ == "__main__":
    if "--recheck" in sys.argv:
        # S6: вердикты восстанавливаемы из полей артефакта (с per-side threshold)
        r = json.load(open(sys.argv[sys.argv.index("--recheck") + 1]))
        for side in ("L23_tf", "L23_gen"):
            blk = r[side]
            got, _sub = make_verdict(blk["delta_median_docs"], blk["ci95"],
                                     blk["threshold_used"])
            if got != blk["verdict"]:
                raise SystemExit(f"S6 FAIL: {side} verdict {got} != {blk['verdict']}")
        print("S6 recheck OK:", r["verdict_tf"], r["verdict_gen"])
    else:
        main()
