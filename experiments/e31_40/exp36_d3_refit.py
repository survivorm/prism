#!/usr/bin/env python3
"""E-36 D3 refit: p̄_med = НОРМИРОВАННАЯ медиана (Σp=1).

Причина (ревью E-36, 2026-07-07): P(p) = diag(p) − ppᵀ — якобиан softmax
определено только для вероятностного p. Ненормированная медиана по строкам
симплекса имеет Σ ≈ 0.006, и P(c·p) = c·diag(p) − c²·ppᵀ ≈ c·diag(p) —
то есть E-36 D3 (и E-33r.2 A) мерили НЕ gauge-оператор, а крошечный
диагональный скалинг. Пересчёт D3 с нормированным p̄_med.

D1/D2/D5/D4 не затрагиваются. Aгрегаты (summary, verdict, MD) перегенерируются.
Вход: results/exp36_gauge_anatomy.json (per_doc из полного прогона).
Выход: тот же файл (per_doc[*][l]["d3"], summary, meta, verdict) + MD.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch

torch.set_num_threads(1)   # не забирать все ядра (другие refit'ы/задачи)

import exp33r_gauge_anatomy as E33R
from exp36_gauge_anatomy import (
    CKPT_DIR, DELTA_GRID, LAYERS, _clean, _fmt, _np2py, dsvd_right,
    p_apply, verdict,
)

R = 16
DT = 1
JSON = "results/exp36_gauge_anatomy.json"
CKPT = "results/exp36_d3_refit.ckpt.json"


def doc_d3_norm(Jdoc, Pdoc, sent, r, dt=1):
    T = Jdoc.shape[0]
    sent_t = torch.as_tensor(sent, dtype=torch.bool)
    alive = ~sent_t
    pbar = Pdoc[alive].median(dim=0).values if alive.any() else Pdoc.mean(dim=0)
    pbar = pbar / pbar.sum().clamp_min(1e-30)
    Ppbar = p_apply(Jdoc, pbar.unsqueeze(0).expand(T, -1))
    Ppt = p_apply(Jdoc, Pdoc)
    DPt = Ppt - Ppbar
    out = {}
    mask = (~(sent_t[:-dt] | sent_t[dt:])).numpy()
    for name, X in (("fp", Ppbar), ("dyn", Ppt), ("dp", DPt)):
        a, fa, fb = dsvd_right(X[:-dt], X[dt:], r)
        out[f"d3_{name}"] = np.where(mask & ~(fa | fb).numpy(), a.numpy(), np.nan)
    return out


def main():
    no_md = "--no-md" in sys.argv
    t0 = time.time()
    res = json.load(open(JSON))
    docs = res["meta"]["n_docs"]
    cks = {l: E33R.load_layer(CKPT_DIR, l) for l in LAYERS}
    ck0 = cks[LAYERS[0]]
    domains = sorted({k[0] for k in ck0["JR"]})
    doclist = [(d, s) for d in domains for s in range(ck0["n_samples"])]
    assert len(doclist) == docs, f"len(doclist)={len(doclist)} vs meta.n_docs={docs}"

    # resume
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        done = {int(k): v for k, v in ck.items()}
        print(f"resume: {len(done)}/{docs} docs already refit")
    else:
        done = {}

    for i, (d, s) in enumerate(doclist):
        if i in done:
            continue
        row = {}
        for l in LAYERS:
            Jdoc, Pdoc, sent = E33R.load_doc(cks[l], d, s)
            row[str(l)] = doc_d3_norm(Jdoc, Pdoc, sent, R, DT)
            del Jdoc, Pdoc, sent
        done[i] = row
        json.dump(_clean({str(k): v for k, v in done.items()}),
                  open(CKPT, "w"), ensure_ascii=False)
        if (i + 1) % 4 == 0:
            print(f"  doc {i+1}/{docs} t={time.time()-t0:.0f}s", flush=True)

    # --- update JSON (ckpt хранит NaN как None через _clean) ---
    for i in range(docs):
        for l in LAYERS:
            res["per_doc"][str(i)][str(l)]["d3"] = {
                k: [None if v is None else v for v in done[i][str(l)][k]]
                for k in done[i][str(l)]}

    # summary d3
    for l in LAYERS:
        d3s = {}
        for v in ("fp", "dyn", "dp"):
            key = "d3_" + v
            arr_raw = np.concatenate(
                [np.asarray(res["per_doc"][str(i)][str(l)]["d3"][key],
                             dtype=float) for i in range(docs)])
            n_total = arr_raw.size
            arr = arr_raw[np.isfinite(arr_raw)]
            d3s[f"d3_{v}"] = {
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "n": int(arr.size),
                "frac_nan": 1.0 - arr.size / max(1, n_total),
            }
        res["summary"][str(l)]["d3"] = d3s

    # meta
    res["meta"]["d3_refit"] = {
        "note": "D3 пересчитан с нормированным p̄_med (Σp=1); p̄ из исходного "
                "прогона — ненормированная медиана (Σ≈0.006), P ≈ diag — не "
                "gauge-оператор. D1/D2/D5/D4 неизменны.",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "refit_timing_sec": round(time.time() - t0, 1),
    }
    for i, dv in enumerate(res["meta"]["deviation"]):
        if dv.startswith("D3:"):
            res["meta"]["deviation"][i] = (
                "D3: p̄ = медиана по non-sentinel позициям, НОРМИРОВАННАЯ (Σp=1) — "
                "P(p) — якобиан softmax только для вероятностного вектора; "
                "ненормированная медиана (E-33r.2 A) давала P ≈ diag, не "
                "gauge-оператор. Пересчитано в exp36_d3_refit (см. meta.d3_refit)")

    res["meta"]["timing_sec"] = res["meta"]["timing_sec"] + time.time() - t0
    res["verdict"] = verdict(res)

    with open(JSON, "w") as f:
        json.dump(_clean(res), f, indent=2, ensure_ascii=False)
    os.remove(CKPT)
    print(f"done in {time.time()-t0:.0f}s; JSON updated")

    # --- MD: перегенерировать только D3-секции ---
    if no_md:
        print("MD skipped (--no-md)")
        return
    md = open("results/exp36_gauge_anatomy.md").read().split("\n")
    out = []
    skip = False
    for line in md:
        if line.startswith("## D3"):
            skip = True
            continue
        if line.startswith("## D4"):
            skip = False
        if not skip:
            out.append(line)
    md = out
    # вставить новую D3 перед D4
    d3 = ["## D3: fixed-P vs dynamic (Δ=1, медианы, rad)",
          "| layer | P(p̄)·J | P(p_t)·J | ΔP_t·J |",
          "|---|---|---|---|"]
    for l in LAYERS:
        d = res["summary"][str(l)]["d3"]
        d3.append(f"| {l} | {_fmt(d['d3_fp'].get('median'))} "
                  f"| {_fmt(d['d3_dyn'].get('median'))} "
                  f"| {_fmt(d['d3_dp'].get('median'))} |")
    d3.append("")
    d3.append("(p̄_med — нормированная медиана, Σp=1 — refit 2026-07-07; "
              "исходный прогон использовал ненормированную медиану, P ≈ diag)")
    d3.append("")
    idx = next(i for i, x in enumerate(md) if x.startswith("## D4"))
    md[idx:idx] = d3
    open("results/exp36_gauge_anatomy.md", "w").write("\n".join(md))
    print("MD updated")


if __name__ == "__main__":
    main()
