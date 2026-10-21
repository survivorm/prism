#!/usr/bin/env python
"""E-38c — Degeneracy pre-flight for holonomy (eigengap_min in top-16).

Post-process exp27b dense cache (0 cost). The non-Abelian formalism
(H_012, {theta_j}) requires (near-)degenerate eigen-subspaces. This
pre-flight checks whether the top-16 singular spectrum of J_t has a
sufficiently small minimum consecutive eigenvalue gap.

Per (layer, t, doc):
    sv(J_t) -> sigma_1 >= ... >= sigma_16
    spectrum = sigma^2 (eigenvalues of J J^T)
    (gap_min, applicable) = eigengap_min(spectrum, eps_frac)
    threshold = eps_frac * sigma_1^2, applicable = gap_min >= threshold

eps_frac = 0.01 (convention; sensitivity sweep {0.005, 0.01, 0.02}).
L23 (lm_head) is trivial by construction — preflight is only informative
for internal layers L18/L22; all 6 layers are computed and recorded.
Layers are NOT averaged (per-TZ).

Per layer: fraction_applicable = median_t (mean_doc applicable);
          min_t_gap_min = min_t (median_doc gap_min).

Pre-registered outcomes:
    non_abelian_not_applicable — min_t_gap_min >= threshold on ALL layers
    near_degeneracy            — min_t_gap_min < threshold on L18 or L22
    l23_architectural          — L23 applicable, L18/L22 not

Deterministic: no randomness.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _e38_cache import LAYERS, S, load_layer  # noqa: E402
from prism.spectral import eigengap_min  # noqa: E402

EPS_MAIN = 0.01
EPS_SWEEP = (0.005, 0.01, 0.02)
R = 16

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38c_degeneracy_preflight.json")


def main():
    t0 = time.time()
    per_layer = {}
    for l in LAYERS:
        ck = load_layer(l)
        J = ck["J"]                     # [N, S, K, D]
        N, Ss, Kk, Dd = J.shape
        del ck
        # accumulate per (t, doc): gap_min (main eps), applicable (per sweep eps)
        gap_min_td = np.zeros((Ss, N))          # main eps
        appl_td = {e: np.zeros((Ss, N), dtype=bool) for e in EPS_SWEEP}
        s1sq = np.zeros((Ss, N))                # sigma_1^2
        for t in range(Ss):
            sv = torch.linalg.svdvals(J[:, t])  # [N, K] descending
            sv16 = sv[:, :R]                    # [N, R]
            lam = (sv16 ** 2)                   # [N, R] eigenvalues
            for i in range(N):
                s1sq[t, i] = float(lam[i, 0])
                for e in EPS_SWEEP:
                    g, a = eigengap_min(lam[i], e)
                    appl_td[e][t, i] = a
                    if e == EPS_MAIN:
                        gap_min_td[t, i] = g
        # per-t medians over docs, then per-layer aggregates
        gap_t_med = np.median(gap_min_td, axis=1)
        ref_threshold = EPS_MAIN * float(np.median(s1sq))
        per_layer[str(l)] = {
            "fraction_applicable": {
                e: float(np.median(appl_td[e].mean(axis=1))) for e in EPS_SWEEP},
            "min_t_gap_min": float(gap_t_med.min()),
            "ref_threshold": ref_threshold,
            "min_t_gap_min_over_threshold": float(gap_t_med.min() / ref_threshold),
            "argmin_t": int(gap_t_med.argmin()),
            "s1sq_median": float(np.median(s1sq)),
            "profile_gap_min_t": [float(x) for x in gap_t_med[::4]],
            "informative": bool(l in (18, 22)),
        }
        r = per_layer[str(l)]
        del J
        print(f"L{l}: fraction_applicable@0.01={r['fraction_applicable'][0.01]:.3f} "
              f"(sweep {[r['fraction_applicable'][e] for e in EPS_SWEEP]}) | "
              f"min_t_gap_min={r['min_t_gap_min']:.4f} thresh={ref_threshold:.4f} "
              f"ratio={r['min_t_gap_min_over_threshold']:.3f} informative={r['informative']}")

    # verdicts (only informative layers drive the aggregate; L23 trivial)
    inf = {l: r for l, r in per_layer.items() if r["informative"]}
    verdicts = {}
    all_not_applicable = all(r["min_t_gap_min_over_threshold"] >= 1.0 for r in per_layer.values())
    near_degen = [l for l, r in inf.items() if r["min_t_gap_min_over_threshold"] < 1.0]
    l23_appl = per_layer["23"]["min_t_gap_min_over_threshold"] >= 1.0
    l1822_not = all(per_layer[l]["min_t_gap_min_over_threshold"] < 1.0 for l in ("18", "22"))
    if all_not_applicable:
        verdicts["aggregate"] = "non_abelian_not_applicable"
    elif near_degen:
        verdicts["aggregate"] = "near_degeneracy"
        verdicts["near_degeneracy_layers"] = near_degen
    if l23_appl and l1822_not:
        verdicts["l23_architectural"] = True
    out = {
        "params": {"eps_frac": EPS_MAIN, "eps_sweep": list(EPS_SWEEP),
                   "r": R, "layers": list(LAYERS), "N_docs": 32, "S": S,
                   "cache": "exp27b_jcache_L{l}.pt"},
        "metrics": per_layer,
        "outcomes": verdicts,
        "meta": {"note": "L23 (lm_head) trivial by construction — preflight "
                         "informative only for L18/L22 (internal). Layers not "
                         "averaged (per-TZ). applicable = gap_min >= eps_frac*sigma_1^2."},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"OUTCOMES {verdicts}")
    print(f"JSON -> {OUT} ({os.path.getsize(OUT)} bytes) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
