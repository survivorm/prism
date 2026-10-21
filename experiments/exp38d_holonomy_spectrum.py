#!/usr/bin/env python
"""E-38d — Full holonomy matrix H_012 and spectrum {θ_j} (docs/E38.md).

Wave 2 (post E-38c; c reported near_degeneracy on L18/L22 → {θ_j} is
content-bearing, per the pre-registered gate).

For each triple (t, t+1, t+2), each doc, each layer:
    V_i = top_r_svd(J_i, 16)[2].T      [D, 16] orthonormal-column basis
    (R_ij, res_ij) = procrustes_align(V_i, V_j)   per-step residual_sq
    (theta, H_012) = holonomy_spectrum([V_0, V_1, V_2], closed_loop=True)
        theta: [r//2] = 8 angles (rad); H_012 = R_total [16, 16] float64
    kappa = ‖H_012 − I‖_F
    tr(H) = 2 Σ cos θ_j ; ‖H−I‖_F² = 4 Σ (1 − cos θ_j)  (exact identities)
    Ω(t) = geodesic_curvature(û_t, û_{t+1}, û_{t+2})

J = RAW J from the exp27b cache (same convention as E-28, which gave
κ = 0.018 on L23 — consistency check, tol 1e-3).

Per layer: {θ_j} distribution over docs, ρ(tr(H), Ω) by position (Spearman),
‖H−I‖_F vs E-28 κ(L23) = 0.018.

Degenerate step: a Procrustes step with residual_sq near 2r (≈ 32 for
r=16; subspaces nearly orthogonal, σ_min(V_i^T V_j) ≈ 0) makes R ill-
conditioned. Triples with any such step are EXCLUDED from the ρ
correlation and recorded (per-TZ).

Trivial H_012 ≈ I → holonomy_spectrum returns theta=[0,...,0] with a
RuntimeWarning: a VALID Abelian result, not a bug (recorded as trivial).

Deterministic: seed=0; Spearman permutation p uses seed=0.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _e38_cache import LAYERS, S, D, load_layer  # noqa: E402
from prism.spectral import top_r_svd  # noqa: E402
from prism.holonomy import (holonomy_spectrum, geodesic_curvature,
                            procrustes_align)  # noqa: E402

SEED = 0
R = 16
KAPPA_L23_E28 = 0.018
# residual_sq near 2r = 32 means subspaces are (nearly) orthogonal. Use a
# conservative threshold well below 2r: 0.9 * 2r flags "near-orthogonal".
DEGEN_RES_FRAC = 0.9 * 2 * R   # = 28.8

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38d_holonomy_spectrum.json")


def main():
    t0 = time.time()
    np.random.seed(SEED)
    per_layer = {}
    for l in LAYERS:
        ck = load_layer(l)
        J, H = ck["J"], ck["H"]           # [N, S, K, D], [N, S, D]
        N, Ss, Kk, Dd = J.shape
        del ck
        norms = H.norm(dim=-1)
        u = H / norms.clamp_min(1e-12)[:, :, None]
        n_triples = Ss - 2
        theta_all = np.zeros((n_triples, N, R // 2))
        kappa_all = np.zeros((n_triples, N))
        trH_all = np.zeros((n_triples, N))
        omega_all = np.zeros((n_triples, N))
        degenerate = np.zeros((n_triples, N), dtype=bool)
        trivial = np.zeros((n_triples, N), dtype=bool)
        for t in range(n_triples):
            for i in range(N):
                V0 = top_r_svd(J[i, t], R)[2].T.contiguous()
                V1 = top_r_svd(J[i, t + 1], R)[2].T.contiguous()
                V2 = top_r_svd(J[i, t + 2], R)[2].T.contiguous()
                res01 = procrustes_align(V0, V1)[1]
                res12 = procrustes_align(V1, V2)[1]
                res20 = procrustes_align(V2, V0)[1]
                step_res = max(res01, res12, res20)
                with warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")
                    th, Htot = holonomy_spectrum([V0, V1, V2],
                                                 closed_loop=True)
                Hn = Htot.numpy()
                kappa_all[t, i] = float(np.linalg.norm(Hn - np.eye(R)))
                # trace from the MATRIX (always exact): a mixed spectrum
                # returns < r//2 angles (trivial +1 pairs dropped per the
                # holonomy_spectrum contract), so 2*sum(cos(th)) would
                # undercount the trace there.
                trH_all[t, i] = float(np.trace(Hn))
                # theta storage: zero-pad for length (does not affect
                # theta_max / degenerate detection)
                theta_all[t, i, :len(th)] = th
                omega_all[t, i] = geodesic_curvature(u[i, t], u[i, t + 1],
                                                     u[i, t + 2])
                if step_res > DEGEN_RES_FRAC:
                    degenerate[t, i] = True
                if any("R_total" in str(w.message) and "0" in str(w.message)
                       for w in wlist) or float(th.max()) < 1e-9:
                    trivial[t, i] = True
        from scipy.stats import spearmanr
        rho = []
        for t in range(0, n_triples, 4):
            m = ~degenerate[t]
            if m.sum() >= 8:
                r_, _ = spearmanr(trH_all[t, m], omega_all[t, m])
                if np.isfinite(r_):
                    rho.append(float(r_))
        rho_med = float(np.median(rho)) if rho else float("nan")
        theta_flat = theta_all[~degenerate]
        spread = float(np.nanmax(theta_flat) - np.nanmin(theta_flat))
        # exact identity check on a sample of non-degenerate triples
        sample = np.argwhere(~degenerate)
        idx = sample[:: max(1, len(sample) // 200)][:200]
        idiff = max(abs(kappa_all[a, b] ** 2 - 4.0 * (1.0 - np.cos(
            theta_all[a, b])).sum()) for a, b in idx) if len(idx) else 0.0
        per_layer[str(l)] = {
            "kappa_median": float(np.median(kappa_all)),
            "kappa_ci": list(np.percentile(kappa_all, [2.5, 97.5])),
            "theta": {
                "median_per_j": [float(np.median(theta_all[:, :, j]))
                                 for j in range(R // 2)],
                "hist_8_angles": [float(c) for c in np.histogram(theta_flat,
                                     bins=[0, 0.01, 0.05, 0.1, 0.5,
                                           1.0, np.pi])[0]],
                "spread_max_minus_min": spread},
            "trH_vs_omega": {"rho_median": rho_med, "n_positions": len(rho)},
            "identity_check_max_absdiff": float(idiff),
            "degenerate_frac": float(degenerate.mean()),
            "trivial_frac": float(trivial.mean()),
            "profile_kappa": [float(np.median(kappa_all[t]))
                              for t in range(0, n_triples, 8)],
        }
        r = per_layer[str(l)]
        del J, H
        if l == 23:
            r["consistency_e28"] = {
                "kappa_L23_this": r["kappa_median"],
                "kappa_L23_E28": KAPPA_L23_E28,
                "abs_diff": abs(r["kappa_median"] - KAPPA_L23_E28),
                "within_tol_1e-3": abs(r["kappa_median"] - KAPPA_L23_E28) < 1e-3}
        print(f"L{l}: κ_med={r['kappa_median']:.5f} "
              f"θ_med={float(np.median(theta_all)):.5f} spread={spread:.4f} "
              f"ρ(trH,Ω)={rho_med:.3f} degen={r['degenerate_frac']:.3f} "
              f"trivial={r['trivial_frac']:.3f}")

    verdicts = {}
    for l, r in per_layer.items():
        v = []
        if l == "23":
            v.append("consistency_pass" if r["consistency_e28"]["within_tol_1e-3"]
                     else "consistency_FAIL")
        sp = r["theta"]["spread_max_minus_min"]
        v.append("effectively_abelian" if sp < 0.1
                 else ("non_abelian_candidate" if sp > 0.5 else "mixed_spectrum"))
        rho_m = r["trH_vs_omega"]["rho_median"]
        if np.isfinite(rho_m):
            v.append("spherical_geometry" if rho_m > 0.7
                     else ("third_mechanism" if rho_m < 0.3 else "partial"))
        verdicts[l] = ",".join(v)
    out = {
        "params": {"seed": SEED, "r": R, "layers": list(LAYERS), "N_docs": 32,
                   "S": S, "J": "raw (E-28 convention)",
                   "degen_res_threshold": DEGEN_RES_FRAC,
                   "cache": "exp27b_jcache_L{l}.pt"},
        "metrics": per_layer,
        "outcomes": {"per_layer": verdicts},
        "meta": {"note": "degenerate = any Procrustes step residual_sq > "
                         "0.9*2r (near-orthogonal subspaces); excluded from "
                         "rho. trivial (theta≈0 + warning) is a valid Abelian "
                         "result, not a bug. tr(H) and ‖H−I‖_F² are exact "
                         "functions of {θ_j} (identity_check_max_absdiff)."},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"VERDICTS {verdicts}")
    print(f"JSON -> {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
