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
κ = 0.018 on L23 — consistency check; PRE-REGISTERED tolerance 5e-3,
1e-3 is a tighter DIAGNOSTIC only — the header previously said 1e-3,
contradicting the code; docs/E38_fixes2.md P1-4).

Per layer: {θ_j} distribution over docs, ρ(tr(H), Ω) by position (Spearman),
‖H−I‖_F vs E-28 κ(L23) = 0.018.

Pre-registered outcomes (per layer; docs/E38_fixes2.md P1-5 — the old
docstring did not list them, so the post-hoc ones are now marked):
    consistency_pass / consistency_redflag (L23 only, tol 5e-3)
    effectively_abelian    — max_theta < 0.1  (PRE-REGISTERED; the old
                             spread-based rule mislabeled rigid rotations
                             with identical angles as abelian —
                             docs/E38_fixes2.md P0-1)
    mixed_spectrum         — 0.1 ≤ max_theta ≤ 0.5 (PRE-REGISTERED)
    non_abelian_candidate  — max_theta > 0.5 (POST-HOC extension, no
                             threshold formula in the TZ — recorded as such)
    spherical_geometry     — ρ(trH, Ω) > 0.7 (POST-HOC extension, same)
    third_mechanism        — ρ < 0.3 (POST-HOC extension, same; ≡ NOT
                             spherical, meta.label_semantics)
    partial                — 0.3 ≤ ρ ≤ 0.7 (POST-HOC extension, same)

Degenerate step: a Procrustes step with residual_sq near 2r (≈ 32 for
r=16; subspaces nearly orthogonal, σ_min(V_i^T V_j) ≈ 0) makes R ill-
conditioned. Triples with any such step are EXCLUDED from the ρ
correlation and recorded (per-TZ).

Trivial H_012 ≈ I → holonomy_spectrum returns theta=[0,...,0] with a
RuntimeWarning: a VALID Abelian result, not a bug (recorded as trivial).

Deterministic: seed=0. spearmanr's p is an asymptotic t-approximation
(NOT a permutation test; seed is irrelevant for it) — the median p is
reported as p_med for diagnostics (docs/E38_fixes.md part-5 §3, M-3).
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
        theta_all = np.full((n_triples, N, R // 2), np.nan)  # NaN padding:
        kappa_all = np.zeros((n_triples, N))
        trH_all = np.zeros((n_triples, N))
        omega_all = np.zeros((n_triples, N))
        step_res_all = np.zeros((n_triples, N))
        degenerate = np.zeros((n_triples, N), dtype=bool)
        trivial = np.zeros((n_triples, N), dtype=bool)
        idiff_max = 0.0
        n_anom = 0            # P0-2: non-trivial holonomy_spectrum warnings
        # P1-7 (docs/E38_fixes2.md): the old counter lumped several warning
        # classes. "lone -1 eigenvalue (reflection axis)" is LEGITIMATE
        # without a det-fix in Procrustes; "degenerate spectrum" is
        # structural too. Count them separately so the JSON can say which
        # structural warnings actually fired (expected 0 on real data:
        # degenerate 16x16 spectra are measure-zero; reflections possible
        # for r=3 frames of near-symmetric Jacobians).
        n_anom_reflection = 0
        n_anom_degenerate = 0
        n_omega_skipped = 0   # P1-5: degenerate h (‖h‖ < 1e-10) triples
        # full H_012 for L23, doc 0 (docs/E38_fixes.md §5.4): kept for
        # future κ_fiber recomputation in the eigenframe; float64, ~126x16x16x8B.
        H012_save = [] if l == 23 else None
        for t in range(n_triples):
            for i in range(N):
                V0 = top_r_svd(J[i, t], R)[2].T.contiguous()
                V1 = top_r_svd(J[i, t + 1], R)[2].T.contiguous()
                V2 = top_r_svd(J[i, t + 2], R)[2].T.contiguous()
                res01 = procrustes_align(V0, V1)[1]
                res12 = procrustes_align(V1, V2)[1]
                res20 = procrustes_align(V2, V0)[1]
                step_res = max(res01, res12, res20)
                step_res_all[t, i] = step_res
                with warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")
                    th, Htot = holonomy_spectrum([V0, V1, V2],
                                                 closed_loop=True)
                # P0-2: inspect the warnings. The trivial-loop RuntimeWarning
                # is a VALID Abelian result (not an anomaly). P1-7:
                # count by class — reflection/degenerate warnings are
                # structural, not anomalies.
                for w in wlist:
                    m = str(w.message)
                    if "trivial" in m:
                        continue
                    n_anom += 1
                    if "reflection" in m:
                        n_anom_reflection += 1
                    elif "degenerate spectrum" in m:
                        n_anom_degenerate += 1
                if H012_save is not None and i == 0:
                    # P2 (docs/E38_fixes2.md): .cpu() guards against a
                    # future GPU-returning holonomy_spectrum (.tolist() on
                    # a CUDA tensor would raise).
                    H012_save.append(Htot.detach().cpu().clone())  # float64 (M-2)
                Hn = Htot.numpy()
                kappa_all[t, i] = float(np.linalg.norm(Hn - np.eye(R)))
                # trace from the MATRIX (always exact): a mixed spectrum
                # returns < r//2 angles (trivial +1 pairs dropped per the
                # holonomy_spectrum contract), so 2*sum(cos(th)) would
                # undercount the trace there.
                trH_all[t, i] = float(np.trace(Hn))
                # theta storage: NaN-pad for length (a mixed spectrum can
                # return < r//2 angles; NaN is ignored by nanmax/nanmin/
                # nanmedian, zero-padding would bias spread high —
                # docs/E38_fixes.md part-3 §2). Does not affect degenerate
                # does not affect degenerate detection (that uses step_res).
                if len(th) > R // 2:  # P1-7: loud, not a shape crash mid-layer
                    raise ValueError(
                        f"holonomy_spectrum returned {len(th)} angles > "
                        f"{R // 2} at L{l} t={t} i={i}")
                theta_all[t, i, :len(th)] = th
                # Exact identity for ANY orthogonal (normal) H, summed over
                # ALL r eigenvalues:  ||H - I||_F^2 = sum_j 2 (1 - cos
                # ang_j).  (The naive 4*sum(1-cos theta) over the r//2
                # reported angles OVER-COUNTS a lone -1 reflection axis,
                # which can appear because a Procrustes step is the polar
                # factor with no det-fix and may be a reflection: the axis
                # contributes 4, the reported pi angle implies 8.)  Computed
                # here from the SAME float64 Htot as kappa_all -> exact.
                ang_all = np.angle(
                    torch.linalg.eigvals(Htot).numpy())
                idiff_max = max(idiff_max, abs(
                    kappa_all[t, i] ** 2
                    - float((2.0 * (1.0 - np.cos(ang_all))).sum())))
                # P1-5: degenerate h (‖h‖ < 1e-10) → geodesic_curvature would
                # raise (unit-check) or divide by clamp_min(1e-12); mark NaN
                # and skip (excluded from rho via the finiteness mask).
                if min(float(norms[i, t]), float(norms[i, t + 1]),
                       float(norms[i, t + 2])) < 1e-10:
                    omega_all[t, i] = np.nan
                    n_omega_skipped += 1
                else:
                    omega_all[t, i] = geodesic_curvature(
                        u[i, t], u[i, t + 1], u[i, t + 2])
                if step_res > DEGEN_RES_FRAC:
                    degenerate[t, i] = True
                # trivial detection: th all ~0. (The "R_total in w.message"
                # branch was dead code — holonomy_spectrum raises a
                # RuntimeError for non-orthogonality, it never emits such a
                # warning; docs/E38_fixes.md §2.2.)
                if float(th.max()) < 1e-9:
                    trivial[t, i] = True
        from scipy.stats import spearmanr

        def rho_at(degen_mask):
            """rho_median (+ median p) over non-degenerate finite positions."""
            rs, ps = [], []
            for t in range(0, n_triples, 4):
                m = (~degen_mask[t] & np.isfinite(omega_all[t])
                     & np.isfinite(trH_all[t]))
                if m.sum() >= 8:  # P2 (docs/E38_fixes2.md): n_docs=32, 25%
                    # survival floor — noisy at 8 points, documented in meta
                    r_, p_ = spearmanr(trH_all[t, m], omega_all[t, m])
                    if np.isfinite(r_):
                        rs.append(float(r_))
                        ps.append(float(p_))
            return (float(np.median(rs)) if rs else float("nan"),
                    float(np.median(ps)) if ps else float("nan"),
                    len(rs))

        rho_med, p_med, n_rho_pos = rho_at(degenerate)
        # P0-4: threshold sweep — is rho an artifact of the 0.9*2r cutoff?
        degen_sweep = {}
        for frac in (0.7, 0.9, 0.95):
            msk = step_res_all > frac * 2 * R
            rm, pm, nm = rho_at(msk)
            degen_sweep[f"{frac}x2r"] = {"degenerate_frac": float(msk.mean()),
                                         "rho_median": rm, "n_positions": nm}
        theta_flat = theta_all[~degenerate]
        spread = float(np.nanmax(theta_flat) - np.nanmin(theta_flat))
        hist = np.histogram(theta_flat,
                            bins=[0, 0.01, 0.05, 0.1, 0.5, 1.0, np.pi])[0]
        per_layer[str(l)] = {
            "kappa_median": float(np.median(kappa_all)),
            "kappa_ci": list(np.percentile(kappa_all, [2.5, 97.5])),
            "theta": {
                "median_per_j": [float(np.nanmedian(theta_all[:, :, j]))
                                 for j in range(R // 2)],
                "hist_8_angles": [float(c) for c in hist],
                "n_dropped_nan": int(theta_flat.size - hist.sum()),  # M-4
                "spread_max_minus_min": spread},
            "trH_vs_omega": {"rho_median": rho_med, "n_positions": n_rho_pos,
                             "p_med": p_med},
            "identity_check_max_absdiff": float(idiff_max),
            "degenerate_frac": float(degenerate.mean()),
            "degen_sweep": degen_sweep,
            "anomaly_count": n_anom,       # P0-2: non-trivial warnings
            "n_omega_skipped": n_omega_skipped,  # P1-5: ‖h‖ < 1e-10 triples
            "trivial_frac": float(trivial.mean()),
            "profile_kappa": [float(np.median(kappa_all[t]))
                              for t in range(0, n_triples, 8)],
        }
        r = per_layer[str(l)]
        del J, H
        if H012_save is not None:
            r["H_012_doc0"] = [H.tolist() for H in H012_save]
        if l == 23:
            abs_diff = abs(r["kappa_median"] - KAPPA_L23_E28)
            r["consistency_e28"] = {
                "kappa_L23_this": r["kappa_median"],
                "kappa_L23_E28": KAPPA_L23_E28,
                "abs_diff": abs_diff,
                # DIAGNOSTIC, not a hard verdict (docs/E38_fixes.md §1.3):
                # 1e-3 is tighter than the rounding of the cited 0.018 and
                # can false-fail; 5e-3 is the pre-registered tolerance.
                "within_tol_5e-3": abs_diff < 5e-3,
                "within_tol_1e-3": abs_diff < 1e-3}
        print(f"L{l}: κ_med={r['kappa_median']:.5f} "
              f"θ_med={float(np.nanmedian(theta_all)):.5f} spread={spread:.4f} "
              f"ρ(trH,Ω)={rho_med:.3f} (p_med={p_med:.3f}) "
              f"degen={r['degenerate_frac']:.3f} "
              f"trivial={r['trivial_frac']:.3f} anom={n_anom}")

    verdicts = {}
    for l, r in per_layer.items():
        v = []
        if l == "23":
            # consistency is a DIAGNOSTIC (tolerance 5e-3 chosen
            # empirically, docs/E38_fixes.md §1.3): red flag only if the
            # cited κ is off by > 5e-3.
            v.append("consistency_pass" if r["consistency_e28"]["within_tol_5e-3"]
                     else "consistency_redflag")
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
        "meta": {"label_semantics": "third_mechanism ≡ NOT spherical "
                                    "(ρ < 0.3): the spherical-geometry "
                                    "hypothesis is falsified and the "
                                    "interpretation requires revision — it "
                                    "does NOT assert that a third mechanism "
                                    "was identified (docs/E38_fixes.md "
                                    "part-5 §5)",
                 "note": "degenerate = any Procrustes step residual_sq > "
                         "0.9*2r (near-orthogonal subspaces); excluded from "
                         "rho. degen_sweep reports degenerate_frac and "
                         "rho_median at 0.7/0.9/0.95*2r (is the 0.9*2r "
                         "cutoff an artifact?). trivial (theta≈0) is a "
                         "valid Abelian result, not a bug. tr(H) and "
                         "‖H−I‖_F² are exact functions of {θ_j} "
                         "(identity_check_max_absdiff). theta_all is "
                         "NaN-padded (mixed spectrum may return < r//2 "
                         "angles); n_dropped_nan counts the histogram "
                         "dropouts. anomaly_count = non-trivial "
                         "holonomy_spectrum warnings (π-boundary splits "
                         "etc.; the π-loss bug was fixed in "
                         "prism/holonomy.py). n_omega_skipped = triples "
                         "with ‖h‖ < 1e-10 (Omega undefined, excluded "
                         "from rho). p_med = median spearmanr p "
                         "(asymptotic t-approximation, NOT permutation). "
                         "H_012_doc0 (L23) is float64. consistency_e28 "
                         "is a diagnostic with an empirical 5e-3 "
                         "tolerance, not a hard verdict (1e-3 is tighter "
                         "than the rounding of the cited 0.018)."},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    def _json_default(o):
        """NaN/Inf → 'nan'/'inf' strings (valid RFC-8259 JSON; cache.py "
        "_json_dump_value does the same). default=float alone would emit "
        "the NaN literal, which strict parsers reject (part-5 §2, HIGH-3)."""
        if isinstance(o, float) and not np.isfinite(o):
            return "nan" if o != o else ("inf" if o > 0 else "-inf")
        return str(o)
    json.dump(out, open(OUT, "w"), indent=1, default=_json_default)
    print(f"VERDICTS {verdicts}")
    print(f"JSON -> {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
