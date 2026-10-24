#!/usr/bin/env python3
"""E-38d — Full holonomy matrix H_012 and spectrum {θ_j} (docs/E38.md).

Wave 2 (post E-38c; c reported near_degeneracy on L18/L22 → {θ_j} is
content-bearing, per the pre-registered gate).

For each triple (t, t+1, t+2), each doc, each layer, each R in R_VALUES:
    V_i = top_r_svd(J_i, R)[2].T        [D, R] orthonormal-column basis
    (R_ij, res_ij) = procrustes_align(V_i, V_j)   per-step residual_sq
    H_012 = R_01 @ R_12 @ R_20 (cycle order, float64 — see _procrustes)
    theta = [R//2] spectral angles of H_012 (same binning contract as
            prism.holonomy.holonomy_spectrum; NaN-padded when fewer)
    kappa = ‖H_012 − I‖_F ; trH = tr(H_012)
    Omega = geodesic_curvature(û_t, û_{t+1}, û_{t+2})   (R-independent)

VECTORIZED (v2, 2026-10-24): the old per-triple Python loop was > 2 h on
CPU (126 triples x 32 docs x 6 layers x {3 SVDs + 3 Procrustes + eig}).
v2 computes, per layer, ONE batched SVD over all (doc, t) positions
[N·S, R, D] and three batched [N·(S-2), R, R] Procrustes SVDs, then the
per-triple holonomy H is a small [r, r] matmul loop (cheap). On a T4 this
fits in ~minutes, which frees the 10 T4-h budget for the R extension:
R_VALUES = (16, 32, 64, 128). R = 16 stays the HEADLINE (pre-registered
value; all verdicts use r=16, JSON metrics."16" keeps the old field
names). R > 16 is a POST-HOC r-sweep (sensitivity: does the spectrum
converge with r?) — it does not change any pre-registered outcome.

J = RAW J from the exp27b cache (same convention as E-28, which gave
κ = 0.018 on L23 — consistency check; PRE-REGISTERED tolerance 5e-3,
1e-3 is a tighter DIAGNOSTIC only — the header previously said 1e-3,
contradicting the code; docs/E38_fixes2.md P1-4).

Per layer (headline r=16): {θ_j} distribution over docs, ρ(tr(H), Ω) by
position (Spearman), ‖H−I‖_F vs E-28 κ(L23) = 0.018.

Pre-registered outcomes (per layer, r = 16; docs/E38_fixes2.md P1-5 — the
old docstring did not list them, so the post-hoc ones are now marked):
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

Trivial H_012 ≈ I → spectrum all +1 → theta = [0,...,0]: a VALID Abelian
result, not a bug (recorded as trivial).

Deterministic: seed=0. spearmanr's p is an asymptotic t-approximation
(NOT a permutation test; seed is irrelevant for it) — the median p is
reported as p_med for diagnostics (docs/E38_fixes.md part-5 §3, M-3).
"""
import json
import os
import sys
import time

import numpy as np
import torch

# REPO/OUT are meaningless when inlined into a Kaggle notebook cell
# (no __file__); the Kaggle path is resolved inside main() via /kaggle/working.
try:
    _HERE = os.path.abspath(os.path.dirname(__file__))
except NameError:  # inlined into a notebook cell
    _HERE = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from _e38_cache import LAYERS, S, D, load_layer  # noqa: E402
from prism.spectral import top_r_svd  # noqa: F401 (kept for traceability)

SEED = 0
R_HEADLINE = 16
R_VALUES = (16, 32, 64, 128)   # v2 extension: r-sweep (budget: T4)
KAPPA_L23_E28 = 0.018
DEGEN_RES_FRAC = 0.9          # degenerate step: residual_sq > frac * 2r

# _HERE is defined above the imports (inlining-safe __file__).
REPO = os.path.abspath(os.path.join(_HERE, ".."))
OUT = os.path.join(REPO, "results", "exp38d_holonomy_spectrum.json")
# OUT is the LOCAL default; main() overrides it on Kaggle (see OUT_PATH).


def _procrustes_batched(V_a, V_b):
    """Vectorized procrustes_align (prism/holonomy.py contract).

    V_a, V_b: [B, D, r] orthonormal columns. M = V_b^T V_a [B, r, r]
    (row-major: M_{ij} = Σ_d V_b_{d i} V_a_{d j}), SVD M = P Σ Qt,
    R = Qt^T P^T — EXACTLY the scalar function's construction (minimizes
    ‖V_a R − V_b‖_F). v2.0 had the factor order swapped (V_a^T V_b),
    which gives the same singular values (hence the same residual_sq) but
    a different R — caught by the scalar-vs-batched cross-check.

    Returns (R [B, r, r] float64, residual_sq [B] float):
        residual_sq = 2r − 2·Σ_i s_i, clamped at 0 — EXACTLY the scalar
        formula (‖V_a R − V_b‖_F² = 2r − 2 tr(M^T R) = 2r − 2 Σ s).
    """
    M = torch.bmm(V_b.transpose(1, 2), V_a).double()
    P, s, Qt = torch.linalg.svd(M, full_matrices=False)
    R = torch.bmm(Qt.transpose(1, 2), P.transpose(1, 2))
    res = (2.0 * V_a.shape[2] - 2.0 * s.sum(dim=-1)).clamp(min=0.0)
    return R, res


def _spectrum_batched(H):
    """Vectorized angle set of H: [B, r] float64 orthogonal → angles.

    Mirrors prism/holonomy.holonomy_spectrum's binning contract:
      * one angle per conjugate pair {e^{iθ}, e^{−iθ}}, sorted ascending;
      * +1 axes dropped; the lone-−1 (reflection) axis lands in the π bin;
      * a π pair split across the ±π boundary is merged into one (counted
        once); repeated angles (degenerate spectrum) keep multiplicity;
      * if fewer than r//2 angles survive, the array is NaN-PADDED
        (v1 convention — zero-padding would bias spread high).
    Returns (theta [B, r//2] with NaN padding, n_angles [B]).
    """
    B, r = H.shape[0], H.shape[1]
    Hn = H.numpy()
    half = r // 2
    theta = np.full((B, half), np.nan, dtype=np.float64)
    n_ang = np.zeros(B, dtype=int)
    for bi in range(B):
        angs = np.angle(np.linalg.eigvals(Hn[bi]))   # (-π, π]
        pos = angs[angs > 1e-6]
        if pos.size == 0:
            continue                                  # H ≈ I (or −I? no:
                                                      # −I gives π angles)
        rounded = np.round(pos, 8)
        uniq, counts = np.unique(rounded, return_counts=True)
        mult = counts.copy()
        pi_mask = uniq > np.pi - 1e-6
        mult[pi_mask] //= 2
        mult[mult == 0] = 1   # π-pair split across the ±π boundary
        vals = np.repeat(uniq, mult)
        vals = vals[vals <= np.pi + 1e-12]            # guard: none > π
        vals.sort()
        take = vals[:half]
        theta[bi, :take.size] = take
        n_ang[bi] = take.size
    return theta, n_ang


def main(results_dir="results", device="auto"):
    t0 = time.time()
    np.random.seed(SEED)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"E-38d v2: device={device}, R_VALUES={R_VALUES} (headline {R_HEADLINE})")

    # Output dir: explicit > /kaggle/working > repo (kaggle.md pattern;
    # /kaggle/input is read-only, /kaggle/working survives the run).
    if results_dir:
        out_dir = results_dir
    elif os.path.isdir("/kaggle/working"):
        out_dir = os.path.join("/kaggle/working", "results")
    else:
        out_dir = os.path.join(REPO, "results")
    os.makedirs(out_dir, exist_ok=True)
    OUT_PATH = os.path.join(out_dir, "exp38d_holonomy_spectrum.json")

    from scipy.stats import spearmanr

    def rho_at(trH, omega, degen_mask, n_triples, n_docs):
        """rho_median (+ median p) over non-degenerate finite positions."""
        rs, ps = [], []
        for t in range(0, n_triples, 4):
            m = (~degen_mask[t] & np.isfinite(omega[t])
                 & np.isfinite(trH[t]))
            if m.sum() >= 8:  # P2 (docs/E38_fixes2.md): n_docs=32, 25%
                # survival floor — noisy at 8 points, documented in meta
                r_, p_ = spearmanr(trH[t, m], omega[t, m])
                if np.isfinite(r_):
                    rs.append(float(r_))
                    ps.append(float(p_))
        return (float(np.median(rs)) if rs else float("nan"),
                float(np.median(ps)) if ps else float("nan"), len(rs))

    per_layer = {}
    for l in LAYERS:
        ck = load_layer(l)
        J, H = ck["J"], ck["H"]           # [N, S, K, D], [N, S, D]
        N, Ss, Kk, Dd = J.shape
        n_triples = Ss - 2
        del ck
        norms = H.norm(dim=-1)
        u = H / norms.clamp_min(1e-12)[:, :, None]

        # Ω(t) — R-independent, computed once (vectorized geodesic curvature;
        # same formula as prism.holonomy.geodesic_curvature; NaN where
        # ‖h‖ < 1e-10 or arc < 1e-8 — P1-5).
        u1, u2, u3 = u[:, :-2], u[:, 1:-1], u[:, 2:]
        arc = (torch.norm(u1 - u2, dim=-1) + torch.norm(u3 - u2, dim=-1)) / 2
        t1 = u1 - (u1 * u2).sum(-1, keepdim=True) * u2
        t3 = u3 - (u3 * u2).sum(-1, keepdim=True) * u2
        n1, n3 = torch.norm(t1, dim=-1), torch.norm(t3, dim=-1)
        cos_turn = ((t1 * t3).sum(-1) / (n1 * n3).clamp_min(1e-12)).clamp(-1, 1)
        turn = np.pi - torch.acos(cos_turn)
        deg_h = (norms[:, :-2].min(dim=-1).values < 1e-10)
        deg_arc = (arc < 1e-8) | (n1 < 1e-8) | (n3 < 1e-8)
        omega_all = torch.where(deg_h | deg_arc,
                                torch.full_like(arc, float("nan")),
                                turn / arc.clamp_min(1e-8)).numpy()
        n_omega_skipped = int((deg_h | deg_arc).sum())

        per_layer[str(l)] = {}
        for R in R_VALUES:
            Jflat = J.reshape(N * Ss, Kk, Dd).to(device, non_blocking=True)
            _, _, V = top_r_svd(Jflat, R)              # [N*S, R, D]
            del Jflat
            Vt = V.transpose(1, 2).contiguous()        # [N*S, D, R]
            # triple (t, t+1, t+2): index (i, t) -> flat i*S + t, t=0..S-3
            idx = torch.arange(N * n_triples, device=device)
            i_doc = idx // n_triples                   # flat doc index
            t_pos = idx % n_triples
            f0 = i_doc * Ss + t_pos
            f1 = f0 + 1
            f2 = f0 + 2
            V0, V1, V2 = Vt[f0], Vt[f1], Vt[f2]        # [N*T, D, R]
            R01, res01 = _procrustes_batched(V0, V1)
            R12, res12 = _procrustes_batched(V1, V2)
            R20, res20 = _procrustes_batched(V2, V0)
            del Vt, V0, V1, V2, V
            step_res = torch.maximum(torch.maximum(res01, res12), res20)
            degen = (step_res > DEGEN_RES_FRAC * 2 * R).numpy()
            # H = R01 @ R12 @ R20, cycle order (v1: R_total = I @ R01 @ R12
            # @ R20, holonomy_accumulate left-multiplication convention)
            B = R01.shape[0]
            Htot = np.empty((B, R, R), dtype=np.float64)
            for bi in range(B):
                Htot[bi] = (R01[bi] @ R12[bi] @ R20[bi]).numpy()
            del R01, R12, R20
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

            kappa_all = np.linalg.norm(Htot - np.eye(R), axis=(1, 2))
            trH_all = np.trace(Htot, axis1=1, axis2=2)
            theta_all, n_ang = _spectrum_batched(torch.tensor(Htot))
            max_theta = np.nanmax(theta_all[~degen], axis=1) if degen.any() \
                else np.nanmax(theta_all, axis=1)   # P0-1: non-degenerate
            # exact identity: ||H−I||_F^2 = Σ_j 2(1−cos ang_j) over ALL r
            # eigenvalues (v1's M-2 check, kept per triple)
            angs = np.angle(np.linalg.eigvals(Htot))
            idiff = (kappa_all ** 2
                     - (2.0 * (1.0 - np.cos(angs))).sum(axis=1))
            idiff_max = float(np.abs(idiff).max())
            trivial = np.all(theta_all <= 1e-9, axis=1)
            rho_med, p_med, n_rho_pos = rho_at(trH_all, omega_all, degen,
                                               n_triples, N)
            degen_sweep = {}
            for frac in (0.7, 0.9, 0.95):
                msk = step_res.numpy() > frac * 2 * R
                rm, pm, nm = rho_at(trH_all, omega_all, msk, n_triples, N)
                degen_sweep[f"{frac}x2r"] = {"degenerate_frac": float(msk.mean()),
                                             "rho_median": rm, "n_positions": nm}
            dmask = ~degen
            theta_flat = theta_all[dmask]
            spread = (float(np.nanmax(theta_flat) - np.nanmin(theta_flat))
                      if theta_flat.size else float("nan"))
            hist = np.histogram(theta_flat,
                                bins=[0, 0.01, 0.05, 0.1, 0.5, 1.0, np.pi])[0]
            entry = {
                "kappa_median": float(np.median(kappa_all)),
                "kappa_ci": [float(x) for x in np.percentile(kappa_all, [2.5, 97.5])],
                "theta": {
                    "median_per_j": [float(np.nanmedian(theta_all[:, j]))
                                     for j in range(R // 2)],
                    "hist_8_angles": [float(c) for c in hist],
                    "n_dropped_nan": int(theta_flat.size - hist.sum()),  # M-4
                    "spread_max_minus_min": spread,
                    "max_theta": float(np.nanmax(max_theta)) if max_theta.size
                    else float("nan"),
                },
                "trH_vs_omega": {"rho_median": rho_med, "n_positions": n_rho_pos,
                                 "p_med": p_med},
                "identity_check_max_absdiff": idiff_max,
                "degenerate_frac": float(degen.mean()),
                "degen_sweep": degen_sweep,
                "trivial_frac": float(trivial.mean()),
                "n_omega_skipped": n_omega_skipped,
                "profile_kappa": [float(np.median(kappa_all[t]))
                                  for t in range(0, n_triples, 8)],
            }
            per_layer[str(l)][str(R)] = entry
            del Htot
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        # headline r=16: consistency vs E-28, H_012_doc0, .npz dump
        rh = per_layer[str(l)][str(R_HEADLINE)]
        if l == 23:
            abs_diff = abs(rh["kappa_median"] - KAPPA_L23_E28)
            rh["consistency_e28"] = {
                "kappa_L23_this": rh["kappa_median"],
                "kappa_L23_E28": KAPPA_L23_E28,
                "abs_diff": abs_diff,
                # DIAGNOSTIC, not a hard verdict (docs/E38_fixes.md §1.3):
                # 1e-3 is tighter than the rounding of the cited 0.018 and
                # can false-fail; 5e-3 is the pre-registered tolerance.
                "within_tol_5e-3": bool(abs_diff < 5e-3),
                "within_tol_1e-3": bool(abs_diff < 1e-3)}
        if l == 23:
            # P2 (docs/E38_fixes2.md): full H_012_doc0 as .npz (float64),
            # not JSON.
            Jt = J.reshape(N * Ss, Kk, Dd)
            _, _, V16 = top_r_svd(Jt.to(device), R_HEADLINE)
            Vt16 = V16.transpose(1, 2).contiguous()
            f0 = Ss + torch.arange(n_triples)   # doc 0
            V0 = Vt16[f0]; V1 = Vt16[f0 + 1]; V2 = Vt16[f0 + 2]
            R01, _ = _procrustes_batched(V0, V1)
            R12, _ = _procrustes_batched(V1, V2)
            R20, _ = _procrustes_batched(V2, V0)
            H012 = torch.stack(
                [R01[b] @ R12[b] @ R20[b] for b in range(n_triples)], dim=0
            ).cpu().double()
            np.savez(os.path.join(out_dir, f"exp38d_H012_doc0_L{l}.npz"),
                     H012=H012.numpy())
            del V16, Vt16, V0, V1, V2, R01, R12, R20
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        r = rh
        del J, H
        print(f"L{l} r={R_HEADLINE}: κ_med={r['kappa_median']:.5f} "
              f"maxθ={r['theta']['max_theta']:.5f} "
              f"ρ(trH,Ω)={r['trH_vs_omega']['rho_median']:.3f} "
              f"degen={r['degenerate_frac']:.3f} "
              f"trivial={r['trivial_frac']:.3f}")

    # ---- verdicts (headline r=16 ONLY — pre-registered) ----
    verdicts = {}
    for lstr, by_r in per_layer.items():
        r = by_r[str(R_HEADLINE)]
        v = []
        if lstr == "23":
            v.append("consistency_pass" if r["consistency_e28"]["within_tol_5e-3"]
                     else "consistency_redflag")
        sp = r["theta"]["max_theta"]     # P0-1: MAGNITUDE, not spread
        v.append("effectively_abelian" if sp < 0.1
                 else ("non_abelian_candidate" if sp > 0.5 else "mixed_spectrum"))
        rho_m = r["trH_vs_omega"]["rho_median"]
        if np.isfinite(rho_m):
            v.append("spherical_geometry" if rho_m > 0.7
                     else ("third_mechanism" if rho_m < 0.3 else "partial"))
        verdicts[lstr] = ",".join(v)

    out = {
        "params": {"seed": SEED, "r": R_HEADLINE, "r_values": list(R_VALUES),
                   "layers": list(LAYERS), "N_docs": 32,
                   "S": S, "J": "raw (E-28 convention)",
                   "degen_res_threshold": f"{DEGEN_RES_FRAC}*2r",
                   "device": device,
                   "cache": "exp27b_jcache_L{l}.pt"},
        "metrics": per_layer,
        "outcomes": {"per_layer": verdicts,
                     "r_sweep_note": "verdicts use r=16 only (pre-registered); "
                                     "r in (32, 64, 128) is a post-hoc "
                                     "sensitivity sweep (spectrum "
                                     "convergence with r)"},
        "meta": {"label_semantics": "third_mechanism ≡ NOT spherical "
                                    "(ρ < 0.3): the spherical-geometry "
                                    "hypothesis is falsified and the "
                                    "interpretation requires revision — it "
                                    "does NOT assert that a third mechanism "
                                    "was identified (docs/E38_fixes.md "
                                    "part-5 §5)",
                 "note": "v2 vectorized (identical contract to v1's "
                         "per-triple prism.holonomy calls, float64). "
                         "degenerate = any Procrustes step residual_sq > "
                         "0.9*2r (near-orthogonal subspaces); excluded "
                         "from rho. degen_sweep reports degenerate_frac "
                         "and rho_median at 0.7/0.9/0.95*2r (is the 0.9*2r "
                         "cutoff an artifact?). trivial (theta≈0) is a "
                         "valid Abelian result, not a bug. tr(H) and "
                         "‖H−I‖_F² are exact functions of {θ_j} "
                         "(identity_check_max_absdiff). theta_all is "
                         "NaN-padded (mixed spectrum may return < r//2 "
                         "angles); n_dropped_nan counts the histogram "
                         "dropouts. n_omega_skipped = triples with "
                         "‖h‖ < 1e-10 or arc < 1e-8 (Omega undefined, "
                         "excluded from rho). p_med = median spearmanr p "
                         "(asymptotic t-approximation, NOT permutation). "
                         "H_012_doc0 (L23, r=16) is float64 in "
                         "exp38d_H012_doc0_L23.npz. consistency_e28 is a "
                         "diagnostic with an empirical 5e-3 tolerance, "
                         "not a hard verdict (1e-3 is tighter than the "
                         "rounding of the cited 0.018)."},
    }

    def _json_default(o):
        """P0-3 (docs/E38_fixes2.md): strict JSON serializer — raise on
        np.bool_, convert numpy numerics explicitly, NaN/Inf → strings
        (valid RFC-8259)."""
        if isinstance(o, np.bool_):
            raise TypeError(f"unwrapped numpy bool_ in JSON: {o!r}")
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            o = float(o)
        if isinstance(o, float) and not np.isfinite(o):
            return "nan" if o != o else ("inf" if o > 0 else "-inf")
        raise TypeError(
            f"object of type {type(o).__name__} is not JSON serializable")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1, default=_json_default)
    print(f"VERDICTS {verdicts}")
    print(f"JSON -> {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
