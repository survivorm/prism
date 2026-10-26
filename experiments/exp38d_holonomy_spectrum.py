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
from prism.json_utils import dump as json_dump  # noqa: E402

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
        """rho_median (+ median p) over non-degenerate finite positions.

        trH/omega/degen_mask are [n_docs, n_triples] (doc=axis0, position=axis1).
        At each fixed position t we correlate trH[:, t] against omega[:, t]
        ACROSS the n_docs=32 documents (the per-position Spearman the TZ wants).
        """
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
        # degenerate triple: any of the 3 unit vectors has ‖h‖ < 1e-10
        # (norms is [N, S]; a triple (t, t+1, t+2) sees rows t..t+2)
        deg_h = (torch.stack([norms[:, :-2], norms[:, 1:-1], norms[:, 2:]], dim=0)
                 .min(dim=0).values < 1e-10)
        deg_arc = (arc < 1e-8) | (n1 < 1e-8) | (n3 < 1e-8)
        omega_all = torch.where(deg_h | deg_arc,
                                torch.full_like(arc, float("nan")),
                                turn / arc.clamp_min(1e-8)).numpy().T   # [T, N] (v1 layout)
        n_omega_skipped = int((deg_h | deg_arc).sum())

        per_layer[str(l)] = {}
        # Memory + speed layout (T4 16 GB):
        # ONE fp32 SVD of J per layer (K=128 => full_matrices=False yields
        # all 128 right-singular vectors), so every R in the sweep is a
        # slice of the SAME decomposition — the sweep is strictly
        # comparable (an earlier kernel version did a separate fp64 SVD
        # per R: 4x4096 128x1024 fp64 SVDs per layer ≈ days on T4 fp64 —
        # the kernel was killed at ~5.5 h). fp32 SVD on T4 ≈ 9 min/layer;
        # Vh [N*S,128,D] = 2.15 GB.
        # Per-R peak: Vh 2.15 + J 2.15 + V 2.15 (R=128) + 3 R-blocks fp64
        # 1.59 GB + Htot fp64 0.53 GB ≈ 8.6 GB — comfortably under 16 GB.
        # Procrustes stays fp64 (M [B,R,R] is tiny; T4 fp64 on 4032 small
        # matrices is seconds). Numbers drift ~1e-6 vs the v1 fp64 SVDs
        # (top-16 subspace is gap-protected; E-28 tol is 5e-3).
        Jflat = J.reshape(N * Ss, Kk, Dd)          # [N*S, K, D] fp32
        _, _, Vh = torch.linalg.svd(
            Jflat.to(device, dtype=torch.float32), full_matrices=False)
        del Jflat
        Vt = Vh[:, :R_HEADLINE].to(J.dtype).contiguous().transpose(1, 2)  # [N*S, D, 16]
        for R in R_VALUES:
            if R == R_HEADLINE:
                V = Vt          # [N*S, D, R], shared with the L23 npz block
            else:
                # .contiguous() makes a fresh tensor per R (at most ~2.15 GB
                # at R=128; at R=128 it is a VIEW of Vh — zero extra). The
                # previous R's tensor is GC'd on reassignment. Vh stays
                # alive across the sweep.
                V = Vh[:, :R].to(J.dtype).contiguous().transpose(1, 2)
            # CHUNK OVER DOCUMENTS (v8 OOM fix): the three full-batch V-
            # slices V0/V1/V2 ([N*T, D, R] each = 2.15 GB at R=128) plus the
            # fp64 R-blocks and M transient pushed peak past T4 16 GB.
            # Chunking caps the per-R working set; R=16 gives chunk == N
            # (single pass, bit-identical to the unchunked path).
            per_row = 3 * Dd * R * 4 + 48 * R * R      # 3 V-slices + 48 R-R fp64
            chunk = max(1, min(N, (1500 * 10**6) // (n_triples * per_row) or 1))
            B_tot = N * n_triples
            step_res_cpu = np.empty(B_tot, dtype=np.float64)
            Htot = np.empty((B_tot, R, R), dtype=np.float64)
            for c0 in range(0, N, chunk):
                c1 = min(c0 + chunk, N)
                b0, b1 = c0 * n_triples, c1 * n_triples
                idx = torch.arange(b0, b1, device=device)
                i_doc = idx // n_triples                # flat doc index
                t_pos = idx % n_triples
                f0 = i_doc * Ss + t_pos
                f1 = f0 + 1
                f2 = f0 + 2
                V0, V1, V2 = V[f0], V[f1], V[f2]        # [cT, D, R]
                # R01 (V0,V1) and R20 (V2,V0) first, then V0 is gone before
                # R12 — peak = 2 of the 3 V-slices + 3 R-blocks.
                R01, res01 = _procrustes_batched(V0, V1)
                R20, res20 = _procrustes_batched(V2, V0)
                del V0
                R12, res12 = _procrustes_batched(V1, V2)
                del V1, V2
                step_res_cpu[b0:b1] = torch.maximum(
                    torch.maximum(res01, res12), res20).cpu().numpy()
                # H = R01 @ R12 @ R20, cycle order (v1: R_total = I @ R01
                # @ R12 @ R20, holonomy_accumulate left-multiplication).
                # einsum index check (unit-tested vs torch.bmm chain):
                # R01[b,i,j] R12[b,j,k] R20[b,k,l] -> H[b,i,l]; a transposed
                # index (e.g. 'bsr') silently swaps the last factor.
                Htot[b0:b1] = torch.einsum(
                    "bij,bjk,bkl->bil", R01, R12, R20).cpu().numpy()
                del R01, R12, R20
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            if R != R_HEADLINE:
                del V
            degen_flat = (step_res_cpu > DEGEN_RES_FRAC * 2 * R)

            # Layouts (match v1 exactly):
            #   per-TRIPLE stats (kappa, theta, trivial, idiff): flat [B],
            #     B = N*n_triples, index = i_doc*n_triples + t_pos (doc-major).
            #   per-POSITION stats (rho, profile_kappa): [T, N] (v1 layout,
            #     position=axis0, doc=axis1).
            def _tn(flat):
                return flat.reshape(N, n_triples).T
            kappa_flat = np.linalg.norm(Htot - np.eye(R), axis=(1, 2))   # [B]
            trH_all = _tn(np.trace(Htot, axis1=1, axis2=2))              # [T, N]
            theta_flat = _spectrum_batched(torch.tensor(Htot))[0]        # [B, R//2]
            max_theta = np.nanmax(theta_flat, axis=1)                    # [B]
            degen_mask = _tn(degen_flat)                                 # [T, N]
            # exact identity: ||H−I||_F^2 = Σ_j 2(1−cos ang_j) over ALL r
            # eigenvalues (v1's M-2 check, kept per triple)
            angs = np.angle(np.linalg.eigvals(Htot))
            idiff_max = float(np.abs(kappa_flat ** 2
                                     - (2.0 * (1.0 - np.cos(angs))).sum(axis=1)).max())
            trivial = np.all(theta_flat <= 1e-9, axis=1)                 # [B]
            rho_med, p_med, n_rho_pos = rho_at(trH_all, omega_all, degen_mask,
                                               n_triples, N)
            degen_sweep = {}
            for frac in (0.7, 0.9, 0.95):
                msk = _tn(step_res_cpu > frac * 2 * R)
                rm, pm, nm = rho_at(trH_all, omega_all, msk, n_triples, N)
                degen_sweep[f"{frac}x2r"] = {"degenerate_frac": float(msk.mean()),
                                             "rho_median": rm, "n_positions": nm}
            th_masked = theta_flat[~degen_flat]                          # [K, R//2]
            th_vals = th_masked[np.isfinite(th_masked)]
            spread = (float(np.nanmax(th_vals) - np.nanmin(th_vals))
                      if th_vals.size else float("nan"))
            hist = np.histogram(th_vals,
                                bins=[0, 0.01, 0.05, 0.1, 0.5, 1.0, np.pi])[0]
            kappa_tn = _tn(kappa_flat)                                   # [T, N]
            entry = {
                "kappa_median": float(np.median(kappa_flat)),
                "kappa_ci": [float(x) for x in np.percentile(kappa_flat, [2.5, 97.5])],
                "theta": {
                    # High-R slices can be EMPTY (0 valid entries for some j):
                    # n_valid per j is recorded, empty slice median = "nan"
                    # (valid JSON string via json_utils.sanitize; the old
                    # default=_json_default was dead for Python floats and
                    # emitted the non-standard NaN literal —
                    # docs/E38_fixes.md §4).
                    "median_per_j": [float(np.nanmedian(theta_flat[:, j]))
                                     for j in range(R // 2)],
                    "n_valid_per_j": [int((theta_flat[:, j].nan_to_num(nan=-1.0)
                                           > -0.5).sum())
                                      for j in range(R // 2)],
                    "hist_8_angles": [float(c) for c in hist],
                    "n_dropped_nan": int(th_masked.size - th_vals.size),  # M-4
                    "spread_max_minus_min": spread,
                    "max_theta": float(np.nanmax(max_theta)) if max_theta.size
                    else float("nan"),
                },
                "trH_vs_omega": {"rho_median": rho_med, "n_positions": n_rho_pos,
                                 "p_med": p_med},
                "identity_check_max_absdiff": idiff_max,
                "degenerate_frac": float(degen_flat.mean()),
                "degen_sweep": degen_sweep,
                "trivial_frac": float(trivial.mean()),
                "n_omega_skipped": n_omega_skipped,
                "profile_kappa": [float(np.median(kappa_tn[t]))
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
            # not JSON. doc 0 = flat indices 0..Ss-1, so the triple
            # (t, t+1, t+2) starts at f0 = t (the old v2 code used
            # f0 = Ss + t — that is doc 1; the .npz was mislabeled).
            f0 = torch.arange(n_triples, device=device)   # doc 0
            V0 = Vt[f0]; V1 = Vt[f0 + 1]; V2 = Vt[f0 + 2]
            R01, _ = _procrustes_batched(V0, V1)
            R12, _ = _procrustes_batched(V1, V2)
            R20, _ = _procrustes_batched(V2, V0)
            H012 = torch.stack(
                [R01[b] @ R12[b] @ R20[b] for b in range(n_triples)], dim=0
            ).cpu().double()
            np.savez(os.path.join(out_dir, f"exp38d_H012_doc0_L{l}.npz"),
                     H012=H012.numpy())
            del V0, V1, V2, R01, R12, R20
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        r = rh
        del J, H, Vt
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
                         "rounding of the cited 0.018). "
                         "median_per_j: at high r many j-slices are EMPTY "
                         "(0 valid entries) — their median is the string "
                         "'nan' (valid JSON); n_valid_per_j records the "
                         "per-j counts (docs/E38_fixes.md §4)."},
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    # P0-3 (docs/E38_fixes2.md) + docs/E38_fixes.md §4: strict JSON —
    # sanitize non-finite floats (the old default= hook was dead for
    # Python floats: json.dump serializes float('nan') natively before
    # default= is consulted) + allow_nan=False as defense in depth.
    with open(OUT_PATH, "w") as f:
        json_dump(out, f, indent=1)
    print(f"VERDICTS {verdicts}")
    print(f"JSON -> {OUT_PATH} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
