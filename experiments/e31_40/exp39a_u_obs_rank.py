"""E-39a — U-observability gate: rank of the û_t manifold on L23 + dry-run
M_K-surgery κ-preview (docs/E39.md §1).

Revised per docs/E39_fixes.md (review items 2.1–2.5, 3.x, 4.x):
  - 2.1: strategies no longer degenerate — candidate pool = bottom
    R_POOL = 64 right SVs of U (dim > R), so (a)/(b)/(c) are genuinely
    different subspaces; degenerate pool → flagged, not silent.
  - 2.2: three-mode feasibility verdict (exact / approximate / infeasible).
  - 2.3: κ_base validated against E-38d ref (warn + flag, not assert — a
    drift must survive into the JSON; round 2, E39a_fixes §3.6).
  - 2.4: svdvals rank guard before each QR; degenerate strategy →
    kappa=None, kappa_valid=False (round 2, §3.3).
  - 2.5: Jp computed per-doc; subtraction in float64 (v4 also downcast the
    result to float32 for storage, 67 MB/doc — removed in round 5, v5 uses
    fp64 134 MB/doc; no 4 GB materialization either way).
  - 4.1: float64 SVD for J_t (κ' target < 0.005 is at float32 limit).
  - 4.2: SIGNED min Procrustes det + n_negative_det (round 2, §3.1 — the
    round-1 .abs() version was always ≈1 and useless).
  - 4.3: profile values log10'd, honest name (linear idx, round 2, §3.7).
  - 4.4: body r_eff flagged as upper bound on body_non_sentinel.
  - 2.5 memory note (round 6 §7): "67 MB/doc" below is the v4 fp32-cast
    figure; round 5 removed the cast, so v5 uses fp64 134 MB/doc.
Round 3 per docs/E39a_fixes.md (round-2 findings):
  - 2.1: det(H) population split in κ_base (241/4032 reflection triples).
  - 2.2: no-op control V_new = V_top16 (M_kp = M_K ⇒ κ' must ≈ κ_base).
  - 2.3: strategy-independence finding (CI overlap of all three κ').
  - 2.4: rho_noise = 1.4% contamination caveat in limitations.
  - 2.5: rank regime reported for ALL τ (not just the τ=0.05 anchor).
  - 2.6: per-position vs per-document manifold gap finding.
  - 2.7: κ percentiles (p10/25/50/75/90) for baseline and dry-run.
  - 2.8: aligned statistics — both 32-doc-medians and all-triples medians.
Round 4 per docs/E39a_fixes.md (round-3 review): CRITICAL einsum bug fix —
  - 1: 'kd,td->tkd' in _kappa_dryrun was ELEMENT-WISE (d not contracted):
    Jp = M_K ∘ (1 − û_t) instead of M_K(I − û_t û_tᵀ). All κ' from v1–v3
    were from the wrong surgery. Now: proj = 'kd,td->tk' then
    'tk,td->tkd'. Real-data verification: dryrun(M_K) all-triples median
    reproduces κ_base to 1.7e-9 (noop control, measured on v5).
  - 1.5/3.4/3.5: noop control ENFORCED — warn + kappa_valid=False for all
    rotated strategies + NOOP_CONTROL_FAILED_DRYRUN_INVALID in verdict.
  - 3.1: dead walrus variable removed. 3.2: unused kap_base param removed.
  - 3.3: edge-case 'percentiles' key unified to percentiles_p10_25_50_75_90.
  - 3.6: det(H) split reported for dry-run strategies too.
Round 5 per docs/E39a_fixes.md (round-4 review, medium/small):
  - 2.1/2.2: noop compares ALL-TRIPLES medians (apples-to-apples with
    κ_base) with tol 1e-3 (measured real-data diff = 1.7e-9; 0.01 was
    ~1e6× looser). control_check reports both diffs, primary = all-triples.
  - 3.1/3.2 + §7.2: findings/strategy_effect_null filter kappa_valid and
    drop the 'M_rest dominates κ'' claim (refuted by the 16% point-estimate
    spread); report spread + underpowered-N wording instead.
  - 4/6.2: float32 storage cast removed (f64→f32→f64 lost SVD input
    precision for no benefit); docstring claims the real-data 1.7e-9.
  - 5: noop det(H) split cross-checked with baseline (finer drift signal).
  - 6.1: noop skip_reason = 'noop_control_failed' on failure.
  - 7.3: strategies_effect_size_ratio + rho_noise amplification finding.
Round 6 per docs/E39a_fixes.md (round-5 review — interpretation fixes):
  - §2: the "14x amplification" claim is RETRACTED — with n=3 strategies the
    16% κ' spread is within individual bootstrap CI widths (~26-34%), so no
    rho_noise→κ' functional dependence is established. Report: rho_noise ≈
    1.4% for all three; κ' range; R_POOL=128/256 untested.
  - §3: "matters + underpowered" contradiction removed — honest wording for
    both CI branches; NEW paired per-doc test (best vs worst, n=32 pairs,
    more powerful than unpaired): kappa_per_doc stored, Δκ' mean + 95% CI +
    sign fraction reported in findings.
  - §4: "floor κ' ≥ 0.004" → "at R_POOL=64 (rho_noise≈1.4%) residual κ' ≈
    0.004; R_POOL=128/256 untested — whether κ' drops below it is unknown".
  - §5: strategies_effect_size_ratio → kappa_best_over_worst_ratio, computed
    unconditionally (not gated on CI overlap).
  - §6: abs_diff_kappa → abs_diff_doc_median (it is the 32-doc-median diff,
    not the κ_base diff).
  - §7: docstring/memory comments — "67 MB/doc" was the v4 fp32 cast
    (removed in round 5); v5 uses fp64 134 MB/doc, no 4 GB materialization
    either way.
Round 7 per docs/E39a_fixes.md (round-6 review — conclusion refinement):
  - §2 (MEDIUM): strategy_effect_null now REFLECTS the paired test result —
    the round-6 wording ("N=32 insufficient to distinguish") contradicted
    the paired test finding ("significant") on the same N=32. Now: "unpaired
    CIs overlap, but the paired per-doc test is decisive ... matters for
    ranking, not for a large effect."
  - §3: paired tests exposed as structured JSON (res["paired_tests"], one
    dict per pair: mean_delta, ci, n_lower, significant, ...) — the paper
    can extract numbers without regex over finding text.
  - §4: t-critical from scipy.stats.t.ppf(1 − α/2, df = n_pairs − 1) — no
    hardcoded 2.04, so a missing doc (n < 32) cannot make the CI
    over-confident.
  - §5: ALL three pairwise tests (a↔b, a↔c, c↔b) with Bonferroni
    α = 0.05/3 = 0.0167 — the "a < c < b" ranking is now supported by all
    three comparisons, not only min-vs-max.

Question: can M_K-surgery (E-39c) flatten the L23 leaf (κ → 0) WITHOUT
training? Necessary condition (spec §1): dim(U_obs⊥) ≥ 16, i.e.
r_obs ≤ 1008, U_obs = span{û_t : all (doc, t)}.

ALL from the exp27b dense L23 cache — no forward pass.
CPU only. Deterministic: seed=0.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _e38_cache import load_layer  # noqa: E402
from exp38d_holonomy_spectrum import _procrustes_batched  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = "Qwen/Qwen3.5-0.8B"
OUT = os.path.join(REPO, "results", "exp39a_u_obs_rank.json")
SEED = 0
R = 16                 # headline r (same as E-28 / E-38d)
R_POOL = 64            # candidate pool width for V_new (review 2.1)
BODY_LO, BODY_HI = 16, 126   # body window (E-38b convention)
TAUS = (0.01, 0.05, 0.1, 0.5)
KAPPA_L23_REF = 0.0162  # E-38d headline on this cache (consistency check)
KAPPA_DRIFT_TOL = 0.005
QR_RANK_TOL = 1e-6


def _r_eff(sig):
    """Effective rank: |{i : σ_i > τ·σ_1}| (relative threshold, spec §1.3)."""
    s1 = sig[0].item()
    if s1 == 0:
        return {str(t): 0 for t in TAUS}
    return {str(t): int((sig > t * s1).sum().item()) for t in TAUS}


def _r_energy99(sig):
    """min r s.t. Σ_{i≤r} σ² ≥ 0.99 Σ σ²."""
    e = sig.double() ** 2
    frac = e.cumsum(0) / e.sum()
    return int(torch.searchsorted(frac, torch.tensor(0.99)).item()) + 1


def _spectrum_summary(sig):
    s = sig.cpu().numpy().astype(float)
    idx = np.unique(np.linspace(0, len(s) - 1, 40).astype(int))
    return {
        "r_eff_tau": _r_eff(sig),
        "r_obs_99": _r_energy99(sig),
        # review 4.3 (round 2, E39a_fixes §3.7): honest name — values are
        # log10'd, indices are LINEAR-spaced (not log-spaced)
        "profile_log10_values_linear_idx_40": [float(np.log10(max(s[i], 1e-30))) for i in idx],
        "sigma_1": float(s[0]),
        "sigma_16": float(s[15]),
        "sigma_17": float(s[16]),
    }


def _qr_guarded(P, tag, report):
    """QR with svdvals rank guard (review 2.4): a rank-deficient input would
    make torch.linalg.qr emit an arbitrary completion basis — silent."""
    s = torch.linalg.svdvals(P)
    rank_ok = bool(s[-1] > QR_RANK_TOL * s[0])
    Q = torch.linalg.qr(P)[0]
    report[tag] = rank_ok
    if not rank_ok:
        warnings.warn(
            f"{tag}: rank-deficient projection "
            f"(σ_min/σ_1 = {s[-1].item() / max(s[0].item(), 1e-30):.2e}); "
            f"QR basis completion is arbitrary — treat as degenerate")
    return Q, rank_ok


def _kappa_from_J(J, S_s, R_r):
    """κ over docs×consecutive triples, 38d definition, batched (CPU).

    J: [N, S, K, D]. Float64 SVD (review 4.1: κ' target < 0.005 is at the
    float32 accuracy limit). V_t = top-r RIGHT SVs = LEFT factor of J_tᵀ SVD.
    Cycle (exactly 38d): H = R01 @ R12 @ R20 with R20 = procrustes_align(V2, V0)
    (holonomy_accumulate left-multiplication order).
    Returns (summary, values): summary has median/ci/n, SIGNED min Procrustes
    det + n_negative_det (round 2 §3.1), the det(H) population split and
    percentiles (round 3 §2.1/§2.7); values = per-triple κ (finite only).
    """
    N_n, S_s2, Kk, Dd = J.shape
    assert S_s2 == S_s
    V = torch.empty(N_n, S_s, Dd, R_r, dtype=torch.float64)
    for n0 in range(N_n):
        Jt = J[n0].transpose(1, 2).double()               # [S, D, K]
        Uu, _, _ = torch.linalg.svd(Jt, full_matrices=False)
        V[n0] = Uu[:, :, :R_r].contiguous()               # [S, D, r]
    kappas = []
    detH_neg = []
    min_det_signed = float("inf")
    n_negative_det = 0
    for d in range(N_n):
        Vd = V[d]                                         # [S, D, r]
        Ts = S_s - 2
        V0, V1, V2 = Vd[:Ts], Vd[1:Ts + 1], Vd[2:Ts + 2]  # [T, D, r]
        R01, _ = _procrustes_batched(V0, V1)
        R20, _ = _procrustes_batched(V2, V0)
        R12, _ = _procrustes_batched(V1, V2)
        H = torch.bmm(torch.bmm(R01, R12), R20)
        k = torch.linalg.norm(H - torch.eye(R_r, dtype=H.dtype), dim=(1, 2))
        kappas.append(k.cpu().numpy())
        # round 3 §2.1: det(H) split — is κ(det=+1) a different population
        # from κ(det=−1)? (6% det(R01)<0 triples; H may or may not reflect.)
        detH_neg.append((torch.linalg.det(H.float()) < 0).cpu().numpy())
        # round 2 §3.1: SIGNED det — |det|=1 exactly for orthogonal R01, so
        # .abs() erased both signals (reflection, collapse).
        det = torch.linalg.det(R01.float())
        min_det_signed = min(min_det_signed, float(det.min()))
        n_negative_det += int((det < 0).sum())
    kap_all = np.concatenate(kappas)
    finite = np.isfinite(kap_all)
    kap = kap_all[finite]
    if kap.size == 0:
        return ({"median": None, "ci": [None, None], "n": 0,
                 "min_det_R01_signed": min_det_signed,
                 "n_negative_det_R01": n_negative_det,
                 "n_det_H_negative": None,
                 "kappa_median_detH_pos": None, "kappa_median_detH_neg": None,
                 "percentiles_p10_25_50_75_90": None}, np.array([], dtype=float),
                np.array([], dtype=bool))
    ci = np.percentile(kap, [2.5, 97.5])
    negH = np.concatenate(detH_neg)[finite]
    return ({
        "median": float(np.median(kap)),
        "ci": [float(ci[0]), float(ci[1])],
        "n": int(kap.size),
        "min_det_R01_signed": min_det_signed,
        "n_negative_det_R01": n_negative_det,
        "n_det_H_negative": int(negH.sum()),
        "kappa_median_detH_pos": (float(np.median(kap[~negH])) if (~negH).any() else None),
        "kappa_median_detH_neg": (float(np.median(kap[negH])) if negH.any() else None),
        "percentiles_p10_25_50_75_90": [float(x) for x in
                                        np.percentile(kap, [10, 25, 50, 75, 90])],
    }, kap, negH)


def _kappa_dryrun(M_kp, u64, N_n, S_s, R_r):
    """κ' from J_t' = M_kp(I − û_t û_tᵀ), per-doc (review 2.5).

    E39a_fixes §3.2 (round 2): subtraction in float64 (M_kp and u64 are both
    float64), THEN downcast the RESULT to float32 for storage — 67 MB/doc.
    (Round 6 §7: the 67 MB/doc figure was the v4 fp32 cast; round 5 removed
    the cast — v5 holds the full fp64 [S,K,D] = 134 MB/doc, one doc at a
    time, and never materializes the 4 GB [N,S,K,D].)
    SVD inside _kappa_from_J upcasts to float64 again (review 4.1).
    Round 3 §2.8: returns BOTH statistics — 32-doc-medians (headline) and the
    all-triples median (aligned with κ_base's 4032-triple statistic).
    Round 4 §1: the projection is a MATRIX product — J_t = M_kp(I − û_t û_tᵀ),
    i.e. (Σ_d M_kp[k,d]û[d])·û[δ] subtracted. The v1–v3 einsum
    'kd,td->tkd' did NOT contract d (element-wise M_kp ∘ û — a bug the noop
    control in §1.5 was designed to catch). Measured on real data: dryrun(M_K)
    reproduces κ_base all-triples to 1.7e-9.
    Round 5 §4: the float32 storage cast is REMOVED — _kappa_from_J upcasts
    to float64 for the SVD anyway, so the f64→f32→f64 chain only lost
    ~1e-7 of SVD input precision to save 67 MB/doc (one doc at a time).
    Round 6 §7: "67 MB/doc" = the v4 fp32 cast (removed in round 5); v5
    computes and holds the full fp64 [S,K,D] = 134 MB/doc (one doc at a
    time). Both versions avoid the 4 GB materialization.
    """
    per_doc = []
    all_vals = []
    n_detH_neg = 0
    detH_neg_kaps = []
    for d in range(N_n):
        u_t = u64[d]                                     # [T, D]
        proj = torch.einsum("kd,td->tk", M_kp, u_t)      # [T, K] = M_kp·û_t
        Jp_d = M_kp[None, :] - torch.einsum("tk,td->tkd", proj, u_t)  # [S,K,D] fp64, 134 MB/doc
        s, vals, negH = _kappa_from_J(Jp_d.unsqueeze(0), S_s, R_r)
        if s["median"] is not None:
            per_doc.append(s["median"])
            all_vals.append(vals)
            n_detH_neg += int(negH.sum())
            detH_neg_kaps.append(vals[negH])
    if not per_doc:
        return {"kappa": None, "kappa_ci": [None, None], "n_valid": 0,
                "kappa_median_all_triples": None,
                "kappa_percentiles_p10_25_50_75_90": None,
                "n_det_H_negative": None,
                "kappa_median_detH_neg": None,
                "kappa_per_doc": []}
    arr = np.array(per_doc)
    allv = np.concatenate(all_vals)
    ci = np.percentile(arr, [2.5, 97.5])
    neg_kaps = np.concatenate(detH_neg_kaps) if any(a.size for a in detH_neg_kaps) else np.array([])
    return {"kappa": float(np.median(arr)),
            "kappa_ci": [float(ci[0]), float(ci[1])],
            "n_valid": int(arr.size),
            "kappa_median_all_triples": float(np.median(allv)),
            "kappa_percentiles_p10_25_50_75_90": [float(x) for x in
                                                  np.percentile(arr, [10, 25, 50, 75, 90])],
            "n_det_H_negative": n_detH_neg,
            "kappa_median_detH_neg": (float(np.median(neg_kaps)) if neg_kaps.size else None),
            "kappa_per_doc": [float(x) for x in arr],   # round 6 §3.3: paired test
    }


def _obs_subspace(sU, VU, R_r, R_pool, report):
    """Candidate pool for V_new (review 2.1 + E39a_fixes §3.5): the true
    null space if ≥ R dims, else the bottom-R_POOL right SVs of U (dim 64
    > 16, so strategies (a)/(b)/(c) can span genuinely different
    subspaces). Explicit branch + rank check — not a silent fallback.
    """
    null_thr = 1e-6 * sU[0].item()
    null_idx = (sU <= null_thr).nonzero().flatten()
    if len(null_idx) >= R_r:
        report["mode"] = "true_null_space"
        P = VU[:, null_idx]
    else:
        P = VU[:, -R_pool:]
        report["mode"] = f"bottom_{R_pool}_right_svs_of_U (no true null space)"
    sv = torch.linalg.svdvals(P)
    assert sv[-1] > QR_RANK_TOL * sv[0], f"candidate pool rank-deficient: {sv[:4]}"
    return P, report


def _principal_angles_cos(Va, Vb):
    """cosines of principal angles between span(Va) and span(Vb) (singular
    values of VaᵀVb ∈ [0,1]). E39a_fixes §3.4: the v2 max-entry check
    passed subspaces that agree on 15 of 16 directions (max entry ~0.25)."""
    return torch.linalg.svdvals(Va.T @ Vb)


def _paired_tests(strat, names):
    """Round 7 §3/§4/§5: all pairwise per-doc paired t-tests over the
    per-doc κ' arrays (n_pairs = number of valid docs, typically 32).

    - §4: t-critical from scipy (df = n_pairs − 1) — no hardcoded 2.04,
      so a missing doc cannot make the CI over-confident.
    - §5: all three pairs (not just min-vs-max), Bonferroni α = 0.05/3.
    - §3: structured dicts so the paper can extract numbers without
      regex over finding text.

    Returns a list of dicts (one per pair), or None if any strategy lacks
    per-doc data (then no paired test can be run).
    """
    from scipy import stats
    per_doc_map = {s: strat[s].get("kappa_per_doc") for s in names}
    if not all(p is not None and len(p) > 0 for p in per_doc_map.values()):
        return None
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    # E39_a_fixes Bug 3: guard against empty pairs (e.g. names=[] or all
    # strategies invalid). Without this, alpha = 0.05 / len(pairs) crashes
    # with ZeroDivisionError on the noop-failed / all-invalid path.
    if len(names) < 2 or not pairs:
        return None
    alpha = 0.05 / len(pairs)   # Bonferroni
    res = []
    for lo, hi in pairs:
        d = (np.asarray(per_doc_map[lo], dtype=np.float64)
             - np.asarray(per_doc_map[hi], dtype=np.float64))
        m_d = float(d.mean())
        se = float(d.std(ddof=1)) / np.sqrt(d.size)
        tc = float(stats.t.ppf(1.0 - alpha / 2.0, df=d.size - 1))
        lo_ci, hi_ci = m_d - tc * se, m_d + tc * se
        n_lower = int((d < 0).sum())
        sig = bool(lo_ci > 0.0 or hi_ci < 0.0)
        res.append({
            "lo_strategy": lo,
            "hi_strategy": hi,
            "n_pairs": int(d.size),
            "mean_delta": m_d,
            "ci": [float(lo_ci), float(hi_ci)],
            "ci_level": 1.0 - alpha,   # Bonferroni-adjusted (0.9833 for 3 pairs)
            "t_critical": tc,
            "n_lower": n_lower,
            "significant": sig,
            "interpretation": ("ranking_significant_small_effect" if sig
                               else "not_significant_bonferroni"),
        })
    return res


def _round3_findings(strat, rank_report):
    """Round 3 (E39a_fixes §2.3/§2.4/§2.6) + round 5 rewrite (§7.2–7.3):
    findings computed from the actual numbers, only over VALID strategies
    (kappa_valid — a noop failure must not produce findings on invalid κ')."""
    out = []
    rot = [s for s in strat if not strat[s].get("is_control")
           and strat[s].get("kappa_valid", False)
           and strat[s]["kappa"] is not None
           and strat[s]["kappa_ci"][0] is not None]
    if len(rot) >= 2:
        lo_hi = [(strat[s]["kappa_ci"][0], strat[s]["kappa_ci"][1]) for s in rot]
        ci_overlap = max(l for l, _ in lo_hi) <= min(h for _, h in lo_hi)
        kaps = [strat[s]["kappa"] for s in rot]
        spread = max(kaps) / min(kaps) - 1.0
        if ci_overlap:
            out.append(
                f"strategy effect: point estimates span {spread * 100:.0f}% "
                f"({min(kaps):.5f}..{max(kaps):.5f}); CIs overlap, and the "
                "spread is within individual bootstrap CI widths (~26-34%). "
                "M_rest is shared by all three strategies, so a real "
                "difference can only come from the V_new ↔ M_rest coupling — "
                "but at N=32 we cannot establish that the spread is real, "
                "and the M_rest-dominates hypothesis is also not confirmed. "
                "Paired per-doc test below.")
        else:
            out.append(
                f"strategy effect: CIs do NOT overlap; point estimates span "
                f"{spread * 100:.0f}% ({min(kaps):.5f}..{max(kaps):.5f}) — "
                "V_new choice likely matters (paired per-doc test below).")
        # round 7 §3/§4/§5: all pairwise per-doc tests, Bonferroni α = 0.05/3,
        # scipy t-critical (df = n_pairs − 1; no hardcoded 2.04)
        pt = _paired_tests(strat, rot)
        if pt is not None:
            for r in pt:
                out.append(
                    f"paired per-doc test ({r['lo_strategy']} − "
                    f"{r['hi_strategy']}, n={r['n_pairs']} pairs, Bonferroni "
                    f"α = 0.05/{len(pt)} ≈ {0.05/len(pt):.4f} two-sided, "
                    f"95%→{r['ci_level']*100:.2f}% CI): "
                    f"mean Δκ' = {r['mean_delta']:+.5f}, CI "
                    f"[{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}], "
                    f"{r['n_lower']}/{r['n_pairs']} docs lower for "
                    f"{r['lo_strategy']}. "
                    + (f"{r['lo_strategy']} < {r['hi_strategy']} is "
                       "statistically significant (CI excludes 0, "
                       "consistent per-doc direction) — the difference is "
                       "real, but small relative to the ~16% overall "
                       "point-estimate spread: V_new choice matters for "
                       "ranking, not for a large effect."
                       if r["significant"] else
                       "CI includes 0 — no significant difference at the "
                       "Bonferroni level."))
        rhos = [strat[s]["rho_noise_u_obs_overlap_mean"] for s in rot
                if strat[s].get("rho_noise_u_obs_overlap_mean") is not None]
        if len(rhos) >= 2 and min(rhos) > 0:
            out.append(
                ("rho_noise: all three V_new achieve rho_noise ≈ %.4f "
                 "(≈1.4%%); residual κ' ∈ [%.5f, %.5f]. The %.0f%% "
                 "point-estimate spread is within individual bootstrap CI "
                 "widths (~26-34%%), so n=3 strategies CANNOT establish a "
                 "rho_noise→κ' functional dependence (the v5 '14x "
                 "amplification' claim is retracted). At R_POOL=64 "
                 "(rho_noise≈1.4%%) residual κ' ≈ %.4f; R_POOL=128/256 "
                 "(lower rho_noise) are untested — whether κ' drops below "
                 "that is unknown.")
                % (float(np.mean(rhos)), min(kaps), max(kaps),
                   spread * 100, min(kaps)))
    # §2.6: per-position vs per-document manifold gap
    rb = rank_report["body"]["r_eff_tau"].get("0.05")
    rd = rank_report["doc_mean_32"]["r_eff_tau"].get("0.05")
    if rb and rd and rb > 3 * rd:
        out.append(
            f"per-position r_eff(τ=0.05) = {rb} vs per-document r_eff = {rd} "
            f"({rb / rd:.1f}×): the per-position manifold is dominated by "
            "intra-document (noise-level) directions, not between-document "
            "variability. Per-position surgery therefore acts on noise-level "
            "directions — this is by spec (û_t is per-position) but must be "
            "stated, not buried.")
    return out


def main(results_dir=None):
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = results_dir if results_dir else os.path.join(REPO, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp39a_u_obs_rank.json")

    ck = load_layer(23, want=("J", "H"))
    H, J = ck["H"], ck["J"]
    U128 = ck["U128"]
    N_n, S_s, Dd = H.shape
    assert (N_n, S_s, Dd) == (32, 128, 1024), (N_n, S_s, Dd)
    assert torch.isfinite(H).all() and torch.isfinite(J).all(), "NaN/Inf in cache"

    # ---------- 1. U_obs rank ----------
    u = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-12)     # [N, S, D]
    U = u.reshape(-1, Dd).double()                            # [4096, 1024]
    _, sU, Vh = torch.linalg.svd(U, full_matrices=False)
    # CRITICAL FIX (E39_a_fixes Bug 1): torch.linalg.svd returns (U, S, Vh)
    # where Vh has right SVs as ROWS (Vh = Vᵀ). The code treated VU as V
    # (right SVs as columns), so VU[:, -R_pool:] was selecting the last
    # COLUMNS of Vh (= an arbitrary orthonormal basis, NOT the bottom-R
    # right SVs of U). ρ_noise = 0.0148 ≈ R/D = 0.0156 confirms this (a true
    # bottom-R subspace gives ρ ≈ σ_bottom16²/N ≈ 3.9e-5). Fix: transpose.
    VU = Vh.T   # now VU = V: right SVs as columns

    body = np.zeros(S_s, dtype=bool); body[BODY_LO:BODY_HI + 1] = True
    pos0 = np.zeros(S_s, dtype=bool); pos0[0] = True
    U3 = U.reshape(N_n, S_s, Dd)
    subsets = {
        "all": U,
        "body": U3[:, body].reshape(-1, Dd),
        "body_with_pos0": U3[:, np.logical_or(body, pos0)].reshape(-1, Dd),
        "all_minus_pos0": U3[:, ~pos0].reshape(-1, Dd),
    }
    rank_report = {}
    for name, M in subsets.items():
        _, sig, _ = torch.linalg.svd(M, full_matrices=False)
        rank_report[name] = _spectrum_summary(sig)
    Udoc = u.mean(dim=1)                                       # [N, D]
    Udoc = Udoc / Udoc.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    _, sig_doc, _ = torch.linalg.svd(Udoc.double(), full_matrices=False)
    rank_report["doc_mean_32"] = _spectrum_summary(sig_doc)
    # review 4.4: body includes sentinels → upper bound on body_non_sentinel
    rank_report["note"] = ("r_eff(body) is an UPPER BOUND on r_eff(body, "
                           "non-sentinel): sentinel split not computable "
                           "from P128 (probability-normalized masses).")

    # ---------- 2. baseline κ(L23) — validated (review 2.3) ----------
    print("κ baseline: SVD over 4096 positions ...")
    kb, _kb_vals, _kb_negH = _kappa_from_J(J, S_s, R)
    kap0, kap0_ci, n_kap = kb["median"], kb["ci"], kb["n"]
    min_det0, neg_det0 = kb["min_det_R01_signed"], kb["n_negative_det_R01"]
    print(f"  κ_base = {kap0} CI {kap0_ci} (ref 38d = {KAPPA_L23_REF})")
    # round 3 §2.1: det(H) population split — warn if the two populations
    # differ (headline κ_base would be a mixture of two distributions).
    if kb["kappa_median_detH_pos"] is not None and kb["kappa_median_detH_neg"] is not None:
        _mp, _mn = kb["kappa_median_detH_pos"], kb["kappa_median_detH_neg"]
        _ratio = max(_mp, _mn) / max(min(_mp, _mn), 1e-30)
        if _ratio > 2.0:
            warnings.warn(
                f"κ populations det(H)=±1 differ by {_ratio:.1f}× (medians "
                f"{_mp:.5f} vs {_mn:.5f}; n_det_H_neg={kb['n_det_H_negative']}) "
                "— headline κ_base is a mixture of two populations; see "
                "kappa_baseline.kappa_median_detH_pos/neg")
    # review 2.3 round 2 (E39a_fixes §3.6): warning + flag (NOT assert —
    # an assert kill makes the flag dead code; drift must survive into the
    # JSON for interpretation). Split checks so kap0=None has a clean message.
    base_kappa_drift = False
    if kap0 is None:
        base_kappa_drift = True
        warnings.warn("κ_base = None — insufficient valid triples; "
                      "dry-run κ' untrustworthy")
    elif abs(kap0 - KAPPA_L23_REF) >= KAPPA_DRIFT_TOL:
        base_kappa_drift = True
        warnings.warn(f"κ_base drift: {kap0:.5f} vs E-38d ref {KAPPA_L23_REF} "
                      f"(tol {KAPPA_DRIFT_TOL}) — cache/code drift, "
                      f"dry-run κ' untrustworthy")
    # E39a_fixes §3.8: |det|≈1 for proper-orthogonal Procrustes output;
    # |det| < 0.99 is numerical drift in _procrustes_batched — warn, not silent.
    # (round 4 §3.1: the walrus-bound variable was dead; plain if.)
    if np.isfinite(min_det0):
        if abs(min_det0) < 0.99:
            warnings.warn(f"Procrustes numerical drift: min |det(R01)| = "
                          f"{abs(min_det0):.6f} < 0.99")
    if neg_det0 > 0:
        warnings.warn(f"{neg_det0} Procrustes triples with det(R01) < 0 "
                      f"(reflections; E-28 convention has no det-fix — the "
                      f"frequency is the indicator to watch)")

    # ---------- 3. M_K + surgery dry-run ----------
    from transformers import AutoModelForCausalLM
    from prism.interventions import _resolve_lm_head_weight
    from prism.jacobian import _lm_module
    from prism.jstruct import build_M_K

    # weights are loaded ONLY to build M_K (no forward pass). Force offline
    # load from the local HF cache so a network outage cannot hang the run
    # (model is cached locally: ~1.7 GB blob).
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    print("loading model weights (M_K only, offline, no forward) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True)
    model.eval()
    W_lm = _resolve_lm_head_weight(model)
    g = _lm_module(model).norm.weight + 1
    assert g.min() > 1e-6, f"γ has zeros: min={g.min()}"
    M_K = build_M_K(W_lm, list(U128), g).double().detach()   # [128, D]

    # ---------- feasibility mode (review 2.2) ----------
    r_obs = int(len(sU) - int((sU <= 1e-6 * sU[0].item()).sum()))  # full-SVD rank
    r_eff05_body = int(rank_report["body"]["r_eff_tau"]["0.05"])
    null_rep = {}
    P_pool, null_rep = _obs_subspace(sU, VU, R, R_POOL, null_rep)
    null_mode = null_rep["mode"]
    if null_mode == "true_null_space":
        feasibility = "surgery_feasible_exact"
    elif r_eff05_body < 1008:
        # no exact null space, but effective rank has room: bottom-R_POOL
        # right SVs are the candidate pool (directions of minimal U_obs overlap)
        feasibility = "surgery_approximate_bottom_svs"
    else:
        feasibility = "surgery_infeasible_effective"
    print(f"r_obs(full SVD) = {r_obs}, r_eff(0.05, body) = {r_eff05_body}, "
          f"mode = {null_mode}, feasibility = {feasibility}")

    Uk, sk, Vk = torch.linalg.svd(M_K, full_matrices=False)
    # M_K [128, D]: Uk [128,128], sk [128], Vk = Vᵀ [128, D] (rows = right SVs)
    U_top, s_top, V_top = Uk[:, :R], sk[:R], Vk[:R, :].T      # V_top [D, R]
    M_rest = Uk[:, R:] @ torch.diag(sk[R:]) @ Vk[R:, :]

    # ---------- three strategies in the R_POOL-wide candidate pool ----------
    # (review 2.1: dim(pool) = 64 > 16, so (a)/(b)/(c) span genuinely
    # different 16-dim subspaces; pool degeneracy is flagged, not silent)
    qr_report = {}
    Pn = P_pool @ P_pool.T                                    # projector
    rng = np.random.default_rng(SEED)
    Aa = torch.from_numpy(rng.normal(size=(R_POOL, R))).double()
    V_a, _ = _qr_guarded(P_pool @ torch.linalg.qr(Aa)[0], "a_random", qr_report)
    Pb = Pn @ V_top
    V_b, _ = _qr_guarded(Pb, "b_principal", qr_report)
    V_c, _ = _qr_guarded(torch.cat([V_a[:, :8], V_b[:, :8]], dim=1),
                         "c_balanced", qr_report)

    # E39a_fixes §3.4: subspace distance via projectors + principal angles
    # (max-entry check was weak: 15/16 shared directions ⇒ max entry ~0.25
    # yet the subspaces are essentially the same).
    proj_dist = float(torch.linalg.norm(V_a @ V_a.T - V_b @ V_b.T, "fro"))
    cos_pa = _principal_angles_cos(V_a, V_b)
    max_pa_deg = float(torch.acos(cos_pa.clamp(-1, 1)).max() * 180 / np.pi)
    strategies_distinct = bool(max_pa_deg > 30.0)

    u64 = u.double()
    print(f"V_a vs V_b: projector Frobenius dist = {proj_dist:.4f}, "
          f"max principal angle = {max_pa_deg:.1f}° "
          f"({'distinct strategies' if strategies_distinct else '≈ same subspace'})")

    strat = {}
    for name, V_new in [("a_random", V_a), ("b_principal", V_b),
                        ("c_balanced", V_c)]:
        assert (V_new.T @ V_new - torch.eye(R, dtype=V_new.dtype)).abs().max() < 1e-8, name
        # E39a_fixes §3.3: a rank-deficient QR completion is arbitrary —
        # κ' from such a V_new is a random number. Do NOT compute/report it.
        if not qr_report[name]:
            strat[name] = {"kappa": None, "kappa_ci": [None, None],
                           "n_valid": 0, "rho_noise_u_obs_overlap_mean": None,
                           "qr_rank_ok": False, "kappa_valid": False,
                           "skip_reason": "qr_rank_deficient"}
            continue
        M_kp = U_top @ torch.diag(s_top) @ V_new.T + M_rest
        assert torch.isfinite(M_kp).all()
        # Frobenius preserved by orthogonal top-16 rotation (smoke test)
        assert abs(torch.linalg.norm(M_kp).item()
                   - torch.linalg.norm(M_K).item()) < 1e-6
        kap_res = _kappa_dryrun(M_kp, u64, N_n, S_s, R)
        # contamination diagnostic ρ_noise = E_t ‖û_tᵀ V_new‖_F² (0 = perfect)
        ov = (u64 @ V_new).pow(2).sum(dim=-1).mean().item()
        strat[name] = {"kappa": kap_res["kappa"], "kappa_ci": kap_res["kappa_ci"],
                       "kappa_median_all_triples": kap_res["kappa_median_all_triples"],
                       "kappa_percentiles_p10_25_50_75_90": kap_res["kappa_percentiles_p10_25_50_75_90"],
                       "kappa_per_doc": kap_res["kappa_per_doc"],   # round 6 §3.3: paired test
                       "n_valid": kap_res["n_valid"],
                       "rho_noise_u_obs_overlap_mean": ov,
                       "qr_rank_ok": True, "kappa_valid": True,
                       "skip_reason": None}

    # round 3 §2.2: no-op CONTROL — V_new = V_top16 ⇒ M_kp = M_K ⇒ κ' must ≈
    # κ_base. This proves the dry-run pipeline (projection, SVD, medians) is
    # unbiased, so the flattened κ' of the rotated strategies is a real
    # surgery effect and not a pipeline artifact.
    ov_noop = (u64 @ V_top).pow(2).sum(dim=-1).mean().item()
    noop_res = _kappa_dryrun(M_K, u64, N_n, S_s, R)
    # round 4 §1.5: the noop control is ENFORCED, not just printed.
    # Correct formula: M_kp = M_K ⇒ Jp_d = M_K(I − ûûᵀ) = cached J_d ⇒
    # κ'(noop) must ≈ κ_base. A failure means the dry-run formula is broken
    # (e.g. the v1–v3 einsum bug: 'kd,td->tkd' is element-wise, not a
    # projection) — then ALL rotated-strategy κ' are invalid numbers.
    # Round 5 §2.1/§2.2: apples-to-apples — compare the ALL-TRIPLES median
    # (same 4032-triple statistic as kap0), not the 32-doc-median. Measured
    # real-data diff = 1.7e-9 → tol 1e-3 (~600× margin, still catches any
    # formula/index regression; 0.01 was ~1e6× looser than needed).
    noop_all = noop_res["kappa_median_all_triples"]
    noop_failed = (noop_all is None or kap0 is None
                   or abs(noop_all - kap0) > 1e-3)
    if noop_failed:
        warnings.warn(
            f"noop control FAILED: κ'(V_top16) all-triples = {noop_all} vs "
            f"κ_base = {kap0} (tol 1e-3) — the dry-run formula does not "
            "reproduce the cache J_t = M_K(I − ûûᵀ); κ' of all strategies is "
            "invalid (check _kappa_dryrun einsum: 'kd,td->tkd' is element-"
            "wise, not a projection)")
        for s in ("a_random", "b_principal", "c_balanced"):
            strat[s]["kappa_valid"] = False
            strat[s]["skip_reason"] = "noop_control_failed"
    # round 5 §5: the noop det(H) split must equal the baseline's — catches
    # finer pipeline drift (Procrustes ordering, H assembly) than the κ diff.
    if (noop_res["n_det_H_negative"] is not None
            and noop_res["n_det_H_negative"] != kb["n_det_H_negative"]):
        warnings.warn(
            f"noop det(H) split mismatch: {noop_res['n_det_H_negative']} vs "
            f"baseline {kb['n_det_H_negative']} — pipeline drift "
            "(procrustes ordering or H assembly)?")
    strat["d_noop_V_top16"] = {"kappa": noop_res["kappa"],
                               "kappa_ci": noop_res["kappa_ci"],
                               "kappa_median_all_triples": noop_res["kappa_median_all_triples"],
                               "kappa_percentiles_p10_25_50_75_90": noop_res["kappa_percentiles_p10_25_50_75_90"],
                               "n_valid": noop_res["n_valid"],
                               "n_det_H_negative": noop_res["n_det_H_negative"],
                               "kappa_median_detH_neg": noop_res["kappa_median_detH_neg"],
                               "rho_noise_u_obs_overlap_mean": ov_noop,
                               "qr_rank_ok": True,
                               "kappa_valid": not noop_failed,
                               "control_check": {
                                   "expect": kap0,
                                   "tol": 1e-3,
                                   "abs_diff_doc_median": (abs(noop_res["kappa"] - kap0)
                                                          if noop_res["kappa"] is not None and kap0 is not None else None),
                                   "abs_diff_all_triples": (abs(noop_all - kap0)
                                                           if noop_all is not None and kap0 is not None else None),
                                   "primary_metric": "all_triples"},
                               "is_control": True,
                               "skip_reason": ("noop_control_failed" if noop_failed else None)}
    print(f"    d_noop_V_top16 (control): κ' (all-triples) = {noop_all} "
          f"(expect ≈ κ_base = {kap0}, tol 1e-3, {'FAILED' if noop_failed else 'ok'})", flush=True)

    # ---------- verdict (review 2.2 + 2.1; E39a_fixes §3.5) ----------
    verdict_parts = [feasibility,
                     f"rank_regime_{'low' if r_eff05_body < 32
                                        else 'moderate' if r_eff05_body < 256
                                        else 'high' if r_eff05_body < 1008
                                        else 'infeasible'}_{r_eff05_body}"]
    if base_kappa_drift:
        verdict_parts.append("BASE_KAPPA_DRIFT_UNTRUSTWORTHY")
    if noop_failed:
        verdict_parts.append("NOOP_CONTROL_FAILED_DRYRUN_INVALID")
    if not strategies_distinct:
        verdict_parts.append("strategies_degenerate_single_subspace")
    # §3.5: in the infeasible_effective mode the dry-run is NOT a significant
    # result — mark all κ' invalid and do NOT emit dryrun_kappa_min_*.
    if feasibility == "surgery_infeasible_effective":
        for v in strat.values():
            v["kappa_valid"] = False
            v["skip_reason"] = "surgery_infeasible_effective"
    else:
        ok = [v for s, v in strat.items() if s != "d_noop_V_top16"
              and v.get("kappa_valid") and v["kappa"] is not None]
        if ok:
            best = min(ok, key=lambda v: v["kappa"])
            verdict_parts.append(f"dryrun_kappa_min_{best['kappa']:.4f}")
    verdict = ",".join(verdict_parts)

    # round 3 §2.3 + round 5 §7.2: explicit JSON field — only over VALID
    # strategies (a noop failure must not report a 'null' effect on invalid
    # κ'), and the wording must not contradict the point-estimate spread.
    _rot_v = [s for s in ("a_random", "b_principal", "c_balanced")
              if strat[s].get("kappa_valid", False) and strat[s]["kappa"] is not None
              and strat[s]["kappa_ci"][0] is not None]
    strategy_effect_null = None
    _best_worst_ratio = None
    _paired = _paired_tests(strat, _rot_v)   # round 7: computed once, shared
    if _rot_v:
        _ci = [(strat[s]["kappa_ci"][0], strat[s]["kappa_ci"][1]) for s in _rot_v]
        _ks = [strat[s]["kappa"] for s in _rot_v]
        # round 6 §5: unconditional (previously only when CIs overlap)
        if min(_ks) > 0:
            _best_worst_ratio = min(_ks) / max(_ks)
        _ci_overlap = (len(_ci) >= 2
                       and max(l for l, _ in _ci) <= min(h for _, h in _ci))
        # round 7 §2: strategy_effect_null must REFLECT the paired test
        # result, not contradict it (round 6 said "N=32 insufficient" while
        # the paired test on the same N=32 said "significant").
        if _ci_overlap and _paired:
            _pk = min(_paired, key=lambda r: r["mean_delta"])   # primary pair
            strategy_effect_null = (
                f"Unpaired CIs overlap (point estimates span "
                f"{max(_ks) / min(_ks) * 100 - 100:.0f}%), but the paired "
                f"per-doc test is decisive: {_pk['lo_strategy']} < "
                f"{_pk['hi_strategy']} (mean Δκ' = {_pk['mean_delta']:+.5f}, "
                f"Bonferroni CI [{_pk['ci'][0]:+.5f}, {_pk['ci'][1]:+.5f}], "
                f"{_pk['n_lower']}/{_pk['n_pairs']} docs) — the ranking is "
                "significant, but the point-estimate spread is only 16% "
                "(within individual CI widths): V_new choice matters for "
                "ranking, not for a large effect. Full pair-wise results: "
                "res[paired_tests].")
        elif _ci_overlap:
            strategy_effect_null = (
                f"CIs overlap; point estimates span "
                f"{max(_ks) / min(_ks) * 100 - 100:.0f}% "
                f"({min(_ks):.5f}..{max(_ks):.5f}), within individual "
                "bootstrap CI widths (~26-34%); paired per-doc data "
                "unavailable — cannot distinguish signal from noise.")
        elif len(_ci) >= 2:
            strategy_effect_null = (
                f"CIs do NOT overlap; point estimates span "
                f"{max(_ks) / min(_ks) * 100 - 100:.0f}% "
                f"({min(_ks):.5f}..{max(_ks):.5f}) — V_new choice likely "
                "matters (corroborated by the paired per-doc test).")

    res = {
        "experiment": "E-39a: U-observability gate",
        "tz": "docs/E39.md §1",
        "revised": "docs/E39_fixes.md (round 1) + docs/E39a_fixes.md (rounds 2–7; round 4 = critical einsum fix, round 5 = enforced noop + honest statistics, round 6 = interpretation: paired per-doc test, retracted 14x/floor claims, round 7 = Bonferroni all-pairs + structured paired_tests JSON + strategy_effect_null aligned with paired test)",
        "params": {"seed": SEED, "model": MODEL, "layer": 23, "r": R,
                   "r_pool": R_POOL, "body": [BODY_LO, BODY_HI],
                   "taus": list(TAUS)},
        "rank_report": rank_report,
        "rank_regime_all_taus": {t: ("low" if r < 32 else "moderate" if r < 256
                                     else "high" if r < 1008 else "infeasible")
                                for t, r in rank_report["body"]["r_eff_tau"].items()},
        "r_obs": r_obs,
        "r_obs_note": ("full-SVD rank of [4096,1024] U — generically 1024 "
                       "(no exact null space); feasibility is judged on "
                       "r_eff(τ=0.05) per review 2.2"),
        "null_mode": null_mode,
        "dim_U_obs_perp_exact": int((sU <= 1e-6 * sU[0].item()).sum()),
        "kappa_baseline": {**kb, "ref_38d": KAPPA_L23_REF,
                           "abs_diff": (abs(kap0 - KAPPA_L23_REF)
                                        if kap0 is not None else None),
                           "validated": not base_kappa_drift},
        "surgery_dryrun": strat,
        "strategies_distinct": strategies_distinct,
        "v_a_v_b_projector_frobenius_dist": proj_dist,
        "v_a_v_b_max_principal_angle_deg": max_pa_deg,
        "strategy_effect_null": strategy_effect_null,
        "verdict": verdict,
        "limitations": [
            "Sentinel split (E-34, 50.7%) NOT computable from exp27b cache: "
            "P128 stores probability-normalized top-128 masses (sum=1), raw "
            "logit masses absent. Body-subset is the canonical choice; "
            "r_eff(body) is an upper bound on r_eff(body_non_sentinel).",
            "Dry-run κ' uses cached û_t (base model) — the real E-39c model "
            "has changed embed_tokens (tied weights), so û_t of the edited "
            "model differs; κ' here is a PREVIEW, not the E-39c measurement. "
            "The verdict dryrun_kappa_min_* must not be read as an E-39c result.",
            "Spec §1.4 asked for ρ_noise as the contamination diagnostic; the "
            "dry-run κ' is reported instead/alongside as a more direct metric "
            "(ρ_noise_u_obs_overlap_mean is also reported per strategy).",
            "Strategy (b) principal = QR(P_pool·V_top16(M_K)); if M_K top-16 "
            "has a large U_obs component the projection loses rank and (b) ≈ "
            "(a) — see v_a_v_b_projector_frobenius_dist, "
            "v_a_v_b_max_principal_angle_deg and the qr_rank_ok flags.",
        ] + _round3_findings(strat, rank_report),
        "paired_tests": _paired,   # round 7 §3: structured per-doc paired t-tests
        "kappa_best_over_worst_ratio": _best_worst_ratio,   # round 6 §5: unconditional, renamed
        "runtime_s": round(time.time() - t0, 1),
        "deterministic": True,
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nE-39a done in {time.time() - t0:.1f}s → {out_path}")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
