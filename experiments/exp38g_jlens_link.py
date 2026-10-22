#!/usr/bin/env python
"""E-38g — J↔h dual: tail-subspace overlap + dual sparsity/tail correlation
(docs/E38.md).

Wave 3 external validation via a J-Lens checkpoint for Qwen3.5-0.8B
(Utkarsh736/qwen3.5-0.8b-hindi-jacobian-lens, D=1024, 24 layers, target
L23, J_ell = E[dh_final/dh_ell]).

WHAT THIS EXPERIMENT MEASURES (rewritten per docs/E38_fixes.md part-2 §1,
options (a)+(b); the previous version only measured lens-INTERNAL
consistency — lens activations in the lens' own basis — which is NOT the
question the TZ asks and is retained here only as a secondary metric):

PRIMARY (b) — direct subspace check, the TZ's "symmetry with Anthropic":
    V_lens_tail  : right singular vectors of the lens J_22 [1024,1024]
                   with σ_k in the bottom 10% of the lens spectrum
                   (102 modes of 1024) — a subspace of R^1024.
    V_our_tail(t): right singular vectors of OUR per-position J_t from
                   exp27b (raw [128,1024], 128 singular values) with
                   σ_k in the bottom 10% of OUR spectrum (12 modes of 128)
                   — a subspace of R^1024.
    overlap(t) = ‖V_our_tail(t)^T @ V_lens_tail‖_F² / r_our,  r_our = 12.
    overlap ∈ [0,1]; overlap ≈ 1 ⇒ our per-position tail subspace lies
    WITHIN the lens tail subspace. Median over all (doc, t) + bootstrap
    CI + position profile.

SECONDARY (a) — dual sparsity/tail correlation, computed in BOTH bases:
    OUR:   full SVD of J_our[i,t] (128x1024) → 128 right singular
           vectors; completed to a full orthonormal basis of R^1024 by
           adding the 896 null-space vectors (scipy.linalg.null_space,
           deterministic LAPACK). c_k(t) = ⟨h_t, v_k⟩;
           S_our(t) = −Σ p log p, p_k = c_k²/Σc²;
           frac_tail_our(t) = energy in our tail modes (bottom 10% of
           the 128 σ). rho_our = median over docs of
           Spearman_t(S_our(t), frac_tail_our(t)).
    LENS:  the same construction with the lens' full 1024-mode basis:
           rho_lens. (This is the metric the old version of the
           experiment reported alone.)
    Reading: if "our tail ≈ lens tail" is true, both rho should be
    high; one high + one low ⇒ the corresponding operator lacks the
    sparsity/tail structure the hypothesis refers to.

CAVEATS (recorded in meta):
- The lens was fit on HINDI Wikipedia (finewiki@hi); the exp27b docs are
  English.
- The lens J_22 is a position-AVERAGED full Jacobian [1024,1024]; our
  J_t is per-position with 128 output-token rows [128,1024]. Both act on
  the SAME input space h_t ∈ R^1024 at L22 — that is the space in which
  the tail subspaces are compared (right singular vectors).
- The author flags the lens "experimental/unvalidated". Pre-registered
  validity gate: J_22 spectrum bounded and moderately close to identity
  (σ1 < 4, ‖J−I‖_F < 3). If failed — `lens_validity: suspect`, and the
  results are interpreted as a sidebar only (per TZ stop conditions: do
  not substitute 9B, do not over-claim from an unvalidated artifact).

Pre-registered interpretation (on the PRIMARY overlap metric):
    overlap_median > 0.9 → our_tail_within_lens_tail
                           (subspace symmetry supported, modulo the
                           language caveat above)
    overlap_median < 0.3 → tails_disjoint
    else                  → partial_overlap
    (+ lens_validity_suspect appended if the gate fails)

Deterministic: seed=0 (bootstrap); SVD/null_space via LAPACK —
deterministic for fixed inputs and platform.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.linalg import null_space
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _e38_cache import S, D, load_layer  # noqa: E402
from prism.cache import bootstrap_ci  # noqa: E402

SEED = 0
B = 1000
T_LAYER = 22          # priority layer per TZ
TAIL_FRAC = 0.10
OVERLAP_HIGH = 0.9
OVERLAP_LOW = 0.3

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LENS = os.path.join(REPO, "results", "exp38g_jlens", "lens.pt")
OUT = os.path.join(REPO, "results", "exp38g_jlens_link.json")


def entropy_frac_tail(c2: torch.Tensor, tail_idx: int):
    """From squared coefficients c2 [..., n_modes] and a tail cutoff index
    (modes with index > tail_idx), return (entropy, frac_tail)."""
    tot = c2.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    p = c2 / tot
    s = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    ft = c2[..., tail_idx + 1:].sum(dim=-1) / tot.squeeze(-1)
    return s, ft


def main():
    t0 = time.time()
    np.random.seed(SEED)

    # --- lens: validity gate + right singular vectors ---
    lens = torch.load(LENS, map_location="cpu", weights_only=False)
    J_l = lens["J"][T_LAYER].float()         # [D, D]
    sigma_l = torch.linalg.svdvals(J_l)
    s1 = float(sigma_l[0])
    jmin_i = float(torch.linalg.norm(J_l - torch.eye(D)))
    _, _, Vh_l = torch.linalg.svd(J_l, full_matrices=False)
    V_lens = Vh_l.T.contiguous()             # [D, D] right vecs, σ-desc
    s1_ratio = s1 / float(sigma_l.median())
    lens_valid = bool(jmin_i < 3.0 and s1 < 4.0)
    tail_l = int((1 - TAIL_FRAC) * D)        # 921 → modes 922..1023 (102)
    V_lens_tail = V_lens[:, tail_l + 1:]     # [D, 102]
    r_lens = V_lens_tail.shape[1]
    print(f"Lens J_{T_LAYER}: σ1={s1:.3f} median σ={float(sigma_l.median()):.3f} "
          f"‖J−I‖_F={jmin_i:.3f} s1/median={s1_ratio:.3f} -> "
          f"{'valid' if lens_valid else 'SUSPECT'}")
    del lens

    # --- activations + our per-position J from exp27b ---
    ck = load_layer(T_LAYER)
    H = ck["H"].float()                       # [N, S, D]
    J_our = ck["J"]                           # [N, S, K, D] raw
    N, Ss, Kk, _ = J_our.shape
    del ck

    # --- per (doc, t): dual projection + overlap ---
    tail_o = int((1 - TAIL_FRAC) * Kk)        # 115 → modes 116..127 (12)
    r_our = Kk - tail_o - 1
    overlap_t = np.empty((N, Ss))
    rho_our_per_doc = []
    rho_lens_per_doc = []
    S_our_med = np.full((N, Ss), np.nan)
    frac_our_med = np.full((N, Ss), np.nan)

    for i in range(N):
        S_o = np.empty(Ss)
        Ft_o = np.empty(Ss)
        S_l = np.empty(Ss)
        Ft_l = np.empty(Ss)
        for t in range(Ss):
            # OUR basis: 128 right singular vectors + 896 null vectors.
            _, _, Vh_o = torch.linalg.svd(J_our[i, t], full_matrices=False)
            V_our = Vh_o.T.contiguous()                    # [D, K]
            V_null = torch.from_numpy(
                null_space(V_our.T.numpy())).float()       # [D, D-K]
            V_full = torch.cat([V_our, V_null], dim=1)     # [D, D]
            c2_o = (H[i, t] @ V_full) ** 2
            S_o[t], Ft_o[t] = entropy_frac_tail(c2_o, tail_o)
            # OVERLAP (primary): our tail inside lens tail?
            V_our_tail = V_our[:, tail_o + 1:]             # [D, r_our]
            gram = V_our_tail.T @ V_lens_tail              # [r_our, r_lens]
            overlap_t[i, t] = float((gram ** 2).sum()) / r_our
            # LENS basis (full 1024 modes).
            c2_l = (H[i, t] @ V_lens) ** 2
            S_l[t], Ft_l[t] = entropy_frac_tail(c2_l, tail_l)
        S_our_med[i], frac_our_med[i] = S_o, Ft_o
        r_o, _ = spearmanr(S_o, Ft_o)
        rho_our_per_doc.append(float(r_o) if np.isfinite(r_o) else float("nan"))
        r_l, _ = spearmanr(S_l, Ft_l)
        rho_lens_per_doc.append(float(r_l) if np.isfinite(r_l) else float("nan"))
    del J_our, H

    ov = overlap_t.ravel()
    ov_med = float(np.nanmedian(ov))
    ov_ci = list(bootstrap_ci(ov[~np.isnan(ov)], B=B, seed=SEED))
    rho_our_arr = np.array(rho_our_per_doc)
    rho_lens_arr = np.array(rho_lens_per_doc)
    rho_our_med = float(np.nanmedian(rho_our_arr))
    rho_lens_med = float(np.nanmedian(rho_lens_arr))
    rho_our_ci = list(bootstrap_ci(rho_our_arr[~np.isnan(rho_our_arr)],
                                   B=B, seed=SEED))
    rho_lens_ci = list(bootstrap_ci(rho_lens_arr[~np.isnan(rho_lens_arr)],
                                    B=B, seed=SEED))

    # --- pre-registered verdict (primary = overlap) ---
    if ov_med > OVERLAP_HIGH:
        verdict = "our_tail_within_lens_tail"
    elif ov_med < OVERLAP_LOW:
        verdict = "tails_disjoint"
    else:
        verdict = "partial_overlap"
    if not lens_valid:
        verdict += ",lens_validity_suspect"

    out = {
        "params": {
            "seed": SEED, "B": B, "layer": T_LAYER, "N_docs": N, "S": Ss,
            "tail_frac": TAIL_FRAC,
            "lens": "Utkarsh736/qwen3.5-0.8b-hindi-jacobian-lens",
            "lens_sha256": None,
            "cache": f"exp27b_jcache_L{T_LAYER}.pt",
            "r_our": r_our, "r_lens": r_lens,
            "J_our": "raw per-position [K=128, D=1024] (E-28 convention)",
        },
        "lens_validity": {
            "sigma1": s1, "sigma_median": float(sigma_l.median()),
            "norm_J_minus_I": jmin_i, "s1_over_median": s1_ratio,
            "valid": lens_valid,
        },
        "metrics": {
            "primary_overlap_our_tail_in_lens_tail": {
                "median": ov_med, "ci": ov_ci,
                "profile_every_8": [float(np.nanmedian(overlap_t[:, t]))
                                    for t in range(0, Ss, 8)],
            },
            "rho_our_S_frac_tail": {
                "median": rho_our_med, "ci": rho_our_ci,
                "per_doc": rho_our_per_doc},
            "rho_lens_S_frac_tail": {
                "median": rho_lens_med, "ci": rho_lens_ci,
                "per_doc": rho_lens_per_doc},
            "S_our_entropy": {"median": float(np.nanmedian(S_our_med))},
            "frac_tail_our": {"median": float(np.nanmedian(frac_our_med))},
        },
        "outcomes": {"verdict": verdict},
        "meta": {
            "what_this_test_is": "PRIMARY (b): direct subspace check — "
                    "per-position tail subspace of OUR J_t (bottom 10% of "
                    "its 128 σ, right singular vectors, R^1024) vs tail "
                    "subspace of the LENS J_22 (bottom 10% of 1024 σ): "
                    "overlap = ‖V_our_tail^T V_lens_tail‖_F²/r_our. This "
                    "is the TZ's 'symmetry with Anthropic' claim. "
                    "SECONDARY (a): dual sparsity/tail correlation in "
                    "both bases (rho_our vs rho_lens; the old version "
                    "reported only the lens-internal rho_lens, which is "
                    "NOT the TZ question). docs/E38_fixes.md part-2 §1.",
            "note": "Lens fit on HINDI (finewiki@hi), docs English. Lens "
                    "J_22 is a position-averaged [1024,1024] Jacobian; "
                    "our J_t is per-position [128,1024]. Both act on the "
                    "same h_t ∈ R^1024 at L22 — the right-singular "
                    "comparison is well-defined there. Author flags the "
                    "lens 'experimental/unvalidated'; validity gate = "
                    "bounded spectrum of J_22. Per TZ: do NOT substitute "
                    "9B results (different D).",
            "basis_completion": "our 128 right singular vectors completed "
                    "to a full R^1024 basis by scipy.linalg.null_space "
                    "(896 deterministic LAPACK null vectors), so "
                    "S_our(t) is over the same 1024-mode p_k as rho_lens.",
            "tail": f"bottom {TAIL_FRAC:.0%} of each spectrum: our modes "
                    f"{tail_o + 1}..{Kk} (of {Kk}), lens modes "
                    f"{tail_l + 1}..{D} (of {D}).",
        },
    }
    sha = hashlib.sha256(open(LENS, "rb").read()).hexdigest()
    out["params"]["lens_sha256"] = sha
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"overlap (our tail in lens tail) median={ov_med:.4f} CI={ov_ci}")
    print(f"rho_our median={rho_our_med:.4f} CI={rho_our_ci} | "
          f"rho_lens median={rho_lens_med:.4f} CI={rho_lens_ci}")
    print(f"-> {verdict}")
    print(f"JSON -> {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
