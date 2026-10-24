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

SECONDARY (a) — dual sparsity/tail correlation, computed in BOTH bases
    (P0 fix, docs/E38_fixes2.md: the OLD version completed our 128
    right singular vectors to a full R^1024 basis with 896 null-space
    vectors; for a typical h the null modes carry ≈ 896/1024 ≈ 0.875
    of the energy, so S_our ≈ log(1024) and frac_tail_our ≈ 0.875 —
    near-constants, rho_our ≈ 0 for ANY completion, and the
    'completion sensitivity' check was vacuously false):
    OUR:   full SVD of J_our[i,t] (128x1024) → 128 right singular
           vectors (the REAL modes only, no null completion).
           c_k(t) = ⟨h_t, v_k⟩; S_our(t) = −Σ p log p, p_k = c_k²/Σc²;
           frac_tail_our(t) = energy in our tail modes (bottom 10% of
           the 128 σ — 12 modes). rho_our = median over docs of
           Spearman_t(S_our(t), frac_tail_our(t)).
    LENS:  the same construction with the lens' full 1024-mode basis:
           rho_lens.
    Reading: if "our tail ≈ lens tail" is true, both rho should be
    high; one high + one low ⇒ the corresponding operator lacks the
    sparsity/tail structure the hypothesis refers to. CAVEAT: the two
    metrics are defined on different mode spaces (128 vs 1024) — they
    are semantically comparable ('sparsity in the spectral basis')
    but NOT by absolute magnitude.

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

    POST-HOC RELABEL (P1, docs/E38_fixes2.md, fixed explicitly per the
    E-38a precedent — a new label must not appear silently): the TZ's
    0.3 threshold is above the random baseline r_lens/D ≈ 0.1, so
    'overlap < 0.3' cannot be read as 'disjoint' (that is random
    INDEPENDENCE). The implemented labels are:
        > 0.9             → our_tail_within_lens_tail
        < 0.5·baseline    → tails_anticorrelated
        < 3·baseline      → at_random   (≈ the TZ's 0.3 zone)
        else              → partial_overlap (wider: 3·baseline..0.9)

Deterministic: seed=0 (bootstrap); SVD via LAPACK — deterministic for
fixed inputs and platform. (The old np.random.seed call was a no-op:
all stochasticity goes through bootstrap_ci's own default_rng.)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
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
    """From squared coefficients c2 [n_modes] (1-D — the assert below
    enforces it; the old docstring's batched claim was stale) and a
    tail cutoff index (modes with index > tail_idx), return
    (entropy, frac_tail)."""
    assert c2.dim() == 1, f"entropy_frac_tail expects 1-D c2, got {c2.dim()}D"
    tot = c2.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    p = c2 / tot
    s = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)
    ft = c2[..., tail_idx + 1:].sum(dim=-1) / tot.squeeze(-1)
    return s, ft


def main():
    t0 = time.time()

    # --- lens: validity gate + right singular vectors ---
    lens = torch.load(LENS, map_location="cpu", weights_only=False)
    # LOW-10: loud index/shape checks (key-based vs list-based layer
    # indexing is not distinguishable otherwise).
    # P1 (docs/E38_fixes2.md): normalize dict/list/ndarray via
    # torch.as_tensor — a bare .float() on an np.ndarray fails with
    # an unhelpful AttributeError.
    J_l = None
    if hasattr(lens["J"], "keys"):
        assert T_LAYER in lens["J"], f"lens J keys {sorted(lens['J'].keys())}, T_LAYER={T_LAYER}"
        J_l = torch.as_tensor(lens["J"][T_LAYER], dtype=torch.float32)
    else:
        assert len(lens["J"]) > T_LAYER, f"lens J len {len(lens['J'])} <= T_LAYER {T_LAYER}"
        J_l = torch.as_tensor(lens["J"][T_LAYER], dtype=torch.float32)
    assert J_l.shape == (D, D), f"lens L{T_LAYER} shape {tuple(J_l.shape)} != ({D},{D})"
    sigma_l = torch.linalg.svdvals(J_l)
    s1 = float(sigma_l[0])
    jmin_i = float(torch.linalg.norm(J_l - torch.eye(D)))
    _, _, Vh_l = torch.linalg.svd(J_l, full_matrices=False)
    V_lens = Vh_l.T.contiguous()             # [D, D] right vecs, σ-desc
    # LOW-9: guard the degenerate-lens division.
    s1_ratio = s1 / max(float(sigma_l.median()), 1e-30)
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
            # OUR basis: the 128 real right singular vectors ONLY.
            # P0 (docs/E38_fixes2.md): the old null-space completion
            # (896 arbitrary modes) made S_our/frac_tail_our ≈ constants
            # (≈ log(1024) / ≈ 0.875), so rho_our ≈ 0 for ANY completion
            # and the sensitivity check was vacuous.
            _, _, Vh_o = torch.linalg.svd(J_our[i, t], full_matrices=False)
            V_our = Vh_o.T.contiguous()                    # [D, K]
            c2_o = (H[i, t] @ V_our) ** 2                  # [K]
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

    # MEDIUM-4: DOC-level bootstrap for overlap (the unit of independence
    # is the doc; positions are correlated — same fix as E-38b/E-38f).
    ov_per_doc = np.nanmedian(overlap_t, axis=1)           # [N]
    ov_valid = ov_per_doc[~np.isnan(ov_per_doc)]
    ov_med = float(np.nanmedian(ov_valid))
    # MEDIUM-7: bootstrap_ci needs >= 2 values; guard all-NaN / 1-value.
    ov_ci = (list(bootstrap_ci(ov_valid, B=B, seed=SEED)) if ov_valid.size >= 2
             else [float("nan"), float("nan")])
    rho_our_arr = np.array(rho_our_per_doc)
    rho_lens_arr = np.array(rho_lens_per_doc)
    rho_our_med = float(np.nanmedian(rho_our_arr))
    rho_lens_med = float(np.nanmedian(rho_lens_arr))
    valid_rho_our = rho_our_arr[~np.isnan(rho_our_arr)]
    valid_rho_lens = rho_lens_arr[~np.isnan(rho_lens_arr)]
    rho_our_ci = (list(bootstrap_ci(valid_rho_our, B=B, seed=SEED))
                  if valid_rho_our.size >= 2 else [float("nan"), float("nan")])
    rho_lens_ci = (list(bootstrap_ci(valid_rho_lens, B=B, seed=SEED))
                   if valid_rho_lens.size >= 2 else [float("nan"), float("nan")])

    # --- pre-registered verdict (primary = overlap, TZ-rewritten) ---
    # MEDIUM-3: random baseline for two independent random subspaces of
    # dims r_our, r_lens in R^D: E[overlap] = r_lens/D ≈ 0.1. "Disjoint"
    # at overlap ≈ 0.1 is wrong: that is random independence, not
    # orthogonality. Reclassify relative to the baseline.
    random_baseline = r_lens / D
    if ov_med > OVERLAP_HIGH:
        verdict = "our_tail_within_lens_tail"
    elif ov_med < 0.5 * random_baseline:
        verdict = "tails_anticorrelated"
    elif ov_med < 3 * random_baseline:
        verdict = "at_random"
    else:
        verdict = "partial_overlap"
    if not lens_valid:
        verdict += ",lens_validity_suspect"
    # HIGH-2: the overlap verdict is a REWRITE of the TZ pre-registration
    # (ρ-based). Report BOTH so the status is traceable to the TZ.
    tz_preregistered_outcome = (
        "jlens_not_tail_modes" if rho_our_med < 0.3 and rho_lens_med < 0.3
        else ("symmetric" if min(rho_our_med, rho_lens_med) > 0.7 else "partial"))

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
                "ci_note": "doc-level: per-doc median, bootstrap over 32 docs",
                "random_baseline": random_baseline,
                "random_baseline_note": "E[overlap] = r_lens/D for two "
                    "independent random subspaces; overlap ~ 0.1 is "
                    "RANDOM INDEPENDENCE, not disjointness",
                "profile_every_8": [float(np.nanmedian(overlap_t[:, t]))
                                    for t in range(0, Ss, 8)],
            },
            "rho_our_S_frac_tail": {
                "median": rho_our_med, "ci": rho_our_ci,
                "per_doc": rho_our_per_doc,
                "basis_note": "128 REAL right singular modes only (no null "
                    "completion — P0, docs/E38_fixes2.md: the old 896-mode "
                    "completion dominated the energy and made this metric "
                    "a near-constant). NOT magnitude-comparable with "
                    "rho_lens (1024 modes).",
            },
            "rho_lens_S_frac_tail": {
                "median": rho_lens_med, "ci": rho_lens_ci,
                "per_doc": rho_lens_per_doc},
            "S_our_entropy": {"median": float(np.nanmedian(S_our_med))},
            "frac_tail_our": {"median": float(np.nanmedian(frac_our_med))},
        },
        "outcomes": {
            "verdict": verdict,
            "rewritten_primary_outcome": verdict,
            "tz_preregistered_outcome": tz_preregistered_outcome,
            "outcome_note": "the TZ pre-registration was ρ-based (ρ>0.7 "
                "symmetric / ρ<0.3 not_tail_modes); the code rewrite makes "
                "the subspace OVERLAP primary (the old version measured "
                "only lens-internal consistency — not the TZ question). "
                "Both outcomes are reported; do not mix them in the "
                "paper (docs/E38_fixes.md part-8 HIGH-1/2).",
        },
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
            "basis_completion": "our 128 right singular vectors (REAL "
                    "modes only; the old scipy null-space completion of "
                    "896 arbitrary modes made S_our/frac_tail_our near-"
                    "constants and rho_our ≈ 0 for any completion — P0, "
                    "docs/E38_fixes2.md). A degenerate σ in the LENS also "
                    "makes V_lens_tail arbitrary within the degenerate "
                    "subspace — the validity gate flags suspect lenses, "
                    "but does not catch degeneracy (MEDIUM-6).",
            "layer_note": "T_LAYER=22 per TZ, while E-38a–e use L23. "
                    "Paper §4.5 (E-37) marks L22 as the GDN dip layer — "
                    "the least stable for interpretation, and the lens J "
                    "is position-averaged (recurrent structure may be "
                    "lost). 'Symmetry with Anthropic' here compares "
                    "layers that are not directly comparable — read the "
                    "overlap as a cross-layer, cross-language sanity "
                    "check only (P2, docs/E38_fixes2.md).",
            "tail": f"bottom {TAIL_FRAC:.0%} of each spectrum: our modes "
                    f"{tail_o + 1}..{Kk} (of {Kk}), lens modes "
                    f"{tail_l + 1}..{D} (of {D}).",
        },
    }
    with open(LENS, "rb") as _lf:          # LOW-8: closed handle
        sha = hashlib.sha256(_lf.read()).hexdigest()
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
