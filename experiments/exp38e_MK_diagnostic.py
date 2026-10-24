#!/usr/bin/env python
"""E-38e — M_K diagnostic: w = α û_t + β v_⊥ (docs/E38.md).

Gate for E-38f. Tests whether the uniform direction 1/√K (token space)
lies on the blind direction û_t (strong-unification test, lemma M-1) or
is orthogonal to it.

Setup (L23):
    W_lm = _resolve_lm_head_weight(model)[token_set]   [128, 1024]
    g    = _lm_module(model).norm.weight + 1           (RMSNorm 1+w)
    M_K  = build_M_K(W_lm, token_set, g)              [128, 1024]
    w    = M_K^T @ 1/√K                                [1024]
    per (doc, t): (α_t, β_t, v_⊥_t) = decompose_w_vs_uhat(w, û_t)
    û_t = h_t / ‖h_t‖ (clamp_min 1e-12), h_t from exp27b cache field H.

Consistency (lemma M-1, E-12a/E-33r, tolerance 15%):
    |β_t|  vs  σ_1(M_K) := svdvals(M_K)[0]
    Derivation ([L-gauge-J], docs/E38_fixes.md part-6 §1 HIGH-1): J = c·M_K·
    (I − ûû^T) with c = √D/r_t ⇒ ‖J^T(1/√K)‖ = c·β = σ_g(J)·cos(u_g,1/√K)
    and σ_g(J) ≈ c·σ_1(M_K). The OLD check normalized c·β by σ_1(M_K),
    whose expected value is c·cos ≈ 0.2–0.4 — ratio ≈ 1 was PHYSICALLY
    IMPOSSIBLE (the old consistency_redflag was a false positive of the
    check, not of the hypothesis). The correct local ratio
    (c·β) / (c·σ_1(M_K)) = β/σ_1(M_K) ≈ cos(u_g, 1/√K) ≈ 0.9, so the
    check is: median β/σ_1(M_K) within 15% of 0.9. The direct
    factorization cross-check β vs ‖J_t^T 1/√K‖ = c·β (ratio = 1/c =
    r_t/√D, set by the hidden-state scale — see r_t_expected, NOT 1)
    closes the loop without the alignment factor.

Pre-registered outcomes:
    gate_failed        — median |α|/|β| > 10 (blind direction ⊥ uniform;
                         E-38f skipped)
    mixed              — median |α|/|β| ≈ 1 (E-38f runs)
    gauge_blind_sep    — median |α|/|β| << 1 (median < 0.1): uniform
                         almost ⊥ û_t, gauge and blind are distinct objects
    consistency_consistent_with_E33r — median |β_t|/σ_1(M_K) within 15% of
                         0.9 (≈ cos(u_g, 1/√K) ≈ 0.9; upper bound 1). This
                         is NOT exact agreement: the ratio = cos ≤ 1 by
                         construction. (Supersedes the old check that
                         normalized c·β by σ_1(M_K) and expected ≈ 1 —
                         impossible, docs/E38_fixes.md part-6 §1 HIGH-1.)
    consistency_redflag— otherwise (check M_K, g, token_set); NOTE:
                         redflag is one-sided — it also fires when
                         v_1(M_K) ∥ û (σ_g(J) << c·σ_1(M_K)), not only
                         for a bad model
    degenerate         — nan_frac > 0.5 (mass ‖h‖ < 1e-6; all medians
                         undefined, fail loud)

Model loaded on CPU, float32. Deterministic: seed=0 recorded.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _e38_cache import S, D, load_layer  # noqa: E402
from prism.cache import bootstrap_ci  # noqa: E402
from prism.interventions import _resolve_lm_head_weight  # noqa: E402
from prism.jacobian import _lm_module  # noqa: E402
from prism.jstruct import build_M_K  # noqa: E402

SEED = 0
B = 1000
MODEL = "Qwen/Qwen3.5-0.8B"
TOL_CONSISTENCY = 0.15

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38e_MK_diagnostic.json")


def _json_default(o):
    """P0 (docs/E38_fixes2.md): the old `default=float` silently converted
    np.bool_ -> 1.0 and leaned on an np.float32-NaN str() coincidence.
    Strict version: raise on np.bool_ (unwrapped numpy scalar = a code
    bug), convert numpy numerics explicitly, NaN/Inf -> strings (valid
    RFC-8259)."""
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


def main():
    t0 = time.time()
    torch.manual_seed(SEED)

    # --- model-side quantities (once) ---
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    ck23 = load_layer(23, want=("J", "H"))  # only J and H are used; skip P
    token_set = ck23["U128"]
    assert len(token_set) == 128, f"token_set len {len(token_set)} != 128"

    W_lm_full = _resolve_lm_head_weight(model)
    # HIGH-3: the cache's U128 must index the SAME model's vocab. The old
    # `ck23.get("V")` guard was dead code (load_layer never returns "V"),
    # so only the range check below remains; a cache from a different
    # checkpoint with the same V would still pass (flagged in meta).
    V_model = W_lm_full.shape[0]
    assert 0 <= min(token_set) and max(token_set) < V_model
    K = len(token_set)
    g = _lm_module(model).norm.weight + 1             # [1024]
    # build_M_K indexes W_lm by token_set internally — pass the FULL weight.
    M_K = build_M_K(W_lm_full, token_set, g)           # [K, 1024]
    with torch.no_grad():
        w = (M_K.T @ (torch.ones(K) / np.sqrt(float(K))))
        # σ_1(M_K) — NOT the gauge σ_g(J); σ_g(J) ≈ c·σ_1(M_K)
        sigma_1_MK = float(torch.linalg.svdvals(M_K)[0])
        w = w.detach()
    w_norm = float(w.norm())
    del W_lm_full, model
    print(f"M_K built: σ_1(M_K)={sigma_1_MK:.4f}, ‖w‖={w_norm:.4f}")

    # --- per (doc, t) decomposition ---
    H = ck23["H"]                                     # [N, S, D]
    J23 = ck23["J"]                                   # raw J [N, S, K, D]
    N, Ss, Dd = H.shape
    norms = H.norm(dim=-1)                            # [N, S]
    u = H / norms.clamp_min(1e-12)[:, :, None]
    sqrtD = np.sqrt(Dd)
    # ‖J_t^T (1/√K)‖ — direct factorization cross-check (J at L23 is raw)
    JtU = torch.einsum("ntkd,k->ntd", J23, torch.ones(K) / np.sqrt(float(K)))
    JtU_norm = JtU.norm(dim=-1)                       # [N, S]

    # vectorized rank-1 split: α = w·û, β = ‖w − αû‖ (decompose_w_vs_uhat
    # returns exactly these: β ≥ 0 norm, v_⊥ unit)
    w2 = (u @ w).detach()                        # [N, S] = α_t
    beta_vec = (w[None, None, :] - w2[:, :, None] * u).norm(dim=-1).detach()
    # MEDIUM-5: ‖h‖ < 1e-6 → û undefined (clamp_min(1e-12) would give u≈0
    # silently); mark NaN and count (nan_frac in meta).
    valid = norms >= 1e-6
    nan_frac = float(1.0 - valid.float().mean())
    w2 = torch.where(valid, w2, torch.tensor(float("nan")))
    beta_vec = torch.where(valid, beta_vec, torch.tensor(float("nan")))

    ratio = w2.abs() / beta_vec.clamp_min(1e-12)     # [N, S] |α|/|β| (NaN-safe)
    beta_scaled = (sqrtD * beta_vec / norms.clamp_min(1e-12)).detach()
    beta_vs_JtU = beta_vec / JtU_norm.clamp_min(1e-12)
    alpha_abs = w2.abs().numpy()
    beta_abs = beta_vec.numpy()
    ratios_docs = [float(np.nanmedian(ratio[i].numpy())) for i in range(N)]
    pos0_valid = valid[:, 0]                       # MEDIUM-4: attention-sink guard
    pos0_vals = ratio[:, 0][pos0_valid]
    # P0-2: guard — if every doc has ‖h_0‖ < 1e-6 (strong sink) the slice
    # is empty: .median() raises on new torch, silently NaNs on old.
    pos0_ratio = float(pos0_vals.median()) if pos0_vals.numel() else float("nan")
    # P2: cap — β ≈ 0 at pos 0 gives ratio ~ 1e9 (meaningless); cap + flag.
    POS0_CAP = 1e6
    pos0_ratio_capped = bool(np.isfinite(pos0_ratio) and pos0_ratio > POS0_CAP)
    if pos0_ratio_capped:
        pos0_ratio = float(POS0_CAP)
    pos0_degenerate_frac = float((~pos0_valid).float().mean())

    ratio_median = float(np.nanmedian(ratios_docs))
    # HIGH-1 fix: the LOCAL ratio (c·β)/(c·σ_1(M_K)) = β/σ_1(M_K)
    # ≈ cos(u_g, 1/√K) ≈ 0.9 (NOT ≈ 1 — see header).
    cons_ratio = float(np.nanmedian(beta_vec.numpy()) / sigma_1_MK)
    jtu_ratio = float(np.nanmedian(beta_vs_JtU.numpy()))
    bs = beta_scaled.numpy()

    gate = ("gate_failed" if ratio_median > 10
            else ("gauge_blind_sep" if ratio_median < 0.1 else "mixed"))
    # P1 (docs/E38_fixes2.md §8): mass degeneracy would make every median
    # NaN and silently fall into "mixed"/"redflag"; fail loud instead.
    if nan_frac > 0.5:
        gate = "degenerate"
    # consistency == cos(u_g, 1/√K) ≈ 0.9 ± 0.15 — the alignment factor is
    # ≤ 1 by construction (docs/E38_fixes.md part-6 §1 HIGH-1).
    cons = ("consistency_consistent_with_E33r" if abs(cons_ratio - 0.9) <= TOL_CONSISTENCY
            else "consistency_redflag")
    # P1-7 (jacobian.py M-1 cross-check): if û is the INPUT to L23, the
    # factorization J = c·M_K·(I−ûû^T) kills û ⇒ ‖J·û‖/‖J‖ << 1; a value
    # ~ 1 would mean an off-by-one hidden-state convention.
    # (J @ u) is [N,S,K] — einsum, because plain matmul [N,S,K,D]x[N,S,D]
    # right-aligns the batch dims and fails (S vs N mismatch). Both
    # norms are [N,S].
    resid = (torch.einsum("ntkd,ntd->ntk", J23, u).norm(dim=-1)
             / J23.norm(dim=(-1, -2)).clamp_min(1e-12))
    j_uhat_resid = float(resid[valid].median())
    per_layer = {
        "L23": {
            "sigma_1_MK": sigma_1_MK, "w_norm": w_norm,
            "ratio_alpha_beta": {
                "median_over_docs": ratio_median,
                "ci": list(bootstrap_ci(np.array(ratios_docs), B=B, seed=SEED)),
                "ci_note": "doc-level bootstrap over the 32 per-doc medians "
                           "(P0, docs/E38_fixes2.md); the old flat "
                           "percentile over N·S positions overstated "
                           "precision ~√(S)×",
                "per_doc": ratios_docs},
            "alpha": {"median_abs": float(np.nanmedian(alpha_abs)),
                      "mean_abs": float(np.nanmean(alpha_abs))},
            "beta": {"median": float(np.nanmedian(beta_abs)),
                     "mean": float(np.nanmean(beta_abs))},
            "consistency_sigma_1_MK": {
                "median_scaled_beta": float(np.nanmedian(beta_scaled.numpy())),
                "over_sigma_g_OLD": float(np.nanmedian(beta_scaled.numpy()) / sigma_1_MK),
                "over_sigma_g_OLD_note": "c·β/σ_1(M_K) ≈ c·cos ≈ 0.2–0.4 — "
                                         "the OLD normalization, kept for "
                                         "traceability; NOT ≈ 1 (false-"
                                         "positive source of the old "
                                         "redflag)",
                "over_sigma_g_J": cons_ratio,
                "expected": 0.9,
                "tolerance": TOL_CONSISTENCY},
            "consistency_factorization": {
                "median_beta_over_JtU": float(np.nanmedian(beta_vs_JtU.numpy())),
                "r_t_expected": float(np.nanmedian(norms.numpy()) / sqrtD),
                "note": "β_t / ‖J_t^T 1/√K‖ = 1/c = r_t/√D (NOT 1) under "
                        "exact J = M_K(I − ûû^T): ‖J^T(1/√K)‖ = c·β with "
                        "c = √D/r_t. The expected value is set by the "
                        "hidden-state scale — compare with r_t_expected = "
                        "median(‖h‖)/√D (typical ~1–4, not a fixed 4); "
                        "agreement with r_t_expected CONFIRMS the exact "
                        "factorization, it is not a redflag (docs/"
                        "E38_fixes.md part-6 §1 HIGH-2)"},
            "pos0_ratio": pos0_ratio,
            "pos0_ratio_capped": pos0_ratio_capped,
            "pos0_degenerate_frac": pos0_degenerate_frac,
            "nan_frac": nan_frac,
            "j_uhat_resid_median": j_uhat_resid,
            "j_uhat_resid_note": "median ‖J·û‖/‖J‖: << 1 confirms û is the "
                                 "INPUT to L23 (projector kills it); ~ 1 "
                                 "would mean an off-by-one hidden-state "
                                 "convention (jacobian.py M-1)"},
    }
    del M_K, H, J23, ck23   # P2: release ~2.2 GB before the (cheap) dump
    out = {
        "params": {"seed": SEED, "B": B, "model": MODEL, "layer": 23,
                   "N_docs": N,
                   "S": S, "D": D, "K": 128, "tolerance": TOL_CONSISTENCY,
                   "token_set": "exp27b U128",
                   "cache": "exp27b_jcache_L23.pt"},
        "metrics": per_layer,
        "outcomes": {"gate": gate, "consistency": cons,
                     "gate_failed": gate == "gate_failed"},
        "meta": {"note": "cos(u_g, 1/√K) alignment factor recorded via the "
                         "direct factorization cross-check (β vs ‖J^T 1/√K‖ "
                         "= c·β, ratio = r_t/√D) and via β/σ_1(M_K) ≈ cos "
                         "— the alignment factor is ≤ 1 and unknown a "
                         "priori",
                 "consistency_semantics": "consistency_consistent_with_E33r "
                                          "means median β/σ_1(M_K) ≈ 0.9 "
                                          "± 15% (≈ cos(u_g, 1/√K)), NOT "
                                          "exact agreement — the old check "
                                          "(c·β vs σ_1(M_K), expected ≈ 1) "
                                          "was impossible by construction "
                                          "and its redflag was a false "
                                          "positive of the check",
                 "gate_rule": ">10 → E-38f skipped; <0.1 → gauge_blind_sep; "
                              "else mixed → E-38f runs; degenerate → "
                              "nan_frac > 0.5 (medians undefined)",
                 "redflag_semantics": "consistency_redflag is one-sided: "
                                      "it also fires when v_1(M_K) lies "
                                      "along û (σ_g(J) << c·σ_1(M_K) ⇒ "
                                      "cons_ratio << cos even for a good "
                                      "model). redflag = 'the check did "
                                      "not confirm', not 'model is bad' "
                                      "(docs/E38_fixes2.md §5)",
                 "vocab_guard": "the old ck23.get('V') guard was dead "
                                "code; only the token-range check against "
                                "the live model's V remains — a cache "
                                "from a different checkpoint with the "
                                "same V would pass (docs/E38_fixes2.md "
                                "§3)",
                 "token_set_range": [int(min(token_set)), int(max(token_set))]},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=_json_default)
    print(f"ratio |α|/|β| median-over-docs = {ratio_median:.4f} -> {gate}")
    print(f"consistency: |β|/σ_1(M_K) = {cons_ratio:.4f} -> {cons}; "
          f"β/‖J^T u‖ = {jtu_ratio:.4f} (r_t_expected = "
          f"{float(np.nanmedian(norms.numpy()) / sqrtD):.2f})")
    print(f"JSON -> {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
