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
    (√D / r_t) · |β_t|  vs  σ_g := svdvals(M_K)[0]
    (cos(u_g, 1/√K) ≤ 1 is the alignment factor — recorded as an upper
    bound; the direct factorization cross-check β_t vs ‖J_t^T 1/√K‖
    closes the loop without the alignment factor.)

Pre-registered outcomes:
    gate_failed        — median |α|/|β| > 10 (blind direction ⊥ uniform;
                         E-38f skipped)
    mixed              — median |α|/|β| ≈ 1 (E-38f runs)
    gauge_blind_sep    — median |α|/|β| << 1 (median < 0.1): uniform
                         almost ⊥ û_t, gauge and blind are distinct objects
    consistency_ok     — (√D/r_t)|β_t| within 15% of σ_g
    consistency_redflag— otherwise (check M_K, g, token_set)

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
from prism.interventions import _resolve_lm_head_weight  # noqa: E402
from prism.jacobian import _lm_module  # noqa: E402
from prism.jstruct import build_M_K  # noqa: E402

SEED = 0
MODEL = "Qwen/Qwen3.5-0.8B"
TOL_CONSISTENCY = 0.15

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38e_MK_diagnostic.json")


def main():
    t0 = time.time()
    torch.manual_seed(SEED)

    # --- model-side quantities (once) ---
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    ck23 = load_layer(23)
    token_set = ck23["U128"]
    assert len(token_set) == 128, f"token_set len {len(token_set)} != 128"

    W_lm_full = _resolve_lm_head_weight(model)
    g = _lm_module(model).norm.weight + 1             # [1024]
    # build_M_K indexes W_lm by token_set internally — pass the FULL weight.
    M_K = build_M_K(W_lm_full, token_set, g)           # [128, 1024]
    with torch.no_grad():
        w = (M_K.T @ (torch.ones(128) / torch.sqrt(torch.tensor(128.0))))
        sigma_g = float(torch.linalg.svdvals(M_K)[0])
        w = w.detach()
    w_norm = float(w.norm())
    del W_lm_full, model
    print(f"M_K built: σ_g={sigma_g:.4f}, ‖w‖={w_norm:.4f}")

    # --- per (doc, t) decomposition ---
    H = ck23["H"]                                     # [N, S, D]
    J23 = ck23["J"]                                   # raw J [N, S, K, D]
    N, Ss, Dd = H.shape
    norms = H.norm(dim=-1)                            # [N, S]
    u = H / norms.clamp_min(1e-12)[:, :, None]
    sqrtD = np.sqrt(Dd)
    # ‖J_t^T (1/√K)‖ — direct factorization cross-check (J at L23 is raw)
    JtU = torch.einsum("ntkd,k->ntd", J23, torch.ones(128) / np.sqrt(128.0))
    JtU_norm = JtU.norm(dim=-1)                       # [N, S]

    # vectorized rank-1 split: α = w·û, β = ‖w − αû‖ (decompose_w_vs_uhat
    # returns exactly these: β ≥ 0 norm, v_⊥ unit)
    w2 = (u @ w).detach()                        # [N, S] = α_t
    beta_vec = (w[None, None, :] - w2[:, :, None] * u).norm(dim=-1).detach()
    ratio = w2.abs() / beta_vec.clamp_min(1e-12)     # [N, S] |α|/|β|
    beta_scaled = (sqrtD * beta_vec / norms.clamp_min(1e-12)).detach()
    beta_vs_JtU = beta_vec / JtU_norm.clamp_min(1e-12)
    alpha_abs = w2.abs().numpy()
    beta_abs = beta_vec.numpy()
    ratios_docs = [float(ratio[i].median()) for i in range(N)]
    pos0_ratio = float(ratio[:, 0].median())

    ratio_median = float(np.median(ratios_docs))
    cons_ratio = float(np.median(beta_scaled.numpy()) / sigma_g)
    jtu_ratio = float(np.median(beta_vs_JtU.numpy()))
    bs = beta_scaled.numpy()

    gate = ("gate_failed" if ratio_median > 10
            else ("gauge_blind_sep" if ratio_median < 0.1 else "mixed"))
    cons = ("consistency_ok" if abs(cons_ratio - 1.0) <= TOL_CONSISTENCY
            else "consistency_redflag")
    per_layer = {
        "L23": {
            "sigma_g": sigma_g, "w_norm": w_norm,
            "ratio_alpha_beta": {
                "median_over_docs": ratio_median,
                "ci": list(np.percentile(ratios_docs, [2.5, 97.5])),
                "per_doc": ratios_docs},
            "alpha": {"median_abs": float(np.median(alpha_abs)),
                      "mean_abs": float(alpha_abs.mean())},
            "beta": {"median": float(np.median(beta_abs)),
                     "mean": float(beta_abs.mean())},
            "consistency_sigma_g": {
                "median_scaled_beta": float(np.median(beta_scaled.numpy())),
                "over_sigma_g": float(np.median(beta_scaled.numpy()) / sigma_g),
                "tolerance": TOL_CONSISTENCY},
            "consistency_factorization": {
                "median_beta_over_JtU": float(np.median(beta_vs_JtU.numpy())),
                "note": "β_t / ‖J_t^T 1/√K‖; = 1 under exact "
                        "J = M_K(I − û û^T)"},
            "pos0_ratio": pos0_ratio,
        }
    }
    del ck23
    out = {
        "params": {"seed": SEED, "model": MODEL, "layer": 23, "N_docs": N,
                   "S": S, "D": D, "K": 128, "tolerance": TOL_CONSISTENCY,
                   "token_set": "exp27b U128",
                   "cache": "exp27b_jcache_L23.pt"},
        "metrics": per_layer,
        "outcomes": {"gate": gate, "consistency": cons,
                     "gate_failed": gate == "gate_failed"},
        "meta": {"note": "cos(u_g, 1/√K) alignment factor recorded via the "
                         "direct factorization cross-check (β vs ‖J^T 1/√K‖) "
                         "rather than an a-priori σ_g·cos value — the "
                         "alignment factor is ≤ 1 and unknown a priori",
                 "gate_rule": ">10 → E-38f skipped; <0.1 → gauge_blind_sep; "
                              "else mixed → E-38f runs"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"ratio |α|/|β| median-over-docs = {ratio_median:.4f} -> {gate}")
    print(f"consistency: (√D/r_t)|β|/σ_g = {cons_ratio:.4f} -> {cons}; "
          f"β/‖J^T u‖ = {jtu_ratio:.4f}")
    print(f"JSON -> {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
