#!/usr/bin/env python
"""E-38g — J↔h dual: projection of activations on tail modes (docs/E38.md).

External validation via a J-Lens checkpoint for Qwen3.5-0.8B
(Utkarsh736/qwen3.5-0.8b-hindi-jacobian-lens, D=1024, 24 layers, target
L23, J_ℓ = E[∂h_final/∂h_ℓ]).

CAVEATS (recorded in meta, per TZ):
- The lens was fit on HINDI Wikipedia (finewiki@hi); the exp27b docs are
  English. The test is structural (projection sparsity vs tail energy),
  not a token readout, so it is more robust to language mismatch than the
  degenerate readouts the author reports — but the caveat stands.
- The author flags the lens "experimental / unvalidated". Pre-registered
  validity gate: J_22 (last source layer, one step from target) should be
  near-identity in the SPECTRAL SENSE (top-16 subspace overlap with I high,
  spectrum bounded). If J_22 is degenerate (σ_1 >> 1 by an order of
  magnitude, or ‖J_22 − I‖_F large), the structural test is reported with
  a `lens_validity: suspect` flag and interpreted as NOT external
  validation (per TZ stop conditions — we do not substitute 9B, and we do
  not over-claim from an unvalidated artifact).

Per (doc, t) at L22 (h_t from exp27b, J_22 from the lens):
    SVD J_22 = U Σ V^T → {v_k} right singular vectors [1024], k = 1..1024
    c_k(t) = ⟨h_t, v_k⟩
    p_k(t) = c_k(t)² / Σ_j c_j(t)² ;  S(t) = −Σ p_k log p_k
    tail = k with σ_k in the bottom 10% of the spectrum (k = 923..1024)
    frac_tail(t) = Σ_{k ∈ tail} c_k(t)² / Σ_k c_k(t)²
    ρ(S, frac_tail) per doc (Spearman) → median over docs + bootstrap CI.

Pre-registered interpretation:
    ρ > 0.7  — J-Space = tail modes, symmetry established
    ρ < 0.3  — J-Space ≠ tail modes; J-Lens is a sidebar, not validation
    else     — partial overlap (requires further analysis)

Deterministic: seed=0 (bootstrap).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _e38_cache import S, load_layer  # noqa: E402
from prism.cache import bootstrap_ci  # noqa: E402

SEED = 0
B = 1000
D = 1024
T_LAYER = 22          # priority layer per TZ
TAIL_FRAC = 0.10

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LENS = os.path.join(REPO, "results", "exp38g_jlens", "lens.pt")
OUT = os.path.join(REPO, "results", "exp38g_jlens_link.json")


def main():
    t0 = time.time()
    np.random.seed(SEED)

    # --- lens: validity gate ---
    lens = torch.load(LENS, map_location="cpu", weights_only=False)
    J_l = lens["J"][T_LAYER].float()         # [D, D]
    I = torch.eye(D)
    sigma = torch.linalg.svdvals(J_l)
    s1 = float(sigma[0])
    jmin_i = float(torch.linalg.norm(J_l - I))
    # spectral overlap of J_l's top-16 right subspace with I's — trivially
    # 1.0; the informative check is the SPECTRUM: a valid transport has
    # bounded singular values and moderate deviation from identity.
    U, _, Vt = torch.linalg.svd(J_l, full_matrices=False)
    V = Vt.T.contiguous()                          # [D, D] right vecs
    s1_ratio = s1 / float(sigma.median())
    lens_valid = bool(jmin_i < 3.0 and s1 < 4.0)
    print(f"Lens J_{T_LAYER}: σ1={s1:.3f} median σ={float(sigma.median()):.3f} "
          f"‖J−I‖_F={jmin_i:.3f} s1/median={s1_ratio:.3f} -> "
          f"{'valid' if lens_valid else 'SUSPECT'}")

    # --- activations L22 from exp27b ---
    ck = load_layer(T_LAYER)
    H = ck["H"]                                     # [N, S, D]
    N, Ss, _ = H.shape
    del ck
    Hc = H.float()
    c_all = Hc @ V                                  # [N, S, D]
    c2 = c_all ** 2
    tot = c2.sum(dim=-1, keepdim=True).clamp_min(1e-30)   # [N, S, 1]
    p = c2 / tot
    S_ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=-1)  # [N, S]
    tail = int((1 - TAIL_FRAC) * D)
    frac_tail = (c2[:, :, tail:].sum(dim=-1) / tot.squeeze(-1))  # [N, S]

    from scipy.stats import spearmanr
    per_doc_rho = []
    for i in range(N):
        r_, _ = spearmanr(S_ent[i].numpy(), frac_tail[i].numpy())
        per_doc_rho.append(float(r_) if np.isfinite(r_) else float("nan"))
    rho_arr = np.array(per_doc_rho)
    rho_med = float(np.nanmedian(rho_arr))
    rho_ci = list(bootstrap_ci(rho_arr[~np.isnan(rho_arr)], B=B, seed=SEED))

    if rho_med > 0.7:
        verdict = "jlens_equals_tail_modes"
    elif rho_med < 0.3:
        verdict = "jlens_not_tail_modes"
    else:
        verdict = "partial_overlap"
    if not lens_valid:
        verdict += ",lens_validity_suspect"

    out = {
        "params": {"seed": SEED, "B": B, "layer": T_LAYER, "N_docs": N,
                   "S": Ss, "tail_frac": TAIL_FRAC,
                   "lens": "Utkarsh736/qwen3.5-0.8b-hindi-jacobian-lens",
                   "lens_sha256": None,
                   "cache": f"exp27b_jcache_L{T_LAYER}.pt"},
        "lens_validity": {
            "sigma1": s1, "sigma_median": float(sigma.median()),
            "norm_J_minus_I": jmin_i, "s1_over_median": s1_ratio,
            "valid": lens_valid,
        },
        "metrics": {
            "rho_S_frac_tail": {"median": rho_med, "ci": rho_ci,
                                "per_doc": per_doc_rho},
            "S_entropy": {"median": float(S_ent.median()),
                          "range": [float(S_ent.min()), float(S_ent.max())]},
            "frac_tail": {"median": float(frac_tail.median()),
                          "range": [float(frac_tail.min()),
                                    float(frac_tail.max())]},
        },
        "outcomes": {"verdict": verdict},
        "meta": {
            "note": "Lens fit on HINDI (finewiki@hi), docs English — "
                    "structural test (sparsity vs tail energy), not token "
                    "readout. Author flags lens 'experimental/unvalidated'; "
                    "validity gate = bounded spectrum of J_22. Per TZ: do "
                    "NOT substitute 9B results (different D).",
            "tail": f"bottom {TAIL_FRAC:.0%} of σ (k > {tail})",
        },
    }
    # record file sha256
    import hashlib
    sha = hashlib.sha256(open(LENS, "rb").read()).hexdigest()
    out["params"]["lens_sha256"] = sha
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"ρ(S, frac_tail) median={rho_med:.4f} CI={rho_ci} -> {verdict}")
    print(f"JSON -> {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
