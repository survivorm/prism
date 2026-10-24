"""E-38i. Readout-only edits: factorization J = (sqrt(D)/r_t) M_K(I - û ûᵀ).

Pre-registered (docs/E38.md): apply the E-16 readout-only ΔW snapshots
and check the L23 factorization (E-35 lemma [M-1]) token-by-token on U128.

Key structural fact (pre-flight): the edit changes ONLY lm_head, so in an
UNTIED model h_t, r_t, û_t, J_norm(t) would be INVARIANT across steps; only
W (hence J(s), M_K(s)) changes. Qwen3.5-0.8B IS TIED (lm_head =
embed_tokens), so in the REAL post-edit model h_t changes too (E38_fixes
P1); here the identity and all drifts are computed on the CACHED h
(original model) only — documented in meta.

The factorization is an EXACT identity for the last-layer readout Jacobian
for ANY W (closed form == dense W @ J_norm) — the residual is structurally
zero for every W, so it verifies the algebraic identity on the edited W
rows (numeric floor), NOT "preservation under edit" (E38_fixes P0, item 2:
the old "preserved" verdict was tautological). The informative output is
the LEAF DRIFT: relative change of J rows / M_K rows per step.
Pre-registered verdict rule: residual(s) <= 1e-6 for all s  =>  "identity
holds on edited W rows (residual at numeric floor; tautology by
construction — drift metrics carry the information)".

State subsample (pre-registered, same as E-38h, bracketing the E-16 knee
knee_step=84): K_STEPS = [10, 20, 40, 80, 90, 100, 200, 400, 800, 1000].

Token-set caveat (honest): the ΔW snapshots cover the E-16 256-token set;
only 13 rows intersect U128. W_ed on U128 = W_orig(U128) + dW[shared rows];
the other 115 rows are edit-invariant by construction.

Artifacts: results/exp38i_factorization.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from prism.exp38_hi_common import (  # noqa: E402
    D, j_raw, K_STEPS, load_snapshots, load_model_tensors,
    factor_residual, load_u128,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38i_factorization.json")

# deterministic token-position sample across all 4 domains
SAMPLES = [
    ("wiki", 0, 0), ("wiki", 0, 60), ("wiki", 2, 127), ("wiki", 7, 45),
    ("math", 3, 100), ("math", 5, 0), ("math", 1, 127), ("math", 6, 70),
    ("code", 5, 10), ("code", 1, 127), ("code", 4, 64), ("code", 0, 30),
    ("chat", 7, 90), ("chat", 2, 0), ("chat", 6, 127), ("chat", 3, 55),
]
RESID_TOL = 1e-6


def main():
    t0 = time.time()
    U, ck = load_u128()
    snaps_all, raw = load_snapshots()
    missing = [k for k in K_STEPS if k not in snaps_all]
    assert not missing, f"missing snapshot steps: {missing}"
    ts256 = torch.tensor(raw["token_set"], dtype=torch.long)
    emb, gamma = load_model_tensors()

    W_full = emb[U]  # tied-embedding rows for U128 (== model W_lm, verified)
    pos = {int(tok): i for i, tok in enumerate(U.tolist())}
    shared_idx256, shared_idxU = [], []
    for i256, tok in enumerate(ts256.tolist()):
        if tok in pos:
            shared_idx256.append(i256)
            shared_idxU.append(pos[tok])
    shared_idx256 = torch.tensor(shared_idx256)
    shared_idxU = torch.tensor(shared_idxU)

    states = {0: W_full.clone()}  # clone: shared reference, not mutated
    for k in K_STEPS:
        W_ed = W_full.clone()
        # P0 fix (E38_fixes): snapshots[k] IS ΔW — do NOT subtract W_orig
        dW_shared = snaps_all[k][shared_idx256]
        W_ed[shared_idxU] = W_ed[shared_idxU] + dW_shared
        states[k] = W_ed

    Hs = [ck["H"][(d, s, t)].double() for d, s, t in SAMPLES]

    # baseline J per position
    J0 = [j_raw(states[0], h, gamma) for h in Hs]
    MK0 = states[0] * gamma[None, :]

    out_states = {}
    for s, W_ed in states.items():
        res = [factor_residual(W_ed, h, gamma) for h in Hs]
        J = [j_raw(W_ed, h, gamma) for h in Hs]
        jdr = []
        for Ji, Ji0 in zip(J, J0):
            rel = (Ji - Ji0).norm(dim=1) / Ji0.norm(dim=1)
            jdr.append(rel.max().item())
        MK = W_ed * gamma[None, :]
        # E38_fixes P1: 115 of 128 rows are edit-invariant (not in token_set),
        # so max/median over ALL rows underreports — report shared-only too.
        J_sh = torch.stack([(((Ji - Ji0).norm(dim=1) / Ji0.norm(dim=1))
                             [shared_idxU])
                            for Ji, Ji0 in zip(J, J0)])  # [16, 13]
        MK_sh = ((MK - MK0).norm(dim=1) / MK0.norm(dim=1))[shared_idxU]
        out_states[s] = {
            "factorization_residual_max": max(res),
            "factorization_residual_mean": sum(res) / len(res),
            "J_rel_drift_max": max(jdr),
            "MK_rel_drift_max": float(
                ((MK - MK0).norm(dim=1) / MK0.norm(dim=1)).max()),
            "J_rel_drift_max_shared": float(J_sh.max()),
            "J_rel_drift_median_shared": float(J_sh.median()),
            "MK_rel_drift_max_shared": float(MK_sh.max()),
            "MK_rel_drift_median_shared": float(MK_sh.median()),
        }

    resid_max_all = max(v["factorization_residual_max"]
                        for v in out_states.values())
    verdict = ("identity holds on edited W rows (residual at numeric floor; "
               "tautological by construction — leaf drifts carry the "
               "information; h_t invariance FALSE for tied model, cached h "
               "used)" if resid_max_all <= RESID_TOL
               else "RESIDUAL GROWTH — inspect")

    out = {
        "experiment": "E-38i",
        "tz": "docs/E38.md (E-38i)",
        "data": {
            "W_orig256": "exp16_dW_snapshots.pt (fp16->fp64), step grid",
            "subsample": K_STEPS,
            "W_full": "tied embed_tokens rows on U128 (fp64)",
            "shared_rows_256_U128": int(shared_idx256.numel()),
            "H": "exp27b L23 cache, sample of %d (dom,doc,t)" % len(SAMPLES),
            "samples": [list(x) for x in SAMPLES],
            "U128_size": int(U.numel()), "D": D,
        },
        "identity": "J(t) = (sqrt(D)/r_t) M_K(I - û ûᵀ) — EXACT for any W "
                    "(E-35 [M-1]); verified dense vs closed form",
        "structural_fact": "readout-only edit leaves h_t, r_t, û_t, J_norm "
                           "invariant ONLY in an untied model; Qwen3.5-0.8B "
                           "is TIED (lm_head = embed_tokens) => in the REAL "
                           "post-edit model h_t changes; all values here are "
                           "computed on the CACHED h (original model) only "
                           "(E38_fixes P1)",
        "per_state": out_states,
        "verdict_rule": "max_s residual <= %.0e => preserved" % RESID_TOL,
        "verdict": verdict,
        "runtime_s": round(time.time() - t0, 1),
        "deterministic": True,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"OK -> {OUT}")
    for s, v in out_states.items():
        print(f"  s={s}: resid_max={v['factorization_residual_max']:.2e} "
              f"J_drift_sh={v['J_rel_drift_max_shared']:.4f} "
              f"MK_drift_sh={v['MK_rel_drift_max_shared']:.4f}")
    print("  verdict:", verdict)


if __name__ == "__main__":
    main()
