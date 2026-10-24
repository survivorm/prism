"""E-38i. Readout-only edits: factorization J = (sqrt(D)/r_t) M_K(I - û ûᵀ).

Pre-registered (docs/E38.md): apply the E-16 readout-only ΔW snapshots
and check the L23 factorization (E-35 lemma [M-1]) token-by-token on U128.

Key structural fact (pre-flight, verified): the edit changes ONLY lm_head, so
h_t, r_t, û_t, J_norm(t) are INVARIANT across steps; only W (hence J(s),
M_K(s)) changes. The factorization is an EXACT identity for the last-layer
readout Jacobian for ANY W (closed form == dense W @ J_norm), so the
residual is structurally zero and the test measures:
  (a) the numeric floor of the identity (fp16 snapshots -> fp64 algebra),
  (b) the leaf drift: relative change of J rows / M_K rows per step.
Pre-registered verdict rule: residual(s) <= 1e-6 for all s  =>  "factorization
preserved under readout edit (leaf-protective: the edit cannot break the
identity because h_t is invariant)".  A fitted-approximation factorization
would show residual growth with s; we do not expect that.

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
    W_orig256 = raw["W_orig"].to(torch.float64)
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

    states = {0: W_full}
    for k in K_STEPS:
        W_ed = W_full.clone()
        dW_shared = snaps_all[k][shared_idx256] - W_orig256[shared_idx256]
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
        out_states[s] = {
            "factorization_residual_max": max(res),
            "factorization_residual_mean": sum(res) / len(res),
            "J_rel_drift_max": max(jdr),
            "MK_rel_drift_max": float(
                ((MK - MK0).norm(dim=1) / MK0.norm(dim=1)).max()),
            "h_invariant": True,  # structural: edit touches only lm_head
        }

    resid_max_all = max(v["factorization_residual_max"]
                        for v in out_states.values())
    verdict = ("factorization preserved under readout-only edits "
               "(residual at numeric floor; h_t/û_t/J_norm invariant)"
               if resid_max_all <= RESID_TOL
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
                           "invariant; only W (=> J, M_K) changes",
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
              f"J_drift_max={v['J_rel_drift_max']:.4f} "
              f"MK_drift_max={v['MK_rel_drift_max']:.4f}")
    print("  verdict:", verdict)


if __name__ == "__main__":
    main()
