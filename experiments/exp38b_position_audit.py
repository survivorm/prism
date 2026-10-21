#!/usr/bin/env python
"""E-38b — Classical position audit: C_b, PR_modes, align_K1 on [16, S-2].

Post-process exp27b dense cache (0 cost). Per (layer, t, doc):
    C_b(t)      — cross-doc cosine of gauge-free J_t (Frobenius); per-doc
                  value = median over the other 31 docs, reported as the
                  median over docs.
    PR_modes(t) — participation_ratio of eigenvalues of J_t J_t^T
                  (= singular values squared), per doc.
    align_K1(t) — |<u_t, 1/√K>| where u_t = top-LEFT singular vector of
                  J_t (E-37 convention: (uK1 * ones).sum(-1).abs()).

Body = t ∈ [16, S-2) (126 NOT included — TZ Pitfall: body excludes tail);
pos 0 is the E-0.1/E-27.6 baseline. Bootstrap CI per (layer, metric) via
prism.cache.bootstrap_ci (B=1000, seed=0, statistic=np.median) over docs.

Pre-registered outcomes (per layer):
    no_artifact        — C_b(body) ≤ 1.5 x C_b(pos0) and (L22: PR body CI
                         contains 4.6) and not (align body > 0.9 with
                         pos0 < 0.7)
    edge_contamination — C_b(body) > 1.5 x C_b(pos0)
    l22_position_sensitive — L22 PR body CI excludes 4.6
    align_edge_effect  — align_K1(body) > 0.9 while pos0 < 0.7

Deterministic: seed=0, B=1000.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _e38_cache import LAYERS, S, K, load_layer, gauge_free_J  # noqa: E402
from prism.cache import bootstrap_ci  # noqa: E402
from prism.spectral import participation_ratio  # noqa: E402

SEED = 0
B = 1000
HEAD_END = 16
BODY_END = S - 2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38b_position_audit.json")

PR_MODES_L22_POS0 = 4.6   # E-27.6 baseline cited in TZ


def main():
    t0 = time.time()
    per_layer = {}
    for l in LAYERS:
        ck = load_layer(l)
        Jraw, P = ck["J"], ck["P"]          # [N, S, K, D], [N, S, K]
        N = Jraw.shape[0]
        Jgf = gauge_free_J(Jraw, P)         # [N, S, K, D]
        del Jraw, ck
        one = torch.ones(K) / torch.sqrt(torch.tensor(float(K)))

        pos0_cos, pos0_pr, pos0_al = [], [], []
        body_cos, body_pr, body_al = [], [], []
        prof_t, prof_cos, prof_pr, prof_al = [], [], [], []
        for t in range(S):
            Jt = Jgf[:, t]                                  # [N, K, D]
            Jf = Jt.flatten(1, 2)                           # [N, K*D]
            nrm = Jf.norm(dim=1)                            # [N]
            cos_mat = (Jf @ Jf.T) / (nrm[:, None] * nrm[None, :]).clamp_min(1e-30)
            mask = ~torch.eye(N, dtype=torch.bool)
            iu, ju = torch.nonzero(mask).T
            cos_docs = cos_mat[iu, ju].reshape(N, N - 1).median(dim=1).values
            # SVD only where needed (pos0, body, profile t)
            if t == 0 or HEAD_END <= t < BODY_END or t % 4 == 0:
                sv = torch.linalg.svd(Jt, full_matrices=False)
                pr_docs = torch.tensor(
                    [float(participation_ratio(sv.S[i] ** 2)) for i in range(N)])
                al_docs = (sv.U[:, :, 0] * one).sum(dim=1).abs()  # [N]
                if t == 0:
                    pos0_cos, pos0_pr, pos0_al = cos_docs, pr_docs, al_docs
                if HEAD_END <= t < BODY_END:
                    body_cos.append(cos_docs)
                    body_pr.append(pr_docs)
                    body_al.append(al_docs)
                if t % 4 == 0:
                    prof_t.append(t)
                    prof_cos.append(float(cos_docs.median()))
                    prof_pr.append(float(pr_docs.median()))
                    prof_al.append(float(al_docs.median()))
        body_cos = torch.cat(body_cos)
        body_pr = torch.cat(body_pr)
        body_al = torch.cat(body_al)

        def block(posv, bodyv):
            return {
                "pos0_median": float(posv.median()),
                "pos0_ci": list(bootstrap_ci(posv.numpy(), B=B, seed=SEED)),
                "body_median": float(bodyv.median()),
                "body_ci": list(bootstrap_ci(bodyv.numpy(), B=B, seed=SEED)),
            }
        c0, cb = float(pos0_cos.median()), float(body_cos.median())
        per_layer[str(l)] = {
            "C_b": {**block(pos0_cos, body_cos), "body_over_pos0": cb / max(c0, 1e-12)},
            "PR_modes": block(pos0_pr, body_pr),
            "align_K1": block(pos0_al, body_al),
            "profiles": {"t": prof_t, "C_b": prof_cos,
                         "PR_modes": prof_pr, "align_K1": prof_al},
        }
        r = per_layer[str(l)]
        del Jgf, sv
        print(f"L{l}: C_b pos0={r['C_b']['pos0_median']:.4f} body={r['C_b']['body_median']:.4f} "
              f"x{r['C_b']['body_over_pos0']:.2f} | PR pos0={r['PR_modes']['pos0_median']:.3f} "
              f"body={r['PR_modes']['body_median']:.3f} | align pos0={r['align_K1']['pos0_median']:.3f} "
              f"body={r['align_K1']['body_median']:.3f}")

    verdicts = {}
    for l, r in per_layer.items():
        reasons = []
        if r["C_b"]["body_over_pos0"] > 1.5:
            reasons.append("edge_contamination")
        if l == "22":
            lo, hi = r["PR_modes"]["body_ci"]
            if not (lo <= PR_MODES_L22_POS0 <= hi):
                reasons.append("l22_position_sensitive")
        if r["align_K1"]["body_median"] > 0.9 and r["align_K1"]["pos0_median"] < 0.7:
            reasons.append("align_edge_effect")
        verdicts[l] = "no_artifact" if not reasons else ",".join(reasons)
    out = {
        "params": {"seed": SEED, "B": B, "S": S, "K": K, "layers": list(LAYERS),
                   "body": f"[{HEAD_END},{BODY_END})", "pos0_baseline": True,
                   "cache": "exp27b_jcache_L{l}.pt"},
        "metrics": per_layer,
        "outcomes": {"per_layer": verdicts,
                     "aggregate": ("no_artifact" if all(v == "no_artifact"
                                                        for v in verdicts.values())
                                   else "see per_layer")},
        "meta": {"PR_modes_L22_pos0_baseline": PR_MODES_L22_POS0,
                 "note": "C_b = per-doc median cross-doc Frobenius cosine of "
                         "gauge-free J (exp01/exp27 family); PR on sv(J_t)^2 "
                         "per doc; align_K1 = |<top-left-singular, 1/sqrt(K)>| "
                         "(E-37 convention)"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"VERDICTS {verdicts}")
    print(f"JSON -> {OUT} ({os.path.getsize(OUT)} bytes) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
