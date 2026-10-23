#!/usr/bin/env python
"""E-38b — Classical position audit: C_b, PR_modes, align_K1 on [16, S-2].

Post-process exp27b dense cache (0 cost). Per (layer, t, doc):
    C_b(t)      — cross-doc cosine of gauge-free J_t (Frobenius); per-doc
                  value = median over the other 31 docs, reported as the
                  median over docs.
    PR_modes(t) — participation_ratio of eigenvalues of J_t J_t^T
                  (= singular values squared), per doc.
    align_K1(t) — max_k |<u_k, 1/√K>| where u_k = k-th LEFT singular
                  vector of J_t. The max over modes (not U[:,0]) is the
                  E-37 convention (u_g = argmax |cos(u, 1/√K)|): on L22 the
                  top mode is NOT uniform (uniformity_0 = 0.504, E-33r),
                  so U[:,0] would measure a different vector and the
                  E-37 baseline 0.710 would be incomparable
                  (docs/E38_fixes.md part-3 §1).
                  NAMING CAVEAT: this is computed on the gauge-free J (the
                  KEPT component), whereas E-37's align_K1 was defined on
                  K = J − P(p)J (the KILLED component) — the values are
                  therefore NOT directly comparable to E-37's
                  0.895→0.710 (docs/E38_fixes.md part-3 §1, HIGH).
                  The max-over-modes is kept (documented, meta.naming).

Body = t ∈ [16, S-2] (126 INCLUDED, per TZ); J_t is defined for t = 0..S-1,
so including S-2 leaves position S-1 as the tail (non-empty).
pos 0 is the E-0.1/E-27.6 baseline; the L22 PR check uses the COMPUTED
pos0 PR median from this cache (not the paper's 4.6 — that figure is the
min over positions per paper §20, recorded as a cross-reference only;
docs/E38_fixes.md §2.3), with a cross-check warning if pos0 PR diverges
from 4.6 by >20% (it likely is a different definition, E-30 off²-PR).

CI: pos0 CI = per-doc bootstrap (32 docs). The L22 verdict and the
pos0-vs-body comparison use PAIRED bootstrapping over docs:
per-doc body = median over t ∈ [16,126] of the PER-DOC lists
(torch.stack(list, dim=0).nanmedian(dim=0) — the flat cat is [t-major]:
cat = [t16_d0..t16_d31, t17_d0..], so reshape(N, -1) rows are MIXED
t/doc and the per-doc medians were garbage; docs/E38_fixes2.md P0-1),
then bootstrap_ci_paired on the 32 (pos0, body) pairs — NOT two
independent bootstraps. The flat body CI is kept for reporting ONLY
under the name body_flat_iid_ci_do_not_use (flattened 3552 values are
NOT i.i.d.; the i.i.d. CI is ~10x too narrow, docs/E38_fixes2.md P1-5).

Degenerate docs: ‖J_t‖ < 1e-6 → NaN in C_b (excluded from medians,
counted in meta.nan_count); PR on a degenerate J returns 0.0
(participation_ratio convention).

Pre-registered outcomes (per layer):
    no_artifact        — C_b(body) ≤ 1.5 x C_b(pos0) and (L22: PR body CI
                         contains the computed pos0 PR median) and not
                         (align body > 0.9 with pos0 < 0.7)
    edge_contamination — C_b(body) > 1.5 x C_b(pos0)
    l22_position_sensitive — L22 PR body CI excludes the computed pos0
                             PR median (paper cross-ref: 4.6)
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
from prism.cache import bootstrap_ci, bootstrap_ci_paired  # noqa: E402
from prism.spectral import participation_ratio  # noqa: E402

SEED = 0
B = 1000
HEAD_END = 16
BODY_END = S - 2

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38b_position_audit.json")

PR_MODES_L22_PAPER = 4.6  # paper §20 min PR_modes on L22 (min over
                          # positions, NOT a pos0 value) — cross-reference
                          # only; the verdict uses the computed pos0 PR
                          # median (docs/E38_fixes.md §2.3).


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
        eye = torch.eye(N, dtype=torch.bool)
        iu, ju = torch.nonzero(~eye).T      # row-major pairs, per-row groups

        pos0_cos, pos0_pr, pos0_al = [], [], []
        body_cos_list, body_pr_list, body_al_list = [], [], []
        prof_t, prof_cos, prof_pr, prof_al = [], [], [], []
        nan_count = 0
        for t in range(S):
            Jt = Jgf[:, t]                                  # [N, K, D]
            Jf = Jt.flatten(1, 2)                           # [N, K*D]
            nrm = Jf.norm(dim=1)                            # [N]
            deg = nrm < 1e-6
            nan_count += int(deg.sum())
            cos_mat = (Jf @ Jf.T) / (nrm[:, None] * nrm[None, :]).clamp_min(1e-30)
            if deg.any():
                cos_mat = torch.where(deg[:, None] | deg[None, :],
                                      torch.tensor(float("nan")), cos_mat)
            cos_docs = torch.nanmedian(cos_mat[iu, ju].reshape(N, N - 1),
                                       dim=1).values
            # SVD only where needed (pos0, body, profile t)
            if t == 0 or HEAD_END <= t <= BODY_END or t % 4 == 0:
                sv = torch.linalg.svd(Jt, full_matrices=False)
                pr_docs = torch.tensor(
                    [float(participation_ratio(sv.S[i] ** 2)) for i in range(N)])
                # align_K1 = max over LEFT modes |<u_k, 1/√K>| (E-37 u_g),
                # NOT the top mode U[:,0] — on L22 they differ (see header).
                # Axes: U is [N, i (comp), k (mode)]; the mode inner product
                # sums over COMPONENTS (dim=1), then max over MODES (dim=-1).
                # The old .sum(dim=-1) summed over modes (wrong vector)
                # .max(dim=-1) took the max over components
                # (docs/E38_fixes2.md P0-2: U=I gave 1/√K by coincidence
                # only; a Haar-random U gave ~1.44 vs the correct ≤ 1).
                al_docs = (sv.U * one[None, :, None]).sum(dim=1).abs().max(dim=-1).values  # [N]
                if t == 0:
                    pos0_cos, pos0_pr, pos0_al = cos_docs, pr_docs, al_docs
                if HEAD_END <= t <= BODY_END:
                    body_cos_list.append(cos_docs)
                    body_pr_list.append(pr_docs)
                    body_al_list.append(al_docs)
                if t % 4 == 0:
                    prof_t.append(t)
                    prof_cos.append(float(cos_docs.nanmedian()))
                    prof_pr.append(float(pr_docs.median()))
                    prof_al.append(float(al_docs.median()))
        # Flat cat is for REPORTING only (non-i.i.d. CI, renamed field).
        # PER-DOC body medians: the lists are [t16: [N], t17: [N], ...];
        # stack → [111, N] and nanmedian over dim=0 → [N]. (The old
        # torch.cat(...).reshape(N, -1) mixed docs across t — the rows
        # were slices of the flat t-major array, not docs;
        # docs/E38_fixes2.md P0-1. nanmedian: an all-NaN doc → NaN,
        # dropped by paired_finite, P2-7.)
        body_cos_flat = torch.cat(body_cos_list)
        body_pr_flat = torch.cat(body_pr_list)
        body_al_flat = torch.cat(body_al_list)
        body_per_doc_cos = torch.stack(body_cos_list, dim=0).nanmedian(dim=0).values
        body_per_doc_pr = torch.stack(body_pr_list, dim=0).nanmedian(dim=0).values
        body_per_doc_al = torch.stack(body_al_list, dim=0).nanmedian(dim=0).values

        def block(posv, body_flat, body_per_doc):
            # Degenerate (doc, t) with ‖J^T u‖ < 1e-9 → NaN (MEDIUM-5).
            # Shapes: posv [32] (per-doc pos0), body_flat [3552] (flat,
            # reporting only), body_per_doc [32] (per-doc body medians).
            # The PAIRED CI drops docs where EITHER 32-vector is NaN —
            # independent .isfinite masks desynchronize when exactly one
            # side is degenerate (the 32 vs 31 crash). body_flat carries
            # its own NaN mask (reporting CI only).
            paired_finite = (torch.isfinite(posv)
                             & torch.isfinite(body_per_doc))
            posv_f = posv[paired_finite]
            bpd_f = body_per_doc[paired_finite]
            body_f = body_flat[torch.isfinite(body_flat)]
            return {
                "pos0_median": float(posv.nanmedian()),
                "pos0_ci": (list(bootstrap_ci(posv_f.numpy(), B=B, seed=SEED))
                            if posv_f.numel() >= 2 else None),
                "body_median": float(body_flat.nanmedian()),
                "body_flat_iid_ci_do_not_use": (list(bootstrap_ci(
                    body_f.numpy(), B=B, seed=SEED)) if body_f.numel() >= 2
                    else None),
                "body_ci_note": "FLAT i.i.d. bootstrap over 3552 correlated "
                                "(t, doc) values — ~10x too narrow; for "
                                "reporting only, do NOT read as significance "
                                "(docs/E38_fixes2.md P1-5). Use the paired "
                                "CI / per-doc median.",
                "body_per_doc_median": (float(bpd_f.nanmedian())
                                        if bpd_f.numel() else float("nan")),
                "paired_diff_ci_pos0_minus_body": (list(bootstrap_ci_paired(
                    posv_f.numpy(), bpd_f.numpy(), B=B, seed=SEED))
                    if posv_f.numel() >= 2 else None),
                "n_docs_paired": int(posv_f.numel()),
            }
        # P1-3: body_over_pos0 compares the SAME aggregate on both sides
        # (median-of-medians; the old flat body median vs per-doc pos0
        # median mixed aggregates, docs/E38_fixes2.md).
        c0 = float(pos0_cos.nanmedian())
        cb = float(body_per_doc_cos.nanmedian())
        per_layer[str(l)] = {
            "C_b": {**block(pos0_cos, body_cos_flat, body_per_doc_cos),
                    "body_over_pos0": cb / max(c0, 1e-12)},
            "PR_modes": block(pos0_pr, body_pr_flat, body_per_doc_pr),
            "align_K1": block(pos0_al, body_al_flat, body_per_doc_al),
            "profiles": {"t": prof_t, "C_b": prof_cos,
                         "PR_modes": prof_pr, "align_K1": prof_al},
            "nan_count": nan_count,
        }
        r = per_layer[str(l)]
        del Jgf
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
            # PAIRED verdict (docs/E38_fixes.md part-3 §4, P0): per-doc body
            # PR vs per-doc pos0 PR on the same 32 docs; sensitive only if
            # the CI of the difference excludes 0. The old check (pos0
            # median outside an independent CI over 3552 flattened values)
            # overstated body precision ~10x → false positives.
            # P0-4: guard — the paired CI is None when fewer than 2 docs
            # survive the finiteness mask (the old unpack raised TypeError
            # on None, docs/E38_fixes2.md P1-4).
            pair_ci = r["PR_modes"]["paired_diff_ci_pos0_minus_body"]
            if pair_ci is not None:
                dlo, dhi = pair_ci
                if not (dlo <= 0 <= dhi):
                    reasons.append("l22_position_sensitive")
        if r["align_K1"]["body_median"] > 0.9 and r["align_K1"]["pos0_median"] < 0.7:
            reasons.append("align_edge_effect")
        verdicts[l] = "no_artifact" if not reasons else ",".join(reasons)

    warnings_ = []
    for l, r in per_layer.items():
        if l == "22":
            pos0_pr_med = r["PR_modes"]["pos0_median"]
            ratio = pos0_pr_med / PR_MODES_L22_PAPER
            r["PR_modes"]["pos0_PR_vs_paper_ratio"] = ratio
            if abs(ratio - 1) > 0.2:
                warnings_.append(
                    f"L22 pos0 PR = {pos0_pr_med:.2f} vs paper 4.6 "
                    f"(ratio {ratio:.2f}) — 4.6 may be a different definition")
    out = {
        "params": {"seed": SEED, "B": B, "S": S, "K": K, "layers": list(LAYERS),
                   "body": f"[{HEAD_END},{BODY_END}]", "pos0_baseline": True,
                   "cache": "exp27b_jcache_L{l}.pt",
                   "ci_policy": "pos0 CI: per-doc bootstrap; pos0-vs-body: "
                                "paired bootstrap over per-doc body medians"},
        "metrics": per_layer,
        "outcomes": {"per_layer": verdicts,
                     "aggregate": ("no_artifact" if all(v == "no_artifact"
                                                        for v in verdicts.values())
                                   else "see per_layer")},
        "meta": {"PR_modes_L22_paper_crossref": PR_MODES_L22_PAPER,
                 "align_K1_target": "J^(p) (gauge-free J), NOT "
                                    "K = J − J^(p); NOT comparable to E-37 "
                                    "numbers",
                 "naming": "align_K1 computed on the gauge-free J (kept "
                           "component) with max-over-modes (E-37 u_g "
                           "convention); E-37's align_K1 was on K = J - "
                           "P(p)J (killed component) — values NOT directly "
                           "comparable to 0.895->0.710 (docs/E38_fixes.md "
                           "part-3 §1)",
                 "warnings": warnings_,
                 "note": "C_b = per-doc median cross-doc Frobenius cosine of "
                         "gauge-free J (exp01/exp27 family); convention = "
                         "median-of-medians (per-doc median over the other "
                         "docs, then median over docs) — NOT the flat median "
                         "over all N*(N-1) pairs (docs/E38_fixes.md §2.4); "
                         "PR on sv(J_t)^2 per doc; L22 verdict: paired "
                         "bootstrap, per-doc body vs pos0 (the paper's 4.6 "
                         "is a min-over-positions cross-ref only); "
                         "degenerate docs (‖J_t‖<1e-6) → NaN in C_b, counted "
                         "per layer in nan_count"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"VERDICTS {verdicts}")
    print(f"JSON -> {OUT} ({os.path.getsize(OUT)} bytes) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
