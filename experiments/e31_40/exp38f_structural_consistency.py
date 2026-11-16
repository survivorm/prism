#!/usr/bin/env python
"""E-38f — Structural consistency: cos(v_g, v_pred) with alignment
normalization (docs/E38.md).

Post-process exp27b dense cache L23 (0 cost). Direct test that the gauge
direction v_g (from SVD J) equals v_pred = J^T·(1/√K)/‖·‖ (lemma M-1).

Per (doc, t) at L23 (RAW J — M-1 is derived for raw J, per TZ Pitfalls):
    SVD J = U Σ V^T
    c_k  = |u_k · 1/√K|, g = argmax_k c_k
    P0 (docs/E38_fixes.md §38f): the Δcos ≥ 0 property is a MATHEMATICAL
    consequence of the lemma ONLY for g = top mode (σ_1). If argmax picks
    k ≠ 0, Δcos can be deeply negative (σ_k < σ_1) WITHOUT the lemma
    being violated — such (doc, t) are excluded from the verdict (NaN +
    g_not_top_frac recorded). The verdict runs on the g == 0 subset.
    v_g  = corresponding right singular vector (top mode)
    v_pred = v_pred_from_JT(J, 1/√K)
    cos(v_g, v_pred) — with |·| (sign(v_g) non-unique, LAPACK)
    cos(u_top, 1/√K) — the alignment factor c_g (c[0], top mode)
    Δcos = cos(v_g, v_pred) − c_g

Pre-registered interpretation (docs/E38_fixes.md part-7 §2, HIGH-2):
    per [L-gauge-J], cos(v_k, v_pred) = σ_k c_k / ‖J^T u‖, so under exact
    factorization Δcos ∈ [0, 1 − c_g] (top-mode dominance gives Δcos → 0;
    weaker dominance gives larger Δcos — ALL compatible with the lemma).
    Δcos < −0.05        — lemma BROKEN (negative is impossible exactly)
    Δcos ≤ 1 − c_g + 0.05 — lemma CONFIRMED (within the lemma's range)
    Δcos > 1 − c_g + 0.05 — methodological anomaly (numeric failure)
(The old rule "Δcos > +0.1 = anomaly" was too strict: it rejected valid
lemma-compatible results.)

Gate: run only if E-38e did not report gate_failed (|α|/|β| > 10) or
gate = "degenerate" (nan_frac > 0.5) — both make E-38f uninformative
(docs/E38_fixes.md §38e P1-3).

Verdict labels (renamed from the pre-registered lemma_confirmed /
lemma_broken — the test measures EMPIRICAL alignment v_⊥ ∥ v_g, not the
lemma itself, which is an exact identity; rename recorded in
meta.labels_renamed, docs/E38_fixes.md §38f):
    empirical_alignment_within_range — Δcos ≤ 1 − c_g + 0.05 (the lemma's
                       range; top-mode dominance shrinks it toward 0)
    empirical_alignment_violated     — Δcos < −0.05 (impossible exactly
                       under the lemma; 0.05 is a STATISTICAL tolerance,
                       not numerical — float32 SVD noise is ~1e-7)
    methodological_anomaly           — Δcos > 1 − c_g + 0.05 (numeric failure)

Bootstrap CI of Δcos over DOCS: per-doc median of Δcos, then bootstrap
over the 32 docs (B=1000, seed=0, statistic=np.median). The old flat
bootstrap over N·S correlated positions overstated precision ~√(S)×
(docs/E38_fixes.md part-7 §3, MEDIUM).

Deterministic: seed=0.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _e38_cache import load_layer  # noqa: E402
from prism.cache import bootstrap_ci  # noqa: E402
from prism.jstruct import v_pred_from_JT  # noqa: E402
from prism.json_utils import dump as json_dump  # noqa: E402

SEED = 0
B = 1000
# Statistical tolerance for the verdict (NOT numerical — float32 SVD noise
# is ~1e-7; 0.05 acknowledges empirical Δcos can drift below 0;
# docs/E38_fixes.md §38f).
EPS = 0.05

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38f_structural_consistency.json")
GATE_FILE = os.path.join(REPO, "results", "exp38e_MK_diagnostic.json")


def main():
    t0 = time.time()
    # gate check — FAIL-CLOSED (P0, docs/E38_fixes2.md): a missing E-38e
    # JSON is NOT 'E-38e passed'. The old code set gate_ok = True on
    # absence, silently running E-38f ungated.
    gate_file_missing = not os.path.exists(GATE_FILE)
    gate_ok = not gate_file_missing
    if gate_file_missing:
        print("WARNING: E-38e JSON missing — gate FAIL-CLOSED, E-38f skipped")
    else:
        with open(GATE_FILE) as f:
            g = json.load(f)
        e_gate = g["outcomes"].get("gate", "")
        # P1-3 (docs/E38_fixes.md §38e): "degenerate" (nan_frac > 0.5)
        # also stops E-38f — defense in depth on top of the propagated
        # gate_failed flag.
        gate_ok = (not g["outcomes"].get("gate_failed", False)
                   and e_gate not in ("gate_failed", "degenerate"))
        print(f"E-38e gate: {e_gate} -> "
              f"{'PASS (run)' if gate_ok else 'gate_failed/degenerate (skip)'}")
    if not gate_ok:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        tmp = OUT + ".tmp"
        with open(tmp, "w") as f:
            json_dump({"params": {"seed": SEED, "gate": "gate_failed"},
                       "outcomes": {"skipped": True},
                       "meta": {"gate_file_missing": gate_file_missing,
                                "note": "E-38e gate failed (|α|/|β| > 10), "
                                        "degenerate (nan_frac > 0.5) or E-38e "
                                        "JSON absent (fail-closed): E-38f "
                                        "uninformative, skipped per TZ"}},
                      f, indent=1)
        os.replace(tmp, OUT)   # atomic: no 0-byte OUT on raise
        print("E-38f SKIPPED")
        return

    ck = load_layer(23, want=("J",))      # J only: H/P not used (~2.1 GB saved)
    J = ck["J"]                           # [N, S, K, D]
    N, Ss, Kk, Dd = J.shape
    del ck
    one = torch.ones(Kk) / math.sqrt(Kk)

    cos_vg_vp = []
    cos_ug_one = []
    dcos = []
    profile = []
    n_g_not_top = 0
    n_vpred_degenerate = 0
    for i in range(N):
        row_dc = []
        for t in range(Ss):
            Jt = J[i, t]
            # MEDIUM-6: loud shape check (K is a file constant; a cache
            # built with a different K must not silently mis-index).
            if Jt.shape[0] != Kk:
                raise ValueError(f"Jt rows {Jt.shape[0]} != K {Kk}")
            U, _, Vt = torch.linalg.svd(Jt, full_matrices=False)  # [K,K],[K],[K,D]
            # c_k = |u_k · 1/√K| = (U.T @ one)[k] — the k-th COLUMN of U
            # dotted with one. (The old code used (U @ one)[k] = row k of
            # U dotted with one — a different, meaningless quantity; it
            # picked the wrong mode and produced a false lemma_broken,
            # docs/E38_fixes.md part-7 §1, HIGH-1.)
            c = (U.T @ one).abs()                # [K], c_k = |u_k·1/√K|
            k = int(c.argmax())
            # P0 (docs/E38_fixes.md §38f): the Δcos ≥ 0 property is a
            # MATHEMATICAL consequence of the lemma ONLY for g = top mode
            # (σ_1). If argmax picks k != 0, Δcos can be deeply negative
            # (σ_k < σ_1) WITHOUT the lemma being violated. Such (doc, t)
            # are excluded from the verdict (NaN) and counted — the verdict
            # runs on the g == 0 subset. The old code used k = argmax
            # unfiltered, so a non-top pick produced a false lemma_broken.
            if k != 0:
                n_g_not_top += 1
                cos_vg_vp.append(float("nan"))
                cos_ug_one.append(float("nan"))   # excluded from cu_med
                row_dc.append(float("nan"))
                continue
            vg = Vt[0]                             # top-mode right vector
            cos_ug_one.append(float(c[0]))         # c_g on the top mode
            try:
                vp = v_pred_from_JT(Jt, one)
            except RuntimeError:
                vp = None
                # P2 (docs/E38_fixes.md §38f): count the degenerate
                # ‖J^T u‖ ≈ 0 subset — previously indistinguishable from
                # "normal" NaN.
                n_vpred_degenerate += 1
            if vp is not None:
                cv = float((vg @ vp).abs())
            else:
                cv = float("nan")
            cos_vg_vp.append(cv)
            row_dc.append(cv - float(c[0]))
        dcos.extend(row_dc)
        if i % 8 == 0:
            # MEDIUM-4: nanmedian — NaN rows (‖J^T u‖ ≈ 0) must not poison
            # the profile.
            profile.append(float(np.nanmedian(row_dc)))

    cv_med = float(np.nanmedian(cos_vg_vp))
    cu_med = float(np.nanmedian(cos_ug_one))
    dc = np.array(dcos).reshape(N, Ss)
    cu = np.array(cos_ug_one).reshape(N, Ss)
    n_total = N * Ss
    g_not_top_frac = float(n_g_not_top) / n_total
    vpred_degenerate_frac = float(n_vpred_degenerate) / n_total
    # MEDIUM-3: DOC-level CI (the unit of independence is the doc;
    # positions within a doc are correlated, E-31 C_F(1) up to 0.996).
    per_doc_dcos = np.array([float(np.nanmedian(dc[i])) for i in range(N)])
    valid = per_doc_dcos[~np.isnan(per_doc_dcos)]
    nan_count = int(np.isnan(dc).sum())
    nan_frac = float(np.isnan(dc).mean()) if dc.size else 0.0
    # MEDIUM-5: bootstrap_ci needs >= 2 values; has_data reflects that.
    has_data = valid.size >= 2
    dc_med = float(np.nanmedian(valid)) if has_data else float("nan")
    dc_ci = (list(bootstrap_ci(valid, B=B, seed=SEED)) if has_data
             else [float("nan"), float("nan")])
    # P1 (docs/E38_fixes.md §38f): per-position methodological-anomaly
    # check. A single (doc, t) with Δcos > 1 − c_g + eps is a numeric
    # failure at THAT position; the per-doc median hides it. c_g varies
    # per position (cu), g != top positions are NaN -> excluded. Sensitivity
    # diagnostic, not a verdict (like degen_sweep in E-38d).
    upper_per_t = 1.0 - cu + EPS
    frac_above = float((dc > upper_per_t).mean())

    # HIGH-2: lemma-compatible range is Δcos ∈ [0, 1 − c_g] (top-mode
    # dominance shrinks it toward 0); negative is impossible exactly.
    # EPS = 0.05 is a STATISTICAL tolerance, not a numerical one: float32
    # SVD noise is ~1e-7, so 0.05 (1e5× that) acknowledges empirical
    # Δcos can drift below 0. The verdict "within_range" on Δcos ∈
    # [−0.05, 0) is therefore empirical, not lemma-exact.
    eps = EPS
    upper = 1.0 - cu_med + eps
    # P1 (docs/E38_fixes.md §38f): the labels are renamed from the
    # pre-registered lemma_confirmed / lemma_broken — the test measures the
    # EMPIRICAL alignment v_⊥ ∥ v_g, not the lemma (an exact identity).
    # Rename recorded in meta.labels_renamed.
    if not has_data:
        verdict = "insufficient_data_nan"
    elif dc_med < -eps:
        verdict = "empirical_alignment_violated"
    elif dc_med <= upper:
        verdict = "empirical_alignment_within_range"
    else:
        verdict = "methodological_anomaly"
    out = {
        "params": {"seed": SEED, "B": B, "layer": 23, "N_docs": N, "S": Ss,
                   "J": "raw (M-1 derived for raw J)",
                   "cache": "exp27b_jcache_L23.pt"},
        "metrics": {
            "cos_vg_vpred": {"median": cv_med},
            "cos_ug_uniform": {"median": cu_med},
            "dcos": {"median": dc_med, "ci": dc_ci,
                     "ci_note": "doc-level: per-doc median, bootstrap over 32 docs",
                     "mean": float(np.nanmean(dc)),
                     "mean_note": "flat mean over N*S positions — NOT comparable to the median (median-of-per-doc-medians): different aggregates (docs/E38_fixes.md §38f)",
                     "nan_count": nan_count, "nan_frac": nan_frac,
                     "g_not_top_frac": g_not_top_frac,
                     "g_not_top_note": "fraction of (doc,t) where argmax_k |c_k| picked a non-top mode: excluded from the verdict (NaN), Δcos >= 0 does not follow for k != 0 (docs/E38_fixes.md §38f P0)",
                     "vpred_degenerate_frac": vpred_degenerate_frac,
                     "frac_positions_above_upper": frac_above,
                     "frac_above_note": "per-position Δcos > 1 - c_g + eps (c_g per position): sensitivity diagnostic, not a verdict (per-doc median hides a single anomalous position), docs/E38_fixes.md §38f"},
            "profile_dc_per_docblock": profile,
            "profile_docs": list(range(0, N, 8)),
        },
        "outcomes": {"verdict": verdict, "gate": "pass"},
        "meta": {"note": "|cos| used (sign(v_g) non-unique, LAPACK); "
                         "c_k = (U.T @ one)[k] = |u_k·1/√K| (the old "
                         "(U @ one)[k] row-metric was a bug, part-7 §1); "
                         "the lemma J^T u = c β v_⊥ is an exact identity "
                         "— the cos test measures the EMPIRICAL alignment "
                         "v_⊥ ∥ v_g. The Δcos ≥ 0 range is a consequence "
                         "of the lemma ONLY for g = top mode: (doc, t) "
                         "with argmax c_k at k != 0 are EXCLUDED from the "
                         "verdict (NaN) and counted in g_not_top_frac — "
                         "a non-top pick can give Δcos << 0 without "
                         "violating the lemma (P0, docs/E38_fixes.md "
                         "§38f; supersedes docs/E38_fixes2.md). Δcos < "
                         "−eps = empirical_alignment_violated, ≤ 1−c_g"
                         "+eps = empirical_alignment_within_range, above "
                         "= methodological_anomaly. eps = 0.05 is a "
                         "STATISTICAL tolerance, not a numerical one "
                         "(float32 SVD noise is ~1e-7; 0.05 = 1e5× that): "
                         "a 'within_range' verdict on Δcos ∈ [−eps, 0) is "
                         "empirical, not lemma-exact. dcos median is a "
                         "NaN-excluding median (nanmedian); "
                         "nan_count/nan_frac record the degenerate "
                         "‖J^T u‖ ≈ 0 subset; vpred_degenerate_frac the "
                         "v_pred_from_JT RuntimeErrors",
                 "labels_renamed": "verdict labels renamed from the "
                                   "pre-registered lemma_confirmed / "
                                   "lemma_broken to "
                                   "empirical_alignment_within_range / "
                                   "empirical_alignment_violated: the "
                                   "lemma is an exact identity, the test "
                                   "measures empirical alignment "
                                   "(docs/E38_fixes.md §38f)"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # P1-1 (docs/E38_fixes.md §38f): strict RFC-8259 — sanitize non-finite
    # floats + allow_nan=False (the old default=_json_default was dead for
    # Python floats; the insufficient_data_nan branch writes NaN CI).
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json_dump(out, f, indent=1)
    os.replace(tmp, OUT)   # atomic: no 0-byte OUT on raise
    print(f"cos(v_g,v_pred) med={cv_med:.4f}  cos(u_g,1/√K) med={cu_med:.4f}  "
          f"Δcos med={dc_med:.4f} CI={dc_ci}")
    print(f"VERDICT: {verdict}")
    print(f"JSON -> {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
