#!/usr/bin/env python
"""E-38f — Structural consistency: cos(v_g, v_pred) with alignment
normalization (docs/E38.md).

Post-process exp27b dense cache L23 (0 cost). Direct test that the gauge
direction v_g (from SVD J) equals v_pred = J^T·(1/√K)/‖·‖ (lemma M-1).

Per (doc, t) at L23 (RAW J — M-1 is derived for raw J, per TZ Pitfalls):
    SVD J = U Σ V^T
    u_g  = left singular vector maximizing |cos(u_g, 1/√K)|  (c_k = u_k·1/√K)
    v_g  = corresponding right singular vector
    v_pred = v_pred_from_JT(J, 1/√K)
    cos(v_g, v_pred) — with |·| (sign(v_g) non-unique, LAPACK)
    cos(u_g, 1/√K)   — the alignment factor c_g
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

Gate: run only if E-38e did not report gate_failed (|α|/|β| > 10).

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

from _e38_cache import S, load_layer  # noqa: E402
from prism.cache import bootstrap_ci  # noqa: E402
from prism.jstruct import v_pred_from_JT  # noqa: E402

SEED = 0
B = 1000
K = 128

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38f_structural_consistency.json")
GATE_FILE = os.path.join(REPO, "results", "exp38e_MK_diagnostic.json")


def _json_default(o):
    """P0 (docs/E38_fixes2.md): the old `default=float` would write the
    bare `NaN` literal in the insufficient_data_nan branch (invalid
    RFC-8259). Strict version: raise on np.bool_, convert numpy
    numerics, NaN/Inf -> strings."""
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
    # gate check — FAIL-CLOSED (P0, docs/E38_fixes2.md): a missing E-38e
    # JSON is NOT 'E-38e passed'. The old code set gate_ok = True on
    # absence, silently running E-38f ungated.
    gate_file_missing = not os.path.exists(GATE_FILE)
    gate_ok = not gate_file_missing
    if gate_file_missing:
        print("WARNING: E-38e JSON missing — gate FAIL-CLOSED, E-38f skipped")
    else:
        g = json.load(open(GATE_FILE))
        gate_ok = not g["outcomes"].get("gate_failed", False)
        print(f"E-38e gate: {g['outcomes']['gate']} -> "
              f"{'PASS (run)' if gate_ok else 'gate_failed (skip)'}")
    if not gate_ok:
        json.dump({"params": {"seed": SEED, "gate": "gate_failed"},
                   "outcomes": {"skipped": True},
                   "meta": {"gate_file_missing": gate_file_missing,
                            "note": "E-38e gate failed (|α|/|β| > 10) or "
                                    "E-38e JSON absent (fail-closed): "
                                    "E-38f uninformative, skipped per TZ"}},
                  open(OUT, "w"), indent=1, default=_json_default)
        print("E-38f SKIPPED")
        return

    ck = load_layer(23)
    J, H = ck["J"], ck["H"]               # [N, S, K, D], [N, S, D]
    N, Ss, Kk, Dd = J.shape
    del ck
    one = torch.ones(Kk) / math.sqrt(Kk)

    cos_vg_vp = []
    cos_ug_one = []
    dcos = []
    profile = []
    for i in range(N):
        row_dc = []
        for t in range(Ss):
            Jt = J[i, t]
            # MEDIUM-6: loud shape check (K is a file constant; a cache
            # built with a different K must not silently mis-index).
            if Jt.shape[0] != Kk:
                raise ValueError(f"Jt rows {Jt.shape[0]} != K {Kk}")
            U, _, Vt = torch.linalg.svd(Jt, full_matrices=False)  # [K,K],[K],[K,D]
            # u_g: left singular vector maximizing |cos(u_k, 1/√K)|.
            # c_k = u_k · 1/√K = (U.T @ one)[k] — the k-th COLUMN of U
            # dotted with one. (The old code used (U @ one)[k] = row k of
            # U dotted with one — a different, meaningless quantity; it
            # picked the wrong mode and produced a false lemma_broken,
            # docs/E38_fixes.md part-7 §1, HIGH-1.)
            c = (U.T @ one).abs()                # [K], c_k = |u_k·1/√K|
            k = int(c.argmax())
            ug = U[:, k]
            vg = Vt[k]                          # [D]
            cos_ug_one.append(float(c[k]))
            try:
                vp = v_pred_from_JT(Jt, one)
            except RuntimeError:
                vp = None
            if vp is not None:
                cv = float((vg @ vp).abs())
            else:
                cv = float("nan")
            cos_vg_vp.append(cv)
            row_dc.append(cv - float(c[k]))
        dcos.extend(row_dc)
        if i % 8 == 0:
            # MEDIUM-4: nanmedian — NaN rows (‖J^T u‖ ≈ 0) must not poison
            # the profile.
            profile.append(float(np.nanmedian(row_dc)))

    cv_med = float(np.nanmedian(cos_vg_vp))
    cu_med = float(np.nanmedian(cos_ug_one))
    dc = np.array(dcos).reshape(N, Ss)
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

    # HIGH-2: lemma-compatible range is Δcos ∈ [0, 1 − c_g] (top-mode
    # dominance shrinks it toward 0); negative is impossible exactly.
    eps = 0.05
    upper = 1.0 - cu_med + eps
    if not has_data:
        verdict = "insufficient_data_nan"
    elif dc_med < -eps:
        verdict = "lemma_broken"
    elif dc_med <= upper:
        verdict = "lemma_confirmed"
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
                     "nan_count": nan_count, "nan_frac": nan_frac},
            "profile_dc_per_docblock": profile,
            "profile_docs": list(range(0, N, 8)),
        },
        "outcomes": {"verdict": verdict, "gate": "pass"},
        "meta": {"note": "|cos| used (sign(v_g) non-unique, LAPACK); "
                         "c_k = (U.T @ one)[k] = |u_k·1/√K| (the old "
                         "(U @ one)[k] row-metric was a bug, part-7 §1); "
                         "the lemma J^T u = c β v_⊥ is an exact identity "
                         "— the cos test measures the EMPIRICAL alignment "
                         "v_⊥ ∥ v_g, restricted to g = top mode (Δcos ≥ 0 "
                         "is impossible to violate otherwise — P0, "
                         "docs/E38_fixes2.md); Δcos < −0.05 = "
                         "lemma_violation, ≤ 1−c_g+0.05 = empirical "
                         "alignment within lemma range, above = "
                         "methodological_anomaly. dcos median is a "
                         "NaN-excluding median (nanmedian); "
                         "nan_count/nan_frac record the degenerate "
                         "‖J^T u‖ ≈ 0 subset"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=_json_default)
    print(f"cos(v_g,v_pred) med={cv_med:.4f}  cos(u_g,1/√K) med={cu_med:.4f}  "
          f"Δcos med={dc_med:.4f} CI={dc_ci}")
    print(f"VERDICT: {verdict}")
    print(f"JSON -> {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
