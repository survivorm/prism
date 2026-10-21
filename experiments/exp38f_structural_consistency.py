#!/usr/bin/env python
"""E-38f — Structural consistency: cos(v_g, v_pred) with alignment
normalization (docs/E38.md).

Post-process exp27b dense cache L23 (0 cost). Direct test that the gauge
direction v_g (from SVD J) equals v_pred = J^T·(1/√K)/‖·‖ (lemma M-1).

Per (doc, t) at L23 (RAW J — M-1 is derived for raw J, per TZ Pitfalls):
    SVD J = U Σ V^T
    u_g  = left singular vector maximizing |cos(u_g, 1/√K)|
    v_g  = corresponding right singular vector
    v_pred = v_pred_from_JT(J, 1/√K)
    cos(v_g, v_pred) — with |·| (sign(v_g) non-unique, LAPACK)
    cos(u_g, 1/√K)   — the alignment factor
    Δcos = cos(v_g, v_pred) − cos(u_g, 1/√K)

Pre-registered interpretation:
    |Δcos| < 0.05 (within CI) — lemma M-1 CONFIRMED, gauge = M_K projection
    Δcos < −0.1               — lemma BROKEN, gauge from another mechanism
    Δcos > +0.1               — methodological anomaly (check M_K, g,
                                token_set): cos ≈ 1 is EXCESS, not agreement

Gate: run only if E-38e did not report gate_failed (|α|/|β| > 10).

Bootstrap CI of Δcos over docs (B=1000, seed=0, statistic=np.median).
Deterministic: seed=0.
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
from prism.jstruct import v_pred_from_JT  # noqa: E402

SEED = 0
B = 1000
K = 128

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38f_structural_consistency.json")
GATE_FILE = os.path.join(REPO, "results", "exp38e_MK_diagnostic.json")


def main():
    t0 = time.time()
    # gate check
    gate_ok = True
    if os.path.exists(GATE_FILE):
        g = json.load(open(GATE_FILE))
        gate_ok = not g["outcomes"].get("gate_failed", False)
        print(f"E-38e gate: {g['outcomes']['gate']} -> "
              f"{'PASS (run)' if gate_ok else 'gate_failed (skip)'}")
    if not gate_ok:
        json.dump({"params": {"seed": SEED, "gate": "gate_failed"},
                   "outcomes": {"skipped": True},
                   "meta": {"note": "E-38e gate failed (|α|/|β| > 10): "
                                     "E-38f uninformative, skipped per TZ"}},
                  open(OUT, "w"), indent=1)
        print("E-38f SKIPPED (gate_failed)")
        return

    ck = load_layer(23)
    J, H = ck["J"], ck["H"]               # [N, S, K, D], [N, S, D]
    N, Ss, Kk, Dd = J.shape
    del ck
    one = torch.ones(Kk) / torch.sqrt(torch.tensor(float(Kk)))

    cos_vg_vp = []
    cos_ug_one = []
    dcos = []
    profile = []
    for i in range(N):
        row_dc = []
        for t in range(Ss):
            Jt = J[i, t]
            U, _, Vt = torch.linalg.svd(Jt, full_matrices=False)  # [K,K],[K],[K,D]
            # u_g: left singular vector maximizing |cos(u_g, 1/√K)|
            c = (U @ one).abs()                 # [K]
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
            profile.append(float(np.median(row_dc)))

    cv_med = float(np.nanmedian(cos_vg_vp))
    cu_med = float(np.nanmedian(cos_ug_one))
    dc = np.array(dcos)
    dc_med = float(np.median(dc))
    dc_ci = list(bootstrap_ci(dc[~np.isnan(dc)], B=B, seed=SEED))

    if abs(dc_med) < 0.05:
        verdict = "lemma_confirmed"
    elif dc_med < -0.1:
        verdict = "lemma_broken"
    elif dc_med > 0.1:
        verdict = "methodological_anomaly"
    else:
        verdict = "intermediate"
    out = {
        "params": {"seed": SEED, "B": B, "layer": 23, "N_docs": N, "S": Ss,
                   "J": "raw (M-1 derived for raw J)",
                   "cache": "exp27b_jcache_L23.pt"},
        "metrics": {
            "cos_vg_vpred": {"median": cv_med},
            "cos_ug_uniform": {"median": cu_med},
            "dcos": {"median": dc_med, "ci": dc_ci, "mean": float(np.nanmean(dc))},
            "profile_dc_per_docblock": profile,
        },
        "outcomes": {"verdict": verdict, "gate": "pass"},
        "meta": {"note": "|cos| used (sign(v_g) non-unique, LAPACK); "
                         "cos(v_g,v_pred) ≈ cos(u_g,1/√K) is full agreement; "
                         "cos ≈ 1 is EXCESS (anomaly), not agreement"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"cos(v_g,v_pred) med={cv_med:.4f}  cos(u_g,1/√K) med={cu_med:.4f}  "
          f"Δcos med={dc_med:.4f} CI={dc_ci}")
    print(f"VERDICT: {verdict}")
    print(f"JSON -> {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
