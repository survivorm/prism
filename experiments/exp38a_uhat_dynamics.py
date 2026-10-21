#!/usr/bin/env python
"""E-38a — û_t trajectory and local curvature through positions (docs/E38.md).

Post-process of the exp27b dense cache (kaggle_kernel/results/exp27b_jcache_L{l}.pt),
0 cost. For each (layer, doc, t):
    û_t = h_t / ‖h_t‖₂ (inline, clamp_min 1e-12 — attention-sink guard)
    s(t) = ‖û_{t+1} − û_t‖₂,                      t = 0..S-2
    κ_g(t) = geodesic_curvature(û_{t-1}, û_t, û_{t+1}),  t = 1..S-2
Segments: head=[0,16), body=[16,S-2), tail={S-2,S-1}. Anomaly if
|median(seg) − median(body)| / IQR(body) > 2.

Pre-registered outcomes:
    edges special   — head and tail both anomalous (>>)
    edges typical   — no segment anomalous
    edges atypical  — head << body
    mixed           — one of head/tail anomalous, other not
Per-layer verdicts; aggregate = majority of layers.

Deterministic: no randomness (medians only), seed=0 recorded for protocol.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _e38_cache import LAYERS, S, load_layer  # noqa: E402
from prism.holonomy import geodesic_curvature  # noqa: E402

SEED = 0
HEAD_END = 16
BODY_END = S - 2          # 126; tail = {126, 127}
T_ANOM = 2.0

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38a_uhat_dynamics.json")


def seg_stats(vals: np.ndarray, body: np.ndarray) -> dict:
    med = float(np.median(vals))
    q1, q3 = np.percentile(vals, [25, 75])
    iqr_b = float(np.percentile(body, 75) - np.percentile(body, 25))
    med_b = float(np.median(body))
    ratio = abs(med - med_b) / iqr_b if iqr_b > 0 else (
        float("inf") if med != med_b else 0.0)
    return {"median": med, "iqr": float(q3 - q1),
            "anomaly_ratio": ratio, "anomalous": bool(ratio > T_ANOM)}


def main():
    t0 = time.time()
    per_layer = {}
    for l in LAYERS:
        ck = load_layer(l)
        H = ck["H"]  # [N, S, D]
        N, _, Ddim = H.shape
        norms = H.norm(dim=-1)                       # [N, S]
        u = H / norms.clamp_min(1e-12)[:, :, None]   # [N, S, D]
        # sink diagnostics: ‖h_0‖ vs median position norm
        pos0_ratio = (norms[:, 0] / norms.median(dim=-1).values.clamp_min(1e-12))
        s = (u[:, 1:] - u[:, :-1]).norm(dim=-1)      # [N, S-1]  t=0..S-2
        kapp = np.zeros((N, S - 1), dtype=np.float64)  # t=0..S-2, kappa at t+1
        for t in range(1, S - 1):
            for i in range(N):
                kapp[i, t] = geodesic_curvature(u[i, t - 1], u[i, t], u[i, t + 1])

        def segs(v: np.ndarray, t_lo: int, t_hi: int) -> np.ndarray:
            return v[:, t_lo:t_hi].ravel()

        s_body = segs(s, HEAD_END, BODY_END)
        k_body = segs(kapp, HEAD_END, BODY_END)
        row = {
            "s": {
                "head": seg_stats(segs(s, 0, HEAD_END), s_body),
                "body": seg_stats(s_body, s_body),
                "tail": seg_stats(segs(s, BODY_END, S - 1), s_body),
            },
            "kappa_g": {
                "head": seg_stats(segs(kapp, 1, HEAD_END), k_body),
                "body": seg_stats(k_body, k_body),
                "tail": seg_stats(segs(kapp, BODY_END, S - 1), k_body),
            },
            "pos0": {
                "median_s0": float(np.median(s[:, 0])),
                "median_kappa_t1": float(np.median(kapp[:, 1])),
                "frac_norm_h0_below_0.1x_median": float(
                    (pos0_ratio < 0.1).float().mean()),
                "median_pos0_norm_ratio": float(pos0_ratio.median()),
            },
            "profile_S": [float(np.median(s[:, t])) for t in range(0, S - 1, 4)],
            "profile_K": [float(np.median(kapp[:, t])) for t in range(1, S, 4)],
        }
        # GDN vs full-attention: L18 (full) vs L19 (GDN) profile distance
        row["is_gdn"] = bool(l in (19, 22))
        per_layer[str(l)] = row
        del ck
        hs = [r["anomalous"] for r in row["s"].values() if r is row["s"]["head"]]
        print(f"L{l}: s head={row['s']['head']['median']:.4f} "
              f"body={row['s']['body']['median']:.4f} "
              f"tail={row['s']['tail']['median']:.4f} | "
              f"k head={row['kappa_g']['head']['median']:.4f} "
              f"body={row['kappa_g']['body']['median']:.4f} "
              f"tail={row['kappa_g']['tail']['median']:.4f} | "
              f"h0/med-norm={row['pos0']['median_pos0_norm_ratio']:.3f}")

    # L18 (full) vs L19 (GDN) profile comparison (Pitfall: GDN recurrence may
    # give a numerically different trajectory)
    p18 = np.array(per_layer["18"]["profile_S"])
    p19 = np.array(per_layer["19"]["profile_S"])
    l18_vs_l19 = {"max_abs_profile_diff": float(np.abs(p18 - p19).max()),
                  "mean_abs_profile_diff": float(np.abs(p18 - p19).mean())}

    # verdicts per layer + aggregate
    verdicts = {}
    for l, row in per_layer.items():
        h_s, t_s = row["s"]["head"]["anomalous"], row["s"]["tail"]["anomalous"]
        h_k, t_k = row["kappa_g"]["head"]["anomalous"], row["kappa_g"]["tail"]["anomalous"]
        h = h_s or h_k
        t = t_s or t_k
        head_below = row["s"]["head"]["median"] < row["s"]["body"]["median"]
        if h and t:
            v = "edges_special"
        elif h or t:
            v = "mixed"
        elif head_below and row["s"]["head"]["anomaly_ratio"] > 0.5:
            v = "edges_atypical_candidate"
        else:
            v = "edges_typical"
        verdicts[l] = v
    counts = {}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    aggregate = max(counts, key=counts.get)

    out = {
        "params": {"seed": SEED, "S": S, "layers": list(LAYERS), "N_docs": 32,
                   "head": f"[0,{HEAD_END})", "body": f"[{HEAD_END},{BODY_END})",
                   "tail": f"{{{BODY_END},{S - 1}}}", "anomaly_threshold": T_ANOM,
                   "cache": "kaggle_kernel/results/exp27b_jcache_L{l}.pt"},
        "metrics": per_layer,
        "l18_vs_l19_gdn": l18_vs_l19,
        "outcomes": {"per_layer": verdicts, "aggregate": aggregate,
                     "counts": counts},
        "meta": {"S": S, "note": "GDN chunk boundary at S=128 (E-27.6) may "
                                 "contaminate tail; recorded in params",
                 "pos0": "attention sink recorded separately (not excluded "
                         "from segments per TZ)"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"VERDICTS {verdicts} aggregate={aggregate}")
    print(f"JSON -> {OUT} ({os.path.getsize(OUT)} bytes) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
