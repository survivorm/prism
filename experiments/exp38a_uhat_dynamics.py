#!/usr/bin/env python
"""E-38a — û_t trajectory and local curvature through positions (docs/E38.md).

Post-process of the exp27b dense cache (kaggle_kernel/results/exp27b_jcache_L{l}.pt).
H-only load (~2.1 GB/layer, no JR stacking — docs/E38_fixes2.md §4). For each (layer, doc, t):
    û_t = h_t / ‖h_t‖₂ (inline, clamp_min 1e-12 — attention-sink guard)
    s(t) = ‖û_{t+1} − û_t‖₂,                      t = 0..S-2
    κ_g(t) = geodesic_curvature(û_{t-1}, û_t, û_{t+1}),  t = 1..S-2
Segments: head=[0,16), body=[16,S-2), tail={S-2} (single position: s is
[N, S-1] with indices 0..S-2 and kappa needs the triple (t-1, t, t+1), so
position S-1 has neither s[S-1] nor kappa[S-1] — {S-2} is the only
possible tail). Anomaly if
|median(seg) − median(body)| / IQR(body) > 2.

Pre-registered outcomes (the post-hoc 'edges atypical_candidate' label was
DEAD CODE and is removed — docs/E38_fixes2.md §1):
    edges special   — head and tail both anomalous AND both ABOVE body ((>>)
    edges typical   — no segment anomalous
    edges atypical  — head << body (head anomalous in the DOWNWARD direction)
    mixed           — everything else: one of head/tail anomalous, other not,
                      or opposite directions (post-hoc extension, recorded as
                      such; docs/E38_fixes.md §1 MEDIUM)
`mixed` is NOT in the TZ's pre-registered label set; it is recorded here
explicitly as a post-hoc extension: a new label cannot be introduced
silently (docs/E38_fixes.md §1, MEDIUM).
Per-layer verdicts; aggregate = majority of layers; on a tie the label
whose layers come FIRST in LAYERS order wins (documented, was implicit).

pos 0 is NOT excluded from `head` — the TZ says «исключать из медиан ИЛИ
записать как отдельный сегмент» and the separate `pos0` block is the
chosen option; a pos0-vs-head-excluding-pos0 diagnostic is recorded for
the L6/L22 re-check (docs/E38_fixes.md §1 LOW). `pos0_ratio` uses the
median over positions 1..S-1 (pos 0 excluded from its own denominator —
a strong sink would inflate the ratio, docs/E38_fixes.md §1 MEDIUM).

NaN handling: triples with any ‖h_t‖ < 1e-10 are skipped in κ_g (NaN),
counted in `kappa_nan_count` and excluded by nan-robust statistics —
exact-zero h would make û = 0 and crash geodesic_curvature's unit-check
mid-run (docs/E38_fixes.md §1 HIGH; E-28 convention «дегенерации → NaN и
исключаются»). Cache check: min ‖h‖ = 5.85 on L22, count(<1e-10) = 0.

tail = {S−2} (single position), not {S−2, S−1} as in the TZ: s has
indices 0..S−2 and κ_g 1..S−2, so position S−1 does not exist in those
fields — the TZ contains an off-by-one, the code is right; recorded in
`meta.tz_deviation` for the report (docs/E38_fixes.md §1 MEDIUM).

JSON guard: `default=_json_default` raises on unwrapped numpy scalars
instead of silently converting np.bool_ → 1.0 (docs/E38_fixes.md §1 LOW).

Deterministic: no randomness (medians only), seed=0 recorded for protocol.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _e38_cache import LAYERS, S, load_layer_H  # noqa: E402
from prism.holonomy import geodesic_curvature  # noqa: E402


SEED = 0
HEAD_END = 16
BODY_END = S - 2          # 126; tail = {126} (see docstring: S-1 undefined for s/kappa)
T_ANOM = 2.0

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38a_uhat_dynamics.json")


def seg_stats(vals: np.ndarray, body: np.ndarray) -> dict:
    # nan-robust: κ_g carries NaN for degenerate triples (‖h‖ < 1e-10)
    med = float(np.nanmedian(vals))
    q1, q3 = np.nanpercentile(vals, [25, 75])
    iqr_b = float(np.nanpercentile(body, 75) - np.nanpercentile(body, 25))
    med_b = float(np.nanmedian(body))
    ratio = abs(med - med_b) / iqr_b if iqr_b > 0 else (
        float("inf") if med != med_b else 0.0)
    return {"median": med, "iqr": float(q3 - q1),
            "anomaly_ratio": ratio, "anomalous": bool(ratio > T_ANOM)}


def main():
    t0 = time.time()
    per_layer = {}
    for l in LAYERS:
        H = load_layer_H(l)  # [N, S, D] — H only (no 2.1 GB JR stack)
        N, _, Ddim = H.shape
        norms = H.norm(dim=-1)                       # [N, S]
        u = H / norms.clamp_min(1e-12)[:, :, None]   # [N, S, D]
        # sink diagnostics: ‖h_0‖ vs median position norm over positions
        # 1..S-1 — pos 0 is NOT in its own denominator (a strong sink
        # would inflate the ratio; docs/E38_fixes.md §1 MEDIUM).
        pos0_ratio = (norms[:, 0] / norms[:, 1:].median(dim=-1).values
                      .clamp_min(1e-12))
        s = (u[:, 1:] - u[:, :-1]).norm(dim=-1)      # [N, S-1]  t=0..S-2
        kapp = np.zeros((N, S - 1), dtype=np.float64)  # t=0..S-2, kappa at t+1
        kapp_nan_count = 0
        for t in range(1, S - 1):
            for i in range(N):
                # exact-zero h would give û = 0 and crash
                # geodesic_curvature's unit-check mid-run: NaN + exclude
                # (docs/E38_fixes.md §1 HIGH, E-28 convention).
                # Guard window = t-1..t+1 INCLUSIVE: the slice must cover
                # all three triple members (t-1, t, t+1); the old t-1:t+1
                # missed t+1 (docs/E38_fixes2.md §2).
                if float(norms[i, t - 1:t + 2].min()) < 1e-10:
                    kapp[i, t] = np.nan
                    kapp_nan_count += 1
                    continue
                try:
                    # geodesic_curvature raises RuntimeError when the three
                    # û coincide (arc < eps) — padding/repeat inputs; NaN +
                    # count, same convention as the norm guard
                    # (docs/E38_fixes2.md §2).
                    kapp[i, t] = geodesic_curvature(
                        u[i, t - 1], u[i, t], u[i, t + 1])
                except (RuntimeError, ValueError):
                    kapp[i, t] = np.nan
                    kapp_nan_count += 1

        def segs(v: np.ndarray, t_lo: int, t_hi: int) -> np.ndarray:
            return v[:, t_lo:t_hi].ravel()

        s_body = segs(s, HEAD_END, BODY_END)
        k_body = segs(kapp, HEAD_END, BODY_END)

        def per_doc_ratio(v: np.ndarray, t_lo: int, t_hi: int,
                          b_lo: int, b_hi: int, iqr_b: float) -> list:
            """Per-doc anomaly ratio (docs/E38_fixes.md §5.1/§2.5): mixed
            seg_stats conflates docs and positions; the per-doc ratio
            distinguishes "edge special on all docs" from "on 5 of 32".
            Denominator = IQR of the MIXED body (same as seg_stats, for
            comparability); medians are per-doc over positions."""
            med_seg = np.nanmedian(v[:, t_lo:t_hi], axis=1)
            med_body = np.nanmedian(v[:, b_lo:b_hi], axis=1)
            if iqr_b > 0:
                return [float(abs(a - b) / iqr_b)
                        for a, b in zip(med_seg, med_body)]
            return [0.0] * v.shape[0]

        iqr_s = float(np.nanpercentile(s_body, 75) - np.nanpercentile(s_body, 25))
        iqr_k = float(np.nanpercentile(k_body, 75) - np.nanpercentile(k_body, 25))
        row = {
            "s": {
                "head": seg_stats(segs(s, 0, HEAD_END), s_body),
                "body": seg_stats(s_body, s_body),
                "tail": seg_stats(segs(s, BODY_END, S - 1), s_body),
                "per_doc_ratio_head": per_doc_ratio(s, 0, HEAD_END,
                                                    HEAD_END, BODY_END, iqr_s),
                "per_doc_ratio_tail": per_doc_ratio(s, BODY_END, S - 1,
                                                    HEAD_END, BODY_END, iqr_s),
            },
            "kappa_g": {
                "head": seg_stats(segs(kapp, 1, HEAD_END), k_body),
                "body": seg_stats(k_body, k_body),
                "tail": seg_stats(segs(kapp, BODY_END, S - 1), k_body),
                "per_doc_ratio_head": per_doc_ratio(kapp, 1, HEAD_END,
                                                    HEAD_END, BODY_END, iqr_k),
                "per_doc_ratio_tail": per_doc_ratio(kapp, BODY_END, S - 1,
                                                    HEAD_END, BODY_END, iqr_k),
            },
            "pos0": {
                "median_s0": float(np.nanmedian(s[:, 0])),
                "median_kappa_t1": float(np.nanmedian(kapp[:, 1])),
                "frac_norm_h0_below_0.1x_median": float(
                    (pos0_ratio < 0.1).float().mean()),
                "median_pos0_norm_ratio": float(pos0_ratio.median()),
                # pos 0 vs head EXCLUDING pos 0 — the TZ's «исключать ИЛИ
                # записать» was resolved as «записать, не исключать»; this
                # diagnostic supports the L6/L22 re-check asked in
                # docs/E38_fixes.md §1 LOW.
                # pos 0 vs head EXCLUDING pos 0 AND the boundary-crossing
                # step 15→16 (s[15] touches body index 16): head positions
                # are 0..15, so 'head excl pos0' = steps 1..14
                # (docs/E38_fixes2.md §2 off-by-one).
                "median_s_head_excl_pos0": float(
                    np.nanmedian(s[:, 1:HEAD_END - 1])),
                "kappa_nan_count": int(kapp_nan_count),
            },
            "profile_S": [float(np.nanmedian(s[:, t])) for t in range(0, S - 1, 4)],
            "profile_K": [float(np.nanmedian(kapp[:, t])) for t in range(1, S, 4)],
        }
        # GDN vs full-attention: L18 (full) vs L19 (GDN) profile distance
        row["is_gdn"] = bool(l in (19, 22))
        per_layer[str(l)] = row
        del H
        print(f"L{l}: s head={row['s']['head']['median']:.4f} "
              f"body={row['s']['body']['median']:.4f} "
              f"tail={row['s']['tail']['median']:.4f} | "
              f"k head={row['kappa_g']['head']['median']:.4f} "
              f"body={row['kappa_g']['body']['median']:.4f} "
              f"tail={row['kappa_g']['tail']['median']:.4f} | "
              f"h0/med-norm={row['pos0']['median_pos0_norm_ratio']:.3f}")

    # L18 (full) vs L19 (GDN) profile comparison (Pitfall: GDN recurrence may
    # give a numerically different trajectory). Guarded: if LAYERS changes
    # and drops 18/19 the keys would be missing (docs/E38_fixes2.md §5).
    l18_vs_l19 = None
    if "18" in per_layer and "19" in per_layer:
        p18 = np.array(per_layer["18"]["profile_S"])
        p19 = np.array(per_layer["19"]["profile_S"])
        l18_vs_l19 = {"max_abs_profile_diff": float(np.abs(p18 - p19).max()),
                      "mean_abs_profile_diff": float(np.abs(p18 - p19).mean())}

    # verdicts per layer + aggregate
    # Direction-aware (docs/E38_fixes2.md §1): the pre-registered
    # 'edges special' requires BOTH edges ABOVE body ((>>)); 'edges atypical'
    # requires the head ANOMALOUS downward (<<); opposite directions fall
    # to 'mixed'. The old code: (a) let (head BELOW, tail above) hit
    # 'edges_special', (b) never assigned 'edges_atypical' (masked by the
    # 'mixed' branch), (c) carried a dead 'edges_atypical_candidate'
    # branch (its condition implied h_s, already False there).
    verdicts = {}
    for l, row in per_layer.items():
        h_s, t_s = row["s"]["head"]["anomalous"], row["s"]["tail"]["anomalous"]
        h_k, t_k = row["kappa_g"]["head"]["anomalous"], row["kappa_g"]["tail"]["anomalous"]
        h = h_s or h_k
        t = t_s or t_k
        head_above = row["s"]["head"]["median"] > row["s"]["body"]["median"]
        tail_above = row["s"]["tail"]["median"] > row["s"]["body"]["median"]
        if h and t and head_above and tail_above:
            v = "edges_special"
        elif h and t:
            # both anomalous but not both above (opposite directions) —
            # the TZ's (>>) is not satisfied; 'mixed' is the honest label.
            v = "mixed"
        elif h and not head_above:
            # pre-registered 'edges atypical': head << body, anomalous
            # ratio (the same T_ANOM as every other label).
            v = "edges_atypical"
        elif h or t:
            v = "mixed"
        else:
            v = "edges_typical"
        verdicts[l] = v
    counts = {}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1
    # AGGREGATE = a LABEL (the old max() returned the LAYER KEY "18", not a
    # label — docs/E38_fixes2.md §1 critical). Tie-breaking: first in LAYERS
    # insertion order wins (documented — previously an implicit dict-order
    # accident, docs/E38_fixes.md §1 LOW).
    best_layer = max(verdicts.keys(), key=lambda k: (counts[verdicts[k]],
                                                     -list(verdicts).index(k)))
    aggregate = verdicts[best_layer]

    out = {
        "params": {"seed": SEED, "S": S, "layers": list(LAYERS), "N_docs": 32,
                   "head": f"[0,{HEAD_END})", "body": f"[{HEAD_END},{BODY_END})",
                   "tail": f"{{{BODY_END}}}", "anomaly_threshold": T_ANOM,
                   "cache": "kaggle_kernel/results/exp27b_jcache_L{l}.pt"},
        "metrics": per_layer,
        "l18_vs_l19_gdn": l18_vs_l19,
        "outcomes": {"per_layer": verdicts, "aggregate": aggregate,
                     "counts": counts},
        "meta": {"S": S, "note": "GDN chunk boundary at S=128 (E-27.6) may "
                                 "contaminate tail; recorded in params",
                 "pos0": "attention sink recorded separately (not excluded "
                         "from segments per TZ)",
                 "labels": "edges special requires BOTH edges anomalous AND "
                         "ABOVE body (TZ (>>)); edges atypical = head "
                         "anomalous downward; opposite directions and "
                         "single-edge anomalies -> mixed (post-hoc "
                         "extension, docs/E38_fixes.md §1 MEDIUM); the dead "
                         "'edges_atypical_candidate' branch was removed "
                         "(docs/E38_fixes2.md §1)",
                 "tz_deviation": "tail = {S-2} single position, not the TZ's "
                         "{S-2, S-1}: s has indices 0..S-2 and kappa_g 1..S-2, "
                         "so position S-1 does not exist in those fields — TZ "
                         "off-by-one, code is right (docs/E38_fixes.md §1 "
                         "MEDIUM); 'mixed' is a post-hoc label extension, "
                         "not in the pre-registered set (same §)",
                 "per_doc_ratio": "anomaly_ratio in seg_stats uses the MIXED "
                         "docs×positions body IQR (IQR-mixing, documented); "
                         "per_doc_ratio_* gives the per-doc medians over "
                         "positions with the same mixed denominator — "
                         "distinguishes 'edge special on all docs' from a "
                         "subset (docs/E38_fixes.md §5.1/§2.5)"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # NaN in a per-doc segment (an all-NaN segment → nanmedian NaN) must
    # not emit the non-standard NaN literal into the JSON (valid RFC-8259;
    # same convention as exp38d's _json_default, docs/E38_fixes2.md §2).
    def _json_default(o):
        if isinstance(o, float) and not np.isfinite(o):
            return "nan" if o != o else ("inf" if o > 0 else "-inf")
        raise TypeError(f"object of type {type(o).__name__} is not JSON serializable")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=_json_default)
    print(f"VERDICTS {verdicts} aggregate={aggregate}")
    print(f"JSON -> {OUT} ({os.path.getsize(OUT)} bytes) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
