#!/usr/bin/env python
"""E-38c — Degeneracy pre-flight for holonomy (eigengap_min in top-16).

Post-process exp27b dense cache (0 cost). The non-Abelian formalism
(H_012, {theta_j}) requires (near-)degenerate eigen-subspaces. This
pre-flight checks whether the top-16 singular spectrum of J_t has a
sufficiently small minimum consecutive eigenvalue gap.

Per (layer, t, doc):
    sv(J_t) -> sigma_1 >= ... >= sigma_16
    spectrum = sigma^2 (eigenvalues of J J^T)
    (gap_min, well_separated) = eigengap_min(spectrum, eps_frac)
    threshold = eps_frac * sigma_1^2, well_separated = gap_min >= threshold

eps_frac = 0.01 (convention; sensitivity sweep {0.005, 0.01, 0.02}).
L23 (lm_head) is trivial by construction — preflight is only informative
for internal layers L18/L22; all 6 layers are computed and recorded.
Layers are NOT averaged (per-TZ).

Per layer: fraction_well_separated = median_t (mean_doc well_separated);
          min_t_gap_min = min_t (median_doc gap_min).

Note (docs/E38_fixes.md §1.1): well_separated = "gap big" = the non-Abelian
formalism is NOT applicable at that (t, doc). The pre-registered verdicts
below are driven by min_ratio_t — the per-t threshold comparison
(threshold = eps*sigma_1^2 LOCAL to each (t, doc), aggregated as the
median over docs of the per-doc ratio, then min over t;
docs/E38_fixes2.md P0-3/P1-4 — the header previously claimed the verdict
used the global-scale min_t_gap_min_over_threshold, which contradicted
the code and meta). The global-scale ratio is reported for reference only.

Pre-registered outcomes (aggregate; l23_architectural is the THIRD
pre-registered outcome, promoted to an aggregate value — docs/E38_fixes2.md
P1-7; a boolean key is ALSO always written for a stable JSON schema):
    non_abelian_not_applicable — min_ratio_t >= 1.0 on ALL layers
    near_degeneracy            — min_ratio_t < 1.0 on L18 or L22
    l23_architectural          — L23 applicable (< 1.0), L18/L22 NOT
                                 applicable (>= 1.0); the old code had this
                                 condition INVERTED (docs/E38_fixes2.md P0-1)
    inconclusive               — fallback: none of the above (e.g. L6 near-
                                 degenerate while L18/L22 are not — the old
                                 code then emitted NO aggregate at all,
                                 docs/E38_fixes2.md P0-2)

Deterministic: no randomness.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from _e38_cache import LAYERS, S, load_layer  # noqa: E402
# NOTE: the vectorized gap below is the SAME math as
# prism.spectral.eigengap_min EXCEPT the zero-spectrum edge: eigengap_min
# special-cases s[0] == 0 -> (0.0, False), while the vectorized predicate
# gets an explicit `valid = thr_base > 0` mask (a fully degenerate J must
# NOT be classified well_separated, docs/E38_fixes2.md P0-5). The import
# was removed — it was dead code that implied validation
# (docs/E38_fixes2.md P1-6).

EPS_MAIN = 0.01
EPS_SWEEP = (0.005, 0.01, 0.02)
R = 16

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38c_degeneracy_preflight.json")


def main():
    t0 = time.time()
    per_layer = {}
    for l in LAYERS:
        ck = load_layer(l, want=("J",))  # J only — P/H not needed (fixes2 §6)
        J = ck["J"]                     # [N, S, K, D]
        N, Ss, Kk, Dd = J.shape
        del ck
        if min(Kk, Dd) < R:  # docs/E38_fixes2.md P1-9: the top-R slice would
            # silently shrink and "top-16" semantics break — fail loudly.
            raise ValueError(f"layer {l}: min(K,D)={min(Kk, Dd)} < R={R}")
        if not torch.isfinite(J).all():
            # docs/E38_fixes2.md P1-12: NaN/Inf in a cache would propagate
            # into ratio_t and make the verdict silently drop the layer.
            raise ValueError(f"layer {l}: non-finite J in cache")
        # accumulate per (t, doc): gap_min (main eps), well_separated (per sweep eps)
        # NB: the predicate is well_separated (NEUTRAL, spectral), NOT
        # "applicable" — [M-4] maps the opposite way (near-degenerate ⇒
        # applicable). Names keep that distinction (docs/E38_fixes.md
        # part-4 §2, MEDIUM): do not rename back to appl_*.
        gap_min_td = np.zeros((Ss, N))          # main eps
        ws_td = {e: np.zeros((Ss, N), dtype=bool) for e in EPS_SWEEP}
        s1sq = np.zeros((Ss, N))                # sigma_1^2
        for t in range(Ss):
            # float32 SVD, same as the original run (9-min baseline).
            # NOTE: an intermediate fix switched this to .double(), which
            # was a 30x slowdown (full float64 SVD per t under machine
            # load, no log progress for 4.5h). Reverted: float32 SVD
            # noise in sigma^2 is ~5e-7 relative — 10^4x below the
            # eps_frac=0.005 threshold — so it cannot flip the
            # well_separated predicate. (docs/E38_fixes2.md does not
            # require float64 for 38c; that was an over-engineering bug.)
            sv = torch.linalg.svdvals(J[:, t])  # [N, K] desc
            sv16 = sv[:, :R]                             # [N, R] float64
            lam = (sv16 ** 2)                            # [N, R] eigenvalues
            # vectorized inner loop (P2): sort once, gaps vectorized;
            # same math as per-doc eigengap_min (which itself is float64).
            s_sorted, _ = torch.sort(lam, dim=-1, descending=True)
            gaps = s_sorted[:, :-1] - s_sorted[:, 1:]    # [N, R-1]
            gap_min_vec = gaps.min(dim=-1).values        # [N]
            thr_base = s_sorted[:, 0]                    # [N] = σ_1²
            s1sq[t, :] = thr_base.cpu().numpy()
            # P0-5: a zero σ_1² (fully degenerate J) must NOT classify as
            # well_separated (0 >= eps*0 is True without the mask).
            valid = thr_base > 0
            for e in EPS_SWEEP:
                wse = torch.zeros(N, dtype=torch.bool)
                wse[valid] = gap_min_vec[valid] >= e * thr_base[valid]
                ws_td[e][t, :] = wse.cpu().numpy()
                if e == EPS_MAIN:
                    gap_min_td[t, :] = gap_min_vec.cpu().numpy()
        # per-t medians over docs, then per-layer aggregates
        gap_t_med = np.nanmedian(gap_min_td, axis=1)  # P1-12: nan-robust
        s1sq_t_med = np.nanmedian(s1sq, axis=1)       # per-t scale over docs
        # PER-T threshold ratio (TZ step 3-4: threshold = eps*σ_1² is local
        # to each (t, doc)). P1-4 (docs/E38_fixes2.md): the ratio is the
        # MEDIAN OF PER-DOC LOCAL RATIOS, not the ratio of medians (the two
        # differ when σ_1² varies across docs at a fixed t). A zero
        # per-doc σ_1² (0/0 → NaN) is maximal degeneracy: recorded as 0.0
        # (explicit, documented) so the layer lands in near_degeneracy
        # instead of silently dropping out of both branches.
        per_doc_ratio = gap_min_td / (EPS_MAIN * s1sq)   # [S, N]
        per_doc_ratio = np.where(np.isfinite(per_doc_ratio),
                                 per_doc_ratio, 0.0)
        ratio_t = np.median(per_doc_ratio, axis=1)
        min_ratio_t = float(ratio_t.min())
        s1sq_global_med = float(np.median(s1sq))
        # P1-11: guard the reference threshold against a zero global scale.
        ref_threshold = EPS_MAIN * s1sq_global_med
        global_ratio = (float(gap_t_med.min() / ref_threshold)
                        if ref_threshold > 0 else float("inf"))
        per_layer[str(l)] = {
            "fraction_well_separated": {
                # P2 (docs/E38_fixes2.md): string keys — a float key
                # becomes "0.005" after json round-trip anyway.
                str(e): float(np.nanmedian(ws_td[e].mean(axis=1)))
                for e in EPS_SWEEP},
            "min_t_gap_min": float(gap_t_med.min()),
            "threshold_used_per_t": [float(x) for x in EPS_MAIN * s1sq_t_med],
            "threshold_used_global_ref": ref_threshold,
            "min_t_gap_min_over_threshold": global_ratio,
            "min_ratio_t": min_ratio_t,
            "min_ratio_t_note": "median over docs of the PER-DOC local ratio "
                                "gap_min/(eps*sigma_1^2), then min over t; "
                                "zero per-doc sigma_1^2 -> 0.0 (maximal "
                                "degeneracy, docs/E38_fixes2.md P1-4)",
            "verdict_metric": "min_ratio_t (per-t threshold)",
            "argmin_t": int(gap_t_med.argmin()),
            "s1sq_median": s1sq_global_med,
            # per-position profile of σ_1² (median over docs) — shows how
            # the threshold scale varies with depth (docs/E38_fixes.md §5.3).
            # P2: s1sq_profile_t has S=128 entries, profile_gap_min_t only
            # S/4=32 (every 4th t) — lengths intentionally differ.
            "s1sq_profile_t": [float(x) for x in np.nanmedian(s1sq, axis=1)],
            "profile_gap_min_t": [float(x) for x in gap_t_med[::4]],
            "profile_gap_min_t_note": "every 4th t (length S/4); "
                                      "s1sq_profile_t has full length S",
            "informative": bool(l in (18, 22)),
        }
        r = per_layer[str(l)]
        del J
        print(f"L{l}: fraction_well_separated@0.01={r['fraction_well_separated']['0.01']:.3f} "
              f"(sweep {[r['fraction_well_separated'][str(e)] for e in EPS_SWEEP]}) | "
              f"min_ratio_t={r['min_ratio_t']:.3f} (global-ref ratio="
              f"{r['min_t_gap_min_over_threshold']:.3f}) informative={r['informative']}")

    # verdicts (only informative layers drive the aggregate; L23 trivial).
    # The verdict uses min_ratio_t (PER-T threshold, TZ step 3-4); the
    # global-scale ratio is reported but not used (docs/E38_fixes.md
    # part-4 §2, HIGH).
    # P0-1 (docs/E38_fixes2.md): the l23_architectural condition was
    # INVERTED — applicable = near-degenerate = min_ratio_t < 1.0 (see
    # meta.direction). P1-7: it is the THIRD pre-registered OUTCOME, so it
    # is an aggregate value, not a side boolean (the boolean is also
    # always written for a stable schema). P0-2: the 'inconclusive'
    # fallback — the old if/elif chain could emit NO aggregate (e.g. L6
    # near-degenerate, L18/L22 not).
    inf = {l: r for l, r in per_layer.items() if r["informative"]}
    verdicts = {}
    all_not_applicable = all(r["min_ratio_t"] >= 1.0
                             for r in per_layer.values())
    near_degen = [l for l, r in inf.items() if r["min_ratio_t"] < 1.0]
    l23_applicable = per_layer["23"]["min_ratio_t"] < 1.0
    l1822_not_applicable = all(per_layer[l]["min_ratio_t"] >= 1.0
                               for l in ("18", "22"))
    l23_arch = bool(l23_applicable and l1822_not_applicable)
    if all_not_applicable:
        verdicts["aggregate"] = "non_abelian_not_applicable"
    elif near_degen:
        verdicts["aggregate"] = "near_degeneracy"
        verdicts["near_degeneracy_layers"] = near_degen
    elif l23_arch:
        verdicts["aggregate"] = "l23_architectural"
    else:
        verdicts["aggregate"] = "inconclusive"
    verdicts["l23_architectural"] = l23_arch  # always present
    out = {
        "params": {"eps_frac": EPS_MAIN, "eps_sweep": list(EPS_SWEEP),
                   "r": R, "layers": list(LAYERS), "N_docs": 32, "S": S,
                   "cache": "exp27b_jcache_L{l}.pt"},
        "metrics": per_layer,
        "outcomes": verdicts,
        "meta": {"note": "L23 (lm_head) trivial by construction — preflight "
                         "informative only for L18/L22 (internal). Layers not "
                         "averaged (per-TZ). well_separated = gap_min >= "
                         "eps_frac*sigma_1^2 (i.e. near-degenerate NOT present; "
                         "the non-Abelian formalism is applicable where this is "
                         "False)",
                 "direction": "INVERSION WARNING: well_separated=True means "
                              "the spectrum is NOT near-degenerate, i.e. the "
                              "non-Abelian formalism is NOT applicable there. "
                              "fraction_well_separated must not be read as "
                              "fraction_applicable — the mapping is inverted "
                              "([M-4]). Verdicts use min_ratio_t (per-t "
                              "threshold = eps*sigma_1^2 at that t); the "
                              "global-scale ratio min_t_gap_min_over_threshold "
                              "is reported for reference only"},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    def _json_default(o):
        """P1-10: do not silently convert np.bool_ → 1.0 (raise), do not
        emit NaN/Inf literals (stringify, valid RFC-8259) — same
        convention as exp38a/exp38d (docs/E38_fixes2.md)."""
        if isinstance(o, np.bool_):
            raise TypeError(f"unwrapped numpy bool_ in JSON: {o!r}")
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (float, np.floating)) and not np.isfinite(o):
            return "nan" if o != o else ("inf" if o > 0 else "-inf")
        return str(o)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1, default=_json_default)
    print(f"OUTCOMES {verdicts}")
    print(f"JSON -> {OUT} ({os.path.getsize(OUT)} bytes) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
