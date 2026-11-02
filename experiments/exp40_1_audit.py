"""
E-40.1 — audit of slices/objects (tier-0 gate, docs/models/1-7.md §5 item 0.0
+ corr/2-1 §2.1 [D4 floor] + corr/3-1 numbering: 0.0 -> E-40.1).

OBSERVATIONAL AUDIT: no new forward passes. Reads the committed artifacts of
E-33r/E-33r.2 (4 JSON variants) and E-27.6 and pins, for every headline claim
the tier-0 tests will consume, the QUADRA object|slice|side|layer (+r, +Delta)
plus a bootstrap-grade resolution floor.

Items (launch doc + corrections):
  A1  object Delta_gf ~ pi/2   : which quantity, r, side, slice, layers
  A2  signs of Delta_gain      : card vs artifacts (4/4 JSONs)
  A3  "98-99%"                 : chain status (E-33r.2 -> E-39-suppl), kill_frac
  A4  r_eff 24-31 vs 178/31    : object collision (ESG-entropy-PR vs U_obs tau-PR)
  A5  D4 resolution floor      : floor(med-of-n) ~ 1.253*IQR/sqrt(n) from stored
                                 per-doc IQRs (e34 protocol stores med+iqr+n)

Verdict semantics (per corr/2-1 §2.1): effect below floor => `not_testable`,
NEVER `refuted`. All deviations from preregistered card text -> D7 tracking.

Output: results/E40_1_audit.json
"""
from __future__ import annotations

import json
import math
import os

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
REPO = os.path.abspath(os.path.join(_HERE, ".."))

FILES = {
    "v1":         "results/exp33r_gauge_anatomy.json",
    "e33r2":      "results/exp33r_gauge_anatomy_e33r2_0830.json",
    "rerun":      "results/exp33r_gauge_anatomy_rerun.json",
    "strict":     "results/exp33r_gauge_anatomy_strict.json",
    "strictdiag": "results/exp33r_gauge_anatomy_strict_diag.json",
}
LAYERS = ["6", "12", "18", "22", "23"]
CARD_GAIN = {"6": 0.404, "12": 0.442, "18": 0.444, "22": 0.380, "23": -1.270}


def _floor(iqr_pair, n: int) -> float:
    """Half-width CI of a median-of-n-medians, normal approx:
    floor = 1.253 * IQR / sqrt(n).  (1.253 = sqrt(2*pi)/2, sd = 1.349*IQR/2 for
    normal; half-width 1.96*sd/sqrt(n)/... -> 1.253*IQR/sqrt(n) upper bound.)"""
    lo, hi = iqr_pair
    return 1.253 * (hi - lo) / math.sqrt(max(n, 2))


def main(results_dir: str | None = None) -> dict:
    results_dir = results_dir or os.path.join(REPO, "results")
    dev = []          # D7 deviations_from_spec

    # ---- A1/A2/A5: E-33r variants ------------------------------------------
    variants, gain_grid, floors = {}, {}, {}
    for name, rel in FILES.items():
        path = os.path.join(REPO, rel)
        if not os.path.exists(path):
            dev.append(f"artifact missing: {rel}")
            continue
        d = json.load(open(path))
        meta = d.get("meta", {})
        variants[name] = {"r": meta.get("r"), "delta_grid": meta.get("delta_grid"),
                          "n_docs": meta.get("n_docs"), "seed": meta.get("seed")}
        L = d.get("layers", {})
        if not L:
            continue
        gain_grid[name] = {}
        for l in LAYERS:
            dl = L.get(l, {}).get("delta", {})
            row = {}
            for dt, v in dl.items():
                g = v.get("gain", {})
                if "med" not in g:
                    continue
                row[dt] = {"gain": round(g["med"], 5),
                           "raw": round(v["raw"]["med"], 5),
                           "gf": round(v["gf"]["med"], 5),
                           "n_docs": g.get("n_docs")}
            gain_grid[name][l] = row
        # floors from the primary variant's stored per-doc IQRs (A5)
        if name == "e33r2":
            for l in LAYERS:
                for dt, v in L[l]["delta"].items():
                    for key in ("raw", "gf", "gain"):
                        if "iqr" in v.get(key, {}):
                            floors[f"{l}|{dt}|{key}"] = round(
                                _floor(v[key]["iqr"], v[key].get("n_docs", 31)), 5)

    # A2 verdict: card signs vs artifacts
    prim = gain_grid.get("e33r2", {})
    a2 = {"card_values": CARD_GAIN, "artifact_sign_r16_d1": {},
          "all_negative_all_layers_all_D_4of4": None, "card_reproducible": False}
    all_neg = True
    for name, grid in gain_grid.items():
        for l, row in grid.items():
            for dt, v in row.items():
                if not (v["gain"] < 0):
                    all_neg = False
    for l in LAYERS:
        g = prim.get(l, {}).get("1", {}).get("gain")
        a2["artifact_sign_r16_d1"][l] = g
    a2["all_negative_all_layers_all_D_4of4"] = all_neg
    card_ok = all(abs(CARD_GAIN[l] - (prim.get(l, {}).get("1", {}).get("gain") or 9e9))
                  < 0.01 for l in LAYERS)
    a2["card_reproducible"] = bool(card_ok)
    if not all_neg:
        dev.append("A2: gain not negative everywhere in artifacts (postscript text wrong)")
    if not card_ok:
        dev.append("A2 CRITICAL: ref-exp E-33r.2 card cites Delta_gain = "
                   "0.404/0.442/0.444/0.380 (L6-L22) — positive values are NOT "
                   "reproducible from ANY of the 4 committed E-33r JSONs "
                   "(all show gain<0 on every layer and Delta); string enters "
                   "history via verbatim restruct 8b0544c/60b8841 (Oct 17). "
                   "Card must be corrected to artifact values.")

    # A1: object quadra (from meta + code reading, exp33r_gauge_anatomy.py L191)
    a1 = {
        "object": "Delta_gf = max principal angle between top-r RIGHT subspaces "
                  "of gauge-free J^(p_t) = P(p_t) J at positions (t, t+Delta); "
                  "E-28 metric (acos of min svdval of Va^T Vb)",
        "side": "right (V)", "r_primary": 16,
        "r_sweep_diag": [1, 2, 4, 8, 16, 32],
        "slice": "canonical J^(p_t) (per-position p_t); raw and fixP = controls",
        "layers": [6, 12, 18, 22, 23], "delta_grid": [1, 4, 8, 16],
        "n_docs_effective": 31, "seed": 42,
        "headline_L23_d1": {"raw": 0.28923, "gf": 1.56652, "fixP_card": 0.152},
        "reproduced_from_json": {
            "raw": prim.get("23", {}).get("1", {}).get("raw"),
            "gf": prim.get("23", {}).get("1", {}).get("gf")},
    }

    # A3: 98-99% chain status + kill_frac cross-check (this artifact, slice 2)
    d2 = json.load(open(os.path.join(REPO, FILES["e33r2"])))
    kills = {l: round(d2["layers"][l]["kill_frac"]["med"], 4)
             for l in LAYERS if isinstance(d2["layers"].get(l, {}).get("kill_frac"), dict)}
    a3 = {"status": "ADJUDICATED upstream (docs/E39+.md ch.2; card E-39-suppl): "
                    "'98-99%' = operator norm of P (O1), NOT energy; "
                    "g_uni (=O2=O3) = 0.126-0.215; energy claims re-based",
          "kill_frac_e33r2": kills,
          "consistent_with_card_range_0.980_0.989":
              all(0.977 <= v <= 0.990 for v in kills.values())}

    # A4: r_eff object collision
    a4 = {"E-27.6_T6": "r_eff = exp(-sum p ln p), ENTROPY participation ratio of "
                       "the ESG (JJ^T) spectrum over K=128 modes, per-sample, "
                       "range 24.4-31.6 — mode-spectrum object",
          "E-39a": "r_eff(tau) = threshold participation ratio of U_obs "
                   "(hidden covariance): body r_eff(0.05)=178, per-doc 31 — "
                   "observation-space object",
          "collision": "two different objects share the name r_eff; the '98-99% "
                       "vs r_eff=24-31' contradiction was ALREADY dissolved by "
                       "E-39-suppl (norm vs energy); tier-0 must always name "
                       "the subscript: r_eff^ESG vs r_eff^U(tau)",
          "open": "which p-slice the E-27.5/27.6 spectra used (filename "
                  "'S64K64p0' suggests fixed p0) — follow-up before E-40.2 "
                  "consumes C_F/C_top16 lineage"}

    # A5: floors + significance of the gain statement (D4 discipline)
    a5 = {"method": "floor = 1.253*IQR/sqrt(n_docs), from stored per-doc IQR "
                    "(e34 protocol); effect < floor => not_testable, not refuted",
          "floors": floors}
    gain_sig = {}
    for l in LAYERS:
        v = prim.get(l, {}).get("1", {})
        if "gain" not in v:
            continue
        f = floors.get(f"{l}|1|gain")
        if f is None:
            continue
        eff = abs(v["gain"])
        gain_sig[l] = {"gain": v["gain"], "floor": f,
                       "status": "testable" if eff > f else "not_testable"}
    a5["gain_d1_per_layer"] = gain_sig
    ntt = [l for l, s in gain_sig.items() if s["status"] == "not_testable"]
    a5["statement_fix"] = ("'gain<0 at ALL layers' must be softened to "
                           f"'significant at {sorted(set(gain_sig)-set(ntt))}, "
                           f"not_testable at {sorted(ntt)} (N=32)'")

    out = {
        "experiment": "E-40.1", "tz": "docs/models/1-7.md §5 (0.0); corr/2-1 §2.1; "
                    "corr/3-1 §1 (renumbering)",
        "phase": "tier0_gate", "observational": True, "new_forwards": 0,
        "artifacts": variants,
        "A1_delta_gf_object": a1,
        "A2_delta_gain_signs": a2,
        "A3_98_99_status": a3,
        "A4_r_eff_collision": a4,
        "A5_D4_floors": a5,
        "gain_grid_all_variants": gain_grid,
        "gate_open_items_for_E40_2": [a4["open"]],
        "deviations_from_spec": dev,
    }
    os.makedirs(results_dir, exist_ok=True)
    p = os.path.join(results_dir, "E40_1_audit.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"saved {p}")
    print("A2 all-negative 4/4 JSONs:", a2["all_negative_all_layers_all_D_4of4"],
          "| card reproducible:", a2["card_reproducible"])
    print("A5 per-layer D4 status (D=1):",
          {l: s["status"] for l, s in gain_sig.items()})
    print("deviations:", len(dev))
    return out


if __name__ == "__main__":
    main()
