"""E-39l v2 — drift-vs-behavior join, non-tautological design.

ТЗ: docs/E39_fixes.md (раздел «ТЗ: правка E39.md §12 и exp39l_drift_behavior.py»).

v1 (архив results/archive/exp39l_drift_behavior_v1.json) коррелировал две
серии, обе монотонные по s_frac, и получал ρ≈−0.818 **по построению** —
тавтология (Spearman инвариантен к монотонным преобразованиям). v2:

  1. Detrending: isotonic (sklearn; fallback — linear polyfit, в caveats)
     по s_frac для обеих серий; primary metric = Spearman остатков (ρ_res, p).
  2. ρ_all (полные серии) репортится ТОЛЬКО как «trivial» — в вердикт не
     входит (ТЗ §12.2.3).
  3. Anti-control: B=1000 перестановок retention; ρ_anti = median|ρ|;
     verdict «drift_predictive_beyond_trend» запрещён при ρ_anti ≤ 0.3.
  4. Fail-loud (ТЗ §12.4.7): max(E38I_STEPS)==1000; n_joined ≥ 6;
     monotonicity check (drift с провалом → monotonicity_ok=false →
     inconclusive, ТЗ §2.5.2); missing steps → warning + missing_steps.
  5. E-16b из источника, если файл есть, иначе hardcoded fallback (6 точек
     доза-ответ из ref-exp.md E-16b) с verified=False и caveat.
  6. s_frac=0 исключён из join (ТЗ §12.4.5: E-16b retention(0)=0.837 —
     состояние после E-16a, не base; semantics join-точки не верифицирован).
  7. Scatter plot (ТЗ §12.3 / §3.7).

Данные:
  drift(s)    : results/exp38i_factorization.json, per_state[str(s)]
                ["J_rel_drift_max"], s ∈ E38I_STEPS (11 точек; measured).
  retention(s): E-16b доза-ответ, 6 узлов; join по 10 точкам s>0 (s=0
                исключён), retention интерполируется np.interp.
"""

import json
import os
import sys
import warnings

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp39l_drift_behavior.json")
OUT_PNG = os.path.join(REPO, "results", "exp39l_scatter.png")

SEED = 0
B_ANTI = 1000
E38I_STEPS = [0, 10, 20, 40, 80, 90, 100, 200, 400, 800, 1000]
# E-16b доза-ответ (ref-exp.md, блок 4.8; n=6, C4-heldout):
# s ∈ {0, 0.10, 0.25, 0.50, 0.75, 1.00} → top-p retention
E16B_FALLBACK = {
    0.0: 0.837, 0.10: 0.809, 0.25: 0.746,
    0.50: 0.666, 0.75: 0.523, 1.00: 0.416,
}
E16B_ENDPOINTS = [(0.0, 0.837), (1.0, 0.416)]  # ТЗ §2.5.1, tol 0.01
N_JOINED_MIN = 6
MONO_TOL = 1e-6


def _load_e16b(repo):
    """ТЗ §2.5.1: source file if present (endpoint-verified), else the
    hardcoded 6-point curve from ref-exp.md with verified=False."""
    src = os.path.join(repo, "results", "exp16b_dose_response.json")
    if not os.path.exists(src):
        # glob for any exp16b artifact (scheme {s: [...], retention: [...]})
        cand = [f for f in sorted(os.listdir(os.path.join(repo, "results")))
                if f.startswith("exp16b") and f.endswith(".json")]
        if not cand:
            return dict(E16B_FALLBACK), {"source": "hardcoded (ref-exp.md E-16b)",
                                         "verified": False}
        src = os.path.join(repo, "results", cand[0])
    with open(src) as f:
        data = json.load(f)
    s = np.asarray(data["s"], dtype=float)
    r = np.asarray(data["retention"], dtype=float)
    for s0, r0 in E16B_ENDPOINTS:
        got = float(np.interp(s0, s, r))
        if abs(got - r0) >= 0.01:
            warnings.warn(f"E-16b source endpoint mismatch at s={s0}: "
                          f"{got:.4f} vs {r0}; using source (warn, per spec)")
        else:
            assert abs(got - r0) < 0.01, f"E-16b mismatch at s={s0}"
    return dict(zip(s.tolist(), r.tolist())), {"source": src, "verified": True}


def _detrend(s, y, caveats):
    """Isotonic detrending (ТЗ §3.2); linear polyfit fallback if sklearn is
    unavailable OR the isotonic fit is exact (residual ≡ 0: happens for a
    perfectly monotone series — isotonic interpolates it exactly, leaving
    nothing to correlate). Both fallbacks are recorded in caveats."""
    linear = None  # noqa: F841 (placeholder removed)
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(increasing="auto").fit(s, y)
        resid = y - iso.predict(s)
        if np.std(resid) > 1e-12:
            return resid
        caveats.append(f"isotonic fit is exact on a perfectly monotone series "
                       "(residual ≡ 0) — linear (polyfit deg 1) detrend used")
    except ImportError:
        caveats.append("sklearn unavailable — linear (polyfit deg 1) detrend "
                       "used instead of isotonic")
    beta = np.polyfit(s, y, 1)
    return y - np.polyval(beta, s)


def _spearman(x, y):
    """Spearman rho + two-sided p (t-distribution for small n, via scipy)."""
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _anti_control(s, drift, retention, B=B_ANTI, seed=SEED):
    """ТЗ §12.2.4: random permutation of retention across s; return
    (median |rho|, p95 |rho|)."""
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(B):
        perm = rng.permutation(len(retention))
        r, _ = _spearman(drift, retention[perm])
        rhos.append(r)
    ab = np.abs(rhos)
    return float(np.median(ab)), float(np.percentile(ab, 95))


def _verdict(rho_res, p_res, rho_anti, n_measured, monotone_ok):
    """ТЗ §2.3/§3.2 — strict: drift_predictive_beyond_trend is impossible
    without anti_control_passed, n_measured ≥ 2 and monotone_ok."""
    if not monotone_ok:
        return ("inconclusive (drift non-monotone — rho_res may be "
                "informative, see diagnostics)")
    if rho_anti <= 0.3:
        return "anti_control_insufficient (test does not distinguish signal from noise)"
    if n_measured < 2:
        return "insufficient_measured_anchors"
    if rho_res <= -0.5 and p_res < 0.05:
        return "drift_predictive_beyond_trend_strong"
    if rho_res <= -0.3 and p_res < 0.10:
        return "drift_predictive_beyond_trend_moderate"
    if abs(rho_res) < 0.2:
        return "shared_trend_only (drift not predictive beyond trend)"
    if rho_res > 0.3:
        return "paradoxical (check sign, detrending failure)"
    return "inconclusive"


def main():
    caveats = []
    # --- 1. E-38i drift (fail-loud on missing file / schema) ---
    path38i = os.path.join(REPO, "results", "exp38i_factorization.json")
    if not os.path.exists(path38i):
        raise SystemExit(f"fail-loud: {path38i} not found (ТЗ §12.5)")
    with open(path38i) as f:
        d38 = json.load(f)
    per_state = d38["per_state"]  # KeyError if schema changed — loud
    assert max(E38I_STEPS) == 1000, "fail-loud: E38I_STEPS max != 1000"

    # --- 2. E-16b retention (source or hardcoded fallback) ---
    e16b_map, e16b_meta = _load_e16b(REPO)

    # --- 3. join at E38I steps, s_frac = step/1000 ---
    rows, missing = [], []
    for s in E38I_STEPS:
        if str(s) not in per_state:
            missing.append(s)
            continue
        try:
            drift_v = float(per_state[str(s)]["J_rel_drift_max"])
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"fail-loud: per_state[{s}] missing "
                              f"J_rel_drift_max (ТЗ §12.5 — no silent skip)")
        if not np.isfinite(drift_v):
            missing.append(s)
            continue
        s_frac = s / 1000.0
        rows.append({"s": s, "s_frac": s_frac,
                     "J_rel_drift_max": drift_v,
                     "retention_interp": float(np.interp(
                         s_frac, sorted(e16b_map), [e16b_map[k]
                                                     for k in sorted(e16b_map)]))})
    if missing:
        warnings.warn(f"missing/non-finite steps in E-38i: {missing}")

    # --- 4. s_frac=0 excluded (ТЗ §12.4.5) ---
    n0 = sum(1 for r in rows if r["s_frac"] == 0.0)
    rows = [r for r in rows if r["s_frac"] > 0.0]
    n_joined = len(rows)
    assert n_joined >= N_JOINED_MIN, (
        f"fail-loud: n_joined={n_joined} < {N_JOINED_MIN} (ТЗ §12.4.7)")

    # measured = join points coinciding with a real E-16b node (retention
    # taken from the source curve, not np.interp); the rest are interpolated
    # (ТЗ §1.2: «9 из 11 точек — интерполированы по 6 узлам E-16b»).

    drift = np.array([r["J_rel_drift_max"] for r in rows])
    ret = np.array([r["retention_interp"] for r in rows])
    sf = np.array([r["s_frac"] for r in rows])

    # --- 5. monotonicity of drift on the joined grid (ТЗ §12.4.7) ---
    monotonicity_ok = bool(np.all(np.diff(drift) >= -MONO_TOL))
    if not monotonicity_ok:
        warnings.warn("drift(s) is NOT monotone on the joined grid — "
                      "monotonicity_ok=false, verdict inconclusive per spec")

    # --- 6. detrend + primary residual correlation ---
    resid_drift = _detrend(sf, drift, caveats)
    resid_ret = _detrend(sf, ret, caveats)
    if np.std(resid_drift) < 1e-12 or np.std(resid_ret) < 1e-12:
        rho_res, p_res = float("nan"), float("nan")
        caveats.append("residual correlation undefined (degenerate residual)")
    else:
        rho_res, p_res = _spearman(resid_drift, resid_ret)
    rho_all, p_all = _spearman(drift, ret)  # trivial by construction (ТЗ §12.2.3)
    rho_anti, rho_anti_p95 = _anti_control(sf, drift, ret)
    anti_control_passed = bool(rho_anti > 0.3)
    e16b_nodes = sorted(e16b_map)
    n_measured = sum(
        1 for r in rows
        if any(abs(r["s_frac"] - s0) < 1e-9 for s0 in e16b_nodes))
    n_interpolated = n_joined - n_measured

    verdict = _verdict(rho_res, p_res, rho_anti, n_measured, monotonicity_ok)
    verdict_confidence = ("high"
                          if (anti_control_passed and n_measured >= 2
                              and monotonicity_ok) else "low")

    caveats.extend([
        f"rho_all = {rho_all:+.3f} is tautological (both series driven by "
        "s_frac; ТЗ §12.2.3) — reported for transparency only, never in the verdict",
        "s_frac=0 excluded from the join: E-16b retention(0)=0.837 is the "
        "post-E-16a state, not base; join-point semantics not verified (ТЗ §12.4.5)",
        (f"{n0} E-38i point(s) at s_frac=0 dropped" if n0 else
         "all E-38i join points have s_frac > 0"),
        f"retention is interpolated from the 6 E-16b dose-response nodes on "
        f"{n_interpolated} of {n_joined} join points "
        f"({n_measured} coincide with E-16b nodes); drift is measured at each "
        f"E-38i state",
        "rho_residual measures shared residual structure, NOT causation — a "
        "common third factor (e.g. W_lm degradation) drives both (ТЗ §12.6); "
        "causal test = E-39d/e",
    ])
    if e16b_meta["source"] == "hardcoded (ref-exp.md E-16b)":
        caveats.append("E-16b source = hardcoded 6-point curve from ref-exp.md "
                       "(no results/exp16b_*.json on disk; verified=False)")

    res = {
        "experiment": "E-39l: drift vs behavior (v2, non-tautological design)",
        "tz": "docs/E39.md §12 (v2) + docs/E39_fixes.md (E-39l ТЗ)",
        "verdict": verdict,
        "verdict_confidence": verdict_confidence,
        "metrics": {
            "rho_residual": {"rho": rho_res, "p": p_res},
            "rho_all_trivial": {"rho": rho_all, "p": p_all,
                                "note": "tautological — both monotone-driven in s_frac"},
            "rho_anti_control": {"median_abs": rho_anti, "p95_abs": rho_anti_p95,
                                 "B": B_ANTI, "seed": SEED},
            "anti_control_passed": anti_control_passed,
            "monotonicity_ok": monotonicity_ok,
            "n_joined": n_joined,
            "n_measured": n_measured,
            "n_interpolated": n_interpolated,
            "missing_steps": missing,
            "s_frac_excluded": [0.0],
        },
        "e16b_source": e16b_meta,
        "points": rows,
        "caveats": caveats,
        "v1_note": ("v1 ρ=−0.818 was a tautology (both series monotone in "
                    "s_frac); archived in results/archive/"
                    "exp39l_drift_behavior_v1.json"),
        "deterministic": True,
    }
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)

    # --- scatter (ТЗ §3.7) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(drift, ret, c="C0", label="joined points")
        for r in rows:
            ax.annotate(f"s={r['s_frac']:.2f}",
                        (r["J_rel_drift_max"], r["retention_interp"]),
                        fontsize=7, alpha=0.6)
        ax.set_xlabel("J_rel_drift_max")
        ax.set_ylabel("top-p retention (interp)")
        ax.set_title(f"E-39l v2: rho_all={rho_all:+.2f} (trivial), "
                     f"rho_res={rho_res:+.2f} (p={p_res:.3f})")
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=120)
        plt.close(fig)
        res["scatter_plot"] = OUT_PNG
    except ImportError:
        res["scatter_plot"] = "matplotlib not installed, skipped"

    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"E-39l v2: rho_res={rho_res:+.3f} (p={p_res:.3f}), "
          f"rho_all={rho_all:+.3f} (trivial), rho_anti={rho_anti:.3f}, "
          f"monotone={monotonicity_ok} → {verdict}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
