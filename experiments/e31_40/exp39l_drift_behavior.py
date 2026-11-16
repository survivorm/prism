"""E-39l v2 — drift-vs-behavior join, non-tautological design.

ТЗ: docs/E39.md §12 (v2) + docs/E39_fixes.md (E-39l ТЗ).
Правки раунда 1: docs/E39l_fixes.md (anti-control criterion/на residual,
NaN early-exit, monotone both directions, fail-loud endpoint, caveats).
Правки раунда 2 (docs/E39l_fixes.md, «после E39l_fixes»):
  §2.1 NaN → None в JSON (RFC 8259, allow_nan); §2.2 moderate tier
  underpowered для малого n — явный caveat + диагностика в verdict;
  §2.3 verdict_confidence по frac_measured (high ≥ 0.5, medium ≥ 0.2,
  low); §2.4 один json.dump (в конце, после matplotlib); §2.5 p_res/
  p95_anti используются в inconclusive-сообщении; §3.1 detrend fallback
  — polyfit deg 2 (n ≥ 4) вместо линейного; §3.2 caveat parametric vs
  permutation; §3.3 retention_source в JSON (measured/interpolated,
  fraction); §3.4 null-распределение (p10/p50/p95/p99); §4.1 warn при
  p_res < 0.05 но permutation NS; §4.3 savefig try/except.

v1 (архив results/archive/exp39l_drift_behavior_v1.json) коррелировал две
серии, обе монотонные по s_frac, и получал ρ≈−0.818 **по построению** —
тавтология (Spearman инвариантен к монотонным преобразованиям). v2:

  1. Detrending: isotonic (sklearn; fallback — polyfit deg 2 if n≥4 else 1,
     в caveats) по s_frac для обеих серий; primary metric = Spearman
     остатков (ρ_res, p).
  2. ρ_all (полные серии) репортится ТОЛЬКО как «trivial» — в вердикт не
     входит (ТЗ §12.2.3).
  3. Anti-control: B=1000 перестановок RESIDUAL-серии retention →
     null-распределение для ρ_res; значимость = |ρ_res| > p95(null)
     (пермутационный тест, не жёсткий порог).
  4. Fail-loud (ТЗ §12.4.7): max(E38I_STEPS)==1000; n_joined ≥ 6;
     monotonicity check — drift не должен менять направление (возрастание
     ИЛИ убывание); провал → monotonicity_ok=false → inconclusive
     (ТЗ §2.5.2); missing steps → warning + missing_steps.
     E-16b endpoint-mismatch → fail-loud.
  5. E-16b из источника, если файл есть, иначе hardcoded fallback (6 точек
     доза-ответ из ref-exp.md E-16b) с verified=False и caveat.
  6. s_frac=0 исключён из join (ТЗ §12.4.5: E-16b retention(0)=0.837 —
     состояние после E-16a, не base; semantics join-точки не верифицирован).
  7. Scatter plot (ТЗ §12.3 / §3.7); savefig failure → caveat, не падение.
  8. NaN (дегенеративные остатки) → явный early-exit в _verdict + None в
     JSON (не `NaN` — invalid по RFC 8259).
  9. Sensitivity (E39l_fixes р2 §3.1): ρ_res при linear vs quadratic
     детрендах; если знаки расходятся — результат = артефакт детренда,
     не сигнал (caveat + sign_consistent=false).

Данные:
  drift(s)    : results/exp38i_factorization.json, per_state[str(s)]
                ["J_rel_drift_max"], s ∈ E38I_STEPS (11 точек; measured).
  retention(s): E-16b доза-ответ, 6 узлов; join по 10 точкам s>0 (s=0
                исключён), 8 из 10 retention-значений интерполированы
                np.interp по 6 узлам E-16b, 2 совпадают с узлами.
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


def _safe(v):
    """E39l_fixes (р2) §2.1: NaN/inf → None (RFC 8259: `NaN` — invalid JSON)."""
    if v is None:
        return None
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def _load_e16b(repo):
    """ТЗ §2.5.1 + E39l_fixes §3.1: source file if present; endpoint
    mismatch → fail-loud (ранее warn+use: assert в else-ветке был dead)."""
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
            raise SystemExit(
                f"fail-loud: E-16b source endpoint mismatch at s={s0}: "
                f"{got:.4f} vs {r0:.3f} (ТЗ §2.5.1; E39l_fixes §3.1) — "
                f"source {src} не согласован с ref-exp.md E-16b")
    return dict(zip(s.tolist(), r.tolist())), {"source": src, "verified": True}


def _detrend_poly(s, y, deg, caveats=None):
    """polyfit detrend, deg (1 или 2)."""
    beta = np.polyfit(s, y, deg)
    return y - np.polyval(beta, s)


def _detrend_pair(s, y_a, y_b, caveats):
    """E39l_fixes р2 §3.1: ОДИН метод детренда для ОБЕХ серий.
    Раньше _detrend вызывался по-отдельности: drift (немонотонный) получал
    isotonic, а retention (монотонный) — polyfit-fallback, и ρ_res
    становился артефактом смешения двух детрендов (не воспроизводим ни
    одним согласованным вариантом). Правило: isotonic, если ОБЕ серии дают
    не-дегенеративный isotonic-остаток; иначе — polyfit deg 2 для обеих."""
    try:
        from sklearn.isotonic import IsotonicRegression
        ra = y_a - IsotonicRegression(increasing="auto").fit(s, y_a).predict(s)
        rb = y_b - IsotonicRegression(increasing="auto").fit(s, y_b).predict(s)
        if np.std(ra) > 1e-12 and np.std(rb) > 1e-12:
            return ra, rb, "isotonic"
        msg = ("isotonic degenerate (residual ≡ 0) on a perfectly-monotone "
               "series — polyfit deg-2 detrend applied CONSISTENTLY to both "
               "series (E39l_fixes р2 §3.1)")
    except ImportError:
        msg = ("sklearn unavailable — polyfit deg-2 detrend applied "
               "CONSISTENTLY to both series (E39l_fixes р2 §3.1)")
    if msg not in caveats:
        caveats.append(msg)
    return _detrend_poly(s, y_a, 2), _detrend_poly(s, y_b, 2), "polyfit2"


def _spearman(x, y):
    """Spearman rho + two-sided p (t-distribution for small n, via scipy)."""
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def _anti_control_residual(resid_drift, resid_ret, B=B_ANTI, seed=SEED):
    """E39l_fixes р1 §2.2: null-distribution для ρ_res — перестановка
    RESIDUAL retention (не raw-серии: raw-anti-control orthogonal к
    primary metric). (median |rho|, p95 |rho|, distribution) по B
    перестановкам. Distribution (p10/p50/p95/p99) — E39l_fixes р2 §3.4:
    читатель может перепроверить p95."""
    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(B):
        perm = rng.permutation(len(resid_ret))
        r, _ = _spearman(resid_drift, resid_ret[perm])
        rhos.append(r)
    ab = np.abs(rhos)
    dist = {
        "p10": float(np.percentile(ab, 10)),
        "p50": float(np.median(ab)),
        "p95": float(np.percentile(ab, 95)),
        "p99": float(np.percentile(ab, 99)),
        "n_perms": int(B),
    }
    return float(np.median(ab)), float(np.percentile(ab, 95)), dist


def _verdict(rho_res, p_res, p95_anti, significant, monotone_ok, sign_flip=False):
    """ТЗ §2.3/§3.2 + E39l_fixes р1 §2.1/р2 §2.2/р3 §3: значимость —
    пермутационный тест (|ρ_res| > p95 null на residual-перестановках).
    sign_flip (р3 §3.4) вычисляется ДО verdict и передаётся сюда —
    verdict не противоречит caveat «do not interpret».
    p_res/p95_anti в inconclusive-диагностике (не dead params, р2 §2.5).
    moderate tier (rho_res ∈ (−0.5, −0.3] & significant) underpowered при
    малом n (null p95 ~ 0.6) — явная пометка (р2 §2.2). NaN → early-exit."""
    if np.isnan(rho_res):
        return ("inconclusive (degenerate residuals — no residual signal to "
                "correlate; E39l_fixes §2.3)")
    if sign_flip:
        return ("inconclusive (rho_res SIGN FLIPS across detrend choices — "
                "detrend artifact, not a robust signal; do not interpret; "
                "E39l_fixes р3 §3)")
    if not monotone_ok:
        return ("inconclusive (drift non-monotone — see diagnostics)")
    if significant and rho_res <= -0.5:
        return "drift_predictive_beyond_trend_strong"
    if significant and rho_res <= -0.3:
        return "drift_predictive_beyond_trend_moderate"
    if abs(rho_res) < 0.2:
        return "shared_trend_only (drift not predictive beyond trend)"
    if rho_res > 0.3 and significant:
        return "paradoxical (check sign, detrending failure)"
    if 0.3 <= abs(rho_res) <= 0.6:
        return (f"inconclusive (|rho_res|={abs(rho_res):.2f} below "
                f"p95={p95_anti:.2f}; underpowered — moderate tier "
                f"unreachable at this n; E39l_fixes р2 §2.2)")
    return (f"inconclusive (rho_res={rho_res:+.2f}, p_res={p_res:.3f}, "
            f"|rho|<p95={p95_anti:.2f} — not significant vs permutation null)")


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

    drift = np.array([r["J_rel_drift_max"] for r in rows])
    ret = np.array([r["retention_interp"] for r in rows])
    sf = np.array([r["s_frac"] for r in rows])

    # --- 5. monotonicity of drift on the joined grid (ТЗ §12.4.7,
    # E39l_fixes §2.4: monotone = без смены направления, оба направления) ---
    d = np.diff(drift)
    mono_inc = bool(np.all(d >= -MONO_TOL))
    mono_dec = bool(np.all(d <= +MONO_TOL))
    monotonicity_ok = mono_inc or mono_dec
    if not monotonicity_ok:
        warnings.warn("drift(s) changes direction on the joined grid — "
                      "monotonicity_ok=false, verdict inconclusive per spec")

    # --- 6. detrend + primary residual correlation ---
    resid_drift, resid_ret, detrend_method = _detrend_pair(sf, drift, ret, caveats)
    if np.std(resid_drift) < 1e-12 or np.std(resid_ret) < 1e-12:
        rho_res, p_res = float("nan"), float("nan")
        anti_med = float("nan")
        p95_anti = float("nan")
        anti_dist = None
        significant = False
        caveats.append("residual correlation undefined (degenerate residual)")
    else:
        rho_res, p_res = _spearman(resid_drift, resid_ret)
        anti_med, p95_anti, anti_dist = _anti_control_residual(
            resid_drift, resid_ret)
        # E39l_fixes §2.1: пермутационный тест (null для ρ_res)
        significant = bool(abs(rho_res) > p95_anti)
        # E39l_fixes р2 §4.1: несогласованность parametric vs permutation
        if np.isfinite(p_res) and p_res < 0.05 and not significant:
            warnings.warn("parametric p_res < 0.05 but permutation test is "
                          "not significant — verdict follows the "
                          "permutation test (E39l_fixes р2 §4.1)")
            caveats.append("parametric p_res < 0.05 but permutation test "
                           "not significant — verdict follows the "
                           "permutation test (E39l_fixes р2 §4.1)")
    rho_all, p_all = _spearman(drift, ret)  # trivial by construction (ТЗ §12.2.3)

    # --- 7. sensitivity: ρ_res при linear vs quadratic детрендах ---
    # (E39l_fixes р2 §3.1: на нелинейных монотонных сериях знак ρ_res
    # зависит от выбора детренда; если знаки расходятся — артефакт, не
    # сигнал.)
    sens = {}
    for name, deg in (("quadratic", 2), ("linear", 1)):
        rd = _detrend_poly(sf, drift, deg)
        rr = _detrend_poly(sf, ret, deg)
        if np.std(rd) > 1e-12 and np.std(rr) > 1e-12:
            rr_, p_ = _spearman(rd, rr)
            sens[name] = {"rho_res": _safe(rr_), "p": _safe(p_)}
        else:
            sens[name] = {"rho_res": None, "p": None,
                          "note": "degenerate residuals (detrend flat)"}
    rhos = [v["rho_res"] for v in sens.values() if v["rho_res"] is not None]
    sign_flip = False  # р3 §3.4: вычисляется ДО _verdict
    if len(rhos) <= 1:
        sens = None
    else:
        sign_flip = any(r > 0 for r in rhos) and any(r < 0 for r in rhos)
        sens["sign_consistent"] = not sign_flip
        if sign_flip:
            caveats.append("rho_res SIGN flips across detrend choices "
                           "(linear vs quadratic, E39l_fixes р2 §3.1) — "
                           "result is a detrend artifact, not a robust "
                           "signal; do not interpret")

    e16b_nodes = sorted(e16b_map)
    _is_measured = lambda x: any(abs(x - s0) < 1e-9 for s0 in e16b_nodes)
    measured_points = [r["s_frac"] for r in rows if _is_measured(r["s_frac"])]
    # р3 §4: interpolated — той же толерантной проверкой (exact `in` для
    # floats мог бы не совпасть с measured при float-ошибках → n_measured +
    # n_interpolated > n_joined)
    interpolated_points = [r["s_frac"] for r in rows
                           if not _is_measured(r["s_frac"])]
    n_measured = len(measured_points)
    n_interpolated = n_joined - n_measured
    frac_measured = n_measured / n_joined

    verdict = _verdict(rho_res, p_res, p95_anti, significant, monotonicity_ok,
                       sign_flip)
    # E39l_fixes р2 §2.3: confidence по доле measured retention, не по
    # наивному n_measured >= 2 (80% интерполяции — не "high")
    ok = bool(monotonicity_ok and not np.isnan(rho_res))
    verdict_confidence = ("high" if ok and frac_measured >= 0.5
                          else "medium" if ok and frac_measured >= 0.2
                          else "low")

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
        "anti-control null is approximate: permutation of RESIDUAL retention "
        "of a fixed series (E39l_fixes р1 §2.2), not raw-data null",
        "rho_residual.p is parametric (t-approx, asymptotic); significant "
        "uses permutation p95 — for small n they diverge, verdict follows "
        "the permutation test (E39l_fixes р2 §3.2)",
        "verdict tiers (strong: rho_res<=-0.5 & significant; moderate: "
        "rho_res<=-0.3 & significant) are underpowered for small n: "
        "permutation null p95 ~ 0.6 at n=10 -> moderate tier unreachable "
        "(E39l_fixes р2 §2.2)",
    ])
    if e16b_meta["source"] == "hardcoded (ref-exp.md E-16b)":
        caveats.append("E-16b source = hardcoded 6-point curve from ref-exp.md "
                       "(no results/exp16b_*.json on disk; verified=False)")

    res = {
        "experiment": "E-39l: drift vs behavior (v2, non-tautological design; "
                      "E39l_fixes rounds 1+2 applied)",
        "tz": "docs/E39.md §12 (v2) + docs/E39_fixes.md (E-39l ТЗ) + "
              "docs/E39l_fixes.md (ревью v2, раунды 1 и 2)",
        "verdict": verdict,
        "verdict_confidence": verdict_confidence,
        "metrics": {
            "rho_residual": {"rho": _safe(rho_res), "p": _safe(p_res)},
            "rho_all_trivial": {"rho": _safe(rho_all), "p": _safe(p_all),
                                "note": "tautological — both monotone-driven in s_frac"},
            "rho_anti_control_residual": {
                "median_abs": _safe(anti_med), "p95_abs": _safe(p95_anti),
                "significant": significant,
                "null_distribution": anti_dist,
                "B": B_ANTI, "seed": SEED,
                "note": "null for rho_res: permutation of RESIDUAL retention "
                        "(E39l_fixes р1 §2.1/§2.2)"},
            "monotonicity_ok": monotonicity_ok,
            "mono_increasing": mono_inc,
            "mono_decreasing": mono_dec,
            "n_joined": n_joined,
            "n_measured": n_measured,
            "n_interpolated": n_interpolated,
            "missing_steps": missing,
            "s_frac_excluded": [0.0],
        },
        "detrend_sensitivity": sens,
        "retention_source": {
            "measured_points": measured_points,
            "interpolated_points": interpolated_points,
            "fraction_measured": frac_measured,
            "note": "E39l_fixes р2 §3.3: interpolation (np.interp, linear "
                    "between 6 nodes) — rho_res on interpolated retention is "
                    "a model-based estimate, not a measurement",
        },
        "e16b_source": e16b_meta,
        "points": rows,
        "caveats": caveats,
        "v1_note": ("v1 ρ=−0.818 was a tautology (both series monotone in "
                    "s_frac); archived in results/archive/"
                    "exp39l_drift_behavior_v1.json"),
        "deterministic": True,
    }

    # --- scatter (ТЗ §3.7; E39l_fixes р2 §4.3: savefig failure → caveat) ---
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
    except Exception as e:
        res["scatter_plot"] = f"savefig failed: {e}"
        res["caveats"].append(f"scatter plot savefig failed: {e} "
                              "(E39l_fixes р2 §4.3)")

    # --- один write в конце (E39l_fixes р2 §2.4: partial-write risk);
    # allow_nan=False — NaN упадёт loudly (RFC 8259), не тихий `NaN` (§2.1) ---
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, allow_nan=False)
    print(f"E-39l v2: rho_res={_safe(rho_res)}, p_res={_safe(p_res)}, "
          f"rho_all={rho_all:+.3f} (trivial), rho_anti_res={_safe(anti_med)} "
          f"(p95={_safe(p95_anti)}, sig={significant}), "
          f"monotone={monotonicity_ok}, frac_measured={frac_measured:.2f} "
          f"→ {verdict}")
    if sens:
        print(f"  sensitivity: " + ", ".join(
            f"{k}: rho={v['rho_res']}" for k, v in sens.items()
            if isinstance(v, dict)) + f", sign_consistent={sens.get('sign_consistent')}")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
