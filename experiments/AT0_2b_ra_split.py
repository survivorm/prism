#!/usr/bin/env python3
"""AT0_2b — T0.2b «R/A-split: кто несёт DC» (disk-only, ~5 мин CPU).

ТЗ: docs/TZ_add_0-3.md §1.5 T0.2b + (G6)/(G6′) §1.3. Пре-рег: docs/AT0_2b_prereg.md.

Δh = r̄·Δu + Δr·ū ТОЧНО (G6): radial R_t = Δr_t·ū_t (приращения нормы),
angular A_t = r̄_t·Δu_t (приращения направления). Вопрос: какая компонента несёт DC
(плато ACF T0.1): v2 (review C1/C2) — carrier_verdict по вКЛАДУ в net displacement
Δh (contrib = share + half-cross) + gate «у Δh есть DC» (p_fdc_dh); старое правило
(f_dc компонент) — dominance_verdict (descriptive). Вердикты: RADIAL_CARRIER /
ANGULAR_CARRIER / CANCELLED / NO_DH_DC / INCONCLUSIVE / not_testable.

Ядро — пинненный prism/field.py (G1′): ra_split, cross_acf, f_dc, degen, jclean,
sha256_file, _selfcheck. ACF — AT0_2_shape.acf_full (FFT zero-padded, пин в main).
Данные/загрузчик — AT0_1_accounting (load_source, LAYERS, пути, L1_DEGRADE_TOL).

Дизайн-решение (зафиксировано в пре-реге §4): null для f_dc — SIGN-FLIP (Rademacher),
НЕ перестановка: mean(x) инвариантен к ЛЮБОЙ перестановке строк (Σ invariant),
перестановочный null для f_dc вырожден (все B «null'ов» = fdc_obs). Sign-flip —
точная randomization-гипотеза «нет систематического среднего» (E[fdc|null] = 1/N).

G-гейты (fail-loud, клетка): closure REL ≤ 1e-12 (ТЗ G6′), additivity REL ≤ 1e-12,
energy-additivity REL ≤ 1e-12 (v2 M1), orth(=t) ≤ 1e-9, gram(G1 на компонентах) < 1e-10,
c0 > 0. v2: per-cell p_fdc_dh/dc_present_dh, contrib_R/A; агрегаты: n_present, absent_frac,
bootstrap-CI (B_BOOT=199, seed_ci = 9001 + pool_idx·1009 + L). v2 C4: null — «mean-direction
null» (НЕ ACF-matched). v2 C6/C7: decode_probe_status/provenance_limited; file_sha256
(canonical payload hash vs file sha — конвенция в claim.meta).
Артефакты: results/AT0/store/<sha16>.json, AT0_2b_RA.json (claim),
AT0_2b_manifest.json, AT0_2b_run.log. СMOKE (--smoke): 2 дока, артефакты НЕ пишет.
"""

import hashlib
import json
import math
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402

# G1′: тот же прибор, что AT0_1/AT0_2 — field.py напрямую из файла.
_fspec = importlib.util.spec_from_file_location(
    "prism_field_pinned", os.path.join(ROOT, "prism/field.py"))
pf = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(pf)
sys.modules["prism_field_pinned"] = pf
vpath, dcurve, degen, sha256_file = (pf.vpath, pf.dcurve, pf.degen, pf.sha256_file)

import AT0_1_accounting as a1  # noqa: E402  (load_source, decode_probe, LAYERS, пути)
import AT0_2_shape as a2  # noqa: E402  (acf_full — FFT-ACF, пин-тест в AT0_2 main)

LAYERS = a1.LAYERS
GEN, OWN, COR = a1.GEN, a1.OWN, a1.COR
OUT_DIR = a1.OUT_DIR
STORE_DIR = a1.STORE_DIR
OUT_JSON = os.path.join(OUT_DIR, "AT0_2b_RA.json")
MAN_JSON = os.path.join(OUT_DIR, "AT0_2b_manifest.json")
RUN_LOG = os.path.join(OUT_DIR, "AT0_2b_run.log")

K_MAX = 96
B_IID = 199          # p_floor = 1/200 = 0.005 (стандарт серии, урок E3)
B_BOOT = 199         # v2 C3: bootstrap-CI (per-doc)
SEED_CI_BASE = 9001  # v2 C3: seed_ci = 9001 + pool_idx·1009 + L (детерминированно)
SEED_BASE = 2026
SEED_STRIDE = 1009   # doc_idx — ГЛОБАЛЬНЫЙ по 96 (конвенция серии, AT0_3 r3.2)
ALPHA = 0.05
CLOSURE_TOL = 1e-12  # REL, scale = max‖Δh_t‖² (ТЗ G6′)
ADD_TOL = 1e-12      # REL: ‖Δh − R − A‖ / sqrt(max‖Δh_t‖²)
ENERGY_TOL = 1e-12   # v2 M1: |c0(Δh) − c0(R) − c0(A)| / c0(Δh)
ORTH_TOL = 1e-9      # max_t |⟨Δu_t, ū_t⟩| (абс, fp64)
GRAM_TOL = 1e-10     # G1 на компонентах (тот же, что AT0_2 G8)
MIN_PRESENT = 5      # v2 C3: тест-ability carrier_verdict
COMP_DEGEN_FRAC = 1e-6  # компонент с c0 < 1e-6·scale(max‖Δh_t‖²) — absent-degenerate (отчёт, не тише)

LOG = []


def logline(s=""):
    LOG.append(str(s))
    print(s, flush=True)


def sign_flip_fdc(x, doc_idx):
    """Mean-direction null для f_dc (v2 C4; пре-рег §4): sign-flip (Rademacher).
    Возвращает (fdc_obs, p_fdc, fdc_perm_median). Denominator mean‖x_t‖² инвариантен
    к знаковым флипам ⇒ меняем только числитель: fdc_s = ‖Σ_t ε_t x_t‖² / (N² ·
    mean‖x_t‖²), E[fdc_s|null] = 1/N. Граница: null НЕ сохраняет ACF/telescoping —
    «DC present» = направленный mean drift, НЕ spectral DC (пре-рег §4)."""
    x = np.asarray(x, dtype=np.float64)
    N = x.shape[0]
    c0 = float(np.einsum("ij,ij->i", x, x).sum())
    if c0 <= 0:
        raise ValueError("sign_flip_fdc: нулевой объект")
    mean_n2 = c0 / N
    fdc_obs = float(np.linalg.norm(x.mean(axis=0)) ** 2 / mean_n2)
    num = np.empty(B_IID)
    for s in range(B_IID):
        rng = np.random.default_rng(SEED_BASE + doc_idx * SEED_STRIDE + s)
        eps = (rng.integers(0, 2, size=N) * 2 - 1).astype(np.float64)
        num[s] = float(np.linalg.norm(eps @ x) ** 2)
    fdc_perm = num / (N * N * mean_n2)
    p = float((int(np.sum(fdc_perm >= fdc_obs)) + 1) / (B_IID + 1))
    return fdc_obs, p, float(np.median(fdc_perm))


def comp_acf(x):
    """(c0, rho96, r_direct, r_fft, gram) — ACF k=1..K_MAX + G1-пин на компоненте.
    Вызывать при c0 > 0 (проверяет caller)."""
    N = x.shape[0]
    c0, c, _ = a2.acf_full(x)
    r_direct = float((x.sum(axis=0) ** 2).sum() / c0)
    r_fft = 1.0 + 2.0 * c.sum() / c0
    gram = abs(r_fft - r_direct) / max(1.0, r_direct)
    K = min(K_MAX, N - 1)
    return {"c0": c0, "rho96": [float(v) for v in c[:K] / c0],
            "r_direct": r_direct, "r_fft": r_fft, "gram": float(gram)}


def cell_ra(u, hn, doc_idx):
    """Одна клетка (док, L): G6-гейты + f_dc/p_fdc (R, A) + ACF + кросс-лаги +
    точное разложение net displacement (ΣΔh = ΣR + ΣA)."""
    s = pf.ra_split(u, hn)
    dh, R, A = s["dh"], s["radial"], s["angular"]
    N = dh.shape[0]
    n2 = np.einsum("ij,ij->i", dh, dh)
    scale = float(n2.max())
    if not np.isfinite(scale) or scale <= 0:
        return {"absent": "dh_zero"}
    # G6-гейты (fail-loud)
    closure_rel = s["closure_max"] / scale
    add_rel = float(np.max(np.linalg.norm(dh - R - A, axis=1))) / math.sqrt(scale)
    orth = s["du_u_cross_max"]
    if closure_rel > CLOSURE_TOL:
        raise ValueError(f"G6 closure: {closure_rel:.3e} > {CLOSURE_TOL:.0e}")
    if add_rel > ADD_TOL:
        raise ValueError(f"G6 additivity: {add_rel:.3e} > {ADD_TOL:.0e}")
    if orth > ORTH_TOL:
        raise ValueError(f"G6 orth(=t): {orth:.3e} > {ORTH_TOL:.0e}")
    # Компоненты: f_dc + sign-flip null + ACF
    out = {"N": int(N), "scale": scale, "closure_rel": float(closure_rel),
           "add_rel": float(add_rel), "orth": float(orth)}
    for nm, x in (("radial", R), ("angular", A)):
        c0 = float(np.einsum("ij,ij->i", x, x).sum())
        if c0 <= 0 or c0 < COMP_DEGEN_FRAC * scale:
            out[nm] = {"absent": "degenerate"}
            continue
        fdc, p, fdc_perm_med = sign_flip_fdc(x, doc_idx)
        ca = comp_acf(x)
        ca.update(fdc=fdc, p_fdc=p, fdc_perm_median=fdc_perm_med, dc_present=p < ALPHA)
        out[nm] = ca
    # Δh: f_dc (уровень плато T0.1) + p (v2 C2: gate carrier_verdict) +
    #     точное разложение net displacement + contribution (v2 C1)
    fdc_dh, p_dh, fdc_dh_perm_med = sign_flip_fdc(dh, doc_idx)
    out["fdc_dh"] = fdc_dh
    out["p_fdc_dh"] = p_dh
    out["fdc_dh_perm_median"] = fdc_dh_perm_med
    out["dc_present_dh"] = bool(p_dh < ALPHA)
    dR, dA, dH = R.sum(axis=0), A.sum(axis=0), dh.sum(axis=0)
    nH = float(np.dot(dH, dH))
    if nH > 0:
        share_rad = float(np.dot(dR, dR) / nH)
        share_ang = float(np.dot(dA, dA) / nH)
        share_cross = float(2.0 * np.dot(dR, dA) / nH)
        out["share_rad"], out["share_ang"], out["share_cross"] = share_rad, share_ang, share_cross
        out["contrib_R"] = float(share_rad + 0.5 * share_cross)   # half-cross (v2 C1)
        out["contrib_A"] = float(share_ang + 0.5 * share_cross)
    else:
        out["share_rad"] = out["share_ang"] = out["share_cross"] = None
        out["contrib_R"] = out["contrib_A"] = None
    # v2 M1: energy-additivity (same-t ортогональность ⇒ c0(Δh) = c0(R) + c0(A))
    c0R_e = out.get("radial", {}).get("c0") if isinstance(out.get("radial"), dict) else None
    c0A_e = out.get("angular", {}).get("c0") if isinstance(out.get("angular"), dict) else None
    c0_dh = float(n2.sum())
    if c0R_e and c0A_e:
        energy_add_rel = abs(c0_dh - c0R_e - c0A_e) / c0_dh
        if energy_add_rel > ENERGY_TOL:
            raise ValueError(f"G6 energy-additivity: {energy_add_rel:.3e} > {ENERGY_TOL:.0e}")
        out["energy_add_rel"] = float(energy_add_rel)
    # Кросс-лаги (ОЧЕРЕДНО, без вердикта; T: ортогональность G6 — только при равных t)
    K = min(K_MAX, N - 1)
    c0R = out.get("radial", {}).get("c0")
    c0A = out.get("angular", {}).get("c0")
    if c0R and c0A:
        cRA = pf.cross_acf(R, A, K)
        cAR = pf.cross_acf(A, R, K)
        norm = math.sqrt(c0R * c0A)
        out["cross_RA"] = [float(v / norm) for v in cRA]
        out["cross_AR"] = [float(v / norm) for v in cAR]
    # Якорь плато T0.1 (самокоррекция пре-рега §1): v_tangent median_t cos(9..32)
    V, _ = pf.vpath(u, hn)
    vc = pf.dcurve(V, np.arange(1, min(97, V.shape[0])), stat="median")
    lo, hi = min(9, len(vc)), min(33, len(vc) + 1)
    out["vt_plateau"] = float(np.median(vc[lo - 1:hi - 1])) if lo < hi else None
    return out


def _boot_ci_joint(vals_r, vals_a, seed):
    """v2 C3: совместный bootstrap (per-doc, B_BOOT) медиан (R, A, ratio R/A).
    Детерминированный сид (пре-рег §7: seed_ci = 9001 + pool_idx·1009 + L)."""
    if not vals_r or not vals_a:
        return None
    r, a = np.asarray(vals_r, dtype=np.float64), np.asarray(vals_a, dtype=np.float64)
    if len(r) != len(a):
        raise ValueError(f"_boot_ci_joint: desync r/a ({len(r)} vs {len(a)}) — joint-сэмплинг нарушен")
    n = len(r)
    rng = np.random.default_rng(seed)
    med_r = np.empty(B_BOOT)
    med_a = np.empty(B_BOOT)
    ratio = np.empty(B_BOOT)
    for s in range(B_BOOT):
        idx = rng.integers(0, n, size=n)
        mr, ma = float(np.median(r[idx])), float(np.median(a[idx]))
        med_r[s], med_a[s] = mr, ma
        ratio[s] = (mr / ma) if ma > 0 else np.nan
    return {
        "r": [float(np.percentile(med_r, 2.5)), float(np.percentile(med_r, 97.5))],
        "a": [float(np.percentile(med_a, 2.5)), float(np.percentile(med_a, 97.5))],
        "ratio": [float(np.nanpercentile(ratio, 2.5)), float(np.nanpercentile(ratio, 97.5))]
                 if np.all(np.isfinite(ratio)) else None,
        "b_boot": B_BOOT,
    }


def _pool_stats(cells, n_total, pool_idx, L):
    """cells — list PRESENT-клеток per-doc за слой. v2: n_present/absent_frac +
    Δh-DC агрегаты + contribution с bootstrap-CI (C1/C2/C3/C5/M3)."""
    stat = {}
    for key in ("fdc", "p_fdc", "fdc_perm_median"):
        for nm in ("radial", "angular"):
            vals = [c[nm][key] for c in cells
                    if isinstance(c.get(nm), dict) and not c.get(nm).get("absent") and key in c.get(nm)]
            stat[f"{nm}_{key}_med"] = float(np.median(vals)) if vals else None
    for nm in ("radial", "angular"):
        vals = [c[nm]["dc_present"] for c in cells
                if isinstance(c.get(nm), dict) and not c.get(nm).get("absent") and "dc_present" in c.get(nm)]
        stat[f"{nm}_frac_dc"] = float(np.mean(vals)) if vals else None
        stat[f"{nm}_n_present"] = len(vals)
        stat[f"{nm}_n_dc_present"] = int(sum(vals)) if vals else 0
        stat[f"{nm}_absent_frac"] = float(1.0 - len(vals) / n_total) if n_total else None
    for key in ("fdc_dh", "p_fdc_dh", "share_rad", "share_ang", "share_cross",
                "contrib_R", "contrib_A", "vt_plateau"):
        vals = [c[key] for c in cells if c.get(key) is not None]
        stat[f"{key}_med"] = float(np.median(vals)) if vals else None
    vals = [c["dc_present_dh"] for c in cells if "dc_present_dh" in c]
    stat["dh_frac_dc"] = float(np.mean(vals)) if vals else None
    # v2 C3: bootstrap-CI (совместный per-doc) для fdc R/A/ratio + contrib R/A
    fdc_r = [c["radial"]["fdc"] for c in cells
             if isinstance(c.get("radial"), dict) and not c.get("radial").get("absent") and "fdc" in c["radial"]]
    fdc_a = [c["angular"]["fdc"] for c in cells
             if isinstance(c.get("angular"), dict) and not c.get("angular").get("absent") and "fdc" in c["angular"]]
    ci_seed = SEED_CI_BASE + pool_idx * 1009 + L
    stat["boot_ci_fdc"] = _boot_ci_joint(fdc_r, fdc_a, ci_seed)
    cr = [c["contrib_R"] for c in cells if c.get("contrib_R") is not None]
    ca = [c["contrib_A"] for c in cells if c.get("contrib_A") is not None]
    stat["boot_ci_contrib"] = _boot_ci_joint(cr, ca, ci_seed)
    # ACF-кривые k=1..16 (медиана по докам)
    for nm in ("radial", "angular"):
        ks = 16
        curve = []
        for k in range(ks):
            vals = [c[nm]["rho96"][k] for c in cells
                    if isinstance(c.get(nm), dict) and not c.get(nm).get("absent")
                    and len(c[nm].get("rho96", [])) > k]
            curve.append(float(np.median(vals)) if vals else None)
        stat[f"{nm}_rho16"] = curve
    for key in ("cross_RA", "cross_AR"):
        ks = 16
        curve = []
        for k in range(ks):
            vals = [c[key][k] for c in cells if c.get(key) and len(c[key]) > k]
            curve.append(float(np.median(vals)) if vals else None)
        stat[f"{key}_med16"] = curve
    stat["n_cells_total"] = n_total
    stat["n_cells_present"] = len(cells)
    return stat


def _dominance_verdict(st):
    """v2: DESCRIPTIVE (пре-рег §5 канал 1) — старое правило v1 без изменений."""
    fr, fa = st.get("radial_frac_dc"), st.get("angular_frac_dc")
    mr, ma = st.get("radial_fdc_med"), st.get("angular_fdc_med")
    if None in (fr, fa, mr, ma):
        return "not_testable"
    if fr > 0.5 and fa > 0.5:
        return "MIXED_DOM"
    if fr > 0.5 and mr > 1.1 * ma:
        return "RADIAL_DOM"
    if fa > 0.5 and ma > 1.1 * mr:
        return "ANGULAR_DOM"
    if fr <= 0.5 and fa <= 0.5:
        return "NO_DC_DOM"
    return "INCONCLUSIVE_DOM"


def _carrier_verdict(st):
    """v2 C1/C2: PRIMARY (пре-рег §5 канал 2) — кто несёт DC плато Δh по вкладу в
    net displacement (contrib = share + half-cross) + gate «у Δh есть DC»."""
    npr, npa = st.get("radial_n_present"), st.get("angular_n_present")
    if npr is None or npa is None or npr < MIN_PRESENT or npa < MIN_PRESENT:
        return "not_testable"
    dh_frac = st.get("dh_frac_dc")
    if dh_frac is None or dh_frac <= 0.5:
        return "NO_DH_DC"
    sc, sr, sa = st.get("share_cross_med"), st.get("share_rad_med"), st.get("share_ang_med")
    if None in (sc, sr, sa):
        return "not_testable"
    if sc < -0.5 and sr > 0.5 and sa > 0.5:
        return "CANCELLED"
    ci = st.get("boot_ci_contrib") or {}
    med_cr, med_ca = st.get("contrib_R_med"), st.get("contrib_A_med")
    if None in (med_cr, med_ca):
        return "not_testable"
    ci_r_low = (ci.get("r") or [None, None])[0]
    ci_a_low = (ci.get("a") or [None, None])[0]
    if ci_r_low is not None and ci_r_low > 0.5 and med_cr > med_ca:
        return "RADIAL_CARRIER"
    if ci_a_low is not None and ci_a_low > 0.5 and med_ca > med_cr:
        return "ANGULAR_CARRIER"
    return "INCONCLUSIVE"


def main():
    smoke = "--smoke" in sys.argv
    os.makedirs(STORE_DIR, exist_ok=True)
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "threads": os.environ.get("OMP_NUM_THREADS"),
           "seed_base": SEED_BASE, "seed_stride": SEED_STRIDE, "b_iid": B_IID}
    logline(f"AT0_2b T0.2b R/A-split {'SMOKE' if smoke else 'FULL'}: "
            f"py{env['python']} np{env['numpy']} thr={env['threads']}")

    # G1′: прибор — sha + selfcheck + пины (additivity ra_split; sign-flip калибровка)
    pf_sha = sha256_file(os.path.join(ROOT, "prism/field.py"))
    pf._selfcheck()
    _rng_t = np.random.default_rng(123)
    _u = _rng_t.standard_normal((384, 64))
    _u /= np.linalg.norm(_u, axis=1, keepdims=True)
    _hn = 10.0 + np.cumsum(_rng_t.standard_normal(384) * 0.01)
    _s = pf.ra_split(_u, _hn)
    _dh = _s["dh"]
    _scale = float(np.einsum("ij,ij->i", _dh, _dh).max())
    _add = float(np.max(np.linalg.norm(_dh - _s["radial"] - _s["angular"], axis=1))) / math.sqrt(_scale)
    assert _add < 1e-13, f"ra_split additivity pin: {_add:.3e}"
    # sign-flip калибровка (правило 6): iid ⇒ median fdc_perm ≈ 1/N; сильный DC ⇒ p ≤ 1/(B+1)
    _x = _rng_t.standard_normal((383, 1024))
    _, _, _med = sign_flip_fdc(_x, 9999)
    assert 0.5 / 383 < _med < 2.0 / 383, f"sign-flip calibration: median fdc {_med:.3e} !~ 1/383"
    _x_dc = _x + 20.0
    _, _p_dc, _ = sign_flip_fdc(_x_dc, 9998)
    assert _p_dc <= 1.0 / (B_IID + 1), f"sign-flip: DC-объект не отвергнут (p={_p_dc})"
    logline(f"G1′: field.py sha {pf_sha[:16]} + _selfcheck ok + additivity pin {_add:.2e} "
            f"+ sign-flip калибровка (iid median {_med:.3e} ~ 1/383; DC p={_p_dc:.4f}) ok")

    # G0: данные — sha + gen_sha + декод-проба (как AT0_2)
    data = {os.path.basename(p): sha256_file(p) for p in (GEN, OWN, COR)}   # v2.1 NIT: basename-ключи (location-invariant)
    gen_meta = json.load(open(GEN.replace(".npz", "_meta.json")))
    gen_sha_ok = int(sum(1 for v in gen_meta.values() if v.get("sha_ok")))
    first_doc = sorted(gen_meta)[0]
    # decode-проба — best-effort: transformers в shared venv может быть сломан
    # параллельной сессией (14.11 13:35: huggingface-hub 2.4.0). G0-целостность данных
    # держится на sha256; проба — только content-sanity. НЕ чинить venv (чужой).
    try:
        dec = a1.decode_probe(np.asarray(np.load(GEN, allow_pickle=True)[f"{first_doc}_tok"]), first_doc)
        dec_note = f"decode_probe[{first_doc}]: {dec['lines'][0]['text'][:40]!r}..."
    except (ImportError, ModuleNotFoundError, OSError, ValueError) as e:   # v2.1 NIT: шире than ImportError
        dec = {"skipped": str(e).splitlines()[0]}
        dec_note = f"decode_probe SKIPPED ({str(e).splitlines()[0]}) — G0 по sha256"
    logline(f"G0: gen sha {data[os.path.basename(GEN)][:16]} gen_sha_ok {gen_sha_ok}/64; "
            f"own sha {data[os.path.basename(OWN)][:16]}; cor sha {data[os.path.basename(COR)][:16]}; {dec_note}")
    dec_status = "skipped" if isinstance(dec, dict) and "skipped" in dec else "ok"
    provenance_limited = "content_sanity_skipped" if dec_status == "skipped" else None

    docs = {}
    docs.update(a1.load_source(GEN, "gen"))
    docs.update(a1.load_source(OWN, "state"))
    docs.update(a1.load_source(COR, "state"))
    n_docs = len(docs)

    # L1-TPU degraded (то же правило, что AT0_1/AT0_2)
    l1_degraded = {}
    for nm, d in docs.items():
        if not nm.startswith(("own_", "cor_")):
            continue
        u1, hn1 = d["layers"][1]
        U1 = u1 / np.maximum(np.linalg.norm(u1, axis=1, keepdims=True), 1e-12)
        V1, _ = vpath(U1, hn1)
        vc1 = dcurve(V1, np.arange(1, min(97, V1.shape[0])))
        meta_dip1 = d["curves_meta"].get(1, {}).get("dip")
        dev = None if (len(vc1) == 0 or meta_dip1 is None) else float(abs(float(vc1[0]) - meta_dip1))
        l1_degraded[nm] = bool(dev is not None and dev > a1.L1_DEGRADE_TOL)
    logline(f"L1-TPU degraded (dev > {a1.L1_DEGRADE_TOL}): "
            f"{[nm for nm, v in sorted(l1_degraded.items()) if v]}")

    # degen-флаги (ТО ЖЕ правило, что AT0_2: L23 vcurve + token share + wiki_0/4)
    degen_flags = {}
    for nm, d in docs.items():
        u23, hn23 = d["layers"][23]
        U23 = u23 / np.maximum(np.linalg.norm(u23, axis=1, keepdims=True), 1e-12)
        V23, _ = vpath(U23, hn23)
        vc23 = dcurve(V23, np.arange(1, min(97, V23.shape[0])))
        fl, mcos, _ = degen(vc23, d["toks"])
        degen_flags[nm] = {"degen": bool(fl or nm in ("wiki_0", "wiki_4")), "mcos": float(mcos)}

    # --- прогон клеток ---
    cells = {}          # (nm, L) -> cell
    n_absent = 0
    if smoke:
        smoke_names = sorted(nm for nm in docs if nm in ("wiki_0", "math_3"))
        doc_names = {nm: i for i, nm in enumerate(smoke_names)}
    else:
        doc_names = {nm: i for i, nm in enumerate(sorted(docs))}   # ГЛОБАЛЬНЫЙ doc_idx
    env["n_docs_run"] = len(doc_names)
    n_gate = {"closure": 0.0, "add": 0.0, "orth": 0.0, "gram": 0.0, "energy": 0.0}
    for nm, doc_idx in doc_names.items():
        d = docs[nm]
        l1_bad = bool(l1_degraded.get(nm)) and (not smoke)
        for L in LAYERS:
            if l1_bad and L == 1:
                cells[(nm, L)] = {"absent": "L1_degraded"}
                n_absent += 1
                continue
            u, hn = d["layers"][L]
            c = cell_ra(u, hn, doc_idx)
            if "absent" not in c:
                for nm2, x in (("radial", c.get("radial")), ("angular", c.get("angular"))):
                    if isinstance(x, dict) and "gram" in x:
                        if x["gram"] > GRAM_TOL:
                            raise ValueError(f"G1-gram {nm2} {nm} L{L}: {x['gram']:.3e} > {GRAM_TOL:.0e}")
                        n_gate["gram"] = max(n_gate["gram"], x["gram"])   # v2.1 MINOR-1: измеренный max
                n_gate["closure"] = max(n_gate["closure"], c["closure_rel"])
                n_gate["add"] = max(n_gate["add"], c["add_rel"])
                n_gate["orth"] = max(n_gate["orth"], c["orth"])
                if "energy_add_rel" in c:
                    n_gate["energy"] = max(n_gate["energy"], c["energy_add_rel"])
            else:
                n_absent += 1
            cells[(nm, L)] = c

    expected = len(doc_names) * len(LAYERS)
    assert len(cells) == expected, f"G2: клеток {len(cells)} != {expected}"
    if not smoke:
        assert n_absent == 2, f"G2: n_absent = {n_absent} != 2 (own_w0/own_w4 × L1 = 2 клетки)"

    # Пулы (как AT0_2: primary = all − corpus − degen; пин 37)
    corpus_docs = {nm for nm, d in docs.items() if d["source"] == "P2_corpus"}
    degen_docs = {nm for nm, d in docs.items() if degen_flags[nm]["degen"]}
    pools = {"all_96": set(doc_names),
             "primary": set(doc_names) - corpus_docs - degen_docs,
             "loopy": set(doc_names) & degen_docs}
    if not smoke:
        assert len(pools["primary"]) == 37, \
            f"совместимость с AT0_2: primary = {len(pools['primary'])} != 37 (degen-флаг отличается?)"
    logline(f"pools: all_96 {len(pools['all_96'])} / primary verified∩nondeg {len(pools['primary'])} / "
            f"loopy {len(pools['loopy'])} (degen); absent {n_absent}; "
            f"gate max: closure {n_gate['closure']:.2e} / add {n_gate['add']:.2e} / "
            f"orth {n_gate['orth']:.2e} / energy {n_gate['energy']:.2e}")

    # Агрегаты + вердикты per (L, пул) (v2: dominance descriptive + carrier primary)
    agg = {}
    for pidx, (pnm, pset) in enumerate(sorted(pools.items())):
        agg[pnm] = {}
        for L in LAYERS:
            cl = [cells[(nm, L)] for nm in sorted(pset) if (nm, L) in cells
                  and "absent" not in cells[(nm, L)]]
            st = _pool_stats(cl, n_total=len(pset), pool_idx=pidx, L=L)
            st["dominance_verdict"] = _dominance_verdict(st)
            st["verdict"] = _carrier_verdict(st)
            agg[pnm][f"L{L}"] = st
            logline(f"{pnm} L{L}: fdc_R {st.get('radial_fdc_med')} fdc_A {st.get('angular_fdc_med')} "
                    f"frac_R {st.get('radial_frac_dc')} frac_A {st.get('angular_frac_dc')} "
                    f"n_pr {st.get('radial_n_present')}/{st.get('angular_n_present')} "
                    f"dh_frac {st.get('dh_frac_dc')} cR {st.get('contrib_R_med')} "
                    f"cA {st.get('contrib_A_med')} vt_plateau {st.get('vt_plateau_med')} "
                    f"-> {st['verdict']} (dom: {st['dominance_verdict']})")

    # SMOKE: артефакты НЕ пишем (паттерн AT0_2, ревью N2)
    if smoke:
        logline("AT0_2b SMOKE OK (артефакты не пишутся)")
        return

    store = {
        "experiment": "AT0_2b T0.2b (R/A-split: кто несёт DC)",
        "tz": "docs/TZ_add_0-3.md §1.5 T0.2b",
        "prereg": "docs/AT0_2b_prereg.md",
        "meta": {"env": env, "field_sha256": pf_sha, "data_sha256": data,
                 "gen_sha_ok": f"{gen_sha_ok}/64", "decode_probe": dec,
                 "decode_probe_status": dec_status,   # v2 C6
                 "provenance_limited": provenance_limited,   # v2 C6
                 "script_sha256": sha256_file(os.path.join(ROOT, "experiments/AT0_2b_ra_split.py")),
                 "n_docs": n_docs, "n_absent": n_absent, "gate_max": n_gate,
                 "layers": [int(l) for l in LAYERS],
                 "degen_docs": sorted(degen_docs), "corpus_docs": sorted(corpus_docs),
                 "null_note": ("f_dc — MEAN-DIRECTION null (v2 C4; sign-flip Rademacher, B=199): "
                               "перестановка вырождена (mean инвариантен к любому reorder); "
                               "E[fdc|null] = 1/N; ГРАНИЦА: null НЕ сохраняет ACF/telescoping — "
                               "«DC present» = направленный mean drift, НЕ spectral DC (не "
                               "мостить к T0.1/T0.2); домен: doc_idx глобальный по 96, seed = "
                               "2026 + doc_idx*1009 + s; bootstrap-CI: seed_ci = 9001 + "
                               "pool_idx*1009 + L, B_BOOT=199"),
                 "hash_convention": ("v2 C7: store_sha256 = canonical payload hash (canonical "
                                     "JSON sort_keys, без self-поля, поле None при расчёте); "
                                     "store_file_sha256/claim_file_sha256/manifest_file_sha256 = "
                                     "sha256 ФАКТИЧЕСКИХ файлов (manifest не может содержать "
                                     "свой file_sha — только store/claim). v2.1 MINOR-3: ключи "
                                     "data_sha256 = basenames файлов (location-invariant: "
                                     "перенос репо не меняет canonical store sha)"),
                 "vt_plateau_note": "vt_plateau — DESCRIPTIVE anchor (якорь плато T0.1), НЕ часть вердикта (v2 M2)"},
        "cells": {f"{nm}|L{L}": c for (nm, L), c in sorted(cells.items())},
    }
    store["store_sha256"] = None
    store_b = json.dumps(pf.jclean(store), sort_keys=True).encode()
    store_sha = hashlib.sha256(store_b).hexdigest()
    store["store_sha256"] = store_sha
    store_path = os.path.join(STORE_DIR, f"{store_sha[:16]}.json")
    with open(store_path, "w") as f:
        json.dump(pf.jclean(store), f, sort_keys=True)
    store_file_sha = hashlib.sha256(open(store_path, "rb").read()).hexdigest()   # v2 C7
    store["meta"]["store_file_sha256"] = store_file_sha

    claim = {
        "experiment": "AT0_2b T0.2b (R/A-split: кто несёт DC)",
        "round": ("round 2 (v2: review C1–C7/M1–M3; v2.1: MINOR-фиксы review round 2 — "
                  "gram-max, tracked-лог полный, data_sha256 basename-ключи; клетки "
                  "детерминированы сидами — вычисления те же, canonical store sha — новый "
                  "(basename-ключи)"),
        "date": "2026-11-15 (UTC+3)",
        "store": f"{store_sha[:16]}.json",
        "store_sha256": store_sha,
        "store_file_sha256": store_file_sha,
        "meta": store["meta"],
        "prereg": {
            "null": ("mean-direction null (sign-flip Rademacher, B=199, p_floor 0.005); "
                     "DC-присутствие p < 0.05; граница C4: НЕ ACF-matched spectral null"),
            "verdict_rule": ("v2: PRIMARY = carrier_verdict по вКЛАДУ в net displacement Δh: "
                             "contrib_R/A = share + 0.5·cross; gate: dh_frac_dc > 0.5 (у Δh есть "
                             "DC), n_present ≥ 5; CANCELLED: cross_med < −0.5 и обе shares > 0.5; "
                             "RADIAL_CARRIER: CI_low(contrib_R) > 0.5 и med_R > med_A; ANGULAR — "
                             "зеркало; NO_DH_DC; INCONCLUSIVE; not_testable (пре-рег §5 канал 2). "
                             "DESCRIPTIVE = dominance_verdict (старое правило v1: frac>0.5 + "
                             "1.1× медиан f_dc — без CI)"),
            "interpretation": ("RADIAL_CARRIER ⇒ DC-плато Δh = drift/рост нормы ‖h_t‖; "
                               "ANGULAR_CARRIER ⇒ когерентный поворот; NO_DH_DC ⇒ нечего "
                               "разводить (территория T0.5/VAR; v1-самокоррекция: f_dc(Δh) ≈ "
                               "1.5e-5, плато T0.1 — персистентность v_tangent, не DC Δh)"),
            "cross_lags": "ОЧЕРЕДНО (ТЗ: «отчитать»), вердикта нет",
        },
        "gates": {"gate_max": n_gate, "n_absent": n_absent,
                  "note": ("closure REL ≤1e-12 (ТЗ G6′), additivity REL ≤1e-12, "
                           "energy-additivity REL ≤1e-12 (v2 M1), orth ≤1e-9, gram(G1) < 1e-10 "
                           "на R/A; mean-direction-калибровка в main (правило 6)"),
                  "provenance_limited": provenance_limited},   # v2 C6
        "aggregates": agg,
        "pool_note": ("pools как в AT0_2: all_96 (sensitivity), primary = verified ∩ nondeg "
                      "(37, пин совместимости), loopy (degen). Δh gen — provenance_limited "
                      "(hn из файла, scale-пин G0a — тот же остаточный риск, что h_raw AT0_2); "
                      "own/corpus — hn = ‖raw h‖ (верифицировано конструкцией). "
                      "vt_plateau — descriptive anchor (v2 M2)."),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(pf.jclean(claim), f, sort_keys=True, indent=1)
    claim_file_sha = hashlib.sha256(open(OUT_JSON, "rb").read()).hexdigest()   # v2 C7
    man = {
        "experiment": "AT0_2b T0.2b",
        "round": "round 2",
        "date": "2026-11-15 (UTC+3)",
        "claim_sha256": claim_file_sha,
        "claim_file_sha256": claim_file_sha,
        "store_sha256": store_sha,
        "store_file_sha256": store_file_sha,
        "script_sha256": store["meta"]["script_sha256"],
        "hash_convention": claim["meta"]["hash_convention"],
        "note": ("claim AT0_2b_RA.json (gates+prereg+aggregates, v2: carrier/dominance verdicts, "
                 "n_present, CI, p_fdc_dh); store = per-cell (rho96 R/A, cross k=1..K, p_fdc, "
                 "p_fdc_dh, contrib, closure); run_log = stdout (tracked)."),
    }
    man_b = json.dumps(man, sort_keys=True).encode()
    man["manifest_sha256"] = hashlib.sha256(man_b).hexdigest()
    with open(MAN_JSON, "w") as f:
        json.dump(man, f, indent=1)
    man_file_sha = hashlib.sha256(open(MAN_JSON, "rb").read()).hexdigest()   # v2 C7 (в log)
    logline(f"store: {store_path} (canonical sha {store_sha[:16]}; file sha {store_file_sha[:16]})")
    logline(f"written: {OUT_JSON} (file sha {claim_file_sha[:16]}), {MAN_JSON} (file sha "
            f"{man_file_sha[:16]}), {RUN_LOG}")
    logline("AT0_2b gates: ALL PASS")
    with open(RUN_LOG, "w") as f:   # v2.1 MINOR-2: ПОСЛЕ всех logline (tracked лог = полный stdout)
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
