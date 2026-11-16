#!/usr/bin/env python3
"""at0_1_r7_annotation.py — post-hoc аннотация AT0_1: claim r6 → r7
(docs/AT0_1_review.md, round 2: C1–C7). НЕ экспериментальный прогон: чисел прогона не
пересчитывается; все изменения — детерминированные пересчёты из закреплённых (sha-пин)
входов + аннотация. Само-аудит — ASSERT'ы (не stdout): любой сбой ⇒ стоп, на диске р7
артефакта нет.

Изменения (C1–C7):
  C1: null_analytic_ma1: старый z = (s_doc + f_dc)/se = mean_band/se тестировал null
      «mean_band = 0»; при matched MA(1) с DC: E[rho_Tk(k≥2)] = ‖m‖²/(‖m‖²+var) = f_dc
      ⇒ E[mean_band] ≈ f_dc, E[s_doc] ≈ 0 ⇒ z_excess = s_doc/se — тест «плато на уровне
      f_dc». Старая статистика — z_raw_band (диагностика). E_doc переписан.
  C2: G0a_tpu_hn_field.max_rel_dev_reconstruction → reconstruction_roundoff_identity:
      U·hn ≡ h тождественно в обеих ветках max(·,1e-12) — измеряет только round-trip
      divide→multiply (~4e-16), провалить не может (кроме NaN/inf) — тождество, НЕ
      provenance-гейт; годность TPU-рядов устанавливает l23_meta_vs_recomputed.
  C3: meta.r7_annotation.chain — r5/r6/r7 claim shas + run/annotation-скрипт shas; r5-claim
      восстановлен перепуском de305869 (threads=4: bit-eq store 23281c64 + run.log);
      полная сверка r5→r6 (diff = ровно декларированные r6-изменения) и r6→r7 — ассертится.
  C4: se_eff = se·sqrt((1+2ρ₁²)·D/D_eff): Bartlett (1+2ρ₁²), ρ₁ = θ/(1+θ²) (matched null);
      D_eff = (Σλ)²/Σλ² participation ratio ковариации строк объекта (zero-mean, L23, per-doc).
      Ambient D=1024 — нижняя граница (se ≤ se_eff). z_excess_eff — отчёт.
  C5: SE_formula: /98 → /88 (= K_MAX−8, 88 лагов; код всегда считал /88, текст был неверен).
  C6: T_none_full — из excess_median/excess_mean в отдельный блок identity_check_G1
      (Gram-тождество (r−1)/2 − (N−1)·f_dc — функция (r, N, f_dc), не независимый excess);
      заголовочный T0 — r/f_dc per object (r_median/f_dc_median в том же блоке).
  C7: self-audit расширен: + l23_meta_vs_recomputed (max_abs_dev_by_source, n_docs,
      worst_doc) + roundoff-идентичность + per-doc theta + z_raw_band = r5-статистика
      (bit-eq) + T_none_full (bit-eq) + pool П4 — все assert'ы.

Входы (sha-пinned):
  claim r5: results/AT0/store/claim_r5_d4a4b101.json   (перепуск de305869, bit-eq store+log)
  claim r6: results/AT0/store/claim_r6_b0f74312.json
  store:    results/AT0/store/23281c64cc418ff1.json
  npz:      /data/prism_kernels/exp62/gen_layers_qwen.npz (f142f297…)
            results/e64/tpu_v8/P1_own.npz (783e1633…), P2_corpus.npz (5f42a084…)
  meta:     results/e64/tpu_v8/P1_own_meta.json, P2_corpus_meta.json
Выходы:
  results/AT0/T0_ACCOUNTING.json        — r7 claim (компактный, формат r5/r6)
  results/AT0/AT0_1_manifest.json       — r7 manifest (claim + manifest-блок, indent=1)
  results/AT0/store/claim_r7_<sha>.json — пин r7
  stdout: self-audit отчёт (JSON)
Запуск: .venv/bin/python experiments/at0_1_r7_annotation.py
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import importlib.util  # noqa: E402

_fspec = importlib.util.spec_from_file_location(
    "prism_field_pinned", os.path.join(ROOT, "prism/field.py"))
pf = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(pf)
sys.modules["prism_field_pinned"] = pf
jclean, vpath, dcurve = pf.jclean, pf.vpath, pf.dcurve

K_MAX = 96
LAYERS = [1, 5, 11, 17, 23]
OBJECTS = ("h_raw", "du_unit", "v_tangent")

CLAIM_R5 = os.path.join(ROOT, "results/AT0/store/claim_r5_d4a4b101.json")
CLAIM_R6 = os.path.join(ROOT, "results/AT0/store/claim_r6_b0f74312.json")
STORE = os.path.join(ROOT, "results/AT0/store/23281c64cc418ff1.json")
GEN = "/data/prism_kernels/exp62/gen_layers_qwen.npz"
OWN = os.path.join(ROOT, "results/e64/tpu_v8/P1_own.npz")
COR = os.path.join(ROOT, "results/e64/tpu_v8/P2_corpus.npz")
OWN_META = os.path.join(ROOT, "results/e64/tpu_v8/P1_own_meta.json")
COR_META = os.path.join(ROOT, "results/e64/tpu_v8/P2_corpus_meta.json")
OUT_DIR = os.path.join(ROOT, "results/AT0")
OUT_CLAIM = os.path.join(OUT_DIR, "T0_ACCOUNTING.json")
OUT_MAN = os.path.join(OUT_DIR, "AT0_1_manifest.json")

R5_SHA = "d4a4b10135e8b63f361f6fe75475cff7d44b6810e9248d03a8bdf96b5b73d94a"
R6_SHA = "b0f74312416285207ba99ae93ee75ecc446938b8b61ba82a19a04ac36b913994"
STORE_SHA = "23281c64cc418ff1fedad2a5013c4bf7c8b7dd7680ac8d4fa7cf746c62fd1732"
GEN_SHA = "f142f297ec16209d1520cf6f4ee46fdd2543b0d73f566a69d93c26771349fdd7"
OWN_SHA = "783e16338cdcb652198722a33ecec8a6e8e98fa3c611c687b17db722b0d58198"
COR_SHA = "5f42a08442a5d3bcb8d7c4de5c9658340c0378a917a5e633b9b2aa589ceac125"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- diff
def leaf_diff(a, b, path=""):
    """Рекурсивный diff: (added, removed, changed) — лист-пути. added: есть в b, нет в a."""
    added, removed, changed = [], [], []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b:
                removed.append(f"{path}.{k}" if path else str(k))
            else:
                added_k, removed_k, changed_k = leaf_diff(a[k], b[k], f"{path}.{k}" if path else str(k))
                added += added_k; removed += removed_k; changed += changed_k
        for k in b:
            if k not in a:
                added.append(f"{path}.{k}" if path else str(k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            changed.append(f"{path}.__len__")
        for i, (x, y) in enumerate(zip(a, b)):
            added_i, removed_i, changed_i = leaf_diff(x, y, f"{path}[{i}]")
            added += added_i; removed += removed_i; changed += changed_i
    else:
        if a != b:
            changed.append(path)
    return added, removed, changed


def patterns_match(paths, patterns):
    """Каждый path ∈ paths совпадает с ≥1 pattern И каждый pattern покрыт ≥1 path.
    Pattern-подстановки: {obj} → h_raw|du_unit|v_tangent, * — суффикс. Точки — литеральные."""
    import re
    objs = "(h_raw|du_unit|v_tangent)"
    ok = True
    for p in paths:
        if not any(re.fullmatch(pat.replace(".", "\\.").replace("{obj}", objs).replace("*", ".*"), p)
                   for pat in patterns):
            ok = False
            print(f"  НЕ ОЖИДАЕМЫЙ PATH: {p}")
    for pat in patterns:
        if not any(re.fullmatch(pat.replace(".", "\\.").replace("{obj}", objs).replace("*", ".*"), p)
                   for p in paths):
            ok = False
            print(f"  ОЖИДАЕМЫЙ PATH НЕ ПОКРЫТ: {pat}")
    return ok


# --------------------------------------------------------------------------- loads
def load23(nm):
    """Зеркало load_source(kind=gen|state) + cell-loop (AT0_1_accounting.py:163-197, 264-276):
    rows L23 → (U, hn) → объекты h_raw=Δh, du_unit=ΔU, v_tangent=V (vpath)."""
    if nm.startswith("own_"):
        path, isgen = OWN, False
    elif nm.startswith("cor_"):
        path, isgen = COR, False
    else:
        path, isgen = GEN, True
    z = np.load(path, allow_pickle=True)
    if isgen:
        u = np.asarray(z[f"{nm}_L23_u"], dtype=np.float64)
        hn = np.asarray(z[f"{nm}_L23_hn"], dtype=np.float64)
    else:
        hraw = np.asarray(z[f"{nm}_L23"], dtype=np.float64)
        hn = np.linalg.norm(hraw, axis=1)
        u = hraw / np.maximum(hn, 1e-12)[:, None]
    U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    h = U * hn[:, None]
    V, _ = vpath(U, hn)
    return {"h_raw": h[1:] - h[:-1], "du_unit": U[1:] - U[:-1], "v_tangent": V}


def deff_rows(x):
    """D_eff = (Σλ)²/Σλ² — participation ratio ковариации строк (zero-mean часть)."""
    xc = x - x.mean(axis=0)
    w = np.linalg.eigvalsh(xc.T @ xc)
    w = w[w > 0]
    assert w.size > 0 and float((w ** 2).sum()) > 0.0, \
        f"deff_rows: вырожденная ковариация (w.size={w.size}) — D_eff неопределён"
    return float(w.sum() ** 2 / (w ** 2).sum())


# --------------------------------------------------------------------------- self-audits
def audit_g0a_evidence(claim):
    """C7 (было в r6): 4 числа evidence_own_w0 — bit-eq воспроизведение."""
    z = np.load(OWN, allow_pickle=True)
    _hn0 = np.asarray(z["own_w0_hn"], dtype=np.float64)
    _col = np.linalg.norm(np.asarray(z["own_w0_L23"], dtype=np.float64)[:, :len(_hn0)], axis=0)
    row_norms = np.linalg.norm(np.asarray(z["own_w0_L23"], dtype=np.float64), axis=1)
    rec = {
        "row_norm_median": float(np.median(row_norms)),
        "hn_median": float(np.median(_hn0)),
        "hn_vs_column_norms_max_rel": float(np.max(np.abs(_hn0 / _col - 1.0))),
        "row_norm_vs_hn_corr": float(np.corrcoef(row_norms, _hn0)[0, 1]),
    }
    cl = claim["gates"]["G0a_tpu_hn_field"]["evidence_own_w0"]
    assert all(rec[k] == cl[k] for k in rec), f"C7: G0a evidence не воспроизведён: {rec} vs {cl}"
    return rec


def audit_roundoff_identity(claim):
    """C2: roundoff-идентичность U·hn vs h — bit-eq воспроизведение r6-значений."""
    overall = (0.0, None, None)
    by_src, by_layer = {}, {}
    for src, path, prefix in (("P1_own", OWN, "own_w"), ("P2_corpus", COR, "cor_w")):
        z = np.load(path, allow_pickle=True)
        for i in range(16):
            n = f"{prefix}{i}"
            for L in LAYERS:
                h = np.asarray(z[f"{n}_L{L}"], dtype=np.float64)
                hn = np.linalg.norm(h, axis=1)
                u = h / np.maximum(hn, 1e-12)[:, None]
                U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
                h_rec = U * hn[:, None]
                rel = np.linalg.norm(h_rec - h, axis=1) / np.maximum(np.linalg.norm(h, axis=1), 1e-300)
                i_mx = int(np.argmax(rel))
                if float(rel[i_mx]) > overall[0]:
                    overall = (float(rel[i_mx]), n, L)
                by_src.setdefault(src, []).append(float(rel.max()))
                by_layer.setdefault(f"L{L}", []).append(float(rel.max()))
    ref_val = claim["gates"]["G0a_tpu_hn_field"]["max_rel_dev_reconstruction"]
    ref = claim["gates"]["G0a_tpu_hn_field"]["max_rel_dev_reconstruction_detail"]
    assert overall[0] == ref_val, f"C2: roundoff max не воспроизведён: {overall[0]} vs {ref_val}"
    assert overall[1] == ref["worst"]["doc"] and overall[2] == ref["worst"]["L"], \
        f"C2: worst не воспроизведён: {overall[1:]} vs {ref['worst']}"
    for k in by_src:
        assert max(by_src[k]) == ref["max_by_source"][k], f"C2: by_source {k} не воспроизведён"
    for k in by_layer:
        assert max(by_layer[k]) == ref["max_by_layer"][k], f"C2: by_layer {k} не воспроизведён"
    return {
        "max_rel_dev": overall[0],
        "worst": {"doc": overall[1], "L": overall[2]},
        "max_by_source": {k: max(v) for k, v in sorted(by_src.items())},
        "max_by_layer": {k: max(v) for k, v in sorted(by_layer.items())},
    }


def audit_l23_meta(claim):
    """C7: l23_meta_vs_recomputed — bit-eq воспроизведение (зеркало gate_g0a_tpu,
    AT0_1_accounting.py:393-418): per TPU-док dip_V,med (k=1) recomputed vs meta curves.23.dip."""
    pairs = [(f"own_w{i}", OWN, OWN_META) for i in range(16)] + \
            [(f"cor_w{i}", COR, COR_META) for i in range(16)]
    dev_l23 = {}
    for nm, path, meta_path in pairs:
        z = np.load(path, allow_pickle=True)
        m = json.load(open(meta_path, encoding="utf-8"))
        hraw = np.asarray(z[f"{nm}_L23"], dtype=np.float64)
        hn = np.linalg.norm(hraw, axis=1)
        u = hraw / np.maximum(hn, 1e-12)[:, None]
        U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        V, _ = vpath(U, hn)
        vc = dcurve(V, np.arange(1, min(K_MAX + 1, V.shape[0])))
        meta_dip = m.get(nm, {}).get("curves", {}).get("23", {}).get("dip")
        if meta_dip is None or len(vc) == 0:
            continue
        dev_l23[nm] = abs(float(vc[0]) - float(meta_dip))
    by_src = {}
    for nm, d in dev_l23.items():
        by_src.setdefault("P1_own" if nm.startswith("own_") else "P2_corpus", []).append(d)
    rec = {
        "max_abs_dev_by_source": {k: float(max(v)) for k, v in sorted(by_src.items())},
        "n_docs": {k: len(v) for k, v in sorted(by_src.items())},
        "worst_doc": max(dev_l23, key=lambda k: dev_l23[k]),
    }
    ref = claim["gates"]["G0a_tpu_hn_field"]["l23_meta_vs_recomputed"]
    for k in ("max_abs_dev_by_source", "n_docs"):
        assert rec[k] == ref[k], f"C7: l23_meta {k} не воспроизведён: {rec[k]} vs {ref[k]}"
    assert rec["worst_doc"] == ref["worst_doc"], f"C7: worst_doc {rec['worst_doc']} vs {ref['worst_doc']}"
    return rec


def audit_pool_and_theta(claim, store, pool):
    """C7: П4-пул + per-doc theta (bit-eq со store null_ma1_z) + z_raw_band = r5-статистика."""
    nondeg = [nm for nm in sorted(store["per_doc"]) if not store["per_doc"][nm]["degen_tz"]]
    pool_calc = [nm for nm in nondeg
                 if all(store["per_doc"][nm]["layers"]["23"][o]["mean"]["K"] == K_MAX
                        for o in OBJECTS)]
    assert pool_calc == pool, f"C7: П4-пул не воспроизведён: n={len(pool_calc)} vs {len(pool)}"
    z_raw_check = {}
    for obj in OBJECTS:
        for nm in pool:
            c = store["per_doc"][nm]["layers"]["23"][obj]["mean"]
            rt = np.asarray(c["rho_Tk"])
            n, D = c["N"], c["D"]
            rho1 = float(np.asarray(c["rho"])[0])
            se = float(np.sqrt(np.sum(1.0 / ((n - np.arange(9, K_MAX + 1)) * D))) / (K_MAX - 8))
            th = 0.0 if rho1 == 0.0 else (1.0 - float(np.sqrt(1.0 - 4.0 * rho1 ** 2))) / (2.0 * rho1)
            nz = store["per_doc"][nm]["null_ma1_z"][obj]
            assert th == nz["theta"], f"C7: theta({nm},{obj}) не воспроизведён: {th} vs {nz['theta']}"
            z_old = float((float(np.mean(rt[8:K_MAX]) - c["f_dc"]) + c["f_dc"]) / se)
            assert z_old == nz["z"], f"C7: z_raw_band({nm},{obj}) != r5-статистика: {z_old} vs {nz['z']}"
            z_raw_check[(obj, nm)] = z_old
    return z_raw_check


def audit_tnone(claim, store):
    """C6/C7: T_none_full median/mean per obj — bit-eq со значением в r6-claim;
    + max |sum_rho_full − (r−1)/2| per obj (Gram-тождество)."""
    rec = {}
    for obj in OBJECTS:
        vals, gram = [], 0.0
        for nm in sorted(store["per_doc"]):
            d = store["per_doc"][nm]
            if d.get("degen_tz"):
                continue
            c = d["layers"]["23"][obj]["mean"]
            if c.get("sum_rho_full") is None or c.get("f_dc") is None or c.get("N") is None:
                continue
            vals.append(c["sum_rho_full"] - (c["N"] - 1) * c["f_dc"])
            gram = max(gram, abs(c["sum_rho_full"] - (c["r"] - 1) / 2))
        med, mean = float(np.median(vals)), float(np.mean(vals))
        ref_m = claim["aggregates"]["t0_headline"][obj]["excess_median"]["T_none_full"]
        ref_mean = claim["aggregates"]["t0_headline"][obj]["excess_mean"]["T_none_full"]
        assert med == ref_m, f"C6: T_none_full median({obj}) не воспроизведён: {med} vs {ref_m}"
        assert mean == ref_mean, f"C6: T_none_full mean({obj}) не воспроизведён: {mean} vs {ref_mean}"
        rec[obj] = {"T_none_full_median": med, "T_none_full_mean": mean,
                    "max_abs_sum_rho_full_minus_(r-1)/2": float(gram), "n_docs": len(vals)}
    return rec


# --------------------------------------------------------------------------- C1/C4
def ma1_recompute(claim, store, pool, z_raw_check):
    """C1+C4: per-doc z_excess / z_raw_band / z_excess_eff + D_eff; агрегаты per obj."""
    out = {}
    per_obj_doc = {obj: {} for obj in OBJECTS}
    for nm in pool:
        xdoc = load23(nm)   # один раз на док (3 объекта)
        for obj in OBJECTS:
            c = store["per_doc"][nm]["layers"]["23"][obj]["mean"]
            x = xdoc[obj]
            n, D = c["N"], c["D"]
            rt = np.asarray(c["rho_Tk"])
            mb = float(np.mean(rt[8:K_MAX]))
            s_doc = mb - c["f_dc"]
            se = float(np.sqrt(np.sum(1.0 / ((n - np.arange(9, K_MAX + 1)) * D))) / (K_MAX - 8))
            th = store["per_doc"][nm]["null_ma1_z"][obj]["theta"]
            zex = s_doc / se
            zraw = z_raw_check[(obj, nm)]
            de = deff_rows(x)
            if th is None:
                per_obj_doc[obj][nm] = {"z_excess": zex, "z_raw_band": zraw,
                                        "z_excess_eff": None, "d_eff": de}
                out.setdefault(obj, {"zex": [], "zraw": [], "zeff": [], "de": [], "fc": []})
                out[obj]["zex"].append(zex); out[obj]["zraw"].append(zraw)
                continue
            rho1n = th / (1.0 + th ** 2)
            fct = float(np.sqrt((1.0 + 2.0 * rho1n ** 2) * D / de))
            per_obj_doc[obj][nm] = {"z_excess": zex, "z_raw_band": zraw,
                                    "z_excess_eff": s_doc / (se * fct), "d_eff": de}
            out.setdefault(obj, {"zex": [], "zraw": [], "zeff": [], "de": [], "fc": []})
            out[obj]["zex"].append(zex); out[obj]["zraw"].append(zraw)
            out[obj]["zeff"].append(s_doc / (se * fct)); out[obj]["de"].append(de)
            out[obj]["fc"].append(fct)
    for obj in OBJECTS:
        zex, zraw, zeff = (np.array(out[obj][k]) for k in ("zex", "zraw", "zeff"))
        de, fc = out[obj]["de"], out[obj]["fc"]
        assert len(de) > 0, f"obj {obj}: нет matched-доков (все |rho1| > 0.5) — агрегаты se_eff не вычисляются"
        out[obj] = {
            "z_excess": {
                "median": float(np.median(zex)), "mean": float(np.mean(zex)),
                "n_abs_z_ge_3": int(np.sum(np.abs(zex) >= 3.0)),
                "def": "s_doc/se, s_doc = mean_{k=9..96} rho_Tk − f_dc; null: E[mean_band] = f_dc (C1)",
            },
            "z_excess_eff": {
                "median": float(np.median(zeff)), "mean": float(np.mean(zeff)),
                "n_abs_z_ge_3": int(np.sum(np.abs(zeff) >= 3.0)),
                "def": "s_doc/se_eff, se_eff = se·sqrt((1+2ρ₁²)·D/D_eff) (C4; ambient D — нижняя граница)",
            },
            "z_raw_band": {
                "median": float(np.median(zraw)), "mean": float(np.mean(zraw)),
                "n_abs_z_ge_3": int(np.sum(np.abs(zraw) >= 3.0)),
                "def": "mean_band/se — старая (r5/r6) статистика, сохранена диагностикой (C1); "
                       "bit-eq со store per_doc.null_ma1_z.z (self-audit)",
            },
            "d_eff": {"min": float(min(de)), "median": float(np.median(de)),
                      "max": float(max(de)),
                      "def": "(Σλ)²/Σλ² participation ratio ковариации строк объекта (zero-mean, L23) (C4)"},
            "se_factor_median": float(np.median(fc)),
            "per_doc": per_obj_doc[obj],
        }
    return out


# --------------------------------------------------------------------------- main
def main():
    report = {}

    # ---- 0. входы: sha-верификация ----
    shas = {
        "claim_r5": sha256_file(CLAIM_R5), "claim_r6": sha256_file(CLAIM_R6),
        "store": sha256_file(STORE), "gen_npz": sha256_file(GEN),
        "own_npz": sha256_file(OWN), "cor_npz": sha256_file(COR),
    }
    assert shas["claim_r5"] == R5_SHA, f"r5-claim sha: {shas['claim_r5']} != {R5_SHA}"
    assert shas["claim_r6"] == R6_SHA, f"r6-claim sha: {shas['claim_r6']} != {R6_SHA}"
    assert shas["store"] == STORE_SHA, f"store sha: {shas['store']} != {STORE_SHA}"
    assert shas["gen_npz"] == GEN_SHA and shas["own_npz"] == OWN_SHA and shas["cor_npz"] == COR_SHA, \
        f"npz sha mismatch: {shas}"
    claim_r5 = json.load(open(CLAIM_R5, encoding="utf-8"))
    claim_r6 = json.load(open(CLAIM_R6, encoding="utf-8"))
    store = json.load(open(STORE, encoding="utf-8"))
    assert claim_r5["store"]["sha256"] == STORE_SHA and claim_r6["store"]["sha256"] == STORE_SHA
    assert claim_r6["meta"]["script_sha256"] == "de3058693022320876da8028c4e63dd1ef57ca8fe9dc565f08f49d396a763f20"
    report["inputs_sha256"] = shas

    # ---- 1. C3: сверка r5→r6 (diff = ровно декларированные r6-изменения) ----
    a, r, c = leaf_diff(claim_r5, claim_r6)
    R6_ADDED = [
        "prereg.4_plateau_vs_fDC.{obj}.direction",
        "prereg.4_plateau_vs_fDC.{obj}.direction_note",
        "prereg.4_plateau_vs_fDC.v_tangent.null_analytic_ma1.applicability",
        "gates.G0a_tpu_hn_field.max_rel_dev_reconstruction",
        "gates.G0a_tpu_hn_field.max_rel_dev_reconstruction_detail*",
        "gates.G8_gram.n_l1_absent_docs",
        "gates.G8_gram.n_l1_absent_cells",
        "gates.G8_gram.n_l1_note",
        "meta.r6_annotation*",
    ]
    R6_CHANGED = ["aggregates.t0_headline_note"]
    assert not r, f"C3: r5→r6 удаляет ключи: {r}"
    assert patterns_match(a, R6_ADDED), f"C3: r5→r6 added: {a}"
    assert patterns_match(c, R6_CHANGED), f"C3: r5→r6 changed: {c}"
    # r5-claim: нет полей r6-аннотации (инвариант)
    assert "r6_annotation" not in claim_r5["meta"]
    report["r5_to_r6_diff"] = {
        "added": a, "changed": c,
        "ok": True,
        "note": "diff = ровно декларированные r6-изменения (r6_annotation.changes, manifest r6) — assert",
    }

    # ---- 2. self-audits (C2/C6/C7) ----
    pool = [nm for nm in sorted(store["per_doc"]) if not store["per_doc"][nm]["degen_tz"]
            and all(store["per_doc"][nm]["layers"]["23"][o]["mean"]["K"] == K_MAX for o in OBJECTS)]
    assert len(pool) == 51, f"П4-пул: n={len(pool)} != 51"
    g0a = audit_g0a_evidence(claim_r6)
    rnd = audit_roundoff_identity(claim_r6)
    l23 = audit_l23_meta(claim_r6)
    z_raw_check = audit_pool_and_theta(claim_r6, store, pool)
    tnone = audit_tnone(claim_r6, store)
    report["self_audit"] = {
        "g0a_evidence_own_w0_bit_eq": g0a,
        "roundoff_identity_bit_eq": rnd,
        "l23_meta_vs_recomputed_bit_eq": l23,
        "pool4_n": len(pool),
        "theta_bit_eq": "asserted (per doc × obj)",
        "z_raw_band_eq_r5_statistic": "asserted (per doc × obj)",
        "tnone_bit_eq": "asserted (per obj)",
    }

    # ---- 3. r7 claim = r6 + C1–C7 ----
    claim = json.loads(json.dumps(claim_r6))  # deep copy

    # C1 + C4 + C5: null_analytic_ma1
    ma1 = ma1_recompute(claim, store, pool, z_raw_check)
    for obj in OBJECTS:
        m = claim["prereg"]["4_plateau_vs_fDC"][obj]["null_analytic_ma1"]
        m["z_excess"] = ma1[obj]["z_excess"]
        m["z_excess_eff"] = ma1[obj]["z_excess_eff"]
        m["z_raw_band"] = ma1[obj]["z_raw_band"]
        m["d_eff"] = ma1[obj]["d_eff"]
        m["se_factor_median"] = ma1[obj]["se_factor_median"]
        m["E_doc"] = ("s_doc = mean_band − f_dc: E ≈ 0 (matched MA(1) с DC: ρ_acf(k≥2) = 0 ⇒ "
                      "E[rho_Tk(k≥2)] = ‖m‖²/(‖m‖²+var) = f_dc ⇒ E[mean_band] ≈ f_dc; r7, C1: "
                      "формулировка r5/r6 «E[mean_band] = 0» неверна для T-k нормировки — DC не "
                      "сокращается: c(0) = (‖m‖²+var)/D, c(k≥2) ≈ ‖m‖²/D)")
        m["SE_formula"] = ("sqrt(Σ_{k=9..96} 1/((N−k)·D)) / 88 — первый порядок, per-dim гауссов "
                           "MA(1), D = скрытое измерение (r7, C5: делитель = K_MAX−8 = 88 лагов; в "
                           "r5/r6 ошибочно записано /98; утверждение «Bartlett-поправка нулевая для "
                           "k > p+1» неверно: var(ρ̂_k) ≈ (1/N)·Σ_j ρ(j)² = (1/N)(1+2ρ₁²) для MA(1))")
        m["SE_eff_formula"] = ("se_eff = se·sqrt((1+2ρ₁²)·D/D_eff) — r7, C4: Bartlett-фактор "
                               "(1+2ρ₁²), ρ₁ = θ/(1+θ²) (matched null); D_eff = (Σλ)²/Σλ² "
                               "participation ratio ковариации строк объекта (zero-mean, L23, per-doc, "
                               "поле d_eff). Ambient D=1024 — нижняя граница (se ≤ se_eff); "
                               "р.178 r_eff(0.05) из addentum — другая метрика (5%-порог ранга), не "
                               "использована")
        m["c1_c4_note"] = ("r7 (docs/AT0_1_review.md, C1/C4): r5/r6 z = (s_doc + f_dc)/se = "
                           "mean_band/se тестировал null «mean_band = 0»; при matched MA(1) с DC "
                           "E[mean_band] ≈ f_dc ⇒ E[s_doc] ≈ 0 ⇒ z_excess = s_doc/se (тест «плато на "
                           "уровне f_dc»). Для v_tangent (f_dc_median ≈ 0.08) старый z (median 381) — "
                           "by-product DC в числителе (r6 applicability), не plateau-сигнал; "
                           "z_excess_ambient ≈ 10.5 (р. C1 ручной счёт), se_eff ×≈5.5 ⇒ "
                           "z_excess_eff ≈ 1.9 — описательно (вердикт — по первичному каналу, r5 D2). "
                           "Треды (round 3 MINOR-2): per_doc d_eff/z_excess_eff и se_factor_median — "
                           "bit-eq только в pinned-окружении (run_one.sh: OpenBLAS/MKL_NUM_THREADS=6; "
                           "r7-прогон: OMP_NUM_THREADS=4 — eigvalsh чувствителен к OpenBLAS/MKL-тредам); "
                           "в другом thread-окружении дрейф 1–2 ulp (≤3.4e-15 rel, проверено "
                           "независимо, ревью round 3). z_raw_band/z_excess/θ/T_none_full/roundoff/"
                           "l23_meta — bit-eq в обоих окружениях. Round 3 MINOR-5/6: guard th is None "
                           "(no_match-док в пуле) — defensive, не достигается на данных (все 51 доки "
                           "пула matched); проверен exec'ом реальной ma1_recompute с имитированным "
                           "no_match. Поля median_z/mean_z/n_abs_z_ge_3 (r5) bit-eq "
                           "z_raw_band.median/mean/n_abs_z_ge_3 — дубль сохранён инвариантом "
                           "r5-листьев; цитировать z_raw_band.")
        m["per_doc_ma1"] = ma1[obj]["per_doc"]

    # C2: tpu_dev → reconstruction_roundoff_identity
    g0a_new = claim["gates"]["G0a_tpu_hn_field"]
    det = g0a_new.pop("max_rel_dev_reconstruction_detail")
    g0a_new.pop("max_rel_dev_reconstruction")
    det["status"] = ("тождество (roundoff), НЕ provenance-гейт (r7, C2): U·hn ≡ h в обеих ветках "
                     "max(·,1e-12) (hn>1e-12: U=u=h/hn ⇒ U·hn=h; hn≤1e-12: U=h/hn ⇒ U·hn=h) — "
                     "измеряет только round-trip divide→multiply, провалить не может (кроме NaN/inf); "
                     "годность TPU-рядов устанавливает l23_meta_vs_recomputed (max dev 2.08e-4 ≤ 1e-3), "
                     "не эта величина")
    det["provenance"] = ("post-hoc детерминированный пересчёт r7 (2026-11-15) из TPU-npz "
                         "(входы sha256 ниже); bit-eq воспроизведение r6-значений — assert "
                         "(r6: experiments/at0_1_tpu_dev_recompute.py, тавтология отмечена r7, C2)")
    g0a_new["reconstruction_roundoff_identity"] = rnd["max_rel_dev"]
    g0a_new["reconstruction_roundoff_identity_detail"] = det

    # C6: T_none_full → identity_check_G1
    for obj in OBJECTS:
        th = claim["aggregates"]["t0_headline"][obj]
        th["excess_median"].pop("T_none_full")
        th["excess_mean"].pop("T_none_full")
        th["identity_check_G1"] = {
            "T_none_full_median": tnone[obj]["T_none_full_median"],
            "T_none_full_mean": tnone[obj]["T_none_full_mean"],
            "identity": ("T_none_full = (r−1)/2 − (N−1)·f_dc — Gram-тождество (G8): функция "
                         "(r, N, f_dc), НЕ независимый excess по всем лагам (r7, C6; r6 — note в "
                         "том же dict, r7 — вынесен в отдельный блок)"),
            "max_abs_sum_rho_full_minus_(r-1)/2": tnone[obj]["max_abs_sum_rho_full_minus_(r-1)/2"],
            "n_docs": tnone[obj]["n_docs"],
            "headline": ("заголовочный T0 — r per object: r_median, f_dc_median (в блоке "
                         "t0_headline); r/f_dc ≡ N (G3-тождество r = N·f_dc) — тривиально, не "
                         "цитировать; информативно: r_v ≈ 24.9 (not DC-free), r_du ≈ 0.0048 "
                         "(DC-suppressed); частичные excessы k=1..96 — столбцы T_none_K96/Tk_none_K96/"
                         "T_bart_K96/Tk_bart_K96"),
        }
    claim["aggregates"]["t0_headline_note"] = (
        claim["aggregates"]["t0_headline_note"] +
        " r7 (docs/AT0_1_review.md, C6): T_none_full вынесен из excess_median/excess_mean в "
        "identity_check_G1 (Gram-тождество, не независимый excess); заголовочный ответ T0 — "
        "r per object (r/f_dc ≡ N — G3-тождество, тривиально; r_v ≈ 24.9 — not DC-free; r_du ≈ "
        "0.0048 — DC-suppressed).")

    # C3: meta.r7_annotation (chain + изменения + инвариант + self-audit)
    claim["meta"]["r7_annotation"] = {
        "date": "2026-11-15",
        "review": "docs/AT0_1_review.md (round 2: C1–C7)",
        "chain": {
            "run_script_sha256": "de3058693022320876da8028c4e63dd1ef57ca8fe9dc565f08f49d396a763f20",
            "r5_claim_sha256": R5_SHA,
            "r5_claim_note": ("восстановлен перепуском de305869 (env r5: python 3.12.9/np 2.5.3, "
                              "threads=4, TZ 67addba/647c03ab): bit-eq store 23281c64cc418ff1 + "
                              "run.log e6dcd6fc62aeced0; пин results/AT0/store/claim_r5_d4a4b101.json; "
                              "sha ранее был записан только в meta.r6_annotation.base_run (без пин-файла) "
                              "— р7 закрепляет файл и воспроизведение"),
            "r6_claim_sha256": R6_SHA,
            "r6_claim_note": ("пин results/AT0/store/claim_r6_b0f74312.json; diff r5→r6 = ровно "
                              "декларированные r6-изменения (assert в r7-скрипте); r6-аннотационный "
                              "скрипт закреплен r7: experiments/at0_1_r6_annotation.py (round 3 "
                              "MINOR-3)"),
            "r6_annotation_script_sha256": sha256_file("experiments/at0/at0_1_r6_annotation.py"),
            "r7_claim_sha256": None,  # заменяется строкой при записи (self-sha невозможен)
            "annotation_script_sha256": sha256_file(os.path.abspath(__file__)),
            "store_sha256": STORE_SHA,
        },
        "changes": [
            "C1: null_analytic_ma1: + z_excess, z_raw_band, per_doc_ma1; E_doc переписан (было: «E[mean_band] = 0»)",
            "C2: G0a_tpu_hn_field: max_rel_dev_reconstruction(+detail) → reconstruction_roundoff_identity(+detail), status = тождество, не гейт",
            "C3: meta.r7_annotation.chain (r5/r6/r7 shas + скрипты); полная сверка r5→r6 и r6→r7 — assert",
            "C4: null_analytic_ma1: + z_excess_eff, d_eff, se_factor_median, SE_eff_formula (Bartlett + participation ratio; ambient D — нижняя граница)",
            "C5: SE_formula: /98 → /88 (K_MAX−8; код всегда считал /88)",
            "C6: t0_headline: T_none_full из excess_median/excess_mean → identity_check_G1; + headline = r/f_dc; t0_headline_note дополнен",
            "C7: self-audit расширен (l23_meta_vs_recomputed, roundoff, theta, z_raw_band=r5, T_none_full, pool) — assert'ы, не stdout",
            "R3 (round 3 MINOR-1..6): дата 2026-11-14 → 2026-11-15; thread-ноты (d_eff/z_eff bit-eq в pinned 6-thread env); r6-скрипт закреплен (experiments/at0_1_r6_annotation.py); headline r/f_dc ≡ N (trivial) — заголовок = r per object; guard th is None — defensive note; дубль median_z/z_raw_band — note. Чисел прогона не тронуто",
        ],
        "invariant": ("числа прогона не изменились: full diff r6→r7 — только декларированные "
                      "пути (assert patterns_match); все остальные листы bit-eq"),
        "self_audit": report["self_audit"],
    }

    # ---- 4. C3: full diff r6→r7 = ровно декларированные пути; остальное bit-eq ----
    a, r, c = leaf_diff(claim_r6, claim)
    R7_ADDED = [
        "prereg.4_plateau_vs_fDC.{obj}.null_analytic_ma1.z_excess*",
        "prereg.4_plateau_vs_fDC.{obj}.null_analytic_ma1.z_excess_eff*",
        "prereg.4_plateau_vs_fDC.{obj}.null_analytic_ma1.z_raw_band*",
        "prereg.4_plateau_vs_fDC.{obj}.null_analytic_ma1.d_eff*",
        "prereg.4_plateau_vs_fDC.{obj}.null_analytic_ma1.se_factor_median",
        "prereg.4_plateau_vs_fDC.{obj}.null_analytic_ma1.per_doc_ma1*",
        "prereg.4_plateau_vs_fDC.{obj}.null_analytic_ma1.SE_eff_formula",
        "prereg.4_plateau_vs_fDC.{obj}.null_analytic_ma1.c1_c4_note",
        "gates.G0a_tpu_hn_field.reconstruction_roundoff_identity",
        "gates.G0a_tpu_hn_field.reconstruction_roundoff_identity_detail*",
        "aggregates.t0_headline.{obj}.identity_check_G1*",
        "meta.r7_annotation*",
    ]
    R7_REMOVED = [
        "gates.G0a_tpu_hn_field.max_rel_dev_reconstruction",
        "gates.G0a_tpu_hn_field.max_rel_dev_reconstruction_detail*",
        "aggregates.t0_headline.{obj}.excess_median.T_none_full",
        "aggregates.t0_headline.{obj}.excess_mean.T_none_full",
    ]
    R7_CHANGED = [
        "prereg.4_plateau_vs_fDC.{obj}.null_analytic_ma1.E_doc",
        "prereg.4_plateau_vs_fDC.{obj}.null_analytic_ma1.SE_formula",
        "aggregates.t0_headline_note",
    ]
    assert patterns_match(a, R7_ADDED), f"C3: r6→r7 added: {a}"
    assert patterns_match(r, R7_REMOVED), f"C3: r6→r7 removed: {r}"
    assert patterns_match(c, R7_CHANGED), f"C3: r6→r7 changed: {c}"
    report["r6_to_r7_diff"] = {"n_added": len(a), "n_removed": len(r), "n_changed": len(c), "ok": True}

    # ---- 5. запись (claim компактный; manifest = claim + manifest-блок, indent=1;
    # self-sha невозможен: r7_claim_sha256 живёт в manifest-блоке, не в claim) ----
    claim["meta"]["r7_annotation"]["chain"]["r7_claim_sha256"] = "см. manifest-блок (self-sha невозможен)"
    claim_bytes = json.dumps(jclean(claim), ensure_ascii=False).encode("utf-8")
    with open(OUT_CLAIM, "wb") as f:
        f.write(claim_bytes)
    r7_sha = hashlib.sha256(claim_bytes).hexdigest()
    assert sha256_file(OUT_CLAIM) == r7_sha, "r7 claim sha не сошёлся после записи"
    man = dict(json.loads(claim_bytes))
    man["manifest"] = {"store_sha256": STORE_SHA, "claim_sha256": r7_sha, "written": True}
    with open(OUT_MAN, "w", encoding="utf-8") as f:
        json.dump(jclean(man), f, ensure_ascii=False, indent=1)
    pin = os.path.join(ROOT, "results/AT0/store/claim_r7_" + r7_sha[:16] + ".json")
    with open(pin, "wb") as f:
        f.write(claim_bytes)
    report["outputs"] = {"r7_claim_sha256": r7_sha, "pinned": pin,
                         "manifest": OUT_MAN, "claim": OUT_CLAIM}
    json.dump(report, sys.stdout, ensure_ascii=False, indent=1)
    print()
    print("R7_ANNOTATION_DONE")


if __name__ == "__main__":
    main()
