#!/usr/bin/env python3
"""E-36: GAUGE ANATOMY (ТЗ docs/Exp36_1-5.md). Эксплораторная карта gauge.

Пять декомпозиций J_t ∈ R^{K×D} (K=128, D=1024), слои {6, 12, 18, 22, 23},
кэш E-27b, протокол E-33r (sentinel: PRaw.sum < 0.5, только middle pairs):

D1: J = J^∥U + J^⊥U, P_U = 1·1ᵀ/K на строки (token-сторона, gauge-конвенция
    E-33r). energy_par_F = ‖J^∥U‖/‖J‖, Δ_SVD (правые D-side субпространства,
    конвенция E-28) при Δt ∈ {1, 4, 8, 16} для ∥U (r=1), ⊥U (r), raw (r).
D2: profile uniformity(u_k) = ⟨u_k, 1/√K⟩², k=1..8 (левые векторы).
D3: Δ_SVD трёх версий: P(p̄)·J (fixed-P), P(p_t)·J (dynamic),
    ΔP_t·J = (P(p_t) − P(p̄))·J.
D4: пары слоёв (6,12),(12,18),(18,22),(22,23): α(t) = ⟨u_1(a,t), u_1(b,t)⟩,
    Δ_SVD в группах Q1 (low |α|) vs Q4 (high |α|) по квартилям.
D5: β(t) = ⟨v_1(t), û_t⟩, û_t = h_t/‖h_t‖ (E-35); L23: β ≈ 0 по предсказанию.

Preflight = D1 + D2 + D5 (все слои). --full добавляет D3 + D4.

Использование:
    .venv/bin/python experiments/exp36_gauge_anatomy.py --selfcheck
    .venv/bin/python experiments/exp36_gauge_anatomy.py [--full]
Выход: results/exp36_gauge_anatomy.json + .md
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import exp33r_gauge_anatomy as E33R

FLAT_REL = 1e-6  # E-28/E-33r (RANK_EPS): «субпространство недефинировано» iff s_r/s_1 < FLAT_REL


# ============================================================
# helpers
# ============================================================

def dangle1(a, b):
    """Угол между 1-мерными подпространствами (|cos|, знак не важен)."""
    return torch.arccos(torch.clamp((a * b).sum(dim=-1).abs(), 0, 1))


def dsvd_right(Ja, Jb, r):
    """E-28 метрика (правые/D-side подпространства): max principal angle между
    top-r правыми сингулярными подпространствами Ja, Jb [T, n, k].
    → (angle [T], flat_a [T], flat_b [T]); flat = s_r/s_1 < FLAT_REL."""
    _, sa, Vha = torch.linalg.svd(Ja, full_matrices=False)
    _, sb, Vhb = torch.linalg.svd(Jb, full_matrices=False)
    Va = Vha[:, :r].transpose(-1, -2)              # [T, k, r]
    Vb = Vhb[:, :r].transpose(-1, -2)
    s = torch.linalg.svdvals(Va.transpose(-1, -2) @ Vb).clamp(max=1.0 - 1e-12)
    a = torch.acos(s).max(-1).values
    fa = sa[:, r - 1] / sa[:, 0].clamp_min(1e-30) < FLAT_REL
    fb = sb[:, r - 1] / sb[:, 0].clamp_min(1e-30) < FLAT_REL
    return a, fa, fb


def _np2py(o):
    """numpy → python, NaN/Inf сохраняются (для чекпоинта, в отличие от _clean)."""
    if isinstance(o, dict):
        return {str(k): _np2py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_np2py(v) for v in o]
    if isinstance(o, np.ndarray):
        return _np2py(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def _clean(o):
    """Нормализация JSON-значений: numpy-типы → python, NaN/Inf → None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        o = o.tolist()
        return _clean(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float):
        if np.isnan(o) or np.isinf(o):
            return None
        return o
    return o


def _fmt(x, nd=4):
    if isinstance(x, dict):
        x = x.get("median")
    if x is None:
        return "—"
    f = float(x)
    if not np.isfinite(f):
        return "NaN"
    return f"{f:.{nd}f}"


# ============================================================
# D1: gauge split (левый uniform, строки/токены — конвенция E-33r)
# ============================================================

def d1_split(Jdoc):
    """TZ D1: J = J∥U + J⊥U. P_U = 1·1ᵀ/K действует на строки (token-сторона):
    J∥U = P_U·J — все строки = token-mean (uniformity-компонента u_k).
    Ортогонально в Frobenius: ⟨J∥U, J⊥U⟩_F = 0."""
    c = Jdoc.mean(dim=1)                           # [T, D] token-mean
    Jpar = c[:, None, :].expand(*Jdoc.shape).contiguous()
    return Jpar, Jdoc - Jpar


def pair_angles(Vh, r_eff, dt):
    """max principal angle top-r_eff правых субпространств (t, t+dt).
    Vh [T, K, D] → [T−dt]."""
    Va = Vh[:-dt, :r_eff].transpose(-1, -2)
    Vb = Vh[dt:, :r_eff].transpose(-1, -2)
    s = torch.linalg.svdvals(Va.transpose(-1, -2) @ Vb).clamp(max=1.0 - 1e-12)
    return torch.acos(s).max(-1).values


def doc_d1(Jdoc, r, delta_grid, sent, s=None, Vh=None):
    """TZ E-36.1: ‖J∥U‖_F/‖J‖_F + Δ_SVD (правые субпространства, E-28)
    для ∥U (r=1), ⊥U (r) и raw J (r) при каждом Δ. Sentinel + flat-маска.
    s, Vh — общий SVD raw J (экономит 1 SVD); иначе считает сам."""
    T, K, D = Jdoc.shape
    sent_t = torch.as_tensor(sent, dtype=torch.bool)
    Jpar, Jperp = d1_split(Jdoc)
    Jn = Jdoc.norm(dim=(-1, -2)).clamp_min(1e-30)
    out = {"energy_par_F": (Jpar.norm(dim=(-1, -2)) / Jn).numpy()}

    # Jpar rank-1 (все строки = token-mean): Δ_SVD = угол топ-1 правых (r=1)
    _, spar, Vhp = torch.linalg.svd(Jpar, full_matrices=False)
    # Jperp: топ-r (r)
    _, sperp, Vhp2 = torch.linalg.svd(Jperp, full_matrices=False)
    flatp2 = (sperp[:, r - 1] / sperp[:, 0].clamp_min(1e-30) < FLAT_REL)
    if s is None:
        _, s, Vh = torch.linalg.svd(Jdoc, full_matrices=False)
    flat1 = (s[:, r - 1] / s[:, 0].clamp_min(1e-30) < FLAT_REL)

    for dt in delta_grid:
        mask = (~(sent_t[:-dt] | sent_t[dt:])).numpy()
        a = pair_angles(Vhp, 1, dt)
        out[f"d1_par_{dt}"] = np.where(mask, a.numpy(), np.nan)   # rank-1: flat невозможен
        a = pair_angles(Vhp2, r, dt)
        out[f"d1_perp_{dt}"] = np.where(mask & ~(flatp2[:-dt] | flatp2[dt:]).numpy(),
                                        a.numpy(), np.nan)
        a = pair_angles(Vh, r, dt)
        out[f"d1_raw_{dt}"] = np.where(mask & ~(flat1[:-dt] | flat1[dt:]).numpy(),
                                           a.numpy(), np.nan)
    return out


def doc_d2(U, kmax=8):
    """TZ E-36.2: profile uniformity(u_k) = ⟨u_k, 1/√K⟩², k=1..kmax.
    U — левые сингулярные векторы raw J [T, K, K]. → [T, kmax].
    Не парная метрика — sentinel-маска не применяется."""
    one = torch.ones(U.shape[1]) / U.shape[1] ** 0.5
    u = U[:, :, :kmax]
    return ((u * one[None, :, None]).sum(dim=1) ** 2).numpy()


# ============================================================
# D2-D5 (per doc, per position)
# ============================================================

def p_apply(J, p):
    """P(p)·J per position [T,K,D] (E-33r): J∘p − (Jᵀp)∘p."""
    w = torch.einsum('tkd,tk->td', J, p)
    return J * p.unsqueeze(2) - w.unsqueeze(1) * p.unsqueeze(2)


def doc_d3(Jdoc, Pdoc, r, sent, dt=1):
    """TZ E-36.3: p_t = p̄ + δp_t. Δ_SVD (Δ=1) трёх версий:
    P(p̄)·J_t (fixed-P), P(p_t)·J_t (dynamic), ΔP_t·J_t = (P(p_t)−P(p̄))·J_t.
    p̄ = медиана по non-sentinel позициям (конвенция E-33r.2 A; см. meta.deviation).
    → dict [T−1] per version."""
    T = Jdoc.shape[0]
    sent_t = torch.as_tensor(sent, dtype=torch.bool, device=Jdoc.device)
    alive = ~sent_t
    pbar = Pdoc[alive].median(dim=0).values if alive.any() else Pdoc.mean(dim=0)
    Ppbar = p_apply(Jdoc, pbar.unsqueeze(0).expand(T, -1))
    Ppt = p_apply(Jdoc, Pdoc)
    DPt = Ppt - Ppbar
    out = {}
    mask = (~(sent_t[:-dt] | sent_t[dt:])).numpy()
    for name, X in (("fp", Ppbar), ("dyn", Ppt), ("dp", DPt)):
        a, fa, fb = dsvd_right(X[:-dt], X[dt:], r)
        out[f"d3_{name}"] = np.where(mask & ~(fa | fb).numpy(), a.numpy(), np.nan)
    return out


def doc_d5(Vh, Hdoc):
    """TZ E-36.5: β(t) = ⟨v_1(t), û_t⟩, û_t = h_t/‖h_t‖ (E-35).
    Vh — правые сингулярные векторы raw J [T, K, D]; Hdoc [T, D] (h_t).
    → beta [T]. Не парная метрика — sentinel-маска не применяется."""
    v1 = Vh[:, 0, :]
    hn = Hdoc.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    uhat = Hdoc / hn
    return (v1 * uhat).sum(dim=-1).numpy()


def doc_d4(Ja, Jb, r=16, Ua=None, Ub=None, dt=1):
    """TZ E-36.4: α(t) = ⟨u_1(a,t), u_1(b,t)⟩ (знак SVD произволен → |α|),
    Δ_SVD (r, правые) пары (t, t+dt) в группах Q1 (|α| ≤ q25) / Q4 (|α| ≥ q75).
    Ua, Ub — левые векторы (экономят 2 SVD); иначе считаются.
    → dict (alpha [T] + групповые медианы)."""
    aA, faA, fbA = dsvd_right(Ja[:-dt], Ja[dt:], r)
    aB, faB, fbB = dsvd_right(Jb[:-dt], Jb[dt:], r)
    if Ua is None:
        Ua, _, _ = torch.linalg.svd(Ja, full_matrices=False)
    if Ub is None:
        Ub, _, _ = torch.linalg.svd(Jb, full_matrices=False)
    alpha = (Ua[:, :, 0] * Ub[:, :, 0]).sum(dim=-1)          # [T]
    absa = np.abs(alpha.numpy())
    m = np.isfinite(absa)
    q1 = float(np.percentile(absa[m], 25)) if m.any() else 0.0
    q4 = float(np.percentile(absa[m], 75)) if m.any() else 0.0
    g1 = m & (absa <= q1)
    g4 = m & (absa >= q4)

    finA = (~(faA | fbA)).numpy()
    finB = (~(faB | fbB)).numpy()

    def med(arr, fin, gm):
        v = arr.numpy()[gm[:-dt] & fin]
        return float(np.median(v)) if v.size else None

    return {
        "alpha": alpha.numpy(),
        "alpha_abs_median": float(np.median(absa[m])) if m.any() else None,
        "q1": q1, "q4": q4,
        "dsvd_a_Q1": med(aA, finA, g1), "dsvd_a_Q4": med(aA, finA, g4),
        "dsvd_b_Q1": med(aB, finB, g1), "dsvd_b_Q4": med(aB, finB, g4),
    }


def verdict(res):
    """Exploratory — карта, не тест. Сводные строки."""
    L = res["meta"]["layers"]
    full = res["meta"]["full"]

    def g(d, k):
        v = (d or {}).get(k)
        return _fmt(v.get("median")) if isinstance(v, dict) else _fmt(v)

    out = ["# Вердикт — E-36: gauge anatomy (exploratory, нет pass/fail)", ""]
    out.append("## D1 — uniform split (Δ=1, медианы)")
    out.append("| layer | energy_par_F | par | perp | raw |")
    out.append("|---|---|---|---|---|")
    for l in L:
        d = res["summary"][str(l)]["d1"]
        out.append(f"| {l} | {g(d, 'energy_par_F')} | {g(d, 'd1_par_1')} "
                   f"| {g(d, 'd1_perp_1')} | {g(d, 'd1_raw_1')} |")
    out.append("")
    out.append("## D2 — profile uniformity(u_k), медиана по (doc, t)")
    for l in L:
        prof = res["summary"][str(l)]["d2_profile"]
        out.append(f"- l={l}: " + " ".join(_fmt(x) for x in prof))
    out.append("")
    out.append("## D5 — |β(t)| = |⟨v_1, û_t⟩| (E-35: L23 → 0)")
    for l in L:
        out.append(f"- l={l}: {_fmt(res['summary'][str(l)]['d5_abs_beta'].get('median'))}")
    if full:
        out.append("")
        out.append("## D3 — fixed-P vs dynamic (Δ=1)")
        out.append("| layer | P(p̄)·J | P(p_t)·J | ΔP_t·J |")
        out.append("|---|---|---|---|")
        for l in L:
            d = res["summary"][str(l)]["d3"]
            out.append(f"| {l} | {g(d, 'd3_fp')} | {g(d, 'd3_dyn')} | {g(d, 'd3_dp')} |")
        out.append("")
        out.append("## D4 — cross-layer α и Δ_SVD по группам")
        for key, d in res["summary"].get("d4", {}).items():
            out.append(f"- {key}: |α| med={_fmt(d['alpha_abs'].get('median'))} "
                       f"(p10–p90: {_fmt(d['alpha_abs'].get('p10'))}–{_fmt(d['alpha_abs'].get('p90'))}); "
                       f"Δ_SVD Q1/Q4: a={_fmt(d.get('dsvd_a_Q1'))}/{_fmt(d.get('dsvd_a_Q4'))}, "
                       f"b={_fmt(d.get('dsvd_b_Q1'))}/{_fmt(d.get('dsvd_b_Q4'))}")
    out.append("")
    out.append("Интерпретация — по таблице в ТЗ (исходы A/B/C).")
    return out


# ============================================================
# selfcheck
# ============================================================

def test_d1():
    """D1: энергия, ортогональность, rank-1 тождество, sentinel, flat-маска."""
    T, K, D, r = 12, 16, 64, 4
    grid = (1, 2)
    sent = np.zeros(T, dtype=bool)
    sent[:2] = sent[-2:] = True
    one = np.ones(K) / K ** 0.5

    # (a) rank-1 поле J = a(t)⊗b(t): energy = |⟨a(t), 1/√K⟩| = sqrt(uniformity_0),
    #     d1_par = угол между b(t), b(t+1); d1_raw = d1_par (rank-1: правое под-пространство = b)
    torch.manual_seed(0)
    A = torch.randn(T, K); A = A / A.norm(dim=-1, keepdim=True)
    B = torch.randn(T, D); B = B / B.norm(dim=-1, keepdim=True)
    J = A.unsqueeze(2) * B.unsqueeze(1)              # [T, K, D]
    o = doc_d1(J, r, grid, sent)
    e_expect = (A * torch.tensor(one)).sum(-1).abs().numpy()
    assert np.allclose(o["energy_par_F"], e_expect.numpy() if hasattr(e_expect, 'numpy') else e_expect, atol=1e-5), (o["energy_par_F"][:3], e_expect[:3])
    v = B / B.norm(dim=-1, keepdim=True)
    ang_expect = dangle1(v[:-1], v[1:]).numpy()
    mfin = np.isfinite(o["d1_par_1"])
    assert mfin.any() and np.allclose(o["d1_par_1"][mfin], ang_expect[mfin], atol=1e-4), \
        "d1_par rank-1 identity"
    # raw при r=4 на rank-1 поле: top-r недефинируемо → flat-маска (NaN) — корректно
    assert np.isnan(o["d1_raw_1"]).all(), "raw flat on rank-1"
    # sentinel: pair (0,1) и (T-2,T-1) — NaN
    assert np.isnan(o["d1_par_1"][0]) and np.isnan(o["d1_par_1"][T - 2]), "sentinel mask"
    assert np.isfinite(o["d1_par_1"][2]), "middle finite"
    # Δ=2: pair (0,2) тоже NaN (sent[0]), pair (1,3) NaN (sent[1])
    assert np.isnan(o["d1_par_2"][:2]).all(), "sentinel d2"

    # (b) orthogonality: energy² + perp² = 1 (Frobenius-ортогональность)
    torch.manual_seed(1)
    J2 = torch.randn(T, K, D)
    Jpar, Jperp = d1_split(J2)
    Jn = J2.norm(dim=(-1, -2)).clamp_min(1e-30)
    assert np.allclose((Jpar.norm(dim=(-1, -2)) / Jn) ** 2 + (Jperp.norm(dim=(-1, -2)) / Jn) ** 2,
                       1.0, atol=1e-5), "frobenius orthogonality"
    # строки Jperp zero-mean
    assert torch.allclose(Jperp.mean(dim=1), torch.zeros(T, D), atol=1e-5), "zero-mean rows"

    # (c) полностью uniform J: Jpar = J → energy = 1, Jperp = 0 → flat → NaN
    C = torch.randn(T, D)
    J3 = C[:, None, :].expand(T, K, D).contiguous()
    o3 = doc_d1(J3, r, grid, sent)
    assert np.allclose(o3["energy_par_F"], 1.0, atol=1e-5)
    assert np.isnan(o3["d1_perp_1"][1:]).all(), "perp flat masked"

    print("test_d1: all passed")


def test_d2():
    T, K, D, kmax = 12, 16, 64, 8
    one = torch.ones(K) / K ** 0.5
    # (a) rank-1 поле: u_1 = a(t)/|a(t)| (знак произвольный) → uniformity_1 =
    #     |⟨a, 1/√K⟩|², остальные u_k — произвольны (вырождение) — только finiteness
    torch.manual_seed(0)
    A = torch.randn(T, K); A = A / A.norm(dim=-1, keepdim=True)
    B = torch.randn(T, D); B = B / B.norm(dim=-1, keepdim=True)
    J = A.unsqueeze(2) * B.unsqueeze(1)
    prof = doc_d2(torch.linalg.svd(J, full_matrices=False)[0], kmax)
    assert prof.shape == (T, kmax) and np.all(np.isfinite(prof))
    e1 = ((A * one).sum(-1) ** 2).numpy()
    assert np.allclose(prof[:, 0], e1, atol=1e-4), (prof[:, 0][:4], e1[:4])
    assert (prof[:, 1:] >= -1e-6).all() and (prof[:, 1:] <= 1 + 1e-6).all()
    # (b) произвольное поле: uniformity ∈ [0,1], sum по всем k ≤ 1 (ортонормальность)
    torch.manual_seed(1)
    J2 = torch.randn(T, K, D)
    p2 = doc_d2(torch.linalg.svd(J2, full_matrices=False)[0], kmax)
    assert np.all((p2 >= -1e-6) & (p2 <= 1 + 1e-6))
    assert np.all(p2.sum(axis=1) <= 1 + 1e-5), "orthonormal: sum uniformity ≤ 1"
    # (c) поле, у которого u_1 = uniform, u_2..u_8 — ортогональны 1: профиль =
    #     [1, ~0, ~0, ...] — колено на k=1
    Qr, _ = torch.linalg.qr(torch.randn(K, K - 1))
    M = Qr - (Qr.T @ one)[None, :] * one[:, None]             # ⊥ one
    U1 = torch.cat([one[:, None], M], dim=1)                 # ортонормальный, 1-й = uniform
    sw = torch.linspace(K, 1, K)                             # строго убывающие σ
    W, _ = torch.linalg.qr(torch.randn(D, K))                # W^T ∈ R^{K×D}
    J3 = (U1 * sw[None, :]) @ W.T                             # SVD-форма [K, D]
    J3 = J3[None].expand(T, K, D).contiguous()
    p3 = doc_d2(torch.linalg.svd(J3, full_matrices=False)[0], kmax)
    assert np.allclose(p3[:, 0], 1.0, atol=1e-4), p3[:, 0]
    assert np.allclose(p3[:, 1:], 0.0, atol=1e-4), p3[:, 1:]
    print("test_d2: all passed")


def test_d3():
    T, K, D, r = 12, 16, 64, 4
    grid = (1,)
    sent = np.zeros(T, dtype=bool)
    sent[:2] = sent[-2:] = True

    # (a) gauge-поле: J = a(t)⊗B, p_t ∥ a_t → P(p_t)·J_t = 0 (всё gauge)
    torch.manual_seed(0)
    A = torch.randn(T, K); A = A / A.norm(dim=-1, keepdim=True)
    B = torch.randn(D)
    J = A.unsqueeze(2) * B[None, None, :]                    # [T, K, D]
    P = A / A.norm(dim=-1, keepdim=True)
    o = doc_d3(J, P, r, sent)
    # P(p_t)·J_t = 0 → flat → NaN везде
    assert np.isnan(o["d3_dyn"]).all(), ("dyn", o["d3_dyn"][:4])
    assert np.isnan(o["d3_dp"]).all(), ("dp", o["d3_dp"][:4])
    # fixed-P: P(p̄)·J_t ≠ 0 (не всё gauge), но rank-1 → top-r не дефинируемо
    assert np.isnan(o["d3_fp"]).all(), ("fp", o["d3_fp"][:4])
    # P(p̄)·J ≠ 0 (в отличие от dyn) — sanity
    pbar = (P.mean(0) / P.mean(0).norm()).unsqueeze(0).expand(T, -1)
    Ppbar = p_apply(J, pbar)
    assert Ppbar.abs().sum() > 1e-3

    # (b) non-gauge поле + constant p: ΔP = 0 → flat; fp finite
    torch.manual_seed(1)
    J2 = torch.randn(T, K, D)
    p = torch.ones(T, K) / K
    o2 = doc_d3(J2, p, r, sent)
    assert np.isnan(o2["d3_dp"]).all(), "ΔP = 0 -> flat"
    mfin = np.isfinite(o2["d3_fp"])
    assert mfin.sum() > 0 and np.all(o2["d3_fp"][mfin] >= 0), "fp finite"

    # (c) P(p)·1 = 0 (gauge убивает uniform-строки)
    J4 = torch.ones(T, K, D)
    assert torch.allclose(p_apply(J4, p), torch.zeros(T, K, D), atol=1e-5), "P(p)·1 = 0"
    print("test_d3: all passed")


def test_d4():
    T, K, D = 12, 16, 64
    torch.manual_seed(0)
    e0, e1, e2 = torch.eye(K)[:3]
    w = torch.randn(D)
    # u_a: чётные t → e0, нечётные → e1; u_b: чётные → e0, нечётные → e2.
    # α(t) = 1 (чётные), 0 (нечётные). w(t) постоянно → Δ_SVD = 0 везде (r=1).
    ua = torch.stack([e0 if t % 2 == 0 else e1 for t in range(T)])
    ub = torch.stack([e0 if t % 2 == 0 else e2 for t in range(T)])
    Ja = (ua.unsqueeze(2) * w[None, None, :]).contiguous()
    Jb = (ub.unsqueeze(2) * w[None, None, :]).contiguous()
    o = doc_d4(Ja, Jb, r=1)
    exp_alpha = np.array([1.0 if t % 2 == 0 else 0.0 for t in range(T)])
    assert np.allclose(o["alpha"], exp_alpha, atol=1e-4), o["alpha"][:6]
    assert abs(o["q1"] - 0.0) < 1e-6 and abs(o["q4"] - 1.0) < 1e-6, (o["q1"], o["q4"])
    for k in ("dsvd_a_Q1", "dsvd_a_Q4", "dsvd_b_Q1", "dsvd_b_Q4"):
        assert o[k] is not None and abs(o[k]) < 1e-3, (k, o[k])
    # (b) полностью ортогональные top-1: α ≡ 0
    Ja2 = (e1.unsqueeze(0).expand(T, K).contiguous().unsqueeze(2) * w[None, None, :]).contiguous()
    Jb2 = (e0.unsqueeze(0).expand(T, K).contiguous().unsqueeze(2) * w[None, None, :]).contiguous()
    o2 = doc_d4(Ja2, Jb2, r=1)
    assert np.allclose(o2["alpha"], 0.0, atol=1e-4), o2["alpha"][:6]
    print("test_d4: all passed")


def test_d5():
    T, K, D = 12, 16, 64
    torch.manual_seed(0)
    # (a) J = a⊗b: v1 = b/‖b‖ (знак произволен); H ∥ b → |β| = 1
    A = torch.randn(T, K); A = A / A.norm(dim=-1, keepdim=True)
    B = torch.randn(D); B = B / B.norm()
    J = A.unsqueeze(2) * B[None, None, :]
    H1 = B.unsqueeze(0).expand(T, D).contiguous()
    assert np.allclose(np.abs(doc_d5(torch.linalg.svd(J, full_matrices=False)[2], H1)), 1.0, atol=1e-4)
    # (b) H ⊥ v1 → β = 0
    Bp = torch.randn(D); Bp = Bp - (Bp @ B) * B; Bp = Bp / Bp.norm()
    H2 = Bp.unsqueeze(0).expand(T, D).contiguous()
    assert np.allclose(doc_d5(torch.linalg.svd(J, full_matrices=False)[2], H2), 0.0, atol=1e-4)
    # (c) h = cos(θ)·b + sin(θ)·b⊥ → |β| = cos(θ)
    th = torch.rand(T) * 0.5 + 0.1
    H3 = torch.cos(th)[:, None] * B.unsqueeze(0) + torch.sin(th)[:, None] * Bp.unsqueeze(0)
    assert np.allclose(np.abs(doc_d5(torch.linalg.svd(J, full_matrices=False)[2], H3)), torch.cos(th).numpy(), atol=1e-4)
    print("test_d5: all passed")



# ============================================================
# main
# ============================================================

LAYERS = [6, 12, 18, 22, 23]
D4_PAIRS = [(6, 12), (12, 18), (18, 22), (22, 23)]
DELTA_GRID = (1, 4, 8, 16)
CKPT_DIR = "/data/prism_kernels/exp27b/results"


def _stats(arr):
    v = np.asarray(arr, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)), "n": int(v.size),
            "frac_nan": 1.0 - v.size / max(1, arr.size)}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--r", type=int, default=16, help="subspace dimension (E-33r: 16)")
    a.add_argument("--sample-cap", type=int, default=0, help="max documents (0=all)")
    a.add_argument("--full", action="store_true", help="+D3 +D4 (priority 2)")
    a.add_argument("--resume", action="store_true",
                   help="resume from results/exp36_gauge_anatomy.ckpt.json (skip done docs)")
    a.add_argument("--selfcheck", action="store_true")
    args = a.parse_args()

    if args.selfcheck:
        for t in (test_d1, test_d2, test_d3, test_d4, test_d5):
            t()
        print("selfcheck: all passed")
        return

    t0 = time.time()
    r = args.r
    layers = LAYERS
    print(f"loading caches: layers={layers} dir={CKPT_DIR}")
    cks = {l: E33R.load_layer(CKPT_DIR, l) for l in layers}
    ck0 = cks[layers[0]]
    domains = sorted({k[0] for k in ck0["JR"]})
    docs = [(d, s) for d in domains for s in range(ck0["n_samples"])]
    ref = set((k[0], k[1]) for k in ck0["JR"])
    for l in layers:
        assert set((k[0], k[1]) for k in cks[l]["JR"]) == ref, \
            f"layer {l}: (domain,sample) set mismatch vs {layers[0]}"
    if args.sample_cap > 0:
        docs = docs[:args.sample_cap]

    res = {
        "meta": {
            "spec": "E-36 gauge anatomy (docs/Exp36_1-5.md)",
            "layers": layers, "r": r, "delta_grid": list(DELTA_GRID),
            "d4_pairs": [list(p) for p in D4_PAIRS],
            "n_docs": len(docs), "S": int(ck0["S"]),
            "sentinel": "PRaw.sum(-1) < 0.5 (E-33r)",
            "full": bool(args.full), "cache_dir": CKPT_DIR, "timing_sec": None,
            "deviation": [
                "D1: P_U действует слева (P_U·J, строки) — формула 'J·P_U' в ТЗ "
                "размерностно невозможна ([K,D]·[K,K]); код = конвенция E-33r/paper §8",
                "D3: p̄ = медиана по non-sentinel позициям (конвенция E-33r.2 A), "
                "а не полное mean_t(p_t) из ТЗ",
                "D2/D5: не парные метрики — sentinel-маска E-33r не применяется, "
                "sentinel-позиции (PRaw.sum<0.5) включены как обычные samples (by design); "
                "проверено: J_t и h_t на sentinel НЕ вырождены (‖J_t‖ и |β| того же "
                "порядка, что и на non-sentinel), т.е. это не шум, а валидные точки; "
                "D4: α(t) — не парная по t метрика, sentinel-позиции тоже валидны",
                "D1 raw: flat-проверка на оба endpoint'а (строже E-33r, который "
                "проверяет только левый; согласовано с dsvd_right E-36)",
            ],
        },
        "per_doc": {},     # [i][str(layer)] = {doc, d1, d2, d5, [d3]}
        "d4_pairs": {},    # [i][f"{la}-{lb}"] = {...}
        "summary": {},
    }

    CKPT = "results/exp36_gauge_anatomy.ckpt.json"
    if args.resume and os.path.exists(CKPT):
        ckpt = json.load(open(CKPT))
        done = {int(k): v for k, v in ckpt["per_doc"].items()}
        res["per_doc"].update(done)
        if args.full:
            res["d4_pairs"].update({int(k): v for k, v in ckpt["d4_pairs"].items()})
        print(f"resume: {len(done)}/{len(docs)} docs already done")

    def save_ckpt():
        with open(CKPT, "w") as f:
            json.dump({"per_doc": _np2py(res["per_doc"]),
                       "d4_pairs": _np2py(res["d4_pairs"])}, f, ensure_ascii=False)

    for i, (d, s) in enumerate(docs):
        if i in res["per_doc"] and (not args.full or i in res["d4_pairs"]):
            continue   # resume: doc уже посчитан
        row = {}
        Ucache = {}      # layer → left singular vectors [S, K, K] (для D4)
        for l in layers:
            ck = cks[l]
            Jdoc, Pdoc, sent = E33R.load_doc(ck, d, s)
            Hdoc = E33R.load_H(ck, d, s).to(torch.float32)
            Jt = Jdoc.to(torch.float32)
            # один SVD raw J на (док, слой): общий для D1-raw, D2, D5, D4-u1
            U, S, Vh = torch.linalg.svd(Jt, full_matrices=False)
            row[str(l)] = {
                "doc": f"{d}_{s}",
                "d1": doc_d1(Jt, r, DELTA_GRID, sent, s=S, Vh=Vh),
                "d2": doc_d2(U, kmax=8),
                "d5": doc_d5(Vh, Hdoc),
            }
            if args.full:
                row[str(l)]["d3"] = doc_d3(Jt, Pdoc, r, sent)
                Ucache[l] = U
            del Jdoc, Pdoc, sent, Hdoc, Jt, S, Vh
        res["per_doc"][i] = row
        if args.full:
            pr = {}
            for (la, lb) in D4_PAIRS:
                Jla = E33R.load_doc(cks[la], d, s)[0].float()
                Jlb = E33R.load_doc(cks[lb], d, s)[0].float()
                pr[f"{la}-{lb}"] = doc_d4(Jla, Jlb, r=r, Ua=Ucache[la], Ub=Ucache[lb])
                del Jla, Jlb
            res["d4_pairs"][i] = pr
            del Ucache
        if (i + 1) % 4 == 0 or i == len(docs) - 1:
            print(f"  doc {i+1}/{len(docs)} ({d}_{s}) t={time.time()-t0:.0f}s", flush=True)
        save_ckpt()

    # ---------------- summary ----------------
    for l in layers:
        summ = {}
        d1s = {}
        for dt in DELTA_GRID:
            for part in ("par", "perp", "raw"):
                d1s[f"d1_{part}_{dt}"] = _stats(np.concatenate(
                    [p[str(l)]["d1"][f"d1_{part}_{dt}"] for p in res["per_doc"].values()]))
        d1s["energy_par_F"] = _stats(np.concatenate(
            [p[str(l)]["d1"]["energy_par_F"] for p in res["per_doc"].values()]))
        summ["d1"] = d1s
        prof = np.concatenate([p[str(l)]["d2"] for p in res["per_doc"].values()])
        summ["d2_profile"] = [float(np.nanmedian(prof[:, k])) for k in range(8)]
        b = np.concatenate([np.abs(p[str(l)]["d5"]) for p in res["per_doc"].values()])
        summ["d5_abs_beta"] = _stats(b)
        if args.full:
            d3s = {}
            for v in ("fp", "dyn", "dp"):
                d3s[f"d3_{v}"] = _stats(np.concatenate(
                    [p[str(l)]["d3"][f"d3_{v}"] for p in res["per_doc"].values()]))
            summ["d3"] = d3s
        res["summary"][str(l)] = summ

    if args.full:
        d4s = {}
        for key in [f"{a}-{b}" for a, b in D4_PAIRS]:
            al = np.concatenate([res["d4_pairs"][i][key]["alpha"] for i in res["d4_pairs"]])
            al = np.abs(al[np.isfinite(al)])
            vals = [res["d4_pairs"][i][key] for i in res["d4_pairs"]]
            d4s[key] = {
                "alpha_abs": _stats(al),
                "q1": float(np.nanmedian([v["q1"] for v in vals])),
                "q4": float(np.nanmedian([v["q4"] for v in vals])),
            }
            for part in ("a", "b"):
                for grp in ("Q1", "Q4"):
                    vv = [v[f"dsvd_{part}_{grp}"] for v in vals if v[f"dsvd_{part}_{grp}"] is not None]
                    d4s[key][f"dsvd_{part}_{grp}"] = float(np.median(vv)) if vv else None
        res["summary"]["d4"] = d4s

    res["meta"]["timing_sec"] = time.time() - t0
    res["verdict"] = verdict(res)

    with open("results/exp36_gauge_anatomy.json", "w") as f:
        json.dump(_clean(res), f, indent=2, ensure_ascii=False)
    if os.path.exists(CKPT):
        os.remove(CKPT)

    # ---------------- markdown ----------------
    md = ["# E-36: Gauge Anatomy", "",
          "## Протокол", "",
          f"- слои {layers}, r={r}, Δt ∈ {list(DELTA_GRID)}, n_docs={len(docs)}, S={ck0['S']}",
          f"- sentinel: PRaw.sum < 0.5 (E-33r), только middle pairs",
          f"- full={args.full} (D3/D4 {'включены' if args.full else 'не включены'})",
          f"- timing: {res['meta']['timing_sec']:.1f}s", "",
          "## D1: uniform split (медианы, rad)"]
    for l in layers:
        d = res["summary"][str(l)]["d1"]
        hdr = "| Δ | ∥U (r=1) | ⊥U (r) | raw (r) |"
        md.append(f"### layer {l}")
        md.append("")
        md.append(hdr)
        md.append("|---|---|---|---|")
        for dt in DELTA_GRID:
            md.append(f"| {dt} | {_fmt((d.get(f'd1_par_{dt}') or {}).get('median'))} "
                      f"| {_fmt((d.get(f'd1_perp_{dt}') or {}).get('median'))} "
                      f"| {_fmt((d.get(f'd1_raw_{dt}') or {}).get('median'))} |")
        e = (d.get("energy_par_F") or {})
        md.append("")
        md.append(f"energy_par_F: med={_fmt(e.get('median'))} (p10–p90: "
                  f"{_fmt(e.get('p10'))}–{_fmt(e.get('p90'))})")
        md.append("")
    md.append("## D2: profile uniformity(u_k)")
    md.append("| k | " + " | ".join(f"L{l}" for l in layers) + " |")
    md.append("|---|" + "---|" * len(layers))
    for k in range(8):
        md.append(f"| {k+1} | " + " | ".join(
            _fmt(res["summary"][str(l)]["d2_profile"][k]) for l in layers) + " |")
    md.append("")
    md.append(f"(случайный уровень: 1/K = {1/128:.4f})")
    md.append("")
    md.append("## D5: |β| = |⟨v_1, û_t⟩|")
    md.append("| layer | median | p10 | p90 |")
    md.append("|---|---|---|---|")
    for l in layers:
        b = res["summary"][str(l)]["d5_abs_beta"]
        md.append(f"| {l} | {_fmt(b.get('median'))} | {_fmt(b.get('p10'))} | {_fmt(b.get('p90'))} |")
    md.append("")
    if args.full:
        md.append("## D3: fixed-P vs dynamic (Δ=1, медианы, rad)")
        md.append("| layer | P(p̄)·J | P(p_t)·J | ΔP_t·J |")
        md.append("|---|---|---|---|")
        for l in layers:
            d = res["summary"][str(l)]["d3"]
            md.append(f"| {l} | {_fmt((d.get('d3_fp') or {}).get('median'))} "
                      f"| {_fmt((d.get('d3_dyn') or {}).get('median'))} "
                      f"| {_fmt((d.get('d3_dp') or {}).get('median'))} |")
        md.append("")
        md.append("## D4: cross-layer")
        for key, d in res["summary"]["d4"].items():
            md.append(f"- {key}: |α| med={_fmt(d['alpha_abs'].get('median'))} "
                      f"(q1={_fmt(d['q1'])}, q4={_fmt(d['q4'])}); "
                      f"Δ_SVD Q1/Q4 a={_fmt(d.get('dsvd_a_Q1'))}/{_fmt(d.get('dsvd_a_Q4'))}, "
                      f"b={_fmt(d.get('dsvd_b_Q1'))}/{_fmt(d.get('dsvd_b_Q4'))}")
        md.append("")
    md += ["## Verdict", ""] + res["verdict"][2:]
    md += ["", f"## Timing", "", f"- total: {res['meta']['timing_sec']:.1f}s", ""]

    with open("results/exp36_gauge_anatomy.md", "w") as f:
        f.write("\n".join(md))

    print(f"\nDone in {res['meta']['timing_sec']:.1f}s")
    print(f"  JSON: results/exp36_gauge_anatomy.json")
    print(f"  MD:   results/exp36_gauge_anatomy.md")


if __name__ == "__main__":
    main()
