#!/usr/bin/env python3
"""E-33r: gauge anatomy — где живёт gauge и что выживает после P(p).

ТЗ: docs/Exp33.md. Только exp27b dense-кэш (JR, P128, P128_RAW, H).

Шесть срезов (per doc per layer, batched SVD):
  1. локализация gauge-режима: u_top, uniformity_0, gauge_share (top-1 proxy),
     gauge_share_exact, align(u_top, u_g_cheap), align(u_top, p_t)
  2. выживание после P(p): survive/kill_frac, sig_gf/sig_raw top-8, rank_ratio
  3. устойчивость по Δ: Δ_SVD_raw vs Δ_SVD_gf (r=16, E-28 метрика), Δ_gain
  4. реконструкция «убитого» K_t = J_t − P(p_t)J_t: SVD, rank_K, align топ-левого с 1
  5. depth coherence raw vs gf: cos(v_top^l(t), v_top^{l'}(t)) vs random R^D pairs
  6. смыкание: L23 Spearman(Δ_SVD_gf, d_S(û)); Δ_gain vs gauge_share_exact; kill vs gauge_share

Отличие от исходного E-33: sentinel-маска — endpoints (para t, t+Δ валидна, если
sent[t] и sent[t+Δ] оба False), не вся цепочка. Исходный E-33 сохраняется как
отрицательный контроль (degenerate_test_result) — результаты из
results/exp33_gauge_holonomy.json.

Исходный E-33 закрыт как тривиальный: u_g ≈ uniform, P(p)·1 = 0.

Самодиагностика: --selfcheck (синтетика rank-1: uniformity_0 = <u, 1/√K>²,
kill_frac = uniformity_0 при p = uniform; Δ_SVD_gf = Δ_SVD_raw при p = uniform).

Использование:
    .venv/bin/python experiments/exp33r_gauge_anatomy.py --selfcheck
    .venv/bin/python experiments/exp33r_gauge_anatomy.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

import exp28_leaf_structure as E28

SEED = 42
LAYERS = [6, 12, 18, 22, 23]
R_MAIN = 16
DELTA_GRID = [1, 4, 8, 16]
DIAG_R = [1, 2, 4, 8, 16, 32]
DEPTH_PAIRS = [(6, 12), (12, 18), (18, 22), (22, 23)]
SENT = 0.5
N_DEP_SHUF = 512
RANK_EPS = 1e-6
# ponytail: FLAT_REL = RANK_EPS — «субпространство недефинировано» iff gap ≤ rank-порог.
# E-28's 1e-3 was for raw spectra (gap ~0.1); gf spectra are flat (gap ~1e-5), so at 1e-3
# every gf pair is marked degenerate and Δ_gf ≡ π/2 by construction. Split if raw/gf
# ever need different thresholds.
FLAT_REL = RANK_EPS


# ---------------- data ----------------

def load_layer(ckpt_dir, layer):
    path = os.path.join(ckpt_dir, f"exp27b_jcache_L{layer}.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["S"] = int(ck["S"]); ck["n_samples"] = int(ck["n_samples"])
    k0 = next(iter(ck["JR"].values()))
    # JR[(dom, si, t)] has shape [K, D] (128 tokens x 1024 hidden)
    ck["K"] = int(k0.shape[0]); ck["D"] = int(k0.shape[1])
    return ck


def load_doc(ck, dom, si):
    """Ключи кэша — кортежи (dom, si, t)."""
    S, K = ck["S"], ck["K"]
    Jdoc = torch.stack([ck["JR"][(dom, si, t)] for t in range(S)]).float()
    Pdoc = torch.stack([ck["P128"][(dom, si, t)] for t in range(S)]).float()
    PRaw = torch.stack([ck["P128_RAW"][(dom, si, t)] for t in range(S)]).float()
    sent = (PRaw.sum(-1) < SENT).numpy()
    return Jdoc, Pdoc, sent


def load_H(ck, dom, si):
    S = ck["S"]
    return torch.stack([ck["H"][(dom, si, t)] for t in range(S)]).float()


# ---------------- math (vectorized over positions) ----------------

def p_apply(J, p):
    """P(p)·J per position: [T,K,D] = J∘p[:,None,:] − (Jᵀp)[:,None,:]∘p[:,None,:].
    w[t,d] = Σ_k p[t,k]·J[t,k,d]."""
    w = torch.einsum('tkd,tk->td', J, p)               # [T, D] = J·p
    return J * p.unsqueeze(2) - w.unsqueeze(1) * p.unsqueeze(2)


def batch_metrics(Jdoc, Pdoc, sent, r=R_MAIN, diag=False):
    """One doc, one layer: SVD of J, PJ, K_t (+ PJ_fp for E-33r.2 diagnostics).
    Returns dict of per-position numpy arrays [S] (NaN where degenerate)."""
    T, K, D = Jdoc.shape
    if Pdoc.shape[0] == 1:
        Pdoc = Pdoc.expand(T, -1)
    S1, s1, Vh1 = torch.linalg.svd(Jdoc, full_matrices=False)   # [T,K,K] [T,K] [T,K,D]
    PJ = p_apply(Jdoc, Pdoc)
    S3, s3, Vh3 = torch.linalg.svd(PJ, full_matrices=False)     # gauge-free
    Kt = Jdoc - PJ
    S4, s4, Vh4 = torch.linalg.svd(Kt, full_matrices=False)     # «убитое»
    if diag:
        # E-33r.2 Diagnostic A: fixed p0 = НОРМИРОВАННАЯ медиана p_t по non-sentinel t
        # (до фикса 2026-07-07 медиана была ненормированной: Σ ≈ 0.006 → P ≈ diag,
        # не gauge-оператор; см. results/exp33r_gauge_anatomy.json meta.refit_note)
        alive = ~torch.as_tensor(sent)
        p0 = Pdoc[alive].median(dim=0).values if alive.any() else Pdoc.mean(dim=0)
        p0 = p0 / p0.sum().clamp_min(1e-30)
        assert abs(float(p0.sum()) - 1.0) < 1e-5, "p0 must be normalized"
        Sfp, sfp, Vhfp = torch.linalg.svd(p_apply(Jdoc, p0.unsqueeze(0).expand(T, -1)),
                                          full_matrices=False)
    s1n = s1 / s1[:, :1].clamp_min(1e-30)
    s3n = s3 / s3[:, :1].clamp_min(1e-30)
    s4n = s4 / s4[:, :1].clamp_min(1e-30)
    one = torch.ones(K, device=Jdoc.device) / K ** 0.5
    u_top = S1[:, :, 0]                                        # [T, K] левый топ-вектор
    uniformity_0 = ((u_top * one).sum(-1)) ** 2                # <u_top, 1>^2 (squared inner product)
    lam2 = s1 ** 2
    trace_e = lam2.sum(-1).clamp_min(1e-30)
    gs_top1 = (lam2[:, 0] * uniformity_0) / trace_e
    gs_exact = ((S1.transpose(-1, -2) @ one) ** 2 * lam2).sum(-1) / trace_e
    J1 = torch.einsum('tkd,tk->td', Jdoc, torch.ones(T, K, device=Jdoc.device))  # [T,D] = J·ones(K)
    Jjt1 = torch.einsum('tkd,td->tk', Jdoc, J1)                   # [T,K] = J·(J·1)
    ugc = Jjt1 / Jjt1.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    align_ug = (u_top * ugc).sum(-1).abs()
    pnorm = Pdoc.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    align_p = (u_top * Pdoc / pnorm).sum(-1).abs()
    survive = (PJ.norm(dim=(-1, -2)) / Jdoc.norm(dim=(-1, -2)).clamp_min(1e-30))
    nrmK = Kt.norm(dim=(-1, -2))
    kill = (1.0 - survive)
    kill_check = nrmK / Jdoc.norm(dim=(-1, -2)).clamp_min(1e-30)
    rank_raw = (s1n > RANK_EPS).sum(-1)
    rank_gf = (s3n > RANK_EPS).sum(-1)
    rank_K = (s4n > RANK_EPS).sum(-1)
    sKtop = s4[:, :4].numpy()
    uKtop = S4[:, :, 0]
    align_K1 = (uKtop * one).sum(-1).abs()
    vtop_raw = Vh1[:, 0, :].contiguous()               # [T, D]
    vtop_gf = Vh3[:, 0, :].contiguous()                # [T, D]
    # E-33r.2 F: top-16 bases for per-r depth coherence (no extra SVD in the depth leg)
    vt16_raw = Vh1[:, :min(16, K), :].contiguous()     # [T, min(16,K), D]
    vt16_gf = Vh3[:, :min(16, K), :].contiguous()
    out = {
        'uniformity_0': uniformity_0.numpy(),
        'gauge_share': gs_top1.numpy(),
        'gauge_share_exact': gs_exact.numpy(),
        'align_ug': align_ug.numpy(),
        'align_p': align_p.numpy(),
        'survive_frac': survive.numpy(),
        'kill_frac': kill.numpy(),
        'kill_frac_check': kill_check.numpy(),
        'rank_raw': rank_raw.numpy(),
        'rank_gf': rank_gf.numpy(),
        'rank_K': rank_K.numpy(),
        'sK_top4': sKtop,
        'align_K1': align_K1.numpy(),
        'vtop_raw': vtop_raw.cpu().numpy(),
        'vtop_gf': vtop_gf.cpu().numpy(),
        'vt16_raw': vt16_raw.cpu().numpy(),
        'vt16_gf': vt16_gf.cpu().numpy(),
    }
    for k in range(8):
        out[f'sig_ratio_{k+1}'] = (s3[:, k] / s1[:, k].clamp_min(1e-30)).numpy()
    # E-33r.2 B: ratio = sig_gf/sig_raw at r, mean over i<r, per position
    for rt in (1, 2, 4, 8, 16, 32):
        if rt > K:
            continue
        out[f'ratio_{rt}'] = (s3n[:, :rt].mean(-1) / s1n[:, :rt].mean(-1).clamp_min(1e-30)).numpy()
    # slice 3: Δ_SVD raw vs gf per Δ (E-28 metric: max principal angle of top-r right subspaces)
    for dt in DELTA_GRID:
        Va = Vh1[:-dt, :r].transpose(-1, -2)                   # [T-dt, D, r]
        Vb = Vh1[dt:, :r].transpose(-1, -2)
        M = Va.transpose(-1, -2) @ Vb                          # [T-dt, r, r]
        sv = torch.linalg.svdvals(M).clamp(max=1.0 - 1e-12)
        out[f'draw_{dt}'] = torch.acos(sv).max(-1).values.numpy()
        Wa = Vh3[:-dt, :r].transpose(-1, -2)
        Wb = Vh3[dt:, :r].transpose(-1, -2)
        M2 = Wa.transpose(-1, -2) @ Wb
        sv2 = torch.linalg.svdvals(M2).clamp(max=1.0 - 1e-12)
        out[f'dgf_{dt}'] = torch.acos(sv2).max(-1).values.numpy()
        # degeneracy: endpoints sentinel OR top-r subspace numerically unidentifiable
        # (spectrum flat: s_r/s_1 < FLAT_REL — E-28 rule, relative version for gf)
        flat_raw = (s1[:-dt, r - 1] / s1[:-dt, 0].clamp_min(1e-30) < FLAT_REL).numpy()
        flat_gf = (s3[:-dt, r - 1] / s3[:-dt, 0].clamp_min(1e-30) < FLAT_REL).numpy()
        mask = ~(sent[:-dt] | sent[dt:])
        out[f'draw_{dt}'] = np.where(mask & ~flat_raw, out[f'draw_{dt}'], np.nan)
        out[f'dgf_{dt}'] = np.where(mask & ~flat_gf, out[f'dgf_{dt}'], np.nan)
        out[f'gain_{dt}'] = np.where(mask & ~flat_raw & ~flat_gf,
                                     out[f'draw_{dt}'] - out[f'dgf_{dt}'], np.nan)
        if diag:
            # E-33r.2 A: fixed-P transport; D: Omega = ||U_gf^T U_raw||_F^2 / r
            Mfp = Vhfp[:-dt, :r] @ Vhfp[dt:, :r].transpose(-1, -2)
            svf = torch.linalg.svdvals(Mfp).clamp(max=1.0 - 1e-12)
            flat_fp = (sfp[:-dt, r - 1] / sfp[:-dt, 0].clamp_min(1e-30) < FLAT_REL).numpy()
            out[f'dgp_{dt}'] = np.where(mask & ~flat_fp,
                                        torch.acos(svf).max(-1).values.numpy(), np.nan)
            Om = Vh3[:-dt, :r] @ Vh1[:-dt, :r].transpose(-1, -2)      # [T-dt, r, r] = U_gfᵀ·U_raw
            out[f'omega_{dt}'] = np.where(mask, ((Om ** 2).sum(-1).sum(-1) / r).numpy(), np.nan)
            # E-33r.2 D: Omega(r) sweep r ∈ {1,2,8,16}; Omega_fp for fixed-p
            for rt in (1, 2, 8, 16):
                if rt > K:
                    continue
                Omr = Vh3[:-dt, :rt] @ Vh1[:-dt, :rt].transpose(-1, -2)
                out[f'omega_{dt}_r{rt}'] = np.where(mask, ((Omr ** 2).sum(-1).sum(-1) / rt).numpy(), np.nan)
                Mfp2 = Vhfp[:-dt, :rt] @ Vhfp[dt:, :rt].transpose(-1, -2)
                svfp = torch.linalg.svdvals(Mfp2).clamp(max=1.0 - 1e-12)
                flatfp = (sfp[:-dt, rt - 1] / sfp[:-dt, 0].clamp_min(1e-30) < FLAT_REL).numpy()
                out[f'dgp_{dt}_r{rt}'] = np.where(mask & ~flatfp,
                                                  torch.acos(svfp).max(-1).values.numpy(), np.nan)
    if diag:
        # E-33r.2 C: r-sweep (B uses dgf[rt][1]); E: fixed-p at r in {16, 32}
        dg = {'dgf': {}, 'draw': {}, 'dgp': {}}
        for rt in DIAG_R:
            if rt > K:
                continue
            dgr, dgg, dgp_ = {}, {}, {}
            for dt in DELTA_GRID:
                mr_ = ~(sent[:-dt] | sent[dt:])
                Va = Vh1[:-dt, :rt].transpose(-1, -2); Vb = Vh1[dt:, :rt].transpose(-1, -2)
                svr = torch.linalg.svdvals(Va.transpose(-1, -2) @ Vb).clamp(max=1.0 - 1e-12)
                fr = (s1[:-dt, rt - 1] / s1[:-dt, 0].clamp_min(1e-30) < FLAT_REL).numpy()
                dgr[dt] = np.where(mr_ & ~fr, torch.acos(svr).max(-1).values.numpy(), np.nan)
                Wa = Vh3[:-dt, :rt].transpose(-1, -2); Wb = Vh3[dt:, :rt].transpose(-1, -2)
                svg = torch.linalg.svdvals(Wa.transpose(-1, -2) @ Wb).clamp(max=1.0 - 1e-12)
                fg = (s3[:-dt, rt - 1] / s3[:-dt, 0].clamp_min(1e-30) < FLAT_REL).numpy()
                dgg[dt] = np.where(mr_ & ~fg, torch.acos(svg).max(-1).values.numpy(), np.nan)
                if rt in (16, 32):
                    Vfa = Vhfp[:-dt, :rt].transpose(-1, -2); Vfb = Vhfp[dt:, :rt].transpose(-1, -2)
                    svf2 = torch.linalg.svdvals(Vfa.transpose(-1, -2) @ Vfb).clamp(max=1.0 - 1e-12)
                    ff = (sfp[:-dt, rt - 1] / sfp[:-dt, 0].clamp_min(1e-30) < FLAT_REL).numpy()
                    dgp_[dt] = np.where(mr_ & ~ff, torch.acos(svf2).max(-1).values.numpy(), np.nan)
            dg['draw'][rt] = dgr
            dg['dgf'][rt] = dgg
            dg['dgp'][rt] = dgp_
        out['diag'] = dg
    return out


def dS_of_H(Hdoc):
    H = Hdoc.float()
    n = H.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    u = H / n
    c = (u[:-1] * u[1:]).sum(-1).clamp(-1, 1)
    return torch.acos(c)                                       # [T-1]


def med_iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return None
    m = float(np.median(x))
    q1, q3 = np.percentile(x, [25, 75])
    return {'med': m, 'iqr': [float(q1), float(q3)]}


def spearman(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return None
    ra = np.argsort(np.argsort(a[m])); rb = np.argsort(np.argsort(b[m]))
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------- depth leg (raw vs gf coherence) ----------------

def depth_leg(Va, Vb, sent, rng):
    """Va, Vb: [T, D] top-right per position. Coherence vs random R^D pairs.
    NOT a T^h transport (no D_l in cache) — coherence only, flagged in JSON."""
    T = min(Va.shape[0], Vb.shape[0])
    va = Va[:T] / np.linalg.norm(Va[:T], axis=1, keepdims=True).clip(1e-30)
    vb = Vb[:T] / np.linalg.norm(Vb[:T], axis=1, keepdims=True).clip(1e-30)
    mask = ~(np.asarray(sent[:T], dtype=bool))
    cos = np.abs((va * vb).sum(-1))[mask]
    if len(cos) < 8:
        return None
    ra = rng.standard_normal((N_DEP_SHUF, Va.shape[1]))
    ra /= np.linalg.norm(ra, axis=1, keepdims=True)
    cos_rand = np.abs(ra @ vb.T).ravel()
    return {'cos_real': float(np.median(cos)), 'cos_rand': float(np.median(cos_rand)),
            'p': float((cos_rand >= np.median(cos)).mean()), 'n': int(len(cos))}


# ---------------- aggregation (E-34 protocol) ----------------

def agg_positions(vals_per_doc, min_n=8):
    """per-doc medians -> median of medians."""
    meds = []
    for v in vals_per_doc:
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if len(v) >= min_n:
            meds.append(float(np.median(v)))
    if not meds:
        return None
    return {'med': float(np.median(meds)), 'iqr': [float(np.percentile(meds, 25)),
                                                    float(np.percentile(meds, 75))],
            'n_docs': len(meds)}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--ckpt-dir', default='/data/prism_kernels/exp27b/results')
    a.add_argument('--out', default='results/exp33r_gauge_anatomy.json')
    a.add_argument('--layers', default='6,12,18,22,23')
    a.add_argument('--r', type=int, default=R_MAIN)
    a.add_argument('--diag', action='store_true',
                   help='E-33r.2 diagnostics A/C/D/B (+E, F) on L23')
    a.add_argument('--sample-cap', type=int, default=0)
    a.add_argument('--selfcheck', action='store_true')
    a = a.parse_args()
    if a.selfcheck:
        selfcheck(); return
    torch.set_num_threads(1)
    E28._pin_p_cores()
    np.random.seed(SEED)
    layers = [int(x) for x in a.layers.split(',')]
    torch_layers = {}
    docs = None
    for l in layers:
        torch_layers[l] = load_layer(a.ckpt_dir, l)
        if docs is None:
            ck = torch_layers[l]
            domains = sorted({k[0] for k in ck['JR']})
            docs = [(d, si) for d in domains for si in range(ck['n_samples'])]
    if a.sample_cap:
        docs = docs[:a.sample_cap]
    S = torch_layers[layers[0]]['S']
    K = torch_layers[layers[0]]['K']

    per_doc = {l: [] for l in layers}          # list of batch_metrics dicts
    dS23 = []
    for i, (dom, si) in enumerate(docs):
        for l in layers:
            ck = torch_layers[l]
            if ck['n_samples'] != torch_layers[layers[0]]['n_samples']:
                raise RuntimeError('n_samples differs across layers')
            Jdoc, Pdoc, sent = load_doc(ck, dom, si)
            m = batch_metrics(Jdoc, Pdoc, sent, r=a.r, diag=a.diag and l == 23)
            m['sent_frac'] = float(sent.mean())
            per_doc[l].append(m)
            del Jdoc, Pdoc
            if l == 23:
                dS23.append(dS_of_H(load_H(ck, dom, si)).numpy())
        print(f'doc {i+1}/{len(docs)} {dom}/{si}', flush=True)

    # ---------------- per-layer aggregation ----------------
    layer_out = {}
    for l in layers:
        rows = per_doc[l]
        if not rows:
            continue
        fields_pos = ['uniformity_0', 'gauge_share', 'gauge_share_exact',
                      'align_ug', 'align_p', 'survive_frac', 'kill_frac',
                      'kill_frac_check', 'rank_raw', 'rank_gf', 'rank_K', 'align_K1']
        for k in range(1, 9):
            fields_pos.append(f'sig_ratio_{k}')
        entry = {'n_docs': len(rows),
                 'sent_frac_med': agg_positions([np.array([r['sent_frac']]) for r in rows], min_n=1) or {}}
        for f in fields_pos:
            entry[f] = agg_positions([r[f] for r in rows])
        entry['sK_top4'] = [agg_positions([np.ravel(r['sK_top4'][:, j]) for r in rows])
                            for j in range(4)]
        entry['delta'] = {}
        for dt in DELTA_GRID:
            entry['delta'][str(dt)] = {
                'raw': agg_positions([r[f'draw_{dt}'] for r in rows]),
                'gf': agg_positions([r[f'dgf_{dt}'] for r in rows]),
                'gain': agg_positions([r[f'gain_{dt}'] for r in rows])}
        if a.diag and rows and all('diag' in r_ for r_ in rows):
            rts = [rt for rt in DIAG_R if rt in rows[0]['diag']['dgf']]
            entry['diag'] = {
                'C_raw': {str(rt): {str(dt): agg_positions([r_['diag']['draw'][rt][dt] for r_ in rows])
                                    for dt in DELTA_GRID} for rt in rts},
                'C_gf': {str(rt): {str(dt): agg_positions([r_['diag']['dgf'][rt][dt] for r_ in rows])
                                   for dt in DELTA_GRID} for rt in rts},
                'C_gain': {str(rt): {str(dt): agg_positions([np.nan_to_num(
                                       np.asarray(r_['diag']['draw'][rt][dt], float) -
                                       np.asarray(r_['diag']['dgf'][rt][dt], float),
                                       nan=np.nan) for r_ in rows])
                                     for dt in DELTA_GRID} for rt in rts},
                'E_fixed_p': {str(rt): {str(dt): agg_positions([r_['diag']['dgp'][rt][dt] for r_ in rows])
                                        for dt in DELTA_GRID} for rt in rts if rt in (16, 32)},
                'E_fixed_p_r': {str(rt): {str(dt): agg_positions([r_[f'dgp_{dt}_r{rt}'] for r_ in rows])
                                          for dt in DELTA_GRID} for rt in (1, 2, 8, 16)},
                'D_omega': {str(dt): agg_positions([r_[f'omega_{dt}'] for r_ in rows])
                            for dt in DELTA_GRID},
                'D_omega_by_r': {str(rt): {str(dt): agg_positions([r_[f'omega_{dt}_r{rt}'] for r_ in rows])
                                           for dt in DELTA_GRID}
                                 for rt in (1, 2, 8, 16)},
            }
            bres = {}   # B: per-doc Spearman(Δ_gf(1, r), d_S), per r
            for rt in rts:
                sps = []
                for m_, ds in zip(rows, dS23):
                    dg1 = np.asarray(m_['diag']['dgf'][rt][1], dtype=float)
                    msk = np.isfinite(dg1) & np.isfinite(ds)
                    if msk.sum() >= 8:
                        sps.append(spearman(dg1[msk], ds[msk]))
                bres[str(rt)] = {'med': float(np.median(sps)), 'n': len(sps)} if sps else None
            entry['diag']['B_rho_by_r'] = bres
        # slice 6: correlations
        if l == 23 and dS23:
            # per-doc: Spearman(Δ_gf(1), d_S) — pooled over docs (Δ_gf(1) [T-1] vs d_S [T-1])
            rows23 = per_doc[23]
            sp_gf, sp_raw = [], []
            for m, ds in zip(rows23, dS23):
                dg = np.asarray(m['dgf_1'], dtype=float)
                dr = np.asarray(m['draw_1'], dtype=float)
                msk = np.isfinite(dg) & np.isfinite(ds)
                if msk.sum() >= 8:
                    sp_gf.append(spearman(dg[msk], ds[msk]))
                    sp_raw.append(spearman(dr[msk], ds[msk]))
            entry['corr_with_dS'] = {'gf_d1': {'med': float(np.median(sp_gf)), 'n': len(sp_gf)} if sp_gf else None,
                                     'raw_d1': {'med': float(np.median(sp_raw)), 'n': len(sp_raw)} if sp_raw else None}
        # gain vs gauge_share_exact (per-doc Spearman, pooled)
        sp_ge, sp_kg = [], []
        for m in rows:
            gain1 = np.asarray(m['gain_1'], dtype=float)            # pair (t, t+1): [S-1]
            gse = np.asarray(m['gauge_share_exact'], dtype=float)[:-1]  # per-position → t
            kf = np.asarray(m['kill_frac'], dtype=float)[:-1]
            msk = np.isfinite(gain1) & np.isfinite(gse) & np.isfinite(kf)
            if msk.sum() >= 8:
                sp_ge.append(spearman(gain1[msk], gse[msk]))
                sp_kg.append(spearman(kf[msk], gse[msk]))
        entry['corr_gain_gauge_share'] = {'med': float(np.median(sp_ge)), 'n': len(sp_ge)} if sp_ge else None
        entry['corr_kill_gauge_share'] = {'med': float(np.median(sp_kg)), 'n': len(sp_kg)} if sp_kg else None
        # E-33r.2 B: ratio = sig_gf/sig_raw, mean over i<r, per r
        for rt in (1, 2, 4, 8, 16, 32):
            if f'ratio_{rt}' in rows[0]:
                entry[f'ratio_{rt}'] = agg_positions([r_[f'ratio_{rt}'] for r_ in rows])
        layer_out[str(l)] = entry

    # ---------------- depth coherence (raw vs gf) ----------------
    depth_out = {}
    rng = np.random.default_rng(SEED + 99)
    # Coherence — not a pair metric, so no sentinel mask (endpoints apply to pairs only).
    for (la, lb) in DEPTH_PAIRS:
        if la not in per_doc or lb not in per_doc:
            continue
        res_raw, res_gf = [], []
        for i in range(min(len(per_doc[la]), len(per_doc[lb]))):
            va = per_doc[la][i]['vtop_raw']; vb = per_doc[lb][i]['vtop_raw']
            wga = per_doc[la][i]['vtop_gf']; wgb = per_doc[lb][i]['vtop_gf']
            rr = depth_leg(va, vb, np.zeros(S, dtype=bool), rng)
            rg = depth_leg(wga, wgb, np.zeros(S, dtype=bool), rng)
            if rr:
                res_raw.append(rr)
            if rg:
                res_gf.append(rg)
        depth_out[f'{la}-{lb}'] = {
            'raw': {'cos': [x['cos_real'] for x in res_raw],
                    'p': [x['p'] for x in res_raw],
                    'med_cos': float(np.median([x['cos_real'] for x in res_raw])) if res_raw else None},
            'gf': {'cos': [x['cos_real'] for x in res_gf],
                   'p': [x['p'] for x in res_gf],
                   'med_cos': float(np.median([x['cos_real'] for x in res_gf])) if res_gf else None},
            'depth_gain': (float(np.median([x['cos_real'] for x in res_gf]) -
                                  np.median([x['cos_real'] for x in res_raw]))
                           if res_gf and res_raw else None),
            # E-33r.2 F: per-r subspace overlap Ω = ‖U_gfᵀU_raw‖²_F/r, r ∈ {1,4,16}
            'by_r': {},
            'note': 'coherence, NOT T^h transport (no D_l in cache)'}
        for rt in (1, 4, 16):
            ov_raw, ov_gf = [], []
            for i in range(len(per_doc[la])):
                a_r = per_doc[la][i]['vt16_raw'][:, :rt, :]
                b_r = per_doc[lb][i]['vt16_raw'][:, :rt, :]
                a_g = per_doc[la][i]['vt16_gf'][:, :rt, :]
                b_g = per_doc[lb][i]['vt16_gf'][:, :rt, :]
                ov_raw.append(float(np.nanmedian(((a_r @ b_r.transpose(0, 2, 1)) ** 2).sum(-1).sum(-1) / rt)))
                ov_gf.append(float(np.nanmedian(((a_g @ b_g.transpose(0, 2, 1)) ** 2).sum(-1).sum(-1) / rt)))
            if ov_raw:
                depth_out[f'{la}-{lb}']['by_r'][str(rt)] = {
                    'raw': float(np.median(ov_raw)), 'gf': float(np.median(ov_gf)),
                    'gain': float(np.median(ov_gf) - np.median(ov_raw))}

    # E-33r.2 F: depth coherence gf on L12-L18, L18-L22 (from the depth leg)
    F_depth = {f'{la}-{lb}': (depth_out.get(f'{la}-{lb}') or {}).get('gf', {}).get('med_cos')
               for la, lb in ((12, 18), (18, 22))}

    # ---------------- negative control (original E-33) ----------------
    neg = None
    neg_path = 'results/exp33_gauge_holonomy.json'
    if os.path.exists(neg_path):
        try:
            nd = json.load(open(neg_path))
            per_l = {}
            for l in layers:
                e = nd['layers'].get(str(l))
                if e:
                    per_l[str(l)] = {'pos1': e.get('pos1'), 'pos8': e.get('pos8')}
            neg = {'note': 'E-33 original: transport of u_g under P(p); u_g ≈ uniform, '
                           'P(p)·1 = 0 — degenerate test, kept as negative control',
                   'layers': per_l,
                   'verdict_original': 'borderline signal L23 (p=0.0445) — fails Bonferroni',
                   'verdict_corrected': 'trivial on all layers (effect size ~0.6σ at K=128)'}
        except Exception:
            neg = None

    # ---------------- verdict ----------------
    gains = {l: (layer_out[str(l)]['delta']['1']['gain'] or {}).get('med') for l in layers}
    gain_pos = sum(1 for v in gains.values() if v is not None and v > 0.05)
    gain_neg = sum(1 for v in gains.values() if v is not None and v < -0.05)
    gain_strong_neg = sum(1 for v in gains.values() if v is not None and v < -0.5)
    gain_l23 = gains.get(23)
    # три ветки: + / 0 / −
    if gain_l23 is not None and abs(gain_l23) < 0.05:
        v23 = 'L23: Δ_gain(1) ≈ 0 → gf ≡ raw (E-35 факторизация точна)'
    elif gain_l23 is None:
        v23 = 'L23: no data'
    elif gain_l23 < -0.05:
        strength = 'сильный' if gain_l23 < -0.5 else 'умеренный'
        v23 = (f'L23: Δ_gain(1) = {gain_l23:+.4f} ({strength}) → лист L23 живёт в '
               f'gauge-компоненте; P(p)·J неструктурирован по позиции и глубине — '
               f'новое утверждение для §18')
    else:
        v23 = f'L23: Δ_gain(1) = {gain_l23:+.4f} (gf стабильнее raw)'
    if gain_pos >= 3:
        vmain = (f'Δ_gain > 0.05 on {gain_pos}/{len(layers)} layers — E-33r входит в §18 '
                 f'как карта raw vs gf')
    elif gain_neg >= 3 or gain_strong_neg >= 1:
        vmain = (f'Δ_gain < −0.05 on {gain_neg}/{len(layers)} layers '
                 f'(сильных < −0.5: {gain_strong_neg}) — новое утверждение: структура J '
                 f'локализована в gauge-компоненте; механизм (изменчивость p_t / фильтр P) — '
                 f'E-33r.2 (docs/Exp33r2.md)')
    else:
        vmain = (f'Δ_gain ≈ 0 на всех слоях ({gain_pos} полож., {gain_neg} отриц.) — '
                 f'количественная картина (gauge = uniform mode), без нового утверждения')
    verdict = f'{v23}; {vmain}'

    report = {
        'meta': {'seed': SEED, 'r': a.r, 'delta_grid': DELTA_GRID,
                 'sentinel': 'endpoints (t, t+Δ)', 'sent_threshold': SENT,
                 'layers': layers, 'n_docs': len(docs),
                 'pool_offsets': 'not needed — all per-position',
                 'depth': 'coherence, not transport (no D_l in cache)',
                 'e34_protocol': 'per-doc median/IQR -> median of 32 docs',
                 'e33r2_diag': bool(a.diag)},
        'layers': layer_out,
        'depth': depth_out,
        'F_depth_gf': F_depth,
        'negative_control': neg,
        'verdict': verdict,
        'verdict_gains': {str(l): g for l, g in gains.items()},
    }

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            f = float(o)
            return f if np.isfinite(f) else None
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(_clean(report), f, indent=2, allow_nan=False, ensure_ascii=False)
    print(f'saved {a.out}', flush=True)
    print('VERDICT:', verdict, flush=True)


# ---------------- selfcheck ----------------

def selfcheck():
    torch.manual_seed(0)
    K, D = 8, 16
    one = torch.ones(K) / K ** 0.5

    # (1) p_apply vs diag(p) − pp^T
    p = torch.rand(K); p /= p.sum()
    J1 = torch.randn(K, D)
    Pm = torch.diag(p) - torch.outer(p, p)
    ref = Pm @ J1
    got = p_apply(J1.unsqueeze(0), p.unsqueeze(0))[0]
    assert (ref - got).abs().max() < 1e-4, 'p_apply'

    # (2) P kills uniform: J = 1·wᵀ/√K (all rows equal) → P·J = 0 exactly
    u1 = torch.ones(K) / K ** 0.5
    w1 = torch.randn(D)
    Jv = (u1.unsqueeze(1) * w1.unsqueeze(0)).unsqueeze(0)          # [1,K,D]
    assert p_apply(Jv, p.unsqueeze(0)).norm() < 1e-5, 'P kills uniform'

    # (3) rank-1 J = u v^T, p = uniform = 1/√K.
    # Exact: P·J = (1/√K)(u − c·Ī)vᵀ, c = <u,Ī> — gauge component c·Īvᵀ killed 100%,
    # top-right subspace (v) unchanged, survive = √(1−c²)/√K (NOT 1−c²).
    u = torch.randn(K); u = u / u.norm()
    v = torch.randn(D); v = v / v.norm()
    Jun = (u.unsqueeze(1) * v.unsqueeze(0)).unsqueeze(0)
    pup = torch.ones(K) / K
    m = batch_metrics(Jun, pup.unsqueeze(0), np.zeros(1, dtype=bool), r=2)
    uf = float((u @ one) ** 2)                                    # <u, 1/√K>^2
    assert abs(m['uniformity_0'][0] - uf) < 1e-4, (m['uniformity_0'][0], uf)
    surv_exact = float(np.sqrt(1 - uf) / K)                    # P(1/K) scales 1^⊥ by 1/K
    assert abs(m['survive_frac'][0] - surv_exact) < 1e-3, (m['survive_frac'][0], surv_exact)
    # gauge share (top-1 and exact) = uniformity_0 for rank-1
    assert abs(m['gauge_share'][0] - uf) < 1e-3, (m['gauge_share'][0], uf)
    assert abs(m['gauge_share_exact'][0] - uf) < 1e-3, (m['gauge_share_exact'][0], uf)
    # top-right subspace preserved: vtop_gf ≈ ±v
    vg = m['vtop_gf'][0]
    v_np = v.numpy() / np.linalg.norm(v.numpy())
    assert abs(float(np.dot(vg, v_np))) > 0.999, 'subspace moved'
    # K_t rank-1 (u not ~ uniform)
    assert m['rank_K'][0] <= 1, m['rank_K'][0]

    # (4) p=uniform: P·J = J/K when Jᵀ1 = 0 → same right subspaces, draw == dgf.
    # Without Jᵀ1=0 the top-r subspaces DO shift (rank-1 correction to JᵀJ) — see test 4b.
    one = torch.ones(K) / K ** 0.5
    TT = 20                                            # > max(DELTA_GRID)
    J2 = torch.randn(TT, K, D)
    J2 = J2 - torch.einsum('tkd,k->td', J2, one)[:, None, :] * one[None, :, None]   # J2ᵀ·1 = 0
    pup2 = torch.ones(K) / K
    m2 = batch_metrics(J2, pup2.unsqueeze(0), np.zeros(TT, dtype=bool), r=2, diag=True)
    for dt in DELTA_GRID:
        d = np.abs(m2[f'draw_{dt}'] - m2[f'dgf_{dt}'])
        assert np.nanmax(d) < 1e-3, (dt, d)
    # (4a) omega: same subspaces (J2^T·1=0, p=uniform) → omega ≈ 1; diag keys present
    for dt in DELTA_GRID:
        om = np.asarray(m2[f'omega_{dt}'], float)
        assert np.nanmax(np.abs(om - 1.0)) < 1e-3, (dt, om)
        om1 = np.asarray(m2[f'omega_{dt}_r1'], float)
        assert np.nanmax(np.abs(om1 - 1.0)) < 1e-3, (dt, 'omega_r1')
        assert np.isfinite(m2[f'ratio_2'][0]), 'ratio_2'
        for rt in (rt for rt in DIAG_R if rt in m2['diag']['dgf']):
            assert np.asarray(m2['diag']['dgf'][rt][dt], float).shape[0] == TT - dt
    # (4c) Omega non-degenerate: random J + random p → top-r subspaces differ,
    # Ω ≈ r/D (r=2, D=16 → 0.125), certainly < 0.8; the inverted formula
    # U_gf·U_rawᵀ gives exactly 1 for ANY orthonormal bases (old test 4a was degenerate).
    Jr2 = torch.randn(TT, K, D)
    Pr2 = torch.rand(TT, K); Pr2 = Pr2 / Pr2.sum(-1, keepdim=True)
    mr2 = batch_metrics(Jr2, Pr2, np.zeros(TT, dtype=bool), r=2, diag=True)
    om_med = float(np.nanmedian(mr2['omega_1']))
    assert om_med < 0.8, (om_med, 'Omega: random subspaces expect ~r/D, not 1')

    # (4b) negative: random J (Jᵀ1 ≠ 0) → draw != dgf at dt=1 (test measures real shift)
    J2r = torch.randn(TT, K, D)
    m2r = batch_metrics(J2r, pup2.unsqueeze(0), np.zeros(TT, dtype=bool), r=2)
    assert np.nanmax(np.abs(m2r['draw_1'] - m2r['dgf_1'])) > 1e-3, 'test 4 would be dead'

    # (5) sentinel endpoints: sent[t]=True removes (t, t+1)
    Js = torch.randn(6, K, D)
    Ps = torch.rand(6, K); Ps = Ps / Ps.sum(-1, keepdim=True)
    sent = np.zeros(6, dtype=bool); sent[2] = True
    ms = batch_metrics(Js, Ps, sent, r=2)
    d1 = ms['draw_1']
    assert not np.isfinite(d1[1]) and not np.isfinite(d1[2]), 'sentinel t=2'
    assert np.isfinite(d1[0]) and np.isfinite(d1[3]), 'sentinel only endpoints'

    # (6) depth_leg: identical vtop -> cos ~ 1
    va = np.random.default_rng(0).standard_normal((10, D))
    va /= np.linalg.norm(va, axis=1, keepdims=True)
    r = depth_leg(va, va.copy(), np.zeros(10, dtype=bool), np.random.default_rng(1))
    assert r['cos_real'] > 0.99, r
    # random vs random -> cos ~ 0
    vb = np.random.default_rng(2).standard_normal((10, D))
    vb /= np.linalg.norm(vb, axis=1, keepdims=True)
    r2 = depth_leg(va, vb, np.zeros(10, dtype=bool), np.random.default_rng(3))
    assert r2['cos_real'] < 0.3, r2

    # (7) med_iqr guard
    assert med_iqr(np.array([1.0] * 5)) is None
    assert med_iqr(np.arange(10.0))['med'] == 4.5

    print('selfcheck OK')


if __name__ == '__main__':
    main()
