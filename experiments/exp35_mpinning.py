#!/usr/bin/env python3
"""Exp 35 full — M-pinning: где «лежит» левый лист J (docs/Exp35.md).

L23, где J_raw = M_K (√D/r_t) (I − û ûᵀ) ТОЧНО (M_K = W_lm[U128] · diag(g),
g = 1+w из RMSNorm; модель read-only, только W_lm и norm.weight):
  уровень 1 (идеал): A^TA = λI на û⊥  =>  Δ_SVD(r) = d_S(û_t, û_{t+1})
  уровень 2: top-r правые подпространства c·P_u·M_K <=> top-r P_u M_KᵀM_K P_u.
             ТОЧНАЯ редукция в span(col M_Kᵀ, û_t) (dim ≤ K+1 = 129):
             Qo = QR([P_u M_Kᵀ, û]), C = Qoᵀ B Qo [129²], eigh, top-r = хвост,
             R^D = Qo Y_r. (B = M_KᵀM_K действует на S, поэтому редукция точна.)
  уровень 3: правые подпространства кэшевого raw J (E-28 метрика).
  Δ_SVD(ур.3) − Δ_SVD(ур.2) | r=16 — эффект весов softmax P(p_t).
r-sweep {1,8,16,32,128}: per-doc Spearman(Δ_SVD, d_S), медиана/IQR по docs.
L6/L12/L18/L22: ТОЛЬКО уровень 3 (факторизация точна только на L23 —
отклонение от ТЗ задокументировано).
Доп.: shuffled-û базлайн (dS-бининг, 20 бин), κ holonomy (ур.3, r=16),
alignment |<û_t, v_top-k(M_K)>| k∈{1,2,4}, SVD M_K top-16 + eigengap,
PCA траекторий û_t (per-doc + pooled).
Артефакт: results/exp35_mpinning_sweep.json.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import gc
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp28_leaf_structure as E28
import exp34_p1_coupling as E34

LAYERS = [6, 12, 18, 22, 23]
RS = [1, 8, 16, 32, 128]
R_MAIN = 16
SEED = 20260928
N_THREADS = 4


def load_model():
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3.5-0.8B', torch_dtype=torch.float32,
        attn_implementation='eager')
    m.eval()
    return m


def build_MK(model, U128):
    """M_K = W_lm[U128] · diag(g). g = 1+w (RMSNorm zero-init) — эмпирика."""
    W_lm = model.lm_head.weight.detach().float()
    w = model.model.norm.weight.detach().float()
    info = {'w_min': float(w.min()), 'w_max': float(w.max()),
            'w_mean': float(w.mean())}
    M = (W_lm[list(U128), :].float() * (1.0 + w)[None, :]).numpy().astype(np.float64)
    return M, info


def dS_pairs(H):
    """d_S(t) = arccos(û_t·û_{t+1}), H [T,D] f64. [T-1]."""
    Hn = H / np.linalg.norm(H, axis=1, keepdims=True)
    c = np.clip((Hn[:-1] * Hn[1:]).sum(-1), -1.0, 1.0)
    return np.arccos(c)


def subspace_angle(A, B):
    """max principal angle; A, B [m,·] — колонки ортонормированы."""
    sv = np.linalg.svdvals(A.T @ B)
    return float(np.arccos(np.clip(sv, 0.0, 1.0 - 1e-12)).max())


def level2_eig(M, u):
    """Q [D,K], w [K], Y [K,K]: точная редукция top-r c·P_u·M (c>0).
    Поддержка T = P_u MᵀM P_u лежит в W = span(P_u col Mᵀ) (dim ≤ K),
    T|_W: W→W: Q = QR(P_u Mᵀ), C = Qᵀ MᵀM Q [K,K]; top-r = хвост eigh."""
    D = M.shape[1]
    Q = np.linalg.qr((np.eye(D) - np.outer(u, u)) @ M.T)[0]  # [D, K]
    C = Q.T @ (M.T @ M) @ Q
    w, Y = np.linalg.eigh(C)
    return Q, w, Y


def level2_subspace(M, u, r):
    Q, w, Y = level2_eig(M, u)
    return Q @ Y[:, -r:]


def _spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(rx, ry)[0, 1])


def _med_pct(a):
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return [None, None, None]
    return [float(np.median(a)), float(np.percentile(a, 25)),
            float(np.percentile(a, 75))]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if np.isfinite(f) else None
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def doc_level3(Jt, Rs):
    """Уровень 3: Δ_SVD для всех r (один SVD) + κ (r=16). [T,K,D] torch."""
    allV = E28.right_subspaces(Jt, sorted(set(Rs)))
    Vm, sig = allV[R_MAIN]
    jn = Jt.norm(dim=(1, 2))
    deg = (sig < E28.DEGEN_REL * jn.clamp_min(1e-30)).numpy()
    out = {}
    for r in Rs:
        Vr = allV[r][0]
        M1 = Vr[:-1].transpose(-1, -2) @ Vr[1:]
        d1 = torch.acos(torch.linalg.svdvals(M1).clamp(max=1.0 - 1e-12)).max(-1).values.numpy()
        d1[deg[:-1] | deg[1:]] = np.nan
        out[r] = d1
    V = allV[R_MAIN][0]
    M01 = V[:-2].transpose(-1, -2) @ V[1:-1]
    M12 = V[1:-1].transpose(-1, -2) @ V[2:]
    M02 = V[:-2].transpose(-1, -2) @ V[2:]
    R01, R12, R02 = (E28.procrustes_batch(M) for M in (M01, M12, M02))
    k2 = np.empty(M01.shape[0])
    for t in range(M01.shape[0]):
        Hm = R01[t] @ R12[t] @ torch.linalg.inv(R02[t])
        k2[t] = float((Hm - torch.eye(R_MAIN, dtype=Hm.dtype)).norm())
    k2[deg[:-2] | deg[1:-1] | deg[2:]] = np.nan
    return out, k2, int(deg.sum())


def shuffled_baseline(d1a, dS, V16, rng):
    """d1_shuffle(t) = угол(V16_t, V16_s), s из того же dS-бина (20 бин), |s−t|>1."""
    T = len(dS) + 1
    d1s = np.full(len(dS), np.nan)
    q = np.quantile(dS, np.linspace(0, 1, 21))
    q[0], q[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(q, dS, 'right') - 1, 0, 19)
    pos = np.arange(len(dS))  # s — позиция пары (s, s+1)
    for t in range(len(dS)):
        if np.isnan(d1a[t]):
            continue
        cands = pos[(idx == idx[t]) & (np.abs(pos - t) > 1)]
        if len(cands) == 0:
            continue
        d1s[t] = subspace_angle(V16[t], V16[int(rng.choice(cands))])
    return d1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=E28.CKPT_DIR)
    ap.add_argument('--out', default='results/exp35_mpinning_sweep.json')
    ap.add_argument('--layers', default='6,12,18,22,23')
    ap.add_argument('--sample-cap', type=int, default=None)
    ap.add_argument('--selfcheck', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(N_THREADS)
    rng = np.random.RandomState(SEED)
    if args.selfcheck:
        selfcheck()
        return

    t0 = time.time()
    layers = [int(x) for x in args.layers.split(',')]
    ckL23 = E28.load_layer(args.ckpt_dir, 23)
    U128 = list(ckL23['U128'])
    S = len(ckL23['JR']) // (ckL23['n_samples'] * len(ckL23['domains']))

    model = load_model()
    MK, ginfo = build_MK(model, U128)
    del model
    _, sM, VhM = np.linalg.svd(MK, full_matrices=False)
    mk_info = {'s_top16': [float(x) for x in sM[:16]],
               'gap_16_17': float(sM[15] / max(sM[16], 1e-12)), 'g': ginfo}

    report = {'experiment': 'E-35-full: M-pinning, r-sweep', 'rs': RS,
              'model': 'Qwen/Qwen3.5-0.8B (read-only: W_lm, norm.weight)',
              'mk': mk_info,
              'note_lv23': 'lv2/1 и Δ3−Δ2 только L23: факторизация M_K точна '
                           'только после L23 (W_lm идёт после L23); L6/12/18/22 — lv3',
              'layers': {}}

    for l in layers:
        ck = ckL23 if l == 23 else E28.load_layer(args.ckpt_dir, l)
        docs = [(dom, si) for dom in ck['domains'] for si in range(ck['n_samples'])]
        if args.sample_cap:
            docs = docs[:args.sample_cap]
        print(f'L{l}: {len(docs)} docs', flush=True)
        L = {}
        for i, (dom, si) in enumerate(docs):
            Jdoc = np.stack([ck['JR'][(dom, si, t)].numpy() for t in range(S)])
            H = np.stack([ck['H'][(dom, si, t)].numpy() for t in range(S)]).astype(np.float64)
            raw = ck.get('P128_RAW', ck['P128'])
            sent = np.stack([raw[(dom, si, t)].numpy() for t in range(S)]).sum(1) < E34.SENTINEL_SUM
            dS = dS_pairs(H)
            vv = ~(sent[:-1] | sent[1:])
            Jt = torch.from_numpy(Jdoc).float()
            d3, k2, nd = doc_level3(Jt, RS)
            row = {'n_valid_pairs': int(vv.sum()), 'n_degen': nd}
            row['spearman_lv3'] = {str(r): _spearman(d3[r][vv], dS[vv]) for r in RS}
            kmask = ~(sent[:-2] | sent[1:-1] | sent[2:])
            row['kappa_r16'] = _med_pct(k2[kmask])
            V16 = E28.right_subspaces(Jt, [R_MAIN])[R_MAIN][0].numpy()
            d1s = shuffled_baseline(d3[R_MAIN], dS, V16, rng)
            row['spearman_lv3_shuffled'] = _spearman(d1s[vv], dS[vv])
            if l == 23:
                u = H / np.linalg.norm(H, axis=1, keepdims=True)
                eig = [level2_eig(MK, u[t]) for t in range(S)]  # один eigh на t
                d2 = {r: np.array([subspace_angle(eig[t][0] @ eig[t][2][:, -r:],
                                                  eig[t + 1][0] @ eig[t + 1][2][:, -r:])
                                   if vv[t] else np.nan for t in range(S - 1)])
                      for r in RS}
                row['spearman_lv2'] = {str(r): _spearman(d2[r][vv], dS[vv]) for r in RS}
                row['d3_minus_d2_r16'] = _med_pct(d3[16] - d2[16])
                row['align_u_topk_MK'] = {str(k_): float(np.median(np.abs(u @ VhM[:k_].T)))
                                          for k_ in (1, 2, 4)}
            L[f'{dom}/{si}'] = row
            del Jdoc, Jt
            if (i + 1) % 8 == 0:
                print(f'    {i + 1}/{len(docs)} {time.time() - t0:.0f}s', flush=True)
        if l != 23:
            del ck
            gc.collect()
        med = {f'spearman_lv3_{r}': _med_pct([v['spearman_lv3'][str(r)] for v in L.values()])
               for r in RS}
        med['spearman_lv3_shuffled_r16'] = _med_pct([v['spearman_lv3_shuffled'] for v in L.values()])
        med['kappa_r16'] = _med_pct([v['kappa_r16'][0] for v in L.values()])
        if l == 23:
            med['spearman_lv2'] = {str(r): _med_pct([v['spearman_lv2'][str(r)] for v in L.values()])
                                   for r in RS}
            med['spearman_lv1_ideal'] = {str(r): 1.0 for r in RS}
            med['d3_minus_d2_r16'] = _med_pct([v['d3_minus_d2_r16'][0] for v in L.values()])
            med['align_u_topk_MK'] = {str(k_): float(np.median([v['align_u_topk_MK'][str(k_)] for v in L.values()]))
                                      for k_ in (1, 2, 4)}
            # PCA траекторий û_t (per-doc и pooled) — L23 только
            Utr = np.stack([np.stack([ck['H'][(dom, si, t)].numpy() for t in range(S)])
                            for dom, si in docs])  # [n,D,1024]
            Utr = Utr / np.linalg.norm(Utr, axis=2, keepdims=True)
            per_doc = []
            for i in range(len(docs)):
                X = Utr[i] - Utr[i].mean(0, keepdims=True)
                sv = np.linalg.svdvals(X)
                per_doc.append([float(sv[0] ** 2 / (sv ** 2).sum()),
                                float(sv[1] ** 2 / (sv ** 2).sum())])
            med['pca_u_perdoc_top2'] = _med_pct(np.array(per_doc)[:, 0])
            med['pca_u_perdoc_top2sum'] = _med_pct(np.array(per_doc).sum(1))
            Xp = Utr.reshape(-1, Utr.shape[2]) - Utr.reshape(-1, Utr.shape[2]).mean(0)
            svp = np.linalg.svdvals(Xp)
            med['pca_u_pooled_top'] = [float(svp[i] ** 2 / (svp ** 2).sum()) for i in range(5)]
        report['layers'][str(l)] = {'docs': L, 'median': _clean(med)}
        print(f'  L{l} done {time.time() - t0:.0f}s', flush=True)

    with open(args.out, 'w') as f:
        json.dump(_clean(report), f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f'Wrote {args.out}  ({time.time() - t0:.0f}s)', flush=True)


def selfcheck():
    torch.manual_seed(SEED)
    rng = np.random.RandomState(1)
    D, K = 32, 8
    M = rng.randn(K, D).astype(np.float64)
    u = rng.randn(D); u /= np.linalg.norm(u)
    # (a) редукция ур.2 vs dense SVD (top-r = ПЕРВЫЕ r колонок: svals по убыванию)
    A = M @ (np.eye(D) - np.outer(u, u))
    U, _, vh = np.linalg.svd(A, full_matrices=True)
    for r in (2, 5):
        ang = subspace_angle(level2_subspace(M, u, r), vh.T[:, :r])
        assert ang < 1e-4, f'level2 r={r}: {ang}'
    # (b) dS: поворот на d
    d = 0.3
    w = rng.randn(D); w -= (w @ u) * u; w /= np.linalg.norm(w)
    u2 = np.cos(d) * u + np.sin(d) * w
    assert abs(dS_pairs(np.stack([u, u2]))[0] - d) < 1e-12
    # (c) kappa = 0 для одинаковых подпространств (E28 procrustes, O(r))
    V = np.linalg.qr(rng.randn(D, 3))[0]
    Rt = E28.procrustes(torch.from_numpy(V), torch.from_numpy(V))
    assert (Rt @ Rt @ torch.linalg.inv(Rt) - torch.eye(3)).norm().item() < 1e-10
    # (d) shuffled: кандидаты из того же бина
    T = 40
    Us = np.cumsum(rng.randn(T, D), 0); Us /= np.linalg.norm(Us, axis=1, keepdims=True)
    dSs = dS_pairs(Us)
    q = np.quantile(dSs, np.linspace(0, 1, 21)); q[0], q[-1] = -np.inf, np.inf
    idx = np.clip(np.searchsorted(q, dSs, 'right') - 1, 0, 19)
    pos = np.arange(len(dSs))
    assert len(pos[(idx == idx[5]) & (np.abs(pos - 5) > 1)]) > 0
    # (e) Spearman: идеальная корреляция = 1
    x = rng.randn(20)
    assert abs(_spearman(x, 2 * x + 1) - 1.0) < 1e-12
    print('selfcheck OK')


if __name__ == '__main__':
    main()
