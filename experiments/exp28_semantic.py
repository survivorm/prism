#!/usr/bin/env python3
"""Exp 28.2 — semantic correlation of the L23 leaf boundaries (docs/Exp28.md step 3,
previously skipped as 'token types not in cache'). The exp27b ckpt DOES contain the
position data the TZ step 3 asks for: P128[(dom,si,t)] = p_t on U128, t = 0..127.
Token IDs themselves stay out (not cached) -> token-type bullet remains skipped;
syntax proxy = p_t dynamics, which is exactly the position signal the TZ bullet
is about.

v2 per docs/Exp28_errors.md review fixes:
  1.1 pooled P-axes via 2D SVD (3D .t() crashed)
  1.2 absz reshape = [N,T1,B_TOP]
  1.3 dom_out masks concatenated over the 8 docs
  1.4 kappa profile .norm(dim=1)
  1.5 H no longer loaded (dynamics on entrop(P) only)
  2.1 ONE SVD per document, V cached at r=32, reused for the r-sweep,
      domain_blocks and kappa
  2.3 vectorized perm_test (exact label counts, B x M boolean matrix)

Tests (CPU, no GPU, no model; r=16 to reproduce exp28 MAD boundaries exactly):
  1. P-axes: top-8 right singular vectors of the pooled (P_t - mean) stack;
     per-token z-scores within doc; |z| boundary vs rest (permutation B=2000);
     boundary-adjacent (t-2..t+2) vs far; per-domain breakdown.
  2. Dynamics: dH = entrop(P_{t+1}) - entrop(P_t); ds = ||P_{t+1}-P_t||_2;
     boundary vs non-boundary, permutation p.
  3. Domain blocks: cosine distance between L23 d(t) profiles -> average
     linkage, k=4, ARI vs domain labels.
  4. r-sensitivity (8/16/32) of the boundary set: Jaccard pooled + per-domain.
  5. Pooled kappa profile L23 (r=16) - check peaks at t ~ 31/63/95.
Boundary definition = exp28: d(t) > median + max(2*MAD, 1e-6) per document
(d over t=0..T1-1, r=16). Output: results/exp28_semantic.json.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CKPT_DIR = '/data/prism_kernels/exp27b/results'
E28_JSON = '/data/prism_kernels/exp28/exp28_leaf_structure.json'
R_MAIN = 16
B_PERM = 2000
B_TOP = 8
LAYER = 23
SEED = 20260928
N_THREADS = 1  # one thread per process; 4 forked processes pinned OUTSIDE via taskset (user rule)

RESULTS = 'results/exp28_semantic.json'


def load_all():
    ck = torch.load(os.path.join(CKPT_DIR, f'exp27b_jcache_L{LAYER}.pt'),
                    map_location='cpu', weights_only=False)
    n, S, Kp = len(ck['domains']) * ck['n_samples'], ck['S'], len(ck['U128'])
    P = torch.stack([ck['P128'][(d, s, t)] for d in ck['domains']
                     for s in range(ck['n_samples']) for t in range(S)]).view(n, S, Kp)
    return ck, P


def d_profile_from_V(V, r):
    """d(t) = max principal angle between V_t and V_{t+1}; V: [T, D, r]."""
    Vr = V[:, :, :r]
    M = Vr[:-1].transpose(-1, -2) @ Vr[1:]
    return torch.acos(torch.linalg.svdvals(M).clamp(max=1.0 - 1e-12)).max(-1).values


def boundary_mask(d, flat_t=1e-6):
    dm = float(torch.nanmedian(d))
    mad = float(torch.nanmedian((d - dm).abs()))
    return (d > dm + max(2.0 * mad, flat_t))


def perm_test(x, labels, B=B_PERM, seed=SEED):
    """Mean x[labels] - mean x[~labels]; one-sided p via B random masks with the
    EXACT same number of positives as the real labels (vectorized)."""
    x = np.asarray(x, float)
    l = np.asarray(labels, bool)
    stat = x[l].mean() - x[~l].mean()
    M = x.size
    g = np.random.default_rng(seed)
    lab = g.random((B, M)) < l.mean()
    sel = lab @ x
    cnt = lab.sum(axis=1)
    stats = sel / np.maximum(cnt, 1) - (x.sum() - sel) / np.maximum(M - cnt, 1)
    return float(stat), float((stats >= stat).sum() + 1) / (B + 1)


def _svd_docs(Js, idxs, conn):
    torch.set_num_threads(1)
    for i, J in zip(idxs, Js):
        _, _, Vh = torch.linalg.svd(J, full_matrices=False)
        conn.send((i, Vh[:, :32].transpose(-1, -2).numpy()))
    conn.close()


def _pin():
    # cores pinned OUTSIDE via taskset -c (user rule); single-threaded here,
    # the SVD batch loop is split across 4 forked processes in main()
    torch.set_num_threads(N_THREADS)


def main():
    _pin()
    ck, P = load_all()
    N, T, K = P.shape
    T1 = T - 1
    docs28_list = json.load(open(E28_JSON))['layers'][str(LAYER)]['docs']
    docs28 = {(d['dom'], d['si']): d for d in docs28_list}
    doms_order = [(d['dom'], d['si']) for d in docs28_list]
    J_all = ck['JR']

    # --- SVD per document, 4 forked processes (batched CPU SVD is sequential) ---
    import multiprocessing as mp
    Js = [torch.stack([J_all[(d, s, t)] for t in range(T)]) for (d, s) in doms_order]
    conns = [mp.Pipe() for _ in range(4)]
    procs = []
    for w in range(4):
        c = mp.Process(target=_svd_docs, args=(Js[w::4], list(range(w, N, 4)), conns[w][1]))
        c.daemon = True
        c.start()
        procs.append(c)
    V_all = [None] * N
    for w in range(4):
        for _ in range(len(range(w, N, 4))):
            i, v = conns[w][0].recv()
            V_all[i] = torch.from_numpy(np.asarray(v))
    for c in procs:
        c.join()
    del Js
    d16 = [d_profile_from_V(V, 16) for V in V_all]
    msk = {r: [boundary_mask(d_profile_from_V(V, r)).numpy() for V in V_all] for r in (8, 16, 32)}

    # self-check: r=16 boundaries match the saved exp28 JSON exactly
    for i, (d, s) in enumerate(doms_order):
        jset = set(docs28[(d, s)]['boundary_positions'])
        a = set(int(t) for t in np.nonzero(msk[16][i])[0])
        assert jset == a, (i, len(jset), len(a))
    print('boundary masks: r=16 matches exp28 JSON for all 32 docs')

    # --- p-axes: top B_TOP right singular directions of the pooled P stack (2D SVD) ---
    Pc = (P - P.mean(0, keepdim=True)).reshape(N * T, K)
    _, _, VH = torch.linalg.svd(Pc, full_matrices=False)    # VH [K, NT]
    Axes = VH.t()[:, :B_TOP]                                 # [K, B_TOP]
    zc = (P - P.mean(0, keepdim=True)) @ Axes                # [N,T,B_TOP]
    Zs = (zc - zc.mean(1, keepdim=True)) / zc.std(1, keepdim=True, unbiased=False).clamp_min(1e-9)

    # --- dynamics axes ---
    lp = torch.log(P.clamp_min(1e-12))
    Hp = -(P * lp).sum(-1)                                   # [N,T]
    dH = Hp[:, 1:] - Hp[:, :-1]                               # [N,T1] (along time, not docs!)
    dP = (P[:, 1:] - P[:, :-1]).norm(dim=-1)                 # [N,T1]
    m, sd = dH.mean(), dH.std(unbiased=False).clamp_min(1e-9)
    dHz = (dH - m) / sd
    m, sd = dP.mean(), dP.std(unbiased=False).clamp_min(1e-9)
    dPz = (dP - m) / sd

    # --- test 1: boundaries (r=16) vs non-boundary, per p-axis ---
    lab = np.zeros(N * T1, bool)
    for i in range(N):
        lab[i * T1:(i + 1) * T1] = msk[16][i]
    absz = Zs[:, :T1, :B_TOP].numpy().reshape(N * T1, B_TOP)
    axes_out = {}
    adj = np.zeros(N * T1, bool)
    for i in range(N):
        for t in np.nonzero(msk[16][i])[0]:
            for u in range(max(0, t - 2), min(T1, t + 3)):
                adj[i * T1 + u] = True
    for a in range(B_TOP):
        stat, pval = perm_test(absz[:, a], lab)
        stat_a, pval_a = perm_test(absz[:, a], adj)
        axes_out[f'axis{a}'] = {
            'mean_absz_boundary_vs_rest': round(stat, 4), 'p_perm': round(pval, 5),
            'adjacent_vs_far': round(stat_a, 4), 'p_perm_adj': round(pval_a, 5),
        }

    # per-position profile of the axes (pooled over docs), thirds
    prof = Zs.numpy().mean(0)                                # [T,B_TOP]
    thirds = [[float(prof[i * T // 3:min((i + 1) * T // 3, T), a].mean()) for i in range(3)]
              for a in range(B_TOP)]
    # per-domain mean |z| at boundaries vs rest
    dom_out = {}
    for di, dom in enumerate(ck['domains']):
        labd = np.concatenate([msk[16][di * 8 + j] for j in range(8)])
        v = absz[di * 8 * T1:(di + 1) * 8 * T1, :].reshape(-1, B_TOP)
        for a in range(B_TOP):
            dom_out.setdefault(dom, {})[f'a{a}'] = round(float(v[labd, a].mean() - v[~labd, a].mean()), 4)

    # --- test 2: dynamics (entropy / p-jump) at boundaries ---
    dyn = {}
    for nm, Z in (('dH', dHz), ('dP', dPz)):
        stat, pval = perm_test(Z.numpy().reshape(-1), lab)
        dyn[nm] = {'boundary_vs_rest': round(stat, 4), 'p_perm': round(pval, 5)}

    # --- test 3: domain blocks — hierarchical clustering of L23 d-profiles ---
    Dmat = np.stack([d16[i].numpy() for i in range(N)])      # [N,T1], reuses SVD
    nr = np.linalg.norm(Dmat, axis=1, keepdims=True).clip(min=1e-12)
    Dn = Dmat / nr
    C = Dn @ Dn.T                                             # cosine [N,N]
    dist = np.clip(1 - C, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import adjusted_rand_score
    Zlink = linkage(squareform(dist, checks=False), method='average')
    lab4 = fcluster(Zlink, t=4, criterion='maxclust')
    domlab = np.array([di for di in range(4) for _ in range(8)])
    ari = float(adjusted_rand_score(domlab, lab4))

    # --- test 4: boundary-set r-sensitivity (Jaccard, pooled + per-domain) ---
    rset = {r: np.zeros(N * T1, bool) for r in (8, 16, 32)}
    for r in (8, 16, 32):
        for i in range(N):
            rset[r][i * T1:(i + 1) * T1] = msk[r][i]
    jac = {}
    for r in (8, 32):
        a, b = rset[r], rset[16]
        pooled_jac = float((a & b).sum() / max((a | b).sum(), 1))
        per_dom = []
        for di in range(4):
            aa, bb = a[di * 8 * T1:(di + 1) * 8 * T1], b[di * 8 * T1:(di + 1) * 8 * T1]
            per_dom.append(round(float((aa & bb).sum() / max((aa | bb).sum(), 1)), 3))
        jac[str(r)] = {'jaccard_vs_r16': round(pooled_jac, 4), 'per_domain': per_dom}

    # --- test 5: pooled kappa profile L23 (r=16) — check peaks at t ~ 31/63/95 ---
    def _pol(M):
        U, _, Wt = torch.linalg.svd(M, full_matrices=False)
        return U @ Wt

    krow = []
    for V in V_all:                                           # reuses the cached SVD
        Vr = V[:, :, :16]
        M01 = Vr[:-2].transpose(-1, -2) @ Vr[1:-1]
        M12 = Vr[1:-1].transpose(-1, -2) @ Vr[2:]
        M02 = Vr[:-2].transpose(-1, -2) @ Vr[2:]
        Hhol = _pol(M01) @ _pol(M12) @ _pol(M02).transpose(-1, -2)
        krow.append((Hhol - torch.eye(16)).flatten(1).norm(dim=1).numpy())
    kprof = np.nanmean(np.stack(krow), axis=0)
    peaks = sorted(range(1, len(kprof) - 1), key=lambda t: -kprof[t])[:5]

    out = {
        'params': {
            'layer': LAYER, 'r_main': R_MAIN, 'b_perm': B_PERM, 'b_top': B_TOP,
            'seed': SEED, 'n_docs': N, 'T': T,
            'boundary_def': 'd(t) > median + max(2*MAD, 1e-6), per doc, r=16 (== exp28 JSON, self-checked)',
            'p_axes': 'top-8 right singular vectors of pooled (P_t - mean); per-doc z-score',
            'token_types': 'NOT in cache — TZ bullet skipped (token IDs unavailable)',
            'perm_test': 'one-sided, vectorized, exact label counts',
            'inputs': f'{CKPT_DIR}/exp27b_jcache_L{LAYER}.pt, {E28_JSON}',
        },
        'p_axes': {'axis_thirds_mean': thirds, 'boundary_vs_rest': axes_out, 'per_domain': dom_out},
        'dynamics': dyn,
        'domain_blocks': {'ari_k4': ari},
        'r_sensitivity': jac,
        'kappa_profile_l23': {'pooled': kprof.tolist(), 'top5_peak_positions': peaks},
    }
    with open(RESULTS, 'w') as f:
        json.dump(out, f)
    print(f'[OK] saved {RESULTS}')
    for a, v in axes_out.items():
        print(f"{a}: {v}")
    print('dynamics:', dyn)
    print('ARI k=4:', round(ari, 3))
    print('r-sensitivity:', jac)
    print('kappa top peaks at t:', peaks, 'values:', [round(float(kprof[t]), 4) for t in peaks])


if __name__ == '__main__':
    main()
