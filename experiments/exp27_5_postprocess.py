#!/usr/bin/env python3
"""Exp 27.5 — post-hoc significance, coupling and clustering diagnostics of J^(p)(x).

Post-processing of results/exp27_noise_floor.json + results/exp27_jcache.pt.
No GPU. Main slice: S=64, K=128, pos=0 (token-gramian C = <J_i,J_j>_F/sqrt(E_i*E_j)).

Blocks per docs/Exp27.5.md:
  A. T_domain (permutation B=1000) + spectral null (B_rot=100) + half-split 70 partitions
  B. Spearman C vs [dh, dH, dsigma_z, dlogE, dg_uni] + entropy-quartile permutation
  C. Agglomerative clustering (average) + silhouette + gap (B_gap=100) + rho_NN
     + half-split silhouette (50) + cophenetic
  D. uniform mode: g_uni, Q=uu^T (u=1/sqrt(K) 1), J_rest, delta_C_uni

Outputs: results/exp27_5_postprocess.json
"""
import argparse
import hashlib
import json
import sys
import time
from itertools import combinations

import numpy as np
import torch

LAYERS = [6, 12, 18, 19, 22, 23]
K_MAX = 128
SEED = 20260918
B_PERM, B_ROT, B_GAP, B_HH = 1000, 100, 100, 50


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def mask_from_labels(labels, n):
    m = np.zeros((n, n), bool)
    for i, j in combinations(range(n), 2):
        if labels[i] == labels[j]:
            m[i, j] = m[j, i] = True
    return m


def gramian_cosine(J, P, U, h):
    """C_ij = <J_i, J_j>_F / sqrt(E_i E_j), E = Frobenius norm^2. (n, n)"""
    E = np.einsum('ikd,ikd->i', J, J)
    C = np.einsum('ikd,jkd->ij', J, J) / np.sqrt(np.outer(E, E) + 1e-30)
    np.fill_diagonal(C, 1.0)
    # per-sample features (gauge-free, no double projection)
    p = P / (P.sum(1, keepdims=True) + 1e-30)
    H = -np.einsum('ik,ik->i', p, np.log(p + 1e-12))
    # sigma_z,i = std over U of logits z_ik. z_k = log p_tilde_k + const,
    # so std_U z = std_U log p_tilde (recoverable from cached p, no logits needed)
    sz = np.log(p + 1e-12).std(axis=1)
    u = np.ones(J.shape[1]) / np.sqrt(J.shape[1])
    uj = np.einsum('k,ikd->id', u, J)                  # rows of Q J_i, Q = u u^T
    g = np.einsum('id,id->i', uj, uj) / (E + 1e-30)    # g_uni = ||QJ_i||^2 / E_i
    dh = np.linalg.norm(h[:, None, :] - h[None, :, :], axis=2)
    dH = np.abs(np.subtract.outer(H, H))
    dsz = np.abs(np.subtract.outer(sz, sz))
    dlogE = np.abs(np.subtract.outer(np.log(E + 1e-12), np.log(E + 1e-12)))
    dg = np.abs(np.subtract.outer(g, g))
    return C, E, dict(dh=dh, dH=dH, dsigma_z=dsz, dlogE=dlogE, dg_uni=dg)


def mean_offdiag(M, mask):
    iu = np.triu_indices_from(M, 1)
    sel = mask[iu]
    return float(M[iu][sel].mean())


def perm_test_domain(C, labels, rng, B=B_PERM):
    """T_domain = C_within - C_cross; permutation over labels."""
    n = len(labels)
    m = mask_from_labels(labels, n)
    T0 = mean_offdiag(C, m) - mean_offdiag(C, ~m)
    cnt = 0
    lab = list(labels)
    for _ in range(B):
        rng.shuffle(lab)
        mp = mask_from_labels(lab, n)
        Tp = mean_offdiag(C, mp) - mean_offdiag(C, ~mp)
        if Tp >= T0:
            cnt += 1
    return T0, (cnt + 1) / (B + 1)


def _rotated_cosine_t(Rn, Jt, En_t):
    """One-shot null cosines. Rn: (B*n, K, K) orthogonal per sample; Jt: (n, K, D)
    -> (B, n, n) cosine matrix per batch row. Kraw[i,j] uses R_i J_i vs R_j J_j —
    per-sample R (batched bmm, 100x faster than per-iter numpy qr under load)."""
    n, K, D = Jt.shape
    B = Rn.shape[0] // n
    Jrep = Jt.float().repeat(B, 1, 1).reshape(B * n, K, D)
    RA = torch.bmm(Rn, Jrep).reshape(B, n, K * D)
    Kraw = torch.bmm(RA, RA.transpose(1, 2))
    den = torch.sqrt(torch.outer(En_t, En_t) + 1e-30)
    return (Kraw / den).numpy()


def spectral_null(J, labels, rng, B=B_ROT):
    """Null: random orthogonal rotations of each J preserving its gramian eigenvalues."""
    n = len(labels)
    K = J.shape[1]
    m = mask_from_labels(labels, n)
    En = np.einsum('ikd,ikd->i', J, J)
    En_t = torch.tensor(En, dtype=torch.float32)
    En = np.einsum('ikd,ikd->i', J, J)
    C0 = np.einsum('ikd,jkd->ij', J, J) / np.sqrt(np.outer(En, En) + 1e-30)
    np.fill_diagonal(C0, 1.0)
    T0 = mean_offdiag(C0, m) - mean_offdiag(C0, ~m)
    Jt = torch.from_numpy(np.ascontiguousarray(J))
    R = torch.linalg.qr(torch.randn(B * n, K, K))[0]  # all rotations in one batched call
    cnt = 0
    for b in range(B):
        Cn = _rotated_cosine_t(R[b * n:(b + 1) * n], Jt, En_t)[0]
        Tn = mean_offdiag(Cn, m) - mean_offdiag(Cn, ~m)
        if Tn >= T0:
            cnt += 1
    return (cnt + 1) / (B + 1)


def halfsplit_reliability(C, labels, n_per, eps=1e-12):
    """All C(8,4)=70 half-splits per domain. Delta_half, S_half per spec 7.4."""
    idx = {d: [i for i, l in enumerate(labels) if l == d] for d in set(labels)}
    diffs, stabs = [], []
    for d, ids in idx.items():
        for A in combinations(ids, 4):
            B = [i for i in ids if i not in A]
            CA = mean_offdiag(C[np.ix_(A, A)],
                              mask_from_labels([0] * 4, 4))
            CB = mean_offdiag(C[np.ix_(B, B)],
                              mask_from_labels([0] * 4, 4))
            diffs.append(abs(CA - CB))
            stabs.append(1 - abs(CA - CB) / (max(abs(CA), abs(CB)) + eps))
    return float(np.mean(diffs)), float(np.mean(stabs))


def spearman(x, y):
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def pair_spearman(C, D):
    """Spearman over all 496 unique pairs (off-diagonal)."""
    iu = np.triu_indices_from(C, 1)
    return spearman(C[iu], D[iu])


def entropy_quartile_test(C, H, rng, B=1000):
    """T_ent: within-quartile C minus cross-quartile C, permuted over quartile labels."""
    n = len(H)
    order = np.argsort(H, kind='stable')
    qlab = [i // (n // 4) for i in range(n)]
    labels = [0] * n
    for i, q in zip(order, qlab):
        labels[i] = q
    m = mask_from_labels(labels, n)
    T0 = mean_offdiag(C, m) - mean_offdiag(C, ~m)
    cnt = 0
    lab = list(labels)
    for _ in range(B):
        pool = list(lab)
        rng.shuffle(pool)
        for i in range(n):
            lab[i] = pool[i]
        # group sizes stay 8/8/8/8 automatically (same multiset)
        mp = mask_from_labels(lab, n)
        Tp = mean_offdiag(C, mp) - mean_offdiag(C, ~mp)
        if Tp >= T0:
            cnt += 1
    return T0, (cnt + 1) / (B + 1), float(mean_offdiag(C, m)), float(mean_offdiag(C, ~m))


def token_gramian_cosine(J):
    """K^G_ij = <G_i, G_j>_F / (||G_i||_F ||G_j||_F), G_i = J_i J_i^T."""
    G = np.einsum('ikd,ild->ikl', J, J)
    nG = np.linalg.norm(G, axis=(1, 2))
    n, K = G.shape[0], G.shape[1]
    Kc = np.einsum('ikl,jkl->ij', G, G) / np.outer(nG, nG)
    np.fill_diagonal(Kc, 1.0)
    return Kc


def _fit_agglo(DG, k):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    dist = squareform(DG, checks=False)
    Z = linkage(dist, method='average')
    labels = fcluster(Z, t=k, criterion='maxclust')
    return Z, labels


def silhouette_avg(DG, labels):
    n = DG.shape[0]
    s = np.zeros(n)
    uniq = np.unique(labels)
    for i in range(n):
        own = labels[i]
        same = [j for j in range(n) if labels[j] == own and j != i]
        if not same:
            s[i] = 0.0
            continue
        a = DG[i, same].mean()
        b = min(DG[i, [j for j in range(n) if labels[j] == c]].mean()
                for c in uniq if c != own)
        s[i] = (b - a) / (max(a, b) + 1e-12)
    return float(s.mean())


def _W_k(DG, labels, k):
    W = 0.0
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) < 2:
            continue
        W += 0.5 / len(idx) * (DG[np.ix_(idx, idx)] ** 2).sum()
    return W


def clustering_diagnostics(DG, J, rng, k_range=range(2, 9), B_gap=B_GAP, B_hh=B_HH):
    from scipy.cluster.hierarchy import cophenet
    from scipy.spatial.distance import squareform
    n = DG.shape[0]
    sil = {}
    Z_all = {}
    for k in k_range:
        Z, labels = _fit_agglo(DG, k)
        Z_all[k] = Z
        sil[k] = silhouette_avg(DG, labels)
    # gap statistic: eigenvalue-preserving nulls
    gap, s_k = {}, {}
    logW0 = {}
    for k in k_range:
        _, labels = _fit_agglo(DG, k)
        logW0[k] = np.log(_W_k(DG, labels, k) + 1e-12)
    # v27.5r: null distances generated once, shared across k (same reference
    # distribution for all k — standard gap practice; per-k fresh nulls would
    # cost 7x more).
    En_np = np.einsum('ikd,ikd->i', J, J)
    Jt = torch.from_numpy(J)
    En_t = torch.tensor(En_np, dtype=torch.float32)
    # B_gap eigenvalue-preserving nulls: one random orthogonal R per null
    torch.manual_seed(SEED + 1000)
    R_all = torch.linalg.qr(torch.randn(B_gap * n, J.shape[1], J.shape[1]))[0]
    Dnulls = [1.0 - _rotated_cosine_t(R_all[b * n:(b + 1) * n], Jt, En_t)[0]
              for b in range(B_gap)]
    for k in k_range:
        Wn = []
        for Dn in Dnulls:
            _, labels = _fit_agglo(Dn, k)
            Wn.append(np.log(_W_k(Dn, labels, k) + 1e-12))
        Wn = np.array(Wn)
        gap[k] = float(Wn.mean() - logW0[k])
        s_k[k] = float(Wn.std(ddof=1) * np.sqrt(1 + 1 / B_gap))
    ksel = None
    ks = list(k_range)
    for pos, k in enumerate(ks[:-1]):
        if gap[k] >= gap[ks[pos + 1]] - s_k[ks[pos + 1]]:
            ksel = k
            break
    best_k = max(ks, key=lambda k: sil[k])
    # rho_NN for the best-silhouette clustering
    _, labels = _fit_agglo(DG, best_k)
    KG = 1.0 - DG
    uniq = sorted(set(labels.tolist()))
    num, den = 0.0, 0.0
    for c in uniq:
        ic = np.where(labels == c)[0]
        nc = len(ic)
        if nc < 2:
            continue
        Kw = KG[np.ix_(ic, ic)]
        iu = np.triu_indices_from(Kw, 1)
        K_within = Kw[iu].mean()
        best_neigh = -2.0
        for d in uniq:
            if d == c:
                continue
            idd = np.where(labels == d)[0]
            K_cd = KG[np.ix_(ic, idd)].mean()
            best_neigh = max(best_neigh, K_cd)
        rho_c = K_within / (best_neigh + 1e-12)
        num += nc * rho_c
        den += nc
    rho_NN = num / den
    # cophenetic (Spearman per spec 9.7)
    Z = Z_all[best_k]
    out = cophenet(Z, squareform(DG, checks=False))
    Dc = out[0] if np.ndim(out[0]) else out[1]  # 1D condensed cophenetic
    iu = np.triu_indices_from(DG, 1)
    coph = spearman(DG[iu], Dc)
    # half-split silhouette 16+16 x 50 (S_max over k=2..8 per half)
    def _smax(DG_sub, kmax=8):
        vals = []
        for k in range(2, min(kmax, DG_sub.shape[0]) + 1):
            _, lb = _fit_agglo(DG_sub, k)
            if len(np.unique(lb)) < 2:
                continue
            vals.append(silhouette_avg(DG_sub, lb))
        return max(vals) if vals else 0.0

    sA, sB = [], []
    for _ in range(B_hh):
        perm = rng.permutation(n)
        iA, iB = perm[:n // 2], perm[n // 2:]
        sA.append(_smax(DG[np.ix_(iA, iA)]))
        sB.append(_smax(DG[np.ix_(iB, iB)]))
    sA, sB = np.array(sA), np.array(sB)
    hh_corr = float(np.corrcoef(sA, sB)[0, 1]) if sA.std() > 0 and sB.std() > 0 else 0.0
    return dict(
        silhouette_curve={str(k): sil[k] for k in ks},
        silhouette_max=float(sil[best_k]),
        silhouette_argmax_k=int(best_k),
        gap_curve={str(k): gap[k] for k in ks},
        gap_selected_k=int(ksel) if ksel is not None else None,
        rho_NN=float(rho_NN),
        cophenetic_corr=float(coph),
        halfsplit_silhouette_mean_A=float(sA.mean()),
        halfsplit_silhouette_mean_B=float(sB.mean()),
        halfsplit_silhouette_corr=hh_corr,
    )


def uniform_mode_block(J, C, E):
    """Block D: g_uni stats, J_rest = (I-Q)J, delta_C_uni."""
    K = J.shape[1]
    u = np.ones(K) / np.sqrt(K)
    c = np.einsum('k,ikd->id', u, J)              # (n,D) coefficient
    QJ = c[:, None, :] * u[None, :, None]          # (n,K,D) = Q J
    Jrest = J - QJ
    g = np.einsum('ikd,ikd->i', QJ, QJ) / (E + 1e-30)
    Erest = np.einsum('ikd,ikd->i', Jrest, Jrest)
    Crest = np.einsum('ikd,jkd->ij', Jrest, Jrest) / np.sqrt(np.outer(Erest, Erest) + 1e-30)
    iu = np.triu_indices_from(C, 1)
    dC = C[iu] - Crest[iu]
    return dict(
        mean_g_uni=float(g.mean()),
        max_g_uni=float(g.max()),
        frac_gt_001=float((g > 0.01).mean()),
        mean_C_drop=float(dC.mean()),
    )


def run_layer(l, J, P, U, h, labels, rng, D):
    print(f'--- layer {l} ---', flush=True)
    E_all = np.einsum('ikd,ikd->i', J, J)
    valid = E_all > 1e-12
    excl = int((~valid).sum())
    if excl:
        J, P, h = J[valid], P[valid], h[valid]
        labels = [x for x, v in zip(labels, valid.tolist()) if v]
    t0 = time.time()
    C, E, Dp = gramian_cosine(J, P, U, h)
    print(f'  gramian C: {time.time()-t0:.1f}s (excluded E<1e-12: {excl})', flush=True)

    t0 = time.time()
    T0, p_perm = perm_test_domain(C, labels, rng)
    print(f'  perm test: T0={T0:.4f} p={p_perm:.4f} ({time.time()-t0:.1f}s)', flush=True)
    t0 = time.time()
    p_rot = spectral_null(J, labels, rng)
    print(f'  spectral null: p={p_rot:.4f} ({time.time()-t0:.1f}s)', flush=True)
    t0 = time.time()
    d_half, s_half = halfsplit_reliability(C, labels, 8)
    print(f'  halfsplit: d={d_half:.4f} s={s_half:.4f} ({time.time()-t0:.1f}s)', flush=True)

    m = mask_from_labels(labels, len(labels))
    iu = np.triu_indices_from(C, 1)
    corr = {'rho_C_' + k: pair_spearman(C, Dp[k]) for k in
            ['dh', 'dH', 'dsigma_z', 'dlogE', 'dg_uni']}
    T_ent, p_ent, cw, cc = entropy_quartile_test(C, -np.einsum(
        'ik,ik->i', P / P.sum(1, keepdims=True),
        np.log(P / P.sum(1, keepdims=True) + 1e-12)), rng, B=1000)

    t0 = time.time()
    KG = token_gramian_cosine(J)
    DG = 1.0 - KG
    clus = clustering_diagnostics(DG, J, rng)
    print(f'  clustering: Smax={clus["silhouette_max"]:.3f} k={clus["silhouette_argmax_k"]} '
          f'rhoNN={clus["rho_NN"]:.3f} ({time.time()-t0:.1f}s)', flush=True)

    unif = uniform_mode_block(J, C, E)
    return dict(
        domain=dict(C_within=mean_offdiag(C, m), C_cross=mean_offdiag(C, ~m),
                    T_domain=T0, p_perm=p_perm, p_rot=p_rot),
        halfsplit=dict(mean_abs_diff=d_half, mean_stability=s_half),
        correlations=corr,
        entropy_quartiles=dict(C_within=cw, C_cross=cc, T_ent=T_ent, p_perm=p_ent),
        uniform_mode=unif,
        clustering=clus,
        n_excluded_energy=excl,
        other_axes=dict(text_length='constant (S=64, pos=0 slice)',
                        position='constant (pos=0)',
                        dataset_source='tested as domain axis (block A)',
                        mean_top_p_size='not_cached (logits not in jcache)'),
    )


def selfcheck():
    # identity: diagonal ~ 1, values in [-1,1]
    Jt = torch.from_numpy(np.random.default_rng(0).standard_normal((8, 16, 32)).astype(np.float32))
    En_t = torch.tensor(np.einsum('ikd,ikd->i', Jt.numpy(), Jt.numpy()), dtype=torch.float32)
    import torch as _t
    Rn = _t.linalg.qr(_t.randn(1 * 8, 16, 16))[0]
    Cn = _rotated_cosine_t(Rn, Jt, En_t)
    assert abs(Cn[0, 0, 0] - 1) < 5e-3, f'diag not 1: {Cn[0, 0, 0]}'
    assert Cn.min() >= -1.001 and Cn.max() <= 1.001
    # T0=0-ish noise -> p >= 1/(B+1)
    rng = np.random.default_rng(1)
    J = np.random.default_rng(2).standard_normal((32, 16, 32))
    labels = ['a'] * 8 + ['b'] * 8 + ['c'] * 8 + ['d'] * 8
    p = spectral_null(J, labels, rng, B=50)
    assert p >= 1 / 51
    # structured: strong shared per-domain base -> perm rejects, rot null does not
    rng = np.random.default_rng(3)
    base = rng.standard_normal((4, 16, 16)) * 5.0
    J = np.stack([base[i // 8] + rng.standard_normal((16, 16)) for i in range(32)])
    P = np.full((32, 16), 1.0 / 16)
    U = np.array([i % 17 for i in range(16)])
    h = rng.standard_normal((32, 16))
    C, E, Dp = gramian_cosine(J, P, U, h)
    T0, p_perm = perm_test_domain(C, labels, np.random.default_rng(4), B=200)
    p_rot = spectral_null(J, labels, np.random.default_rng(5), B=50)
    print(f'selfcheck: T0={T0:.3f} p_perm={p_perm:.4f} p_rot={p_rot:.4f}')
    assert p_perm < 0.05, f'perm should reject structured data: p={p_perm}'
    # shared per-domain base: orientation alignment is a real signal the
    # eigenvalue-preserving null cannot reproduce -> null rejects
    assert p_rot < 0.05, f'rot null should reject oriented structure: p={p_rot}'
    print('selfcheck OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='results/exp27_noise_floor.json')
    ap.add_argument('--jcache', default='results/exp27_jcache.pt')
    ap.add_argument('--out', default='results/exp27_5_postprocess.json')
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        return

    exp = json.load(open(a.json))
    layers = exp['params']['layers']
    domains = exp['params']['domains']
    n_samples = exp['params']['n_samples']  # per domain
    print(f'exp27: {a.json}  jcache: {a.jcache}', flush=True)

    labels = [d for d in domains for _ in range(n_samples)]
    rng = np.random.default_rng(SEED)

    per_layer = {}
    ck = torch.load(a.jcache, map_location='cpu', weights_only=False)
    U = np.asarray(ck['U128'], dtype=np.int64)
    for l in LAYERS:
        J = np.stack([ck['JR'][(dom, i, l, 64, 0)][:K_MAX].numpy()
                      for dom in domains for i in range(n_samples)], 0)
        P = np.stack([ck['P128'][(dom, i, l, 64, 0)][:K_MAX].numpy()
                      for dom in domains for i in range(n_samples)], 0)
        h = np.stack([ck['H'][(dom, i, l, 64, 0)].numpy()
                      for dom in domains for i in range(n_samples)], 0)
        per_layer[str(l)] = run_layer(l, J, P, U, h, labels, rng, int(ck['D']))

    out = dict(
        params=dict(layers=LAYERS, domains=domains, S=64, K=K_MAX, position=0,
                    n_samples=len(labels), n_permutations=B_PERM,
                    n_rot_null=B_ROT, n_gap_null=B_GAP,
                    n_halfsplit_cluster=B_HH, seed=SEED),
        per_layer=per_layer,
    )
    with open(a.out, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'saved: {a.out}', flush=True)

    print()
    print('| слой | C_within | C_cross | p_perm | p_rot | half_stab | rho(C,Delh) | rho(C,DelH) | rho(C,DellogE) | S_max | gap_k | rho_NN | coph |')
    print('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for l in LAYERS:
        r = per_layer[str(l)]
        print(f"| {l} | {r['domain']['C_within']:.3f} | {r['domain']['C_cross']:.3f} | "
              f"{r['domain']['p_perm']:.3f} | {r['domain']['p_rot']:.3f} | "
              f"{r['halfsplit']['mean_stability']:.3f} | "
              f"{r['correlations']['rho_C_dh']:.3f} | {r['correlations']['rho_C_dH']:.3f} | "
              f"{r['correlations']['rho_C_dlogE']:.3f} | "
              f"{r['clustering']['silhouette_max']:.3f} | {r['clustering']['gap_selected_k']} | "
              f"{r['clustering']['rho_NN']:.3f} | {r['clustering']['cophenetic_corr']:.3f} |")
    for l in LAYERS:
        r = per_layer[str(l)]
        dom, cl, hs = r['domain'], r['clustering'], r['halfsplit']
        verdict = ('domain axis REAL (p_perm<0.05, p_rot<0.05)'
                   if dom['p_perm'] < 0.05 and dom['p_rot'] < 0.05 else
                   'domain axis NOT supported' if dom['p_perm'] >= 0.05 else
                   'spectral-artifact risk (p_rot>=0.05)')
        print(f"L{l}: {verdict}; T_domain={dom['T_domain']:+.3f} "
              f"(p_perm={dom['p_perm']:.3f}, p_rot={dom['p_rot']:.3f}); "
              f"halfstab={hs['mean_stability']:.3f}; "
              f"S_max={cl['silhouette_max']:.3f}@k{cl['silhouette_argmax_k']}, "
              f"gap_k={cl['gap_selected_k']}, rho_NN={cl['rho_NN']:.3f}, "
              f"coph={cl['cophenetic_corr']:.3f}")



if __name__ == '__main__':
    main()
