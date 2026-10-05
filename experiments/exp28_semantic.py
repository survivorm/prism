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

v3 per docs/Exp28_errors.md review round 2 (2026-10-04):
  2.1 perm_test EXACT positives via argsort (binomial masks were not exact)
  2.2/2.6 worker computes d(r) r=8/16/32 + kappa profile and sends ~1KB;
      V_all (500MB) and the Js stacks (2GB) leave the parent, J is read in
      the worker from the COW-forked ckpt
  2.4 boundary alignment: d(t) is the V_t -> V_{t+1} transition, absz is
      reported for BOTH z(t) (pre-jump) and z(t+1) (post-jump)
  2.5 os.sched_getaffinity saved to params for reproducibility

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
  6. Axis-2 attribution on U128 (top token weights, p_t ratios at boundaries)
     + real next-token types from the rebuilt pool-ids cache (token-type TZ bullet).
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
IDS_PT = os.path.join(CKPT_DIR, 'exp27b_pool_ids_L23.pt')  # rebuilt input_ids (re-streamed pool == ckpt pool_fp)
R_MAIN = 16
B_PERM = 2000
B_TOP = 8
LAYER = 23
SEED = 20260928
N_THREADS = 1  # one thread per process; 4 forked processes pinned OUTSIDE via taskset (user rule)
T_TOTAL = 128  # exp27b dense cache: t = 0..127

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
    # numpy mask: np.nonzero on a torch tensor is broken in numpy>=2.5 (returns 1 idx)
    d = d.numpy() if torch.is_tensor(d) else np.asarray(d, float)
    dm = float(np.nanmedian(d))
    mad = float(np.nanmedian(np.abs(d - dm)))
    return (d > dm + max(2.0 * mad, flat_t))

def perm_test(x, labels, B=B_PERM, seed=SEED):
    """Mean x[labels] - mean x[~labels]; one-sided p via B random masks with the
    EXACT same number of positives as the real labels (argsort selection)."""
    x = np.asarray(x, float)
    l = np.asarray(labels, bool)
    stat = float(x[l].mean() - x[~l].mean()) if l.sum() else 0.0
    M = x.size
    k = int(l.sum())
    if k == 0 or k == M:
        return stat, 1.0
    g = np.random.default_rng(seed)
    idx = np.argsort(g.random((B, M)), axis=1)[:, :k]
    lab = np.zeros((B, M), bool)
    lab[np.arange(B)[:, None], idx] = True
    sel = lab @ x
    stats = sel / k - (x.sum() - sel) / (M - k)
    return stat, float((stats >= stat).sum() + 1) / (B + 1)


def kappa_from_V(V, r):
    """kappa(t) = ||procr(V_t,V_{t+1}) procr(V_{t+1},V_{t+2}) procr(V_t,V_{t+2})^-1 - I||_F."""
    def _pol(M):
        U, _, Wt = torch.linalg.svd(M, full_matrices=False)
        return U @ Wt
    Vr = V[:, :, :r]
    M01 = Vr[:-2].transpose(-1, -2) @ Vr[1:-1]
    M12 = Vr[1:-1].transpose(-1, -2) @ Vr[2:]
    M02 = Vr[:-2].transpose(-1, -2) @ Vr[2:]
    Hhol = _pol(M01) @ _pol(M12) @ _pol(M02).transpose(-1, -2)
    return (Hhol - torch.eye(r)).flatten(1).norm(dim=1)


def _worker(ck, keys, idxs, conn):
    """SVD + d(r) profiles + kappa in the worker; send only profiles (~1KB)."""
    torch.set_num_threads(1)
    for i, (d, s) in zip(idxs, keys):
        J = torch.stack([ck['JR'][(d, s, t)] for t in range(T_TOTAL)])
        _, _, Vh = torch.linalg.svd(J, full_matrices=False)
        V = Vh[:, :32].transpose(-1, -2)                      # [T, D, 32]; Vh is [T,K,D] batched SVD
        drs = tuple(d_profile_from_V(V, r).numpy() for r in (8, 16, 32))
        krow = kappa_from_V(V, 16).numpy()                    # [T-2]
        conn.send((i, drs, krow))
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

    # --- SVD + d(r) + kappa per document in 4 forked workers; profiles only back ---
    import multiprocessing as mp
    conns = [mp.Pipe() for _ in range(4)]
    procs = []
    for w in range(4):
        c = mp.Process(target=_worker, args=(ck, doms_order[w::4], list(range(w, N, 4)), conns[w][1]))
        c.daemon = True
        c.start()
        procs.append(c)
    profs = [None] * N
    krows = [None] * N
    for w in range(4):
        for _ in range(len(range(w, N, 4))):
            i, drs, krow = conns[w][0].recv()
            profs[i], krows[i] = drs, krow
    for c in procs:
        c.join()
    msk = {r: [boundary_mask(p[i]) for p in profs] for i, r in ((0, 8), (1, 16), (2, 32))}

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
    absz = Zs[:, :T1, :B_TOP].numpy().reshape(N * T1, B_TOP)      # pre-jump: z(t)
    absz_post = Zs[:, 1:, :B_TOP].numpy().reshape(N * T1, B_TOP)  # post-jump: z(t+1)
    axes_out = {}
    adj = np.zeros(N * T1, bool)
    for i in range(N):
        for t in np.nonzero(msk[16][i])[0]:
            for u in range(max(0, t - 2), min(T1, t + 3)):
                adj[i * T1 + u] = True
    for a in range(B_TOP):
        stat, pval = perm_test(absz[:, a], lab)
        stat_p, pval_p = perm_test(absz_post[:, a], lab)
        stat_a, pval_a = perm_test(absz[:, a], adj)
        axes_out[f'axis{a}'] = {
            'mean_absz_boundary_vs_rest': round(stat, 4), 'p_perm': round(pval, 5),
            'mean_absz_post': round(stat_p, 4), 'p_perm_post': round(pval_p, 5),
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
    Dmat = np.stack([p[1] for p in profs])                          # [N,T1], r=16
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

    # --- test 5: pooled kappa profile L23 (r=16) - check peaks at t ~ 31/63/95 ---
    kprof = np.nanmean(np.stack(krows), axis=0)
    peaks = sorted(range(1, len(kprof) - 1), key=lambda t: -kprof[t])[:5]

    # --- test 6: axis-2 token attribution + real next-token types (TZ bullet) ---
    tok = torch.load(IDS_PT, map_location='cpu', weights_only=False)
    Ua = np.asarray(ck['U128'])
    assert tok['meta']['pool_fp'] == {d: ck['pool_fp'][d] for d in ck['domains']}, 'pool-ids fp mismatch'
    idsA = np.array([list(ids) for d in ck['domains'] for ids in tok['ids'][d]])   # [N,T]
    V2 = Axes[:, 2].numpy()
    w2 = V2 / np.abs(V2).sum()
    top8 = np.argsort(-np.abs(w2))[:8]
    energy = float((w2[top8] ** 2).sum())
    top8_tok = [[int(Ua[j]), round(float(w2[j]), 4)] for j in top8]
    STRUCT = {220: 'space', 198: 'nl', 11: 'comma', 271: 'nnl'}
    nU = np.isin(idsA[:, 1:], Ua)                                             # [N,T1], next token in U128
    nU_idx = {int(t): i for i, t in enumerate(Ua)}
    is_struct = np.isin(idsA[:, 1:], list(STRUCT))                            # [N,T1]
    pos = (np.arange(T1)[None, :] + T1 * np.arange(N)[:, None]).reshape(-1)  # flat [N*T1]
    lab_f = lab.reshape(-1); nU = nU.reshape(-1); is_struct = is_struct.reshape(-1)
    Pn = P.numpy()
    pbv = Pn[pos[lab_f] // T1, pos[lab_f] % T1, :].mean(0)
    prv = Pn[pos[~lab_f] // T1, pos[~lab_f] % T1, :].mean(0)
    corr_v2 = float(np.corrcoef(V2, pbv - prv)[0, 1])
    pstruct = {nm: {'boundary': float((idsA[:, 1:] == tid).reshape(-1)[pos[lab_f]].mean()),
                    'rest': float((idsA[:, 1:] == tid).reshape(-1)[pos[~lab_f]].mean())}
               for tid, nm in STRUCT.items()}
    pstruct_all = {'boundary': float(is_struct[pos[lab_f]].mean()), 'rest': float(is_struct[pos[~lab_f]].mean())}
    # p_t of the REAL next token, on the nU subset (next token in U128)
    Uinv = {int(t): j for j, t in enumerate(Ua)}
    sel = pos[nU]
    nxt = idsA[sel // T1, sel % T1 + 1]
    pnext = Pn[sel // T1, sel % T1, [Uinv[t] for t in nxt.tolist()]]
    pnext_mean = {'boundary': float(pnext[lab_f[nU]].mean()), 'rest': float(pnext[~lab_f[nU]].mean())}
    bi, ri = np.flatnonzero(lab_f), np.flatnonzero(~lab_f)
    ratio_struct = {}
    for tid, nm in STRUCT.items():
        j = int(np.where(Ua == tid)[0][0])
        ratio_struct[nm] = {'boundary': float(Pn[bi // T1, bi % T1, j].mean()), 'rest': float(Pn[ri // T1, ri % T1, j].mean())}
    att = {
        'axis_energy_top8': round(energy, 4), 'top8_token_weight': top8_tok,
        'struct_next_type': pstruct_all, 'struct_next_type_by_token': pstruct,
        'p_real_next_token': pnext_mean, 'n_next_in_U128': int(nU.sum()), 'frac_next_in_U128': round(float(nU.mean()), 3),
        'p_struct_at_boundary': ratio_struct, 'corr_V2_pb_minus_pr': round(corr_v2, 3),
    }

    out = {
        'params': {
            'layer': LAYER, 'r_main': R_MAIN, 'b_perm': B_PERM, 'b_top': B_TOP,
            'seed': SEED, 'n_docs': N, 'T': T,
            'boundary_def': 'd(t) > median + max(2*MAD, 1e-6), per doc, r=16 (== exp28 JSON, self-checked)',
            'p_axes': 'top-8 right singular vectors of pooled (P_t - mean); per-doc z-score',
            'token_types': f'real next tokens from rebuilt pool-ids cache ({IDS_PT}); fp matched vs ckpt pool_fp, all 4 domains',
            'perm_test': 'one-sided, B random masks with EXACT positive count (argsort)',
            'alignment': 'lab[t] = transition V_t -> V_{t+1}; pre = z(t), post = z(t+1)',
            'affinity': sorted(os.sched_getaffinity(0)), 'threads_per_proc': N_THREADS,
            'inputs': f'{CKPT_DIR}/exp27b_jcache_L{LAYER}.pt, {E28_JSON}, {IDS_PT}',
        },
        'p_axes': {'axis_thirds_mean': thirds, 'boundary_vs_rest': axes_out, 'per_domain': dom_out},
        'dynamics': dyn,
        'domain_blocks': {'ari_k4': ari},
        'r_sensitivity': jac,
        'kappa_profile_l23': {'pooled': kprof.tolist(), 'top5_peak_positions': peaks},
        'axis2_token_attribution': att,
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
    print('axis2 attribution:', att)


if __name__ == '__main__':
    main()
