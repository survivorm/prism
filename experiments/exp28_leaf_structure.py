#!/usr/bin/env python3
"""Exp 28 — leaf structure and local curvature of the J field (docs/Exp28.md).

Post-processing of the exp27b dense J-caches. No GPU, no model.

exp27b ckpt exp27b_jcache_L{l}.pt: gauge-free J[(dom, si, t)] in R^{K x D}
(K=128, D=1024) at EVERY token position t = 0..127 of the 128-token prefix
(t = absolute token index; J at t sees tokens 0..t, causal).
n_samples=8 per domain, 4 domains, 6 layers.

Metrics (per document (dom, si) and layer l):
  1. Delta_SVD(t), t=0..126: max principal angle between the top-r right
     singular subspaces of J_t and J_{t+1} (V = right singular vectors,
     D-side; the field lives in hidden space — TZ: V, not U).
  2. kappa(t), t=0..125: holonomy of the triple (t, t+1, t+2):
     R_ab = argmin_{R in O(r)} ||V_b - V_a R||_F (polar factor, O(r) — no
     det fix, so a pure sign-gauge gives H = I),
     H = R_01 R_12 R_02^{-1}, kappa = ||H - I||_F.
  3. Boundaries: Delta_SVD(t) > median(profile) + 2*MAD (relative spike, TZ).
     Positions saved per document (review item: list, not only the count).
  4. Random baselines: B_RAND random r-subspaces of R^D per document
     (QR of a Gaussian). NOT a rotation of the data subspace — principal
     angles and holonomy are basis-invariant, so V @ Q(r) is a no-op
     baseline (the old-version bug, see docs/Exp28_errors.md item 1).
  5. r sweep {8,16,32} on profile means (stability); main r=16.
     All r share ONE batched SVD per document (Vh prefixes) — no re-SVD.
  Degenerate positions (sigma_r < 1e-9 * ||J||_F): NaN in the profile,
  excluded from summaries, count saved (TZ item 3).
  Token types are not in the cache -> TZ step-3 token-type bullet skipped.
  Relative position: profiles summarized per document third on the common
  absolute-position grid (S//3, 2S//3) — same positions for d1 and k2.
  flat_frac = frac(t: d(t)<T and k(t)<T); kappa_flat_frac = frac(k<T)
  (review item 1: separate metric, since TZ defines flatness via kappa).
  Single-model cache: the multi-model TZ bullet is not applicable.

Speed (09-30, review «Оптимизации»): one batched SVD [T,K,D] per document
(4096 individual SVD -> 1), r-sweep = Vh prefixes (3x win), sign-gauging
dropped (metrics are sign-invariant), pinned to P-cores, 4 BLAS threads.

Output: results/exp28_leaf_structure.json (saved after each layer).
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

LAYERS = [6, 12, 18, 19, 22, 23]
R_MAIN = 16
R_SWEEP = [8, 16, 32]
B_RAND = 200
SEED = 20260928
CKPT_DIR = '/data/prism_kernels/exp27b/results'
DEGEN_REL = 1e-9
FLAT_T = 1e-3  # ТЗ 10.3: плоское поле — kappa < 1e-3
N_THREADS = 4  # max 4 threads, on P-cores (user rule 09-30)


def _pin_p_cores(n=N_THREADS):
    """Pin to P-cores (hyperthread pairs) on hybrid CPUs, cap BLAS threads."""
    try:
        cores = {}
        for i in range(os.cpu_count() or 1):
            with open(f'/sys/devices/system/cpu/cpu{i}/topology/core_id') as fh:
                cores.setdefault(fh.read().strip(), []).append(i)
        p = [c for cs in cores.values() if len(cs) >= 2 for c in cs]
        p = (p or list(range(os.cpu_count())))[:n]
        os.sched_setaffinity(0, set(p))
    except Exception:
        pass
    torch.set_num_threads(n)


def right_subspaces(J_stack, r_list):
    """Top-r right singular subspaces for ALL positions in ONE batched SVD.
    J_stack [T,K,D] -> {r: (V [T,D,r], sig_r [T])}.
    Sign-gauging skipped: principal angles and ||H-I||_F are invariant to
    per-column signs (SVD of Va^T Vb unchanged under V -> V diag(+-1))."""
    _, S, Vh = torch.linalg.svd(J_stack, full_matrices=False)  # Vh [T,K,D]
    return {r: (Vh[:, :r].transpose(-1, -2).contiguous(),
                S[:, r - 1] if r <= S.shape[1] else S[:, -1]) for r in r_list}


def principal_angles_max(Va, Vb):
    """Max principal angle between two r-dim subspaces (single pair)."""
    s = torch.linalg.svdvals(Va.t() @ Vb).clamp(max=1.0 - 1e-12)
    return float(torch.acos(s).max())


def haar_subspace(D, r, device, generator=None):
    """Uniform random r-subspace of R^D (QR of a Gaussian)."""
    A = torch.randn(D, r, device=device, generator=generator)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :r]


def procrustes(Va, Vb):
    """Polar factor of Va^T Vb: argmin_{R in O(r)} ||Vb - Va R||_F.
    No det-fix: with sign-gauged bases M = Va^T Vb is a reflection (det -1)
    and the SO(r) projection flips a column, breaking H = I for identical
    subspaces. Metrics use O(r), so the polar factor in O(r) is correct."""
    M = Va.t() @ Vb
    U, _, Wt = torch.linalg.svd(M, full_matrices=False)
    return U @ Wt


def procrustes_batch(M):
    """Polar factor per batch element. M [T,r,r] -> [T,r,r]."""
    U, _, Wt = torch.linalg.svd(M, full_matrices=False)
    return U @ Wt


def load_layer(ckpt_dir, l):
    p = os.path.join(ckpt_dir, f'exp27b_jcache_L{l}.pt')
    ck = torch.load(p, map_location='cpu', weights_only=False)
    return ck


def document_profile(J_stack, r_list, r_main=R_MAIN):
    """Per-position profiles for ONE (dom, si, layer) from a [T,K,D] stack.
    One batched SVD; all r in r_list are prefixes of the same Vh.
    Returns ({r: (d1 [T-1], k2 [T-2])}, d1_main, k2_main, n_degen).
    NaN at positions touching a degenerate token (sigma_r < DEGEN_REL*||J||_F,
    evaluated at r_main)."""
    T = J_stack.shape[0]
    allV = right_subspaces(J_stack, r_list)
    V, sig = allV[r_main]
    jnorm = J_stack.norm(dim=(1, 2))
    deg = sig < DEGEN_REL * jnorm.clamp_min(1e-30)
    n_degen = int(deg.sum())
    out = {}
    for r in r_list:
        Vr = allV[r][0]
        M1 = Vr[:-1].transpose(-1, -2) @ Vr[1:]          # [T-1, r, r]
        d1 = torch.acos(torch.linalg.svdvals(M1).clamp(max=1.0 - 1e-12)).max(-1).values
        M01 = Vr[:-2].transpose(-1, -2) @ Vr[1:-1]       # [T-2, r, r]
        M12 = Vr[1:-1].transpose(-1, -2) @ Vr[2:]
        M02 = Vr[:-2].transpose(-1, -2) @ Vr[2:]
        H = procrustes_batch(M01) @ procrustes_batch(M12) @ procrustes_batch(M02).transpose(-1, -2)
        k2 = (H - torch.eye(r, dtype=H.dtype)).flatten(1).norm(dim=1)
        out[r] = (d1, k2)
    d1m, k2m = out[r_main]
    dm = d1m.masked_fill(deg[:-1] | deg[1:], float('nan'))
    km = k2m.masked_fill(deg[:-2] | deg[1:-1] | deg[2:], float('nan'))
    return out, dm, km, n_degen


def rand_baseline(D, r, B, seed):
    """B random subspaces -> baseline Delta_SVD (consecutive) and kappa (triples).
    kappa baseline uses independent random subspaces on the 3 legs
    (a random triangle in the space of r-subspaces is not flat)."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    R = [haar_subspace(D, r, 'cpu', generator=g) for _ in range(B)]
    d_r = [principal_angles_max(R[i], R[i + 1]) for i in range(len(R) - 1)]
    k_r = []
    for i in range(len(R) - 2):
        R01 = procrustes(R[i], R[i + 1])
        R12 = procrustes(R[i + 1], R[i + 2])
        R02 = procrustes(R[i], R[i + 2])
        H = R01 @ R12 @ R02.t()
        k_r.append(float((H - torch.eye(r)).norm()))
    return float(np.mean(d_r)), float(np.mean(k_r))


def _thirds(arr, T):
    """Mean per document third on the common absolute-position grid
    (0..T//3, T//3..2T//3, 2T//3..T) — same positions for d1 and k2
    (review item 4). NaN-aware."""
    a = np.asarray(arr, dtype=float)
    b = a.shape[-1]
    return [float(np.nanmean(a[:, i * T // 3:min((i + 1) * T // 3, b)], axis=0)) for i in range(3)]


def summarize_profile(d1, k2):
    """Profile stats: NaN-aware means + third-wise means + flat fractions.
    flat_frac = frac(d<T and k<T); kappa_flat_frac = frac(k<T) (review item 1).
    boundary_positions = absolute t where d(t) > median + max(2*MAD, 1e-6)."""
    d = np.asarray(d1, dtype=float)
    k = np.asarray(k2, dtype=float)
    dn = d[~np.isnan(d)]
    kn = k[~np.isnan(k)]
    dm = float(np.median(dn))
    mad = float(np.median(np.abs(dn - dm)))
    thresh = dm + max(2 * mad, 1e-6)
    mask = ~(np.isnan(d[:len(k)]) | np.isnan(k))
    return {
        'delta_mean': float(dn.mean()), 'delta_std': float(dn.std()), 'delta_p95': float(np.percentile(dn, 95)),
        'kappa_mean': float(kn.mean()), 'kappa_med': float(np.median(kn)), 'kappa_p95': float(np.percentile(kn, 95)),
        'n_boundaries': int(np.sum(dn > thresh)),
        'boundary_positions': [int(i) for i in np.nonzero(~np.isnan(d) & (d > thresh))[0]],
        'flat_frac': float(np.mean((d[:len(k)][mask] < FLAT_T) & (k[mask] < FLAT_T))) if mask.any() else float('nan'),
        'kappa_flat_frac': float(np.mean(k[mask] < FLAT_T)) if mask.any() else float('nan'),
        'n_degen_pairs': int(np.isnan(d).sum()),
    }


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _git_hash():
    import subprocess
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return 'unknown'


def _arr_to_json(a):
    return [round(float(x), 5) if not np.isnan(x) else None for x in a]


def main():
    ap = argparse.ArgumentParser(description='Exp 28 leaf structure + curvature, exp27b dense (CPU)')
    ap.add_argument('--ckpt-dir', default=CKPT_DIR)
    ap.add_argument('--layers', default='6,12,18,19,22,23')
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--b-rand', type=int, default=B_RAND)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    _pin_p_cores()
    layers = [int(x) for x in a.layers.split(',')]
    os.makedirs(a.results_dir, exist_ok=True)
    out_path = os.path.join(a.results_dir, 'exp28_leaf_structure.json')
    ck0 = load_layer(a.ckpt_dir, layers[0])
    D, S, n, domains = ck0['D'], ck0['S'], ck0['n_samples'], ck0['domains']
    out = {
        'params': {'ckpt_dir': a.ckpt_dir, 'layers': layers, 'D': D, 'S': S,
                   'r_main': R_MAIN, 'r_sweep': R_SWEEP, 'b_rand': a.b_rand,
                   'seed': a.seed, 'n_degen_rel': DEGEN_REL,
                   'threads': torch.get_num_threads(),
                   'token_types': 'not in cache — TZ step 3 (token types) skipped',
                   'baseline': 'B random r-subspaces of R^D per document (basis-invariant metrics — data-rotation is a no-op)'},
        'git': _git_hash(),
        'layers': {},
    }
    out['manifest'] = {l: {'sha256_16': sha_file(os.path.join(a.ckpt_dir, f'exp27b_jcache_L{l}.pt'))[:16],
                           'pool_fp': load_layer(a.ckpt_dir, l)['pool_fp']} for l in layers}
    t0 = time.time()
    for l in layers:
        ck = load_layer(a.ckpt_dir, l)
        assert ck['D'] == D and ck['S'] == S
        doc_rows = []
        n_degen = 0
        A1 = np.full((n * len(domains), S - 1), np.nan)
        A2 = np.full((n * len(domains), S - 2), np.nan)
        rs = {r: {'delta': [], 'kappa': []} for r in R_SWEEP}
        row = 0
        for dom in domains:
            for si in range(n):
                J_stack = torch.stack([ck['JR'][(dom, si, t)] for t in range(S)])
                allr, d1, k2, nd = document_profile(J_stack, R_SWEEP, R_MAIN)
                del J_stack
                n_degen += nd
                st = summarize_profile(d1, k2)
                doc_rows.append({'dom': dom, 'si': si, **st})
                A1[row], A2[row] = np.array(d1), np.array(k2)
                row += 1
                for r in R_SWEEP:
                    dr, kr = allr[r]
                    rs[r]['delta'].append(float(dr.nanmean()))
                    rs[r]['kappa'].append(float(kr.nanmean()))
        r_sweep = {str(r): {'delta_mean': float(np.mean(rs[r]['delta'])),
                            'kappa_mean': float(np.mean(rs[r]['kappa']))} for r in R_SWEEP}
        # pooled per-position profiles (mean over documents)
        pd1 = np.nanmean(A1, axis=0)
        pk2 = np.nanmean(A2, axis=0)
        # third-wise means over all documents, common absolute grid
        d_thirds = _thirds(A1, S - 1)
        k_thirds = _thirds(A2, S - 2)
        d_rand, k_rand = rand_baseline(D, R_MAIN, a.b_rand, a.seed + l)
        L = str(l)
        out['layers'][L] = {
            'delta_mean': float(np.nanmean(A1)),
            'delta_std_across_docs': float(np.nanstd([r['delta_mean'] for r in doc_rows])),
            'delta_p95': float(np.nanmean([r['delta_p95'] for r in doc_rows])),
            'kappa_mean': float(np.nanmean(A2)),
            'kappa_med': float(np.nanmean([r['kappa_med'] for r in doc_rows])),
            'kappa_p95': float(np.nanmean([r['kappa_p95'] for r in doc_rows])),
            'delta_thirds': d_thirds,
            'kappa_thirds': k_thirds,
            'n_boundaries_total': int(sum(r['n_boundaries'] for r in doc_rows)),
            'flat_frac': float(np.nanmean([r['flat_frac'] for r in doc_rows])),
            'kappa_flat_frac': float(np.nanmean([r['kappa_flat_frac'] for r in doc_rows])),
            'n_degen': n_degen,
            'baseline': {'delta_rand_mean': d_rand, 'kappa_rand_mean': k_rand, 'b_rand': a.b_rand},
            'r_sweep': r_sweep,
            'pooled_delta_profile': _arr_to_json(pd1),
            'pooled_kappa_profile': _arr_to_json(pk2),
            'docs': doc_rows,
        }
        with open(out_path, 'w') as f:
            json.dump(out, f)
        print(f'[OK] layer {l}: delta={out["layers"][L]["delta_mean"]:.4f} '
              f'(rand {d_rand:.4f}), kappa={out["layers"][L]["kappa_mean"]:.4f} '
              f'(rand {k_rand:.4f}) [{time.time()-t0:.0f}s]', flush=True)
    print(f'[OK] exp28 saved {out_path}')


def selfcheck():
    torch.manual_seed(0)
    D, K, r, T = 64, 16, 8, 5
    Q = torch.linalg.qr(torch.randn(D, r))[0][:, :r]

    def flat_field():
        # exactly rank-r, orthogonal left factor, right subspace = Q for all t
        Js = []
        for _ in range(T):
            M = torch.linalg.qr(torch.randn(K, r))[0][:, :r]
            Js.append(M @ Q.t())
        return torch.stack(Js)

    def curved_field():
        Js = []
        for _ in range(T):
            Qr = torch.linalg.qr(torch.randn(D, r))[0][:, :r]
            M = torch.linalg.qr(torch.randn(K, r))[0][:, :r]
            Js.append(M @ Qr.t())
        return torch.stack(Js)

    sdecay = torch.arange(r).float() * -0.25 + 2.0  # strict sigma decay (r-prefix stable)
    f = torch.stack([torch.linalg.qr(torch.randn(K, r))[0][:, :r] @ torch.diag(sdecay) @ Q.t() for _ in range(T)])
    f, c = f, curved_field()
    fl, _, _, _ = document_profile(f, [r], r)
    cu, _, _, _ = document_profile(c, [r], r)
    d_f = fl[r][0].max()
    k_f = fl[r][1][0]
    d_c = cu[r][0].max()
    k_c = cu[r][1][0]
    d_r, k_r = rand_baseline(D, r, 50, 0)
    d_r2, k_r2 = rand_baseline(D, r, 50, 0)
    assert d_r == d_r2 and k_r == k_r2, 'rand_baseline not reproducible'
    sf = summarize_profile(fl[r][0], fl[r][1])
    sc = summarize_profile(cu[r][0], cu[r][1])
    assert d_f < 0.05, f'flat delta too large: {d_f}'
    assert k_f < 1e-4, f'flat kappa too large: {k_f}'
    assert sf['flat_frac'] == 1.0, f'flat field flat_frac must be 1.0: {sf["flat_frac"]}'
    assert sf['kappa_flat_frac'] == 1.0
    assert d_c > 0.3, f'curved delta too small: {d_c}'
    assert k_c > 0.5, f'curved kappa too small: {k_c}'
    assert sc['flat_frac'] == 0.0, f'curved field flat_frac must be 0.0: {sc["flat_frac"]}'
    assert sc['kappa_flat_frac'] == 0.0
    assert d_r > 0.5, f'random delta should be O(1): {d_r}'
    # r-prefix: r=4 profile also flat (top-4 of the same Vh, one SVD)
    fl4, _, _, _ = document_profile(f, [4], 4)
    assert fl4[4][0].max() < 1e-3, f'r=4 profile not flat: {float(fl4[4][0].max())}'
    # sign invariance: flip a column of every J row -> metrics unchanged
    sign = torch.ones(T, K); sign[:, 3] = -1.0
    fs = f * sign.unsqueeze(-1)
    fls, _, _, _ = document_profile(fs, [r], r)
    assert float((fls[r][0] - fl[r][0]).abs().max()) < 1e-5, 'sign invariance broken'
    print(f"flat: d={d_f:.5f} k={k_f:.2e} flat_frac={sf['flat_frac']:.2f} | curved: d={d_c:.3f} k={k_c:.3f} flat_frac={sc['flat_frac']:.2f} | rand: d={d_r:.3f} k={k_r:.3f}")
    print('selfcheck OK')


if __name__ == '__main__':
    main()
