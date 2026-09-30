#!/usr/bin/env python3
"""Exp 28 — leaf structure and local curvature of the J field (docs/Exp28.md).

Post-processing of the exp27 v11 J-cache. No GPU, no model.

Objects from cache: gauge-free J[(dom, si, l, S, p)] in R^{K x D} (K=128, D=1024),
positions p are from the END of the S-token sequence (p=0 -> token S-1).
Available: S=32/64/256 -> p in {0,1,2}; S=64 -> {0,1,2,4,8,16}; S=128 -> {0,1,2,32,33}.

Metrics:
  1. Delta_SVD(t) on consecutive pairs: principal angles of the top-r right
     singular subspaces V_t, V_{t+1} (V = right singular vectors of J, D-side).
     Delta = max principal angle. Consecutive pairs in cache:
       end: (0,1) and (1,2) for S=32/64/128/256 (tokens S-1,S-2 / S-2,S-3),
       mid: (33,32) @ S=128 (tokens 94,95).
     Bonus decay: gap pairs (2,4),(4,8),(8,16) @ S=64 (dt=2,4,8 tokens).
  2. Curvature via holonomy on triples (0,1,2) for S=32/64/128/256:
     R_ab = argmin_{R in O(r)} ||V_b - V_a R||_F (Procrustes),
     H = R_01 R_12 R_02^{-1}, kappa = ||H - I||_F.
  3. Position correlation: end vs mid, decay by dt. Token types are NOT in the
     cache -> that TZ bullet is skipped (recorded in findings).
Random baselines: Haar-random rotation of V_b per pair/triple (B=200), same r.
r sweep: {8, 16, 32} (stability), main r=16.

Output: results/exp28_leaf_structure.json
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

LAYERS = [6, 12, 18, 19, 22, 23]
R_MAIN = 16
R_SWEEP = [8, 16, 32]
B_RAND = 200
SEED = 20260928
CKPT_DEFAULT = '/data/prism_kernels/exp27/results/exp27_jcache.pt'



def right_subspace(J, r):
    """Top-r right singular vectors V in R^{D x r} of J (K x D).
    Sign convention: first nonzero element of each column positive (TZ item 2)."""
    _, _, Vh = torch.linalg.svd(J, full_matrices=False)
    V = Vh[:r].t().contiguous()
    for j in range(r):
        col = V[:, j]
        nz = torch.nonzero(col.abs() > 1e-12)
        if len(nz) > 0 and col[nz[0]] < 0:
            V[:, j] = -col
    return V


def principal_angles_max(Va, Vb):
    """Max principal angle between column subspaces (rational, degenerate-safe)."""
    M = Va.t() @ Vb
    s = torch.linalg.svdvals(M)
    s = s.clamp(max=1.0 - 1e-12)
    return float(torch.acos(s).max())


def procrustes(Va, Vb):
    """R = argmin_{R in O(r)} ||Vb - Va R||_F, from SVD of Va^T Vb."""
    # TZ: argmin over O(r) (not SO) — polar factor of Va^T Vb. No det fix:
    # with the det flip, holonomy H != I for identical subspaces (gauge artifact).
    M = Va.t() @ Vb
    U, _, Wt = torch.linalg.svd(M, full_matrices=False)
    return U @ Wt


def haar_random(r, device):
    """Random rotation in O(r): Haar via Gaussian QR (TZ random baseline)."""
    A = torch.randn(r, r, device=device)
    Q, _ = torch.linalg.qr(A)
    d = torch.sign(torch.linalg.det(Q))
    Q[:, 0] *= d
    return Q



# pair spec: (S, p1, p2, dt, zone)  — absolute gap dt in tokens
PAIRS = [
    (64, 0, 1, 1, 'end'), (64, 1, 2, 2, 'end'),
    (128, 33, 32, 1, 'mid'),
    (256, 0, 1, 1, 'end'), (256, 1, 2, 2, 'end'),
    (32, 0, 1, 1, 'end'), (32, 1, 2, 2, 'end'),
    (64, 2, 4, 2, 'gap'), (64, 4, 8, 4, 'gap'), (64, 8, 16, 8, 'gap'),
]
# triple spec: (S, p0, p1, p2) consecutive, curvature at middle
TRIPLES = [(32, 0, 1, 2), (64, 0, 1, 2), (128, 0, 1, 2), (256, 0, 1, 2)]
TRIPLE_ZONES = {'(32,0)': 'end', '(64,0)': 'end', '(128,0)': 'end', '(256,0)': 'end'}


def sample_delta_svd(J1, J2, r, rng):
    """Data Delta_SVD + Haar baseline for one pair."""
    V1 = right_subspace(J1, r)
    V2 = right_subspace(J2, r)
    d_data = principal_angles_max(V1, V2)
    d_rand = principal_angles_max(V1, V2 @ haar_random(r, V1.device))
    return d_data, d_rand


def sample_kappa(J0, J1, J2, r, rng):
    """Data kappa + Haar baseline for one triple.
    Baseline: independent Haar rotations on the 1->2 and 0->2 legs,
    H = R01 R12 R02^{-1}, R02 = R01 R12 (consistent) -> H = R02 R02^{-1}...
    instead: rotate middle subspace randomly: V1' = V1 Q.
    H = R01' R12' R02'^{-1}, all three recomputed. This isolates the
    triangle closure: for flat field kappa~0, random middle -> kappa ~ O(1)."""
    V0 = right_subspace(J0, r)
    V1 = right_subspace(J1, r)
    V2 = right_subspace(J2, r)

    def hol(Va, Vb, Vc):
        R01 = procrustes(Va, Vb)
        R12 = procrustes(Vb, Vc)
        R02 = procrustes(Va, Vc)
        H = R01 @ R12 @ R02.t()
        return float((H - torch.eye(r)).norm())

    k_data = hol(V0, V1, V2)
    V1r = V1 @ haar_random(r, V1.device)
    k_rand = hol(V0, V1r, V2)
    return k_data, k_rand



def load_jcache(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return ck


def main():
    ap = argparse.ArgumentParser(description='Exp 28 leaf structure + curvature (CPU)')
    ap.add_argument('--ckpt', default=CKPT_DEFAULT)
    ap.add_argument('--layers', default='6,12,18,19,22,23')
    ap.add_argument('--results-dir', default='results')
    ap.add_argument('--b-rand', type=int, default=B_RAND)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    layers = [int(x) for x in a.layers.split(',')]
    rng = np.random.default_rng(a.seed)
    ck = load_jcache(a.ckpt)
    JR, D = ck['JR'], ck['D']
    n_tot = ck['n_samples'] * len(ck['domains'])
    os.makedirs(a.results_dir, exist_ok=True)
    out = {'params': {'ckpt': a.ckpt, 'layers': layers, 'D': D, 'r_main': R_MAIN,
                      'r_sweep': R_SWEEP, 'b_rand': a.b_rand, 'seed': a.seed,
                      'token_types': 'not in cache — TZ bullet 3 (token type) skipped'},
           'manifest': {'ckpt_sha256_16': sha_file(a.ckpt)[:16],
                        'git': _git_hash(),
                        'pool_fp': {d: (fp if isinstance(fp, str) else fp)
                                     for d, fp in (ck.get('pool_fp') or {}).items()}
                                   if isinstance(ck.get('pool_fp'), dict) else ck.get('pool_fp')},
           'pairs': {}, 'triples': {}, 'r_sweep': {}}
    for l in layers:
        # --- Delta_SVD per pair ---
        pdata = {}
        for (S, p1, p2, dt, zone) in PAIRS:
            ds = []
            for dom in ck['domains']:
                for si in range(ck['n_samples']):
                    d_d, d_r = sample_delta_svd(JR[(dom, si, l, S, p1)],
                                                 JR[(dom, si, l, S, p2)], R_MAIN, rng)
                    ds.append((d_d, d_r))
            ds = np.array(ds)
            pdata[(S, p1, p2, zone, dt)] = {
                'delta_svd_mean': float(ds[:, 0].mean()), 'delta_svd_std': float(ds[:, 0].std()),
                'delta_rand_mean': float(ds[:, 1].mean()), 'delta_rand_std': float(ds[:, 1].std()),
                'excess': float(ds[:, 0].mean() - ds[:, 1].mean()), 'n': len(ds)}
        # --- kappa per triple ---
        tdata = {}
        for (S, q0, q1, q2) in TRIPLES:
            ks = []
            for dom in ck['domains']:
                for si in range(ck['n_samples']):
                    k_d, k_r = sample_kappa(JR[(dom, si, l, S, q0)],
                                             JR[(dom, si, l, S, q1)],
                                             JR[(dom, si, l, S, q2)], R_MAIN, rng)
                    ks.append((k_d, k_r))
            ks = np.array(ks)
            tdata[(S, q0, q1, q2)] = {
                'kappa_mean': float(ks[:, 0].mean()), 'kappa_std': float(ks[:, 0].std()),
                'kappa_rand_mean': float(ks[:, 1].mean()),
                'kappa_frac_gt_1e-3': float((ks[:, 0] > 1e-3).mean()),
                'n': len(ks)}
        out['pairs'][str(l)] = {f'{S}_{p1}_{p2}_{zone}_dt{dt}': v
                                for (S, p1, p2, zone, dt), v in pdata.items()}
        out['triples'][str(l)] = {f'{S}_{q0}_{q1}_{q2}': v
                                  for (S, q0, q1, q2), v in tdata.items()}
        # --- r sweep (end pair 64,0,1 + triple 64) ---
        rs = {}
        for r in R_SWEEP:
            ds, ks = [], []
            for dom in ck['domains']:
                for si in range(ck['n_samples']):
                    d_d, _ = sample_delta_svd(JR[(dom, si, l, 64, 0)], JR[(dom, si, l, 64, 1)], r, rng)
                    ds.append(d_d)
                    k_d, _ = sample_kappa(JR[(dom, si, l, 64, 0)], JR[(dom, si, l, 64, 1)],
                                          JR[(dom, si, l, 64, 2)], r, rng)
                    ks.append(k_d)
            rs[str(r)] = {'delta_svd_mean': float(np.mean(ds)), 'kappa_mean': float(np.mean(ks))}
        out['r_sweep'][str(l)] = rs
        print(f'layer {l}: done', flush=True)
    path = os.path.join(a.results_dir, 'exp28_leaf_structure.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'[OK] saved {path}')


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



def selfcheck():
    torch.manual_seed(0)
    D, K, r = 64, 16, 8
    V = torch.randn(D, r)
    Q, _ = torch.linalg.qr(V)  # orthonormal base subspace
    # flat: same V, varying left vectors
    flat = [torch.linalg.qr(torch.randn(K, r))[0] @ torch.diag(torch.rand(r) + 0.5) @ Q.t() for _ in range(3)]
    # curved: V_t = Q_t V
    def rot():
        A = torch.randn(D, D)
        Uq, _ = torch.linalg.qr(A)
        return Uq
    curved = [torch.linalg.qr(torch.randn(K, r))[0] @ torch.diag(torch.rand(r) + 0.5) @ (rot() @ Q).t() for _ in range(3)]
    d_flat = max(principal_angles_max(right_subspace(flat[0], r), right_subspace(flat[1], r)),
                 principal_angles_max(right_subspace(flat[1], r), right_subspace(flat[2], r)))
    k_flat = sample_kappa(flat[0], flat[1], flat[2], r, None)[0]
    d_curv = max(principal_angles_max(right_subspace(curved[0], r), right_subspace(curved[1], r)),
                 principal_angles_max(right_subspace(curved[1], r), right_subspace(curved[2], r)))
    k_curv = sample_kappa(curved[0], curved[1], curved[2], r, None)[0]
    assert d_flat < 0.05, f'flat delta too large: {d_flat}'
    assert k_flat < 1e-4, f'flat kappa too large: {k_flat}'
    assert d_curv > 0.3, f'curved delta too small: {d_curv}'
    assert k_curv > 0.5, f'curved kappa too small: {k_curv}'
    print(f'selfcheck OK: flat d={d_flat:.5f} k={k_flat:.2e}, curved d={d_curv:.3f} k={k_curv:.3f}')


if __name__ == '__main__':
    main()

