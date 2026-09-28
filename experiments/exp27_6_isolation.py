#!/usr/bin/env python
"""
Exp 27.6: Domain isolation, block structure, null-baselines for Exp 27.5 verdicts.
P0 addendum per docs/Exp27.6.md. Post-processing only: results/exp27_jcache.pt.
No GPU, no new forwards.

Tasks (per docs/Exp27.6.md):
  T1  T_domain isolation: all / no_math / no_chat / no_code (K^F and K^G)
  T2  Bock 4x4 block matrix on K^G (+PNG)
  T3  Bootstrap CI (resample SAMPLES) on T_domain and T_ent
  T4  Permutation-null for ARI (fixed cluster labels, permute domains)
  T5  Hartigan dip of off-diagonal K^G + normal threshold + beta moment-matched null
  T6  Eigengap spectrum of K^F / K^G
  T7  Spearman rho(layer, C_a) with bootstrap CI + permutation p
  T8  Per-sample features: E, logE, r_eff, r_stable, norm_h, g_uni, H, sigma_z=not_cached

Inputs: results/exp27_jcache.pt (JR/P128/H/U128, v11 format).
NOTE: pair_matrix JSON and ari/dip fields from the TZ do not exist; K^F/K^G
matrices, cluster labels (agglomerative, seed=42, same as exp27.5) and dip
are computed here. Main slice S=64, K=128, pos=0 (exp27.5 main slice; the TZ
T7 line says K=64 'совпадает с main slice' which it does not — K=128 used).

Deterministic: seed 42 everywhere.
"""
import os, sys, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JCACHE_PATH = os.path.join(REPO, 'results', 'exp27_jcache.pt')
OUT_JSON = os.path.join(REPO, 'results', 'exp27_6_isolation.json')
OUT_PNG = os.path.join(REPO, 'results', 'exp27_6_bock_4x4.png')
DOMAINS = ['wiki', 'math', 'code', 'chat']
LAYERS = [6, 12, 18, 19, 22, 23]
N_SAMPLES = 8
S, K, POS = 64, 128, 0
SEED = 42
B_BOOT = 1000
B_ARI = 1000
B_DIP = 500

def _load():
    ck = torch.load(JCACHE_PATH, map_location='cpu', weights_only=False)
    jr, p128, h = ck['JR'], ck['P128'], ck['H']
    Js, Ps, Hs = [], [], []
    for l in LAYERS:
        for d in DOMAINS:
            for si in range(N_SAMPLES):
                Js.append(jr[(d, si, l, S, POS)][:K].float())
                Ps.append(p128[(d, si, l, S, POS)][:K].float())
                Hs.append(h[(d, si, l, S, POS)].float())
    J = torch.stack(Js); P = torch.stack(Ps); H = torch.stack(Hs)  # [L*n, ...]
    labels = [f"{d}_{si}" for d in DOMAINS for si in range(N_SAMPLES)]
    doms = np.array([d for d in DOMAINS for _ in range(N_SAMPLES)])
    # T5 mid-context slice: S=128, pos=32 (v11 dip pair 32_33), K=128
    J2s = [jr[(d, si, l, 128, 32)][:K].float() for l in LAYERS for d in DOMAINS for si in range(N_SAMPLES)]
    J2 = torch.stack(J2s)
    P2s = [p128[(d, si, l, 128, 32)][:K].float() for l in LAYERS for d in DOMAINS for si in range(N_SAMPLES)]
    P2 = torch.stack(P2s)
    return J, P, H, J2, P2, labels, doms


def _gram(J, P, l):
    """K^F (n x n) and K^G (n x n) for one layer. J is (n, K, D):
    K^F = cos of full gauge-free J (Frobenius; same as exp27 C_ij metric);
    K^G = cos of per-token gramians G_i = J_i J_i^T (K x K, = token_gramian in 27.5)."""
    i0 = LAYERS.index(l) * 32
    Jl = J[i0:i0 + 32]
    Pl = P[i0:i0 + 32]
    Jg = torch.einsum('ik,ikd->id', Pl, Jl)            # (n,D) = J^T p
    Jb = Pl[:, :, None] * (Jl - Jg[:, None, :])        # gauge-free (n,K,D)
    E = torch.einsum('ikd,ikd->i', Jb, Jb)
    Kf = torch.einsum('ikd,jkd->ij', Jb, Jb) / (E[:, None] * E[None, :]).clamp_min(1e-30).sqrt()
    Gb = torch.einsum('nkd,nld->nkl', Jb, Jb)          # (n,K,K) token-gramian
    gk = Gb.flatten(1).norm(dim=1)
    Kg = (Gb.flatten(1) @ Gb.flatten(1).T) / (gk[:, None] * gk[None, :]).clamp_min(1e-30)
    return Kf.numpy(), Kg.numpy()


def t1_isolation(Kf, Kg, doms):
    """T_domain for domain subsets: all / no_math / no_chat / no_code."""
    n = len(doms)
    out = {}
    for name, K in (('K_F', Kf), ('K_G', Kg)):
        row = {}
        for variant, excl in (('all', []), ('no_math', ['math']),
                              ('no_chat', ['chat']), ('no_code', ['code'])):
            keep = np.array([i for i, d in enumerate(doms) if d not in excl])
            within = cross = 0.0
            nw = nc = 0
            for i in keep:
                for j in keep:
                    if i == j:
                        continue
                    v = K[i, j]
                    if doms[i] == doms[j]:
                        within += v; nw += 1
                    else:
                        cross += v; nc += 1
            row[variant] = (within / nw - cross / nc)
        out[name] = row
    return out


def t2_bock(Kg, doms):
    """Full 4x4 block means of K^G. TZ 5.2: off-diag all 64 cells; diag = per-domain mean (i!=j)."""
    n = len(doms)
    M = np.zeros((4, 4))
    for a, da in enumerate(DOMAINS):
        for b, db in enumerate(DOMAINS):
            ia = np.where(doms == da)[0]
            ib = np.where(doms == db)[0]
            if a == b:
                M[a, b] = np.mean([Kg[i, j] for i in ia for j in ib if i != j])
            else:
                M[a, b] = Kg[np.ix_(ia, ib)].mean()
    return M


def _t_domain_of(K, doms, valid=None):
    """valid: (n,n) bool mask excluding resampled duplicate pairs idx[i]==idx[j]."""
    n = len(doms)
    valid = valid if valid is not None else np.ones((n, n), bool)
    within = [K[i, j] for i in range(n) for j in range(n)
              if i != j and valid[i, j] and doms[i] == doms[j]]
    cross = [K[i, j] for i in range(n) for j in range(n)
             if i != j and valid[i, j] and doms[i] != doms[j]]
    return np.mean(within) - np.mean(cross)


def t3_bootstrap(K, doms, H_q, B=B_BOOT):
    """Bootstrap CI on T_domain and T_ent. Resample SAMPLES (rows+cols), not pairs."""
    n = len(doms)
    rng = np.random.default_rng(SEED)
    Td = np.zeros(B); Te = np.zeros(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        Kb = K[np.ix_(idx, idx)]
        doms_b = doms[idx]
        valid = idx[:, None] != idx[None, :]   # exclude resampled duplicate pairs (Kb[i,j]=1)
        Td[b] = _t_domain_of(Kb, doms_b, valid)
        qb = H_q[idx]
        w = [Kb[i, j] for i in range(n) for j in range(n) if valid[i, j] and qb[i] == qb[j]]
        c = [Kb[i, j] for i in range(n) for j in range(n) if valid[i, j] and qb[i] != qb[j]]
        Te[b] = np.mean(w) - np.mean(c)
    def _ci(v):
        return {'mean': float(v.mean()), 'p2.5': float(np.percentile(v, 2.5)),
                'p50': float(np.percentile(v, 50)), 'p97.5': float(np.percentile(v, 97.5)),
                'ci_lo': float(np.percentile(v, 2.5)), 'ci_hi': float(np.percentile(v, 97.5)),
                'n_valid_boot': B}
    return _ci(Td), _ci(Te)


def _cluster_labels(DG, ks=range(2, 9)):
    """Same as exp27.5: agglomerative average, k = argmax silhouette."""
    import scipy.cluster.hierarchy as sch
    from scipy.spatial.distance import squareform
    import sklearn.metrics as skm
    Z = sch.linkage(squareform(DG, checks=False), method='average')
    best_k, best_s = None, -1
    for k in ks:
        lab = sch.fcluster(Z, t=k, criterion='maxclust')
        if len(set(lab)) < 2:
            continue
        s = skm.silhouette_score(DG, lab, metric='precomputed')
        if s > best_s:
            best_k, best_s = k, s
    lab = sch.fcluster(Z, t=best_k, criterion='maxclust')
    return lab, best_k, best_s


def t4_ari_null(K, doms, B=B_ARI):
    """Permutation-null for ARI: fixed cluster labels, permute domains."""
    from sklearn.metrics import adjusted_rand_score
    n = len(doms)
    DG = 1.0 - (K + K.T) / 2
    np.fill_diagonal(DG, 0.0)
    lab, k, sil = _cluster_labels(DG)
    ari_real = adjusted_rand_score(lab, doms)
    rng = np.random.default_rng(SEED + 1)
    nulls = np.zeros(B)
    for b in range(B):
        nulls[b] = adjusted_rand_score(lab, rng.permutation(doms))
    p = float((nulls >= ari_real - 1e-12).mean())
    return {'ari_real': float(ari_real), 'k': int(k), 'silhouette': float(sil),
            'ari_null_mean': float(nulls.mean()), 'ari_null_p95': float(np.percentile(nulls, 95)),
            'p_value': p}


def _dip_stat(x):
    """Hartigan dip statistic via the diptest library (Maechler port). kmax not provided by lib -> 0."""
    from diptest import dipstat
    xs = np.asarray(x, dtype=np.float64)
    if np.ptp(xs) <= 0:
        return 0.0, 0
    return float(dipstat(xs)), 0


def _fit_beta_moments(x):
    m = x.mean(); s = x.std()
    if s < 1e-12:
        return None
    cv = s / m if abs(m) > 1e-12 else np.inf
    if abs(m) < 1e-12 or not np.isfinite(cv):
        return None
    if cv >= 1:
        a = 1.0 / cv ** 2
        b = 1.0 / a - 1.0
    else:
        a = m / s ** 2 - 1
        b = 1 / a - 1
    return (a, b) if (a > 0 and b > 0) else None


def t5_dip(K, B=B_DIP, threshold=0.05):
    """Dip of off-diagonal K^G values + beta moment-matched null + normal threshold."""
    n = K.shape[0]
    off = K[~np.eye(n, dtype=bool)]
    dip_real, kmax = _dip_stat(off)
    rng = np.random.default_rng(SEED + 2)
    nulls = np.zeros(B)
    bm = _fit_beta_moments(off)
    if bm is not None:
        for b in range(B):
            nulls[b], _ = _dip_stat(rng.beta(bm[0], bm[1], len(off)))
    else:
        mu, sd = off.mean(), off.std()
        for b in range(B):
            nulls[b], _ = _dip_stat(rng.normal(mu, max(sd, 1e-12), len(off)))
    return {'dip': float(dip_real), 'k_max': int(kmax),
            'normal_threshold': threshold,
            'dip_null_p95': float(np.percentile(nulls, 95)),
            'reject_unimodal': bool(dip_real > np.percentile(nulls, 95))}  # review B.4: null-based


def t6_eigengap(Kf, Kg):
    """Spectrum + eigengap ratios of symmetrized K^F and K^G."""
    out = {}
    for name, K in (('K_F', Kf), ('K_G', Kg)):
        Ks = (K + K.T) / 2
        ev = np.sort(np.linalg.eigvalsh(Ks))[::-1]
        ev = np.clip(ev, 1e-12, None)
        p = ev / ev.sum()
        r_eff = float(np.exp(-(p * np.log(p)).sum()))
        gaps = {f'{i}/{i+1}': float(ev[i - 1] / ev[i]) for i in (2, 3, 4)}
        am = int(np.argmax(list(gaps.values()))) + 1
        out[name] = {'lambda_top': [float(x) for x in ev[:31]],
                     'gap_ratios': gaps, 'argmax_gap': am, 'r_eff_K': r_eff,
                     'has_structure': bool(max(gaps.values()) >= 1.5)}
    return out


def t7_depth_spearman(exp27_json):
    """Spearman rho(layer, C_a) at S64 K64 pair0_1 with bootstrap CI + permutation p."""
    from scipy.stats import spearmanr
    vals = {}
    for key in ('C_a_pospk', 'C_a_samepk_late', 'C_a_raw'):
        vals[key] = np.array([exp27_json['layers'][str(l)]['configs']['S64_K64_pair0_1'][key]
                              for l in LAYERS])
    layers = np.array(LAYERS, dtype=float)
    rng = np.random.default_rng(SEED + 3)
    out = {}
    for key, x in vals.items():
        rho, _ = spearmanr(layers, x)
        boot = np.zeros(B_BOOT)
        for b in range(B_BOOT):
            idx = rng.integers(0, 6, 6)
            if len(set(layers[idx])) < 2:
                boot[b] = np.nan
            else:
                r, _ = spearmanr(layers[idx], x[idx])
                boot[b] = r if np.isfinite(r) else np.nan
        perm = np.zeros(B_ARI)
        for b in range(B_ARI):
            r, _ = spearmanr(layers, rng.permutation(x))
            perm[b] = abs(r) if np.isfinite(r) else 0.0
        out[key] = {'rho': float(rho),
                    'ci_lo': float(np.nanpercentile(boot, 2.5)),
                    'ci_hi': float(np.nanpercentile(boot, 97.5)),
                    'p_perm': float((perm >= abs(rho) - 1e-12).mean())}
    return out


def t8_per_sample(J, P, H, labels):
    """Per-sample features: E, logE, r_eff, r_stable, norm_h, g_uni, H (entropy)."""
    out = {}
    for l in LAYERS:
        row = {}
        i0 = LAYERS.index(l) * 32
        Jl = J[i0:i0 + 32]
        Pl = P[i0:i0 + 32]
        Hl = H[i0:i0 + 32]
        for i in range(32):
            Jraw = Jl[i]
            c = Jraw.T @ Pl[i]                       # J^T p
            Jg = (Pl[i][:, None] * (Jraw - c[None, :]))  # gauge-free (same as _gram)
            E = float((Jg ** 2).sum())  # TZ 11.2: E = ||J_i||_F^2 (gauge-free J)

            # TZ 11.2: g_uni = ||Q J_i||^2_F / E_i, Q = u u^T, u = 1/sqrt(K)
            # (uniform direction, not p-direction; gauge-free projection kills it — review A.3)
            u = np.ones(K) / np.sqrt(K)
            uJ = Jg.T @ u
            g_uni = float((uJ ** 2).sum()) / max(E, 1e-12)
            sv = torch.linalg.svdvals(Jg)
            lam = sv ** 2
            p = lam / lam.sum().clamp_min(1e-12)
            r_eff = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
            r_stable = float((lam.sum() / lam.max().clamp_min(1e-12)))
            pk = Pl[i] / Pl[i].sum().clamp_min(1e-12)
            Hent = float(-(pk * torch.log(pk.clamp_min(1e-12))).sum())
            row[labels[i]] = {'E': E, 'logE': float(np.log(E + 1e-12)),
                              'r_eff': r_eff, 'r_stable': r_stable,
                              'norm_h': float(Hl[i].norm()),
                              'g_uni': g_uni, 'H': Hent, 'sigma_z': 'not_cached'}
        out[str(l)] = row
    return out


def main():
    t0 = time.time()
    np.random.seed(SEED); torch.manual_seed(SEED)
    J, P, H, J2, P2, labels, doms = _load()
    print(f'loaded J={tuple(J.shape)} P={tuple(P.shape)} H={tuple(H.shape)}')

    # entropy quartile labels per layer (for T3 T_ent)
    exp27 = json.load(open(os.path.join(REPO, 'results', 'exp27_noise_floor.json')))

    per_layer = {}
    for l in LAYERS:
        Kf, Kg = _gram(J, P, l)
        Kf = (Kf + Kf.T) / 2; Kg = (Kg + Kg.T) / 2
        # entropy quartiles: entropy of P (per sample, this layer)
        i0 = LAYERS.index(l) * 32
        Pl = P[i0:i0 + 32].numpy()
        pk = Pl / Pl.sum(1, keepdims=True)
        Hs = -(pk * np.log(pk + 1e-12)).sum(1)
        H_q = np.digitize(Hs, np.percentile(Hs, [25, 50, 75]))
        r = {}
        r['T1'] = t1_isolation(Kf, Kg, doms)
        r['T2'] = t2_bock(Kg, doms)
        td, te = t3_bootstrap(Kf, doms, H_q)
        r['T3'] = {'T_domain': td, 'T_ent': te}
        r['T4'] = t4_ari_null(Kf, doms)
        Kg128 = _gram(J2, P2, l)[1]
        Kg128 = (Kg128 + Kg128.T) / 2
        r['T5'] = {'S64_pos0': t5_dip(Kg), 'S128_mid32_33': t5_dip(Kg128)}
        r['T6'] = t6_eigengap(Kf, Kg)
        per_layer[str(l)] = r
        print(f'  L{l}: T1 K_F all={r["T1"]["K_F"]["all"]:+.4f} no_math={r["T1"]["K_F"]["no_math"]:+.4f} | '
              f'T3 T_dom mean={td["mean"]:+.4f} [{td["ci_lo"]:+.4f},{td["ci_hi"]:+.4f}] | '
              f'T4 ari={r["T4"]["ari_real"]:+.4f} p={r["T4"]["p_value"]:.3f} | '
              f'T5 dip S64={r["T5"]["S64_pos0"]["dip"]:.3f} S128mid={r["T5"]["S128_mid32_33"]["dip"]:.3f} | '
              f'T6 K_G struct={r["T6"]["K_G"]["has_structure"]}')

    # --- TZ 13: per-task verdicts ---
    def _t1v():
        # review B.3: ratio undefined at noise level; sign flips invalidate ratios
        if max(abs(per_layer[l]['T1']['K_F']['all']) for l in per_layer) < 0.01:
            return ('inconclusive: max |T_domain(all)| < 0.01 — signal at noise floor, '
                    'ratios meaningless')
        bad = [l for l in per_layer
               if per_layer[l]['T1']['K_F']['all'] > 0
               and per_layer[l]['T1']['K_F']['no_math'] < 0.2 * per_layer[l]['T1']['K_F']['all']]
        ok = [l for l in per_layer
              if per_layer[l]['T1']['K_F']['all'] > 0
              and per_layer[l]['T1']['K_F']['no_math'] >= 0.5 * per_layer[l]['T1']['K_F']['all']]
        if len(ok) == 6: return 'pass'
        if bad: return f'fail: T1 no_math < 0.2*all at {bad}'
        return 'inconclusive'
    def _t2v():
        isolated = all(
            per_layer[l]['T2'][1, 1] < 0.5 * min(per_layer[l]['T2'][a, a] for a in (0, 2, 3))
            and per_layer[l]['T2'][1, :].mean() < per_layer[l]['T2'].mean()
            for l in per_layer)
        return 'pass' if isolated else 'inconclusive (math isolation not uniform)'
    def _t3v():
        z = [l for l in per_layer if per_layer[l]['T3']['T_domain']['ci_lo'] <= 0 <= per_layer[l]['T3']['T_domain']['ci_hi']]
        return ('fail: T_domain CI contains 0 on all layers (domain axis ~ 0 at N=32)'
                if len(z) == 6 else ('pass' if not z else 'inconclusive'))
    def _t4v():
        # review B.2: Bonferroni across 6 layers, alpha = 0.05/6
        alpha = 0.05 / 6
        p = [per_layer[l]['T4']['p_value'] for l in per_layer]
        sig = [l for l, x in zip(per_layer, p) if x < 0.05]
        corr = [l for l, x in zip(per_layer, p) if x < alpha]
        if not sig: return 'pass'
        if corr: return f'fail: ARI significant (Bonferroni) at {corr}'
        return f'inconclusive: raw p<0.05 at {sig} but not Bonferroni alpha={alpha:.4f}'
    def _t5v():
        rej = [l for l in per_layer if per_layer[l]['T5']['S64_pos0']['reject_unimodal']]
        return 'pass' if len(rej) >= 3 else ('inconclusive' if rej else 'fail: unimodal')
    def _t6v():
        s = [per_layer[l]['T6']['K_G']['has_structure'] for l in per_layer]
        return ('pass: no spectral structure -> silhouette block of 27.5 inapplicable'
                if not any(s) else f'inconclusive: structure at {[l for l in per_layer if per_layer[l]["T6"]["K_G"]["has_structure"]]}')
    t7r = out7 = t7_depth_spearman(exp27)
    def _t7v():
        r = t7r['C_a_pospk']
        return 'pass' if (r['p_perm'] < 0.05 and r['ci_lo'] > 0.5) else (
            'fail: depth trend not distinguished from noise' if 0 in (r['ci_lo'], r['ci_hi']) or (r['ci_lo'] < 0 < r['ci_hi']) else 'inconclusive')
    t8r = t8_per_sample(J, P, H, labels)
    def _t8v():
        for l in t8r:
            re = [t8r[l][k]['r_eff'] for k in t8r[l]]
            if max(re) / max(min(re), 1e-9) >= 5:
                return 'pass'
        return 'inconclusive: r_eff range < 5x'
    verdicts = {'T1': _t1v(), 'T2': _t2v(), 'T3': _t3v(), 'T4': _t4v(),
                'T5': _t5v(), 'T6': _t6v(), 'T7': _t7v(), 'T8': _t8v()}
    print('VERDICTS:', json.dumps(verdicts, indent=1))

    out = {
        'params': {'S': S, 'K': K, 'pos': POS, 'seed': SEED,
                   'B_perm': B_ARI, 'B_boot': B_BOOT, 'B_dip': B_DIP,
                   'layers': LAYERS, 'domains': DOMAINS,
                   'note': 'main slice S64 K128 pos0 (exp27.5 main slice); TZ T7 says K=64, '
                           'exp27 baseline S64_K64 pair0_1 used for C_a there'},
        'T1_isolation': {l: per_layer[l]['T1'] for l in per_layer},
        'T2_bock': {l: {'matrix': per_layer[l]['T2'].tolist(), 'domains': DOMAINS} for l in per_layer},
        'T3_bootstrap': {l: per_layer[l]['T3'] for l in per_layer},
        'T4_ari_null': {l: per_layer[l]['T4'] for l in per_layer},
        'T5_dip': {l: per_layer[l]['T5'] for l in per_layer},
        'T6_eigengap': {l: per_layer[l]['T6'] for l in per_layer},
        'T7_depth_spearman': t7r,
        'T8_per_sample': t8r,
        'verdicts': verdicts,
    }
    json.dump(out, open(OUT_JSON, 'w'), indent=1, default=float)
    print(f'JSON {os.path.getsize(OUT_JSON)} bytes -> {OUT_JSON}')

    # PNG: 2x3 heatmap 4x4 K^G (+K^F diagonal in title), layers 6/12/18/22 + L19/L23
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, l in zip(axes.ravel(), LAYERS):
        Kg = _gram(J, P, l)[1]
        M = t2_bock(Kg, doms)
        im = ax.imshow(M, cmap='RdBu_r', vmin=-0.1, vmax=0.1)
        Kfdiag = np.array([t2_bock(_gram(J, P, l)[0], doms)[a, a] for a in range(4)])
        ax.set_title(f'L{l} K^G blocks (K^F diag: ' +
                     ' '.join(f'{v:.3f}' for v in Kfdiag) + ')')
        ax.set_xticks(range(4), ['wiki', 'math', 'code', 'chat'], fontsize=7)
        ax.set_yticks(range(4), ['wiki', 'math', 'code', 'chat'], fontsize=7)
        for a in range(4):
            for b in range(4):
                ax.text(b, a, f'{M[a, b]:.3f}', ha='center', va='center', fontsize=6)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f'PNG -> {OUT_PNG}')
    print(f'DONE in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
