"""Аудит E-39b, шаг 2: отвечает ли κ вообще на дрожание топ-r субпространства того же размера.
Если да — κ живой метр и «калибровочная выровненность» требует когерентного null'а;
если нет — κ при 12 % дрейфе слеп и вывод E-39b пуст."""
import sys, json, numpy as np, torch
sys.path.insert(0, ".")
import exp28_leaf_structure as E28
from _e38_cache import jcache_path
L, r, S = 23, 16, 128
ck = torch.load(jcache_path(L), map_location="cpu", weights_only=False)
DOMS, NS = ck["domains"], ck["n_samples"]
J = np.stack([ck["JR"][(d, s, t)].numpy().astype(np.float32) for d in DOMS for s in range(NS) for t in range(S)])
del ck
N = len(J)

def tops(Jst):
    """_batched: правые векторы топ-r (numpy, по 256 позиций за раз)."""
    out = []
    for i in range(0, len(Jst), 256):
        _, _, Vh = np.linalg.svd(Jst[i:i + 256], full_matrices=False)
        out.append(Vh[:, :r, :].astype(np.float32))
    return np.concatenate(out)                      # [N, r, D]

def kappa(Jst):
    ks = []
    for i in range(0, N, S):
        _, _, k2, _ = E28.document_profile(torch.from_numpy(Jst[i:i + S]), [1, r, 32], r_main=r)
        ks.append(k2[~np.isnan(k2)])
    return float(np.median(np.concatenate(ks)))

V = tops(J)
JV = np.einsum('nkd,nrd->nkr', J, V)                 # координаты в топ-r
k0 = kappa(J)
res = {"kappa_base": k0, "sweep": []}
print("κ базы %.6f" % k0, flush=True)
rng = np.random.default_rng(0)
for delta in (0.0, 0.02, 0.05, 0.12, 0.25, 0.5):
    # Q_r — случайное единичное r-подпространство, ортогональное V_r (по подпространству, не по позиции)
    G = rng.standard_normal((N, V.shape[-1], r)).astype(np.float32)
    Q = G - np.einsum('ndx,nrd,ner->ndx', V.transpose(0, 2, 1), V, np.ones((1, r, 1), np.float32))
    Q = np.linalg.qr(Q.reshape(-1, V.shape[-1]))[0].reshape(N, -1)[:, :r].transpose(0, 2, 1)  # [N,r,D]
    Jp = np.einsum('nkr,nrd->nkd', np.cos(delta) * JV, V) + \
         np.einsum('nkr,nrd->nkd', np.sin(delta) * JV, Q) + \
         (J - np.einsum('nkr,nrd->nkd', JV, V))
    d = float(np.median(np.linalg.norm((Jp - J).reshape(N, -1), axis=1) / np.linalg.norm(J.reshape(N, -1), axis=1)))
    k = kappa(Jp.astype(np.float32))
    res["sweep"].append({"delta_rad": delta, "drift_median": d, "kappa": k, "ratio": k / k0})
    print("δ=%.2f рад  drift %.4f  κ %.6f  ratio %.4f" % (delta, d, k, k / k0), flush=True)
json.dump(res, open("/tmp/null39b2.json", "w"), indent=1)
