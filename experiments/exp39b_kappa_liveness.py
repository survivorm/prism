"""E-39b шаг 2 (лёгкая версия): живость κ — дрожание топ-r субпространства на угол δ при
сохранённых нормах, 8 документов L23, память по документу."""
import sys, json, numpy as np, torch
sys.path.insert(0, ".")
import exp28_leaf_structure as E28
from _e38_cache import jcache_path
L, r, S = 23, 16, 128
DOMS = ["wiki", "math", "code", "chat"]
ck = torch.load(jcache_path(L), map_location="cpu", weights_only=False)
DEL = [0.0, 0.02, 0.05, 0.12, 0.25, 0.5]
acc = {d: [] for d in DEL}
drift = {d: [] for d in DEL}
rng = np.random.default_rng(0)
for dom in DOMS:
    for si in range(2):
        Jd = np.stack([ck["JR"][(dom, si, t)].numpy().astype(np.float32) for t in range(S)])
        T = torch.from_numpy(Jd)
        V, _ = E28.right_subspaces(T, [r])[r]                # [S, D, r]
        Vn = V.numpy()                                        # [n, D, r]
        JV = np.einsum('nkd,ndr->nkr', Jd, Vn)
        Jr = Jd - np.einsum('nkr,ndr->nkd', JV, Vn)           # остаток за пределами топ-r
        G = rng.standard_normal((S, Vn.shape[1], r)).astype(np.float32)
        VG = np.einsum('ndr,nds->nrs', Vn, G)                 # [n, r, r]
        Q = G - np.einsum('ndr,nrs->nds', Vn, VG)             # ортогонально V_n
        Q = np.linalg.qr(Q)[0]                                # [n, D, r]
        for d in DEL:
            Jp = np.cos(d) * np.einsum('nkr,ndr->nkd', JV, Vn) + np.sin(d) * np.einsum('nkr,ndr->nkd', JV, Q) + Jr
            drift[d].append(float(np.median(np.linalg.norm((Jp - Jd).reshape(S, -1), axis=1) /
                                           np.linalg.norm(Jd.reshape(S, -1), axis=1))))
            _, _, k2, _ = E28.document_profile(torch.from_numpy(Jp.astype(np.float32)), [1, r, 32], r_main=r)
            acc[d].append(k2[~np.isnan(k2)])
        print("док %s/%d готов" % (dom, si), flush=True)
res = {"n_docs": 8, "sweep": []}
k0 = float(np.median(np.concatenate(acc[0.0])))
for d in DEL:
    k = float(np.median(np.concatenate(acc[d]))) if d else k0
    res["sweep"].append({"delta_rad": d, "drift_median": float(np.mean(drift[d])), "kappa": k, "ratio": k / k0})
    print("δ=%.2f  drift %.4f  κ %.6f  ratio %.4f" % (d, np.mean(drift[d]), k, k / k0), flush=True)
json.dump(res, open("/tmp/liveness2.json", "w"), indent=1)
