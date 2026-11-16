#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# E-76 EIGENVECTORS: «одна мода или две подсистемы» (теория Т-1(в) / Т-А, docs/dynamic/3-5).
# Чтение (corpus TF, чистый текст) vs сатурация (own greedy, 63% лупов): если дип и петля —
# одна period-2 мода выученного оператора на разных амплитудах, то λ₁-подпространства lift-
# оператора в обеих режимах совпадают с точностью до базиса (principal angles ~0 против
# matched-rotated контроля). Не совпали — луч (dip, loopy) = две подсистемы, третья координата.
# Клетки: (corpus, own) × слои [1,11,23] × rank lift 64; контроли: (а) random-rotated той же
# пары на N=64 (null совпадения), (б) half-vs-half ВНУТРИ corpus — потолок стабильности оценки.
# Пре-рег (заранее): mean cos(real) > верх 95-перцентиля rotated-контроля И > 0.7 ⇒ «одна мода»;
#   cos(real) < нижняя половина rotated-полосы ⇒ «разные подсистемы»; между — «не разрешено».
import ast, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
D = ROOT/"results"/"e64"/"tpu"
LAYERS = [1, 11, 23]
R = 64
SEED = 0
src = open(ROOT/"experiments"/"kaggle_e64gpu"/"e64_gpu.py").read()
tree = ast.parse(src)
fns = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef)}
ns = {"np": np}
for w in ["vpath_units", "dcurve", "curves_series", "subspaces"]:
    exec(fns[w], ns)
dcurve = ns["dcurve"]; subspaces = ns["subspaces"]

def lift_docs(path, layer):
    """список V [T,R] по документам npz-файла"""
    z = np.load(path)
    out = []
    for k in sorted(z.files):
        if not k.startswith(f"h_L{layer}_doc"):
            continue
        H = z[k].astype(np.float64)
        H = H - H.mean(0)
        U = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
        # PCA-R базис от конкатенации документа
        C = np.cov(U.T)
        w, V = np.linalg.eigh(C)
        B = V[:, -R:]                       # [1024,R]
        A = U @ B                            # [T,R]
        Ad = np.concatenate([A[:-1], A[1:]], axis=1)  # lift 2R? нет: keep R из [:-1]->[1:]
        # М-lite в духе λ1_lite: X=A[:-1], Y=A[1:], M=(X^T Y)(X^T X)^-1
        X = A[:-1]; Y = A[1:]
        Gxx = X.T @ X + 1e-9*np.eye(R)
        M = np.linalg.solve(Gxx.T, (X.T @ Y).T).T
        out.append(dict(M=M, X=X, Y=Y, T=len(A)))
    return out

def lam1_subspace(rows, k=2):
    """объединённый М по документам (нормировка по ||X||^2), λ₁-подпространство [R,k]"""
    Sxy = np.zeros((R, R)); Sxx = np.zeros((R, R))
    for r in rows:
        Sxy += r["X"].T @ r["Y"]; Sxx += r["X"].T @ r["X"]
    M = np.linalg.solve(Sxx.T, Sxy.T).T
    # эрмитова часть (period-2 сидит в симметричной части flip-моды)
    Ms = 0.5*(M + M.T)
    w, V = np.linalg.eigh(Ms)
    idx = np.argsort(w)[:k]                 # самые отрицательные λ
    return V[:, idx], w[idx]

def pangles(Ua, Ub):
    s = np.linalg.svd(Ua.T @ Ub, compute_uv=False)
    return float(np.clip(s, 0, 1).mean())   # mean cos углов

def main():
    rng = np.random.default_rng(SEED)
    out = {}
    cells = {}
    for L in LAYERS:
        corpus = lift_docs(D/"P2_corpus.npz", L)
        own = lift_docs(D/"P1_own.npz", L)
        U_c, w_c = lam1_subspace(corpus)
        U_o, w_o = lam1_subspace(own)
        real = pangles(U_c, U_o)
        # контроль (а): rotated own-подпространство (qr от random gaussian) × 64
        rot = []
        for _ in range(64):
            Q = np.linalg.qr(rng.standard_normal((R, 2)))[0]
            rot.append(pangles(U_c, Q))
        rot = np.array(rot)
        # контроль (б): half-vs-half внутри corpus (стабильность оценки)
        h1 = corpus[::2]; h2 = corpus[1::2]
        Uh1, _ = lam1_subspace(h1); Uh2, _ = lam1_subspace(h2)
        stab = pangles(Uh1, Uh2)
        cells[f"L{L}"] = dict(cos_real=round(real, 3), rot_p95=round(float(np.percentile(rot, 95)), 3),
                              rot_med=round(float(np.median(rot)), 3), stab_half=round(stab, 3),
                              lam_corpus=[round(float(x), 3) for x in w_c],
                              lam_own=[round(float(x), 3) for x in w_o], n_c=len(corpus), n_o=len(own))
        print(f"L{L:2d} cos_real {real:.3f} | rot med {np.median(rot):.3f} p95 {np.percentile(rot,95):.3f} "
              f"| half-stab {stab:.3f} | λc {w_c.round(2)} λo {w_o.round(2)}", flush=True)
    # вердикт по слоям
    verd = {}
    for L in LAYERS:
        c = cells[f"L{L}"]
        one = c["cos_real"] > max(c["rot_p95"], 0.7)
        diff = c["cos_real"] < (c["rot_med"] + c["rot_p95"])/2 - 0.1
        verd[f"L{L}"] = "одна мода" if one else ("разные подсистемы" if diff else "не разрешено")
    out = dict(cells=cells, verdict=verd,
               note="principal angles λ₁(2D)-подпространств corpusTF vs ownGreedy; контроль=rotated; "
                    "stab=предел стабильности half-vs-half",
               prereg="cos>max(rot_p95,0.7) ⇒ одна мода; cos<середина rot-полосы ⇒ разные", seed=SEED, layers=LAYERS, rank=R)
    (ROOT/"results"/"e64"/"tpu"/"EIGEN.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(verd, ensure_ascii=False), flush=True)
    print("EIGEN DONE", flush=True)

main()
