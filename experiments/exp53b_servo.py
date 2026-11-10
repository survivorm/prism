#!/usr/bin/env python
"""E-53b (docs/E53b_v2.md): damped servo + slow drift для v_t листа.

Стадии: -1 синтетика (iso/weak-w/strong-w/null; аудит разрешения),
0a скалярный b (v_t = b v_{t-1}+), 0b CCA-канон.корреляции vs emp. shuffle-null,
1 (a,b) c out-of-sample c* + bootstrap CI, 2/4 oracle-vs-model (ridge на PCA-фичах),
3 Δ-кривая измеренная vs предсказанная из fitted-симуляции, m_t гейт vs per-doc константа.

Данные: /data/prism_kernels/exp47b (T=384, d=1024, 64 дока).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import json
import numpy as np

RNG = np.random.default_rng(532)
D, T = 1024, 384
SHRINK = 0.10          # рег. к CCA: Σ + SHRINK·tr/d·I
KPCA = 256
LAGS = (1, 2, 3, 4, 8, 16)

# ---------- данные ----------

def load_real():
    z = np.load("/data/prism_kernels/exp47b/exp47b_gen_kinematics.npz", allow_pickle=True)
    docs = json.load(open("../results/exp47b_gen_kinematics.json"))["docs"]
    out = []
    for d in docs:
        k = f"{d['dom']}_{d['si']}"
        u = z[k + "_u"].astype(np.float32)[:T]
        hn = np.asarray(z[k].item()["hnorm"], dtype=np.float32)[:T]
        U = u * hn[:, None]
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        out.append(dict(name=k, dom=d["dom"], si=d["si"], U=U))
    return out


def derive(doc):
    """v_t (не нормиров.), Vc (per-doc centering), c* по первой половине."""
    U = doc["U"]
    Vo = np.zeros((T - 1, D), np.float32)
    for t in range(T - 1):
        Vo[t] = U[t + 1] - float(U[t + 1] @ U[t]) * U[t]
    ok = np.linalg.norm(Vo, axis=1) > 1e-6
    m = Vo[ok].mean(0, keepdims=True) if ok.any() else np.zeros((1, Vo.shape[1]), Vo.dtype)
    doc.update(V=Vo, ok=ok, cstar=U[:T // 2].mean(0) / np.linalg.norm(U[:T // 2].mean(0)),
               eta=m[0] / max(np.linalg.norm(m[0]), 1e-12), Vc=Vo - m)
    return doc

# ---------- нули и CI ----------

def bci(v, B=3000):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if not len(v):
        return [float("nan")] * 3
    idx = RNG.integers(0, len(v), (B, len(v)))
    bs = np.median(v[idx], 1)
    return [float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def shuffle_seq(seq):
    p = RNG.permutation(len(seq))
    return seq[p]

# ---------- стадия 0 ----------

def scalar_ar(pairs_ok, Vc):
    """b = Σ⟨v_t,v_{t-1}⟩/Σ||v_{t-1}||² per doc; pairs_ok: список (t-1,t)."""
    if len(pairs_ok) < 2:
        return 0.0, 0.0
    P = Vc[np.array([a for a, _ in pairs_ok])]
    N = Vc[np.array([b for _, b in pairs_ok])]
    return float((P * N).sum()), float((P * P).sum())


def cca_spectrum(Ps, Ns, reps=1):
    """канон.корреляции M=Σyy^-1/2 Σxy Σxx^-1/2 со shrinkage; пул (Ps[t], Ns[t])=(prev, cur)."""
    def build(pairs):
        X = np.concatenate([p for p, _ in pairs])
        Y = np.concatenate([n for _, n in pairs])
        Sxx = X.T @ X / len(X); Syy = Y.T @ Y / len(Y); Sxy = Y.T @ X / len(X)
        f = SHRINK * np.trace(Sxx) / D
        Sxx = Sxx + f * np.eye(D, dtype=np.float32)
        f = SHRINK * np.trace(Syy) / D
        Syy = Syy + f * np.eye(D, dtype=np.float32)
        def wisqrt(S):
            w, Q = np.linalg.eigh(S.astype(np.float64))
            return ((Q / np.sqrt(np.maximum(w, 1e-30))) @ Q.T).astype(np.float32)
        M = wisqrt(Syy) @ Sxy @ wisqrt(Sxx)
        return np.linalg.svd(M, compute_uv=False)
    real = build(pairs)
    nulls = []
    for _ in range(reps):
        sh = [(shuffle_seq(p), shuffle_seq(n)) for p, n in pairs]
        nulls.append(build(sh))
    tau = float(np.median([n[0] for n in nulls]))   # верх null'а (медиана по репликам)
    return real[:15].astype(float).tolist(), tau, int((real > tau).sum()), [float(n[0]) for n in nulls]


def stage0(docs, tag, lag1_null_from_curve=None):
    res = {}
    # 0a скаляр
    xs, ss, per = [], [], []
    for d in docs:
        pr = [(t - 1, t) for t in range(1, T - 1) if d["ok"][t - 1] and d["ok"][t]]
        x, s = scalar_ar(pr, d["Vc"])
        xs.append(x); ss.append(s); per.append((x, s))
    b = float(np.sum(xs) / np.sum(ss))
    bx = []
    for _ in range(2000):
        i = RNG.integers(0, len(docs), len(docs))
        bx.append(sum(per[j][0] for j in i) / max(sum(per[j][1] for j in i), 1e-30))
    res["scalar_b"] = b
    res["scalar_beta"] = -b                      # модель: v_t = -β v_{t-1}
    res["scalar_ci"] = [float(np.median(bx)), float(np.percentile(bx, 2.5)), float(np.percentile(bx, 97.5))]
    # null: shuffle порядка Vc внутри дока
    xn = []
    for rep in range(3):
        xx = ss_ = 0.0
        for d in docs:
            Vs = d["Vc"][d["ok"]]
            if len(Vs) < 3:
                continue
            Vn = shuffle_seq(Vs)
            xx += float((Vn[1:] * Vn[:-1]).sum())
            ss_ += float((Vn[:-1] * Vn[:-1]).sum())
        if ss_ > 1e-30:
            xn.append(xx / ss_)
    res["scalar_b_null"] = float(np.median(xn)) if xn else float("nan")
    # гетерогенность β̄ per-doc (ревью 3.2): b_i = Σx_t/Σs_t, CI — bootstrap по t внутри дока
    bper, sdper = [], []
    for d in docs:
        idx = np.nonzero(d["ok"][1:] & d["ok"][:-1])[0]
        if len(idx) < 10:
            continue
        P = d["Vc"][idx]; N = d["Vc"][idx + 1]
        xt = (P * N).sum(1); st = (P * P).sum(1)
        if st.sum() <= 1e-30:
            continue
        bs = []
        for _ in range(300):
            j = RNG.integers(0, len(idx), len(idx))
            bs.append(xt[j].sum() / max(st[j].sum(), 1e-30))
        bper.append(float(xt.sum() / st.sum()))
        sdper.append(float(np.std(bs)))
    res["scalar_b_perdoc"] = dict(n=len(bper), median=float(np.median(bper)),
                                  sd_between=float(np.std(bper)),
                                  median_ci_halfwidth=float(np.median(sdper)),
                                  iqr=[float(np.percentile(bper, 25)), float(np.percentile(bper, 75))],
                                  frac_neg=float(np.mean([b < 0 for b in bper])))
    # 0b CCA
    pairs = [(d["Vc"][np.array([t - 1 for t in range(1, T - 1) if d["ok"][t - 1] and d["ok"][t]])],
              d["Vc"][np.array([t for t in range(1, T - 1) if d["ok"][t - 1] and d["ok"][t]])])
             for d in docs]
    top, tau, cnt, nmax = cca_spectrum(pairs, None, reps=2)
    res.update(cca_top15=top, cca_null_tau=tau, cca_count_above_null=cnt, cca_null_max=nmax)
    return res

# ---------- стадия 1: (a,b) с out-of-sample c* ----------

def stage1(docs, tag):
    per = []
    for d in docs:
        g = d["cstar"][None, :] - d["U"][T // 2:T - 1]          # v_t для t∈[T/2, T-2]
        p = d["V"][T // 2 - 1:T - 2]
        y = d["V"][T // 2:T - 1]
        m = d["ok"][T // 2:T - 1] & d["ok"][T // 2 - 1:T - 2]
        g, p, y = g[m].astype(np.float64), p[m].astype(np.float64), y[m].astype(np.float64)
        S = np.array([[ (g * g).sum(), (g * p).sum()], [(g * p).sum(), (p * p).sum()]])
        r = np.array([(g * y).sum(), (p * y).sum()])
        yty = float((y * y).sum())
        per.append((S, r, yty, int(m.sum())))
    Sp = sum(x[0] for x in per); rp = sum(x[1] for x in per)
    coef = np.linalg.solve(Sp, rp)
    sst = sum(x[2] for x in per)
    R2_abs = float(1 - (sst - float(rp @ coef)) / sst)   # residual SS = SST − r·coef
    bb = []
    for _ in range(2000):
        i = RNG.integers(0, len(per), len(per))
        S = sum(per[j][0] for j in i); r = sum(per[j][1] for j in i)
        try:
            c = np.linalg.solve(S, r)
            bb.append(c)
        except np.linalg.LinAlgError:
            pass
    bb = np.array(bb)
    gg, gp, pp = Sp[0, 0], Sp[0, 1], Sp[1, 1]
    return dict(a=float(coef[0]), b=float(coef[1]),
                a_ci=[float(np.percentile(bb[:, 0], q)) for q in (2.5, 50, 97.5)],
                b_ci=[float(np.percentile(bb[:, 1], q)) for q in (2.5, 50, 97.5)],
                R2_abs=R2_abs,
                cos_regressors=float(gp / np.sqrt(gg * pp)),
                cond=float(np.linalg.cond(Sp / Sp[1, 1])))

# ---------- стадии 2/4: oracle vs model ----------

def pca_fit(X, k):
    mu = X.mean(0)
    Xc = X - mu
    # ковариация k-largest через SVD на Xc (n×2d): экономно — через Xc^T Xc (2d×2d float32)
    C = (Xc.T @ Xc).astype(np.float32)
    w, Q = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1][:k]
    return mu, Q[:, idx].astype(np.float32)


def pca_apply(X, mu, Q):
    return (X - mu) @ Q


def ridge_fit(F, Y, lams=(1e-2, 1, 10, 100, 1e3, 1e5)):
    n = len(F)
    cut = int(0.8 * n)
    G = F.T @ F
    R = (F.T @ Y).astype(np.float64)
    best, bl = -1e18, None
    for lam in lams:
        B = np.linalg.solve(G + lam * np.eye(len(G), dtype=np.float32), R)
        pred = F[:cut] @ B
        sse = float(((Y[:cut] - pred) ** 2).sum())
        r2 = 1 - sse / float((Y[:cut] ** 2).sum())
        if r2 > best:
            best, bl = r2, lam
    B = np.linalg.solve(G + bl * np.eye(len(G), dtype=np.float32), R)
    return B, bl, float(best)


def r2_on(B, F, Y):
    pred = F @ B
    return float(1 - float(((Y - pred) ** 2).sum()) / float((Y ** 2).sum()))


def feat_blocks(docs_sel, half_from, centerY=True):
    """G,P,U,Y по t∈[half_from, T-2] с масками; Y центрируется per-doc mean(V) (фиксированный η-эффект)."""
    Gs, Ps, Us, Ys = [], [], [], []
    for d in docs_sel:
        m = d["ok"][T // 2:T - 1] & d["ok"][T // 2 - 1:T - 2]
        idx = np.nonzero(m)[0]
        g = d["cstar"][None, :] - d["U"][T // 2 + idx]
        p = d["V"][T // 2 - 1 + idx]
        u = d["U"][T // 2 + idx]
        y = d["V"][T // 2 + idx]
        if centerY:
            y = y - d["V"][d["ok"]].mean(0, keepdims=True)
        Gs.append(g); Ps.append(p); Us.append(u); Ys.append(y)
    return np.concatenate(Gs), np.concatenate(Ps), np.concatenate(Us), np.concatenate(Ys)


def stage24(docs):
    tr = [d for d in docs if d["si"] % 4 != 3]
    te = [d for d in docs if d["si"] % 4 == 3]
    Gtr, Ptr, Utr, Ytr = feat_blocks(tr, 0)
    Gte, Pte, Ute, Yte = feat_blocks(te, 0)
    muM, QM = pca_fit(np.concatenate([Gtr, Ptr], 1), KPCA)
    muO, QO = pca_fit(np.concatenate([Utr, Ptr], 1), KPCA)
    Fm_tr = pca_apply(np.concatenate([Gtr, Ptr], 1), muM, QM)
    Fo_tr = pca_apply(np.concatenate([Utr, Ptr], 1), muO, QO)
    Fm_te = pca_apply(np.concatenate([Gte, Pte], 1), muM, QM)
    Fo_te = pca_apply(np.concatenate([Ute, Pte], 1), muO, QO)
    Bm, lm, _ = ridge_fit(Fm_tr, Ytr)
    Bo, lo, _ = ridge_fit(Fo_tr, Ytr)
    folds = np.array_split(np.arange(len(tr)), 5)
    r2of = []
    for f in folds:
        keep = np.setdiff1d(np.arange(len(tr)), f)
        G1, P1, U1, Y1 = feat_blocks([tr[i] for i in keep], 0)
        G2, P2, U2, Y2 = feat_blocks([tr[i] for i in f], 0)
        muo, Qo = pca_fit(np.concatenate([U1, P1], 1), KPCA)
        B, _, _ = ridge_fit(pca_apply(np.concatenate([U1, P1], 1), muo, Qo), Y1)
        r2of.append(r2_on(B, pca_apply(np.concatenate([U2, P2], 1), muo, Qo), Y2))
    out = dict(
        R2_model_test=r2_on(Bm, Fm_te, Yte),
        R2_oracle_test=r2_on(Bo, Fo_te, Yte),
        R2_oracle_train_crossfitted=[float(x) for x in r2of],
        lam_model=lm, lam_oracle=lo, n_train=len(Ytr), n_test=len(Yte))
    out["ratio_model_oracle_test"] = (out["R2_model_test"] / out["R2_oracle_test"]
                                      if out["R2_oracle_test"] > 0.005 else None)
    return out

# ---------- m_t и гейт per-doc константы ----------

def smooth_ma(V, k=9):
    c = np.concatenate([np.zeros((1, V.shape[1]), V.dtype), np.cumsum(V, 0)], 0)
    idx = np.arange(len(V))
    lo = np.maximum(idx - k // 2, 0); hi = np.minimum(idx + k // 2 + 1, len(V))
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def slow_mode(docs):
    sm = [smooth_ma(d["Vc"][d["ok"]]) for d in docs]
    S = np.concatenate(sm).astype(np.float32)
    S -= S.mean(0)
    w, Q = np.linalg.eigh((S.T @ S / len(S)).astype(np.float32))
    m1 = Q[:, -1]
    gate = [float(abs(float(np.dot(d["eta"], m1)))) for d in docs]
    return dict(m1_share=float(w[-1] / max(w.sum(), 1e-30)),
                cos_m_perdoc_mean=dict(median=float(np.median(gate)),
                                       frac_gt_09=float(np.mean([g > 0.9 for g in gate]))))

# ---------- Δ-кривая ----------

def lag_curve(docs, null=True):
    out = {}
    for k in LAGS:
        real, nl = [], []
        for d in docs:
            Vn = d["V"][d["ok"]]
            Vn = Vn / np.maximum(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-12)
            real.append(np.median([float(Vn[i + k] @ Vn[i]) for i in range(len(Vn) - k)]))
            if null:
                W = shuffle_seq(Vn)
                nl.append(np.median([float(W[i + k] @ W[i]) for i in range(len(W) - k)]))
        real = np.array(real)
        e = dict(median=float(np.median(real)))
        if null:
            dl = real - np.array(nl)
            e["delta"] = bci(dl.tolist())
            e["null_median"] = float(np.median(nl))
        out[k] = e
    return out

# ---------- симулятор (стадия −1 и 3) ----------

def simulate(n_docs, alpha=0.07, beta=0.0, beta_w=0.0, gamma=0.0, sig=0.018,
             floor=0.004, w_mode=0.5, m_phi=0.95, seed=1):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(D); c /= np.linalg.norm(c)
    w = rng.standard_normal(D); w /= np.linalg.norm(w)
    docs = []
    for i in range(n_docs):
        U = np.zeros((T, D), np.float32)
        x = c + 0.3 * rng.standard_normal(D) / np.sqrt(D)
        U[0] = x / np.linalg.norm(x)
        eta = rng.standard_normal(D).astype(np.float32) * (floor / np.sqrt(D))
        v = np.zeros(D, np.float32)
        s = 0.0
        for t in range(T - 1):
            g = c - U[t]
            damp = -beta * v if beta else np.zeros(D, np.float32)
            if beta_w:
                damp = damp - beta_w * float(v @ w) * w
            s = m_phi * s + rng.normal(0, 0.002)
            mm = gamma * s * w if gamma else np.zeros(D, np.float32)
            v = (alpha * g + damp + mm + eta
                 + rng.standard_normal(D).astype(np.float32) * (sig / np.sqrt(D)))
            nx = U[t] + v
            U[t + 1] = nx / np.linalg.norm(nx)
        docs.append(dict(name=f"syn_{i}", dom="syn", si=i % 4, U=U.astype(np.float32)))
    return [derive(d) for d in docs]

# ---------- main ----------

def main():
    out = {"tz": "docs/E53b_v2.md", "controls": {}}
    # стадия −1
    cfgs = dict(
        iso=dict(alpha=0.07, beta=0.09, gamma=0.02),
        weak_w=dict(alpha=0.07, beta_w=0.09, gamma=0.02),
        strong_w=dict(alpha=0.07, beta_w=0.5, gamma=0.02),
        null=dict(alpha=0.0, beta=0.0, gamma=0.0))
    for name, cfg in cfgs.items():
        sd = simulate(24, seed=11 + len(name), **cfg)
        r = stage0(sd, name)
        out["controls"][name] = dict(truth=cfg, scalar_beta=r["scalar_beta"],
                                     scalar_beta_null=r["scalar_b_null"],
                                     cca_count_above_null=r["cca_count_above_null"],
                                     cca_top5=[round(x, 4) for x in r["cca_top15"][:5]],
                                     cca_tau=round(r["cca_null_tau"], 4))
        print("[-1]", name, out["controls"][name]["scalar_beta"], out["controls"][name]["cca_count_above_null"], flush=True)

    docs = [derive(d) for d in load_real()]
    print("[data] loaded", len(docs), flush=True)
    out["stage0"] = stage0(docs, "real")
    print("[0]", {k: v for k, v in out["stage0"].items() if k != "cca_top15"}, flush=True)
    out["stage1"] = stage1(docs, "real")
    print("[1]", out["stage1"], flush=True)
    out["slow_gate"] = slow_mode(docs)
    print("[m]", out["slow_gate"], flush=True)
    out["delta_curve"] = lag_curve(docs)
    for k, e in out["delta_curve"].items():
        print("[Δ]", k, {a: (round(b, 4) if isinstance(b, float) else [round(x, 4) for x in b])
                         for a, b in e.items()}, flush=True)
    out["stage24"] = stage24(docs)
    print("[2/4]", out["stage24"], flush=True)

    # стадия 3: Δ-кривая fitted-симуляции
    s1 = out["stage1"]
    fit = simulate(16, alpha=max(s1["a"], 0.0), beta=max(s1["b"], 0.0), gamma=0.02, seed=7)
    out["stage3_pred_curve"] = lag_curve(fit, null=False)
    print("[3] pred", {k: round(v["median"], 4) for k, v in out["stage3_pred_curve"].items()}, flush=True)

    json.dump(out, open("../results/exp53b_servo.json", "w"), indent=1, ensure_ascii=False)
    print("saved -> results/exp53b_servo.json")


if __name__ == "__main__":
    main()
