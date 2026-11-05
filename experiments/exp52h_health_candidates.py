#!/usr/bin/env python
"""E-52h: кандидаты health по программe архитектора (E52_review, финальная секция).

G  sens(ε): локальная чувствительность спектра K СВОЕЙ точке весов (абсолютная,
   без reference): sens_v(ε) = median_t ‖Δσ_top64‖/ε при δA = ε·rms(A)·ξ (ξ фикс.
   сид). ε ∈ {1e-3,1e-2,1e-1}: const=линейный режим, растёт/падает — нелинейность.
D  ACF ρ(1),ρ(2),ρ(4),ρ(8),ρ(16) траектории Σ(t) (по координатам, среднее),
   Hurst R/S; AR(1) ρ(1) — baseline. Тест ревьюера: ρ(1) base vs rtn2 на 4B —
   выше=setка (но E-52g это уже опроверг), ниже=реальное падение персистентности.
E  tail(r) = Σ_{k>r}σ²/Σσ² для r=8,16,32,64; Δtail под rtn — хвост или топ.
F  power-law: log σ = −α·log k + c по медианному спектру; ГЕЙТ R²>0.9, иначе F мёртв.

B (стратификация по энтропии) — не тестировается без logits (по ревьюеру);
C — нужны multi-layer traces.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, os.path.join(_HERE, "..")):
    if p not in sys.path:
        sys.path.insert(0, p)
import exp47_common as C  # noqa: E402
from exp49_leaf_info import load_model, smoothed_us  # noqa: E402
from exp52a_perturb_axis import rtn  # noqa: E402

TMAX, W, RT, TSTRIDE = 96, 4, 64, 4
EPS_G = (1e-3, 1e-2, 1e-1)
R_TAIL = (8, 16, 32, 64)


def sigs(A, us, eps, stride=TSTRIDE):
    out = []
    for i, x in enumerate(us):
        if i % stride:
            continue
        J, _, _ = C.J_analytic(A, x, eps)
        out.append(np.linalg.svd(J.double().numpy(), full_matrices=False)[1])
    return np.array(out)                       # [T', Dmin]


def sens_of(A, us, eps):
    # ξ ФИКСИРОВАН (сид 52) для всех вариантов: Δσ сравниваются при одном
    # направлении возмущения (paired). Brittleness-оговорка (ревью): рост sens
    # при ε=1e-1 у rtn = перещёлкивание сетки, не sensitivity; линейный режим =
    # sens(1e-3)≈sens(1e-2), скачок на 1e-1 читается отдельно.
    xi = torch.randn(A.shape, dtype=A.dtype,
                     generator=torch.Generator().manual_seed(52))
    rms = float(A.pow(2).mean().sqrt())
    res = {}
    for e in EPS_G:
        Ap = A + e * rms * xi
        d = []
        for i, x in enumerate(us):
            if i % TSTRIDE:
                continue
            J0, _, _ = C.J_analytic(A, x, eps)
            J1, _, _ = C.J_analytic(Ap, x, eps)
            s0 = np.linalg.svd(J0.double().numpy(), full_matrices=False)[1][:RT]
            s1 = np.linalg.svd(J1.double().numpy(), full_matrices=False)[1][:RT]
            d.append(float(np.linalg.norm(s1 - s0)) / e)
        res[str(e)] = float(np.median(d))
    return res


def acf_hurst(S):                                             # S: [T,R]
    X = S - S.mean(0)
    den = (X * X).sum(0).clip(min=1e-30)
    out = {}
    for L in (1, 2, 4, 8, 16):
        if S.shape[0] > L:
            out[f"rho{L}"] = float(((X[:-L] * X[L:]).sum(0) / den).mean())
    # Hurst R/S по каждой координате, медиана
    hs = []
    for j in range(S.shape[1]):
        y = np.cumsum(S[:, j] - S[:, j].mean())
        R = y.max() - y.min()
        Sd = S[:, j].std()
        if Sd > 0 and R > 0:
            hs.append(float(np.log(R / Sd) / np.log(len(S))))
    out["hurst"] = float(np.median(hs)) if hs else None
    return out


def tail_fr(S):
    E = (S ** 2)
    tot = E.sum(1).clip(min=1e-30)
    return {str(r): float(np.median(E[:, r:].sum(1) / tot)) for r in R_TAIL}


def powerlaw_gate(Smed):
    k = np.arange(3, 35)   # σ₃..σ₃₄ ↔ k=3..34 (off-by-one fix)
    y = np.log(np.maximum(Smed[2:34], 1e-12))
    a = np.polyfit(np.log(k), y, 1)
    pred = np.polyval(a, np.log(k))
    ss = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return {"alpha": float(a[0]), "r2": float(ss), "alive": bool(ss > 0.9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "lfm3"), default="qwen")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--npz", default=None); ap.add_argument("--docs-json", default=None)
    ap.add_argument("--families", default="rtn")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fams = set(a.families.split(","))
    ns = argparse.Namespace(model=a.model)
    A0, U, eps, npz, jp = load_model(ns)
    g = load_model.last_g
    if a.npz:
        npz = a.npz
    if a.model_dir:
        from exp52c_symptom import head_from_ckpt
        z0 = np.load(npz, allow_pickle=True)
        head, normw = head_from_ckpt(a.model_dir, torch.as_tensor(U))
        if "g_norm" in z0.files:
            g = torch.from_numpy(z0["g_norm"].astype(np.float64)); eps = float(z0["eps"])
        else:
            g = normw.double() if float(normw.abs().mean()) > 0.5 else 1.0 + normw.double()
        A0 = C.head_M(head[torch.as_tensor(U)], g)
    docs = json.load(open(a.docs_json or "../results/exp47b_gen_kinematics.json"))["docs"][: a.n_docs]
    z = np.load(npz, allow_pickle=True)
    base_us = {}
    for d in docs:
        k = f"{d['dom']}_{d['si']}"
        u = torch.from_numpy(z[k + "_u"].astype(np.float64))[:TMAX]
        hn = np.asarray(z[k].item()["hnorm"], dtype=float)[:TMAX]
        base_us[k] = smoothed_us(u, hn, W)

    variants = [("base", A0)]
    if "flat" in fams and a.model == "qwen" and not a.model_dir:
        fl = torch.load("../results/exp45_flat_rows.pt", map_location="cpu", weights_only=False)
        for nm in ("flat_l0.1", "flat_noleash"):
            variants.append((nm, C.head_M(fl[nm].float(), g)))
    if "rtn" in fams:
        variants += [(f"rtn{b}", rtn(A0, b)) for b in (8, 4, 2)]

    res = {}
    for nm, A in variants:
        Gv, Dv, Ev, Smed = [], [], [], []
        for k, us in base_us.items():
            S = sigs(A, us, eps, stride=1)   # ACF/tail/спектр — ПОЛНАЯ трасса
            Gv.append(sens_of(A, us, eps))   # G на TSTRIDE (дорогая SVD×3ε)
            Dv.append(acf_hurst(S))
            Ev.append(tail_fr(S))
            Smed.append(S.mean(0))
        med = lambda lst, key: float(np.median([x[key] for x in lst]))  # noqa: E731
        res[nm] = {"G_sens": {str(e): float(np.median([g_[str(e)] for g_ in Gv])) for e in EPS_G},
                   "D_acf": {kk: med(Dv, kk) for kk in Dv[0]},
                   "E_tail": {str(r): med(Ev, str(r)) for r in R_TAIL},
                   "F_powerlaw": powerlaw_gate(np.median(Smed, axis=0))}
        print(nm, "G:", {e: round(v, 4) for e, v in res[nm]["G_sens"].items()},
              "rho1:", round(res[nm]["D_acf"]["rho1"], 3),
              "H:", round(res[nm]["D_acf"]["hurst"], 3),
              "tail16:", round(res[nm]["E_tail"]["16"], 4),
              "F_r2:", round(res[nm]["F_powerlaw"]["r2"], 2), flush=True)
    json.dump(res, open(a.out, "w"), indent=1, allow_nan=False)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
