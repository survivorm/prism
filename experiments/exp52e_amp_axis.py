#!/usr/bin/env python
"""E-52e: вторая ось — амплитуда. radial_share кинетического разложения (E-40.5)
на трассах/вариантах E-52a, парно к τ_geo.

Каналы (prism.kinetic.kinetic_decomposition, r=16, замыкание ассертится внутри):
  radial_share = Σ E_radial / Σ ‖ΔJ‖²        (доля «смены синг. значений»)
  τ_amp²       = медиана (‖Δs‖₁₂+‖Δs‖₂₃−‖Δs₁₃)/‖Δs‖₁₂ на s_t=σ₁..₁₆(J_t)
                 (аддитивность амплитудной траектории — та же формула, что τ_geo)
Статика (контекст-оси): tr(JJᵀ), r_eff(энтропийный по σ²), R_can=σ₁²/tr.
Флаги честности: доля пар с ambiguous E_radial (σ-пермютэйшн), linearization_valid.
Варианты — зеркало e52a (rtn/flat/u_ns; модель-дири для 4B). BLAS=1 тред (урок E-40.5).
"""
# BLAS-треды ПИНЯТСЯ ДО numpy/torch (замер E-40.5: SVD [128,1024] 38мс@1 vs 1162мс@12)
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse, json, os as _os, sys  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = _os.path.dirname(_os.path.abspath(__file__))
for p in (_HERE, _os.path.join(_HERE, "..")):
    if p not in sys.path:
        sys.path.insert(0, p)
import exp47_common as C  # noqa: E402
from exp49_leaf_info import load_model, smoothed_us, boot_ci  # noqa: E402
from exp52a_perturb_axis import rtn  # noqa: E402
from prism.kinetic import kinetic_decomposition  # noqa: E402

TMAX = 96
W = 4
R = 16


def per_doc_metrics(A, us, eps):
    J = []
    for x in us:
        Jt, _, _ = C.J_analytic(A, x, eps)
        J.append(Jt.double().numpy())
    # ЕДИНЫЙ SVD на все кадры (аудит ревью E52 §6): раньше tau_amp2 пересчитывал
    # SVD, а kinetic брал кеш; теперь один список и там и там
    allsv = [np.linalg.svd(j, full_matrices=False) for j in J]
    num = den = 0.0
    amb = lin_bad = 0
    n_pair = 0
    for t in range(len(J) - 1):
        kd = kinetic_decomposition(J[t], J[t + 1], r=R, svds=(allsv[t], allsv[t + 1]))
        num += kd["E_radial"]; den += kd["norm_dJ_sq"]; n_pair += 1
        if "E_radial" in kd.get("ambiguous", []):
            amb += 1
        lin_bad += 0 if kd.get("linearization_valid", True) else 1
    # τ_amp² на s_[:R] (тот же SVD)
    S = [sv[1][:R] for sv in allsv]
    ad = []
    for t in range(len(S) - 2):
        a = float(np.linalg.norm(S[t] - S[t + 1])); b = float(np.linalg.norm(S[t + 1] - S[t + 2]))
        c = float(np.linalg.norm(S[t] - S[t + 2]))
        if a > 1e-12:
            ad.append((a + b - c) / a)
    # σ-перескоки: доля соседних пар, где разрыв σ_r-границы меньше шага —
    # τ_amp² на СОРТИРОВАННЫХ σ чувствителен к перестановке (ревью §3)
    jump = float(np.median([float(np.abs(S[t] - S[t + 1]).max()) for t in range(len(S) - 1)]))
    gap = float(np.median([float(np.median(np.diff(-np.sort(-allsv[t][1])))) for t in range(len(allsv))]))
    st = [sv[1] for sv in allsv]
    tr = [float((s ** 2).sum()) for s in st]
    reff = [float(np.exp(-(p := (s ** 2) / max((s ** 2).sum(), 1e-30)) *
                          np.log(np.maximum(p, 1e-30))).sum()) for s in st]
    rcan = [float((s ** 2)[0] / max((s ** 2).sum(), 1e-30)) for s in st]
    return {"radial_share": num / max(den, 1e-30), "n_pairs": n_pair,
            "amb_frac": amb / max(n_pair, 1), "linbad_frac": lin_bad / max(n_pair, 1),
            "jump_gap_ratio": jump / max(gap, 1e-30),
            "tau_amp2": float(np.median(ad)) if ad else None,
            "tr_med": float(np.median(tr)), "reff_med": float(np.median(reff)),
            "rcan_med": float(np.median(rcan))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("qwen", "lfm3"), default="qwen")
    ap.add_argument("--n-docs", type=int, default=16)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--npz", default=None); ap.add_argument("--docs-json", default=None)
    ap.add_argument("--families", default="rtn")   # rtn,flat,u_ns
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = _os.path.join(_HERE, "..")
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
        print("[model-dir] A пересобрана", flush=True)
    docs = json.load(open(a.docs_json or _os.path.join(root, jp)))["docs"][: a.n_docs]
    z = np.load(npz, allow_pickle=True)
    base = {}
    for d in docs:
        k = f"{d['dom']}_{d['si']}"
        u = torch.from_numpy(z[k + "_u"].astype(np.float64))[:TMAX]
        hn = np.asarray(z[k].item()["hnorm"], dtype=float)[:TMAX]
        base[k] = (u, hn)

    variants = [("base", A0, None)]
    if "flat" in fams and a.model == "qwen" and not a.model_dir:
        fl = torch.load(_os.path.join(root, "results", "exp45_flat_rows.pt"),
                        map_location="cpu", weights_only=False)
        for nm in ("flat_l0.01", "flat_l0.1", "flat_noleash"):
            variants.append((nm, C.head_M(fl[nm].float(), g), None))
    if "rtn" in fams:
        variants += [(f"rtn{b}", rtn(A0, b), None) for b in (8, 4, 2)]
    res = {}
    for nm, A, _ in variants:
        res[nm] = {}
        for k, (u, hn) in base.items():
            res[nm][k] = per_doc_metrics(A, smoothed_us(u, hn, W), eps)
        m = [res[nm][k]["radial_share"] for k in res[nm]]
        print(nm, "radial_share med", round(float(np.median(m)), 4), flush=True)
    if "u_ns" in fams:
        import hashlib
        for s in (0.01, 0.02, 0.04):
            sd = int.from_bytes(hashlib.sha256(f"u{s}".encode()).digest()[:8], "little") % (1 << 62)
            xi = torch.randn((1, A0.shape[1]), dtype=torch.float64,
                             generator=torch.Generator().manual_seed(sd))
            nm = f"u_ns{s}"
            res[nm] = {}
            for k, (u, hn) in base.items():
                un = u + s * u.norm(dim=1, keepdim=True) * xi.to(u.dtype)
                res[nm][k] = per_doc_metrics(A0, smoothed_us(un, hn, W), eps)
            print(nm, "done", round(float(np.median([v["radial_share"] for v in res[nm].values()])), 4), flush=True)

    summ = {}
    b0 = res["base"]
    for nm in res:
        row = {f"{s}_med": float(np.median([v[s] for v in res[nm].values()]))
               for s in ("radial_share", "tau_amp2", "tr_med", "reff_med", "rcan_med")}
        if nm != "base":
            d = [res[nm][k]["radial_share"] - b0[k]["radial_share"] for k in b0]
            row["d_radial_share"] = [float(np.median(d)), boot_ci(d)]
            d2 = [(res[nm][k]["tau_amp2"] or 0) - (b0[k]["tau_amp2"] or 0) for k in b0]
            row["d_tau_amp2"] = [float(np.median(d2)), boot_ci(d2)]
        row["amb_frac"] = float(np.mean([v["amb_frac"] for v in res[nm].values()]))
        row["linbad_frac"] = float(np.mean([v["linbad_frac"] for v in res[nm].values()]))
        summ[nm] = row
    out = {"model": a.model, "model_dir": a.model_dir, "T": TMAX, "W": W, "r": R,
           "summary": summ, "per_doc": res}
    json.dump(out, open(a.out, "w"), indent=1, allow_nan=False)
    print("WROTE", a.out)


if __name__ == "__main__":
    main()
