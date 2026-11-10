#!/usr/bin/env python
"""E-52j: порог дозы кванта (ревью E52h §6): rtn3 между rtn4 и rtn2.
tail16 + alpha на тех же функциях, что e52h; base/rtn8/rtn4/rtn2 пересчитываются
в том же прогоне (одна опора), 3-bit — новый уровень lev=3.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import exp47_common as C  # noqa: E402
from exp49_leaf_info import load_model, smoothed_us  # noqa: E402
from exp52a_perturb_axis import rtn  # noqa: E402
from exp52h_health_candidates import sigs, tail_fr, powerlaw_gate  # noqa: E402

TMAX, W = 96, 4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--npz", default=None); ap.add_argument("--docs-json", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    A0, U, eps, npz, jp = load_model(argparse.Namespace(model="qwen"))
    g = load_model.last_g
    if a.npz: npz = a.npz
    if a.model_dir:
        from exp52c_symptom import head_from_ckpt
        z0 = np.load(npz, allow_pickle=True)
        head, normw = head_from_ckpt(a.model_dir, torch.as_tensor(U))
        if "g_norm" in z0.files:
            g = torch.from_numpy(z0["g_norm"].astype(np.float64)); eps = float(z0["eps"])
        else:
            g = normw.double() if float(normw.abs().mean()) > 0.5 else 1.0 + normw.double()
        A0 = C.head_M(head[torch.as_tensor(U)], g)
    docs = json.load(open(a.docs_json or "../results/exp47b_gen_kinematics.json"))["docs"]
    z = np.load(npz, allow_pickle=True)
    out = {}
    variants = [("base", A0)] + [(f"rtn{b}", rtn(A0, b)) for b in (8, 4, 3, 2)]
    tails = {nm: [] for nm, _ in variants}; smeds = {nm: [] for nm, _ in variants}
    for d in docs:
        k = f"{d['dom']}_{d['si']}"
        u = torch.from_numpy(z[k + "_u"].astype(np.float64))[:TMAX]
        hn = np.asarray(z[k].item()["hnorm"], dtype=float)[:TMAX]
        us = smoothed_us(u, hn, W)
        for nm, A in variants:
            S = sigs(A, us, eps, stride=1)
            tails[nm].append(tail_fr(S)["16"])
            smeds[nm].append(S.mean(0))
        print(k, "done", flush=True)
    for nm, _ in variants:
        out[nm] = {"tail16": float(np.median(tails[nm])),
                   "alpha": powerlaw_gate(np.median(smeds[nm], axis=0))["alpha"]}
        print(nm, "tail16", round(out[nm]["tail16"], 4), "alpha", round(out[nm]["alpha"], 3), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
