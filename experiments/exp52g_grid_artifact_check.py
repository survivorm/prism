#!/usr/bin/env python
"""E-52g: прямой тест «сеточного» объяснения 4B rtn2 (ревью E52, опц. пункт).

Доля шагов σ(t), где max_i|σ_i(t)−σ_i(t+1)| ≈ 0 на уровне сетки: если под rtn2
она велика (base — нет), «преимущественно артефакт» подтверждается; иначе —
переформулировать в «bounded by».
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

_HERE2 = os.path.dirname(os.path.abspath(__file__))
from exp49_leaf_info import load_model, smoothed_us  # noqa: E402
from exp52a_perturb_axis import rtn  # noqa: E402

TMAX, W, R = 96, 4, 16


def sig_series(A, us, eps):
    out = []
    for x in us:
        J, _, _ = C.J_analytic(A, x, eps)
        out.append(np.linalg.svd(J.double().numpy(), full_matrices=False)[1][:R])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=8)
    ap.add_argument("--model-dir", default=os.path.join(_HERE2, "..", "models", "Qwen3.5-4B"))
    ap.add_argument("--npz", default=os.path.join(_HERE2, "..", "results", "e52_4b", "exp47b_gen_kinematics.npz"))
    ap.add_argument("--docs-json", default=os.path.join(_HERE2, "..", "results", "e52_4b", "exp47b_gen_kinematics.json"))
    a = ap.parse_args()
    ns = argparse.Namespace(model="qwen")
    A0, U, eps, npz, jp = load_model(ns)
    from exp52c_symptom import head_from_ckpt
    z0 = np.load(a.npz, allow_pickle=True)
    head, normw = head_from_ckpt(a.model_dir, torch.as_tensor(U))
    if "g_norm" in z0.files:
        g = torch.from_numpy(z0["g_norm"].astype(np.float64)); eps = float(z0["eps"])
    else:
        g = normw.double() if float(normw.abs().mean()) > 0.5 else 1.0 + normw.double()
    A0 = C.head_M(head[torch.as_tensor(U)], g)
    A2 = rtn(A0, 2)
    # U128 из кэша 0.8B: для model-dir-моделей обязателен общий словарь (ревью §6)
    from transformers import AutoTokenizer  # type: ignore
    tk0 = AutoTokenizer.from_pretrained(os.environ.get("PRISM_QWEN_DIR", "/data/Qwen/Qwen3.5-0.8B"))
    tk1 = AutoTokenizer.from_pretrained(a.model_dir)
    assert tk0.vocab_size == tk1.vocab_size, "словари разные — U128 невалиден"
    assert all(tk0.convert_ids_to_tokens(int(i)) == tk1.convert_ids_to_tokens(int(i))
               for i in list(U)[:64]), "U128 токенизуется в 4B иначе"
    print("[U-check] словарь и первые 64 id U128 совпадают", flush=True)
    # разрешение сетки A: rtn2 симметричный lev=1 ⇒ A ∈ {−sc_row, 0, +sc_row}
    docs = json.load(open(a.docs_json))["docs"][: a.n_docs]
    z = np.load(a.npz, allow_pickle=True)
    res = {"base": [], "rtn2": []}
    exact0 = {"base": [], "rtn2": []}
    for d in docs:
        k = f"{d['dom']}_{d['si']}"
        u = torch.from_numpy(z[k + "_u"].astype(np.float64))[:TMAX]
        hn = np.asarray(z[k].item()["hnorm"], dtype=float)[:TMAX]
        us = smoothed_us(u, hn, W)
        for nm, A in (("base", A0), ("rtn2", A2)):
            S = sig_series(A, us, eps)
            step = np.abs(np.diff(S, axis=0)).max(1)
            scale = float(np.median(S[:, 0]))
            res[nm].append(float((step < 1e-3 * scale).mean()))
            exact0[nm].append(float((step <= 1e-12).mean()))
        print(k, "base zero-frac %.3f rtn2 %.3f" % (res["base"][-1], res["rtn2"][-1]), flush=True)
    print("\nИТОГ доля «нулевых» шагов (max|Δσ|<1e-3·σ̄1): base med %.3f, rtn2 med %.3f"
          % (np.median(res["base"]), np.median(res["rtn2"])))
    print("точные нули: base %.3f, rtn2 %.3f" % (np.median(exact0["base"]), np.median(exact0["rtn2"])))
    json.dump({"zero_frac": res, "exact_zero": exact0},
              open("../results/exp52g_grid_check_4b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
