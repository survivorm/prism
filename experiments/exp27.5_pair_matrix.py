#!/usr/bin/env python
"""Exp 27.5 — full between-sample pair matrix C_b(i, j).

Post-processing only: reads exp27_jcache.pt, computes the complete
496-pair gauge-free cosine matrix at S=64, K=64, pos=0, and
sanity-checks the within-domain means against the kernel JSON C_b_mean.
No GPU, no model.

Outputs:
  results/exp27_pair_matrix_S64K64p0.json   (stats + full 32x32 per layer)
  results/exp27_pair_matrix_S64K64p0.pt     (raw matrices, for plotting)

Usage:
  python experiments/exp27.5_pair_matrix.py [--ckpt PATH] [--results-dir PATH]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism.jacobian import softmax_jacobian  # noqa: E402

S, K, POS = 64, 64, 0
LAYER_SET = [6, 12, 18, 19, 22, 23]
DOMAINS = ["wiki", "code", "chat", "math"]
N_PER_DOMAIN = 8


def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
    u, v = u.flatten(), v.flatten()
    den = u.norm() * v.norm()
    if den.item() < 1e-12:
        return float("nan")
    return float((u @ v).item()) / den.item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/exp27_jcache.pt")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    JR, H, P128 = ckpt["JR"], ckpt["H"], ckpt["P128"]
    pts = [(d, si) for d in DOMAINS for si in range(N_PER_DOMAIN)]
    rd = Path(args.results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    kernel_json = rd / "exp27_noise_floor.json"
    rjson = json.loads(kernel_json.read_text()) if kernel_json.exists() else None

    import hashlib
    out_json: dict = {
        "params": {
            "S": S, "K": K, "pos": POS, "layers": LAYER_SET,
            "n_pairs": len(pts) * (len(pts) - 1) // 2,
            "domains": DOMAINS, "n_per_domain": N_PER_DOMAIN,
            "metric": "cosine(softmax_jacobian(J_raw[:K], p[:K]) / delta_ij)",
            "gauge": "pos-pk at pos 0 (p renormalized on K-subset)",
            "ckpt": str(args.ckpt),
            "ckpt_sha256": hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest()[:16],
            "ckpt_U128_sha": ckpt.get("U128_sha", None),
            "ckpt_model_sha": ckpt.get("model_sha", None),
        },
        "layers": {},
    }
    outs_pt: dict = {}
    sanity_ok = True

    for l in LAYER_SET:
        Jg, hs = [], []
        for (di, si) in pts:
            key = (di, si, l, S, POS)
            Jk = JR[key][:K]
            pkv = P128[key][:K]
            pkv = pkv / pkv.sum()
            Jg.append(softmax_jacobian(Jk, pkv).cpu())
            hs.append(H[key])
        n = len(pts)
        M = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                d = (hs[i] - hs[j]).norm().item()
                M[i, j] = M[j, i] = _cos(Jg[i] / d, Jg[j] / d)
        v = M[~np.isnan(M)]
        same = [M[i, j] for i in range(n) for j in range(i + 1, n)
                if pts[i][0] == pts[j][0]]
        cross = [M[i, j] for i in range(n) for j in range(i + 1, n)
                 if pts[i][0] != pts[j][0]]
        per_domain: dict = {}
        for d in DOMAINS:
            idx = [i for i in range(n) if pts[i][0] == d]
            vals = [M[i, j] for i in idx for j in idx if i < j]
            per_domain[d] = {"mean": float(np.mean(vals)),
                             "min": float(np.min(vals)),
                             "max": float(np.max(vals)),
                             "n_pairs": len(vals)}
        blk = {
            "min": float(v.min()), "p5": float(np.percentile(v, 5)),
            "mean": float(v.mean()), "p50": float(np.percentile(v, 50)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max()),
            "frac_neg": float((v < 0).mean()),
            "same_domain_mean": float(np.mean(same)),
            "cross_domain_mean": float(np.mean(cross)),
            "per_domain": per_domain,
            "samples": [f"{d}_{si}" for (d, si) in pts],
            "matrix": M.tolist(),
        }
        out_json["layers"][str(l)] = blk
        outs_pt[str(l)] = M
        line = (f"L{l:2d}: min={v.min():+.3f} p5={np.percentile(v, 5):+.3f} "
                f"mean={v.mean():+.3f} p95={np.percentile(v, 95):+.3f} "
                f"max={v.max():+.3f} | same {np.mean(same):+.3f} "
                f"cross {np.mean(cross):+.3f}")
        if rjson is not None:
            jb = rjson["layers"][str(l)]["configs"][f"S{S}_K{K}_p0"]["C_b_mean"]
            ok = abs(float(np.mean(same)) - jb) < 1e-4
            sanity_ok = sanity_ok and ok
            line += f" | JSON C_b_mean {jb:+.4f} match={ok}"
        print(line)

    jout = rd / f"exp27_pair_matrix_S{S}K{K}p{POS}.json"
    jout.write_text(json.dumps(out_json, indent=1))
    tout = rd / f"exp27_pair_matrix_S{S}K{K}p{POS}.pt"
    torch.save(outs_pt, tout)
    print(f"saved {jout.name} + {tout.name}")
    if rjson is not None:
        assert sanity_ok, "within-domain means do not match kernel C_b_mean"
        print("SANITY OK: within-domain means == kernel C_b_mean (all 6 layers)")
    else:
        print("WARN: kernel JSON not found, sanity check skipped")


if __name__ == "__main__":
    main()
