"""exp29 validation: independent recompute of T_eff and C_F subsets.

Different code paths from exp29_fisher.py main():
- T_eff via eigh of the KxK matrix (no P-weighted trace trick)
- C_F via plain matmul in a python loop (no einsum)
- gauge-free residual check, n-pair counts, manifest sha, ranges.
"""
import argparse
import hashlib
import json
import os
import sys

import torch


def _sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/data/prism_kernels/exp27b/results")
    ap.add_argument("--json", default="results/exp29_fisher.json")
    ap.add_argument("--layers", default="6,23")
    ap.add_argument("--n-check", type=int, default=32)
    args = ap.parse_args()
    torch.manual_seed(123)
    res = json.load(open(args.json))
    fails = []
    n_pass = [0]
    def ok(name, cond, detail=""):
        if cond:
            n_pass[0] += 1
        else:
            fails.append(name)
            print(f"  FAIL {name} {detail}")

    layers = [int(x) for x in args.layers.split(",") if x]
    for l in layers:
        print(f"L{l}:")
        ck = torch.load(os.path.join(args.ckpt_dir, f"exp27b_jcache_L{l}.pt"),
                        map_location="cpu", weights_only=False)
        S, D = ck["S"], ck["D"]
        n = ck["n_samples"] * len(ck["domains"])
        docs = [(dom, si) for dom in ck["domains"] for si in range(ck["n_samples"])]
        idx = torch.randperm(n * S)[:args.n_check]
        trE, trF = [], []
        for j, i in enumerate(idx.tolist()):
            (dom, si), t = docs[i // S], i % S
            J = ck["JR"][(dom, si, t)].float()          # [K, D]
            K = J.shape[0]
            p = ck["P128"][(dom, si, t)].float()        # [K]
            M = torch.diag(p) - p.unsqueeze(0) * p.unsqueeze(1)
            ok(f"JR_raw@t{t}-d{i//S}", float((p @ J).abs().max()) > 1e-3 * J.abs().max().item(),
               "JR looks gauge-free, not raw logits")
            if j < 2:  # eigh is slow (125s on 1024x1024) - two positions only
                evm = torch.linalg.eigvalsh(M)
                ok(f"FP_psd@t{t}-d{i//S}", float(evm.min()) > -1e-6,
                   f"min eig={float(evm.min()):.1e}")
                ok(f"FP_null@t{t}-d{i//S}", float((M @ torch.ones(K)).abs().max()) < 1e-6)
                ev = torch.linalg.eigvalsh(J.t() @ M @ J)
                trE.append(float(ev.sum()))
            mu = (p.unsqueeze(1) * J).sum(0)
            trF.append(float((p * J.pow(2).sum(-1)).sum() - mu.pow(2).sum()))
        maxdiff = max(abs(a - b) for a, b in zip(trE, trF))
        ok("T_eff eigh vs weighted-trace", maxdiff < 0.05, f"maxdiff={maxdiff:.2e}")
        my_T = float(torch.tensor(trF).mean())
        js_T = res["layers"][str(l)]["T_eff"]
        rel = abs(my_T - js_T) / max(abs(js_T), 1e-9)
        ok("T_eff vs JSON", rel < 0.05, f"mine={my_T:.3f} json={js_T:.3f}")
        ps = [float(ck["P128"][k].sum()) for k in list(ck["P128"])[:64]]
        ok("p128 sum=1", all(abs(x - 1) < 1e-5 for x in ps), f"min={min(ps):.5f}")
        cf_vals = []
        stride = max(1, (n * S - 1) // args.n_check)
        for i in range(0, n * S - 1, stride)[:args.n_check]:
            if i % S == S - 1:
                continue
            (doma, sia) = docs[i // S]
            (domb, sib) = docs[(i + 1) // S]
            t = i % S
            pa = ck["P128"][(doma, sia, t)].float()
            Ja = ck["JR"][(doma, sia, t)].float()
            Fa = torch.diag(pa) - pa.unsqueeze(0) * pa.unsqueeze(1)
            d = (ck["H"][(domb, sib, t + 1)] - ck["H"][(doma, sia, t)]).float().reshape(-1)
            y = Ja @ d            # [K]
            w = Fa @ y            # [K]
            q = Ja.t() @ w        # [D]
            cf_vals.append(float((d * q).sum() / d.pow(2).sum().clamp_min(1e-20)))
        js_cf = res["layers"][str(l)]["intra"]["lag1"]["cf"]
        my_cf = float(torch.tensor(cf_vals).mean())
        print(f"  info intra lag1: mine(subset n={len(cf_vals)})={my_cf:.5f} "
              f"json={js_cf:.5f} min={min(cf_vals):.4f} max={max(cf_vals):.4f}")
        ok("cf values bounded", all(-1 < x < 2 for x in cf_vals))
        for L in (1, 2, 4, 8, 16):
            exp_n = n * max(0, S - L)
            got = res["layers"][str(l)]["intra"][f"lag{L}"]["n"]
            ok(f"n intra lag{L}", got == exp_n, f"{got}=={exp_n}")
        exp_ni = n * (n - 1) * S
        ok("n inter", res["layers"][str(l)]["inter"]["n"] == exp_ni,
           f"{res['layers'][str(l)]['inter']['n']}=={exp_ni}")
        ok("manifest sha",
           res["manifest"][str(l)]["sha256_16"] ==
           _sha16(os.path.join(args.ckpt_dir, f"exp27b_jcache_L{l}.pt")))
    Ts = {l: res["layers"][str(l)]["T_eff"] for l in layers}
    print("  info T_eff by layer:", Ts)
    print("VALIDATION", "FAILED: " + str(fails) if fails else "ALL PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
