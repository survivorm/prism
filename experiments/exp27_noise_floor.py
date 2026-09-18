"""Exp 27: Noise-floor calibration and robustness sweep for J^(p)(x).

docs/Exp27.md. Closes the five holes of exp01 v6:
  1. noise floor 1/sqrt(D) is 3-8x above the effective-dimension floor;
     now F = 1/sqrt(r_eff*(K+D-r_eff)), r_eff = entropic rank (median per layer).
  2. all samples truncated at S=64 — S-sweep {32,64,128,256} on prefixes of
     one pool (256-token texts; left-aligned, so same prefix at all S).
  3. positions only {0,1,2} — position-sweep {0,1,2,4,8,16} at S=64.
  4. K=64 untested — K-sweep {32,64,128}, nested from one U_128.
  5. layers 6/12/18/22 are GDN (linear_attention) — add 19, 23 full-attention
     (23 = last layer, J ~ W lm_head = control).

Config per (sample, layer):
  S=64: K in {32,64,128} x pos in {0,1,2,4,8,16}          (18 J at U_128)
  S in {32,128,256}: K=64 x pos in {0,1,2}                (9 J)
  S=128 extra: pos=64 (truncation test 6.2: vs S=64 pos=0)
~28 J/sample/layer x 32 samples x 6 layers = ~5.4k calls, ~2.5-3h T4.
J computed once per (sample, layer, S, pos) at U_128; K subsets = row slices.

Outputs: results/exp27_noise_floor.json (+manifest: git/model/token-set hashes);
half-split bootstrap + PNG + findings = local post (no GPU).
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from prism.calibration import load_calibration_data
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian

MIN_BIN_PAIRS = 5
N_BINS = 14

def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
    num = (u * v).sum()
    den = u.pow(2).sum().sqrt() * v.pow(2).sum().sqrt()
    if den.item() < 1e-12:
        return float("nan")
    return (num / den).item()

def _ranks(J: torch.Tensor) -> tuple[float, float, float]:
    """(entropic, stable, participation-ratio) rank of J [K, D]."""
    s2 = torch.linalg.svdvals(J).pow(2)
    tot = s2.sum().clamp_min(1e-30)
    p = s2 / tot
    p = p[p > 1e-12]
    r_ent = float(torch.exp(-(p * torch.log(p)).sum()))
    r_stable = float(tot / s2[0])
    r_pr = float((s2.sum() ** 2) / s2.pow(2).sum())
    return r_ent, r_stable, r_pr

def _load_pool(model_name: str, domains: list[str], n: int, L: int = 256) -> dict:
    """Per domain: n texts of exactly L tokens. ONE stream pass (min_length=1):
    long texts kept, shorts concatenated (left-aligned prefix property holds).
    Retries on rate-limited (near-empty) streams."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    per: dict[str, list[dict]] = {}
    for dom in domains:
        big: list[dict] = []
        for att in range(5):
            big = load_calibration_data(
                model_name, max_length=L, min_length=1, n_samples=n * 32, domain=dom,
            )
            if len(big) >= n * 4:
                break
            print(f"  pool {dom}: stream gave {len(big)} < {n * 4}, retry {att + 1}...")
            time.sleep(10 * (att + 1))
        data = [s for s in big if s["input_ids"].shape[1] >= L]
        if len(data) < n:  # intrinsically short (gsm8k): concatenate
            buf = ""
            out = []
            for t in big:
                buf += " " + tok.decode(t["input_ids"][0], skip_special_tokens=True).strip()
                words = buf.split(" ")
                while len(out) < n and len(words) >= 64:
                    ids = tok(" ".join(words[:320]), max_length=L, truncation=True, return_tensors="pt")
                    if ids["input_ids"].shape[1] >= L:
                        out.append(ids)
                        words = words[200:]  # consume so next sample is mostly fresh
                        continue
                    break
            data = (data + out)[:n]
        else:
            data = data[:n]
        print(f"  pool {dom}: {len(data)} @ {L}")
        per[dom] = data
    return per

def _forward_prefix(m, s: dict, S: int, layers: list[int], positions: list[int]) -> dict:
    """One forward on the S-prefix: logits + hs at `positions` (from the end)."""
    ids = s["input_ids"][:, :S].to(m.device)
    attn = s["attention_mask"][:, :S].to(m.device)
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    logits = out.logits[0].float().cpu()
    hs: dict[int, dict[int, torch.Tensor]] = {}
    for l in layers:
        hs[l] = {
            p: out.hidden_states[l + 1][0, S - 1 - p, :].float().cpu()
            for p in positions
            if p < S
        }
    return {"logits": logits, "hs": hs}

def _j_configs() -> list[tuple[int, int, int]]:
    """(S, K_of_128, pos) — J computed at U_128; K subsets = row slices."""
    cfgs = [(64, 128, p) for p in (0, 1, 2, 4, 8, 16)]
    for S in (32, 128, 256):
        cfgs += [(S, 128, p) for p in (0, 1, 2)]
    cfgs += [(128, 128, 64), (128, 128, 65)]  # truncation test 6.2 (both positions)
    return cfgs

def main(
    model="Qwen/Qwen3.5-0.8B",
    device="auto",
    results_dir="results",
    layers=(6, 12, 18, 19, 22, 23),
    n_samples=8,
    domains=("wiki", "math", "code", "chat"),
    make_png: bool = False,
):
    torch.manual_seed(42)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM

    print(f"Loading {model}...")
    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    m.eval()
    layers = list(layers)
    domains = list(domains)
    D = m.config.hidden_size
    V = m.config.vocab_size

    print(f"Loading pool: {n_samples} x {len(domains)} @ 256 tokens...")
    pool = _load_pool(model, domains, n_samples, 256)
    for dom in domains:
        assert len(pool[dom]) == n_samples, f"pool {dom}: {len(pool[dom])} < {n_samples}"
    n = n_samples * len(domains)  # 32
    print(f"Pool OK: {n} samples")

    # Nested token set U_128: top-128 by final-logit frequency at S=256.
    counts: dict[int, int] = {}
    for dom in domains:
        for si, s in enumerate(pool[dom]):
            with torch.no_grad():
                out = m(input_ids=s["input_ids"].to(m.device),
                        attention_mask=s["attention_mask"].to(m.device))
            p = torch.softmax(out.logits[0, -1].float(), dim=-1)
            for t in p.topk(128).indices.tolist():
                counts[t] = counts.get(t, 0) + 1
    U128 = sorted([t for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:128]])
    print(f"U_128 built ({len(U128)} tokens)")

    cfgs = _j_configs()
    positions_all = sorted({p for _, _, p in cfgs})
    Ss = sorted({S for S, _, _ in cfgs})
    print(f"J configs: {len(cfgs)} per sample/layer; positions {positions_all}; S {Ss}")

    # --- checkpoint: resume J-cache from a prior run (same pool/cfgs/U128) ---
    pool_fp = hashlib.sha256(
        b"".join(pool[d][0]["input_ids"].flatten().numpy().tobytes()
                 for d in domains)).hexdigest()[:12]
    ckpt = os.path.join(results_dir, "exp27_jcache.pt")
    do_j = True
    if os.path.exists(ckpt):
        try:
            ck = torch.load(ckpt, map_location="cpu")
            ok = (ck["cfgs"] == cfgs and ck["U128"] == U128
                  and ck["pool_fp"] == pool_fp and ck["layers"] == layers)
            print(f"J-cache ckpt: match={ok} (n={len(ck.get('JR', {}))})")
            if ok:
                JR, P128, RANKS, H = ck["JR"], ck["P128"], ck["RANKS"], ck["H"]
                do_j = False
                print(f"J-cache RESUMED from {ckpt} — skipping J-loop")
        except Exception as e:
            print(f"J-cache ckpt load failed: {e}; recomputing")

    # One forward per (sample, S): logits + hs at all positions.
    # (only needed for the J-loop; on resume H comes from the checkpoint)
    fwd: dict[tuple, dict] = {}
    if do_j:
        for dom in domains:
            for si in range(n_samples):
                s = pool[dom][si]
                for S in Ss:
                    fwd[(dom, si, S)] = _forward_prefix(m, s, S, layers, positions_all)
        print(f"Forward cache: {len(fwd)} forwards")

    # J-cache at U_128 per (dom, si, S, pos), plus per-sample ranks.
    JR: dict[tuple, torch.Tensor] = {}
    P128: dict[tuple, torch.Tensor] = {}
    RANKS: dict[tuple, tuple[float, float, float]] = {}
    H: dict[tuple, torch.Tensor] = {}

    total = n * len(layers) * len(cfgs)
    done = 0
    for dom in domains:
        for si in range(n_samples):
            for l in layers:
                for (S, _, p) in cfgs:
                    f = fwd[(dom, si, S)]
                    p128 = torch.softmax(f["logits"][-(p + 1)], dim=-1)[U128]
                    p128 = (p128 / p128.sum()).to(m.device)
                    ids_k = pool[dom][si]["input_ids"][:, : S - p]
                    mask_k = pool[dom][si]["attention_mask"][:, : S - p]
                    Jr = compute_jacobian_for_tokens(
                        m, l, ids_k.to(m.device), mask_k.to(m.device), U128,
                    ).float()
                    JR[(dom, si, l, S, p)] = Jr.cpu()
                    P128[(dom, si, l, S, p)] = p128.cpu()
                    RANKS[(dom, si, l, S, p)] = _ranks(Jr)
                    H[(dom, si, l, S, p)] = f["hs"][l][p]
                    done += 1
                print(f"  {dom} s{si} L{l}: {len(cfgs)} J done ({done}/{total})")
    if do_j:
        torch.save({"JR": JR, "P128": P128, "RANKS": RANKS, "H": H,
                    "cfgs": cfgs, "U128": U128, "pool_fp": pool_fp,
                    "layers": layers, "D": D, "n_samples": n_samples,
                    "domains": domains}, ckpt)
        print(f"J-cache checkpoint saved: {ckpt}")
    del fwd
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    print(f"J cache: {done} raw J [128, {D}] on CPU")

    per_layer = _metrics(JR, P128, RANKS, H, D, layers, domains,
                          n_samples, cfgs)
    # ---- save --------------------------------------------------------------
    import subprocess

    def _git_hash():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                text=True,
            ).strip()
        except Exception:
            return "unknown"

    U_bytes = torch.tensor(U128).numpy().tobytes()
    manifest = {
        "git_hash": _git_hash(),
        "model": model,
        "model_state_sha256_8": hashlib.sha256(
            torch.flatten(m.state_dict()[list(m.state_dict().keys())[0]])
            .cpu().numpy().tobytes()
        ).hexdigest()[:16],
        "U128_sha256_8": hashlib.sha256(U_bytes).hexdigest()[:16],
        "device": device,
    }
    result = {
        "params": {
            "model": model, "layers": layers, "n_samples": n_samples,
            "domains": domains, "Ss": Ss, "Ks": list(Ks),
            "D": D, "token_set": "nested U128 (freq of final logits at S=256)",
        },
        "manifest": manifest,
        "layers": per_layer,
    }
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp27_noise_floor.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[OK] exp27 saved to {out_path}")
    print(json.dumps(manifest, indent=1))

def _metrics(JR, P128, RANKS, H, D, layers, domains, n_samples, cfgs):
    # ---- per-layer metrics -------------------------------------------------
    n = n_samples * len(domains)
    Ss = sorted({S for S, _, _ in cfgs})
    Ks = (32, 64, 128)
    pos_pairs_S64 = ((0, 1), (1, 2), (2, 4), (4, 8), (8, 16))
    pos_pairs_Sother = ((0, 1), (1, 2))
    pos_pairs_S128 = ((0, 1), (1, 2), (64, 65))  # truncation test 6.2
    per_layer: dict[str, dict] = {}
    for l in layers:
        Lres: dict = {"D": D}
        Lres["configs"] = {}
        # sigma_d per S (pos 0, full samples of the S-prefix pool).
        sig_d: dict[int, float] = {}
        for S in Ss:
            stds = []
            for dom in domains:
                hs_d = torch.stack([H[(dom, si, l, S, 0)] for si in range(n_samples)])
                stds.append(hs_d.std(dim=0, unbiased=False).norm().item())
            sig_d[S] = float(np.mean(stds))
        Lres["sigma_d"] = {str(S): sig_d[S] for S in Ss}

        # C_b per (S, K): within-domain sample pairs at pos 0.
        for S in Ss:
            for K in Ks:
                cbs, lbs = [], []
                for dom in domains:
                    js, hs = [], []
                    for si in range(n_samples):
                        Jk = JR[(dom, si, l, S, 0)][:K]
                        pkv = P128[(dom, si, l, S, 0)][:K]
                        pkv = pkv / pkv.sum()
                        js.append(softmax_jacobian(Jk, pkv).cpu())
                        hs.append(H[(dom, si, l, S, 0)])
                    for i in range(n_samples):
                        for j2 in range(i + 1, n_samples):
                            d = (hs[i] - hs[j2]).norm().item()
                            cbs.append(_cos(js[i], js[j2]))
                            lbs.append((js[i] - js[j2]).norm().item() / max(d, 1e-12))
                Lres["configs"].setdefault(f"S{S}_K{K}_p0", {})
                Lres["configs"][f"S{S}_K{K}_p0"]["C_b_mean"] = float(np.mean(cbs))
                Lres["configs"][f"S{S}_K{K}_p0"]["C_b_std"] = float(np.std(cbs))

        # per-config: ranks, F, gauge_share; C_a on pos pairs; L_p90.
        for S in Ss:
            pairs = pos_pairs_S64 if S == 64 else (pos_pairs_S128 if S == 128 else pos_pairs_Sother)
            for K in Ks:
                for p in {q for pr in pairs for q in pr}:
                    key = f"S{S}_K{K}_p{p}"
                    rk_ent, rk_st, rk_pr, gs = [], [], [], []
                    for dom in domains:
                        for si in range(n_samples):
                            Jk = JR[(dom, si, l, S, p)][:K]
                            rk = _ranks(Jk)
                            rk_ent.append(rk[0]); rk_st.append(rk[1]); rk_pr.append(rk[2])
                            pkv = P128[(dom, si, l, S, p)][:K]
                            pkv = pkv / pkv.sum()
                            Jp = softmax_jacobian(Jk, pkv)
                            gs.append(float(Jp.pow(2).sum() / Jk.pow(2).sum().clamp_min(1e-12)))
                    r_eff = float(np.median(rk_ent))
                    F = 1.0 / np.sqrt(r_eff * (K + D - r_eff))
                    Lres["configs"].setdefault(key, {})
                    Lres["configs"][key].update({
                        "r_eff_entrop": r_eff, "r_eff_stable": float(np.median(rk_st)),
                        "r_eff_PR": float(np.median(rk_pr)), "F": float(F),
                        "gauge_share": float(np.median(gs)),
                    })
        Lres["pos_pairs"] = {str(S): [list(pr) for pr in
            (pos_pairs_S64 if S == 64 else pos_pairs_Sother)] for S in Ss}
        per_layer[str(l)] = Lres
        print(f"L{l}: configs done")

        # C_a per (S, K) on pos pairs: 3 conventions + L; per-sample lists
        # for the baseline (S=64, K=64) feed the local half-split bootstrap.
        base_samples: list[tuple[str, float, float, float]] = []  # (dom, Cpos, Csame, craw)
        base_cos_p: list[float] = []
        for S in Ss:
            pairs = pos_pairs_S64 if S == 64 else (pos_pairs_S128 if S == 128 else pos_pairs_Sother)
            for K in Ks:
                for pi, pr in enumerate(pairs):
                    p1, p2 = pr  # p1 < p2: t1 = S-1-p1 (later), t2 = S-1-p2 (earlier)
                    cpos, csame_l, csame_r, craw, lpos, lsame, lraw = [], [], [], [], [], [], []
                    for dom in domains:
                        for si in range(n_samples):
                            J1, J2 = JR[(dom, si, l, S, p1)][:K], JR[(dom, si, l, S, p2)][:K]
                            H1, H2 = H[(dom, si, l, S, p1)], H[(dom, si, l, S, p2)]
                            d = (H1 - H2).norm().item()
                            pk1 = (P128[(dom, si, l, S, p1)][:K] / P128[(dom, si, l, S, p1)][:K].sum())
                            pk2 = (P128[(dom, si, l, S, p2)][:K] / P128[(dom, si, l, S, p2)][:K].sum())
                            J1p = softmax_jacobian(J1, pk1)
                            J2p = softmax_jacobian(J2, pk2)
                            J1s_l = softmax_jacobian(J1, pk1)
                            J2s_l = softmax_jacobian(J2, pk1)
                            J1s_r = softmax_jacobian(J1, pk2)
                            J2s_r = softmax_jacobian(J2, pk2)
                            cpos.append(_cos(J1p, J2p))
                            csame_l.append(_cos(J1s_l, J2s_l))
                            csame_r.append(_cos(J1s_r, J2s_r))
                            craw.append(_cos(J1, J2))
                            lpos.append((J1p - J2p).norm().item() / max(d, 1e-12))
                            lsame.append((J1s_l - J2s_l).norm().item() / max(d, 1e-12))
                            lraw.append((J1 - J2).norm().item() / max(d, 1e-12))
                            if S == 64 and K == 64:
                                base_samples.append((dom, cpos[-1], csame_l[-1], craw[-1]))
                                base_cos_p.append(_cos(P128[(dom, si, l, S, p1)], P128[(dom, si, l, S, p2)]))
                    key = f"S{S}_K{K}_pair{p1}_{p2}"
                    Lres["configs"].setdefault(key, {})
                    Lres["configs"][key].update({
                        "C_a_pospk": float(np.mean(cpos)), "C_a_pospk_std": float(np.std(cpos)),
                        "C_a_samepk_late": float(np.mean(csame_l)),
                        "C_a_samepk_early": float(np.mean(csame_r)),
                        "C_a_raw": float(np.mean(craw)),
                        "L_pos90_pospk": float(np.percentile(lpos, 90)),
                        "L_mean_pospk": float(np.mean(lpos)),
                        "n_pairs": len(cpos),
                    })
        Lres["halfsplit_baseline"] = {
            "per_sample": [[d, c1, c2, c3] for (d, c1, c2, c3) in base_samples],
            "cos_p": base_cos_p,
        }
        # truncation test 6.2: C_a(S=128, pos 64->65) vs C_a(S=64, pos 0->1).
        Lres["trunc_test"] = Lres["configs"].get("S128_K64_pair64_65", {})
        per_layer[str(l)] = Lres
    return per_layer

def _selfcheck() -> None:
    """Model-free checks: ranks of known matrices, F formula, cos, config set."""
    # ranks: rank-1 matrix 8x16 -> entropic = stable = PR = 1.
    J1 = torch.randn(8, 1) @ torch.randn(1, 16)
    r = _ranks(J1)
    assert abs(r[0] - 1.0) < 1e-4 and abs(r[1] - 1.0) < 1e-3, f"rank-1: {r}"
    # rank-8 block: [I8 | 0] -> all three ranks = 8 exactly.
    Jf = torch.zeros(8, 16); Jf[:, :8] = torch.eye(8)
    r = _ranks(Jf)
    assert all(abs(x - 8.0) < 1e-4 for x in r), f"rank-8: {r}"
    # F formula: r_eff = min dim -> F = 1/sqrt(min*(K+D-min)).
    D, K = 1024, 64
    r_eff = 8.0
    F = 1.0 / (r_eff * (K + D - r_eff)) ** 0.5
    assert abs(F - 1.0 / np.sqrt(r_eff * (K + D - r_eff))) < 1e-12
    assert F < 1.0 / np.sqrt(D), "F must be < 1/sqrt(D) when r_eff < D (1/sqrt(D) is overestimated)"
    # cos sanity.
    v = torch.randn(64, 1024)
    assert abs(_cos(v, v) - 1.0) < 1e-5
    assert abs(_cos(v, -v) + 1.0) < 1e-5
    # config set.
    cfgs = _j_configs()
    assert (64, 128, 16) in cfgs and (256, 128, 0) in cfgs and (128, 128, 64) in cfgs
    assert len(cfgs) == 6 + 9 + 2, f"cfg count {len(cfgs)}"
    print("selfcheck OK (ranks, F, cos, configs)")

def main_cli():
    ap = argparse.ArgumentParser(
        description="Exp 27: noise-floor calibration and robustness sweep (docs/Exp27.md)"
    )
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--layers", default="6,12,18,19,22,23")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--domains", default="wiki,math,code,chat")
    ap.add_argument("--make-png", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="path to exp27_jcache.pt: metrics only, no model")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        per_layer = _metrics(ck["JR"], ck["P128"], ck["RANKS"], ck["H"],
                             ck["D"], ck["layers"], ck["domains"],
                             ck["n_samples"], ck["cfgs"])
        out_path = os.path.join(args.results_dir, "exp27_noise_floor.json")
        result = {"params": {"mode": "resume", "D": ck["D"],
                             "layers": ck["layers"], "n_samples": ck["n_samples"],
                             "domains": ck["domains"]},
                  "manifest": {"mode": "resume", "ckpt": args.resume},
                  "layers": per_layer}
        os.makedirs(args.results_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)
        print(f"[OK] resume metrics saved to {out_path}")
        return
    main(
        model=args.model,
        device=args.device,
        results_dir=args.results_dir,
        layers=tuple(int(x) for x in args.layers.split(",")),
        n_samples=args.n_samples,
        domains=tuple(args.domains.split(",")),
        make_png=args.make_png,
    )

if __name__ == "__main__":
    main_cli()
