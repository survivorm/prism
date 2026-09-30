"""Exp 28b — dense J-profile cache, S=128, t=0..127.

One-off GPU cache to feed exp28 (leaf structure / curvature) and future
post-processing: gauge-free J(t) at EVERY position of the 128-prefix for all
32 samples and 6 layers. Causal prefixes make S=64/32 dense profiles free
slices of this cache (exp27 causal-identity, verified).

24,576 J calls (128 t x 32 samples x 6 layers) ~ 12-16h T4.
Per-layer checkpoint files results/exp28_jcache_L{l}.pt (~2.1GB, RAM-safe,
layer-granular resume). Post-processing stays local (no GPU).
"""

import argparse
import hashlib
import os
import time

import numpy as np
import torch

from prism.calibration import load_calibration_data
from prism.jacobian import compute_jacobian_for_tokens

S_DENSE = 128
LAYERS_DEFAULT = (6, 12, 18, 19, 22, 23)

def mad_boundary(values: list[float]) -> float | None:
    """median + 2*MAD boundary; None if <4 values (Exp28 §4/§8)."""
    if len(values) < 4:
        return None
    v = np.asarray(values)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med + 2.0 * 1.4826 * mad

def _load_pool(model_name: str, domains: list[str], n: int, L: int = 256) -> dict:
    """Per domain: n texts of exactly L tokens. ONE stream pass, shorts
    concatenated (exp27 v11/v12: full 320-word non-overlapping windows +
    dedup; math via nvidia/OpenMathInstruct-2 in load_calibration_data)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    per: dict[str, list[dict]] = {}
    for dom in domains:
        big: list[dict] = []
        for att in range(5):
            big = load_calibration_data(
                model_name, max_length=L, min_length=1, n_samples=n * 64, domain=dom,
            )
            data0 = [s for s in big if s["input_ids"].shape[1] >= L]
            if len(data0) >= n or len(big) >= n * 64:
                break
            print(f"  pool {dom}: stream gave {len(data0)} >=L, retry {att + 1}...")
            time.sleep(10 * (att + 1))
        data = data0
        if len(data) < n:  # intrinsically short sources: concatenate
            buf = ""
            out = []
            consumed = 0
            for t in big:
                buf += " " + tok.decode(t["input_ids"][0], skip_special_tokens=True).strip()
                words = buf.split(" ")
                while len(out) < n and len(words) - consumed >= 64:
                    ids = tok(" ".join(words[consumed:consumed + 320]),
                              max_length=L, truncation=True, return_tensors="pt")
                    if ids["input_ids"].shape[1] >= L:
                        out.append(ids)
                        consumed += 320
                        continue
                    break
            data = data + out
            seen, ded = set(), []
            for x in data:
                t = tuple(x["input_ids"][0].tolist())
                if t not in seen:
                    seen.add(t)
                    ded.append(x)
            if len(ded) < n:
                print(f"  pool {dom}: WARNING dedup -> {len(ded)} unique < {n}")
            data = (ded + data)[:n]
        else:
            data = data[:n]
        print(f"  pool {dom}: {len(data)} @ {L}")
        per[dom] = data
    return per

def _forward_dense(m, s: dict, S: int, layers: list[int]) -> dict:
    """One forward on the S-prefix: logits + hidden states at ALL positions
    (from the end, p=0..S-1) for each layer."""
    ids = s["input_ids"][:, :S].to(m.device)
    attn = s["attention_mask"][:, :S].to(m.device)
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    logits = out.logits[0].float().cpu()
    hs = {}
    for l in layers:
        hs[l] = out.hidden_states[l + 1][0].float().cpu()  # [S, D], all pos
    return {"logits": logits, "hs": hs}

def main(
    model="Qwen/Qwen3.5-0.8B",
    device="auto",
    results_dir="results",
    layers=LAYERS_DEFAULT,
    n_samples=8,
    domains=("wiki", "math", "code", "chat"),
    seed: int = 42,
    s_dense: int = S_DENSE,
):
    S = s_dense
    torch.manual_seed(seed)
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

    print(f"Loading pool: {n_samples} x {len(domains)} @ 256 tokens...")
    pool = _load_pool(model, domains, n_samples, 256)
    for dom in domains:
        assert len(pool[dom]) == n_samples, f"pool {dom}: {len(pool[dom])} < {n_samples}"
    n = n_samples * len(domains)
    print(f"Pool OK: {n} samples")

    # U_128: top-128 by final-logit frequency at S=256 (identical to exp27 v11).
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
    print(f"U_128 built fresh ({len(U128)} tokens)")

    pool_fp = {}
    for d in domains:
        blob = b"".join(x["input_ids"].flatten().numpy().tobytes() for x in pool[d])
        pool_fp[d] = hashlib.sha256(blob).hexdigest()[:12]

    t0 = time.time()
    for l in layers:
        ck_l = os.path.join(results_dir, f"exp28_jcache_L{l}.pt")
        JR: dict[tuple, torch.Tensor] = {}
        P128: dict[tuple, torch.Tensor] = {}
        RANKS: dict[tuple, tuple[float, float, float]] = {}
        H: dict[tuple, torch.Tensor] = {}
        if os.path.exists(ck_l):
            ck = torch.load(ck_l, map_location="cpu")
            if ck["U128"] == U128 and ck["pool_fp"] == pool_fp and ck.get("S") == S:
                JR, P128, RANKS, H = ck["JR"], ck["P128"], ck["RANKS"], ck["H"]
                print(f"L{l}: resumed {len(JR)}/{n * S} J from ckpt")
            else:
                print(f"L{l}: ckpt U128/pool mismatch -> recompute layer")
        fwd: dict[tuple, dict] = {}
        for dom in domains:
            for si in range(n_samples):
                if any((dom, si, t) in JR for t in range(S)):
                    continue
                fwd[(dom, si)] = _forward_dense(m, pool[dom][si], S, layers)
        done = len(JR)
        total = n * S
        for dom in domains:
            for si in range(n_samples):
                for t in range(S):
                    if (dom, si, t) in JR:
                        continue
                    f = fwd[(dom, si)]
                    p128 = torch.softmax(f["logits"][S - 1 - t], dim=-1)[U128]
                    p128 = (p128 / p128.sum()).to(m.device)
                    ids_k = pool[dom][si]["input_ids"][:, :S - t]
                    mask_k = pool[dom][si]["attention_mask"][:, :S - t]
                    pc = None
                    if S - t == 1:
                        # J at position 0: prefix is a single token -> the
                        # prefix pass would run the model on 0 tokens, which
                        # the GDN layers cannot handle (reshape [1,0,-1,H]).
                        # An empty DynamicCache is exactly the state after
                        # 0 tokens; the decoder pass is a fresh 1-token pass.
                        pc = DynamicCache(config=m.config)
                    Jr = compute_jacobian_for_tokens(
                        m, l, ids_k.to(m.device), mask_k.to(m.device), U128
                    ).float()
                    JR[(dom, si, t)] = Jr.cpu()
                    P128[(dom, si, t)] = p128.cpu()
                    RANKS[(dom, si, t)] = _ranks(Jr)
                    H[(dom, si, t)] = f["hs"][l][S - 1 - t]
                    done += 1
                    if done % 512 == 0:
                        print(f"  L{l} J {done}/{total} "
                              f"({(time.time() - t0) / 60:.1f} min total)")
                fwd.pop((dom, si), None)  # free this sample's logits+hs
        del fwd
        torch.save({"JR": JR, "P128": P128, "RANKS": RANKS, "H": H,
                    "U128": U128, "pool_fp": pool_fp,
                    "layers": layers, "D": D, "n_samples": n_samples,
                    "domains": domains, "S": S}, ck_l)
        print(f"L{l}: saved {ck_l} ({len(JR)}/{total} J) "
              f"[{(time.time() - t0) / 60:.1f} min total]")
    print(f"[OK] exp28_dense cache: {len(layers)} layers x {n} samples x {S} t "
          f"in {(time.time() - t0) / 60:.0f} min, S={S}")

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

def selfcheck():
    # constant field: MAD=0 -> boundary = median
    assert abs(mad_boundary([0.01] * 10) - 0.01) < 1e-12
    v = [0.01] * 20
    v[7] = 0.5
    b = mad_boundary(v)
    assert b is not None and v[7] > b and all(x <= b for x in v if x != 0.5)
    # short list -> None
    assert mad_boundary([0.1, 0.2]) is None
    print("selfcheck OK")

def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS_DEFAULT))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--s-dense", type=int, default=S_DENSE)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        return
    main(model=a.model, device=a.device, results_dir=a.results_dir,
         layers=a.layers, n_samples=a.n_samples, seed=a.seed, s_dense=a.s_dense)

if __name__ == "__main__":
    main_cli()
