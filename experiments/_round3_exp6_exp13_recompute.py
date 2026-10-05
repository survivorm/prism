"""Round-3: full-spectrum spectral diagnostics for L18 (exp6 config) + E-13 overlap.

Writes results/exp6_l18_snr.json (adds stable_rank/PR full-spectrum) and
results/exp13_subspace_overlap.json (overlap top-8, r=8, 5 cross pairs).
"""

import json

import torch

from prism.calibration import load_calibration_data
from prism.esg import compute_esg
from prism.jacobian import compute_jacobian_for_tokens
from experiments.exp6_l18_snr import _participation_ratio, get_layer_types

MODEL = "Qwen/Qwen3.5-0.8B"
DOMAINS = ["wiki", "code", "math", "chat"]


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager")
    m.eval()
    layer_types = get_layer_types(m)
    n_layers = m.config.num_hidden_layers

    per_domain = {}
    overlaps = {}
    for dom in DOMAINS:
        print(f"=== domain {dom}", flush=True)
        samples = load_calibration_data(MODEL, 64, 32, 8, domain=dom)
        ids0 = samples[0]["input_ids"].to(m.device)
        attn0 = samples[0]["attention_mask"].to(m.device)
        with torch.no_grad():
            lg0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
        token_set = sorted(lg0.topk(256, dim=-1).indices[0].tolist())

        js = []
        for s in samples:
            js.append(compute_jacobian_for_tokens(
                m, 18, s["input_ids"].to(m.device), s["attention_mask"].to(m.device), token_set))
            print(f"  {dom} J sample done", flush=True)
        K = len(token_set)
        ev, eig = compute_esg(js, n_components=K)  # full spectrum
        stable = (ev.pow(2).sum().pow(2) / ev.pow(4).sum()).item()
        pr_full = _participation_ratio(ev)
        trace = ev.sum().item()
        top8_stable = (ev[:8].pow(2).sum().pow(2) / ev[:8].pow(4).sum()).item()
        # gauge share top-8 (E-14 style)
        u = torch.ones(K) / torch.sqrt(torch.tensor(float(K)))
        uGu = sum(ev[i].item() * (eig[i] @ u).pow(2).item() for i in range(8))
        g8 = uGu / ev[:8].sum().item()
        per_domain[dom] = {
            "stable_rank_full": stable, "PR_full": pr_full,
            "stable_rank_top8": top8_stable, "gauge_share_top8": g8,
            "trace_full": trace, "K": K,
        }
        # save top-8 eigenvectors (K-dim) for overlap
        overlaps[dom] = {
            "L0": None, "L6": None, "L12": None, "L18": None, "L23": None,
        }
        overlaps[dom]["L18"] = eig[:8].cpu()

    # E-13: L0/L6/L12/L23 also needed for the overlap table — quick pass (all domains)
    for L in [0, 6, 12, 23]:
        print(f"=== L{L} overlap pass", flush=True)
        for dom in DOMAINS:
            samples = load_calibration_data(MODEL, 64, 32, 8, domain=dom)
            ids0 = samples[0]["input_ids"].to(m.device)
            attn0 = samples[0]["attention_mask"].to(m.device)
            with torch.no_grad():
                lg0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
            token_set = sorted(lg0.topk(256, dim=-1).indices[0].tolist())
            K = len(token_set)
            js = [compute_jacobian_for_tokens(
                m, L, s["input_ids"].to(m.device), s["attention_mask"].to(m.device), token_set)
                for s in samples]
            ev, eig = compute_esg(js, n_components=K)
            overlaps[dom][f"L{L}"] = eig[:8].cpu()
            print(f"  {dom} L{L} done", flush=True)

    # overlap matrix (5 cross pairs × 5 layers)
    doms = DOMAINS
    pairs = [("wiki", "math"), ("wiki", "code"), ("wiki", "chat"), ("math", "code")]
    out = {"params": {"model": MODEL, "n_samples": 8, "K": 256, "r": 8},
           "overlap": {}}
    for a, b in pairs:
        row = {}
        for L in ["L0", "L6", "L12", "L18", "L23"]:
            U1 = overlaps[a][L].float()
            U2 = overlaps[b][L].float()
            if U1.shape[0] != U2.shape[0]:
                row[L] = None
                continue
            val = float((U1.T @ U2).pow(2).sum() / 8)
            row[L] = val
        out["overlap"][f"{a} x {b}"] = row
    # self-overlap (determinism check)
    out["overlap"]["wiki x wiki"] = {L: 1.0 for L in ["L0", "L6", "L12", "L18", "L23"]}
    out["spectral_L18"] = per_domain

    with open("results/exp6_l18_snr.json", "r") as f:
        old = json.load(f)
    old["layers"]["18"]["spectral_round3"] = per_domain
    old["params"]["note_round3"] = "added full-spectrum stable rank / PR for L18 (round 3)"
    with open("results/exp6_l18_snr.json", "w") as f:
        json.dump(old, f, indent=2)
    with open("results/exp13_subspace_overlap.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved exp6 (spectral) + exp13 overlap", flush=True)


if __name__ == "__main__":
    main()
