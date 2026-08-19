"""Experiment 13: Domain-specific ESG and bridge losses.

Check how much ESG maps and bridge losses depend on calibration domain
(wiki / math / code / chat). Universal vs domain-specific subspaces.

FIXES (v2):
  - Common token set for all domains (softmax sum of first sample per domain).
  - Two overlap variants: token_space (ESG eigenvectors on shared set) and
    hidden_space (right singular vectors of mean-J, independent of token set).
  - Bridge on shared token set, probs from large model on shared set.
  - n_samples=8.
"""

import argparse
import json
import os

import torch
from scipy.stats import spearmanr  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

from prism.calibration import (
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.bridge import jacobian_bridge
from prism.esg import compute_esg
from prism.jacobian import compute_jacobian_for_tokens


def _build_shared_token_set(model, all_samples, domain_list, K):
    """Build a SHARED token set: sum softmax probabilities across domains, take top-K."""
    log_probs = torch.zeros(1, device=model.device)
    for d in domain_list:
        samples = all_samples[d]
        if not samples:
            continue
        ids = samples[0]["input_ids"].to(model.device)
        attn = samples[0]["attention_mask"].to(model.device)
        with torch.no_grad():
            lg = model(input_ids=ids, attention_mask=attn).logits[:, -1, :]
        log_probs = log_probs + torch.log_softmax(lg, dim=-1)
    return sorted(log_probs.topk(K, dim=-1).indices[0].tolist())


def _subspace_overlap(U1, U2):
    """||U1^T @ U2||^2_F / n  — normalized to [0, 1] when unit-length vectors."""
    n = U1.shape[0]
    return (U1 @ U2.T).pow(2).sum().item() / n


def _bridge_losses(mL, mS, EL, ES, samples, ts, pairs, n_samples):
    """Bridge loss averaged over all samples per pair."""
    losses = []
    for lL, lS in pairs:
        pair_losses = []
        for si in range(n_samples):
            ids, attn = samples[si]["input_ids"], samples[si]["attention_mask"]
            with torch.no_grad():
                lg = mL(
                    input_ids=ids.to(mL.device),
                    attention_mask=attn.to(mL.device),
                ).logits
            probs = torch.softmax(lg[0, -1, ts], dim=-1).to(mS.device)
            ids = samples[si]["input_ids"].to(mL.device)
            attn = samples[si]["attention_mask"].to(mL.device)
            JL = compute_jacobian_for_tokens(mL, lL, ids, attn, ts).to(mS.device)
            ids = samples[si]["input_ids"].to(mS.device)
            attn = samples[si]["attention_mask"].to(mS.device)
            JS = compute_jacobian_for_tokens(mS, lS, ids, attn, ts)
            pair_losses.append(jacobian_bridge(JL, JS, EL, ES, probs))
            del JL, JS
        losses.append(
            {
                "layer": f"{lL}->{lS}",
                "loss": sum(pair_losses) / len(pair_losses),
                "per_sample": [float(x) for x in pair_losses],
            }
        )
    return losses


def _domain_esg(mS, samples, selected, ts, nc, n_samples):
    """ESG on small model with given token set.

    Returns spectra (eigenvectors in token space) and mean-Jacobians (for hidden-space overlap).
    """
    spectra = {}
    mean_js = {}
    for layer in selected:
        js = []
        for si in range(n_samples):
            J = compute_jacobian_for_tokens(
                mS,
                layer,
                samples[si]["input_ids"].to(mS.device),
                samples[si]["attention_mask"].to(mS.device),
                ts,
            )
            js.append(J)
        print(f"    layer {layer}")
        ev, eig = compute_esg(js, n_components=nc)
        ev_cpu = ev.cpu()
        n_eig = min(8, len(ev_cpu))
        spectra[layer] = {
            "trace": ev_cpu.sum().item(),
            "top5": ev_cpu[:5].tolist(),
            "eig_top8": eig[:n_eig].cpu().tolist(),
            "n_eig": n_eig,
        }
        # Mean Jacobian for hidden-space overlap (SVD of mean-J)
        mean_j = torch.stack(js).mean(dim=0)
        _, _, vh = torch.linalg.svd(mean_j)
        # vh is [D, D] (or [min(K,D), D]); take top-8 right singular vectors -> [8, D]
        mean_js[layer] = vh[:8].T  # [D, 8]
        del mean_j, vh
    return spectra, mean_js


def _compute_overlap_token(all_res, domain_list, selected):
    """Overlap in token space (ESG eigenvectors on shared token set)."""
    overlap = {}
    for d1 in domain_list:
        for d2 in domain_list:
            k1, k2 = f"esg_{d1}", f"esg_{d2}"
            if k1 not in all_res or k2 not in all_res:
                continue
            ov = {}
            for layer in selected:
                U1 = torch.tensor(all_res[k1][layer]["eig_top8"])
                U2 = torch.tensor(all_res[k2][layer]["eig_top8"])
                ov[layer] = _subspace_overlap(U1, U2)
            overlap[f"{d1}x{d2}"] = ov
    return overlap


def _compute_overlap_hidden(all_mean_js, domain_list, selected):
    """Overlap in hidden space (right singular vectors of mean-J, independent of token set).

    V[d][layer] has shape [D, 8]; overlap = ||V1^T @ V2||^2_F / 8.
    """
    overlap = {}
    for d1 in domain_list:
        for d2 in domain_list:
            k1, k2 = f"meanjs_{d1}", f"meanjs_{d2}"
            if k1 not in all_mean_js or k2 not in all_mean_js:
                continue
            ov = {}
            for layer in selected:
                V1 = all_mean_js[k1][layer]  # [D, 8]
                V2 = all_mean_js[k2][layer]
                n = min(V1.shape[1], V2.shape[1])
                ov[layer] = (V1[:, :n].T @ V2[:, :n]).pow(2).sum().item() / n
            overlap[f"{d1}x{d2}"] = ov
    return overlap


def _print_summary(all_res, domain_list, selected, n_layers):
    mono = all_res.get("monotonicity", {})
    last_layer = selected[-1] if selected else n_layers - 1
    print("\n=== Domain ESG ===")
    print(f"{'Domain':<8} {'Tr(L0)':>12} {'Tr(L{last_layer})':>12} {'Mono':>8}")
    for d in domain_list:
        ek = f"esg_{d}"
        if ek not in all_res:
            continue
        t0 = all_res[ek].get(0, {}).get("trace", 0)
        tl = all_res[ek].get(last_layer, {}).get("trace", 0)
        print(f"{d:<8} {t0:>12.1f} {tl:>12.1f} {mono.get(d, float('nan')):>8.3f}")
    print(f"\n{'Domain':<8} {'AvgBridge':>10}")
    for d in domain_list:
        bk = f"bridge_{d}"
        if bk not in all_res:
            continue
        avg = sum(r["loss"] for r in all_res[bk]) / len(all_res[bk])
        print(f"{d:<8} {avg:>10.4f}")
    for suffix in ("_token", "_hidden"):
        key = f"overlap{suffix}"
        ov = all_res.get(key, {})
        if not ov:
            continue
        print(f"\n=== Overlap ({suffix}) ===")
        hdr = f"{'Pair':<16}" + "".join(f" {f'L{li}':>6}" for li in selected)
        print(hdr)
        for pair, vals in ov.items():
            rows = [vals.get(li, -1) for li in selected]
            print(f"{pair:<16}" + "".join(f" {r:>6.3f}" for r in rows))


def main(
    large_model="/data/Qwen/Qwen3.5-4B",
    small_model="Qwen/Qwen3.5-0.8B",
    n_samples=8,
    K=64,
    max_length=64,
    min_length=32,
    n_components=32,
    domains="wiki,math,code,chat",
    max_pairs=5,
    device_L="auto",
    device_S="auto",
    results_dir="results",
):
    torch.manual_seed(42)

    if device_L == "auto":
        device_L = "cuda" if torch.cuda.is_available() else "cpu"
    if device_S == "auto":
        device_S = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Devices: L={device_L}, S={device_S}")
    if device_L == "cpu" or device_S == "cpu":
        torch.set_num_threads(4)

    domain_list = [d.strip() for d in domains.split(",")]
    print(f"Domains: {domain_list}")

    # Load calibration data per domain
    all_samples = {}
    for d in domain_list:
        print(f"Loading {n_samples} samples for domain {d}...")
        all_samples[d] = load_calibration_data(
            small_model,
            max_length=max_length,
            min_length=min_length,
            n_samples=n_samples,
            domain=d,
        )
        got = len(all_samples[d])
        print(f"  Got {got} samples")
        if got == 0:
            print(f"  Warning: no samples for {d}, skipping")
            continue

    # Load models

    print(f"Loading {large_model}...")
    dtype_L = (
        torch.float16
        if torch.cuda.is_available() and device_L.startswith("cuda")
        else torch.float32
    )
    mL = AutoModelForCausalLM.from_pretrained(
        large_model, torch_dtype=dtype_L, attn_implementation="eager"
    ).to(device_L)
    mL.eval()

    print(f"Loading {small_model}...")
    mS = AutoModelForCausalLM.from_pretrained(
        small_model, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device_S)
    mS.eval()

    # Shared token set (sum of softmax probs across domains)
    active_domains = [d for d in domain_list if all_samples[d]]
    shared_ts = _build_shared_token_set(mL, all_samples, active_domains, K)
    print(f"Shared token set size: {len(shared_ts)}")

    n_layers = mS.config.num_hidden_layers
    selected = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    print(f"Selected layers (0.8B): {selected}")

    mapping = uniform_layer_mapping(
        mL.config.num_hidden_layers, mS.config.num_hidden_layers
    )
    step = max(1, len(mapping) // 5)
    sampled_pairs = list(mapping.items())[::step]
    if len(sampled_pairs) > max_pairs:
        idx = list(range(0, len(sampled_pairs), len(sampled_pairs) // max_pairs))[
            :max_pairs
        ]
        sampled_pairs = [sampled_pairs[i] for i in idx]
    print(f"Sampled pairs (large->small): {len(sampled_pairs)}")

    EL = mL.model.embed_tokens.weight.detach().to(mS.device)
    ES = mS.model.embed_tokens.weight.detach()

    # ---- Per-domain ESG + bridge (on shared token set) ----
    all_results = {}
    all_mean_js = {}
    for d in active_domains:
        samples = all_samples[d]
        n_s = len(samples)
        print(f"\n--- Domain: {d} ---")

        spectra, mean_js = _domain_esg(
            mS, samples, selected, shared_ts, n_components, n_s
        )
        all_results[f"esg_{d}"] = spectra
        all_mean_js[f"meanjs_{d}"] = mean_js

        all_results[f"bridge_{d}"] = _bridge_losses(
            mL, mS, EL, ES, samples, shared_ts, sampled_pairs, n_s
        )

    # ---- Overlap: token space (ESG eigenvectors on shared set) ----
    all_results["overlap_token"] = _compute_overlap_token(
        all_results, active_domains, selected
    )

    # ---- Overlap: hidden space (right singular vectors of mean-J) ----
    all_results["overlap_hidden"] = _compute_overlap_hidden(
        all_mean_js, active_domains, selected
    )

    # ---- Spearman monotonicity (bridge losses per domain) ----
    mono = {}
    for d in active_domains:
        bk = f"bridge_{d}"
        if bk not in all_results:
            continue
        losses = [r["loss"] for r in all_results[bk]]
        if len(losses) >= 3:
            rho, _ = spearmanr(range(len(losses)), losses)
            mono[d] = float(rho)  # type: ignore[arg-type]
    all_results["monotonicity"] = mono

    # ---- Cross-check: Spearman between token and hidden overlap ----
    # Flatten all pairs x layers
    token_vals = []
    hidden_vals = []
    for pair_key in all_results["overlap_token"]:
        for layer in selected:
            tv = all_results["overlap_token"][pair_key].get(layer)
            hv = all_results["overlap_hidden"][pair_key].get(layer)
            if tv is not None and hv is not None:
                token_vals.append(tv)
                hidden_vals.append(hv)
    if len(token_vals) >= 3:
        rho_cross, _ = spearmanr(token_vals, hidden_vals)
        all_results["overlap_cross_spearman"] = float(rho_cross)  # type: ignore[arg-type]

    # ---- Save ----
    all_results["shared_token_set_size"] = len(shared_ts)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp13_domain_esg.json")
    all_results["params"] = {large_model: large_model, small_model: small_model, n_samples: n_samples, K: K, max_length: max_length, min_length: min_length, n_components: n_components, domains: domains, max_pairs: max_pairs, device_L: device_L, device_S: device_S}
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    _print_summary(all_results, domain_list, selected, n_layers)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 13: Domain-specific ESG + bridge")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--domains", default="wiki,math,code,chat")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--max-pairs", type=int, default=5)
    args = parser.parse_args()
    main(**vars(args))
