"""Experiment 15: Anchor layer shifts and computation phases.

Check that bridge convergence trend is not an artifact of hidden-state sampling
phase (exit of full-attention vs GDN block). Three schemes:
A (full-attention), B (post-full = first GDN after full), C (pre-full = last GDN before full).
"""

import argparse
import json
import os

import torch
from scipy.stats import spearmanr  # type: ignore[import-untyped]
from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

from prism.bridge import jacobian_bridge
from prism.calibration import (
    extract_embeddings,
    get_layer_types,
    load_calibration_data,
)
from prism.esg import compute_esg
from prism.jacobian import compute_jacobian_for_tokens


def _get_full_indices(layer_types):
    """Return indices of full_attention layers."""
    return [i for i, t in enumerate(layer_types) if t == "full_attention"]


def _get_anchors(layer_types, scheme):
    """Build anchor list for scheme A/B/C/D, clipping to valid range."""
    full = _get_full_indices(layer_types)
    n = len(layer_types)
    if scheme == "A":
        return full
    if scheme == "B":
        return [min(i + 1, n - 1) for i in full]
    if scheme == "C":
        return [max(i - 1, 0) for i in full]
    # D: full-2, control phase (GDN anchors, not adjacent to full)
    return [max(i - 2, 0) for i in full]


def _build_pairs(anchors_large, anchors_small, max_pairs):
    """Pair anchors proportionally by position in each list, capped at max_pairs."""
    m = len(anchors_large)
    n = len(anchors_small)
    pairs = []
    for i in range(m):
        j = round(i * max(n - 1, 0) / max(m - 1, 1))
        pairs.append((anchors_large[i], anchors_small[j]))
    if max_pairs is not None and len(pairs) > max_pairs:
        idx = list(range(0, len(pairs), len(pairs) // max_pairs))[:max_pairs]
        pairs = [pairs[i] for i in idx]
    return pairs


def _run_scheme(
    scheme_name,
    m_large,
    m_small,
    pairs,
    samples,
    token_set,
    probs_cache,
    E_L,
    E_S,
    n_components=32,
):
    """Run bridge + ESG trace for one anchor scheme."""
    results = {
        "scheme": scheme_name,
        "pairs": pairs,
        "bridge": {},
        "esg_trace": {},
    }

    for li, si in pairs:
        losses = []
        js_small = []
        for si_idx, s in enumerate(samples):
            ids_l = s["input_ids"].to(m_large.device)
            attn_l = s["attention_mask"].to(m_large.device)
            ids_s = s["input_ids"].to(m_small.device)
            attn_s = s["attention_mask"].to(m_small.device)
            J_L = compute_jacobian_for_tokens(m_large, li, ids_l, attn_l, token_set).to(
                m_small.device
            )
            J_S = compute_jacobian_for_tokens(m_small, si, ids_s, attn_s, token_set)
            js_small.append(J_S)
            probs = probs_cache[si_idx][token_set].to(m_small.device)
            loss = jacobian_bridge(J_L, J_S, E_L, E_S, probs)
            losses.append(float(loss))
            del J_L, J_S

        results["bridge"][f"{li}->{si}"] = {
            "mean": sum(losses) / len(losses),
            "values": losses,
        }

        ev, _ = compute_esg(js_small, n_components=n_components)
        results["esg_trace"][str(si)] = ev.sum().item()
        del js_small

    return results


def main(
    model="Qwen/Qwen3.5-0.8B",
    large_model="/data/Qwen/Qwen3.5-4B",
    n_samples=4,
    K=256,
    max_length=64,
    min_length=32,
    max_pairs=4,
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

    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )

    print(f"Loading small model {model}...")
    m_small = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device_S)
    m_small.eval()

    print(f"Loading large model {large_model}...")
    dtype_L = torch.float16 if device_L.startswith("cuda") else torch.float32
    m_large = AutoModelForCausalLM.from_pretrained(
        large_model, torch_dtype=dtype_L, attn_implementation="eager"
    ).to(device_L)
    m_large.eval()

    types_small = get_layer_types(m_small)
    types_large = get_layer_types(m_large)

    E_L = extract_embeddings(m_large).to(device_S)
    E_S = extract_embeddings(m_small)

    # Token set + probs cache from large model
    ids0 = samples[0]["input_ids"].to(device_L)
    attn0 = samples[0]["attention_mask"].to(device_L)
    with torch.no_grad():
        lg0 = m_large(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    token_set = sorted(lg0.topk(K, dim=-1).indices[0].tolist())

    # Precompute probs for all samples
    probs_cache = []
    for s in samples:
        with torch.no_grad():
            out = m_large(
                s["input_ids"].to(device_L),
                attention_mask=s["attention_mask"].to(device_L),
            )
            p = torch.softmax(out.logits[:, -1, :], dim=-1)[0]
        probs_cache.append(p)

    # Build anchors for each scheme
    schemes = {}
    for name in ("A", "B", "C", "D"):
        anc_s = _get_anchors(types_small, name)
        anc_l = _get_anchors(types_large, name)
        pairs = _build_pairs(anc_l, anc_s, max_pairs)
        schemes[name] = {
            "anchors_large": anc_l,
            "anchors_small": anc_s,
            "pairs": pairs,
        }
        print(f"Scheme {name}: large={anc_l}, small={anc_s}, pairs={len(pairs)}")

    # Run each scheme
    all_results = {}
    for name in ("A", "B", "C", "D"):
        print(f"\n=== Running scheme {name} ===")
        res = _run_scheme(
            name,
            m_large,
            m_small,
            schemes[name]["pairs"],
            samples,
            token_set,
            probs_cache,
            E_L,
            E_S,
        )
        all_results[name] = res

    # Compute Spearman correlations
    correlations = {}
    for name in ("A", "B", "C", "D"):
        bridge_vals = [
            all_results[name]["bridge"][f"{li}->{si}"]["mean"]
            for li, si in schemes[name]["pairs"]
        ]
        depths = [si for li, si in schemes[name]["pairs"]]
        if len(bridge_vals) >= 3:
            rho, _ = spearmanr(depths, bridge_vals)
            correlations[f"{name}_monotonicity"] = float(rho)  # type: ignore[arg-type]

    # Spearman(A,B), Spearman(A,C), Spearman(A,D): correlate bridge losses by position
    for n2 in ("B", "C", "D"):
        vA = [
            all_results["A"]["bridge"][f"{li}->{si}"]["mean"]
            for li, si in schemes["A"]["pairs"]
        ]
        vB = [
            all_results[n2]["bridge"][f"{li}->{si}"]["mean"]
            for li, si in schemes[n2]["pairs"]
        ]
        min_len = min(len(vA), len(vB))
        if min_len >= 3:
            rho, _ = spearmanr(vA[:min_len], vB[:min_len])
            correlations[f"A_vs_{n2}"] = float(rho)  # type: ignore[arg-type]

    # Max absolute difference across schemes at comparable depths
    max_abs_diff = 0.0
    for (li_a, si_a), _ in schemes["A"]["pairs"]:
        for n2 in ("B", "C", "D"):
            val_b_list = [
                all_results[n2]["bridge"][f"{ll}->{ss}"]["mean"]
                for ll, ss in schemes[n2]["pairs"]
            ]
            for idx, val_b in enumerate(val_b_list):
                si_b = schemes[n2]["pairs"][idx][1]
                if abs(si_a - si_b) <= 2:  # within 2 layers
                    val_a = all_results["A"]["bridge"][f"{li_a}->{si_a}"]["mean"]
                    max_abs_diff = max(max_abs_diff, abs(val_a - val_b))
    correlations["max_abs_diff_schemes"] = float(max_abs_diff)

    # Save
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "exp15_gdn_anchors.json")
    out = {"schemes": all_results, "correlations": correlations}
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # Summary
    print("\n=== Summary ===")
    for name in ("A", "B", "C", "D"):
        bridge_means = [
            all_results[name]["bridge"][f"{li}->{si}"]["mean"]
            for li, si in schemes[name]["pairs"]
        ]
        print(f"\nScheme {name}:")
        for (li, si), val in zip(schemes[name]["pairs"], bridge_means, strict=True):
            print(
                f"  L{li} -> L{si} ({types_small[si]}): "
                f"loss={val:.4f} "
                f"trace={all_results[name]['esg_trace'].get(str(si), 'N/A')}"
            )

    print(f"\nCorrelations: {json.dumps(correlations, indent=2)}")
    print(f"\nSaved to {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 15: Anchor layer shifts and computation phases"
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
