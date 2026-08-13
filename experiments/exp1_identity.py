"""Experiment 1: Identity test — same model as large and small.

With pinv Gramian correction, J @ E^T @ E @ (E^T @ E)^{-1} = J,
so identity loss is ~1e-7. The validation is:
1. L_bridge ≈ 0 for J == J (projection correction works)
2. Determinism: J computed twice is identical
"""

import argparse
import json
import os

import torch

from transformers import AutoModelForCausalLM

from prism.calibration import (
    build_global_token_set,
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.jacobian import compute_jacobian_for_tokens
from prism.bridge import jacobian_bridge


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=4,
    K=32,
    max_length=256,
    min_length=128,
    results_dir="results",
):
    torch.set_num_threads(4)
    print("Loading calibration data...")
    samples = load_calibration_data(
        model, max_length=max_length, min_length=min_length, n_samples=n_samples
    )
    print(f"Loaded {len(samples)} samples")

    print("Loading model (once)...")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model_obj.eval()
    n_layers = model_obj.config.num_hidden_layers
    print(f"Model loaded, {n_layers} layers")

    print("Building global token set...")
    token_set = build_global_token_set(model_obj, samples, K=K)
    print(f"Token set size: {len(token_set)}")

    E = extract_embeddings(model_obj)
    mapping = uniform_layer_mapping(n_layers, n_layers)

    results = {}
    print("Running identity test...")

    for l_L, l_S in mapping.items():
        losses = []
        for sample in samples:
            ids = sample["input_ids"].to(model_obj.device)
            attn = sample["attention_mask"].to(model_obj.device)

            J = compute_jacobian_for_tokens(model_obj, l_L, ids, attn, token_set)
            with torch.no_grad():
                logits = model_obj(input_ids=ids).logits[:, -1, :]
                probs = torch.softmax(logits[0, token_set], dim=-1)

            loss = jacobian_bridge(J, J, E, E, probs)
            losses.append(loss)

        avg = sum(losses) / len(losses)
        results[f"{l_L}->{l_S}"] = avg
        print(f"  Layer {l_L}->{l_S}: {avg:.6f}")

    # Determinism check
    det_ok = True
    for layer in [0, n_layers // 2, n_layers - 1]:
        sample = samples[0]
        ids = sample["input_ids"].to(model_obj.device)
        attn = sample["attention_mask"].to(model_obj.device)
        J1 = compute_jacobian_for_tokens(model_obj, layer, ids, attn, token_set)
        J2 = compute_jacobian_for_tokens(model_obj, layer, ids, attn, token_set)
        diff = (J1 - J2).abs().max().item()
        if diff > 1e-6:
            det_ok = False
            print(f"  Determinism FAIL at layer {layer}: diff={diff}")

    max_loss = max(results.values()) if results else 0
    os.makedirs(results_dir, exist_ok=True)
    out = {
        "type": "identity",
        "model": model,
        "n_samples": len(samples),
        "K": K,
        "results": results,
        "mean_loss": sum(results.values()) / len(results),
        "max_loss": max_loss,
        "determinism_pass": det_ok,
    }
    with open(os.path.join(results_dir, "exp1_identity.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nMean L_bridge: {out['mean_loss']:.6f}")
    print(f"Max L_bridge:  {max_loss:.6f}")
    print(f"Determinism:   {'PASS' if det_ok else 'FAIL'}")
    return out, det_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 1: Identity test")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--min-length", type=int, default=128)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(
        model=args.model,
        n_samples=args.n_samples,
        K=args.K,
        max_length=args.max_length,
        min_length=args.min_length,
        results_dir=args.results_dir,
    )
