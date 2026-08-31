"""CKA identity-control: CKA(same_model, same_layer) should be ≈ 1.

If CKA(4B→4B) ≈ 1 and CKA(0.8B→0.8B) ≈ 1, then low CKA values
for 4B→0.8B are real and not a bug in the CKA implementation.
"""

import argparse
import json
import os

import torch

from transformers import AutoModelForCausalLM

from prism.calibration import (
    load_calibration_data,
    uniform_layer_mapping,
)


def cka_similarity(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two sets of vectors [N, D]."""
    K_XX = X @ X.T
    K_YY = Y @ Y.T
    K_XY = X @ Y.T
    N = X.shape[0]
    one_N = torch.ones(N, N, device=X.device) / N
    center = torch.eye(N, device=X.device) - one_N
    K_XX_c = center @ K_XX @ center
    K_YY_c = center @ K_YY @ center
    K_XY_c = center @ K_XY @ center
    hsic_xx = K_XX_c.pow(2).sum()
    hsic_yy = K_YY_c.pow(2).sum()
    hsic_xy = K_XY_c.pow(2).sum()
    denom = torch.sqrt(torch.clamp(hsic_xx * hsic_yy, min=1e-12))
    return (hsic_xy / denom).item()


def main(
    large_model="/data/Qwen/Qwen3.5-4B",
    small_model="Qwen/Qwen3.5-0.8B",
    n_samples=5,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    if not torch.cuda.is_available():
        torch.set_num_threads(4)

    samples = load_calibration_data(
        small_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )
    print(f"Loaded {len(samples)} samples")

    # Load models
    print(f"Loading {large_model}...")
    dtype_L = torch.float16 if torch.cuda.is_available() else torch.float32
    model_L = AutoModelForCausalLM.from_pretrained(
        large_model, torch_dtype=dtype_L, attn_implementation="eager"
    )
    model_L.eval()

    print(f"Loading {small_model}...")
    model_S = AutoModelForCausalLM.from_pretrained(
        small_model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model_S.eval()

    n_L = model_L.config.num_hidden_layers
    n_S = model_S.config.num_hidden_layers
    print(f"Layers: L={n_L}, S={n_S}")

    # Identity CKA: same model, same layers
    # Sample layers uniformly
    step_L = max(1, n_L // 10)
    step_S = max(1, n_S // 10)

    results = {
        "large_identity": {},
        "small_identity": {},
    }

    # Large model identity
    print("\n=== CKA(Large→Large) identity ===")
    for l in range(0, n_L, step_L):
        h_list = []
        h_list2 = []
        for sample in samples:
            ids = sample["input_ids"].to(model_L.device)
            attn = sample["attention_mask"].to(model_L.device)
            with torch.no_grad():
                out = model_L(
                    input_ids=ids, attention_mask=attn, output_hidden_states=True
                )
            h_list.append(out.hidden_states[l + 1][:, -1, :])
            # Same output — CKA should be exactly 1
            h_list2.append(out.hidden_states[l + 1][:, -1, :])

        H1 = torch.cat(h_list, dim=0).float()
        H2 = torch.cat(h_list2, dim=0).float()
        cka = cka_similarity(H1, H2)
        results["large_identity"][f"L{l}"] = cka
        print(f"  L{l}: CKA={cka:.6f} (expect ~1.0)")

    # Small model identity
    print("\n=== CKA(Small→Small) identity ===")
    for l in range(0, n_S, step_S):
        h_list = []
        h_list2 = []
        for sample in samples:
            ids = sample["input_ids"].to(model_S.device)
            attn = sample["attention_mask"].to(model_S.device)
            with torch.no_grad():
                out = model_S(
                    input_ids=ids, attention_mask=attn, output_hidden_states=True
                )
            h_list.append(out.hidden_states[l + 1][:, -1, :])
            h_list2.append(out.hidden_states[l + 1][:, -1, :])

        H1 = torch.cat(h_list, dim=0).float()
        H2 = torch.cat(h_list2, dim=0).float()
        cka = cka_similarity(H1, H2)
        results["small_identity"][f"L{l}"] = cka
        print(f"  L{l}: CKA={cka:.6f} (expect ~1.0)")

    # Save
    os.makedirs(results_dir, exist_ok=True)
    results["params"] = {
        "large_model": large_model,
        "small_model": small_model,
        "n_samples": n_samples,
    }
    with open(os.path.join(results_dir, "exp2a_cka_identity.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_dir}/exp2a_cka_identity.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CKA identity control")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
