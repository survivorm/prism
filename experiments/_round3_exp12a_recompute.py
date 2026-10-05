"""Round-3 recompute of E-12a: add first-order predictor + spectral diagnostics.

Same protocol as experiments/exp12a_perturbation_test.py (seed 42, 4 samples,
K=64, layers {6,12,18}, eps=1.0, full decoder), plus per-sample:
  sigma_g, sigma_d, norm_h, KL_lin_g, KL_lin_d (= ½‖J^(p) v‖², [L-fisher]).
Writes results/exp12a_perturbation_test.json on success only.
"""

import json
import os

import torch

from experiments.exp12a_perturbation_test import (
    kl_div,
    load_calibration_data,
    run_full_decoder,
)
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian


def main():
    model_name = "Qwen/Qwen3.5-0.8B"
    torch.set_num_threads(4)
    torch.manual_seed(42)

    samples = load_calibration_data(model_name, 64, 32, 8)

    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager"
    )
    m.eval()

    s0 = samples[0]
    with torch.no_grad():
        lg0 = m(input_ids=s0["input_ids"].to(m.device),
                attention_mask=s0["attention_mask"].to(m.device)).logits[:, -1, :]
    token_set = sorted(lg0.topk(64, dim=-1).indices[0].tolist())

    all_results = {}
    for layer in [6, 12, 18]:
        print(f"Layer {layer}:", flush=True)
        layer_results = []
        for si, sample in enumerate(samples):
            ids = sample["input_ids"]
            attn = sample["attention_mask"]
            S = ids.shape[1]

            J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)

            with torch.no_grad():
                out = m(input_ids=ids, attention_mask=attn,
                        output_hidden_states=True, use_cache=True)
            prefix_cache = out.past_key_values
            h_base = out.hidden_states[layer + 1][:, -1:, :]
            p_orig = torch.softmax(
                m.lm_head(m.model.norm(h_base))[:, -1, :][0], dim=-1)

            p_tokens = torch.softmax(out.logits[0, -1, token_set], dim=-1)
            J_p = softmax_jacobian(J, p_tokens)

            U_j, S_j, Vt_j = torch.linalg.svd(J.float(), full_matrices=False)
            ones = torch.ones(J.shape[0]) / torch.sqrt(torch.tensor(float(J.shape[0])))
            uniformity = torch.abs(U_j.T @ ones)
            gauge_idx = uniformity.argmax()
            v_g = Vt_j[gauge_idx] / Vt_j[gauge_idx].norm()
            sigma_g = float(S_j[gauge_idx])

            _, S_p, Vt_p = torch.linalg.svd(J_p.float(), full_matrices=False)
            v_d = Vt_p[0] / Vt_p[0].norm()
            sigma_d = float(S_p[0])

            kl_lin_g = 0.5 * float((J_p @ v_g).norm() ** 2)
            kl_lin_d = 0.5 * float((J_p @ v_d).norm() ** 2)
            h_norm = float(h_base.norm())

            entry = {
                "layer": layer, "sample": si,
                "uniformity_vg": float(uniformity[gauge_idx]),
                "sigma_g": sigma_g, "sigma_d": sigma_d, "norm_h": h_norm,
                "KL_lin_g": kl_lin_g, "KL_lin_d": kl_lin_d,
                "ratio_lin": kl_lin_d / (kl_lin_g + 1e-30),
            }
            for sign in [-1, +1]:
                p_g = run_full_decoder(m, prefix_cache, h_base + sign * v_g, S, layer)
                p_d = run_full_decoder(m, prefix_cache, h_base + sign * v_d, S, layer)
                entry["KL_g_minus" if sign < 0 else "KL_g_plus"] = kl_div(p_orig, p_g)
                entry["KL_d_minus" if sign < 0 else "KL_d_plus"] = kl_div(p_orig, p_d)
            layer_results.append(entry)
            print(f"  sample {si}: sig_g={sigma_g:.2f} sig_d={sigma_d:.2f} "
                  f"||h||={h_norm:.1f} ratio_lin={entry['ratio_lin']:.1f} "
                  f"KL_g={[round(entry['KL_g_minus'], 3), round(entry['KL_g_plus'], 3)]} "
                  f"KL_d={[round(entry['KL_d_minus'], 3), round(entry['KL_d_plus'], 3)]}",
                  flush=True)
        all_results.setdefault("eps_1.0", {})[str(layer)] = layer_results

    all_results["params"] = {
        "model": model_name, "n_samples": 8, "K": 64,
        "max_length": 64, "min_length": 32, "eps_list": [1.0],
        "note": "round3 recompute: added sigma_g/sigma_d, norm_h, KL_lin_* "
                "(first-order predictor ½||J^(p) v||^2); KL values unchanged protocol",
    }
    with open("results/exp12a_perturbation_test.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
