"""Experiment 11A: Distribution fidelity with/without defragmentation.

For layers {6, 12, 18}, perturb hidden state along gauge/semantic/random directions.
Compare KL(p_true || p_hat) for:
  - raw: p = softmax(z0 + J @ delta)
  - gaugefree: p = p0 + J^(p) @ delta
  - defrag_r: p = p0 + U_r @ (J_r @ delta), r in {4, 8, 16, 32}

Ground truth: full forward at h + delta.
"""

import argparse
import json
import os

import torch
from scipy.stats import entropy

from prism.calibration import load_calibration_data
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    """KL(p || q) with epsilon smoothing."""
    p_s = (p + eps) / (p + eps).sum()
    q_s = (q + eps) / (q + eps).sum()
    return float(entropy(p_s.detach().numpy(), q_s.detach().numpy()))


def tv_distance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Total variation distance."""
    return float(0.5 * torch.abs(p - q).sum().item())


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    a_n = a / (a.norm() + 1e-10)
    b_n = b / (b.norm() + 1e-10)
    return float((a_n @ b_n).item())


def top_k_agreement(p_true: torch.Tensor, p_hat: torch.Tensor, k: int) -> float:
    """Fraction of top-k tokens that agree."""
    true_topk = p_true.topk(k).indices
    pred_topk = p_hat.topk(k).indices
    return (
        float((true_topk.unsqueeze(0) == pred_topk.unsqueeze(1)).any(dim=0).sum()) / k
    )


def run_full_decoder(
    model, prefix_cache, h: torch.Tensor, S: int, layer_idx: int
) -> torch.Tensor:
    """Run h through remaining layers with cache, return probs."""
    from transformers.cache_utils import DynamicLayer, LinearAttentionLayer
    from transformers.masking_utils import (  # type: ignore[import-not-found]
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    # Clone cache
    from transformers import DynamicCache

    cache_clone = DynamicCache()
    for idx in range(len(prefix_cache.layers)):
        layer = prefix_cache.layers[idx]
        if isinstance(layer, DynamicLayer):
            new_k = [k.clone() if k is not None else None for k in layer.keys]
            new_v = [v.clone() if v is not None else None for v in layer.values]
            cache_clone.layers.append(DynamicLayer(keys=new_k, values=new_v))
        else:
            new_conv = {k: v.clone() for k, v in layer.conv_states.items()}
            new_recur = {k: v.clone() for k, v in layer.recurrent_states.items()}
            cache_clone.layers.append(
                LinearAttentionLayer(conv_states=new_conv, recurrent_states=new_recur)
            )

    text_model = model.model
    pos = torch.full((4, 1, 1), S - 1, dtype=torch.long, device=h.device)
    text_pos = pos[0]
    rope_pos = pos[1:]
    pos_emb = text_model.rotary_emb(h, rope_pos)

    # Build masks
    dummy = h.new_zeros(1, 1, text_model.config.hidden_size)
    mask_kwargs = {
        "config": text_model.config,
        "inputs_embeds": dummy,
        "attention_mask": None,
        "past_key_values": cache_clone,
        "position_ids": text_pos,
    }
    masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
    }

    x = h
    n_layers = model.config.num_hidden_layers
    for i in range(layer_idx + 1, n_layers):
        layer_type = model.config.layer_types[i]
        x = text_model.layers[i](
            x,
            position_embeddings=pos_emb,
            attention_mask=masks[layer_type],
            position_ids=text_pos,
            past_key_values=cache_clone,
        )

    x = text_model.norm(x)
    logits = model.lm_head(x[:, -1, :])
    return torch.softmax(logits[0], dim=-1)


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=4,
    K=64,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    torch.set_num_threads(4)
    torch.manual_seed(42)

    print(f"Loading {n_samples} samples...")
    samples = load_calibration_data(model, max_length, min_length, n_samples)

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    m.eval()

    selected = [6, 12, 18]
    print(f"Layers: {selected}")

    # Token set
    s0 = samples[0]
    ids0 = s0["input_ids"].to(m.device)
    attn0 = s0["attention_mask"].to(m.device)
    with torch.no_grad():
        lg0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = lg0.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    D = m.config.hidden_size
    scales = [0.5, 1.0, 2.0]
    rs = [4, 8, 16, 32]

    all_results = {}

    for layer in selected:
        print(f"\nLayer {layer}:")
        layer_results = []

        for si, sample in enumerate(samples):
            ids = sample["input_ids"]
            attn = sample["attention_mask"]
            S = ids.shape[1]

            # Compute Jacobian
            J = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)
            # J shape: [K, D]

            # Extract hidden state for full forward
            with torch.no_grad():
                out = m(
                    input_ids=ids,
                    attention_mask=attn,
                    output_hidden_states=True,
                    use_cache=True,
                )
            prefix_cache = out.past_key_values
            h_base = out.hidden_states[layer + 1][:, -1:, :]  # [1, 1, D]
            z0 = m.lm_head(m.model.norm(h_base))[:, -1, :]  # [1, V]
            p0 = torch.softmax(z0[0], dim=-1)

            # J^(p) gauge-free — last-token logits on token_set
            p_tokens = torch.softmax(out.logits[0, -1, token_set], dim=-1)
            J_p = softmax_jacobian(J, p_tokens)

            # Defrag basis: top-r eigenvectors of G^(p) = J_p @ J_p.T
            # For a single sample, G^(p) is rank-J_p
            G_p = J_p @ J_p.T
            ev, eig = torch.linalg.eigh(G_p)
            idx = ev.argsort(descending=True)
            U_all = eig[:, idx]  # [K, K] eigenvectors

            # Generate perturbations
            perturbations = []

            # 1. Gauge direction: J^T @ 1
            delta_g = J.sum(dim=0)  # [D]
            perturbations.append(("gauge", delta_g))

            # 2. Semantic directions (top-5 eigenvectors of G^(p))
            for ei in range(min(5, K)):
                v = U_all[:, ei].unsqueeze(1)  # [K, 1]
                # delta = J_p^T @ v
                delta = (J_p.T @ v).squeeze(1)  # [D]
                perturbations.append((f"sem_{ei}", delta))

            # 3. Random gaussian
            for ri in range(3):
                delta = torch.randn(D, device=h_base.device)
                perturbations.append((f"rand_{ri}", delta))

            for pname, delta in perturbations:
                # Normalize to target scale
                J_delta = J @ delta  # [K]
                norm = J_delta.norm()
                if norm < 1e-6:
                    continue

                scale_factors = {s: s / norm for s in scales}

                for s, factor in scale_factors.items():
                    delta_scaled = delta * factor

                    # Ground truth: full forward
                    h_pert = h_base + delta_scaled  # [1, 1, D]
                    with torch.no_grad():
                        p_true = run_full_decoder(m, prefix_cache, h_pert, S, layer)

                    # Predictor 1: raw (update only token_set positions)
                    z_pert = z0.clone()
                    z_pert[0, token_set] += J @ delta_scaled
                    p_raw = torch.softmax(z_pert[0], dim=-1)

                    # Predictor 2: gaugefree (in p-space)
                    dp_gf = J_p @ delta_scaled  # [K]
                    # Extend to full V via top-k only
                    p_gf = p0.clone()
                    p_gf[token_set] += dp_gf
                    p_gf = torch.softmax(p_gf, dim=-1)

                    # Predictor 3: defrag_r
                    entry = {
                        "layer": layer,
                        "sample": si,
                        "perturbation": pname,
                        "scale": s,
                        "KL_true_raw": kl_divergence(p_true, p_raw),
                        "KL_true_gf": kl_divergence(p_true, p_gf),
                        "TV_true_raw": tv_distance(p_true, p_raw),
                        "cos_true_raw": cosine_similarity(p_true - p0, p_raw - p0),
                        "top1_raw": top_k_agreement(p_true, p_raw, 1),
                        "top5_raw": top_k_agreement(p_true, p_raw, 5),
                    }

                    for r in rs:
                        U_r = U_all[:, :r]  # [K, r]
                        # J_r = U_r^T @ J_p  (r, D)
                        J_r = U_r.T @ J_p  # [r, D]
                        # Reconstruction: J_tilde_p = U_r @ J_r
                        J_tilde_p = U_r @ J_r  # [K, D]
                        dp_dr = J_tilde_p @ delta_scaled  # [K]
                        p_dr = p0.clone()
                        p_dr[token_set] += dp_dr
                        p_dr = torch.softmax(p_dr, dim=-1)

                        entry[f"KL_true_dr{r}"] = kl_divergence(p_true, p_dr)
                        entry[f"cos_true_dr{r}"] = cosine_similarity(
                            p_true - p0, p_dr - p0
                        )

                    layer_results.append(entry)

        all_results[layer] = layer_results

        # Summary
        for pname in ["gauge", "sem_0"]:
            for s in scales:
                kl_raws = [
                    r["KL_true_raw"]
                    for r in layer_results
                    if r["perturbation"] == pname and r["scale"] == s
                ]
                kl_gfs = [
                    r["KL_true_gf"]
                    for r in layer_results
                    if r["perturbation"] == pname and r["scale"] == s
                ]
                print(
                    f"  {pname} s={s}: KL(raw)={sum(kl_raws) / len(kl_raws):.4f}, KL(gf)={sum(kl_gfs) / len(kl_gfs):.4f}"
                )

    os.makedirs(results_dir, exist_ok=True)
    all_results["params"] = {model: model, n_samples: n_samples, K: K, max_length: max_length, min_length: min_length}
    with open(os.path.join(results_dir, "exp11a_defrag_fidelity.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_dir}/exp11a_defrag_fidelity.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 11A: Defrag fidelity")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
