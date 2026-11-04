"""Experiment 12A: Perturbation test via sensitivity map.

For layers {6,12,18}:
  - v_g: right singular vector of J whose left vector ≈ uniform (common mode)
  - v_d: top right singular vector of J^(p) (differential direction)

Perturb h by ±ε·v at same norm. Measure KL(p_orig ‖ p_pert).
Prediction: KL_g ≪ KL_d (orders of magnitude).
"""

import argparse
import json
import os

import torch
from scipy.stats import entropy

from prism.calibration import load_calibration_data
from prism.jacobian import (
    compute_jacobian_for_tokens,
    softmax_jacobian,
    _run_prefix_pass,
    last_position_logits,
)


def kl_div(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> float:
    ps = (p + eps) / (p + eps).sum()
    qs = (q + eps) / (q + eps).sum()
    # .cpu() обязателен: на cuda-тензоре scipy .numpy() кидает
    return float(entropy(ps.detach().cpu().numpy(), qs.detach().cpu().numpy()))


def run_full_decoder(
    model, prefix_cache, h: torch.Tensor, S: int, layer_idx: int
) -> torch.Tensor:
    """Истинный хвост модели (путь prism, согласованный с якобианом).

    Аудит 2026-11-05, баг 2: прежняя копия декодировала h поверх кэша ПОЛНОГО
    forward (0..S-1) — двойной учёт последнего токена, KL(δ=0) ≈ 7–9 натов.
    prefix_cache обязан быть из _run_prefix_pass ([:, :-1]).
    """
    with torch.no_grad():
        logits = last_position_logits(model, h, S, layer_idx, prefix_cache)
    return torch.softmax(logits[0].float(), dim=-1)


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=8,
    K=64,
    max_length=64,
    min_length=32,
    eps_list=None,
    results_dir="results",
    device="cpu",
):
    torch.set_num_threads(4)
    torch.manual_seed(42)

    if eps_list is None:
        eps_list = [0.01, 0.1, 0.5, 1.0]

    samples = load_calibration_data(model, max_length, min_length, n_samples)

    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    m.eval()
    m.to(device)  # Kaggle-GPU: драйвер ядра передаёт device="cuda:0"

    selected = [6, 12, 18]

    s0 = samples[0]
    ids0 = s0["input_ids"].to(m.device)
    attn0 = s0["attention_mask"].to(m.device)
    with torch.no_grad():
        lg0 = m(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    top_k = lg0.topk(K, dim=-1).indices[0].tolist()
    token_set = sorted(top_k)

    all_results = {}
    _jcache: dict[tuple[int, int], torch.Tensor] = {}  # J не зависит от eps (аудит: был в eps-цикле ×4)

    for eps_val in eps_list:
        eps_key = f"eps_{eps_val}"
        print(f"\n{'='*60}")
        print(f"Epsilon = {eps_val}")
        print(f"{'='*60}")
        eps_results = {}

        for layer in selected:
            print(f"\nLayer {layer}:")
            layer_results = []

            for si, sample in enumerate(samples):
                ids = sample["input_ids"].to(m.device)
                attn = sample["attention_mask"].to(m.device)
                S = ids.shape[1]

                _k = (layer, si)
                if _k not in _jcache:
                    _jcache[_k] = compute_jacobian_for_tokens(m, layer, ids, attn, token_set)
                J = _jcache[_k]

                # Forward for baseline
                with torch.no_grad():
                    out = m(
                        input_ids=ids,
                        attention_mask=attn,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    # кэш 0..S-2 (аудит 2026-11-05: полный кэш = двойной учёт S-1)
                prefix_cache = _run_prefix_pass(m, ids, attn)
                h_base = out.hidden_states[layer + 1][:, -1:, :]  # [1,1,D]
                # ЯКОРЬ = истинный выход модели z(h_base) (tail+norm+head), а не
                # срезанный lm_head(norm(h_base)): на слоях 6/12/18 второй даёт
                # KL≈8.9 ната при δ=0, тогда как run_full_decoder (ground truth)
                # и J (полный дека-якобиан) согласованы с out.logits. Аудит
                # 2026-11-05: абсолютные KL_g/KL_d были загрязщены константой
                # якоря (симметрия KL(±) и ratio_lin выживали, модуль — нет).
                p_orig = torch.softmax(out.logits[:, -1, :][0].detach(), dim=-1)

                # p for J^(p)
                p_tokens = torch.softmax(out.logits[0, -1, token_set], dim=-1)
                J_p = softmax_jacobian(J, p_tokens)

                # SVD of J → find gauge direction (left vector ≈ uniform)
                U_j, S_j, Vt_j = torch.linalg.svd(J.float(), full_matrices=False)
                ones = torch.ones(J.shape[0], device=J.device) / torch.sqrt(
                    torch.tensor(float(J.shape[0]), device=J.device)
                )
                uniformity = torch.abs(U_j.T @ ones)  # [min(K,D)]
                gauge_idx = uniformity.argmax()
                v_g = Vt_j[gauge_idx]
                sigma_g = float(S_j[gauge_idx])

                # Top right singular vector of J^(p)
                _, S_p, Vt_p = torch.linalg.svd(J_p.float(), full_matrices=False)
                v_d = Vt_p[0]
                sigma_d = float(S_p[0])

                # Normalize
                v_g = v_g / v_g.norm()
                v_d = v_d / v_d.norm()

                # First-order predictor, L2-on-δp metric: KL_lin(v) = ½ ‖J^(p) v‖²
                # ⚠ НЕ равен ½ Var_p(Jv): ‖M_p z‖² = zᵀM_p²z (веса [p(1−p)]², top-K
                # окно), тогда как истинная квадратичная форма KL = ½ zᵀM_p z =
                # ½ Var_p(z) (M_p D⁻¹ M_p = M_p). В gauge-направлении L2-версия
                # недооценивает KL в κ_g≈19–28×, в differential в κ_d≈0.2–3.4×
                # (аудит v4.1: ratio_lin 7–59× — артефакт метрики, в KL-метрике ≈1×).
                # [L-fisher]; compares the full-decoder KL below with the linear term
                kl_lin_g = 0.5 * float((J_p @ v_g).norm() ** 2)
                kl_lin_d = 0.5 * float((J_p @ v_d).norm() ** 2)
                # истинная KL-метрика: ½ Var_p(J v) по полному словарю недоступна
                # (J строится на top-K), считаем по top-K с p-весами:
                _zg = J @ v_g
                _zd = J @ v_d
                kl_fisher_g = 0.5 * float(
                    (p_tokens * _zg * _zg).sum() - (p_tokens * _zg).sum() ** 2
                )
                kl_fisher_d = 0.5 * float(
                    (p_tokens * _zd * _zd).sum() - (p_tokens * _zd).sum() ** 2
                )
                h_norm = float(h_base.norm())

                results_g = {}
                results_d = {}

                for sign in [-1, +1]:
                    delta_g = sign * eps_val * v_g
                    h_g = h_base + delta_g
                    p_g = run_full_decoder(m, prefix_cache, h_g, S, layer)
                    results_g[sign] = kl_div(p_orig, p_g)

                    delta_d = sign * eps_val * v_d
                    h_d = h_base + delta_d
                    p_d = run_full_decoder(m, prefix_cache, h_d, S, layer)
                    results_d[sign] = kl_div(p_orig, p_d)

                entry = {
                    "layer": layer,
                    "sample": si,
                    "uniformity_vg": uniformity[gauge_idx].item(),
                    "sigma_g": sigma_g,
                    "sigma_d": sigma_d,
                    "norm_h": h_norm,
                    "KL_lin_g": kl_lin_g,
                    "KL_lin_d": kl_lin_d,
                    "KL_fisher_g": kl_fisher_g,
                    "KL_fisher_d": kl_fisher_d,
                    "ratio_lin": kl_lin_d / (kl_lin_g + 1e-30),
                    "KL_g_minus": results_g[-1],
                    "KL_g_plus": results_g[+1],
                    "KL_d_minus": results_d[-1],
                    "KL_d_plus": results_d[+1],
                }
                layer_results.append(entry)
                print(
                    f"  sample {si}: KL_g={[round(v, 4) for v in results_g.values()]}, KL_d={[round(v, 4) for v in results_d.values()]}"
                )

            eps_results[layer] = layer_results

            avg_klg = sum(r["KL_g_plus"] for r in layer_results) / len(layer_results)
            avg_kld = sum(r["KL_d_plus"] for r in layer_results) / len(layer_results)
            print(
                f"  avg: KL_g+={avg_klg:.4f}, KL_d+={avg_kld:.4f}, ratio={avg_kld / (avg_klg + 1e-8):.1f}x"
            )

        all_results[eps_key] = eps_results

    os.makedirs(results_dir, exist_ok=True)
    all_results["params"] = {"model": model, "n_samples": n_samples, "K": K, "max_length": max_length, "min_length": min_length, "eps_list": eps_list,
                             "note": "round3: added sigma_g/sigma_d, norm_h, KL_lin_* (first-order predictor ½‖J^(p)v‖²)"}
    with open(os.path.join(results_dir, "exp12a_perturbation_test.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_dir}/exp12a_perturbation_test.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 12A: Perturbation test")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--eps-list", default="0.01,0.1,0.5,1.0",
                        help="Comma-separated epsilon values, e.g. 0.01,0.1,0.5,1.0")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    args.eps_list = [float(x) for x in args.eps_list.split(",")]
    main(**vars(args))
