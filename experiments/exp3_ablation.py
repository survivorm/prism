"""Experiment 3: Ablation — without Gramian correction.

Compare identity loss WITH pinv Gramian (should be ~0)
vs WITHOUT (should be ~0.5-0.8 for real embeddings).
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
from prism.bridge import (
    _project_to_vocab,
    _valid_mask,
    _weighted_cosine_loss,
)
from prism.bridge import jacobian_gap


def jacobian_gap_raw(
    J_L: torch.Tensor,
    J_S: torch.Tensor,
    E_L: torch.Tensor,
    E_S: torch.Tensor,
    probs_L: torch.Tensor,
    threshold: float = 1e-6,
) -> float:
    """Bridge WITHOUT Gramian correction — raw J @ E^T @ E."""
    J_L_proj = _project_to_vocab(J_L, E_L)
    J_S_target = J_L_proj @ E_S.float()
    mask = _valid_mask(J_S, J_S_target, threshold)
    if mask.sum() == 0:
        return 1.0
    return _weighted_cosine_loss(J_S, J_S_target, probs_L, mask)


def random_orthogonal_matrix(rows: int, cols: int, device, seed: int = 0) -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition.

    багфикс аудита (v6 F7): было `Q * torch.sign(d).clamp(min=0)` — clamp(min=0)
    превращает -1 в 0, из-за чего ~55 % столбцов обнулялись и QQᵀ ≠ I
    (max|QQᵀ−I| ≈ 0.71): «случайная ортогональная» матрица ортогональной не была.
    Канонический знак-фикс — Q ← Q·diag(sign d). Плюс явный seed: прежний RNG
    не сеялся, значит null-столбец карточки E-3 был невоспроизводим.
    """
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(rows, cols, generator=g).to(device=device, dtype=torch.float32)
    Q, R = torch.linalg.qr(M)
    return Q * torch.where(torch.diagonal(R) >= 0, 1.0, -1.0).to(Q.dtype)


def _check_orthogonal(n: int = 128) -> None:
    """Один запускаемый контроль на F7: QQᵀ обязан быть единичным."""
    Q = random_orthogonal_matrix(n, n, torch.device("cpu"))
    err = float((Q @ Q.T - torch.eye(n)).abs().max())
    assert err < 1e-4, f"QR sign-fix сломан: max|QQᵀ−I| = {err:.3e}"
    nz = float((Q != 0).float().mean())
    assert abs(nz - 1.0) < 1e-6, f"обнулённые столбцы: ненулевых {nz:.1%}"
    print(f"ok: max|QQᵀ−I| = {err:.2e}, ненулевых элементов {nz:.0%}")


def jacobian_gap_random(
    J: torch.Tensor,  # [K, D] — original jacobian at layer
    E: torch.Tensor,  # [V, D] — embeddings
    probs_L: torch.Tensor,
    threshold: float = 1e-6,
) -> float:
    """Bridge with RANDOM orthogonal R replacing learned W.

    Protocol:
      1. Build R: random orthogonal in R^{D×D} — детерминирована (seed=0), одна и та же
         на все слои, то есть null парный сравнению, а не перерисовывается на слой
      2. J_rand = J @ R^T
      3. Project both through vocab: J_proj = J @ E^T, J_rand_proj = J_rand @ E^T
      4. Compare cosine similarity
    """
    D = J.shape[1]
    R = random_orthogonal_matrix(D, D, J.device)
    J_rand = J @ R.T  # [K, D]
    mask = _valid_mask(J_rand, J, threshold)
    if mask.sum() == 0:
        return 1.0
    return _weighted_cosine_loss(J_rand, J, probs_L, mask)


def main(
    model="Qwen/Qwen3.5-0.8B",
    n_samples=1,
    K=8,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    torch.set_num_threads(4)
    print("Loading calibration data...")
    samples = load_calibration_data(
        model, max_length=max_length, min_length=min_length, n_samples=n_samples
    )
    print(f"Loaded {len(samples)} samples")

    print("Loading model...")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model_obj.eval()
    n_layers = model_obj.config.num_hidden_layers
    mapping = uniform_layer_mapping(n_layers, n_layers)

    token_set = build_global_token_set(model_obj, samples, K=K)
    E = extract_embeddings(model_obj)

    results_corrected = {}
    results_raw = {}
    results_random = {}

    print("Running ablation (corrected vs raw vs random)...")

    for l_L, l_S in mapping.items():
        losses_c, losses_r, losses_rand = [], [], []
        for sample in samples:
            ids = sample["input_ids"].to(model_obj.device)
            attn = sample["attention_mask"].to(model_obj.device)
            J = compute_jacobian_for_tokens(model_obj, l_L, ids, attn, token_set)

            with torch.no_grad():
                logits = model_obj(input_ids=ids).logits[:, -1, :]
                probs = torch.softmax(logits[0, token_set], dim=-1)

            l_c = jacobian_gap(J, J, E, E, probs)
            l_r = jacobian_gap_raw(J, J, E, E, probs)
            l_rand = jacobian_gap_random(J, E, probs)
            losses_c.append(l_c)
            losses_r.append(l_r)
            losses_rand.append(l_rand)

        results_corrected[f"{l_L}->{l_S}"] = sum(losses_c) / len(losses_c)
        results_raw[f"{l_L}->{l_S}"] = sum(losses_r) / len(losses_r)
        results_random[f"{l_L}->{l_S}"] = sum(losses_rand) / len(losses_rand)

    os.makedirs(results_dir, exist_ok=True)
    output = {
        "with_correction": results_corrected,
        "without_correction": results_raw,
        "random_orthogonal": results_random,
        "params": {
            "model": model,
            "n_samples": n_samples,
            "K": K,
            "max_length": max_length,
            "min_length": min_length,
        },
    }
    with open(os.path.join(results_dir, "exp3_ablation.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(
        f"\nMean corrected:  {sum(results_corrected.values()) / len(results_corrected):.8f}"
    )
    print(f"Mean without:    {sum(results_raw.values()) / len(results_raw):.8f}")
    print(f"Mean random:     {sum(results_random.values()) / len(results_random):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 3: Ablation")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--check-qr", action="store_true",
                        help="только контроль ортогональности null-матрицы (аудит v6 F7)")
    args = parser.parse_args()
    if args.check_qr:
        args = vars(args)
        args.pop("check_qr")
        _check_orthogonal()
        raise SystemExit(0)
    main(**vars(args))
