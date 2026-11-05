"""Experiment 7: CKA (Centered Kernel Alignment) baseline between hidden states.

Computes CKA between hidden states of 4B and 0.8B models across multiple samples,
then compares with Jacobian gap (1-cos) on the same layer pairs.
"""

import argparse
import json
import os

import torch

from transformers import AutoModelForCausalLM

from prism.calibration import (
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.bridge import jacobian_gap
from prism.jacobian import compute_jacobian_for_tokens


def cka_similarity(
    X: torch.Tensor,
    Y: torch.Tensor,
    form: str = "standard",
) -> float:
    """Линейный CKA двух наборов векторов [N, D] — двух форм, нарочно раздельно.

    `standard` (то и есть CKA,Kernel Alignment с линейным ядром):
        CKA = tr(K_c L_c) / sqrt(tr(K_c²) tr(L_c²)),   K = XXᵀ, L = YYᵀ,  M_c = H M H,
    работает и при разных размерностях представлений (4B D=2560 против 0.8B D=1024),
    потому что оба грама — N×N.

    `legacy` — то, что в этом репозитории считалось «CKA» до 2026-11-05:
        ‖H (XYᵀ) H‖²_F / (‖H K H‖_F ‖H L H‖_F).
    Это НЕ CKA: числитель — норма Фробениуса перекрёстного грама, а не tr(K_c L_c). На
    независимых матрицах формы расходятся на два порядка (N=5, D=1024: legacy ≈ 0.0035,
    standard ≈ 0.995), поэтому числа, записанные legacy-формой, нельзя сравнивать с нулём
    ни от какой публикации. Числа `results/exp7_cka_baseline.json` (0.0003–0.0007) и
    `results/exp2a_cka_identity.json` сняты именно legacy — см. --self-check.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"CKA требует равного числа объектов: N={X.shape[0]} против M={Y.shape[0]} "
            "(матрица центрирования одна)"
        )
    N = X.shape[0]
    C = torch.eye(N, device=X.device, dtype=torch.float64) - torch.ones(N, N, device=X.device, dtype=torch.float64) / N
    Kc = C @ (X.double() @ X.T.double()) @ C
    Lc = C @ (Y.double() @ Y.T.double()) @ C
    if form == "standard":
        num = (Kc * Lc).sum()
        den = ((Kc * Kc).sum() * (Lc * Lc).sum()).sqrt()
    elif form == "legacy":
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"legacy-форма требует равной размерности (X @ Yᵀ): D={X.shape[1]} против {Y.shape[1]}"
            )
        Xy = C @ (X.double() @ Y.T.double()) @ C
        num = Xy.pow(2).sum()
        den = (Kc.pow(2).sum() * Lc.pow(2).sum()).sqrt()
    else:
        raise ValueError(f"неизвестная форма CKA: {form}")
    return (num / torch.clamp(den, min=1e-12)).item()


def self_check(seed: int = 0, reps: int = 200, results_dir: str = "results") -> dict:
    """Механика обеих форм CKA + нуль и положительный контроль на тех формах, что цитируют docs.

    Найдено при разборе R3 (2026-11-05). Два независимых нахождения:

    1. Реализация в репозитории до этой правки считала НЕ CKA: числитель ``||H (X Y^T) H||^2_F``
       вместо ``tr(K_c L_c)``. На независимых матрицах при N=5, D=1024 это 0.0035 вместо 0.995,
       то есть числящиеся в ``results/exp7_cka_baseline.json`` 0.0003-0.0007 — величина другой
       функциональности, и сравнивать её с литературным нулём нельзя.
    2. Оговорка в карточке E-7 («нуль ~ 0.34») снята с матриц 64x128. На настоящей форме E-7
       (N=5 объектов, D=2560x1024) настоящий CKA вырождается: при N<<D центрированные граммы
       концентрируются к матрице центрирования H, поэтому и нуль, и положительный контроль дают
       ~1 — динамического диапазона нет вовсе.

    Отсюда: вывод E-7 «CKA ~ 0 => представления не линейно скоррелированы => мост измеряет
    новое» опирался только на legacy-функцию; «новое» держится на независимых признаках моста
    (монотонность по глубине, сходимость по K >= 32, отсутствие корреляции с послойным KL),
    и в docs он должен стоять именно на них.
    """
    torch.manual_seed(seed)
    out: dict = {"identity": {}, "null": {}, "control": {}}

    def med(vals):
        vals = sorted(vals)
        return vals[len(vals) // 2]

    for form in ("standard", "legacy"):
        X = torch.randn(5, 1024)
        out["identity"][form] = {
            "identity": cka_similarity(X, X.clone(), form=form),
            "scale": cka_similarity(X, 3.7 * X, form=form),
        }
        assert out["identity"][form]["identity"] > 0.999, form
        for tag, (n, da, db) in (
            ("64x128", (64, 128, 128)),
            ("e7_5_2560x1024", (5, 2560, 1024)),
            ("e7_5_1024x1024", (5, 1024, 1024)),
            ("n2_1024", (2, 1024, 1024)),
        ):
            if form == "legacy" and da != db:
                # legacy-форма не умеет разных размерностей вообще: то есть записанные в
                # exp7_cka_baseline.json cross-D числа (4B 2560 против 0.8B 1024) сделаны НЕ
                # этим кодом — ни legacy-, ни standard-версией на HEAD. Их происхождение в
                # истории репозитория не восстановлено, и цитировать их как CKA нельзя.
                continue
            out["null"][f"{form}/{tag}"] = med(
                [cka_similarity(torch.randn(n, da), torch.randn(n, db), form=form) for _ in range(reps)]
            )
            # положительный контроль: тот же объект, другое представление (X @ W)
            ctl = []
            for _ in range(reps):
                A = torch.randn(n, da)
                W = torch.randn(da, db) / max(da, db) ** 0.5
                ctl.append(cka_similarity(A, A @ W, form=form))
            out["control"][f"{form}/{tag}"] = med(ctl)
    # N=2 в legacy-форме вырождается в cos^2 между разностными векторами: именно поэтому
    # «identity-контроль» exp2a на группах по 2 объекта дал 1e-5…1e-2, а не ~1
    A, B = torch.randn(2, 1024), torch.randn(2, 1024)
    cos2 = torch.nn.functional.cosine_similarity((A[0] - A[1])[None], (B[0] - B[1])[None]).pow(2).item()
    out["n2_legacy_is_cos2_abs_err"] = abs(cka_similarity(A, B, form="legacy") - cos2)
    assert out["n2_legacy_is_cos2_abs_err"] < 1e-9
    out["note"] = (
        "стандартный CKA при N<<D даёт ~1 и на нуле, и на контроле => метрика не имеет диапазона; "
        "legacy-форма — не CKA"
    )
    print("self-check:", json.dumps(out, indent=2, ensure_ascii=False))
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "exp7_cka_selfcheck.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out


def main(
    large_model="/data/Qwen/Qwen3.5-4B",
    small_model="Qwen/Qwen3.5-0.8B",
    n_samples=5,
    K=8,
    max_length=64,
    min_length=32,
    results_dir="results",
):
    torch.set_num_threads(4)
    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        large_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )
    print(f"Loaded {len(samples)} samples")

    print(f"Loading {large_model}...")
    model_L = AutoModelForCausalLM.from_pretrained(
        large_model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model_L.eval()

    print(f"Loading {small_model}...")
    model_S = AutoModelForCausalLM.from_pretrained(
        small_model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model_S.eval()

    n_L = model_L.config.num_hidden_layers
    n_S = model_S.config.num_hidden_layers
    mapping = uniform_layer_mapping(n_L, n_S)

    # Sample layers (every N-th) to keep runtime manageable
    step = max(1, len(mapping) // 10)
    sampled = list(mapping.items())[::step]
    print(f"Testing {len(sampled)} layer pairs")

    # Collect hidden states for CKA across all samples
    print("Collecting hidden states for CKA...")
    cka_data = {}

    for l_L, l_S in sampled:
        h_L_list: list[torch.Tensor] = []
        h_S_list: list[torch.Tensor] = []

        for sample in samples:
            ids = sample["input_ids"]
            attn = sample["attention_mask"]

            with torch.no_grad():
                out_L = model_L(
                    input_ids=ids.to(model_L.device),
                    attention_mask=attn.to(model_L.device),
                    output_hidden_states=True,
                )
                out_S = model_S(
                    input_ids=ids.to(model_S.device),
                    attention_mask=attn.to(model_S.device),
                    output_hidden_states=True,
                )

            # Last token hidden state per sample
            h_L_list.append(out_L.hidden_states[l_L + 1][:, -1, :])
            h_S_list.append(out_S.hidden_states[l_S + 1][:, -1, :])

        H_L = torch.cat(h_L_list, dim=0)  # [N, D_L]
        H_S = torch.cat(h_S_list, dim=0)  # [N, D_S]

        cka = cka_similarity(H_L.float(), H_S.float())
        cka_data[f"{l_L}->{l_S}"] = cka
        print(f"  {l_L}->{l_S}: CKA={cka:.6f}")

    # Jacobian gap on same layers (single sample)
    print("\nComputing Jacobian gap...")
    E_L = extract_embeddings(model_L)
    E_S = extract_embeddings(model_S)

    sample = samples[0]
    ids = sample["input_ids"].to(model_L.device)
    attn = sample["attention_mask"].to(model_L.device)

    with torch.no_grad():
        logits_L = model_L(input_ids=ids, attention_mask=attn).logits[:, -1, :]
    top_k = logits_L.topk(K, dim=-1).indices[0].tolist()
    probs = torch.softmax(logits_L[0, top_k], dim=-1)
    token_set = sorted(top_k)

    gap_data = {}

    for l_L, l_S in sampled:
        J_L = compute_jacobian_for_tokens(model_L, l_L, ids, attn, token_set)
        J_S = compute_jacobian_for_tokens(model_S, l_S, ids, attn, token_set)
        loss = jacobian_gap(J_L, J_S, E_L, E_S, probs)
        gap_data[f"{l_L}->{l_S}"] = loss
        print(f"  {l_L}->{l_S}  : gap={loss:.6f}")

    # Save results + correlation
    os.makedirs(results_dir, exist_ok=True)
    import scipy

    cka_vals = list(cka_data.values())
    gap_vals = list(gap_data.values())
    corr = scipy.stats.spearmanr(cka_vals, gap_vals).correlation

    output = {
        "cka": cka_data,
        "gap": gap_data,
        "spearman_corr": corr,
        "n_samples": len(samples),
        "params": {
            "large_model": large_model,
            "small_model": small_model,
            "n_samples": n_samples,
            "K": K,
            "max_length": max_length,
            "min_length": min_length,
        },
    }
    with open(os.path.join(results_dir, "exp7_cka_baseline.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSpearman correlation (CKA vs gap): {corr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 7: CKA baseline")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--self-check", action="store_true", help="проверить механику метрики и нули, модели не нужны")
    args = parser.parse_args()
    kw = vars(args)
    if kw.pop("self_check"):  # в main такого параметра нет — иначе TypeError (урок аудита v6 F3)
        self_check()
    else:
        main(**kw)
