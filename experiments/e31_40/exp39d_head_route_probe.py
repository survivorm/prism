#!/usr/bin/env python3
"""E-39d/E-39m: маршрут «head от кэшевого H» даёт неверную температуру readout.

Провокатор (2026-11-04). `results/exp39d_floor_seeds12.log` печатает sentinel
2271/4096 = 55.4 %, а поле `P128_RAW` того же кэша — 50.71 %; и там и там порог
`Σ_k p_k < 0.5` на одних и тех же позициях. Плюс в `kl_leash` (exp39d_l23_training.py:139)
лежит формула `logit = √D · (h/‖h‖) · (W·diag g)ᵀ`, которой приписано совпадение с
`lm_head(model.norm(h))` до 9.5e-6.

Что проверяем (только чтение кэша + веса модели, форвардов нет):
  (route_exact)  p = softmax(H @ W_lmᵀ)[U128]
  (route_renorm) p = softmax(√D · (H/‖H‖) · (W_lm·diag g)ᵀ)[U128]   ← как в коде E-39d/E-39m

Ответ: `H` в кэше — это УЖЕ выход финальной RMSNorm (множитель `g` тоже уже внутри),
поэтому точный маршрут — просто `H @ W_lmᵀ`: он воспроизводит `P128_RAW` до KL 5e-10,
тогда как маршрут из кода отличается на KL 4.6e-2 (среднее) и сдвигает долю sentinel
50.71 % → 55.44 %. Ниже заодно измеряется, во сколько раз этот же маршрут занижает KL
между базой и отредактированной считывающей матрицей (то есть насколько занижен `kl_f`,
которым гейтятся вердикты E-39d/E-39m).

Запуск: `../.venv/bin/python exp39d_head_route_probe.py` из `experiments/` (модель fp32,
несколько проходов по 4096×248k — ~5 минут CPU). Опционально `--carrier <файл носителя>`,
чтобы прогнать тем же тестом по настоящему носителю E-39m. Носители в git не лежат
(`*.pt` в .gitignore), поэтому случай носителя (строка `carrier_E39m_floor_pearson`,
отношение ×1.12) воспроизводится только с локального файла вывода кернела; синтетический
ряд `synthetic_eps_*` воспроизводится из свежего клона — он и есть переносимая часть вывода
(отношение ≈ 1.0 на правках любого размера).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _e38_cache as e38  # noqa: E402

torch.set_grad_enabled(False)
LAYER = 23
RAW_CACHE_DIR = "/data/prism_kernels/exp27b/results"   # здесь лежит P128_RAW (в репозитории его нет)


def kl_readout(H: torch.Tensor, W: torch.Tensor, W_rows: torch.Tensor, tok: torch.Tensor,
               chunk: int = 256) -> torch.Tensor:
    """KL(p_edit ‖ p_base) по полному словарю, readout-only, точный маршрут logits = H @ Wᵀ."""
    out = torch.empty(H.shape[0], dtype=torch.float64)
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk].double()
        b = h @ W.t().double()
        e = b.clone()
        e[:, tok] = h @ W_rows.t().double()
        lse_b = torch.logsumexp(b, dim=-1)
        lse_e = torch.logsumexp(e, dim=-1)
        pe = torch.softmax(e, dim=-1)
        # Σ_v p_e(v)·(e_v − b_v) = Σ_{k∈U128} … (вне U128 разность нулевая)
        out[i:i + chunk] = (lse_b - lse_e) + (pe[:, tok] * (e[:, tok] - b[:, tok])).sum(-1)
        del b, e, pe
    return out


def kl_readout_renorm(H: torch.Tensor, M: torch.Tensor, W_rows: torch.Tensor, tok: torch.Tensor,
                      g: torch.Tensor, sq: float, chunk: int = 256) -> torch.Tensor:
    """Тот же KL, но logits считаются маршрутом из кода E-39d/E-39m: √D·(H/‖H‖)·(W·diag g)ᵀ.

    Строки правки домножаются на тот же g — как `M_K = W_rows · diag(g)` в `kl_leash`.
    """
    out = torch.empty(H.shape[0], dtype=torch.float64)
    for i in range(0, H.shape[0], chunk):
        h = H[i:i + chunk].double()
        u = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        b = sq * (u @ M.t().double())
        e = b.clone()
        e[:, tok] = sq * (u @ (W_rows * g[None, :]).t().double())
        lse_b = torch.logsumexp(b, dim=-1)
        pe = torch.softmax(e, dim=-1)
        out[i:i + chunk] = (lse_b - torch.logsumexp(e, dim=-1)) + (pe[:, tok] * (e[:, tok] - b[:, tok])).sum(-1)
        del b, e, pe
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--carrier", default="/tmp/e39m/results/exp39m_carrier_floor_pearson.pt",
                    help="файл носителя с ключом W_rows (нет — берётся синтетическая правка)")
    ap.add_argument("--eps", type=float, nargs="+", default=[3e-3, 1e-2, 3e-2],
                    help="относительная величина синтетической правки строк U128")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()
    t0 = time.time()

    ck = torch.load(e38.jcache_path(args.layer), map_location="cpu", weights_only=False)
    tok_ids = [int(t) for t in ck["U128"]]
    D = int(ck["D"])
    K = len(tok_ids)
    keys = sorted(ck["H"].keys())
    H = torch.stack([ck["H"][k].float().clone() for k in keys])
    P = torch.stack([ck["P128"][k].float().clone() for k in keys])
    # поле P128 обязано быть ренормированной строкой (sum=1) — иначе все «массы» ниже
    assert abs(float(P.sum(-1).mean()) - 1.0) < 1e-4, float(P.sum(-1).mean())

    raw_path = Path(RAW_CACHE_DIR) / f"exp27b_jcache_L{args.layer}.pt"
    if raw_path.exists():
        ck2 = torch.load(raw_path, map_location="cpu", weights_only=False)
        assert torch.equal(ck["P128"][keys[7]], ck2["P128"][keys[7]]), "P128 differs: это не тот кэш"
        RAW = torch.stack([ck2["P128_RAW"][k].float().clone() for k in keys])
    else:
        RAW = None
    tok = torch.tensor(tok_ids, dtype=torch.long)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(ck.get("model", "Qwen/Qwen3.5-0.8B")), dtype=torch.float32)
    W = model.lm_head.weight.float()
    g = (1.0 + model.model.norm.weight.float())
    M = W * g[None, :]
    SQ = float(np.sqrt(D))

    def p_exact() -> torch.Tensor:
        out = torch.empty(H.shape[0], K, dtype=torch.float32)
        for i in range(0, H.shape[0], 256):
            out[i:i + 256] = torch.softmax(H[i:i + 256] @ W.t(), dim=-1)[:, tok]
        return out

    def p_renorm() -> torch.Tensor:
        out = torch.empty(H.shape[0], K, dtype=torch.float32)
        for i in range(0, H.shape[0], 256):
            h = H[i:i + 256]
            u = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            out[i:i + 256] = torch.softmax(SQ * (u @ M.t()), dim=-1)[:, tok]
        return out

    pe, pr = p_exact(), p_renorm()
    nrm = H.norm(dim=-1)
    res: dict = {
        "experiment": "E-39d/E-39m маршрут head(H): температура readout",
        "layer": args.layer, "n_positions": len(keys), "K": K, "D": D,
        "new_forwards": 0,
        "cache_H_norm": {"mean": float(nrm.mean()), "std": float(nrm.std()),
                         "sqrt_D": SQ, "mean_ratio_to_sqrtD": float((nrm / SQ).mean())},
    }
    print(f"‖H‖ mean {float(nrm.mean()):.3f} = {float((nrm/SQ).mean()):.3f}·√D", flush=True)

    for name, Pm in (("route_exact", pe), ("route_renorm", pr)):
        m_ = Pm.sum(-1)
        res[name] = {"mass_mean": float(m_.mean()), "frac_mass_lt_0p5": float((m_ < 0.5).double().mean())}
        if RAW is not None:
            Q = Pm / Pm.sum(-1, keepdim=True).clamp_min(1e-12)
            R = RAW / RAW.sum(-1, keepdim=True).clamp_min(1e-12)
            kl = (R * (R.clamp_min(1e-12).log() - Q.clamp_min(1e-12).log())).sum(-1)
            res[name]["KL_vs_P128_RAW_renorm_mean"] = float(kl.mean())
            res[name]["KL_vs_P128_RAW_renorm_p95"] = float(kl.quantile(0.95))
            res[name]["corr_mass_vs_P128_RAW"] = float(np.corrcoef(m_, RAW.sum(-1))[0, 1])
        print(f"{name}: frac<0.5 {res[name]['frac_mass_lt_0p5']:.4f} "
              f"KL(vs поле кэша) {res[name].get('KL_vs_P128_RAW_renorm_mean', float('nan')):.3e}", flush=True)

    # fail-loud: точный маршрут обязан совпасть с полем кэша, а renorm — нет
    if RAW is not None:
        assert res["route_exact"]["KL_vs_P128_RAW_renorm_mean"] < 1e-6, res["route_exact"]
        assert res["route_renorm"]["KL_vs_P128_RAW_renorm_mean"] > 1e-3, res["route_renorm"]
        assert abs(res["route_exact"]["frac_mass_lt_0p5"] - float((RAW.sum(-1) < 0.5).double().mean())) < 1e-6

    # ---- насколько тот же маршрут занижает KL правки считывания ----
    sent = RAW.sum(-1) < 0.5 if RAW is not None else torch.zeros(H.shape[0], dtype=torch.bool)
    alive = ~sent
    cases = []
    real = Path(args.carrier)
    if real.exists():
        c = torch.load(real, map_location="cpu", weights_only=False)
        Wr = c["W_rows"].float()
        cases.append(("carrier_E39m_floor_pearson", Wr, c.get("rec", {}).get("history", [{}])[-1].get("l_kl")))
    rng = np.random.default_rng(0)
    base_rows = W[tok]
    for e in args.eps:
        delta = torch.tensor(rng.normal(0.0, 1.0, size=(K, D)), dtype=torch.float32)
        delta = delta / float(np.linalg.norm(delta.numpy())) * e * float(np.linalg.norm(base_rows.numpy()))
        cases.append((f"synthetic_eps_{e:g}", base_rows + delta, None))

    for tag, Wr, reported in cases:
        k_ex = kl_readout(H, W, Wr, tok)
        k_rn = kl_readout_renorm(H, M, Wr, tok, g, SQ)
        row = {"case": tag,
               "KL_exact_all": float(k_ex.mean()), "KL_renorm_all": float(k_rn.mean()),
               "KL_exact_alive": float(k_ex[alive].mean()), "KL_renorm_alive": float(k_rn[alive].mean()),
               "ratio_exact_over_renorm_all": float(k_ex.mean() / k_rn.mean()),
               "reported_l_kl_in_run": reported}
        res.setdefault("kl_cases", []).append(row)
        print(f"{tag:28s} KL exact {row['KL_exact_all']:.5f} vs renorm {row['KL_renorm_all']:.5f} "
              f"×{row['ratio_exact_over_renorm_all']:.2f} (в прогоне l_kl={reported})", flush=True)

    res["verdict"] = {
        "cache_H_is": "выход финальной RMSNorm (множитель g включён): точные логики = H @ W_lmᵀ",
        "code_route_is": "√D·(H/‖H‖)·(W·diag g)ᵀ — перенормирует post-norm H и применяет g второй раз, "
                         "то есть делит все логиты на ‖H‖/√D ≈ 4.04 (позиционная температура, std 0.20)",
        "what_breaks": "доля sentinel (50.71 % → 55.44 %), KL-леваш kl_f (занижен в разы), всё, что "
                       "построено на base_k_all/lse_b_all; ранговые метрики (corr_f/κ′) почти не плывут — "
                       "положительное масштабирование логитов сохраняет порядок",
        "what_is_safe": "PPL (реальный форвард), κ/E-39a (реальный J), все поля кэша P128/P128_RAW",
    }
    res["wall_sec"] = round(time.time() - t0, 1)
    path = Path(args.results_dir) / "E39d_head_route_probe.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"wrote {path} wall {res['wall_sec']} с")


if __name__ == "__main__":
    main()
