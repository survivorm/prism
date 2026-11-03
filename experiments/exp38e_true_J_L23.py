#!/usr/bin/env python3
"""E-38e/E-39: кэшевый J слоя 23 — суррогат (формула нормировки, вычисленная на её же выходе).

Продюсер кэша (`kaggle_kernel/kernel-exp27b/kernel.ipynb`, `_last_layer_jacobian`) пишет:
«J is evaluated at the norm INPUT (h), never at h_norm — evaluating the formula at the norm
output gives a matrix with ~0.87 relative error». И тут же зовёт это на
`h = outputs.hidden_states[layer_idx+1]`, что для последнего слоя в этой версии transformers
есть РОВНО выход финальной RMSNorm (замер: `hidden_states[24]` совпадает с кэшевым `H` до
cos = 1.000000, `‖H‖ = 131`; у входа `‖h‖ = 10.1`, `cos(ĥ,Ĥ) = 0.685`). Значит JR(L23) =
(√D/‖H_post‖)·M_K(I − Ĥ Ĥᵀ) — проверено с точностью 1e-07, — а настоящий `∂logits/∂h_pre`
отличается в 13 раз по норме и на ~7 % по направлению после подбора масштаба.

Скрипт перестраивает НАСТОЯЩИЙ J (закрытая форма, сверена с autograd) и считает на нём то,
что считалось на суррогате: κ(E-38d/E-39a) и headline-отношения E-39-suppl (O1/iso, g_uni),
на одних и тех же позициях. Выход — число: на сколько живёт серия.

Запуск: `../.venv/bin/python exp38e_true_J_L23.py [--docs 4]` из `experiments/`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _e38_cache as e38  # noqa: E402

torch.set_grad_enabled(False)
POOL = "/data/prism_kernels/exp27b/results/exp27b_pool_ids_L23.pt"
DOMAINS = ("wiki", "math", "code", "chat")
N_PER_DOMAIN = 8


def load_exp39a():
    """_kappa_from_J из E-39a — тот же код, что дал опубликованный κ(L23) = 0.0162."""
    path = Path(__file__).resolve().parent / "exp39a_u_obs_rank.py"
    spec = importlib.util.spec_from_file_location("exp39a_for_kappa", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp39a_for_kappa"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=8, help="документов на домен (8 = весь корпус)")
    ap.add_argument("--carrier", default="",
                    help="файл носителя с ключом W_rows (например floor_pearson E-39m): тогда κ "
                         "считается ещё и для него, на обоих операторах — честный ответ, "
                         "переносится ли плоскость на настоящий J")
    ap.add_argument("--results-dir", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = ap.parse_args()
    t0 = time.time()

    ck = torch.load(e38.jcache_path(23), map_location="cpu", weights_only=False)
    ids = torch.load(POOL, map_location="cpu", weights_only=False)["ids"]
    tok = torch.tensor([int(t) for t in ck["U128"]])
    S = int(ck["S"]); D = int(ck["D"]); K = len(tok)
    assert S == 128 and K == 128

    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32)
    m.eval()
    w = m.model.norm.weight.float(); g = 1.0 + w
    W = m.lm_head.weight.float()[tok]                   # [K, D]
    M_K = W * g[None, :]                                # [K, D] — множитель g В ЯКОБИАНЕ верен
    eps = float(getattr(m.model.norm, "eps", 1e-5))
    SQ = float(np.sqrt(D))

    keys_cached = sorted(ck["JR"].keys())
    n_doc = args.docs
    docs = [(dom, si) for dom in DOMAINS for si in range(n_doc)]

    J_true_all, J_cache_all, P_all = [], [], []
    chk = {}
    res_selfcheck: dict = {}
    for (dom, si) in docs:
        x = torch.as_tensor(ids[dom][si]).view(1, -1).long()
        out = m(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True)
        h_pre = out.hidden_states[-2][0].float()         # [S, D] вход в финальную RMSNorm
        h_post = out.hidden_states[-1][0].float()        # [S, D] — обязан равняться кэшевому H
        # сверка: кэш хранит post-norm состояние
        d_post = max(float((h_post[S - 1 - t] - ck["H"][(dom, si, t)]).abs().max()) for t in (0, 63, 127))
        assert d_post < 1e-3, f"{dom}/{si}: кэшевый H не равен post-norm состоянию (Δ={d_post})"
        rms = h_pre.pow(2).mean(-1, keepdim=True).add(eps).sqrt()          # [S,1]
        xhat = h_pre / rms                                                 # = rsqrt·h
        # J = r·(diag(1+w) − xhat xᵀ/D)   [D,D] — и на него слева W: J_logits = M_K·J_normᵀ
        # построчно: J[k,d] = r·(M_K[k,d] − xhat_d·(M_K·x̂)_k / D)·... (норма по x̂, не по ĥ)
        r = 1.0 / rms                                                    # [S,1]
        Mx = xhat @ M_K.t()                                              # [S,K] = M_K·x̂
        # J[k,d] = r·(M_K[k,d] − x̂_d·(M_K·x̂)_k/D) — множитель r относится к обеим частям
        Jt = r[..., None] * (M_K[None, :, :] - (Mx[..., None] * xhat[:, None, :]) / D)  # [S,K,D]
        if not J_true_all and dom == DOMAINS[0] and si == 0:
            # самопроверка закрытой формы против autograd на одной позиции
            hp = h_pre.clone().requires_grad_(True)
            with torch.enable_grad():
                lg = m.lm_head(m.model.norm(hp).unsqueeze(0))[0][:, tok]
                rows = [torch.autograd.grad(lg[:, k][S - 1], hp, retain_graph=True,
                                            allow_unused=True)[0][S - 1].detach().float()
                        for k in range(K)]
            Jaut = torch.stack(rows)
            dev = float((Jaut - Jt[S - 1]).norm() / Jaut.norm())
            assert dev < 1e-4, f"закрытая форма J не совпала с autograd: rel dev {dev:.2e}"
            res_selfcheck = {"rel_dev_closed_form_vs_autograd": dev, "position": "wiki/0 t=0"}
            print(f"[selfcheck] closed form vs autograd: rel dev {dev:.2e}", flush=True)
        J_true_all.append(Jt.contiguous())
        J_cache_all.append(torch.stack([ck["JR"][(dom, si, t)] for t in range(S)]).float())
        P_all.append(torch.stack([ck["P128"][(dom, si, t)] for t in range(S)]).float())
        chk[f"{dom}/{si}"] = {"max_abs_H_vs_post": d_post,
                              "norm_h_pre_mean": float(h_pre.norm(dim=-1).mean()),
                              "norm_H_mean": float(h_post.norm(dim=-1).mean())}

    JT = torch.stack(J_true_all)          # [N,S,K,D]
    JC = torch.stack(J_cache_all)
    n_pos = JT.shape[0] * JT.shape[1]

    # масштаб: во сколько раз настоящий J больше кэшевого (построчно)
    ratio = (JT.flatten(2).norm(dim=-1) / JC.flatten(2).norm(dim=-1))
    cos_pf = torch.nn.functional.cosine_similarity(JT.flatten(2).double(), JC.flatten(2).double(), dim=-1)

    res: dict = {
        "experiment": "E-38e/E-39: настоящий J(L23) против кэшевого-суррогата",
        "selfcheck_closed_form_vs_autograd": res_selfcheck,
        "n_docs": len(docs), "n_positions": int(n_pos), "new_forwards": len(docs),
        "per_doc_checks": chk,
        "scale_true_over_cached": {"median": float(ratio.median()), "p10": float(ratio.quantile(0.1)),
                                   "p90": float(ratio.quantile(0.9))},
        "cos_true_vs_cached": {"median": float(cos_pf.median()), "min": float(cos_pf.min())},
    }

    # ---- headline-отношения E-39-suppl: масштаб по позиции сокращается, направление — нет ----
    def ratios(J_docs: list[torch.Tensor], P_docs: list[torch.Tensor]) -> dict:
        """Энергетические shares E-39-suppl, по-доково в fp64 (полный fp64-стек не заводим)."""
        o1_rel, g_uni = [], []
        for J, P in zip(J_docs, P_docs):
            Jd = J.double()
            Pd = P.double()
            PJ = Pd[..., None] * (Jd - torch.einsum("sk,skd->sd", Pd, Jd))
            o1 = PJ.pow(2).sum((-1, -2)) / Jd.pow(2).sum((-1, -2))
            iso = (Pd.pow(2).sum(-1) - 2 * Pd.pow(3).sum(-1) + Pd.pow(2).sum(-1) ** 2) / K
            Jt1 = torch.einsum("skd,k->sd", Jd, torch.ones(K, dtype=torch.float64))
            gu = Jt1.pow(2).sum(-1) / (K * Jd.pow(2).sum((-1, -2)))       # ‖Jᵀ1‖²/(K‖J‖²)
            o1_rel.append(o1 / iso)
            g_uni.append(gu)
        o1_rel = torch.cat([x.flatten() for x in o1_rel])
        g_uni = torch.cat([x.flatten() for x in g_uni])
        # среднее по O1_rel не читается: iso → 0 на почти дельта-позициях (правило №7
        # supplement'а: для этого отношения единственная осмысленная агрегация — медиана)
        return {"O1_rel_median": float(o1_rel.median()), "g_uni_median": float(g_uni.median()),
                "g_uni_mean": float(g_uni.mean())}

    res["supplement_ratios"] = {"cached": ratios(J_cache_all, P_all), "true": ratios(J_true_all, P_all)}

    # ---- κ на том же коде, что дал опубликованный κ ----
    a39 = load_exp39a()
    kap = {}
    cases = [("cached", JC), ("true", JT)]
    if args.carrier:
        c = torch.load(args.carrier, map_location="cpu", weights_only=False)
        Wr = c["W_rows"].float()
        assert Wr.shape == (K, D), f"W_rows {tuple(Wr.shape)} != {(K, D)}"
        Mk_c = Wr * g[None, :]
        for tag, hlist, base_M in (("carrier_true", "pre", Mk_c), ("carrier_surrogate", "post", Mk_c)):
            Jc = []
            for (dom, si) in docs:
                x = torch.as_tensor(ids[dom][si]).view(1, -1).long()
                o = m(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True)
                hv = (o.hidden_states[-2] if hlist == "pre" else o.hidden_states[-1])[0].float()
                rms = hv.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
                xh = hv / rms
                rr = 1.0 / rms
                Mx = xh @ base_M.t()
                Jc.append((rr[..., None] * (base_M[None] - (Mx[..., None] * xh[:, None, :]) / D)).contiguous())
            cases.append((tag, torch.stack(Jc)))
    for tag, J in cases:
        kb, vals, negH = a39._kappa_from_J(J, S, a39.R)
        kap[tag] = {"kappa": kb["median"], "ci": kb["ci"], "n": kb["n"],
                    "n_neg_det": int(np.sum(negH))}
    res["kappa"] = kap
    if kap["cached"]["kappa"] and kap["true"]["kappa"]:
        res["kappa_ratio_true_over_cached"] = kap["true"]["kappa"] / kap["cached"]["kappa"]
    res["wall_sec"] = round(time.time() - t0, 1)
    path = Path(args.results_dir) / "E38e_true_J_L23.json"
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_doc_checks",)}, indent=2, ensure_ascii=False))
    print(f"wrote {path} wall {res['wall_sec']} с")


if __name__ == "__main__":
    main()
