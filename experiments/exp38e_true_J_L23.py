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
import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:                    # в ноутбуке Kaggle `__file__` не определён (урок E-39m v1)
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# На Kaggle соседей (`_e38_cache`, `exp39a_u_obs_rank`, `exp28_leaf_structure` и пакет `prism`)
# берём из датасета кода — обычный импорт, поэтому `__main__`-гарды работают как локально.
KAGGLE_CODE = tuple(p for p in ("/kaggle/input/prism-exp-code",
                               "/kaggle/input/datasets/voroninmichael/prism-exp-code")
                    if os.path.isdir(p))
for _p in KAGGLE_CODE:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _e38_cache as e38  # noqa: E402

torch.set_grad_enabled(False)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# ids пула exp27b (32 док × 128 позиций): без них не восстановить те же позиции; на Kaggle
# файл кладётся в датасет кода, локально — лежит в дереве продакшена кэша.
POOL = "/data/prism_kernels/exp27b/results/exp27b_pool_ids_L23.pt"
DOMAINS = ("wiki", "math", "code", "chat")
N_PER_DOMAIN = 8


def _find(name: str) -> str:
    """Файл-сосед: рядом со скриптом, потом в датасете кода Kaggle."""
    for d in (_HERE, *KAGGLE_CODE):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"{name}: не найден ни в {_HERE}, ни в {KAGGLE_CODE}")


# Носители E-39m — выход ядра `e-39m-flat-carrier-vs-ptq`, на Kaggle оно смонтировано как
# kernel_source в /kaggle/input; локально — /tmp после `kaggle kernels output`.
CARRIER_DIRS = ("/kaggle/input", "/tmp/e39m/results", os.path.join(_HERE, "..", "results"))


def _find_carrier(spec: str) -> str:
    """`cheat` / `floor_pearson` / путь к .pt → путь к файлу носителя (fail-loud)."""
    if os.path.isfile(spec):
        return spec
    name = spec if spec.endswith(".pt") else f"exp39m_carrier_{spec}.pt"
    for d in CARRIER_DIRS:
        for root, _dirs, files in os.walk(d) if os.path.isdir(d) else []:
            if name in files:
                return os.path.join(root, name)
    raise FileNotFoundError(f"{name}: носитель не найден в {CARRIER_DIRS}")


def load_exp39a():
    """_kappa_from_J из E-39a — тот же код, что дал опубликованный κ(L23) = 0.0162."""
    path = Path(_find("exp39a_u_obs_rank.py"))
    spec = importlib.util.spec_from_file_location("exp39a_for_kappa", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp39a_for_kappa"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exp28():
    """Оценщики E-28 (Δ_SVD и голономия Procrustes) — тот же код, что дал карточку L23 0.266/0.018."""
    path = Path(_find("exp28_leaf_structure.py"))
    spec = importlib.util.spec_from_file_location("exp28_for_metrics", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp28_for_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod


def main(docs: int = 8, carrier: str = "", results_dir: str | None = None,
         device: str | None = None) -> None:
    """docs — документов на домен (8 = весь корпус); carrier — файл(ы) носителя через запятую.

    Сигнатура именованная, потому что на Kaggle `main` вызывается из runner-ячейки с kwargs
    (argparse в ноутбуке разобрал бы argv ipkernel'а).
    """
    t0 = time.time()
    dev = torch.device(device or DEV)
    specs = [c.strip() for c in carrier.replace(",", " ").split() if c.strip()]
    carriers = [_find_carrier(c) for c in specs]
    if results_dir:
        rd = Path(results_dir)
    elif "__file__" in globals():
        rd = Path(__file__).resolve().parents[1] / "results"
    else:
        rd = Path.cwd() / "results"

    ck = torch.load(e38.jcache_path(23), map_location="cpu", weights_only=False)
    pool = POOL if os.path.isfile(POOL) else _find(os.path.basename(POOL))
    ids = torch.load(pool, map_location="cpu", weights_only=False)["ids"]
    tok = torch.tensor([int(t) for t in ck["U128"]])
    S = int(ck["S"])
    D = int(ck["D"])
    K = len(tok)
    assert S == 128 and K == 128

    from transformers import AutoModelForCausalLM
    G = globals()
    if "load_model" in G:      # Kaggle: resolve_model + кэш модели в /kaggle/working
        m = G["load_model"](AutoModelForCausalLM, G.get("SMALL", "Qwen/Qwen3.5-0.8B"),
                            dtype=torch.float32)
    else:
        m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32)
    m = m.to(dev).eval()
    w = m.model.norm.weight.float().cpu()
    g = 1.0 + w
    W = m.lm_head.weight.float().cpu()[tok]              # [K, D]
    M_K = W * g[None, :]                                # [K, D] — множитель g В ЯКОБИАНЕ верен
    eps = float(getattr(m.model.norm, "eps", 1e-5))
    print(f"[phase] модель на {dev} за {time.time() - t0:.0f} с", flush=True)

    def forward_h(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(h_pre, h_post) финальной RMSNorm, fp32 на CPU — математика J остаётся CPU-ной."""
        with torch.no_grad():
            out = m(input_ids=x.to(dev), attention_mask=torch.ones_like(x).to(dev),
                    output_hidden_states=True)
        return out.hidden_states[-2][0].float().cpu(), out.hidden_states[-1][0].float().cpu()

    def j_closed(h: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """J = r·(M − (M x̂) x̂ᵀ) для x̂ = h/rms(h), r = 1/rms — закрытая форма ∂(M·x̂)/∂h."""
        rms = h.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
        xhat = h / rms
        r = 1.0 / rms
        Mx = xhat @ M.t()
        return (r[..., None] * (M[None, :, :] - (Mx[..., None] * xhat[:, None, :]) / D)).contiguous()

    doc_list = [(dom, si) for dom in DOMAINS for si in range(docs)]

    J_true_all, J_cache_all, P_all = [], [], []
    chk = {}
    res_selfcheck: dict = {}
    for (dom, si) in doc_list:
        x = torch.as_tensor(ids[dom][si]).view(1, -1).long()
        h_pre, h_post = forward_h(x)
        # сверка: кэш хранит post-norm состояние
        d_post = max(float((h_post[S - 1 - t] - ck["H"][(dom, si, t)]).abs().max()) for t in (0, 63, 127))
        assert d_post < 1e-3, f"{dom}/{si}: кэшевый H не равен post-norm состоянию (Δ={d_post})"
        Jt = j_closed(h_pre, M_K)
        # контроль закрытой формы на кэше: та же формула, вычисленная на post-norm входе,
        # обязана совпасть с JR(L23) — это и есть тезис о суррогате. Кэш индексует позиции
        # НАИЗОБОРОТ: JR[t] = формула на h_post[S-1-t] (замер 2026-11-04: rel 1.7e-07 против
        # 2.7e-01 при прямом индексе), поэтому все наши стеки кладём в порядке кэша.
        if not J_true_all:
            Jsurr = j_closed(h_post, M_K)
            rel_jr = float((Jsurr[S - 1] - ck["JR"][(dom, si, 0)]).norm()
                           / ck["JR"][(dom, si, 0)].norm())
            chk["closed_form_at_post_vs_cached_JR_rel"] = rel_jr
            assert rel_jr < 1e-4, f"формула на post-norm не равна кэшевому JR: {rel_jr:.2e}"
            hp = h_pre.clone().to(dev).requires_grad_(True)
            with torch.enable_grad():
                lg = m.lm_head(m.model.norm(hp)).squeeze(0)[:, tok]
                rows = [torch.autograd.grad(lg[:, k][S - 1], hp, retain_graph=True,
                                            allow_unused=True)[0][S - 1].detach().float().cpu()
                        for k in range(K)]
            Jaut = torch.stack(rows)
            rel_dev = float((Jaut - Jt[S - 1]).norm() / Jaut.norm())
            assert rel_dev < 1e-4, f"закрытая форма J не совпала с autograd: rel dev {rel_dev:.2e}"
            res_selfcheck = {"rel_dev_closed_form_vs_autograd": rel_dev,
                             "rel_closed_form_at_post_vs_cached_JR": rel_jr,
                             "position": "wiki/0 t=0"}
            print(f"[selfcheck] closed vs autograd {rel_dev:.2e} | at-post vs JR {rel_jr:.2e}",
                  flush=True)
        J_true_all.append(Jt.flip(0))          # в порядок индексов кэша (t ↔ S-1-t)
        J_cache_all.append(torch.stack([ck["JR"][(dom, si, t)] for t in range(S)]).float())
        P_all.append(torch.stack([ck["P128"][(dom, si, t)] for t in range(S)]).float())
        chk[f"{dom}/{si}"] = {"max_abs_H_vs_post": d_post,
                              "norm_h_pre_mean": float(h_pre.norm(dim=-1).mean()),
                              "norm_H_mean": float(h_post.norm(dim=-1).mean())}
    t_fwd = time.time() - t0
    print(f"[phase] форварды+J: {len(doc_list)} док за {t_fwd:.0f} с", flush=True)

    JT = torch.stack(J_true_all)          # [N,S,K,D] — в порядке индексов кэша
    JC = torch.stack(J_cache_all)
    # словарь кэша весит ~2.2 ГБ (JR+P128+P128_RAW+H): на инстансе Kaggle 13-15 ГБ RAM, а
    # нужны ещё стеки JT/JC и носителя → освобождаем сразу, P_all и ids уже извлечены
    for _k in ("JR", "P128", "P128_RAW", "H"):
        ck.pop(_k, None)
    gc.collect()
    n_pos = JT.shape[0] * JT.shape[1]

    # масштаб: во сколько раз настоящий J больше кэшевого (построчно). Косинус считаем в fp32:
    # .double() удвоил бы оба стека (4.3 ГБ сверху) и на инстансе Kaggle это уже пик памяти
    flat_t, flat_c = JT.flatten(2), JC.flatten(2)
    ratio = flat_t.norm(dim=-1) / flat_c.norm(dim=-1)
    cos_pf = torch.nn.functional.cosine_similarity(flat_t, flat_c, dim=-1)

    res: dict = {
        "experiment": "E-38e/E-39: настоящий J(L23) против кэшевого-суррогата",
        "carriers": carrier, "device": str(dev), "pool_file": pool,
        "selfcheck_closed_form_vs_autograd": res_selfcheck,
        "n_docs": len(doc_list), "n_positions": int(n_pos), "new_forwards": len(doc_list),
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
    del J_true_all, J_cache_all          # списки по доку нужны только для ratios
    gc.collect()
    print(f"[phase] shares за {time.time() - t0 - t_fwd:.0f} с", flush=True)

    # ---- κ на том же коде, что дал опубликованный κ ----
    a39 = load_exp39a()

    def kappas(J_stack: torch.Tensor) -> dict:
        kb, _vals, negH = a39._kappa_from_J(J_stack, S, a39.R)
        return {"kappa": kb["median"], "ci": kb["ci"], "n": kb["n"],
                "n_neg_det": int(np.sum(negH))}

    kap = {"cached": kappas(JC), "true": kappas(JT)}
    res["kappa"] = kap
    if kap["cached"]["kappa"] and kap["true"]["kappa"]:
        res["kappa_ratio_true_over_cached"] = kap["true"]["kappa"] / kap["cached"]["kappa"]
    print(f"[phase] kappa за {time.time() - t0:.0f} с", flush=True)

    # ---- Δ_SVD и голономия E-28 на обоих операторах (те же оценщики, r = 16, без новых форвардов) ----
    # Карточка E-28 (results/exp28_leaf_structure.json, L23): delta_mean 0.26618594716955707,
    # kappa_mean 0.01752786685915092 — они обязаны сойтись на кэше, иначе сверяем не то.
    e28 = load_exp28()
    e28._pin_p_cores()
    dsc: dict = {"r": e28.R_MAIN, "degen_rel": e28.DEGEN_REL,
                 "anchor_e28": {"delta_mean": 0.26618594716955707, "kappa_mean": 0.01752786685915092}}
    for tag, J in (("cached", JC), ("true", JT)):
        A1, A2 = [], []
        for i in range(J.shape[0]):
            _, d1, k2, _nd = e28.document_profile(J[i], [e28.R_MAIN], e28.R_MAIN)
            A1.append(d1.numpy())
            A2.append(k2.numpy())
        dsc[tag] = {"delta_svd": float(np.nanmean(np.stack(A1))),
                    "kappa_holonomy": float(np.nanmean(np.stack(A2))),
                    "n_nan_delta": int(np.isnan(np.stack(A1)).sum())}
        del A1, A2
    d_rand, k_rand = e28.rand_baseline(D, e28.R_MAIN, e28.B_RAND, e28.SEED + 23)
    dsc["random_baseline"] = {"delta_svd": d_rand, "kappa_holonomy": k_rand}
    assert len(doc_list) != 32 or abs(dsc["cached"]["delta_svd"] - dsc["anchor_e28"]["delta_mean"]) < 2e-3, \
        f"Δ_SVD на кэше не воспроизводит E-28: {dsc['cached']}"
    assert len(doc_list) != 32 or abs(dsc["cached"]["kappa_holonomy"] - dsc["anchor_e28"]["kappa_mean"]) < 2e-3, \
        f"голономия на кэше не воспроизводит E-28: {dsc['cached']}"
    dsc["delta_ratio_true_over_cached"] = dsc["true"]["delta_svd"] / dsc["cached"]["delta_svd"]
    dsc["kappa_holonomy_ratio_true_over_cached"] = (dsc["true"]["kappa_holonomy"]
                                                   / dsc["cached"]["kappa_holonomy"])
    res["e28_metrics"] = dsc
    # стеки базовых операторов больше не нужны (все метрики по ним сняты) — освобождаем
    # до циклов по носителям: 2 × 2.1 ГБ RAM
    del JT, JC
    gc.collect()
    print(f"[phase] e28 за {time.time() - t0:.0f} с", flush=True)

    # ---- носители E-39m: переносится ли плоскость на настоящий оператор ----
    # Носитель меняет только 128 строк связной матрицы, значит J носителя = та же закрытая
    # форма с M = W_rows·g; новых backward-прогонов не нужно, только forward за h_pre/h_post.
    if carriers:
        res["carrier_metrics"] = {}
        for cf in carriers:
            tag = Path(cf).stem.replace("exp39m_carrier_", "")
            c = torch.load(cf, map_location="cpu", weights_only=False)
            Wr = c["W_rows"].float()
            assert Wr.shape == (K, D), f"W_rows {tuple(Wr.shape)} != {(K, D)}"
            rec = c.get("rec", {})
            Mk_c = Wr * g[None, :]
            ent = {"file": cf, "kappa": {}, "supplement_ratios": {},
                   "published_e39m": {k: rec.get(k) for k in ("kappa", "corr_f", "ppl",
                                                              "ppl_delta_pct") if k in rec}}
            # стек носителя держим по одному оператору: 2.1 ГБ на стек, 13-15 ГБ RAM инстанса
            # не резиновые (словарь кэша к этому моменту уже освобождён)
            for op, which in (("true", 0), ("surrogate", 1)):
                J_c = []
                for (dom, si) in doc_list:
                    x = torch.as_tensor(ids[dom][si]).view(1, -1).long()
                    J_c.append(j_closed(forward_h(x)[which], Mk_c).flip(0))
                ent["kappa"][op] = kappas(torch.stack(J_c))
                ent["supplement_ratios"][op] = ratios(J_c, P_all)
                del J_c
                gc.collect()
                print(f"[carrier {tag} {op}] kappa {ent['kappa'][op]['kappa']:.6f} "
                      f"wall {time.time() - t0:.0f} s", flush=True)
            # fail-loud: суррогатная ветка обязана воспроизвести κ' из .pt (иначе это не тот
            # носитель). Сверка осмысленна только на полном корпусе: κ' E-39m — медиана по 4096
            # позициям, на подвыборке она законно уходит (на 4 документах — 5.4 %).
            if rec.get("kappa") and len(doc_list) == 32:
                err = abs(ent["kappa"]["surrogate"]["kappa"] - rec["kappa"]) / rec["kappa"]
                ent["surrogate_vs_published_rel_err"] = err
                assert err < 0.03, (f"{tag}: κ на суррогате {ent['kappa']['surrogate']['kappa']} "
                                   f"не совпал с κ' из .pt ({rec['kappa']}, отн. {err:.1%})")
            if kap["true"]["kappa"]:
                ent["kappa_true_rel_to_base"] = (ent["kappa"]["true"]["kappa"]
                                                 / kap["true"]["kappa"] - 1.0)
            res["carrier_metrics"][tag] = ent
            print(f"[carrier {tag}] готов за {time.time() - t0:.0f} с", flush=True)

    res["wall_sec"] = round(time.time() - t0, 1)
    path = rd / "E38e_true_J_L23.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_doc_checks",)}, indent=2, ensure_ascii=False))
    print(f"wrote {path} wall {res['wall_sec']} с")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=8, help="документов на домен (8 = весь корпус)")
    ap.add_argument("--carrier", default="", nargs="*",
                    help="файл(ы) носителя с ключом W_rows (например floor_pearson E-39m): тогда κ "
                         "считается ещё и для него, на обоих операторах — честный ответ, "
                         "переносится ли плоскость на настоящий J")
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    main(docs=a.docs, carrier=",".join(a.carrier), results_dir=a.results_dir, device=a.device)
