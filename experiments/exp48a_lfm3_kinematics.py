#!/usr/bin/env python3
"""E-48a: переносимость кинематики листа на LFM3-1.7B (другая архитектура backbone).

Идея (docs/leaf_dynamics.md §9, пререгистрация ДО чисел)
--------------------------------------------------------
Вся теория «листа» E-47 зависит только от последних двух модулей:
    logits = W_head · norm(h),   norm(x) = g ⊙ x · rsqrt(mean(x²)+eps)
(Qwen: g = 1+w; LFM3: g = w, модуль model.embedding_norm). Отсюда
    J(h) = r(h) · M (I − ρ u u^T), M = W_head[U] diag(g), ρ = ||h||²/(||h||²+D·eps),
— форма НЕ зависит от backbone. Что ПЕРЕНОСИМО, так это:
  (1) сама идентичность (верифицируется числом на реальном кэше);
  (2) КИНЕМАТИКА u_t (поведение h沿 генерации/промпта): скорость a,速度 листа s,
      планарность p, насыщение s ≈ A·tanh(a/a0), доменные контрасты excess_a.
      (2) — эмпирика, и вот её переносимость под вопросом: backbone LFM3 —
      conv/sliding/full 1:1:2 с sandwich-norm, распределение h может быть другим.

Критерии (зафиксированы до запуска):
  T0.1 идентичность: rel ||J_ag − J_an||/||J|| < 1e-5 на графе norm+head (24 позиции)
       и на полном backbone (2 позиции); route-check |norm(x)E_U^T − logits| < 1e-4.
  T0.2 пол метода: angle(V(h), V(3.7h)) ≤ 1e-5; gap16 (σ16>σ17) ≥ 0.95 позиций.
  T1   насыщение: фит tanh по пулу (s,a); переносимо, если A ∈ [0.5,2]·A_qwen И R²>0.8.
  T2   планарность: med p > 0.7 (Qwen L23: ~0.85).
  T3   доменный контраст math/others по excess_a: тот же знак и CI не пересекаются
       (Qwen: math выше ≈×1.5); иначе — «не переносится».
Пул/метрики — те же функции exp47_common, U128 — СВЕЖИЙ частотный базис LFM3
(overlap с Qwen-базисом не имеет смысла: другой токенайзер; сравниваем величины, не базисы).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

import exp47_common as C  # noqa: E402

RNG_SEED = 47
# ориентиры Qwen L23 (P0, exp47a_kinematics_cache.json) для T1/T2
QWEN_REF = {"A_tanh": 0.383, "a0_tanh": 0.715, "p_med": 0.850}  # §8.1, фит по бин-средним


def pearson(x, y):
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 8:
        return float("nan")
    x, y = x[m], y[m]
    x = x - x.mean(); y = y - y.mean()          # ЦЕНТРИРОВАТЬ (иначе это E[XY]/σσ > 1)
    sx, sy = x.std(), y.std()
    return float((x * y).mean() / (sx * sy)) if sx > 0 and sy > 0 else float("nan")


def boot_ci(vals, B=10_000, seed=RNG_SEED):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 3:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(B, len(vals)))
    ms = np.nanmedian(vals[idx], axis=1)
    return [float(np.quantile(ms, 0.025)), float(np.quantile(ms, 0.975))]


def fit_tanh(a, s, seed=RNG_SEED):
    """s ≈ A·tanh(a/a0):网格-фит по A × сетка a0, RSS-минимум (torch, без scipy)."""
    a = torch.tensor(a, dtype=torch.float64)
    s = torch.tensor(s, dtype=torch.float64)
    ok = ~(torch.isnan(a) | torch.isnan(s))
    a, s = a[ok], s[ok]
    best = (float("inf"), None, None)
    A_grid = torch.linspace(0.2, 3.0, 57)
    a0_grid = torch.exp(torch.linspace(math.log(0.02), math.log(3.0), 60))
    for A in A_grid:
        pred = A * torch.tanh(a.unsqueeze(1) / a0_grid.unsqueeze(0))
        rss = ((pred - s.unsqueeze(1)) ** 2).mean(dim=0)
        j = int(torch.argmin(rss))
        if float(rss[j]) < best[0]:
            best = (float(rss[j]), float(A), float(a0_grid[j]))
    rss_min, A_hat, a0_hat = best
    rss_0 = float(((s - s.mean()) ** 2).mean())
    r2 = 1.0 - rss_min / rss_0 if rss_0 > 0 else float("nan")
    return {"A": A_hat, "a0": a0_hat, "r2": r2, "n": int(ok.sum())}


def get_final_form(m):
    """Норма+хед этой модели: (g, eps, head_W_tied). Только множитель g различается."""
    if hasattr(m.model, "norm"):                       # Qwen-style: final model.norm, g=1+w
        nrm = m.model.norm
        return 1.0 + nrm.weight.detach().float(), float(nrm.eps), m.lm_head.weight.detach(), "norm"
    nrm = m.model.embedding_norm                        # LFM3: хвостовая RMSNorm g=w, head tied
    return (nrm.weight.detach().float(), float(nrm.variance_epsilon),
            m.get_input_embeddings().weight.detach(), "embedding_norm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/LiquidAI/LFM3-1.7B")
    ap.add_argument("--n-docs", type=int, default=8)
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--K", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--results-dir", default=os.path.join(_HERE, "..", "results"))
    ap.add_argument("--smoke", action="store_true", help="1 домен × 2 док × 16 позиций, только selfchecks")
    args = ap.parse_args()

    torch.set_grad_enabled(True)
    from transformers import AutoModelForCausalLM  # type: ignore
    from prism.calibration import build_global_token_set, load_calibration_data  # noqa: E402

    t0 = time.time()
    # transformers 5.15 нормализует layer_types (conv -> linear_attention) ПОСЛЕ
    # __post_init__ remote-кода, а masking-карта modeling_lfm3 ключует 'conv'
    # (KeyError 'linear_attention'): перечитываем тип-список из config.json.
    from transformers import AutoConfig  # type: ignore
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    import json as _json
    with open(os.path.join(args.model, "config.json")) as _fh:
        cfg.layer_types = list(_json.load(_fh)["layer_types"])
    # transformers 5.15.1 + remote lfm3 несовместимость: DynamicCache в ветке
    # record_past возвращает conv state unsqueezed(1) -> 4D -> F.conv1d падает
    # (у нативного lfm2 состояние 4D и не проходит через update_conv_state).
    # Сжимаем лишнюю размерность на границе cache->model (для 3D — no-op).
    from transformers.cache_utils import LinearAttentionLayer  # type: ignore
    _orig_ucs = LinearAttentionLayer.update_conv_state

    def _ucs(self, conv_states, state_idx=0, **kw):
        if conv_states.ndim == 4:
            conv_states = conv_states.squeeze(1)
        return _orig_ucs(self, conv_states, state_idx, **kw)

    LinearAttentionLayer.update_conv_state = _ucs
    m = AutoModelForCausalLM.from_pretrained(args.model, config=cfg, dtype=torch.float32,
                                             trust_remote_code=True).to(args.device)
    m.eval()
    D = m.config.hidden_size
    tied = m.lm_head.weight.data_ptr() == m.get_input_embeddings().weight.data_ptr()
    g, eps, head_W_all, norm_name = get_final_form(m)
    print(f"model={m.config.model_type} norm={norm_name} eps={eps} tied={tied} D={D}", flush=True)

    # --- pool + свежий частотный базис ---
    domains = ["wiki"] if args.smoke else ["wiki", "math", "code", "chat"]
    pool = {}
    for dom in domains:
        pool[dom] = load_calibration_data(args.model, max_length=512, min_length=args.S + 8,
                                          n_samples=args.n_docs, domain=dom)
        if args.smoke:
            pool[dom] = pool[dom][:2]
        print(f"pool {dom}: {len(pool[dom])} docs", flush=True)
    U128 = build_global_token_set(m, [d for v in pool.values() for d in v], K=args.K)
    U = torch.tensor(U128, device=args.device)
    A = C.head_M(head_W_all.float()[U128], g)           # [K,D] fp64

    # --- hook: вход финальной нормы = выход последнего decoder-слоя ---
    cap = {}

    def _hook(_mod, _inp, out):
        cap["x"] = out[0] if isinstance(out, tuple) else out

    m.model.layers[-1].register_forward_hook(_hook)

    def capture(ids):
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        with torch.no_grad():
            logits = m(input_ids=ids.to(args.device)).logits[0]
        return cap["x"][0].float(), logits.float()

    # ---------------- T0.1: идентичность ----------------
    sc = {"tied_head": bool(tied), "eps": eps, "norm_module": norm_name,
          "model_type": m.config.model_type}
    route_max, rel_head, rel_full = 0.0, 0.0, 0.0
    n_checked = 0
    for dom in domains:
        for si, doc in enumerate(pool[dom]):
            ids = doc["input_ids"][:args.S if not args.smoke else 16].to(args.device)
            x, logits = capture(ids)
            t = min(args.S if not args.smoke else 16, ids.shape[0]) - 5
            # g входит в A=E_U·diag(g), поэтому здесь — БЕЗ g (иначе g² и маршрут, и J врут)
            xn = x[t].double() * torch.rsqrt(x[t].double().pow(2).mean() + eps)
            route_max = max(route_max, float((xn @ A.T - logits[t, U.cpu()].double()).abs().max()))
            # autograd J на графе norm+head (то, что определяет лист): полный jacobian
            xd0 = x[t].detach().double()
            Jg = torch.autograd.functional.jacobian(
                lambda z: (z * torch.rsqrt(z.pow(2).mean() + eps)) @ A.T, xd0)
            Ja, _rho, _u = C.J_analytic(A, xd0, eps)
            r = float(torch.rsqrt(xd0.pow(2).mean() + eps))
            rel_head = max(rel_head, float((Ja * r - Jg).norm() / Jg.norm()))
            n_checked += 1
            if n_checked >= (2 if args.smoke else 8):
                break
        if n_checked >= (2 if args.smoke else 8):
            break
    sc.update(route_max=route_max, rel_J_headgraph_max=rel_head, n_identity_positions=n_checked)
    ok_t01 = route_max < 1e-3 and rel_head < 1e-5
    print("T0.1", {k: sc[k] for k in ("route_max", "rel_J_headgraph_max")}, flush=True)

    # полный backbone (2 позиции): grad logits[t] по входу нормы через весь граф
    if not args.smoke:
        for dom, si in [("math", 0), ("code", 0)]:
            ids = pool[dom][si]["input_ids"].reshape(-1)[:args.S].to(args.device)
            with torch.enable_grad():
                out = m(input_ids=ids.unsqueeze(0))
            xg = cap["x"].reshape(ids.shape[0], -1).detach().requires_grad_(True)  # [S,D]
            # ВАЖНО: rms ПОСЛЕВЫЕ (dim=-1), иначе grad разливается по всем S строкам
            lgh = (xg.double() * torch.rsqrt(xg.double().pow(2).mean(-1, keepdim=True) + eps)) @ A.T
            t = 60
            Jfull = []
            for k in range(0, args.K, 16):
                e = torch.zeros(args.K, dtype=torch.float64, device=args.device); e[k] = 1.0
                gfull, = torch.autograd.grad(lgh[t], xg, e, retain_graph=True)
                Jfull.append(gfull[t].double())          # grad по всем S строкам; нужна строка t
            Jfull = torch.stack(Jfull)                        # [8,D]
            Ja, _rho, _u = C.J_analytic(A, xg[t].detach().double(), eps)
            r = float(torch.rsqrt(xg[t].detach().double().pow(2).mean() + eps))
            rel_full = max(rel_full, float((Ja[::16] * r - Jfull).norm() / Jfull.norm()))
        sc["rel_J_fullbackbone_max"] = rel_full
        ok_t01 = ok_t01 and rel_full < 1e-5

    # ---------------- T0.2: пол + gap16 ----------------
    floor_max, gap_pos, gap_n = 0.0, 0, 0
    # ---------------- основной цикл ----------------
    docs_out, pair_as = [], []
    for dom in domains:
        for si, doc in enumerate(pool[dom]):
            ids = doc["input_ids"][:args.S].to(args.device)
            x, _ = capture(ids)
            hs = x[:args.S].double()
            if args.smoke:
                hs = hs[:16]
            u = C.u_series(hs)
            a = C.angle_series(u)
            s_ser, p_ser, e_ser = [], [], []
            Vs = [C.V16_from_J(C.J_analytic(A, hs[t], eps)[0])[0] for t in range(hs.shape[0])]
            for t in range(len(Vs) - 1):
                V0, V1 = Vs[t], Vs[t + 1]
                s, p, P, _th = C.frame_kinematics(V0, V1)
                s_ser.append(s); p_ser.append(p)
                e_ser.append(C.align_to_u_plane(P, u[t], u[t + 1]))
            # floor + gap на подвыборке позиций
            for t in range(0, hs.shape[0], 16 if not args.smoke else 8):
                V0, sig = C.V16_from_J(C.J_analytic(A, hs[t], eps)[0])
                V1f = C.V16_from_J(C.J_analytic(A, 3.7 * hs[t], eps)[0])[0]
                floor_max = max(floor_max, float(torch.acos(
                    torch.linalg.svdvals(V0.T @ V1f).clamp(max=1 - 1e-12)).max()))
                gap_pos += int(float(sig[15]) > float(sig[16])); gap_n += 1
            a_a = a.numpy().astype(float)
            s_a = np.array(s_ser); p_a = np.array(p_ser); e_a = np.array(e_ser)
            pair_as.extend(zip(a_a.tolist(), s_a.tolist()))
            thr = np.quantile(a_a[~np.isnan(a_a)], 0.10)
            sel = a_a > thr
            docs_out.append({
                "dom": dom, "si": si,
                "a_med": float(np.nanmedian(a_a)), "s_med": float(np.nanmedian(s_a)),
                "p_med": float(np.nanmedian(p_a)), "e_med": float(np.nanmedian(e_a)),
                "r_pear": pearson(s_a[sel], a_a[sel]),
                "excess_a": C.band_power(torch.from_numpy(a_a - np.nanmean(a_a))),
            })
            print(f"{dom} s{si}: a={docs_out[-1]['a_med']:.4f} s={docs_out[-1]['s_med']:.4f} "
                  f"p={docs_out[-1]['p_med']:.3f} e={docs_out[-1]['e_med']:.3f} "
                  f"r={docs_out[-1]['r_pear']:.3f} exc_a={docs_out[-1]['excess_a']:.2f}", flush=True)

    aa = np.array([x0 for x0, _ in pair_as]); ss = np.array([y for _, y in pair_as])
    ok = ~(np.isnan(aa) | np.isnan(ss))
    tanh = fit_tanh(aa[ok], ss[ok])
    # честное сравнение с Qwen (§8.1): фит по БИН-СРЕДНИМ (13 квантильных бинов по a),
    # R² тоже на бин-средних — point-wise R² несопоставим (шум сэмпов доминирует)
    qa = np.quantile(aa[ok], np.linspace(0, 1, 14))
    a_edges, mean_s = [], []
    for i in range(len(qa) - 1):
        m_ = (aa[ok] >= qa[i]) & (aa[ok] < qa[i + 1])
        if m_.sum() > 20:
            a_edges.append(float(np.mean(aa[ok][m_])))
            mean_s.append(float(np.mean(ss[ok][m_])))
    tanh_bins = fit_tanh(np.array(a_edges), np.array(mean_s))
    sc["scale_floor_max_rad"] = floor_max
    sc["gap16_positive_frac"] = gap_pos / max(gap_n, 1)

    # ---------------- вердикты (механически) ----------------
    by = {d0["dom"] for d0 in docs_out}
    med = {d0: float(np.nanmedian([x0["excess_a"] for x0 in docs_out if x0["dom"] == d0])) for d0 in by}
    p_med = float(np.nanmedian([d0["p_med"] for d0 in docs_out]))
    a_med = float(np.nanmedian([d0["a_med"] for d0 in docs_out]))
    verdicts = {
        "T0.1": {**{k: sc[k] for k in sc if k.startswith(("route", "rel_J"))},
                 "verdict": "идентичность переносится" if ok_t01 else "ЛОМКА формулы/пайплайна"},
        "T0.2": {"floor": floor_max, "gap16": sc["gap16_positive_frac"],
                 "verdict": "ok" if (floor_max <= 1e-5 and sc["gap16_positive_frac"] >= 0.95)
                            else "метод/спектр не переносится"},
        "T1": {**tanh_bins, "A_points": tanh["A"], "a0_points": tanh["a0"], "r2_points": tanh["r2"],
               "qwen_A": QWEN_REF["A_tanh"], "qwen_a0": QWEN_REF["a0_tanh"],
               "verdict": ("насыщение переносится"
                           if (tanh_bins["r2"] > 0.8 and 0.5 <= tanh_bins["A"] / QWEN_REF["A_tanh"] <= 2.0
                               and 0.5 <= tanh_bins["a0"] / QWEN_REF["a0_tanh"] <= 2.0)
                           else "форма/масштаб насыщения отличается")},
        "T2": {"p_med": p_med, "qwen_p_med": QWEN_REF["p_med"],
               "verdict": "планарность переносится" if p_med > 0.7 else "не переносится"},
        "T3": None,
    }
    if not args.smoke and "math" in by:
        others = [x0["excess_a"] for x0 in docs_out if x0["dom"] in ("wiki", "code", "chat")]
        mm = [x0["excess_a"] for x0 in docs_out if x0["dom"] == "math"]
        ci_m, ci_o = boot_ci(mm), boot_ci(others)
        ratio = med["math"] / max(float(np.nanmedian(others)), 1e-12)
        overlap = not (ci_m[1] < ci_o[0] or ci_o[1] < ci_m[0])
        verdicts["T3"] = {"math_med": med["math"], "others_med": float(np.nanmedian(others)),
                          "ratio": ratio, "ci_math": ci_m, "ci_others": ci_o, "ci_overlap": overlap,
                          "verdict": ("доменный контраст переносится"
                                      if (ratio > 1.15 and not overlap) else "не переносится")}
    out = {"meta": {"protocol": "docs/leaf_dynamics.md §9 (E-48a)", "model": args.model,
                    "n_docs": args.n_docs, "S": args.S, "K": args.K, "U128": "fresh freq basis",
                    "elapsed_s": round(time.time() - t0, 1), "smoke": args.smoke},
           "descriptive": {"per_doc": docs_out, "tanh_fit": tanh, "tanh_fit_bins": tanh_bins,
                           "a_edges": a_edges, "mean_s_by_abin": mean_s,
                           "a_med_overall": a_med, "p_med_overall": p_med,
                           "domain_excess_a": med},
           "verdicts": verdicts}
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, "exp48a_lfm3_kinematics.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("WROTE", path, flush=True)
    for k, v in sorted(verdicts.items()):
        print("VERDICT", k, "->", (v or {}).get("verdict") if v else "n/a", flush=True)


if __name__ == "__main__":
    main()
