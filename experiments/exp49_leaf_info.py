#!/usr/bin/env python
"""E-49: НЕСЁТ ЛИ ЛИСТ ИНФОРМАЦИЮ О ТРАЕКТОРИИ — размороженный протокол с контролями.

Контекст: тождество J и «облако» V-рам воспроизводятся на двух архитектурах (E-48), но
попарные эвклидовы метрики r=16 оказались null-эквивалентны (§10). Проверяем три гипотезы
о том, что МЫ МЕРИЛИ НЕ ТАК, каждая — несколькими методами:

  H1 «не тот top»     — r ∈ {1,2,4} (верхние моды, σ1/σ2≈2.3) vs 16 (вырожденный хвост).
  H2 «не та гранулярн.»— сглаживание h по окну W ∈ {1,4,16} открывает медленный канал.
  H3 «не та метрика»  — углы в pullback-метрике ридаута G = AᵀA (M-метрика, G-ортонорм.
                        базисы через L=chol(G)).

Методы на каждом (r, W, метрика):
  M1 лаг-кривая s(lag), lag ∈ {1,2,4,8,16,32} (full-length серии, без прореживания);
  M2 когерентность τ = [s(t,t+1)+s(t+1,t+2)−s(t,t+2)]/s(t,t+1)  (0 = связное движение,
     1 = белое; главный дискриминант, работает даже когда лаг-кривая насыщена);
  M3 голономия окна L=8 — НЕ ЗАПУСКАЛАСЬ в production-сетке (функция holonomy()
     есть, не вызывалась; исправлено по ревью E49 §1.2 — заявка снята);
  M4 v1-угол со знакомым транспортом (r=1 чувствителен к знаку — фиксируем трассировкой).

КОНТРОЛИ (падение этапа, если не различились):
  C-A white u (независимые направления через тот же конвейер): τ≈1, лаг flat;
  C-B coherent u (шаги δ=0.02 по геодезическим, медленно поворачивающаяся плоскость):
      τ<0.5, лаг-кривая растёт.

Запуск: --stage 1 (только контроли, быстро) → --stage 2 (данные) → обе модели.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import exp47_common as C  # noqa: E402

EPS_LFM, EPS_QWEN = 1e-5, 1e-6
RNG = np.random.default_rng(47)
LAGS = (1, 2, 4, 8, 16, 32)


def med(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return round(float(np.median(x)), 4) if len(x) else None


def boot_ci(v, B=2000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return [None, None]
    # seed фиксирован НА ВЫЗОВ (ревью §2.6): CI не зависит от порядка вызовов
    idx = np.random.default_rng(47).integers(0, len(v), (B, len(v)))
    m = np.median(v[idx], axis=1)
    return [round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


# ------------------------------------------------------------------ каркас метрик

def frame_in_metric(J, r, L):
    """Базис top-r инвариантного подпространства JᵀJ, G-ортонормированный.
    Фикс (ревью E49 §1.1): torch.linalg.cholesky даёт G = L·Lᵀ (НЕ LᵀL), значит
    ‖x‖_G = ‖Lᵀx‖₂ и G-ортонорм — qr(LᵀV), а не qr(LV). Возвращаемый базис —
    Lᵀ·span(V) (НЕ span(V)); углы между подпространствами в G-метрике равны
    евклидовым углам между Lᵀ-образами — это то, зачем встроивание нужно.
    ПРЕЖНИЕ M-ячейки сетки
    exp49_leaf_info_*.json считали метрику LᵀL ≠ G — НЕИНТЕРПРЕТИРУЕМЫ (евклидовы
    H2-выводы exp49b2 не затронуты: там L=None)."""
    V = C.V16_from_J(J, r=r)[0]                       # [D,r] ортонорм (евклид)
    if L is None:
        return V
    Wm = L.T @ V
    return torch.linalg.qr(Wm)[0]


def tau_triplets(Vs):
    a = [C.frame_kinematics(Vs[t], Vs[t + 1])[0] for t in range(len(Vs) - 2)]
    b = [C.frame_kinematics(Vs[t + 1], Vs[t + 2])[0] for t in range(len(Vs) - 2)]
    c = [C.frame_kinematics(Vs[t], Vs[t + 2])[0] for t in range(len(Vs) - 2)]
    return [(x + y - z) / x for x, y, z in zip(a, b, c) if x > 1e-6]


def v1_signed_series(us, A, eps):
    out, prev = [], None
    for u in us:
        J, _, _ = C.J_analytic(A, u, eps)
        G = J @ J.T; G = 0.5 * (G + G.T)
        lam, Ue = torch.linalg.eigh(G)
        v1 = J.T @ Ue[:, -1]
        v1 = v1 / v1.norm()
        if prev is not None and float(v1 @ prev) < 0:
            v1 = -v1
        out.append(v1); prev = v1
    return out


def v1_lag(v1s):
    return {L: [float(torch.acos((v1s[t] @ v1s[t + L]).abs().clamp(max=1 - 1e-12)))
                for t in range(len(v1s) - L)] for L in LAGS}


def holonomy(Vs, L=8):
    out = []
    r = Vs[0].shape[1]
    for t in range(len(Vs) - L):
        P = torch.eye(r, dtype=torch.float64)
        for k in range(L):
            P = P @ C.procrustes_R(Vs[t + k], Vs[t + k + 1])
        P = C.procrustes_R(Vs[t + L], Vs[t]) @ P
        th, _, _ = C.schur_angles(P)
        out.append(float(torch.sqrt((th * th).sum())))
    return out


def doc_stats(Vs, v1s=None):
    d = {"s": {str(L): [C.frame_kinematics(Vs[t], Vs[t + L])[0] for t in range(len(Vs) - L)]
               for L in LAGS},
         "tau": tau_triplets(Vs)}
    if v1s is not None:
        d["v1"] = {str(L): v for L, v in v1_lag(v1s).items()}
    return d


def agg(docss):
    out = {"s_lag": {str(L): med([x for d in docss for x in d["s"][str(L)]]) for L in LAGS},
           "tau": med([x for d in docss for x in d["tau"]])}
    if docss and "v1" in docss[0]:
        out["v1_lag"] = {str(L): med([x for d in docss for x in d["v1"][str(L)]]) for L in LAGS}
    # bootstrap по документам на τ и на s(1)
    taus = [med(d["tau"]) for d in docss]
    out["tau_doc_boot95"] = boot_ci(taus)
    return out


# ------------------------------------------------------------------ контроли

def nz(x):
    return x / x.norm()


def control_white(T, D):
    g = torch.Generator().manual_seed(7)
    return [nz(torch.randn(D, dtype=torch.float64, generator=g))
            for _ in range(T)]


def control_coherent(T, D, delta=0.02):
    """ГЕОДЕЗИЧЕСКИЙ контроль (фикс ревью E49-4 §2.1 — та же ортогонализация, что в
    synth_coh exp49b2; прежняя версия без w⟂u давала невалидный CB)."""
    g = torch.Generator().manual_seed(11)
    u = nz(torch.randn(D, dtype=torch.float64, generator=g))
    w = nz(torch.randn(D, dtype=torch.float64, generator=g))
    w = nz(w - u * (u @ w))
    out = []
    for t in range(T):
        out.append(u.clone())
        u2 = math.cos(delta) * u + math.sin(delta) * w
        u = nz(u2)
        z = nz(torch.randn(D, dtype=torch.float64, generator=g))
        w = nz(w + 0.05 * (z - (z @ u) * u - (z @ w) * w))
        w = nz(w - u * (u @ w))
    return out


def run_stage1(A, eps, D, T=96):
    res = {}
    for name, us in [("C-A_white", control_white(T, D)), ("C-B_coherent", control_coherent(T, D))]:
        Vs = [frame_in_metric(C.J_analytic(A, u, eps)[0], 16, None) for u in us]
        v1s = v1_signed_series(us, A, eps)
        d = doc_stats(Vs, v1s)
        res[name] = {"s_lag": {k: med(v) for k, v in d["s"].items()},
                     "tau": med(d["tau"]),
                     "v1_lag": {k: med(v) for k, v in d["v1"].items()}}
    ok = (res["C-A_white"]["tau"] > 0.7 and res["C-B_coherent"]["tau"] < 0.5
          and res["C-B_coherent"]["s_lag"]["16"] and res["C-B_coherent"]["s_lag"]["1"]
          and res["C-B_coherent"]["s_lag"]["16"] > res["C-B_coherent"]["s_lag"]["1"])
    res["SELFTEST"] = "ok" if ok else "ВАЛИТСЯ: конвейер не различает белое/связное"
    return res


# ------------------------------------------------------------------ данные

def smoothed_us(u, nrm, W):
    h = u * torch.tensor(nrm, dtype=torch.float64).unsqueeze(1)
    if W > 1:
        hs = torch.stack([h[max(0, t - W + 1):t + 1].mean(0) for t in range(len(h))])
    else:
        hs = h
    return [hs[t] / hs[t].norm() for t in range(len(hs))]


def load_model(args):
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
    if args.model == "qwen":
        mp = os.environ.get("PRISM_QWEN_DIR", "/data/Qwen/Qwen3.5-0.8B")
        ck = torch.load(os.environ.get("PRISM_JCACHE",
                        "/data/prism_kernels/exp27b/results/exp27b_jcache_L23.pt"),
                        map_location="cpu", weights_only=False)
        U = ck["U128"]
        U = U.tolist() if torch.is_tensor(U) else list(U)
        cfg = AutoConfig.from_pretrained(mp)
        m = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.float32).eval()
        g = 1.0 + m.model.norm.weight.detach().float()
        npz, jp, eps = (os.environ.get("PRISM_NPZ",
                        "/data/prism_kernels/exp47b/exp47b_gen_kinematics.npz"),
                        os.environ.get("PRISM_DOCS_JSON",
                                       "results/exp47b_gen_kinematics.json"), EPS_QWEN)
    else:
        from transformers.cache_utils import LinearAttentionLayer  # type: ignore
        mp = os.environ.get("PRISM_LFM3_DIR", "/data/LiquidAI/LFM3-1.7B")
        cfg = AutoConfig.from_pretrained(mp, trust_remote_code=True)
        cfg.layer_types = list(json.load(open(os.path.join(mp, "config.json")))["layer_types"])
        _o = LinearAttentionLayer.update_conv_state
        LinearAttentionLayer.update_conv_state = (
            lambda self, cs, si=0, **kw: _o(self, cs.squeeze(1) if cs.ndim == 4 else cs, si, **kw))
        m = AutoModelForCausalLM.from_pretrained(mp, config=cfg, dtype=torch.float32,
                                                 trust_remote_code=True).eval()
        g = m.model.embedding_norm.weight.detach().float()
        U = list(range(0, 384, 3))[:128]
        npz, jp, eps = ("/data/prism_kernels/exp48b/exp48b_lfm3_gen_kinematics.npz",
                        "results/exp48b_lfm3_gen_kinematics.json", EPS_LFM)
    W_head = m.get_input_embeddings().weight.detach()
    # tied-проверка универсальная (ревью E49 §1.4/1.6): у LFM3-1.7B
    # tie_word_embeddings=True и lm_head в чекпоинте ОТСУТСТВУЕТ (проверено
    # 2026-11-06 по config/safetensors) — head ≡ embeddings, A был верным;
    # для untied-моделей берём output head.
    head = getattr(m, "lm_head", None)
    if head is not None and head.weight.data_ptr() != W_head.data_ptr():
        W_head = head.weight.detach()
    A = C.head_M(W_head.float()[U], g)
    return A, U, eps, npz, jp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=2)
    ap.add_argument("--model", choices=("qwen", "lfm3"), default="qwen")
    args = ap.parse_args()
    root = os.path.join(_HERE, "..")
    A, U, eps, npz, jp = load_model(args)
    D = int(A.shape[1])
    out = {"model": args.model, "U_head": U[:6], "eps": eps}

    if args.stage == 1:
        out["controls"] = run_stage1(A, eps, D)
        print("CONTROLS:", json.dumps(out["controls"], indent=1), flush=True)
        if out["controls"]["SELFTEST"] != "ok":
            sys.exit("selftest failed")

    if args.stage >= 2:
        G = A.T @ A
        Lch = torch.linalg.cholesky(G + 1e-10 * torch.eye(D, dtype=torch.float64) * float(G.diag().mean()))
        docs = json.load(open(os.path.join(root, jp)))["docs"]
        z = np.load(npz, allow_pickle=True)
        ud = {}
        for doc in docs:
            f = z[f"{doc['dom']}_{doc['si']}"].item()
            ud[f"{doc['dom']}_{doc['si']}"] = (
                torch.from_numpy(z[f"{doc['dom']}_{doc['si']}_u"].astype(np.float64)),
                np.asarray(f["hnorm"], dtype=float))
        rep = {}
        # стриминг: один (W, doc) в памяти; для него eig один раз, все (r,metric) из него
        comb = {}
        for W in (1, 4, 16):
            for key, (u, nrm) in ud.items():
                us = smoothed_us(u, nrm, W)
                eig = []
                for x in us:
                    J, _, _ = C.J_analytic(A, x, eps)
                    Gm = J @ J.T; Gm = 0.5 * (Gm + Gm.T)
                    lam, Ue = torch.linalg.eigh(Gm)
                    eig.append((J, Ue))
                for r in (1, 2, 4, 16):
                    for metric in ("eucl", "M"):
                        Vs = []
                        for (J, Ue) in eig:
                            V = J.T @ Ue[:, -r:]
                            V = V / V.norm(dim=0, keepdim=True)
                            V = torch.linalg.qr(V)[0]
                            if metric == "M":
                                # ревью E49-2 §2.1: cholesky даёт G=L·Lᵀ ⇒ G-ортонорм
                                # = qr(LᵀV); инлайн-ветка ранее крутила метрику LᵀL
                                V = torch.linalg.qr(Lch.T @ V)[0]
                            Vs.append(V)
                        comb.setdefault(f"W{W}_r{r}_{metric}", []).append(doc_stats(Vs, None))
                del eig
            print(f"W{W} done", flush=True)
        for k, ds in comb.items():
            rep[k] = agg(ds)
            print(f"{k}: tau={rep[k]['tau']} s1={rep[k]['s_lag']['1']} "
                  f"s32={rep[k]['s_lag']['32']}", flush=True)
        # v1-метрики отдельно (W=1): угол между v1 со знакомым транспортом
        v1ds = {}
        for key, (u, nrm) in ud.items():
            us = smoothed_us(u, nrm, 1)
            v1ds[key] = v1_signed_series(us, A, eps)
        rep["v1"] = {str(L): med([float(torch.acos((v1ds[k][t] @ v1ds[k][t + L]).abs().clamp(max=1 - 1e-12)))
                                 for k in v1ds for t in range(len(v1ds[k]) - L)]) for L in LAGS}
        out["grid"] = rep

    path = os.path.join(root, "results", f"exp49_leaf_info_{args.model}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print("WROTE", path)


if __name__ == "__main__":
    main()
