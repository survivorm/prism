#!/usr/bin/env python3
"""E-37 (финальный): анатомия P(p_t)-чувствительной компоненты Якобиана.

ТЗ: docs/Exp37.md. Только exp27b dense-кэш (слои {6,12,18,19,22,23}, 32 документа).
Термин «gauge» снят. Три диагностики:

  A. SVD убитой компоненты K_t = J_t − P(p_t)·J_t
     A.0 калибровка порога (fp32 vs fp64, RANK_EPS_calibrated) — ОБЯЗАТЕЛЬНА
     A.1 полный прогон: спектр, energy, rank_loss, dim ker, align_K, null-модели
  B. атрибуция оси u_g(t) = J Jᵀ 1 / ‖·‖ по 128 токенам U128 (синтаксис vs содержание)
  C. L22: тип слоя из config + сравнение 18/19/22/23

Никаких новых форвардов. CPU-only.

Узлы (ТЗ §5):
  1. K не ортогонален PJ — energy_K_topk доля ВНУТРИ K; отдельно frac_K_from_J
  2. u_g — cheap-proxy (L22 align_ug=0.85)
  3. U128 — глобальный top-128 (содержательные токены могут быть вне)
  4. per-sample: НЕ усреднять K_t по сэмплам до SVD (SVD нелинеен)
  5. p̄_med — медиана, не mean; не смешивать med/mean в одном прогоне
  6. категориальная база — per-position (топ-128 по p_t), не глобальная
  7. пиковый p_t (p_max>0.9) — baseline вырожден, исключать, фиксировать долю
  8. калиброванный порог — единый для A.1 и B

Запуск:
    .venv/bin/python experiments/exp37_p_sensitivity.py --selfcheck
    .venv/bin/python experiments/exp37_p_sensitivity.py
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

CKPT_DIR = "/data/prism_kernels/exp27b/results"
MODEL = "/data/Qwen/Qwen3.5-0.8B"
LAYERS = [6, 12, 18, 19, 22, 23]
C_LAYERS = [18, 19, 22, 23]   # диагностика C
SENT = 0.5
RANK_EPS_CANDIDATES = [1e-4, 1e-5, 1e-6]
K_TOP = 32                      # спектр K — топ-32 целиком
TAIL_FROM = 96                  # хвост σ_i для i >= 96
N_A0_POS = 8                    # первые 8 позиций каждого документа
PEAK_PMAX = 0.9                 # пиковый p_t — baseline вырожден
SEED = 42
RESULTS = "results"


# ---------------- data (самодостаточный загрузчик — не зависит от версии exp33r) ----------------

def load_layer(layer):
    path = os.path.join(CKPT_DIR, f"exp27b_jcache_L{layer}.pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    ck["S"] = int(ck["S"]); ck["n_samples"] = int(ck["n_samples"])
    k0 = next(iter(ck["JR"].values()))
    ck["K"] = int(k0.shape[0]); ck["D"] = int(k0.shape[1])
    return ck


def load_doc(ck, dom, si):
    """Ключи кэша — кортежи (dom, si, t)."""
    S = ck["S"]
    Jdoc = torch.stack([ck["JR"][(dom, si, t)] for t in range(S)]).float()
    Pdoc = torch.stack([ck["P128"][(dom, si, t)] for t in range(S)]).float()
    if "P128_RAW" in ck:
        PRaw = torch.stack([ck["P128_RAW"][(dom, si, t)] for t in range(S)]).float()
        sent = (PRaw.sum(-1) < SENT).numpy()
    else:
        # L19: кэш без P128_RAW — sentinel-маска недоступна, все позиции валидны
        sent = np.zeros(S, dtype=bool)
    return Jdoc, Pdoc, sent


def p_apply(J, p):
    """P(p)·J per position: [T,K,D] = J∘p − (Jᵀp)∘p.  (E-33r convention)"""
    w = torch.einsum('tkd,tk->td', J, p)
    return J * p.unsqueeze(2) - w.unsqueeze(1) * p.unsqueeze(2)


def doc_ids(ck):
    domains = sorted({k[0] for k in ck["JR"]})
    return [(d, s) for d in domains for s in range(ck["n_samples"])]


def k_left_svd(Kx):
    """Левые сингулярные векторы Kx [T,K,D] через eigh(K·Kᵀ) — ДЕТЕРМИНИРОВАННО.
    (torch.linalg.svd в этом env возвращает U с недетерминированным layout:
    2D (T,K) «folded» vs 3D (T,K,K) — зависит от процесса/кэша; eigh(Gram) — нет.)
    → (sigma [T,K] desc, U [T,K,K] — столбец k = u_k)."""
    G = Kx @ Kx.transpose(-1, -2)          # [T, K, K], K=128 — дёшево
    ev, ec = torch.linalg.eigh(G)          # по возрастанию
    sigma = ev.flip(-1).clamp_min(0).sqrt()
    U = ec.flip(-1)                          # [T, K, K]: U[t,:,k] = u_k(t)
    return sigma, U


# ---------------- categories (B) ----------------

def build_categories(ck):
    """U128 → token id → category {whitespace, punctuation, newline, content}."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    U = [int(x) for x in np.asarray(ck["U128"]).tolist()]
    cats = {}
    for t in U:
        d = tok.decode([t])
        if d == " ":
            c = "whitespace"
        elif d in ("\n", "\r", "\t") or (d.isspace() and d != " "):
            c = "newline"
        elif d in ",.!?;:()[]{}\"'`/\\|@#$%^&*~+=<>-_":
            c = "punctuation"
        else:
            c = "content"
        cats[t] = c
    return U, cats


# ---------------- A.0: threshold calibration ----------------

def calibrate(ck, docs, n_pos=N_A0_POS):
    """fp32 vs fp64 SVD K_t, σ_noise из хвоста σ_i(K_fp64) i>=TAIL_FROM,
    RANK_EPS = 3·σ_noise (relative to σ_1)."""
    out_positions = []
    sigma_tail_fp64 = []
    for (dom, si) in docs:
        Jdoc, Pdoc, sent = load_doc(ck, dom, si)
        for t in range(min(n_pos, Jdoc.shape[0])):
            J32 = Jdoc[t]                                   # [K, D] fp32
            p32 = Pdoc[t]
            K32_t = J32 - p_apply(J32.unsqueeze(0), p32.unsqueeze(0))[0]  # fp32
            s32 = torch.linalg.svdvals(K32_t)                # [D] fp32
            s64 = torch.linalg.svdvals(K32_t.double())       # [D] fp64
            # top-32 fp32/fp64 ratio
            top_ratio = (s32[:K_TOP] / s64[:K_TOP].clamp_min(1e-300)).numpy()
            # tail (i >= TAIL_FROM) fp64
            tail = s64[TAIL_FROM:]
            sigma_tail_fp64.append(tail.numpy())
            out_positions.append({
                "dom": dom, "t": t,
                "s64_top32": s64[:K_TOP].numpy(),
                "s32_top32": s32[:K_TOP].numpy(),
                "top32_fp32_over_fp64": top_ratio,
                "tail_95fp64": float(np.percentile(tail, 95)),
                "tail_mean_fp64": float(tail.mean()),
            })
        del Jdoc, Pdoc
    tail_all = np.concatenate(sigma_tail_fp64)
    sigma_noise = float(np.percentile(tail_all, 95))
    # σ_1 typical (median of top-1 across positions) for relative scaling
    s1 = np.array([p["s64_top32"][0] for p in out_positions])
    s1_med = float(np.median(s1))
    rank_eps = 3.0 * sigma_noise / max(s1_med, 1e-300)     # relative to σ_1
    top_ratio_all = np.concatenate([p["top32_fp32_over_fp64"] for p in out_positions])
    return {
        "n_positions": len(out_positions),
        "sigma_noise_fp64_tail95": sigma_noise,
        "s1_median_fp64": s1_med,
        "rank_eps_calibrated": rank_eps,
        "top32_fp32_over_fp64_med": float(np.median(top_ratio_all)),
        "top32_fp32_over_fp64_min": float(top_ratio_all.min()),
        "top32_fp32_over_fp64_p05": float(np.percentile(top_ratio_all, 5)),
        "fp32_reliable": bool(np.percentile(top_ratio_all, 5) > 0.98),
        "positions": out_positions,
    }


# ---------------- A.1: K anatomy ----------------

def k_metrics(Jdoc, Pdoc, sent, eps_cal, eps_grid):
    """K_t = J_t − P(p_t)·J_t (perP) и K_t = J_t − P(p̄)·J_t (fixP).
    p̄ — нормированная медиана по non-sentinel. Возвращает (dict метрик [T], pbar)."""
    T, K, D = Jdoc.shape
    sent_t = torch.as_tensor(sent, dtype=torch.bool)
    alive = ~sent_t
    # pbar — НОРМИРОВАННАЯ медиана (Σp=1); p_apply — якобиан softmax
    pbar = Pdoc[alive].median(dim=0).values if alive.any() else Pdoc.mean(dim=0)
    pbar = pbar / pbar.sum().clamp_min(1e-30)
    assert abs(float(pbar.sum()) - 1.0) < 1e-5, "pbar must be normalized"

    PJ_per = p_apply(Jdoc, Pdoc)
    PJ_fix = p_apply(Jdoc, pbar.unsqueeze(0).expand(T, -1))
    K_per = Jdoc - PJ_per
    K_fix = Jdoc - PJ_fix

    out = {}
    ones = torch.ones(K, device=Jdoc.device) / K ** 0.5

    for name, Kx in (("perP", K_per), ("fixP", K_fix)):
        Sx = torch.linalg.svdvals(Kx)                      # [T, K] — надёжно (скаляры)
        sigma, U = k_left_svd(Kx)                           # [T,K] desc, [T,K,K]
        out[f"s_{name}_top32"] = Sx[:, :K_TOP].numpy()
        # energy внутри K (топ-k / ‖K‖²_F)
        nrm2 = Sx.pow(2).sum(-1).clamp_min(1e-30)
        out[f"energy_top1_{name}"] = (Sx[:, 0] ** 2 / nrm2).numpy()
        out[f"energy_top8_{name}"] = (Sx[:, :8].pow(2).sum(-1) / nrm2).numpy()
        # rank по порогам (relative: σ_i/σ_1 > eps)
        s1 = Sx[:, 0].clamp_min(1e-300)
        out[f"rank_{name}_cal"] = ((Sx / s1.unsqueeze(1)) > eps_cal).sum(-1).numpy()
        for e, tag in zip(eps_grid, ("1e4", "1e5", "1e6")):
            out[f"rank_{name}_{tag}"] = ((Sx / s1.unsqueeze(1)) > e).sum(-1).numpy()
        # dim ker = #{σ_i < eps_cal * σ_1}
        out[f"dimker_{name}"] = (K - (Sx / s1.unsqueeze(1) > eps_cal).sum(-1)).numpy()
        # align_K_1 = |<u_K1, 1/√K>| — левый топ-1 (eigh-Gram, детерминированно)
        uK1 = U[:, :, 0]                                    # [T, K]
        out[f"align_K1_{name}"] = (uK1 * ones).sum(-1).abs().numpy()
        # align_K_p = |<u_K1, p_t/‖p_t‖>|
        pn = Pdoc / Pdoc.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        out[f"align_Kp_{name}"] = (uK1 * pn).sum(-1).abs().numpy()
    # frac_K_from_J = ‖K‖²_F / ‖J‖²_F (отдельно perP/fixP)
    Jn2 = Jdoc.pow(2).sum((-1, -2))
    out["frac_K_from_J_perP"] = (K_per.pow(2).sum((-1, -2)) / Jn2).numpy()
    out["frac_K_from_J_fixP"] = (K_fix.pow(2).sum((-1, -2)) / Jn2).numpy()
    out["sent_frac"] = float(sent_t.float().mean())
    return out, pbar


def null_shuffled(Jdoc, Pdoc, sent, n_shuf=16, seed=SEED):
    """Shuffled-J null: на каждый прогон — своя перестановка (баг-фикс: раньше одна al ×16).
    Ограничение (зафиксировано): K = J − P·J сохраняет row-mean (1ᵀP(p)J=0),
    поэтому align_K1 у shuffled-J ≈ реального — нуль НЕ тестирует структурность
    align_K1; для неё — null_random_vector. Shuffled-J остаётся вторым нулём."""
    T, K, D = Jdoc.shape
    vals = []
    for i in range(n_shuf):
        g = torch.Generator().manual_seed(seed + i)
        al = torch.argsort(torch.rand(T, K, generator=g), dim=1)
        Jsh = torch.gather(Jdoc, 1, al.unsqueeze(2).expand(T, K, D))
        Ksh = Jsh - p_apply(Jsh, Pdoc)
        _, U = k_left_svd(Ksh)
        ones = torch.ones(K, device=Jdoc.device) / K ** 0.5
        vals.append((U[:, :, 0] * ones).sum(-1).abs().numpy())
    return np.median(np.stack(vals), axis=0)


def null_random_vector(Jdoc, n_rep=16, seed=SEED):
    """ИСТИННЫЙ ноль для align_K1: w ~ S^{K-1} (случайный единичный), |<w, 1/√K>|
    ~ sqrt(2/(πK)) ≈ 0.07 — baseline без SVD (row-mean-инвариантность не мешает)."""
    T, K, D = Jdoc.shape
    ones = torch.ones(K, device=Jdoc.device) / K ** 0.5
    vals = []
    for i in range(n_rep):
        g = torch.Generator().manual_seed(seed + i)
        w = torch.randn(T, K, generator=g)
        w = w / w.norm(dim=-1, keepdim=True)
        vals.append((w * ones).sum(-1).abs().numpy())
    return np.median(np.stack(vals), axis=0)


def null_random_p(Jdoc, Pdoc, sent, n_rep=16, seed=SEED):
    """Random-p null: p_t → случайное Дирихле.
    КОНТРПРИМЕР, не baseline: real p_t даёт align_Kp ≈ 0.03–0.19, random p — 0.63–0.78.
    P(p_t) убивает p_t-направление в top-векторе K, P(random) — нет.
    Значит real ≪ random — сильный сигнал осмысленности оператора (не «нет эффекта»)."""
    g = torch.Generator().manual_seed(seed)
    T, K, D = Jdoc.shape
    vals = []
    for _ in range(n_rep):
        p = torch.rand(T, K, generator=g)
        p = p / p.sum(-1, keepdim=True)
        Ksh = Jdoc - p_apply(Jdoc, p)
        _, U = k_left_svd(Ksh)
        pn = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        vals.append((U[:, :, 0] * pn).sum(-1).abs().numpy())
    return np.median(np.stack(vals), axis=0)


# ---------------- B: u_g attribution ----------------

def ug_attribution(Jdoc, Pdoc, sent, U, cats, eps_cal):
    """u_g = J Jᵀ 1 / ‖·‖, u_g⊥ = u_g − <u_g,1/√K>1/√K, атрибуция по U128.
    Per-position baseline: доля категорий в топ-128 по p_t (здесь весь набор = U128)."""
    T, K, D = Jdoc.shape
    sent_t = torch.as_tensor(sent, dtype=torch.bool)
    ones = torch.ones(K, device=Jdoc.device) / K ** 0.5
    Jt1 = torch.einsum('tkd,tk->td', Jdoc, torch.ones(T, K, device=Jdoc.device))
    Jjt1 = torch.einsum('tkd,td->tk', Jdoc, Jt1)
    ugc = Jjt1 / Jjt1.norm(dim=-1, keepdim=True).clamp_min(1e-30)   # cheap-proxy u_g
    proj = (ugc * ones).sum(-1, keepdim=True)
    ugc_perp = ugc - proj * ones[None, :]
    perp_share = (ugc_perp.pow(2).sum(-1) / ugc.pow(2).sum(-1).clamp_min(1e-30)).numpy()
    # веса α_i на U128 (весь K-размер = U128, K=128)
    alpha = ugc_perp
    a2 = alpha.pow(2)
    a2n = a2 / a2.sum(-1, keepdim=True).clamp_min(1e-30)
    cat_energy = {}
    for c in ("whitespace", "punctuation", "newline", "content"):
        mask = torch.zeros(K, dtype=torch.bool)
        for i, t in enumerate(U):
            if cats.get(t) == c:
                mask[i] = True
        cat_energy[c] = a2n[:, mask].sum(-1).numpy()
    # entropy = -Σ a2n log a2n (nats)
    ent = -(a2n * a2n.clamp_min(1e-12).log()).sum(-1).numpy()
    # top-8 токенов по |α_i| (пер-позиционный)
    top8 = torch.topk(alpha.abs(), 8, dim=-1).indices.numpy()
    # per-position baseline: доля категорий в топ по p_t
    pmax = Pdoc.max(-1).values.numpy()
    peaked = pmax > PEAK_PMAX
    # alive: non-sentinel AND non-peaked (фиксация доли исключённых — ТЗ §5.7, §3)
    alive_b = (~sent_t) & torch.as_tensor(~peaked, dtype=torch.bool, device=Jdoc.device)
    # baseline = cat share в p_t (весь набор U128, так что это глобально по p_t)
    base = {}
    for c in ("whitespace", "punctuation", "newline", "content"):
        mask = torch.zeros(K, dtype=torch.bool)
        for i, t in enumerate(U):
            if cats.get(t) == c:
                mask[i] = True
        base[c] = (Pdoc[:, mask].sum(-1) / Pdoc.sum(-1).clamp_min(1e-30)).numpy()
    return {
        "perp_share": perp_share,
        "cat_share": {c: cat_energy[c] for c in cat_energy},
        "base_share": {c: base[c] for c in base},
        "entropy_ug_perp": ent,
        "top8_tokens": top8,
        "pmax": pmax,
        "peaked": peaked,
        "alive": alive_b.numpy(),
    }


# ---------------- C: L22 type ----------------

def l22_type():
    import json as _json
    cfg = _json.load(open(os.path.join(MODEL, "config.json")))
    tc = cfg.get("text_config", cfg)
    lt = tc.get("layer_types", [])
    n = len(lt)
    full = [i for i, x in enumerate(lt) if "full" in str(x)]
    lin = [i for i, x in enumerate(lt) if "linear" in str(x).lower()]
    # cache 0-indexed: layer index in cache == 0-indexed model layer
    return {
        "n_layers": n,
        "full_attention_interval": tc.get("full_attention_interval"),
        "layer_types_full_0idx": full,
        "layer_types_linear_0idx": lin,
        "cache_full": [l for l in LAYERS if (l) in full],
        "cache_linear": [l for l in LAYERS if (l) in lin],
        "L22_is_full": (22 in full),
        "L23_is_full": (23 in full),
    }


# ---------------- aggregation (E-30/E-34 protocol) ----------------

def agg_positions(vals_per_doc, min_n=8):
    """med per doc (per-position) -> med over docs. Поддерживает 1D [T] и 2D [T, n] (по-σ)."""
    meds = []
    for v in vals_per_doc:
        v = np.asarray(v, dtype=float)
        if v.ndim == 2:
            # баг-фикс: плоская v[isfinite] плющила [T,n] → 1D; сохраняем shape
            v = v[np.all(np.isfinite(v), axis=-1)]
            if len(v) >= min_n:
                meds.append(np.median(v, axis=0))
        else:
            v = v[np.isfinite(v)]
            if len(v) >= min_n:
                meds.append(float(np.median(v)))
    if not meds:
        return None
    if isinstance(meds[0], (int, float, np.floating)):
        m = np.asarray(meds)
        return {"med": float(np.median(m)),
                "iqr": [float(np.percentile(m, 25)), float(np.percentile(m, 75))],
                "n_docs": len(meds)}
    stack = np.stack(meds)  # [n_docs, n]
    return {"med_per_sigma": [float(x) for x in np.median(stack, axis=0)],
            "n_docs": len(meds)}


# ---------------- selfcheck ----------------

def selfcheck():
    torch.manual_seed(0)
    K, D, T = 128, 64, 16
    # (a) pbar normalized: p_apply(P(pbar)) kills uniform
    p = torch.rand(T, K); p = p / p.sum(-1, keepdim=True)
    pbar = p.median(0).values; pbar = pbar / pbar.sum()
    assert abs(float(pbar.sum()) - 1.0) < 1e-5
    J = torch.rand(T, K, D)
    PJf = p_apply(J, pbar.unsqueeze(0).expand(T, -1))
    Kx = J - PJf
    # P(p)·1 = 0 → token-mean of P(p)·J is 0 (gauge kills uniform mode in its own rows)
    ones = torch.ones(K) / K ** 0.5
    assert (PJf.mean(dim=1)).abs().max() < 1e-3, "fixP: P(p)·J has zero token-mean"
    # (b) K not orthogonal to PJ: ‖J‖² != ‖PJ‖² + ‖K‖²
    PJ = p_apply(J, p)
    Jn = J.pow(2).sum(); Pn = PJ.pow(2).sum(); Kn = Kx.pow(2).sum()
    assert not torch.allclose(Jn, Pn + Kn, atol=1e-2), "K not orthogonal (TZ 5.1)"
    # (c) ug attribution: uniform J → u_g⊥ = 0 (perp_share ≈ 0)
    C = torch.randn(T, D)
    Ju = C[:, None, :].expand(T, K, D).contiguous()
    attr = ug_attribution(Ju, p, np.zeros(T, dtype=bool),
                          list(range(K)), {i: "content" for i in range(K)}, 1e-6)
    assert np.median(attr["perp_share"]) < 1e-4, "uniform J -> u_g perp ~ 0"
    # (d) k_left_svd (Gram-eigh) == linalg.svd left-vec (на 2D-матрице)
    Kd, Dd = 128, 512
    Md = torch.randn(Kd, Dd)
    sig, Ul = k_left_svd(Md.unsqueeze(0))
    sig, Ul = sig[0], Ul[0]                       # [K], [K,K] столбцы
    assert torch.allclose(sig, torch.linalg.svdvals(Md), atol=1e-3), "Gram sigma == svdvals"
    assert (Ul.t() @ Ul - torch.eye(Kd)).abs().max() < 1e-3, "U orthonormal"
    v1 = (Md.T @ Ul[:, 0]) / sig[0].clamp_min(1e-30)
    assert abs(float(v1.norm()) - 1.0) < 1e-3, "M u1 == sigma1 v1"
    print("selfcheck: all passed")


def _clean(o):
    if isinstance(o, torch.Tensor):
        o = o.detach().cpu().numpy()
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    return o


# ---------------- main ----------------

def main():
    a = argparse.ArgumentParser()
    a.add_argument("--selfcheck", action="store_true")
    a.add_argument("--skip-a1", action="store_true", help="только A.0 + B + C")
    a.add_argument("--out-prefix", default="results/exp37")
    a = a.parse_args()
    if a.selfcheck:
        selfcheck()
        return
    torch.set_num_threads(1)
    E28 = __import__("exp28_leaf_structure")
    E28._pin_p_cores()
    np.random.seed(SEED)

    # C: L22 type (30 сек)
    ctype = l22_type()
    json.dump(_clean(ctype), open(f"{a.out_prefix}_C_L22_type.json", "w"), indent=2)
    print("C:", ctype["cache_full"], ctype["cache_linear"], "L22_full=", ctype["L22_is_full"])

    ck = load_layer(23)
    U, cats = build_categories(ck)
    docs = doc_ids(ck)
    n_docs = len(docs)

    # A.0: калибровка (ОБЯЗАТЕЛЬНА)
    cal = calibrate(ck, docs)
    eps_cal = cal["rank_eps_calibrated"]
    json.dump(_clean(cal), open(f"{a.out_prefix}_A0_calibration.json", "w"), indent=2)
    print(f"A.0: RANK_EPS_calibrated={eps_cal:.3e} fp32_reliable={cal['fp32_reliable']} "
          f"(fp32/fp64 p05={cal['top32_fp32_over_fp64_p05']:.4f})")

    # A.1 + B: полный прогон по всем слоям (append JSONL per doc — resumable)
    os.makedirs(RESULTS, exist_ok=True)
    per_layer = {}
    for l in LAYERS:
        ck_l = load_layer(l)
        jp = f"{a.out_prefix}_L{l}_docs.jsonl"
        done_keys = set()
        if os.path.exists(jp):
            for x in open(jp):
                if x.strip():
                    r = json.loads(x)
                    done_keys.add((r["dom"], r["si"]))
            if done_keys:
                print(f"  L{l} resume: {len(done_keys)}/{n_docs} already done", flush=True)
        jl = open(jp, "a")
        for i, (dom, si) in enumerate(docs):
            if (dom, si) in done_keys:
                continue
            Jdoc, Pdoc, sent = load_doc(ck_l, dom, si)
            m, pbar = k_metrics(Jdoc, Pdoc, sent, eps_cal, RANK_EPS_CANDIDATES)
            # nulls на всех слоях (баг-фикс: раньше только L23); n=4 (дешёво, медиана по докам)
            m["null_align_K1_shuf"] = null_shuffled(Jdoc, Pdoc, sent, n_shuf=4, seed=SEED + l)
            m["null_align_K1_rndvec"] = null_random_vector(Jdoc, n_rep=4, seed=SEED + 1000 + l)
            m["null_align_Kp_rndp"] = null_random_p(Jdoc, Pdoc, sent, n_rep=4, seed=SEED + 2000 + l)
            b = ug_attribution(Jdoc, Pdoc, sent, U, cats, eps_cal)
            rec = {"dom": dom, "si": si, "m": _clean(m), "b": _clean(b), "pbar": _clean(pbar)}
            jl.write(json.dumps(rec) + "\n")   # ponytail: per-doc SVD-метрики ~50KB; спектр не хранится по 128 позициям
            del Jdoc, Pdoc
            if (i + 1) % 4 == 0:
                print(f"  L{l} doc {i+1}/{n_docs}", flush=True)
        jl.close()
        print(f"  L{l} docs done ({a.out_prefix}_L{l}_docs.jsonl)", flush=True)

    # Агрегация из JSONL (median по позициям -> median по докам)
    for l in LAYERS:
        path = f"{a.out_prefix}_L{l}_docs.jsonl"
        if not os.path.exists(path):
            continue
        rows = [json.loads(x) for x in open(path) if x.strip()]
        mkeys = [k for k in rows[0]["m"] if isinstance(rows[0]["m"][k], list)]
        entry = {"n_docs": len(rows),
                 "layer_type": ("full" if l in ctype["cache_full"] else "linear")}
        # sent_frac — скаляр per doc (баг-фикс: раньше терялся при агрегации, ТЗ §5.7)
        entry["sent_frac"] = float(np.median([r["m"]["sent_frac"] for r in rows])) if rows else None
        for f in mkeys:
            entry[f] = agg_positions([r["m"][f] for r in rows])
        b_entry = {}
        # B — только по alive позициям (non-sentinel AND non-peaked, ТЗ §3: p_max>0.9 исключать)
        for c in ("whitespace", "punctuation", "newline", "content"):
            b_entry[f"cat_{c}"] = agg_positions([np.asarray(r["b"]["cat_share"][c])[r["b"]["alive"]] for r in rows])
            b_entry[f"base_{c}"] = agg_positions([np.asarray(r["b"]["base_share"][c])[r["b"]["alive"]] for r in rows])
        b_entry["perp_share"] = agg_positions([np.asarray(r["b"]["perp_share"])[r["b"]["alive"]] for r in rows])
        b_entry["entropy"] = agg_positions([np.asarray(r["b"]["entropy_ug_perp"])[r["b"]["alive"]] for r in rows])
        n_all = sum(len(r["b"]["alive"]) for r in rows)
        n_alive = sum(int(np.sum(r["b"]["alive"])) for r in rows)
        # excluded_frac = sentinel ∪ peaked; peaked_only_frac = excluded − sentinel
        # (sentinel-доля = sent_frac, константа по слою: P128_RAW не зависит от слоя)
        b_entry["excluded_frac"] = float(1 - n_alive / max(n_all, 1)) if rows else None
        b_entry["peaked_only_frac"] = float(b_entry["excluded_frac"] - entry["sent_frac"]) if rows else None
        entry["B"] = b_entry
        per_layer[str(l)] = entry
        print(f"  L{l} agg (B alive: cat_content={b_entry['cat_content'].get('med'):.3f} "
              f"whitespace={b_entry['cat_whitespace'].get('med'):.3f} perp={b_entry['perp_share'].get('med'):.3f} "
              f"excluded={b_entry['excluded_frac']:.3f} peaked_only={b_entry['peaked_only_frac']:.3f}) "
              f"sent_frac={entry['sent_frac']:.3f}", flush=True)

    json.dump(_clean(per_layer), open(f"{a.out_prefix}_A1_K_anatomy.json", "w"), indent=2)
    print(f"A.1 saved → {a.out_prefix}_A1_K_anatomy.json")

    # C: явное сравнение 18/19/22/23 (ТЗ §4 шаг 2 — раньше только чтение config.json)
    cmp_keys = ("align_K1_perP", "align_Kp_perP", "energy_top1_perP", "energy_top8_perP",
                "rank_perP_cal", "dimker_perP", "frac_K_from_J_perP")
    cmp = {"layers": {}, "note": "L22=GDN vs L23=full; L19=full (контрольный); локальная аномалия vs общий GDN-vs-full"}
    for l in (18, 19, 22, 23):
        e = per_layer.get(str(l))
        if not e:
            continue
        row = {"layer_type": e["layer_type"]}
        for k in cmp_keys:
            row[k] = e[k]["med"] if isinstance(e.get(k), dict) else e.get(k)
        row["B_cat_content"] = e["B"]["cat_content"]["med"]
        row["B_cat_syntactic"] = (e["B"]["cat_whitespace"]["med"] + e["B"]["cat_punctuation"]["med"]
                                   + e["B"]["cat_newline"]["med"])
        row["B_perp_share"] = e["B"]["perp_share"]["med"]
        cmp["layers"][str(l)] = row
    json.dump(_clean(cmp), open(f"{a.out_prefix}_C_L22_comparison.json", "w"), indent=2)
    print(f"C saved → {a.out_prefix}_C_L22_comparison.json")
    print("done.")


if __name__ == "__main__":
    main()
