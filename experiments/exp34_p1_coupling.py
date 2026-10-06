"""E-34 — Pre-flight 1: где сидит cross-coupling токенов (docs/Exp 31-35.md §E-34).

Per sample, per position, per layer: ESG = J J^T, три уровня снятия
cross-talk (raw / gauge-free P(p)·ESG·P(p) / top-1 removed), K-инвариантная
метрика ρ = RMS(offdiag)/RMS(diag) = r/√(K−1), калибровка по нуль-модели,
контроль ρ_offdiag vs |Δp|.

Математика (все формулы сверены с прямой плотной реализацией в selfcheck):
  off²(ESG)  = ‖ESG‖_F² − ‖diag(ESG)‖_F²                       (тотождество)
  P(p)       = diag(p) − pp^T,  P не проектор (P² ≠ P)
  A = P·ESG·P = (E∘pp^T) − outer(p, E p) − outer(E p, p) + (p^T E p)·pp^T
  top-1 пара ESG — power iteration (ESG PSD), uᵀEu = λ₁,
  off²(ESG − λ₁u₁u₁ᵀ) = ‖ESG‖²_F − λ₁² − ‖diag(ESG − λ₁u₁u₁ᵀ)‖²_F
  ρ_null — E_Q[ρ] эмпирически по 16 зафиксированным ортогональным Q (seed),
  J_rand = Qᵀ·diag(√ESG_ii)  [в тексте ТЗ «diag·Q» — опечатка: тогда
  ESG_rand = diag(d), offdiag ≡ 0 и калибровка вырождается]
  uniformity_0 = (u₁·1)²/K — проекция топ-1 режима на uniform-направление

Вход: dense-кэш exp27b (JR raw, P128 нормализованный, sum=1, + P128_RAW —
сырой sub-stochastic, sum<1). CPU-only, без модели, батченный torch по всем
позициям слоя. Слои {6,12,18,22,23}: L0 в кэше нет, L19 исключаем
(консистентность с E-14).
Агрегация: per-sample → медиана/IQR (пул ESG до отношения некорректен).
Sentinel: позиция с RAW p128 sum < 0.5 исключается из её сэмпла, доля
фиксируется. Читается из P128_RAW (сырые суммы); без P128_RAW в кэше — доля
0 = «нет данных для проверки», а не «сентинелов нет» (механизм на месте).
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

LAYERS = (6, 12, 18, 22, 23)
K = 128
SENTINEL_SUM = 0.5
SEED = 42
PI_ITERS = 60


def _rho_r(off2: float, n_diag: float) -> tuple[float, float]:
    """(ρ, r) из ‖offdiag‖²_F и ‖diag‖_F. 0/0 → (0, 0); off>0, diag=0 → inf."""
    off2 = max(off2, 0.0)
    if n_diag < 1e-12:
        r = 0.0 if off2 < 1e-24 else float("inf")
    else:
        r = np.sqrt(off2) / n_diag
    return r / np.sqrt(K - 1), r


def p_esg_p(ESG: np.ndarray, p: np.ndarray) -> np.ndarray:
    """P·ESG·P (полная матрица): (E∘pp^T) − outer(p,pe) − outer(pe,p) + (p^TEp)pp^T."""
    pe = ESG @ p
    pEp = float(p @ pe)
    pp = np.outer(p, p)
    p2e = p * pe
    return ESG * pp - np.outer(p2e, p) - np.outer(p, p2e) + pEp * pp


def rho_offdiag(M: np.ndarray) -> float:
    off2 = float((M * M).sum() - np.sum(np.diag(M) ** 2))
    n_diag = float(np.linalg.norm(np.diag(M)))
    return _rho_r(off2, n_diag)[0]


def null_rho_batched(row2: "torch.Tensor", Qs: list["torch.Tensor"]) -> "torch.Tensor":
    """ρ_null = E_Q[ρ] для J_rand = Qᵀ·diag(√ESG_ii), Q — случайный ортогональный.

    Честное E_Q[ρ]: для каждого Q отдельно off2 и diag2, затем ρ_Q, затем
    среднее (отношение ожиданий — другое число, ρ нелинеен).
    ‖Q D Q^T‖_F² = Σd² — строго (отмена в f32 катострофична);
    diag(E) = (Q²)ᵀd считается в f64: off2 = s2 − Σ_i E_ii²."""
    import torch
    r64 = row2.double().T.contiguous()  # [K, N] f64
    s2 = (r64 * r64).sum(0)             # [N] = Σd²
    rhos = []
    for q in Qs:
        q2 = (q * q).double().T.contiguous()   # [K, K]
        diag2 = ((q2 @ r64) ** 2).sum(0)       # [N]: Σ_i E_ii²
        off2 = (s2 - diag2).clamp_min(0.0)
        nd = diag2.sqrt()
        r = torch.where(nd < 1e-12,
                        torch.where(off2 < 1e-24, torch.zeros_like(off2),
                                    torch.full_like(off2, float("inf"))),
                        off2.sqrt() / nd) / float(np.sqrt(K - 1))
        rhos.append(r)
    return torch.stack(rhos).mean(0)


def batched_layer(JT: "torch.Tensor", PT: "torch.Tensor",
                  rng: np.random.Generator, Qs: list["torch.Tensor"]) -> np.ndarray:
    """JT [N,K,D] f32, PT [N,K] f32 → [N, 5]: ρ_raw, ρ_gf, ρ_top1, ρ_null, unif0."""
    import torch
    JTc = JT.contiguous()
    ESG = JTc @ JTc.transpose(1, 2)                 # [N,K,K]
    row2 = (JTc * JTc).sum(-1)                       # [N,K] = diag(ESG)
    n_f2 = (ESG * ESG).sum(-1, dtype=torch.float64).sum(-1)  # [N] f64: ‖ESG‖_F²
    n_diag2 = (row2 * row2).sum(-1)                  # [N] ‖diag‖_F²
    off2_raw = n_f2 - n_diag2.double()

    pe = ESG @ PT.unsqueeze(-1)                      # [N,K,1]
    pe = pe.squeeze(-1)
    pEp = (pe * PT).sum(-1)                          # [N]
    pp = PT.unsqueeze(2) * PT.unsqueeze(1)           # [N,K,K] outer(p,p)
    p2e = PT * pe
    A = (ESG * pp - p2e.unsqueeze(2) * PT.unsqueeze(1)
         - PT.unsqueeze(2) * p2e.unsqueeze(1) + pEp.view(-1, 1, 1) * pp)
    # diag(A)_i = p_i²·(row2_i − 2·pe_i + pEp)
    diagv = (PT * PT) * (row2 - 2.0 * pe + pEp.unsqueeze(1))   # [N,K]
    n_diagA2 = (diagv * diagv).sum(-1)                          # [N] ‖diag(A)‖²_F
    off2_gf = (A * A).sum(-1).sum(-1) - n_diagA2

    # power iteration, батченно (фиксированные 60 итер: f32 не сходится до
    # tol 1e-9 за 60, check каждый шаг — лишний sync; проще без break)
    v = torch.from_numpy(rng.standard_normal((JT.shape[0], K)).astype(np.float32)).to(JTc.dtype)  # f64 selfcheck
    v = v / v.norm(dim=1, keepdim=True)
    lam = torch.zeros(JT.shape[0])
    for _ in range(PI_ITERS):
        w = (ESG @ v.unsqueeze(-1)).squeeze(-1)          # [N,K]: ESG·v (O(N·K²))
        wn = w.norm(dim=1, keepdim=True).clamp_min(1e-30)
        v = w / wn
        lam = (v * w).sum(1)
    u = v  # v — топ-1 eigenvector ESG (в token-пространстве [N,K])
    u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-30)
    # canonical sign: dominant element positive
    dom = u.abs().argmax(-1)
    sign = torch.where(u.gather(1, dom.unsqueeze(1)).squeeze(1) < 0,
                       torch.tensor(-1.0, device=u.device), torch.tensor(1.0, device=u.device))
    u = u * sign.unsqueeze(1)
    unif0 = (u.sum(-1) ** 2 / K)
    # offdiag²(ESG − λuuᵀ) = ‖ESG‖²_F − λ² − ‖diag(E1)‖²_F
    diagT = row2 - lam.unsqueeze(1) * u * u
    diagT64 = diagT.double().detach().numpy()            # f64: анти-отмена
    diagT2 = (diagT64 ** 2).sum(-1)
    off2_t1 = n_f2.cpu().numpy() - lam.cpu().numpy() ** 2 - diagT2
    R_null = null_rho_batched(row2, Qs)
    out = np.empty((JT.shape[0], 5), dtype=np.float64)
    oR = off2_raw.cpu().numpy(); dR = n_diag2.cpu().numpy()
    oA = off2_gf.cpu().numpy(); dA2 = n_diagA2.cpu().numpy()
    for i in range(out.shape[0]):
        out[i, 0] = _rho_r(float(oR[i]), np.sqrt(dR[i]))[0]
        out[i, 1] = _rho_r(float(oA[i]), np.sqrt(dA2[i]))[0]
        out[i, 2] = _rho_r(float(off2_t1[i]), float(np.sqrt(diagT2[i])))[0]
        out[i, 3] = float(R_null[i])
        out[i, 4] = float(unif0[i])
    return out


def make_Qs(n: int = 16, seed: int = SEED) -> list["torch.Tensor"]:
    import torch
    rng = np.random.default_rng(seed)
    return [torch.from_numpy(np.linalg.qr(rng.standard_normal((K, K)))[0].astype(np.float32))
            for _ in range(n)]


def process_layer(ck: dict, rng: np.random.Generator, Qs: list["torch.Tensor"]) -> dict:
    import torch
    JR, P128 = ck["JR"], ck["P128"]
    domains, n_samples, S = ck["domains"], ck["n_samples"], ck["S"]
    order = [(dom, si, t) for dom in domains for si in range(n_samples) for t in range(S)]
    JT = torch.stack([JR[k].float() for k in order])            # [N,K,D]
    PT = torch.stack([P128[k].float() for k in order])          # [N,K]
    del JR, P128
    R = batched_layer(JT, PT, rng, Qs)                              # [N,5]
    # Sentinel uses the RAW (sub-stochastic) p128 row-sum: the normalized P128
    # always sums to 1.0, so its row-sum can never trip the <0.5 threshold
    # (that is why sentinel_fraction was stuck at 0 = "no data"). P128_RAW is
    # the raw softmax mass over U128 (sum<1). Fallback: if the ckpt predates the
    # P128_RAW field, we have no raw sums -> sentinel disabled, fraction 0 =
    # "no data to check", NOT "no sentinels" (matches the TЗ sentinel note).
    raw = ck.get("P128_RAW")
    if raw and len(raw) == len(order):
        PT_raw = torch.stack([raw[k].float() for k in order])       # [N,K]
        sent = PT_raw.sum(-1).numpy() < SENTINEL_SUM                # [N]
        sentinel_source = "P128_RAW (raw sub-stochastic sum)"
    else:
        sent = np.zeros(len(order), dtype=bool)                     # [N]
        sentinel_source = "none (no P128_RAW in ckpt)"

    P_all = PT.numpy().reshape(len(domains) * n_samples, S, K)  # [32, S, K]
    dP_all = np.linalg.norm(P_all[:, 1:] - P_all[:, :-1], axis=-1)  # [32, S-1]

    out, med_list, ratios, corr = {}, [], [], []
    n_sent = 0
    for dom_i, dom in enumerate(domains):
        for si in range(n_samples):
            s = dom_i * n_samples * S + si * S
            sl = sent[s:s + S]
            n_sent += int(sl.sum())
            Rm = R[s:s + S][~sl]
            if not Rm.size:
                continue
            med = np.median(Rm, axis=0)
            med_list.append(med)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratios.append([float(np.nanmedian(Rm[:, 0] / Rm[:, 3])),
                               float(np.nanmedian(Rm[:, 1] / Rm[:, 3]))])
            # контроль: ρ_raw(t) vs |Δp(t)| — пары (t,t+1), обе не-sentinel
            dp = dP_all[dom_i * n_samples + si]
            mask = ~sl[:-1] & ~sl[1:]                  # обе позиции пары не-sentinel
            if int(mask.sum()) >= 4:
                c, _ = spearmanr(R[s + np.arange(S)[:-1]][mask, 0], dp[mask])
                corr.append(float(c))
            out[f"{dom}/{si}"] = {
                "median": med.tolist(),
                "iqr": (np.percentile(Rm, 75, axis=0) - np.percentile(Rm, 25, axis=0)).tolist(),
                "uniformity_0": float(np.median(Rm[:, 4])),
                "excluded": int(sl.sum()), "n": int(Rm.shape[0])}
    M = np.stack(med_list)                              # агрегация по per-sample медианам (ТЗ)
    return {
        "per_sample": out,
        "layer_median": np.median(M, axis=0).tolist(),
        "layer_iqr": (np.percentile(M, 75, axis=0) - np.percentile(M, 25, axis=0)).tolist(),
        "null_ratio_median": [float(np.median([r[0] for r in ratios])),
                              float(np.median([r[1] for r in ratios]))],
        "sentinel_positions_excluded": n_sent,
        "sentinel_fraction": n_sent / (len(domains) * n_samples * S),
        "sentinel_source": sentinel_source,
        "correlation_rho_dP": {
            "median": float(np.median(corr)) if corr else None, "per_sample": corr},
    }


def main(ckpt_dir: str, out_path: str, layers=LAYERS, seed: int = SEED,
         sample_cap: int | None = None) -> None:
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rng = np.random.default_rng(seed)
    Qs = make_Qs()
    report: dict = {
        "params": {"ckpt_dir": ckpt_dir, "layers": list(layers), "K": K,
                   "sentinel_sum": SENTINEL_SUM, "seed": seed,
                   "metric": "ρ = RMS(offdiag)/RMS(diag) = r/√(K−1)",
                   "thresholds": "ρ ≪ 0.1 diagonal; ρ ~ 0.3 comparable; ρ ≫ 0.3 offdiag",
                   "layers_note": "L0 в кэше нет; L19 исключён (консистентность с E-14)",
                   "null_model": "E_Q[ρ] (эмпирически, 16 seed-детерминированных Q), "
                                 "J_rand = Qᵀ·diag(√ESG_ii); в тексте ТЗ «diag·Q» — опечатка "
                                 "(ESG_rand = diag(d), offdiag ≡ 0)"},
        "layers": {}}
    for l in layers:
        ck = torch.load(os.path.join(ckpt_dir, f"exp27b_jcache_L{l}.pt"),
                        map_location="cpu", weights_only=False)
        assert len(ck["U128"]) == K
        block = process_layer(ck, rng, Qs)
        del ck
        if sample_cap:
            keys = list(block["per_sample"])[:sample_cap]
            block["per_sample"] = {k: block["per_sample"][k] for k in keys}
            block["smoke_sample_cap"] = sample_cap
        report["layers"][str(l)] = block
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[OK] E-34 -> {out_path}")
    for l, b in report["layers"].items():
        m = b["layer_median"]
        print(f"  L{l}: ρ_raw={m[0]:.4f} ρ_gf={m[1]:.4f} ρ_top1={m[2]:.4f} "
              f"ρ_null={m[3]:.4f} ρ_gf/ρ_null={b['null_ratio_median'][1]:.2f} "
              f"sent={b['sentinel_fraction']:.3f} [{b['sentinel_source']}] "
              f"corr(ρ_raw,dP)={b['correlation_rho_dP']['median']}")


def selfcheck() -> None:
    import torch
    rng = np.random.default_rng(0)
    torch.set_num_threads(4)
    J = rng.standard_normal((K, 1024)).astype(np.float32)
    p = rng.dirichlet(np.ones(K)).astype(np.float32)
    ESG = J @ J.T
    # 1) ρ = r/√(K−1), ρ=0 на диагонали
    Jd = np.diag(np.sqrt(np.linspace(1.0, 0.1, K)))
    assert rho_offdiag(Jd @ Jd.T) < 1e-12
    off2 = float((ESG * ESG).sum() - (np.diag(ESG) ** 2).sum())
    rho_ref = _rho_r(off2, float(np.linalg.norm(np.diag(ESG))))[0]
    r_ref = _rho_r(off2, float(np.linalg.norm(np.diag(ESG))))[1]
    assert abs(rho_ref - r_ref / np.sqrt(K - 1)) < 1e-12
    # 2) p_esg_p vs прямой P·ESG·P
    Pp = np.diag(p) - np.outer(p, p)
    assert np.abs(p_esg_p(ESG, p) - Pp @ ESG @ Pp).max() < 1e-9 * np.abs(ESG).max()
    # P не проектор
    assert np.abs(Pp @ Pp - Pp).max() > 1e-6
    # 3) batched_layer vs поэлементные плотные реализации
    JT = torch.from_numpy(J)
    PT = torch.from_numpy(p)
    Qs = make_Qs(n=16, seed=7)
    R = batched_layer(JT.unsqueeze(0), PT.unsqueeze(0), rng, Qs)
    assert abs(R[0, 0] - rho_ref) < 1e-4, (R[0, 0], rho_ref)
    A = p_esg_p(ESG, p)
    offA2 = float((A * A).sum() - (np.diag(A) ** 2).sum())
    rho_gf_ref = _rho_r(offA2, float(np.linalg.norm(np.diag(A))))[0]
    assert abs(R[0, 1] - rho_gf_ref) < 1e-4, (R[0, 1], rho_gf_ref)
    w, v = np.linalg.eigh(ESG)
    l1 = w[-1]; u1 = v[:, -1]
    if u1[np.argmax(np.abs(u1))] < 0:
        u1 = -u1
    E1 = ESG - l1 * np.outer(u1, u1)
    off12 = float((E1 * E1).sum() - (np.diag(E1) ** 2).sum())
    rho_t1_ref = _rho_r(off12, float(np.linalg.norm(np.diag(E1))))[0]
    assert abs(R[0, 2] - rho_t1_ref) < 1e-3, (R[0, 2], rho_t1_ref)
    assert abs(R[0, 4] - (u1.sum() ** 2 / K)) < 1e-3  # f32: PI tol 1e-9
    # 4) null: batched vs прямое эмпирическое среднее по тем же Q (f64: в f32
    # катострофическая отмена при ‖E‖²−‖diag‖²)
    row2t = torch.from_numpy(np.einsum("ij,ij->i", J, J).astype(np.float64)).unsqueeze(0)
    R_null = null_rho_batched(row2t, Qs)
    r2n = row2t[0].numpy()
    emp = []
    for q in Qs:
        qn = q.numpy().astype(np.float64)
        sq = np.sqrt(r2n)
        Er = (qn * sq) @ (qn * sq).T
        emp.append(rho_offdiag(Er))
    assert abs(float(R_null[0]) - float(np.mean(emp))) < 1e-3, (float(R_null[0]), float(np.mean(emp)))
    # 5) uniform: ker P = span{1} → P·(c·11^T)·P = 0 → ρ_gf = 0 (f64: в f32 шум ~O(1))
    Ju64 = np.ones((K, 1024), dtype=np.float64) * 0.1
    pu64 = np.full(K, 1.0 / K, dtype=np.float64)
    Ru = batched_layer(torch.from_numpy(Ju64).unsqueeze(0),
                       torch.from_numpy(pu64).unsqueeze(0), rng, Qs)
    assert Ru[0, 1] < 1e-9, Ru[0, 1]
    # 6) sentinel-агрегация: доля, пары корреляции, пер-сэмпли медианы.
    # Sentinel читает P128_RAW (raw sub-stochastic): P128 (нормир.) всегда sum=1
    # и не может триггерить порог 0.5. На sentinel-позициях raw sum=0.2 < 0.5.
    S_t = 8
    frng = np.random.default_rng(1)
    JRf, P128f, P128rawf = {}, {}, {}
    for dom in ("a", "b"):
        for t in range(S_t):
            JRf[(dom, 0, t)] = torch.from_numpy(frng.standard_normal((K, 1024)).astype(np.float32))
            praw = frng.dirichlet(np.ones(K)).astype(np.float32)   # sum=1 (raw here)
            if dom == "a" and t in (3, 7):
                praw = praw / praw.sum() * 0.2                     # sentinel: raw sum < 0.5
            P128rawf[(dom, 0, t)] = torch.from_numpy(praw)
            P128f[(dom, 0, t)] = torch.from_numpy(praw / praw.sum())  # normalized, sum=1
    ck = {"JR": JRf, "P128": P128f, "P128_RAW": P128rawf,
          "domains": ["a", "b"], "n_samples": 1, "S": S_t}
    b = process_layer(ck, rng, Qs)
    assert b["sentinel_positions_excluded"] == 2, b["sentinel_positions_excluded"]
    assert abs(b["sentinel_fraction"] - 2.0 / 16.0) < 1e-12
    assert "P128_RAW" in b["sentinel_source"]
    assert b["per_sample"]["a/0"]["n"] == S_t - 2
    assert b["per_sample"]["b/0"]["n"] == S_t
    # пары (t,t+1): в a — (2,3),(3,4),(6,7) исключены → 4 пары; в b — 7
    csp = b["correlation_rho_dP"]["per_sample"]
    assert len(csp) == 2
    # 7) fallback: ckpt без P128_RAW -> sentinel отключён, доля 0 ("нет данных")
    ck_noraw = {"JR": JRf, "P128": P128f, "domains": ["a", "b"], "n_samples": 1, "S": S_t}
    b2 = process_layer(ck_noraw, rng, Qs)
    assert b2["sentinel_positions_excluded"] == 0
    assert "no P128_RAW" in b2["sentinel_source"]
    assert b2["per_sample"]["a/0"]["n"] == S_t
    print("selfcheck OK")


def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/data/prism_kernels/exp27b/results")
    ap.add_argument("--out", default="results/exp34_p1_coupling.json")
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sample-cap", type=int, default=None,
                    help="smoke: первые N сэмплов в отчёте")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        return
    main(a.ckpt_dir, a.out, a.layers, a.seed, a.sample_cap)


if __name__ == "__main__":
    main_cli()
