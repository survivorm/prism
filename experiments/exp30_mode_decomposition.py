#!/usr/bin/env python
"""E-30. Mode decomposition: где сидит остаточный ρ_gf.

ТЗ: docs/E-30.md. Чистая линейная алгебра на exp27b dense-кэше (JR, P128).

Математика (ТЗ):
  A_r = P · ESG_r · P,  ESG_r = Σ_{k≤r} λ_k u_k u_kᵀ   (ТОП-r усечение по модам ESG)
  off²(r) = ‖A_r‖_F² − ‖diag(A_r)‖_F²
  ρ_gf(r) = √(off²(r)/(K−1)) / ‖diag(A_r)‖_F          (r=K → ρ_gf из E-34)
  Δoff²(r) = off²(r) − off²(r−1);  w_r = Δoff²(r)/off²(K)
  PR_modes = 1/Σ_r w_r² — по ВЕСЬМУ спектру r=1..K (consecutive); отрицательные
  Δoff² — вклад в знаменатель, доля отрицательных отчитана (non-additivity = сигнал)
  ρ_null(r) = E_Q[ρ] на diag(ESG_r) = Σ_{k≤r} λ_k u_k[i]², те же 16-Q, что E-34.

Полный спектр off²(1..K) считается Gram-трюком: w_k = P u_k,
  ‖A_r‖_F² = Σ_{k,k'≤r} λ_k λ_{k'} (w_k·w_{k'})²  (2-axis cumsum по G = (WWᵀ)²),
  diag(A_r)_i = Σ_{k≤r} λ_k w_{ki}².
Чанки по позициям (peak ~1-2GB).

Диагностика: топ-5 μ_j(A), v_j (eigh A per chunk), alignment |⟨v_j, u_k⟩|,
unif0(u_k) для топ-5 u_k.

Агрегация: per-sample медианы по позициям, затем медиана + IQR по сэмплам
(протокол E-34). Sentinel: sum(P128) < 0.5 → exclude (неактивен на текущем
кэше, sentinel_fraction ≡ 0 = нет данных, механизм на месте).

Стоимость: torch.set_num_threads(1) — обязательно: на 12 потоках batched
torch.linalg.eigh падает в oversubscription (~2750 мс/позиция против 1 мс).
~1-1.5 мин на слой, <10 мин на 5 слоёв.
"""
import argparse
import json
import os

import numpy as np
import torch

import exp34_p1_coupling as E34

LAYERS = (6, 12, 18, 22, 23)
K = 128
RS_REPORT = (1, 2, 3, 4, 5, 8, 16, 32, 64, 128)
RS_DELTA = 10          # Δoff²(r) для r ∈ 1..10 (consecutive)
TOP = 5                # топ-5 μ_j(A), alignment, unif0
CHUNK = 1024           # чанк по позициям (память)


def _rho(off2: float, n_diag: float) -> float:
    off2 = max(float(off2), 0.0)
    if n_diag < 1e-12:
        return 0.0 if off2 < 1e-24 else float("inf")
    return float(np.sqrt(off2) / n_diag) / np.sqrt(K - 1)


def _rho_batch(off2: torch.Tensor, n_diagA2: torch.Tensor) -> torch.Tensor:
    off2d = off2.double()
    nd = n_diagA2.double().sqrt()
    r = torch.where(nd < 1e-12,
                    torch.where(off2d < 1e-24, torch.zeros_like(off2d),
                                torch.full_like(off2d, float("inf"))),
                    off2d.clamp_min(0.0).sqrt() / nd)
    return (r / np.sqrt(K - 1)).double()


def decompose_batched(JT: torch.Tensor, PT: torch.Tensor,
                      Qs: list[torch.Tensor],
                      rs: tuple = RS_REPORT) -> dict:
    """JT [N,K,D], PT [N,K] → кривая ρ_gf(r)/ρ_null(r)/off²(r) по r ∈ rs,
    off² по всему спектру [N,K], B, lam, топ-5 μ(A)/v(A).

    Вычисления чанками по позициям; выходы [N, ...].
    """
    JTc = JT.contiguous()
    PTf = PT.contiguous()
    N = JTc.shape[0]
    n_rs = len(rs)
    rho_gf = np.empty((N, n_rs), dtype=np.float64)
    off2_rep = np.empty((N, n_rs), dtype=np.float64)
    off2_full = np.empty((N, K), dtype=np.float64)
    B_all = np.empty((N, K, K), dtype=np.float32)
    lam_all = np.empty((N, K), dtype=np.float32)
    row2_rep = np.empty((N, n_rs), dtype=np.float32)
    mu5 = np.empty((N, TOP), dtype=np.float64)
    V5 = np.empty((N, K, TOP), dtype=np.float32)
    for c0 in range(0, N, CHUNK):
        c1 = min(c0 + CHUNK, N)
        Jc, Pc = JTc[c0:c1], PTf[c0:c1]
        ESG = Jc @ Jc.transpose(1, 2)
        lam, B = torch.linalg.eigh(ESG)        # eigh: возрастание
        lam = lam.flip(-1)                      # → убывание: u_1 = топ-мода (ТЗ: λ_1 ≥ λ_2 ≥ …)
        B = B.flip(-1)
        lam_all[c0:c1] = lam.numpy()
        B_all[c0:c1] = B.numpy()
        # --- полный спектр off²(1..K) через Gram-трюк ---
        # оси матрицы: ось 1 = ТОКЕН i, ось 2 = МОДА k.
        pe = (B.transpose(1, 2) @ Pc.unsqueeze(2)).squeeze(-1)   # (pᵀB)_k [nc,K]
        L = lam.clamp_min(0.0)                                   # [nc,K]
        Wt = Pc.unsqueeze(2) * (B - pe.unsqueeze(1))             # w_k[i] [nc,tok,mod]
        M = ((Wt.transpose(1, 2) @ Wt) ** 2) * (L.unsqueeze(2) * L.unsqueeze(1))  # G[k,k'] = W_k·W_k'
        nA2 = torch.diagonal(M.cumsum(-1).cumsum(1), dim1=1, dim2=2)   # [nc,mod] ‖A_r‖²
        del M
        # diag prefix: D[nc,mod,tok] = Σ_{k≤mod} λ_k w_k[i]²
        # (Wt².transpose: [nc,tok,mod]→[nc,mod,tok]; λ_k — по модной оси 1)
        D = (L.unsqueeze(2) * (Wt ** 2).transpose(1, 2)).cumsum(1)
        ndA2 = (D * D).sum(2)                                    # [nc,mod] ‖diag A_r‖²
        del Wt
        del B
        off2f = (nA2 - ndA2).double()
        off2_full[c0:c1] = off2f.numpy()
        for i, r in enumerate(rs):
            off2_rep[c0:c1, i] = off2f[:, r - 1].numpy()
            rho_gf[c0:c1, i] = _rho_batch(off2f[:, r - 1], ndA2[:, r - 1]).numpy()
        # --- топ-5 μ, v по A = P·ESG·P (ESG тот же, что для eigh) ---
        peA = (ESG @ Pc.unsqueeze(2)).squeeze(-1)
        pEp = (peA * Pc).sum(-1)
        pp = Pc.unsqueeze(2) * Pc.unsqueeze(1)
        p2e = Pc * peA
        A = (ESG * pp - p2e.unsqueeze(2) * Pc.unsqueeze(1)
             - Pc.unsqueeze(2) * p2e.unsqueeze(1) + pEp.view(-1, 1, 1) * pp)
        del ESG
        muA, VA = torch.linalg.eigh(A)
        top_mu = muA[:, -TOP:].flip(1)               # по убыв.
        top_v = VA.flip(-1)[:, :, :TOP].contiguous()
        mu5[c0:c1] = top_mu.numpy()
        V5[c0:c1] = top_v.numpy()
    # diag(ESG_r) = Σ_{k≤r} λ_k u_k[i]² → row2 для null (per-r)
    # diag(ESG_r) = Σ_{k≤r} λ_k u_k[i]² — cumsum по МОДОВОЙ оси (axis 2),
    # lam/b построчно совпадают по порядку (ascending eigh, B_all уже перевёрнут)
    B_t = torch.from_numpy(B_all)
    lt = torch.from_numpy(lam_all).clamp_min(0).unsqueeze(1)       # [N,1,K(mode)]
    cum1 = (lt * (B_t ** 2)).cumsum(2)                             # [N, K(pos), K(mode)]
    row2_rep = torch.stack([cum1[:, :, r - 1] for r in rs], dim=1).numpy().astype(np.float32)
    rho_null = np.empty((N, n_rs), dtype=np.float64)
    for i in range(n_rs):
        rho_null[:, i] = E34.null_rho_batched(torch.from_numpy(row2_rep[:, i]), Qs).numpy()
    return dict(rho_gf_r=rho_gf, rho_null_r=rho_null, off2=off2_rep,
                off2_full=off2_full, B=B_all, lam=lam_all,
                mu5=mu5, V5=V5)


def per_sample_stats(vals: np.ndarray, n_blocks: int, n_pos: int):
    """[N, *shape] (порядок dom→si→t) → [med, q1, q3] покомпонентно: reshape на
    блоки (dom,si), медиана по позициям внутри блока, затем медиана/IQR по блокам
    на каждом компоненте *shape."""
    v = vals.reshape(n_blocks, n_pos, *vals.shape[1:])
    sm = np.median(v, axis=1)
    med = np.median(sm, axis=0)
    q1 = np.percentile(sm, 25, axis=0)
    q3 = np.percentile(sm, 75, axis=0)
    return [med.tolist() if hasattr(med, "tolist") else float(med),
            q1.tolist() if hasattr(q1, "tolist") else float(q1),
            q3.tolist() if hasattr(q3, "tolist") else float(q3)]


def process_layer(ck: dict, Qs: list[torch.Tensor], sample_cap: int | None = None) -> dict:
    domains, S, n_samples = ck["domains"], ck["S"], ck["n_samples"]
    P128, JR = ck["P128"], ck["JR"]
    keys = [(dom, si, t) for dom in domains for si in range(n_samples) for t in range(S)]
    if sample_cap is not None:
        keys = [k for k in keys if k[1] < sample_cap]
    PT_raw = torch.stack([P128[k].float() for k in keys])
    sent = (PT_raw.sum(-1).numpy() < E34.SENTINEL_SUM)
    n_sent = int(sent.sum())
    mask = ~torch.from_numpy(sent)
    PTv = PT_raw[mask]
    JRv = torch.stack([JR[k].float() for k, s in zip(keys, sent) if not s]).contiguous()
    Nv = PTv.shape[0]
    out = decompose_batched(JRv, PTv, Qs)
    Bn, lam = out["B"], out["lam"]
    u_top5 = Bn[:, :, :TOP]                         # u_1 = топ-мода, ..., u_5 (убыв.)
    # i — токенная ось: |⟨v_j, u_k⟩| по токенам → [N,5,5]
    al = np.abs(np.einsum("nij,nik->njk", out["V5"], u_top5))
    unif0_u = (u_top5.sum(1) ** 2) / K               # [N,5] (Σ_i u_k[i])²/K
    lam_top5 = lam[:, :TOP]                          # [N,5] по убыв.
    rho_gf_r, rho_null_r, off2 = out["rho_gf_r"], out["rho_null_r"], out["off2"]
    off2f = out["off2_full"]                         # [N,K] весь спектр
    idx = {r: i for i, r in enumerate(RS_REPORT)}
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rho_null_r < 1e-12, np.inf, rho_gf_r / rho_null_r)
    # Δoff²(r), consecutive, r=1..K; отчёт r=1..10
    d_full = np.empty_like(off2f)
    d_full[:, 0] = off2f[:, 0]
    d_full[:, 1:] = np.diff(off2f, axis=1)
    offK = off2f[:, -1]
    safe = np.where(offK < 1e-24, 1.0, offK)
    wk = d_full / safe[:, None]
    n_samp_eff = n_samples if sample_cap is None else min(sample_cap, n_samples)
    n_blk = len(domains) * n_samp_eff
    npos = Nv // n_blk
    # E-30b: полный спектр Δoff²(r), r=1..K: кумулята, per-r статистики, top-20
    off2_cum = off2f / np.where(offK < 1e-12, 1.0, offK)[:, None]
    delta_stats = per_sample_stats(d_full, n_blk, npos)
    cum_stats = per_sample_stats(off2_cum, n_blk, npos)
    neg_mask = (d_full < 0).astype(np.float32)
    neg_frac_per_r = np.median(neg_mask.reshape(n_blk, npos, K).mean(1), axis=0)
    med_delta = np.array(delta_stats[0])
    abs_med = np.abs(med_delta)
    top_idx = np.argsort(abs_med)[::-1][:20]
    top20 = [{"r": int(i + 1), "med_delta": float(med_delta[i]),
              "abs_med_delta": float(abs_med[i]),
              "neg_frac": float(neg_frac_per_r[i])} for i in top_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = (wk.sum(1) ** 2) / np.where((wk * wk).sum(1) < 1e-30, np.inf, (wk * wk).sum(1))
    res = {
        "n_positions": int(Nv),
        "sentinel_positions_excluded": n_sent,
        "sentinel_fraction": n_sent / (len(domains) * n_samples * S),
        "rho_gf_r": {str(r): per_sample_stats(rho_gf_r[:, idx[r]], n_blk, npos)
                     for r in RS_REPORT},
        "rho_gf_over_null_r": {str(r): per_sample_stats(ratio[:, idx[r]], n_blk, npos)
                               for r in RS_REPORT},
        "delta_off2_w_top10": [per_sample_stats(d_full[:, j] / safe, n_blk, npos)
                               for j in range(RS_DELTA)],
        "PR_modes": per_sample_stats(pr, n_blk, npos),
        "neg_doff2_fraction": float(np.median((d_full < 0).mean(1))),
        "full_spectrum": {
            "r": list(range(1, K + 1)),
            "delta_off2_med": delta_stats[0], "delta_off2_q1": delta_stats[1],
            "delta_off2_q3": delta_stats[2],
            "off2_cum_med": cum_stats[0], "off2_cum_q1": cum_stats[1],
            "off2_cum_q3": cum_stats[2],
            "neg_frac_per_r": neg_frac_per_r.tolist(),
            "top20_by_abs_delta": top20,
        },
        "spectrum_top5": {
            "mu_A": per_sample_stats(out["mu5"], n_blk, npos),
            "lam_ESG": per_sample_stats(lam_top5, n_blk, npos),
            "alignment_v_u": per_sample_stats(al, n_blk, npos),
            "unif0_u_k": per_sample_stats(unif0_u, n_blk, npos),
        },
    }
    return res


def main(ckpt_dir: str, out_path: str, layers=LAYERS, seed: int = E34.SEED,
         sample_cap: int | None = None, full_spectrum_layers=(18, 22)) -> dict:
    torch.set_num_threads(1)
    Qs = E34.make_Qs(16, seed)
    rep = {"experiment": "E-30 mode decomposition",
           "layers": list(layers), "K": K, "rs": list(RS_REPORT),
           "null_model": "16-Q empirical E_Q[rho], J_rand = Q^T diag(sqrt(ESG_ii)), per-r on diag(ESG_r)",
           "aggregation": "per-sample medians -> median/IQR over samples (E-34 protocol)",
           "sentinel_sum": E34.SENTINEL_SUM, "seed": seed}
    rep_full = {}
    for l in layers:
        ck = torch.load(f"{ckpt_dir}/exp27b_jcache_L{l}.pt",
                                      map_location="cpu", weights_only=True)
        res = process_layer(ck, Qs, sample_cap)
        rep[l] = res
        pr = res["PR_modes"][0]
        print(f"L{l}: PR_modes={pr:.1f}  rho_gf(5)={res['rho_gf_r']['5'][0]:.4f} "
              f"rho_gf(128)={res['rho_gf_r']['128'][0]:.4f}  "
              f"ratio(5)={res['rho_gf_over_null_r']['5'][0]:.2f} "
              f"ratio(128)={res['rho_gf_over_null_r']['128'][0]:.2f}", flush=True)
        if l in full_spectrum_layers:
            rep_full[str(l)] = res.pop("full_spectrum")
        else:
            res.pop("full_spectrum")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if rep_full:
        fpath = out_path.replace("mode_decomposition", "full_spectrum")
        with open(fpath, "w") as f:
            json.dump({"experiment": "E-30b full spectrum d(off^2)(r), r=1..K",
                       "layers": {str(l): s for l, s in rep_full.items()},
                       "aggregation": rep["aggregation"]},
                      f, indent=2, ensure_ascii=False)
        print(f"wrote {fpath}")
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path}")
    return rep


def _dense_rho_gf(Jn: np.ndarray, Pn: np.ndarray, kcheck: int) -> float:
    """Dense-референс ρ_gf(r) для одной позиции: топ-k по eigh (возр. → с конца). f64."""
    Jn, Pn = Jn.astype(np.float64), Pn.astype(np.float64)
    E = Jn @ Jn.T
    w, U = np.linalg.eigh(E)
    Er = (U[:, -kcheck:] * w[-kcheck:]) @ U[:, -kcheck:].T
    p = Pn
    pe = Er @ p
    pEp = float(p @ pe)
    Ar = (Er * np.outer(p, p) - np.outer(p * pe, p)
          - np.outer(p, p * pe) + pEp * np.outer(p, p))
    return _rho(float((Ar * Ar).sum() - np.sum(np.diag(Ar) ** 2)),
                float(np.linalg.norm(np.diag(Ar))))


def selfcheck() -> None:
    """Dense-референс (numpy eigh) vs batched: ρ_gf(r) r∈{2,5} + r=1, null(r) r=3,
    топ-5 μ(A) vs eigh, Gram off²(1) vs dense, PR конечность и ≥1, r=K = E-34 ρ_gf."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    N, D = 64, 256
    JT = torch.from_numpy(rng.standard_normal((N, K, D)).astype(np.float32))
    PT = torch.from_numpy(rng.dirichlet(np.ones(K), size=N).astype(np.float32))
    Qs = E34.make_Qs(4, 0)
    out = decompose_batched(JT, PT, Qs)
    Jn, Pn = JT.numpy(), PT.numpy()
    # 1) кривая ρ_gf(r) dense (топ-k с конца спектра)
    for kcheck in (1, 2, 5):
        ref = [_dense_rho_gf(Jn[i], Pn[i], kcheck) for i in range(N)]
        ik = {r: i for i, r in enumerate(RS_REPORT)}[kcheck]
        got = out["rho_gf_r"][:, ik]
        assert np.allclose(got, ref, atol=1e-4), (kcheck, np.abs(got - np.array(ref)).max())
    # 1b) off²(1) из Gram vs dense: off² = ρ(1)² · (K−1) · ‖diag A_1‖²
    for i in range(4):
        ref_rho = _dense_rho_gf(Jn[i], Pn[i], 1)
        Jn_i, Pn_i = Jn[i].astype(np.float64), Pn[i].astype(np.float64)
        E_i = Jn_i @ Jn_i.T
        w_i, U_i = np.linalg.eigh(E_i)
        u_i = U_i[:, -1]
        Er_i = np.outer(u_i, u_i) * w_i[-1]
        pe_i = Er_i @ Pn_i
        Ar_i = (Er_i * np.outer(Pn_i, Pn_i) - np.outer(Pn_i * pe_i, Pn_i)
                - np.outer(Pn_i, Pn_i * pe_i) + float(Pn_i @ pe_i) * np.outer(Pn_i, Pn_i))
        diag2 = float(np.sum(np.diag(Ar_i) ** 2))
        ref_off2 = ref_rho ** 2 * (K - 1) * diag2
        assert abs(out["off2_full"][i, 0] - ref_off2) < 1e-3 * max(1.0, ref_off2), i
    # 2) r=K совпадает с E-34 gauge-free
    iK = {r: i for i, r in enumerate(RS_REPORT)}[K]
    e34 = E34.batched_layer(JT, PT, np.random.default_rng(0), E34.make_Qs(16, E34.SEED))
    assert np.allclose(out["rho_gf_r"][:, iK], e34[:, 1], atol=1e-5), \
        np.abs(out["rho_gf_r"][:, iK] - e34[:, 1]).max()
    # 2b) off2_full[:, -1] == off2(r=K)
    assert np.allclose(out["off2_full"][:, -1], out["off2"][:, iK], rtol=1e-9, atol=0), \
        np.abs(out["off2_full"][:, -1] - out["off2"][:, iK]).max()
    # 3) null(r) dense: E_Q[ρ] на diag(ESG_r), r=3 (топ-3)
    i3 = {r: i for i, r in enumerate(RS_REPORT)}[3]
    for i in range(8):
        E = Jn[i] @ Jn[i].T
        w, U = np.linalg.eigh(E)
        d3 = (U[:, -3:] ** 2 * w[-3:]).sum(1)   # diag(ESG_3)_i по токенам
        rhos = []
        for q in Qs:
            qn = q.numpy()
            # ESG_rand = Q·diag(d3)·Qᵀ (d3 = diag(ESG_r); Q·diag(√d)·Qᵀ — это J, а не JJᵀ)
            Er = qn @ np.diag(d3.astype(np.float64)) @ qn.T
            Er = (Er + Er.T) / 2
            rhos.append(_rho(float((Er * Er).sum() - np.sum(np.diag(Er) ** 2)),
                             float(np.linalg.norm(np.diag(Er)))))
        assert abs(out["rho_null_r"][i, i3] - np.mean(rhos)) < 1e-3, i
    # 4) топ-5 μ(A) vs eigh (eigh в коде → точное сравнение)
    i = 0
    ESG = Jn[i] @ Jn[i].T
    p = Pn[i]
    pe = ESG @ p
    pEp = float(p @ pe)
    A = (ESG * np.outer(p, p) - np.outer(p * pe, p)
         - np.outer(p, p * pe) + pEp * np.outer(p, p))
    A = (A + A.T) / 2
    w_ref = np.linalg.eigvalsh(A)[::-1][:TOP]
    assert np.allclose(out["mu5"][i], w_ref, rtol=1e-4, atol=1e-3), (out["mu5"][i], w_ref)
    # 5) PR: Δoff² consecutive, конечность и ≥1; w суммируется ≈ 1
    off2f = out["off2_full"][:8]
    d = np.empty_like(off2f)
    d[:, 0] = off2f[:, 0]
    d[:, 1:] = np.diff(off2f, axis=1)
    offK = off2f[:, -1]
    wk = d / np.where(offK < 1e-24, 1.0, offK)[:, None]
    pr = (wk.sum(1) ** 2) / np.where((wk * wk).sum(1) < 1e-30, np.inf, (wk * wk).sum(1))
    assert np.all(np.isfinite(pr)), pr
    assert np.all(pr >= 1.0 - 1e-9), pr
    # 5b) full-spectrum off²(r) и PR_modes: независимый dense-референс (numpy f64)
    # J1 — low-rank с разделёнными сингулярными значениями (невырожденная кривая)
    P0 = np.random.default_rng(3).dirichlet(np.ones(K) * 4.0)
    rng_j = np.random.default_rng(5)
    Uq, _ = np.linalg.qr(rng_j.normal(size=(K, 4)))
    Vq, _ = np.linalg.qr(rng_j.normal(size=(K, 4)))
    J1 = (Uq @ np.diag([8.0, 4.0, 2.0, 1.0]) @ Vq.T)   # [K,K] rank-4
    P1v = P0 / P0.sum()
    ESG1 = J1 @ J1.T
    pe1 = ESG1 @ P1v
    A1 = (ESG1 * np.outer(P1v, P1v) - np.outer(P1v * pe1, P1v)
          - np.outer(P1v, P1v * pe1) + float(P1v @ pe1) * np.outer(P1v, P1v))
    # плотный референс: A_r = P·E_r·P, E_r = Σ_{k≤r} λ_k u_k u_kᵀ (по убыванию)
    eA1 = np.linalg.eigh((ESG1 + ESG1.T) / 2)          # ascending
    wA, UA = eA1[0][::-1], eA1[1][:, ::-1]
    def _pM(M):
        peM = M @ P1v
        return (M * np.outer(P1v, P1v) - np.outer(P1v * peM, P1v)
                - np.outer(P1v, P1v * peM) + float(P1v @ peM) * np.outer(P1v, P1v))
    Er, s_ref = np.zeros((K, K)), []
    for kk in range(K):
        Er += wA[kk] * np.outer(UA[:, kk], UA[:, kk])
        Ar = _pM(Er)
        s_ref.append(float(np.sum(Ar * Ar) - np.sum(np.diag(Ar) ** 2)))
    out1 = decompose_batched(torch.from_numpy(J1[None].astype(np.float32)),
                             torch.from_numpy(P1v[None].astype(np.float32)), Qs)
    got_curve = out1["off2_full"][0]
    # кривая off²(r): плотный референс на всём спектре (f32 eigh, rel ~1e-3)
    tol = 2e-3 * float(np.max(np.abs(s_ref)))
    assert np.max(np.abs(got_curve - np.array(s_ref))) < tol, \
        np.max(np.abs(got_curve - np.array(s_ref)))
    # PR из кривой кода
    d1 = np.diff(np.concatenate([[0.0], got_curve]))
    w1 = d1 / got_curve[-1]
    pr_code = float(1.0 / (w1 * w1).sum())
    # PR из плотной кривой
    d1r = np.diff(np.concatenate([[0.0], s_ref]))
    w1r = d1r / s_ref[-1]
    pr_ref = float(1.0 / (w1r * w1r).sum())
    assert abs(pr_code - pr_ref) / pr_ref < 1e-2, (pr_code, pr_ref)
    assert pr_code >= 1.0 - 1e-9
    # 5c) alignment |⟨v_j,u_k⟩| и unif0(u_k): плотный референс (топенная ось = K)
    V5t = rng.normal(size=(3, K, TOP))
    ut  = rng.normal(size=(3, K, TOP))
    al_t = np.abs(np.einsum("nij,nik->njk", V5t, ut))
    assert al_t.shape == (3, TOP, TOP)
    for i in range(3):
        ref_al = np.abs(V5t[i].T @ ut[i])          # [5,5]
        assert np.max(np.abs(al_t[i] - ref_al)) < 1e-9
    uf_t = (ut.sum(1) ** 2) / K                     # [3,5]
    for i in range(3):
        assert np.max(np.abs(uf_t[i] - (ut[i].sum(0) ** 2) / K)) < 1e-9
    print("selfcheck OK")


def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/data/prism_kernels/exp27b/results")
    ap.add_argument("--out", default="results/exp30_mode_decomposition.json")
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    ap.add_argument("--sample-cap", type=int, default=None)
    ap.add_argument("--full-spectrum-layers", type=int, nargs="+",
                    default=[18, 22],
                    help="слои, для которых полный спектр d(off^2)(r) пишется в "
                         "results/exp30_full_spectrum.json (по умолчанию 18 22)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        return
    main(a.ckpt_dir, a.out, a.layers, sample_cap=a.sample_cap,
         full_spectrum_layers=tuple(a.full_spectrum_layers))


if __name__ == "__main__":
    main_cli()
