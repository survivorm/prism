#!/usr/bin/env python3
"""E-39 supplement: что на самом деле стоит за «98–99%» (ТЗ `docs/E39+.md`, ч. 2).

Шесть раундов в отчётах ходило «98–99%» под одной вывеской. Здесь считается три объекта
по слоям 6/12/18/22/23 на кэше exp27b (N=32 доков, K=128, S=128), без новых форвардов:

  O1  operator-kill      `‖P(p_t)J_t‖²_F / ‖J_t‖²_F`        — доля, убитая оператором;
      + изотропный референс `‖P(p)‖²_F / K` (замкнутая форма, без gauge-информации).
  O2  `g_uni`            `‖Q J‖²_F / ‖J‖²_F`, `Q = 11ᵀ/K`   — реальная uniform-доля
      энергии сырого J (paper §2.5).
  O3  ESG `gauge_share`  `Σ_k (u_kᵀ1)² λ_k / Σ_k λ_k` по ESG `G = Σ_t J_t J_tᵀ`
      (paper §2.3) + топ-1 прокси `λ₁·uniformity(u₁)/tr G` (именно его отчётен E-10a).

Плюс контрольные алгебраические соотношения (§1.4 ТЗ):
  (a) O3 = `1ᵀG1 / (K·tr G)` = O2 на том же G — тождество, проверяется численно;
  (b) `g_uni(J^gf) ≡ 0`, потому что `P(p)·1 = p(1 − Σp) = 0` при нормированном p —
      «доля gauge» после применения P пуста по построению;
  (c) O1 vs `(1 − g_uni)·O1_iso` — выполняется ли «операторная» доля вообще без данных.

Якоря (не «улучшать», сверять): E-39k `results/exp39k_gauge_aware_importance.json`
(O1 L23 = 4.72e-4, iso = 6.90e-4, g_uni L23 = 0.140 / L18 = 0.142) и E-33r
`survive_frac = ‖PJ‖/‖J‖ ∈ [0.010, 0.023]` (docs/ref-exp.md, карточка E-33r).

Запуск: `python experiments/exp39_supp_three_objects.py` → `results/exp39_supp_three_objects.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _e38_cache import DOMAINS, N_PER_DOMAIN, S, load_layer  # noqa: E402
from exp39k_gauge_importance import _iso_P_reference  # noqa: E402  (та же замкнутая форма)

LAYERS = (6, 12, 18, 22, 23)
CHUNK = 32                      # позиций за раз: J-чанк в fp64 = CHUNK·K·D·8 ≈ 34 MB
# якоря для самопроверок (единственные числовые литералы, кроме порогов сверки) — сняты
# побитово с results/exp39k_gauge_aware_importance.json : layers_diag
ANCHOR_E39K = {23: {"o1": 0.0004724755626467771, "iso": 0.0006900831821251604,
                     "g_uni": 0.14033701687050754},
                18: {"o1": 0.0002057266238312365, "iso": 0.0006900831821251604,
                     "g_uni": 0.14248665098072735}}
E33R_SURVIVE = (0.010, 0.023)   # docs/ref-exp.md, карточка E-33r
E10A_TOP1_PROXY = {6: 0.620, 12: 0.474, 18: 0.349, 23: 0.165, 0: 0.111}
ANCHOR_TOL = 0.05               # 5% на O1/iso, g_uni; E-39k считал iso по 1024 позициям
GF_UNI_MAX = 1e-12              # «доля gauge» у J^gf обязана быть нулевой алгебраически
NULL_STRIDE = 1                 # null-контроли по ВСЕМ позициям (раунд 4: шаг 32 давал 4
                               # позиции на документ — per-doc/per-domain оценки не сходились
                               # с корпусными даже по величине)
NULL_CHUNK = 32                 # позиций в чанке null-контроля
NULL_SEED = 0
BOOT = 2000                     # парный bootstrap по документам (для real/dir)
# E-37 sentinel: позиция — артефакт обрезки, если суммарная масса P128_RAW < 0.5
# (experiments/exp37_p_sensitivity.py:45). E-37 считает свои метрики по alive-позициям,
# весь наш ряд (E-33r/E-39k/это дополнение) — по всем, поэтому оба варианта пишем отдельно.
SENT_RAW_MASS = 0.5
# Kопия кэша с полем P128_RAW (его дописывал E-37; рядом лежат *.pre_raw.bak). В файлах под
# kaggle_kernel/results поля P128_RAW нет, так что sentinel-маска берётся отсюда.
RAW_CACHE_DIR = "/data/prism_kernels/exp27b/results"
# frac_K_from_J = ‖K‖²_F/‖J‖²_F (ЭНЕРГИЯ, не норма: exp37_p_sensitivity.py:248),
# медиана по alive-позициям; снято с results/exp37_A1_K_anatomy.json : <L>/frac_K_from_J_perP/med
E37_FRAC_K_ENERGY = {6: 0.996558, 12: 0.996758, 18: 0.998127, 22: 0.998288, 23: 0.993934}


def _spearman(x, y) -> float:
    """Ранговая корреляция без scipy (128 строк, связей нет)."""
    rx = np.argsort(np.argsort(np.asarray(x, dtype=np.float64)))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=np.float64)))
    return float(np.corrcoef(rx, ry)[0, 1])


def _o1_rel(a, e, iso) -> float:
    """O1/iso по группе позиций: (Σa/Σe)/среднего ‖P‖²_F/K.

    a_i = ‖P(p_i)J_i‖²_F, e_i = ‖J_i‖²_F, iso_i = ‖P(p_i)‖²_F/K — то же определение,
    что и в корпусе, только на группе (нужно для per-doc/per-domain null-контролей).
    """
    a = np.asarray(a, dtype=np.float64); e = np.asarray(e, dtype=np.float64)
    iso = np.asarray(iso, dtype=np.float64)
    return float((a.sum() / e.sum()) / iso.mean())


def _boot_median_ratio(vals, boot=BOOT, seed=0):
    """Медиана по документам + bootstrap CI (парный: один и тот же бут документов)."""
    v = np.asarray(vals, dtype=np.float64)
    gen = np.random.default_rng(seed)
    meds = np.median(v[gen.integers(0, v.size, (boot, v.size))], axis=1)
    return [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]


def _sentinel_mask(layer: int, P: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """E-37 sentinel: позиция — артефакт обрезки, если P128_RAW.sum(-1) < SENT_RAW_MASS.

    Поле P128_RAW есть только в дополненной копии кэша (RAW_CACHE_DIR), поэтому перед
    употреблением проверяем, что это ТОТ ЖЕ кэш: P128 совпадает с загруженным, JR —
    побитово на 8 детерминированно выбранных ключах.
    """
    path = os.path.join(RAW_CACHE_DIR, f"exp27b_jcache_L{layer}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"L{layer}: нет копии кэша с P128_RAW ({path})")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if "P128_RAW" not in raw:
        raise KeyError(f"L{layer}: в {path} нет P128_RAW")
    out = torch.empty(P.shape, dtype=torch.float32)
    pr = torch.empty(P.shape, dtype=torch.float32)
    di = 0
    for d_i, dom in enumerate(DOMAINS):
        for si in range(N_PER_DOMAIN):
            for t in range(S):
                out[di, t] = raw["P128_RAW"][(dom, si, t)].float()
                pr[di, t] = raw["P128"][(dom, si, t)].float()
            di += 1
    # сверка «это тот же кэш»: P128 целиком, JR — побитово на 8 детерминированных ключах
    assert (pr - P).abs().max() <= 1e-6, f"L{layer}: P128 в копии кэша не совпадает с рабочим"
    for k in range(8):
        key = (DOMAINS[(k * 7919) % len(DOMAINS)], (k * 6151) % N_PER_DOMAIN,
               (k * 104729) % S)
        doc = DOMAINS.index(key[0]) * N_PER_DOMAIN + key[1]
        assert torch.equal(raw["JR"][key], J[doc, key[2]].to(raw["JR"][key].dtype)), \
            f"L{layer}: JR[{key}] в копии кэша не совпадает с рабочим"
    del raw
    return out.sum(-1) < SENT_RAW_MASS


def _layer_numbers(layer: int) -> dict:
    """Слой → O1/O2/O3 + контрольные соотношения. Один проход по позициям."""
    ck = load_layer(layer, want=("J", "P"))
    J, P = ck["J"], ck["P"]                      # [N, S, K, D], [N, S, K]
    u128 = list(ck["U128"])
    n_docs, s_len, K, D = J.shape
    assert (s_len, n_docs) == (S, len(DOMAINS) * N_PER_DOMAIN), f"L{layer}: {J.shape}"
    n_pos = n_docs * s_len

    e_raw_s, e_P_s, uni_s, iso_s = [], [], [], []          # per position (fp64 скаляры)
    cross_s, p2_s, p3_s, k_raw_s = [], [], [], []          # ⟨J,PJ⟩, Σp², Σp³, ‖J−PJ‖²
    b_s, bc_s = [], []                                     # b_t = Σᵢ dᵢ,  b_t·Σp²
    dev_row = torch.zeros(K, dtype=torch.float64)           # Σₜ ‖J_{t,i} − m_t‖²
    p_row = torch.zeros(K, dtype=torch.float64)             # Σₜ p_{t,i}
    gf_uni_max = 0.0
    G_doc = torch.zeros(K, K, dtype=torch.float64)          # Σ_t J_t J_tᵀ по всем докам
    per_doc = []                                            # (Σe_raw, Σe_P, Σuni, trG, 1ᵀG1, iso)
    for d in range(n_docs):
        ed, ep, un, isod = 0.0, 0.0, 0.0, 0.0
        G = torch.zeros(K, K, dtype=torch.float64)
        for c0 in range(0, s_len, CHUNK):
            js = slice(c0, min(c0 + CHUNK, s_len))
            Jc = J[d][js].double()                        # [c, K, D]
            Pc = P[d][js].double()                        # [c, K]
            raw = Jc.pow(2).sum(dim=(1, 2))                # [c] = ‖J_t‖²_F
            m = torch.einsum("sk,skd->sd", Pc, Jc)         # [c, D] = Σ_j p_j J_j
            PJ = Pc.unsqueeze(-1) * (Jc - m.unsqueeze(1))  # [c, K, D] = P(p_t) J_t
            eP = PJ.pow(2).sum(dim=(1, 2))
            j1 = Jc.sum(dim=1)                             # [c, D] = J_tᵀ 1
            uni = j1.pow(2).sum(dim=1)                     # ‖J_tᵀ1‖²
            j1gf = PJ.sum(dim=1)                           # (P J)ᵀ1 = JᵀPᵀ1 = 0 по алгебре
            gf_uni_max = max(gf_uni_max, float((j1gf.pow(2).sum(1)
                                                / raw.clamp_min(1e-300)).max()))
            s2 = Pc.pow(2).sum(-1)
            iso = (s2 - 2 * Pc.pow(3).sum(-1) + s2.pow(2)) / K   # ‖P(p)‖²_F / K
            # точная форма действия P: (P(p)J)ᵢ = pᵢ(Jᵢ − m), m = ΣⱼpⱼJⱼ ⇒ ‖PJ‖² = Σᵢpᵢ²dᵢ
            dev = (Jc - m.unsqueeze(1)).pow(2).sum(-1)      # [c, K] = dᵢ
            b = dev.sum(1)                                   # [c]
            Kc = Jc - PJ                                     # = (I−P)J, объект E-37
            cross_s.append((Jc * PJ).sum(dim=(1, 2)))         # ⟨J_t, P J_t⟩
            k_raw_s.append(Kc.pow(2).sum(dim=(1, 2)))         # ‖K_t‖²_F
            p2_s.append(s2); p3_s.append(Pc.pow(3).sum(-1))
            del Kc
            dev_row += dev.sum(0); p_row += Pc.sum(0)
            b_s.append(b); bc_s.append(b * s2)
            e_raw_s.append(raw); e_P_s.append(eP); uni_s.append(uni); iso_s.append(iso)
            ed += float(raw.sum()); ep += float(eP.sum()); un += float(uni.sum())
            isod += float(iso.sum())
            # G = Σ_t J_t J_tᵀ одним BLAS-матмулом: A[k, (t,d)] = J[t,k,d]
            A = J[d][js].permute(1, 0, 2).reshape(K, -1).double()
            GG = A @ A.t()
            G += GG
            del Jc, Pc, PJ, A, GG
        G_doc += G
        per_doc.append((ed, ep, un, float(G.trace()), float(G.sum()), isod / s_len))
    e_raw = torch.cat(e_raw_s); e_P = torch.cat(e_P_s)
    uni = torch.cat(uni_s); iso = torch.cat(iso_s)
    cross = torch.cat(cross_s); k_raw = torch.cat(k_raw_s)
    p2 = torch.cat(p2_s); p3 = torch.cat(p3_s)
    del ck
    sent = _sentinel_mask(layer, P, J)                  # [N, S] — тот же срез, что у E-37
    alive = (~sent).reshape(-1)                          # [n_pos] в том же порядке
    assert int(alive.sum()) > 0
    b_all = torch.cat(b_s); bc_all = torch.cat(bc_s)
    assert e_raw.min() > 0
    # (P J)ᵢ = pᵢ(Jᵢ − m) ⇒ Σ‖PJ‖² обязано совпасть с Σᵢ p²dᵢ — это и есть a_sum ниже
    a_sum, b_sum, bc_sum = float(e_P.sum()), float(b_all.sum()), float(bc_all.sum())

    o1_corpus = float(e_P.sum() / e_raw.sum())
    o1_pos = (e_P / e_raw).numpy()
    iso_mean = float(iso.mean())
    # изотропный референс считается дважды: inline и замкнутой формой из E-39k
    iso_ref = _iso_P_reference(P.reshape(-1, K), max_pos=10 ** 9)
    assert abs(iso_ref - iso_mean) <= 1e-9 * abs(iso_mean), \
        f"L{layer}: iso {iso_mean!r} расходится с E-39k-версией {iso_ref!r}"
    p_norm2 = iso_mean * K                                # средний ‖P(p)‖²_F
    # точное (не приближённое) разложение: O1/iso = conc × V × sharpness
    #   conc      = K·Σa/Σ(b·c) — где энергия отклонений строк относительно p² (=1 если ровно)
    #   V         = Σb/Σe_raw  — энергия отклонений строк против сырой энергии
    #   sharpness = c_eff/‖P‖²_F, c = Σp², c_eff = Σ(b·c)/Σb — насколько ‖P‖² ниже Σp²
    conc = float(K * a_sum / bc_sum)
    v_dev = float(b_sum / float(e_raw.sum()))
    c_eff = float(bc_sum / b_sum)
    sharp = c_eff / p_norm2
    closure = conc * v_dev * sharp
    assert abs(closure - o1_corpus / iso_mean) < 1e-9 * (o1_corpus / iso_mean), \
        f"L{layer}: разложение O1/iso не замкнулось: {closure} vs {o1_corpus / iso_mean}"
    rho_p_dev = _spearman((p_row / n_pos).numpy(), dev_row.numpy())
    # --- K = (I−P)J: кросс-член, угол между J и PJ, изотропный референс самого (I−P)
    #     ‖K‖² = ‖J‖² − 2⟨J,PJ⟩ + ‖PJ‖² (тождество); tr P = 1 − Σp², значит
    #     ‖I−P‖²_F = K − 2(1 − Σp²) + ‖P‖²_F — у метрики E-37 есть свой изотропный референс.
    ip_norm2 = K - 2.0 * (1.0 - p2) + (p2 - 2.0 * p3 + p2.pow(2))    # per position
    frac_pos = k_raw / e_raw                                          # ‖K‖²/‖J‖² (энергия)
    ip_ref_pos = ip_norm2 / K
    ident_err = float((k_raw - (e_raw - 2.0 * cross + e_P)).abs().max() / e_raw.max())
    assert ident_err < 1e-12, f"L{layer}: тождество ‖K‖² не замкнулось, rel {ident_err:.2e}"
    assert float(cross.min()) >= -1e-9 * float(e_raw.mean()), (
        f"L{layer}: ⟨J,PJ⟩ < 0 — P не PSD? min = {float(cross.min()):.3e}")
    ones_mask = torch.ones(n_pos, dtype=torch.bool)

    def _kstats(mask):
        er, xp, ep_ = float(e_raw[mask].sum()), float(cross[mask].sum()), float(e_P[mask].sum())
        fr, ir = frac_pos[mask], ip_ref_pos[mask]
        return {"c_cross": xp / er,                        # ⟨J,PJ⟩/‖J‖²
                "cos_J_PJ": xp / float(np.sqrt(er * ep_)),  # угол между J и PJ
                # E6: агрегатные s и frac на том же знаменателе Σ‖J‖², что и c_cross/cos —
                # тройка (c, s, cos, frac_agg) самосогласованна; медиана/среднее позиций —
                # другие агрегаты, тождество frac=1−2c+s² для них не обязательно.
                "s_agg": float(np.sqrt(ep_ / er)),          # ‖PJ‖/‖J‖ (агрегат)
                "frac_K_energy_agg": (er - 2.0 * xp + ep_) / er,  # = 1 − 2c + s² точно
                "frac_K_energy_median": float(fr.median()),
                "frac_K_energy_mean": float(fr.mean()),
                "ip_ref_energy_median": float(ir.median()),
                "ip_ref_energy_mean": float(ir.mean()),
                "frac_over_ip_ref": float(fr.median() / ir.median()),
                "n_positions": int(mask.sum())}
    kstats = {"all_positions": _kstats(ones_mask), "alive_non_sentinel": _kstats(alive)}
    kstats["e37_frac_reference"] = E37_FRAC_K_ENERGY.get(layer)
    if layer in E37_FRAC_K_ENERGY:
        for kk, vv in kstats.items():
            if isinstance(vv, dict):
                vv["minus_e37"] = vv["frac_K_energy_median"] - E37_FRAC_K_ENERGY[layer]
    # ---sentinel---
    sent_frac = float(sent.float().mean())
    # два null-контроля на подвыборке позиций (шаг 32), со своими же p:
    #  iid        — J ~ N(0,1), учебниковый изотропный null (O1/iso обязан выйти ~1)
    #  direction  — направления случайны, нормы строк как у реального J: разделяет
    #               «структуру направлений» и «структуру норм строк» как источник O1 < iso
    idx = torch.arange(0, n_pos, NULL_STRIDE)
    rr_all, cc_all = (idx // s_len).tolist(), (idx % s_len).tolist()
    n_null = idx.numel()
    doc_of_null = np.array(rr_all)                # 128 позиций на документ
    dom_of_null = doc_of_null // N_PER_DOMAIN
    gen = torch.Generator(device="cpu").manual_seed(NULL_SEED)
    nulls = {}
    # per-position вклады по всем трём вариантам: реальный J, iid, direction-matched
    pos = {"real": (e_P[idx].numpy(), e_raw[idx].numpy(), iso[idx].numpy())}
    for kind in ("iid", "direction"):
        a_l, e_l, iso_l = [], [], []
        s_a = s_b = s_e = s_bc = 0.0
        for c0 in range(0, n_null, NULL_CHUNK):
            # чанк собирается по индексам, а не стекуется целиком: стек [n_pos, K, D] в fp64
            # это 4.3 ГБ, а 32 позиции — 34 МБ
            rs, cs = rr_all[c0:c0 + NULL_CHUNK], cc_all[c0:c0 + NULL_CHUNK]
            Pu = torch.stack([P[r, c] for r, c in zip(rs, cs)]).double()
            Ju = torch.stack([J[r, c] for r, c in zip(rs, cs)]).double()
            g = torch.randn(Ju.shape, generator=gen, dtype=torch.float64)
            if kind == "direction":      # сохраняем нормы строк реального J
                g = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-300) \
                    * Ju.norm(dim=-1, keepdim=True)
            mu = torch.einsum("sk,skd->sd", Pu, g)
            dv = (g - mu.unsqueeze(1)).pow(2).sum(-1)
            s2u = Pu.pow(2).sum(-1)
            iso_u = (s2u - 2 * Pu.pow(3).sum(-1) + s2u.pow(2)) / K
            a = Pu.pow(2) * dv                                # [chunk, K] = pᵢ²dᵢ
            a_l.append(a.sum(1)); e_l.append(g.pow(2).sum((1, 2))); iso_l.append(iso_u)
            s_a += float(a.sum()); s_b += float(dv.sum()); s_e += float(g.pow(2).sum())
            s_bc += float(dv.sum(1).mul(s2u).sum())
            del Pu, Ju, g
        a_n, e_n, iso_n = torch.cat(a_l), torch.cat(e_l), torch.cat(iso_l)
        pos[kind] = (a_n.numpy(), e_n.numpy(), iso_n.numpy())
        iso_u_mean = float(iso_n.mean())
        cn = K * s_a / s_bc
        vn = s_b / s_e
        sn = (s_bc / s_b) / (iso_u_mean * K)
        nulls[kind] = {"n_positions": int(n_null), "conc": cn, "V": vn,
                       "sharpness": sn, "O1_rel": cn * vn * sn,
                       "O1_rel_check": _o1_rel(*pos[kind])}
        assert abs(nulls[kind]["O1_rel"] - nulls[kind]["O1_rel_check"]) \
            <= 1e-9 * abs(nulls[kind]["O1_rel"]), f"L{layer}: {kind}: формы O1_rel разъехались"
    # реальный J на той же подвыборке — иначе отношение real/null сравнивает разные наборы
    a_r, e_r, iso_r = pos["real"]
    o1r = {k: _o1_rel(*pos[k]) for k in pos}
    by_doc = {k: np.array([_o1_rel(a[doc_of_null == d], e[doc_of_null == d],
                                   iso[doc_of_null == d]) for d in range(n_docs)])
              for k, (a, e, iso) in ((k, pos[k]) for k in pos)}
    by_dom = {k: {DOMAINS[j]: _o1_rel(a[dom_of_null == j], e[dom_of_null == j],
                                      iso[dom_of_null == j])
                  for j in range(len(DOMAINS))}
              for k, (a, e, iso) in ((k, pos[k]) for k in pos)}
    ratio_doc = by_doc["real"] / by_doc["direction"]
    three = {"n_positions_per_kind": int(n_null),
             "O1_rel_by_kind": o1r,
             "real_over_direction": o1r["real"] / o1r["direction"],
             "real_over_iid": o1r["real"] / o1r["iid"],
             "O1_rel_by_domain": {d: {k: by_dom[k][d] for k in pos} for d in DOMAINS},
             "real_over_direction_by_domain":
                 {d: by_dom["real"][d] / by_dom["direction"][d] for d in DOMAINS},
             "real_over_direction_by_doc": [float(x) for x in ratio_doc],
             "real_over_direction_doc_median": float(np.median(ratio_doc)),
             "real_over_direction_doc_p25_p75": [float(np.percentile(ratio_doc, 25)),
                                                 float(np.percentile(ratio_doc, 75))],
             "real_over_direction_doc_boot_ci95": _boot_median_ratio(ratio_doc),
             "note": ("всё на полном корпусе позиций (тот же срез, что и у корпусных "
                      "O1/iso); по документам дана медиана и парный bootstrap по 32 "
                      "документам (по 128 позиций на документ)")}
    # эластичность: как быстро энергия отклонений строк растёт с p (а P взвешивает как p²)
    _lp = np.log(np.maximum((p_row / n_pos).numpy(), 1e-12))
    _ld = np.log(np.maximum(dev_row.numpy(), 1e-300))
    alpha_dev = float(np.polyfit(_lp, _ld, 1)[0])
    g_uni_corpus = float(uni.sum() / (K * e_raw.sum()))
    # O3 по определению paper §2.3 на агрегированной ESG G = Σ_t J_t J_tᵀ
    tr_g, ones_g = float(G_doc.trace()), float(G_doc.sum())
    o3_corpus = ones_g / (K * tr_g)
    # топ-1 прокси E-10a: λ₁ · uniformity(u₁) / tr G,  uniformity(u) = ⟨u, 1/√K⟩²
    Gs = (G_doc / 2 + G_doc.t() / 2).numpy()
    lam, U = np.linalg.eigh(Gs)
    lam1, u1 = float(lam[-1]), U[:, -1]
    unif_u1 = float(np.dot(u1, np.ones(K) / np.sqrt(K)) ** 2)
    top1_proxy = lam1 * unif_u1 / tr_g
    # (c) O1 vs (1 − g_uni)·O1_iso — «операторное» объяснение без данных
    pred = (1.0 - g_uni_corpus) * iso_mean

    doc = np.array(per_doc, dtype=np.float64)               # [32, 6]
    per_doc_g_uni = doc[:, 2] / (K * doc[:, 0])
    per_doc_o1 = doc[:, 1] / doc[:, 0]
    per_doc_o1_rel = per_doc_o1 / doc[:, 5]
    per_doc_o3 = doc[:, 4] / (K * doc[:, 3])
    survive = np.sqrt(o1_pos)
    dom_of = np.repeat(np.arange(len(DOMAINS)), N_PER_DOMAIN)

    out = {
        "layer": layer, "K": K, "n_positions": int(n_pos), "U128": u128,
        "p_normalized_max_dev": float((P.sum(-1) - 1.0).abs().max()),
        "p_checksum": float(P.double().sum()),
        # --- O1: оператор
        "O1_corpus": o1_corpus,
        "O1_pos_median": float(np.median(o1_pos)),
        "O1_iso_mean": iso_mean,
        "O1_rel": o1_corpus / iso_mean,
        "survive_frac_median": float(np.median(survive)),
        "survive_frac_doc_median_range": [float(np.sqrt(per_doc_o1).min()),
                                          float(np.sqrt(per_doc_o1).max())],
        "O1_rel_doc_median": float(np.median(per_doc_o1_rel)),
        "O1_rel_doc_min": float(per_doc_o1_rel.min()),
        "O1_rel_doc_max": float(per_doc_o1_rel.max()),
        "O1_rel_by_domain": {d: float(np.median(per_doc_o1_rel[dom_of == k]))
                             for k, d in enumerate(DOMAINS)},
        # --- точное разложение O1/iso (ассерт на замыкание, не приближение)
        "O1_rel_closure": closure,
        "O1_rel_factor_conc": conc,
        "O1_rel_factor_V": v_dev,
        "O1_rel_factor_sharpness": sharp,
        "rho_pbar_row_deviation_energy": rho_p_dev,
        "nulls": nulls,
        "sentinel": {"raw_mass_threshold": SENT_RAW_MASS, "sent_frac": sent_frac,
                     "note": ("E-37 считает метрики по alive-позициям, весь остальной ряд "
                              "(E-33r/E-39k/это дополнение) — по всем; поэтому K-блок дан "
                              "в обоих срезах")},
        "K_from_I_minus_P": kstats,
        "nulls_same_slice": three,
        "O1_rel_over_null_direction": o1_corpus / iso_mean / nulls["direction"]["O1_rel"],
        "O1_rel_over_null_iid": o1_corpus / iso_mean / nulls["iid"]["O1_rel"],
        "alpha_deviation_energy_vs_p": alpha_dev,
        "pbar_row_range": [float((p_row / n_pos).min()), float((p_row / n_pos).max())],
        "iso_cross_check_exp39k": {"inline": iso_mean, "closed_form": iso_ref},
        "O1_pred_no_data": pred,               # (1 − g_uni)·O1_iso
        "O1_over_pred": o1_corpus / pred,
        # --- O2: uniform-доля энергии сырого J
        "g_uni_corpus": g_uni_corpus,
        "g_uni_pos_median": float(np.median((uni / (K * e_raw)).numpy())),
        "g_uni_doc_median": float(np.median(per_doc_g_uni)),
        "g_uni_doc_min": float(per_doc_g_uni.min()),
        "g_uni_doc_max": float(per_doc_g_uni.max()),
        "g_uni_by_domain": {d: float(per_doc_g_uni[dom_of == k].mean())
                            for k, d in enumerate(DOMAINS)},
        # --- O3: ESG gauge_share
        "gauge_share_ESG": o3_corpus,
        "gauge_share_top1_proxy": top1_proxy,
        "top1_proxy_over_g_uni": top1_proxy / g_uni_corpus,
        "gauge_share_doc_median": float(np.median(per_doc_o3)),
        "ESG_lambda1": lam1, "ESG_uniformity_u1": unif_u1,
        "ESG_r_stable": tr_g / lam1, "ESG_PR": float(tr_g ** 2 / (lam ** 2).sum()),
        # --- контроль (b): доля uniform-режима у J^gf (должна быть нулевой)
        "g_uni_gauge_free_max": gf_uni_max,
        # --- тождество (a): O2 vs O3 на одном G
        "identity_O2_vs_O3_rel": abs(g_uni_corpus - o3_corpus) / max(o3_corpus, 1e-300),
        # --- распределения по докам (для CI в будущих сводках)
        "per_doc": {"sum_e_raw": doc[:, 0].tolist(), "sum_e_P": doc[:, 1].tolist(),
                    "sum_uni": doc[:, 2].tolist(), "tr_G": doc[:, 3].tolist(),
                    "ones_G_ones": doc[:, 4].tolist(), "iso_mean": doc[:, 5].tolist()},
        "per_doc_usage": "per-doc суммы — заготовка под парный bootstrap (сравнения слоёв и "
                         "режимов по одним и тем же документам); сами по себе цитируются "
                         "только как разброс, не как CI",
    }
    del J, P
    return out


def _selfcheck(res: dict, p_identical: bool) -> dict:
    """Сверка с E-39k / E-33r / алгеброй. Ничего не подгоняем: пишем факт."""
    chk = {"anchor_e39k": {}, "e33r_survive_in_range": {}, "no_anchor_band": {},
           "algebra": {}}
    field_of = {"o1": "O1_corpus", "iso": "O1_iso_mean", "g_uni": "g_uni_corpus"}
    for lay in LAYERS:
        r = res["layers"][str(lay)]
        if lay in ANCHOR_E39K:
            chk["anchor_e39k"][str(lay)] = {
                k: {"measured": r[fld], "ref": ANCHOR_E39K[lay][k],
                    "rel_err": abs(r[fld] - ANCHOR_E39K[lay][k]) / ANCHOR_E39K[lay][k]}
                for k, fld in field_of.items()}
        lo, hi = E33R_SURVIVE
        chk["e33r_survive_in_range"][str(lay)] = {
            "median": r["survive_frac_median"],
            "in_range": bool(lo <= r["survive_frac_median"] <= hi),
            "e33r_ref": [lo, hi],
            "e10a_top1_proxy_ref": E10A_TOP1_PROXY.get(lay),
        }
        if lay not in ANCHOR_E39K:
            # якоря для этого слоя нет — минимум: грубая граница + структура «ниже iso»
            assert 1e-5 < r["O1_corpus"] < 1e-2, f"L{lay}: O1 = {r['O1_corpus']} вне правдоподобного"
            chk["no_anchor_band"][str(lay)] = {
                "O1": r["O1_corpus"], "below_iso": bool(r["O1_corpus"] < r["O1_iso_mean"]),
                "closure_rel_err": abs(r["O1_rel_closure"] - r["O1_rel"]) / r["O1_rel"],
            }
    chk["p_is_readout_level"] = {
        "identical_across_layers": bool(p_identical),
        "note": ("если p одинаков во всех слоях, то P(p) и его изотропный референс — одна "
                 "константа по слоям: сравнивать O1_rel между слоями корректно, а сам iso "
                 "про слой ничего не говорит")}
    chk["algebra"] = {
        "O2_equals_O3_max_rel": max(res["layers"][str(l)]["identity_O2_vs_O3_rel"]
                                    for l in LAYERS),
        "O2_equals_O3_tol": 1e-4,
        "g_uni_gauge_free_max": max(res["layers"][str(l)]["g_uni_gauge_free_max"]
                                    for l in LAYERS),
        "g_uni_gauge_free_tol": GF_UNI_MAX,
        "note": ("O2 = O3, потому что обе меры = ‖Jᵀ1‖²/(K‖J‖²_F): "
                 "Σ_k(u_kᵀ1)²λ_k = 1ᵀJJᵀ1 = ‖Jᵀ1‖². У J^gf uniform-доля нулевая, "
                 "ибо P(p)·1 = p(1−Σp) = 0 при нормированном p."),
    }
    for lay in LAYERS:
        for k, v in chk["anchor_e39k"].get(str(lay), {}).items():
            if k == "iso" and not p_identical:
                continue    # iso зависит только от p; при разном p якорь не обязан совпасть
            assert v["rel_err"] <= ANCHOR_TOL, (
                f"L{lay} {k}: измерено {v['measured']:.6g}, якорь E-39k {v['ref']:.6g} "
                f"(rel {v['rel_err']:.3%} > {ANCHOR_TOL:.0%}) — протокол разъехался")
    assert chk["algebra"]["O2_equals_O3_max_rel"] <= 1e-4, "тождество O2=O3 не подтвердилось"
    chk["null_iid_control"] = {
        "O1_rel_iid_by_layer": {str(l): res["layers"][str(l)]["nulls"]["iid"]["O1_rel"]
                                for l in LAYERS},
        "tol": 0.25,
        "note": ("на изотропном J отношение O1/iso обязано быть ~1: это проверка всего "
                 "механизма разложения, а не данных")}
    for l, v in chk["null_iid_control"]["O1_rel_iid_by_layer"].items():
        assert abs(v - 1.0) <= chk["null_iid_control"]["tol"], \
            f"L{l}: изотропный null дал O1/iso = {v} — разложение или iso неверны"
    # отношения на ОДНОЙ подвыборке (real/iid/direction) + то, что подвыборка не уехала
    chk["same_subsample_ratios"] = {}
    for l in LAYERS:
        r = res["layers"][str(l)]
        t = r["nulls_same_slice"]
        rel_sub, rel_corp = t["O1_rel_by_kind"]["real"], r["O1_rel"]
        assert abs(rel_sub / rel_corp - 1.0) <= 0.05, (
            f"L{l}: real O1/iso на срезе null'ов {rel_sub:.4g} не совпадает с корпусным "
            f"{rel_corp:.4g} — срез позиций собран неверно")
        k = r["K_from_I_minus_P"]["alive_non_sentinel"]
        chk["same_subsample_ratios"][str(l)] = {
            "real_over_direction_same_subsample": t["real_over_direction"],
            "real_over_direction_old_mixed_agg": r["O1_rel_over_null_direction"],
            "iid_by_domain_max_dev": max(abs(v - 1.0) for v in
                                         (t["O1_rel_by_domain"][d]["iid"] for d in DOMAINS)),
            "cos_J_PJ_alive": k["cos_J_PJ"],
            "c_cross_alive": k["c_cross"],
            "frac_vs_ip_ref_alive": k["frac_over_ip_ref"],
            "frac_minus_e37_alive": k["minus_e37"],
            "sent_frac": r["sentinel"]["sent_frac"],
        }
        assert 0.0 < k["cos_J_PJ"] <= 1.0, f"L{l}: cos(J,PJ) = {k['cos_J_PJ']} вне (0,1]"
    assert chk["algebra"]["g_uni_gauge_free_max"] <= GF_UNI_MAX, (
        "у J^gf осталась uniform-компонента — p не нормирован или P не тот")
    return chk


def main(results_dir: str = "results", device: str = "cpu", **kw) -> str:
    t0 = time.time()
    res = {"experiment": "E-39-supplement (three objects behind «98–99%»)",
           "spec": "docs/E39+.md, ч. 2 «ТЗ на дополнение E-39»",
           "params": {"layers": list(LAYERS), "N_docs": len(DOMAINS) * N_PER_DOMAIN,
                      "S": S, "cache": "exp27b dense (read-only)",
                      "objects": {"O1": "‖P(p)J‖²/‖J‖² + iso ‖P‖²_F/K",
                                  "O2": "g_uni = ‖QJ‖²/‖J‖²",
                                  "O3": "ESG gauge_share = Σ(u_kᵀ1)²λ_k/Σλ_k по G=Σ_t J_tJ_tᵀ"}},
           "layers": {}, "deviations_from_spec": {}}
    u_ref = None
    p_sum_ref = None
    for lay in LAYERS:
        r = _layer_numbers(lay)
        if u_ref is None:
            u_ref = list(r["U128"])
        assert list(r["U128"]) == u_ref, f"L{lay}: U128 расходится со слоем {LAYERS[0]}"
        r.pop("U128")
        if p_sum_ref is None:
            p_sum_ref = r["p_checksum"]
        res["layers"][str(lay)] = r
        print(f"L{lay}: O1 {r['O1_corpus']:.4g} (iso {r['O1_iso_mean']:.4g}, "
              f"rel {r['O1_rel']:.3f}×) survive {r['survive_frac_median']:.4f} | "
              f"g_uni {r['g_uni_corpus']:.4f} = ESG {r['gauge_share_ESG']:.4f} "
              f"(top1-прокси {r['gauge_share_top1_proxy']:.4f}) | "
              f"g_uni(J^gf) {r['g_uni_gauge_free_max']:.2g}", flush=True)
        kk = r["K_from_I_minus_P"]
        print(f"    K=(I−P)J: ‖K‖²/‖J‖² alive {kk['alive_non_sentinel']['frac_K_energy_median']:.6f} "
              f"(E-37 {kk['e37_frac_reference']}) all {kk['all_positions']['frac_K_energy_median']:.6f}"
              f" | референс (I−P) {kk['alive_non_sentinel']['ip_ref_energy_median']:.6f} "
              f"| c={kk['alive_non_sentinel']['c_cross']:.3e} cos(J,PJ)={kk['alive_non_sentinel']['cos_J_PJ']:.3f}"
              f" | sent {r['sentinel']['sent_frac']:.3f} | real/dir(тот же срез) "
              f"{r['nulls_same_slice']['real_over_direction']:.3f}", flush=True)
    res["selfchecks"] = _selfcheck(
        res, p_identical=all(abs(v["p_checksum"] - p_sum_ref) <= 1e-9 * abs(p_sum_ref)
                             for v in res["layers"].values()))
    res["deviations_from_spec"]["D1"] = (
        "ТЗ §1.4 спрашивает, сводятся ли O2 и O3: сводятся не «при определённых условиях», "
        "а тождественно — обе меры равны ‖Jᵀ1‖²/(K‖J‖²_F) на одном и том же G. Значит "
        "независимых объекта два (операторная доля и uniform-доля), а не три; «98–99%» "
        "принадлежит первому и то в форме нормы (survive_frac), а не энергии.")
    res["wall_sec"] = round(time.time() - t0, 1)
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "exp39_supp_three_objects.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n→ {path} (wall {res['wall_sec']} с)", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    main(results_dir=ap.parse_args().results_dir)
