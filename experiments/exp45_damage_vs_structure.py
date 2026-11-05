#!/usr/bin/env python
"""E-45: цена правки readout — ущерб распределению токенов против разрушения листа.

Вопрос (пользователь, 2026-11-04): «насколько каждая из таких правок портит распределение
токенов — GPTQ, случайная, уплощающая (эту считали) — и вообще, что нам эта структура может
дать, пока не ясно». Одна и та же правка меряется двумя независимыми линейками:

  структура     : κ(r=16) (голономия топ-r правых подпространств J, E-28) и ‖ΔJ‖/‖J‖;
  распределение : KL(base‖edit), KL(edit‖base), TV, доля смены top-1 на двух наборах позиций
                  (A = 4096 позиций кэша exp27b; B = C4-holdout 32 дока × 128) и PPL на B.

Семейства правок (все — только `W_LM` слоя 23, то есть одна линейная правка считывания):
  base                — контроль (все ущербы обязаны быть 0);
  gptq4 / rtn8        — вся матрица [V,D], протокол E-39b (якоря PPL 34.707, κ ratio 0.9883);
  rand_K_d{d}         — iid ΔW только по 128 строкам U128, величина bisection'ом под медианный
                        дрейф J d ∈ {0.06, 0.12, 0.25} (аудит E-39b: κ таких правок не различает);
  rand_full_d{d}      — та же величина J-дрейфа, но правятся ВСЕ V строк (режет «сколько строк
                        шевелится» при фиксированном дрейфе того же J);
  flat_l{λ}           — уплощение листа минимизацией `flatness_loss` (E-39d, импорт как есть)
                        с KL-поводком `kl_leash` (LAMBDA_KL=0.1);
  flat_noleash        — то же без поводка: сколько стоит сплющить лист, если распределение не
                        беречь.

Главный вывод здесь считается, а не декларируется: несёт ли падение κ информацию об ущербе
распределению сверх ‖ΔJ‖. Если нет — κ не метрика качества, а описание того, КАК правка
поворачивает координаты (обсуждение в docs/E45_damage.md).

Две реализации ущерба, сверенные друг с другом и с `kl_leash`:
  * плотная — полный matmul [CH,D]×[D,V] (правки всей матрицы);
  * разреженная — точные замкнутые формулы через logsumexp, когда изменены только K=128 строк
    (вывод в `_sparse_terms`, self-test против плотной на 64 позициях, допуск 1e-4, и против
    самоспроверенного `kl_leash`, допуск 1e-5).

Тяжесть: только CPU, новых GPU-форвардов нет. h для B снимается один раз (readout-only ⇒ h от
правки не зависит), GPTQ-Гессиан строится на 4096 кэшевых позициях вместо 16k токенов C4
(отклонение от E-39b сознательное, проверяется якорем κ ratio).

Запуск:  cd experiments && ../.venv/bin/python exp45_damage_vs_structure.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exp28_leaf_structure as E28           # noqa: E402  (κ = document_profile)
from _e38_cache import jcache_path           # noqa: E402
from exp39b_ptq_perturbation import (_c4_docs, _quantize_gptq,   # noqa: E402
                                     _quantize_rtn)
from exp39d_l23_training import flatness_loss, kl_leash          # noqa: E402

# ---- константы -------------------------------------------------------------
MODEL_DIR = "/data/Qwen/Qwen3.5-0.8B"
L, R, S, D = 23, 16, 128, 1024
K_TOK = 128                    # |U128|
N_PPL, S_PPL = 32, 128         # set B: как E-39b (якорь PPL base 34.317338)
GPTQ_BITS, GPTQ_GROUP, GPTQ_DAMP = 4, 128, 0.01
RTN_BITS, RTN_GROUP = 8, 128
DRIFTS = (0.06, 0.12, 0.25)
SEEDNULL_DRIFT = 0.12        # дрейф для iid null'а E45_SEEDNULL (середина сетки DRIFTS)
FLAT_LAMS = (0.01, 0.1)            # сетка E-39d была (0.01, 0.1, 1.0, 10.0); верх отброшен:
                                  # в exp39d_l23_training.json λ=1 даёт PPL 248–320 (+624…+832 %)
                                  # и κ'=0.0043, ХУЖЕ чем λ=0.1 (0.0026), а λ=10 расходился.
                                  # Для вопроса «есть ли незаметное уплощение» они не информативны.
LAMBDA_KL = 0.1                # вес KL-поводка E-39d (spec §4.4)
N_STEPS, LR, MB = 60, 1e-3, 64    # E-39d учил 100 шагов; траектория κ из того же артефакта:
                                  # 0.0080 (20) → 0.0063 (40) → 0.0056 (60) → 0.0044 (100), то
                                  # есть к 60 шагам уже ×2.8 уплощения — достаточно для теста
                                  # «невидимого» разрушения, а стоит на треть меньше времени.
P_STEP = 256                   # позиций на шаг уплощения; E-39d использовал 512, здесь вдвое
                               # меньше (замер: ~6 с/шаг при 512 на 12 ядрах CPU, пять конфигов
                               # по 100 шагов = ~2.5 часа). Отклонение фиксируется и влияет
                               # только на силу уплощения, не на корректность метрик.
CH = 128                       # позиций в чанке логиков (fp32 matmul → fp64 суммы)
METRIC_STRIDE = 2              # подыборка позиций для линейки ущерба (см. комментарий в main;
                               # stride 4 сдвигал долю sentinel-позиций на 1.4 п.п. — индикатор
                               # автокоррелирован вдоль позиций, stride 2 остаётся в допуске)
SP_CH = 4 * CH                 # чанк разреженного пути (там нет [m,V])
SENTINEL_SUM = 0.5             # E-34: сырая масса Σ p_k по U128 < 0.5 → sentinel
C4_PPL_SKIP = 24576            # fallback как в E-39b, если val недоступен

ANCH = {
    "kappa_base": 0.01623332313133695,      # аналитический J на кэше (аудит E-39b)
    "gptq_kappa_ratio": 0.9882754466931893,
    "ppl_base": 34.317338, "ppl_gptq4": 34.707,
}
SQD = math.sqrt(D)
TOK_COL = None                 # [V] → номер столбца U128 либо -1; ставится в main()


def log(*a):
    print(*a, flush=True)


# ---- структурная линейка: J и κ --------------------------------------------

def make_j_fn(Hst: np.ndarray, g: np.ndarray):
    """J_of(WU) = √D/‖h‖·(M − (M u)uᵀ), M = W_U ⊙ g. fp64, векторно по всем N позициям."""
    n = np.linalg.norm(Hst, axis=1, keepdims=True)
    u = Hst / n
    cs = (SQD / n)[:, 0]

    def J_of(WU_):
        M = WU_ * g[None, :]
        Mu = u @ M.T                                     # [N, K]
        return cs[:, None, None] * (M[None, :, :] - Mu[:, :, None] * u[:, None, :])
    return J_of


def drift_med(J0, J1, chunk=512):
    """Медиана/среднее ‖ΔJ_t‖/‖J_t‖ по позициям, чанками: разность (J1 − J0) целиком
    добавляет ещё 4.3 ГБ к двум полным якобианам, и на этом пике процесс умирал без
    traceback (не хватает памяти)."""
    r = []
    for c0 in range(0, len(J0), chunk):
        c1 = min(c0 + chunk, len(J0))
        d = (J1[c0:c1] - J0[c0:c1]).reshape(c1 - c0, -1)
        n0 = np.linalg.norm(J0[c0:c1].reshape(c1 - c0, -1), axis=1)
        r.append(np.linalg.norm(d, axis=1) / n0)
        del d
    d = np.concatenate(r)
    return float(np.median(d)), float(np.mean(d))


def kappa_of(J):
    """Медиана κ(r=16) по документам: стек [128,K,D] на документ, как E-28/E-39b."""
    ks = []
    for i in range(0, len(J), S):
        if i % (4 * S) == 0:
            log(f"    κ: документ {i // S + 1}/{len(J) // S}")
        Jd = torch.from_numpy(J[i:i + S].astype(np.float32))
        _, _, k2, _ = E28.document_profile(Jd, [1, R, 32], r_main=R)
        k2 = np.asarray(k2, dtype=np.float64)
        ks.append(k2[~np.isnan(k2)])
    return float(np.median(np.concatenate(ks)))


# ---- линейка ущерба распределению ------------------------------------------

def _lse_from_sparse(b_k, e_k, lse_b):
    """lse_edit = log(e^{lse_b} − Σ_k e^{b_k} + Σ_k e^{e_k}), правки только в K столбцах."""
    m = torch.cat([lse_b[:, None], e_k, b_k], 1).max(1).values
    return m + torch.log(torch.exp(lse_b - m)
                         - torch.exp(b_k - m[:, None]).sum(1)
                         + torch.exp(e_k - m[:, None]).sum(1))


def _sparse_terms(b_k, e_k, lse_b, lse_e):
    """Точные KL/TV, если изменены только столбцы K. Всё fp64, [m,·].

    Для v∉K: log p_e,v − log p_b,v = lse_b − lse_e (константа), откуда
      KL(b‖e) = −Σ_k p_b,k Δ_k + (lse_e − lse_b),
      KL(e‖b) = Σ_k p_e,k Δ_k + (lse_b − lse_e)    (это же считает kl_leash),
      Σ_{v∉K}|p_e,v − p_b,v| = (1 − a_b)·|e^{lse_b − lse_e} − 1|.
    """
    db = lse_b - lse_e
    pb = torch.exp(b_k - lse_b[:, None])
    pe = torch.exp(e_k - lse_e[:, None])
    dk = e_k - b_k
    kl_be = -(pb * dk).sum(1) - db
    kl_eb = (pe * dk).sum(1) + db
    tv = 0.5 * ((pe - pb).abs().sum(1) + (1.0 - pb.sum(1)) * (torch.exp(db) - 1.0).abs())
    return kl_be.clamp_min(0.0), kl_eb.clamp_min(0.0), tv.clamp_min(0.0)


def _sparse_flip(e_k, b_max_nonK, argK):
    """Смена top-1 при правке только K столбцов.

    argK: номер столбца текущего top-1 внутри U128, либо -1 (top-1 вне U128);
    b_max_nonK: лучший базовый логик вне U128 (при правке он не меняется).
    """
    m_e, j_e = e_k.max(1)
    new_in_K = m_e > b_max_nonK
    was_in_K = argK >= 0
    kept = torch.where(new_in_K, argK == j_e, ~was_in_K)
    return (~kept).double()


def base_pre(Hq, W_base, tok_ids, tgt=None):
    """Базовые величины по позициям Hq: логики U128, lse, лучший логик вне U128, столбец
    текущего top-1 (−1 если вне), и логик целевого токена (если задан tgt).

    Матричное умножение в fp32 (быстрое), накапливаем в fp64.
    """
    N = Hq.shape[0]
    tk = torch.as_tensor(tok_ids, dtype=torch.long)
    b_k = torch.empty(N, K_TOK, dtype=torch.float64)
    lse = torch.empty(N, dtype=torch.float64)
    mx_nonK = torch.empty(N, dtype=torch.float64)
    argK = torch.empty(N, dtype=torch.long)
    b_y = torch.empty(N, dtype=torch.float64)
    for c0 in range(0, N, CH):
        c1 = min(c0 + CH, N)
        lb = (Hq[c0:c1] @ W_base.t()).double()
        lse[c0:c1] = torch.logsumexp(lb, 1)
        b_k[c0:c1] = lb[:, tk]
        _, ik = lb.max(1)
        lbm = lb.clone()
        lbm[:, tk] = float("-inf")                       # лучший логик вне U128
        mx_nonK[c0:c1] = lbm.max(1).values
        col = TOK_COL[ik]                                # ≥ 0 если top-1 ∈ U128
        argK[c0:c1] = torch.where(col >= 0, col, torch.full_like(col, -1))
        if tgt is not None:
            b_y[c0:c1] = lb.gather(1, tgt[c0:c1, None].long())[:, 0]
        del lb, lbm, ik, col
    return b_k, lse, mx_nonK, argK, (b_y if tgt is not None else None)


def dense_metrics(Hq, W_edit, W_base, tgt=None):
    """Ущерб полным matmul'ом по всем позициям Hq (правки всей матрицы)."""
    acc = {k: [] for k in ("kl_be", "kl_eb", "tv", "flip", "nll")}
    N = Hq.shape[0]
    # Матричное умножение и элементы — в fp32 (быстро, без fp64-временных на [CH,V]);
    # сводки по словарю
    # accumulate'ятся в fp64 через dtype=, иначе на 248k слагаемых накопленная ошибка
    # сопоставима с малыми KL.
    for c0 in range(0, N, CH):
        c1 = min(c0 + CH, N)
        lb = Hq[c0:c1] @ W_base.t()
        le = Hq[c0:c1] @ W_edit.t()
        lb_e, le_e = torch.logsumexp(lb, 1), torch.logsumexp(le, 1)
        lpb, lpe = lb - lb_e[:, None], le - le_e[:, None]
        pb, pe = torch.exp(lpb), torch.exp(lpe)
        acc["kl_be"].append((pb * (lpb - lpe)).sum(1, dtype=torch.float64).clamp_min(0.0))
        acc["kl_eb"].append((pe * (lpe - lpb)).sum(1, dtype=torch.float64).clamp_min(0.0))
        acc["tv"].append((0.5 * (pb - pe).abs()).sum(1, dtype=torch.float64))
        acc["flip"].append((lb.argmax(1) != le.argmax(1)).double())
        if tgt is not None:
            acc["nll"].append((le_e - le.gather(1, tgt[c0:c1, None].long())[:, 0]).double())
        del lb, le, pb, pe, lpb, lpe
    return {k: torch.cat(v) for k, v in acc.items() if v}


def sparse_metrics(Hq, tok_ids, W_rows_new, pre, tgt=None):
    """Ущерб для правки только строк tok_ids — точные формулы, без [m,V] матричных произведений."""
    b_k, lse_b, mx_nonK, argK, b_y = pre
    acc = {k: [] for k in ("kl_be", "kl_eb", "tv", "flip", "nll")}
    Wr = W_rows_new.double()
    N = Hq.shape[0]
    for c0 in range(0, N, SP_CH):
        c1 = min(c0 + SP_CH, N)
        bk = b_k[c0:c1]
        e_k = Hq[c0:c1].double() @ Wr.t()
        lse_e = _lse_from_sparse(bk, e_k, lse_b[c0:c1])
        kl_be, kl_eb, tv = _sparse_terms(bk, e_k, lse_b[c0:c1], lse_e)
        acc["kl_be"].append(kl_be)
        acc["kl_eb"].append(kl_eb)
        acc["tv"].append(tv)
        acc["flip"].append(_sparse_flip(e_k, mx_nonK[c0:c1], argK[c0:c1]))
        if tgt is not None:
            # целевой логик меняется только если токен ∈ U128
            col = TOK_COL[tgt[c0:c1]].long()
            dk_y = (e_k - bk).gather(1, col.clamp_min(0)[:, None])[:, 0]
            dk_y = torch.where(col >= 0, dk_y, torch.zeros_like(dk_y))
            acc["nll"].append(lse_e - (b_y[c0:c1] + dk_y))
        del e_k, lse_e
    return {k: torch.cat(v) for k, v in acc.items() if v}


SPEC_DOCS = tuple(range(0, 32, 4))     # 8 документов на спектр: fp64-диагонализация дорогая,
                                      # медиане и доле позиций это не вредит


FLAT_CACHE = "../results/exp45_flat_rows.pt"
PARTIAL = "../results/exp45_damage_vs_structure.partial.json"
RESUME = "--resume" in sys.argv


def _dump_partial(out):
    """После каждой правки — частичный артефакт (атомарно): часовой прогон не должен сгорать
    целиком из-за одного OOM в конце. --resume доучивает/домеряет только недостающее."""
    tmp = PARTIAL + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    os.replace(tmp, PARTIAL)


def spectral_flags(J1, docs=SPEC_DOCS):
    """Медианные спектральные характеристики J после правки (по 8 документам).

    Смысл: κ и V_r определены только если на разрезе r есть зазор. Правка может «уронить κ» не
    потому, что лист стал другим, а потому что σ_16 и σ_17 слиплись и V_16 превратилась в шум
    (E-39d раунд 9 предупреждал про вырожденную M_K, которая играмит сурогат плоскости). Поэтому
    вместе с κ обязана идти пара: gap16 = (σ_16 − σ_17)/σ_1 и доля позиций с gap16 < 0.01.

    Считается в fp64: у fp32 eigvalsh абсолютный пол ~eps·‖J‖², из-за чего на J с σ_16/σ_1 = 1e-6
    он возвращает σ_16 ≈ 3e-4 (замерено на синтетике, /tmp/tspec.py). Порог 0.01 при этом на
    ~4 порядка выше собственного шума, так что «зазора нет» не может быть артефактом точности.
    """
    gap, pr, rat = [], [], []
    for di in docs:
        Jd = torch.from_numpy(J1[di * S:(di + 1) * S])
        s2 = torch.linalg.eigvalsh(Jd @ Jd.transpose(1, 2)).clamp_min(0.0).flip(1)
        s = s2.sqrt()
        gap.append(((s[:, R - 1] - s[:, R]) / s[:, 0].clamp_min(1e-30)).numpy())
        pr.append(((s2.sum(1) ** 2) / s2.pow(2).sum(1).clamp_min(1e-30)).numpy())
        rat.append((s[:, R - 1] / s[:, 0].clamp_min(1e-30)).numpy())
        del Jd, s2, s
    gap, pr, rat = (np.concatenate(v) for v in (gap, pr, rat))
    return dict(spec_gap16_med=float(np.median(gap)),
                spec_frac_gap_lt_1pct=float((gap < 0.01).mean()),
                spec_pr_med=float(np.median(pr)),
                spec_s16_over_s1_med=float(np.median(rat)))


def agg(m, mask):
    """Среднее и медиана по замаскированным позициям."""
    out = {}
    for k, v in m.items():
        if v.numel() == 0:
            continue
        vv = v[mask]
        out[k] = float(vv.mean())
        out[k + "_median"] = float(vv.median())
    return out


def selftest_formulas():
    """Проверка замкнутых формул `_sparse_terms`/`_lse_from_sparse`/`_sparse_flip` против
    грубого softmax на случайных логиках (это же гоняется перед полным прогоном; правая
    половина скрипта целиком на них держится, а знак в KL(b‖e) уже был испорчен здесь).

    Запуск:  ../.venv/bin/python exp45_damage_vs_structure.py --selftest
    """
    global TOK_COL
    torch.manual_seed(0)
    m, V, K = 40, 700, K_TOK
    lb = torch.randn(m, V, dtype=torch.float64) * 3
    tk = torch.randperm(V)[:K]
    TOK_COL = torch.full((V,), -1, dtype=torch.long)
    TOK_COL[tk] = torch.arange(K)
    b_k, lse_b = lb[:, tk], torch.logsumexp(lb, 1)
    _, ik = lb.max(1)
    lbm = lb.clone()
    lbm[:, tk] = float("-inf")
    mxn = lbm.max(1).values
    col = TOK_COL[ik]
    argK = torch.where(col >= 0, col, torch.full_like(col, -1))
    for scale, tag in ((2.0, "слабая"), (15.0, "сильная")):
        eb = lb.clone()
        eb[:, tk] += torch.randn(m, K, dtype=torch.float64) * scale
        e_k = eb[:, tk]
        lse_e = _lse_from_sparse(b_k, e_k, lse_b)
        assert torch.allclose(lse_e, torch.logsumexp(eb, 1), atol=1e-10), f"lse, {tag}"
        kl_be, kl_eb, tv = _sparse_terms(b_k, e_k, lse_b, lse_e)
        pb, pe = torch.softmax(lb, 1), torch.softmax(eb, 1)
        refs = (("kl_be", kl_be, (pb * (pb / pe).log()).sum(1)),
                ("kl_eb", kl_eb, (pe * (pe / pb).log()).sum(1)),
                ("tv", tv, 0.5 * (pb - pe).abs().sum(1)))
        for nm, a, r in refs:
            d = float((a - r).abs().max())
            log(f"  [selftest {tag}] {nm}: max|Δ| = {d:.2e}")
            assert d < 1e-9, f"{tag} {nm}: {d:.2e}"
        fl = _sparse_flip(e_k, mxn, argK)
        d = float((fl - (eb.argmax(1) != ik).double()).abs().max())
        log(f"  [selftest {tag}] flip: max|Δ| = {d:.2e}")
        assert d == 0, f"{tag} flip: {d}"
    TOK_COL = None
    log("selftest формул ущерба: OK")


def main():
    global TOK_COL
    t0 = time.time()
    torch.set_num_threads(len(os.sched_getaffinity(0)))   # не os.cpu_count(): на коробке 64 при 12 ядрах
    out = {"experiment": "E-45", "L": L, "r": R, "anchors": ANCH, "edits": [],
           "selfcheck": {}}

    # ---------------- кэш L23 = set A ----------------
    log("loading jcache L23 ...")
    ck = torch.load(jcache_path(L), map_location="cpu", weights_only=False)
    U128 = [int(t) for t in (ck["U128"].numpy() if hasattr(ck["U128"], "numpy")
                             else ck["U128"])]
    DOMS, NS = list(ck["domains"]), int(ck["n_samples"])
    Hst = np.stack([ck["H"][(d, s, t)].double().numpy()
                    for d in DOMS for s in range(NS) for t in range(S)])
    Ha = torch.from_numpy(Hst).float()
    log(f"set A: {Hst.shape} ({len(DOMS)} доменов × {NS} документов × {S})")
    del ck

    # ---------------- модель ----------------
    log("loading model (fp32) ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    torch.set_grad_enabled(False)
    W = model.lm_head.weight.detach().float().clone()            # [V, D]
    g_np = (1.0 + model.model.norm.weight.detach().float()).numpy().astype(np.float64)
    g_t = torch.from_numpy(g_np).float()
    V = W.shape[0]
    tok_ids = torch.tensor(U128, dtype=torch.long)
    TOK_COL = torch.full((V,), -1, dtype=torch.long)
    TOK_COL[tok_ids] = torch.arange(K_TOK)
    WU0 = W[U128].numpy().astype(np.float64)
    # Геометрия (κ, спектр, дрейф) считается по ПОДЫБОРУ позиций, а не по всем 4096: полный
    # якобиан — это 4.3 ГБ fp64 на каждую правку, а медиана по 1024 позициям (8 документов) от
    # медианы по 4096 не уходит (проверка — assert ниже: база на подыборке обязана сойтись с
    # якорем E-39d/E-28, посчитанным по всему корпусу). Считается на ОДНИХ и тех же документах
    # для базы и для всех правок, поэтому относительные числа корректны в любом случае.
    SPEC_POS = np.concatenate([np.arange(i * S, (i + 1) * S) for i in SPEC_DOCS])
    J_of_spec = make_j_fn(Hst[SPEC_POS], g_np)

    # ---------------- set B: C4-holdout, h снимается один раз ----------------
    log("capturing h for set B (C4 val; readout-only ⇒ h от правки не зависит) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    try:
        ids_B, attn_B = _c4_docs(N_PPL, S_PPL, tok, skip=0, split="val")
    except Exception as e:
        log(f"  val недоступен ({e}), fallback train skip={C4_PPL_SKIP}")
        ids_B, attn_B = _c4_docs(N_PPL, S_PPL, tok, skip=C4_PPL_SKIP)
    parts = []
    with torch.no_grad():
        for c0 in range(0, N_PPL, 8):
            c1 = min(c0 + 8, N_PPL)
            parts.append(model.model(
                input_ids=ids_B[c0:c1], attention_mask=attn_B[c0:c1]
            ).last_hidden_state.float())
    del model
    HB = torch.cat(parts, 0)[:, :-1, :].reshape(-1, D)           # позиции, предсказывающие t+1
    mask_B = attn_B[:, 1:].reshape(-1) > 0
    tgt_B = ids_B[:, 1:].reshape(-1)
    HB[~mask_B] = 0.0                                            # padding не участвует
    log(f"set B: {tuple(HB.shape)}, реальных токенов {int(mask_B.sum())}")

    # Ущерб распределения — по детерминированному подыбору позиций (каждый METRIC_STRIDE-й):
    # полный [m,V] matmul на 4096 + 4064 позициях стоит ~5 минут на правку, а медианы KL/TV и
    # доля смены top-1 на 1024 позициях те же; PPL_B становится оценкой по подвыборке, ОДИНАКОВОЙ
    # для всех правок, поэтому разности и отношения остаются корректными.
    Ha_s = Ha[::METRIC_STRIDE]
    HB_s = HB[::METRIC_STRIDE]
    tgt_B_s, mask_B_s = tgt_B[::METRIC_STRIDE], mask_B[::METRIC_STRIDE]

    # ---------------- базовые предрасчёты ----------------
    log("base logits over A ...")
    preA = base_pre(Ha_s, W, U128)
    preB = base_pre(HB_s, W, U128, tgt=tgt_B_s)
    raw_mass = torch.exp(preA[0] - preA[1][:, None]).sum(1)
    keep = raw_mass >= SENTINEL_SUM
    log(f"  sentinel A (масса < {SENTINEL_SUM}): {int((~keep).sum())}/{len(keep)} "
        f"= {100 * float((~keep).float().mean()):.1f} %")
    out["selfcheck"]["sent_frac_A"] = float((~keep).float().mean())
    assert abs(float((~keep).float().mean()) - 0.50708) < 0.01, "sentinel_frac ушёл от 50.7 %"

    # самопроверка 1: KL(e‖b) обязан совпасть с самоспроверенным kl_leash
    sub = torch.where(keep)[0][:2048]
    Wr_test = W[U128] * 1.02
    e_test = Ha_s[sub].double() @ Wr_test.double().t()
    lse_test = _lse_from_sparse(preA[0][sub], e_test, preA[1][sub])
    kl_ours = _sparse_terms(preA[0][sub], e_test, preA[1][sub], lse_test)[1].mean()
    kl_ref = kl_leash(Ha_s[sub], Wr_test, g_t, preA[0][sub].float(), preA[1][sub].float())
    absdiff = abs(float(kl_ours) - float(kl_ref))
    rel = absdiff / max(float(kl_ref), 1e-12)
    log(f"  cross-check против kl_leash: ours={float(kl_ours):.8f} "
        f"ref={float(kl_ref):.8f} abs={absdiff:.2e} rel={rel:.2e}")
    # kl_leash считается в fp32, наш разреженный путь — в fp64, поэтому допуск 1e-4 с запасом:
    # на согласованной выборке наблюдение было 2.3e-06. Страж fp64-против-fp64 — самопроверка 2
    # ниже (sparse vs dense, max|Δ| ~2e-06). Историческая заметка: этот assert падал с
    # rel = 1.04e-05 (abs = 7.7e-05), когда самопроверка брала индексы из подыборки, а h — из
    # полного массива Ha: сравнивались РАЗНЫЕ позиции, то есть ловила она себя, а не эталон.
    assert rel < 1e-4, f"KL(e‖b) расходится с kl_leash: abs={absdiff:.2e} rel={rel:.2e}"
    out["selfcheck"]["kl_vs_kl_leash_rel"] = rel
    out["selfcheck"]["kl_vs_kl_leash_abs"] = absdiff

    # самопроверка 2: разреженный путь против плотного (KL/TV/flip) на 64 позициях
    sub64 = torch.where(keep)[0][:64]
    Wsp = W.clone()
    Wsp[U128] = Wr_test
    p64 = (preA[0][sub64], preA[1][sub64], preA[2][sub64], preA[3][sub64], None)
    dm = dense_metrics(Ha_s[sub64], Wsp, W)
    sm = sparse_metrics(Ha_s[sub64], tok_ids, Wr_test, p64)
    for k in ("kl_be", "kl_eb", "tv"):
        d = float((dm[k] - sm[k]).abs().max())
        log(f"  sparse-vs-dense {k}: max|Δ| = {d:.2e}")
        assert d < 1e-4, f"разреженный путь расходится с плотным по {k}: {d:.2e}"
        out["selfcheck"][f"sparse_vs_dense_{k}"] = d
    # top-1 — единственная целочисленная метрика: два маршрута могут разойтись только там, где
    # зазор между двумя лучшими логиками ничтожен, поэтому считаем расхождением >1 позиции.
    nflip_diff = int((dm["flip"] != sm["flip"]).sum())
    log(f"  sparse-vs-dense flip: расхождений {nflip_diff} из {dm['flip'].numel()}")
    assert nflip_diff <= 1, f"разреженный top-1 расходится с плотным: {nflip_diff}"
    out["selfcheck"]["sparse_vs_dense_flip_diffs"] = nflip_diff
    del Wsp, dm, sm

    # ---------------- семейства правок ----------------
    # base — КОПИЯ весов: иначе KL ≡ 0 тривиально (тот же объект), а нужно, чтобы «нулевой
    # ущерб» был получен двумя независимыми проходами по одним и тем же числам.
    edits = [dict(name="base", kind="dense", W=W.clone())]
    log("quantizing GPTQ 4-bit (Гессиан = 4096 позиций кэша) ...")
    edits.append(dict(name="gptq4", kind="dense",
                      W=_quantize_gptq(W.double(), Ha.double(), GPTQ_BITS,
                                       GPTQ_GROUP, GPTQ_DAMP,
                                       torch.device("cpu")).float()))
    log("quantizing int8 RTN ...")
    edits.append(dict(name="rtn8", kind="dense",
                      W=_quantize_rtn(W.double(), RTN_BITS, RTN_GROUP).float()))
    log("  RTN готов, строю J базы …")
    J0 = J_of_spec(WU0)
    log("  J базы готов, считаю κ базы …")
    KAPPA_SUB = kappa_of(J0)        # база на той же подыборке документов, что и все правки
    log(f"  база κ(r={R}) на подыборке {len(SPEC_DOCS)} доков = {KAPPA_SUB:.6f} "
        f"(якорь по корпусу {ANCH['kappa_base']:.6f})")
    assert abs(KAPPA_SUB / ANCH["kappa_base"] - 1) < 0.25, (
        f"база на подыборке ({KAPPA_SUB:.6f}) не сходится с корпусным якорем "
        f"({ANCH['kappa_base']:.6f}): подыборка документов непредставительна")
    G_U = np.random.default_rng(0).standard_normal((K_TOK, D)) * float(WU0.std())
    G_all = np.random.default_rng(11).standard_normal((V, D)) * float(W.std())
    G_U_full = G_all[U128].astype(np.float64)
    # Величина iid-возмущения подбирается ОДНОЙ оценкой, а не бисекцией: ΔJ линеен по σ, значит
    # постреляционный дрейф ‖ΔJ_t‖/‖J_t‖ линеен по σ точно, и медиана — тоже. Бисекция из шести
    # итераций стоила ~12 минут и ничего не давала сверх этого.
    lin_K = drift_med(J0, J_of_spec(WU0 + G_U))[0]
    lin_F = drift_med(J0, J_of_spec(WU0 + G_U_full))[0]
    log(f"  линейность дрейфа по σ: drift_K(σ=1) = {lin_K:.4f}, drift_full(σ=1) = {lin_F:.4f}")
    for d_target in DRIFTS:
        s_, s2 = d_target / lin_K, d_target / lin_F
        edits.append(dict(name=f"rand_K_d{d_target}", kind="sparse",
                          rows=W[U128] + s_ * torch.from_numpy(G_U).float(), sigma=s_))
        edits.append(dict(name=f"rand_full_d{d_target}", kind="dense",
                          W=W + s2 * torch.from_numpy(G_all).float(), sigma=s2))
        log(f"  drift {d_target}: σ_K = {s_:.4f}·σ_U, σ_full = {s2:.5f}·σ_W "
            f"(фактический дрейф печатается в строке правки)")
    del G_all, G_U_full                       # 2 ГБ fp64 больше не нужны, впереди J-фаза

    # iid null того же семейства, НЕСКОЛЬКО draw'ов. Строки rand_K/rand_full — по ОДНОЙ выборке
    # на семейство, поэтому их взаимные различия (знак Δκ, величина gap16) могут быть разбросом
    # выборки: якобиан L23 есть функция только 128 строк U128, так что по закону J у rand_K и
    # rand_full одинаковый. Считается только при E45_SEEDNULL>0 (отдельный --resume-проход),
    # иначе гонять восемь лишних строк бессмысленно.
    n_sn = int(os.environ.get("E45_SEEDNULL", "0"))
    for k in range(n_sn):
        Gk = np.random.default_rng(100 + k).standard_normal((K_TOK, D)) * float(WU0.std())
        # ОЦЕНКА σ: drift_med берётся от ПРАВКИ WU0+Gk, как в lin_K выше. Первая версия звала
        # drift_med(J0, J_of_spec(Gk)), то есть меряла расстояние между J шума и J базы вместо
        # нормы J самого шума: числитель завышен в √2, σ занижен в те же 0.71, и фактический
        # дрейф draw'ей выходил 0.0855 вместо заказанных 0.12 (полоса разброса при этом была
        # бы не на том дрейфе, что строки rand_*).
        sk = SEEDNULL_DRIFT / drift_med(J0, J_of_spec(WU0 + Gk))[0]   # дрейф линеен по σ
        edits.append(dict(name=f"seednull_s{k}", kind="sparse",
                          rows=W[U128] + sk * torch.from_numpy(Gk).float(), sigma=sk))
    if n_sn:
        log(f"  iid null: {n_sn} draw'ов семейства U128 при дрейфе {SEEDNULL_DRIFT}")

    # уплощение листа (только 128 строк U128), гиперпараметры E-39d
    l_flat_base = None
    kidx = torch.where(keep)[0]
    for lam in list(FLAT_LAMS) + [None]:
        lam_kl = LAMBDA_KL if lam is not None else 0.0
        name = f"flat_l{lam}" if lam is not None else "flat_noleash"
        if RESUME and os.path.exists(FLAT_CACHE):
            cached = torch.load(FLAT_CACHE, map_location="cpu", weights_only=False)
            if name in cached:
                edits.append(dict(name=name, kind="sparse", rows=cached[name],
                                  lam_kl=lam_kl, l_flat_base=None))
                log(f"  {name}: взято из кэша, обучение пропущено")
                continue
        torch.manual_seed(0)
        np.random.seed(0)
        Wr = W[U128].clone().requires_grad_(True)
        opt = torch.optim.Adam([Wr], lr=LR)
        torch.set_grad_enabled(True)
        for step in range(1, N_STEPS + 1):
            opt.zero_grad()
            idx = kidx[torch.randint(0, kidx.numel(), (P_STEP,))]
            H_sub = Ha_s[idx]          # kidx — индексы подыборки, не полного A
            lf = flatness_loss(Wr * g_t[None, :], H_sub, R, MB)
            if l_flat_base is None:
                l_flat_base = float(lf.detach())
            l_kl = kl_leash(H_sub, Wr, g_t, preA[0][idx].float(), preA[1][idx].float())
            loss = (1.0 if lam is None else lam) * (lf / l_flat_base) + lam_kl * l_kl
            assert torch.isfinite(loss), f"{name}: non-finite loss на шаге {step}"
            loss.backward()
            torch.nn.utils.clip_grad_norm_([Wr], 1.0)
            opt.step()
            assert torch.isfinite(Wr.detach()).all(), f"{name}: params non-finite {step}"
        torch.set_grad_enabled(False)
        edits.append(dict(name=name, kind="sparse", rows=Wr.detach().clone(),
                          lam_kl=lam_kl, l_flat_base=l_flat_base))
        log(f"  {name}: готово (l_flat_base={l_flat_base:.4f}), t={time.time() - t0:.0f} с")
        cache = (torch.load(FLAT_CACHE, map_location="cpu", weights_only=False)
                 if os.path.exists(FLAT_CACHE) else {})
        cache[name] = Wr.detach().clone()
        torch.save(cache, FLAT_CACHE)         # перезапуск не переучивает готовые конфиги

    # ---------------- прогон: обе линейки на каждой правке ----------------
    # ---------------- прогон: обе линейки на каждой правке ----------------
    DONE = {}
    if RESUME and os.path.exists(PARTIAL):
        with open(PARTIAL) as f:
            DONE = {r["name"]: r for r in json.load(f).get("edits", [])}
        log(f"resume: {len(DONE)} готовых строк из partial")
    for e in edits:
        name, t1 = e["name"], time.time()
        if name in DONE:
            out["edits"].append(DONE[name])
            log(f"{name:>16s}: готовая строка из partial (пропуск)")
            continue
        WU_new = (e["W"][U128].numpy().astype(np.float64) if e["kind"] == "dense"
                  else e["rows"].numpy().astype(np.float64))
        J1 = J_of_spec(WU_new)
        dmed, dmean = drift_med(J0, J1)
        log(f"  [{name}] дрейф {dmed:.4f} готов, считаю κ …")
        kap_A = kappa_of(J1)
        log(f"  [{name}] κ = {kap_A:.6f}, спектр …")
        spec = spectral_flags(J1, docs=range(len(SPEC_DOCS)))
        del J1
        log(f"  [{name}] спектр готов ({time.time() - t1:.0f} с), метрики A и B …")
        if e["kind"] == "dense":
            mA = dense_metrics(Ha_s, e["W"], W)
            mB = dense_metrics(HB_s, e["W"], W, tgt=tgt_B_s)
        else:
            mA = sparse_metrics(Ha_s, tok_ids, e["rows"], preA)
            mB = sparse_metrics(HB_s, tok_ids, e["rows"], preB, tgt=tgt_B_s)
        aA, aA_all = agg(mA, keep), agg(mA, torch.ones_like(keep))
        log(f"  [{name}] метрики готовы ({time.time() - t1:.0f} с), агрегаты …")
        aB = agg(mB, mask_B_s)
        ppl_B = float(torch.exp(mB["nll"][mask_B_s].mean()))
        rec = dict(name=name, kind=e["kind"], drift_med=dmed, drift_mean=dmean,
                   kappa_A=kap_A, ratio_A=kap_A / KAPPA_SUB,
                   ratio_A_vs_corpus=kap_A / ANCH["kappa_base"],
                   kl_be_A=aA_all["kl_be"], kl_be_A_alive=aA["kl_be"],
                   kl_be_A_median=aA_all["kl_be_median"], kl_eb_A=aA_all["kl_eb"],
                   tv_A=aA_all["tv"], flip_A=aA_all["flip"],
                   kl_be_B=aB["kl_be"], kl_eb_B=aB["kl_eb"], tv_B=aB["tv"],
                   flip_B=aB["flip"], ppl_B=ppl_B, **spec)
        for extra in ("sigma", "lam_kl"):
            if extra in e:
                rec[extra] = e[extra]
        out["edits"].append(rec)
        log(f"{name:>16s}: drift {dmed:.4f}  κ {kap_A:.6f} (×{rec['ratio_A']:.3f})  "
            f"KL_A {rec['kl_be_A']:.5f}  KL_B {rec['kl_be_B']:.5f}  TV_B {rec['tv_B']:.4f}  "
            f"flip_B {rec['flip_B']:.4f}  PPL_B {ppl_B:.4f}  gap16 {rec['spec_gap16_med']:.4f} "
            f"({100 * rec['spec_frac_gap_lt_1pct']:.1f} % < 1 %)   "
            f"[{time.time() - t1:.0f} с]")
        del mA, mB
        _dump_partial(out)

    # ---------------- якоря и сводка ----------------
    by = {r["name"]: r for r in out["edits"]}
    base_ppl = by["base"]["ppl_B"]
    out["gptq_hessian_note"] = ("Гессиан GPTQ = 4096 позиций кэша A (E-39b использовал 16k "
                                "токенов C4): отклонение сознательное, проверка — якорь κ ratio")
    out["kappa_base_subset"] = KAPPA_SUB
    out["metric_stride"] = METRIC_STRIDE
    out["kappa_docs"] = list(SPEC_DOCS)
    out["slice_note"] = (f"геометрия (drift, κ, спектр) — на {len(SPEC_POS)} позициях "
                         f"({len(SPEC_DOCS)} документов из 32); ущерб (KL, TV, top-1, PPL) — на "
                         f"каждом {METRIC_STRIDE}-й позиции A и B; база и все правки на одних и "
                         f"тех же позициях, ratio_A считается на подыборочной базе, "
                         f"ratio_A_vs_corpus — против корпусного якоря")
    sp_rows = {e["name"]: e["rows"].numpy() for e in edits if e["kind"] == "sparse"}
    np.savez_compressed("../results/exp45_edits_sparse.npz", **sp_rows)
    out["saved_sparse_edits"] = sorted(sp_rows)
    out["base_ppl_B"] = base_ppl
    out["ppl_delta_pct"] = {k: 100 * (r["ppl_B"] / base_ppl - 1)
                            for k, r in by.items() if k != "base"}
    # Якорные сверки: два разных утверждения, которые раньше были слиты в один список.
    # (а) самоточность конвейера — те же позиции, тот же код ⇒ совпадение до 1e-12 (см. ratio_A
    #     базы ниже); (б) сопоставимость с ЧУЖИМ срезом — E-28/E-39b считали κ по 4096 позициям,
    #     PPL по 8 документам C4, поэтому разность там есть выборка, а не ошибка: несогласованный
    #     док-бутстрап уровня κ на L23 даёт ±9.5 % при 8 документах
    #     (results/exp45_kappa_seed_null.json). Допуски обязаны быть порядка этой полосы.
    rel_self = abs(by["base"]["ratio_A"] - 1.0)
    log(f"  самоточность: ratio_A базы = {by['base']['ratio_A']:.12f} (rel {rel_self:.2e})")
    assert rel_self < 1e-12, f"база не совпадает сама с собой: {by['base']['ratio_A']}"
    out["selfcheck"]["kappa_selfcheck_rel"] = rel_self
    for k, v, tol, why in (
            ("kappa_base", by["base"]["kappa_A"], 0.25,
             "другой срез позиций: 8 документов против 4096 (полоса ±9.5 %)"),
            ("ppl_base", base_ppl, 0.15, "другой корпус C4: 32 документа против 8 у E-39b"),
            ("ppl_gptq4", by["gptq4"]["ppl_B"], 0.15, "то же, на правке GPTQ"),
            ("gptq_kappa_ratio", by["gptq4"]["ratio_A"], 5e-2,
             "тот же объект и тип отношения, допуск узкий")):
        rel = abs(v / ANCH[k] - 1)
        log(f"anchor {k}: {v:.6g} vs {ANCH[k]:.6g} (rel {rel:.2e}, допуск {tol:.0e}; {why})")
        assert rel < tol, f"якорь {k} не сошёлся: rel={rel:.2e} — {why}"
        out["selfcheck"][f"anchor_{k}_rel"] = rel
    assert by["base"]["kl_be_B"] < 1e-9, "base должен давать KL = 0 точно"
    for lam in FLAT_LAMS:
        r = by[f"flat_l{lam}"]
        assert r["kappa_A"] < by["base"]["kappa_A"], f"λ={lam}: κ не уменьшился"
        assert r["kl_be_B"] > by["base"]["kl_be_B"], f"λ={lam}: KL не вырос"
    out["wall_sec"] = time.time() - t0
    with open("../results/exp45_damage_vs_structure.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    log(f"готово за {out['wall_sec']:.0f} с → results/exp45_damage_vs_structure.json")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest_formulas()
    else:
        main()
