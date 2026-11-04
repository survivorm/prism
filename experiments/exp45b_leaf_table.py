#!/usr/bin/env python
"""E-45b: помеченная таблица «лист ↔ поведение» по позициям — общий артефакт E-46…E-50.

Зачем. Все вопросы «что даёт найденный лист» упираются в отсутствие набора, где у ОДНОЙ и той же
позиции есть (а) величины листа и (б) размеченное поведение и текст. Кэш exp27b даёт (а), но в нём
нет идентификаторов токенов: ни NLL, ни текста, ни границ, ни «копировался ли токен». Здесь —
4 домена × 8 документов × S=128 (ровно форма кэша) со снятием скрытых состояний с живой модели,
одна строка на позицию.

Артефакты:
  results/exp45b_leaf_table.json.gz — скалярные колонки по всем позициям (коммитится);
  results/exp45b_leaf_table.npz     — V_r [N,S,D,r] и h [N,S,D] fp16 (локальный кэш, в git не
                                      входит, пересоздаётся этой же командой).

Колонки позиции:
  лист        : d1_r = Δ_SVD пары (t,t+1) и k2_r = локальная голономия для r ∈ R_LIST,
                ‖J‖_F, PR(σ²), g_uni позиции, ‖h‖ (главный кандидат в конфаунды);
  поведение   : nll, энтропия, top-1, ранг цели, in_top{5,50}, copy@{1,4,16} (цель встречалась в
                предыдущих k токенах), dnll = |nll(t) − nll(t−1)|;
  лист↔токен  : align_next = ‖P_{V_r,t} ŵ(y_{t+1})‖ — насколько строка считывания СЛЕДУЮЩЕГО токена
                лежит в листе позиции t; align_emit — для только что вышедшего токена; align_rand —
                для 8 случайных строк — это Haar-нуль: E‖P_V ŵ‖² = r/D, поэтому для r=16,D=1024
                среднее ‖P_V ŵ‖ ≈ √(r/D) = 0.125 с поправкой ~1 % (норма, не квадрат);
  текст       : длина декодированного токена, перевод строки, конец предложения, цифра; смещение
                конца токена в тексте (у E-31 границы предложений считались так же).

Границы — по строкам токена, не по offset_mapping в текст: сырой текст датасета не сохраняется,
так что это прокси, и в выводах это надо называть прокси.

Дорожка токенизации: `save_offset_mapping=True` идёт по ветке `add_special_tokens=False` (см.
NOTE в prism/calibration.py: offset_mapping обязан оставаться выровнен 1:1 с текстом). Значит
последовательность `ids` в этой таблице НЕ то же самое, что mainline-калибровка E-16 на том же
тексте, — смешивать их идентификаторы нельзя (для нас важны только сами токены и их порядок).

Маршрут логиков: `H @ W_lmᵀ`, где H = `model.model(...).last_hidden_state` (уже выход финальной
RMSNorm — проверено `exp39d_head_route_probe.py`, cos = 1.000000). Здесь это сверяется с
`model(...).logits` на каждом домене и падает громко.

Запуск:  cd experiments && ../.venv/bin/python exp45b_leaf_table.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exp28_leaf_structure as E28                     # noqa: E402
from _e38_cache import jcache_path                     # noqa: E402
from prism.calibration import load_calibration_data    # noqa: E402

MODEL_DIR = "/data/Qwen/Qwen3.5-0.8B"
L, R_MAIN, S, D = 23, 16, 128, 1024
K_TOK = 128
R_LIST = [1, 4, 16, 32, 64]
DOMAINS = ["wiki", "math", "code", "chat"]
N_PER_DOMAIN = 8
CH = 512                        # чанк логиков [CH, V]
SQD = float(D) ** 0.5
KAPPA_CACHE = 0.016233          # κ(r=16) на кэше exp27b — сверка масштаба на свежем тексте
SENT_PUNCT = tuple('.?!:;"”’)')


def log(*a):
    print(*a, flush=True)


def analytic_J(M: torch.Tensor, Hdoc: torch.Tensor) -> torch.Tensor:
    """J_t = √D/‖h‖·(M − (M û)ûᵀ), M = W_U ⊙ g [K,D], Hdoc [T,D] → J [T,K,D]."""
    hn = Hdoc.norm(dim=-1).clamp_min(1e-12)
    u = Hdoc / hn[:, None]
    Mu = u @ M.T                                        # [T,K]
    return (SQD / hn)[:, None, None] * (M[None, :, :] - Mu[:, :, None] * u[:, None, :])


def leaf_scalars(J: torch.Tensor) -> dict:
    """Скаляры листа по позициям по НАСТОЯЩЕМУ спектру якобиана.

    σ² = собственные числа Грамма J Jᵀ (K×K на позицию). ВАЖНО: `(J**2).sum(2)` — это суммы
    квадратов по строкам (нормы строк J), их сумма равна ‖J‖²_F = Σσ², но поэлементно это НЕ σ²;
    первая версия функции именно так и считала PR, то есть называла участием главных
    направлений участие строк. Проверяется на случайной J: PR по строкам ≠ PR по σ.

    Возвращает: ‖J‖_F, PR(σ²) = (Σσ²)²/Σσ⁴, g_uni = ‖Jᵀ1‖²/(K‖J‖²_F),
    gap_r16 = (σ_16 − σ_17)/σ_1 — зазор на разрезе r: при малом зазоре сам разрез (а значит
    и V_r, и κ) неустойчив, и выводы «лист разрушен/укреплён» на таких позициях невалидны;
    s16_over_s1 = σ_16/σ_1.
    """
    Jd = J.double()                                       # см. докстринг: fp32 съедает малые σ
    s2 = torch.linalg.eigvalsh(Jd @ Jd.transpose(1, 2)).clamp_min(0.0).flip(1)  # σ² по убыв.
    s = s2.sqrt()
    jnorm = s2.sum(1).sqrt()
    pr = (s2.sum(1) ** 2) / s2.pow(2).sum(1).clamp_min(1e-30)
    g_uni = (J.sum(1) ** 2).sum(1) / (K_TOK * (jnorm ** 2)).clamp_min(1e-30)
    return dict(j_norm=jnorm.numpy(), pr_sigma=pr.numpy(), g_uni=g_uni.numpy(),
                gap_r16=((s[:, R_MAIN - 1] - s[:, R_MAIN]) / s[:, 0].clamp_min(1e-30)).numpy(),
                s16_over_s1=(s[:, R_MAIN - 1] / s[:, 0].clamp_min(1e-30)).numpy())


def main():
    t0 = time.time()
    torch.set_num_threads(len(os.sched_getaffinity(0)))   # не os.cpu_count(): на этой коробке
    # он возвращает 64 при 12 доступных ядрах, и OpenMP-потоки задыхаются на барьерах
    torch.set_grad_enabled(False)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ck = torch.load(jcache_path(L), map_location="cpu", weights_only=False)
    U128 = [int(t) for t in (ck["U128"].numpy() if hasattr(ck["U128"], "numpy")
                             else ck["U128"])]
    assert len(U128) == K_TOK
    del ck

    log("loading model (fp32) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    W = model.lm_head.weight.detach().float()            # [V, D]
    g = (1.0 + model.model.norm.weight.detach().float())  # [D]
    V = W.shape[0]
    M = W[U128] * g[None, :]                             # [K,D]

    rows, Vbuf, Hbuf, meta = [], [], [], {}
    rng = np.random.default_rng(7)
    for dom in DOMAINS:
        log(f"domain {dom}: load + capture ...")
        data = load_calibration_data(MODEL_DIR, max_length=S, min_length=S,
                                     n_samples=N_PER_DOMAIN, domain=dom,
                                     save_offset_mapping=True)
        if len(data) < N_PER_DOMAIN:
            raise RuntimeError(f"{dom}: {len(data)} доков, ожидалось {N_PER_DOMAIN}")
        ids = torch.stack([torch.as_tensor(d["input_ids"], dtype=torch.long)[:S]
                           for d in data])
        offs = torch.stack([torch.as_tensor(d["offset_mapping"], dtype=torch.int32)[:S]
                            for d in data])
        if ids.shape != (N_PER_DOMAIN, S):
            raise RuntimeError(f"{dom}: форма ids {tuple(ids.shape)} ≠ {(N_PER_DOMAIN, S)}")
        H = model.model(input_ids=ids).last_hidden_state.float()      # [N,S,D]
        Hbuf.append(H.half())

        ref = model(input_ids=ids[:2]).logits.float()
        dmax = float((H[:2] @ W.T - ref).abs().max())
        log(f"  route check |H@Wᵀ − logits|max = {dmax:.3e}")
        assert dmax < 0.2, f"{dom}: маршрут логиков не сошёлся ({dmax:.3e})"
        meta[f"route_check_{dom}"] = dmax

        N = ids.shape[0]
        tgt = torch.cat([ids[:, 1:], torch.zeros(N, 1, dtype=torch.long)], 1)
        tflat = tgt.reshape(-1)
        Hf = H.reshape(-1, D)
        n = N * S
        nll = np.empty(n, np.float32)
        ent = np.empty(n, np.float32)
        top1 = np.empty(n, np.float32)
        rank = np.empty(n, np.int32)
        for c0 in range(0, n, CH):
            c1 = min(c0 + CH, n)
            lg = Hf[c0:c1] @ W.T
            lse = torch.logsumexp(lg, 1)
            lsm = lg - lse[:, None]
            nll[c0:c1] = (lse - lg.gather(1, tflat[c0:c1, None])[:, 0]).numpy()
            ent[c0:c1] = (-(torch.exp(lsm) * lsm).sum(1)).numpy()
            top1[c0:c1] = torch.exp(lsm).max(1).values.numpy()
            rank[c0:c1] = ((lg > lg.gather(1, tflat[c0:c1, None])).sum(1) + 1).numpy()
            del lg, lse, lsm
        nll2 = nll.reshape(N, S)
        dnll = np.zeros_like(nll2)
        dnll[:, 1:] = np.abs(nll2[:, 1:] - nll2[:, :-1])
        # у ПОСЛЕДНЕЙ позиции документа цели нет: tgt там = pad, и nll/rank/align_next там
        # осмысленны только как «лог-вероятность паддинга». Ставим флаг has_target и по нему
        # режут все, кто берёт из таблицы средние (самопроверка PPL ниже тоже по нему).
        has_t = np.ones((N, S), np.float32)
        has_t[:, S - 1] = 0.0

        idn, tgn = ids.numpy(), tgt.numpy()
        # copy@k = цель (токен t+1) уже встречался в окне из k последних токенов, ВКЛЮЧАЯ
        # текущий: copy1 = буквальное повторение выданного токена, copy4/16 = короткое/среднее
        # окно (прокси induction-поведения, не сам induction-хед).
        copyk = {k: np.zeros((N, S), np.float32) for k in (1, 4, 16)}
        for i in range(N):
            for t in range(S):
                y = int(tgn[i, t])
                for k in copyk:
                    copyk[k][i, t] = float(y in idn[i, max(0, t - k + 1):t + 1])

        d1s = {r: np.full((N, S), np.nan, np.float32) for r in R_LIST}
        k2s = {r: np.full((N, S), np.nan, np.float32) for r in R_LIST}
        scal = {k: np.zeros((N, S), np.float32) for k in
                ("j_norm", "pr_sigma", "g_uni", "h_norm", "gap_r16", "s16_over_s1",
                 "align_next", "align_emit", "align_rand", "align_pca_next", "align_pca_rand",
                 "align_mean_next")}
        Vd = torch.empty(N, S, D, R_MAIN, dtype=torch.float16)

        def align(col_ids, V16):
            """‖P_{V_t} ŵ(y)‖ по позициям; ŵ = нормированная строка считывания с метрикой g."""
            m = W[col_ids] * g[None, :]
            m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            return (V16.transpose(1, 2) @ m[:, :, None])[:, :, 0].norm(dim=-1).numpy()

        for i in range(N):
            J = analytic_J(M, H[i])                      # [S,K,D]
            prof, _dm, _km, _nd = E28.document_profile(J, R_LIST, r_main=R_MAIN)
            V16 = E28.right_subspaces(J, [R_MAIN])[R_MAIN][0]     # [S,D,r]
            Vd[i] = V16.half()
            sc = leaf_scalars(J)
            scal["h_norm"][i] = H[i].norm(dim=-1).numpy()
            for key in sc:
                scal[key][i] = sc[key]
            for r in R_LIST:
                d1r, k2r = prof[r]
                d1s[r][i, :len(d1r)] = d1r.numpy()
                k2s[r][i, :len(k2r)] = k2r.numpy()
            scal["align_next"][i] = align(tgt[i], V16)
            scal["align_emit"][i] = align(ids[i], V16)
            rr = rng.integers(0, V, size=(S, 8))
            scal["align_rand"][i] = np.mean(
                [align(torch.from_numpy(rr[:, c]), V16) for c in range(8)], axis=0)
            # Конкуренты той же размерности: нецентрированный PCA облака h по документу (линейный
            # метод имеет полное право занять «массивное» среднее направление, листу оно недоступно
            # по построению) и ранг-1 направление среднего (несправедлив по размерности, отмечаем).
            Hd = H[i].double()
            V_pca = torch.linalg.svd(Hd, full_matrices=False)[2][:R_MAIN].T.contiguous()
            mu = Hd.mean(0)
            U_mu = (mu / mu.norm().clamp_min(1e-12))[:, None]
            Vp16 = V_pca.float().unsqueeze(0).expand(S, D, R_MAIN)
            scal["align_pca_next"][i] = align(tgt[i], Vp16)
            # «Случайный токен» в БАЗИСЕ PCA — иначе сравнивать основания нельзя: align растёт с
            # захваченной энергией, и разница bases в абсолютных числах ничего не значит.
            scal["align_pca_rand"][i] = np.mean(
                [align(torch.from_numpy(rr[:, c]), Vp16) for c in range(8)], axis=0)
            scal["align_mean_next"][i] = align(
                tgt[i], U_mu.float().unsqueeze(0).expand(S, D, 1))
            del J, V16, Hd, V_pca, U_mu

        toks = [tok.decode([int(x)]) for x in ids.reshape(-1)]
        tk_len = np.zeros(n, np.float32)
        tk_nl = np.zeros(n, np.float32)
        tk_sent = np.zeros(n, np.float32)
        tk_digit = np.zeros(n, np.float32)
        for j, s in enumerate(toks):
            tk_len[j] = len(s)
            tk_nl[j] = float("\n" in s)
            tk_sent[j] = float(("\n" in s) or s.rstrip().endswith(SENT_PUNCT))
            tk_digit[j] = float(s.strip().isdigit())
        off_end = offs[:, :, 1].numpy().reshape(-1)

        for i in range(N):
            for t in range(S):
                j = i * S + t
                row = dict(domain=dom, doc=i, t=t, tok=int(idn[i, t]), nxt=int(tgn[i, t]),
                           nll=float(nll[j]), ent=float(ent[j]), top1=float(top1[j]),
                           rank=int(rank[j]), in5=float(rank[j] <= 5),
                           in50=float(rank[j] <= 50),
                           copy1=float(copyk[1][i, t]), copy4=float(copyk[4][i, t]),
                           copy16=float(copyk[16][i, t]), dnll=float(dnll[i, t]),
                           has_target=float(has_t[i, t]),
                           j_norm=float(scal["j_norm"][i, t]),
                           pr_sigma=float(scal["pr_sigma"][i, t]),
                           g_uni=float(scal["g_uni"][i, t]),
                           h_norm=float(scal["h_norm"][i, t]),
                           gap_r16=float(scal["gap_r16"][i, t]),
                           s16_over_s1=float(scal["s16_over_s1"][i, t]),
                           align_next=float(scal["align_next"][i, t]),
                           align_emit=float(scal["align_emit"][i, t]),
                           align_rand=float(scal["align_rand"][i, t]),
                           align_pca_next=float(scal["align_pca_next"][i, t]),
                           align_pca_rand=float(scal["align_pca_rand"][i, t]),
                           align_mean_next=float(scal["align_mean_next"][i, t]),
                           tok_len=float(tk_len[j]), tok_nl=float(tk_nl[j]),
                           tok_sent_end=float(tk_sent[j]), tok_digit=float(tk_digit[j]),
                           off_end=int(off_end[j]))
                for r in R_LIST:
                    row[f"d1_r{r}"] = None if np.isnan(d1s[r][i, t]) else float(d1s[r][i, t])
                    row[f"k2_r{r}"] = None if np.isnan(k2s[r][i, t]) else float(k2s[r][i, t])
                rows.append(row)
        Vbuf.append(Vd)
        log(f"  {dom}: {n} позиций, PPL={np.exp(nll.mean()):.2f}, t={time.time() - t0:.0f} с")

    # ---------------- самопроверки ----------------
    k2 = np.array([r["k2_r16"] for r in rows], dtype=np.float64)
    kmed = float(np.nanmedian(k2))
    log(f"κ(r=16) медиана на свежем тексте = {kmed:.5f}  (кэш exp27b: {KAPPA_CACHE})")
    assert 0.5 * KAPPA_CACHE < kmed < 2.0 * KAPPA_CACHE, f"масштаб κ ушёл: {kmed:.5f}"
    ar = np.array([r["align_rand"] for r in rows])
    theory = (R_MAIN / D) ** 0.5
    log(f"align_rand = {ar.mean():.4f} против Haar-ориентира √(r/D) = {theory:.4f} "
        f"(E‖P ŵ‖² = r/D; для нормы поправка ~1 %)")
    assert abs(ar.mean() / theory - 1) < 0.05, "Haar-нулевой уровень проекций не сошёлся"
    ht = np.array([r["has_target"] for r in rows], dtype=bool)
    ppl = float(np.exp(np.mean(np.array([r["nll"] for r in rows], dtype=np.float64)[ht])))
    log(f"PPL таблицы = {ppl:.2f} (позиции всех четырёх доменов, {int(ht.sum())} из {len(rows)}"
        f" с целью)")
    assert 5.0 < ppl < 200.0, f"PPL таблицы неправдоподобен: {ppl:.2f}"
    d1 = np.array([r["d1_r16"] for r in rows], dtype=np.float64)
    assert np.nanmax(np.abs(d1)) <= 1.5708 + 1e-6, "Δ_SVD вне [0, π/2]"

    meta.update(dict(experiment="E-45b", model=MODEL_DIR, L=L, r_main=R_MAIN,
                     r_list=R_LIST, domains=DOMAINS, n_per_domain=N_PER_DOMAIN, S=S,
                     n_rows=len(rows), n_with_target=int(ht.sum()),
                     kappa_median_r16=kmed, kappa_cache=KAPPA_CACHE,
                     align_rand_mean=float(ar.mean()), align_haar_theory=theory,
                     ppl_table=ppl, wall_sec=time.time() - t0,
                     note=("границы предложений — прокси по строкам токена; "
                           "J — readout-линеаризованный слой 23")))
    with gzip.open("../results/exp45b_leaf_table.json.gz", "wt") as f:
        json.dump({"meta": meta, "rows": rows}, f, ensure_ascii=False)
    np.savez_compressed("../results/exp45b_leaf_table.npz",
                        V=torch.cat(Vbuf, 0).numpy(), H=torch.cat(Hbuf, 0).numpy())
    log(f"готово за {time.time() - t0:.0f} с → results/exp45b_leaf_table.json.gz (+.npz)")


if __name__ == "__main__":
    main()
