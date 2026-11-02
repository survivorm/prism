"""E-39m — Робастен ли плоский носитель к PTQ (mini-тест на честном носителе E-39d).

Зачем. E-39g/h были приостановлены с основанием «носитель (плоский лист) недостижим»
(E-39c REFUTED, E-39d не добрал κ < 0.005 с целым PPL). Основание ложно: честный floor
E-39d (λ=0.01, λo=1.0, seed=0) дал κ' = 0.009206 = −42% от базы прогона при PPL +0.99%
и corr_f = 0.9992 (`results/exp39d_stage1_ordering.log:34`). Значит носитель с существенно
более плоским листом при сохранном поведении СУЩЕСТВУЕТ, и вопрос «даёт ли плоскость что-нибудь
downstream» можно закрыть на нём, без новой тренировки.

Вопрос мини-теста (ветка E-39g/h = «плоский лист снижает урон от PTQ»):

    у носителя с κ' на 42% ниже, меньше ли урон от ОДНОГО И ТОГО ЖЕ GPTQ-4bit (g=128)?

Дизайн (пререгистрация до запуска; пороги не правятся задним числом):

  Носители (все — одна и та же модель, отличается только W_lm[U128]):
    base            — без тренировки, κ' = 0.015923 (база этого протокола)
    cheat           — λ=0.01, λo=0    → κ' 0.004443, PPL +12.01%, corr_f 0.5936
                      (плоскость через перестановку рангов; sanity-строка, не вердикт)
    floor_pearson   — λ=0.01, λo=1.0, order_mode=pearson
    floor_pairwise  — λ=0.01, λo=1.0, order_mode=pairwise
  Режим ordering в логе E-39d не записан, поэтому воспроизводятся ОБА, носителем
  объявляется тот, который совпал с логом (|κ' − 0.009206| и |PPL − 34.6523|); второй
  остаётся в JSON как отбракованный. Якорь детерминизма: cheat обязан дать κ' ≈ 0.004443.

  Воздействие: GPTQ 4-bit group=128 по всей tied-матрице W_lm (то есть и readout, и
  эмбеддинги — как настоящий PTQ; см. E-39f). Hessian калибровки считается один раз на
  базовой модели и общий для всех носителей: разница между носителями — только в весах.

  Метрики на носитель: κ' (E-39a dry-run, M_K = W_rows ⊙ g), PPL на C4-holdout
  (`_c4_docs(8, 128, skip=4096)` — та же шкала, что у E-39d, НЕсравнимо с PPL 30.47 из
  E-39b/E-39f), corr_f и KL квантованного носителя против его же fp-версии.

  Гейты (прereg, ΔPPL в процентах от базы своего носителя):
    flat_protects  ΔPPL(floor) < ΔPPL(base) − 0.5 п.п.
    flat_neutral   |ΔPPL(floor) − ΔPPL(base)| ≤ 0.5 п.п.
    flat_worse     ΔPPL(floor) > ΔPPL(base) + 0.5 п.п.
  Sanity: ΔPPL(cheat) > ΔPPL(base) (плоскость, купленная поведением, не защищает).

Механика: `experiments/exp39d_l23_training.py` не меняется (файл параллельной сессии и
носитель обязан быть ровно его конфигурацией) — тренировка вызывается его же функциями
потеря-в-точность, тот же seed/бюджет шагов/сплиты.

Гейт (по разбору `docs/E39m_critique.md` §2: мерится СУММАРНЫЙ урон, не инкрементальный).
Носитель стартует уже с уроном (+0.99%), поэтому «плоскость защищает» означает «после PTQ
носитель лучше базы», а не «носитель менее чувствителен к PTQ». Для режима r:

    inc(c)   = PPL(c+Q)/PPL(c_fp) − 1        инкрементальный урон (шкала E-39f)
    total(c) = PPL(c+Q)/PPL(base_fp) − 1     суммарный урон от базы
    cmp(c)   = PPL(c+Q)/PPL(base+Q) − 1      носитель против базы ПОСЛЕ PTQ (тот же holdout)

`flat_protects` требует `cmp(floor) < −GATE_PP` И `inc(floor) − inc(base) < −GATE_PP`, причём
оба — по верхней границе парного bootstrap CI на 32 документах (конъюнкция: нельзя объявить
защиту, выиграв только на одной шкале). `flat_worse` — симметрично по нижней границе CI.

Запуск: локально `--phase smoke` (трубка), на Kaggle `--phase all` (три носителя + два
режима квантования одним процессом; ~80 мин T4, якорь E-39d: 1287 с на один носитель).
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

# диагностика краша v3 на T4 (device-side assert без traceback'а): синхронный CUDA
# убирает переупорядочивание ядер, цена ~2× на мелких операциях. Снять после отладки.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

try:                    # в ноутбуке Kaggle `__file__` не определён (upstream-краш v1)
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# На Kaggle этого файла нет рядом: соседи берутся из датасета с кодом (обычный import,
# поэтому `if __name__ == "__main__"`-гарды и цепочки импортов работают как локально).
for _p in ("/kaggle/input/prism-exp-code",
           "/kaggle/input/datasets/voroninmichael/prism-exp-code"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import torch.nn.functional as F             # noqa: E402

import exp39d_l23_training as d39          # noqa: E402  (носитель = его конфигурация)
from exp39f_full_tied import _quantize_gptq  # noqa: E402

REPO = d39.REPO
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GPTQ_BITS, GPTQ_GROUP, DAMP_FRAC = 4, 128, 0.01
# режимы квантования: name -> (bits, group). g128_b4 = спецификация; g128_b3 — режим, где
# эффект велик (E-39f: PPL +9.6% против +1.76%), то есть разрешение по 0.5 п.п. на порядок
# лучше (разбор §4: на g128 тест мерит 0.5 п.п. на фоне шума).
REGIMES = {"g128_b4": (4, 128), "g128_b3": (3, 128)}
C4_CAL_SKIP, N_CAL, S_CAL = 8192, 128, 128       # вербатим E-39f (GPTQ-калибровка)
LAMBDA_FLAT, LAMBDA_ORDER, SEED = 0.01, 1.0, 0

# якоря из `results/exp39d_stage1_ordering.log` (E-39d round 11) — сверка воспроизведения
LOG_CHEAT = {"kappa": 0.004443, "ppl": 38.4368, "corr_f": 0.5936}
LOG_FLOOR = {"kappa": 0.009206, "ppl": 34.6523, "corr_f": 0.9992, "kl": 0.0043}
KAPPA_TOL_REL = 0.03          # κ' носителя обязан сойтись с логом в пределах 3%
PPL_TOL_REL = 0.005           # и PPL — в пределах 0.5%
GATE_PP = 0.5                 # п.п. ΔPPL, порог `neutral`
PPL_DOCS = 32                 # разбор §3: 8 доков E-39d не дают разрешения 0.5 п.п.
PPL_CHUNK = 4                 # doc за раз: логиты 4×128×248320 fp32 ≈ 0.5 GB
BOOT = 2000                   # парный bootstrap по документам
BOOT_SEED = 0
CORR_F_HONEST = 0.95          # разбор §5: ниже — носитель уже «куплен поведением»
PPL_BASE_ANCHOR = 34.3142     # results/exp39d_stage1_ordering.log, шкала 8 доков
OUT_DIR = os.path.join(REPO, "results")

CARRIERS = {                  # name -> (lambda_flat, lambda_order, order_mode)
    "cheat": (LAMBDA_FLAT, 0.0, "pearson"),
    "floor_pearson": (LAMBDA_FLAT, LAMBDA_ORDER, "pearson"),
    "floor_pairwise": (LAMBDA_FLAT, LAMBDA_ORDER, "pairwise"),
}


# ---------------------------------------------------------------- носитель
def build_context():
    """Всё, что нужно для тренировки и оценки: модель, кэш L23, g, базовые логиты, PPL-доки.

    Повторяет Шаг 0 `exp39d_l23_training.main()` (тот же порядок, те же константы), но
    ничего не тренирует и ничего не пишет.
    """
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    kw = {"dtype": torch.float32} if _ver >= (4, 35) else {"torch_dtype": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(d39.MODEL, **kw)
    model.eval()
    model.to(DEV)
    W_lm = model.lm_head.weight.detach().clone()
    norm_mod = getattr(model.model, "norm", None)
    g_base = ((1.0 + norm_mod.weight.float()).detach().clone()
              if norm_mod is not None and hasattr(norm_mod, "weight")
              else torch.ones(d39.D))

    ck = torch.load(d39._jcache_l23_path(), map_location="cpu", weights_only=False)
    H_dict, U128, S = ck["H"], [int(t) for t in ck["U128"]], ck["S"]
    assert ck["D"] == d39.D
    # всё, что участвует в матричных операциях, сразу на DEV: на GPU `matmul` CPU×CUDA
    # падает (краш v1 на `u_all @ M_K_full.t()`), на CPU это no-op
    tok_ids = torch.tensor(U128, dtype=torch.long, device=DEV)
    K = len(U128)

    docs = {}
    for (dom, si, t) in H_dict:
        docs.setdefault((dom, si), {})[int(t)] = H_dict[(dom, si, t)]
    all_keys = sorted(H_dict.keys())
    H_all = torch.stack([H_dict[k].float() for k in all_keys]).to(DEV)
    N_total = H_all.shape[0]

    doc_ids = sorted(docs.keys())
    u64 = []
    for did in doc_ids:
        seq = docs[did]
        assert len(seq) == S, f"doc {did}: {len(seq)} != {S}"
        arr = torch.stack([seq[t].float() for t in range(S)])
        arr = arr / arr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u64.append(arr.double())
    u64_docs = torch.stack(u64)
    N_docs = u64_docs.shape[0]

    # ACTUAL logits через факторизацию E-38e (round 8 CRITICAL #1), вербатим exp39d
    u_all = H_all / H_all.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    M_K_full = W_lm * g_base[None, :]
    base_k_all = torch.empty(N_total, K, device=DEV)
    lse_b_all = torch.empty(N_total, device=DEV)
    raw_sum_all = torch.empty(N_total, device=DEV)
    SQRT_D = math.sqrt(d39.D)
    for i in range(0, N_total, 256):
        bl = SQRT_D * (u_all[i:i + 256] @ M_K_full.t())
        lse_b_all[i:i + 256] = torch.logsumexp(bl, dim=-1)
        base_k_all[i:i + 256] = bl[:, tok_ids]
        raw_sum_all[i:i + 256] = torch.exp(bl[:, tok_ids] - lse_b_all[i:i + 256, None]).sum(1)
        del bl
    mask = raw_sum_all >= d39.SENTINEL_SUM
    keep_idx = torch.where(mask)[0]
    if keep_idx.numel() == 0:
        keep_idx = torch.arange(N_total)

    tok = AutoTokenizer.from_pretrained(d39.MODEL)
    # первые d39.N_DOCS_PPL документов совпадают с якорным набором E-39d (_c4_docs берёт
    # первые n с того же skip), поэтому из этого же прогона воспроизводится и шкала лога
    ppl_ids, ppl_attn = d39._c4_docs(PPL_DOCS, d39.S_LEN, tok)
    ppl_ids, ppl_attn = ppl_ids.to(DEV), ppl_attn.to(DEV)

    ctx = dict(model=model, W_lm=W_lm, g_base=g_base.to(DEV), U128=U128,
               tok_ids=tok_ids.to(DEV), K=K, S=S, N_total=N_total, N_docs=N_docs,
               H_all=H_all.to(DEV), u64_docs=u64_docs.to(DEV),
               base_k_all=base_k_all.to(DEV), lse_b_all=lse_b_all.to(DEV),
               keep_idx=keep_idx.to(DEV), ppl_ids=ppl_ids, ppl_attn=ppl_attn,
               n_sent=int((~mask).sum()))
    # нормирователь flatness-лосса — та же схема, что exp39d round 9 #6
    torch.manual_seed(SEED)
    with torch.no_grad():
        norm_idx = keep_idx[torch.randint(0, keep_idx.numel(), (d39.P_STEP,),
                                          device=keep_idx.device)]
        M0 = (W_lm[tok_ids] * g_base[None, :])
        l_flat_base = float(d39.flatness_loss(M0, H_all[norm_idx], d39.R, d39.MB))
    ctx["l_flat_base"] = l_flat_base
    # fail-loud на индексы: device-side assert без traceback'а (краш v3) ищется тяжело
    assert int(tok_ids.max()) < W_lm.shape[0] and int(tok_ids.min()) >= 0
    assert int(keep_idx.max()) < N_total and int(keep_idx.min()) >= 0
    assert H_all.device.type == DEV.split(":")[0] and W_lm.device.type == DEV.split(":")[0]
    print(f"[ctx] N={N_total} docs={N_docs} sentinel={ctx['n_sent']} "
          f"l_flat_base={l_flat_base:.4f}", flush=True)
    return ctx


def train_carrier(ctx, lam_flat: float, lam_order: float, order_mode: str) -> dict:
    """Один прогон тренировки W_lm[U128] — шаг-в-шаг как в exp39d.main()."""
    W_lm, g, tok_ids, H_all = ctx["W_lm"], ctx["g_base"], ctx["tok_ids"], ctx["H_all"]
    keep_idx, base_k_all, lse_b_all = ctx["keep_idx"], ctx["base_k_all"], ctx["lse_b_all"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    W_rows = W_lm[tok_ids].detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([W_rows], lr=d39.LR)
    hist = []
    t0 = time.time()
    for step in range(1, d39.N_STEPS + 1):
        opt.zero_grad()
        M_K = W_rows * g[None, :]
        idx = keep_idx[torch.randint(0, keep_idx.numel(), (d39.P_STEP,),
                                     device=keep_idx.device)]
        H_sub = H_all[idx]
        l_flat = d39.flatness_loss(M_K, H_sub, d39.R, d39.MB) / ctx["l_flat_base"]
        l_kl = d39.kl_leash(H_sub, W_rows, g, base_k_all[idx], lse_b_all[idx])
        use_ord = (order_mode != "none" and lam_order > 0)
        if use_ord:
            fn = (d39.ordering_loss_pearson if order_mode == "pearson"
                  else d39.ordering_loss_pairwise)
            l_order, corr_run = fn(H_sub, W_rows, g, base_k_all[idx])
        else:
            with torch.no_grad():
                _, corr_run = d39.ordering_corr(H_sub, W_rows.detach(), g,
                                                base_k_all[idx])
                l_order = 1.0 - corr_run.mean()
        loss = lam_flat * l_flat + d39.LAMBDA_KL * l_kl
        if use_ord:
            loss = loss + lam_order * l_order
        if not torch.isfinite(loss.detach()):
            print(f"  [{order_mode} λo={lam_order}] шаг {step}: не-конечный лосс, стоп",
                  flush=True)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([W_rows], 1.0)
        opt.step()
        if step <= 5 or step % 10 == 0:
            print(f"    step {step}: loss={float(loss.detach()):.6f} "
                  f"finite={bool(torch.isfinite(W_rows).all())}", flush=True)
        if step % d39.METRIC_EVERY == 0 or step == d39.N_STEPS:
            with torch.no_grad():
                kp = d39._kappa_dryrun((W_rows * g[None, :]).double(), ctx["u64_docs"],
                                       ctx["N_docs"], ctx["S"], d39.R)
                corr_f = _corr_f(ctx, W_rows)
            hist.append({"step": step, "kappa": kp["kappa"], "corr_run": float(corr_run.mean()),
                         "corr_f": corr_f, "l_flat": float(l_flat), "l_kl": float(l_kl)})
            print(f"  [{order_mode} λo={lam_order}] st{step}: κ'={kp['kappa']:.6f} "
                  f"corr_run={float(corr_run.mean()):.4f} corr_f={corr_f:.4f}", flush=True)
    W_out = W_lm.clone()
    W_out[tok_ids] = W_rows.detach()
    return {"lambda_flat": lam_flat, "lambda_order": lam_order, "order_mode": order_mode,
            "seed": SEED, "W_rows": W_rows.detach().clone(), "W": W_out,
            "history": hist, "kappa_final": hist[-1]["kappa"] if hist else None,
            "corr_f_final": hist[-1]["corr_f"] if hist else None,
            "train_wall_sec": time.time() - t0}


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def _corr_f(ctx, W_rows: torch.Tensor, n_pos: int = 512) -> float:
    """corr_f: Pearson ранга K-логитов носителя против базы, усреднение по n_pos позиций.

    Это главный контроль «плоскость не купена поведением»: PPL +1% не различает честный
    floor и rank-permutation (у обоих ≈ +1%), поэтому полагаться можно только на него.
    """
    idx = torch.as_tensor(np.linspace(0, ctx["N_total"] - 1, n_pos).astype(np.int64),
                          device=DEV)
    u = ctx["H_all"][idx]
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    mk = (W_rows * ctx["g_base"][None, :]).t()          # [D, K], колонки = строки U128 по порядку
    assert mk.shape[1] == ctx["K"] == ctx["base_k_all"].shape[1], \
        f"corr_f: {mk.shape[1]} колонок против K={ctx['K']}"
    # ВАЖНО: tok_ids — это id в словаре (до 151k), а не столбцы mk; повторная индексация
    # столбцов давала device-side assert на T4 (краш v3/v4). Порядок столбцов уже U128.
    lg = (math.sqrt(d39.D) * (u @ mk)).detach().cpu().numpy()
    bk = ctx["base_k_all"][idx].detach().cpu().numpy()
    return float(np.mean([_pearson(lg[i], bk[i]) for i in range(len(idx))]))

# ---------------------------------------------------------------- метрики
def kappa_of(ctx, W_rows: torch.Tensor) -> float:
    Mk = (W_rows * ctx["g_base"][None, :]).double()
    return d39._kappa_dryrun(Mk, ctx["u64_docs"], ctx["N_docs"], ctx["S"], d39.R)["kappa"]


def ppl_docs_of(ctx, W: torch.Tensor | None):
    """Пер-доковый средний NLL на holdout'е при tied-матрице W (None = базовые веса).

    Возвращает (nll[n_docs], ntok[n_docs]). Полный forward модели, у которой tied-матрица
    заменена на W: меняются и эмбеддинги, и head (одна и та же память в настоящей модели),
    поэтому измеряется полный tied-канал PTQ, как в E-39f. После замера веса обязательно
    восстанавливаются — это предусловие того, что следующий носитель стартует из базы.

    Агрегат `ppl_from` на этом же наборе совпадает с `d39._ppl` (взвешивание по токенам),
    поэтому одно измерение даёт и якорную шкалу E-39d (первые 8 доков), и рабочую (32).
    """
    model = ctx["model"]
    emb = model.get_input_embeddings().weight
    lm_head = model.lm_head.weight
    ids, attn = ctx["ppl_ids"], ctx["ppl_attn"]
    swap = W is not None
    if swap:
        saved = emb.data.clone()
        assert torch.equal(saved, lm_head.data), "tied-матрица уже разделена — протокол сломан"
        emb.data.copy_(W.to(emb.dtype))
        lm_head.data.copy_(W.to(lm_head.dtype))
    nll, ntok = [], []
    try:
        with torch.no_grad():
            for c0 in range(0, ids.shape[0], PPL_CHUNK):
                i, a = ids[c0:c0 + PPL_CHUNK], attn[c0:c0 + PPL_CHUNK]
                lg = model(input_ids=i, attention_mask=a).logits
                sl = lg[:, :-1, :].contiguous()
                st, sa = i[:, 1:].contiguous(), a[:, 1:].contiguous()
                loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                                       reduction="none").view(st.shape)
                for b in range(i.shape[0]):
                    m = sa[b] > 0
                    nll.append(float(loss[b][m].mean()))
                    ntok.append(int(m.sum()))
                del lg, sl, loss
    finally:
        if swap:
            emb.data.copy_(saved)
            lm_head.data.copy_(saved)
            assert torch.equal(emb.data, ctx["W_lm"].to(emb.dtype)), "restore не сработал"
    return np.array(nll, dtype=np.float64), np.array(ntok, dtype=np.int64)


def ppl_from(nll, ntok, first: int | None = None) -> float:
    """PPL = exp(Σ nll_d·n_d / Σ n_d) — та же свёртка, что в `d39._ppl`."""
    n = np.asarray(nll, dtype=np.float64)
    t = np.asarray(ntok, dtype=np.float64)
    if first:
        n, t = n[:first], t[:first]
    return float(math.exp(float((n * t).sum() / t.sum())))


def _rel_pct(diff, ntok, idx=None) -> float:
    """Парный сдвиг NLL (одни и те же доки) → % прироста PPL; idx — bootstrap-выборка доков."""
    d = np.asarray(diff, dtype=np.float64)
    t = np.asarray(ntok, dtype=np.float64)
    if idx is not None:
        d, t = d[idx], t[idx]
    return (math.exp(float((d * t).sum() / t.sum())) - 1.0) * 100.0


def _boot_ci(fn, n: int, boot: int = BOOT, seed: int = BOOT_SEED):
    """CI95 для fn(индексы доков); fn обязан опираться на одни и те же доки во всех
    сравниваемых условиях (парный bootstrap: документ переворачивается для всех моделей сразу).
    """
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(boot)], dtype=np.float64)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def ppl_of(ctx, W: torch.Tensor) -> float:
    """PPL по всем holdout-докам (обёртка над ppl_docs_of — для смоук-проверки)."""
    nll, ntok = ppl_docs_of(ctx, W)
    return ppl_from(nll, ntok)


def calib_activations(ctx) -> torch.Tensor:
    """Hessian-активации: пост-norm состояние БАЗОВОЙ модели на C4 (вербатим E-39f:
    128 документов × 128 токенов, skip 8192; калибровка одна на все носители)."""
    from transformers import AutoTokenizer
    model = ctx["model"]
    tok = AutoTokenizer.from_pretrained(d39.MODEL)
    ids, attn = d39._c4_docs(N_CAL, S_CAL, tok, skip=C4_CAL_SKIP)
    # v5 упало здесь: _c4_docs отдаёт CPU-тензоры, а модель на GPU (E-39f делал .to() сам).
    ids, attn = ids.to(DEV), attn.to(DEV)
    caps = []
    text = model.model
    with torch.no_grad():
        for c0 in range(0, N_CAL, 8):
            c1 = min(c0 + 8, N_CAL)
            o = text(input_ids=ids[c0:c1], attention_mask=attn[c0:c1])
            caps.append(o.last_hidden_state[attn[c0:c1] > 0].float())
            del o
            print(f"  calib {c1}/{N_CAL}", flush=True)
    X = torch.cat(caps, dim=0)
    del caps
    return X


# ---------------------------------------------------------------- фазы
def phase_smoke(out_path: str) -> dict:
    """Дешёвая проверка трубки перед часами счёта: контекст, база, один GPTQ, restore."""
    t0 = time.time()
    ctx = build_context()
    W_rows = ctx["W_lm"][ctx["tok_ids"]].clone()
    k_base = kappa_of(ctx, W_rows)
    nll_b, nt = ppl_docs_of(ctx, None)
    ppl_base = ppl_from(nll_b, nt)
    ppl_log = ppl_from(nll_b, nt, first=d39.N_DOCS_PPL)
    print(f"[base] κ'={k_base:.6f} PPL={ppl_base:.4f} (якорная шкала 8 доков {ppl_log:.4f})",
          flush=True)
    # свёртка пер-докового NLL обязана совпасть с d39._ppl на том же наборе доков
    chk = ppl_from(*ppl_docs_of(ctx, None), first=d39.N_DOCS_PPL)
    assert abs(chk - ppl_log) < 1e-9
    assert abs(ppl_log - PPL_BASE_ANCHOR) < 0.02, f"шкала E-39d не сошлась: {ppl_log}"
    # self-check канала corr_f: на T4 он падал на индексации (краш v3/v4), ловить до GPTQ
    cf_self = _corr_f(ctx, ctx["W_lm"][ctx["tok_ids"]].detach().clone(), n_pos=16)
    assert abs(cf_self - 1.0) < 1e-6, f"corr_f(base, base) = {cf_self} — канал сломан"
    Wp = ctx["W_lm"][ctx["tok_ids"]].detach().clone()
    cf_perm = _corr_f(ctx, Wp[torch.randperm(ctx["K"], device=Wp.device)], n_pos=16)
    print(f"[corr_f] base-vs-base {cf_self:.6f}, perm-stub {cf_perm:.4f}", flush=True)
    assert cf_perm < 0.9, f"corr_f на перемешанных строках = {cf_perm} — метрика пуста"
    X = calib_activations(ctx)
    t1 = time.time()
    W_q = _quantize_gptq(ctx["W_lm"], X[:4096], GPTQ_BITS, GPTQ_GROUP, DAMP_FRAC,
                         torch.device(DEV))
    print(f"[gptq] wall {time.time() - t1:.0f}s", flush=True)
    k_q = kappa_of(ctx, W_q[ctx["tok_ids"]].clone())
    ppl_q = ppl_of(ctx, W_q)
    out = {"device": DEV, "ppl_docs": int(nt.shape[0]),
           "base": {"kappa": k_base, "ppl": ppl_base, "ppl_log_scale": ppl_log,
                    "n_sent": ctx["n_sent"], "l_flat_base": ctx["l_flat_base"]},
           "quant_small_calib": {"kappa": k_q, "ppl": ppl_q,
                                 "ppl_delta_pct": (ppl_q / ppl_base - 1) * 100},
           "gptq_wall_sec": time.time() - t1, "wall_sec": time.time() - t0}
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def phase_train(out_path: str) -> dict:
    ctx = build_context()
    return run_train(ctx, out_path)


def run_train(ctx, out_path: str) -> dict:
    t0 = time.time()
    W_rows_base = ctx["W_lm"][ctx["tok_ids"]].clone()
    k_base = kappa_of(ctx, W_rows_base)
    nll_base, ntok = ppl_docs_of(ctx, None)
    ppl_base = ppl_from(nll_base, ntok, first=d39.N_DOCS_PPL)   # якорная шкала E-39d (8 доков)
    print(f"[base] κ'={k_base:.6f} PPL={ppl_base:.4f} (8-док шкала) / "
          f"{ppl_from(nll_base, ntok):.4f} (рабочая, {ntok.size} доков)", flush=True)
    assert abs(k_base - 0.015923) < 3e-4, f"база не сходится с логом E-39d: {k_base}"
    assert abs(ppl_base - PPL_BASE_ANCHOR) < 0.02, f"шкала E-39d не сошлась: {ppl_base}"

    out = {"experiment": "E-39m flat carrier vs PTQ (фаза train: воспроизведение носителя)",
           "spec": "PLAN.md (мини-тест на честном носителе E-39d)",
           "device": DEV, "ppl_docs": int(ntok.size),
           "base": {"kappa": k_base, "ppl_log_scale": ppl_base,
                    "ppl": ppl_from(nll_base, ntok)},
           "log_anchors": {"cheat": LOG_CHEAT, "floor": LOG_FLOOR},
           "carriers": {}}
    for name, (lf, lo, mode) in CARRIERS.items():
        print(f"=== carrier {name} (λ={lf}, λo={lo}, {mode})", flush=True)
        c = train_carrier(ctx, lf, lo, mode)
        nll_c, nt_c = ppl_docs_of(ctx, c["W"])
        assert np.array_equal(nt_c, ntok), "сдвинулся holdout — парный bootstrap невалиден"
        ppl = ppl_from(nll_c, nt_c, first=d39.N_DOCS_PPL)
        rec = {"kappa": c["kappa_final"], "corr_f": c["corr_f_final"], "ppl": ppl,
               "ppl_full": ppl_from(nll_c, nt_c),
               "ppl_delta_pct": (ppl / ppl_base - 1) * 100,
               "history": c["history"], "train_wall_sec": c["train_wall_sec"],
               "kappa_rel_to_base": c["kappa_final"] / k_base - 1}
        anchor = LOG_CHEAT if name == "cheat" else LOG_FLOOR
        rec["anchor_match"] = {
            "kappa_rel_err": abs(c["kappa_final"] / anchor["kappa"] - 1),
            "ppl_rel_err": abs(ppl / anchor["ppl"] - 1),
            "ok": (abs(c["kappa_final"] / anchor["kappa"] - 1) <= KAPPA_TOL_REL
                   and abs(ppl / anchor["ppl"] - 1) <= PPL_TOL_REL),
        }
        print(f"  → κ'={rec['kappa']:.6f} PPL={ppl:.4f} "
              f"({rec['ppl_delta_pct']:+.2f}%) corr_f={rec['corr_f']:.4f} "
              f"anchor_ok={rec['anchor_match']['ok']}", flush=True)
        torch.save({"W_rows": c["W_rows"].cpu(), "rec": rec,
                    "nll_fp": nll_c, "ntok": nt_c}, _cpath(name))
        out["carriers"][name] = rec
    out["wall_sec"] = time.time() - t0
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"output: {out_path} wall {out['wall_sec']:.0f}s", flush=True)
    return out


def phase_quant(out_path: str) -> dict:
    ctx = build_context()
    return run_quant(ctx, out_path)


def phase_all(train_path: str, quant_path: str) -> dict:
    """Один процесс (Kaggle): контекст, три носителя, затем оба режима PTQ на них же."""
    ctx = build_context()
    run_train(ctx, train_path)
    return run_quant(ctx, quant_path)


def _cpath(name: str) -> str:
    return os.path.join(OUT_DIR, f"exp39m_carrier_{name}.pt")


def _load_carrier(ctx, name: str):
    """Носитель из фазы train: (W, W_rows, rec, nll_fp, ntok) или None, если файла нет."""
    p = _cpath(name)
    if not os.path.exists(p):
        return None
    blob = torch.load(p, map_location="cpu", weights_only=False)
    W_rows = blob["W_rows"].to(DEV)
    W = ctx["W_lm"].clone()
    W[ctx["tok_ids"]] = W_rows
    nll = blob.get("nll_fp")
    return W, W_rows, blob["rec"], (None if nll is None else np.asarray(nll, np.float64)), \
        (None if nll is None else np.asarray(blob["ntok"], np.int64))


def _verdict(ci) -> str:
    """Вердикт по CI разности: «лучше» только если ВЕСЬ CI ниже −GATE_PP (консервативно)."""
    if ci[1] < -GATE_PP:
        return "floor_better"
    if ci[0] > GATE_PP:
        return "floor_worse"
    return "neutral"


def run_quant(ctx, out_path: str) -> dict:
    t0 = time.time()
    base_rows = ctx["W_lm"][ctx["tok_ids"]].clone()
    k_base = kappa_of(ctx, base_rows)
    nll_base, ntok = ppl_docs_of(ctx, None)
    ppl_base = ppl_from(nll_base, ntok)
    assert abs(k_base - 0.015923) < 3e-4, f"база не сходится с логом E-39d: {k_base}"
    assert abs(ppl_from(nll_base, ntok, first=d39.N_DOCS_PPL) - PPL_BASE_ANCHOR) < 0.02
    print(f"[quant] база: κ'={k_base:.6f} PPL={ppl_base:.4f}  "
          f"holdout {ntok.size} доков × {int(ntok.mean())} ток.", flush=True)
    print("[quant] GPTQ-калибровка (общая для всех носителей)...", flush=True)
    X = calib_activations(ctx)
    print(f"  X: {tuple(X.shape)}", flush=True)

    items = {"base": (ctx["W_lm"].clone(), base_rows,
                      {"kappa": k_base, "corr_f": 1.0, "anchor_match": None},
                      nll_base, ntok)}
    for name in CARRIERS:
        got = _load_carrier(ctx, name)
        if got is None:
            print(f"  [skip] нет носителя {name}", flush=True)
            continue
        W, W_rows, rec, nll_fp, nt_fp = got
        if nll_fp is None:                      # носитель из старой версии фазы train
            nll_fp, nt_fp = ppl_docs_of(ctx, W)
        assert np.array_equal(nt_fp, ntok), f"{name}: holdout не совпадает с базовым"
        items[name] = (W, W_rows, rec, nll_fp, nt_fp)

    res, nllq = {}, {}
    for name, (W, W_rows, rec0, nll_fp, _nt) in items.items():
        k_fp = float(rec0["kappa"])
        res[name] = {"kappa_fp": k_fp, "corr_f": rec0.get("corr_f"),
                     "anchor_match": rec0.get("anchor_match"),
                     "ppl_fp": ppl_from(nll_fp, ntok),
                     "ppl_fp_total_pct": (ppl_from(nll_fp, ntok) / ppl_base - 1) * 100,
                     "regimes": {}}
        nllq[name] = {"fp": nll_fp}
        for rname, (bits, group) in REGIMES.items():
            tq = time.time()
            # ВЕРБАТИМ E-39f: квантуется ВСЯ tied-матрица [V, D] (у `_quantize_gptq` в
            # докстринге «W: [V, D]»); W_rows — только 128 обученных строк U128, на них
            # результат был бы несопоставим с якорями E-39f и упал бы по форме в ppl_docs_of.
            W_q = _quantize_gptq(W, X, bits, group, DAMP_FRAC, torch.device(DEV))
            k_q = kappa_of(ctx, W_q[ctx["tok_ids"]].clone())
            nll, _ = ppl_docs_of(ctx, W_q)
            nllq[name][rname] = nll
            r = res[name]["regimes"][rname] = {
                "bits": bits, "group": group, "kappa_q": k_q,
                "kappa_rel_shift": k_q / k_fp - 1,
                "ppl_q": ppl_from(nll, ntok),
                "inc_pct": _rel_pct(nll - nll_fp, ntok),
                "inc_ci95": _boot_ci(lambda i, a=nll, b=nll_fp: _rel_pct(a - b, ntok, i), ntok.size),
                "total_pct": _rel_pct(nll - nll_base, ntok),
                "total_ci95": _boot_ci(lambda i, a=nll, b=nll_base: _rel_pct(a - b, ntok, i), ntok.size),
                "gptq_wall_sec": time.time() - tq}
            del W_q
            print(f"  [{name:14s} {rname}] κ {k_fp:.6f}→{k_q:.6f} ({r['kappa_rel_shift']:+.1%})  "
                  f"PPL→{r['ppl_q']:.4f}  inc {r['inc_pct']:+.2f}%  total {r['total_pct']:+.2f}%",
                  flush=True)
        items[name] = None          # [W] носителя больше не нужен: держим только nll (T4, 15 GB)

    floor = _pick_floor(res)
    n = ntok.size
    comparison = {}
    for rname in REGIMES:
        c = {}
        for who, name in (("floor", floor), ("cheat", "cheat")):
            if not name or name not in res:
                continue
            a, b = nllq[name][rname], nllq["base"][rname]
            af, bf = nllq[name]["fp"], nllq["base"]["fp"]
            c[f"cmp_{who}_pct"] = _rel_pct(a - b, ntok)               # после PTQ: носитель vs база
            c[f"cmp_{who}_ci95"] = _boot_ci(lambda i, a=a, b=b: _rel_pct(a - b, ntok, i), n)
            c[f"d_inc_{who}_pp"] = _rel_pct(a - af, ntok) - _rel_pct(b - bf, ntok)
            c[f"d_inc_{who}_ci95"] = _boot_ci(
                lambda i, a=a, af=af, b=b, bf=bf: _rel_pct(a - af, ntok, i) - _rel_pct(b - bf, ntok, i), n)
        if "cmp_floor_pct" in c:
            c["gate_absolute"] = _verdict(c["cmp_floor_ci95"])
            c["gate_incremental"] = _verdict(c["d_inc_floor_ci95"])
            c["gate"] = ("flat_protects" if c["gate_absolute"] == "floor_better"
                         and c["gate_incremental"] == "floor_better" else
                         "flat_worse" if c["gate_absolute"] == "floor_worse"
                         and c["gate_incremental"] == "floor_worse" else "flat_neutral")
            # разбор §5: cheat — не «sanity», а разделитель H_flat и H_any_low_κ
            c["h_any_low_kappa"] = bool("cmp_cheat_ci95" in c
                                        and c["cmp_cheat_ci95"][1] < -GATE_PP
                                        and (res["cheat"]["corr_f"] or 1.0) < CORR_F_HONEST)
            c["discriminates_honest_vs_cheat"] = bool(c["gate_absolute"] == "floor_better"
                                                     and not c["h_any_low_kappa"])
        comparison[rname] = c

    spec_gate = comparison.get("g128_b4", {}).get("gate")
    out = {"experiment": "E-39m flat carrier vs PTQ (фаза quant: tied GPTQ, два режима)",
           "protocol": {
               "regimes": {k: {"bits": v[0], "group": v[1]} for k, v in REGIMES.items()},
               "scope": "вся tied-матрица W_lm (readout + embeddings, как E-39f)",
               "calibration": f"C4 train skip {C4_CAL_SKIP}, n={N_CAL}×{S_CAL}, "
                              "активации базовой модели, одна на все носители",
               "ppl_holdout": f"C4 skip {d39.C4_SKIP}, n={PPL_DOCS}×{d39.S_LEN} (парный bootstrap "
                              f"по документам; шкала E-39d, не сравнима с PPL 30.47 из E-39b/E-39f)",
               "gate_pp": GATE_PP, "gate_rule": "конъюнкция cmp(floor)<−GATE и d_inc<−GATE, обе по "
                                                "верхней границе CI95 (разбор §2-§3)",
               "floor_carrier": floor, "device": DEV},
           "carriers": res, "comparison": comparison,
           "gate": spec_gate,
           "gate_by_regime": {k: v.get("gate") for k, v in comparison.items()},
           "limitations": {
               "seed": "носитель = seed 0; других seed E-39d не прогонял, так что κ'=0.0092 — "
                       "одна реализация, а не медиана (разбор §6)",
               "model": "одна модель 0.8B, tied weights; L23 единственный слой с живым J",
               "ppl_scale": "PPL в шкале E-39d (S=128, skip 4096), не сравнима с E-39b/E-39f",
               "kappa_resolution": "на 32 документах разрешающая способность κ ≈ ±0.0012 (E-39f), "
                                   "поэтому κ здесь — описание носителя, не величина гейта",
           },
           "wall_sec": time.time() - t0}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"gate(g128_b4)={spec_gate} | gate_by_regime={out['gate_by_regime']} "
          f"(floor={floor})\noutput: {out_path} wall {out['wall_sec']:.0f}s", flush=True)
    return out


def _pick_floor(res: dict) -> str | None:
    """Какой из двух order_mode дал честный floor E-39d.

    Разбор §7: если якорь прошли оба, берём ближайший по κ' к 0.009206, а при расхождении
    κ' больше 1e-3 — падает loudly, режим выбора не должен решаться порядком в словаре.
    """
    ok = [n for n in ("floor_pearson", "floor_pairwise")
          if n in res and res[n].get("anchor_match") and res[n]["anchor_match"]["ok"]]
    if not ok:
        return None
    if len(ok) == 2:
        kp = {n: float(res[n]["kappa_fp"]) for n in ok}
        if abs(kp["floor_pearson"] - kp["floor_pairwise"]) > 1e-3:
            raise AssertionError(f"оба order_mode прошли якорь, но κ' расходятся: {kp}")
    return min(ok, key=lambda n: abs(float(res[n]["kappa_fp"]) / LOG_FLOOR["kappa"] - 1))


def main(results_dir: str | None = None, phase: str = "all") -> dict:
    global OUT_DIR
    if results_dir:
        OUT_DIR = results_dir
        os.makedirs(OUT_DIR, exist_ok=True)
    paths = {p: os.path.join(OUT_DIR, f"exp39m_flat_ptq_{p}.json")
             for p in ("smoke", "train", "quant")}
    if phase == "all":
        return phase_all(paths["train"], paths["quant"])
    return {"smoke": phase_smoke, "train": phase_train, "quant": phase_quant}[phase](paths[phase])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "train", "quant", "all"], default="train")
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args()
    main(results_dir=a.results_dir, phase=a.phase)
