"""E94/Q3cpu — CPU-аналог Q3: ОДИН forward на документ (не на позицию).

Зачем: внешнее ревью Q3 (docs/dynamic/2_REVIEW_E94_Q3_exec_r1.md, MAJOR-3) показало, что я ошибся —
CPU-вторая-кандидат-трек стоит 16 форвардов, а не 6144, поэтому обещанное ТЗ сравнение «корреляция
против корреляции» выполнимо сегодня.

Конвенция — ДОСЛОВНО как в exp94_q3.py (иначе сравнение движков невалидно):
  ids448 = prompt(64) + REPLAY.gen_ids[key](384); t = OFF + j метит ГЕНЕРИРУЕМЫЙ токен g_j;
  margin_t = p1 - p2 из softmax(логитов позиции OFF-1+j) = распределение, породившее g_j;
  flag_t  = rep_flags(ids448, 8)[OFF + j] (дословная копия exp81b);
  воспроизведение: p1(OFF-1+j) обязан совпасть с REPLAY.gen_p1[j] (тот же трек, та же конвенция).

Встроенный гейт (не «поверь мне»): если max|Δp1| > REPRO_TOL хотя бы на REPRO_MIN_DOCS документах —
rc=2 и CPU-числа не публикуются. rc: 0 ок, 2 гейт воспроизведения не пройден, 1 ошибка.
Артефакт: results/e94/Q3cpu.json (пер-доковые массивы обязательны).
"""

import json
import os
import sys
import time

# ВАЖНО (найдено этой ночью): torch-сборка на этой коробке не даёт triton выбрать драйвер
# (RuntimeError: 0 active drivers ([])), а `fla` (flash-linear-attention) зовёт triton-ядра.
# Блокируем импорт fla ДО импорта transformers — тогда transformers сам берёт torch-реализацию
# `torch_chunk_gated_delta_rule` (см. transformers/integrations/hub_kernels.py: порядок приоритетов
# hub -> пакет -> torch). Численность этой замены проверяется встроенным гейтом gen_p1.
sys.modules["fla"] = None
os.environ.setdefault("USE_HUB_KERNELS", "NO")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp94_sycl_common import rep_flags, spear, bootstrap_ci_median, san  # noqa: E402

BASE = "/data/Qwen/Qwen3.5-0.8B"
OFF, GEN = 64, 384
LAGS = (1, 4, 8, 16)
REPRO_TOL, REPRO_MIN_DOCS = 1e-3, 3
BOOT_N, BOOT_SEED, N_PERM = 10000, 8181, 2000
PROMPTS = "results/e93/prompts.json"
REPLAY = "results/e81/REPLAY.json"
Q3 = "results/e94/Q3.json"
OUT = "results/e94/Q3cpu.json"


def doc_metrics(key, rec, lg, gp1):
    """Один док: lg = torch-логиты (448, V) fp32, gp1 = замороженный трек p_max."""
    import torch

    ids = list(rec["ids"]) + list(rec["gen"])
    flag = rep_flags(ids, 8)
    pos, m, fl, d_p1 = [], [], [], []
    id_mismatch = 0
    for j in range(GEN):
        row = lg[OFF - 1 + j]
        lse = torch.logsumexp(row, dim=-1)
        top2 = torch.topk(row, k=2).values
        p1 = float(torch.exp(top2[0] - lse))
        p2 = float(torch.exp(top2[1] - lse))
        arg = int(torch.argmax(row))
        pos.append(OFF + j)
        m.append(p1 - p2)
        fl.append(bool(flag[OFF + j]))
        d_p1.append(abs(p1 - float(gp1[j])))
        if arg != int(rec["gen"][j]):
            id_mismatch += 1
    mv, fv = np.array(m), np.array(fl)
    e = {"doc": key, "n_used": int(len(mv)), "margin": [round(x, 8) for x in m],
         "flag": [int(x) for x in fl], "pos": [int(p) for p in pos],
         "flag_full": [int(x) for x in flag],
         "repro_max_abs_d_p1": float(np.max(d_p1)),
         "repro_median_abs_d_p1": float(np.median(d_p1)),
         "repro_pass": bool(np.max(d_p1) < REPRO_TOL),
         "argmax_vs_gen_ids_mismatches": int(id_mismatch),
         "n_flag1": int(fv.sum()), "n_flag0": int(len(fv) - fv.sum())}
    e["spear_a"] = spear(mv, fv)
    b, dropped = {}, {}
    for L in LAGS:
        pair = [(mv[i], flag[p + L]) for i, p in enumerate(pos) if p + L < len(ids)]
        b[L] = spear(np.array([x[0] for x in pair]), np.array([x[1] for x in pair]))
        dropped[L] = int(len(mv) - len(pair))
    e["spear_b"], e["n_dropped_lag"] = {str(k): v for k, v in b.items()}, dropped
    f1, f0 = mv[fv == 1], mv[fv == 0]
    e["med_flag1"] = float(np.median(f1)) if len(f1) else None
    e["med_flag0"] = float(np.median(f0)) if len(f0) else None
    e["delta_median"] = (e["med_flag1"] - e["med_flag0"]) if len(f1) and len(f0) else None
    e["delta_ci_2.5_97.5"] = (list(bootstrap_ci_median(
        f1 - np.median(f0), n=1000, seed=BOOT_SEED)[1]) if len(f1) > 20 and len(f0) > 20 else None)
    return e


def main():
    t0 = time.time()
    R = {"phase": "q3cpu", "engine": "HF fp32 CPU (transformers), model " + BASE,
         "convention": {"ids448": "prompt(64) + REPLAY.gen_ids[key](384)",
                        "margin_t": "p1-p2 из softmax(логитов позиции OFF-1+j), t = OFF+j",
                        "flag_t": "rep_flags(ids448, 8)[OFF+j] — дословно exp81b/exp94_q3",
                        "repro_gate": "p1(OFF-1+j) против REPLAY.gen_p1[j], tol %g на >=%d доках"
                                      % (REPRO_TOL, REPRO_MIN_DOCS)},
         "unit_of_analysis": "document (первично); пул позиций — справочно",
         "per_doc": [], "rc": 1, "halt_reason": None}
    try:
        prompts = json.load(open(PROMPTS))["docs"]
        replay = json.load(open(REPLAY))
        q3 = json.load(open(Q3))
        import torch
        from transformers import AutoModelForCausalLM

        torch.set_num_threads(min(8, os.cpu_count() or 4))
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).eval()
        for rec in prompts:
            key = rec["key"]
            rec = dict(rec)
            rec["gen"] = replay["gen_ids"][key]
            if len(rec["gen"]) != GEN:
                raise RuntimeError("%s: gen_ids len %d != %d" % (key, len(rec["gen"]), GEN))
            with torch.no_grad():
                lg = model(input_ids=torch.tensor([list(rec["ids"]) + list(rec["gen"])]),
                           ).logits[0].float()
            e = doc_metrics(key, rec, lg, replay["gen_p1"][key])
            del lg
            R["per_doc"].append(e)
            R["rc"] = -1  # промежуточная запись: частичный результат переживает обрыв
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(san(R), f, ensure_ascii=False, indent=1)
            print("%s: repro max|dp1|=%.3e pass=%s argmax_mismatch=%d rho_a=%s" % (
                key, e["repro_max_abs_d_p1"], e["repro_pass"],
                e["argmax_vs_gen_ids_mismatches"],
                None if e["spear_a"] is None else round(e["spear_a"], 4)), flush=True)

        n_pass = sum(1 for e in R["per_doc"] if e["repro_pass"])
        R["repro_gate_pass_docs"] = n_pass
        R["repro_max_abs_all"] = float(max(e["repro_max_abs_d_p1"] for e in R["per_doc"]))
        if n_pass < REPRO_MIN_DOCS:
            R["halt_reason"] = "гейт воспроизведения gen_p1: только %d/%d доков с max|dp1| < %g" % (
                n_pass, len(R["per_doc"]), REPRO_TOL)
            R["rc"] = 2
            return R

        rho = [e["spear_a"] for e in R["per_doc"] if e["spear_a"] is not None]
        med, ci = bootstrap_ci_median(rho, n=BOOT_N, seed=BOOT_SEED)
        S = {"d_per_doc_rho": {"median": med, "ci_2.5_97.5": ci, "n_docs": len(rho),
                               "n_positive": int(sum(1 for v in rho if v > 0)),
                               "n_bootstrap": BOOT_N, "seed": BOOT_SEED,
                               "resample_unit": "document"},
             "p_note": "первичная статистика — d_per_doc_rho; пермутационный p по пулу позиций "
                       "оптимистичен из-за внутридоковой автокорреляции"}
        allm = np.concatenate([np.array(e["margin"]) for e in R["per_doc"]])
        allf = np.concatenate([np.array(e["flag"]) for e in R["per_doc"]])
        o, p, nci = __import__("exp94_sycl_common").perm_null(allm, allf, n=N_PERM, seed=BOOT_SEED)
        S["pooled_positions"] = {"rho": o, "p_perm": p, "null_ci": nci, "n_positions": int(len(allm)),
                                 "n_perm": N_PERM, "seed": BOOT_SEED,
                                 "p_definition": "two-sided on |rho|, MC доля без +1",
                                 "status": "справочно, НЕ первичная статистика"}
        lag = {}
        for L in LAGS:
            v = [e["spear_b"][str(L)] for e in R["per_doc"] if e["spear_b"].get(str(L)) is not None]
            lm, lci = bootstrap_ci_median(v, n=BOOT_N, seed=BOOT_SEED)
            lag["L%d" % L] = {"median_per_doc_rho": lm, "ci_2.5_97.5": lci, "n_docs": len(v)}
        S["lags_per_doc"] = lag
        dl = [e["delta_median"] for e in R["per_doc"] if e["delta_median"] is not None]
        dm, dci = bootstrap_ci_median(dl, n=BOOT_N, seed=BOOT_SEED)
        S["delta_median_margin"] = {"median_over_docs": dm, "ci_2.5_97.5": dci, "n_docs": len(dl),
                                    "resample_unit": "document",
                                    "sign_note": "положительно = повторы на боЛЕЕ уверенных позициях"}
        # сравнение движков: корреляции как интервалы (llama-числа берём из Q3.json, не пересчитываем)
        lrho = [x.get("spear_a") for x in q3["per_doc"] if x.get("spear_a") is not None]
        lmed, lci = bootstrap_ci_median(lrho, n=BOOT_N, seed=BOOT_SEED)
        S["cross_engine"] = {"cpu": {"median_per_doc_rho": med, "ci_2.5_97.5": ci},
                             "llama": {"median_per_doc_rho": lmed, "ci_2.5_97.5": lci,
                                       "source": Q3},
                             "per_doc_rho_diff_cpu_minus_llama": {
                                 e["doc"]: round(e["spear_a"] - l, 6)
                                 for e, l in zip(R["per_doc"], lrho)
                                 if e["spear_a"] is not None and l is not None},
                             "reading": "сравниваются корреляции, не позиции; смешения позиций между "
                                        "движками нет по построению (у каждого свой поток)"}
        R["summary"] = S
        R["rc"] = 0
        return R
    except Exception as exc:  # noqa: BLE001
        R["halt_reason"] = "exception: %r" % (exc,)
        R["rc"] = 1
        return R
    finally:
        R["t_total_s"] = round(time.time() - t0, 2)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(san(R), f, ensure_ascii=False, indent=1)
        print("E94 q3cpu rc=%s halt=%s" % (R["rc"], R["halt_reason"]), flush=True)
        sys.exit(R["rc"])


if __name__ == "__main__":
    main()
