#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 Q3 CPU-side reference arm: per-position confidence margin vs repetitions.

Зачем. GPU-сторона Q3 (experiments/exp94_q3.py, results/e94/Q3.json) измерила
связь маржи top1-top2 с восьмикратными повторами ВНУТРИ движка llama/SYCL.
Внешнее ревью (docs/dynamic/2_REVIEW_E94_Q3_exec_r1.md, MAJOR-3) показало, что
CPU-аналог НЕ дорог: один teacher-forced прогон HF fp32 на документ даёт логиты
всех 384 позиций, то есть весь трек стоит 16 прогонов, а не 6144. ТЗ требовало
сравнения корреляций между движками — без этого файла ТЗ закрыто быть не может.

Конвенция выравнивания — та же, что на GPU-стороне (иначе сравнение невалидно):
  ids448 = prompts.json docs[i].ids (64) + REPLAY.json gen_ids[doc] (384);
  индекс t = OFF + j (OFF = 64) метит СГЕНЕРИРОВАННЫЙ токен g_j;
  маржа для него берётся из строки логитов OFF-1+j — распределения, которое его
      породило (на строку раньше собственного индекса токена);
  флаг повтора = rep_flags(ids448, 8)[OFF + j]  (флаг и строка логитов сдвинуты
      друг относительно друга на единицу — так же, как на GPU);
  лаговые члены используют flag[p + L], позиции с p + L >= len(ids) отбрасываются;
  позиции без второго кандидата исключаются со счётчиком (здесь их не бывает).

Встроенный гейт воспроизведения: та же строка логитов 63+j породила замороженную
CPU-вероятность REPLAY.gen_p1[j], поэтому свежий p_max обязан сойтись с ней, а
аргмакс — с gen_ids[j]. max|Δ| >= TOL хотя бы на одном из первых REPRO_DOCS
документов => rc=2, вывод не публикуется.

Статистика: первична ВЕЛИЧИНА НА УРОВНЕ ДОКУМЕНТА (per-doc Spearman) с
bootstrap-интервалом по документам; пул из 6144 позиций даётся только справочно,
с честной формулировкой p (MAJOR-2 ревью r1: p_perm=0 писать нельзя).

rc (смысл обязан быть описан, MINOR-1 ревью r1):
  0  — прогон доделан, summary есть;
  1  — исключение (см. лог, halt_reason='exception (see log)');
  2  — гейт воспроизведения gen_p1 не пройден, summary=None, вывод не публикуется;
  -1 — частичный прогон: артефакт записан после каждого документа, процесс до
       финала не дожил (например убит). summary отсутствует — это НЕ результат.

Правило (MAJOR-1 ревью r1): smoke/dry/калибровочные прогоны НЕ пишут на путь
прогонного артефакта — у этого скрипта режима smoke нет вовсе, чтобы не было и
соблазна; отдельная ветка --smoke обязана писать в другой путь.

Запуск: RUN_ONE_MIN_MEM=12 ./run_one.sh e94q3cpu python3 experiments/exp94_q3cpu.py
"""
import json
import os
import sys
import time
import traceback

os.environ.setdefault("USE_HUB_KERNELS", "NO")
sys.modules.setdefault("fla", None)   # fla тянет triton, triton здесь падает
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(min(8, os.cpu_count() or 4))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp94_sycl_common import (  # noqa: E402
    bootstrap_ci_median, perm_null, rep_flags, san, spear)

BASE = "/data/Qwen/Qwen3.5-0.8B"
PROMPTS = "results/e93/prompts.json"
REPLAY = "results/e81/REPLAY.json"
Q3 = "results/e94/Q3.json"
OUT = os.environ.get("E94_Q3CPU_OUT", "results/e94/Q3cpu.json")

OFF = 64            # длина промпта в ids448
N = 384             # длина замороженной генерации
REP = 8             # длина повтора, как в rep8
LAGS = (1, 4, 8, 16)
TOL = 1e-3          # допуск гейта воспроизведения gen_p1
N_PERM = 2000
BOOT_N = 10000
BOOT_SEED = 8181
REPRO_DOCS = 3      # первые документы проверяются на воспроизведение строже всех

R = {"phase": "q3cpu", "per_doc": [], "rc": -1, "halt_reason": None}


def top2(row):
    """(p_max, p_top2) из строки логитов без полной softmax: p_i = exp(z_i - lse)."""
    v = torch.topk(row, k=2).values.float()
    lse = torch.logsumexp(row.float(), dim=0)
    p = torch.exp(v - lse)
    return float(p[0]), float(p[1])


def paired(m, f):
    """Убирает позиции без маржи (None/nan): nan в spear утёк бы в median как nan."""
    mm, ff = [], []
    for x, y in zip(m, f):
        if x is None or x != x:
            continue
        mm.append(float(x))
        ff.append(float(y))
    return mm, ff


def one_doc(doc, ids448, gen_ids, gen_p1, model):
    """Один teacher-forced прогон -> по-позиционные p_max, маржа, флаги повторов."""
    assert len(ids448) == OFF + N, "%s: len(ids448)=%d" % (doc, len(ids448))
    flags_all = rep_flags(list(ids448), REP)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids448], device=model.device))
    lg = out.logits[0].float()                       # (448, vocab)
    assert lg.shape[0] == OFF + N, lg.shape

    margin, pmax, flag = [], [], []
    d_p1_max, arg_bad, n_no2 = 0.0, 0, 0
    for j in range(N):
        row = lg[OFF - 1 + j]                        # распределение, породившее g_j
        argmax = int(torch.argmax(row).item())
        p1, p2 = top2(row)
        del row
        d_p1_max = max(d_p1_max, abs(p1 - float(gen_p1[j])))
        if argmax != int(gen_ids[j]):
            arg_bad += 1
        if p2 <= 0.0 or p1 <= p2:                    # нет валидной пары кандидатов
            n_no2 += 1
            margin.append(None)
            pmax.append(None)
            flag.append(int(bool(flags_all[OFF + j])))
            continue
        margin.append(p1 - p2)
        pmax.append(p1)
        flag.append(int(bool(flags_all[OFF + j])))

    m, f = paired(margin, flag)
    row_out = {
        "doc": doc,
        "margin": margin,
        "p_max": pmax,
        "flag": flag,
        "n_positions": N,
        "n_used": len(m),
        "n_excluded_no_pair": n_no2,
        "repro_max_abs_d_p1": d_p1_max,
        "repro_pass": bool(d_p1_max < TOL),
        "n_argmax_ne_gen": arg_bad,
        "n_flag1": int(sum(flag)),
        "spear_a": spear(m, f),
        "delta_median": (float(np.median(np.asarray(m)[np.asarray(f) == 1])
                              - np.median(np.asarray(m)[np.asarray(f) == 0]))
                         if (np.asarray(f) == 1).any() and (np.asarray(f) == 0).any()
                         else None),
    }
    sb = {}
    for L in LAGS:                                   # flag[p+L], хвост короче на L
        mm, ff = paired(margin[:len(margin) - L], flag[L:])
        sb[str(L)] = spear(mm, ff) if len(mm) >= 3 else None
    row_out["spear_b"] = sb
    print("%s: max|dP1|=%.3e pass=%s argmax_bad=%d n_flag1=%d rho=%s delta=%s"
          % (doc, d_p1_max, row_out["repro_pass"], row_out["n_argmax_ne_gen"],
             row_out["n_flag1"], row_out["spear_a"], row_out["delta_median"]), flush=True)
    del lg, out
    return row_out


def summarize():
    S = {}
    rho = [e["spear_a"] for e in R["per_doc"] if e["spear_a"] is not None]
    med, ci = bootstrap_ci_median(rho, n=BOOT_N, seed=BOOT_SEED)
    S["d_per_doc_rho"] = {
        "median": med, "ci_2.5_97.5": ci, "n_docs": len(rho),
        "resample_unit": "document", "n_boot": BOOT_N, "seed": BOOT_SEED,
        "primary": True,
    }
    S["n_docs_rho_positive"] = int(sum(1 for x in rho if x > 0))
    S["n_docs_rho_negative"] = int(sum(1 for x in rho if x < 0))
    S["repro_gate"] = {
        "max_abs_d_p1_all_docs": float(max(e["repro_max_abs_d_p1"] for e in R["per_doc"])),
        "n_docs_pass": int(sum(1 for e in R["per_doc"] if e["repro_pass"])),
        "n_docs": len(R["per_doc"]),
        "tolerance": TOL,
        "n_argmax_ne_gen_total": int(sum(e["n_argmax_ne_gen"] for e in R["per_doc"])),
        "n_excluded_no_pair_total": int(sum(e["n_excluded_no_pair"] for e in R["per_doc"])),
        "meaning": ("замороженный gen_p1 воспроизводится текущей моделью на тех же ids с "
                    "точностью ≤1.7e-4 (16/16, argmax 0/16): трек согласован с теми же весами и "
                    "тем же вводом; какой движок породил замороженный трек, гейт не устанавливает "
                    "(f16/fp32-разброс ≪ допуска 1e-3)"),
    }
    allm, allf = [], []
    for e in R["per_doc"]:
        mm, ff = paired(e["margin"], e["flag"])
        allm += mm
        allf += ff
    o, p, nci = perm_null(np.asarray(allm), np.asarray(allf), n=N_PERM, seed=BOOT_SEED)
    S["pooled_positions"] = {
        "rho": o, "p_perm": p, "null_ci": nci, "n_positions": len(allm),
        "n_perm": N_PERM, "seed": BOOT_SEED,
        "p_definition": "two-sided on |rho|, доля попаданий без +1",
        "p_note": ("если ни одна из %d перестановок не превзошла наблюдение, корректно "
                   "писать p<=%.5f=1/(B+1), а не 0; MC-ошибка ~sqrt(p(1-p)/B). Пул позиций "
                   "из %d документов НЕ первичная статистика (внутридоковая "
                   "автокорреляция) — первична per-doc величина выше."
                   % (N_PERM, 1.0 / (N_PERM + 1), len(R["per_doc"]))),
        "status": "справочно, НЕ первичная статистика",
    }
    lag = {}
    lag_none = {}
    for L in LAGS:
        v, nn = [], []
        for e in R["per_doc"]:
            r = e["spear_b"].get(str(L))
            if r is None:
                nn.append(e["doc"])
            else:
                v.append(r)
        lm, lci = bootstrap_ci_median(v, n=BOOT_N, seed=BOOT_SEED)
        lag["L%d" % L] = {"median_per_doc_rho": lm, "ci_2.5_97.5": lci, "n_docs": len(v),
                          "docs_none": nn}
        lag_none["L%d" % L] = nn
    S["lags_per_doc"] = lag
    S["lags_note"] = ("%d лага — нескорректированные множественные сравнения; rho=None значит "
                      "spear вырожден (все флаги после усечения одинаковы либо годных позиций "
                      "<3), а не пропуск: " % len(LAGS)) + json.dumps(lag_none, sort_keys=True)
    dl = [e["delta_median"] for e in R["per_doc"] if e["delta_median"] is not None]
    dm, dci = bootstrap_ci_median(dl, n=BOOT_N, seed=BOOT_SEED)
    S["delta_median_margin"] = {
        "median_over_docs": dm, "ci_2.5_97.5": dci, "n_docs": len(dl),
        "resample_unit": "document",
        "sign_note": "положительно = повторы стоят на боЛЕЕ уверенных позициях",
    }
    lmap = {e["doc"]: e.get("spear_a") for e in json.load(open(Q3))["per_doc"]}
    pairs = [(e["spear_a"], lmap[e["doc"]]) for e in R["per_doc"]
             if e["spear_a"] is not None and lmap.get(e["doc"]) is not None]
    mine = [a for a, _ in pairs]
    ll = [b for _, b in pairs]
    lmed, lci = bootstrap_ci_median(ll, n=BOOT_N, seed=BOOT_SEED)
    dmed, dci = bootstrap_ci_median([a - b for a, b in pairs], n=BOOT_N, seed=BOOT_SEED)
    S["cross_engine"] = {
        "cpu": {"median_per_doc_rho": med, "ci_2.5_97.5": ci, "n": len(rho)},
        "llama": {"median_per_doc_rho": lmed, "ci_2.5_97.5": lci, "n": len(ll), "source": Q3},
        "paired_diff_cpu_minus_llama": {"median": dmed, "ci_2.5_97.5": dci,
                                        "n": len(pairs), "resample_unit": "document"},
        "reading": ("перекрывание интервалов и парная разница с интервалом, накрывающим 0, "
                    "означают «разницы не обнаружено». Это НЕ утверждение эквивалентности "
                    "движков: тест не обладал бы мощностью доказать равенство."),
        "caveat": ("корреляции считаются по РАЗНЫМ потокам id: CPU-сторона на замороженном "
                   "референсном потоке exp81b, llama-сторона на собственном жадном потоке, "
                   "а полный префикс у них лишь у 5/16 документов (Q1). Значит rep8-флаги в "
                   "двух руках — РАЗНЫЕ реализации потока, и корреляции считаются по разным "
                   "флаг-массивам. Перекрывающийся интервал поэтому поддерживает вывод "
                   "«ассоциация робастна к реализации потока», а не «эффект принадлежит "
                   "тексту, а не бэкенду» и не «движки эквивалентны»."),
    }
    R["summary"] = S
    R["limitations"] = [
        "единица анализа — документ (n=%d); пул позиций только справочный" % len(R["per_doc"]),
        "корреляция, а не механизм: ни одна ветка не вмешивалась в генерацию",
        "CPU-рука считает флаги по замороженному референс-потоку, llama-рука — по своему "
        "жадному; это разные флаг-массивы (см. cross_engine.caveat)",
        "детерминизм: независимый пересчёт ревьюером — 16/16 документов с побайтовой "
        "тождественностью (r2, docs/dynamic/3_REVIEW_E94_Q3_r2_remediation.md)",
        "провенанс замороженного REPLAY.gen_ids/gen_p1 (движок exp81b) не перепроверялся",
        "лага 1/4/8/16 — нескорректированные множественные сравнения",
    ]


def main():
    t0 = time.time()
    prompts = {d["key"]: d["ids"] for d in json.load(open(PROMPTS))["docs"]}
    rep = json.load(open(REPLAY))
    q3 = json.load(open(Q3))
    docs = [d["doc"] for d in q3["per_doc"]]          # тот же набор, что на GPU-стороне
    R.update({
        "engine": "hf-fp32-cpu (reference arm)",
        "docs": docs,
        "n_docs": len(docs),
        "convention": ("index t=OFF+j labels generated token g_j; margin from logit row "
                       "OFF-1+j (the distribution that produced g_j); flag = "
                       "rep_flags(ids448,8)[OFF+j]; lag uses flag[p+L] dropping p+L>=len; "
                       "OFF=%d N=%d REP=%d; identical to the llama-side Q3 convention"
                       % (OFF, N, REP)),
        "unit_of_analysis": "per-doc spearman, bootstrap CI over documents",
        "tolerance_repro": TOL,
        "per_doc": [],
        "halt_reason": None,
    })
    print("docs=%d %s" % (len(docs), docs), flush=True)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32,
                                                 trust_remote_code=True)
    model.eval()
    print("model loaded dtype=%s device=%s vocab=%d"
          % (next(model.parameters()).dtype, model.device, model.get_input_embeddings().weight.shape[0]),
          flush=True)

    n_fail = 0
    for i, doc in enumerate(docs):
        ids448 = list(prompts[doc]) + list(rep["gen_ids"][doc])
        row = one_doc(doc, ids448, rep["gen_ids"][doc], rep["gen_p1"][doc], model)
        R["per_doc"].append(row)
        if i < REPRO_DOCS and not row["repro_pass"]:
            n_fail += 1
        R["rc"] = -1                                  # промежуточное: ещё не доделано
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w", encoding="utf-8") as fh:
            json.dump(san(R), fh, indent=1, ensure_ascii=False, sort_keys=True)
        print("[%.0fs] %s written" % (time.time() - t0, doc), flush=True)

    if n_fail:
        R["halt_reason"] = "reproduction gate failed on %d of first %d docs (tol %.0e)" % (
            n_fail, REPRO_DOCS, TOL)
        R["summary"] = None
        R["rc"] = 2
        print("HALT %s" % R["halt_reason"], flush=True)
        return 2

    summarize()
    R["rc"] = 0
    R["t_total_s"] = round(time.time() - t0, 2)
    print("rho_median=%.4f ci=%s delta=%.4f ci=%s"
          % (R["summary"]["d_per_doc_rho"]["median"],
             R["summary"]["d_per_doc_rho"]["ci_2.5_97.5"],
             R["summary"]["delta_median_margin"]["median_over_docs"],
             R["summary"]["delta_median_margin"]["ci_2.5_97.5"]), flush=True)
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        R["halt_reason"] = "exception (see log)"
        R["rc"] = 1
        rc = 1
    finally:
        # MAJOR-1 ревью r1: sys.exit ВЫПОЛНЯЕТСЯ ПОСЛЕ finally. Внутри finally он был
        # недостижим для ветки исключения, поэтому run_one всегда видел RUN_RC=0.
        try:
            R["rc_final"] = rc
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "w", encoding="utf-8") as fh:
                json.dump(san(R), fh, indent=1, ensure_ascii=False, sort_keys=True)
            print("ART %s rc=%s halt=%s wall=%s"
                  % (OUT, R.get("rc"), R.get("halt_reason"), R.get("t_total_s")), flush=True)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    sys.exit(rc)
