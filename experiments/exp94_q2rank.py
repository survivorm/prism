#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 Q2-rank: переанализ ЗАКОММИЧЕННЫХ raw — ранг эмитированного токена в собственном
отчёте сервера (emitted vs его же top_logprobs), плюс срез частоты флипов по разрыву p1-p2.

Это НЕ новое измерение: новых запросов к серверу нет, GPU не нужен. Источник —
results/e94/raw_q1.json.tar.gz (свободная генерация, 16 доков × r1/r2 × 384 позиции) и
results/e94/raw_q2.json.tar.gz (teacher forcing, 6 доков × 384) + results/e94/Q2.json
(чтобы сверить число флипов с прогонным артефактом).

Определения (обязаны быть записаны, иначе два человека разойдутся):
  rank(emitted) = индекс id в top_logprobs ТОЙ ЖЕ позиции + 1; если id нет в списке — 'absent';
  gap = p1 - p2 того же списка, p_i = exp(logprob_i);
  flip = rank != 1;
  Q1-уникальные = только r1 (r1 и r2 дают одинаковые позиции флипов; проверяется, а не
  предполагается: n_r1_eq_r2 должно равняться числу доков).

Что утверждает анализ (см. docs/dynamic/0_E94_Q2_supervisor_analysis.md §11):
  во всех наблюдаемых флипах rank ∈ {2,3}; при gap >= GAP_BOUND позиций много и флипов ноль.
GAP_BOUND здесь НЕ порог, а строка таблицы; вывод о пороге — «между max_gap_flip и 0.6».

Запуск: ./run_one.sh e94q2rank python3 experiments/exp94_q2rank.py
"""
import glob
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = "results/e94/Q2rank.json"
GAP_BINS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.6), (0.6, 1.01)]
R = {"phase": "q2rank", "rc": -1, "halt_reason": None}


def extract(tar_rel):
    """Достаёт архив raw во временный каталог, возвращает путь к корню содержимого."""
    path = os.path.join(REPO, tar_rel)
    tmp = tempfile.mkdtemp(prefix="e94q2rank_")
    with tarfile.open(path) as tf:
        tf.extractall(tmp)          # архив внутри содержит каталог e94raw/ или raw/
    hits = glob.glob(os.path.join(tmp, "**", "*.json"), recursive=True)
    if not hits:
        raise RuntimeError("no json inside %s" % tar_rel)
    return os.path.dirname(hits[0]), len(hits)


def per_position(cp, tag, out_rows):
    """Один поток completion_probabilities -> строки (позиция, ранг, gap)."""
    for j, c in enumerate(cp):
        tl = c.get("top_logprobs") or []
        if not tl:
            out_rows.append({"tag": tag, "j": j, "rank": "no_probs", "gap": None, "p1": None})
            continue
        ids = [t.get("id") for t in tl]
        em = c.get("id")
        rank = ids.index(em) + 1 if em in ids else "absent"
        p = [math.exp(t["logprob"]) for t in tl]
        out_rows.append({"tag": tag, "j": j, "rank": rank,
                         "gap": (p[0] - p[1]) if len(p) > 1 else 0.0,
                         "p1": p[0], "emitted": em, "top": ids[0],
                         "p_emitted": math.exp(c["logprob"])})


def bin_table(rows):
    t = {}
    for lo, hi in GAP_BINS:
        sel = [r for r in rows if r["gap"] is not None and lo <= r["gap"] < hi]
        fl = [r for r in sel if r["rank"] not in (1,)]
        t["%.2f-%.2f" % (lo, hi)] = {"n": len(sel), "n_flip": len(fl),
                                     "rate": (len(fl) / len(sel)) if sel else None}
    return t


def summarize(rows, arm, extra=None):
    fl = [r for r in rows if r["rank"] not in (1, "no_probs")]
    ranks = {}
    for r in fl:
        ranks[str(r["rank"])] = ranks.get(str(r["rank"]), 0) + 1
    gaps = [r["gap"] for r in fl if r["gap"] is not None]
    s = {
        "arm": arm,
        "n_positions": len(rows),
        "n_flip": len(fl),
        "rank_hist": ranks,
        "n_rank_gt3": sum(v for k, v in ranks.items() if k.isdigit() and int(k) > 3),
        "n_absent_from_top20": ranks.get("absent", 0),
        "max_gap_among_flips": max(gaps) if gaps else None,
        "max_p1_among_flips": max((r["p1"] for r in fl), default=None),
        "gap_bins": bin_table(rows),
        "flip_positions": sorted("%s/%d" % (r["tag"], r["j"]) for r in fl),
    }
    hi = [r for r in rows if r["gap"] is not None and r["gap"] >= 0.6]
    s["gap_ge_0.6"] = {"n": len(hi), "n_flip": sum(1 for r in hi if r["rank"] != 1)}
    if extra:
        s.update(extra)
    return s


def main():
    os.chdir(REPO)
    q2 = json.load(open("results/e94/Q2.json"))
    R["definitions"] = ("rank = индекс id в top_logprobs той же позиции + 1; gap = p1-p2; "
                        "flip = rank != 1; exp(logprob) без перенормировки")
    R["sources"] = ["results/e94/raw_q1.json.tar.gz", "results/e94/raw_q2.json.tar.gz",
                    "results/e94/Q2.json"]

    # ---------- Q1: свободная генерация ----------
    root, nfiles = extract("results/e94/raw_q1.json.tar.gz")
    rows1, by_doc = [], {}
    for f in sorted(glob.glob(os.path.join(root, "q1_*.json"))):
        tag = os.path.basename(f)[:-5]
        cp = json.load(open(f))["response"]["completion_probabilities"]
        per_position(cp, tag, rows1)
        doc, run = tag.split("_")[1], tag.split("_")[2]
        fl = sorted(r["j"] for r in rows1 if r["tag"] == tag and r["rank"] not in (1,))
        by_doc.setdefault(doc, {})[run] = fl
    n_r1 = sum(1 for d, v in by_doc.items() if v.get("r1") == v.get("r2"))
    rows1_r1 = [r for r in rows1 if r["tag"].endswith("_r1")]
    R["arm_q1_free"] = summarize(rows1, "llama-server/SYCL, свободная генерация r1+r2", {
        "n_files": nfiles,
        "r1_r2_same_flip_positions_docs": n_r1,
        "n_docs": len(by_doc),
        "unique_positions_note": "r1 и r2 — один и тот же промпт дважды; уникальные позиции = r1",
    })
    R["arm_q1_free_r1_only"] = summarize(rows1_r1, "только r1 (уникальные позиции)")

    # ---------- Q2: teacher forcing ----------
    root2, nfiles2 = extract("results/e94/raw_q2.json.tar.gz")
    rows2 = []
    idx = {os.path.basename(f): f for f in glob.glob(os.path.join(root2, "q2_*.json"))}
    missing = []
    for doc, rr in q2["per_position"].items():
        for r in rr:
            name = "q2_%s_k%d.json" % (doc, r["k"])
            f = idx.get(name)
            if f is None:
                missing.append(name)
                continue
            cp = json.load(open(f))["response"]["completion_probabilities"]
            per_position(cp, doc + "_k%d" % r["k"], rows2)
    agree = {"n_flip_artifact": sum(1 for doc, rr in q2["per_position"].items()
                                    for r in rr if not r["top1_agree"]),
             "n_flip_recomputed": sum(1 for r in rows2 if r["rank"] not in (1, "no_probs")),
             "top1_agreement_artifact": q2["summary"]["overall"]["top1_agreement"]}
    R["arm_q2_forced"] = summarize(rows2, "llama-server/SYCL, teacher forcing", {
        "n_files": nfiles2, "missing_raw": missing[:5], "n_missing_raw": len(missing),
        "cross_check_with_artifact": agree,
    })
    if missing:
        R["halt_reason"] = "raw missing for %d probes (see arm_q2_forced.missing_raw)" % len(missing)
        R["rc"] = 2
        return 2

    R["pooled"] = {
        "n_positions": R["arm_q1_free"]["n_positions"] + R["arm_q2_forced"]["n_positions"],
        "n_flip": R["arm_q1_free"]["n_flip"] + R["arm_q2_forced"]["n_flip"],
        "n_rank_gt3": R["arm_q1_free"]["n_rank_gt3"] + R["arm_q2_forced"]["n_rank_gt3"],
        "n_absent_from_top20": (R["arm_q1_free"]["n_absent_from_top20"]
                                + R["arm_q2_forced"]["n_absent_from_top20"]),
        "gap_ge_0.6": {"n": (R["arm_q1_free"]["gap_ge_0.6"]["n"]
                             + R["arm_q2_forced"]["gap_ge_0.6"]["n"]),
                       "n_flip": (R["arm_q1_free"]["gap_ge_0.6"]["n_flip"]
                                  + R["arm_q2_forced"]["gap_ge_0.6"]["n_flip"])},
    }
    R["interpretation"] = {
        "claim": ("всех наблюдаемых флипах эмитированный токен стоит на 2-м или 3-м месте того же "
                  "отчёта сервера; ниже 4-го и вне топ-20 — ни разу; при gap >= 0.6 флипов нет"),
        "not_claim": [
            "это не доказанный off-by-one: при малом возмущении логитов аргмакс и так уходит "
            "преимущественно на #2, поэтому «ранг <= 3» — довод умеренной мощности",
            "граница gap не измеренный порог: между max_gap_flip и 0.6 наблюдений мало",
            "ранг считается по списку top_logprobs, чьё происхождение помечено "
            "provenance-unverified (для верхнего кандидата сверено с HF fp32 до 1e-4, для #2/#3 — нет)",
            "46 флипов — мало для оценки формы зависимости",
        ],
        "next_probe_with_power": ("logit_bias -0.5 на вершину отчёта в w4 k=70 (разрыв 0.638 логита, "
                                  "после bias 0.138 — вершина остаётся вершиной): «сэмплер = аргмакс» "
                                  "предсказывает 318, «сдвиг в буфере сэмплера» — 310/#3"),
    }
    R["rc"] = 0
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
        try:
            sys.path.insert(0, os.path.join(REPO, "experiments"))
            from exp94_sycl_common import san
            R["rc_final"] = rc
            os.makedirs(os.path.dirname(os.path.join(REPO, OUT)), exist_ok=True)
            with open(os.path.join(REPO, OUT), "w", encoding="utf-8") as fh:
                json.dump(san(R), fh, indent=1, ensure_ascii=False, sort_keys=True)
            print("ART %s rc=%s halt=%s" % (OUT, R.get("rc"), R.get("halt_reason")), flush=True)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
    sys.exit(rc)
