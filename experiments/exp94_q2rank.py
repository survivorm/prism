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

Раздел replica_r1_vs_r2 (добавлен 12.11): детерминизм измерен, а не обещан, И заодно измерен
масштаб межпрогонного шума самого отчёта. Это опровергает более раннюю фразу «вероятности между
r1 и r2 различаются ~1e-4»: по факту |Δlogprob| доходит до ~1e-2 и переставляет порядок близких
кандидатов, хотя эмиссия и вершина списка не меняются ни в одной позиции.

Раздел null_perturbation (добавлен 12.11): честная нулевая для «выбор = аргмакс ВОЗМУЩЁННЫХ
логитов» — условная P(rank=k | rank!=1) = p_k / sum_{j>=2} p_j по тому же списку кандидатов.
Независимая проверка того, имеет ли «ранг всегда 2-3» какую-либо мощность против неё.

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
OUT = os.environ.get("E94_Q2RANK_OUT", "results/e94/Q2rank.json")  # переопределяемо: проверочный
# прогон обязан иметь возможность НЕ перезаписывать закоммиченный артефакт
GAP_BINS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.6), (0.6, 1.01)]
R = {"phase": "q2rank", "rc": -1, "halt_reason": None}


def extract(tar_rel):
    """Достаёт архив raw во временный каталог, возвращает путь к корню содержимого."""
    path = os.path.join(REPO, tar_rel)
    tmp = tempfile.mkdtemp(prefix="e94q2rank_")
    with tarfile.open(path) as tf:
        tf.extractall(tmp, filter='data')   # каталоги внутри: e94raw/ или raw/
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
                         # logit_gap = log(p1/p2) = разрыв в единицах ЛОГИТА: чтобы аргмакс
                         # сместился на #2, возмущение логитов обязано его превысить
                         "logit_gap": (math.log(p[0] / p[1]) if len(p) > 1 and p[1] > 0 else None),
                         "n_cand": len(tl),
                         "p_tail": p[1:],          # знаменатель условной нулевой
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
    lg = sorted(r["logit_gap"] for r in fl if r.get("logit_gap") is not None)
    if lg:
        s["flip_logit_gap"] = {
            "median": lg[len(lg) // 2], "max": lg[-1],
            "n_gt_0.1": sum(1 for v in lg if v > 0.1),
            "n_gt_0.4": sum(1 for v in lg if v > 0.4),
            "n_gt_1.0": sum(1 for v in lg if v > 1.0),
            "units": "логит (log p1 - log p2); возмущение меньшего порядка аргмакс не меняет",
        }
    s["null_perturbation"] = null_perturbation(fl)
    if extra:
        s.update(extra)
    return s


def null_perturbation(fl, n_mc=10000, seed=8181):
    """Нулевая «выбор = аргмакс возмущённых логитов» (не epsilon-greedy, а малый сдвиг логитов).
    Условно на то, что выбор НЕ вершина: P(rank=k | rank!=1) = p_k / sum_{j>=2} p_j по тому же
    отчёту. Если наблюдаемое (почти всё на #2, ни одного >=4) согласуется с этой нулевой, то
    «ранг 2-3» против неё мощности не имеет."""
    import numpy as np
    vecs, obs2, obsge4 = [], 0, 0
    for r in fl:
        v = r.get("p_tail") or []
        if not v or sum(v) <= 0:
            continue
        vecs.append(np.asarray(v, dtype=float))
        obs2 += (r["rank"] == 2)
        obsge4 += (r["rank"] != 1 and r["rank"] != 2 and r["rank"] != 3)
    if not vecs:
        return {"status": "нет годных флипов"}
    p2_cond = [float(v[0] / v.sum()) for v in vecs]
    rng = np.random.default_rng(seed)
    cnt2 = np.zeros(n_mc, dtype=np.int32)
    cntge4 = np.zeros(n_mc, dtype=np.int32)
    for v in vecs:
        draw = rng.choice(len(v), size=n_mc, p=v / v.sum()) + 2   # ранг = индекс хвоста + 2
        cnt2 += (draw == 2)
        cntge4 += (draw >= 4)
    return {
        "definition": ("P(rank=k | rank!=1) = p_k / sum_{j>=2} p_j по top_logprobs той же позиции; "
                       "монте-карло n=%d, seed=%d, один рисунок на флип-позицию" % (n_mc, seed)),
        "n_flip_positions_used": len(vecs),
        "mean_P_rank2_given_notop": float(np.mean(p2_cond)),
        "expected_n_at_rank2": float(np.sum(p2_cond)),
        "observed_n_at_rank2": int(obs2),
        "p_obs_at_rank2_ge": float(np.mean(cnt2 >= obs2)),
        "expected_n_rank_ge4": float(np.mean(cntge4)),
        "observed_n_rank_ge4": int(obsge4),
        "p_obs_rank_ge4_le": float(np.mean(cntge4 <= obsge4)),
        "ge4_note": "в счётчик >=4 входят и позиции 'absent' (эмитента нет в списке); в этих данных их 0",
    }


def divergence_alignment(q1, rows1):
    """Совпадает ли ПЕРВЫЙ флип с точкой расхождения потока llama с etalon-потоком.

    prefix_len (из results/e94/Q1.json) = число совпавших первых токенов; значит первое
    расхождение — на индексе prefix_len. Флип = выбор сервера не совпал с вершиной его же
    отчёта. Если первое имеет место ровно там же, где второе, то расхождение движков —
    это и есть флип, а не «f16 против fp32 потихоньку разошлись».
    """
    import numpy as np
    pl = {d["doc"]: d.get("prefix_len") for d in q1["per_doc"]}
    first, cnt = {}, {}
    for r in rows1:
        if not r["tag"].endswith("_r1") or r["rank"] in (1, "no_probs"):
            continue
        doc = r["tag"].split("_")[1]
        cnt[doc] = cnt.get(doc, 0) + 1
        if doc not in first or r["j"] < first[doc]:
            first[doc] = r["j"]
    docs = sorted(pl)
    eq = [d for d in first if pl.get(d) == first[d]]
    div = [d for d in docs if pl.get(d) is not None and pl[d] < 384]
    flip_docs = sorted(first)
    # Бернуллиева модель: единый hazard по всем позициям; сколько раз первый флип упрётся
    # ровно в prefix_len, если флипы «просто случаются» с этой частотой.
    n_pos = sum(1 for r in rows1 if r["tag"].endswith("_r1"))
    n_flips_r1 = sum(1 for r in rows1 if r["tag"].endswith("_r1")
                     and r["rank"] not in (1, "no_probs"))
    haz = n_flips_r1 / float(max(n_pos, 1))
    rng = np.random.default_rng(8181)
    n_mc, hit_mc, nodiv_mc = 20000, 0, 0
    for _ in range(n_mc):
        ok = 0
        for d in div:
            j = int(rng.geometric(haz)) - 1 if haz > 0 else 10**9   # 0-based первый флип
            ok += (j == pl[d])
        hit_mc += (ok == len(div))
        # сколько документов вообще без флипов (сравнить с числом неразшедшихся)
        nodiv_mc += sum(1 for d in docs if rng.geometric(haz) - 1 >= 384) / float(len(docs))
    return {
        "definitions": ("prefix_len = число совпавших первых токенов llama/эталон (Q1.json); "
                        "первое расхождение = индекс prefix_len; первый флип = минимальный j с "
                        "rank != 1 по r1"),
        "per_doc": {d: {"prefix_len": pl.get(d), "first_flip_j": first.get(d),
                        "n_flips_r1": cnt.get(d, 0),
                        "first_flip_eq_divergence": pl.get(d) == first.get(d)} for d in docs},
        "n_docs": len(docs),
        "n_docs_diverging": len(div),
        "n_docs_with_flips": len(flip_docs),
        "diverging_eq_flipdocs": sorted(div) == flip_docs,
        "first_flip_eq_divergence": "%d/%d" % (len(eq), len(first)),
        "hazard_per_position": haz,
        "mc_same_alignment_all_docs": {"n": n_mc, "hits": hit_mc,
                                       "p": hit_mc / float(n_mc),
                                       "note": "BERNOULLI-модель с тем же hazard: первый флип ровно "
                                               "в точке расхождения во ВСЕХ расходящихся документах"},
        "mc_mean_docs_without_flip": nodiv_mc / float(n_mc),
        "observed_docs_without_flip": len([d for d in docs if d not in first]),
        "reading": "",
    }


def raw_recheck(q1root, q1, rows1):
    """Независимая перепроверка divergence_alignment прямо из raw + REPLAY, минуя сводки.

    prefix_len берётся из Q1.json, а raw руки Q3 в git не архивирован — значит единственный
    способ проверить главный вывод раунда заново: восстановить поток llama из response.tokens,
    сравнить с cpu-эталоном (results/e81/REPLAY.json gen_ids) и пересчитать флипы заново.
    Ловушка поля: у top_logprobs[i] есть 'token' (строка) и 'id' (число); сравнивать надо id,
    иначе получается 384/384 «флипов» на каждом документе.
    """
    rp = json.load(open(os.path.join(REPO, "results/e81/REPLAY.json")))["gen_ids"]
    pl_stored = {d["doc"]: d.get("prefix_len") for d in q1["per_doc"]}
    first, cnt = {}, {}
    for r in rows1:
        if not r["tag"].endswith("_r1") or r["rank"] in (1, "no_probs"):
            continue
        doc = r["tag"].split("_")[1]
        cnt[doc] = cnt.get(doc, 0) + 1
        if doc not in first or r["j"] < first[doc]:
            first[doc] = r["j"]
    p_bad, f_bad, n_bad, nof = [], [], [], 0
    for doc, cpu in sorted(rp.items()):
        f = os.path.join(q1root, "q1_%s_r1.json" % doc)
        if not os.path.exists(f):
            nof += 1
            continue
        resp = json.load(open(f))["response"]
        tk, cp = resp["tokens"], resp["completion_probabilities"]
        fp = [j for j, c in enumerate(cp) if c["id"] != c["top_logprobs"][0]["id"]]
        cpu = list(cpu)
        L = min(len(tk), len(cpu))
        pl = L
        for j in range(L):
            if tk[j] != cpu[j]:
                pl = j
                break
        if pl != pl_stored.get(doc):
            p_bad.append([doc, pl, pl_stored.get(doc)])
        if len(fp) != cnt.get(doc, 0):
            n_bad.append([doc, len(fp), cnt.get(doc, 0)])
        if (min(fp) if fp else None) != first.get(doc):
            f_bad.append([doc, min(fp) if fp else None, first.get(doc)])
    return {
        "note": ("пересчитано напрямую: поток = response.tokens, эталон = "
                 "results/e81/REPLAY.json gen_ids, флип = (id != top_logprobs[0].id). Смысл: "
                 "raw руки Q3 в архиве не лежит, поэтому это единственный способ проверить "
                 "главный вывод заново, а не пересказать Q1.json"),
        "n_docs": len(rp) - nof,
        "missing_raw": nof,
        "prefix_len_mismatch": p_bad,
        "n_flips_mismatch": n_bad,
        "first_flip_mismatch": f_bad,
        "clean": not (p_bad or f_bad or n_bad) and nof == 0,
    }


def raw_integrity(ars):
    """Целостность самих архивов сырья: не симлинки ли, не дубликаты ли. Вопрос поднят внешним
    ревью и обязан закрываться числом в артефакте, а не словами."""
    import hashlib
    out = {}
    for ar in ars:
        with tarfile.open(ar) as tf:
            ms = tf.getmembers()
            reg = [m for m in ms if m.isreg()]
            kinds = {"members": len(ms), "regular": len(reg),
                     "symlink": sum(1 for m in ms if m.issym()),
                     "hardlink": sum(1 for m in ms if m.islnk())}
            md5 = {}
            for m in reg:
                if not m.name.endswith(".json"):
                    continue
                b = tf.extractfile(m).read()
                md5.setdefault(hashlib.md5(b).hexdigest(), []).append(m.name)
            dup = {k: v for k, v in md5.items() if len(v) > 1}
            sizes = [m.size for m in reg if m.name.endswith(".json")]
            kinds.update({"n_json": sum(len(v) for v in md5.values()),
                          "distinct_md5": len(md5),
                          "identical_content_groups": len(dup),
                          "identical_examples": {k: v for k, v in list(dup.items())[:3]},
                          "size_min": min(sizes), "size_max": max(sizes)})
        out[os.path.basename(ar)] = kinds
    return out


def replica(cps):
    """r1 против r2 одного и того же сервера: что именно детерминировано, а что нет.
    Сравнивается полезная нагрузка ответов, а не байты файла: wall_s различается по определению."""
    import numpy as np
    docs = sorted({d for d, _ in cps if (d, "r1") in cps and (d, "r2") in cps})
    n_pos = top1_same = ord_diff = set_diff = seq_same = 0
    dshare = []
    for d in docs:
        a, b = cps[(d, "r1")], cps[(d, "r2")]
        seq_same += ([x.get("id") for x in a] == [x.get("id") for x in b])
        for x, y in zip(a, b):
            n_pos += 1
            ix = [t.get("id") for t in (x.get("top_logprobs") or [])]
            iy = [t.get("id") for t in (y.get("top_logprobs") or [])]
            if ix[:1] and ix[:1] == iy[:1]:
                top1_same += 1
            if ix != iy:
                ord_diff += 1
            if set(ix) != set(iy):
                set_diff += 1
            for t, u in zip(x.get("top_logprobs") or [], y.get("top_logprobs") or []):
                if t.get("id") == u.get("id"):
                    dshare.append(abs(t["logprob"] - u["logprob"]))
    q = [float(x) for x in np.percentile(dshare, [50, 90, 99])] if dshare else [0.0, 0.0, 0.0]
    mx = float(max(dshare)) if dshare else 0.0
    return {
        "n_docs_compared": len(docs),
        "n_positions": n_pos,
        "emitted_id_sequence_identical_docs": seq_same,
        "reported_top1_identical_positions": top1_same,
        "candidate_order_differs_positions": ord_diff,
        "candidate_set_differs_positions": set_diff,
        "abs_dlogprob_shared_ids": {"median": q[0], "p90": q[1], "p99": q[2], "max": mx,
                                    "n_compared_ids": len(dshare)},
        "reading": ("эмиссия и вершина отчёта совпали во ДВУХ прогонах (%d/%d позиций); порядок "
                    "близких кандидатов в отчёте — нет (%d позиций из %d), состав списка меняется "
                    "в %d, макс |Δlogprob| = %.2e (медиана %.2e) — это измеренный масштаб шума "
                    "самого сервера, а не 1e-4. Слово «детерминированы» ниже не используется: "
                    "оба прогона шли с одним seed, так что это совпадение двух прогонов, а не "
                    "доказательство детерминизма") % (top1_same, n_pos, ord_diff, n_pos, set_diff,
                                             mx, q[0]),
        "seed_note": ("seed обеих реплик = 8181 (прочитан из request каждого участника архива), "
                      "поэтому r2 — технический повтор, а не независимое наблюдение: "
                      "независимых позиций в Q1 не 12288, а 6144, флипов не 32, а 16"),
    }


def main():
    os.chdir(REPO)
    q2 = json.load(open("results/e94/Q2.json"))
    R["definitions"] = ("rank = индекс id в top_logprobs той же позиции + 1; gap = p1-p2; "
                        "flip = rank != 1; exp(logprob) без перенормировки; "
                        "rank = 'absent' — эмитированного id нет среди присланных кандидатов "
                        "(в этот запуск n_absent = 0 во всех армах, т.к. n_probs = 20 и эмитент "
                        "всегда приходил); rank = 'no_probs' — позиция без отчёта (пустой "
                        "top_logprobs), она НЕ считается флипом и в знаменатель флипов не входит; "
                        "значения 'absent' и 'no_probs' в сравнениях ранга с 1 дают True, поэтому "
                        "все агрегаты отбрасывают их явно (rank not in (1, 'no_probs'))")
    R["sources"] = ["results/e94/raw_q1.json.tar.gz", "results/e94/raw_q2.json.tar.gz",
                    "results/e94/Q2.json"]

    # ---------- Q1: свободная генерация ----------
    root, nfiles = extract("results/e94/raw_q1.json.tar.gz")
    rows1, by_doc, cps = [], {}, {}
    for f in sorted(glob.glob(os.path.join(root, "q1_*.json"))):
        tag = os.path.basename(f)[:-5]
        cp = json.load(open(f))["response"]["completion_probabilities"]
        per_position(cp, tag, rows1)
        doc, run = tag.split("_")[1], tag.split("_")[2]
        cps[(doc, run)] = cp
        fl = sorted(r["j"] for r in rows1 if r["tag"] == tag and r["rank"] not in (1,))
        by_doc.setdefault(doc, {})[run] = fl
    R["replica_r1_vs_r2"] = replica(cps)
    da = divergence_alignment(json.load(open("results/e94/Q1.json")), rows1)
    da["raw_recheck"] = raw_recheck(root, json.load(open("results/e94/Q1.json")), rows1)
    R["divergence_alignment"] = da
    R["divergence_alignment"]["reading"] = (
        "документы с флипами и документы, где поток llama расходится с эталоном, — ОДНИ И ТЕ ЖЕ "
        "(diverging_eq_flipdocs=%s), и первый флип стоит ровно на индексе расхождения в %s "
        "документов; в BERNOULLI-модели с тем же hazard такое выравнивание получилось в "
        "p=%s MC-прогонов (0 из них при p=0.0) — то есть при малом p модель его НЕ воспроизводит, "
        "и совпадение требует механизма, а не считается случайной удачей"
        % (R["divergence_alignment"]["diverging_eq_flipdocs"],
           R["divergence_alignment"]["first_flip_eq_divergence"],
           R["divergence_alignment"]["mc_same_alignment_all_docs"]["p"]))
    R["raw_integrity"] = raw_integrity(["results/e94/raw_q1.json.tar.gz",
                                        "results/e94/raw_q2.json.tar.gz"])
    R["raw_integrity_note"] = ("проверено над закоммиченными архивами напрямую (tarfile), не над "
                              "распаковкой: симлинков/хардлинков нет, побайтово одинаковых файлов "
                              "нет; из 16 пар r1/r2 побайтово не совпадает ни одна (см. "
                              "replica_r1_vs_r2: совпадают только эмиссия и вершина отчёта)")
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
        "claim": ("во всех наблюдаемых флипах эмитированный токен стоит на 2-м или 3-м месте того "
                  "же отчёта сервера; ниже 4-го — ни разу; при gap >= 0.6 флипов нет"),
        "not_claim": [
            "это не доказанный off-by-one: при малом возмущении логитов аргмакс и так уходит "
            "преимущественно на #2 — см. null_perturbation: если она согласуется с наблюдениями, "
            "«ранг 2-3» мощности против неё не имеет",
            "список кандидатов устроен так, что ранг > n_cand наблюдаем быть не может (n_cand <= 20 "
            "по устройству запроса), поэтому «ни одного вне топ-20» — свойство отчёта, а не факт о словаре",
            "граница gap не измеренный порог: между max_gap_flip и 0.6 наблюдений мало",
            "ранг считается по списку top_logprobs, чьё происхождение помечено "
            "provenance-unverified (для верхнего кандидата сверено с HF fp32 до 1e-4, для #2/#3 — нет)",
            "46 флипов (и 16 уникальных в Q1) — мало для оценки формы зависимости",
        ],
        "retracted": [
            "«r1 и r2 дают одинаковые позиции флипа, хотя вероятности различаются ~1e-4 — значит это "
            "не ножевой float-tie»: измерение (replica_r1_vs_r2) даёт max |Δlogprob| ~1e-2 и "
            "перестановку кандидатов в ~5% позиций; сопоставлять надо с этой величиной, а не с 1e-4",
        ],
        "power_test_12_11": {
            "question": "имеет ли «ранг всегда 2-3» мощность против «выбор = аргмакс возмущённых "
                        "логитов» (то есть против скучного объяснения без дефекта)",
            "answer": ("имеет: условная нулевая по тому же отчёту предсказывает 6.5 из 16 на #2 и "
                       "6.9 на рангах >=4, наблюдено 13 и 0; обе хвостовые вероятности < 1e-3 "
                       "(arm_q1_free_r1_only.null_perturbation)"),
            "second_argument": ("разрыв в единицах логита во флипах: медиана 0.65, максимум 0.99, "
                                "10 из 16 выше 0.4 — при измеренном межпрогонном шуме самого "
                                "сервера max |dlogprob| = 9.5e-03 (replica_r1_vs_r2). Случайное "
                                "возмущение такого размера аргмакс не сдвигает"),
            "caveats": [
                "нулевая предполагает НЕКОГЕРЕНТНОЕ возмущение; когерентный сдвиг (систематически "
                "иная позиция/контекст у сэмплера) концентрируется именно на #2 и этим тестом "
                "не отсекается",
                "монте-карло трактует флип-позиции как независимые, а они кластеризованы внутри "
                "дока (в Q1 по одному на док — мягко; в Q2 5 из 14 в w14 и 4 в w4)",
                "список кандидатов обрезан (<=20), поэтому масса за пределами списка вычтена из "
                "знаменателя — это завышает P(rank=2) и делает отказ КОНСЕРВАТИВНЫМ",
                "шум r1-vs-r2 измеряет межпрогонную стохастичность, а не систематическое "
                "расхождение двух внутренних вычислений: последнее может быть больше",
            ],
        },
        "next_probe_with_power": ("logit_bias -0.5 на вершину отчёта в w4 k=70 (разрыв 0.638 логита, "
                                  "после bias 0.138 — вершина остаётся вершиной): «сэмплер = аргмакс» "
                                  "предсказывает 318, «сдвиг в буфере сэмплера» — 310/#3"),
        "correction_12_11": (
            "две правки к этому артефакту по итогам ночи 12-11 (детали: §24, §27 в "
            "docs/dynamic/0_E94_Q2_supervisor_analysis.md); числа ниже не меняются, меняется то, "
            "сколько наблюдений за ними стоит. (1) НЕЗАВИСИМОСТЬ: r1 и r2 делали с одним seed 8181, "
            "поэтому 12288 позиций и 32 флипа Q1 — это 6144 независимых позиции и 16 независимых "
            "флипов; ветка Q2 (2304 разных запроса) независима целиком. Любые точные тесты по Q1 "
            "обязаны идти по r1 (arm_q1_free_r1_only), иначе p-values завышаются в разы порядков. "
            "(2) АДРЕС: «w4 k=70» — адрес ветки Q2 (teacher forcing); в свободной генерации тот же "
            "сайт есть w4 j=6 (тот же топ-5 [318, 310, 326, 369, 783], тот же эмитент 310 против "
            "вершины 318), а w4 j=70 в Q1 — мёртвая позиция (g12 = 8.14, дефицит 0): проба по "
            "нему была бы пустой. Пробу g12=0.638 можно и дешевле делать в форме n_predict=1."),
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
