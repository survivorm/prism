#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 Q2 (ТЗ §2.2 + steering r3): teacher-forced id-probe, iGPU/SYCL vs замороженный CPU-референс.

Последовательность: ids448 = prompts.json.docs[i].ids (64) + REPLAY.gen_ids["w<i>"] (384).
Префиксы режутся ТОЛЬКО от 448-id последовательности (не от 64-id промпта — на этом
умер прошлый спайк: те=64 для всех k). Для k = 64..447: POST /completion
prompt=ids448[:k], n_predict=1, канон из body_template артефакта Q1.json
(cache_prompt: TRUE, top_logprobs:20/n_probs:20, прореживание снято, seed 8181,
temp 0, stream false, return_tokens true).

Конвенция позиции (замороженная, exp81b gen_one: nxt=argmax(logits[0,-1]),
p1=softmax(logits[0,-1])[nxt]): на позиции k = 64+j CPU-следующий id = gen_ids[j]
(0-based; «gen_ids[j+1]» из задачи читается как (j+1)-й токен, j=k-64), и
gen_p1[j] = p_max ПОСЛЕ prompt+gen[:j]. Свежий HF-прогон сегодня НЕ делается
(отклонение в NOTES); референс — замороженный REPLAY.gen_p1 (e81b replay-гейт, corr>0.99).

Метрики на позицию: p_max_gpu = exp(completion_probabilities[0].logprob);
top1_gpu = cp[0].id; rank_cpu — 1-based позиция gen_ids[j] в top_logprobs (20),
None = цензура (rank>20); d_pmax = p_max_gpu - gen_p1[j].
Сводки: медиана |Δp_max| (+bootstrap 10000, seed 8181), p90/p99, доля |Δ|>0.01,
top1_agreement, top20_hit_rate, медиана rank_cpu по хитам; per_position в
results/e94/Q2.json — пересчёт без перезапуска.

ГЕЙТЫ (спайк + прогон):
  G1  tokens_evaluated == k на КАЖДОМ запросе → иначе rc=13.
  G2  кэш (recalibrated по steering 12.11): w0, порядок [64,64,128,128,192,192]
      (пары по k; первый запрос каждого k холодный по построению, второй —
      гейтный); гейт — timings.prompt_n <= 8 со ВТОРОГО запроса каждого k
      (измерено: cache_n = ровно k-4, prompt_n=4 на повторе — кэш РАБОТАЕТ,
      старый порог ТЗ cache_n>=k-2 строг); cache_n — наблюдение + клиентский wall-clock на позицию не растёт с k (≤1.5x).
      tokens_cached и /slots — мёртвые метрики, не используются.
      Провал → НЕ чинить: halt_reason + значения по k в Q2.json, rc=2,
      NOTES предлагает скоуп 2 дока вместо 6.
      Основная фаза (МИНOR Q2, мягкий — объявленное отклонение): со второго
      запроса каждого документа (k=65) prompt_n <= 8 и плоский wall по окнам 32
      (медиана последних 32 / первых 32 <= 1.5); нарушения НЕ убивают прогон —
      в артефакт g2_violations ({doc,k,prompt_n}) + g2_violations_n; г2_
      offline_check — пересчёт из сырья (одна процедура для live и from-raw).
Запуск: python3 -u experiments/exp94_q2.py [--from-raw] (--from-raw: пересборка
из /tmp/e94raw без сервера, поля прогона сохраняются из Q2.json, идемпотентность ×2).
  G3  fd-гейт устройства во время decode (20 сэмплов × 0.05 с, паттерн exp94_q1).
  Device-гейт: /dev/dri в /proc/<pid>/fd после health → иначе rc=10.

RUN_RC: 0 ok · 2 spike/cache · 10 device-гейт · 11 ≥2 упавших дока ·
13 prompt-count · 1 иное (вкл. server died, бюджет 40 мин).
Артефакт results/e94/Q2.json пишется при ЛЮБОМ исходе (halt_reason). sys.exit(rc) обязателен.
"""
import json, math, os, sys, threading, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp94_sycl_common import (LlamaServer, pick_port, san, check_schema,
                               check_prompt_count, bootstrap_ci_median)
from exp94_sycl_stream import save_raw, fd_sampler
import numpy as np

DOCS = ["w0", "w2", "w4", "w9", "w13", "w14"]   # предобъявленный скоуп (rep8 0.26–0.97, вкл. w4)
PORTS = [8110]
RAW = "/tmp/e94raw"
OUT = "results/e94"
PROMPTS = "results/e93/prompts.json"
REPLAY = "results/e81/REPLAY.json"
Q1ART = "results/e94/Q1.json"
SERVER_LOG = "/tmp/e94q2_server.log"
KS = range(64, 448)                              # k = 64..447 (k=448 не шлём)
SPIKE_ORDER = [64, 64, 128, 128, 192, 192]       # G2: три k по парам; r2 каждого k — гейт
FD_SAMPLES, FD_INTERVAL = 20, 0.05
BUDGET_S = 40 * 60
CANON_FIELDS = ("temperature", "stream", "return_tokens", "logprobs",
                "top_logprobs", "seed", "top_k", "top_p", "min_p", "typical_p",
                "xtc_probability")


def canon_sha(obj):
    """канонический sha256 (sort_keys, компактные separators) — идемпотентность."""
    import hashlib
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_artifact(R):
    """san(R) → results/e94/Q2.json (пишется и при STOP)."""
    R = san(R)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "Q2.json"), "w") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    return R


def finish(R, rc, halt, t0):
    """R["rc"]/R["halt_reason"]/t_total → артефакт → RUN_RC → rc."""
    R["rc"], R["halt_reason"] = rc, halt
    R["t_total_s"] = round(time.time() - t0, 2)
    write_artifact(R)
    print("E94 q2 rc=%d%s" % (rc, (" halt=" + halt) if halt else ""), flush=True)
    print("RUN_RC=%d" % rc, flush=True)
    return rc


def load_data():
    """{doc: {"ids448": [448], "gen_ids": [384], "gen_p1": [384]}}; assert 64+384.
    ids448 = prompts.json.docs[i].ids (МАССИВ объектов) + REPLAY.gen_ids["w<i>"]
    (ВЕРХНИЙ dict, не docs.w*)."""
    p = json.load(open(PROMPTS))
    pdocs = {d["key"]: list(d["ids"]) for d in p["docs"]}
    r = json.load(open(REPLAY))
    out = {}
    for k_ in DOCS:
        i64, g, p1 = pdocs[k_], list(r["gen_ids"][k_]), [float(x) for x in r["gen_p1"][k_]]
        if not (len(i64) == 64 and len(g) == 384 and len(p1) == 384):
            raise RuntimeError("%s: длины %d/%d/%d (ждём 64/384/384)" % (k_, len(i64), len(g), len(p1)))
        out[k_] = {"ids448": i64 + g, "gen_ids": g, "gen_p1": p1}
    return out


def build_body(template, prompt):
    """канон = body_template(Q1.json) с prompt, n_predict=1, cache_prompt=True."""
    b = dict(template)
    b["prompt"] = list(prompt)
    b["n_predict"] = 1
    b["cache_prompt"] = True
    return b


def get_raw(srv, path, body, fd_first=None):
    """идемпотентный запрос: существующий raw → читаем; иначе POST + save_raw;
    fd_first: один раз, при первом реальном (не кэш-хите) запросе — fd_sampler
    в потоке (G3: 20×0.05 с во время decode). Если первый запрос — кэш-хит сырья,
    сэмплер не стартует: main_run фиксирует это как device_gate_source="reused_raw"."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["response"], True
    if fd_first is not None and "th" not in fd_first:
        fd_first["samples"] = []
        fd_first["th"] = threading.Thread(target=fd_sampler, args=(srv.pid, FD_SAMPLES,
                                                                   FD_INTERVAL,
                                                                   fd_first["samples"]))
        fd_first["th"].start()
    try:
        resp, dt = srv.completion(body, timeout=600)
    finally:
        if fd_first is not None and "th" in fd_first:
            fd_first["th"].join()
            fd_first["n_with_dri"] = sum(1 for s in fd_first["samples"] if s)
            fd_first["dri_union"] = sorted({x for s in fd_first["samples"] for x in s})
    save_raw(os.path.basename(path), {"request": body, "response": resp,
                                     "wall_s": round(dt, 3)})
    return resp, False


def spike_cache(srv, ids448, tpl):
    """G2 на w0: порядок SPIKE_ORDER (второй запрос каждого k — гейтный; первый
    холодный по построению). Гейты: G1 на каждом (te==k); на r2 каждого k — prompt_n <= 8 (recalibrated:
    форк отдаёт cache_n = ровно k-4 при работающем кэше — старый порог строг);
    cache_n пишется как наблюдение (tokens_cached//slots — мёртвые);
    wall на позицию (wall_s/k) не растёт с k (≤1.5x первого r2). → (ok, rows, fails)."""
    rows, fails = [], []
    for i, k in enumerate(SPIKE_ORDER):
        try:
            resp, dt = srv.completion(build_body(tpl, ids448[:k]), timeout=300)
        except Exception as e:
            return False, rows, ["k=%d req%d: request exception: %s" % (k, i + 1, e)]
        t = resp.get("timings") or {}
        rows.append({"k": k, "req": i + 1, "r2": (i + 1) % 2 == 0,
                     "tokens_evaluated": resp.get("tokens_evaluated"),
                     "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
                     "prompt_ms": t.get("prompt_ms"),
                     "wall_s": round(dt, 3), "wall_per_pos": round(dt / k, 6)})
        if resp.get("tokens_evaluated") != k:  # G1 — на каждом запросе
            fails.append("G1 k=%d req%d: tokens_evaluated=%r != %d" %
                         (k, i + 1, resp.get("tokens_evaluated"), k))
    r2 = [r for r in rows if r["r2"]]
    for r in r2:
        if r["prompt_n"] is None or r["prompt_n"] > 8:
            fails.append("G2 k=%d: prompt_n=%r > 8 (префикс не переиспользован); cache_n(obs)=%r"
                         % (r["k"], r["prompt_n"], r["cache_n"]))
    if r2:
        w0 = r2[0]["wall_per_pos"]
        for r in r2[1:]:
            if w0 > 0 and r["wall_per_pos"] > 1.5 * w0:
                fails.append("G2 wall: wall_per_pos(k=%d)=%.5f > 1.5*first(%.5f) — "
                             "растёт с k" % (r["k"], r["wall_per_pos"], w0))
    return (len(fails) == 0), rows, fails


def position_metrics(resp, k, gen_ids, gen_p1):
    """на позицию k (после ids448[:k]): cp[0] — distribution. CPU-следующий id =
    gen_ids[k-64] (0-based). rank_cpu — 1-based позиция в top_logprobs(20),
    None = цензура (rank>20). → (m, None) или (None, err)."""
    cp = resp.get("completion_probabilities")
    if not isinstance(cp, list) or not cp:
        return None, "no completion_probabilities"
    p0 = cp[0]
    if not isinstance(p0, dict) or not isinstance(p0.get("logprob"), (int, float)):
        return None, "cp[0] keys=%s (logprob не число)" % list(p0 or {})
    j = k - 64
    cpu_id = int(gen_ids[j])
    p_max = math.exp(p0["logprob"]) if p0["logprob"] > -745.0 else 0.0
    tl = p0.get("top_logprobs")
    ids20 = [t.get("id") for t in tl] if isinstance(tl, list) else []
    rank = (ids20.index(cpu_id) + 1) if cpu_id in ids20 else None
    return {"k": k, "j": j, "p_max_gpu": p_max, "top1_gpu": p0.get("id"),
            "p_ref": gen_p1[j], "cpu_id": cpu_id, "d_pmax": p_max - gen_p1[j],
            "rank_cpu": rank, "top1_agree": bool(p0.get("id") == cpu_id)}, None


def main_run(srv, R, data, tpl, t0):
    """6 доков × k=64..447; raw /tmp/e94raw/q2_<doc>_k<k>.json (идемпотентно);
    halt-каскад: G1 (13), server died, бюджет 40 мин. G2 основной фазы — мягкий
    (см. rebuild_from_raw; сюда не входит, чтобы live и from-raw считали
    одинаково). device_gate_source: "live" — fd-сэмплер отработал,
    "reused_raw" — первый запрос из кэша сырья (сэмпл не пополнен). → (halt, n_fail_docs)."""
    fd_first, halt, fail_docs = {}, None, []
    for doc in DOCS:
        if halt:
            break
        if time.time() - t0 > BUDGET_S:
            halt = "бюджет фазы 40 мин исчерпан перед %s" % doc
            break
        ids448 = data[doc]["ids448"]
        n_err = 0
        for k in KS:
            if halt:
                break
            path = os.path.join(RAW, "q2_%s_k%d.json" % (doc, k))
            try:
                resp, cached = get_raw(srv, path, build_body(tpl, ids448[:k]), fd_first)
            except Exception as e:
                n_err += 1
                print("%s k=%d: request exception: %s" % (doc, k, e), flush=True)
                if not srv.alive():
                    halt = "server died during %s k=%d (returncode=%s)" % (
                        doc, k, srv.proc.returncode if srv.proc else None)
                    break
                continue
            ok_pc, detail = check_prompt_count(resp, k)  # G1 — на каждом
            if not ok_pc:
                halt = "G1 prompt-count гейт (%s k=%d): %s" % (doc, k, detail)
                break
            ok_sc, detail = check_schema(resp)
            if not ok_sc:
                n_err += 1
                print("%s k=%d schema: %s" % (doc, k, detail), flush=True)
                continue
            if k % 128 == 64:
                print("%s k=%d: cached=%s" % (doc, k, cached), flush=True)
        if n_err > 0 or (halt is None and any(
                not os.path.exists(os.path.join(RAW, "q2_%s_k%d.json" % (doc, k))) for k in KS)):
            fail_docs.append(doc)
            print("%s: не полный (n_err=%d)" % (doc, n_err), flush=True)
    R["fd_first_request"] = san({"n_samples": len(fd_first.get("samples", [])),
                                  "n_samples_with_dri": fd_first.get("n_with_dri"),
                                  "dri_union": fd_first.get("dri_union")})
    R["device_gate_source"] = "live" if "th" in fd_first else "reused_raw"
    return halt, fail_docs


def make_summary(per_position):
    """per-doc + overall: медиана |Δp_max| + bootstrap (10000, 8181), p90/p99,
    frac |Δ|>0.01, top1_agreement, top20_hit_rate, медиана rank_cpu по хитам,
    цензура (rank>20). Одна процедура для run и from-raw (одинаковая статистика)."""
    S = {"overall": None, "per_doc": {}}
    for scope, rows in ([("overall", [m for rr in per_position.values() for m in rr])] +
                       [("doc:%s" % d, rr) for d, rr in per_position.items()]):
        if not rows:
            if scope == "overall":
                S["overall"] = None
            else:
                S["per_doc"][scope] = None
            continue
        d = np.array([abs(m["d_pmax"]) for m in rows])
        med, ci = bootstrap_ci_median(d, n=10000, seed=8181)
        ranks = [m["rank_cpu"] for m in rows if m["rank_cpu"] is not None]
        out = {"n_pos": len(rows),
               "median_abs_d_pmax": float(med),
               "bootstrap_ci_2.5_97.5": [float(ci[0]), float(ci[1])],
               "p90_abs_d_pmax": float(np.percentile(d, 90)),
               "p99_abs_d_pmax": float(np.percentile(d, 99)),
               "frac_abs_d_gt_0.01": float((d > 0.01).mean()),
               "top1_agreement": float(sum(1 for m in rows if m["top1_agree"]) / len(rows)),
               "top20_hit_rate": float(len(ranks) / len(rows)),
               "median_rank_cpu_hits": float(np.median(ranks)) if ranks else None,
               "n_censored_rank_gt_20": len(rows) - len(ranks),
               "n_bootstrap": 10000, "seed": 8181}
        if scope == "overall":
            S["overall"] = out
        else:
            S["per_doc"][scope] = out
    return S


def g2_main_check(doc, pwalls, ppn):
    """G2 основной фазы (мягкий, объявленное отклонение): со второго запроса
    документа (k=65) prompt_n <= 8; плоский wall по окнам 32: медиана последних
    32 / первых 32 <= 1.5. Нарушения — в список, прогон не убиваем.
    → (viol, wall_ratio)."""
    viol = []
    for k in KS:
        if k == 64:
            continue  # первый запрос документа — холодный по построению
        pn = ppn.get(k)
        if pn is not None and pn > 8:
            viol.append({"doc": doc, "k": k, "prompt_n": pn})
    f32 = [pwalls[k] for k in range(64, 96) if k in pwalls]
    l32 = [pwalls[k] for k in range(416, 448) if k in pwalls]
    ratio = None
    if f32 and l32:
        ratio = float(np.median(l32) / np.median(f32))
        if ratio > 1.5:
            viol.append({"doc": doc, "k": 416, "prompt_n": None, "wall_ratio": ratio})
    return viol, ratio


def rebuild_from_raw():
    """пересборка per_position + summary + doc_status + G2 (мягкий) +
    g2_offline_check ИСКЛЮЧИТЕЛЬНО из /tmp/e94raw/q2_*.json (0 запросов к
    серверу; G1 per-k перепроверяется из raw). ОДНА процедура для live и
    from-raw (одинаковая статистика, AGENTS #1). → dict полей артефакта."""
    data = load_data()
    per_position, status = {}, {}
    g2_viol, g2_ratio = [], {}
    max_pn_2nd, n_pn_2nd = None, 0
    for doc in DOCS:
        rows, missing, bad = [], 0, 0
        pwalls, ppn = {}, {}
        for k in KS:
            path = os.path.join(RAW, "q2_%s_k%d.json" % (doc, k))
            if not os.path.exists(path):
                missing += 1
                continue
            r = json.load(open(path))
            resp = r["response"]
            if isinstance(r.get("wall_s"), (int, float)):
                pwalls[k] = float(r["wall_s"])
            pn = (resp.get("timings") or {}).get("prompt_n")
            if isinstance(pn, int):
                ppn[k] = pn
                if k > 64:  # со второго запроса документа
                    n_pn_2nd += 1
                    max_pn_2nd = pn if max_pn_2nd is None else max(max_pn_2nd, pn)
            if resp.get("tokens_evaluated") != k:  # G1 из raw
                bad += 1
                continue
            m, err = position_metrics(resp, k, data[doc]["gen_ids"], data[doc]["gen_p1"])
            if m is None:
                bad += 1
                continue
            rows.append(m)
        viol, ratio = g2_main_check(doc, pwalls, ppn)
        g2_viol.extend(viol)
        if ratio is not None:
            g2_ratio[doc] = ratio
        per_position[doc] = rows
        status[doc] = {"n_pos": len(rows), "missing_raw": missing, "bad": bad,
                       "complete": bool(len(rows) == len(list(KS)) and not missing and not bad)}
    S = make_summary(per_position)
    n_ok = sum(1 for s in status.values() if s["complete"])
    return {"per_position": per_position, "summary": S, "doc_status": status,
            "n_docs_ok": n_ok, "n_docs_failed": len(DOCS) - n_ok,
            "g2_violations": g2_viol, "g2_violations_n": len(g2_viol),
            "g2_wall_last32_first32": g2_ratio,
            "g2_offline_check": {"max_prompt_n_from_second": max_pn_2nd,
                                 "n_checked_prompt_n_from_second": n_pn_2nd,
                                 "wall_last32_first32": g2_ratio,
                                 "max_wall_ratio": max(g2_ratio.values()) if g2_ratio else None}}


def run_q2_from_raw():
    """--from-raw: пересборка ИСКЛЮЧИТЕЛЬНО из /tmp/e94raw/q2_*.json (0 запросов
    к серверу) + идемпотентность (ДВЕ пересборки, canonical sha256). Поля прогона
    (port/pid/rc/halt/times/fd) сохраняются из существующего Q2.json;
    device_gate_source — из сохранённого fd_first_request (n_samples>0 ⇒ исходный
    прогон live). → rc."""
    t0 = time.time()
    art = os.path.join(OUT, "Q2.json")
    if os.path.exists(art):
        R = json.load(open(art))  # сохранить ВСЕ поля прогона
    else:
        R = {"phase": "q2", "started_unix": int(t0), "rc": 1,
             "halt_reason": "нет Q2.json от прогона для сохранения полей"}
    try:
        d1 = rebuild_from_raw()
        sha1 = canon_sha(d1)
        d2 = rebuild_from_raw()  # прогон 2 (снова с диска)
        sha2 = canon_sha(d2)
        R.update(d2)
        R["idempotence"] = {"runs": 2, "artifact_identical": bool(sha1 == sha2),
                            "sha256": sha2}
        R["t_rebuild_s"] = round(time.time() - t0, 2)
        if "rc" not in R:
            R["rc"] = 0 if R["n_docs_failed"] == 0 else 1
        if "device_gate_source" not in R:
            ff = R.get("fd_first_request") or {}
            R["device_gate_source"] = "live" if (ff.get("n_samples") or 0) > 0 else "reused_raw"
        write_artifact(R)
        o = R["summary"]["overall"]
        g = R["g2_offline_check"]
        print("E94 q2 from-raw: ok=%d/%d med_abs_d_pmax=%.3e top1_agreement=%s "
              "median_rank_cpu=%s g2_violations=%d max_prompt_n_from_2nd=%s "
              "max_wall_ratio=%s idempotent=%s sha=%s" % (
                  R["n_docs_ok"], len(DOCS), o["median_abs_d_pmax"], o["top1_agreement"],
                  o["median_rank_cpu_hits"], R["g2_violations_n"],
                  g["max_prompt_n_from_second"], g["max_wall_ratio"],
                  R["idempotence"]["artifact_identical"], sha2), flush=True)
        print("RUN_RC=%d" % R["rc"], flush=True)
        return int(R["rc"])
    except Exception:
        traceback.print_exc()
        R["halt_reason"] = "from-raw exception (см. traceback выше)"
        R["rc"] = 1
        write_artifact(R)
        print("RUN_RC=1", flush=True)
        return 1


def run_q2():
    """load_data → body_template(Q1) → server → health → device-fd → G2 →
    main run (G3 в первом реальном запросе) → rebuild_from_raw → артефакт."""
    t0 = time.time()
    R = {"phase": "q2", "started_unix": int(t0), "docs": list(DOCS),
         "per_position": {}, "summary": {}}
    srv = None
    try:
        data = load_data()  # len(ids448)!=448 недостижим: load_data бросает (64+384)
        q1 = json.load(open(Q1ART))
        tpl = q1.get("body_template") or {}
        for f in CANON_FIELDS:
            if f not in tpl:
                return finish(R, 1, "Q1 body_template без поля %s" % f, t0)
        R["body_template"] = san(build_body(tpl, []))
        port = pick_port(PORTS)
        srv = LlamaServer(port, SERVER_LOG)
        R["port"] = int(port)
        srv.start()
        R["server_pid"] = int(srv.pid)
        print("E94 q2: server pid=%d port=%d, жду /health" % (srv.pid, port), flush=True)
        srv.wait_health(timeout=300)
        R["t_load_s"] = round(time.time() - t0, 2)
        fd_health = srv.device_fd()
        R["fd_after_health"] = fd_health
        if not fd_health:
            return finish(R, 10, "device-гейт: нет /dev/dri fd в /proc/%d/fd после health" % srv.pid, t0)
        print("health ok, load=%.1fs, fd=%s" % (R["t_load_s"], fd_health), flush=True)
        # G2 спайк (w0, секунды) — ДО основного прогона
        ok2, rows2, fails2 = spike_cache(srv, data["w0"]["ids448"], tpl)
        R["spike_g2"] = san({"order": list(SPIKE_ORDER), "rows": rows2,
                              "pass": bool(ok2), "failures": fails2})
        if not ok2:
            R["spike_g2_note"] = ("спайк-гейт упал — НЕ чиним; в NOTES предлагается скоуп "
                                  "2 дока (w0, w4) вместо 6")
            rc = 13 if any(f.startswith("G1") for f in fails2) else 2
            return finish(R, rc, ("спайк: " + "; ".join(fails2)) if fails2
                          else "спайк: провал (см. spike_g2)", t0)
        print("G2 ok: %s" % san(rows2), flush=True)
        # основной прогон (G3 — fd_sampler в первом реальном запросе)
        halt, fail_docs = main_run(srv, R, data, tpl, t0)
        R.update(rebuild_from_raw())  # per_position/summary/doc_status + G2 (мягкий)
        if halt is None and R["n_docs_failed"] >= 2:
            halt = "%d упавших дока (>=2): %s" % (R["n_docs_failed"], sorted(fail_docs))
        if halt is None:
            rc = 0
        elif halt.startswith("G1"):
            rc = 13
        elif halt.startswith("упавших"):
            rc = 11
        else:  # server died, бюджет, иное
            rc = 1
        return finish(R, rc, halt, t0)
    except Exception:
        traceback.print_exc()
        return finish(R, 1, "exception (см. traceback выше)", t0)
    finally:
        if srv is not None:
            srv.kill()


if __name__ == "__main__":
    if "--from-raw" in sys.argv[1:]:
        sys.exit(run_q2_from_raw())
    sys.exit(run_q2())
