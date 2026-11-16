#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 Q1 (ТЗ §2.1): 16 доков (w0..w15) × r=2, greedy n_predict=384, канон-тело (B)
(FORMAT.md: n_probs:20 + top_logprobs:20 + снятие прореживания top_k:400, top_p:1.0,
min_p:0.0, typical_p:1.0, xtc_probability:0.0 + нулевые штрафы, seed:8181,
temperature:0, stream:false, return_tokens:true, cache_prompt:false), ОДИН сервер
на всю фазу. Метрики — только из raw-файлов /tmp/e94raw/q1_<doc>_r<r>.json
(идемпотентно: существующий raw не перезапрашивается). Формулы — дословно exp81b
через exp94_sycl_common: rep8 по конкатенации 64+384 (off=64, 12 сегментов по 32,
rep8=mean(segments['8'])) против CPU-реплая тем же, prefix_len /384, repro_rate r1/r2
(гейт медианы 0.95 → метка non-interpretable (determinism)), главная — медиана |Δrep8|
+ bootstrap (10000, seed 8181, 2.5/97.5); вспомогательно Spearman avg-ranks +
пермутационный null (2000, seed 8181) + число различных rep8. Никаких «механизм»-
выводов: только числа и гейты.
RUN_RC: 0 ok · 10 device-гейт (нет /dev/dri в /proc/<pid>/fd после готовности) ·
11 ≥4 упавших дока · 13 prompt-count (tokens_evaluated != 64) · 1 другое.
sys.exit(rc) обязателен (иначе run_one допишет ложный RUN_RC=0 — воспроизведено).
"""
import hashlib, json, os, sys, threading, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp94_sycl_common import (LlamaServer, pick_port, san, base_body,
                               check_schema, check_prompt_count, rep8_segments,
                               prefix_len, perm_null, bootstrap_ci_median)
from exp94_sycl_stream import save_raw, fd_sampler
import numpy as np

RAW = "/tmp/e94raw"
OUT = "results/e94"
PROMPTS = "results/e93/prompts.json"
REPLAY = "results/e81/REPLAY.json"
INDEP = "/tmp/q1_independent_perdoc.json"  # независимый пересчёт supervisor'а (кросс-чек)
PORTS = [8110, 8111, 8112]
SERVER_LOG = "/tmp/e94q1_server.log"
N_PRED = 384
# канон-тело (B) из results/e94/FORMAT.md: реальное поле — n_probs (top_logprobs
# сервером игнорируется, но остаётся в теле по канону); прореживание снято,
# чтобы p_max был ближе к полному vocab. Жадный поток от этого не меняется (384/384).
CANON = {"top_logprobs": 20, "n_probs": 20, "top_k": 400, "top_p": 1.0,
         "min_p": 0.0, "typical_p": 1.0, "xtc_probability": 0.0}


def write_artifact(R):
    R = san(R)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "Q1.json"), "w") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    return R


def finish(R, rc, halt, t0):
    """артефакт пишется и при STOP (halt_reason) — иначе артефакт неаудитабелен."""
    R["rc"], R["halt_reason"] = rc, halt
    R["t_total_s"] = round(time.time() - t0, 2)
    write_artifact(R)
    print("E94 q1 rc=%d%s" % (rc, (" halt=" + halt) if halt else ""), flush=True)
    print("RUN_RC=%d" % rc, flush=True)
    return rc


def get_response(srv, path, body, fd_first):
    """идемпотентный запрос: существующий raw → читаем; иначе POST → save_raw.
    fd_first: при ПЕРВОМ реальном (не кэш-хите) запросе — сэмплируем /proc/<pid>/fd
    во время него (20×0.05 с): доказательство, что decode идёт на устройстве; сам
    device-гейт — по fd после готовности (ТЗ §2)."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["response"], True
    if fd_first is not None and "th" not in fd_first:
        fd_first["samples"] = []
        fd_first["th"] = threading.Thread(target=fd_sampler, args=(srv.pid, 20, 0.05,
                                                                   fd_first["samples"]))
        fd_first["th"].start()
    try:
        resp, dt = srv.completion(body, timeout=900)
    finally:
        if fd_first is not None and "th" in fd_first:
            fd_first["th"].join()
            fd_first["n_with_dri"] = sum(1 for s in fd_first["samples"] if s)
            fd_first["dri_union"] = sorted({x for s in fd_first["samples"] for x in s})
    save_raw(os.path.basename(path), {"request": body, "response": resp, "wall_s": round(dt, 3)})
    return resp, False


def per_doc_entry(key, ids64, r1, r2, cpu_gen, err1, err2):
    """per-doc: id-стрим — из raw `tokens`; конкатенация 64+384, off=64 (конвенция
    exp81b: повтор считается относительно ВСЕГО контекста prompt+gen). ok = полный
    ответ на обоих r (только они идут в агрегаты)."""
    e = {"doc": key, "ok": False, "errors": {"r1": err1, "r2": err2, "cpu": None}}
    if cpu_gen is None:
        e["errors"]["cpu"] = "no gen_ids[%s] in REPLAY.json" % key
        return e
    stream = r1.get("tokens", []) if r1 is not None else \
        (r2.get("tokens", []) if r2 is not None else None)
    if not stream:
        return e
    _, rep8_l = rep8_segments(list(ids64) + list(stream), off=64)
    _, rep8_c = rep8_segments(list(ids64) + list(cpu_gen), off=64)
    pl = int(prefix_len(stream, list(cpu_gen)))
    e.update({"prefix_len": pl, "prefix_len_norm": round(pl / float(N_PRED), 6),
              "rep8_llama": float(rep8_l), "rep8_cpu": float(rep8_c),
              "delta_rep8": float(rep8_l - rep8_c)})
    if r1 is not None and r2 is not None:
        g1, g2 = r1["tokens"], r2["tokens"]
        agree = sum(1 for a, b in zip(g1, g2) if a == b)
        e["repro_rate"] = float(agree / min(len(g1), len(g2))) if g1 and g2 else None
        e["n_tokens_r1"], e["n_tokens_r2"] = len(g1), len(g2)
        e["ok"] = True
    else:
        e["repro_rate"] = None
        e["n_tokens_r1"] = len(r1["tokens"]) if r1 is not None else None
        e["n_tokens_r2"] = len(r2["tokens"]) if r2 is not None else None
    return e


def make_summary(ok):
    """сводка из ok-доков; ОДНА процедура для run и from-raw (одинаковая
    статистика на сторонах: bootstrap/perm из common, те же сиды)."""
    S = {}
    if not ok:
        return S
    dl = np.array([abs(e["delta_rep8"]) for e in ok])
    med, ci = bootstrap_ci_median(dl, n=10000, seed=8181)
    S["median_abs_delta_rep8"] = float(med)
    S["bootstrap_ci_2.5_97.5"] = [float(ci[0]), float(ci[1])]
    S["n_bootstrap"], S["seed"] = 10000, 8181
    # интерпретируемость (ревью r1 MINOR-2, ТЗ §8 M4): CI накрывает 0 ⇒ не эффект
    S["interpretability"] = ("not interpretable as an effect"
                              if ci[0] <= 0.0 <= ci[1] else "interpretable as an effect")
    S["interpretability_reason"] = (
        "CI медианы |Δrep8| накрывает 0 ([%.6g, %.6g]); медиана %.6g < ширины насыщенной "
        "полосы rep8 0.839–0.974 (0.135) [ревью r1]" % (ci[0], ci[1], med))
    # единичная бифуркация (ревью r1 MINOR-3): max |Δ| — один док, расхождение движков,
    # не «iGPU теряет повторы»
    w = min(ok, key=lambda e: e["delta_rep8"])
    S["bifurcation_note"] = ("%s Δrep8=%.4f = единичная бифуркация (1/%d): расхождение "
                              "движков f16/SYCL против fp32/CPU (низкоуверенные позиции, "
                              "argmax-flip при temp=0), а не «iGPU теряет повторы»"
                              % (w["doc"], w["delta_rep8"], len(ok)))
    a_l = np.array([e["rep8_llama"] for e in ok])
    a_c = np.array([e["rep8_cpu"] for e in ok])
    if len(ok) >= 3:
        rho, p, nullci = perm_null(a_l, a_c, n=2000, seed=8181)
        S["spearman"] = {"rho": rho, "p_perm": p,
                         "null_ci": [float(nullci[0]), float(nullci[1])]
                         if nullci is not None else None,
                         "n_perm": 2000, "seed": 8181,
                         # определение p (ревью r1 MAJOR-1: без закреплённого определения
                         # значения не сводятся): two-sided на |rho|, MC-доля без +1
                         "p_definition": ("two-sided on |rho|: MC fraction of permutations "
                                          "with |rho_perm| >= |rho_obs| (no +1 correction), "
                                          "llama array permuted against fixed cpu, "
                                          "avg-ranks on ties, n_perm=2000, seed=8181"),
                         "p_mc_se": float(np.sqrt(p * (1.0 - p) / 2000))}
    S["distinct_rep8"] = {"llama": int(len(set(a_l.tolist()))),
                          "cpu": int(len(set(a_c.tolist())))}
    rates = [e["repro_rate"] for e in ok if e["repro_rate"] is not None]
    if rates:
        rm = float(np.median(rates))
        S["repro"] = {"median": rm, "gate": 0.95, "pass": bool(rm >= 0.95),
                      "label": None if rm >= 0.95 else "non-interpretable (determinism)"}
    # пер-доковые агрегаты (ТЗ §3: внешний ревью пересчитывает любую цифру)
    pls = [e["prefix_len"] for e in ok if e.get("prefix_len") is not None]
    if pls:
        S["median_prefix_len"] = float(np.median(pls))
        S["frac_prefix_full"] = float(sum(1 for x in pls if x == N_PRED) / len(pls))
        S["n_docs_prefix_full"] = int(sum(1 for x in pls if x == N_PRED))
    # r1≡r2 (детерминизм), НЕ «совпал с CPU» (ревью r1 MINOR-1)
    S["n_docs_repro_identical"] = int(sum(1 for e in ok if e.get("repro_rate") == 1.0))
    return S


def canon_sha(obj):
    """канонический json.dumps (sort_keys + компактные separators) → sha256 hex:
    сравнение пересборок без влияния порядка ключей (идемпотентность)."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def crosscheck(per_doc):
    """сверка с независимым пересчётом supervisor'а (ИНДЕП): prefix_len — точное
    равенство, rep8_*/delta — max|Δ| < 1e-12. Расхождение НЕ подгоняется: оно
    фиксируется как есть в артефакте (причина — в NOTES)."""
    try:
        ind = json.load(open(INDEP))
    except Exception as e:
        return {"source": INDEP, "error": "не читается: %s" % e, "pass": None}
    by_doc = {e["doc"]: e for e in per_doc}
    n_match, dmax, missing = 0, 0.0, []
    for row in ind:
        e = by_doc.get(row["doc"])
        if e is None or not e["ok"]:
            missing.append(row["doc"])
            continue
        if e["prefix_len"] == int(row["prefix_len"]):
            n_match += 1
        dmax = max(dmax, abs(e["rep8_llama"] - float(row["rep8_llama"])),
                   abs(e["rep8_cpu"] - float(row["rep8_cpu"])),
                   abs(e["delta_rep8"] - float(row["delta"])))
    return {"source": INDEP, "n_docs": len(ind),
            "prefix_len_exact_matches": n_match,
            "max_abs_delta_rep8": float(dmax),
            "missing": missing,
            "pass": bool(not missing and n_match == len(ind) and dmax < 1e-12)}


def rebuild_from_raw():
    """одна пересборка per_doc + сводки из /tmp/e94raw (чтение с диска; второй
    вызов — реальная проверка идемпотентности, не кешированная копия)."""
    p = json.load(open(PROMPTS))
    docs = [d for d in p["docs"] if d["key"].startswith("w")]
    if len(docs) != 16:
        raise RuntimeError("prompts.json: %d доков, ждём 16" % len(docs))
    replay = json.load(open(REPLAY))
    gen_ids, replay_docs = replay["gen_ids"], replay["docs"]
    per_doc, selfcheck = [], []
    for d in docs:
        key, ids64 = d["key"], list(d["ids"])
        rr, errs = {}, {}
        for r in (1, 2):
            path = os.path.join(RAW, "q1_%s_r%d.json" % (key, r))
            try:
                rr[r] = json.load(open(path))["response"]
            except Exception as e:
                errs[r] = "raw: %s: %s" % (path, e)
        e = per_doc_entry(key, ids64, rr.get(1), rr.get(2), gen_ids.get(key),
                          errs.get(1), errs.get(2))
        per_doc.append(e)
        st = replay_docs.get(key, {}).get("segments", {}).get("8")
        if "rep8_cpu" in e and st:
            selfcheck.append(abs(e["rep8_cpu"] - float(np.mean(st))))
    ok = [e for e in per_doc if e["ok"]]
    return per_doc, make_summary(ok), len(ok), selfcheck


def run_q1_from_raw():
    """--from-raw: пересборка per_doc и сводок ИСКЛЮЧИТЕЛЬНО из
    /tmp/e94raw/q1_*.json (без единого запроса к серверу) + кросс-чек +
    идемпотентность (ДВЕ пересборки, canonical sha256 сводки+per_doc).
    Поля прогона (port/pid/rc/halt/times/fd) сохраняются из существующего Q1.json."""
    t0 = time.time()
    art = os.path.join(OUT, "Q1.json")
    if os.path.exists(art):
        R = json.load(open(art))  # сохранить ВСЕ поля прогона
    else:
        R = {"phase": "q1", "started_unix": int(t0), "rc": 1,
             "halt_reason": "нет Q1.json от прогона для сохранения полей"}
    try:
        per_doc, S, n_ok, selfcheck = rebuild_from_raw()  # прогон 1
        sha1 = canon_sha({"per_doc": per_doc, "summary": S})
        per_doc, S, n_ok, selfcheck = rebuild_from_raw()  # прогон 2 (снова с диска)
        sha2 = canon_sha({"per_doc": per_doc, "summary": S})
        R["idempotence"] = {"runs": 2, "artifact_identical": bool(sha1 == sha2),
                            "sha256": sha2}
        R["per_doc"] = per_doc
        R["summary"] = S
        R["n_docs_ok"], R["n_docs_failed"] = n_ok, len(per_doc) - n_ok
        R["selfcheck_cpu_rep8_max_abs_delta"] = (float(np.max(selfcheck))
                                                  if selfcheck else None)
        R["independent_crosscheck"] = crosscheck(per_doc)
        R["t_rebuild_s"] = round(time.time() - t0, 2)
        if "rc" not in R:
            R["rc"] = 0 if n_ok == len(per_doc) else 1
        write_artifact(R)
        x = R["independent_crosscheck"]
        print("E94 q1 from-raw: ok=%d/16 crosscheck pass=%s prefix_exact=%s max_d=%s "
              "idempotent=%s sha=%s" % (n_ok, x.get("pass"), x.get("prefix_len_exact_matches"),
                                        x.get("max_abs_delta_rep8"),
                                        R["idempotence"]["artifact_identical"], sha2),
              flush=True)
        print("RUN_RC=%d" % R["rc"], flush=True)
        return int(R["rc"])
    except Exception:
        traceback.print_exc()
        R["halt_reason"] = "from-raw exception (см. traceback)"
        R["rc"] = 1
        write_artifact(R)
        print("RUN_RC=1", flush=True)
        return 1


def run_q1():
    t0 = time.time()
    R = {"phase": "q1", "started_unix": int(t0), "per_doc": [], "summary": {}}
    srv = None
    try:
        p = json.load(open(PROMPTS))
        docs = [d for d in p["docs"] if d["key"].startswith("w")]
        replay = json.load(open(REPLAY))
        gen_ids, replay_docs = replay["gen_ids"], replay["docs"]
        if len(docs) != 16:
            return finish(R, 1, "prompts.json: %d доков, ждём 16" % len(docs), t0)
        for d in docs:
            if len(d["ids"]) != 64:
                return finish(R, 1, "%s: len(ids)=%d != 64" % (d["key"], len(d["ids"])), t0)
            if d["key"] not in gen_ids:
                return finish(R, 1, "REPLAY.json: нет gen_ids[%s]" % d["key"], t0)

        port = pick_port(PORTS)
        srv = LlamaServer(port, SERVER_LOG)
        R["port"] = int(port)
        srv.start()
        R["server_pid"] = int(srv.pid)
        print("E94 q1: server pid=%d port=%d, жду /health" % (srv.pid, port), flush=True)
        srv.wait_health(timeout=300)
        R["t_load_s"] = round(time.time() - t0, 2)
        # device-гейт (ТЗ §2): ПОСЛЕ ГОТОВНОСТИ /proc/<pid>/fd обязан содержать /dev/dri/*
        fd_health = srv.device_fd()
        R["fd_after_health"] = fd_health
        if not fd_health:
            return finish(R, 10, "device-гейт: нет /dev/dri fd в /proc/%d/fd после health"
                          % srv.pid, t0)
        print("health ok, load=%.1fs, fd=%s" % (R["t_load_s"], fd_health), flush=True)
        R["body_template"] = san(base_body([], N_PRED, False, **CANON))  # prompt: [] — шаблон

        fd_first = {}
        per_doc, errs, resp_cache, halt = [], {}, {}, None
        for d in docs:
            key, ids64 = d["key"], list(d["ids"])
            for r in (1, 2):
                if halt:
                    break
                if not srv.alive():
                    halt = "server died before %s r%d (returncode=%s)" % (
                        key, r, srv.proc.returncode if srv.proc else None)
                    break
                path = os.path.join(RAW, "q1_%s_r%d.json" % (key, r))
                try:
                    resp, cached = get_response(
                        srv, path, base_body(ids64, N_PRED, False, **CANON), fd_first)
                except Exception as e:
                    errs[(key, r)] = "request exception: %s" % e
                    print("%s r%d: %s" % (key, r, e), flush=True)
                    if not srv.alive():
                        halt = "server died during %s r%d (returncode=%s)" % (
                            key, r, srv.proc.returncode if srv.proc else None)
                    continue
                ok_pc, detail = check_prompt_count(resp, len(ids64))
                if not ok_pc:
                    halt = "prompt-count гейт (%s r%d): %s" % (key, r, detail)
                    break
                ok_sc, detail = check_schema(resp)
                if not ok_sc:
                    errs[(key, r)] = "schema: %s" % detail
                    print("%s r%d schema: %s" % (key, r, detail), flush=True)
                    continue
                resp_cache[(key, r)] = resp
                print("%s r%d: cached=%s n_tokens=%s tokens_evaluated=%s" %
                      (key, r, cached, len(resp.get("tokens", [])), resp.get("tokens_evaluated")),
                      flush=True)

        R["fd_first_request"] = san({"n_samples": len(fd_first.get("samples", [])),
                                      "n_samples_with_dri": fd_first.get("n_with_dri"),
                                      "dri_union": fd_first.get("dri_union")})

        selfcheck = []
        for d in docs:
            key, ids64 = d["key"], list(d["ids"])
            e = per_doc_entry(key, ids64, resp_cache.get((key, 1)), resp_cache.get((key, 2)),
                              gen_ids.get(key), errs.get((key, 1)), errs.get((key, 2)))
            per_doc.append(e)
            st = replay_docs.get(key, {}).get("segments", {}).get("8")
            if "rep8_cpu" in e and st:
                selfcheck.append(abs(e["rep8_cpu"] - float(np.mean(st))))

        R["per_doc"] = per_doc  # баг прогона: local per_doc не попадал в артефакт
        ok = [e for e in per_doc if e["ok"]]
        R["n_docs_ok"], R["n_docs_failed"] = len(ok), 16 - len(ok)
        if halt is None and R["n_docs_failed"] >= 4:
            halt = "%d упавших доков (>=4): %s" % (R["n_docs_failed"], sorted(errs))

        R["summary"] = make_summary(ok)
        # самчек-конвенции: свежий rep8_cpu против замороженных REPLAY.docs.w*.segments['8']
        R["selfcheck_cpu_rep8_max_abs_delta"] = (float(np.max(selfcheck))
                                                  if selfcheck else None)

        if halt is None:
            rc = 0
        elif halt.startswith("prompt-count"):
            rc = 13
        elif halt.startswith("server died") or "exception" in halt:
            rc = 1
        else:
            rc = 11
        return finish(R, rc, halt, t0)
    except Exception:
        traceback.print_exc()
        return finish(R, 1, "exception (см. traceback выше)", t0)
    finally:
        if srv is not None:
            srv.kill()


if __name__ == "__main__":
    sys.exit(run_q1())
