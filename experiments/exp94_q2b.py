#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 Q2b — диагностика флипа на СВЕЖЕМ iGPU-сервере (порт 8111), док w4,
ids448 = prompts.json ids(64) + REPLAY.gen_ids(384) — та же последовательность,
что и в exp94_q2.py. Все ответы — РЕЗУЛЬТАТЫ, гейтов-остановок нет.

Тело ВСЕХ запросов = канон Q2 (body_template Q1.json): temperature 0, stream
false, return_tokens true, logprobs true, top_logprobs 20, n_probs 20,
cache_prompt true, seed 8181, нейтральные штрафы, top_k 400, top_p 1.0,
min_p 0.0, typical_p 1.0. Исключения: D2 — БЕЗ logprobs/top_logprobs/n_probs;
E — top_k 1.

A: ids448[:70], n_predict=1, 5× подряд — emitted / top1 / tokens_evaluated /
   timings.cache_n / prompt_n (детерминирован ли флип k=70 при повторе).
B: «продувка кэша» — 71/70 чередованием × 3 (71,70,71,70,71,70), запись та же.
C: ids448[:64], n_predict=16, один запрос — на КАЖДУЮ из 16 позиций emitted vs
   top1_of_list (флип w4 k=70 → индекс 6; воспроизводится ли в многошаговом декоде).
D: ids448[:64], n_predict=384 × 2: (1) канон с top_logprobs 20; (2) то же тело
   без logprobs/top_logprobs/n_probs. Сравнение потоков id + число флипов.
   flips_with_probs = позиций, где emitted != top1 СВОЕГО top_logprobs(20);
   stream_identity_no_vs_with_probs (бывш. flips_no_probs, MINOR Q2 fix) —
   число позиций, где no_probs-поток отличается от with_probs-потока;
   БЕЗ запроса вероятностей флипы не вычислимываемы (списка top_logprobs нет),
   поэтому метрика = тождественность потоков (не счётчик флипов), см. D.note.
E: A, но top_k=1 (1 запрос; «вершина списка» vs «что видит сэмплер»);
   отказ сервера — текст ошибки как факт.

Артефакт results/e94/Q2b.json пишется при ЛЮБОМ исходе (halt_reason): плоские
поля прогона (server_cmd, port, pid, t_load_s, fd-снимок) + A/B/C/D/E с сырыми
ответами (D — извлечённые поля: 2×384 позиции × 20 велики) и счётчиками
A.emitted_ids/A.distinct, B.emitted_ids, C.per_position, D.with_probs_ids/
D.no_probs_ids/D.identical/D.first_diff/D.flips_with_probs/D.stream_identity_
no_vs_with_probs/D.note,
E.emitted/E.top1.
rc: 0 — все ответы получены (даже когда всё расходится); 1 — запрос/сервер
сломались. Печатается RUN_RC, sys.exit(rc).
Запуск: RUN_ONE_MIN_MEM=3 ./run_one.sh e94q2b "python3 -u experiments/exp94_q2b.py".
"""
import json, os, sys, threading, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp94_sycl_common import LLAMA_BIN, SERVER_ARGS, LlamaServer, pick_port, san
from exp94_sycl_stream import fd_sampler

PORT = 8111
OUT = "results/e94"
PROMPTS = "results/e93/prompts.json"
REPLAY = "results/e81/REPLAY.json"
Q1ART = "results/e94/Q1.json"
SERVER_LOG = "/tmp/e94q2b_server.log"
BUDGET_S = 20 * 60
FD_SAMPLES, FD_INTERVAL = 20, 0.05
PROB_KEYS = ("logprobs", "top_logprobs", "n_probs")


def load_ids448():
    """w4: prompts.json ids(64) + REPLAY.gen_ids(384) — дословно exp94_q2.load_data."""
    p = json.load(open(PROMPTS))
    w4 = next(d for d in p["docs"] if d["key"] == "w4")
    r = json.load(open(REPLAY))
    ids = list(w4["ids"]) + list(r["gen_ids"]["w4"])
    if len(ids) != 448:
        raise RuntimeError("w4 ids448 len=%d != 448" % len(ids))
    return ids


def build_body(tpl, prompt, n_predict, drop_probs=False, **overrides):
    b = dict(tpl)
    b["prompt"] = list(prompt)
    b["n_predict"] = int(n_predict)
    b["cache_prompt"] = True
    if drop_probs:
        for k in PROB_KEYS:
            b.pop(k, None)
    b.update(overrides)
    return b


def pos_report(resp):
    out = []
    for i, p in enumerate(resp.get("completion_probabilities") or []):
        tl = p.get("top_logprobs") or []
        out.append({"pos": i, "emitted": p.get("id"),
                    "top1_of_list": tl[0].get("id") if tl else None,
                    "logprob": p.get("logprob")})
    return out


def flips_of(rows):
    return sum(1 for r in rows
               if r["top1_of_list"] is not None and r["emitted"] != r["top1_of_list"])


def post(srv, body, timeout=900):
    try:
        resp, dt = srv.completion(body, timeout=timeout)
        return resp, dt, None
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
        if hasattr(e, "read"):  # HTTPError: текст ответа сервера — факт
            try:
                err += " | body=" + e.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
        return None, None, err


def one_request(srv, i, body, label):
    """один запрос n_predict=1 → (ok, row|None, err|None)."""
    if not srv.alive():
        return False, None, "server died (returncode=%s)" % (
            srv.proc.returncode if srv.proc else None)
    resp, dt, err = post(srv, body)
    if err:
        return False, {"req": i + 1, "error": err, "body": san(body)}, err
    t = resp.get("timings") or {}
    cp = resp.get("completion_probabilities") or []
    p0 = cp[0] if cp else {}
    tl = p0.get("top_logprobs") or []
    row = {"req": i + 1, "emitted": p0.get("id"),
           "top1": tl[0].get("id") if tl else None,
           "tokens_evaluated": resp.get("tokens_evaluated"),
           "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
           "wall_s": round(dt, 3), "raw": san(resp)}
    print("%s req%d: emitted=%s top1=%s te=%r cache_n=%r prompt_n=%r" %
          (label, i + 1, row["emitted"], row["top1"], row["tokens_evaluated"],
           row["cache_n"], row["prompt_n"]), flush=True)
    return True, row, None


def run_repeats(srv, R, section, ids448, tpl, ks):
    """A/B: серия одиночных запросов (ks — длины prompt) → R[section] со
    счётчиками emitted_ids/top1_ids/distinct/tokens_evaluated/cache_n/prompt_n
    + raw каждого запроса. → (ok, err|None)."""
    reqs, errs = [], []
    for i, k in enumerate(ks):
        ok, row, err = one_request(srv, i, build_body(tpl, ids448[:k], 1), section)
        reqs.append(row if row else {"req": i + 1, "error": err})
        if not ok:
            errs.append("req%d: %s" % (i + 1, err))
            break
    done = [r for r in reqs if "emitted" in r]
    R[section] = {"n_reqs": len(done),
                  "emitted_ids": [r["emitted"] for r in done],
                  "top1_ids": [r["top1"] for r in done],
                  "distinct": len({r["emitted"] for r in done}),
                  "tokens_evaluated": [r["tokens_evaluated"] for r in done],
                  "cache_n": [r["cache_n"] for r in done],
                  "prompt_n": [r["prompt_n"] for r in done],
                  "reqs": reqs, "errors": errs}
    return (len(errs) == 0), ("; ".join(errs) if errs else None)


def run_C(srv, R, ids448, tpl):
    body = build_body(tpl, ids448[:64], 16)
    resp, dt, err = post(srv, body)
    if err:
        R["C"] = {"error": err, "body": san(body)}
        return False, err
    rows = pos_report(resp)
    per = [{"pos": r["pos"], "k": 64 + r["pos"], "emitted": r["emitted"],
            "top1_of_list": r["top1_of_list"],
            "flip": (r["top1_of_list"] is not None and r["emitted"] != r["top1_of_list"])}
           for r in rows]
    R["C"] = {"n_pos": len(per), "per_position": per,
              "n_flips": sum(1 for x in per if x["flip"]),
              "flip_positions": [x["pos"] for x in per if x["flip"]],
              "tokens_evaluated": resp.get("tokens_evaluated"),
              "wall_s": round(dt, 3), "raw": san(resp)}
    print("C: n_pos=%d flips=%s (известный fлип w4: индекс 6, k=70)" %
          (R["C"]["n_pos"], R["C"]["flip_positions"]), flush=True)
    return True, None


def run_D(srv, R, ids448, tpl):
    b1 = build_body(tpl, ids448[:64], 384)
    b2 = build_body(tpl, ids448[:64], 384, drop_probs=True)
    samples = []
    th = threading.Thread(target=fd_sampler, args=(srv.pid, FD_SAMPLES, FD_INTERVAL, samples))
    th.start()
    r1, dt1, e1 = post(srv, b1, timeout=1800)
    th.join()
    fd = {"n_samples": len(samples), "n_samples_with_dri": sum(1 for s in samples if s),
          "dri_union": sorted({x for s in samples for x in s})}
    if e1:
        R["D"] = {"error": "with_probs: " + e1, "bodies": {"with_probs": san(b1), "no_probs": san(b2)}}
        return False, e1
    r2, dt2, e2 = post(srv, b2, timeout=1800)
    if e2:
        R["D"] = {"error": "no_probs: " + e2,
                  "bodies": {"with_probs": san(b1), "no_probs": san(b2)}, "fd_d1": fd}
        return False, e2
    rows1 = pos_report(r1)
    ids1, ids2 = r1.get("tokens") or [], r2.get("tokens") or []
    n = min(len(ids1), len(ids2))
    diff = [i for i in range(n) if ids1[i] != ids2[i]]
    tl_present2 = any((p.get("top_logprobs") or []) for p in (r2.get("completion_probabilities") or []))
    R["D"] = {"bodies": {"with_probs": san(b1), "no_probs": san(b2)},
              "with_probs_ids": ids1, "no_probs_ids": ids2,
              "len1": len(ids1), "len2": len(ids2),
              "identical": (ids1 == ids2),
              "first_diff": diff[0] if diff else None,
              "n_diff": len(diff), "diff_positions": diff[:64],
              "flips_with_probs": flips_of(rows1),
              "flip_positions_with_probs": [r["pos"] for r in rows1
                                            if r["top1_of_list"] is not None
                                            and r["emitted"] != r["top1_of_list"]],
              "stream_identity_no_vs_with_probs": len(diff),
              "stream_identity_no_vs_with_probs_definition": ("позиции, где no_probs-поток "
                                            "отличается от with_probs-потока"),
              "note": ("без запроса вероятностей (no_probs-ответ не несёт top_logprobs) флипы "
                       "не вычислимываемы — метрика равна тождественности потоков (identical/n_diff), "
                       "а не числу флипов; переименование из flips_no_probs (MINOR Q2 fix)"),
              "d1_tokens_eq_cp_ids": (ids1 == [x.get("id") for x in (r1.get("completion_probabilities") or [])]),
              "d1_top1_per_pos": [r["top1_of_list"] for r in rows1],
              "no_probs_response_has_top_logprobs": bool(tl_present2),
              "no_probs_n_cp": len(r2.get("completion_probabilities") or []),
              "wall_s": [round(dt1, 3), round(dt2, 3)],
              "timings": {"d1": san((r1.get("timings") or {})), "d2": san((r2.get("timings") or {}))},
              "fd_d1": fd}
    print("D: len1=%d len2=%d identical=%s first_diff=%s n_diff=%d flips_with_probs=%d" %
          (len(ids1), len(ids2), R["D"]["identical"], R["D"]["first_diff"],
           len(diff), R["D"]["flips_with_probs"]), flush=True)
    return True, None


def run_E(srv, R, ids448, tpl):
    body = build_body(tpl, ids448[:70], 1, top_k=1)
    resp, dt, err = post(srv, body)
    if err:  # отказ сервера — текст ошибки как факт (запрос считается полученным)
        R["E"] = {"error": err, "body": san(body), "emitted": None, "top1": None}
        print("E: server rejected: %s" % err[:200], flush=True)
        return True, None
    cp = resp.get("completion_probabilities") or []
    p0 = cp[0] if cp else {}
    tl = p0.get("top_logprobs") or []
    t = resp.get("timings") or {}
    R["E"] = {"emitted": p0.get("id"), "top1": tl[0].get("id") if tl else None,
              "tokens_evaluated": resp.get("tokens_evaluated"),
              "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n"),
              "wall_s": round(dt, 3), "error": None, "raw": san(resp)}
    print("E (top_k=1): emitted=%s top1=%s" % (R["E"]["emitted"], R["E"]["top1"]), flush=True)
    return True, None


def finish(R, rc, halt, t0):
    R["rc"], R["halt_reason"] = rc, halt
    R["t_total_s"] = round(time.time() - t0, 2)
    R = san(R)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "Q2b.json"), "w") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    print("E94 q2b rc=%d%s" % (rc, (" halt=" + halt) if halt else ""), flush=True)
    print("RUN_RC=%d" % rc, flush=True)
    return rc


def run_q2b():
    t0 = time.time()
    R = {"phase": "q2b", "started_unix": int(t0), "doc": "w4"}
    srv = None
    try:
        ids448 = load_ids448()
        q1 = json.load(open(Q1ART))
        tpl = q1.get("body_template") or {}
        for f in ("temperature", "stream", "return_tokens", "top_logprobs", "seed",
                  "top_k", "top_p", "min_p", "n_probs"):
            if f not in tpl:
                return finish(R, 1, "Q1 body_template без поля %s" % f, t0)
        R["body_template"] = san(build_body(tpl, [], 1))
        port = pick_port([PORT])
        srv = LlamaServer(port, SERVER_LOG)
        R["port"] = int(port)
        R["server_cmd"] = [LLAMA_BIN] + SERVER_ARGS + ["--port", str(port), "--log-disable"]
        srv.start()
        R["server_pid"] = int(srv.pid)
        print("E94 q2b: server pid=%d port=%d, жду /health" % (srv.pid, port), flush=True)
        srv.wait_health(timeout=300)
        R["t_load_s"] = round(time.time() - t0, 2)
        R["fd_after_health"] = srv.device_fd()
        print("health ok, load=%.1fs, fd=%s" % (R["t_load_s"], R["fd_after_health"]), flush=True)
        for section, fn in (("A", lambda: run_repeats(srv, R, "A", ids448, tpl, [70] * 5)),
                            ("B", lambda: run_repeats(srv, R, "B", ids448, tpl, [71, 70] * 3)),
                            ("C", lambda: run_C(srv, R, ids448, tpl)),
                            ("D", lambda: run_D(srv, R, ids448, tpl)),
                            ("E", lambda: run_E(srv, R, ids448, tpl))):
            if time.time() - t0 > BUDGET_S:
                return finish(R, 1, "бюджет %ds исчерпан перед %s" % (BUDGET_S, section), t0)
            ok, err = fn()
            if not ok:
                return finish(R, 1, "%s: %s" % (section, err), t0)
            print("E94 q2b: %s done (t=%.0fs)" % (section, time.time() - t0), flush=True)
        return finish(R, 0, None, t0)
    except Exception:
        traceback.print_exc()
        return finish(R, 1, "exception (см. traceback выше)", t0)
    finally:
        if srv is not None:
            srv.kill()


if __name__ == "__main__":
    sys.exit(run_q2b())
