#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 Q2c — одна конкурирующая гипотеза к Q2b: топ-список prefill-шага берётся
от распределения ПРЕДЫДУЩЕЙ позиции (сдвиг на 1). 4 запроса, тот же сервер
(порт из кандидатов 8112+, iGPU/SYCL) и каноническое тело Q2 (body_template Q1.json).
Остановок по результатам НЕ делать — всё результат.

1. /tokenize {"content": "The capital of France is"} и
   /tokenize {"content": "The capital of France"} — сохранить оба списка id.
2. prompt = ids("...is"), n_predict=1, top_logprobs=5 — emitted + ВЕСЬ список
   (id, logprob, текст токена) + tokens_evaluated (prefill-only, резковершинный
   промпт: после "is" ждётся ~1.0 на "Paris").
3. prompt = ids("...France"), n_predict=3, top_logprobs=5 — per_position:
   emitted / вершина своего списка / flip; pos0 = prefill-шаг, pos1,pos2 = decode.
4. Контроль на w4: prompt = ids448[:69], n_predict=2, top_logprobs=20 —
   per_position + сравнение вершин: вершина pos0 vs вершина из Q2b-A (k=70),
   вершина pos1 vs вершина из Q2b-A.

verdict_shift: «сдвиг подтверждён», если на резковершинных промптах (2,3) на
prefill-шагах вершина != emitted (2/2), а на decode-шагах == (0/2);
«сдвиг отвергнут», если вершина == emitted везде (0/4); иначе «смешанный: ...»
с числами (оба правила не выполняются). Числа рядом с выводом.
Артефакт results/e94/Q2c.json при ЛЮБОМ исходе (halt_reason).
rc: 0 — все ответы получены; 1 — запрос/сервер сломались. sys.exit(rc).
Запуск: ./run_one.sh e94q2c "python3 -u experiments/exp94_q2c.py".
"""
import json, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp94_sycl_common import LLAMA_BIN, SERVER_ARGS, LlamaServer, pick_port, san
from exp94_q2b import build_body, pos_report, post, load_ids448

PORTS = [8112, 8113, 8114, 8115]
OUT = "results/e94"
Q1ART = "results/e94/Q1.json"
Q2B = "results/e94/Q2b.json"
SERVER_LOG = "/tmp/e94q2c_server.log"
TXT_A = "The capital of France is"
TXT_B = "The capital of France"


def run_single(srv, R, i, body, label):
    """один completion: (ok, (resp, dt, pos_rows) | None, err|None); raw в R["reqs"]."""
    if not srv.alive():
        return False, None, "server died (returncode=%s)" % (
            srv.proc.returncode if srv.proc else None)
    resp, dt, err = post(srv, body, timeout=900)
    if err:
        R["reqs"].append({"label": label, "error": err, "body": san(body)})
        return False, None, "%s: %s" % (label, err)
    rows = pos_report(resp)
    R["reqs"].append({"label": label, "body": san(body), "wall_s": round(dt, 3),
                      "tokens_evaluated": resp.get("tokens_evaluated"),
                      "timings": san(resp.get("timings") or {}),
                      "emitted_ids": resp.get("tokens"), "raw": san(resp)})
    print("%s: te=%r n_pos=%d" % (label, resp.get("tokens_evaluated"), len(rows)), flush=True)
    return True, (resp, dt, rows), None


def full_list(p0, n=5):
    """весь список top_logprobs позиции: [{id, logprob, token}] (текст — поле token)."""
    tl = (p0.get("top_logprobs") or [])[:n]
    return [{"id": e.get("id"), "logprob": e.get("logprob"), "token": e.get("token")}
            for e in tl]


def verdict_shift(flip_map):
    """flip_map: {label_pos: flip_bool}. Правило ТЗ: подтверждён = prefill 2/2
    flipped и decode 0/2; отвергнут = 0/4; иначе смешанный с числами."""
    pre = [v for k, v in flip_map.items() if "prefill" in k]
    dec = [v for k, v in flip_map.items() if "decode" in k]
    nf, nd = sum(pre), sum(dec)
    if len(pre) == 2 and nf == 2 and nd == 0:
        return "сдвиг подтверждён (prefill flip 2/2, decode flip 0/2: %s)" % (
            {k: int(v) for k, v in sorted(flip_map.items())})
    if not any(flip_map.values()):
        return "сдвиг отвергнут (вершина == emitted везде, flip 0/4: %s)" % (
            {k: int(v) for k, v in sorted(flip_map.items())})
    return ("смешанный: prefill flip %d/%d, decode flip %d/%d — ни «подтверждён», "
            "ни «отвергнут» не выполняется (map: %s)" % (
                nf, len(pre), nd, len(dec), {k: int(v) for k, v in sorted(flip_map.items())}))


def finish(R, rc, halt, t0):
    R["rc"], R["halt_reason"] = rc, halt
    R["t_total_s"] = round(time.time() - t0, 2)
    R = san(R)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "Q2c.json"), "w") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)
    print("E94 q2c rc=%d%s" % (rc, (" halt=" + halt) if halt else ""), flush=True)
    print("RUN_RC=%d" % rc, flush=True)
    return rc


def run_q2c():
    t0 = time.time()
    R = {"phase": "q2c", "started_unix": int(t0), "doc": "w4", "reqs": []}
    srv = None
    try:
        ids448 = load_ids448()
        q1 = json.load(open(Q1ART))
        tpl = q1.get("body_template") or {}
        for f in ("temperature", "stream", "return_tokens", "top_logprobs", "seed",
                  "top_k", "top_p", "min_p", "n_probs"):
            if f not in tpl:
                return finish(R, 1, "Q1 body_template без поля %s" % f, t0)
        q2b = json.load(open(Q2B))
        a_top1 = (q2b.get("A", {}).get("top1_ids") or [None])[0]
        R["a_top1_from_q2b_k70"] = a_top1
        port = pick_port(PORTS)
        srv = LlamaServer(port, SERVER_LOG)
        R["port"] = int(port)
        R["server_cmd"] = [LLAMA_BIN] + SERVER_ARGS + ["--port", str(port), "--log-disable"]
        srv.start()
        R["server_pid"] = int(srv.pid)
        print("E94 q2c: server pid=%d port=%d, жду /health" % (srv.pid, port), flush=True)
        srv.wait_health(timeout=300)
        R["t_load_s"] = round(time.time() - t0, 2)
        R["fd_after_health"] = srv.device_fd()
        # 1) tokenize оба промпта (tokenize → (resp, dt))
        tk_a, _ = srv.tokenize(TXT_A)
        tk_b, _ = srv.tokenize(TXT_B)
        ids_a, ids_b = tk_a.get("tokens"), tk_b.get("tokens")
        if not isinstance(ids_a, list) or not isinstance(ids_b, list):
            return finish(R, 1, "tokenize: не массив id (a=%r b=%r)"
                          % (tk_a.get("error"), tk_b.get("error")), t0)
        R["tokenize"] = {"A_text": TXT_A, "A_ids": san(ids_a),
                         "B_text": TXT_B, "B_ids": san(ids_b),
                         "raw": san([tk_a, tk_b])}
        print("tokenize: A=%d id, B=%d id (B ⊂-префикс A: %s)" % (
            len(ids_a), len(ids_b), list(ids_b) == list(ids_a)[:len(ids_b)]), flush=True)
        # 2) "…is", n_predict=1, top_logprobs=5 (реальное поле — n_probs)
        body2 = build_body(tpl, ids_a, 1, top_logprobs=5, n_probs=5)
        ok, out2, err = run_single(srv, R, 0, body2, "req2_is_np1")
        if not ok:
            return finish(R, 1, err, t0)
        resp2, dt2, rows2 = out2
        p0 = (resp2.get("completion_probabilities") or [{}])[0]
        R["req2"] = {"prompt_len": len(ids_a), "n_predict": 1,
                     "emitted": p0.get("id"), "top1": (p0.get("top_logprobs") or [{}])[0].get("id"),
                     "list": full_list(p0, 5), "tokens_evaluated": resp2.get("tokens_evaluated")}
        # 3) "…France", n_predict=3, top_logprobs=5
        body3 = build_body(tpl, ids_b, 3, top_logprobs=5, n_probs=5)
        ok, out3, err = run_single(srv, R, 1, body3, "req3_france_np3")
        if not ok:
            return finish(R, 1, err, t0)
        resp3, dt3, rows3 = out3
        per3 = [{"pos": r["pos"],
                 "stage": "prefill" if r["pos"] == 0 else "decode",
                 "emitted": r["emitted"], "top1_of_list": r["top1_of_list"],
                 "flip": (r["top1_of_list"] is not None and r["emitted"] != r["top1_of_list"])}
                for r in rows3]
        R["req3"] = {"prompt_len": len(ids_b), "n_predict": 3,
                     "per_position": per3, "tokens_evaluated": resp3.get("tokens_evaluated")}
        # 4) контроль w4: ids448[:69], n_predict=2, top_logprobs=20 (канон)
        body4 = build_body(tpl, ids448[:69], 2)
        ok, out4, err = run_single(srv, R, 2, body4, "req4_w4_69_np2")
        if not ok:
            return finish(R, 1, err, t0)
        resp4, dt4, rows4 = out4
        per4 = [{"pos": r["pos"], "k": 69 + r["pos"],
                 "emitted": r["emitted"], "top1_of_list": r["top1_of_list"],
                 "flip": (r["top1_of_list"] is not None and r["emitted"] != r["top1_of_list"])}
                for r in rows4]
        t0_4 = per4[0]["top1_of_list"] if len(per4) > 0 else None
        t1_4 = per4[1]["top1_of_list"] if len(per4) > 1 else None
        R["req4"] = {"prompt_len": 69, "n_predict": 2, "per_position": per4,
                     "tokens_evaluated": resp4.get("tokens_evaluated"),
                     "pos0_top1_eq_A_top1": (t0_4 == a_top1), "pos0_top1": t0_4,
                     "pos1_top1_eq_A_top1": (t1_4 == a_top1), "pos1_top1": t1_4}
        print("req4: pos0 top1=%s (==A? %s), pos1 top1=%s (==A? %s)" % (
            t0_4, t0_4 == a_top1, t1_4, t1_4 == a_top1), flush=True)
        # verdict
        flip_map = {"req2_prefill": bool(R["req2"]["top1"] is not None
                                         and R["req2"]["emitted"] != R["req2"]["top1"])}
        for r in per3:
            flip_map["req3_%s_pos%d" % (r["stage"], r["pos"])] = r["flip"]
        R["verdict_shift"] = verdict_shift(flip_map)
        print("verdict_shift: %s" % R["verdict_shift"], flush=True)
        return finish(R, 0, None, t0)
    except Exception:
        traceback.print_exc()
        return finish(R, 1, "exception (см. traceback выше)", t0)
    finally:
        if srv is not None:
            srv.kill()


if __name__ == "__main__":
    sys.exit(run_q2c())
