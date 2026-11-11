#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 spike: iGPU SYCL llama-server — hard-гейты + доступность top-k (ТЗ r5.2 + steering).
Hard-гейты: device-fd(сэмплирование в decode)→10, prompt-count→13, структурная
схема/кэш/tokenize→2, успех→0. Доступность top-k — ЗАПИСЬ (не hard-гейт): свип
top_logprobs {2,5,20} + реальное поле n_probs {2,5}, (A)/(B) с ренормировкой p_max,
(C) жадность (B)-тела против Q1-тела. Q1/Q2/Q3-метрики = TODO-заглушки (spike-only).
Raw → /tmp/e94raw/*; числа → results/e94/SPIKE.json + FORMAT.md. sys.exit(rc) —
чтобы RUN_RC скрипта == RUN_RC обёртки run_one.
"""
import argparse, json, math, os, sys, threading, time, traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp94_sycl_common import (LLAMA_BIN, GGUF, SERVER_ARGS, LlamaServer, pick_port,
                               san, base_body, check_schema, check_prompt_count,
                               slot_cache_ok)
import numpy as np

RAW = "/tmp/e94raw"
OUT = "results/e94"
PROMPTS = "results/e93/prompts.json"
PORTS = [8110, 8111, 8112]
SERVER_LOG = "/tmp/e94spike_server.log"
FD_SAMPLES, FD_INTERVAL = 20, 0.05  # steering: 20 сэмплов × 0.05 с во время decode


def save_raw(name, obj):
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, name), "w") as f:
        json.dump(san(obj), f, ensure_ascii=False, indent=1)


def schema_tree(obj, prefix=""):
    """полные имена ключей (FORMAT.md); в массивах — первые 2 элемента."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = "%s.%s" % (prefix, k) if prefix else str(k)
            out.append(p)
            out.extend(schema_tree(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:2]):
            out.extend(schema_tree(v, "%s[%d]" % (prefix, i)))
    return out


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return {"idx": i, "a": x, "b": y}
    if len(a) != len(b):
        return {"idx": min(len(a), len(b)), "note": "len mismatch", "a": len(a), "b": len(b)}
    return None


def pos_stats(cp, n):
    """на позицию: id, logprob (p_max = exp(logprob)), len(top_logprobs), сумма p по списку.
    logprob в JSON может быть float('-inf')-заменой (lowest) — exp безопасен."""
    out = []
    for i in range(min(n, len(cp))):
        p = cp[i]
        lp = p.get("logprob")
        tl = p.get("top_logprobs", []) or []
        ps = []
        for e in tl:
            v = e.get("logprob")
            ps.append(math.exp(v) if isinstance(v, (int, float)) and math.isfinite(v)
                      else (0.0 if v == float("-inf") else None))
        out.append({"pos": i, "id": p.get("id"), "logprob": lp,
                    "p_max": math.exp(lp) if isinstance(lp, (int, float)) and math.isfinite(lp)
                    else (0.0 if lp == float("-inf") else None),
                    "n_tl": len(tl),
                    "sum_list": float(sum(x for x in ps if x is not None)) if ps else None,
                    "all_finite": all(x is not None for x in ps)})
    return out


def len_hist(stats):
    return dict(Counter(s["n_tl"] for s in stats))


def fd_sampler(pid, n_samples, interval, out):
    """сэмплирование /proc/<pid>/fd потоком (steering): dri в любом сэмпле → гейт пройден."""
    for _ in range(n_samples):
        dri = []
        try:
            for f in os.listdir("/proc/%d/fd" % pid):
                try:
                    t = os.readlink("/proc/%d/fd/%s" % (pid, f))
                except OSError:
                    continue
                if t.startswith("/dev/dri/"):
                    dri.append(t)
        except (FileNotFoundError, ProcessLookupError):
            pass
        out.append(dri)
        time.sleep(interval)


def do_spike(R):
    """возвращает (rc, halt_reason); все числа — из raw-ответов."""
    t0 = time.time()
    p = json.load(open(PROMPTS))
    w0 = next(d for d in p["docs"] if d["key"] == "w0")
    ids64, text0 = list(w0["ids"]), w0["text"]
    if len(ids64) != 64:
        return 2, "w0 ids len=%d != 64 (prompts.json)" % len(ids64)

    port = pick_port(PORTS)
    srv = LlamaServer(port, SERVER_LOG)
    R.update({"port": int(port), "model": os.path.basename(GGUF),
              "server_cmd": [LLAMA_BIN] + SERVER_ARGS + ["--port", str(port), "--log-disable"],
              "started_unix": int(t0)})
    try:
        srv.start()
        R["server_pid"] = int(srv.pid)
        print("E94 spike: server pid=%d port=%d, жду /health" % (srv.pid, port), flush=True)
        srv.wait_health(timeout=300)
        R["t_load_s"] = round(time.time() - t0, 2)
        fd_health = srv.device_fd()
        R["fd_after_health"] = fd_health  # контекст, НЕ гейт
        print("health ok, load=%.1fs, fd after health: %s" % (R["t_load_s"], fd_health), flush=True)

        # (a) основной запрос w0, n_predict=64; fd-сэмплирование ВО ВРЕМЯ запроса
        body = base_body(ids64, 64, cache=False)
        samples = []
        th = threading.Thread(target=fd_sampler, args=(srv.pid, FD_SAMPLES, FD_INTERVAL, samples))
        th.start()
        resp, dt = srv.completion(body, timeout=900)
        th.join()
        save_raw("w0_completion.json", {"request": body, "response": resp, "wall_s": dt})
        n_with = sum(1 for s in samples if s)
        union = sorted({x for s in samples for x in s})
        R["fd_gate"] = {"n_samples": len(samples), "interval_s": FD_INTERVAL,
                        "n_samples_with_dri": int(n_with), "dri_union": union,
                        "request_wall_s": round(dt, 3),
                        "request_timings": san(resp.get("timings")),
                        "ok": n_with > 0}
        print("fd gate: %d/%d samples with /dev/dri, union=%s" % (n_with, len(samples), union),
              flush=True)
        if not R["fd_gate"]["ok"]:
            return 10, ("no /dev/dri fd in %d samples (0.05s) during n_predict=64 decode; "
                        "after-health fd was %s" % (len(samples), fd_health))

        R["main"] = {"wall_s": round(dt, 3), "per_pos_s": round(dt / 64, 4),
                     "n_tokens": len(resp.get("tokens", [])), "tokens": resp.get("tokens", []),
                     "content": resp.get("content"), "tokens_evaluated": resp.get("tokens_evaluated"),
                     "tokens_cached": resp.get("tokens_cached"), "id_slot": resp.get("id_slot")}
        # (b) гейты
        ok, detail = check_prompt_count(resp, 64)
        R["gate_prompt_count"] = {"ok": ok, "detail": detail}
        if not ok:
            return 13, detail
        ok, detail = check_schema(resp)
        R["gate_schema"] = {"ok": ok, "detail": detail}
        if not ok:
            return 2, detail
        tl_lens = [len(x.get("top_logprobs", []) or []) for x in resp.get("completion_probabilities", [])]
        R["gate_top_logprobs_len"] = {"tz_required": 2, "observed": dict(Counter(tl_lens)),
                                      "ok": set(tl_lens) == {2},
                                      "treated_as": "finding (steering): определяет контракт Q2, не hard-гейт"}
        print("top_logprobs len observed: %s (ТЗ требует 2)" % R["gate_top_logprobs_len"]["observed"],
              flush=True)
        gs = resp.get("generation_settings", {})
        want = {"seed": 8181, "temperature": 0.0, "repeat_penalty": 1.0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0, "repeat_last_n": 0,
                "dry_multiplier": 1.0, "dry_base": 1.0, "xtc_probability": 0.0,
                "mirostat": 0, "n_predict": 64}
        got = {k: gs.get(k) for k in want}
        mism = {k: {"want": want[k], "got": got[k]} for k in want if got[k] != want[k]}
        R["gate_generation_settings"] = {"ok": not mism,
                                         "missing_field": "generation_settings" not in resp,
                                         "mismatch": mism, "got": san(got),
                                         "note": "logprobs:true(bool)→n_probs=1; top_logprobs вне legacy-схемы "
                                                 "(server-schema.cpp:179 поле n_probs, alias logprobs)"}
        print("main: wall=%.1fs eval=%r; prompt_count ok; schema ok; gs mism=%s" %
              (dt, resp.get("tokens_evaluated"), mism or "none"), flush=True)

        # (A) свип top_logprobs {2,5,20} (n_predict=4) + реальное поле n_probs {2,5}
        sweep = {}
        for tag, over in (("tl2", {"top_logprobs": 2}), ("tl5", {"top_logprobs": 5}),
                          ("tl20", {"top_logprobs": 20, "n_probs": 20}),
                          ("np2", {"n_probs": 2}), ("np5", {"n_probs": 5})):
            b = base_body(ids64, 4, cache=False, **over)
            r, dtr = srv.completion(b, timeout=600)
            save_raw("sweep_%s.json" % tag, {"request": b, "response": r, "wall_s": dtr})
            st = pos_stats(r.get("completion_probabilities", []), 4)
            sweep[tag] = {"overrides": over, "len_hist": len_hist(st), "pos": san(st),
                          "wall_s": round(dtr, 3)}
            print("sweep %s: len_hist=%s" % (tag, sweep[tag]["len_hist"]), flush=True)
        R["sweep"] = sweep
        A = sweep["tl20"]["pos"]  # (A): цепочка как есть, k=20
        R["A"] = {"len_hist": len_hist(A), "pos": A}

        # (B) снятие прореживания в том же теле + N-пробы (n_probs=400, n_predict=1)
        B_ov = {"top_logprobs": 20, "n_probs": 20, "top_k": 400, "top_p": 1.0,
                "min_p": 0.0, "typical_p": 1.0, "xtc_probability": 0.0}
        b = base_body(ids64, 4, cache=False, **B_ov)
        rB, dtB = srv.completion(b, timeout=600)
        save_raw("B_k4.json", {"request": b, "response": rB, "wall_s": dtB})
        B = pos_stats(rB.get("completion_probabilities", []), 4)
        R["B"] = {"overrides": B_ov, "len_hist": len_hist(B), "pos": B, "wall_s": round(dtB, 3)}
        for tag, ov in (("NprobeA", {"n_probs": 400}),
                        ("NprobeB", {"n_probs": 400, "top_k": 400, "top_p": 1.0, "min_p": 0.0,
                                     "typical_p": 1.0, "xtc_probability": 0.0})):
            bp = base_body(ids64, 1, cache=False, **ov)
            rp, dtp = srv.completion(bp, timeout=600)
            save_raw("%s.json" % tag, {"request": bp, "response": rp, "wall_s": dtp})
            n_tl = len(rp.get("completion_probabilities", [{}])[0].get("top_logprobs", [])) \
                if rp.get("completion_probabilities") else None
            R.setdefault("n_probe", {})[tag] = {"overrides": ov, "N_observed": n_tl}
            print("N probe %s: N_observed=%s" % (tag, n_tl), flush=True)
        pa = [x["p_max"] for x in A if x["p_max"] is not None]
        pb = [x["p_max"] for x in B if x["p_max"] is not None]
        d = [abs(a - b2) for a, b2 in zip(pa, pb)]
        R["d_pmax"] = {"per_pos": san(d),
                       "median_abs_diff": float(np.median(d)) if d else None,
                       "a_renormalized": bool(d and float(np.median(d)) > 0)}
        print("d_pmax: median=%.4e per_pos=%s" % (float(np.median(d)) if d else -1, d), flush=True)

        # (C) жадность: Q1-тело (384) против (B)-тела (384)
        c1b = base_body(ids64, 384, cache=False)
        c1, dt1 = srv.completion(c1b, timeout=1800)
        save_raw("C1_q1_body.json", {"request": c1b, "response": c1, "wall_s": dt1})
        c2b = base_body(ids64, 384, cache=False, **B_ov)
        c2, dt2 = srv.completion(c2b, timeout=1800)
        save_raw("C2_B_body.json", {"request": c2b, "response": c2, "wall_s": dt2})
        t1 = c1.get("tokens", [])
        i2 = [x.get("id") for x in c2.get("completion_probabilities", [])]
        n_cmp = min(len(t1), len(i2))
        agree = sum(1 for x, y in zip(t1[:n_cmp], i2[:n_cmp]) if x == y)
        R["greedy"] = {"n_cmp": int(n_cmp), "agree": int(agree),
                       "frac": (agree / n_cmp) if n_cmp else None,
                       "first_diff": first_diff(t1, i2),
                       "c1_wall_s": round(dt1, 3), "c2_wall_s": round(dt2, 3),
                       "q1_body_is_B": (agree / n_cmp) < 1.0 if n_cmp else False,
                       "note": "C1 = Q1-тело (top_k=40/top_p=0.95/min_p=0.05 из GGUF), C2 = (B)-тело"}
        print("greedy: %d/%d agree (%.1f%%), C1=%.0fs C2=%.0fs" %
              (agree, n_cmp, 100.0 * (agree / n_cmp) if n_cmp else 0.0, dt1, dt2), flush=True)

        # (d) кэш-гейт: k=64/128/192 × 2, /slots после второго (r3a-B3)
        cache_ok_all = True
        for k in (64, 128, 192):
            r2 = None
            for i in (1, 2):
                b = base_body(ids64[:k], 1, cache=True)
                r2, dt2 = srv.completion(b, timeout=600)
                save_raw("cache_k%d_r%d.json" % (k, i), {"request": b, "response": r2, "wall_s": dt2})
            sl = srv.slots()
            save_raw("slots_k%d_r2.json" % k, sl)
            okc, info = slot_cache_ok(sl, r2.get("id_slot"), k)
            R.setdefault("gate_cache", {})[str(k)] = {"wall_s_r2": round(dt2, 3),
                                                       "per_pos_s": round(dt2, 3),
                                                       "slots": san(info), "ok": okc}
            cache_ok_all = cache_ok_all and okc
            print("cache k=%d: ok=%s slots=%s" % (k, okc, info), flush=True)
        if not cache_ok_all:
            return 2, "cache gate failed: %s" % json.dumps(R["gate_cache"], ensure_ascii=False)

        # (e) /tokenize {"content": текст w0} → сверка с HF-пинами
        tk, dtk = srv.tokenize(text0)
        save_raw("tokenize_w0.json", {"request": {"content": text0}, "response": tk, "wall_s": dtk})
        tk_ids = tk.get("tokens", [])
        tk_ok = list(tk_ids) == ids64
        R["gate_tokenize"] = {"ok": tk_ok, "n_tokens": len(tk_ids), "n_pinned": len(ids64),
                              "first_diff": first_diff(tk_ids, ids64), "wall_s": round(dtk, 3)}
        if not tk_ok:
            return 2, "tokenize mismatch with HF pins: %s" % R["gate_tokenize"]["first_diff"]
        print("tokenize ok: %d ids == HF pins" % len(tk_ids), flush=True)

        # (f) выводы по доступности top-k (steering: однозначный)
        n_b = max((x["n_tl"] for x in B), default=0)
        n_A = R["n_probe"].get("NprobeA", {}).get("N_observed")
        n_B = R["n_probe"].get("NprobeB", {}).get("N_observed")
        if n_b == 20:
            R["topk_availability"] = ("top-k на позицию ДОСТУПЕН (k до 20; поле n_probs; тело (B): "
                                      "len=20 на всех позициях, N(B)=%s кандидатов; поле top_logprobs "
                                      "в /completion игнорируется — свип len_hist: %s)"
                                      % (n_B, {t: R["sweep"][t]["len_hist"] for t in ("tl2", "tl5", "tl20", "np2", "np5")}))
            if R["d_pmax"]["a_renormalized"]:
                R["q2_contract"] = ("контракт Q2 = тело (B), потому что p_max(A) ренормирована по "
                                    "прореженному множеству (median |Δp_max|=%.3e)" % R["d_pmax"]["median_abs_diff"])
            else:
                R["q2_contract"] = "p_max(A)==p_max(B) (median |Δ|=0) → ренормировка не влияет; контракт Q2 = (B) на будущее"
        else:
            R["topk_availability"] = ("top-k на позицию НЕДОСТУПЕН (len(B)=%s != 20) ⇒ margin и top-2 hit rate "
                                      "объявляются N/A, Q2 = |Δp_max| + top1_agree; |Δp_max| непригоден, "
                                      "остаётся top1_agree" % n_b)
            R["q2_contract"] = "top1_agree only"
        R["pmax_normalization"] = ("p_max = softmax ПО N КАНДИДАТАМ (не по полному vocab): "
                                   "A: N=%s (top_k=40∩top_p=0.95∩min_p=0.05), B: N=%s (top_k=400)"
                                   % (n_A, n_B))
        R["gates_all_ok"] = True
        R["t_total_s"] = round(time.time() - t0, 2)
        return 0, None
    finally:
        srv.kill()


def write_artifacts(R):
    R = san(R)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "SPIKE.json"), "w") as f:
        json.dump(R, f, ensure_ascii=False, indent=1)

    g = R.get("gate_cache", {})
    per_pos = [v["per_pos_s"] for v in g.values() if v.get("per_pos_s") is not None]
    per_pos.append(R.get("main", {}).get("per_pos_s"))
    per_pos = [v for v in per_pos if v is not None]
    est = max(per_pos) if per_pos else float("nan")
    lines = ["# E94 SPIKE — схема ответа, гейты, доступность top-k", ""]
    lines.append("## Запуск")
    lines.append("- gguf: `%s`" % R.get("model"))
    lines.append("- argv: `%s`" % " ".join(R.get("server_cmd", [])))
    lines.append("- port=%s server_pid=%s load=%.1fs total=%.1fs" %
                 (R.get("port"), R.get("server_pid"), R.get("t_load_s", -1), R.get("t_total_s", -1)))
    fd = R.get("fd_gate", {})
    lines.append("- fd-гейт: %d/%d сэмплов (0.05с) с /dev/dri во время decode (n_predict=64, wall=%.1fs); "
                 "union=%s; fd after health (контекст): %s" %
                 (fd.get("n_samples_with_dri", -1), fd.get("n_samples", -1),
                  fd.get("request_wall_s", -1), fd.get("dri_union"), R.get("fd_after_health")))
    lines.append("")
    lines.append("## Фактическая схема ответа /completion (полные имена ключей, w0)")
    raw = os.path.join(RAW, "w0_completion.json")
    if os.path.exists(raw):
        resp = json.load(open(raw))["response"]
        lines += ["- `%s`" % k for k in schema_tree(resp)]
    lines.append("")
    lines.append("## Гейты")
    for gg in ("fd_gate", "gate_prompt_count", "gate_schema", "gate_top_logprobs_len",
               "gate_generation_settings", "gate_tokenize"):
        lines.append("- %s: %s" % (gg, json.dumps(R.get(gg), ensure_ascii=False)))
    lines.append("- gate_cache:")
    for k in sorted(g, key=int):
        lines.append("  - k=%s: %s" % (k, json.dumps(g[k], ensure_ascii=False)))
    lines.append("")
    lines.append("## Свип top-k (w0, n_predict=4) — len(top_logprobs) по позициям")
    for t in ("tl2", "tl5", "tl20", "np2", "np5"):
        s = R.get("sweep", {}).get(t, {})
        lines.append("- %s (overrides=%s): len_hist=%s" % (t, s.get("overrides"), s.get("len_hist")))
    lines.append("- A (tl20, цепочка как есть): len_hist=%s; N(A)=%s (проба n_probs=400)" %
                 (R.get("A", {}).get("len_hist"), R.get("n_probe", {}).get("NprobeA", {}).get("N_observed")))
    lines.append("- B (tl20+n_probs=20+top_k:400,top_p:1.0,min_p:0.0,typical_p:1.0,xtc:0.0): len_hist=%s; "
                 "N(B)=%s (проба n_probs=400)" %
                 (R.get("B", {}).get("len_hist"), R.get("n_probe", {}).get("NprobeB", {}).get("N_observed")))
    lines.append("")
    lines.append("## (A) vs (B): p_max и ренормировка")
    for tag in ("A", "B"):
        lines.append("- %s: %s" % (tag, json.dumps(
            [{k: x[k] for k in ("pos", "id", "p_max", "n_tl", "sum_list")} for x in R.get(tag, {}).get("pos", [])],
            ensure_ascii=False)))
    dp = R.get("d_pmax", {})
    lines.append("- median |p_max(B)−p_max(A)| = %s (per_pos=%s) → %s" %
                 (dp.get("median_abs_diff"), dp.get("per_pos"),
                  "(A) ренормирована по выжившему множеству" if dp.get("a_renormalized") else "разницы нет"))
    lines.append("")
    lines.append("## (C) жадность")
    gr = R.get("greedy", {})
    lines.append("- C1 (Q1-тело, 384 токов, wall=%.0fs) vs C2 ((B)-тело, 384, wall=%.0fs): agree %d/%d = %.1f%%; "
                 "first_diff=%s; Q1 переводится на (B): %s" %
                 (gr.get("c1_wall_s", -1), gr.get("c2_wall_s", -1), gr.get("agree", -1), gr.get("n_cmp", -1),
                  100.0 * (gr.get("frac") or 0), gr.get("first_diff"), gr.get("q1_body_is_B")))
    lines.append("")
    lines.append("## Выводы (канон)")
    lines.append("- %s" % R.get("topk_availability"))
    lines.append("- %s" % R.get("q2_contract"))
    lines.append("- %s" % R.get("pmax_normalization"))
    lines.append("- КАНОН: тело запроса для Q1/Q2/Q3 = (B) {n_probs:20, top_logprobs:20, top_k:400, top_p:1.0, "
                 "min_p:0.0, typical_p:1.0, xtc_probability:0.0} + поля ТЗ (temperature:0, stream:false, "
                 "return_tokens:true, cache_prompt, seed:8181, нейтральные штрафы), потому что только в (B) "
                 "len(top_logprobs)=20 стабилен, p_max нормирована по фиксированному N=%s, а жадность совпадает "
                 "с Q1-телом на %s" % (R.get("n_probe", {}).get("NprobeB", {}).get("N_observed"),
                                      ("100%%" if (R.get("greedy", {}).get("frac") or 0) == 1.0
                                       else "%.1f%%" % (100.0 * (R.get("greedy", {}).get("frac") or 0)))))
    lines.append("")
    lines.append("## Время на позицию (клиентский wall-clock, с)")
    lines.append("- main (n_predict=64): wall=%.1f → per_pos=%.4f" %
                 (R.get("main", {}).get("wall_s", -1), R.get("main", {}).get("per_pos_s", -1)))
    for k in sorted(g, key=int):
        lines.append("- cache k=%s: per_pos=%.3f" % (k, g[k]["per_pos_s"]))
    lines.append("")
    lines.append("## Рекомендованный скоуп Q2")
    lines.append("t_per_pos ≈ %.4f с (max). Полный: 13 точек k × 16 доков = 208 запросов (decode ≈ %.0f с); "
                 "минимальный: 2 дока (w0, w8) × 13 точек. Сторона transformers: fp32 .venv-референс "
                 "(валидирован: max|Δ| против REPLAY.gen_p1 = 2.2e-6, top1 6144/6144) — teacher-forced один "
                 "forward на (prompt+gen) даёт p на всех k сразу, поэтому Q2 не ограничено 90 мин." %
                 (est, 208 * est))
    lines.append("")
    lines.append("## Отклонения")
    lines.append("- `--slots` оставлен явно, но он включён ПО УМОЛЧАНИЮ (common/common.h:671 "
                 "endpoint_slots = true).")
    lines.append("- тело запроса: `top_logprobs` не является полем legacy-/completion-схемы и игнорируется; "
                 "реальное поле — `n_probs` (alias `logprobs`; bool true → 1). Фиксируется свипом.")
    lines.append("- серверный лог: %s; raw: %s/*.json" % (SERVER_LOG, RAW))
    lines.append("")
    lines.append("## Метрики Q1/Q2/Q3")
    lines.append("- TODO-заглушки в этом прогоне (spike only); см. docs/dynamic/0_E94_EXEC_NOTES.md.")
    if R.get("halt_reason"):
        lines += ["", "## HALT", "- rc=%s halt_reason=%s" % (R.get("rc"), R["halt_reason"]), ""]
    with open(os.path.join(OUT, "FORMAT.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--from-raw", action="store_true",
                    help="q1: пересборка артефакта из /tmp/e94raw без сервера")
    a = ap.parse_args()
    if a.phase == "q1":
        from exp94_q1 import run_q1, run_q1_from_raw
        sys.exit(run_q1_from_raw() if a.from_raw else run_q1())
    if a.phase != "spike":
        print("E94: phase=%s не реализован в этом прогоне исполнителя (spike only);"
              " см. 0_E94_EXEC_NOTES.md" % a.phase)
        print("RUN_RC=1")
        sys.exit(1)
    R = {"phase": "spike", "started_unix": int(time.time())}
    rc, halt = 1, None
    try:
        rc, halt = do_spike(R)
    except Exception:
        traceback.print_exc()
        rc, halt = 1, "exception (см. traceback выше)"
    R["rc"], R["halt_reason"] = rc, halt
    try:
        write_artifacts(R)
    except Exception:
        traceback.print_exc()
        print("E94: не удалось записать артефакты", flush=True)
    print("E94 spike rc=%d%s" % (rc, (" halt=" + halt) if halt else ""), flush=True)
    print("RUN_RC=%d" % rc, flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
