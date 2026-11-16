#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 S6: пересвип Q2 с dry_multiplier=0.0 (единственное изменение канона).
TZ: docs/dynamic/0_PACKET_E94_S6_TZ.md

База: results/e94/Q2.json (14 флипов из 2304, teacher forcing, 6 доков × k=64..447).
Гипотеза H0-DRY: при dry_multiplier=0.0 флипов = 0 ⇒ все 14 флипов Q2 = DRY-артефакт.

ЕДИНСТВЕННОЕ отличие от Q2-канона: dry_multiplier 1.0 -> 0.0. Проверяется в рантайме
(canonical diff == {'dry_multiplier'} иначе rc=14) и ЭХОМ сервера в каждом ответе
(generation_settings.dry_multiplier == 0.0, иначе halt).

Запуск:
  смоук:  python3 experiments/exp94_s6_dry0.py --smoke
  полный: ./run_one.sh e94_s6 "python3 -u experiments/exp94_s6_dry0.py"
  из raw: python3 experiments/exp94_s6_dry0.py --from-raw
RUN_RC: 0 ok · 2 spike/cache · 10 device-гейт · 11 ≥2 упавших дока ·
13 prompt-count · 14 канон-дрейф или эхо (зарегистрированный канон не применён)
· 1 иное (бюджет/exception/server died).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys
import tarfile
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp94_sycl_common import (LlamaServer, pick_port, san, check_schema,
                               check_prompt_count, bootstrap_ci_median)
from exp94_sycl_stream import fd_sampler
from exp94_q2 import (DOCS, KS, SPIKE_ORDER, BUDGET_S, CANON_FIELDS,
                      FD_SAMPLES, FD_INTERVAL,
                      load_data, spike_cache, g2_main_check)
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "results/e94"
RAW = "/tmp/e94s6raw"
SMOKE_RAW = "/tmp/e94s6smoke"
Q2ART = os.path.join(REPO, "results/e94/Q2.json")
Q1ART = os.path.join(REPO, "results/e94/Q1.json")
Q2RAW = os.path.join(REPO, "results/e94/raw_q2.json.tar.gz")
SERVER_LOG = "/tmp/e94s6_server.log"
PORTS = [8111, 8112]
DRY_OFF = 0.0

R = {"phase": "s6", "rc": -1, "halt_reason": None}


def canon_sha(obj):
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_body_s6(template, prompt):
    """канон Q2 (body_template Q1 + prompt + n_predict=1 + cache_prompt=true)
    с ЕДИНСТВЕННЫМ изменением dry_multiplier=0.0."""
    b = dict(template)
    b["prompt"] = list(prompt)
    b["n_predict"] = 1
    b["cache_prompt"] = True
    b["dry_multiplier"] = DRY_OFF
    return b


def get_raw_s6(srv, path, body, fd_first=None):
    """идемпотентный запрос с записью raw в ЯВНЫЙ путь (S6). ВНИМАНИЕ: get_raw
    из exp94_q2 пишет в /tmp/e94raw (save_raw с константой sycl_stream) — для S6
    не пригоден: rebuild читает другой каталог. Логика — дословно get_raw,
    только save в path."""
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(san({"request": body, "response": resp, "wall_s": round(dt, 3)}),
                  f, ensure_ascii=False, indent=1)
    return resp, False


def check_canon_diff(tpl):
    """§2 ТЗ: diff канона Q2 и S6 обязан быть ровно {'dry_multiplier': (1.0, 0.0)}.
    → (ok, diff_fields, s6_sha, q2_sha)."""
    q2 = dict(tpl)
    q2.update({"prompt": [], "n_predict": 1, "cache_prompt": True})
    s6 = build_body_s6(tpl, [])
    diff = {}
    for k in sorted(set(q2) | set(s6)):
        if q2.get(k) != s6.get(k):
            diff[k] = [q2.get(k), s6.get(k)]
    ok = diff == {"dry_multiplier": [1.0, DRY_OFF]}
    return ok, diff, canon_sha(s6), canon_sha(q2)


def check_echo(resp):
    """эхо сервера: generation_settings.dry_multiplier обязан быть 0.0 (как в S5).
    Проверяется ВСЕГДА, включая кэш-хит: кэш от старого канона (dry=1.0) —
    halt, а не молчаливое чтение чужих данных."""
    gs = resp.get("generation_settings") or {}
    v = gs.get("dry_multiplier")
    if v is None:
        return False, "generation_settings.dry_multiplier отсутствует"
    if abs(float(v) - DRY_OFF) > 1e-12:
        return False, "echo dry_multiplier=%r != 0.0" % v
    return True, None


def position_metrics_s6(resp, k, gen_ids, gen_p1):
    """§4 ТЗ: flip = rank эмиссента в СОБСТВЕННОМ отчёте (top_logprobs) ≠ 1;
    'absent' — флип (отдельный счёт). Перекрёстно: cpu_id (REPLAY), top1_agree_cpu.
    g12 = logit-разрыв top2 отчёта. → (m, None) или (None, err)."""
    cp = resp.get("completion_probabilities")
    if not isinstance(cp, list) or not cp:
        return None, "no completion_probabilities"
    p0 = cp[0]
    if not isinstance(p0, dict) or not isinstance(p0.get("logprob"), (int, float)):
        return None, "cp[0] keys=%s (logprob не число)" % list(p0 or {})
    j = k - 64
    emitted = p0.get("id")
    cpu_id = int(gen_ids[j])
    p_max = math.exp(p0["logprob"]) if p0["logprob"] > -745.0 else 0.0
    tl = p0.get("top_logprobs")
    if not (isinstance(tl, list) and tl):
        # 'no_probs' (пустой top_logprobs) — НЕ флип, конвенция Q2rank
        # (флип = rank not in (1, "no_probs")); отдельный счёт.
        return {"k": k, "j": j, "emitted": emitted, "top1_reported": None,
                "rank_reported": "no_probs", "g12": None, "cpu_id": cpu_id,
                "top1_agree_cpu": bool(emitted == cpu_id),
                "p_max_gpu": p_max, "d_pmax": p_max - float(gen_p1[j]),
                "flip": False, "n_absent": 0, "top20_ids": []}, None
    ids20 = [t.get("id") for t in tl]
    lps20 = [t.get("logprob") for t in tl]
    if emitted in ids20:
        rank_reported = ids20.index(emitted) + 1
        n_absent = 0
    else:
        rank_reported = "absent"  # эмитента нет в НЕПУСТОМ списке — флип
        n_absent = 1
    top1_reported = ids20[0]
    g12 = (lps20[0] - lps20[1]) if len(lps20) > 1 else None
    return {"k": k, "j": j, "emitted": emitted, "top1_reported": top1_reported,
            "rank_reported": rank_reported, "g12": g12, "cpu_id": cpu_id,
            "top1_agree_cpu": bool(emitted == cpu_id),
            "p_max_gpu": p_max, "d_pmax": p_max - float(gen_p1[j]),
            "flip": (rank_reported != 1), "n_absent": n_absent, "top20_ids": ids20}, None


def baseline_q2_flips():
    """база флипов Q2 — ТЕМ ЖЕ ПРОЦЕДУРОМ (position_metrics_s6), что и S6,
    пересчитанная из committed raw (results/e94/raw_q2.json.tar.gz).
    Кросс-чек с артефактом Q2.json (14 по top1_agree — сравнение с CPU-референсом):
    в этом корпусе два определения совпадают, но вердикт сравнивает наборы,
    посчитанные ОДНОЙ процедурой (AGENTS #1).
    → (flip_set, artifact_set, agree, n_raw, q2_emitted: {tag: id эмиссента на Q2-флип-позициях})."""
    data = load_data()
    flip_set, q2_emitted, rows_total = {}, {}, 0
    with tarfile.open(Q2RAW) as tf:
        members = {os.path.basename(n): n for n in tf.getnames() if n.endswith(".json")}
        for doc in DOCS:
            ks = []
            for k in KS:
                key = "q2_%s_k%d.json" % (doc, k)
                if key not in members:
                    continue
                resp = json.load(tf.extractfile(members[key]))["response"]
                m, err = position_metrics_s6(resp, k, data[doc]["gen_ids"], data[doc]["gen_p1"])
                if m is None:
                    continue
                rows_total += 1
                if m["flip"]:
                    ks.append(k)
                    q2_emitted["%sk%d" % (doc, k)] = m["emitted"]  # тот же формат тега, что в verdict_rules
            flip_set[doc] = ks
    q2 = json.load(open(Q2ART))
    artifact_set = {d: [m["k"] for m in q2["per_position"][d] if not m["top1_agree"]] for d in DOCS}
    flat_a = {f"{d}k{k}" for d, ks in flip_set.items() for k in ks}
    flat_b = {f"{d}k{k}" for d, ks in artifact_set.items() for k in ks}
    return flip_set, artifact_set, (flat_a == flat_b), rows_total, q2_emitted


def flip_sets(per_position):
    out = {}
    for doc, rows in per_position.items():
        out[doc] = [m["k"] for m in rows if m["flip"]]
    return out


def verdict_rules(s6_set, q2_set, run_ok, per_position=None):
    """§4 ТЗ, строго по приоритету R1 -> R2 -> R3.
    run_ok = полный прогон (6/6 дока, без halt): на неполных данных вердикт не
    считается (R1 мог бы сработать на частичном наборе).
    resolved_by_top1_agree_cpu — разбивка RESOLVED по S6-отчёту (MINOR-1):
    позиции с вершиной≠cpu — шумовая составляющая, не DRY-объяснение."""
    if not run_ok:
        return {"rule": "HALT", "H0_dry": False, "n_s6_flips": None,
                "n_q2_flips": None, "resolved": None, "residual": None, "new": None,
                "resolved_by_top1_agree_cpu": None,
                "detail": "прогон неполный (halt или упавший док) — правила R1–R3 неприменимы"}
    flat_s6 = {f"{d}k{k}" for d, ks in s6_set.items() for k in ks}
    flat_q2 = {f"{d}k{k}" for d, ks in q2_set.items() for k in ks}
    resolved = sorted(flat_q2 - flat_s6)
    residual = sorted(flat_q2 & flat_s6)
    new = sorted(flat_s6 - flat_q2)
    agree = {"true": 0, "false": 0, "missing": 0}
    for tag in resolved:
        d, ks = tag.split("k", 1)
        row = next((m for m in (per_position or {}).get(d, []) if m["k"] == int(ks)), None)
        agree["true" if row and row["top1_agree_cpu"] else ("false" if row else "missing")] += 1
    rb = ({"agree": agree["true"], "disagree": agree["false"], "missing": agree["missing"]}
          if resolved else None)
    if len(flat_s6) == 0:
        rule, h0 = "R1", True
        detail = ("H0-DRY подтверждена: 0 флипов при dry_multiplier=0.0 — все 14 "
                  "флипов Q2-свопа = DRY-артефакт; линия учительского форсинга "
                  "переписывается как измерение DRY "
                  "(разбивка RESOLVED по top1_agree_cpu: %s agree / %s disagree — "
                  "disagree — шумовая составляющая, не DRY-объяснение)"
                  % (agree["true"], agree["false"]))
    elif not new:
        rule, h0 = "R2", False
        detail = ("остаток %d/%d (все из Q2_FLIPS): %s — для отдельного объяснения"
                  % (len(residual), len(flat_q2), residual))
    else:
        rule, h0 = "R3", False
        detail = ("NEW-флипы (чистые в Q2): %s — DRY-off ввёл новый эффект; "
                  "вывод «всё = DRY» блокируется" % new)
    return {"rule": rule, "H0_dry": h0, "n_s6_flips": len(flat_s6),
            "n_q2_flips": len(flat_q2),
            "resolved": resolved, "residual": residual, "new": new,
            "resolved_by_top1_agree_cpu": rb,
            "detail": detail}


def rebuild_from_raw(raw_dir):
    """per_position + summary + doc_status + G2 (мягкий) ИСКЛЮЧИТЕЛЬНО из raw
    (одна процедура для live и from-raw, AGENTS #1)."""
    data = load_data()
    per_position, status = {}, {}
    g2_viol, g2_ratio = [], {}
    max_pn_2nd, n_pn_2nd = None, 0
    n_echo_checked, n_echo_bad = 0, 0
    for doc in DOCS:
        rows, missing, bad = [], 0, 0
        pwalls, ppn = {}, {}
        for k in KS:
            path = os.path.join(raw_dir, "s6_%s_k%d.json" % (doc, k))
            if not os.path.exists(path):
                missing += 1
                continue
            r = json.load(open(path))
            resp = r["response"]
            n_echo_checked += 1
            ok_echo, _ = check_echo(resp)  # офлайн из raw (защита от чужого канона)
            if not ok_echo:
                n_echo_bad += 1
            if isinstance(r.get("wall_s"), (int, float)):
                pwalls[k] = float(r["wall_s"])
            pn = (resp.get("timings") or {}).get("prompt_n")
            if isinstance(pn, int):
                ppn[k] = pn
                if k > 64:
                    n_pn_2nd += 1
                    max_pn_2nd = pn if max_pn_2nd is None else max(max_pn_2nd, pn)
            if resp.get("tokens_evaluated") != k:  # G1 из raw
                bad += 1
                continue
            m, err = position_metrics_s6(resp, k, data[doc]["gen_ids"], data[doc]["gen_p1"])
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
    # summary (одна процедура; статистика Q2-совместимая + flip-метрики)
    S = {"overall": None, "per_doc": {}}
    for scope, rows in ([("overall", [m for rr in per_position.values() for m in rr])] +
                       [("doc:%s" % d, rr) for d, rr in per_position.items()]):
        if not rows:
            S["overall" if scope == "overall" else scope] = None
            continue
        d = np.array([abs(m["d_pmax"]) for m in rows])
        med, ci = bootstrap_ci_median(d, n=10000, seed=8181)
        out = {"n_pos": len(rows),
               "median_abs_d_pmax": float(med),
               "bootstrap_ci_2.5_97.5": [float(ci[0]), float(ci[1])],
               "top1_agreement_reported": float(sum(1 for m in rows if not m["flip"]) / len(rows)),
               "top1_agreement_cpu": float(sum(1 for m in rows if m["top1_agree_cpu"]) / len(rows)),
               "n_flip": int(sum(1 for m in rows if m["flip"])),
               "n_absent": int(sum(m["n_absent"] for m in rows))}
        if scope == "overall":
            S["overall"] = out
        else:
            S["per_doc"][scope] = out
    n_ok = sum(1 for s in status.values() if s["complete"])
    return {"per_position": per_position, "summary": S, "doc_status": status,
            "n_docs_ok": n_ok, "n_docs_failed": len(DOCS) - n_ok,
            "g2_violations": g2_viol, "g2_violations_n": len(g2_viol),
            "g2_wall_last32_first32": g2_ratio,
            "g2_offline_check": {"max_prompt_n_from_second": max_pn_2nd,
                                 "n_checked_prompt_n_from_second": n_pn_2nd},
            "echo_offline": {"n_checked": n_echo_checked, "n_bad": n_echo_bad,
                             "ok": bool(n_echo_checked > 0 and n_echo_bad == 0)}}


def effect_side(per_position, resolved, q2_emitted):
    """§4 ТЗ: для каждой RESOLVED-позиции — g12, rank_reported (для RESOLVED
    тавтологично 1) и q2_emitted_rank_in_s6 — ранг Q2-эмиссента в top_logprobs
    отчёта S6 ('absent'/'no_probs' по той же конвенции): «насколько далеко
    DRY сдвигал вершину» (k=117: 4383 — rank 3, g12=0.008<1.0)."""
    out = {}
    for tag in resolved:
        doc, ks = tag.split("k", 1)
        k = int(ks)
        for m in per_position.get(doc, []):
            if m["k"] == k:
                qe = q2_emitted.get(tag)
                t20 = m.get("top20_ids") or []
                if not t20:
                    qrank = "no_probs"
                elif qe is None:
                    qrank = None
                else:
                    qrank = t20.index(qe) + 1 if qe in t20 else "absent"
                out[tag] = {"g12": m["g12"], "rank_reported": m["rank_reported"],
                            "emitted": m["emitted"], "top1_reported": m["top1_reported"],
                            "q2_emitted": qe, "q2_emitted_rank_in_s6": qrank}
    return out


def write_artifact(R):
    R = san(R)
    os.makedirs(os.path.join(REPO, OUT), exist_ok=True)
    with open(os.path.join(REPO, OUT, "S6.json"), "w") as f:
        json.dump(R, f, ensure_ascii=False, indent=1, allow_nan=False)


def finish(R, rc, halt, t0):
    R["rc"], R["halt_reason"] = rc, halt
    R["t_total_s"] = round(time.time() - t0, 2)
    write_artifact(R)
    print("E94 s6 rc=%d%s" % (rc, (" halt=" + halt) if halt else ""), flush=True)
    print("RUN_RC=%d" % rc, flush=True)
    return rc


def main_run(srv, R, data, tpl, t0, raw_dir):
    """6 доков × k=64..447; raw raw_dir/s6_<doc>_k<k>.json (идемпотентно);
    halt-каскад: канон-эхо (1), G1 (13), server died, бюджет 40 мин."""
    fd_first, halt, fail_docs = {}, None, []
    echo_first = None
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
            path = os.path.join(raw_dir, "s6_%s_k%d.json" % (doc, k))
            try:
                resp, cached = get_raw_s6(srv, path, build_body_s6(tpl, ids448[:k]), fd_first)
            except Exception as e:
                n_err += 1
                print("%s k=%d: request exception: %s" % (doc, k, e), flush=True)
                if not srv.alive():
                    halt = "server died during %s k=%d (returncode=%s)" % (
                        doc, k, srv.proc.returncode if srv.proc else None)
                    break
                continue
            if echo_first is None:  # эхо ПЕРВОГО запроса — в артефакт (MAJOR-1)
                gs = resp.get("generation_settings") or {}
                echo_first = {"dry_multiplier": gs.get("dry_multiplier"),
                              "dry_base": gs.get("dry_base"),
                              "dry_penalty_last_n": gs.get("dry_penalty_last_n"),
                              "doc": doc, "k": k, "cached": bool(cached)}
            ok_pc, detail = check_prompt_count(resp, k)  # G1 — на каждом
            if not ok_pc:
                halt = "G1 prompt-count гейт (%s k=%d): %s" % (doc, k, detail)
                break
            ok_echo, edetail = check_echo(resp)  # ВСЕГДА (вкл. кэш: защита от stale)
            if not ok_echo:
                halt = ("эхо-гейт (%s k=%d, cached=%s): %s — сервер не отключил DRY "
                        "или кэш от другого канона" % (doc, k, cached, edetail))
                break
            ok_sc, detail = check_schema(resp)
            if not ok_sc:
                n_err += 1
                print("%s k=%d schema: %s" % (doc, k, detail), flush=True)
                continue
            if k % 128 == 64:
                print("%s k=%d: cached=%s" % (doc, k, cached), flush=True)
        if n_err > 0 or (halt is None and any(
                not os.path.exists(os.path.join(raw_dir, "s6_%s_k%d.json" % (doc, k))) for k in KS)):
            fail_docs.append(doc)
            print("%s: не полный (n_err=%d)" % (doc, n_err), flush=True)
    R["fd_first_request"] = san({"n_samples": len(fd_first.get("samples", [])),
                                  "n_samples_with_dri": fd_first.get("n_with_dri"),
                                  "dri_union": fd_first.get("dri_union")})
    R["device_gate_source"] = "live" if "th" in fd_first else "reused_raw"
    R["echo_first_request"] = san(echo_first)
    return halt, fail_docs


def run_s6(raw_dir, spike_doc="w0"):
    t0 = time.time()
    R.update({"phase": "s6", "tz": "docs/dynamic/0_PACKET_E94_S6_TZ.md",
              "started_unix": int(t0), "docs": list(DOCS),
              "per_position": {}, "summary": {}, "rc": -1, "halt_reason": None})
    srv = None
    try:
        data = load_data()
        q1 = json.load(open(Q1ART))
        tpl = q1.get("body_template") or {}
        for f in CANON_FIELDS:
            if f not in tpl:
                return finish(R, 1, "Q1 body_template без поля %s" % f, t0)
        # §2: канон-дифф ДО первого запроса
        ok, diff, s6_sha, q2_sha = check_canon_diff(tpl)
        R["canon"] = {"s6_sha": s6_sha, "q2_sha": q2_sha, "diff_fields": diff}
        if not ok:
            return finish(R, 14, "канон-дрейф: diff != {'dry_multiplier'}: %s" % diff, t0)
        print("canon diff ok: %s" % diff, flush=True)
        # база Q2: те же 14 флипов, но пересчитанные из raw ОДНОЙ процедурой (AGENTS #1)
        q2_set, q2_art_set, agree, n_raw, q2_emitted = baseline_q2_flips()
        R["baseline"] = {"q2_sha16": hashlib.sha1(open(Q2ART, "rb").read()).hexdigest()[:16],
                         "n_positions_raw": n_raw,
                         "n_flip_from_raw": sum(len(v) for v in q2_set.values()),
                         "n_flip_artifact": sum(len(v) for v in q2_art_set.values()),
                         "raw_eq_artifact": bool(agree),
                         "flip_set": q2_set,
                         "flip_set_artifact": q2_art_set,
                         "q2_emitted": q2_emitted}
        port = pick_port(PORTS)
        srv = LlamaServer(port, SERVER_LOG)
        R["port"] = int(port)
        srv.start()
        R["server_pid"] = int(srv.pid)
        print("E94 s6: server pid=%d port=%d, жду /health" % (srv.pid, port), flush=True)
        srv.wait_health(timeout=300)
        R["t_load_s"] = round(time.time() - t0, 2)
        fd_health = srv.device_fd()
        R["fd_after_health"] = fd_health
        if not fd_health:
            return finish(R, 10, "device-гейт: нет /dev/dri fd в /proc/%d/fd после health" % srv.pid, t0)
        print("health ok, load=%.1fs, fd=%s" % (R["t_load_s"], fd_health), flush=True)
        ok2, rows2, fails2 = spike_cache(srv, data[spike_doc]["ids448"],
                                         {**tpl, "dry_multiplier": DRY_OFF})
        R["spike_g2"] = san({"order": list(SPIKE_ORDER), "rows": rows2,
                              "pass": bool(ok2), "failures": fails2})
        R["g1_ok"] = bool(ok2)        # явные flat-гейт-поля (контракт §6)
        R["g3_fd"] = bool(fd_health)  # (G3 = device-гейт после health)
        if not ok2:
            rc = 13 if any(f.startswith("G1") for f in fails2) else 2
            return finish(R, rc, ("спайк: " + "; ".join(fails2)) if fails2
                          else "спайк: провал (см. spike_g2)", t0)
        print("G2 ok", flush=True)
        halt, fail_docs = main_run(srv, R, data, tpl, t0, raw_dir)
        R.update(rebuild_from_raw(raw_dir))
        # флипы + вердикт + эффект-сайд (вердикт только на полном прогоне)
        run_ok = (halt is None and R["n_docs_ok"] == len(DOCS))
        s6_set = flip_sets(R["per_position"])
        v = verdict_rules(s6_set, q2_set, run_ok, R["per_position"])
        R["flips"] = {"s6_set": s6_set, "q2_set": q2_set,
                      "resolved": v["resolved"], "residual": v["residual"],
                      "new": v["new"]}
        R["effect_side"] = effect_side(R["per_position"], v["resolved"] or [], q2_emitted)
        R["verdict"] = {k: v[k] for k in ("rule", "H0_dry", "n_s6_flips",
                                           "resolved_by_top1_agree_cpu", "detail")}
        if halt is None and R["n_docs_failed"] >= 2:
            halt = "%d упавших дока (>=2): %s" % (R["n_docs_failed"], sorted(fail_docs))
        if halt is None:
            rc = 0
        elif halt.startswith("G1"):
            rc = 13
        elif halt.startswith("эхо"):
            rc = 14  # зарегистрированный канон (dry=0.0) не применён сервером
        elif halt.startswith("упавших"):
            rc = 11
        else:
            rc = 1
        # raw -> tar (артефакт корпуса, как raw_q2)
        tar_path = os.path.join(REPO, OUT, "raw_s6.json.tar.gz")
        with tarfile.open(tar_path, "w:gz") as tf:
            for p in sorted(glob.glob(os.path.join(raw_dir, "s6_*.json"))):
                tf.add(p, arcname="raw/" + os.path.basename(p))
        R["raw_tar"] = os.path.relpath(tar_path, REPO)
        o = R["summary"]["overall"]
        print("S6: ok=%d/%d n_flip_s6=%d rule=%s H0_dry=%s top1_agreement_reported=%s" % (
            R["n_docs_ok"], len(DOCS), v["n_s6_flips"], v["rule"], v["H0_dry"],
            o["top1_agreement_reported"] if o else None), flush=True)
        print("verdict", json.dumps(R["verdict"], ensure_ascii=False), flush=True)
        return finish(R, rc, halt, t0)
    except Exception:
        traceback.print_exc()
        return finish(R, 1, "exception (см. traceback выше)", t0)
    finally:
        if srv is not None:
            srv.kill()


def run_smoke():
    """§7 ТЗ: w4, k=64..75 (12 запросов, вкл. флип k=70) + канон-дифф.
    Ожидание: 0 флипов; k=70 резолвится (emitted == top1_reported == 318)."""
    t0 = time.time()
    os.makedirs(SMOKE_RAW, exist_ok=True)
    data = load_data()
    q1 = json.load(open(Q1ART))
    tpl = q1.get("body_template") or {}
    ok, diff, s6_sha, q2_sha = check_canon_diff(tpl)
    print("canon diff: %s (ok=%s)" % (diff, ok), flush=True)
    if not ok:
        print("SMOKE_FAIL canon")
        return 1
    port = pick_port(PORTS)
    srv = LlamaServer(port, SERVER_LOG + ".smoke")
    srv.start()
    try:
        srv.wait_health(timeout=300)
        ids448 = data["w4"]["ids448"]
        fails = []
        for k in range(64, 76):
            path = os.path.join(SMOKE_RAW, "s6_w4_k%d.json" % k)
            resp, cached = get_raw_s6(srv, path, build_body_s6(tpl, ids448[:k]), None)
            ok_pc, detail = check_prompt_count(resp, k)
            ok_echo, edetail = check_echo(resp)
            m, err = position_metrics_s6(resp, k, data["w4"]["gen_ids"], data["w4"]["gen_p1"])
            print("k=%d cached=%s pc=%s echo=%s emitted=%s top1_reported=%s rank=%s flip=%s" % (
                k, cached, ok_pc, ok_echo, m["emitted"] if m else None,
                m["top1_reported"] if m else None, m["rank_reported"] if m else None,
                m["flip"] if m else None), flush=True)
            if not ok_pc:
                fails.append("k=%d G1: %s" % (k, detail))
            if not ok_echo:
                fails.append("k=%d echo: %s" % (k, edetail))
            if m is not None and m["flip"]:
                fails.append("k=%d флип (ожидание 0)" % k)
        # TZ §7: k=70 обязан резолвиться на вершину S4e (318); raw уже в кэше — чтение
        resp70, _ = get_raw_s6(srv, os.path.join(SMOKE_RAW, "s6_w4_k70.json"),
                               build_body_s6(tpl, ids448[:70]), None)
        m70, _ = position_metrics_s6(resp70, 70, data["w4"]["gen_ids"], data["w4"]["gen_p1"])
        if m70 is None or m70["emitted"] != 318 or m70["top1_reported"] != 318:
            fails.append("k=70 не резолвился на 318: %s" % (m70,))
        if fails:
            print("SMOKE_FAIL:", "; ".join(fails), flush=True)
            return 1
        print("SMOKE_OK (12/12, 0 флипов, k=70 резолвлен)", flush=True)
        return 0
    finally:
        srv.kill()


def run_from_raw():
    """--from-raw: пересборка из /tmp/e94s6raw + идемпотентность ×2 + офлайн-
    эхо-чек из raw (raw чужого канона → HALT/rc=14, MAJOR-1)."""
    t0 = time.time()
    art = os.path.join(REPO, OUT, "S6.json")
    if os.path.exists(art):
        R = json.load(open(art))
    else:
        R = {"phase": "s6", "tz": "docs/dynamic/0_PACKET_E94_S6_TZ.md",
             "started_unix": int(t0), "rc": 1,
             "halt_reason": "нет S6.json от прогона"}
    d1 = rebuild_from_raw(RAW)
    sha1 = canon_sha(d1)
    d2 = rebuild_from_raw(RAW)
    sha2 = canon_sha(d2)
    R.update(d2)
    echo_ok = bool((R.get("echo_offline") or {}).get("ok"))
    if "baseline" in R:
        q2_set = R["baseline"]["flip_set"]
        q2_emitted = R["baseline"].get("q2_emitted") or {}
        if echo_ok:
            v = verdict_rules(flip_sets(R["per_position"]), q2_set,
                              R.get("n_docs_ok") == len(DOCS), R["per_position"])
        else:
            off = R.get("echo_offline") or {}
            v = {"rule": "HALT", "H0_dry": False, "n_s6_flips": None, "n_q2_flips": None,
                 "resolved": None, "residual": None, "new": None,
                 "resolved_by_top1_agree_cpu": None,
                 "detail": ("офлайн-эхо: raw пуст/отсутствует (n_checked=0) — "
                            "правила R1–R3 неприменимы" if off.get("n_checked", 0) == 0
                            else "офлайн-эхо-чек не прошёл (raw чужого канона, "
                                 "n_bad=%d) — правила R1–R3 неприменимы" % off.get("n_bad", 0))}
        R["flips"] = {"s6_set": flip_sets(R["per_position"]), "q2_set": q2_set,
                      "resolved": v["resolved"], "residual": v["residual"], "new": v["new"]}
        R["effect_side"] = effect_side(R["per_position"], v["resolved"] or [], q2_emitted)
        R["verdict"] = {k: v[k] for k in ("rule", "H0_dry", "n_s6_flips",
                                           "resolved_by_top1_agree_cpu", "detail")}
    R["idempotence"] = {"runs": 2, "artifact_identical": bool(sha1 == sha2),
                        "sha256": sha2}
    R["t_rebuild_s"] = round(time.time() - t0, 2)
    if not echo_ok:
        R["rc"] = 14
    else:
        R["rc"] = 0 if R.get("n_docs_failed", 1) == 0 else 1
    write_artifact(R)
    print("E94 s6 from-raw: ok=%s idempotent=%s sha=%s rc=%d" % (
        R.get("n_docs_ok"), R["idempotence"]["artifact_identical"], sha2, R["rc"]), flush=True)
    print("RUN_RC=%d" % R["rc"], flush=True)
    return int(R["rc"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--from-raw", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        sys.exit(run_smoke())
    if a.from_raw:
        sys.exit(run_from_raw())
    sys.exit(run_s6(RAW))
