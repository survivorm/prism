#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 S4d — какой именно сэмплер возвращает флип (продолжение S4c, правило 1).

S4c (results/e94/S4c.json, предрегистр 2_REVIEW_E94_probeS3S2S4_exec_r2.md §4) показал:
со штатной цепью на Q2 w13 k=117 эмиссент = 4383 при вершине отчёта 3039, а с цепью
[top_k, temperature] эмиссент = 3039, т.е. флип создаёт цепь. Здесь — локализация.

Предрегистр (записан до кода и до запуска). Адрес тот же, payload тот же, температура 0,
seed 8181. Цепи (из canonical-имён common/sampling.cpp:812-825, подаются МАССИВОМ):

  S4d-pen : [penalties,    top_k, temperature]
  S4d-dry : [dry,          top_k, temperature]
  S4d-sig : [top_n_sigma,  top_k, temperature]
  S4d-bare-rep : [top_k, temperature]            — контроль шума тем же миром

Правило:
  1. цепь вернула 4383 (архивный флип-эмиссент) => этот сэмплер достаточен для флипа;
  2. вернула 3039 (вершину отчёта)              => этот сэмплер не при чём;
  3. вернула третье                             => записать как есть, выводов нет;
  4. ни один не вернул 4383 => флип требует КОМПОЗИЦИИ (не одного элемента); это
     отрицательный результат локализации, дальше нужны пары — не эта рука;
  5. `echo.samplers` не совпал с запрошенным => строка невалидна;
  6. `S4d-bare-rep` обязан дать 3039; иначе в этом сеансе недетерминизм и локализация
     нечитаема.
"""
import hashlib, json, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp94_s4pp as P
from exp94_sycl_common import LlamaServer, pick_port, san, SERVER_ARGS

RAW_Q2, OUT, ART, JL = P.RAW_Q2, "results/e94", "results/e94/S4d.json", "/tmp/e94_s4d_reqs.jsonl"
ADDR_KEY = "probe2"
FLIP, VERTEX = 4383, 3039
CHAINS = [("S4d-pen", ["penalties", "top_k", "temperature"]),
          ("S4d-dry", ["dry", "top_k", "temperature"]),
          ("S4d-sig", ["top_n_sigma", "top_k", "temperature"]),
          ("S4d-bare-rep", ["top_k", "temperature"])]
ARCHIVE_SHA = hashlib.sha256(open(RAW_Q2, "rb").read()).hexdigest()[:16]


def main():
    t0, recs, halt = time.time(), [], None
    try:
        with P.tarfile.open(RAW_Q2) as tf:
            b = P.load_addr(tf, ADDR_KEY)
    except Exception as e:
        halt = "guard_abort: адрес не совпал с предрегистрацией: %r" % (e,)
        _write(dict(exp="E94_S4d", requests=[]), halt); return 3
    print("[gate] addr=%s k=%d archive=%s flip=%d vertex=%d"
          % (ADDR_KEY, b["k"], ARCHIVE_SHA, FLIP, VERTEX), flush=True)

    port = pick_port(P.PORTS)
    srv = LlamaServer(port, "/tmp/e94_s4d_server.log")
    try:
        srv.start(); srv.wait_health(timeout=300)
    except Exception as e:
        halt = "guard_abort: сервер не поднялся: %r" % (e,)
        _write(dict(exp="E94_S4d", requests=[]), halt); return 3
    props = {}
    try:
        props = P.fetch_props(port)
    except Exception as e:
        halt = "guard_abort: /props недоступен: %r" % (e,)
    if not halt and os.path.basename(P.MODEL) != os.path.basename(str(props.get("model_path"))):
        halt = "guard_abort: сервер отдал другую модель: %r" % (props.get("model_path"),)
    if halt:
        _write(dict(exp="E94_S4d", requests=[]), halt)
        try: srv.kill()
        except Exception: pass
        return 3

    for tag, chain in CHAINS:
        bd = P.body(b["prompt"], b["k"], samplers=list(chain))
        rec = dict(tag=tag, chain=list(chain), addr="Q2 w%d k=%d" %
                   (P.ADDR[ADDR_KEY]["w"], b["k"]), payload=bd)
        try:
            resp, wall = srv.completion(bd, timeout=600)
            rec.update(P.readout(resp), status="ok", wall_s=round(wall, 2))
            got = (rec.get("echo") or {}).get("samplers")
            rec["echo_samplers"] = got
            if [str(x) for x in (got or [])] != chain:
                rec.update(verdict="mech_void",
                           reading="правило 5: echo.samplers=%r != %r" % (got, chain))
            elif rec.get("emit") == FLIP:
                rec.update(verdict="flip_restored",
                           reading="правило 1: цепь %s ВЕРНУЛА флип 4383 => этот сэмплер "
                                   "достаточен" % "+".join(chain))
            elif rec.get("emit") == VERTEX:
                rec.update(verdict="no_flip",
                           reading="правило 2: цепь %s дала вершину 3039 => сэмплер не при чём"
                                   % "+".join(chain))
            else:
                rec.update(verdict="other",
                           reading="правило 3: эмиссент %r — ни 4383, ни 3039" % rec.get("emit"))
        except Exception as e:
            rec.update(status="error", err=repr(e), verdict="mech_void", reading="нет ответа")
        recs.append(rec)
        try:
            with open(JL, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception: pass
        print("[req] %-13s emit=%-6s verdict=%s" % (tag, rec.get("emit"), rec.get("verdict")),
              flush=True)
    try: srv.kill()
    except Exception: pass

    g = {r["tag"]: r for r in recs if r.get("status") == "ok" and r.get("verdict") != "mech_void"}
    resp = sorted(t for t, r in g.items() if r.get("verdict") == "flip_restored")
    bare_ok = (g.get("S4d-bare-rep") or {}).get("emit") == VERTEX
    if not bare_ok:
        verdict = "правило 6: контроль [top_k,temperature] дал %r, а не 3039 — сеанс нечитаем" \
                  % (g.get("S4d-bare-rep") or {}).get("emit")
    elif resp:
        verdict = "локализовано: %s — добавление этого сэмплера к голой цепи возвращает флип" % resp
    else:
        verdict = ("правило 4: ни одна одиночная добавка не вернула 4383 => флип требует "
                   "композиции, локализация одним сэмплером не удалась")
    summ = dict(addr_k=b["k"], flip_tok=FLIP, vertex_tok=VERTEX, emits={t: r.get("emit") for t, r in
                [(x, g.get(x) or {}) for x, _ in CHAINS]}, responsible=resp, rule_verdict=verdict,
                requests_n=len(recs), requests_ok=len(g))
    _write(dict(exp="E94_S4d", spec="этот файл, доктрина S4c правило 1",
                started_local=time.strftime("%Y-%m-%d %H:%M"),
                wall_s=round(time.time() - t0, 1), archive_sha16=ARCHIVE_SHA,
                server=dict(port=port, model_reported=str(props.get("model_path"))),
                summary=san(summ), requests=san(recs)), halt)
    print("[done] %s ok=%d/%d wall=%.1fs\n%s" % (ART, len(g), len(CHAINS), time.time() - t0, verdict),
          flush=True)
    print("S4D_DONE", flush=True)
    return 0


def _write(art, halt):
    art = dict(art or {}); art["halt_reason"] = halt
    art.setdefault("exp", "E94_S4d"); art.setdefault("archive_sha16", ARCHIVE_SHA)
    os.makedirs(OUT, exist_ok=True)
    with open(ART, "w", encoding="utf-8") as f:
        json.dump(san(art), f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        try:
            with open(ART, "w", encoding="utf-8") as f:
                json.dump(dict(exp="E94_S4d", halt_reason="crash",
                               trace=traceback.format_exc()[-2000:]), f, ensure_ascii=False, indent=1)
        except Exception:
            pass
        sys.exit(3)
