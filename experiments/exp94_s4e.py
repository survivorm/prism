#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 S4e — предследствие локализации S4d: выключить DRY по-настоящему (multiplier=0.0).

Цепочка фактов этой ночи:
  S4c  со штатной цепью на Q2 w13 k=117 эмиссент=4383 при вершине отчёта 3039; с цепью
       [top_k,temperature] эмиссент=3039  => флип создаёт цепь сэмплеров (правило 1);
  S4d из трёх подозреваемых один возвращает флип к голой цепи — `dry` (penalties и
       top_n_sigma дают вершину);
  исходник: llama-sampler.cpp:3645 `dry_enabled = (multiplier != 0 && base >= 1 && last_n != 0)`,
       :3584 `penalty = multiplier * pow(base, repeat_exp)`. При значениях НАШЕГО канона
       (dry_multiplier=1.0, dry_base=1.0) это НЕ «выключено», а штраф ровно 1.0 логит-юнита
       на токен, повторяющийся в контексте. Разрыв вершин на адресах флипа — 0.0085 ната.

Предрегистр (записан до кода и до запуска). Payload канонический,samplers штатные, T=0,
seed 8181, изменено РОВНО ОДНО поле: dry_multiplier 1.0 -> 0.0.

  S4e-probe   адрес Q2 w4  k=70  (архивный флип: эмиссент 310 при вершине 318)
  S4e-probe2  адрес Q2 w13 k=117 (архивный флип: эмиссент 4383 при вершине 3039)
  S4e-ctrlA   адрес Q2 w13 k=101 (флипа нет, вершина 488)  — контроль, что выключение ничего не сломало
  S4e-repro   адрес-репро (вершина 8810)                    — второй контроль

Правило:
  1. на обоих флиповых адресах эмиссент == вершина отчёта  => флип объяснён DRY-штрафом;
     формулировка результата: «аномалия „эмиссент != вершина отчёта“ есть штатный DRY-штраф
     1.0, включённый НАШИМ же каноном запроса (dry_multiplier=1.0 читается как «нейтрально»,
     а по исходнику это полный штраф)»;
  2. на любом флиповом адресе эмиссент != вершина          => DRY не единственная причина,
     аномалия частично жива — записывается как есть;
  3. на контроле эмиссент != его архивная вершина          => изменённый payload сам что-то
     сломал, проба невалидна;
  4. `echo.dry_multiplier` (если сервер его отдаёт) != 0.0  => строка невалидна;
  5. меньше 4 успешных строк => проба невалидна.
"""
import hashlib, json, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp94_s4pp as P
from exp94_sycl_common import LlamaServer, pick_port, san, SERVER_ARGS

RAW_Q2, OUT, ART, JL = P.RAW_Q2, "results/e94", "results/e94/S4e.json", "/tmp/e94_s4e_reqs.jsonl"
ROWS = [("S4e-probe", "probe", True), ("S4e-probe2", "probe2", True),
        ("S4e-ctrlA", "ctrlA", False), ("S4e-repro", "repro", False)]
ARCHIVE_SHA = hashlib.sha256(open(RAW_Q2, "rb").read()).hexdigest()[:16]


def main():
    t0, recs, halt = time.time(), [], None
    bases = {}
    try:
        with P.tarfile.open(RAW_Q2) as tf:
            for _, key, _ in ROWS:
                bases[key] = P.load_addr(tf, key)
    except Exception as e:
        halt = "guard_abort: адрес не совпал с предрегистрацией: %r" % (e,)
        _write(dict(exp="E94_S4e", requests=[]), halt); return 3
    print("[gate] %d адресов сверены, archive=%s" % (len(bases), ARCHIVE_SHA), flush=True)

    port = pick_port(P.PORTS)
    srv = LlamaServer(port, "/tmp/e94_s4e_server.log")
    try:
        srv.start(); srv.wait_health(timeout=300)
    except Exception as e:
        halt = "guard_abort: сервер не поднялся: %r" % (e,)
        _write(dict(exp="E94_S4e", requests=[]), halt); return 3
    props = {}
    try:
        props = P.fetch_props(port)
    except Exception as e:
        halt = "guard_abort: /props недоступен: %r" % (e,)
    if not halt and os.path.basename(P.MODEL) != os.path.basename(str(props.get("model_path"))):
        halt = "guard_abort: другая модель: %r" % (props.get("model_path"),)
    if halt:
        _write(dict(exp="E94_S4e", requests=[]), halt)
        try: srv.kill()
        except Exception: pass
        return 3

    for tag, key, isflip in ROWS:
        b = bases[key]
        bd = P.body(b["prompt"], b["k"], dry_multiplier=0.0)   # ровно одно изменение
        rec = dict(tag=tag, key=key, isflip=isflip, vertex=b["top1"], archive_emit=b["emit"],
                   payload=bd)
        try:
            resp, wall = srv.completion(bd, timeout=600)
            rec.update(P.readout(resp), status="ok", wall_s=round(wall, 2))
            e = rec.get("emit"); v = b["top1"]
            if isflip:
                ok = (e == v)
                rec.update(verdict="flip_gone" if ok else "flip_stands",
                           reading=("при dry_multiplier=0.0 эмиссент %s СОВПАЛ с вершиной "
                                    "отчёта %s — флип объяснён DRY-штрафом" % (e, v)) if ok else
                                   ("при dry_multiplier=0.0 эмиссент %s != вершина %s — DRY не "
                                    "единственная причина" % (e, v)))
            else:
                ok = (e == b["emit"])
                rec.update(verdict="ctrl_ok" if ok else "ctrl_broken",
                           reading=("контроль цел: эмиссент %s = архивная вершина" % e) if ok else
                                  ("контроль сломан: %s != %s — изменённый payload что-то сдвинул"
                                   % (e, b["emit"])))
        except Exception as ex:
            rec.update(status="error", err=repr(ex), verdict="mech_void", reading="нет ответа")
        recs.append(rec)
        try:
            with open(JL, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception: pass
        print("[req] %-11s emit=%-6s vertex=%-6s verdict=%s"
              % (tag, rec.get("emit"), rec.get("vertex"), rec.get("verdict")), flush=True)
    try: srv.kill()
    except Exception: pass

    g = [r for r in recs if r.get("status") == "ok" and r.get("verdict") != "mech_void"]
    flips = [r for r in g if r["isflip"]]
    ctrls = [r for r in g if not r["isflip"]]
    if len(g) < len(ROWS):
        verdict = "правило 5: успешных строк %d/%d — проба невалидна" % (len(g), len(ROWS))
    elif any(r["verdict"] == "ctrl_broken" for r in ctrls):
        verdict = "правило 3: контроль сломан, проба невалидна"
    elif flips and all(r["verdict"] == "flip_gone" for r in flips):
        verdict = ("правило 1: на ОБОИХ адресах флипа при dry_multiplier=0.0 эмиссент = вершина "
                   "отчёта => аномалия «эмиссент != вершина» есть штатный DRY-штраф 1.0, "
                   "включённый нашим каноном (dry_multiplier=1.0 != выключено)")
    else:
        verdict = "правило 2: хотя бы на одном флиповом адресе флип остался — DRY не единственная причина"
    _write(dict(exp="E94_S4e", spec="этот файл (предрегистр в docstring)",
                started_local=time.strftime("%Y-%m-%d %H:%M"), wall_s=round(time.time() - t0, 1),
                archive_sha16=ARCHIVE_SHA,
                server=dict(port=port, model_reported=str(props.get("model_path"))),
                summary=dict(change="dry_multiplier 1.0 -> 0.0",
                             emits={r["tag"]: r.get("emit") for r in recs},
                             rule_verdict=verdict, requests_n=len(recs), requests_ok=len(g)),
                requests=san(recs)), halt)
    print("[done] %s ok=%d/%d wall=%.1fs\n%s" % (ART, len(g), len(ROWS), time.time() - t0, verdict), flush=True)
    print("S4E_DONE", flush=True)
    return 0


def _write(art, halt):
    art = dict(art or {}); art["halt_reason"] = halt
    art.setdefault("exp", "E94_S4e"); art.setdefault("archive_sha16", ARCHIVE_SHA)
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
                json.dump(dict(exp="E94_S4e", halt_reason="crash",
                               trace=traceback.format_exc()[-2000:]), f, ensure_ascii=False, indent=1)
        except Exception:
            pass
        sys.exit(3)
