#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 S4c — решающая проверка BLOCK'а r10: создан ли флип штатной цепью сэмплеров.

Основание: docs/dynamic/2_REVIEW_E94_probeS3S2S4_exec_r2.md, раздел 4 (предрегистр записан
ДО этого кода). Кратко: на адресе флипа Q2 w13 k=117 эмиссент после бана ушёл в ранг 4
(5522) при вершине отчёта 3039. Ревьюер r10: confound — в цепи ДО top_k стоят
penalties/dry/top_n_sigma, и они могли переупорядочить логов так, что ранг-2/4 и есть
штатный аргмакс. Тест: тот же адрес с ЦЕПЬЮ ТОЛЬКО [top_k, temperature].

Правило решения (предрегистрировано, иного чтения не будет):
  1. bare выдал 3039 (вершину отчёта)  => флип делает цепь; вывод M1 СНИМАЕТСЯ;
  2. bare выдал 5522 (как со штатной)  => цепь не при чём, M1-ветка чиста от confound'а (n=1);
  3. третье значение                   => проба невалидна для вывода M1;
  4. echo.samplers != запрошенному      => проба невалидна (отключение не применилось);
  5. bare != bare-rep по эмиссенту     => недетерминизм при T=0, выводы по n=1 аннулируются;
     при равных эмиссентах дельта логов заносится как допуск шума.

Запроса 4. GPU/сервер не трогаются, если не пройдены гейты: совпадение адресов с
предрегистрацией (load_addr) и идентичность модели из /props. Abort любой охраны пишется в
артефакт как halt_reason="guard_abort" (r10/MINOR: предсказуемый аборт — не crash).
"""
import hashlib, json, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp94_s4pp as P                      # переиспользую load_addr/body/readout/ADDR — не переписываю
from exp94_sycl_common import LlamaServer, pick_port, san, SERVER_ARGS

RAW_Q2 = P.RAW_Q2
OUT = "results/e94"
ART = OUT + "/S4c.json"
JL = "/tmp/e94_s4c_reqs.jsonl"
ADDR_KEY = "probe2"                          # Q2 w13 k=117 — тот самый адрес M1-строки
BARE = ["top_k", "temperature"]              # server-schema.cpp:501-506: samplers — МАССИВ имён
PLAN = ["S4c-base-rep", "S4c-base-rep2", "S4c-bare", "S4c-bare-rep"]
ARCHIVE_SHA = hashlib.sha256(open(RAW_Q2, "rb").read()).hexdigest()[:16]


def main():
    t0 = time.time()
    recs, notes, halt = [], [], None
    try:
        # --- гейт 1: адреса обязаны совпасть с предрегистрацией (до сервера, до запросов)
        with P.tarfile.open(RAW_Q2) as tf:
            b = P.load_addr(tf, ADDR_KEY)
    except Exception as e:
        halt = "guard_abort: адрес не совпал с предрегистрацией: %r" % (e,)
        _write({}, halt, recs, t0)
        return 3
    exp = {"emit": b["emit"], "top1": b["top1"], "rank": b["rank"]}
    print("[gate] addr %s w%d k=%d emit=%s top1=%s rank=%s archive_sha=%s"
          % (ADDR_KEY, P.ADDR[ADDR_KEY]["w"], b["k"], b["emit"], b["top1"], b["rank"],
             ARCHIVE_SHA), flush=True)

    port = pick_port(P.PORTS)
    srv = LlamaServer(port, "/tmp/e94_s4c_server.log")
    try:
        srv.start(); srv.wait_health(timeout=300)
    except Exception as e:
        halt = "guard_abort: сервер не поднялся: %r" % (e,)
        _write({"addr_expected": exp}, halt, recs, t0); return 3

    props = None
    try:
        props = P.fetch_props(port)
    except Exception as e:
        halt = "guard_abort: /props недоступен: %r" % (e,)
    if not halt:
        seen = str((props or {}).get("model_path") or "")
        if os.path.basename(P.MODEL) != os.path.basename(seen):
            halt = ("guard_abort: сервер отдал другую модель: /props=%r != %r"
                    % (seen, P.MODEL))
    if halt:
        _write({"addr_expected": exp}, halt, recs, t0, props)
        try: srv.kill()
        except Exception: pass
        return 3

    def do(tag, bare):
        ov = {"samplers": list(BARE)} if bare else {}
        bd = P.body(b["prompt"], b["k"], **ov)
        rec = dict(tag=tag, addr="Q2 w%d k=%d" % (P.ADDR[ADDR_KEY]["w"], b["k"]),
                   chain=("bare:%s" % "+".join(BARE)) if bare else "stock",
                   want_samplers=(list(BARE) if bare else None),
                   base=dict(emit=b["emit"], top1=b["top1"], rank=b["rank"]),
                   payload=bd)
        try:
            resp, wall = srv.completion(bd, timeout=600)
            rec.update(P.readout(resp), status="ok", wall_s=round(wall, 2))
            echo_s = (rec.get("echo") or {}).get("samplers")
            rec["echo_samplers"] = echo_s
            if bare and (echo_s is None or [str(x) for x in echo_s] != BARE):
                rec.update(verdict="mech_void",
                           reading="правило 4: echo.samplers=%r != %r — отключение цепи не "
                                   "применилось, выводов нет" % (echo_s, BARE))
            else:
                rec["verdict"] = "mech_ok"
                rec["reading"] = "эмиссент %s при вершине отчёта %s" % (rec.get("emit"), rec.get("top1"))
        except Exception as e:
            rec.update(status="error", err=repr(e), verdict="mech_void",
                       reading="запрос не выполнен")
        recs.append(rec)
        try:
            with open(JL, "a") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception: pass
        print("[req] %-14s %-6s emit=%-6s top1=%-6s verdict=%s"
              % (tag, rec["chain"][:6], rec.get("emit"), rec.get("top1"), rec.get("verdict")),
              flush=True)
        return rec

    for tag in PLAN:
        do(tag, bare=tag.startswith("S4c-bare"))
    try: srv.kill()
    except Exception: pass

    # --- разбор по предрегистрированному правилу -------------------------------
    g = {r["tag"]: r for r in recs if r.get("status") == "ok"}
    bare = (g.get("S4c-bare") or {}).get("emit")
    bare2 = (g.get("S4c-bare-rep") or {}).get("emit")
    lps = [(r["tag"], (r.get("echo") or {}).get("samplers")) for r in recs]
    if len(g) < len(PLAN):
        verdict = "невалидно: выполнены не все 4 запроса (%d/%d)" % (len(g), len(PLAN))
    elif bare != bare2:
        verdict = ("правило 5: недетерминизм при T=0 (bare=%r, bare-rep=%r) — выводы по n=1 "
                   "аннулируются" % (bare, bare2))
    elif (g["S4c-bare"].get("verdict") == "mech_void"
          or g["S4c-bare-rep"].get("verdict") == "mech_void"):
        verdict = "правило 4: цепь не отключилась (echo.samplers), проба невалидна"
    elif bare == 3039:
        verdict = ("правило 1: голая цепь выдала ВЕРШИНУ отчёта 3039 => флип создаёт цепь "
                   "сэмплеров; вывод M1 по этой точке СНИМАЕТСЯ")
    elif bare == 5522:
        verdict = ("правило 2: голая цепь выдала то же 5522, что и штатная => цепь не при чём; "
                   "M1-ветка чиста от confound'а r10, но это по-прежнему n=1")
    else:
        verdict = ("правило 3: голая цепь дала третье значение %r (штатная давала 5522) => "
                   "цепь влияет, но не объясняет; проба невалидна для вывода M1" % bare)

    # шумовой пол (правило 5, вторая половина)
    def lp(tag):
        return (g.get(tag) or {}).get("top1_lp")
    noise = None
    try:
        noise = abs(lp("S4c-base-rep") - lp("S4c-base-rep2"))
    except Exception:
        pass
    summ = dict(addr=P.ADDR[ADDR_KEY]["w"], addr_k=b["k"], expected=exp,
                emit_bare=bare, emit_stock=(g.get("S4c-base-rep") or {}).get("emit"),
                rule_verdict=verdict,
                noise_floor_top1_lp=noise,
                echo_samplers=lps,
                requests_n=len(recs), requests_ok=len(g))
    art = dict(exp="E94_S4c", spec="docs/dynamic/2_REVIEW_E94_probeS3S2S4_exec_r2.md §4",
               started_local=time.strftime("%Y-%m-%d %H:%M"), wall_s=round(time.time() - t0, 1),
               halt_reason=halt, archive_sha16=ARCHIVE_SHA,
               server=dict(port=port, args=SERVER_ARGS,
                           model_reported=str((props or {}).get("model_path")),
                           model_matches=True),
               summary=san(summ), requests=san(recs))
    _write(art, halt, recs, t0, props)
    print("[done] %s halt=%s ok=%d/%d wall=%.1fs" % (ART, halt, len(g), len(PLAN),
                                                     time.time() - t0), flush=True)
    print("S4C_DONE", flush=True)
    return 0 if not halt else 2


def _write(art, halt, recs, t0, props=None):
    art = dict(art or {})
    art.setdefault("exp", "E94_S4c")
    art["halt_reason"] = halt
    art.setdefault("archive_sha16", ARCHIVE_SHA)
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
                json.dump(dict(exp="E94_S4c", halt_reason="crash",
                               trace=traceback.format_exc()[-2000:]), f,
                          ensure_ascii=False, indent=1)
        except Exception:
            pass
        sys.exit(3)
