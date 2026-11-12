#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 S4″ / S2 / S3 — раунд различающей пробы над адресами ветки Q2 (teacher forcing).

Спецификация (единственный исполняемый текст):
  docs/dynamic/0_E94_PROBES_S3S2S4.md, раздел «ИСПРАВЛЕНИЯ ПО r2-РЕВЬЮ (05:08)»
  (MAJOR-1..4, MINOR-1..6; редакция S4′ из старого раздела НЕ исполняется).
Основание по статистике: docs/dynamic/0_E94_Q2_supervisor_analysis.md §24-§28.

Развилка:
  M1 — отчётные top_logprobs привязаны к чужой копии логов, внутренний аргмакс = выданный;
  M2 — отчёт верен, жадная ступень при temperature=0 не схлопывает дистрибуцию.

Различающий ридаут (НЕ «меняется/не меняется»: logit_bias применяется внутри
сэмплера, server-schema.cpp:430-462, поэтому ban меняет ответ в обоих мирах).
Различает ИДЕНТИЧНОСТЬ нового эмиссента на адресе флипа, и ветка ОДНА:
  эмиссент вне {эмиссент B, вершина A}    => M1 УСТАНОВЛЕН (решающая ветка)
  эмиссент == вершина отчёта A            => НЕ РАЗЛИЧАЕТ (совместимо с M1 и M2, r3 MAJOR-1)
  эмиссент == B (ban проигнорирован)      => проба НЕВАЛИДНА, выводов нет
  HTTP != 200 / сервер лёг                => проба невалидна, ветка зафиксирована
Значит M2 этой пробой НЕ доказывается; если обе точки дали A, результат — «не разрешено».

Форма logit_bias: ЦЕЛОЧНЫЙ id `[[310, -100]]`, а не строка `' to'` как в карте.
Обоснование (grep по форку, сегодня): server-schema.cpp:447 принимает целое id как
есть, а строка прогоняется через common_tokenize(vocab, s, false) — для `' to'`
это не обязан быть один токен 310, т.е. строковая форма привносит ровно тот
способ отказа, который карта заранее объявила «невалидностью». Манипуляционная
проверка: эхо `generation_settings.logit_bias` = [{"bias":..,"token":..}]
(server-task.cpp:19-27, 128) — сверяется со ОТПРАВЛЕННЫМ id.

n = число РАЗЛИЧНЫХ АДРЕСОВ, не число запросов (при T=0 источника случайности
нет, seed инертен — MINOR-4). Повторы с тем же seed не считаются повторами.

Запросы (20, все n_predict=1, cache_prompt=false, один запуск сервера):
  S4pp-ref 1, S4pp-1 1, S4pp-2 1, ban-ctrl x4, S4pp-0 1, S4pp-repro 1, S2a/S2b 2,
  S3 (top_k x 2 адреса) 12 = 23. Число сверяется ассертом перед лончем сервера.
Порядок — обратный по различающей силе (MAJOR-4): сначала главный вопрос.

Артефакт results/e94/S4pp.json пишется при ЛЮБОМ исходе, включая смерть сервера;
каждый запрос дописывается в /tmp/e94_s4pp_reqs.jsonl СРАЗУ (пиши-первым).
"""
import json, os, sys, tarfile, time, traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from exp94_sycl_common import LlamaServer, pick_port, san, SERVER_ARGS

RAW_Q2 = "results/e94/raw_q2.json.tar.gz"
OUT = "results/e94"
ART = OUT + "/S4pp.json"
JL = "/tmp/e94_s4pp_reqs.jsonl"
PORTS = [8114]
BAN = -100.0

# --- предзарегистрированные адреса (ветка, документ, индекс) + ожидания ---------
# читать ТОЛЬКО из архива той ветки, на которую идёт запрос (MAJOR-3): адреса
# свободной генерации (Q1, индекс j) и teacher-forcing (Q2, индекс k) несравнимы.
ADDR = {
    "probe":   dict(w=4,  k=70,  exp_emit=310, exp_top1=318, exp_rank=2,
                    role="адрес пробы (флип есть)"),
    "probe2":  dict(w=13, k=117, exp_emit=4383, exp_top1=3039, exp_rank=3,
                    role="второй адрес флипа (единственная ничья Q2 с флипом)"),
    "ctrlA":   dict(w=13, k=101, exp_emit=488, exp_top1=488, exp_rank=1,
                    role="контроль той же зоны и класса разрыва, флипа нет"),
    "ctrlB":   dict(w=2,  k=67,  exp_emit=310, exp_top1=310, exp_rank=1,
                    role="второй контроль той же зоны, флипа нет"),
    "tieA":    dict(w=0,  k=65,  exp_emit=18, exp_top1=18, exp_rank=1,
                    role="ничья без флипа"),
    "tieB":    dict(w=13, k=95,  exp_emit=3240, exp_top1=3240, exp_rank=1,
                    role="ничья без флипа"),
    "repro":   dict(w=4,  k=69,  exp_emit=8810, exp_top1=8810, exp_rank=1,
                    role="контроль воспроизводимости адреса (зона уверенности)"),
}
GATE = dict(temperature=0.0, top_k=400, top_p=1.0, min_p=0.0, typical_p=1.0,
            xtc_probability=0.0, repeat_penalty=1.0, seed=8181)


def load_addr(tf, key):
    """Тело запроса и фактический базис адреса — из закоммиченного архива Q2."""
    a = ADDR[key]
    mem = "e94raw/q2_w%d_k%d.json" % (a["w"], a["k"])
    d = json.load(tf.extractfile(mem))
    req, r = d["request"], d["response"]
    cp = r["completion_probabilities"][0]
    tl = cp["top_logprobs"]
    ids = [t["id"] for t in tl]
    emit = r["tokens"][0]
    rank = (ids.index(emit) + 1) if emit in ids else None
    base = dict(mem=mem, prompt=list(req["prompt"]), k=a["k"],
                emit=emit, emit_in_report=emit in ids, rank=rank,
                top1=tl[0]["id"], top1_lp=tl[0]["logprob"],
                top2_lp=tl[1]["logprob"], g12=tl[0]["logprob"] - tl[1]["logprob"],
                rank2=ids[1] if len(ids) > 1 else None,
                n_tl=len(tl), te=r["tokens_evaluated"])
    # гейт совпадения с предрегистрацией: расхождение = читаем не тот адрес
    for f, got, want in (("emit", emit, a["exp_emit"]),
                         ("top1", tl[0]["id"], a["exp_top1"]),
                         ("rank", rank, a["exp_rank"])):
        if got != want:
            raise AssertionError("addr %s: %s got %r want %r (адрес уплыл)"
                                 % (key, f, got, want))
    if base["te"] != a["k"]:
        raise AssertionError("addr %s: tokens_evaluated %r != k %r"
                             % (key, base["te"], a["k"]))
    return base


def body(prompt, k, **ov):
    """Канон Q2 (эксперт: request-поле архива), n_predict=1, cache_prompt=false."""
    b = {"prompt": list(prompt), "n_predict": 1, "temperature": 0, "stream": False,
         "return_tokens": True, "logprobs": True, "top_logprobs": 20,
         "n_probs": 20, "cache_prompt": False, "seed": 8181,
         "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
         "repeat_last_n": 0, "dry_multiplier": 1.0, "dry_base": 1.0,
         "xtc_probability": 0.0, "mirostat": 0, "top_k": 400, "top_p": 1.0,
         "min_p": 0.0, "typical_p": 1.0}
    b.update(ov)
    return b


def readout(resp):
    """Ридаут: ЭМИТЕНТ (не заявленная вершина) + полный отчёт + эхо настроек."""
    r = resp
    t = r.get("tokens") or []
    emit = t[0] if t else None
    out = dict(emit=emit, content=r.get("content"), te=r.get("tokens_evaluated"),
               tp=r.get("tokens_predicted"), stop_type=r.get("stop_type"))
    gs = r.get("generation_settings") or {}
    out["echo"] = {k: gs.get(k) for k in ("temperature", "top_k", "top_p", "min_p",
                                          "typical_p", "xtc_probability", "seed",
                                          "cache_prompt", "backend_sampling",
                                          "post_sampling_probs", "n_probs",
                                          "logit_bias", "samplers",
                                          "speculative.types")}
    cp = r.get("completion_probabilities") or []
    if cp:
        tl = cp[0].get("top_logprobs") or []
        ids = [x["id"] for x in tl]
        out["top1"] = tl[0]["id"] if tl else None
        out["top1_lp"] = tl[0]["logprob"] if tl else None
        out["top2_lp"] = tl[1]["logprob"] if len(tl) > 1 else None
        out["g12"] = ((tl[0]["logprob"] - tl[1]["logprob"]) if len(tl) > 1 else None)
        out["rank"] = (ids.index(emit) + 1) if (emit is not None and emit in ids) else None
        out["emit_lp"] = next((x["logprob"] for x in tl if x["id"] == emit), None)
        out["n_tl"] = len(tl)
        out["ids_head"] = ids[:6]
        out["sorted_desc"] = all(tl[i]["logprob"] >= tl[i + 1]["logprob"]
                                 - 1e-12 for i in range(len(tl) - 1))
        out["max_is_first"] = bool(tl) and abs(tl[0]["logprob"] -
                                               max(x["logprob"] for x in tl)) < 1e-12
    else:
        out.update(top1=None, top1_lp=None, g12=None, rank=None, emit_lp=None,
                   n_tl=0, ids_head=[], sorted_desc=None, max_is_first=None)
    return out


def classify(kind, res, base, ban_tok):
    """Предрегистрированные чтения (раздел «ИСПРАВЛЕНИЯ ПО r3-РЕВЬЮ» карты).

    Ключевой факт (MAJOR-1): ban — параметр сэмплера (server-schema.cpp:430-434), поэтому
    он сносит и внутренний аргмакс. Значит «эмиссент = вершина отчёта A» совместим с обеими
    моделями (при M1 вторым кандидатом собственной копии часто оказывается ровно A), и
    различающая ветка ОДНА: эмиссент вне {B, A}. M2 этой пробой НЕ доказывается.
    """
    e, A, B = res.get("emit"), base["top1"], base["emit"]
    if e is None:
        return ("неизвестно", "невалидна: нет эмиссента в ответе")
    if kind == "ref":
        if e == B:
            return ("ref_ok", "опора есть: флип воспроизвёлся в этом запуске (эмиссент = B)")
        if e == A:
            return ("ref_bad", "адрес в этой сессии НЕ флипует (эмиссент = вершина) — "
                               "весь S4pp объявляется невалидным")
        return ("ref_bad", "эмиссент вне {B, A} уже без смещения — новый класс, "
                           "проба невалидна, описывать отдельно")
    if e == ban_tok:
        return ("invalid", "НЕВАЛИДНА: ban проигнорирован (эмиссент = забаненный)")
    if kind == "probe":
        if e not in (ban_tok, A):
            return ("M1", "M1 УСТАНОВЛЕН: выбор = %s, а это не вершина отчёта (%s) "
                          "и не забаненный" % (e, A))
        return ("undec", "НЕ РАЗЛИЧАЕТ: эмиссент = вершина отчёта %s; совместимо и с M2 "
                         "(следует за отчётом), и с M1 (собственный второй кандидат совпал). "
                         "Вывод о механизме запрещён." % A)
    if kind == "top1ban":
        if e == A:
            return ("ctrl_fail", "смещение не применилось: вершина осталась эмиссентом")
        return ("ctrl_ok", "смещение применимо: эмиссент ушёл с забаненной вершины (%s) "
                           "на %s — не различающая, вопрос только о снимке отчёта" % (A, e))
    if kind == "ctrl":
        # случай e == ban_tok перехвачен выше веткой "ban проигнорирован": для контроля
        # это то же самое событие (собственная вершина осталась эмиссентом), семантика
        # «смещение не применилось», а не «вывод о механизме невалиден»
        if e == A:
            return ("ctrl_fail", "смещение не применилось: эмиссент = забаненная вершина")
        return ("ctrl_ok", "смещение применимо: эмиссент ушёл с вершины %s на %s" % (A, e))
    if kind == "sweep":
        if e == A:
            return ("sweep_same", "эмиссент = вершина, как в архиве (зависимости от top_k нет)")
        return ("sweep_move", "эмиссент %s != вершина %s при top_k — ветка усечения "
                              "(инцидентальный дефект, в механизм не идёт)" % (e, A))
    return ("unknown", "чтение не пред registered для kind=%s" % kind)


def main():
    t_start = time.time()
    halt, notes, first_err = None, [], None
    recs = []

    def emit_line(dct):
        with open(JL, "a") as f:
            f.write(json.dumps(san(dct), ensure_ascii=False) + "\n")

    os.makedirs(OUT, exist_ok=True)
    open(JL, "w").close()

    with tarfile.open(RAW_Q2) as tf:
        bases = {k: load_addr(tf, k) for k in ADDR}

    for k, b in sorted(bases.items()):
        print("[addr] %-7s w%-2d k=%-4d emit=%-6s top1=%-6s rank=%s g12=%.4f"
              % (k, ADDR[k]["w"], b["k"], b["emit"], b["top1"], b["rank"], b["g12"]),
              flush=True)

    N_PLAN = 23  # предрегистрировано в карте (раздел ИСПРАВЛЕНИЯ ПО r3-РЕВЬЮ, п.3)
    port = pick_port(PORTS)
    srv = LlamaServer(port, "/tmp/e94_s4pp_server.log")
    print("[srv] port=%d args=%s" % (port, " ".join(SERVER_ARGS)), flush=True)
    try:
        srv.start()
        srv.wait_health(timeout=300)
        print("[srv] healthy pid=%s dri=%s" % (srv.pid, srv.device_fd()), flush=True)
    except Exception as e:
        halt = "server start failed: %r" % (e,)

    def do(tag, role, bkey, kind, ban_tok=None, extra=None):
        """Один запрос; пишет строку немедленно; ничего не молчит."""
        nonlocal halt, first_err  # без этого присваивания делают halt/first_err локальными
        b = bases[bkey]
        ov = dict(extra or {})
        if ban_tok is not None:
            ov["logit_bias"] = [[int(ban_tok), BAN]]
        bd = body(b["prompt"], b["k"], **ov)
        sent_bias = [int(ban_tok)] if ban_tok is not None else []
        rec = dict(tag=tag, role=role, kind=kind, addr="%s w%d k=%d" % ("Q2", ADDR[bkey]["w"], b["k"]),
                   ban_tok=ban_tok, sent_logit_bias=ov.get("logit_bias"),
                   base=dict(emit=b["emit"], top1=b["top1"], rank=b["rank"],
                             g12=round(b["g12"], 4), rank2=b["rank2"]))
        if halt:
            rec.update(status="skipped", why=halt)
            recs.append(rec); emit_line(rec)
            print("[req] %-12s SKIP (%s)" % (tag, halt), flush=True)
            return rec
        try:
            resp, wall = srv.completion(bd, timeout=600)
            # код ответа / отсутствие эмиссента при живом соединении = тоже остановка:
            # иначе «сервер лёг» молча проходит как успешная строка с emit=None
            if not srv.alive():
                halt = "server died during %s (alive()=False)" % tag
                rec.update(status="error", err=halt)
                recs.append(rec); emit_line(rec)
                print("[req] %-12s DEAD" % tag, flush=True)
                return rec
            ro = readout(resp)
            rec.update(status="ok", wall_s=round(wall, 2), **ro)
            # гейты
            if ro.get("te") != b["k"]:
                rec["gate_te"] = "FAIL te=%r != k=%r" % (ro.get("te"), b["k"])
            echo_bias = (ro.get("echo") or {}).get("logit_bias")
            got_tok = sorted(int(x["token"]) for x in echo_bias
                             if isinstance(x, dict)) if isinstance(echo_bias, list) else []
            rec["echo_bias_tokens"] = got_tok
            if ban_tok is not None:
                rec["manip_ok"] = (got_tok == sent_bias)
                if not rec["manip_ok"]:
                    rec["manip_note"] = ("смещение не применено или применено к другому "
                                         "id — вывод по строке запрещён")
            else:
                rec["manip_ok"] = (got_tok == [])
            if not ro.get("max_is_first"):
                rec["note_report"] = "вершина отчёта не максимум массива (см. §23)"
            v, txt = classify(kind, ro, b, ban_tok)
            if ban_tok is not None and not rec["manip_ok"] and v in ("M1", "undec",
                                                                     "ctrl_ok", "sweep_move"):
                v, txt = "invalid", ("НЕВАЛИДНА: манипуляционный контроль не пройден "
                                     "(эхо logit_bias=%r, ждали %r) — вывод по строке запрещён"
                                     % (rec["echo_bias_tokens"], sent_bias))
            rec["verdict"], rec["reading"] = v, txt
        except Exception as e:
            rec.update(status="error", err=repr(e)[:300])
            if not first_err:
                first_err = "%s: %s" % (tag, repr(e)[:200])
            if not srv.alive():
                halt = "server died at %s (%r)" % (tag, e)
        recs.append(rec); emit_line(rec)
        print("[req] %-12s %-7s emit=%-6s top1=%-6s rank=%-4s echo_bias=%-8s %s"
              % (tag, rec["status"], rec.get("emit"), rec.get("top1"),
                 rec.get("rank"), rec.get("echo_bias_tokens"),
                 rec.get("reading") or rec.get("err") or ""), flush=True)
        time.sleep(1.0)
        return rec

    if not halt:
        # --- 1. главный вопрос: идентичность нового эмиссента -------------------
        do("S4pp-ref", "базис того же адреса в том же запуске (MINOR-6)", "probe", "ref")
        do("S4pp-1", "главная: ban эмиссента B на адресе флипа", "probe", "probe",
           ban_tok=bases["probe"]["emit"])
        do("S4pp-2", "то же различение на втором адресе флипа", "probe2", "probe",
           ban_tok=bases["probe2"]["emit"])
        # --- 2. контроли бана: смещение обязано быть применимым ------------------
        for key, tag in (("ctrlA", "S4pp-ctrlA"), ("ctrlB", "S4pp-ctrlB"),
                         ("tieA", "S4pp-ctrlTieA"), ("tieB", "S4pp-ctrlTieB")):
            do(tag, "контроль: ban СОБСТВЕННОЙ вершины нефлипающей позиции "
                    "(эмиссент обязан уйти)", key, "ctrl",
           ban_tok=bases[key]["top1"])
        # --- 3. не различающая: где снимается отчёт ------------------------------
        do("S4pp-0", "ban вершины A на адресе пробы (не различающая, объявлено заранее)",
           "probe", "top1ban", ban_tok=bases["probe"]["top1"])
        do("S4pp-repro", "контроль воспроизводимости адреса, зона уверенности",
           "repro", "ref")
        # --- 4. S2: пара backend_sampling (сравнивать только между собой) -------
        do("S2a", "backend_sampling true + post_sampling_probs true", "probe", "s2",
           extra={"backend_sampling": True, "post_sampling_probs": True})
        do("S2b", "то же, backend_sampling false (единственный изменённый фактор)",
           "probe", "s2", extra={"backend_sampling": False, "post_sampling_probs": True})
        # --- 5. S3: свип top_k на фиксированном адресе и его контроле -----------
        for kk in (1, 2, 3, 4, 8, 400):
            do("S3-k%d-probe" % kk, "свип top_k, адрес пробы", "probe", "sweep",
               extra={"top_k": kk})
            do("S3-k%d-ctrlA" % kk, "свип top_k, контроль w13 k=101", "ctrlA", "sweep",
               extra={"top_k": kk})

    # --- сводки (только из строк выше, никаких сторонних чисел) ----------------
    def find(tag):
        return next((r for r in recs if r["tag"] == tag), None)

    def verdict_of(tag):
        r = find(tag)
        return (r or {}).get("reading")

    CTRL_TAGS = {"S4pp-ctrlA": "ctrlA", "S4pp-ctrlB": "ctrlB",
                 "S4pp-ctrlTieA": "tieA", "S4pp-ctrlTieB": "tieB"}
    controls = []
    for tag, bkey in CTRL_TAGS.items():
        r = find(tag) or {}
        controls.append(dict(tag=tag, addr=r.get("addr"),
                             ban_sent=r.get("ban_tok"),
                             emit=r.get("emit"),
                             manip_ok=r.get("manip_ok"),
                             moved=bool(r.get("emit") not in (None, bases[bkey]["top1"])),
                             verdict=r.get("verdict"), reading=r.get("reading")))

    probes = [find("S4pp-1") or {}, find("S4pp-2") or {}]
    pv = [p.get("verdict") for p in probes]
    ref = find("S4pp-ref") or {}
    if ref.get("verdict") != "ref_ok":
        verdict = "невалидна: опора не подтверждена (S4pp-ref gave %r)" % ref.get("verdict")
    elif any(v == "invalid" for v in pv):
        verdict = "невалидна: ban или манипуляционный контроль не прошли"
    elif any(v == "M1" for v in pv):
        verdict = ("M1 установлен" if pv.count("M1") == 2 else
                   "M1 в одной точке из двух (n=2 не выдержан, читать как намёк)")
    elif all(v == "undec" for v in pv):
        verdict = ("не разрешено: обе точки дали вершину отчёта — совместимо с M1 и M2; "
                   "вывод о механизме запрещён; рука B — по пункту 2 гейта (0_DECISION_E94_ARM_B.md)")
    else:
        verdict = "нет данных (части запросов не выполнились), halt=%s" % halt

    s3 = [r for r in recs if r["tag"].startswith("S3-")]
    k1 = find("S3-k1-probe") or {}
    ctrl_moves = [r for r in s3 if r["tag"].endswith("ctrlA")
                  and r.get("verdict") not in (None, "sweep_same")]

    summary = dict(
        verdict=verdict,
        ref=dict(emit=ref.get("emit"), verdict=ref.get("verdict"),
                 reading=ref.get("reading")),
        probes=[dict(tag=p.get("tag"), addr=p.get("addr"), ban_sent=p.get("ban_tok"),
                     emit=p.get("emit"), top1=p.get("top1"), rank=p.get("rank"),
                     echo_bias=p.get("echo_bias_tokens"), manip_ok=p.get("manip_ok"),
                     verdict=p.get("verdict"), reading=p.get("reading"))
                for p in probes],
        controls=controls,
        S4pp_0=dict(emit=(find("S4pp-0") or {}).get("emit"),
                    verdict=(find("S4pp-0") or {}).get("verdict"),
                    reading=(find("S4pp-0") or {}).get("reading")),
        repro=dict(emit=(find("S4pp-repro") or {}).get("emit"),
                   verdict=(find("S4pp-repro") or {}).get("verdict")),
        S2_pair=dict(S2a=dict(emit=(find("S2a") or {}).get("emit"),
                              top1=(find("S2a") or {}).get("top1"),
                              n_tl=(find("S2a") or {}).get("n_tl"),
                              echo_backend=(find("S2a") or {}).get("echo", {}).get("backend_sampling")),
                     S2b=dict(emit=(find("S2b") or {}).get("emit"),
                              top1=(find("S2b") or {}).get("top1"),
                              n_tl=(find("S2b") or {}).get("n_tl"),
                              echo_backend=(find("S2b") or {}).get("echo", {}).get("backend_sampling"))),
        S3_topk=[dict(tag=r["tag"], top_k=(r.get("echo") or {}).get("top_k"),
                      emit=r.get("emit"), top1=r.get("top1"), rank=r.get("rank"),
                      verdict=r.get("verdict")) for r in s3],
        S3_control_expectation=dict(
            expected="все 6 значений top_k на контроле w13 k=101 дают эмиссента 488",
            n_moves=len(ctrl_moves), moved=[r["tag"] for r in ctrl_moves]),
        k1_note=dict(
            probe_addr=("k=1 на адресе пробы дал %r; адрес конфаунден (эмиссент-флип лежит "
                        "на ранге 2), поэтому это НЕ диагностика ветки k=1"
                        % k1.get("emit")) if k1.get("emit") is not None else "нет данных",
            decisive=("безусловный дефект ветки k=1: на контроле эмиссент ушёл с вершины"
                      if ctrl_moves else
                      "ветка k=1 на контроле даёт вершину отчёта — безусловного дефекта нет"),
        ),
    )

    n_done = sum(1 for r in recs if r.get("status") == "ok")
    # план сверяется ВСЕГДА: иное число запросов = дефект скрипта, а не исход прогона.
    if len(recs) != N_PLAN:
        halt = halt or ("план исполнен не полностью: %d запросов вместо %d"
                        % (len(recs), N_PLAN))
    rc = 0
    if n_done != N_PLAN and not halt:
        halt = "запросы %d/%d успешны без явной причины; первая ошибка: %s" % (
            n_done, N_PLAN, first_err)

    art = dict(
        exp="E94_S4pp", spec="docs/dynamic/0_E94_PROBES_S3S2S4.md (раздел ИСПРАВЛЕНИЯ ПО r3-РЕВЬЮ)",
        started_local=time.strftime("%Y-%m-%d %H:%M", time.localtime(t_start)),
        wall_s=round(time.time() - t_start, 1), halt_reason=halt,
        server=dict(port=port, pid=srv.pid, dri=srv.device_fd() if srv.pid else None,
                    args=SERVER_ARGS, model="/data/Qwen/qwen3.5-0.8B-f16.gguf"),
        prereg=dict(
            fork="llama.cpp fork 0b8cfb8fe, SYCL/iGPU, /completion",
            question="M1 отчёт привязан к чужой копии логов против M2 жадная ступень не схлопывается",
            readout="идентичность нового эмиссента, а не факт изменения ответа",
            readings=[
                "эмиссент вне {B, A}            => M1 УСТАНОВЛЕН (единственная решающая ветка)",
                "эмиссент == вершина отчёта A   => НЕ РАЗЛИЧАЕТ: при M1 вторым кандидатом "
                "собственной копии часто оказывается ровно A; вывод о механизме запрещён",
                "эмиссент == B                  => НЕВАЛИДНА (ban проигнорирован)",
                "манипуляционный контроль не пройден (эхо logit_bias != отправленный id) "
                "=> НЕВАЛИДНА",
                "S4pp-ref дал не B              => весь раунд НЕВАЛИДЕН (адрес не флипует)",
                "HTTP != 200 или сервер лёг     => НЕВАЛИДНА, ветка зафиксирована",
            ],
            power_note=("при T=0 probe монотонна и детерминирована; n = 2 адреса, обе точки "
                        "в одной паре токенов, значит это НИЖНЯЯ граница независимости"),
            n_unit="число различных адресов (при T=0 seed инертен)",
            bias_form="целочисленный id в logit_bias, обоснование server-schema.cpp:447",
            noninformative=["видимость бана в отчёте (отчёт снимается до сэмплирования)",
                            "tokens_cached (это slot.prompt.n_tokens, не кэш запроса)"],
        ),
        addresses={k: dict(addr="Q2 w%d k=%d" % (ADDR[k]["w"], v["k"]),
                           emit=v["emit"], top1=v["top1"], rank=v["rank"],
                           g12=round(v["g12"], 4), rank2=v["rank2"],
                           role=ADDR[k]["role"])
                   for k, v in bases.items()},
        requests_n=len(recs), requests_ok=sum(1 for r in recs if r.get("status") == "ok"),
        notes=notes,
        summary=san(summary),
        requests=san(recs),
    )
    if srv.pid:
        try:
            srv.kill()
        except Exception as e:
            notes.append("kill: %r" % (e,))
    with open(ART, "w") as f:
        json.dump(san(art), f, ensure_ascii=False, indent=1)
    print("[done] %s halt=%s done=%d/%d wall=%.1fs"
          % (ART, halt, n_done, N_PLAN, time.time() - t_start), flush=True)
    print("S4PP_DONE", flush=True)
    return rc if not halt else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        try:
            with open(ART, "w") as f:
                json.dump(dict(exp="E94_S4pp", halt_reason="crash",
                               trace=traceback.format_exc()[-2000:]), f,
                          ensure_ascii=False, indent=1)
        except Exception:
            pass
        sys.exit(3)
