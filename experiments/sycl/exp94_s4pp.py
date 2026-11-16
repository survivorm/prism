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

Запросы (24, все n_predict=1, cache_prompt=false, один запуск сервера):
  S4pp-ref 1, S4pp-1 1, S4pp-2 1, S4pp-noop 1, ban-ctrl x4, S4pp-0 1, S4pp-repro 1,
  S2a/S2b 2, S3 (top_k x 2 адреса) 12 = 24. Число сверяется ассертом перед лончем сервера.
Порядок — обратный по различающей силе (MAJOR-4): сначала главный вопрос.

Артефакт results/e94/S4pp.json пишется при ЛЮБОМ исходе, включая смерть сервера;
каждый запрос дописывается в /tmp/e94_s4pp_reqs.jsonl СРАЗУ (пиши-первым).
"""
import json, os, subprocess, sys, tarfile, time, traceback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from exp94_sycl_common import LlamaServer, pick_port, san, SERVER_ARGS

RAW_Q2 = "results/e94/raw_q2.json.tar.gz"
OUT = "results/e94"
ART = OUT + "/S4pp.json"
JL = "/tmp/e94_s4pp_reqs.jsonl"
PORTS = [8114]
BAN = -100.0
MODEL = "/data/Qwen/qwen3.5-0.8B-f16.gguf"
assert MODEL in " ".join(SERVER_ARGS), "MODEL разъехался с SERVER_ARGS (константа стала мёртвой)"

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
# r7/MAJOR-1: заведомо no-op цепь для S4pp-noop (доказательства в предрегистрации,
# раздел ИСПРАВЛЕНИЯ ПО r7-РЕВЬЮ): top_k=0 (top_k_impl: k <= 0 → return,
# llama-sampler.cpp:325), dry_multiplier=0.0 (dry_enabled = multiplier != 0.0, :3648),
# top_n_sigma=-1.0 (n <= 0.0f → empty, :3300; r11/F1: было 0.0 — тоже empty; -1.0 =
# серверный дефолт штатной руки, и дифф цепей рук = ровно 2 поля);
# xtc=0.0 уже в body() (p <= 0 → empty, :2417).
NOOP_EXTRA = dict(top_k=0, dry_multiplier=0.0, top_n_sigma=-1.0)


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


def fetch_props(port, timeout=5):
    """Идентичность сервера словами сервера, а не словами конфига в артефакте.

    Без этого артефакт писал бы «model=/data/Qwen/...» как литерал, и чужой сервер на том же
    порту дал бы прогон с неправильной моделью и правильными на вид числами.
    """
    import urllib.request
    with urllib.request.urlopen("http://127.0.0.1:%d/props" % port, timeout=timeout) as r:
        return json.loads(r.read().decode())


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
                                          "speculative.types",
                                          # r7-ремедиация: полнота эхо no-op цепи —
                                          # эффективные значения всех сдвигателей,
                                          # а не только те, что были в первом эхо
                                          "dry_multiplier", "dry_base", "top_n_sigma",
                                          "repeat_penalty", "presence_penalty",
                                          "frequency_penalty", "repeat_last_n",
                                          "mirostat", "xtc_threshold")}
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
        # exec-r1/Self-2: при пустом top_logprobs оба флага вакуумны (all([]) -> True,
        # max_is_first -> False) и читаются как находка; при n_tl==0 кладём None.
        out["sorted_desc"] = (all(tl[i]["logprob"] >= tl[i + 1]["logprob"] - 1e-12
                                  for i in range(len(tl) - 1)) if tl else None)
        out["max_is_first"] = ((tl and abs(tl[0]["logprob"] -
                                           max(x["logprob"] for x in tl)) < 1e-12) or None)
        if not tl:
            out["max_is_first"] = None
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
        # r8/F3: verdict строки — ИДЕНТИФИКАТОР сработавшей ветки (§A), а не
        # механизм: «M1» в поле строки читался как утверждение о механизме и
        # тянулся pv.count/таблицами; механизм — в summary.probes[i].verdict_mechanism.
        if e not in (ban_tok, A):
            return ("branchA", "РЕШАЮЩАЯ ВЕТКА §A: выбор = %s, а это не вершина отчёта (%s) "
                    "и не забаненный — срабатывание ветки; механизм — в "
                    "verdict_mechanism сводки" % (e, A))
        return ("undec", "НЕ РАЗЛИЧАЕТ: эмиссент = вершина отчёта %s; совместимо и с M2 "
                         "(следует за отчётом), и с M1 (собственный второй кандидат совпал). "
                         "Вывод о механизме запрещён." % A)
    if kind == "top1ban":
        # r4/MINOR-2: на этом адресе A и так не была бы эмиссентом (здесь флип), поэтому
        # строка НЕ доказывает применимость смещения (это делают ctrl-ветки) и не различает
        # M1/M2 — она про то, на какой набор действует ban.
        # r7/MINOR-2: «сработал» ничем не был обеспечен — эмиссент 310 = базовый B,
        # поэтому чтение: «применён, эффекта нет, контроль применимости этой строкой не
        # покрыт» (manip_ok — в строке и в сводке).
        if e == A:
            return ("ctrl_fail", "вершина отчёта осталась эмиссентом despite ban — "
                               "невалидно для всего бана: смещение не применилось")
        if e == B:
            return ("nondec_snip", "бан вершины отчёта применён (manip_ok по эхо), эффекта на "
                                   "эмиссент нет (эмиссент %s = базовый B %s); НЕ РАЗЛИЧАЕТ "
                                   "и НЕ контроль применимости: на этом адресе A и так не "
                                   "эмитировалась, вывод о механизме запрещён" % (e, B))
        return ("nondec_snip", "бан вершины отчёта применён (manip_ok по эхо), эмиссент %s "
                               "∉ {A %s, B %s} — новый класс, описывать отдельно"
                               % (e, A, B))
    if kind == "noop":
        # r7/MAJOR-1: изоляция третьей ветки. С no-op цепью сдвигать эмиссию нечему:
        # возврат на вершину отчёта = третья ветка жива; эмиссент вне {B, A} = третья
        # ветка не подтверждена, ветка §A для S4pp-2 в силе.
        if e == A:
            return ("third_branch", "ТРЕТЬЯ ВЕТКА ЖИВА: с no-op цепью эмиссент вернулся на "
                                    "вершину отчёта %s — штатная цепь (top_k=400, dry=1.0, "
                                    "top_n_sigma=-1.0) была тем, что сдвигало эмиссию; "
                                    "элиминация: при T=0 усечательные сэмплеры (top_k, "
                                    "top_n_sigma, min_p/typ_p/top_p, xtc) не могут снять "
                                    "argmax — маскируют только кандидатов с logit ниже "
                                    "argmax; пере-ранжируют только добавляющие к logits: "
                                    "penalties/dry/mirostat/logit_bias; в этом прогоне "
                                    "penalties нейтральны (1.0/0.0/0.0, repeat_last_n=0), "
                                    "mirostat=0, logit_bias одинаков в обеих руках (фактические "
                                    "цепи рук — в summary, поле chain) — значит кандидат-сдвигатель "
                                    "= dry (dry_multiplier=1.0, dry_base=1.0: штраф = "
                                    "1.0*pow(1.0, repeat_exp) = 1.0 — пере-ранжирование "
                                    "повторяющихся токенов, активен при "
                                    "dry_penalty_last_n=64) — только где argmax СТРОГИЙ "
                                    "(S4pp-1: g12=0.639 > tie-порога 0.025). На S4pp-2 "
                                    "(g12=0.0085 < 0.025) вершина может быть классом "
                                    "равенства; при значениях этой цепи "
                                    "(top_p=1.0, min_p=0.0, typ_p=1.0, xtc=0.0, "
                                    "top_n_sigma=-1.0) прочие усечатели инертны по "
                                    "значению, и top_k=400 не может СНЯТЬ argmax "
                                    "(3039 в топ-400), но тезис «обрезка ≠ "
                                    "перестановка» ложен как факт о top_k: top_k_impl "
                                    "делает partial_sort in-place (llama-sampler.cpp:321) "
                                    "— буфер ПЕРЕУПОРЯДОЧИВАЕТСЯ, и ничья равных logits "
                                    "решается ПОЗИЦИЕЙ в буфере после сортировки "
                                    "(tie-break по позиции — ровно этот случай). "
                                    "«Рез по границе» исключен числами v7: под штатной "
                                    "цепью top1 отчёта = 3039 не срезан, а эмиссия 5522 "
                                    "= rank 4 того же отчёта (ids_head) — чтобы выиграть, "
                                    "её надо ПОДНЯТЬ по рангу; обрезка меняет head "
                                    "только удалив его, head жив. Значит сдвиг на "
                                    "S4pp-2 = пере-ранжирование re-ranker'ом "
                                    "(dry-штраф поднимает ранг) ИЛИ переупорядочение "
                                    "буфера top_k (tie-break по позиции); оба — сдвиг "
                                    "цепью (r12/F8/F10, r13/F2). Tie-контроли "
                                    "(ctrlTieA/B): бан сместил эмиссент со своей вершины "
                                    "как ожидается (ctrl_ok, moved=true: 18→19, "
                                    "3240→11639) — аномального флипа (эмиссент вне "
                                    "{вершина, бан}) на tie-адресах нет; третья ветка на "
                                    "них не проверялась (нет noop-плеча, r13/F7). "
                                    "Вывод «M1 снят / "
                                    "третья ветка жива» НЕ зависит от элиминации: "
                                    "он опирается на noop-плечо (тот же бан 4383, "
                                    "вернулась вершина отчёта). Строка читается "
                                    "«сдвиг цепью (не различает)»" % A)
        return ("noop_out", "ТРЕТЬЯ ВЕТКА НЕ ПОДТВЕРЖДЕНА: с no-op цепью сдвигать эмиссию "
                            "нечему (эмиссент %s ∉ {ban %s, top1 %s}); различающая ветка §A "
                            "для S4pp-2 в силе (вилка M1/M2)" % (e, ban_tok, A))
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
        return ("sweep_move", "эмиссент %s != вершина %s при top_k — ветка усечения; "
                              "не-довод только если то же есть на контроле (тогда это "
                              "безусловный дефект k=1); при чистом контроле это и есть "
                              "сигнал гейта руки B" % (e, A))
    if kind == "repro":
        # r4/MINOR-1: адрес репро НЕ флипует (A == B == 8810), поэтому ветка "флип
        # воспроизвёлся" здесь физически невозможна; это проверка стабильности сессии.
        if e == A:
            return ("repro_ok", "адрес-репро стабилен: эмиссент = вершина %s, как в архиве "
                                "(это НЕ проверка флипа — здесь флипа нет)" % A)
        return ("repro_drift", "адрес-репро дрейфнул: эмиссент %s != архивная вершина %s — "
                               "предупреждение о нестабильности сессии, вывод по пробе "
                               "сопровождать этой пометкой" % (e, A))
    if kind == "s2":
        # r4/MINOR-5: пара backend_sampling pre-registered как НЕ различающая M1/M2;
        # сравнение парное, оно считается в summary.S2_pair.diff, здесь — только факт
        if e is None:
            return ("s2_invalid", "нет эмиссента: строка невалидна")
        return ("s2_fact", "эмиссент %s, вершина %s, n_top_logprobs=%s — сверяется ТОЛЬКО с "
                           "парной строкой, о механизме не говорит" % (e, A, res.get("n_tl")))
    return ("unknown", "чтение не предрегистрировано для kind=%s" % kind)


def main():
    t_start = time.time()
    halt, notes, first_err = None, [], None
    # r11/F5: семантика тега — в артефакте, чтобы grep по тегу не читал «ветка M1»
    notes.append("тег строки verdict='branchA' = идентификатор решающей ветки §A "
                 "предрегистрации (r8/F3), а НЕ механизм M1: с r7/MAJOR-1 механизм "
                 "определяется noop-плечом (S4pp-noop) и записан в "
                 "summary.probes[i].verdict_mechanism и summary.verdict; чтение "
                 "тега 'решающая ветка = M1' запрещено")
    # r12/F11: статистическая сила решающего флипа — в артефакте; r13/F3: поле + адрес,
    # «экстраполяция» помечена; числа ratio — динамическая финальная note (из ЭТОГО
    # артефакта; SYCL-шум: spread гуляет между прогонами, r8/F1: 2e-4..6.5e-4)
    notes.append("решающий флип (S4pp-2, w13 k=117) одиночен и лежит над шумом "
                 "top1_lp_spread (поле summary.S3_report_diag.top1_lp_spread; SYCL-шум: "
                 "значение гуляет между прогонами без зафиксированной границы, r8/F1); spread мерян на адресах "
                 "S3-свипа (w4 k=70, w13 k=101), а не на самом адресе флипа "
                 "(w13 k=117) — экстраполяция (r13/F3); численное отношение g12/spread — "
                 "в финальной note (вычислено из этого артефакта); конкурентное "
                 "объяснение «дребезг» слабо; в любом случае вывод НЕ зависит от "
                 "величины g12 — он опирается на noop-плечо (детерминировано при T=0, "
                 "seed=8181)")
    # r13/F8: атрибуция dry vs top_k невозможна — в артефакте
    notes.append("атрибуция сдвига dry vs top_k в этом прогоне невозможна: дифф цепей рук = "
                 "ровно 2 поля (top_k, dry_multiplier), меняются ВМЕСТЕ (H8); поле "
                 "'samplers' эха — статическая цепь сервера (одинаково во всех 24 строках, "
                 "включая noop-руку с dry выключенным) — кандидатов не разделяет "
                 "(0 новых запросов, r13/F8)")
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

    N_PLAN = 24  # предрегистрировано в карте (раздел ИСПРАВЛЕНИЯ ПО r7-РЕВЬЮ: +S4pp-noop)

    # r7/MINOR-5: доказательство fail-fast (мир tamper) — в артефакте, а не только в логе.
    # Копия скрипта с заведомо неверным литералом пробы обязана умереть в load_addr
    # ДО лонча сервера: rc=3, halt_reason="crash", AssertionError в trace, ноль
    # процессов llama-server, ноль запросов. Рекурсии нет: tamper-копия крашится в
    # load_addr ДО своей tamper-ветки. При любом расхождении — halt ДО запросов.
    tamper = {}
    tpdir = "/tmp/e94_s4pp_tamper"
    try:
        os.makedirs(tpdir, exist_ok=True)
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "exp94_s4pp.py"), encoding="utf-8").read()
        lit = "exp_emit=310, exp_top1=318, exp_rank=" + "2"  # split: полный литерал не
        # встречается в источнике (там только ADDR) — иначе count(lit) == 2 и гейт
        # ложно «не уникален»
        if src.count(lit) != 1:
            tamper = dict(ok=False, why="литерал пробы не уникален в скрипте")
            halt = "tamper-репетиция: литерал пробы не уникален — гейт не проверяем"
        else:
            # r8/F5: подмена OUT гейтится тем же count==1 (иначе tamper-копия
            # могла бы писать в настоящий results/e94/S4pp.json)
            lit_out = 'OUT = "results/' + 'e94"'  # split: полный литерал не встречается
            # в источнике (там только константа OUT) — иначе count == 2
            lit_out_ok = (src.count(lit_out) == 1)
            if not lit_out_ok:
                tamper = dict(ok=False, why="литерал OUT не уникален в скрипте")
                halt = ("tamper-репетиция: литерал OUT не уникален — подмена "
                        "артефакта не обеспечена, гейт не проверяем")
            if lit_out_ok:
                tp = os.path.join(tpdir, "tampered_exp.py")
                with open(tp, "w", encoding="utf-8") as f:
                    f.write(src.replace(lit, "exp_emit=999999, exp_top1=318, exp_rank=2")
                             .replace(lit_out, 'OUT = "%s"' % tpdir))
                before = subprocess.run(["pgrep", "-x", "llama-server"],
                                        capture_output=True, text=True).stdout.split()
                p = subprocess.run([sys.executable, tp],
                                   env=dict(os.environ,
                                            PYTHONPATH=os.path.dirname(os.path.abspath(__file__))),
                                   capture_output=True, text=True, timeout=300)
                after = subprocess.run(["pgrep", "-x", "llama-server"],
                                       capture_output=True, text=True).stdout.split()
                art_t = ({})
                if os.path.exists(os.path.join(tpdir, "S4pp.json")):
                    art_t = json.load(open(os.path.join(tpdir, "S4pp.json")))
                out = (p.stderr or "") + (p.stdout or "")
                tamper = dict(
                    rc=p.returncode,
                    halt_reason=art_t.get("halt_reason"),
                    assertion="AssertionError" in out and "уплыл" in out,
                    servers_before=before, servers_after=after,
                    has_requests_key=("requests" in art_t),
                    # r9/F5: признаки сработавших гейтов (true = count==1 продержался)
                    lit_unique=True, out_lit_unique=lit_out_ok,
                )
                tamper["ok"] = (p.returncode == 3
                                and art_t.get("halt_reason") == "crash"
                                and tamper["assertion"] and before == after
                                and not tamper["has_requests_key"])
                if not tamper["ok"]:
                    halt = ("tamper-репетиция не прошла (ждём rc=3/crash/"
                            "AssertionError/0 серверов/0 запросов): %r" % tamper)
    except Exception as e:
        tamper = dict(ok=False, why=repr(e)[:200])
        halt = "tamper-репетиция упала: %r" % e
    print("[tamper] %s" % tamper, flush=True)

    port = pick_port(PORTS)
    srv = LlamaServer(port, "/tmp/e94_s4pp_server.log")
    print("[srv] port=%d args=%s" % (port, " ".join(SERVER_ARGS)), flush=True)
    try:
        srv.start()
        srv.wait_health(timeout=300)
        print("[srv] healthy pid=%s dri=%s" % (srv.pid, srv.device_fd()), flush=True)
    except Exception as e:
        halt = "server start failed: %r" % (e,)

    props = None
    if not halt:
        try:
            props = fetch_props(port)
        except Exception as e:
            halt = "props недоступен: %r" % (e,)
    if not halt:
        seen = str(props.get("model_path") or props.get("model_alias") or "")
        if os.path.basename(MODEL) != os.path.basename(seen):
            halt = ("сервер отдал другую модель: /props=%r, ожидалось %r — прогон запрещён"
                    % (seen, MODEL))
        elif not props.get("webui", True):
            notes.append("в /props webui=false (не ожидаемо, но не блокирует)")
        print("[srv] props model=%s n_ctx=%s" % (
            seen, ((props or {}).get("default_generation_settings") or {}).get("n_ctx")),
              flush=True)

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
                   sent_extra=({k: v for k, v in ov.items() if k != "logit_bias"} or None),
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
            if ban_tok is not None and not rec["manip_ok"] and v in ("branchA", "undec",
                                                                     "ctrl_ok", "sweep_move",
                                                                     "nondec_snip", "noop_out",
                                                                     "third_branch"):
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
        # r7/MAJOR-1: изоляция третьей ветки (no-op цепь на адресе S4pp-2)
        do("S4pp-noop", "r7/MAJOR-1: бан эмиссента на адресе S4pp-2 при заведомо no-op "
                        "цепи (top_k=0, dry=0.0, top_n_sigma=-1.0)", "probe2", "noop",
           ban_tok=bases["probe2"]["emit"], extra=NOOP_EXTRA)
        # --- 2. контроли бана: смещение обязано быть применимым ------------------
        for key, tag in (("ctrlA", "S4pp-ctrlA"), ("ctrlB", "S4pp-ctrlB"),
                         ("tieA", "S4pp-ctrlTieA"), ("tieB", "S4pp-ctrlTieB")):
            do(tag, "контроль: ban СОБСТВЕННОЙ вершины нефлипающей позиции "
                    "(эмиссент обязан уйти)", key, "ctrl",
           ban_tok=bases[key]["top1"])
        # --- 3. не различающая: где снимается отчёт ------------------------------
        do("S4pp-0", "ban вершины A на адресе пробы (не различающая, объявлено заранее)",
           "probe", "top1ban", ban_tok=bases["probe"]["top1"])
        do("S4pp-repro", "контроль стабильности сессии, зона уверенности (флипа нет)",
           "repro", "repro")
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
    elif any(v == "branchA" for v in pv):
        # r7/MAJOR-2: агрегат для пар кроме (undec, undec) предрегистрирован не был —
        # вердикт = пересказ по точкам (тексты зафиксированы в предрегистрации r7);
        # r7/MAJOR-1: точка S4pp-2 сопровождается чтением S4pp-noop (третья ветка);
        # r8/F7: B/A — числами из строк, не литералами.
        noop_v = (find("S4pp-noop") or {}).get("verdict")

        def _pt(p, i):
            if p.get("verdict") == "branchA":
                base = ("сработала решающая ветка §A (эмиссент %r ∉ {ban %r, top1 %r})"
                        % (p.get("emit"), p.get("ban_tok"), p.get("top1")))
                if i != 1:
                    return base
                if noop_v == "third_branch":
                    return base + ("; НО ТРЕТЬЯ ВЕТКА ЖИВА: S4pp-noop (no-op цепь) вернул "
                                   "эмиссию на вершину отчёта — «M1» по этой строке "
                                   "СНЯТ, читается «сдвиг цепью (не различает)»")
                if noop_v == "noop_out":
                    return base + ("; изолирован от третьей ветки: S4pp-noop (no-op "
                                   "цепь) не вернул эмиссию на вершину отчёта")
                return base + (" с оговоркой: не изолирован от третьей ветки (сдвиг "
                               "цепью) — см. строку S4pp-noop")
            return ("не различает (эмиссент = вершина отчёта %r; совместимо с M1 и M2)"
                    % p.get("top1"))
        # r8/F7: числа, а не литерал «вне {B, A}» — артефакт самодостаточен
        if pv.count("branchA") == 2:
            verdict = ("S4pp-1: %s; S4pp-2: %s; обе точки дали эмиссент вне {B, A}: S4pp-1 эмиссент %r ∉ {B=%r, A=%r}; S4pp-2 эмиссент %r ∉ {B=%r, A=%r}. "
                       "Агрегат (branchA, branchA) не предрегистрирован — это пересказ, "
                       "не агрегатное чтение." % (_pt(probes[0], 0), _pt(probes[1], 1), probes[0].get("emit"), probes[0].get("ban_tok"), probes[0].get("top1"), probes[1].get("emit"), probes[1].get("ban_tok"), probes[1].get("top1")))
        else:
            t = [_pt(p, i) for i, p in enumerate(probes)]
            verdict = ("S4pp-1: %s; S4pp-2: %s. Агрегат для этой пары не "
                       "предрегистрирован — это пересказ, не агрегатное чтение."
                       % (t[0], t[1]))
    elif all(v == "undec" for v in pv):
        verdict = ("не разрешено: обе точки дали вершину отчёта — совместимо с M1 и M2; "
                   "вывод о механизме запрещён; рука B — по пункту 2 гейта (0_DECISION_E94_ARM_B.md)")
    else:
        verdict = "нет данных (части запросов не выполнились), halt=%s" % halt

    s3 = [r for r in recs if r["tag"].startswith("S3-")]
    k1 = find("S3-k1-probe") or {}
    ctrl_moves = [r for r in s3 if r["tag"].endswith("ctrlA")
                  and r.get("verdict") not in (None, "sweep_same")]

    # exec-r1/Self-1: «кусается» ли top_k вообще. Это ДИАГНОСТИКА, предикат гейта не
    # меняет: если отчёт не меняется от ширины, свип не показывал применения ширины и
    # «нет зависимости эмиссии от ширины» надо читать «не вычислено», а не «показано».
    def s3diag(rows, sfx):
        rs = [r for r in rows if r["tag"].endswith(sfx)]
        ntl = sorted({r.get("n_tl") for r in rs}, key=lambda x: (x is None, x))
        heads = sorted({str(r.get("ids_head") or []) for r in rs})
        # r7/MINOR-1: численная нестабильность (top1_lp гуляет по k до 5.43e-4) —
        # r8/F1: пол — ФАКТИЧЕСКОЕ поле (max−min по top1_lp за 6 k на адресе);
        # литерал 5.43e-4 из v1 удалён (замеры между прогонами 2e-4..6.5e-4, SYCL-шум).
        lps = [r.get("top1_lp") for r in rs if r.get("top1_lp") is not None]
        return dict(
            n_rows=len(rs), distinct_n_tl=ntl, distinct_ids_head=len(heads),
            top1_lp_spread=(max(lps) - min(lps)) if lps else None,
            # r9/F1: подпись — по 6 k ОДНОГО адреса и ОДНОЙ руки, не глобальный шум
            top1_lp_spread_note="max−min по top1_lp за 6 k одного адреса и одной руки",
            distinct_emits=sorted({r.get("emit") for r in rs}, key=lambda x: (x is None, x)),
            report_changed_by_topk=bool(len(ntl) > 1 or len(heads) > 1),
            reading=("отчёт меняется от top_k — свип валиден как проверка ширины"
                     if (len(ntl) > 1 or len(heads) > 1) else
                     "ОТЧЁТ НЕ МЕНЯЕТСЯ от top_k (n_tl=%s при всех k, порядок id один) — "
                     "свип не показал, что ширина применяется; при таком диагнозе "
                     "probe_dependence=false читается «зависимость НЕ ВЫЧИСЛЕНА», а не "
                     "«независимость показана»" % ntl))

    # r8/F3: branch (branchA/undec) — идентификатор ветки строки; статус механизма —
    # отдельное поле verdict_mechanism (spec: third_branch / «механизм НЕ УСТАНОВЛЕН»)
    def _chain(r):
        # r10/M1: фактическая (ЭФФЕКТИВНАЯ) цепь руки = поля body() (GATE + dry/
        # penalty/mirostat) + top_n_sigma=-1.0 (серверный дефолт, из эха штатной
        # руки) + sent_extra; совпадает с эхо, дифф цепей читается из сводки
        c = dict(GATE, presence_penalty=0.0, frequency_penalty=0.0, repeat_last_n=0,
                 dry_multiplier=1.0, dry_base=1.0, top_n_sigma=-1.0, mirostat=0.0)
        c.update((r or {}).get("sent_extra") or {})
        return c

    def _mech(p):
        # r9/F3: явная ветка «отсутствует» (noop is None) + нейтральный фолбэк;
        # r9/N2: формулировка согласована с reading (снимается M1, механизм — кандидат)
        v = p.get("verdict")
        noop = find("S4pp-noop")
        if v == "branchA":
            if noop is None:
                return "branchA (S4pp-noop отсутствует: третья ветка не изолирована)"
            nv = noop.get("verdict")
            if nv == "third_branch":
                return ("M1 НЕ УСТАНОВЛЕН: третья ветка жива (no-op цепь при том же "
                        "бане вернула эмиссент на вершину отчёта); сдвиг — цепью "
                        "(кандидат dry: штраф 1.0, активен при dry_penalty_last_n=64; "
                        "на S4pp-2 (g12=0.0085 < 0.025) — пере-ранжирование re-ranker'ом "
                        "(dry поднимает ранг) ИЛИ переупорядочение буфера top_k "
                        "(partial_sort, tie-break по позиции); оба — сдвиг цепью, "
                        "r12/F8, r13/F2)")
            if nv == "noop_out":
                return ("branchA (третья ветка изолирована: no-op цепь не вернула "
                        "эмиссент на вершину отчёта)")
            return "branchA (S4pp-noop verdict=%r — предрегистрированного чтения нет)" % nv
        if v == "undec":
            return "не различает (эмиссент = вершина отчёта, совместимо с M1 и M2)"
        return "нет данных: %r" % v

    summary = dict(
        verdict=verdict,
        ref=dict(emit=ref.get("emit"), addr=ref.get("addr"), top1=ref.get("top1"),
                 ban_sent=ref.get("ban_tok"),  # r9/F6: рука без бана на адресе S4pp-1
                 verdict=ref.get("verdict"),
                 reading=ref.get("reading")),
        probes=[dict(tag=p.get("tag"), addr=p.get("addr"), ban_sent=p.get("ban_tok"),
                     emit=p.get("emit"), top1=p.get("top1"), rank=p.get("rank"),
                     n_tl=p.get("n_tl"),  # r8/F7: per-row n_tl в выжимку
                     echo_bias=p.get("echo_bias_tokens"), manip_ok=p.get("manip_ok"),
                     verdict=p.get("verdict"), reading=p.get("reading"),
                     chain=_chain(p),  # r10/M1: фактическая цепь руки (dict)
                     sent_logit_bias=p.get("sent_logit_bias"),  # r10/M1
                     verdict_mechanism=_mech(p))  # r8/F3: механизм отдельным полем
                for p in probes],
        controls=controls,
        S4pp_0=dict(emit=(find("S4pp-0") or {}).get("emit"),
                    addr=(find("S4pp-0") or {}).get("addr"),  # r11/F7: addr в сводке
                    manip_ok=(find("S4pp-0") or {}).get("manip_ok"),  # r7/MINOR-2
                    verdict=(find("S4pp-0") or {}).get("verdict"),
                    reading=(find("S4pp-0") or {}).get("reading")),
        # r9/N1: решающая рука самодостаточна — addr + ban_sent + chain видно,
        # что бан идентичен S4pp-2 (4383), отличается только цепь
        S4pp_noop=dict(emit=(find("S4pp-noop") or {}).get("emit"),
                       top1=(find("S4pp-noop") or {}).get("top1"),
                       addr=(find("S4pp-noop") or {}).get("addr"),
                       ban_sent=(find("S4pp-noop") or {}).get("ban_tok"),
                       chain=_chain(find("S4pp-noop")),
                       sent_logit_bias=(find("S4pp-noop") or {}).get("sent_logit_bias"),  # r10/M1
                       manip_ok=(find("S4pp-noop") or {}).get("manip_ok"),
                       verdict=(find("S4pp-noop") or {}).get("verdict"),
                       reading=(find("S4pp-noop") or {}).get("reading")),
        repro=dict(emit=(find("S4pp-repro") or {}).get("emit"),
                   verdict=(find("S4pp-repro") or {}).get("verdict"),
                   reading=(find("S4pp-repro") or {}).get("reading")),
        S2_pair=dict(S2a=dict(emit=(find("S2a") or {}).get("emit"),
                              top1=(find("S2a") or {}).get("top1"),
                              n_tl=(find("S2a") or {}).get("n_tl"),
                              echo_backend=(find("S2a") or {}).get("echo", {}).get("backend_sampling")),
                     S2b=dict(emit=(find("S2b") or {}).get("emit"),
                              top1=(find("S2b") or {}).get("top1"),
                              n_tl=(find("S2b") or {}).get("n_tl"),
                              echo_backend=(find("S2b") or {}).get("echo", {}).get("backend_sampling")),
                     # r8/F4: n_tl=0 с обеих сторон = мёртвый контроль (null-результат),
                     # в вердикт не вносится
                     # r9/F4 + r10/M2: n_tl с ОБЕИХ сторон (and, согласовано с
                     # report_nonvacuous/reading) И echo.post_sampling_probs
                     informative=bool(
                         (find("S2a") or {}).get("n_tl")
                         and (find("S2b") or {}).get("n_tl")
                         and (find("S2a") or {}).get("echo", {}).get("post_sampling_probs")
                         and (find("S2b") or {}).get("echo", {}).get("post_sampling_probs"))),
        # r4/MINOR-5: парное сравнение S2 считается числом, а не оставляется читающему
        S2_pair_diff=dict(
            differs=bool((find("S2a") or {}).get("emit") != (find("S2b") or {}).get("emit")
                         or (find("S2a") or {}).get("top1") != (find("S2b") or {}).get("top1")
                         or (find("S2a") or {}).get("n_tl") != (find("S2b") or {}).get("n_tl")),
            pairs_same=bool((find("S2a") or {}).get("status") == "ok"
                            and (find("S2b") or {}).get("status") == "ok"),
            # r9/F4: два пустых отчёта = vacuous-true, а не «идентичность показана»
            report_nonvacuous=bool((find("S2a") or {}).get("n_tl")
                                   and (find("S2b") or {}).get("n_tl")),
            reading=("пара различается: эмиссент/отчёт зависит от ветки backend_sampling — "
                     "это различие ветки, НЕ доказательство M1/M2"
                     if ((find("S2a") or {}).get("emit") != (find("S2b") or {}).get("emit")
                         or (find("S2a") or {}).get("top1") != (find("S2b") or {}).get("top1")
                         or (find("S2a") or {}).get("n_tl") != (find("S2b") or {}).get("n_tl"))
                     else ("эмиссент не изменился (%r); ПРО ОТЧЁТ НИЧЕГО НЕ ИЗВЕСТНО: он пуст "
                           "с обеих сторон (n_tl=%r и %r при post_sampling_probs=true), "
                           "сравнивать нечего" % ((find("S2a") or {}).get("emit"),
                                                  (find("S2a") or {}).get("n_tl"),
                                                  (find("S2b") or {}).get("n_tl"))
                           if not ((find("S2a") or {}).get("n_tl")
                                   and (find("S2b") or {}).get("n_tl"))
                           else "пара идентична: backend_sampling не меняет ни эмиссента, "
                                "ни отчёт"))),
        S3_topk=[dict(tag=r["tag"], top_k=(r.get("echo") or {}).get("top_k"),
                      emit=r.get("emit"), top1=r.get("top1"), rank=r.get("rank"),
                      verdict=r.get("verdict")) for r in s3],
        S3_report_diag=dict(probe=s3diag(s3, "probe"), ctrlA=s3diag(s3, "ctrlA")),
        S3_control_expectation=dict(
            # r4/MINOR-4: ожидаемое значение берётся из загруженного адреса, а не литералом
            expected=("все 6 значений top_k на контроле w13 k=101 дают эмиссента %s"
                      % bases["ctrlA"]["top1"]),
            expected_id=bases["ctrlA"]["top1"],
            n_moves=len(ctrl_moves), moved=[r["tag"] for r in ctrl_moves]),
        k1_note=dict(
            probe_addr=("k=1 на адресе пробы дал %r; адрес конфаунден (эмиссент-флип лежит "
                        "на ранге 2), поэтому это НЕ диагностика ветки k=1"
                        % k1.get("emit")) if k1.get("emit") is not None else "нет данных",
            decisive=("безусловный дефект ветки k=1: на контроле эмиссент ушёл с вершины"
                      if ctrl_moves else
                      "ветка k=1 на контроле даёт вершину отчёта — безусловного дефекта нет"),
        ),
        # r4/MAJOR-1 + MINOR-3: вычисляемый предикат гейта руки B. Замечание ревьюера
        # принято частично: «зависимость от ширины на пробе + её отсутствие на контроле» и
        # есть довод, а не дефект; дефектом он становится только при движении на контроле.
        # Формулировка ревьюера («k=1 остался B => M1-leaning, стал A => M2-leaning»)
        # ОТВЕРГНУТА: при M2 эмиссия по построению не следует усечённому набору, поэтому
        # «остался B» совместим с обеими моделями, а «стал A» значит «усечение ограничивает
        # эмиссию», то есть M2 слабит, а не крепит. Единственный информативный исход k=1 на
        # пробе — эмиссент = A, и он работает ПРОТИВ M2.
        gate2=dict(
            rule=("рука B запускается только если эмиссент на адресе пробы зависит от ширины "
                  "набора И на контроле такой зависимости нет; зависимость вместе с движением "
                  "на контроле = безусловный дефект ветки k=1, не довод"),
            probe_dependence=None, control_dependence=None, gate2_met=None,
            k1_probe_emit=k1.get("emit"),
            gate2_caveat=("нижняя правда этого поля ограничена S3_report_diag: если отчёт "
                          "не меняется от ширины, предикат зависимости не вычислен"),
            k1_note_e=("k=1 на пробе дал вершину отчёта A: усечение ограничивает эмиссию — "
                       "это подрывает M2 (эмиссия следует за набором), а не крепит его"
                       if k1.get("emit") == bases["probe"]["top1"] else
                       "k=1 на пробе дал %r: не различает (при M1 эмиссент = аргмакс своего "
                       "набора = B; при M2 эмиссия не следует усечению и тоже даёт B)"
                       % k1.get("emit")),
        ),
    )
    # «зависимость от ширины» = эмиссент МЕНЯЕТСЯ с k, а не «эмиссент != вершина отчёта»:
    # второе истинно на адресе флипа при любом исходе (там эмиссент и так не вершина) и
    # включало бы гейт на пустом месте — это тот самый «почти всегда истинный» предикат.
    def emits_by_k(sfx, bkey):
        m = {}
        for r in s3:
            if r["tag"].endswith(sfx):
                m[(r.get("echo") or {}).get("top_k")] = r.get("emit")
        vals = [v for v in m.values() if v is not None]
        return m, (len(set(vals)) > 1 if len(m) == 6 else None), sorted(set(vals))
    pm, pdep, pvals = emits_by_k("probe", "probe")
    cm, cdep, cvals = emits_by_k("ctrlA", "ctrlA")
    g2 = summary["gate2"]
    g2["probe_emits_by_k"] = san(pm)
    g2["probe_dependence"] = pdep
    g2["probe_distinct_emits"] = pvals
    g2["probe_stays_at_archive_emit"] = bool(pvals == [bases["probe"]["emit"]])
    g2["control_emits_by_k"] = san(cm)
    g2["control_dependence"] = cdep
    g2["control_distinct_emits"] = cvals
    g2["gate2_met"] = (bool(pdep and not cdep) if (pdep is not None and cdep is not None)
                       else None)
    g2["gate2_note"] = ("неполный свип: зависимость не определена (нужны все 6 значений на "
                        "обеих сторонах), рука B не запускается" if g2["gate2_met"] is None
                        else "условие 2 гейта выполнено" if g2["gate2_met"] else
                        "условие 2 гейта НЕ выполнено")

    n_done = sum(1 for r in recs if r.get("status") == "ok")
    # план сверяется ВСЕГДА: иное число запросов = дефект скрипта, а не исход прогона.
    if len(recs) != N_PLAN:
        halt = halt or ("план исполнен не полностью: %d запросов вместо %d"
                        % (len(recs), N_PLAN))
    if n_done != N_PLAN and not halt:
        halt = "запросы %d/%d успешны без явной причины; первая ошибка: %s" % (
            n_done, N_PLAN, first_err)

    # r13/F3: отношение g12/spread — вычислено из ЭТОГО артефакта (SYCL-шум: значение
    # гуляет между прогонами, r8/F1 — поэтому литералами в note не записывается)
    _sp_p = ((summary.get("S3_report_diag") or {}).get("probe") or {}).get("top1_lp_spread")
    _sp_c = ((summary.get("S3_report_diag") or {}).get("ctrlA") or {}).get("top1_lp_spread")
    _g12 = ((find("S4pp-2") or {}).get("base") or {}).get("g12")
    if _sp_p and _g12:
        notes.append("отношение решающего флипа к шуму, этот артефакт: g12(S4pp-2)=%s, "
                     "top1_lp_spread: probe=%s, ctrlA=%s — ratio ~%dx "
                     "(g12/spread probe); spread — SYCL-шум без зафиксированной "
                     "границы (r8/F1), значение вычислено из этого артефакта; "
                     "флип остаётся одиночным во всех прогонах (24/24, эмиссии "
                     "идентичны, сверка в пакетах)" % (
                         _g12, "%.3e" % _sp_p,
                         ("%.3e" % _sp_c) if _sp_c else "н/д", round(_g12 / _sp_p)))

    art = dict(
        exp="E94_S4pp", spec=("docs/dynamic/0_E94_PROBES_S3S2S4.md (разделы ИСПРАВЛЕНИЯ ПО "
                              "r3/r7/r8/r9/r10-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r13-РЕВЬЮ; r11/r12-ремедиации — "
                              "доп. регистрация в том же файле + 2_REVIEW_E94_S4pp_exec_r11/r12.md)"),
        started_local=time.strftime("%Y-%m-%d %H:%M", time.localtime(t_start)),
        wall_s=round(time.time() - t_start, 1), halt_reason=halt,
        server=dict(port=port, pid=srv.pid, dri=srv.device_fd() if srv.pid else None,
                    args=SERVER_ARGS, model=MODEL,
                    model_reported=(props or {}).get("model_path") or
                    (props or {}).get("model_alias"),
                    model_matches=bool(props is not None and os.path.basename(MODEL) ==
                                      os.path.basename(str((props or {}).get("model_path") or
                                                          (props or {}).get("model_alias") or ""))),
                    n_ctx=(props or {}).get("default_generation_settings", {}).get("n_ctx")),
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
        tamper_rehearsal=tamper,  # r7/MINOR-5: доказательство fail-fast в артефакте
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
    return 0 if not halt else 2


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
