# ПАКЕТ ДЛЯ РЕВЬЮ E94/S4pp (r7). Всё нужное здесь — файлы читать НЕ РАЗРЕШАЕТСЯ.
# Пакет собран скриптом из кода и артефакта, вручную не переписывался.

## A. Определения развилки и различающий ридаут (experiments/exp94_s4pp.py, строки 10-24)
```
  10| Развилка:
  11|   M1 — отчётные top_logprobs привязаны к чужой копии логов, внутренний аргмакс = выданный;
  12|   M2 — отчёт верен, жадная ступень при temperature=0 не схлопывает дистрибуцию.
  13| 
  14| Различающий ридаут (НЕ «меняется/не меняется»: logit_bias применяется внутри
  15| сэмплера, server-schema.cpp:430-462, поэтому ban меняет ответ в обоих мирах).
  16| Различает ИДЕНТИЧНОСТЬ нового эмиссента на адресе флипа, и ветка ОДНА:
  17|   эмиссент вне {эмиссент B, вершина A}    => M1 УСТАНОВЛЕН (решающая ветка)
  18|   эмиссент == вершина отчёта A            => НЕ РАЗЛИЧАЕТ (совместимо с M1 и M2, r3 MAJOR-1)
  19|   эмиссент == B (ban проигнорирован)      => проба НЕВАЛИДНА, выводов нет
  20|   HTTP != 200 / сервер лёг                => проба невалидна, ветка зафиксирована
  21| Значит M2 этой пробой НЕ доказывается; если обе точки дали A, результат — «не разрешено».
  22| 
  23| Форма logit_bias: ЦЕЛОЧНЫЙ id `[[310, -100]]`, а не строка `' to'` как в карте.
  24| Обоснование (grep по форку, сегодня): server-schema.cpp:447 принимает целое id как
```
## B. Гейт совпадения адреса с предрегистрацией (load_addr, строки 88-106)
```
  88|     emit = r["tokens"][0]
  89|     rank = (ids.index(emit) + 1) if emit in ids else None
  90|     base = dict(mem=mem, prompt=list(req["prompt"]), k=a["k"],
  91|                 emit=emit, emit_in_report=emit in ids, rank=rank,
  92|                 top1=tl[0]["id"], top1_lp=tl[0]["logprob"],
  93|                 top2_lp=tl[1]["logprob"], g12=tl[0]["logprob"] - tl[1]["logprob"],
  94|                 rank2=ids[1] if len(ids) > 1 else None,
  95|                 n_tl=len(tl), te=r["tokens_evaluated"])
  96|     # гейт совпадения с предрегистрацией: расхождение = читаем не тот адрес
  97|     for f, got, want in (("emit", emit, a["exp_emit"]),
  98|                          ("top1", tl[0]["id"], a["exp_top1"]),
  99|                          ("rank", rank, a["exp_rank"])):
 100|         if got != want:
 101|             raise AssertionError("addr %s: %s got %r want %r (адрес уплыл)"
 102|                                  % (key, f, got, want))
 103|     if base["te"] != a["k"]:
 104|         raise AssertionError("addr %s: tokens_evaluated %r != k %r"
 105|                              % (key, base["te"], a["k"]))
 106|     return base
```
## C. Порядок: загрузка адреса ДО запуска сервера (строки 255-266)
```
 255| 
 256|     with tarfile.open(RAW_Q2) as tf:
 257|         bases = {k: load_addr(tf, k) for k in ADDR}
 258| 
 259|     for k, b in sorted(bases.items()):
 260|         print("[addr] %-7s w%-2d k=%-4d emit=%-6s top1=%-6s rank=%s g12=%.4f"
 261|               % (k, ADDR[k]["w"], b["k"], b["emit"], b["top1"], b["rank"], b["g12"]),
 262|               flush=True)
 263| 
 264|     N_PLAN = 23  # предрегистрировано в карте (раздел ИСПРАВЛЕНИЯ ПО r3-РЕВЬЮ, п.3)
 265|     port = pick_port(PORTS)
 266|     srv = LlamaServer(port, "/tmp/e94_s4pp_server.log")
```
## D. Диагностика ширины и предикат гейта (строки 436-452 и 493-512)
```
 436|     def s3diag(rows, sfx):
 437|         rs = [r for r in rows if r["tag"].endswith(sfx)]
 438|         ntl = sorted({r.get("n_tl") for r in rs}, key=lambda x: (x is None, x))
 439|         heads = sorted({str(r.get("ids_head") or []) for r in rs})
 440|         return dict(
 441|             n_rows=len(rs), distinct_n_tl=ntl, distinct_ids_head=len(heads),
 442|             distinct_emits=sorted({r.get("emit") for r in rs}, key=lambda x: (x is None, x)),
 443|             report_changed_by_topk=bool(len(ntl) > 1 or len(heads) > 1),
 444|             reading=("отчёт меняется от top_k — свип валиден как проверка ширины"
 445|                      if (len(ntl) > 1 or len(heads) > 1) else
 446|                      "ОТЧЁТ НЕ МЕНЯЕТСЯ от top_k (n_tl=%s при всех k, порядок id один) — "
 447|                      "свип не показал, что ширина применяется; при таком диагнозе "
 448|                      "probe_dependence=false читается «зависимость НЕ ВЫЧИСЛЕНА», а не "
 449|                      "«независимость показана»" % ntl))
 450| 
 451|     summary = dict(
 452|         verdict=verdict,
```
```
 493|                                    and (find("S2b") or {}).get("n_tl"))
 494|                            else "пара идентична: backend_sampling не меняет ни эмиссента, "
 495|                                 "ни отчёт"))),
 496|         S3_topk=[dict(tag=r["tag"], top_k=(r.get("echo") or {}).get("top_k"),
 497|                       emit=r.get("emit"), top1=r.get("top1"), rank=r.get("rank"),
 498|                       verdict=r.get("verdict")) for r in s3],
 499|         S3_report_diag=dict(probe=s3diag(s3, "probe"), ctrlA=s3diag(s3, "ctrlA")),
 500|         S3_control_expectation=dict(
 501|             # r4/MINOR-4: ожидаемое значение берётся из загруженного адреса, а не литералом
 502|             expected=("все 6 значений top_k на контроле w13 k=101 дают эмиссента %s"
 503|                       % bases["ctrlA"]["top1"]),
 504|             expected_id=bases["ctrlA"]["top1"],
 505|             n_moves=len(ctrl_moves), moved=[r["tag"] for r in ctrl_moves]),
 506|         k1_note=dict(
 507|             probe_addr=("k=1 на адресе пробы дал %r; адрес конфаунден (эмиссент-флип лежит "
 508|                         "на ранге 2), поэтому это НЕ диагностика ветки k=1"
 509|                         % k1.get("emit")) if k1.get("emit") is not None else "нет данных",
 510|             decisive=("безусловный дефект ветки k=1: на контроле эмиссент ушёл с вершины"
 511|                       if ctrl_moves else
 512|                       "ветка k=1 на контроле даёт вершину отчёта — безусловного дефекта нет"),
```
## E. Артефакт results/e94/S4pp.json (сжатая выжимка, числа как в файле)
```json
{
 "verdict": "M1 в одной точке из двух: различение есть, воспроизведения нет (вторая точка противоречит — читать как намёк, не как результат)",
 "ref": {
  "emit": 310,
  "verdict": "ref_ok",
  "reading": "опора есть: флип воспроизвёлся в этом запуске (эмиссент = B)"
 },
 "probes": [
  {
   "tag": "S4pp-1",
   "addr": "Q2 w4 k=70",
   "ban_sent": 310,
   "emit": 318,
   "top1": 318,
   "rank": 1,
   "echo_bias": [
    310
   ],
   "manip_ok": true,
   "verdict": "undec",
   "reading": "НЕ РАЗЛИЧАЕТ: эмиссент = вершина отчёта 318; совместимо и с M2 (следует за отчётом), и с M1 (собственный второй кандидат совпал). Вывод о механизме запрещён."
  },
  {
   "tag": "S4pp-2",
   "addr": "Q2 w13 k=117",
   "ban_sent": 4383,
   "emit": 5522,
   "top1": 3039,
   "rank": 4,
   "echo_bias": [
    4383
   ],
   "manip_ok": true,
   "verdict": "M1",
   "reading": "M1 УСТАНОВЛЕН: выбор = 5522, а это не вершина отчёта (3039) и не забаненный"
  }
 ],
 "controls": [
  {
   "tag": "S4pp-ctrlA",
   "addr": "Q2 w13 k=101",
   "ban_sent": 488,
   "emit": 369,
   "manip_ok": true,
   "moved": true,
   "verdict": "ctrl_ok"
  },
  {
   "tag": "S4pp-ctrlB",
   "addr": "Q2 w2 k=67",
   "ban_sent": 310,
   "emit": 364,
   "manip_ok": true,
   "moved": true,
   "verdict": "ctrl_ok"
  },
  {
   "tag": "S4pp-ctrlTieA",
   "addr": "Q2 w0 k=65",
   "ban_sent": 18,
   "emit": 19,
   "manip_ok": true,
   "moved": true,
   "verdict": "ctrl_ok"
  },
  {
   "tag": "S4pp-ctrlTieB",
   "addr": "Q2 w13 k=95",
   "ban_sent": 3240,
   "emit": 11639,
   "manip_ok": true,
   "moved": true,
   "verdict": "ctrl_ok"
  }
 ],
 "S4pp_0": {
  "emit": 310,
  "verdict": "nondec_snip",
  "reading": "бан вершины отчёта сработал (эмиссент 310 != A 318); НЕ РАЗЛИЧАЕТ и НЕ контроль применимости: на этом адресе A и так не эмитировалась, вывод о механизме запрещён"
 },
 "repro": {
  "emit": 8810,
  "verdict": "repro_ok",
  "reading": "адрес-репро стабилен: эмиссент = вершина 8810, как в архиве (это НЕ проверка флипа — здесь флипа нет)"
 },
 "S2_pair": {
  "S2a": {
   "emit": 310,
   "top1": null,
   "n_tl": 0,
   "echo_backend": true
  },
  "S2b": {
   "emit": 310,
   "top1": null,
   "n_tl": 0,
   "echo_backend": false
  }
 },
 "S2_pair_diff": {
  "differs": false,
  "pairs_same": true,
  "reading": "эмиссент не изменился (310); ПРО ОТЧЁТ НИЧЕГО НЕ ИЗВЕСТНО: он пуст с обеих сторон (n_tl=0 и 0 при post_sampling_probs=true), сравнивать нечего"
 },
 "S3_report_diag": {
  "probe": {
   "n_rows": 6,
   "distinct_n_tl": [
    20
   ],
   "distinct_ids_head": 1,
   "distinct_emits": [
    310
   ],
   "report_changed_by_topk": false,
   "reading": "ОТЧЁТ НЕ МЕНЯЕТСЯ от top_k (n_tl=[20] при всех k, порядок id один) — свип не показал, что ширина применяется; при таком диагнозе probe_dependence=false читается «зависимость НЕ ВЫЧИСЛЕНА», а не «независимость показана»"
  },
  "ctrlA": {
   "n_rows": 6,
   "distinct_n_tl": [
    20
   ],
   "distinct_ids_head": 1,
   "distinct_emits": [
    488
   ],
   "report_changed_by_topk": false,
   "reading": "ОТЧЁТ НЕ МЕНЯЕТСЯ от top_k (n_tl=[20] при всех k, порядок id один) — свип не показал, что ширина применяется; при таком диагнозе probe_dependence=false читается «зависимость НЕ ВЫЧИСЛЕНА», а не «независимость показана»"
  }
 },
 "gate2": {
  "rule": "рука B запускается только если эмиссент на адресе пробы зависит от ширины набора И на контроле такой зависимости нет; зависимость вместе с движением на контроле = безусловный дефект ветки k=1, не довод",
  "probe_dependence": false,
  "control_dependence": false,
  "gate2_met": false,
  "k1_probe_emit": 310,
  "gate2_caveat": "нижняя правда этого поля ограничена S3_report_diag: если отчёт не меняется от ширины, предикат зависимости не вычислен",
  "k1_note_e": "k=1 на пробе дал 310: не различает (при M1 эмиссент = аргмакс своего набора = B; при M2 эмиссия не следует усечению и тоже даёт B)",
  "probe_emits_by_k": {
   "1": 310,
   "2": 310,
   "3": 310,
   "4": 310,
   "8": 310,
   "400": 310
  },
  "probe_distinct_emits": [
   310
  ],
  "probe_stays_at_archive_emit": true,
  "control_emits_by_k": {
   "1": 488,
   "2": 488,
   "3": 488,
   "4": 488,
   "8": 488,
   "400": 488
  },
  "control_distinct_emits": [
   488
  ],
  "gate2_note": "условие 2 гейта НЕ выполнено"
 },
 "server_model_matches": true,
 "samplers": [
  "penalties",
  "dry",
  "top_n_sigma",
  "top_k",
  "typ_p",
  "top_p",
  "min_p",
  "xtc",
  "temperature"
 ],
 "spec": "none",
 "S3_rows": [
  {
   "tag": "S3-k1-probe",
   "emit": 310,
   "top1": 318,
   "rank": 2,
   "n_tl": 20,
   "ids_head": [
    318,
    310,
    326,
    369,
    783,
    4981
   ],
   "top_k": 1,
   "top1_lp": -0.476978
  },
  {
   "tag": "S3-k1-ctrlA",
   "emit": 488,
   "top1": 488,
   "rank": 1,
   "n_tl": 20,
   "ids_head": [
    488,
    369,
    668,
    421,
    310,
    11
   ],
   "top_k": 1,
   "top1_lp": -0.452254
  },
  {
   "tag": "S3-k2-probe",
   "emit": 310,
   "top1": 318,
   "rank": 2,
   "n_tl": 20,
   "ids_head": [
    318,
    310,
    326,
    369,
    783,
    4981
   ],
   "top_k": 2,
   "top1_lp": -0.477155
  },
  {
   "tag": "S3-k2-ctrlA",
   "emit": 488,
   "top1": 488,
   "rank": 1,
   "n_tl": 20,
   "ids_head": [
    488,
    369,
    668,
    421,
    310,
    11
   ],
   "top_k": 2,
   "top1_lp": -0.452042
  },
  {
   "tag": "S3-k3-probe",
   "emit": 310,
   "top1": 318,
   "rank": 2,
   "n_tl": 20,
   "ids_head": [
    318,
    310,
    326,
    369,
    783,
    4981
   ],
   "top_k": 3,
   "top1_lp": -0.477386
  },
  {
   "tag": "S3-k3-ctrlA",
   "emit": 488,
   "top1": 488,
   "rank": 1,
   "n_tl": 20,
   "ids_head": [
    488,
    369,
    668,
    421,
    310,
    11
   ],
   "top_k": 3,
   "top1_lp": -0.452306
  },
  {
   "tag": "S3-k4-probe",
   "emit": 310,
   "top1": 318,
   "rank": 2,
   "n_tl": 20,
   "ids_head": [
    318,
    310,
    326,
    369,
    783,
    4981
   ],
   "top_k": 4,
   "top1_lp": -0.477176
  },
  {
   "tag": "S3-k4-ctrlA",
   "emit": 488,
   "top1": 488,
   "rank": 1,
   "n_tl": 20,
   "ids_head": [
    488,
    369,
    668,
    421,
    310,
    11
   ],
   "top_k": 4,
   "top1_lp": -0.452233
  },
  {
   "tag": "S3-k8-probe",
   "emit": 310,
   "top1": 318,
   "rank": 2,
   "n_tl": 20,
   "ids_head": [
    318,
    310,
    326,
    369,
    783,
    4981
   ],
   "top_k": 8,
   "top1_lp": -0.476843
  },
  {
   "tag": "S3-k8-ctrlA",
   "emit": 488,
   "top1": 488,
   "rank": 1,
   "n_tl": 20,
   "ids_head": [
    488,
    369,
    668,
    421,
    310,
    11
   ],
   "top_k": 8,
   "top1_lp": -0.452344
  },
  {
   "tag": "S3-k400-probe",
   "emit": 310,
   "top1": 318,
   "rank": 2,
   "n_tl": 20,
   "ids_head": [
    318,
    310,
    326,
    369,
    783,
    4981
   ],
   "top_k": 400,
   "top1_lp": -0.477162
  },
  {
   "tag": "S3-k400-ctrlA",
   "emit": 488,
   "top1": 488,
   "rank": 1,
   "n_tl": 20,
   "ids_head": [
    488,
    369,
    668,
    421,
    310,
    11
   ],
   "top_k": 400,
   "top1_lp": -0.452347
  }
 ]
}
```
## F. Утверждения исполнителя, которые надо проверить (каждое — кликабельный факт)

F1. Отказ на замечание r4/MAJOR-2 («ref несамостоятелен: сравнивается с тем же B из того же
    архива»). Ответ: `load_addr` (пакет B) бросает `AssertionError` при расхождении любого из
    трёх литералов и при `tokens_evaluated != k`; загрузка идёт до `pick_port/LlamaServer`
    (пакет C), поэтому запросов не тратится вовсе. Проверено исполняемым тестом: копия скрипта
    с заведомо неверным `exp_emit` даёт rc=3, артефакт `halt_reason="crash"`, ни одного
    запущенного сервера, ни одного запроса (`experiments/e94_s4pp_rehearsal.py`, мир `tamper`).

F2. Отказ на лекарство r4 (само замечание принято). Предрегистрировано: «k=1 остался B ⇒
    M1-leaning, стал A ⇒ M2-leaning» инвертирует импликацию, потому что при M2 (пакет A:
    «отчёт верен, жадная ступень при T=0 не схлопывает дистрибуцию») эмиссия не обязана
    следовать усечённому набору; единственный информативный исход k=1 — эмиссент = A, и он
    работает ПРОТИВ M2. Вопрос: корректно ли, что наблюдение «при k=1 выдан 310=B» помечено
    «не различает»?

F3. MAJOR-Self-1: свип ширины не показал применения ширины (`n_tl=20` и один порядок id при
    k=1 и k=400; см. `S3_rows` в пакете E), поэтому `probe_dependence=false` надо читать
    «зависимость НЕ ВЫЧИСЛЕНА», а не «независимость показана».

F4. По исходнику форка (`server-context.cpp:1790`) отчёт при `post_sampling_probs=false`
    берётся из логов ДО сэмплера — значит путь отчёта в принципе не может показать действие
    `top_k`; а ветка `post_sampling` обрывает цикл на первом `p == 0.0` (1967-1992), из-за чего
    при T=0 отчёт пуст (`n_tl=0` у `S2a/S2b`). Следствие: предлагаемый контроль
    «`top_k=1` vs `400` при `post_sampling_probs=true`, T=0» мёртв.

F5. Эхо `S4pp-2` (пакет E: `samplers`, `spec`): `top_k` в цепи есть, драфта нет. Значит при
    T=0 и `top_k=1` цепь оставила бы ровно вершину 318, а выдан ранг-2 310 — «либо `top_k` не
    кусается, либо эмиссия не из цепи».

## G. Формат ответа (строго, ≤ 5000 символов, без цитат пакета длиннее 2 строк)

```
ВЕРДИКТ: PASS | PASS_WITH_MINORS | BLOCK
F1: <да/нет/частично + обоснование ≤3 строк>
F2: <то же>
F3: <то же>
F4: <то же + жив ли предложенный контроль, и если мёртв — точная замена>
F5: <то же>
ЧИСЛА: <расхождения выжимки E с §0 само-ревью, или "расхождений нет">
FINDINGS: - [BLOCK|MAJOR|MINOR] <что> <что делать>   (≤8)
```
Отвечать только по пакету. Если данных нет — так и написать, не читать репозиторий.