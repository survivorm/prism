# ПАКЕТ: внешнее ревью E94/S4pp v3 (r9 — ремедиация PASS_WITH_MINORS r8). Репозиторий НЕ читать — всё нужное здесь.

Предыдущее ревью результата (r8, `2_REVIEW_E94_S4pp_exec_r8.md`): PASS_WITH_MINORS,
0 MAJOR/BLOCK, 6 MINOR (F1–F6) + 2 COSMETIC (F7–F8), перезапуск не требовался по
вердикту — но F1/F3/F5/F6/F7 правят КОД, поэтому артефакт регенерирован ЧЕСТНЫМ
прогоном v3 (RUN_RC=0, 24/24, halt=None, wall=55.3s, T=0 детерминирован — числа
идентичны v2). Ремедиация предрегистрирована ДО кода и прогона (spec, раздел
«ИСПРАВЛЕНИЯ ПО r8-РЕВЬЮ», файл docs/dynamic/0_E94_PROBES_S3S2S4.md).
Само-аудит `e94_s4pp_audit.py` после v3: PASS, 0 fail.

Ниже: §A — код ремедиации по каждому finding r8; §B — выжимка артефакта v3;
§C — ответ на F8 (косметика); §D — утверждения исполнителя для проверки (H1–H6);
§I — формат вердикта.

## A. Ремедиация findings r8 (код v3)

### F1 (MINOR): floor 5.43e-4 — фактическое поле
```
   583|         # r7/MINOR-1: численная нестабильность (top1_lp гуляет по k до 5.43e-4) —
   584|         # r8/F1: пол — ФАКТИЧЕСКОЕ поле (max−min по top1_lp за 6 k на адресе);
   585|         # литерал 5.43e-4 из v1 удалён (замеры между прогонами 2e-4..6.5e-4, SYCL-шум).
   586|         lps = [r.get("top1_lp") for r in rs if r.get("top1_lp") is not None]
   587|         return dict(
   588|             n_rows=len(rs), distinct_n_tl=ntl, distinct_ids_head=len(heads),
   589|             top1_lp_spread=(max(lps) - min(lps)) if lps else None,
   590|             distinct_emits=sorted({r.get("emit") for r in rs}, key=lambda x: (x is None, x)),
   591|             report_changed_by_topk=bool(len(ntl) > 1 or len(heads) > 1),
```
Численный литерал 5.43e-4 удалён из кода; top1_lp_spread считается max−min по
top1_lp за 6 k на адресе. В v3: probe spread=0.000517, ctrlA spread=0.000156
(замеры между прогонами 2e-4..7e-4 — SYCL-шум, пол живой).

### F2 (MINOR): элиминация по типу (3 параметра, не «dry=1.0»)
```
   238|         # ветка не подтверждена, ветка §A для S4pp-2 в силе.
   239|         if e == A:
   240|             return ("third_branch", "ТРЕТЬЯ ВЕТКА ЖИВА: с no-op цепью эмиссент вернулся на "
   241|                                     "вершину отчёта %s — штатная цепь (top_k=400, dry=1.0, "
   242|                                     "top_n_sigma=-1.0) была тем, что сдвигало эмиссию; "
   243|                                     "элиминация: при T=0 усечательные сэмплеры (top_k, "
   244|                                     "top_n_sigma, min_p/typ_p/top_p, xtc) не могут снять "
   245|                                     "argmax — маскируют только кандидатов с logit ниже "
   246|                                     "argmax; пере-ранжируют только добавляющие к logits: "
   247|                                     "penalties/dry/mirostat/logit_bias; в этом прогоне "
   248|                                     "penalties нейтральны (1.0/0.0/0.0, repeat_last_n=0), "
   249|                                     "mirostat=0, logit_bias одинаков в обеих руках — значит "
   250|                                     "сдвигатель = dry=1.0; «M1» по S4pp-2 СНИМАЕТСЯ, "
   251|                                     "строка читается «сдвиг цепью (не различает)»" % A)
```
В reading зафиксированы ВСЕ три различающихся параметра (top_k 400→0,
dry 1.0→0.0, top_n_sigma −1.0→0.0) и элиминация: усечательные сэмплеры при T=0
не снимают argmax; пере-ранжируют только penalties/dry/mirostat/logit_bias;
penalties нейтральны, mirostat=0, logit_bias одинаков в обеих руках ⇒ сдвигатель
= dry=1.0.

### F3 (MINOR): branch(§A) + verdict_mechanism
Возврат ветки строки:
```
   209|         # тянулся pv.count/таблицами; механизм — в summary.probes[i].verdict_mechanism.
   210|         if e not in (ban_tok, A):
   211|             return ("branchA", "РЕШАЮЩАЯ ВЕТКА §A: выбор = %s, а это не вершина отчёта (%s) "
   212|                     "и не забаненный — срабатывание ветки; механизм — в "
   213|                     "verdict_mechanism сводки" % (e, A))
```
Механизм — отдельное поле summary.probes[i].verdict_mechanism:
```
   599|     # r8/F3: branch (branchA/undec) — идентификатор ветки строки; статус механизма —
   600|     # отдельное поле verdict_mechanism (spec: third_branch / «механизм НЕ УСТАНОВЛЕН»)
   601|     def _mech(p):
   602|         v = p.get("verdict")
   603|         noop_v = (find("S4pp-noop") or {}).get("verdict")
   604|         if v == "branchA":
   605|             if noop_v == "third_branch":
   606|                 return ("механизм НЕ УСТАНОВЛЕН (третья ветка: no-op цепь вернула "
   607|                         "эмиссент на вершину отчёта)")
   608|             if noop_v == "noop_out":
   609|                 return ("branchA (третья ветка изолирована: no-op цепь не вернула "
   610|                         "эмиссент на вершину отчёта)")
   611|             return "branchA (третья ветка не изолирована: S4pp-noop отсутствует)"
   612|         if v == "undec":
   613|             return "не различает (эмиссент = вершина отчёта, совместимо с M1 и M2)"
   614|         return "нет данных: %r" % v
   615| 
   616|     summary = dict(
```
### F4 (MINOR): S2_pair.informative (мёртвый контроль)
```
   649|                      # r8/F4: n_tl=0 с обеих сторон = мёртвый контроль (null-результат),
   650|                      # в вердикт не вносится
   651|                      informative=bool((find("S2a") or {}).get("n_tl")
   652|                                       or (find("S2b") or {}).get("n_tl"))),
```
В v3: informative=False (n_tl=0 и 0 — отчёт пуст с обеих сторон), reading
S2_pair_diff: «ПРО ОТЧЁТ НИЧЕГО НЕ ИЗВЕСТНО ... сравнивать нечего»; в вердикт
не вносится. (В репетиции тот же код дал n_tl=20/20 — поле двусторонне живое,
в обоих случаях читается корректно.)

### F5 (MINOR): подмена OUT гейтится count==1
```
   328|             # r8/F5: подмена OUT гейтится тем же count==1 (иначе tamper-копия
   329|             # могла бы писать в настоящий results/e94/S4pp.json)
   330|             lit_out = 'OUT = "results/' + 'e94"'  # split: полный литерал не встречается
   331|             # в источнике (там только константа OUT) — иначе count == 2
   332|             lit_out_ok = (src.count(lit_out) == 1)
   333|             if not lit_out_ok:
   334|                 tamper = dict(ok=False, why="литерал OUT не уникален в скрипте")
   335|                 halt = ("tamper-репетиция: литерал OUT не уникален — подмена "
```
При не-уникальном литерале OUT — halt ДО запросов (tamper-копия не может писать
в настоящий results/e94/S4pp.json). В v3: tamper-репетиция rc=3, halt_reason=
crash, assertion=True, серверов 0/0, keys «requests» нет.

### F6 (MINOR): summary.ref +addr +top1
```
   617|         verdict=verdict,
   618|         ref=dict(emit=ref.get("emit"), addr=ref.get("addr"), top1=ref.get("top1"),  # r8/F6
   619|                  verdict=ref.get("verdict"),
```
v3: ref={emit:310, addr:'Q2 w4 k=70', top1:318, verdict:'ref_ok', ...} — рука
«без бана + обычная цепь» на probe2: эмиссент 310 = B, а вершина отчёта 318 —
dry сносит вершину САМ, без бана.

### F7 (COSMETIC→исполнено): числа вместо литералов + per-row n_tl
Вердикт (обе точки branchA — ветка с числами):
```
   555|         # r8/F7: числа, а не литерал «вне {B, A}» — артефакт самодостаточен
   556|         if pv.count("branchA") == 2:
   557|             verdict = ("S4pp-1: %s; S4pp-2: %s; обе точки дали эмиссент вне {B, A}: S4pp-1 эмиссент %r ∉ {B=%r, A=%r}; S4pp-2 эмиссент %r ∉ {B=%r, A=%r}. "
   558|                        "Агрегат (branchA, branchA) не предрегистрирован — это пересказ, "
   559|                        "не агрегатное чтение." % (_pt(probes[0], 0), _pt(probes[1], 1), probes[0].get("emit"), probes[0].get("ban_tok"), probes[0].get("top1"), probes[1].get("emit"), probes[1].get("ban_tok"), probes[1].get("top1")))
   560|         else:
   561|             t = [_pt(p, i) for i, p in enumerate(probes)]
```
per-row n_tl в выжимке:
```
   621|         probes=[dict(tag=p.get("tag"), addr=p.get("addr"), ban_sent=p.get("ban_tok"),
   622|                      emit=p.get("emit"), top1=p.get("top1"), rank=p.get("rank"),
   623|                      n_tl=p.get("n_tl"),  # r8/F7: per-row n_tl в выжимку
   624|                      echo_bias=p.get("echo_bias_tokens"), manip_ok=p.get("manip_ok"),
```
## B. Выжимка артефакта v3 (results/e94/S4pp.json, sha256_16=804699e7261ba3d8, 59756 B)

```json
{
  "started": "2026-11-12 16:06",
  "wall": 55.2,
  "halt": null,
  "req_ok": "24/24",
  "verdict": "S4pp-1: не различает (эмиссент = вершина отчёта 318; совместимо с M1 и M2); S4pp-2: сработала решающая ветка §A (эмиссент 5522 ∉ {ban 4383, top1 3039}); НО ТРЕТЬЯ ВЕТКА ЖИВА: S4pp-noop (no-op цепь) вернул эмиссию на вершину отчёта — «M1» по этой строке СНЯТ, читается «сдвиг цепью (не различает)». Агрегат для этой пары не предрегистрирован — это пересказ, не агрегатное чтение.",
  "ref": {
    "emit": 310,
    "addr": "Q2 w4 k=70",
    "top1": 318,
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
      "n_tl": 20,
      "echo_bias": [
        310
      ],
      "manip_ok": true,
      "verdict": "undec",
      "reading": "НЕ РАЗЛИЧАЕТ: эмиссент = вершина отчёта 318; совместимо и с M2 (следует за отчётом), и с M1 (собственный второй кандидат совпал). Вывод о механизме запрещён.",
      "verdict_mechanism": "не различает (эмиссент = вершина отчёта, совместимо с M1 и M2)"
    },
    {
      "tag": "S4pp-2",
      "addr": "Q2 w13 k=117",
      "ban_sent": 4383,
      "emit": 5522,
      "top1": 3039,
      "rank": 4,
      "n_tl": 20,
      "echo_bias": [
        4383
      ],
      "manip_ok": true,
      "verdict": "branchA",
      "reading": "РЕШАЮЩАЯ ВЕТКА §A: выбор = 5522, а это не вершина отчёта (3039) и не забаненный — срабатывание ветки; механизм — в verdict_mechanism сводки",
      "verdict_mechanism": "механизм НЕ УСТАНОВЛЕН (третья ветка: no-op цепь вернула эмиссент на вершину отчёта)"
    }
  ],
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
    },
    "informative": false
  },
  "S2_pair_diff": {
    "differs": false,
    "pairs_same": true,
    "reading": "эмиссент не изменился (310); ПРО ОТЧЁТ НИЧЕГО НЕ ИЗВЕСТНО: он пуст с обеих сторон (n_tl=0 и 0 при post_sampling_probs=true), сравнивать нечего"
  },
  "tamper": {
    "rc": 3,
    "halt_reason": "crash",
    "assertion": true,
    "servers_before": [],
    "servers_after": [],
    "has_requests_key": false,
    "ok": true
  },
  "S4pp_noop": {
    "emit": 3039,
    "top1": 3039,
    "manip_ok": true,
    "verdict": "third_branch",
    "reading": "ТРЕТЬЯ ВЕТКА ЖИВА: с no-op цепью эмиссент вернулся на вершину отчёта 3039 — штатная цепь (top_k=400, dry=1.0, top_n_sigma=-1.0) была тем, что сдвигало эмиссию; элиминация: при T=0 усечательные сэмплеры (top_k, top_n_sigma, min_p/typ_p/top_p, xtc) не могут снять argmax — маскируют только кандидатов с logit ниже argmax; пере-ранжируют только добавляющие к logits: penalties/dry/mirostat/logit_bias; в этом прогоне penalties нейтральны (1.0/0.0/0.0, repeat_last_n=0), mirostat=0, logit_bias одинаков в обеих руках — значит сдвигатель = dry=1.0; «M1» по S4pp-2 СНИМАЕТСЯ, строка читается «сдвиг цепью (не различает)»"
  },
  "S4pp_0": {
    "emit": 310,
    "manip_ok": true,
    "verdict": "nondec_snip",
    "reading": "бан вершины отчёта применён (manip_ok по эхо), эффекта на эмиссент нет (эмиссент 310 = базовый B 310); НЕ РАЗЛИЧАЕТ и НЕ контроль применимости: на этом адресе A и так не эмитировалась, вывод о механизме запрещён"
  },
  "repro": {
    "emit": 8810,
    "verdict": "repro_ok",
    "reading": "адрес-репро стабилен: эмиссент = вершина 8810, как в архиве (это НЕ проверка флипа — здесь флипа нет)"
  },
  "gate2": {
    "probe_dependence": false,
    "control_dependence": false,
    "gate2_met": false,
    "k1_probe_emit": 310,
    "gate2_caveat": "нижняя правда этого поля ограничена S3_report_diag: если отчёт не меняется от ширины, предикат зависимости не вычислен"
  },
  "S3_report_diag": {
    "probe": {
      "n_rows": 6,
      "distinct_n_tl": [
        20
      ],
      "distinct_ids_head": 1,
      "top1_lp_spread": 0.0005173087120056152,
      "report_changed_by_topk": false
    },
    "ctrlA": {
      "n_rows": 6,
      "distinct_n_tl": [
        20
      ],
      "distinct_ids_head": 1,
      "top1_lp_spread": 0.00015610456466674805,
      "report_changed_by_topk": false
    }
  },
  "controls": [
    {
      "tag": "S4pp-ctrlA",
      "addr": "Q2 w13 k=101",
      "ban_sent": 488,
      "emit": 369,
      "manip_ok": true,
      "moved": true,
      "verdict": "ctrl_ok",
      "reading": "смещение применимо: эмиссент ушёл с вершины 488 на 369"
    },
    {
      "tag": "S4pp-ctrlB",
      "addr": "Q2 w2 k=67",
      "ban_sent": 310,
      "emit": 364,
      "manip_ok": true,
      "moved": true,
      "verdict": "ctrl_ok",
      "reading": "смещение применимо: эмиссент ушёл с вершины 310 на 364"
    },
    {
      "tag": "S4pp-ctrlTieA",
      "addr": "Q2 w0 k=65",
      "ban_sent": 18,
      "emit": 19,
      "manip_ok": true,
      "moved": true,
      "verdict": "ctrl_ok",
      "reading": "смещение применимо: эмиссент ушёл с вершины 18 на 19"
    },
    {
      "tag": "S4pp-ctrlTieB",
      "addr": "Q2 w13 k=95",
      "ban_sent": 3240,
      "emit": 11639,
      "manip_ok": true,
      "moved": true,
      "verdict": "ctrl_ok",
      "reading": "смещение применимо: эмиссент ушёл с вершины 3240 на 11639"
    }
  ],
  "spec": "docs/dynamic/0_E94_PROBES_S3S2S4.md (раздел ИСПРАВЛЕНИЯ ПО r3-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r7-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r8-РЕВЬЮ)"
}
```

## C. F8 (COSMETIC) — оставлено по предрегистрации
pgrep-снимок до/после — перестраховка (ловит висящий, а не короткоживущий
сервер). Основная защита: rc=3 + нет ключа «requests» + halt_reason=crash.
В v3 всё так же: servers_before=[] servers_after=[].

## D. Утверждения исполнителя (проверять только по тексту пакета)

- H1: ЧИСЛА v3 идентичны v2 по всем 24 запросам (T=0, seed=8181, один сервер):
  S4pp-1 emit=318 (undec), S4pp-2 emit=5522 (branchA, 5522 ∉ {4383, 3039}),
  S4pp-noop emit=3039 (third_branch), S4pp-0 emit=310 (nondec_snip),
  ref emit=310 (ref_ok), repro emit=8810 (repro_ok), ctrlA 488, ctrlB 310,
  tieA 18, tieB 3240, S3: probe 310 при всех k, ctrlA 488 при всех k.
- H2: Ни одно из 6 MINOR r8 не осталось в артефакте: F1 — spread фактический
  (0.000517/0.000156, литерала 5.43e-4 в артефакте нет); F2 — reading
  S4pp-noop содержит все 3 параметра + элиминацию; F3 — verdict строки
  branchA/undec (нет «M1»), verdict_mechanism на обеих точках; F4 —
  S2_pair.informative=False; F5 — tamper rc=3/0 серверов; F6 — ref.addr и
  ref.top1 есть; F7 — n_tl построчно (20/20) и вердикт с числами (ветка
  count==2 не срабатывает в v3 — там одна точка branchA; ЧИСЛОВАЯ формулировка
  существует в коде, см. §A F7a).
- H3: Главный вывод НЕ изменился: третья ветка жива (S4pp-noop вернул эмиссию
  на вершину 3039), «M1 established» по S4pp-2 снят, механизм НЕ УСТАНОВЛЕН
  (сдвигатель = dry=1.0 по элиминации).
- H4: gate2_met=false, оговорка к чтению (gate2_caveat) на месте; S3-диагноз:
  отчёт НЕ менялся от top_k на обеих сторонах (distinct_n_tl=1) — «зависимость
  НЕ ВЫЧИСЛЕНА», а не «независимость показана».
- H5: spec-поле артефакта ссылается на r3+r7+r8 разделы предрегистрации.
- H6: В артефакте нет ни одной строки/поля, противоречащей канону S5/S6
  (DRY=1.0 объясняет Q2-флипы): вердикт v3 — пересказ по точкам, слово
  «противоречит» отсутствует.

## I. Формат вердикта (строго, суммарно <= 4000 символов)

```
VERDICT: PASS | PASS_WITH_MINORS | REVISE | BLOCK
MAJOR: <n>, BLOCK: <n>, MINOR: <n>, COSMETIC: <n>
(обоснование 1–3 строки)
FINDINGS:
- [F1|MAJOR|MINOR|COSMETIC] <пункт>: <что>
...
Строка для grep-гейта: VERDICT=<...>
S4PP_R9_DONE
```
