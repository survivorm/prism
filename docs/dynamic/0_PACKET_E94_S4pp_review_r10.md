# ПАКЕТ: внешнее ревью E94/S4pp v4 (r10 — ремедиация PASS_WITH_MINORS r9). Репозиторий НЕ читать — всё нужное здесь.

Предыдущее ревью результата (r9, `2_REVIEW_E94_S4pp_exec_r9.md`): PASS_WITH_MINORS,
0 MAJOR/BLOCK, 6 MINOR + 2 COSMETIC, «перезапуск не требуется» — но F2/F3/F5/F6/N1/N2
правят КОД, поэтому артефакт регенерирован ЧЕСТНЫМ прогоном v4 (RUN_RC=0, 24/24,
halt=None, wall_s=54.2 — число из самого артефакта). Ремедиация предрегистрирована
ДО кода и прогона (spec, раздел «ИСПРАВЛЕНИЯ ПО r9-РЕВЬЮ»,
docs/dynamic/0_E94_PROBES_S3S2S4.md). Само-аудит `e94_s4pp_audit.py` после v4:
PASS, 0 fail (44 чека, включая новые H1–H6).

Ниже: §A — код ремедиации по каждому finding r9; §B — выжимка артефакта v4;
§C — таблица чисел v2/v3/v4 (ответ на r9/H1); §D — утверждения H1–H6;
§I — формат вердикта.

## A. Ремедиация findings r9 (код v4)

### F1 (COSMETIC): подпись spread
```
   597|             top1_lp_spread_note="max−min по top1_lp за 6 k одного адреса и одной руки",
   598|             distinct_emits=sorted({r.get("emit") for r in rs}, key=lambda x: (x is None, x)),
```
### F2 (MINOR): элиминация — только для строгого argmax + фактические цепи рук
```

```
Цепи рук — в артефакте (поле chain в summary.probes и summary.S4pp_noop):
```
   609|     def _chain(r):
   610|         # r9/F2: фактическая цепь руки в сводке; None = доп. полей нет (цепь = GATE),
   611|         # полное эхо цепи — в эхо строки
   612|         e = (r or {}).get("sent_extra")
   613|         return e if e else "штатная цепь = GATE (доп. полей нет); полное эхо — в строке"
```
### F3 (MINOR): _mech — явная ветка «отсутствует» + нейтральный фолбэк
```
   615|     def _mech(p):
   616|         # r9/F3: явная ветка «отсутствует» (noop is None) + нейтральный фолбэк;
   617|         # r9/N2: формулировка согласована с reading (снимается M1, механизм — кандидат)
   618|         v = p.get("verdict")
   619|         noop = find("S4pp-noop")
   620|         if v == "branchA":
   621|             if noop is None:
   622|                 return "branchA (S4pp-noop отсутствует: третья ветка не изолирована)"
   623|             nv = noop.get("verdict")
   624|             if nv == "third_branch":
   625|                 return ("M1 НЕ УСТАНОВЛЕН: третья ветка жива (no-op цепь при том же "
   626|                         "бане вернула эмиссент на вершину отчёта); сдвиг — цепью, "
   627|                         "кандидат dry=1.0 (элиминация для строгого argmax)")
   628|             if nv == "noop_out":
   629|                 return ("branchA (третья ветка изолирована: no-op цепь не вернула "
   630|                         "эмиссент на вершину отчёта)")
   631|             return "branchA (S4pp-noop verdict=%r — предрегистрированного чтения нет)" % nv
   632|         if v == "undec":
   633|             return "не различает (эмиссент = вершина отчёта, совместимо с M1 и M2)"
   634|         return "нет данных: %r" % v
```
Ветки _mech протестированы exec'ом точного фрагмента (noop=None / third_branch /
noop_out / чужой verdict / undec / invalid) — все дают предрегистрированные чтения.

### F4 (COSMETIC): informative гейтится n_tl И echo.post_sampling_probs
```
   678|                      # r9/F4: гейт по n_tl И echo.post_sampling_probs с обеих сторон
   679|                      informative=bool(
   680|                          ((find("S2a") or {}).get("n_tl")
   681|                           or (find("S2b") or {}).get("n_tl"))
   682|                          and (find("S2a") or {}).get("echo", {}).get("post_sampling_probs")
   683|                          and (find("S2b") or {}).get("echo", {}).get("post_sampling_probs"))),
```
```
   692|             report_nonvacuous=bool((find("S2a") or {}).get("n_tl")
   693|                                    and (find("S2b") or {}).get("n_tl")),
```
### F5 (MINOR): признаки сработавших гейтов в артефакте
```
   364|                     # r9/F5: признаки сработавших гейтов (true = count==1 продержался)
   365|                     lit_unique=True, out_lit_unique=lit_out_ok,
   366|                 )
```
v4: lit_unique=true, out_lit_unique=true — count==1-гейты продержались в ЭТОМ
прогоне (evidence H2 r9 стал дискриминирующим).

### F6 (MINOR): ref — рука без бана на адресе S4pp-1 (Q2 w4 k=70)
```
   639|                  ban_sent=ref.get("ban_tok"),  # r9/F6: рука без бана на адресе S4pp-1
   640|                  verdict=ref.get("verdict"),
```
Атрибуция: S4pp-ref — «без бана + ШТАТНАЯ цепь» на адресе S4pp-1 (НЕ probe2 —
ошибка пакета r9). ref: emit=310 (B) и top1=318: цепь сносит вершину и без бана.

### N1 (MINOR): решающая рука самодостаточна
```
   655|         # r9/N1: решающая рука самодостаточна — addr + ban_sent + chain видно,
   656|         # что бан идентичен S4pp-2 (4383), отличается только цепь
   657|         S4pp_noop=dict(emit=(find("S4pp-noop") or {}).get("emit"),
   658|                        top1=(find("S4pp-noop") or {}).get("top1"),
   659|                        addr=(find("S4pp-noop") or {}).get("addr"),
   660|                        ban_sent=(find("S4pp-noop") or {}).get("ban_tok"),
   661|                        chain=_chain(find("S4pp-noop")),
   662|                        manip_ok=(find("S4pp-noop") or {}).get("manip_ok"),
```
Сводка: addr='Q2 w13 k=117' (тот же адрес, что S4pp-2), ban_sent=4383 (ИДЕНТИЧЕН
ban у S4pp-2), chain={'top_k': 0, 'dry_multiplier': 0.0, 'top_n_sigma': 0.0} —
отличается только цепь. logit_bias в обеих руках пуст (см. эхо строк S4pp-2 и
S4pp-noop в артефакте, поле sent_logit_bias; в сводке — echo_bias=[]).

### N2 (MINOR): verdict_mechanism согласован с reading
См. §A F3: для third_branch — «M1 НЕ УСТАНОВЛЕН: третья ветка жива (no-op цепь
при том же бане вернула эмиссент на вершину отчёта); сдвиг — цепью, кандидат
dry=1.0 (элиминация для строгого argmax)». Снимается M1 (внутреннее
предпочтение), механизм сдвига — кандидат, не факт; канону S5/S6 (DRY=1.0
объясняет Q2-флипы) не противоречит (тот же dry).

## B. Выжимка артефакта v4 (results/e94/S4pp.json, sha256_16=c22c8711b2ef7291, 61464 B)

```json
{
  "started": "2026-11-12 16:35",
  "wall": 54.2,
  "halt": null,
  "req_ok": "24/24",
  "verdict": "S4pp-1: не различает (эмиссент = вершина отчёта 318; совместимо с M1 и M2); S4pp-2: сработала решающая ветка §A (эмиссент 5522 ∉ {ban 4383, top1 3039}); НО ТРЕТЬЯ ВЕТКА ЖИВА: S4pp-noop (no-op цепь) вернул эмиссию на вершину отчёта — «M1» по этой строке СНЯТ, читается «сдвиг цепью (не различает)». Агрегат для этой пары не предрегистрирован — это пересказ, не агрегатное чтение.",
  "ref": {
    "emit": 310,
    "addr": "Q2 w4 k=70",
    "top1": 318,
    "ban_sent": null,
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
      "chain": "штатная цепь = GATE (доп. полей нет); полное эхо — в строке",
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
      "chain": "штатная цепь = GATE (доп. полей нет); полное эхо — в строке",
      "verdict_mechanism": "M1 НЕ УСТАНОВЛЕН: третья ветка жива (no-op цепь при том же бане вернула эмиссент на вершину отчёта); сдвиг — цепью, кандидат dry=1.0 (элиминация для строгого argmax)"
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
    "report_nonvacuous": false,
    "reading": "эмиссент не изменился (310); ПРО ОТЧЁТ НИЧЕГО НЕ ИЗВЕСТНО: он пуст с обеих сторон (n_tl=0 и 0 при post_sampling_probs=true), сравнивать нечего"
  },
  "tamper": {
    "rc": 3,
    "halt_reason": "crash",
    "assertion": true,
    "servers_before": [],
    "servers_after": [],
    "has_requests_key": false,
    "lit_unique": true,
    "out_lit_unique": true,
    "ok": true
  },
  "S4pp_noop": {
    "emit": 3039,
    "top1": 3039,
    "addr": "Q2 w13 k=117",
    "ban_sent": 4383,
    "chain": {
      "top_k": 0,
      "dry_multiplier": 0.0,
      "top_n_sigma": 0.0
    },
    "manip_ok": true,
    "verdict": "third_branch",
    "reading": "ТРЕТЬЯ ВЕТКА ЖИВА: с no-op цепью эмиссент вернулся на вершину отчёта 3039 — штатная цепь (top_k=400, dry=1.0, top_n_sigma=-1.0) была тем, что сдвигало эмиссию; элиминация: при T=0 усечательные сэмплеры (top_k, top_n_sigma, min_p/typ_p/top_p, xtc) не могут снять argmax — маскируют только кандидатов с logit ниже argmax; пере-ранжируют только добавляющие к logits: penalties/dry/mirostat/logit_bias; в этом прогоне penalties нейтральны (1.0/0.0/0.0, repeat_last_n=0), mirostat=0, logit_bias одинаков в обеих руках (фактические цепи рук — в summary, поле chain) — значит кандидат-сдвигатель = dry=1.0. Оговорка: элиминация «усечатель не снимает argmax» верна для СТРОГОГО argmax (g12 > SYCL-шум на обеих пробах: 0.639 / 0.009); для класса равенства усечатель вырезает членов класса — см. tie-контроли (g12≈0.025, флипа нет). «M1» по S4pp-2 СНИМАЕТСЯ, строка читается «сдвиг цепью (не различает)»"
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
      "top1_lp_spread": 0.00016394257545471191,
      "top1_lp_spread_note": "max−min по top1_lp за 6 k одного адреса и одной руки",
      "report_changed_by_topk": false
    },
    "ctrlA": {
      "n_rows": 6,
      "distinct_n_tl": [
        20
      ],
      "distinct_ids_head": 1,
      "top1_lp_spread": 0.00012180209159851074,
      "top1_lp_spread_note": "max−min по top1_lp за 6 k одного адреса и одной руки",
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

## C. Числа v2/v3/v4 (ответ на r9/H1 — теперь проверяемо по пакету)

| строка | v2 (р8-пакет) | v3 (р9-пакет) | v4 (этот артефакт) |
|---|---|---|---|
| S4pp-ref emit/top1 | 310 / 318 | 310 / 318 | 310 / 318 |
| S4pp-1 emit/top1/verdict | 318 / 318 / undec | 318 / 318 / undec | 318 / 318 / undec |
| S4pp-2 emit/top1/verdict | 5522 / 3039 / (M1 снят) | 5522 / 3039 / branchA | 5522 / 3039 / branchA |
| S4pp-noop emit/top1/verdict | 3039 / 3039 / third_branch | 3039 / 3039 / third_branch | 3039 / 3039 / third_branch |
| S4pp-0 emit/verdict | 310 / nondec_snip | 310 / nondec_snip | 310 / nondec_snip |
| S4pp-repro emit/verdict | 8810 / repro_ok | 8810 / repro_ok | 8810 / repro_ok |
| S4pp-ctrlA emit/top1 (ban) | moved=true | 369 / 488 | 369 / 488 |
| S4pp-ctrlB emit/top1 (ban) | moved=true | 364 / 310 | 364 / 310 |
| S4pp-ctrlTieA emit/top1 (ban) | moved=true | 19 / 18 | 19 / 18 |
| S4pp-ctrlTieB emit/top1 (ban) | moved=true | 11639 / 3240 | 11639 / 3240 |
| S3 probe emit при k=1,2,3,4,8,400 | 310 x6 | 310 x6 | 310 x6 |
| S3 ctrlA emit при k=1,2,3,4,8,400 | 488 x6 | 488 x6 | 488 x6 |

sha: v1 (git a9a6099) = 05:47; v2 = be1b74d6c90c4c3c (13:50);
v3 = 804699e7261ba3d8 (16:06); **v4 = c22c8711b2ef7291 (16:35, этот артефакт)**.
wall (из артефактов/пакетов): v2=48.0, v3=55.2, v4=54.2.
Сверка v1-git и v4 по ban-контролям: идентичны (369/364/19/11639).
Примечание: в v2 verdict строки S4pp-2 = «M1» (переименование в branchA — r8/F3,
с v3); все ЭМИССИИ во всех версиях одинаковы (T=0, seed=8181, один сервер).

## D. Утверждения исполнителя (проверять только по тексту пакета)

- H1: Числа v4 = числам v3 = числам v2 по всем 24 запросам (таблица §C; v1-git
  по ban-контролям тоже идентичен). T=0, seed=8181, один сервер; wall отличается
  (SYCL), эмиссии — нет.
- H2: Ни один из 6 MINOR r9 не остался: F1 — spread_note в S3_report_diag (§B);
  F2 — chain в probes/S4pp_noop + оговорка строгого argmax в reading (verdict §B);
  F3 — _mech без лжи «отсутствует» (код §A); F4 — informative гейтится echo-флагами
  + report_nonvacuous (§B); F5 — lit_unique/out_lit_unique=true (§B); F6 —
  ref.ban_sent=None + правильная атрибуция (§A); N1 — addr/ban_sent/chain у
  S4pp_noop (§B); N2 — verdict_mechanism = «M1 НЕ УСТАНОВЛЕН...» (§B).
- H3: Главный вывод не изменился: третья ветка жива (S4pp-noop: тот же бан 4383,
  no-op цепь -> эмиссент 3039 = вершина отчёта), M1 снят, сдвиг — цепью (кандидат
  dry=1.0 для строгого argmax).
- H4: gate2_met=false + gate2_caveat; S3: distinct_n_tl=1, report_changed_by_topk
  = false на обеих сторонах => «зависимость НЕ ВЫЧИСЛЕНА», а не «независимость
  показана».
- H5: spec-поле артефакта = r3+r7+r8 разделы предрегистрации; r9-раздел добавлен
  в spec ДО кода v4 и прогона (код, генерировавший v4, ссылался на r3+r7+r8 —
  строка не правилась после прогона; это зафиксировано в r9-спеке).
- H6 (по содержанию, не по слову): вердикт v4 — пересказ по точкам; сдвиг
  атрибутирован dry=1.0 — канону S5/S6 (DRY=1.0 объясняет Q2-флипы) в ту же
  сторону; ни одно поле артефакта не утверждает «M1 установлен».

## I. Формат вердикта (строго, суммарно <= 4000 символов)

```
VERDICT: PASS | PASS_WITH_MINORS | REVISE | BLOCK
MAJOR: <n>, BLOCK: <n>, MINOR: <n>, COSMETIC: <n>
(обоснование 1–3 строки)
FINDINGS:
- [F1|MAJOR|MINOR|COSMETIC] <пункт>: <что>
...
Строка для grep-гейта: VERDICT=<...>
S4PP_R10_DONE
```
