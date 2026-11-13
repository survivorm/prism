# ПАКЕТ: внешнее ревью E94/S4pp v5 (r11 — ремедиация PASS_WITH_MINORS r10). Репозиторий НЕ читать — всё нужное здесь.

Предыдущее ревью результата (r10, `2_REVIEW_E94_S4pp_exec_r10.md`):
PASS_WITH_MINORS, 0 MAJOR/BLOCK, 4 MINOR (M1–M4) + 4 COSMETIC (C1–C4),
«перезапуск не требуется» — но M1/M2/M3/C3/C4 правят КОД, поэтому артефакт
регенерирован ЧЕСТНЫМ прогоном v5 (RUN_RC=0, 24/24, halt=None, wall_s=60.8 —
число из самого артефакта). Ремедиация предрегистрирована ДО кода и прогона
(spec, раздел «ИСПРАВЛЕНИЯ ПО r10-РЕВЬЮ», docs/dynamic/0_E94_PROBES_S3S2S4.md).
Само-аудит `e94_s4pp_audit.py` после v5: PASS, 0 fail (48 чеков, включая новые
H7–H9: sent_logit_bias, дифф цепей, informative=>nonvacuous).

Ниже: §A — код ремедиации по каждому finding r10; §B — выжимка артефакта v5;
§C — полная таблица 24/24 (ответ на r10/M4); §D — утверждения H1–H6;
§I — формат вердикта.

## A. Ремедиация findings r10 (код v5)

### M1 (MINOR): единая dict-форма chain + sent_logit_bias в сводку
```
   611|     def _chain(r):
   612|         # r10/M1: фактическая (ЭФФЕКТИВНАЯ) цепь руки = поля body() (GATE + dry/
   613|         # penalty/mirostat) + top_n_sigma=-1.0 (серверный дефолт, из эха штатной
   614|         # руки) + sent_extra; совпадает с эхо, дифф цепей читается из сводки
   615|         c = dict(GATE, presence_penalty=0.0, frequency_penalty=0.0, repeat_last_n=0,
   616|                  dry_multiplier=1.0, dry_base=1.0, top_n_sigma=-1.0, mirostat=0.0)
   617|         c.update((r or {}).get("sent_extra") or {})
   618|         return c
   619| 
   620|     def _mech(p):
```
```
   654|                      sent_logit_bias=p.get("sent_logit_bias"),  # r10/M1
   655|                      verdict_mechanism=_mech(p))  # r8/F3: механизм отдельным полем
```
```
   669|                        sent_logit_bias=(find("S4pp-noop") or {}).get("sent_logit_bias"),  # r10/M1
   670|                        manip_ok=(find("S4pp-noop") or {}).get("manip_ok"),
```
Следствие в артефакте: chain(S4pp-2) и chain(S4pp-noop) — два dict, дифф =
РОВНО no-op тройка {top_k: 400→0, dry_multiplier: 1.0→0.0, top_n_sigma:
−1.0→0.0} (авто-чек H8). sent_logit_bias(S4pp-2) = [[4383, −100.0]] =
sent_logit_bias(S4pp-noop) — бан идентичен в обеих руках (авто-чек H7).

### M2 (MINOR): informative — and по n_tl обеих сторон
```
   686|                      # r9/F4 + r10/M2: n_tl с ОБЕИХ сторон (and, согласовано с
   687|                      # report_nonvacuous/reading) И echo.post_sampling_probs
   688|                      informative=bool(
   689|                          (find("S2a") or {}).get("n_tl")
   690|                          and (find("S2b") or {}).get("n_tl")
   691|                          and (find("S2a") or {}).get("echo", {}).get("post_sampling_probs")
   692|                          and (find("S2b") or {}).get("echo", {}).get("post_sampling_probs"))),
```
Согласовано с report_nonvacuous (auto-чек H9: informative ⇒ nonvacuous).

### M3 (MINOR): элиминация — только при строгом argmax
```
   245|                                     "argmax — маскируют только кандидатов с logit ниже "
   246|                                     "argmax; пере-ранжируют только добавляющие к logits: "
   247|                                     "penalties/dry/mirostat/logit_bias; в этом прогоне "
   248|                                     "penalties нейтральны (1.0/0.0/0.0, repeat_last_n=0), "
   249|                                     "mirostat=0, logit_bias одинаков в обеих руках (фактические "
   250|                                     "цепи рук — в summary, поле chain) — значит кандидат-сдвигатель "
   251|                                     "= dry=1.0 — только где argmax СТРОГИЙ (S4pp-1: g12=0.639 > "
   252|                                     "tie-порога 0.025); на S4pp-2 (g12=0.009 < 0.025) вершина может "
   253|                                     "быть классом равенства и сдвиг может включать резание усечателем "
   254|                                     "члена класса — это тоже сдвиг цепью (tie-контроли: флипа нет). "
   255|                                     "Вывод «M1 снят / третья ветка жива» НЕ зависит от элиминации: "
   256|                                     "он опирается на noop-плечо (тот же бан 4383, вернулась вершина "
   257|                                     "отчёта). Строка читается «сдвиг цепью (не различает)»" % A)
   258|         return ("noop_out", "ТРЕТЬЯ ВЕТКА НЕ ПОДТВЕРЖДЕНА: с no-op цепью сдвигать эмиссию "
```
### C3 (COSMETIC): verdict_mechanism с той же оговоркой
```
   630|                 return ("M1 НЕ УСТАНОВЛЕН: третья ветка жива (no-op цепь при том же "
   631|                         "бане вернула эмиссент на вершину отчёта); сдвиг — цепью "
   632|                         "(кандидат dry=1.0 при строгом argmax; на S4pp-2 допустим и "
   633|                         "рез в tie-классе — g12=0.009 < 0.025)")
   634|             if nv == "noop_out":
```
### C4 (COSMETIC): spec-строка = r3+r7+r8+r9+r10 (правка ДО прогона v5)
```
   797|                               "r3-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r7-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r8-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО "
   798|                               "r9-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r10-РЕВЬЮ)"),
   799|         started_local=time.strftime("%Y-%m-%d %H:%M", time.localtime(t_start)),
```
### C1/C2 (COSMETIC): кодовый блок элиминации — выше (§A M3); distinct_emits —
в выжимке §B (S3_report_diag).

## B. Выжимка артефакта v5 (results/e94/S4pp.json, sha256_16=1289fd4ace5f0483, 63054 B)

```json
{
  "started": "2026-11-12 17:24",
  "wall": 60.8,
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
      "chain": {
        "temperature": 0.0,
        "top_k": 400,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "xtc_probability": 0.0,
        "repeat_penalty": 1.0,
        "seed": 8181,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repeat_last_n": 0,
        "dry_multiplier": 1.0,
        "dry_base": 1.0,
        "top_n_sigma": -1.0,
        "mirostat": 0.0
      },
      "sent_logit_bias": [
        [
          310,
          -100.0
        ]
      ],
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
      "chain": {
        "temperature": 0.0,
        "top_k": 400,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "xtc_probability": 0.0,
        "repeat_penalty": 1.0,
        "seed": 8181,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repeat_last_n": 0,
        "dry_multiplier": 1.0,
        "dry_base": 1.0,
        "top_n_sigma": -1.0,
        "mirostat": 0.0
      },
      "sent_logit_bias": [
        [
          4383,
          -100.0
        ]
      ],
      "verdict_mechanism": "M1 НЕ УСТАНОВЛЕН: третья ветка жива (no-op цепь при том же бане вернула эмиссент на вершину отчёта); сдвиг — цепью (кандидат dry=1.0 при строгом argmax; на S4pp-2 допустим и рез в tie-классе — g12=0.009 < 0.025)"
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
      "temperature": 0.0,
      "top_k": 0,
      "top_p": 1.0,
      "min_p": 0.0,
      "typical_p": 1.0,
      "xtc_probability": 0.0,
      "repeat_penalty": 1.0,
      "seed": 8181,
      "presence_penalty": 0.0,
      "frequency_penalty": 0.0,
      "repeat_last_n": 0,
      "dry_multiplier": 0.0,
      "dry_base": 1.0,
      "top_n_sigma": 0.0,
      "mirostat": 0.0
    },
    "sent_logit_bias": [
      [
        4383,
        -100.0
      ]
    ],
    "manip_ok": true,
    "verdict": "third_branch",
    "reading": "ТРЕТЬЯ ВЕТКА ЖИВА: с no-op цепью эмиссент вернулся на вершину отчёта 3039 — штатная цепь (top_k=400, dry=1.0, top_n_sigma=-1.0) была тем, что сдвигало эмиссию; элиминация: при T=0 усечательные сэмплеры (top_k, top_n_sigma, min_p/typ_p/top_p, xtc) не могут снять argmax — маскируют только кандидатов с logit ниже argmax; пере-ранжируют только добавляющие к logits: penalties/dry/mirostat/logit_bias; в этом прогоне penalties нейтральны (1.0/0.0/0.0, repeat_last_n=0), mirostat=0, logit_bias одинаков в обеих руках (фактические цепи рук — в summary, поле chain) — значит кандидат-сдвигатель = dry=1.0 — только где argmax СТРОГИЙ (S4pp-1: g12=0.639 > tie-порога 0.025); на S4pp-2 (g12=0.009 < 0.025) вершина может быть классом равенства и сдвиг может включать резание усечателем члена класса — это тоже сдвиг цепью (tie-контроли: флипа нет). Вывод «M1 снят / третья ветка жива» НЕ зависит от элиминации: он опирается на noop-плечо (тот же бан 4383, вернулась вершина отчёта). Строка читается «сдвиг цепью (не различает)»"
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
      "top1_lp_spread": 0.00030285120010375977,
      "top1_lp_spread_note": "max−min по top1_lp за 6 k одного адреса и одной руки",
      "distinct_emits": [
        310
      ],
      "report_changed_by_topk": false
    },
    "ctrlA": {
      "n_rows": 6,
      "distinct_n_tl": [
        20
      ],
      "distinct_ids_head": 1,
      "top1_lp_spread": 0.00011205673217773438,
      "top1_lp_spread_note": "max−min по top1_lp за 6 k одного адреса и одной руки",
      "distinct_emits": [
        488
      ],
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
  "spec": "docs/dynamic/0_E94_PROBES_S3S2S4.md (раздел ИСПРАВЛЕНИЯ ПО r3-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r7-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r8-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r9-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r10-РЕВЬЮ)"
}
```

## C. Числа v2/v3/v5 — полная таблица 24/24 (ответ на r10/M4)

| строка | v2 (r8-пакет) | v3 (r9-пакет) | v5 (этот артефакт) |
|---|---|---|---|
| S4pp-ref emit/top1 | 310 / 318 | 310 / 318 | 310 / 318 |
| S4pp-1 emit/top1/verdict | 318 / 318 / undec | 318 / 318 / undec | 318 / 318 / undec |
| S4pp-2 emit/top1/verdict | 5522 / 3039 / (M1 снят) | 5522 / 3039 / branchA | 5522 / 3039 / branchA |
| S4pp-noop emit/top1/verdict | 3039 / 3039 / third_branch | 3039 / 3039 / third_branch | 3039 / 3039 / third_branch |
| S4pp-0 emit/verdict | 310 / nondec_snip | 310 / nondec_snip | 310 / nondec_snip |
| S4pp-repro emit/verdict | 8810 / repro_ok | 8810 / repro_ok | 8810 / repro_ok |
| S4pp-ctrlA emit/top1 (ban) | 369 / 488 (v1-git, сверено с v5) | 369 / 488 | 369 / 488 |
| S4pp-ctrlB emit/top1 (ban) | 364 / 310 (v1-git, сверено с v5) | 364 / 310 | 364 / 310 |
| S4pp-ctrlTieA emit/top1 (ban) | 19 / 18 (v1-git, сверено с v5) | 19 / 18 | 19 / 18 |
| S4pp-ctrlTieB emit/top1 (ban) | 11639 / 3240 (v1-git, сверено с v5) | 11639 / 3240 | 11639 / 3240 |
| S2a emit/n_tl | 310 / 0 | 310 / 0 | 310 / 0 |
| S2b emit/n_tl | 310 / 0 | 310 / 0 | 310 / 0 |
| S3 probe emit при k=1,2,3,4,8,400 | 310 x6 | 310 x6 | 310 x6 |
| S3 ctrlA emit при k=1,2,3,4,8,400 | 488 x6 | 488 x6 | 488 x6 |

sha: v2 = be1b74d6c90c4c3c (13:50); v3 = 804699e7261ba3d8 (16:06);
v4 = c22c8711b2ef7291 (16:35); **v5 = 1289fd4ace5f0483 (17:24, этот артефакт)**.
wall (из артефактов/пакетов): v2=48.0, v3=55.2, v4=54.2, v5=60.8.
Исход v2-контролей и S2: v1-git (a9a6099, 05:47) — сверка с v5 выполнена
исполнителем: 10/10 значений (4 контроля + S2a/S2b + S4pp-1/2/ref/noop по
логам v5) идентичны; v2-артефакт не сохранился (перезаписан v3), его числа —
из r8-пакета (проза §E) и совпадают с v1-git по всем общим точкам.
Примечание: в v2 verdict строки S4pp-2 = «M1» (переименование в branchA —
r8/F3, с v3); все ЭМИССИИ во всех версиях одинаковы (T=0, seed=8181,
один сервер).

## D. Утверждения исполнителя (проверять только по тексту пакета)

- H1: Числа v5 = числам v3 = числам v2 по всем 24 запросам (таблица §C —
  теперь полная: S2a/S2b и все 4 контроля числами). T=0, seed=8181, один
  сервер; wall отличается (SYCL), эмиссии — нет.
- H2: Ни один из 4 MINOR r10 не остался: M1 — dict-chain + sent_logit_bias
  (коды §A, поля в §B; дифф цепей = no-op тройка); M2 — and-гейт (код §A,
  informative=False в §B); M3 — элиминация для строгого argmax с явным
  исключением S4pp-2 (reading в §B); C3 — verdict_mechanism с оговоркой (§B);
  C4 — spec = r3+r7+r8+r9+r10 (§B).
- H3: Главный вывод не изменился: третья ветка жива (S4pp-noop: тот же бан
  4383 — sent_logit_bias идентичен S4pp-2, различается только цепь (дифф =
  no-op тройка) → эмиссент 3039 = вершина отчёта). M1 снят; сдвиг — цепью
  (пере-ранжирующие сэмплеры; при строгом argmax — кандидат dry=1.0; на
  S4pp-2 допустим и рез в tie-классе). Вывод опирается на noop-плечо, НЕ на
  элиминацию.
- H4: gate2_met=false + gate2_caveat; S3: distinct_n_tl=1,
  report_changed_by_topk=false на обеих сторонах, distinct_emits=[310] /
  [488] (§B) ⇒ «зависимость НЕ ВЫЧИСЛЕНА», а не «независимость показана».
- H5: spec-поле артефакта v5 = r3+r7+r8+r9+r10 (правка строки выполнена ДО
  прогона v5; после прогона код не правился — артефакт = выход этого кода).
- H6 (по содержанию): вердикт v5 — пересказ по точкам; сдвиг атрибутирован
  цепью (кандидат dry=1.0 для строгого argmax) — канону S5/S6 (DRY=1.0
  объясняет Q2-флипы) в ту же сторону; ни одно поле артефакта не утверждает
  «M1 установлен».

## I. Формат вердикта (строго, суммарно <= 4000 символов)

```
VERDICT: PASS | PASS_WITH_MINORS | REVISE | BLOCK
MAJOR: <n>, BLOCK: <n>, MINOR: <n>, COSMETIC: <n>
(обоснование 1–3 строки)
FINDINGS:
- [F1|MAJOR|MINOR|COSMETIC] <пункт>: <что>
...
Строка для grep-гейта: VERDICT=<...>
S4PP_R11_DONE
```
