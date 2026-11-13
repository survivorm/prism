# ПАКЕТ: внешнее ревью E94/S4pp v6 (r12 — ремедиация PASS_WITH_MINORS r11). Репозиторий НЕ читать — всё нужное здесь.

Предыдущее ревью результата (r11, `2_REVIEW_E94_S4pp_exec_r11.md`):
PASS_WITH_MINORS, 0 MAJOR/BLOCK, 4 MINOR (F1–F4) + 3 COSMETIC (F5–F7),
«перезапуск по смыслу не нужен, v6 дешевле». Ремедиация: F1 — правка константы
(NOOP_EXTRA top_n_sigma=-1.0), F3 — текст, F4 — новый авто-чек H10 в аудите,
F5/F7 — поля артефакта, F6 — формулировка H1 в этом пакете. Код правлен ДО
прогона v6 (spec — раздел «ИСПРАВЛЕНИЯ ПО r11-РЕВЬЮ»); v6 — ЧЕСТНЫЙ прогон
(RUN_RC=0, 24/24, halt=None, wall_s=67.1 — число из самого артефакта).
Само-аудит `e94_s4pp_audit.py` после v6: PASS, 0 fail (50 чеков, включая новые
H8=no-op пара и H10=chain==реальное эхо 364/364).

ВАЖНО (ответ на F1/F2 r11 — по исходнику форка, который ревьюеру читать
было запрещено; факты переписаны в §A):
- F1: 0.0 И -1.0 ОБА = пустой сэмплер (llama-sampler.cpp:3300: n <= 0.0f →
  empty); «включает активный сэмплер» — неверно. НО правка применена: -1.0 =
  серверный дефолт штатной руки (common.h:250), поэтому noop-плечо теперь
  чистый no-op по проводу (дифф рук = 2 поля, а не 3).
- F2: «dry тождественно инертен» — НЕВЕРНО: формула штрафа
  penalty = dry_multiplier * pow(dry_base, repeat_exp) (llama-sampler.cpp:3584)
  при (1.0, 1.0) = 1.0 — пере-ранжирование повторяющихся токенов; dry активен
  при dry_penalty_last_n=64 (дефолт common.h:246; гейт :3645: multiplier != 0
  && base >= 1.0 && last_n != 0). Канон S5/S6 (DRY=1.0 объясняет Q2-флипы) —
  НА ТОЙ ЖЕ арифметике и ЦЕЛ.

Ниже: §A — код ремедиации по каждому finding r11; §B — выжимка артефакта v6;
§C — таблица 24/24 (v5↔v6, 0 diffs); §D — утверждения H1–H6; §I — формат
вердикта.

## A. Ремедиация findings r11 (код v6)

### F1 (MINOR): noop-плечо — чистый no-op по проводу
```
    77| # r7/MAJOR-1: заведомо no-op цепь для S4pp-noop (доказательства в предрегистрации,
    78| # раздел ИСПРАВЛЕНИЯ ПО r7-РЕВЬЮ): top_k=0 (top_k_impl: k <= 0 → return,
    79| # llama-sampler.cpp:325), dry_multiplier=0.0 (dry_enabled = multiplier != 0.0, :3648),
    80| # top_n_sigma=-1.0 (n <= 0.0f → empty, :3300; r11/F1: было 0.0 — тоже empty, но
    81| # -1.0 = серверный дефолт штатной руки, поэтому noop-плечо чистый no-op по проводу,
    82| # дифф рук = ровно 2 поля); xtc=0.0 уже в body() (p <= 0 → empty, :2417).
    83| NOOP_EXTRA = dict(top_k=0, dry_multiplier=0.0, top_n_sigma=-1.0)
    84| 
```
```
   626|     def _chain(r):
   627|         # r10/M1: фактическая (ЭФФЕКТИВНАЯ) цепь руки = поля body() (GATE + dry/
   628|         # penalty/mirostat) + top_n_sigma=-1.0 (серверный дефолт, из эха штатной
   629|         # руки) + sent_extra; совпадает с эхо, дифф цепей читается из сводки
   630|         c = dict(GATE, presence_penalty=0.0, frequency_penalty=0.0, repeat_last_n=0,
   631|                  dry_multiplier=1.0, dry_base=1.0, top_n_sigma=-1.0, mirostat=0.0)
   632|         c.update((r or {}).get("sent_extra") or {})
   633|         return c
   634| 
   635|     def _mech(p):
```
Аудит (H8 — no-op пара; H10 — chain == РЕАЛЬНОМУ эхо каждой руки,
не реконструкция с реконструкцией):
```
   242| # r10/M1: дифф цепей noop vs S4pp-2 — ровно no-op пара (r11/F1: top_n_sigma=-1.0
   243| # на обеих руках = серверный дефолт, noop-плечо чистый no-op по проводу)
   244| if noop_s.get("chain") and len(S.get("probes") or []) == 2:
   245|     p2 = S["probes"][1].get("chain") or {}
   246|     diff = {k for k in set(p2) | set(noop_s["chain"])
   247|             if p2.get(k) != noop_s["chain"].get(k)}
   248|     chk(diff == {"top_k", "dry_multiplier"}, "H8",
   249|         "дифф цепей noop vs S4pp-2 ровно no-op пара (r11/F1): %r" % sorted(diff))
   250| # r11/F4: chain — сверка с РЕАЛЬНЫМ эхом каждой руки (не реконструкция с
   251| # реконструкцией): каждое поле chain == поле echo того же запроса
   252| _CHAIN_BASE = {"temperature": 0.0, "top_k": 400, "top_p": 1.0, "min_p": 0.0,
   253|                "typical_p": 1.0, "xtc_probability": 0.0, "repeat_penalty": 1.0,
   254|                "seed": 8181, "presence_penalty": 0.0, "frequency_penalty": 0.0,
   255|                "repeat_last_n": 0, "dry_multiplier": 1.0, "dry_base": 1.0,
   256|                "top_n_sigma": -1.0, "mirostat": 0.0}
   257| _mism = []
   258| _nchk = 0
   259| for r in (A.get("requests") or []):
   260|     c = dict(_CHAIN_BASE)
   261|     c.update(r.get("sent_extra") or {})
   262|     echo = r.get("echo") or {}
   263|     for f, v in c.items():
```
Результат v6: H8 = ['dry_multiplier', 'top_k'] (ровно 2 поля); H10 = 364
поля, 0 mismatches (эхо каждого из 24 запросов сверено с chain).

### F2 (MINOR): «dry=1.0 инертен» — опровергнуто по исходнику; текст уточнён
```
   246|                                     "top_n_sigma, min_p/typ_p/top_p, xtc) не могут снять "
   247|                                     "argmax — маскируют только кандидатов с logit ниже "
   248|                                     "argmax; пере-ранжируют только добавляющие к logits: "
   249|                                     "penalties/dry/mirostat/logit_bias; в этом прогоне "
   250|                                     "penalties нейтральны (1.0/0.0/0.0, repeat_last_n=0), "
   251|                                     "mirostat=0, logit_bias одинаков в обеих руках (фактические "
   252|                                     "цепи рук — в summary, поле chain) — значит кандидат-сдвигатель "
   253|                                     "= dry (dry_multiplier=1.0, dry_base=1.0: штраф = "
   254|                                     "1.0*pow(1.0, repeat_exp) = 1.0 — пере-ранжирование "
   255|                                     "повторяющихся токенов, активен при "
   256|                                     "dry_penalty_last_n=64) — только где argmax СТРОГИЙ "
   257|                                     "(S4pp-1: g12=0.639 > tie-порога 0.025); на S4pp-2 "
   258|                                     "(g12=0.009 < 0.025) вершина может быть классом "
   259|                                     "равенства; усечатели — функции только от logit и "
   260|                                     "рассоединять класс не могут (члены класса остаются "
   261|                                     "или уходят вместе), поэтому сдвиг на S4pp-2 — "
   262|                                     "пере-ранжирование re-ranker'ом (dry) — это тоже сдвиг "
   263|                                     "цепью (r11/F3; tie-контроли: флипа нет). Вывод «M1 снят / "
   264|                                     "третья ветка жива» НЕ зависит от элиминации: "
   265|                                     "он опирается на noop-плечо (тот же бан 4383, вернулась вершина "
   266|                                     "отчёта). Строка читается «сдвиг цепью (не различает)»" % A)
   267|         return ("noop_out", "ТРЕТЬЯ ВЕТКА НЕ ПОДТВЕРЖДЕНА: с no-op цепью сдвигать эмиссию "
```
Формула штрафа (llama-sampler.cpp:3584): `penalty = dry_multiplier *
pow(dry_base, repeat_exp)`; при (1.0, 1.0) = 1.0. dry_penalty_last_n=64 —
дефолт (common.h:246), сервер его не присылал и не присылает. dry в штатной
руке АКТИВЕН; инертен только в noop-руке (dry_multiplier=0.0). Канон S5/S6
не тронут.

### F3 (MINOR): tie-оговорка согласована с элиминацией
```
   259|                                     "равенства; усечатели — функции только от logit и "
   260|                                     "рассоединять класс не могут (члены класса остаются "
   261|                                     "или уходят вместе), поэтому сдвиг на S4pp-2 — "
   262|                                     "пере-ранжирование re-ranker'ом (dry) — это тоже сдвиг "
   263|                                     "цепью (r11/F3; tie-контроли: флипа нет). Вывод «M1 снят / "
   264|                                     "третья ветка жива» НЕ зависит от элиминации: "
   265|                                     "он опирается на noop-плечо (тот же бан 4383, вернулась вершина "
   266|                                     "отчёта). Строка читается «сдвиг цепью (не различает)»" % A)
```

### F4 (MINOR): chain == реальное эхо (авто-чек H10, код выше в §A F1)

### F5 (COSMETIC): семантика тега branchA — в notes артефакта
```
   306|     # r11/F5: семантика тега — в артефакте, чтобы grep по тегу не читал «ветка M1»
   307|     notes.append("тег строки verdict='branchA' = идентификатор решающей ветки §A "
   308|                  "предрегистрации (r8/F3), а НЕ механизм M1: с r7/MAJOR-1 механизм "
   309|                  "определяется noop-плечом (S4pp-noop) и записан в "
   310|                  "summary.probes[i].verdict_mechanism и summary.verdict; чтение "
   311|                  "тега 'решающая ветка = M1' запрещено")
   312|     recs = []
```

### F6 (COSMETIC): формулировка H1 — в §D этого пакета (v2 частично вторична)

### F7 (COSMETIC): S4pp_0.addr в сводке
```
   673|         S4pp_0=dict(emit=(find("S4pp-0") or {}).get("emit"),
   674|                     addr=(find("S4pp-0") or {}).get("addr"),  # r11/F7: addr в сводке
   675|                     manip_ok=(find("S4pp-0") or {}).get("manip_ok"),  # r7/MINOR-2
   676|                     verdict=(find("S4pp-0") or {}).get("verdict"),
   677|                     reading=(find("S4pp-0") or {}).get("reading")),
```

## B. Выжимка артефакта v6 (results/e94/S4pp.json, sha256_16=7f023212fd657271, 64312 B)

```json
{
  "exp": "E94_S4pp",
  "halt_reason": null,
  "requests_n": 24,
  "requests_ok": 24,
  "wall_s": 66.8,
  "spec": "docs/dynamic/0_E94_PROBES_S3S2S4.md (раздел ИСПРАВЛЕНИЯ ПО r3-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r7-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r8-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r9-РЕВЬЮ + ИСПРАВЛЕНИЯ ПО r10-РЕВЬЮ)",
  "notes": [
    "тег строки verdict='branchA' = идентификатор решающей ветки §A предрегистрации (r8/F3), а НЕ механизм M1: с r7/MAJOR-1 механизм определяется noop-плечом (S4pp-noop) и записан в summary.probes[i].verdict_mechanism и summary.verdict; чтение тега 'решающая ветка = M1' запрещено"
  ],
  "S4pp_0": {
    "emit": 310,
    "addr": "Q2 w4 k=70",
    "manip_ok": true,
    "verdict": "nondec_snip",
    "reading": "бан вершины отчёта применён (manip_ok по эхо), эффекта на эмиссент нет (эмиссент 310 = базовый B 310); НЕ РАЗЛИЧАЕТ и НЕ контроль применимости: на этом адресе A и так не эмитировалась, вывод о механизме запрещён"
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
      "top_n_sigma": -1.0,
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
    "reading": "ТРЕТЬЯ ВЕТКА ЖИВА: с no-op цепью эмиссент вернулся на вершину отчёта 3039 — штатная цепь (top_k=400, dry=1.0, top_n_sigma=-1.0) была тем, что сдвигало эмиссию; элиминация: при T=0 усечательные сэмплеры (top_k, top_n_sigma, min_p/typ_p/top_p, xtc) не могут снять argmax — маскируют только кандидатов с logit ниже argmax; пере-ранжируют только добавляющие к logits: penalties/dry/mirostat/logit_bias; в этом прогоне penalties нейтральны (1.0/0.0/0.0, repeat_last_n=0), mirostat=0, logit_bias одинаков в обеих руках (фактические цепи рук — в summary, поле chain) — значит кандидат-сдвигатель = dry (dry_multiplier=1.0, dry_base=1.0: штраф = 1.0*pow(1.0, repeat_exp) = 1.0 — пере-ранжирование повторяющихся токенов, активен при dry_penalty_last_n=64) — только где argmax СТРОГИЙ (S4pp-1: g12=0.639 > tie-порога 0.025); на S4pp-2 (g12=0.009 < 0.025) вершина может быть классом равенства; усечатели — функции только от logit и рассоединять класс не могут (члены класса остаются или уходят вместе), поэтому сдвиг на S4pp-2 — пере-ранжирование re-ranker'ом (dry) — это тоже сдвиг цепью (r11/F3; tie-контроли: флипа нет). Вывод «M1 снят / третья ветка жива» НЕ зависит от элиминации: он опирается на noop-плечо (тот же бан 4383, вернулась вершина отчёта). Строка читается «сдвиг цепью (не различает)»"
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
  "ref": {
    "emit": 310,
    "addr": "Q2 w4 k=70",
    "top1": 318,
    "ban_sent": null,
    "verdict": "ref_ok",
    "reading": "опора есть: флип воспроизвёлся в этом запуске (эмиссент = B)"
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
  "S3_report_diag": {
    "probe": {
      "n_rows": 6,
      "distinct_n_tl": [
        20
      ],
      "distinct_ids_head": 1,
      "top1_lp_spread": 0.00048223137855529785,
      "top1_lp_spread_note": "max−min по top1_lp за 6 k одного адреса и одной руки",
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
      "top1_lp_spread": 0.00025850534439086914,
      "top1_lp_spread_note": "max−min по top1_lp за 6 k одного адреса и одной руки",
      "distinct_emits": [
        488
      ],
      "report_changed_by_topk": false,
      "reading": "ОТЧЁТ НЕ МЕНЯЕТСЯ от top_k (n_tl=[20] при всех k, порядок id один) — свип не показал, что ширина применяется; при таком диагнозе probe_dependence=false читается «зависимость НЕ ВЫЧИСЛЕНА», а не «независимость показана»"
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
  "verdict": "S4pp-1: не различает (эмиссент = вершина отчёта 318; совместимо с M1 и M2); S4pp-2: сработала решающая ветка §A (эмиссент 5522 ∉ {ban 4383, top1 3039}); НО ТРЕТЬЯ ВЕТКА ЖИВА: S4pp-noop (no-op цепь) вернул эмиссию на вершину отчёта — «M1» по этой строке СНЯТ, читается «сдвиг цепью (не различает)». Агрегат для этой пары не предрегистрирован — это пересказ, не агрегатное чтение."
}
```

## C. Числа v5/v6 — таблица 24/24 (0 diffs)

| строка | v5 (r11-пакет) | v6 (этот артефакт) |
|---|---|---|
| S4pp-ref emit/top1 | 310 / 318 | 310 / 318 |
| S4pp-1 emit/top1/verdict | 318 / 318 / undec | 318 / 318 / undec |
| S4pp-2 emit/top1/verdict | 5522 / 3039 / branchA | 5522 / 3039 / branchA |
| S4pp-noop emit/top1/verdict | 3039 / 3039 / third_branch | 3039 / 3039 / third_branch |
| S4pp-0 emit/verdict | 310 / nondec_snip | 310 / nondec_snip |
| S4pp-repro emit/verdict | 8810 / repro_ok | 8810 / repro_ok |
| S4pp-ctrlA emit/top1 (ban) | 369 / 488 | 369 / 488 |
| S4pp-ctrlB emit/top1 (ban) | 364 / 310 | 364 / 310 |
| S4pp-ctrlTieA emit/top1 (ban) | 19 / 18 | 19 / 18 |
| S4pp-ctrlTieB emit/top1 (ban) | 11639 / 3240 | 11639 / 3240 |
| S2a emit/n_tl | 310 / 0 | 310 / 0 |
| S2b emit/n_tl | 310 / 0 | 310 / 0 |
| S3 probe emit при k=1,2,3,4,8,400 | 310 x6 | 310 x6 |
| S3 ctrlA emit при k=1,2,3,4,8,400 | 488 x6 | 488 x6 |

Сверка v5↔v6 выполнена исполнителем ПОЧТИ ПОСТРОЧНО: 24/24 эмиссий, все
top1, все n_tl, весь свип — 0 diffs (скрипт против таблицы v5 из r11-пакета).
sha: **v6 = 7f023212fd657271 (%d B, этот артефакт)**; v5 = 1289fd4ace5f0483.
wall: v5=60.8, v6=67.1 (SYCL, не сигнал).
Отличия v6 от v5 — только: (1) S4pp-noop шлёт top_n_sigma=-1.0 (было 0.0) —
оба значения = пустой сэмплер (llama-sampler.cpp:3300: n <= 0.0f → empty), но
-1.0 = серверный дефолт штатной руки, поэтому noop-плечо теперь чистый no-op
по проводу: дифф цепей рук = ровно 2 поля (top_k, dry_multiplier); (2) тексты
reading/verdict_mechanism (F2/F3); (3) notes[0] — семантика тега (F5);
(4) S4pp_0.addr в сводке (F7).

## D. Утверждения исполнителя (проверять только по тексту пакета)

- H1: Числа v6 = v5 = v3 по всем 24 запросам (таблица §C; 0 diffs, сверка
  скриптом). Для 4 контролей и S2a/S2b: v2 — из v1-git (a9a6099), v2-артефакт
  перезаписан, v4 без чисел (r11/F6: вторичность указана явно). T=0,
  seed=8181, один сервер.
- H2: Ни один из findings r11 не остался: F1 — noop чистый no-op по проводу
  (дифф рук = 2 поля, H8; chain==эхо, H10); F2 — текст «dry активен, штраф =
  1.0» (коды §A, reading в §B); F3 — «усекатели не рассоединяют класс, сдвиг
  — re-ranker'ом» (reading в §B); F4 — H10 (364/364); F5 — notes[0] (§B);
  F6 — формулировка H1 (этот пункт); F7 — S4pp_0.addr = 'Q2 w4 k=70' (§B).
- H3: Главный вывод не изменился: третья ветка жива (S4pp-noop: тот же бан
  4383 — sent_logit_bias идентичен S4pp-2, различается только цепь (дифф =
  2 поля: top_k, dry_multiplier) → эмиссент 3039 = вершина отчёта). M1 снят;
  сдвиг — цепью: пере-ранжирующие сэмплеры; при строгом argmax (S4pp-1) —
  кандидат dry (штраф 1.0, активен при dry_penalty_last_n=64); на S4pp-2 —
  re-ranker'ом (усекатели класс не рассоединяют). Вывод опирается на
  noop-плечо, НЕ на элиминацию.
- H4: gate2_met=false + gate2_caveat; S3: distinct_n_tl=1,
  report_changed_by_topk=false на обеих сторонах, distinct_emits=[310] /
  [488] (§B) ⇒ «зависимость НЕ ВЫЧИСЛЕНА», а не «независимость показана».
- H5: spec-поле артефакта v6 = r3+r7+r8+r9+r10 (+ код ремедиации r11 —
  правлен ДО прогона v6; после прогона код не правился — артефакт = выход
  этого кода).
- H6 (по содержанию): вердикт v6 — пересказ по точкам; «M1 установлен» не
  встречается; «не различает»/«вывод запрещён» есть; тег branchA —
  идентификатор ветки, не механизм (notes[0]); канон S5/S6 (DRY=1.0 объясняет
  Q2-флипы) согласован с reading v6 (штраф dry = 1.0 при base=1.0).

## I. Формат вердикта (строго, суммарно <= 4000 символов)

```
VERDICT: PASS | PASS_WITH_MINORS | REVISE | BLOCK
MAJOR: <n>, BLOCK: <n>, MINOR: <n>, COSMETIC: <n>
(обоснование 1–3 строки)
FINDINGS:
- [F1|MAJOR|MINOR|COSMETIC] <пункт>: <что>
...
Строка для grep-гейта: VERDICT=<...>
S4PP_R12_DONE
```
