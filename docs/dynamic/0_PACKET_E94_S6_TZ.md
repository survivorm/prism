# E94/S6: ТЗ — пересвип Q2 с `dry_multiplier=0.0` (GPU, iGPU/SYCL)

Статус: предрегистр ДО исполнения. Один исполнитель, один прогон.

## §0. Вопрос и гипотеза

S5 (CPU-перечит Q1) подтвердил H0: DRY-штраф 1.0 нат объясняет эмиссент на всех 6144
позициях Q1, 16/16 флипов — артефакт сэмплера. S4e показал на 2 адресах Q2-свопа
(`w4 k=70`: 310→318, `w13 k=117`: 4383→3039), что при `dry_multiplier=0.0`
эмиссент = вершина отчёта.

**S6 вопрос:** объясняет ли DRY **все** 14 флипов исходного Q2-свопа (teacher forcing,
6 доков × 384 позиции) — или остаётся остаток, который надо объяснять отдельно?

**H0-DRY:** при `dry_multiplier=0.0` на той же сетке флипов = 0. Тогда вся линия
«учительного форсинга» (Q2, Q2rank-forced) переписывается как измерение DRY, а не
свойства модели/устройства.

## §1. База (замороженные факты)

- Базовый артефакт: `results/e94/Q2.json` (закоммичен). 6 доков: `w0, w2, w4, w9, w13, w14`;
  k = 64..447 (384 позиции/док, всего 2304); seed 8181, T=0, n_predict=1,
  teacher forcing (prompt = ids448[:k]).
- **14 флипов** (top1_agree==false; совпадает с Q2rank arm_q2_forced n_flip=14):
  - w0: k=87 (40501/32085), k=94 (16/15)
  - w2: k=78 (1599/1248), k=88 (364/30)
  - w4: k=70 (310/318), k=90 (16/15), k=91 (16/15), k=92 (16/15)
  - w13: k=117 (4383/3039)
  - w14: k=86 (5995/264), k=87 (6311/220), k=101 (198/561), k=119 (198/561), k=137 (198/561)
- Канон Q2: `body_template` из `results/e94/Q1.json` + `prompt`, `n_predict=1`,
  `cache_prompt=true` (это ровно `build_body` из `experiments/exp94_q2.py`).
  Поля канона: temperature=0, stream=false, return_tokens=true, logprobs=true,
  top_logprobs=20, n_probs=20, seed=8181, top_k=400, top_p=1.0, min_p=0.0,
  typical_p=1.0, xtc_probability=0.0, mirostat=0, repeat_penalty=1.0,
  presence_penalty=0.0, frequency_penalty=0.0, repeat_last_n=0,
  **dry_multiplier=1.0, dry_base=1.0**.
- REPLAY-референс: `results/e81/REPLAY.json` (gen_ids, gen_p1 — CPU-вершина).
- Промпты: `results/e93/prompts.json` (docs[.ids] по 64). Построение ids448 — ровно
  `load_data` из `experiments/exp94_q2.py`: `ids448(doc) = ids(prompts.json[doc], 64) +
  gen_ids(REPLAY[doc], 384)` (448 id); `prompt(k) = ids448[:k]` — для k=64 это только
  64 промпта, для k=65..447 — префикс конкатенации (просто «docs[.ids] по 64»
  недостаточно для k>64).

## §2. ЕДИНСТВЕНное изменение

`dry_multiplier: 1.0 → 0.0`. Всё остальное — дословно канон Q2.

Проверка в рантайме (обязательна, до первого запроса):
- `diff_fields = {k: (q2[k], s6[k]) for k in set(q2)|set(s6) if q2.get(k)!=s6.get(k)}`
  (без prompt) обязана быть **ровно** `{'dry_multiplier': (1.0, 0.0)}`;
- `canon_sha` обоих bodyTemplate (sort_keys, компактные separators) записаны в артефакт.
При любом ином diff — `rc=14`, прогон не запускается (канонический дрейф).

Эхо сервера (обязательный гейт, live И офлайн из raw): diff проверяет то, что
ОТПРАВЛЕНО, а не что ПРИМЕНЕНО. В `generation_settings` каждого ответа основного
прогона (включая первый live-запрос и каждый кэш-хит) и каждого raw при пересборке
`dry_multiplier == 0.0` (ответы G2-спайка эхо-чеку не подвергаются — гейт сработает
не позднее первого data-запроса, до того как посчитана хоть одна позиция); нарушение
→ `rc=14` (зарегистрированный канон не применён — R1/R2/R3 легли бы на ложное
предположение). В артефакт — эхо ПЕРВОГО запроса: `dry_multiplier`, `dry_base`,
`dry_penalty_last_n`.

## §3. Протокол

Сервер — как в Q2 (`experiments/exp94_sycl_common.py`): `llama-server -m
/data/Qwen/qwen3.5-0.8B-f16.gguf -sm none -mg 0 -ngl 99 -fa on -c 2048 -np 1 --slots
--host 127.0.0.1`, порт из `[8111, 8112]` (pick_port), iGPU/SYCL (ZES_ENABLE_SYSMAN=1).
Пре-лонч: watchdog-лог + `ps` — чужих SYCL-серверов нет; RAM avail ≥ 10G.

Запросы: для каждого дока w ∈ {w0,w2,w4,w9,w13,w14}, для k = 64..447:
`POST /completion`, prompt = ids448[:k], канон §2. Idempotent-raw: кэш запросов в
`/tmp/e94s6raw/{w}_{k}.json` (повторный прогон не ре-POST). После прогона raw →
`results/e94/raw_s6.json.tar.gz`.

Гейты (как в Q2, `experiments/exp94_q2.py`):
- **G1** `tokens_evaluated == k` на каждом запросе → иначе `rc=13`.
- **G2** (кач): спайк на w0 ДО полного прогона (порядок k [64,64,128,128,192,192]) —
  жЁсткий гейт: провал → `rc=2` (или `rc=13` при G1), полный прогон не запускается;
  G2 основной фазы (со второго запроса документа, k=65..447; каждый k
  запрашивается один раз, `prompt_n <= 8`) — мягкий: нарушения
  → в артефакт `g2_violations`, не убивают прогон.
- **G3** fd-гейт устройства во время первого реального decode (20×0.05 с, fd_sampler);
  device-гейт: `/dev/dri` в `/proc/<pid>/fd` после health → иначе `rc=10`.
- Бюджет 40 мин → превышение `rc=1`; ≥2 упавших дока → `rc=11`.
- RUN_RC: 0 ok · 2 spike/cache-спайк · 10 device-гейт · 11 дока · 13 prompt-count ·
  14 канон-дрейф или эхо (канон не применён сервером) · 1 иное. Артефакт пишется при
  ЛЮБОМ исходе (`halt_reason`).

## §4. Определения и правила вердикта (предрегистр)

**flip(k)** = rank эмиссентного id в `top_logprobs` **собственного отчёта** сервера
на этой позиции ≠ 1 (rank по списку кандидатов; 'absent' = флип, фиксируется отдельно
`n_absent`). В определении явно: 'no_probs' (пустой/отсутствующий `top_logprobs`)
НЕ флип — конвенция Q2rank, отдельный счёт. В каждом raw: `emitted = completion_probabilities[0].id`,
`top1_reported = completion_probabilities[0].top_logprobs[0].id`.
Перекрёстно (информация, не гейт): `cpu_id` = gen_ids[j] (REPLAY), `top1_agree_cpu`.

Наборы: `S6_FLIPS`, `Q2_FLIPS` (14 позиций §1), `RESOLVED = Q2_FLIPS \ S6_FLIPS`,
`RESIDUAL = S6_FLIPS ∩ Q2_FLIPS`, `NEW = S6_FLIPS \ Q2_FLIPS`.

**Правила (строго по приоритету):**
- **R1 (H0-DRY):** `|S6_FLIPS| == 0` → H0-DRY подтверждена: все 14 флипов Q2-свопа =
  DRY-артефакт; линия учительского форсинга переписывается как измерение DRY. В
  detail — разбивка 14 RESOLVED-позиций по `top1_agree_cpu` (S6): позиции с
  вершиной ≠ cpu — шумовая составляющая, они не являются DRY-объяснением.
- **R2 (остаток):** `0 < |S6_FLIPS|` и `NEW == ∅` (все остатки из Q2_FLIPS) →
  DRY объясняет часть; по каждой позиции RESIDUAL: gap, rank, g12 — для отдельного
  объяснения (не в S6).
- **R3 (новый эффект):** `NEW ≠ ∅` → DRY-off ввёл новый эффект; вывод «всё = DRY»
  БЛОКИРУЕТСЯ, каждая позиция NEW — в артефакт с полным разбором.
- **HALT (неполный прогон):** halt или упавший док (в from-raw — офлайн-эхо: две
  ветки — raw пуст/отсутствует `n_checked=0` и raw чужого канона `n_bad>0`, обе
  rc=14 с разными detail) → R1–R3 на частичных данных не считаются (защита от
  ложного R1): `verdict.rule = "HALT"`, `rc` по причине halt (§3).
- **Контроль:** позиции, чистые в Q2 (2290), обязаны остаться чистыми; любое
  нарушение — это NEW (R3). «Чистый» = та же самосогласованность, что flip
  (`emitted == top1_reported` в отчёте S6), а НЕ CPU-референс — иначе device-шум дал бы
  ложный R3 / сбой смоука. Адреса S4e-контролей — w13 k=101 (вершина 488) и
  w4 k=69 (вершина 8810); обе позиции чисты в Q2 и лежат внутри этих 2290,
  отдельным контролем не выделяются.
- **Эффект-сайд (обязательно в артефакт):** для каждой позиции RESOLVED — `g12`
  (разрыв top2 в отчёте S6, logit-единицы), `rank_reported` (для RESOLVED тавтологично
  1) и `q2_emitted_rank_in_s6` — ранг Q2-эмиссента в `top_logprobs` отчёта S6
  (w13 k=117: 4383 — rank 3, g12=0.008<1.0): «насколько далеко DRY сдвигал вершину».
- **Канон после R1 (решение утром):** если сработал R1, канон E94 = body_template S6
  (`s6_sha` в S6.json); прежние артефакты (Q1, Q2, Q2rank, Q3, S3b) помечаются
  «DRY-on» — их числа валидны только как измерения DRY-on.

## §5. Что НЕ проверяется (радиус)

1. Не доказывает, что флипы Q1 free-generation = DRY (это уже сделано в S5 отдельно,
   16 док × 384, H0 подтверждена).
2. Не измеряет величину штрафа при ДРУГИХ настройках (base≠1.0, last_n≠64).
3. Не утверждает, что «сервер сломан»: поведение штатное, сломан наш канон запроса.
4. Сетка = ровно 6 доков × k 64..447 исходного Q2. Выезжать за её пределы S6 не обязан.
5. Не пересчитывает логитные метрики до сэмплера (g12-контрасты и т.п. — отдельный
   вопрос; S6 измеряет только эмиссию).

## §6. Контракт артефакта `results/e94/S6.json`

```
{
  "tz": "docs/dynamic/0_PACKET_E94_S6_TZ.md",
  "baseline": {"q2_sha16": "...", "n_positions_raw": 2304,
               "n_flip_from_raw": 14, "n_flip_artifact": 14, "raw_eq_artifact": true,
               "flip_set": {"w0": [87,94], ...},
               "flip_set_artifact": {...},
               "q2_emitted": {"w13k117": 4383, ...}},  # Q2-флип-позиции из raw
  "canon": {"s6_sha": "...", "q2_sha": "...", "diff_fields": {...}},
  "port": N, "server_pid": N, "fd_after_health": [...], "device_gate_source": "...",
  "fd_first_request": {"n_samples": N, "n_samples_with_dri": N, "dri_union": [...]},
  "docs": ["w0","w2","w4","w9","w13","w14"],
  "per_position": { "<w>": [{k, j, emitted, top1_reported, cpu_id, rank_reported,
                             g12, top1_agree_cpu, p_max_gpu, d_pmax, top20_ids}, ...] },
  "summary": {"overall": {"n_pos": 2304, "median_abs_d_pmax": ...,
                          "bootstrap_ci_2.5_97.5": [...], "top1_agreement_reported": ...,
                          "top1_agreement_cpu": ..., "n_flip": N, "n_absent": N},
              "per_doc": {"doc:w0": {...}, ...}},
  "doc_status": {"w0": {"n_pos": N, "missing_raw": 0, "bad": 0, "complete": true}, ...},
  "n_docs_ok": 6, "n_docs_failed": 0,
  "flips": {"s6_set": {...}, "q2_set": {...}, "resolved": [...],
            "residual": [...], "new": [...]},
  "effect_side": { "<w>k": {"g12": ..., "rank_reported": 1,
                           "q2_emitted": ..., "q2_emitted_rank_in_s6": ...}, ... },  # RESOLVED
  "spike_g2": {"order": [64,64,128,128,192,192], "rows": [...], "pass": true, "failures": []},
  "g1_ok": true, "g3_fd": true,
  "g2_violations": [], "g2_violations_n": 0, "g2_wall_last32_first32": {...},
  "g2_offline_check": {...},
  "echo_offline": {"n_checked": 2304, "n_bad": 0, "ok": true},
  "echo_first_request": {"dry_multiplier": 0.0, "dry_base": ..., "dry_penalty_last_n": ...},
  "verdict": {"rule": "R1|R2|R3|HALT", "H0_dry": bool, "n_s6_flips": N,
              "resolved_by_top1_agree_cpu": {"agree": N, "disagree": M, "missing": 0} | null,
              "detail": "..."},  # null при resolved=∅ или HALT; в R1 missing всегда 0
  "rc": N, "halt_reason": null, "t_total_s": ...,
  "raw_tar": "results/e94/raw_s6.json.tar.gz"
}
```

Все float — через `san()` (AGENTS.md #4); артефакт пишется до `sys.exit(rc)`.

## §7. Самодиагностика и запуск

- `--smoke`: 1 док `w4`, k ∈ {64..75} (12 запросов, включает флип k=70) + канон-diff
  чек + эхо-чек каждого ответа смоук-прогона (включая повторное чтение k=70, кэш-хит). Ожидание: k=70 резолвится (emitted == top1_reported ==
  318), остальные чисты (emitted == top1_reported), `diff_fields == {'dry_multiplier'}`.
  Провал смоука — НЕ запускать полный прогон.
- Полный: `./run_one.sh e94_s6 "python3 -u experiments/exp94_s6_dry0.py"`.
- Само-аудит кода — ДО запуска (чек-лист AGENTS.md: одинаковая статистика сторон,
  ветки/guard'ы, json.dump-sanitize, setsid-nohup не нужен — run_one сам detached,
  контроли читаются первыми).
