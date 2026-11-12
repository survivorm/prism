# Ревью исполнения E94/S6 (внешний, r1)

Ты — внешний ревьюер результата. Прогон S6 завершён: `RUN_RC=0`, вердикт **R1 / H0-DRY**.
Твоя задача — проверить, что вердикт следует из артефактов (не из доверия), что контракт
соблюден, и что нет методологических дыр. Формулировка R1 в ТЗ предрегистрирована — сверяй
её с фактами, а факты — с сырьём.

## Файлы

- Артефакт: `results/e94/S6.json` (main-прогон) — после `--from-raw` в нём же поля
  `idempotence` (rebuild ×2) и обновлённые `echo_offline`/`t_rebuild_s`
- Raw-корпус: `results/e94/raw_s6.json.tar.gz` (2304 raw, каталог `raw/`)
- ТЗ (предрегистрация): `docs/dynamic/0_PACKET_E94_S6_TZ.md` — определения §4, контракт §6,
  RUN_RC §3, смоук §7
- Код: `experiments/exp94_s6_dry0.py` (процедура одна для live и from-raw)
- База: `results/e94/Q2.json` (DRY-on: 14 флипов), `results/e94/Q1.json` (body_template DRY-on),
  `results/e94/raw_q2.json.tar.gz` (Q2-raw, источник baseline)
- Смоук-лог: `/tmp/e94_s6_smoke.log`, основной лог: `/tmp/e94_s6.log`,
  from-raw лог: `/tmp/e94_s6_fromraw.log`

## Что проверять (по пунктам)

1. **Вердикт R1 из артефакта**: `per_position` (2304 позиции) — 0 флипов по определению
   §4 (flip = rank эмиссента в СОБСТВЕННОМ top_logprobs отчёте ≠ 1; 'no_probs' — не флип;
   'absent' — флип). Пересчитай flip-счёт из raw-тар-архива самостоятельно (не верь
   `summary`) и сравни с `flips.s6_set`.
2. **Baseline 14**: `baseline.flip_set` == 14 позиций из `Q2.json` AND из `raw_q2.json.tar.gz`
   (одна процедура; `raw_eq_artifact` должен быть true). Список позиций сверь с ТЗ §1.
3. **Эхо-канон**: `canon.diff_fields == {'dry_multiplier': [1.0, 0.0]}`; `echo_first_request`
   (dry_multiplier=0.0); `echo_offline` — n_checked=2304, n_bad=0 (проверка по ВСЕМ raw).
   Спот-чек: открой 3–5 raw случайно, посмотри generation_settings.dry_multiplier.
4. **G1/G2**: g1_ok (tokens_evaluated == k на всех 2304), g2_violations_n=0,
   g2_offline_check (prompt_n со второго запроса документа <= 8 — факт кэширования промпта).
   Спот-чек prompt_n в raw.
5. **Идемпотентность**: `idempotence.artifact_identical` == true (rebuild из raw ×2, sha256).
6. **Контракт §6**: все ключи на месте, flat-гейт-поля (g1_ok/g3_fd/g2_violations/echo_offline),
   verdict-блок, effect_side (14 RESOLVED, w13k117: q2_emitted=4383, rank 3, g12≈0.0096).
7. **Эффект-сайд интерпретация**: 13 позиций с q2_emitted_rank_in_s6=2, 1 — с 3: «DRY-демоция
   на 1–2 ранга» — сверь: для каждой из 14 позиций возьми raw Q2 (эмитент) и raw S6 (top_logprobs),
   пересчитай ранг эмиссента Q2 в отчёте S6.
8. **S4e-контроли**: w13 k=101 (488) и w4 k=69 (8810) — чисты в S6 (emitted == top1_reported).
9. **Методологические дыры**: что НЕ доказывает R1 (радиус §5)? Есть ли путь, при котором
   0 флипов в S6 — артефакт настройки (напр. n_predict=1 vs free-generation), а не DRY?
   Оцени, насколько R1 переносится на free-generation (S5: 16 док × 384 уже подтвердили H0
   отдельно — это в артефакте S5.json).
10. **Цифры в логах**: `S6: ok=6/6 n_flip_s6=0 rule=R1 H0_dry=True top1_agreement_reported=1.0`
    — сверь с артефактом. t_total_s=214.9 (бюджет 40 мин).

## Формат

`docs/dynamic/2_REVIEW_E94_S6_exec_r1.md`: ВЕРДИКТ (PASS / PASS_WITH_MINORS / FAIL),
затем findings `[MAJOR]`/`[MINOR]` с line/файл-ссылками, затем «Проверено дополнительно».
Числа — только из артефактов/логов, каждый пересчёт — с указанием, откуда взяты данные.
Финальный маркер в конце файла: `REVIEW_DONE`.
