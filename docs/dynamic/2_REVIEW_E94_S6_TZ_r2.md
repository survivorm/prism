ВЕРДИКТ: PASS_WITH_MINORS

Ревью r2: сверка diff-области (правки по r1) ТЗ `docs/dynamic/0_PACKET_E94_S6_TZ.md`
с `experiments/exp94_s6_dry0.py` (+ `exp94_q2.py`, `exp94_q2rank.py`, `results/e94/Q2.json`,
`results/e94/S4e.json` для чисел). База не пересмотрена (r1: PASS_WITH_MINORS).

По пунктам:

1. MAJOR-1 (echo-гейт) — ЗАКРЫТО. §2 (строки 54–59): обязательный эхо-гейт live и офлайн из raw,
   `generation_settings.dry_multiplier == 0.0` на каждом ответе (вкл. первый live-запрос и каждый raw),
   нарушение → rc=14; в артефакт — эхо первого запроса (dry_multiplier/dry_base/dry_penalty_last_n).
   Код совпадает: `check_echo` (L114–125) падает при отсутствии поля и при отклонении >1e-12;
   в `main_run` вызывается на КАЖДОМ ответе, включая кэш-хит (L407, get_raw_s6 возвращает кэшированный
   ответ и чек за ним), `echo_first` снимается с первого ответа (L396–401) → `R["echo_first_request"]` (L427),
   halt «эхо-гейт…» → rc=14 (L499–500); `rebuild_from_raw` — check_echo по каждому raw (L272) →
   `echo_offline {n_checked, n_bad, ok}` (L325–327); `run_from_raw` — при `n_bad>0` (ok=False)
   `R["rc"]=14` + вердикт HALT (L591, L599–604, L612–614). §6: `echo_first_request` (L154) и
   `gates.echo_offline` (L153) на месте; RUN_RC §3 (L82–83) — «14 канон-дрейф или эхо»; §7 смоук —
   «эхо-чек каждого ответа» (код run_smoke L549). (См. новые findings 1 и 4 про место/объём эхо.)

2. MINOR-1 (R1-разбивка) — ЗАКРЫТО. §4 R1 (строки 100–103): «в detail — разбивка 14 RESOLVED-позиций
   по top1_agree_cpu (S6): позиции с вершиной ≠ cpu — шумовая составляющая, они не являются DRY-объяснением».
   Код: `verdict_rules(s6_set, q2_set, run_ok, per_position)` (L207–250) считает разбивку по
   `resolved` через `top1_agree_cpu` строк per_position (L223–227), выносит
   `resolved_by_top1_agree_cpu` (L228–229, L249) и в R1-detail пишет «разбивка RESOLVED по
   top1_agree_cpu: N agree / M disagree — disagree — шумовая составляющая, не DRY-объяснение» (L234–239).
   §6-контракт включает поле (L156). (Код добавляет третий ключ — finding 2.)

3. MINOR-2 (G2-фразировка) — ЗАКРЫТО. §3 (строки 75–78): спайк на w0 ДО полного прогона — жёсткий гейт
   (rc=2 или rc=13 при G1), «полный прогон не запускается»; G2 основной фазы — мягкий (g2_violations,
   «не убивают прогон»). Код: `run_s6` (L462–470) — `spike_cache` → `if not ok2: rc = 13 if any(
   f.startswith("G1")...) else 2; return finish(...)` — ДО вызова main_run, т.е. прогон не запускается;
   G2 основной фазы — только в `rebuild_from_raw` через `g2_main_check` (L291) → `g2_violations` в
   артефакт, без halt. (Недословность «со второго запроса каждого k» — finding 5.)

4. MINOR-3 (q2_emitted_rank_in_s6) — ЗАКРЫТО. §4 эффект-сайд (строки 113–117): g12, rank_reported
   (для RESOLVED тавтологично 1), q2_emitted_rank_in_s6 — ранг Q2-эмиссента в top_logprobs отчёта S6
   (w13 k=117: 4383 — rank 3, g12=0.008<1.0). Код: `effect_side(per_position, resolved, q2_emitted)`
   (L329–352) — qrank = index(qe в top20_ids S6-отчёта)+1 / «absent» / «no_probs», выводит
   `q2_emitted` и `q2_emitted_rank_in_s6` (L345–348). Q2-эмиссент — из Q2-raw (raw_q2.json.tar.gz)
   тем же `position_metrics_s6`, не из артефакта: `baseline_q2_flips` (L167–205, q2_emitted — L191),
   плюс кросс-чек raw_eq_artifact в baseline. Пример сверен по S4e.json (S4e-probe2, w13 k=117, dry=0:
   ids_head [3039, 3240, 4383, …] → rank(4383)=3, g12=0.00813). §6: `baseline.q2_emitted` (L139) и
   `effect_side.q2_emitted_rank_in_s6` (L151) на месте.

5. MINOR-4 (ids448) — ЗАКРЫТО. §1 (строки 38–42): явная конкатенация `ids448(doc) = ids(prompts.json[doc], 64)
   + gen_ids(REPLAY[doc], 384)` (448 id), `prompt(k) = ids448[:k]`, и явное замечание, что «docs[.ids] по 64»
   недостаточно для k>64. Код: S6 импортирует `load_data` из `exp94_q2` (L39–41); `exp94_q2.load_data`
   (L117–131) строит ровно `ids448 = i64(64) + g(384)` с assert длин 64/384/384; main_run шлёт
   `build_body_s6(tpl, ids448[:k])` (L393).

6. MINOR-5 («чистый») — ЗАКРЫТО. §4-Контроль (строки 108–111): «„Чистый" = та же самосогласованность,
   что flip (emitted == top1_reported в отчёте S6), а НЕ CPU-референс — иначе device-шум дал бы ложный
   R3 / сбой смоука». §7 (строки 168–169): смоук-ожидание «остальные чисты (emitted == top1_reported)».
   Код: flip = `rank_reported != 1` по СОБСТВЕННОМУ отчёту (position_metrics_s6 L156–164; no_probs-ветка
   flip=False L145–149); смоук проверяет `m["flip"]` (L552–553) и явно k=70 `emitted == 318 и
   top1_reported == 318` (L556–561) — та же метрика, CPU-референс в определениях не участвует
   (cpu_id/top1_agree_cpu — перекрёстно, «информация, не гейт», §4 строки 92–93).

7. MINOR-6 (решение утром) — ЗАКРЫТО. §4 (строки 118–120): «Канон после R1 (решение утром): если сработал
   R1, канон E94 = body_template S6 (s6_sha в S6.json); прежние артефакты (Q1, Q2, Q2rank, Q3, S3b)
   помечаются „DRY-on" — их числа валидны только как измерения DRY-on». Покрывает список r1 (Q2/Q2rank/Q3…)
   с излишком (Q1, S3b). Код пишет `R["canon"]["s6_sha"]` (L446–447).

8. MINOR-7 (no_probs) — ЗАКРЫТО. §4 (строки 89–91): «'no_probs' (пустой/отсутствующий top_logprobs) НЕ
   флип — конвенция Q2rank, отдельный счёт» и «'absent' (эмитента нет в непустом списке) — флип».
   Код: ветка no_probs в position_metrics_s6 — `rank_reported="no_probs"`, `flip=False`, `n_absent=0`
   (L141–149); absent — `n_absent=1`, `flip = rank_reported != 1` → True (L156–157, L164). Конвенция
   Q2rank сверена: exp94_q2rank.py L97 `fl = [r for r in rows if r["rank"] not in (1, "no_probs")]` —
   идентична.

Новые findings (все MINOR, diff-область):

- [MINOR] §6: контракт объявляет обёртку `gates` (g1_ok/g2_violations/g3_fd/echo_offline), но код обёртки
  не создаёт вовсе: `echo_offline` и `g2_violations` пишутся в верхнем уровне артефакта
  (rebuild_from_raw L325–327, L300; `R.update(...)`), ключей g1_ok/g3_fd в коде нет (G3 — как
  `fd_first_request`/`device_gate_source`, L423–426). Артефакт не совпадёт с контрактом по этому блоку;
  `echo_offline` — новое (diff) поле, лежащее в несуществующем в артефакте wrapper'е. Выровнять §6 с
  кодом (flat-поля) или обернуть в коде.
- [MINOR] §6: `verdict.resolved_by_top1_agree_cpu` в коде ВСЕГДА содержит третий ключ `missing`
  (verdict_rules L228; при R1 всегда 0, т.к. run_ok ⇒ полные per_position), в контракте только
  `{"agree", "disagree"}` (L156).
- [MINOR] §6/§4: при неполном прогоне код пишет `verdict.rule="HALT"` (verdict_rules L214, run_from_raw
  L599), а контракт — `"rule": "R1|R2|R3"` (L155); §4 не описывает вердикт при halt. Поведение правильное
  (R1–R3 не считаются на частичных данных — защита от ложного R1), но контракт должен это фиксировать.
- [MINOR] §2: «в generation_settings каждого ответа» — 6 ответов G2-спайка в коде эхо-чеку НЕ подвергаются
  (spike_cache не вызывает check_echo). Влияние нулевое: гейт сработает не позднее первого data-запроса
  (w0 k=64, L407) → rc=14 до того, как посчитана хоть одна позиция, ложный R1/R2/R3 исключён. Уточнить
  формулировку («каждого ответа основного прогона и каждого raw») или добавить чек в спайк.
- [MINOR] §3: G2 основной фазы — «со второго запроса каждого k» неточно: в основной фазе каждый k
  запрашивается один раз. Код (g2_main_check, exp94_q2.py L322–331) чекнет prompt_n ≤ 8 со второго
  запроса ДОКУМЕНТА (пропуск только k==64, т.е. k=65..447). Намерение совпадает; формулировку выровнять
  с docstring Q2 («со второго запроса документа (k=65)»).
- [MINOR] код (run_from_raw, L612–614): при пустом/отсутствующем /tmp/e94s6raw —
  echo_offline.n_checked=0 → ok=False → rc=14 с текстом «raw чужого канона». Fail-closed (безопасно),
  но сообщение вводит в заблуждение в случае «raw просто нет»; разделить ветки n_checked==0 и n_bad>0.

Проверено дополнительно (не findings): канон-дифф — `check_canon_diff` (L100–111) даёт ровно
{'dry_multiplier': [1.0, 0.0]} на фактическом Q1 body_template (dry_multiplier=1.0, dry_base=1.0 —
сверено по Q1.json); RUN_RC-строка кода (docstring) дословно совпадает с §3; `run_one.sh` — детачит
сам (setsid nohup), что согласуется с §7; 14 флипов/2304 позиции §1 сверены по Q2.json; смоук w4
k=64..75 = 12 запросов, кэш-идемпотентен (повторное чтение k=70 не ре-POST, L556–561).

PASS_WITH_MINORS
