# MARKERS — реестр метрик и конвенций (R9-8)
> Гейт программы: имя метрики без записи здесь = анализ невалиден. Каждая запись: формула,
> реализация (файл), конвенция, пины/допуски. Дополняется, не переопределяется молча —
> правка записи = строка CHANGELOG внизу.

## M1. dip (V-оценщик)
- Формула: u_t=h_t/‖h_t‖; V_t=u_{t+1}−(u_{t+1}·u_t)u_t; dip=median cos(V_t,V_{t+1}).
- Реализация: curve_u в exp62_batt_lib.py (dip инлайн в run()/dip1); dcurve — копии в exp63_t17_dipnull.py/exp64_*; пин восстановленного
  оригинала results/e64/ESTIMATOR_PIN.json (sha pyc 3/3 совпадение).
- **Границы (r1-D1): при <2 пар — nan (fail-loud). Ноль — только результат, не default.**
- Канон-уровень: corpus real dip=−0.0722 (L23, pooled nonwiki, ladder v2b); регресс-допуск
  прогона ±0.002. Контрасты-пины: iid-вход 0.000; AR(1)+0.414.

## M2. null-конвенции (три, не смешивать)
- perm-mean2: ядерные K(k) — 2 пермутации/док, среднее (e83/baseline v2).
- iid-sign floor≈0.07: R-ряды/послесвечение (e82b).
- cluster-bootstrap по докам (сида внутри дока = кластер) для попарных дельт;
  moving-block (блок16) для рядов с автокорреляцией. IID-позиционный bootstrap запрещён.

## M3. lift^state vs lift^incr (два объекта! — B5)
- lift^state: V=состояния−mean, SVD, lstsq, r=48, без ridge (exp64_lambda1.py).
- lift^incr: V=H[1:]−H[:-1], ridge 1e-3, r=64 (exp67_quantolike.py).
- Числа между ними не сравнивать; λ₁ отчитывать только с субскриптом.

## M4. plateau (три окна — не путать)
- readmatrix: mean(k=9..24); P3: median(k=9..96); TPU-ядро: med(8..96). В новых доках
  только с субскриптом окна.

## M5. seam_delta (лестница v2b, r1-C1)
- По доку: unmasked−(1−1/b)·masked; репорт = медиана per-doc (cluster bootstrap по докам,
  axis=1!). Разность пуловых медиан отличается до 0.0014 (blk16 знак меняется) — ручной
  чек по pole: поле seam_pooled в results/e90/LADDER_V2.json.

## M6. post-novelty cost kernel (бывш. «novelty», r1-D4)
- Формула: mean(cost[i+k] по i∈first-occurrence, i>4) / mean(cost[4:]), k≤12
  (exp85a_mdl.py:29). КОРПУС-калибр ≡1.0; own-ген ≡1.0 = результат, не тавтология.
  Запрещённое имя: «novelty/доля новизны».

## M7. engine-determinism pass (бывш. «REPLAY r») 
- Реплей greedy на тех же id: r=1.000 — пин детерминизма движка, НЕ доказательство
  лупов. Доказательство луп: rep8/onset8 (e81b: rep8=1.0 у 14/16, onset8≈17).

## M8. HEAT направление (пре-рег r2)
- Порог «подавление нагревом»: ≥12/16 доков с rep8(T)↓ при T∈{0.7,1.0} (p=0.038);
  фолбэк n=8: ≥7/8 (p=0.035). fdr по докам. T>0 междвижковое сравнение — только
  дистрибуционное (побитовой пин доказан для greedy).

## M9. random-init сид (R9-2)
- torch.manual_seed(S) перед from_config; S=11,12 (exp91); значение фиксируется до
  прогона и пишется в артефакт. Вердиктная клетка — half-random (обученные emb/norms).

## M10. E94 Q1 id-поток/повтор (SYCL vs CPU-реплей)
- Формулы (ТЗ E94 §2.1, дословно exp81b через exp94_sycl_common): prefix_len = длина общего
  префикса llama-gen vs REPLAY.gen_ids, norm /384; rep8 = mean(12 сегментов по 32, off=64)
  по конкатенации 64+384 id обеих сторон (тот же реплай на CPU-стриме); главная = медиана |Δrep8|
  + bootstrap 10000 (seed 8181, 2.5/97.5); Spearman avg-ranks + перм-null 2000 (seed 8181);
  repro_rate = доля совпавших id r1/r2 (тот же сервер/тело/флаги), гейт медианы ≥0.95, иначе
  метка `non-interpretable (determinism)`.
- Реализация: experiments/exp94_q1.py (запуск: exp94_sycl_stream.py --phase q1), артефакт
  results/e94/Q1.json, raw /tmp/e94raw/q1_*.json. RUN_RC: 0/10/11/13/1 (ТЗ §3).
- Поля сводки (ревью r1, 12.11): `n_docs_repro_identical` (r1≡r2, экс-`n_identical_docs`),
  `n_docs_prefix_full` (prefix_len==384), `spearman.p_definition` (two-sided на |rho|, MC-доля
  без +1, n_perm=2000, seed 8181, avg-ranks, пермутируется llama-массив против фикс. cpu),
  `spearman.p_mc_se` = sqrt(p(1-p)/n_perm), `interpretability`/`interpretability_reason`
  (CI медианы |Δrep8| накрывает 0 ⇒ "not interpretable as an effect"), `bifurcation_note`
  (w4 — единичная бифуркация 1/16: f16/SYCL vs fp32/CPU, не «iGPU теряет повторы»),
  `idempotence` (2 пересборки из raw, canonical sha256 сводки+per_doc).
- Конвенция: только числа+гейты, «механизм»-выводы запрещены (ТЗ §2.1, M4-аналог: CI накрывает 0 →
  эффект не формулируется).

## CHANGELOG
- 11.11: создан по r1/r2 ревью (D1,D4,C1,H1,H6,E3; B5,H4 — старые долги).
- 12.11: +M10 (E94 Q1: prefix_len/rep8-Δ bootstrap/Spearman-perm/repro_rate, exp94_q1.py).
- 12.11: M10 hygiene (ревью r1): n_docs_repro_identical/n_docs_prefix_full, p_definition/p_mc_se,
  interpretability, bifurcation_note, idempotence (exp94_q1.py --from-raw ×2).
- 12.11: E94 Q2b/Q2c диагностика флипа (experiments/exp94_q2b.py, exp94_q2c.py;
  results/e94/Q2b.json, Q2c.json): новые имена — счётчики A.emitted_ids/A.distinct,
  C.per_position, D.flips_with_probs/D.flips_no_probs/D.first_diff, E.emitted/E.top1,
  Q2c.verdict_shift (результаты, не гейты); итог — флип детерминирован (A 5/5),
  кэш-/наблюдатель-инвариантен (B, D), top_k=1 эмитит тот же (E), сдвиг списка отвергнут (Q2c).
- 12.11 (MINOR Q2, review 2_REVIEW_E94_Q2_exec_r1): D.flips_no_probs →
  D.stream_identity_no_vs_with_probs (+D.note: без запроса вероятностей флипы не
  вычислимываемы — метрика = тождественность потоков); переименование вступает в силу
  на следующем прогоне Q2b (артефакт Q2b.json с прежним ключом не перегонялся, GPU).
  Q2 (experiments/exp94_q2.py): g2_violations/g2_violations_n ({doc,k,prompt_n}, мягкий
  G2 основной фазы — нарушения фиксируются, прогон не останавливается),
  g2_offline_check (max prompt_n со 2-го запроса документа, медианы wall last32/first32),
  device_gate_source ("live"/"reused_raw", рядом с fd_first_request), --from-raw с
  idempotence (2 пересборки, canonical sha256).
