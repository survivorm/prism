Ты — внешний ревьюер прогона AT2b (T2.1b) в /home/survivor/work/prism.
Твоя задача — ОДНА: проверить ПРОГОН (артефакт + results-док + код) и написать
ревью-ФАЙЛ docs/AT2b_results_review.md. Не правь ничего, кроме нового ревью-файла.

## Канон-файлы (проверяй sha перед чтением; если sha не совпадает — STOP + запись в ревью)
- experiments/at2/AT2b.py            sha256 0adac2cb83b61ce9f3f43d7bf220db0da19dee624324a0b19b9435a82d6a5f07 (после post-run фиксов; на момент прогона 22cb816907bf87d998c2...)
- docs/AT2b_prereg.md                sha256 729d22e2960892d441fa8f96fecc9f6e06ab8173f6fe0ae7c18a275e4701f26d
- docs/AT2b_review.md (TZ-ревью)     sha256 acec436a2fa4d140ef9c8d3b411ce495cc670837b155eed5e452515ebfab6999
- docs/AT2b_review_answers.md        sha256 c0d227dc15108e723f788eb301aa73b6fcd2f8b8479d689cc69cc2f5899f7385
- docs/AT2b_results.md               sha256 a96359d8bb15f6d671cb40bf640fac6616c6f7e5e60701e04dcea350068931a8 (объект ревью; CJK-проверь: grep -nP "[\x{3000}-\x{9fff}\x{3040}-\x{30ff}]")
- results/AT2b/AT2b.json             sha256 81fd7342c479e7435056cf9784265fdf74ee532af123e0e165c2ac1a0ee26095 (38 MB; читай по json, не целиком в память строкой)
- results/AT2b/AT2b_run.log
- /tmp/at2b.log                      (канон полный лог; AT2b_run.log не содержит последнюю DONE-строку — зафиксировано в results-доке)
- results/AT2b_smoke/AT2b_smoke_det.json  md5 acc76839a4096b72a91fd66f3146a6ac (S9: ×2 bit-identical — проверь md5)

## Что проверить (каждый пункт — с числом из артефакта/лога в ревью)
1. **Провенанс гейтов S1-S7**: gates в AT2b.json все True; строки лога (S1 7/7, S2 1.885e-08,
   S3 0.381/0.490/0.014/0.017, S4 90/4096 + 8/41/36/5, S5 True, S6/S6b True, S7 860000 set-eq)
   совпадают с полями артефакта (s2_kl_report, lk4.hit_total/hit_per_dom, S7_expected,
   s6_pzero, per_pos_cos len=860000).
2. **Вердикт по пре-рег-дереву**: control CIs k=0 из bootstrap.control_k0_top_minus_weak_CI:
   L6 [-0.0131,-0.0004], L12 [-0.0066,+0.0033], L18 [+0.0133,+0.0315], L22 [+0.0406,+0.0740]
   (пересчитай из per_pos? НЕТ — из артефактных CI; сверь с results-доком). control_ok=False,
   verdict=SUBSPACE_CONTROL_FAIL, в verdict_detail НЕТ ключей a_scratchpad/b_deadweight/...
   (дерево коротко замкнуто на control — как в пре-реге §4).
3. **Числа results-дока vs артефакт** (пересчитай скриптом):
   - k=0 cos_median top/weak per слой (4×2 числа).
   - «полоса k≥1 −0.007…−0.040»: min/max cos_median по J-армам {top,weak,pca,haar} на
     ВСЕХ (L,k), k>=1.
   - «R²_c 0.715…0.775»: min/max R2_mean по J-армам на всех (L,k).
   - floor uniform: диапазон медиан и max ширины CI (сверь с delta_min_calibration:
     spread 0.0716, floor_CI_width_max 0.1004, DELTA_MIN 0.01).
   - d_eff rematching: r_final=7, rematched=True.
   - lk4: n_hits 90/90/78/41/14; coverage 0.478/0.478/0.387/0.731/0.733 c CI;
     onehot 0.328/0.327/0.350/0.360/0.353.
   - L23: k0 pca 0.066 / haar 0.286 / uniform −0.026.
   - wall 457.7 (артефакт) / 460.4 (лог).
4. **Переоценка/недооценка**: results-док НЕ должен интерпретировать k≥1 как ответ на T2
   (контроль не пройден). Проверь, что формулировки это уважают (цитаты из дока).
5. **Post-run фиксы** (results-док §post-run): LOG_PATH (results/AT2b_run.log →
   results/AT2b/AT2b_run.log; файл перенесён — проверь, что results/AT2b/AT2b_run.log
   существует и его строки ⊆ /tmp/at2b.log) и DONE-строка. Оцени: фиксы не затрагивают
   чисел? (в AT2b.py: grep LOG_PATH, tail main()).
6. **Скрипт-аудит** (поверхностный, на bugs, влияющие на числа):
   - S6/S6b, bootstrap (ci_of* — ресэмплинг доков, SEED), win_ci (односторонний),
     per_pos генерация (t+k индексация), N8 (tsel - k), one-hot y_idx.
   - Порядок определений (фаз 5 delta_calib после PAIRS/floor_out).
   - json: numpy-типы/NaN (в AT2b.json grep "NaN" = 0 — проверь).
7. **trackio**: .trackio/prism.db, run_name='at2b_t21b_r1': 8 точек, стадии в порядке,
   fin status=ok, verdict совпадает.
8. **Согласованность с пре-регом**: S7-формула (4×32×625×10+32×625×3=860000), DELTA_MIN=0.01,
   KS=(0,1,2,4,8), r_final=7 (ожидание 7 — совпало), SEED=0, B=1000.

## Формат ревью (docs/AT2b_results_review.md)
- Шапка: что ревьюится (run at2b_t21b_r1), sha проверенных файлов.
- Разделы по пунктам 1-8: каждый — «OK / ПРОБЛЕМА» + числа (только из артефактов/логов).
- MAJOR (ломает вывод) / MINOR (нужно поправить) / NIT.
- ИТОГ: `VERDICT: PASS | REVISE | FAIL` (PASS = можно коммитить; REVISE = мелкие правки
  в results-доке; FAIL = выводы не верны).
- ПОСЛЕДНЯЯ СТРОКА ФАЙЛА ОБЯЗАТЕЛЬНО: `REVIEW_AT2B_DONE`

## Правила
- Числа — ТОЛЬКО из артефактов/логов (python3 + json/sqlite). Никаких «по памяти».
- Сомнение → STOP, запиши «STOP: ...» в ревью и остановись (не выдумывай).
- Память: не грузи AT2b.json целиком строкой; json.load по окном (38 MB — ок для python).
- Не запускай эксперименты, не правь AT2b.py/пре-рег/артефакт — только docs/AT2b_results_review.md.
- Финал: `REVIEW_AT2B_DONE` в конце вывода И в конце файла.
