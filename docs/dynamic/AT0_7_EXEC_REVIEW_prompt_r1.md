# Внешнее ревью AT0_7 (T0.7) — код + артефакт (после исполнителя, перед коммитом)

**Роль.** Независимый критик (не автор, не исполнитель). Протокол: ревью обязан проходить и КОД/артефакт, а не только ТЗ. Числа — пересчётом (read-only, OMP_NUM_THREADS=4), не верой.

**Объекты:**
1. Код: `experiments/AT0_7_loop.py` (427), `experiments/at0_7_core.py` (263), `experiments/at0_7_tests.py` (329), `experiments/at06_v22_pinned.py` (байт-пин AT0_6 v2.2 из git HEAD).
2. ТЗ: `docs/TZ_add_0-3.md` → «### T0.7 —» (v3; PASS по r3 — docs/dynamic/AT0_7_TZ_REVIEW_r3.md; ретроспективные r0-числа там верифицированы).
3. Артефакт: `results/AT0/AT0_7_LOOP.json` (claim), `AT0_7_manifest.json`, `AT0_7_run.log`, `results/AT0/store/7173b42fa0468fe8.json`.
4. Контекст: AT0_3 (`results/AT0/AT0_3_AREA.json`, sha 8fe61affb13b443f), AT0_6 v2.2 store (`results/AT0/store/767d623f7233b1f4.json`).

**Итог исполнителя (проверить, не верить):** VERDICT CONFIRMED; H1 f1=30/30 (min rep_frac 0.8880 code_14, r0 воспроизведена в 1e-12); H2: LU 6/11 (5/8 unique), clean37 0/37 (0/36 unique), rem 2/2; H3: 38/80 стабильных, доля loopy 30/38=0.7895; H4: ρ_s(Q,maxrun)=0.4947 (≥0.4), ρ_s(Q,rep_frac)=−0.0316; H5: R loopy30 ∈ [2.88e-31, 3.66e-30], clean37 max 7.38e-28, Rtok max 0.223, identity R<1e-20; G0/G1a/G1b/G1d/G2/G7a/G7b/POS1/POS2 — все PASS; Q+p_perm: 30/30 committed, p_perm 1/200 (floor). Wall ~3 мин.

**Что проверять (по пунктам):**

1. **Код ↔ ТЗ (MAJOR-уровень — расхождение до запуска = MAJOR):**
   - Пулы: 30 loopy-detected (p_T0.3 из AT0_3), 11 loopy-undetected, 37 clean non-corpus, 16 corpus not_testable, 2 remainder {own_w0, own_w4}; H2-пул = 37 clean + 2 remainder + (11 LU отдельным подпулом); H3 — 80 токенизируемых. Сверить с AT0_3 пересчётом.
   - rep_frac(p) = PPF(p) = #{t: tok[t]==tok[t+p]}/(T−p); C(p) = Σ_v f(v)²; RATIO = PPF/C; maxrun = ЧИСЛО последовательных t в длиннейшем беге (чистый p-repeat → T−p, НЕ m+p−1). Guard p ≥ T → None.
   - H1: f1 = #{rep_frac(p_T0.3) ≥ 0.8}/30, PASS при f1 ≥ 28 (pre-reg). R0-якорь: min 0.8880 (code_14), med 0.9973 — воспроизведение 1e-12.
   - H5: (токен,фаза)-lookup x^sur_{t,d} = mean_{t': tok[t']=tok[t] ∧ t'≡t (mod p)} x_{t',d}; R(p) = Σ_d|S_x^sur−S_x|²/Σ_d|S_x|² ≡ 0 (тождество); Rtok (без фазы) — вырожден при V≫T. Границы: fp64 < 1e-20.
   - H4: Spearman(Q(p_T0.3), maxrun(p_T0.3)) по 30 loopy; порог 0.4 (descriptive).
   - Q(p) + p_perm: формулы = AT0_6 v2.2 (пин at06_v22_pinned.py, sha c24cb668…): S_d(p)=Σ_t x_{t,d}e^{−2πit/p}, Q=Σ_d|S_d|²/(T'·σ²_d), перестановки B=199, сид 2026+doc_idx·1009+s (doc_idx — sorted(96)), p=(#≥+1)/200.
   - Гейты: G1a (rep_frac(13)=1.0 точно, C=11287/146689=0.07694510, RATIO=12.996279≠13, maxrun=370, p=13, T=383, 13 различных v_i=i); G1b (cos a=1 σ=1 p0=13 D=32: Q/D≥50, off/on≤0.2); G1d (iid V=4096 сид 777 T=383: max_p RATIO ≤ 1.1, замер 1.0118 @p37); G2 (96 доков, n_tok 30+39+11=80, corpus 0/16 assert); G7a (20 iid-синтетиков, словарь 4096, сид 777+k: count-гейт ≤1/20 с max_p RATIO ≥ 3); G7b (Q-null k ≤ 3); POS1 (чистый p-repeat p=13, 13 различных v_i, сид 7771: rep_frac=1.0); POS2 (чистая h-осцилляция + iid токены: max_p PPF < 0.2 ∧ Q/D ≥ 50).
2. **Артефакт ↔ цифры исполнителя + р0-якоря (ТЗ r3):** все числа выше пересчитать из store. sha-цепочка: store sha256 = claim.store_sha256; claim_sha256 в manifest; manifest self-hash; field.py sha; at03 sha 8fe61affb13b443f; data shas gen/own/cor.
3. **Честность:** (a) H1 — ретроспективный (r0 уже считала rep_frac на тех же данных): артефакт это маркирует (не «нашли впервые»)? (b) H5 — не тест, а граница (R≡0): артефакт/вердикт не подаёт H5 как тест? (c) K4: T0.7 НЕ влечёт T0.6 (h-линия ≠ tok-repeat; носители разные) — есть ли в вердикте? (d) «80 уникальных текстов, не 96» (own≡wiki дубли по токенам) — per-unique-text отчёт в H2?
4. **Детерминизм/чистота:** все сиды из ТЗ (2026+doc_idx·1009+s; 777+k; 7770+k; 7771/7772); нет global np.random; jclean (NaN/Inf); store sort_keys.
5. **Резидуальные риски исполнителя (оцени, принимай/отклоняй):** (a) H5 R-диапазон [2.9e-31, 3.7e-30] vs r0 [2.5e-32, 3.5e-30] — fp64-резиды/порядок суммирования; (b) ρ_s=0.495 близок к 0.4 (n=30, descriptive); (c) p_perm 30/30 на полу 1/200; (d) пин v2.2 vs WIP v3 в репо (чужая сессия).
6. **НОВЫЕ ДЕФЕКТЫ в коде** (самое важное): мёртвые параметры, guard'ы, каскад (None/None), json-dump sanitize, ветки после if/elif — глазами НЕ считается, где надо — exec точного фрагмента.
7. **Файлы параллельной сессии** (AT0_7_loop_mine.py, AT0_8_threshold.py): не должны влиять на канонический AT0_7 (импорты? name-коллизии в run_one? writes в те же paths?). Проверь, что канонический артефакт НЕ перепутан с чужим.

**Формат выхода (ОДИН файл):** `/home/survivor/work/prism/docs/dynamic/AT0_7_EXEC_REVIEW_r1.md`
- Вердикт: PASS / PASS с MINOR / MAJOR-N
- Таблица: № | MAJOR/MINOR/космет | файл:строка/артефакт-путь | проблема | доказательство (пересчёт/цитата) | фикс
- «Артефакт-сверка»: таблица заявл. число → пересчёт → совпадение
- «Честность» (п. 3): verdict
- «Резидуальные риски»: принимаю/отклоняю с обоснованием
- В конце: `AT0_7_EXEC_R1_DONE`

**Ограничения.** Read-only (кроме одного review-файла). Пересчёты — /tmp/rev_at07_*.py. НЕ запускать прогоны экспериментов (AT0_6 v3 сейчас идёт ~5 мин — не трогай, не записывай в results/AT0/). Данные read-only через AT0_1_accounting / AT0_3.

Начинай.
