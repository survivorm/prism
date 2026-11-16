# Внешнее ревью: AT8_2 (T8.2) «Оконное восстановление + доза» — код + артефакт

CWD: /home/survivor/work/prism. Читай только, НЕ изменяй файлы (кроме ревью-файла).
Результат — ОДИН файл: `docs/dynamic/AT8_2_results_review.md`.

## Объекты (sha256 — проверь перед чтением)
- Код: `experiments/at8/AT8_2_recovery.py`
- TZ (v1.1): `docs/AT8_2_TZ.md`; TZ-ревью r1 (исходные MAJOR/MINOR): `docs/dynamic/AT8_2_TZ_review.md`
- Claim: `results/AT8/AT8_2_RECOVERY.json` (sha 779008c46a770b0c…)
- Store: `results/AT8/AT8_2_RECOVERY_store.json`; manifest: `results/AT8/AT8_2_RECOVERY_manifest.json`
- Run-log: `logs/AT8/AT8_2_run.log`
- Референсы: `experiments/e71_80/exp71_soliton.py` (часть B), `experiments/kaggle_e64gpu/e64_gpu.py` (dip)
- DB: `.trackio/prism.db` (run at8_2_recovery_r1) — read-only sqlite

## Суть
Обобщение E-71B: вставка `phrase×n` (n∈{2,4,8,16}) в середину 8 wiki-доков, dip-окна
W∈{8,16,32,64} ПОСЛЕ вставки vs то же окно контроля, ℓ_rec(n) (ось A по W=64),
dilution-check (ось B, to_global с dip_control_whole — MAJOR-1 фикс). Pre-reg: ELASTIC.
Итог прогона: VERDICT ELASTIC, match_pre_reg=True, wall 257s, 42 форварда (40 + 2 S8).

## Чек-лист (каждый пункт: OK / ПРОБЛЕМА + число)
1. **Hash-цепь:** claim file sha == manifest.claim_file_sha256; store file sha ==
   manifest.store_file_sha256; e64_gpu sha == 8c517e18…; pool_fp == 144c722df3046f2d;
   docs_sha256 == a8f13abd… (пересчитай по пину: data/calib_pins/wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json —
   fp, 8 первых текстов ≥64 ток., токенизация plain-путь).
2. **Гейты S1–S8:** все PASS? S8: dev_post/dev_ctrl = 0.0 (bit-identical, in-process)?
   S2b: control_sha256 — 8 sha, n_dose_reuse=32?
3. **Полнота:** 128 ячеек (8 doc × 4 n × 4 W); 120 measurable; not_measurable=8
   (wiki_5/wiki_6 W=64, D_max<0); 4 ячейки с D_max<8 (wiki_4 W=64, l_rec не отчитан)?
   Пересчитай D_max = T − T//2 − W для всех 8 длин [164,256,254,205,132,125,126,256].
4. **ℓ_rec-определение (TZ §4):** пересчитай ℓ_rec для 3–4 ячеек вручную из store
   (pert[d] = |dip_inj[d] − dip_ctrl[d]|; pert0<0.01 → 0; иначе min d: max(pert[d:])≤0.25·pert0;
   нет → D_max). Совпадает с claim?
5. **Оси A/B:** lrec W64 = {2: 61.5, 4: 0.0, 8: 49.0, 16: 62.0} — пересчитай медианы
   по ячейкам (только rec_status recovered/no_effect, W=64). ratio=1.0081, b=1.8543, r2=0.1528 —
   пересчитай. axis_B: frac_to_global=0.2328 по 116 пар (to_global: residual_global<residual_ctrl
   И |dip_global − dip_control_whole| > 0.25·pert0) — пересчитай на выборке 10 пар.
6. **Вердикт:** ELASTIC по порогам (ratio≤2, frac≤0.4, n_pos≥2)? n_pos=3?
   Уместен ли вердикт с учётом пре-рег-ноута (низковозмущённый режим: pert0-медианы
   0.010–0.017, 0.25·pert0 ≈ пол шума → ось A неинформативна; вердикт через ось B)?
   Есть ли альтернативное чтение (например, «нет эффекта» — no_effect=32/128)?
7. **S8 как проверка:** честно ли «in-process bit-identical» (один forward на h0/h1chk,
   те же срезы)? Не таутология ли это (проверка кода самим собой) — что она реально
   гарантирует и чего НЕ гарантирует (сверка с E-71B-артефактом невозможна — его нет на диске)?
8. **Код-аудит (AGENTS чек-лист):** (a) одна статистика контроля на обеих сторонах (S2b —
   реализовано?); (b) ветки/guard'ы (D_max<0, D_max<8, not_finite — покрыты?); (c)
   json-sanitize (numpy-типы, inf — в claim нет NaN/inf — проверь grep); (d) resume
   по ключам — корректен (control_sha при resume)?; (e) m.model.layers[23] — hook-уход
   после каждого measure?; (f) trackio (TRACKIO_DIR → .trackio, crash-handler atexit).
9. **Числа vs лог:** log (logs/AT8/AT8_2_run.log) vs claim — dip_global, nll, VERDICT
   совпадают? wall 257s?
10. **Согласованность с E-71B:** механика окна (h1[cut+L_n+d:…+W] vs h0[cut+d:…+W]) —
    точное обобщение? S8-значения (post −0.2189, ctrl −0.2067) — правдоподобны
    (E-71B-сдвиг ≤0.03–0.07)?

## Формат
Шапка (объекты + sha), разделы по пунктам, MAJOR/MINOR/NIT,
ИТОГ: `VERDICT: PASS | REVISE | FAIL` (PASS = можно коммитить и вносить в доки).
ПОСЛЕДНЯЯ СТРОКА ФАЙЛА: `REVIEW_AT8_2_DONE` (и последняя строка твоего ответа).

## Правила
Числа — только из артефактов/логов/DB (python3 + json/sqlite). Не запускай модель.
Store можно читать json.load (~? MB — проверь размер; если велик — читай по ячейкам).
Сомнение → STOP-запись в ревью.
