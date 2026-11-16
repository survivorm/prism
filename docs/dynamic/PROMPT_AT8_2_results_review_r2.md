# Внешнее ревью ROUND 2: AT8_2 (T8.2) — фиксы по results-ревью r1 + артефакт v2

CWD: /home/survivor/work/prism. Читай только, НЕ изменяй файлы (кроме ревью-файла).
Результат — ОДИН файл: `docs/dynamic/AT8_2_results_review_r2.md`.

## Контекст
Round 1 (`docs/dynamic/AT8_2_results_review.md`, sha 6acfdd64…, VERDICT: REVISE,
1 MAJOR + 3 MINOR + 5 NIT) нашёл: MAJOR-1 (медианы оси A — по цензурированной
популяции, против TZ §5), MINOR-1 (не реализованный NOT_RESOLVED-гейт «большинство
not_recovered»), MINOR-2 (trackio: нет stage-точек/md5/seed), MINOR-3 (resume не
exercised), NIT-1..5. Фикс выполнен БЕЗ перепрогона (store на месте, прецедент
AT8_1_v2): v2-реанализ (disk-only) + r2-код для будущих прогонов.

## Объекты (sha256 — проверь перед чтением)
- R1-ревью (исходные замечания): `docs/dynamic/AT8_2_results_review.md`
- **TZ v1.2**: `docs/AT8_2_TZ.md` (уточнения популяций — §5; история r1→v2 — §9)
- **Код r2**: `experiments/at8/AT8_2_recovery.py` (будущие прогоны),
  `experiments/at8/at8_2_analysis.py` (ОБЩИЙ анализный фрагмент),
  `experiments/at8/AT8_2_v2_reanalysis.py` (disk-only reanalysis)
- **Exec-тесты**: `experiments/at8/test_at8_2_resume.py`, `test_at8_2_cells.py`,
  `test_at8_2_verdict.py` (прогони: /opt/miniconda3/bin/python3.12 — должны PASS)
- **V2-артефакт**: `results/AT8/AT8_2_RECOVERY_v2.json` (sha d073a1f4… — пересчитай),
  `results/AT8/AT8_2_v2_manifest.json`, `logs/AT8/AT8_2_v2_run.log`
- R1-снапшот (не тронут): `results/AT8/AT8_2_RECOVERY.json` (sha 779008c4…),
  `results/AT8/AT8_2_RECOVERY_store.json` (sha 69f32421…),
  `results/AT8/AT8_2_RECOVERY_manifest.json`, `logs/AT8/AT8_2_run.log`
- Референсы: `experiments/e71_80/exp71_soliton.py` (часть B),
  `experiments/kaggle_e64gpu/e64_gpu.py` (sha 8c517e18…)

## Чек-лист (каждый пункт: OK / ПРОБЛЕМА + число)

### A. Закрытие MAJOR-1
1. TZ v1.2 §5: популяция оси A зафиксирована (полная = все измеримые (doc,W) с ℓ_rec;
   censored_excluded — вторичный, явно помечен)? Конвенция «медиана доков» для
   combined зафиксирована (сначала по W внутри дока, потом по докам)?
2. V2-claim: `axis_A.W64.lrec` = полная популяция {2: 60, 4: 39, 8: 49, 16: 62}?
   Пересчитай сам из store (все измеримые W64, медиана по докам). `lrec_censored_excluded`
   = {61.5, 0, 49, 62} (r1-числа, вторичные)?
3. ratio/b/r2 (W64, полная): 1.0333 / 0.7391 / 0.1842 — пересчитай. Совпадает с
   независимым пересчётом r1-ревьюера (1.0333)?
4. combined_secondary: {35, 51, 60, 62.75}, ratio 1.7929 — пересчитай по конвенции
   TZ v1.2 (медиана по W внутри дока → по докам). ВНИМАНИЕ: r1-ревьюер пересчитал
   combined как прямой median по всем (doc,W) ({44,48,55,62}, ratio 1.4091) —
   разность = конвенция (теперь зафиксирована в TZ), не ошибка? combined — справочный,
   в вердикте не участвует?
5. Вердикт: ELASTIC (match pre-reg) по ОСНОВНОЙ (полной) оси A — пороговые условия
   (ratio≤2, frac≤0.4, n_pos≥2) выполнены по полным числам? n_pos=4?

### B. Закрытие MINOR-1
6. NOT_RESOLVED-гейты реализованы (at8_2_analysis.compute_verdict):
   «<6 пар (doc,n) с ℓ_rec>0 на W=64» + «большинство not_recovered по полной
   измеримой популяции (все W, D_max≥8)»? TZ v1.2 §5 популяции зафиксированы?
   W64-чтение (75% not_recovered) — diagnostic `axis_W64_notrecovered_share`,
   не гейт — в claim?
7. Реальные числа: 55/116 = 0.4741 < 0.5 (гейт не сработал) — пересчитай;
   n_meas_doc_n (W64) = 19 ≥ 6 — пересчитай из store.
8. Границы гейтов: exec-тесты test_at8_2_verdict.py — прогони; покрывают оба гейта
   + границу 0.5 + границы «<6»?

### C. Закрытие MINOR-2/3 + NIT
9. Код r2 (AT8_2_recovery.py): trackio — seed в config («n/a — детерминированный
   forward»), stage-точки (tstage: model_loaded/docs_loaded/forwards_done/analysis/
   artifact), log_md5 в finish? setdefault TRACKIO_DIR убран (AGENTS п.6 — run_one.sh
   экспортирует)?
10. resume_plan (at8_2_analysis) — чистая функция, вызывается из main(); exec-тест
    test_at8_2_resume.py — прогони; покрывает partial-store + fresh?
11. NIT-2 (not_finite-ветка): exec-тест test_at8_2_cells.py — прогони; ветка
    покрыта (NaN + None)? NIT-3 (try/finally hook в measure)? NIT-5 (n_pos≥2
    задокументирован в TZ v1.2)?
12. r1-снапшот НЕ тронут: sha r1-claim 779008c4…, store 69f32421… совпадают с
    r1-manifest; v2-manifest ссылается на r1 (r1_claim_file_sha256)?

### D. V2-реанализ как артефакт (прецедент AT8_1_v2)
13. Hash-цепь v2: v2-claim sha == manifest.claim_file_sha256; store sha ==
    manifest.store_file_sha256; r1-claim sha == manifest.r1_claim_file_sha256?
14. Ячейки v2 == r1-claim побитово (128)? (v2-скрипт fail-loud — проверь, что
    самопроверка реально в коде и сработала: строка в v2-логе).
15. V2-лог (logs/AT8/AT8_2_v2_run.log): полная трасса (sha-верификация → пересчёт →
    ОСИ → VERDICT → written → маркер AT8_2_V2_DONE)?
16. NaN/inf в v2-claim? (grep)
17. Числа v2-лога vs v2-claim: ratio/frac/verdict/wall совпадают?

### E. Общий код-аудит r2 (AGENTS чек-лист)
18. Единый фрагмент: AT8_2_recovery.py (main) и AT8_2_v2_reanalysis.py используют
    ОДИН и тот же at8_2_analysis (build_cells_from_store/axis_A/axis_B/compute_verdict)
    — нет расхождений копий? Мёртвый код/случайные мелочи в r2?
19. Гейты S1–S8 в r2-коде — не сломаны при р2-правках (S8 in-process, S2b, try/finally)?
20. Резюме: все 1 MAJOR + 3 MINOR r1-ревью закрыты? Что осталось (NIT — допустимо)?

## Формат
Шапка (объекты + sha), разделы A–E, новые MAJOR/MINOR/NIT (если есть),
ИТОГ: `VERDICT: PASS | REVISE | FAIL` (PASS = фиксы приняты, можно вносить в доки
и коммитить). ПОСЛЕДНЯЯ СТРОКА ФАЙЛА: `REVIEW_AT8_2_R2_DONE` (и последняя строка
твоего ответа).

## Правила
Числа — только из артефактов/логов (python3 + json; store 229K можно json.load).
Exec-тесты прогони (read-only для репо — только чтение + запуск тестов; тесты не
пишут в репо, кроме stdout). Не запускай модель. Сомнение → STOP-запись в ревью.
