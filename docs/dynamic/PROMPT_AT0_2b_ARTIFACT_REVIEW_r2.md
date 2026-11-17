Ты — НЕЗАВИСИМЫЙ внешний ревьюер артефакта AT0_2b (round 2). Репо: /home/survivor/work/prism.

ЗАДАЧА: проверить артефакт AT0_2b (RAG-дистанционный эффект) ПОСЛЕ внедрения замечаний ревью r1.
r1: P0×2 + P1×4 + P2×5 — обработаны (ответка: docs/AT0_2b_review_r3_answers.md).

ВХОДНЫЕ ФАЙЛЫ:
- docs/TZ_AT0_2b.md — ТЗ v4 (с правками по r1)
- docs/AT0_2b_prereg.md — пре-рег v4.1
- docs/AT0_2b_review.md — ревью r1 (проверь закрытие каждого пункта)
- docs/AT0_2b_review_r3_answers.md — ответка на r1
- experiments/AT0_2b_ra_split.py — код
- results/AT0/AT0_2b_RA.json — артефакт
- results/AT0/AT0_2b_manifest.json — манифест (commit-хеши)
- logs/AT0/AT0_2b_run.log — лог (marker AT0_2B_DONE)

ЧТО ПРОВЕРЯТЬ (приоритетно):
1. P0-1 (r1): n_eff/overlap — реальная overlap-матрица в артефакте? Числа?
2. P0-2 (r1): bootstrap CI — пересчитанные? Совпадают с артефактом?
3. Все P1/P2 из r1 — закрыты в коде/артефакте?
4. **ВНИМАНИЕ**: commit в manifest (7f4d1c8) НЕ соответствует текущему HEAD — коммит 14788e0
   затронул AT0_2b-файлы, но артефакт не пересобран. Проверь: `git show 14788e0 -- experiments/AT0_2b_ra_split.py`
   и ответку r3 — влияют ли изменения на ЧИСЛЕННЫЕ результаты AT0_2b?
5. Независимая пересчитка ключевых метрик (delta_d3, delta_l23) по данным/логам, если возможно.
6. Статистическая корректность: floor-коррекция, multiple-testing, direction gates.

ФОРМАТ:
- sha256sum всех входных файлов — в начало (фиксация актуальности).
- ВЕРДИКТ: PASS / REVISE / FAIL (строка "VERDICT: <PASS|REVISE|FAIL>")
- Найдено: P0×<N>, P1×<N>, P2×<N>
- Таблица пунктов (ID, severity, суть, файл:строка, проверка, статус).
- Если REVISE/FAIL — точные формулировки правок.
- Финальный маркер: REVIEW_DONE

РЕЗУЛЬТАТ: файл docs/dynamic/AT0_2b_ARTIFACT_REVIEW_R2.md (только этот файл, ничего не менять).
Числа — только из файлов, не домысливать.
