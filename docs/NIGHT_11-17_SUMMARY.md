# Ночная автономка 17.11 — итог (AT2c / T2.1c)

Коммиты: 6923b00 (AT2c: код+ТЗ+ревью+результаты+артефакты-провенанс),
ba311b1 (docs/AT2_results.md §6 — итоги T2-цепочки).

## Что сделано

AT2c (T2.1c — стационарный J-базис + centered-цель + weighted cos) — полный
протокол:
- ТЗ: 7 раундов внешнего ревью (r1 REVISE → r2 REVISE → r3 PASS → r4 PASS →
  r5 PASS → r6 PASS → r7 PASS; docs/AT2c_review*.md).
- Код-ревью: 3 узких воркера + recheck (docs/AT2c_review_code_*.md,
  AT2c_review_recheck.md): 1 MAJOR (wcos-таблица в results) + 2 MINOR (N8 4→6
  армов; S6c unweighted→weighted) — закрыты; 5 NIT — зафиксированы.
- S6c-инцидент (r3-stop): weighted-конфаунд max_d-pairwise 1.03e-2 >
  DELTA_MIN (порог откалиброван по unweighted 8.7e-3) → ТЗ r7: статистика
  конфаунда = verdict-количество (per-arm weighted-wmean, 7.6e-3 < 0.01,
  запас 32%); max_d-pairwise — report-only.
- Смоки: S9 детерминизм (md5 бит-в-бит, ×2), pre-run §11.0 (control PASS на 4
  слоях, S8=0.0), arm-swap (swapped CI lower=−0.4571 < 0).
- Main: 4 революции (r1 N8-фикс, r2 N8-6арм, r3 S6c-stop, r4 официальный) —
  наука r1=r2=r4 бит-в-бит.

## Результат (at2c_t21c_r4)

**VERDICT: NOT_RESOLVED (control_ok=True)** — все 11 гейтов PASS; S8 = 0.0
(бит-в-бит воспроизведение просмотренной диагностики C1c).

- Control k=0 PASS: weighted CI top−weak 0.40…0.54 (4 слоя).
- **(a) scratchpad: НЕТ** — weak-tail хуже max(pca,haar) на всех k=2,4,8.
- **(b) deadweight: НЕТ** — top до 0.40 (далеко за DELTA_MIN=0.01).
- **(c) redundant: НЕТ** — top значимо лучше всех остальных армов на всех
  (L,k)∈{1,2,4,8} (2 граничных CI на k=8).
- **(d) metric: НЕТ** — max попарное расхождение 0.41.
- Монотонность: top > weak_mid > weak-tail (по энергии Gagg); pca (per-doc
  SVD) и haar (глобальный) ≈ 0 при k≥1.

**Интерпретация:** disk-only state-readout не подтверждает scratchpad-гипотезу
T2 ни в r1-метрике (floor-dominiрована), ни в T2.1c (stationary, centered,
weighted): будущее НЕ читается из weak-стороны J-спектра — наоборот,
top-сторона доминирует на всех k.

## Нужно решение утром

1. **T2: закрыть как «не подтверждено»** (рекомендация: ДА — обе конструкции
   сходятся; пре-рег §8 допускает ветвь (b)/(c)-стиля), либо
2. **T2.2** — альтернативная конструкция (не readout-клетка), либо
3. уход к другим вопросам моста Якобиана.

## Провенанс

- Артефакт: results/AT2c/AT2c.json (23.4MB, sha e04ae156f337ea647c3bb6704ab974c9af6a26508a36aa803d738fe3ffee5870;
  в git НЕ входит, >10MB; бэкапы /tmp/recovered/AT2c/AT2c_{r1,r2,r4}.json).
- Лог md5 76bab4b7d068c1797e38de5b321334d1 = trackio log_md5 (run
  at2c_t21c_r4, project prism, group AT0).
- Подробности: docs/AT2c_results.md (полный), docs/AT2_results.md §6 (цепочка).

Система: процессов AT2c нет, RAM 25G, watchdog чист.

Отдельно: убит зомби-вотчер AT-цепочки «Пробуждение» (от 08.11, PID
1499557/1499560) — бесконечный цикл `while ! grep ALL DONE /tmp/ATlocal.log`
на мёртвом прогоне (лог обрывается на «C1b code_2 36518s», маркера нет,
процесса нет). Цепочка ATlocal2/ATbatt/ATneedle за 9 дней не стартовала.
Решение утром: перезапускать AT-цепочку (exp64_at_local.py, ~10 ч) или нет.
