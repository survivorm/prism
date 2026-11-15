# Prompt внешнего ревью: AT0_2 r6 (T0.2 shape) — артефакт полного прогона

Роль: внешнее ревью post-hoc артефакта AT0_2 round 2 (полный прогон r6, ревью-луп
r1–r6 ДО прогона пройден). Ты НЕ автор. Задача: независимая верификация (1) артефакта,
(2) provenance-цепочки, (3) чисел, (4) выполнения пунктов р6-ревью (E1–E8, M1–M5).
Формат вывода: `docs/dynamic/AT0_2_r6_REVIEW.md` — таблица проверок + находки MAJOR/MINOR
+ итог «Вердикт: PASS/FAIL».

## Контекст (коротко)

- AT0_2 = T0.2 «Форма профиля при фиксированной сумме» (ТЗ docs/TZ_add_0-3.md §1.5, v1.1).
  Метрика D (participation ratio) объектов Δ-профилей real vs null-распределения (9 null-типов:
  iid, blk2/4/8/16/32, ar1/ar2/ar4); правило решения: refuted, если Δmean > DELTA_MIN=0.05
  (и статистика по cell p-values, frac(p<α) — reportable).
- Provenance: 3 объекта × 96 доков (64 gen + 16 P1_own + 16 P2_corpus) × L{1,5,11,17,23};
  P2_corpus not_testable (G16: 0/16 токенов — provenance_limited); L1-TPU degraded absent:
  ровно own_w0, own_w4 (как AT0_1); degen-доки 43/96 (флаги T0.1).
- Пулы (р6 E6/E8): primary = verified ∩ non-degen (37 доков, 185 клеток = 37×5 слоёв);
  sensitivity: verified_only (80), all_96 (96, 480 клеток).
- Прогон r6 завершён 15.11: gates ALL PASS, артефакты записаны.

## Файлы для ревью (актуальные)

Код:
- `experiments/AT0_2_shape.py` — run-скрипт (sha256 e58b7d1911cb05c7701904c84e934f2b5fcf3d250cf5a8005626b180375bc868 — проверь!).
- `prism/field.py` — vpath/ra_split/f_dc (sha-пин bcc607c561f95e3e…; в claim gates).

Артефакты:
- `results/AT0/AT0_2_SHAPE.json` — claim (sha256 b3cad772b3b3190f5f0c552c189e528d59676b2f8bd050e163580349eb2d83d7).
- `results/AT0/store/f7cc011ef856a6dc.json` — store, 222.8MB (sha256 f7cc011ef856a6dc30dcb1727dbd2948f61ca7a0d96cec119549eedb9340db77); структура: {cells: {obj: {doc|L: …}}, experiment, meta, prereg, tz}.
- `results/AT0/AT0_2_manifest.json` — manifest (claim_sha256/store_sha256/script_sha256/manifest_sha256; self-hash — по blob ДО вставки поля — проверь convention).
- `results/AT0/AT0_2_run.log` — run-лог (tracked); `results/AT0/AT0_2_r6_stdout.log` — stdout прогона (последние строки: per-null Δmean/CI/D_real/frac + «AT0_2 gates: ALL PASS»).

Доки:
- `docs/AT0_2_prereg.md` — пре-рег (round 1; единственный источник правды = хедер скрипта).
- `docs/AT0_2_REVIEW_r6.md` — р6-ревью (E1–E8, M1–M5) — проверяемые требования.
- История: docs/AT0_2_REVIEW*.md (r1–r5) — только для контекста.

## Обязательные проверки (минимум)

1. **Provenance-цепочка:**
   - sha256 script/claim/store совпадают с заявленными (manifest + claim.meta).
   - claim.store_sha256 == sha256 store-файла; manifest.claim_sha256 == sha256 claim-файла.
   - data_sha256 (gen_layers_qwen.npz f142f297…, P1_own, P2_corpus) — в claim gates и
     store.meta; файлы существуют, sha совпадает (пересчитай!).
   - field.py sha-пин.
   - store.prereg == claim.prereg (bit-eq? json-равенство) — хедер-скрипт не дрейфует.

2. **Gates (claim.gates):** все PASS; декомпозиция:
   - G0: data sha + decode-probe (3 gen + 1 own); corpus 0/16 → not_testable (G16).
   - degen 43/96 == флаги T0.1 (сверь с results/AT0/store/23281c64cc418ff1.json, per_doc degen-флаги).
   - absent ровно own_w0|L1, own_w4|L1 (L1-TPU degraded) — 6 absent-клеток (2 дока × … проверь число).
   - store: 3 объекта × 480 клеток (96×5), per-cell ключи полные (выбери 5 клеток и проверь set ключей).

3. **Пулы и арифметика (claim):**
   - primary: n_docs=37, n_cells=185 в каждом (obj,null) слоте; verified_only=80; all_96=96.
   - ЛИНЕЙНОСТЬ: для ВСЕХ 3 пулов × 3 объекта × 9 null: delta_mean == D_real_mean − D_null_mean
     (tol 1e-12). Это дешёвая глобальная проверка — сделай по всему claim.
   - CI: D_real_ci/D_null_ci — симметричны вокруг mean? (Wilson/normal — уточни по коду,
     какая методика; не требуй симметрии, если методика асимметричная — проверь код).
   - Monotonicity null-иерархии (reportable): D_null(iid) > D_null(ar1) > blk2 > … — проверь
     для 3 объектов по all_96 (нарушение = MINOR с объяснением).

4. **Независимый пересчёт (2–3 дока, из npz + store):**
   - Выбери 2 gen-дока + 1 own-док (не degen). Для каждого:
     (a) D_real(h_raw) пересчитай по данным (Δh = h[1:]−h[:-1], h = U·hn; D = participation ratio
         per-dim дисперсий — по формуле из скрипта; N_i = min(T,384)−1); сверь со store cell (tol rel 1e-9).
     (b) D_null(iid) — среднее по null-деталям store для этой клетки (store хранит per-null
         распределение? — посмотри структуру клетки; если хранит только агрегаты — пересчитай
         минимум 1 null от нуля: seed'ы из meta, B из store).
     (c) delta и frac(p<α) для 1 клетки — сверь с log-строкой (log печатает all_96-пул;
         для конкретного дока сравни с per-cell значениями в store).
   - Проверь log↔claim: последние 27 строк log (3 объекта × 9 null: Δmean/CI/D_real/frac)
     == claim.aggregates (all_96 пул): h_raw iid Δmean 1.2569, v_tangent blk16 Δmean 0.0922,
     D_real 0.1385 и т.д. — bit-eq (round'инг до 4 знаков в log допустим, сверь по 4 знакам).

5. **Р6-ревью (E1–E8, M1–M5):** по каждому пункту docs/AT0_2_REVIEW_r6.md — закрыт ли в
   артефакте/коде (покажи path в claim/коде). Особо: E6 (пулы), E8 (object_provenance:
   h_raw/du_unit/v_tangent per source; gen hn — provenance_limited), G16 (corpus).

6. **Код (AT0_2_shape.py):**
   - assert-самоаудит реально останавливает запись (запись — ПОСЛЕ всех assert'ов; см.
     «SMOKE: артефакты НЕ пишем» ~строка 1012 и блок store/claim ~1023).
   - Детерминизм: seed'ы фиксированы (meta.seed); перепуск дал бы бит-в-бит (не запускай —
     только проверь отсутствие rng без seed / time / dict-итерации в численных путях).
   - json.dump: sanitize (NaN/Inf → None? jclean?) — проверь, что в claim нет NaN
     (python-скан по claim).
   - Границы: не трогает field.py / AT0_1-скрипты / данные (read-only).

7. **Пре-рег-соответствие:** хедер скрипта == docs/AT0_2_prereg.md (расхождение = нарушение,
   урок N4); DELTA_MIN=0.05, α, 9 null-типов, пулы — как в пре-реге/р6.

## Ограничения

- НЕ меняй файлы репо (только чтение + вычисления в /tmp). Store 222.8MB — читай
  выборочно (json.load целиком влезает в память; либо ijson по клеткам).
- Числа в вердикте — только из фактических пересчётов/артефактов (не из головы).
- Расхождение = MAJOR/MINOR + точный path + ОБА числа (заявлено vs пересчитано).
- В конце: «Вердикт: PASS/FAIL» + список MAJOR/MINOR + что проверить при ответке.

Финальный маркер в конце вывода: AT0_2_R6_REVIEW_DONE
