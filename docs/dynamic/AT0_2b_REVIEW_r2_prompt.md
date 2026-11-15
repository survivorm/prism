# Ревью AT0_2b r2: R/A-split «кто несёт DC» (код + артефакт)

Роль: внешнее ревью (round 2) экспериментальной клетки AT0_2b (T0.2b). Предыдущее ревью
(docs/AT0_2b_review.md, «PASS как R/A decomposition / REVISE verdict rule») → r2-патч:
carrier_verdict по вкладу в net displacement Δh + Δh-DC gate + n_present/CI + прочее
(C1–C7, M1–M3). Задача: независимая верификация, что r2 закрывает замечания, и свой
независимый аудит.

## Что ревьюить (файлы)

Код:
- `experiments/AT0_2b_ra_split.py` — r2-скрипт (полный, disk-only, ~10 мин CPU).
- `docs/AT0_2b_prereg.md` — пре-рег v2 (зафиксирован ДО прогона: правило carrier_verdict
  §5 канал 2, CI-сиды §7, гейт energy §3.6, границы null §4).
- `prism/field.py` — пинненное ядро (ra_split, cross_acf, vpath, dcurve, degen; sha в claim).
- `experiments/AT0_1_accounting.py`, `experiments/AT0_2_shape.py` — зависимости
  (load_source/decode_probe/LAYERS; acf_full).
- `docs/AT0_2b_review.md` — исходные требования (C1–C7, M1–M3).

Артефакты:
- `results/AT0/AT0_2b_RA.json` — claim (round 2, date 2026-11-15).
- `results/AT0/store/b7e90c9366d9bbad.json` — store (480 клеток; canonical sha b7e90c93…,
  file sha e7ad8c05…).
- `results/AT0/AT0_2b_manifest.json` — manifest.
- `results/AT0/AT0_2b_run.log` + `/tmp/at02b_r2.log` — stdout (маркер «AT0_2b gates: ALL PASS»,
  RUN_RC=0).

## Обязательные проверки (минимум)

1. **Пре-рег ↔ код:** правило carrier_verdict в `_carrier_verdict` БИТ-точно соответствует
   пре-регу §5 канал 2 (порядок веток: not_testable → NO_DH_DC → CANCELLED → RADIAL/ANGULAR
   → INCONCLUSIVE; пороги: MIN_PRESENT=5, dh_frac_dc ≤ 0.5, cross < −0.5 ∧ обе shares > 0.5,
   CI_low > 0.5 ∧ med > med). CI-сиды: seed_ci = 9001 + pool_idx·1009 + L (pool_idx — index
   в sorted(pools)); seed f_dc = 2026 + doc_idx·1009 + s (doc_idx глобальный по 96).
   dominance_verdict = старое правило v1 без изменений (descriptive).
2. **Цепочка provenance:** manifest.claim_sha256 = sha256 файла AT0_2b_RA.json;
   manifest.store_file_sha256 = sha256 файла store; claim.meta.script_sha256 = sha256
   experiments/AT0_2b_ra_split.py; field_sha256 = sha256 prism/field.py; data_sha256 (gen/own/
   cor) — сверь с файлами; decode_probe_status = ok (не skipped); store_sha256 = canonical
   payload hash (пересчитай сам: canonical JSON sort_keys со store_sha256=None → sha).
   Конвенция C7 записана в meta.hash_convention — честная?
3. **C1 (contribution):** contrib_R = share_rad + 0.5·share_cross, contrib_A — зеркало;
   contrib_R + contrib_A = 1 ТОЧНО по всем 480 клеткам (проверь). Пересчитай shares/contrib
   для 3–5 клеток из store (у тебя нет исходных u/hn — пересчитай то, что посчитаемо из
   store-полей; полный пересчёт из npz — по 2–3 докам: загрузи gen/own npz через
   AT0_1_accounting.load_source, ра_split из field.py, сравни bit-eq/rel-1e-12).
4. **C2 (Δh-DC gate):** per-cell p_fdc_dh/fdc_dh_perm_median/dc_present_dh в store;
   агрегаты dh_frac_dc/dh_p_med/fdc_dh_med; gate сработал: везде dh_frac_dc = 0.0 ⇒
   NO_DH_DC. Пересчитай sign_flip_fdc для 2 доков (тот же сид) — bit-eq?
   Калибровка в main (iid median ∈ [0.5/N, 2/N]; DC p ≤ 1/(B+1)) — в логе?
5. **C3/C5 (n_present + CI):** n_cells_total/present, {radial,angular}_n_present/n_dc_present/
   absent_frac в агрегатах; bootstrap-CI: пересчитай для primary L23 (seed_ci, B_BOOT=199)
   — bit-eq с claim? Проверь, что CI — совместный (одни индексы для R и A и для ratio).
6. **C4 (null):** null_note в claim — «mean-direction null», НЕ ACF-matched (граница
   записана); в вердиктах/интерпретации нет смешения со spectral DC T0.1.
7. **C6 (provenance):** decode_probe_status/provenance_limited в meta и gates; при ok —
   provenance_limited = null; pool_note про Δh gen (provenance_limited, hn из файла) —
   сохранён?
8. **C7 (hash semantics):** file_sha256 для store/claim/manifest; manifest не содержит свой
   file_sha (только canonical manifest_sha256) — согласуется с конвенцией?
9. **M1 (energy gate):** per-cell energy_add_rel в store; max по клеткам = gate_max.energy
   в claim/логе (5.28e-16); tolerance 1e-12 в коде; тест-сценарий: если баг в ra_split —
   gate реально сработал бы (исполни фрагмент с искусственным нарушением).
10. **M2/M3:** vt_plateau — descriptive (note в meta + claim, НЕ в вердикте); absent_frac
    агрегируется.
11. **Гейты и инварианты:** n_absent = 2 (pin; v1-ошибка «6» поймана fail-loud — зафиксировано
    в пре-реге v2?); 480 клеток (96×5); primary = 37 (пин); gate max: closure 2.45e-15 /
    add 1.78e-16 / orth 8.88e-16 / energy 5.28e-16 (лог).
12. **Детерминизм:** перепуск скрипта в изолированной КОПИИ репо в /tmp (не в репо!) —
    canonical store sha совпадёт? (data-пути абсолютные; script_sha256 — содержимое файла).
    Внимание: повторный запуск в РЕПО перезапишет артефакты — бэкап в /tmp или копия репо.
13. **Код:** ветки/guard'ы (th is None-аналоги: absent-компоненты, nH=0, ma=0 в ratio,
    пустые lists в _boot_ci_joint); json.dump (jclean? NaN/None); запись после assert'ов;
    dead code; dead params.
14. **Ответ (claim.interpretation + pool_note):** «NO_DH_DC ⇒ нечего разводить (территория
    T0.5/VAR)» — согласуется с self-коррекцией пре-рега v1 (f_dc(Δh) ≈ 1.5e-5)? Нет ли
    заявлений, не подтверждённых артефактом?

## Ограничения
- НЕ менять файлы репо (чтение + вычисления в /tmp; запись — только в
  docs/dynamic/AT0_2b_REVIEW_r2.md).
- Числа в вердикте — только из фактических пересчётов/артефактов.
- Если находишь расхождение — MAJOR/MINOR с точным путём и числами.

## Формат вывода
`docs/dynamic/AT0_2b_REVIEW_r2.md`: (1) таблица «пункт ревью round 1 → статус в r2 →
доказательство»; (2) свои находки MAJOR/MINOR; (3) итог: PASS / PASS_WITH_MINORS / FAIL.
Финальный маркер в stdout: `REVIEW_DONE`.
