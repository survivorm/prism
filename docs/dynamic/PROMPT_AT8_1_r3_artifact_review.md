# Промт: внешнее ревью артефакта AT8_1 машина v3 (T8.1-r3)

Ты — независимый ревьюер экспериментов prism. Проверь **артефакт машины v3** против
пре-рега в TZ и раунд-2 ревью. Только read-only + ОДИН файл ревью.

## Что читать

1. **Артефакт (объект ревью):**
   - `results/AT8/AT8_1_THRESHOLD_v3.json` — claim v3 (объект)
   - `results/AT8/AT8_1_v3_manifest.json`, `results/AT8/AT8_1_v3_run.log`
2. **Пре-рег (контракт):** `docs/TZ_add_4-8.md` — секция «Машина v3 (T8.1-r3)»
   (строки ~908–975; git: закоммичено в `790a375`).
3. **Ревью-основание (что машина обязана закрыть):** `docs/AT8_1_review.md`
   (sha `56edfeba…`) — минимальный патч §5 пункты 1–5 + §6 «что прислать» (F1–F5).
4. **Входы (read-only, сверка provenance):**
   - `results/AT8/AT8_1_THRESHOLD.json` (claim v1, sha `b2923d74…`)
   - `results/AT8/store/018c3dac90eb1bba.json` (store, sha `28a0c28d…`)
   - `results/AT8/AT8_1_manifest.json` (sha-верификация входов)
   - `results/AT8/AT8_1_THRESHOLD_v2.json` (claim v2 — должен НЕ быть тронут)
5. **Код машины:** `experiments/at8/AT8_1_r3_reanalysis.py` (sha в v3-manifest).
6. **Контекст:** `docs/AT8_1_review_r3_answer.md` (ответка — проверь, не утаила ли она
   расхождений), `docs/dynamic/AT8_1_r3_TZ_review.md` (TZ-ревью, уже PASS).

## Что проверить

1. **Provenance:** sha-цепочка (claim v3 ↔ manifest ↔ run.log); входы не тронуты
   (sha claim v1 / store / v2-claim совпадают с декларированными); store не изменён.
2. **Самопроверки артефакта:** `selfcheck.recompute_dev` (dip/eff_rank/nll_holdout) и
   `u_dip_dev` — должны быть 0.0 (пересчёт из store == артефакт v1).
3. **Патч раунда 2 — пункт за пунктом (F1–F5 + O2):**
   - O2/п.1: V3 — знаковый детектор `min dip_seedavg < dip(1.0)` + корроборация
     C1∨C2∨C3 в λ*-клетке; hinge-условие УБРАНО; max_u — описательный.
   - F1/п.2: таблица 3 локаторов (L1 u=0.5 / L2 onset >2×медианы / L3 max-скорость)
     × {dip, nll_holdout, margin}; если вердикт отделения меняется между локаторами
     ⇒ V2 not_resolved (НЕ V2).
   - F2/п.3: per-seed L3-клетки + span-правило (>1 шага ⇒ unresolved, отчётное).
   - F3/п.4: mode → vacuous (через no_transition; предпосылка ревью «rep16(0)=0»
     опровергнута данными — проверь, что это disclosed и выводит mode в vacuous).
   - F4/п.5: S9 — grid-limited декларация (ширина 0.05, не гейт).
   - F5/п.6 + §6: raw-отчёт dip(0)/dip(1)/Δ (знак явный) + min/max dip (с λ) +
     полные per-seed dip-кривые (3×11) + per-seed cond1/λ*.
4. **Независимый пересчёт (обязательно, python3 + json + numpy, без torch):**
   - Из store: 6-doc медиана (wiki_0..wiki_5) → 3-seed среднее (s11/s12/s13) по 11 λ —
     сверь с `raw_report.dip_seedavg`, `min_dip`, `dip_1`, Δ, per-seed кривые.
   - Условие cond1 и C1/C2/C3 в λ*-клетке — пересчитай и сверь с `verdict.details.v3`.
   - Локаторы (u из сырых кривых: u(λ)=(O(λ)−O(0))/(O(1)−O(0))) для dip/nll_holdout/margin —
     пересчитай L1/L2/L3 и gaps (gap = мин. шагов сетки между граничными узлами интервалов)
     и сверь с `verdict.details.v2`.
   - Per-seed L3-клетки и span = (max−min) индексов клеток (левые узлы) — сверь.
   - families: informative-правило (u определена; бинарные — с переходом; no_transition —
     исключены; <2 информативных ⇒ vacuous) — сверь с `families_v3`.
5. **Вердикт:** порядок V3 > V2 > V1 > V0; `expectation_match` (pre-reg = V3_PATHOLOGY);
   v2-вердикт не переписан (v2-claim intact).
6. **Код:** sha256 скрипта == manifest; правила кода == TZ (нет скрытых отступлений:
   gap-конвенция, span-конвенция, tie-правила, per-seed C2/C3-агрегация).
7. **Ответка:** `docs/AT8_1_review_r3_answer.md` — каждое число в ней сверь с артефактом
   (нет расхождений, нет утаенного); A1–A8-соответствие — честное (A2/A3 — «НЕ закрыто»).

## Формат

Файл: `docs/dynamic/AT8_1_r3_artifact_review.md`
- Заголовок, дата, список прочитанных файлов с sha256.
- Таблица: пункт проверки → статус (ок / MAJOR / MINOR / NIT) → аргумент с числами.
- Найденные дыры — с формулировками фикса.
- Итог: **VERDICT: PASS | REVISE | FAIL** (REVISE только при MAJOR).
- Финальный маркер: `REVIEW_AT8_1_R3_ARTIFACT_DONE`

Числа — только из файлов (git show/артефакты/store), не из памяти. Не правь ничего,
кроме своего ревью-файла.
