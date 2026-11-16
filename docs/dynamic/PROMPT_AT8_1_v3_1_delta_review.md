# Промт: внешнее DELTA-ревью AT8_1 артефакта v3.1 (T8.1-r3, per-seed C1 фикс)

Ты — независимый ревьюер экспериментов prism. Предыдущее внешнее ревью артефакта v3 —
**PASS** (`docs/AT8_1_artifact_r3_review.md`, sha `3acac879…`, два прохода: REVISE[MAJOR
провенанс] + PASS; MAJOR закрыт коммитом 7d26167). Теперь — **только дельта v3 → v3.1**:
исправленный перепуск машины v3 по MINOR-1 (per-seed C1 + ложный note). НЕ пересматривать
целое заново — проверять дельту + интактность v3/v1/v2. Только read-only + ОДИН файл ревью.

## Что читать

1. **Артефакт (объект):** `results/AT8/AT8_1_THRESHOLD_v3.1.json` +
   `results/AT8/AT8_1_v3.1_manifest.json` + `logs/AT8/AT8_1_v3_1_run.log`
   (маркер `AT8_1_R3_DONE verdict=V3_PATHOLOGY match=True`, `RUN_RC=0`).
2. **База (заморожен):** `results/AT8/AT8_1_THRESHOLD_v3.json` (sha `c4fcb1ef5ad0d22f…`,
   manifest `AT8_1_v3_manifest.json`, `script_sha256 = 6b064565d9353a97…`).
3. **Скрипт (исполняемый v3.1):** `experiments/at8/AT8_1_r3_reanalysis.py` (текущий файл на
   диске; `script_sha256` в v3.1-manifest обязан ему равняться).
4. **Store (read-only, источник per-seed loopy):** `results/AT8/store/018c3dac90eb1bba.json`
   (sha `28a0c28d94d9…`); пер-seed loopy — `cells[s{sd}_lam{λ}].gen.loopy`.
5. **TZ (пре-рег):** `docs/TZ_add_4-8.md` §T8.1 «Машина v3 (T8.1-r3)» п.6 — per-seed C1/C2/C3.
6. **Ответка:** `docs/AT8_1_artifact_r3_answer.md` (разбор MINOR-1 + v3.1-секция).

## Проверки (только дельта)

| # | Проверка |
|---|---|
| D1 | **Per-seed C1 числится и верен:** в `raw_report.per_seed.s{11,12,13}.c1_loopy` — `c1 = gen.loopy(λ*_seed) > 0`, λ*_seed — per-seed argmin dip (из того же артефакта: `lam_star`). Независимый пересчёт из store: s11 λ*=0.70 → loopy=0 → FALSE; s12 λ*=0.70 → FALSE; s13 λ*=0.65 → FALSE. Совпадает? |
| D2 | **Согласованность:** seed-avg по сидам `gen.loopy` на каждой точке λ == loopy-кривой claim v1 (`AT8_1_THRESHOLD.json`, curves_seedavg.loopy); selfcheck `per_seed_loopy_vs_claim_dev` ≤ 1e-6. |
| D3 | **Note честен:** per-seed note артефакта v3.1 больше НЕ содержит «per-seed loopy в store отсутствует»; формула C1 указана. |
| D4 | **Дельта только заявленная:** diff v3-артефакта vs v3.1-артефакта = per-seed C1-блок + note + selfcheck-поля (no_nan_nonfinite_count, per_seed_loopy_vs_claim_dev, v1_intact, v2_intact) + revision/machine/note-строки + date. ВСЕ остальные числа (cond1, C1–C3 seed-avg, локаторы, gaps, per-seed dip/cond1/C2/C3, F5 Δ, families, reachability, verdict) — побитово равны v3. |
| D5 | **Вердикт не изменён:** `V3_PATHOLOGY`, `expectation_match=true`; маркер в логе; `recompute_dev` ≤ 1e-9; `no_nan_nonfinite_count = 0`. |
| D6 | **Провенанс:** `script_sha256` (v3.1-manifest) = sha256 текущего скрипта на диске; `claim_file_sha256` (v3.1-manifest) = sha256 v3.1-артефакта на диске; входы (claim v1 `b2923d74…`, store `28a0c28d…`, manifest `159967235670…`) интактны. |
| D7 | **Замороженные не тронуты:** v3-артефакт sha = `c4fcb1ef5ad0d22feabf42f8fad819f120f13787671015ecbd5b9f941910ebd6`; v2 = `bb507dac4d6969814980dc65d3f52b37f009745f7436a9df9b48db4652179157`; v1 = `b2923d74324e9180407f143a83fcffeaba22634ffe212cf723f86456457699a0`; selfcheck v1_intact/v2_intact = true. |

## Формат

Файл: `docs/AT8_1_artifact_v3_1_review.md`. Структура: заголовок (дата, объект, sha
прочитанного), таблица D1–D7 со статусами и цифрами, замечания (MAJOR/MINOR/NIT),
«Итог», затем **VERDICT: PASS|REVISE|FAIL** и финальная строка **REVIEW_DONE**.
Числа — только из файлов; каждое утверждение — с доказательством (пересчёт/sha/grep).
