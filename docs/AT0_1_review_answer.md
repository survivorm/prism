# AT0_1 — ответка на ревью (docs/AT0_1_review.md)

**Статус: все пункты закрыты пост-хок аннотацией (r6-cosmetic), без перепуска.**
Дата: 2026-11-13. Правки: `T0_ACCOUNTING.json` (claim) + `AT0_1_manifest.json` (манифест) +
`AT0_1_artifact_hashes.json` (пины). Чисел прогона не изменено: store `23281c64…` и скрипт
`de305869…` не тронуты; манифест = claim + manifest-блок (инвариант сохранён, проверено).

## Пины после правки

| файл | sha256 |
|---|---|
| `T0_ACCOUNTING.json` (claim, аннотированный) | `b0f74312416285207ba99ae93ee75ecc446938b8b61ba82a19a04ac36b913994` |
| `AT0_1_manifest.json` | `ffcdca23a3abdfba283cd91af619bf4b079505207f8fe59daef7d7ee794a71de` |
| `store/23281c64cc418ff1.json` | `23281c64cc418ff1…` (не изменён) |
| `experiments/AT0_1_accounting.py` | `de30586930223208…` (не изменён) |
| run-пин (TZ_add_0-3.md v1.2, D4) | `d4a4b10135e8…` — сборка прогона; аннотированный файл = run + текстовые пометки |

## По пунктам

### #1. Версии артефактов не согласованы — принято

`AT0_1_artifact_hashes.json` перепинен на текущую сборку (claim/manifest — аннотированные;
store/script — без изменений). Старый round-4 пин (claim `1c6b0b40` / store `1af1e9f3` /
script `dc5bc3b2` / manifest `44de85ee`) **сохранён в поле `superseded_by_current`** вместе с
FIX-нотой 08:05 о 16f90dbd — история не теряется. В `meta.r6_annotation` зафиксировано, что
аннотированный claim = run `d4a4b101` + текстовые пометки (числа прогона и store не менялись).

### #2. P4 `refuted` — артефакт null, не свойство real — принято

Добавлено в `prereg.4_plateau_vs_fDC.<obj>` (claim + manifest):
- `h_raw`: `direction: "real_at_fdc"` + `direction_note` (est = −9.7e-5, CI содержит 0;
  real на ~19× ближе к f_dc, чем сдвиг null E = −1.85e-3);
- `du_unit`: `direction: "real_at_fdc"` + `direction_note` (est = −9.1e-5, CI [−2.1e-4, −1.2e-5]
  — ниже 0 с точностью CI, но на ~20× меньше сдвига null);
- `v_tangent`: `direction: "real_above_fdc"` + `direction_note` (est = +1.94e-3, CI
  [+6.9e-4, +3.4e-3] — плато выше f_dc, null E = −2.23e-3).

Само `verdict`-поле не переписано (пре-рег r4/r5 сохранён как зафиксированный до прогона);
разбор — в `direction`/`direction_note`.

### #3. `T_none_full = −12.87` — тривиальная разность — принято

В `aggregates.t0_headline_note` добавлено: `T_none_full = (r−1)/2 − (N−1)·f_dc` — **разность
двух больших слагаемых** (Gram-тождество), не «excess по всем лагам»; для v_tangent (r_median
≈ 24.9, f_dc_median ≈ 0.08) ≈ −12.87 — знак противоположен band mean P4 (+0.00194), не
«плато провалено». Проверено пересчётом (r6): max |sum_rho_full − (r−1)/2| = 1.07e-14 и
медиана T_none_full по 53 докам nondegen — **bit-eq** со значением в claim.

### #4. `ma1 median_z = 381` — MA(1) неадекватна — принято

В `prereg.4_plateau_vs_fDC.v_tangent.null_analytic_ma1` добавлено `applicability`:
f_dc ≈ 0.08 (DC-dominant), z = (s_doc + f_dc)/SE = mean_band/SE — blow-up — by-product
DC-доминирования в числителе, не «огромная структура»; описательно только.

### #5. `tpu_dev` не отчитан — принято

Добавлено `G0a_tpu_hn_field.max_rel_dev_reconstruction = 3.79e-16` (+ detail: определение,
worst = own_w6/L5, max по source/по слоям, self-audit, sha входов, провенанс). Значение —
детерминированный post-hoc пересчёт из исходных TPU-npz (pin'ы sha256 в артефакте),
`experiments/at0_1_tpu_dev_recompute.py`; self-audit — bit-eq воспроизведение отчётных
G0a evidence_own_w0 (4 значения). Машичная точность — как предсказано в ревью.

### #6. `n_l1_absent` не в артефакте — принято, с уточнением

Число **6 корректно как число L1-клеток**: 2 TPU-дока (own_w0, own_w4) × 3 объекта = 6
L1-клеток с K=0 (проверено по store). В `G8_gram` добавлено явно: `n_l1_absent_docs = 2`,
`n_l1_absent_cells = 6` + `n_l1_note`. (Пересечение с AT0_2: там `n_l1_absent_exec = 2` дока →
`n_absent = 6` клеток — та же структура, число 6 совпадает.)

### #7–#10. Мелкие — действия не требуются

- #7 (`t25_ekurve_sha256` — чистый provenance) — согласны, «хорошо документировано»;
- #8 (`p_two_sided_binom05` — неверный null, оставлен для сравнения) — согласны, OK;
- #9 (`perm_spread = 1.665e-16 < 1e-12`) — согласны, OK;
- #10 (`med_theta_deg = 55.15`, `dip_V − dip_du = 0.2575` — совпадает с AT0_3) — согласны,
  согласовано.

## Провенанс правки

- Скрипт правки: `/tmp/recovered/AT0_1_r6/apply_r6_edits.py` (+ уточнение #6) — формат
  идентичен генерации (claim компактный / manifest indent=1, `jclean`, ensure_ascii=False);
  пре- и пост-проверки инварианта manifest = claim + manifest-блок — bit-eq.
- Пересчёт: `experiments/at0_1_tpu_dev_recompute.py` (в репозитории).
- Бэкап исходных артефактов (run `d4a4b101`): `/tmp/recovered/AT0_1_r6/`.
