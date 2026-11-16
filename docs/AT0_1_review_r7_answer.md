# AT0_1 — ответка на ревью round 2 (docs/AT0_1_review.md, C1–C7)

**Статус: C1–C7 закрыты аннотацией r7 (детерминированный пересчёт из закреплённых входов,
без перепуска эксперимента) + перепуск r5 для восстановления цепочки provenance + 6 MINOR
round 3 закрыты (r7 v2).**
Дата: 2026-11-15. Скрипт: `experiments/at0/at0_1_r7_annotation.py` (assert'ы само-аудита;
любой сбой ⇒ артефакт не записывается). r7-claim: 82f44f0813f6adde… (пин
`results/AT0/store/claim_r7_82f44f0813f6adde.json`). Changelog r7 — в claim
`meta.r7_annotation`. Внешнее ревью round 3 — docs/dynamic/AT0_1_r7_REVIEW.md (PASS, MAJOR: нет).

## Согласие/несогласие по пунктам

| пт | вердикт | действие |
|---|---|---|
| C1 (MA(1)-центрирование) | **согласен — баг** | z_excess = s_doc/se; z_raw_band (старая статистика) — диагностика; E_doc переписан |
| C2 (tpu_dev = тавтология) | **согласен — баг** | rename в `reconstruction_roundoff_identity`, status «тождество, не provenance-гейт» |
| C3 (claim ≠ script_sha) | **согласен — баг** | перепуск r5 (bit-eq store+log) → r5-claim закреплён; full-diff r5→r6 и r6→r7 — assert; chain-sha в meta |
| C4 (SE на ambient D) | **согласен по существу; уточнение по метрике** | se_eff = se·sqrt((1+2ρ₁²)·D/D_eff); D_eff — participation ratio (25–35, не 178 — см. ниже) |
| C5 (SE_formula /98 vs /88) | **согласен — опечатка в тексте** | /88 (код всегда считал K_MAX−8 = 88) |
| C6 (T_none_full в excess) | **согласен** | вынесен в `identity_check_G1`; headline = r/f_dc |
| C7 (self-audit не те числа) | **согласен** | + l23_meta_vs_recomputed, roundoff, theta, z_raw_band=r5, T_none_full, pool — assert'ы |

## C1 — детали и числа

Проверено численно по store (per_doc: rho_Tk, f_dc, N, D):

- Старая статистика `z = (s_doc + f_dc)/se = mean_band/se` воспроизводится bit-eq
  (store per_doc.null_ma1_z.z, медианы 381.325 / −0.2157 / −0.2156 — как в r5/r6 claim).
- Корректный null: при matched MA(1) **с DC** и T-k нормировке
  E[rho_Tk(k≥2)] = ‖m‖²/(‖m‖²+var) = f_dc (c(0) = (‖m‖²+var)/D, c(k≥2) ≈ ‖m‖²/D — DC не
  сокращается) ⇒ E[mean_band] ≈ f_dc, E[s_doc] ≈ 0. Т.е. r5/r6 z тестировал «mean_band = 0»,
  а не «плато на уровне f_dc».
- Числа (r7 claim, пул П4, n=51):

| объект | z_raw_band (старая) | z_excess (ambient) | z_excess_eff (C4) |
|---|---:|---:|---:|
| h_raw | med −0.2157 | med −0.4109 (mean −0.2016) | med −0.0431 (mean −0.0352) |
| du_unit | med −0.2156 | med −0.4958 (mean −0.1909) | med −0.0646 (mean −0.0351) |
| v_tangent | med **381.325** | med **10.5475** (mean 17.4853) | med **1.9309** (mean 2.6412) |

  (н. C1 ручной счёт ревью: 10.5 — совпадает; для h_raw/du_unit сдвиг (z_raw − z_excess) =
  f_dc/se per-doc: median ≈ 0.067–0.068 (N=383: ~0.07, как в ревью), max ≈ 0.38–0.46 (малые N)
  — мал по сравнению с 370 у v_tangent, но не нулевой; на первичный канал не влияет.)
- Вывод: «381 = plateau-сигнал 30–150× завышен» — верно по арифметике; после поправки
  v_tangent: z_excess (ambient SE) med 10.5 / mean 17.5, n|z|≥3 = 45/51; по se_eff:
  med 1.93 / mean 2.64, n|z|≥3 = 21/51 — т.е. «MA(1) не объясняет v_tangent-плато» выживает
  как 2–3σ-наблюдение, описательное (вердикт по-прежнему — по первичному каналу, r5 D2).

## C4 — уточнение по метрике

Ревью ссылается на `r_eff^U(0.05)=178` (addentum) — это **5%-пороговый ранг спектра U-состояний**
(другая метрика). Вария-релевантная величина для se — participation ratio ковариации строк
объекта (zero-mean): **D_eff(med) = 25.0 / 26.0 / 34.5** (min 8–9, max 52–68) для
h_raw / du_unit / v_tangent ⇒ фактор sqrt((1+2ρ₁²)·D/D_eff) = 7.2 / 7.0 / 5.5 (не 2.4–4.5).
Ambient D=1024 оставлен нижней границей (se ≤ se_eff); Bartlett-фактор (1+2ρ₁²) включён
(формула «поправка нулевая для k > p+1» в r5/r6 неверна — var(ρ̂_k) ≈ (1/N)(1+2ρ₁²) для MA(1)).

## C3 — цепочка provenance

- **Цепочка provenance (C3):**
  - Перепуск `de305869` (env r5: python 3.12.9/np 2.5.3, threads=4, TZ 67addba/647c03ab):
    **bit-eq** store `23281c64cc418ff1` + run.log `e6dcd6fc62aeced0` ⇒ r5-claim
    `d4a4b10135e8…` восстановлен и закреплён `results/AT0/store/claim_r5_d4a4b101.json`
    (+ manifest_r5_4099c305). sha r5-claim ранее был записан только в `meta.r6_annotation.base_run`
    (без пин-файла) — р7 закрепляет файл и воспроизведение. TZ на момент прогона = 67addba
    (подтверждено коммитом 2dab888: «прогон на TZ-sha 647c03ab»).
- r6-claim закреплён `results/AT0/store/claim_r6_b0f74312.json` (до перепуска).
- Full diff **r5→r6** = ровно 14 декларированных r6-путей (direction/direction_note ×3,
  applicability, t0_headline_note, max_rel_dev_reconstruction+detail, n_l1_* ×3, meta.r6_annotation)
  — assert в r7-скрипте. Full diff **r6→r7** = ровно декларированный r7-набор (30 added /
  8 removed / 7 changed) — assert; все остальные листы bit-eq (т.е. «числа прогона не изменено»
  теперь проверено, а не заявлено).
- Meta-цепочка в claim: `meta.r7_annotation.chain` — run_script / r5 / r6 / r7 shas +
  annotation_script_sha256. r6-аннотационный скрипт в репозитории не был (только
  at0_1_tpu_dev_recompute.py — пересчёт чисел); зафиксировано в chain-ноуте.

## C2 — детали

U·hn ≡ h в обеих ветках max(·,1e-12) (hn>1e-12: U = u = h/hn; hn≤1e-12: U = h/hn) — значение
3.79e-16 = round-trip divide→multiply, провалить не может (кроме NaN/inf). В claim переименовано
`reconstruction_roundoff_identity` (+detail) со статусом «тождество, не provenance-гейт»;
годность TPU-рядов — `l23_meta_vs_recomputed` (max dev 2.08e-4 ≤ 1e-3) — теперь и в self-audit
(r7, C7).

## C5/C6/C7 — коротко

- C5: SE_formula переписана: /88 (= K_MAX−8, 88 лагов; arange(9,97) — 88 лагов); код не тронут.
- C6: `t0_headline.<obj>.identity_check_G1` — T_none_full_median/mean + max |sum_rho_full −
  (r−1)/2| + пояснение; T_none_full удалён из excess_median/excess_mean; headline = r/f_dc
  (r_v ≈ 24.9 — not DC-free; r_du ≈ 0.0048 — DC-suppressed) — в note.
- C7: self-audit r7 — assert'ы (не stdout): G0a evidence_own_w0 (4 числа), roundoff (max/worst/
  by_source/by_layer), l23_meta_vs_recomputed (2 max-dev, n_docs, worst_doc), theta (51×3),
  z_raw_band = r5-статистика (51×3), T_none_full (3×2), П4-пул (51). Все воспроизведены bit-eq.

## Round 3 (внешнее ревью) — PASS, MAJOR: нет; 6 MINOR закрыты (r7 v2, claim 82f44f08)

- **MINOR-1** дата: 2026-11-14 → 2026-11-15 (meta.r7_annotation.date + provenance-ноты).
- **MINOR-2** BLAS-треды: в claim (c1_c4_note): per_doc d_eff/z_excess_eff и se_factor_median —
  bit-eq только в pinned-окружении (run_one.sh: OpenBLAS/MKL_NUM_THREADS=6; eigvalsh
  чувствителен к ним; r7-прогон: OMP_NUM_THREADS=4); в другом thread-окружении дрейф 1–2 ulp
  (≤3.4e-15 rel, независимо проверено ревьюером); z_raw_band/z_excess/θ/T_none_full/roundoff/
  l23_meta — bit-eq в обоих окружениях.
- **MINOR-3** r6-скрипт закреплён: `experiments/at0/at0_1_r6_annotation.py` (= apply_r6_edits.py из
  /tmp/recovered/AT0_1_r6/, sha 0d52eda5…) + sha в meta.r7_annotation.chain.
- **MINOR-4** headline r/f_dc ≡ N (G3-тождество r = N·f_dc) — «тривиально, не цитировать»;
  заголовочный ответ = r per object (r_v ≈ 24.9 / r_du ≈ 0.0048).
- **MINOR-5** guard `th is None` — в claim: defensive, не достигается на данных (все 51 доки
  пула matched); проверен exec'ом реальной ma1_recompute с имитированным no_match (ревьюером).
- **MINOR-6** дубль median_z/z_raw_band — в claim: сохранён инвариантом r5-листьев,
  цитировать z_raw_band.

Проверено ревьюером round 3 (независимые пересчёты): C1 z-статистики по 153 (док, объект)
bit-eq + MC-проверка E_doc; C2 roundoff + ветки тавтологии; C3 полная цепочка + независимый
diff r5→r6 (34 листа) и r6→r7 (744/21/7 leaf'ов) + повторный запуск r7-скрипта в изолированной
копии репо (bit-eq claim+manifest); C4 d_eff/se_factor bit-eq в pinned-6-thread; C5 «/88» +
r5-claim содержит старый «/ 98» (правка только в артефакте); C6 T_none_full bit-eq с r6;
C7 l23_meta bit-eq. Ответка сверена: каждое число — из артефакта/лога, «числа из головы» нет.

## Чего НЕ сделано (и почему)

- **Вердикты/приговоры P4/P5 не переписаны** — первичный канал (p_perm против pair-block null,
  σ/эффект) не затронут; MA(1)-столбец описательный (r5 D2) — его числа исправлены, статус
  прежний (ревью round 2 не требовало переписывать вердикты — проверено round 3).
- **Перепуск эксперимента (чисел прогона)** не нужен: все изменения — пересчёты из store/npz
  (детерминированы, входы sha-pinned; детерминизм прогона доказан перепуском r5, bit-eq
  store+log).
- **d_eff по «приращениям»** (формулировка ревью) — объекты AT0_1 (Δh, Δu, V) сами приращения;
  использована ковариация строк объекта (тот же объект, что в rho_Tk) — вария-релевантная
  спектральная метрика (см. C4).
- **Внимание к окружению:** перепуск r5 потребовал (a) python 3.12.9/np 2.5.3 (env r5; .venv —
  3.13.14/2.5.2, meta.env в claim'е зависело бы от этого), (b) TZ в версии 67addba (на момент
  r5; позже TZ патчили — 2dab888/f2604a1/6c1cdf3), (c) установку packaging/tqdm/pyyaml в
  uv-python 3.12 (transformers в decode_probe). Всё зафиксировано в chain-ноуте claim.
