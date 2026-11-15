# AT0_2b — T0.2b «R/A-split: кто несёт DC» — пре-рег (зафиксирован ДО прогона, 14.11; v2 — 15.11, после review)

**v2 (15.11):** правки по `docs/AT0_2b_review.md` (C1–C7, M1–M3): (1) вердикт = carrier
по вкладу в net displacement Δh (contribution = share + half-cross), а не по f_dc
компоненты; (2) DC-присутствие Δh (p_fdc_dh) — gate перед carrier-вердиктом;
(3) n_present + bootstrap-CI (B=199) в агрегатах; (4) sign-flip null — «mean-direction
null», НЕ ACF-matched spectral null; (5) n_cells_total/present + absent_frac;
(6) decode_probe_status → provenance_limited; (7) file_sha256 (конвенция: store_sha256 =
canonical payload hash без self-поля; *_file_sha256 = sha256 фактического файла);
(8) energy-additivity gate; (9) vt_plateau — descriptive; (10) старое правило —
dominance_verdict (descriptive, не primary). Чисел уже посчитанных не меняется — r2 =
пере-агрегация тех же клеток + новые поля (store r1 не существует — клетки считаются заново,
детерминизм гарантирован сидами).

ТЗ: `docs/TZ_add_0-3.md` §1.5 T0.2b + (G6)/(G6′) §1.3. Драйвер: `experiments/AT0_2b_ra_split.py`.
Ядро — пинненный `prism/field.py` (G1′, sha в манифесте): `ra_split`, `cross_acf`, `f_dc`,
`degen`, `jclean`. ACF — `AT0_2_shape.acf_full` (FFT, zero-padded ≥ 2N−1, пин-тест в main).

## 1. Объект и вопрос

Точный ортогональный R/A-split (G6): `Δh = r̄·Δu + Δr·ū`, `r̄=(r_t+r_{t+1})/2`,
`ū=(u_t+u_{t+1})/2`, `⟨Δu,ū⟩ = 0` ТОЧНО при равных t. Компоненты:
- **radial** `R_t = Δr_t·ū_t` — приращения НОРМЫ ‖h_t‖;
- **angular** `A_t = r̄_t·Δu_t` — приращения НАПРАВЛЕНИЯ (взвешенные средней нормой).

Вопрос T0.2b (пре-рег ТЗ): **какая компонента несёт DC**:
- **RADIAL_DC** ⇒ DC объясняется ростом/дрейфом нормы ‖h_t‖;
- **ANGULAR_DC** ⇒ когерентный поворот (систематический Δu);
- **NO_DC** ⇒ ни у какой компоненты DC нет.

**Самокоррекция (зафиксирована ДО прогона, 14.11):** ТЗ-формулировка «плато T0.1,
f_DC(Δh) ≈ 0.057–0.065» — КОНФУНД. Измерено по T0.1 store (23281c64cc418ff1, пересчёт
совпадает бит-в-бит, max |Δ| = 8e-17): «плато 0.057–0.065» — это АCF-УРОВЕНЬ **v_tangent**
(medians по докам rho_Tk(9..32) = 0.070 L11 / 0.075 L17 / 0.081 L23, 0.095 L5 / 0.130 L1),
а не f_dc(Δh). Для h_raw/du_unit medians по докам rho_Tk(9..32) ≈ −0.0045 (плато нет),
и f_dc(Δh) ≈ 1.5e-5 (med r(Δh) = 0.0056 — h блуждает вокруг опорной точки, нет дрейфа).
«f_DC» в G8 ТЗ — параметр-уровень DC+MA(1)-фита к кривой v_tangent, не измеренный f_dc.
Следствие: ожидаемый исход — **NO_DC** (плато v_tangent — персистентность направления,
не DC компонент Δh ⇒ территория T0.5/VAR), и в отчёт входит привязка: medians по докам
v_tangent rho_Tk(9..32) per слой (vt_plateau) — якорь «плато» к реальному объекту T0.1.
ТЗ-пре-рег (радиальный/угловой DC) применяется к компонентам Δh без изменения.

## 2. Данные и клетки

- 96 доков (gen 64 + own 16 + corpus 16) × L{1,5,11,17,23} — те же npz, что AT0_1/AT0_2
  (read-only; sha в манифесте; gen_sha 64/64; decode-проба).
- L1-TPU degraded (правило AT0_1/AT0_2: |recomputed dip_V,med L1 − meta| > 1e-3):
  absent own_w0, own_w4 ⇒ **480 клеток (96×5), n_absent = 2 (пин, fail-loud; v2: 6 в v1 —
  арифметическая ошибка, поймана fail-loud при первом полном прогоне)**.
- K = min(96, N−1), N = T_states−1 (gen/own 383; corpus 82..254 — v2.1 MINOR-4: 123..254 в v2 —
  неточно; фактически cor_w8 N=82, cor_w11 N=94).
- Δh для gen: h = u·hn (hn из файла — provenance_limited, scale-пин AT0_2 G0a; тот же
  остаточный риск, что h_raw в AT0_2). Для own/corpus: h raw fp16, hn = ‖h‖ (верифицировано
  конструкцией).

## 3. Гейты (fail-loud, на каждой клетке)

1. **Closure (ТЗ G6′, REL):** `max_t |‖Δh_t‖² − ‖R_t‖² − ‖A_t‖²| / max_t ‖Δh_t‖² ≤ 1e-12`
   (абсолютное 1e-12 недостижимо при ‖Δh‖² ~ 3.6e4 — ТЗ; измерено в field._selfcheck).
2. **Additivity (REL):** `max_t ‖Δh_t − R_t − A_t‖ / sqrt(max_t ‖Δh_t‖²) ≤ 1e-12` —
   раскладка точна, не только по энергии.
3. **Ортогональность при равных t:** `max_t |⟨Δu_t, ū_t⟩| ≤ 1e-9` (field: du_u_cross_max).
4. **G1 на компонентах:** `|1 + 2·Σ_{k<N} ρ̂(k) − ‖Σx‖²/c0| / max(1, r) < 1e-10` для R и A
   (тот же GRAM_TOL, что AT0_2 G8).
5. `c0(R) > 0`, `c0(A) > 0` — иначе компонент absent-degenerate (отчёт, не тише).
6. **v2 (M1): energy-additivity:** при обеих компонентах present: `|c0(Δh) − c0(R) −
   c0(A)| / c0(Δh) ≤ 1e-12` (следствие same-t ортогональности; ловит баг ra_split/
   нормировки раньше verdict).

## 4. Null и p (стандарт серии)

- **Null для f_dc — SIGN-FLIP (Rademacher), B = 199** (p_floor = 1/200 = 0.005 — урок E3:
  B=20 → p_floor 0.0476 saturated; AT0_2 round 6). **v2 (C4): название — «mean-direction
  null»**: это randomization-гипотеза «нет систематического среднего направления при
  фиксированных ‖x_t‖». Перестановка строк для f_dc ВЫРОЖДЕНА — mean(x) инвариантен к
  ЛЮБОЙ перестановке (Σ invariant), все B «permutation-null'ов» = fdc_obs тождественно.
  Sign-flip — точная randomization-гипотеза «нет систематического среднего»: x* = ε⊙x,
  ε_t ∈ {±1}; E[fdc_s|null] = 1/N (denominator mean‖x_t‖² инвариантен к знакам — меняется
  только числитель ‖Σ ε_t x_t‖²). **Граница (v2 C4): null НЕ сохраняет ACF/telescoping/
  bounded-path — «DC present» = «направленный mean drift», НЕ «DC-мощность значима
  относительно stationary-процесса с той же ACF» (не мостить к spectral DC T0.1/T0.2).**
- Сиды: `seed = 2026 + doc_idx·1009 + s`, **doc_idx — глобальный по 96** (конвенция серии,
  фикс AT0_3 r3.2 — не позиционный внутри стратума).
- `p_fdc = (#{s: fdc_s ≥ fdc_obs} + 1) / (B + 1)`; **DC-присутствие компонента:
  p_fdc < 0.05.**
- f_dc (G3): `f_dc(x) = ‖mean(x)‖² / mean(‖x‖²)`; для Δh: f_dc = r/N (G3-тождество,
  т.е. DC-уровень Δh — тот же r-объект T0, делённый на N).
- Калибровка (читать ПЕРВОЙ, fail-loud в main): на iid-синтетике median fdc_s обязан
  лежать в [0.5/N, 2/N]; на синтетике с сильным DC p обязан ≤ 1/(B+1) (правило 6).

## 5. Вердикт (L, пул) — правило решения (v2: два канала)

Пулы: `all_96` (96), `primary` = verified ∩ nondeg (flag AT0_2: L23 vcurve + token share,
пин n = 37 — пере-счёт обязан дать 37, иначе несовместимость с AT0_2), `loopy` = degen.

Для пула и слоя: `frac_R` = доля PRESENT-доков с DC у radial (p_fdc < 0.05), `frac_A` —
у angular; `med_R`, `med_A` — медианы f_dc; `n_present_R/A` — число present-клеток
(v2 C3: denominator всегда виден); `dh_frac_dc` = доля present-доков с DC у Δh.

**Канал 1 — dominance_verdict (v2: DESCRIPTIVE, не primary)** — старое правило v1 без
изменений (множитель 1.1 — описательный, без CI):

| условие | dominance_verdict |
|---|---|
| frac_R > 0.5 и med_R > 1.1·med_A | **RADIAL_DOM** |
| frac_A > 0.5 и med_A > 1.1·med_R | **ANGULAR_DOM** |
| frac_R > 0.5 и frac_A > 0.5 | **MIXED_DOM** |
| frac_R ≤ 0.5 и frac_A ≤ 0.5 | **NO_DC_DOM** |
| иначе | **INCONCLUSIVE_DOM** |

**Канал 2 — carrier_verdict (v2: PRIMARY, «кто несёт DC плато Δh»)** — по вкладу
компонент в net displacement Δh = ΣΔh (C1/C2):

```text
входы: dh_frac_dc; med_cR, ci_cR_low (bootstrap 95% CI нижней границы медианы contrib_R);
       med_cA, ci_cA_low; share_rad_med, share_ang_med, share_cross_med;
       n_present_R, n_present_A (>= 5 для тест-ability)

if n_present_R < 5 или n_present_A < 5:           → not_testable
if dh_frac_dc is None or dh_frac_dc <= 0.5:       → NO_DH_DC   (у Δh нет DC — нечего разводить)
elif share_cross_med < −0.5 и share_rad_med > 0.5 и share_ang_med > 0.5: → CANCELLED
     (обе большие, но почти отменяются — «оба несут» неверно)
elif ci_cR_low > 0.5 и med_cR > med_cA:           → RADIAL_CARRIER  (плато Δh = drift нормы)
elif ci_cA_low > 0.5 и med_cA > med_cR:           → ANGULAR_CARRIER (когерентный поворот)
else:                                             → INCONCLUSIVE
```

(Совместное ci_cR_low > 0.5 ∧ ci_cA_low > 0.5 невозможно — cR + cA = 1; порядок веток
задаёт детерминированность. Ветка CANCELLED проверяется ДО доминантных — cancellation
маскирует contribution.)

**Interpretation:** RADIAL_CARRIER ⇒ DC-плато Δh = drift/рост нормы ‖h_t‖;
ANGULAR_CARRIER ⇒ когерентный поворот; NO_DH_DC ⇒ нечего разводить (территория T0.5/VAR —
как самокоррекция пре-рега v1 §1: f_dc(Δh) ≈ 1.5e-5, плато T0.1 — персистентность
v_tangent, не DC Δh).

## 6. Кросс-лаги и точное разложение net displacement (ОЧЕРЕДНО, без вердикта)

`c_RA(k) = Σ_t R_t·A_{t+k}`, `c_AR(k) = Σ_t A_t·R_{t+k}`, k = 1..K;
нормировка `ρ_RA(k) = c_RA(k)/sqrt(c0_R·c0_A)`. Г6-ортогональность — только при равных t;
кросс-лаг ≠ 0 — ожидаемо, отчитывается (claim: медиана по докам k=1..16; store — полный
профиль). Значение |ρ_RA(1)| > 0 — НЕ нарушение G6.

Точное разложение: `Σ_t Δh_t = h_end − h_start = ΣR + ΣA` ⇒
`share_rad = ‖ΣR‖²/‖ΣΔh‖²`, `share_ang = ‖ΣA‖²/‖ΣΔh‖²`, `share_cross = 2⟨ΣR,ΣA⟩/‖ΣΔh‖²`
(сумма = 1 точно; кросс-член ≠ 0 — ортогональность G6 только при равных t). Отчёт: медианы
по пулу/слою.

**v2 (C1): contribution (half-cross attribution)** — вклад компоненты в net displacement:
`contrib_R = share_rad + 0.5·share_cross`, `contrib_A = share_ang + 0.5·share_cross`
(contrib_R + contrib_A = 1 точно). Агрегаты: медианы + bootstrap-CI (per-doc, B_BOOT = 199,
детерминированный сид на (пул, слой): `seed_ci = 9001 + pool_idx·1009 + L`, pool_idx —
index в sorted(pools); CI = перцентиль [2.5, 97.5] bootstrap-медиан). Это primary-метрика
carrier_verdict (§5 канал 2).

## 7. Отчёт (claim: `results/AT0/AT0_2b_RA.json`)

- gates (максимумы closure_rel/add_rel/orth/energy_add_rel по клеткам; n_absent = 2);
- per (L, пул): med fdc_R/fdc_A, frac_R/frac_A, **n_present_R/A (v2 C3/C5)**,
  **n_cells_total/n_cells_present (v2 C5)**, med p_fdc, **dominance_verdict (descriptive)**;
  **carrier_verdict (primary) + входы: med/ci_low contrib_R/A, med share_rad/ang/cross,
  dh_frac_dc/dh_fdc_med/dh_p_med, dc_present_dh per doc (store)** (v2 C1/C2);
  **absent_frac R/A (v2 M3)**;
  med fdc_dh (DC-уровень Δh для контекста; измерен ≈ 1.5e-5, НЕ 0.057);
  **med vt_plateau (v_tangent rho_Tk(9..32) — якорь плато T0.1; v2 M2: DESCRIPTIVE anchor,
  НЕ часть вердикта)**;
  med ACF k=1..16 для R и A; med кросс-лаг k=1..16; med share_rad/share_ang/share_cross;
- store: per-cell полный профиль (rho96 R/A, cross k=1..K, p_fdc, p_fdc_dh, closure,
  N, scale, vt_plateau, **contrib_R/A (v2)**);
- manifest: sha claim/store/скрипта/данных/field.py (формат AT0_2) + **v2 (C7):
  `store_file_sha256`/`claim_file_sha256`/`manifest_file_sha256` = sha256 фактических
  файлов; `store_sha256` = canonical payload hash (canonical JSON без self-поля) —
  конвенция записана в claim.meta**;
- **v2 (C6): `decode_probe_status: ok|skipped`; при skipped — `provenance_limited =
  "content_sanity_skipped"` (claim-уровень; ограничивает content-claims, не R/A-математику).**

## 8. Чего НЕТ в T0.2b (границы)

- Нет blk/AR/Fourier-null'ов (ТЗ требует только DC-вердикт + кросс-лаги отчётом);
- Нет сравнения Σ с real (вырождено G7 — Σ инвариантно к перестановкам);
- Не трогает AT0_1/AT0_2/AT0_3/AT0_5, field.py, данные.

Цена: <5 мин CPU (4 тред) — 1434 клетки × (2 FFT-ACF + 2×96 кросс-einsum + 199×2
перестановки). ТЗ-оценка 20 мин — верх.
