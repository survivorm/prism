# AT0_2 — пре-рег T0.2 «Форма профиля при фиксированной сумме» (round 1)

Статус: прогнан ревью-лупом ДО полного прогона (docs/AT0_2_REVIEW.md, раунд 1,
fixes N2/N4/N5; субагент-ревью + само-ревью исполнителя). Текст = блок пре-рега
в хедере `experiments/AT0_2_shape.py` (единственный источник правды; расхождение
хедер↔код — нарушение, см. урок N4).

Скрипт: experiments/AT0_2_shape.py (sha256 зафиксирован в store/claim после прогона).
ТЗ: docs/TZ_add_0-3.md §1.5 T0.2 (v1.1).

## Зафиксировано ДО прогона

- Объекты: h_raw (Δh), du_unit (Δu), v_tangent (v̂, frame u_t) × 96 доков
  (64 gen + 16 P1_own + 16 P2_corpus) × L{1,5,11,17,23}.
  L1-TPU degraded — absent: ровно own_w0, own_w4 (как AT0_1 excluded_L1).
  Клеток: 1440 всего (1434 обычных + 6 absent = 2 дока × 3 объекта);
  G2 — set-равенство всех 1440 ключей + число absent = 6.

- Полосы — буква ТЗ: {1},{2},{3..8},{9..32},{33..96}. Хвост {97..N−1} НЕ входит в
  вектор профиля (объявленная K=96-обрезка, урок AT0_1); отчитан отдельным скаляром
  tail_share на каждом realization (real и null).

- Нормировка: (r−1)/2 = sum_rho_full — ПОЛНОЕ суммирование всех лагов (G1). Для
  permutation-null'ов (iid/blk/fourier) сумма инвариантна ТОЧНО (G7) ⇒ тот же
  знаменатель, что у real (гейт: |r_null − r_real| ≤ 1e-9·max(1,r_real)).
  Для AR-null знаменатель — собственная сумма суррогата (ТЗ: «Σ отчитывать отдельно»);
  r_AR, sum_rho_AR — в артефакте, с real НЕ СРАВНИВАЮТСЯ.

- Расстояние: L1 по 5-компонентному вектору долей (основное), L2 — отчёт.
  D_center = ‖P_real − C‖, C = mean_s(P_s); D'_s = ‖P_s − C‖;
  p_doc = (#{s: D'_s ≥ D_center} + 1)/(n_valid + 1).

- Сиды: B = 20 на каждый (iid/blk/AR), отдельный сид на (док, s):
  seed = 2026 + doc_idx·1009 + s. Fourier: B = 3 (positive control, не null).

- Null'ы (ТЗ):
  * iid-перестановка строк — null полной независимости (маргинал; Σ — автоматически).
  * blk-shuffle БЕЗ seam-exclusion, b ∈ {2,4,8,16,32} — null локальности (локальная
    структура ≤ b сохраняется).
  * AR(p), p ∈ {1,2,4} — null «линейный процесс»: per-dim Yule-Walker на demeaned
    приращениях (согласовано по ρ̄(1) построением), симуляция, burn-in 256.
  * Fourier phase-randomization (per dim, zero-padded n_fft ≥ 2N−1, DC-бин фиксирован) —
    POSITIVE CONTROL: сохраняет спектр ⇒ обязан воспроизвести круговой ACF (спектральный
    объект) до 1e-6 (k=1..min(96,N−1)) и r по полному ряду точно (DC+Parseval).
    Линейный (обрезанный до N) ACF первых N сэмплов отличается на wrap-around хвостовой
    член — свойство ANY spectrum-preserving суррогата; отчитывается отдельно, не null.
    Не null; в вердикты не входит.

- Seam-excluded blk: ТОЛЬКО отдельный оценщик — within-block ACF (k < b), seed-средний
  профиль по доступным k, нормировка на собственную сумму; Σ с real НЕ СРАВНИВАТЬ (ТЗ).

## Вердикт (совпадает с кодом `aggregate()` — N4: старая pooled-rank формула p_agg удалена,
в коде её не было)

Hypothesis H = «форма профиля объясняется null».
D_real = per-doc mean по L (кластер = док); CI — cluster-бутстрап по ДОКАМ
(B=1000, seed=117); D_null — аналогично per-doc mean(Dnull_l1_mean).
DELTA_MIN (§0.4): если D_real < пол-ширины его CI (ниже разрешения) → not_testable
(проверяется ДО интерпретации).
D_real > CI_null верх → refuted (форма выходит за null);
D_real < CI_null низ → confirmed_anomaly (real ближе к центру, чем типичный null);
D_real ∈ CI_null → confirmed.
Доля клеток frac_cells_p_lt_alpha (p_doc < 0.05) + n_docs_unique_p_lt_alpha +
Wilson-CI — описательные, в вердикт не входят.

## G-гейты

- G0: sha 3 npz + gen_sha 64/64 + декод-проба 3 строк.
- G1′ (до клеток): field.py sha + _selfcheck(); FFT-ACF == полагам-einsum (1e-9);
  batched == построчный; band_sums_direct (cumsum) == полагам-einsum (1e-9);
  r два способами (‖Σy‖²/c0 == 1+2Σρ̂ FFT-full, 1e-9); c0 два способами.
- G1: r(‖Σx‖²) == 1+2Σρ̂ asserted в каждой real-клетке (1e-12, fail-loud).
- G2: полнота (1440 ключей, set-равенство, включая 6 absent; число absent = n_docs_absent × 3).
- G7: hash-store + claim + manifest + run.log; G7-invariance: |r_null − r_real| ≤ 1e-9·max(1, r_real)
  по permutation-null'ам (гейт).
- G8: gram_resid < 1e-10 на real-клетках.
- Positive control (ДО интерпретации null'ов): круговой ACF ≤ 1e-6 и r_full ≤ 2e-4;
  сбой → результат null'ов не интерпретируем.
- AR σ² ≤ 0 → clip + счётчик (σ² < −1e-8·r0 → fail-loud ValueError);
  |sum_rho_full| < 1e-6 → клетка absent-degenerate.
- Домены: доменное утверждение только при n_nondeg ≥ 12 (гейт).
- Smoke: `--smoke` (2 дока × 5 L × 3 объекта, B=3) — ТОЛЬКО stdout, артефакты НЕ пишет
  (N2: smoke не затирает production claim/manifest/run.log).

## Артефакты

results/AT0/store/<sha16>.json (full), results/AT0/AT0_2_SHAPE.json (claim),
results/AT0/AT0_2_manifest.json, results/AT0/AT0_2_run.log (tracked).

## Round 2 (ревью r6, docs/AT0_2_REVIEW_r6_review.md → ответы docs/AT0_2_REVIEW_r6.md, 14.11)

Источники правды: хедер `experiments/AT0_2_shape.py` (обновлён) + ответка. Отклонения
round 1 → round 2 (все — по ревью r6):

- **E3**: B = 20 → **199** (p_floor = 1/200 = 0.005; стандарт T0.1 NNULL=199). Сиды 0..19
  репродуктивны из round 1 (гейт: per-seed r_null/DtoN + real — бит-в-бит, проверено на 4
  клетках; per-seed Dnull НЕ гейт — зависит от B через C = mean по B сидов).
- **E4/M4/M5**: вердикт — **paired** delta_d = D_real(doc) − D_null(doc), cluster-бутстрап
  по докам; **DELTA_MIN = 0.05 pre-registered** (порядок эффекта T0.1 по полосам);
  |Δmean| < floor → not_testable; Δ ≥ floor и CI_Δ > 0 → refuted; Δ ≤ −floor и CI_Δ < 0 →
  confirmed_anomaly (флаг «более типичен, чем типично»); иначе inconclusive.
  Separate-CI — вторичное.
- **E1**: estimand объявлен явно: T0.2 = форма при фиксированной сумме; тест «Σ против
  полосы» ВЫРОЖДЕН (ТЗ v1.2 G7: Σx, c0 инвариантны к перестановкам; численный чек
  sum_invariance_max_dev). AR — Σ своё, mismatch отчитан, с real не сравнивается.
- **E2**: blk — **damped/seamed null** (w(k,b) = max(0,1−k/b)), НЕ «локальность ≤ b»;
  locality-прибор — P3-ladder. Identity-check: within-block(blk-null) ≡ within-block(real)
  (fail-loud, G-seam ≤ 1e-12) ⇒ seam-excluded within-block не может быть null'ом (D ≡ 0).
  **Самокоррекция**: пойман latent-баг round 1 в null_blk_seam_excl (фиксированная сетка
  y[i·b:(i+1)·b] сдвигала блоки после короткого N%b≠0 — seam-prof b=4/8/16/32 rounds 1–5
  слегка загрязнены; фикс — границы по пермутированным блокам).
- **E6/E7**: primary pool = **verified ∩ non-degen (37 доков)**; sensitivity: 80/96;
  corpus — provenance_limited (G16); degen (loopy) — вне primary (AT0_3: loopy несут
  периодическую лаговую структуру).
- **E5**: AR diagnostics: ar_max_root, n_nonstationary, burn_mix_ratio, sigma2_clip_frac,
  rho1..4 real vs null, sum_mismatch; wording AR1/AR2 ≈ тавтология, AR4 — conditional.
- **E8**: object provenance: gen h_raw/v_tangent — provenance_limited (per-cell флаг);
  du_unit — hn-независим (primary на gen); state — верифицирован конструкцией.
- **M1**: Wilson clamp [0,1]. **M2**: rename tested_D_center_l1/l2, descriptive_D_bands_L1.
  **M3**: primary family = 9 тестов (pool × 3 objects × {iid, blk32, ar4}), FWER не
  применяется — объявлено.

Цена round 2: ~35 с/клетку (бенч 14.11) → ~10–14 ч CPU (round 1: ~108 мин при B=20).
