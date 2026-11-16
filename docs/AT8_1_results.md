# AT8_1: результаты полного прогона (round 1)

> 16.11: эксперимент переименован AT0_8 → AT8_1 (имя AT0_8 освобождено под T0.8); артефакты
> — results/AT8/, логи — logs/AT8/ (содержимое и sha не тронуты).

Дата: 2026-11-16. Карриер: Qwen3.5-0.8B (tie_word_embeddings). Кернел:
`voroninmichael/prism-at08-threshold` v2 (T4, GPU). Пререгистрация:
`docs/AT8_1_prereg.md` (неизменяемо). Ревью: `docs/AT8_1_review_r1.md`.

## Прогон

- v1 упала в бисекции (TypeError в `np.mean` по dict_values) — фикс:
  `_bisect_lambda` (векторный mean по (сид,слою)), v2.
- v2: **gates S1–S9 ALL PASS, wall 841.0 s**. Т: 256 токенов, 3 сида ×
  4 домена × 32 доков, λ-грид 0.55…1.0, слои 11/12/13 (линейные).
- Артефакты: `results/AT8/AT8_1_THRESHOLD.json` (claim, file sha
  `b2923d74324e9180`), `results/AT8/AT8_1_manifest.json` (self-sha
  `74571c823aa5aa8b`), `results/AT8/store/018c3dac90eb1bba.json`
  (canonical sha `018c3dac90eb1bba`), `logs/AT8/AT8_1_run.log` (tracked).
  Все sha перепроверены локально по конвенции AT0_2b-v2 C7 (canonical
  = json.dumps sort_keys, self-поле = None).

## Вердикт

**V0_NOT_RESOLVED** — основной observable `dip` не дал resolved λ_c:
seed-avg кривая не монотонна (u: +1.71 при λ=0.55 → −1.74 при 0.70 →
−1.13 при 0.85 → +1.0 при 1.0) и исключена правилом non-monotone-excluded
пререга. Защита пререга сработала по назначению.

Вторичные observables (не являются вердиктом, пререг §4):

| observable | outcome | λ_c (L11/12/13) | bisect |
|---|---|---|---|
| nll_holdout | resolved | 0.85 / 0.85 / 0.85 | [0.75, 0.85] |
| margin | resolved | 0.85 / — / 0.85 | [0.75, 0.85] |

dip/hump/plateau/rep16/loopy/pr_sigma/eff_rank — non_monotone_excluded.

## Сравнение со смоком (локальный CPU, 1 сид, T=64)

Смок дал V2_FLAT, λ_c(dip) ≈ 0.92. На полном прогоне (3 сида, T=256,
GPU) dip не резолвится: dip-минимум существует (отрицательный u на
0.65–0.85), но при λ=0.55–0.60 u положительный — форма «горб-долина»,
не ступенька. Т.е. сигнал 1-сидного смока не подтвердился по dip;
nll_holdout и margin сходятся в λ_c ≈ 0.85 (интервал [0.75, 0.85]),
что согласуется с λ_c ≈ 0.85–0.92 смока по порядку величин.

## Что НЕ сделано (по пререгу)

- Round 2 (если нужен): сужение грида до [0.6, 0.9] и 5 сидов — только
  при явном задании; сейчас вердикт V0 закрыт, round 2 не пререгистрирован.
- Никаких правок пререга/скрипта под результат — фикс v1 касался только
  бага (тип данных в бисекции), не логики.

## Машина v2 (T8.1-r2, 16.11) — disk-only реанализ

Внешнее ревью полного прогона: `docs/AT8_1_review.md` (гейты/провенанс **ACCEPT**,
машина v1 — **REVISE**; разбор F1–F6 с числами — раздел 11, r2-exec). Пре-рег машины v2:
`docs/TZ_add_4-8.md` §T8.1. Скрипт: `experiments/at8/AT8_1_r2_reanalysis.py` (без RNG, без
перепрогона; вход — committed store, sha-верификация по manifest).

**Вердикт v2: V2_SOLITON_CANDIDATE** (coincides с pre-reg ожиданием, match=True):
- полевой событияный эстиманд dip = интервал max-скорости **[0.60, 0.65]** (Δu = −2.099);
  hump/plateau — тот же интервал ([0.60, 0.65], Δu = +0.910 / +0.652);
- отделён от λ_c компетентности (nll_holdout ∧ margin, resolved) = **0.85** на 3 шага сетки;
- V3 НЕ сработал: max(u_dip) = 1.708 > 1 (cond1 ✓), НО hinge(0.55) = 0.0340 <
  hinge(0.00) = 0.0480 (cond2 ✗, seed-avg) — патология не подтверждена;
- V1 структурно недостижим (dip немонотонен → λ_c^dip нет);
- families 3/3 `intra_consistent=false` — нет единого координированного порога;
- sensitivity F6: dip-сигнатура жива на не-луповом {wiki_2, wiki_3} (u: пик +1.844@0.55,
  провал −3.056@0.7, max-скорость [0.60, 0.65]) — составный артефакт луповых доков отклонён;
- r2-exec-опровержение F3b: raw rep16(1.0) = 0.0 (фразовый цикл ≈ 20 токенов > 16-окна),
  sample λ=1 — «The program is designed to help you master the art of making delicious BBQ.»,
  цитата «ame ame» из ревью не из AT8_1.

Две события (в данных с v1-прогона, впервые выражены машиной): поле реорганизуется на
λ ≈ 0.60–0.65; компетентность растёт к λ = 0.85; mode (loopy) включается на 0.85–1.0.

Артефакты (на диске, в git — results/AT8/): `results/AT8/AT8_1_THRESHOLD_v2.json`
(claim, file sha `bb507dac4d696981`), `results/AT8/AT8_1_v2_manifest.json` (self-sha `548eec26f37a5d19`), `logs/AT8/AT8_1_v2_run.log` (tracked; маркер
`AT0_8_R2_DONE verdict=V2_SOLITON_CANDIDATE match=True`).

> По ревью AT8_2 (16.11): в тексте выше «T8.2 (реальная траектория обучения)» — на самом
> деле T8.4 (отложено); T8.2 = оконное восстановление + доза (AT8_2).

## Машина v3 (T8.1-r3, 16.11) — disk-only реанализ (раунд 2 ревью)

Внешнее ревью v2-машины + черновика T8.2: `docs/AT8_1_review.md` (раунд 2, 197 строк;
вердикт: «дисциплина v2 — ACCEPT; вердикт V2_SOLITON_CANDIDATE — REVISE; ТЗ 8.2 — REVISE
по существу»). Пре-рег: `docs/TZ_add_4-8.md` «Машина v3 (T8.1-r3)»; TZ-ревью:
`docs/dynamic/AT8_1_r3_TZ_review.md` (PASS, 0 MAJOR, 5 MINOR + 6 NIT — применены до
запуска); скрипт: `experiments/at8/AT8_1_r3_reanalysis.py` (без RNG, без перепрогона;
тот же committed store, sha-верификация по manifest).

**Вердикт v3: V3_PATHOLOGY** (coincides с pre-reg ожиданием, match=True):
- **V3 (знаковый детектор + корроборация):** cond1: min dip_seedavg = −0.3052@0.70 <
  dip(1.0) = −0.1738; корроборация в λ*-клетке: **C2 (eff_rank): 6.39 < min(17.56, 13.20)** —
  rank-collapse ниже обоих концов (истинно для всех 3 сида); C1 (loopy) = FALSE (0.0),
  C3 (nll_holdout) = FALSE (9.75 < max(12.76, 2.75)). max_u_dip = 1.708@0.55 —
  описательный (не детектор — O2-алгебра: старое hinge-условие требовало u > 2 или u < 0
  и промахивало эталонную E-71-подпись u≈1.19);
- **V2 — not_resolved (расхождение локаторов):** L1 (u=0.5-crossing): dip [0.20,0.40] vs
  nll/margin [0.75,0.85] — gap 6, отделено; L2 (onset укручения > 2× медианы): dip
  [0.60,0.65] vs nll/margin [0.60,0.65] — **gap 0, тот же интервал, НЕ отделено**;
  L3 (max-скорость): dip [0.60,0.65] vs nll/margin [0.85,1.00] — gap 3, отделено.
  Честная формулировка: резкое полевое событие в [0.60,0.65] на фоне широкого подъёма
  компетентности 0.6→1.0 (u=0.5 на 0.85);
- **per-seed (F2):** L3-клетки dip: s11 [0.60,0.65], s12 [0.65,0.70], s13 [0.60,0.65];
  span = 1 шаг (конвенция: (max−min) индексов клеток) — правило «span > 1 ⇒ unresolved»
  не срабатывает; per-seed cond1: TRUE×3, λ*: 0.70/0.70/0.65, C2: TRUE×3;
- **mode → vacuous (F3):** предпосылка ревью «rep16(0) = 0» опровергнута данными
  (raw(0) = 0.8824, raw(1) = 0.0, u определена); вывод удержан через no_transition
  (F4-правило: переход уже при λ=0 — baseline-луп) — disclosed отклонение от буквального
  патча; families: field = inconsistent, leaf = inconsistent, mode = vacuous
  (2/2 измеримых — inconsistent);
- **S9 (F4):** декларация v2 перенесена (один раунд, ширина 0.05 = ½ зазора сетки,
  «≤ 0.02» недостижимо — декларатив, не гейт; capped = false);
- **raw-отчёт (F5):** dip(0) = −0.22183, dip(1) = −0.17382, **Δ = +0.04802 (знак явный)**,
  min dip = −0.30524@0.70, max dip = −0.13982@0.55; полные per-seed dip-кривые (3×11)
  в артефакте.

**Чтение:** на λ-прямой в λ* = 0.70 — патологический overshoot (dip глубже обученного
конца) с rank-collapse — подпись интерполяционной патологии, не упругого события.
Вердикт v2 (V2_SOLITON_CANDIDATE) остаётся артефактом v2-машины (не переписывается).
Т8.2-половина ревью (A1–A8) — point-by-point в `docs/AT8_1_review_r3_answer.md`
(AT8_2: ELASTIC = релаксация к базовой линии, не солитонная упругость; T8.2-r2
декларирован).

Артефакты: `results/AT8/AT8_1_THRESHOLD_v3.json` (claim, file sha `c4fcb1ef5ad0d22f`),
`results/AT8/AT8_1_v3_manifest.json`, `results/AT8/AT8_1_v3_run.log` (маркер
`AT8_1_R3_DONE verdict=V3_PATHOLOGY match=True`, wall 0.0 s, RUN_RC=0). Selfcheck:
пересчёт dip/eff_rank/nll_holdout из store == артефакту (dev 0.0).
