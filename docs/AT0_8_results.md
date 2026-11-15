# AT0_8: результаты полного прогона (round 1)

Дата: 2026-11-16. Карриер: Qwen3.5-0.8B (tie_word_embeddings). Кернел:
`voroninmichael/prism-at08-threshold` v2 (T4, GPU). Пререгистрация:
`docs/AT0_8_prereg.md` (неизменяемо). Ревью: `docs/AT0_8_review_r1.md`.

## Прогон

- v1 упала в бисекции (TypeError в `np.mean` по dict_values) — фикс:
  `_bisect_lambda` (векторный mean по (сид,слою)), v2.
- v2: **gates S1–S9 ALL PASS, wall 841.0 s**. Т: 256 токенов, 3 сида ×
  4 домена × 32 доков, λ-грид 0.55…1.0, слои 11/12/13 (линейные).
- Артефакты: `results/AT0_8_THRESHOLD.json` (claim, file sha
  `b2923d74324e9180`), `results/AT0_8_manifest.json` (self-sha
  `74571c823aa5aa8b`), `results/store/018c3dac90eb1bba.json`
  (canonical sha `018c3dac90eb1bba`), `results/AT0_8_run.log` (tracked).
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
