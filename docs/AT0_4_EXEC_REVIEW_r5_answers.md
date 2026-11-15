# AT0_4 EXEC REVIEW r5 — ответы (по пунктам: принято / спорю / самокоррекция)

Ревью: `docs/AT0_4_EXEC_REVIEW.md` (2026-11-14, «ACCEPT as own-doc r/DC pilot, with
wording/consistency fixes»). Ответы — этот файл (протокол серии: отдельный md на раунд).

Артефакты после патча (БЕЗ форвардов, rebuild r5, golden-test):
- store `store/ab8ec0ed2df2f8c7.json` (sha = имя; old store `e852a4bd88c92649` сохранён,
  untracked — удаление в коммите);
- `AT0_4_MIX.json` (claim) + `AT0_4_manifest.json` (цепочка 4/4, см. C6);
- `experiments/AT0_4_mix.py` (fff89220…), `experiments/AT0_4_rebuild_r5.py` (6a2d3729…,
  NOT REUSABLE); лог `results/AT0/AT0_4_rebuild_r5.log`.
- **pretest_results бит-неизменен — ни один вердикт не изменился** (vacuous ×3, same ×3 +
  trunk_carries, descriptive ×2, null ×6, outliers 2/6+6/6); diff store = ровно 8 ключей
  (P1, P2, r_decomposition, doc_order_provenance, verdict.summary, meta.mix_note,
  meta.doc_stratification, provenance).

Итог: **5 пунктов принято и сделано** (C1.1–C1.3, C2, C4, C7 — C7 уже был, цитирую),
**2 пункта приняты с оговоркой** (C3 — спор об «arbitrary»; C5 — моя самокоррекция
чистого набора ревью: own_w7 тоже периодический), **C6 — не код**: файлы в репо не
пустые, верификация ниже.

---

## C1. c0 headline — принято (всё три подпункта)

- **C1.1 (post-hoc)**: принято. `trained_vs_random.status` = «post-hoc, descriptive — НЕ
  pre-registered confirmatory test (r4 §8.3, r5 C1.1)»; summary_line теперь:
  «post-hoc, descriptive — НЕ pre-registered confirmatory test (exec-review r4 §8.3, r5 C1.1),
  без форвардов».
- **C1.2 (pairwise ≠ 48 независимых)**: принято. Новый `pairwise_note`: «48 = 8 доков × 6
  random-инстансов; инстансы делят doc/trunk/seed structure — это НЕ 48 независимых
  наблюдения; inferential sign test — per-doc (n=8) и per-instance (n=8); pairwise —
  описательно». В summary_line: «pairwise 48/48 > 1 (descriptive — НЕ 48 независимых
  наблюдения, см. pairwise_note)».
- **C1.3 (sensitivity)**: принято, отчитано в `trained_vs_random.sensitivity` (независимый
  пересчёт из store.per_doc + G_nd, до 1e-15):

  | срез | n_docs | median ratio [CI95] | >1 | pairwise |
  |---|---:|---|---:|---|
  | all_8 | 8 | 1.4419 [1.2866, 1.6564] | 8/8 | 48/48, med 1.5097 |
  | no_period1_docs (без w0, w4) | 6 | 1.3682 [1.2599, 1.7457] | 6/6 | 36/36, med 1.4753 |
  | gnd_filtered_pairs (per-instance G_nd ok) | 7 (w5 исключён целиком) | 1.4426 [1.2866, 1.5534] | 7/7 | 40/40, med 1.4991 |
  | no_periodic_docs (без w5, w7 — см. C5) | 4 | 1.3061 [1.2332, 1.4108] | 4/4 | 24/24, med 1.3589 |

  Все срезы > 1 — headline «8/8» устойчив: signal живёт на clean-доках (no_periodic 4/4),
  а не только на вырожденных. Частичные пары gnd_filtered: own_w2 5/6 (R2_S13 not-ok),
  own_w7 5/6 (R1_S13 not-ok).

## C2. P2 «trunk_carries» — принято (с технической оговоркой)

- Новый `P2.cell_relative_note` (data-driven): «„trunk_carries“ = trunk_carries_absolute_scale:
  „same“ — вердикт абсолютного порога 0.1, НЕ zero-effect verdict; tied input embedding даёт
  ненулевую относительную модуляцию r: median |Δr|/r = S11 0.1736, S12 0.1627, S13 0.1027».
  `scale_note` дополнен: «„trunk_carries“ читать как trunk_carries_absolute_scale: „same“ —
  вердикт абсолютного порога, НЕ zero-effect — tied input embedding даёт ненулевую
  relative-модуляцию 0.10–0.17».
- **Оговорка (техническая)**: саму строку пре-рег-вердикта `trunk_carries` НЕ переименовываю
  в «trunk_carries_absolute_scale» — пре-рег v3 фиксирован, а правило серии (r3–r5):
  пре-рег-вердикты не пересматриваются, корректируется только чтение (note + summary).
  Значение cell_verdict в pretest_results поэтому не меняется (бит-неизменность — golden-ассерт).
- **Моя поправка к числам ревью**: «~16–18%» — это S11/S12; S13 = 0.1027. Записано
  «0.10–0.17» (измеренный диапазон по инстансам), не «16–18%».

## C3. P1 — бэнд не детектор когерентного дрейфа — принято по существу, СПОРЮ об «arbitrary»

- **Принято и записано** (`P1.band_meaning`): «Бэнд — НЕ детектор когерентного дрейфа
  (synthetic null: drift_smooth_walk r_med≈192 — вне бэнда) и на данных c0~260
  structurally unreachable — P1 здесь: исторический пре-рег, не содержательный тест (r5 C3)».
  Переименовывать в «coherent_drift» и не собираемся (р4-ответка §4 — без изменений).
- **Спорю о слове «arbitrary»**: окно не произвольное — оно очерчивает режим E[r]=1 для
  iid-малых-шагов под гейтом r_max≥1, и таблица null'ов ревью это сама показывает:
  три режима дают три разных поведения (low-motion iid — в бэнде; drift r≈192 — вне;
  iid-сфера r≈1/N — vacuous и по r_max). «Бэнд ≠ детектор дрейфа» и «окно имеет
  генерирующий механизм» — совместимые утверждения: механизм E[r]=1 не делает бэнд
  детектором дрейфа, потому что у дрейфа r≈N. Формулировка «arbitrary ratio window»
  в артефакт не внесена; внесён совместимый с ревью вывод (не детектор дрейфа,
  исторический пре-рег).

## C4. own_w5 (и own_w7) — принято

`doc_order_provenance` теперь несёт per-doc `note` (разделение token_periodicity /
order_sensitivity — как просили):
- **own_w5**: «token periodicity (period-13, n_uniq=12) present, но absolute order-флаг
  absent — пары не live: не входят в per_doc ни в одном live-инстансе (G_nd not-ok во всех
  random-инстансах); детектор непротиворечив». Факт: G_nd not-ok для w5 во ВСЕХ 6
  random-инстансах (R1/R2 × S11/S12/S13) — max_abs_diff_r в store = None.
- **own_w7** (см. C5 — ревью его не заметило): «token periodicity (period-16, n_uniq=21)
  present, но absolute order-флаг absent — |Δr| ниже абсолютного порога 0.1 во всех
  live-парах (макс 0.0017); детектор непротиворечив».
- `interp` для w0/w4: «single-token mass ≥93%» → по max_share: w0 ≈93% (0.9297), w4 ≈94%
  (0.9401).
- Добавлен `note_key`: «order_flag ≠ token-периодичность: периодический док может не иметь
  order-флага (не live / ниже порога), и наоборот».

## C5. Стратификация — принято + САМОКОРРЕКЦИЯ чистого набора ревью

`meta.doc_stratification` (по token_provenance.min_period_90pct + c0):
- period_1_low_motion: **own_w0, own_w4** (c0 23.4 / 19.9);
- periodic_non_low_motion: **own_w5 (period-13), own_w7 (period-16)**;
- clean_no_periodic: **own_w1, own_w2, own_w3, own_w6**.

**Самокоррекция ревью (важно)**: в stratum-таблице ревью clean = {w1, w2, w3, w6, **w7**}.
Это неверно: `token_provenance.own_w7` = min_period_90pct **16** (rep16 ≥ 0.9), n_uniq=21,
max_share 0.065 — w7 тоже периодический (не low-motion, но период-16). И G_nd-факты:
S13 — own_w7 not-ok (R1_S13), own_w2 not-ok (R2_S13). Поэтому честная стратификация
4/2/2, а не 5/2/1; c0-sensitivity это учитывает — строже всех срез no_periodic
(только w1/w2/w3/w6): **4/4 > 1, median 1.3061** — сигнал не теряет значимости даже
на самом чистом наборе.

## C6. Hash-chain / rebuild r4 «не видны» — файлы в репо НЕ пустые (транспорт)

В контексте ревью пришли пустыми — это не состояние репо. Факты (проверено sha256 на
диск-копии; текущие артефакты rebuild r5):

| файл | размер | sha256 (16 hex) |
|---|---:|---|
| results/AT0/AT0_4_manifest.json | 576 B | c4a7078432be41d2 |
| results/AT0/AT0_4_MIX.json (claim) | 147 504 B | 41de97c209f69353 |
| experiments/AT0_4_rebuild_r4.py | 27 288 B | e2820c876c4e4e32 |

(размеры/sha — на момент верификации; после r5 скрипт rebuild r4 удалён — в репо
только актуальный AT0_4_rebuild_r5.py)
| experiments/AT0_4_rebuild_r5.py | 29 326 B | 6a2d37291974b73e |
| experiments/AT0_4_mix.py | 82 352 B | fff89220f202b905 |
| results/AT0/store/ab8ec0ed2df2f8c7.json | — | ab8ec0ed2df2f8c7 (первое имя = префикс sha) |

Верификация цепочки (rebuild r5, все ассерты зелёные, лог `AT0_4_rebuild_r5.log`):
1. `sha256(claim file) == manifest.claim_sha256`; `sha256(store file) == store-имя ==
   manifest.store_sha256 == claim.store.sha256` — 4/4;
2. rebuild r5 — golden-test на pinned store e852a4bd… (full sha в скрипте): P4/P3/P5
   БИТ-идентичны (их build-функции r5 не тронуты), P1 vacuous ×3 + per_doc/c0_stats
   бит-идентичны, P2 same ×3 + per_doc бит-идентичны, synthetic_null бит-идентичен
   (seed 117, B=200), c0 per_doc/per_instance — независимый пересчёт из store.per_doc
   (до 1e-15), sensitivity — независимый пересчёт из store.per_doc + G_nd;
3. pretest_results бит-неизменен; diff = ровно 8 ожидаемых ключей;
4. script_diff_chain (provenance.aggregation_rebuild_r5): f59035c0… (модуль, store.meta)
   → fff89220… (r5-фиксы) → 6a2d3729… (этот rebuild); более ранние звенья — в
   aggregation_rebuild_r4/r3.

## C7. Mix permutation note — объявлено (уже было в meta, цитирую)

`meta.mix_note`: «mix = одна фиксированная перестановка на док (seed 7) — **mix treatment
fixed; no permutation-seed sensitivity** (новых форвардов нет); эффект mix может зависеть от
конкретной перестановки (r3 C6.4, r5 C7)». Строка была в store ещё в r4 (meta в claim) —
в контексте ревью не видна (пустые файлы, C6). Усилен wording под запрос.

---

## Что НЕ изменено (границы)

- Ни один вердикт pretest_results; per_doc / gates / profiles / token_provenance —
  бит-неизменны; forward-окно 12.11 05:19–13:52 (112 клеток) не тронуто — rebuild data-only;
- docs/AT0_4_prereg.md v3 — не правится; пре-рег-строки вердиктов — фикс (C2-оговорка);
- T0.4 corpus/profile — остаётся open (согласно вердикту ревью).

## Согласие с итогом

AT0_4 закрыт как **own-doc r/DC pilot**; главный результат — c0 trained/random, а не r:
headline 1.4419 [1.287, 1.656], 8/8, и устойчив к любой стратификации (7/7 G_nd, 6/6
no-period1, 4/4 no-periodic). P1 — vacuous/исторический пре-рег, P2 — абсолютный масштаб
+ relative-модуляция 0.10–0.17, P4 — null по медиане с doc-исключениями на period-1
low-motion. Corpus-вопрос «кто несёт профиль» — open.
