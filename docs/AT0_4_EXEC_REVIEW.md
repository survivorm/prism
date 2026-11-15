Принял r4. В контексте виден в основном `AT0_4_MIX.json`; `AT0_4_manifest.json`, `AT0_4_rebuild_r4.py`, `AT0_4_EXEC_REVIEW_r4_answers.md`, `AT0_4_mix.py` пришли пустыми. Поэтому **код-фиксы и sha-цепочку я не верифицирую**, но артефакта достаточно, чтобы ревьюить вердикты и числа.

Короткий вердикт:

- **r4 закрывает большинство прежних дыр AT0_4**: scope, P4 doc-flags, P3/P5 inference, P1 diagnostics, P2 tied-note, `gnd_filter`, относительные дельты, `endpoint_cos`, c0 trained/random.
- **самый сильный новый результат — c0 trained/random**, а не `r`: trained path energy систематически выше при том же порядке endpoint-косинусов.
- **оставшиеся проблемы — в основном wording/consistency**, не арифметика: P2 “trunk_carries” слишком силён относительно relative-дельт; P1 synthetic null показывает, что бэнд не “coherent drift”; c0 headline post-hoc и должен иметь G_nd/clean sensitivity; own_w5 периодичен, но не order-flagged.
- **AT0_4 можно закрывать как own-doc r/DC pilot**, но не как T0.4 corpus/profile.

---

# 1. Что r4 закрыл

| прежний пункт | состояние в артефакте |
|---|---|
| C1 scope: own-doc, не corpus/profile | summary явно: own-doc r/DC carrier pilot; ρ̄(k) описательно; corpus T0.4 open |
| C2 P4 median masks doc-level | `doc_order_provenance`: own_w4 6/6 systematic, own_w0 2/6 partial; связаны с period-1 loop / low c0 |
| C3 P3/P5 “confirmed” без inference | `descriptive`, sign test p=0.2891, CI, Δmin, n=8 power note |
| C4 P1 diagnostics | c0_stats, reachable_frac=0, low_motion_docs, synthetic_null, reachability_note |
| C5 P2 head/embed naming | tied_note: hidden-state r — input-эмбеддинг, не head |
| C6 gnd per-doc not used | `gnd_filter: true` в P1/P2/P4; n_docs падает до 7/6 |
| относительные дельты | `median_abs_diff_rel`, `scale_note` в P4 |
| r decomposition | `endpoint_cos` добавлен; r/r_max = (1−cos)/2 проверяется |
| c0 trained/random | новый post-hoc блок: median ratio 1.4419, CI [1.2866, 1.6564], 8/8, pairwise 48/48 |

Это хорошо.

---

# 2. Главный новый результат: c0, не r

Артефакт:

```text
c0 trained/random median ratio = 1.4419
CI95 [1.2866, 1.6564]
n=8 paired, 8/8 > 1
pairwise 48/48 > 1, median 1.5097
post-hoc, exec-review r4 §8.3, без форвардов
```

Это сильнее, чем P1–P5 по `r`.

Почему:

\[
r=\frac{\|u_N-u_0\|^2}{c_0}
=\frac{2(1-\cos(u_0,u_N))}{c_0}.
\]

`endpoint_cos` в артефакте показывает, что на clean-доках trained/random endpoint-косинусы одного порядка, а `c0` систематически выше у trained.

Примеры:

| doc | c0 trained | c0 random median | ratio |
|---|---:|---:|---:|
| own_w1 | 324.15 | 251.95 | 1.287 |
| own_w2 | 324.13 | 229.75 | 1.411 |
| own_w3 | 321.16 | 260.42 | 1.233 |
| own_w5 | 391.05 | 193.76 | 2.018 |
| own_w6 | 363.19 | 273.98 | 1.326 |
| own_w0 | 23.37 | 14.11 | 1.656 |
| own_w4 | 19.90 | 12.81 | 1.553 |

То есть trained-траектория имеет больше суммарной энергии шагов при сопоставимом net displacement. Это можно интерпретировать как “trained path more wiggly / more step energy”, но **не как DC-profile carrier**.

Важно: артефакт правильно пишет `post-hoc`. Это не пре-рег-подтверждение. Это сильный описательный результат, который надо вынести в headline pilot’а.

---

# 3. Оставшиеся проблемы

## C1. c0 headline: post-hoc + G_nd consistency + pairwise independence

### C1.1. Post-hoc статус должен остаться

Блок честно помечен:

```text
post-hoc (exec-review r4 §8.3), без форвардов
```

Это правильно. Но если c0 станет главным результатом, нельзя подавать его как pre-registered confirmatory test.

Формулировка:

> Post-hoc paired c0 comparison: trained step energy exceeds random in 8/8 docs; descriptive, not pre-registered.

### C1.2. Pairwise 48/48 — не 48 независимых наблюдений

`pairwise 48/48 > 1` — это 8 доков × 6 random instances. Инстансы делят doc и trunk/seed structure. Их нельзя использовать как независимый sign test на 48.

Артефакт, кажется, это понимает:

```text
per-instance — sign test
```

Хорошо. Но в summary надо явно:

```text
pairwise 48/48 descriptive; inferential sign test is per-doc / per-instance, not 48 independent trials.
```

### C1.3. c0 uses all docs, while P1/P2/P4 use gnd_filter

В P1/P2/P4:

```text
gnd_filter: true
n_docs = 7 или 6
```

В c0 trained/random:

```text
n=8 paired
```

При этом G_nd per-instance имеет `ok=false` для own_w5 / own_w2 в некоторых инстансах.

Это не обязательно баг: c0 может быть валиден даже если cv_du low. Но consistency требует sensitivity:

```text
c0 ratio all 8
c0 ratio clean-only (exclude own_w0/own_w4 period-1 loops)
c0 ratio G_nd-filtered pairs
```

По видимым числам clean-only тоже должен быть >1, но это надо отчитать.

Иначе headline “8/8” включает вырожденные low-motion доки, а P-тесты их частично исключают.

---

## C2. P2 “trunk_carries” слишком силён на фоне relative-дельт

P2 S11:

```text
median_abs_diff_r = 0.001299
median_abs_diff_rel = 0.17365
verdict = same
threshold = 0.1
```

S12:

```text
median_abs_diff_rel = 0.16268
```

То есть tied input embedding меняет `r` на **~16–18% относительно**, но абсолютная дельта мала из-за `r ∝ 1/c0`.

Текущий verdict:

```text
trunk_carries
```

можно читать как “embed не влияет”. Это неверно.

Правильная формулировка:

> Trunk dominates the absolute r scale; trained tied input embedding produces a nonzero relative modulation (~16–18% median). “same” is absolute-threshold verdict, not zero-effect verdict.

P4 scale_note уже делает это для order. P2 тоже нужен такой note.

---

## C3. P1 synthetic null показывает, что бэнд не “coherent drift”

Synthetic null:

```text
iid_sphere:
  c0 median 765.96
  r median 0.002592
  band_frac 0
  reachable_frac 0

iid_lowmotion_c0_2:
  c0 median 2.0
  r median 1.0017
  band_frac 1.0
  reachable_frac 1.0

iid_lowmotion_c0_4:
  c0 median 4.0
  r median 0.9955
  band_frac 1.0
  reachable_frac 0.585

drift_smooth_walk:
  c0 median 0.00781
  r median 191.64
  band_frac 0
  reachable_frac 1.0
```

Это важно.

Бэнд `[0.75, 1.5]` **не является детектором когерентного дрейфа**. Smooth drift даёт `r ≈ 192`, потому что `c0` очень мал, а endpoint displacement конечен.

Бэнд ловит某种 low-motion constructions с `endpoint² ≈ c0`, но не generic drift.

Следствие:

- P1 на данных vacuous — правильно.
- Но не надо переименовывать P1 в “coherent_drift” и не надо интерпретировать бэнд как естественную гипотезу.
- Лучше:

```text
P1_r_band_vacuous:
  r ∈ [0.75,1.5] is an arbitrary ratio window;
  structurally unreachable on data c0 ~ 260;
  synthetic null shows it is not a generic drift detector.
```

P1 можно оставить как исторический пре-рег, но не как содержательный тест.

---

## C4. own_w5: period-13 есть, order_flag absent

`token_provenance` / `doc_order_provenance`:

```text
own_w5:
  min_period_90pct = 13
  n_uniq = 12
  max_share = 0.1536
  order_flag = absent
  max_abs_diff_r = null
```

Это не обязательно ошибка, но требует объяснения.

own_w5 имеет token periodicity, но не low-motion period-1 loop. Возможно, он исключён из P4 live pairs из-за G_nd или относительная дельта мала.

Надо разделить:

```text
token_periodicity: own_w5 period-13
order_sensitivity_under_absolute_thr: not flagged / not live
```

Иначе читатель увидит `min_period_90pct=13` и `order_flag=absent` и решит, что детектор противоречив.

Также own_w0 interp говорит “single-token mass ≥93%”, а `max_share=0.9296875` — чуть ниже 93%. Мелочь, но лучше “~93%”.

---

## C5. G_nd не фильтрует token-degenerate low-motion доки

G_nd ok=true для own_w0/own_w4 во многих инстансах, потому что cv_du высокий / d_eff низкий, но это не “невырожденный вход”. Token provenance показывает period-1 loop.

Это нормально для pilot scope, но headline должен стратифицировать:

```text
clean docs: own_w1, own_w2, own_w3, own_w6, own_w7
period-1 low-motion: own_w0, own_w4
periodic non-low-motion: own_w5
```

Сейчас summary говорит “все flags — вырожденные low-motion”, это верно для P4 flags. Но c0 headline включает все. Нужна clean-only sensitivity.

---

## C6. Hash-chain / rebuild r4 не видны

`AT0_4_manifest.json` и `AT0_4_rebuild_r4.py` пустые в контексте.

Артефакт даёт:

```text
store.path = store/e852a4bd88c92649.json
store.sha256 = e852a4bd88c9264954a294f7a194825c85ab34a95dceb3c1fd0f6402e6e4a877
```

Но я не могу проверить:

```text
manifest.claim_sha256
manifest.store_sha256
rebuild r4 asserts
script_sha256
```

Если r4 делал агрегационный rebuild, нужен тот же паттерн, что r3:

- bit-identity старых чисел;
- expected diff list;
- independent recompute новых полей;
- sha chain;
- old store sha in provenance.

Пока это не видно.

---

## C7. Mix permutation sensitivity не видна

В артефакте есть paired real/mix, но не видно:

```text
multiple mix permutations
shuffle seed sensitivity
```

Если осталась одна перестановка на док, надо явно:

```text
mix treatment fixed; no permutation-seed sensitivity.
```

Это не флипает verdict, но ограничивает P3/P4.

---

# 4. Что можно публиковать уже сейчас

## 4.1. c0 trained/random

Можно:

> Post-hoc paired comparison on own_w0..7: trained total step energy c0 exceeds random median by 1.44× [CI95 1.29, 1.66], 8/8 docs; pairwise 48/48 descriptive. Endpoint cosine does not show a comparable systematic shift; r variation is largely c0-driven.

Нельзя:

> pre-registered proof of profile carrier.

## 4.2. P4

Можно:

> Random models show null median absolute order sensitivity under threshold 0.1, but relative sensitivity is ~13–27% on live docs. Systematic absolute flags occur only on degenerate low-motion period-1 docs: own_w4 6/6, own_w0 2/6.

Нельзя:

> random models are order-invariant.

## 4.3. P3/P5

Можно:

> Positive median real-vs-mix r/f_DC difference, descriptive only: sign test p=0.289, CI includes 0; n=8 has power only for unanimity. P5 is algebraic mirror of P3.

Нельзя:

> mix significantly increases DC.

## 4.4. P1

Можно:

> P1 is structurally vacuous on this data: c0 median ~260 ⇒ r_max ≪ 1. Synthetic null shows the r-band is not a generic drift detector.

Нельзя:

> random weights fail orthogonal walk / coherent drift.

## 4.5. P2

Можно:

> R1 vs R2 absolute r difference is below 0.1; trunk dominates absolute r scale. Tied input embedding still produces ~16–18% relative median modulation. Hidden-state r does not identify lm_head contribution.

Нельзя:

> embed has no effect / head contributes.

---

# 5. Минимальный патч перед закрытием AT0_4

Без новых форвардов:

1. **c0 sensitivity**
   ```text
   all 8
   clean-only
   G_nd-filtered pairs
   ```
   и явно post-hoc / pairwise not independent.

2. **P2 wording**
   ```text
   trunk_carries_absolute_scale
   tied_embed_relative_modulation ~0.16–0.18
   ```

3. **P1 wording**
   ```text
   P1_r_band_vacuous
   synthetic drift_smooth_walk misses band ⇒ not coherent-drift detector
   ```

4. **own_w5 note**
   ```text
   token period-13 present; absolute order flag absent/not live
   ```

5. **G_nd stratification**
   ```text
   clean / period-1 low-motion / periodic non-low-motion
   ```

6. **Hash chain**
   прислать manifest/rebuild r4 или подтвердить assert-цепочку.

7. **Mix permutation note**
   если одна перестановка — declare.

---

# 6. Итоговый вердикт

**AT0_4 r4: ACCEPT as own-doc r/DC pilot, with wording/consistency fixes.**

Главные закрытия:

- scope honest;
- P4 doc-level flags + degenerate provenance;
- P3/P5 downgraded to descriptive with power note;
- P1 vacuous diagnostics;
- P2 tied input embedding note;
- gnd_filter applied;
- endpoint_cos/c0 decomposition added;
- c0 trained/random emerged as strong post-hoc signal.

Главные оставшиеся риски:

- c0 headline post-hoc and needs clean/G_nd sensitivity;
- P2 “trunk_carries” overclaims relative zero-effect;
- P1 band interpretation unsupported by synthetic null;
- own_w5 periodicity vs order-flag absence needs note;
- hash-chain/rebuild r4 not visible.

**T0.4 corpus/profile claim remains open.**  
AT0_4 не закрывает “кто несёт профиль” и не закрывает corpus real/mix random×trained. Он закрывает более узкий, но полезный вопрос: на own-doc inputs `r` — это 2-DoF сводка endpoint cosine и c0, а c0 даёт систематический trained/random сигнал.