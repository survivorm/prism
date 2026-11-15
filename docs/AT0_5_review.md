Принял обновлённый `AT0_5_gencheck.py`. Ревью по видимому коду r4.

Коротко:

- **почти все прежние C1–C7 закрыты**: S2 target, fitted-DC null, K/N-правило, in-sample GOF wording, VAR BIC, hard gates/RC, S4 docstring.
- **но branch1-вердикт ещё нельзя финализировать**: остались три дыры, которые могут повлиять на `frac_covered ≥ 0.5` — primary-фракция без CI/source-coverage, d_eff/null-calibration не калибрует real coverage, и seed-коллизии в calibration/null.
- Это уже не «сломанная машина», а «машина правильная, но вердикт-агрегация и калибровка ещё не защищены».

---

# 1. Что r4 закрыл

| прежний пункт | фикс в коде |
|---|---|
| **C1** S2 target `−1/(N−1)` | `gate_s2`: target **0**, uncentered iid `E[ρ̂(k)]=0`; docstring явно объясняет old target как demeaned-артефакт |
| **C2** D-null инжектил real DC | `dcma1_sim_d_fit(mvec, svec, fit["f"], fit["a"])`: total power сохранён, DC-доля = fitted `f`; old circular version удалена |
| **C3** dip coverage как generative | docstring/verdict: **in-sample GOF**, generative-claim не делается; OOS только `plat_hi_oos`, dip не OOS |
| **C5** K-правило / short cells | `k = min(96, n−1)`, T0.1 K-алфавит `{0,24,32,96}`; S9 проверяет K/N; primary = non-degen ∩ K96; short — sensitivity; T0.1 K=0 skip |
| **C6** VAR BIC trace/underpenalized | primary BIC = `n·logdet Σ + (p·r²+r)·log n`; TZ BIC — sensitivity |
| **C7** gates/RC | `ok is True`; при FAIL store+log пишутся, claim/manifest нет, `RUN_RC=1` |
| S4 docstring | odd/even boundary исправлен |
| OOS norm | per-cell `n`, не глобальный N |
| quantile MC error | `qmc` отчитывается |

Это хорошо.

---

# 2. Оставшиеся проблемы, влияющие на branch1

## C1. Primary `frac_covered` без CI и без source-coverage

Код:

```python
elig = [nm for nm in nd if cells[(nm,L)]["k"] == K]
cov_elig = [cells[(nm,L)]["part1"]["cov"]["covered"] for nm in elig]
per_layer[L]["frac_covered"] = mean(cov_elig)
...
branch1 ⇔ frac_covered ≥ 0.5 на каждом слое
```

Но:

1. **Нет CI для `frac_covered`.**  
   Это доля булевых per-doc исходов. При `n_eligible ~ 50` биномиальная/Wilson CI шириной ±0.14. Если `frac ≈ 0.5`, вердикт fragile.

2. **`source_split` — только counts, не coverage.**

```python
src[docs[nm]["source"]] += 1
```

Это говорит, сколько gen/own/corpus в eligible, но **не говорит, кто покрывается**. Corpus без токенов может драйвить branch1, а мы не видим.

3. **Corpus остаётся в primary.**  
   `pool_note`: corpus h-данные есть, токены 0/16, G16 not_testable — только provenance-оговорка. Это допустимо, но тогда source-coverage обязателен.

**Фикс:**

```python
per_layer[L]["frac_covered_ci95"] = wilson_or_bootstrap(cov_elig)
per_layer[L]["frac_covered_by_source"] = {
    "gen": mean(cov for nm in elig if source=="gen"),
    "own": ...,
    "corpus": ...,
}
per_layer[L]["frac_covered_gen_only"] = ...
```

И вердикт:

```text
branch1 only if frac_covered ≥ 0.5 on every layer
AND CI lower bound / source sensitivity not contradict
```

или хотя бы:

```text
if CI includes 0.5 → inconclusive, not branch1/branch2
```

Сейчас `frac=0.501` и `frac=0.90` дают один и тот же verdict.

---

## C2. `d_eff` — diagonal participation ratio, не effective dof covariance

Код:

```python
s2 = ((dh - mean)**2).mean(axis=0)
d_eff = (s2.sum())**2 / (s2*s2).sum()
```

Это participation ratio **per-dim variances**, то есть diagonal PR:

\[
d_{\text{diag}}=\frac{(\sum_j \Sigma_{jj})^2}{\sum_j \Sigma_{jj}^2}.
\]

Но effective dof covariance:

\[
d_{\text{full}}=\frac{(\mathrm{tr}\Sigma)^2}{\|\Sigma\|_F^2}
=\frac{(\sum_j \Sigma_{jj})^2}{\sum_{ij}\Sigma_{ij}^2}.
\]

Если cross-dim correlations сильны, \(d_{\text{full}} \ll d_{\text{diag}}\).

Следствие:

- `d_eff` в артефакте может сильно завышать effective dof;
- independent D=1024 null может давать слишком узкие/широкие CI;
- `null-on-null` calibration это **не поймает**, потому что обе партии из того же independent null.

**Фикс:**

```python
dhc = dh - dh.mean(axis=0)
Sigma = (dhc.T @ dhc) / n
d_eff_full = trace(Sigma)**2 / (Sigma*Sigma).sum()
```

и отчитывать оба:

```text
d_eff_diag
d_eff_full
```

Для калибровки real coverage нужен хотя бы один из:

1. **vector residual bootstrap** fitted MA(1):
   - fit `a`, получить residual vectors;
   - resample blocks of residual vectors;
   - simulate MA(1) with bootstrapped vector innovations;
   - preserves cross-dim covariance and non-Gaussianity.

2. **covariance-matched Gaussian null**:
   - simulate innovations with real residual covariance / low-rank factor;
   - expensive but doable via SVD.

3. **d_eff sensitivity**:
   - independent null with `d_eff_full` factors instead of D=1024;
   - report coverage shift.

Сейчас `null-on-null` проверяет только то, что симулятор самосогласован. Это полезно, но не калибрует real coverage.

---

## C3. Seed-коллизии в calibration и layer-independent null seeds

### C3.1. Calibration seeds collide

Код:

```python
Xa = dcma1_sim_d_fit(..., SEED_CAL_A + idx_of[nm] + li, ...)
Xb = dcma1_sim_d_fit(..., SEED_CAL_B + idx_of[nm] + li, ...)
```

`idx` 0..95, `li` 0..4.

Коллизия:

```text
(idx=1, li=0) → seed 500001
(idx=0, li=1) → seed 500001
```

То есть разные клетки получают одинаковые random draws.

Part1/part2 делают правильно:

```python
SEED1_OFF + NSEED * idx + li   # NSEED=5
```

Calibration должен так же:

```python
SEED_CAL_A + 5 * idx + li
SEED_CAL_B + 5 * idx + li
```

Это reportable calibration, не primary gate, но коллизии делают её коррелированной и занижают variance.

### C3.2. Part3 null seeds не включают layer

Код:

```python
null_cell(idx, pth, scores, seed_i)
...
SEEDN_OFF + NSEED * idx + seed_i
```

Нет `li`.

Следствие: один и тот же doc на разных layers получает одинаковые sim seeds.

Также:

```python
perm = block_perm(n, BLK, 1001 + seed_i)
```

Permutation pattern зависит только от `seed_i`, не от doc/layer.

Это может быть intentional paired null, но тогда надо declare:

```text
null permutations paired across docs/layers by seed_i
```

Если null cells считаются независимыми для median/CI — это неверно.

**Фикс:**

```python
SEEDN_OFF + NSEED * (5 * idx + li) + seed_i
block_perm_seed = base + 5 * idx + li + seed_i
```

или явно paired design.

---

# 3. Менее критичные, но фиксирую

## M1. `dcma1_sim_d_fit`: edge case `M=0, f>0`

Код:

```python
m2 = mvec * sqrt(Mp/M) if M > 0 else zeros_like(mvec)
```

Если real mean exactly zero (`M=0`), но fitted `f>0`, sim DC остаётся 0, а не fitted `f`.

Вероятно редко, но это нарушает контракт:

```text
DC-доля = fitted f
```

Фикс:

```python
if M == 0 and f > 0:
    # choose direction from svec or random unit vector
    m2 = unit_direction * sqrt(Mp)
```

или fail-loud / flag `dc_direction_undefined`.

## M2. `sum96` — имя врёт для short cells

```python
"sum96": float(rho.sum())
```

Для `k<96` это sumK. Primary K96, но short sensitivity есть. Rename `sumK` или `sum_rho_k`.

## M3. `gate_s5/s8` используют глобальный N/K

`gate_s5(dh)` внутри использует `N=383`, `K=96`.

Вызывается на `paths[s5_key]`, где `s5_key=(docnames[0],23)`. Если первый doc full — ок. Но gate должен быть robust:

```python
n = dh.shape[0]
k = min(K, n-1)
```

Иначе short-first smoke может сломаться.

## M4. VAR/AR1 simulation без burn-in

`gen_sim`: VAR initial zeros, AR1 starts at mean n_real.

Если selected VAR near-unit / AR1 beta near 1, early transient может повлиять на lag metrics. Part2 reportable, не branch1, но стоит добавить burn-in или declare.

## M5. `null_cell` scores — проверить, не real ли

`null_cell(idx, pth, scores, seed_i)` получает `scores_cache[(nm,L)]`.

Если ladder на null считается из **real scores**, а не shuffled `dh_s`, то `selected_dist_null` невалиден.

Видимый фрагмент не показывает использование `scores`. Надо проверить:

```python
scores_s = svd_scores(dh_s / norms_s)
ladder_null = var_ladder(scores_s)
```

Если используется real `scores` — баг. G8 gate это не использует, но reportable null ladder — да.

---

# 4. Что уже можно принять

Как machine:

```text
S0 identity
S1 vector telescope
S2 uncentered iid target 0
S3 AR(1) positive control
S4 odd/even boundary
S5 circular Fourier positive control + linear wrap
S6/S7 gram/r=NfDC
S8 permutation invariance
S9 T0.1 store anchor + K/N check
fitted-DC D-null
in-sample GOF wording
primary K96 ∩ nondegen
VAR BIC logdet full df
hard gates + RC=1 on FAIL
```

Как design:

```text
branch1 = in-sample DC+MA(1) GOF, not generative transfer
OOS = plat_hi only, dip circular
short/degen separate
corpus provenance caveat
quantile MC error reported
```

Но **branch1 verdict** пока conditional.

---

# 5. Минимальный патч перед финализацией branch1

Без новых форвардов, из cells/store:

1. **Per-layer coverage CI**
   ```python
   wilson_ci(n_covered, n_eligible)
   ```
   и verdict:
   ```text
   CI includes 0.5 → inconclusive
   ```

2. **Source coverage**
   ```python
   frac_covered_by_source
   frac_covered_gen_only
   frac_covered_excl_corpus
   ```

3. **d_eff full**
   ```python
   d_eff_full = trace(Sigma)**2 / ||Sigma||_F**2
   ```
   и report diag/full.

4. **Real coverage calibration sensitivity**
   - vector residual bootstrap;
   - или covariance-matched null;
   - или d_eff_full factor null.

5. **Seed fixes**
   ```python
   cal_seed = SEED_CAL_A + 5*idx + li
   null_seed = SEEDN_OFF + NSEED*(5*idx + li) + seed_i
   ```
   или declare paired null.

6. **M1/M2/M3/M5**
   - M=0 f>0 edge;
   - sumK rename;
   - gate_s5/s8 per-cell n;
   - null ladder uses shuffled scores.

---

# 6. Итоговый вердикт

**AT0_5 r4 как estimator/gate machine: ACCEPT.**  
Прежние C1–C7 в основном закрыты.

**AT0_5 как branch1 verdict: REVISE / conditional.**

Главные оставшиеся риски:

1. `frac_covered ≥ 0.5` без CI и source-coverage;
2. `d_eff` diagonal PR + independent D null не калибрует real cross-dim coverage;
3. calibration/null seed collisions.

После фикса честная формулировка будет:

> In-sample DC+MA(1) GOF covers dip/plateau in ≥50% non-degenerate K96 docs on every layer, with CI/source sensitivity and calibrated null. Three-mode decomposition not needed as in-sample account; bump remains for T3.

Сейчас код уже близок, но вердикт-агрегация и калибровка ещё не защищены.