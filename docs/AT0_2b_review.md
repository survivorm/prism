Принял `AT0_2b_ra_split.py`. Ревью по видимому коду.

Короткий вердикт:

- **математическое ядро R/A-split хорошее**: точное разложение, same-t ортогональность, sign-flip null выбран правильно, гейты жёсткие, primary-пул pinned, L1-degrade учитывается.
- **но текущий `_verdict` ещё не отвечает на вопрос “кто несёт DC плато Δh”**: он смотрит на `frac_dc` и медианы `f_dc` компонент, но **игнорирует вклад компонент в net displacement Δh** и **не проверяет, есть ли DC у самого Δh в этом пуле**.
- Это может привести к неверному `RADIAL_DC / ANGULAR_DC / MIXED`, даже если все per-component гейты зелёные.

Ниже — по пунктам.

---

# 1. Что сделано правильно

## 1.1. Точное разложение Δh

Код использует:

```text
Δh = r̄·Δu + Δr·ū
R_t = Δr_t·ū_t
A_t = r̄_t·Δu_t
```

При средних:

```text
r̄ = (r_{t+1}+r_t)/2
ū = (u_{t+1}+u_t)/2
```

это тождество:

\[
\Delta h_t
=
\frac{r_{t+1}+r_t}{2}(u_{t+1}-u_t)
+
(r_{t+1}-r_t)\frac{u_{t+1}+u_t}{2}
=
r_{t+1}u_{t+1}-r_tu_t.
\]

И same-t ортогональность:

\[
\langle \Delta u_t,\bar u_t\rangle
=
\frac12(\|u_{t+1}\|^2-\|u_t\|^2)=0
\]

при unit `u`.

Гейты:

```python
closure_rel <= 1e-12
add_rel <= 1e-12
orth <= 1e-9
gram < 1e-10
```

Это правильный набор.

## 1.2. Sign-flip null для `f_dc` — правильный выбор

Ключевая мысль верная:

```text
mean(x) инвариантен к любой перестановке строк
⇒ permutation null для f_dc вырожден
```

Поэтому他们用 Rademacher sign-flip:

```python
eps = ±1
fdc_s = ||Σ eps_t x_t||² / (N² · mean||x_t||²)
E[fdc_s | null] = 1/N
```

Это корректная randomization-гипотеза “нет систематического среднего направления”.

Калибровка тоже правильная:

```python
iid ⇒ median fdc_perm ≈ 1/N
x + 20 ⇒ p ≤ 1/(B+1)
```

B=199 ⇒ p_floor=0.005 — соответствует уроку E3.

## 1.3. Global doc_idx

```python
doc_names = {nm: i for i, nm in enumerate(sorted(docs))}
seed = SEED_BASE + doc_idx * SEED_STRIDE + s
```

Это закрывает cross-stratum seed correlation, которую вы уже ловили в AT0_3.

## 1.4. Primary pool pinned

```python
primary = all_96 − corpus − degen
assert len(primary) == 37
```

Это хорошо:

- corpus без токенов исключён;
- degen исключён;
- совместимость с AT0_2 проверяется явно.

`all_96` и `loopy` остаются sensitivity/strata.

## 1.5. L1-degrade и component-degenerate guards

```python
own_w0/own_w4 × L1 → absent
c0 <= 0 or c0 < 1e-6·scale → component absent
```

Это诚实но. Не молча считает мусор.

## 1.6. Cross-lags без вердикта

```python
cross_RA / cross_AR
```

и note:

```text
ортогональность G6 — только при равных t
```

Правильно. Cross-lag не обязан быть нулём.

---

# 2. Критические проблемы, влияющие на вердикт

## C1. `_verdict` игнорирует вклад компонент в net displacement Δh

Текущий verdict:

```python
fr = radial_frac_dc
fa = angular_frac_dc
mr = radial_fdc_med
ma = angular_fdc_med

if fr > 0.5 and fa > 0.5: MIXED
if fr > 0.5 and mr > 1.1 * ma: RADIAL_DC
if fa > 0.5 and ma > 1.1 * mr: ANGULAR_DC
if fr <= 0.5 and fa <= 0.5: NO_DC
else: INCONCLUSIVE
```

Но вопрос клетки:

```text
кто несёт DC плато f_DC(Δh)
```

А `f_dc` компоненты — это **DC-доминированность внутри самой компоненты**, нормированная на её собственную энергию:

\[
f_{DC}(R)=\frac{\|\bar R\|^2}{\mathbb E\|R_t\|^2},
\qquad
f_{DC}(A)=\frac{\|\bar A\|^2}{\mathbb E\|A_t\|^2}.
\]

Это не вклад в:

\[
\Delta h = R + A.
\]

Код считает правильное разложение net displacement:

```python
dR = R.sum(axis=0)
dA = A.sum(axis=0)
dH = dh.sum(axis=0)

share_rad   = ||dR||² / ||dH||²
share_ang   = ||dA||² / ||dH||²
share_cross = 2 dR·dA / ||dH||²
```

Но `_verdict` их **не использует**.

Это может дать неверный ответ.

Пример:

```text
R: f_dc большой, p значим, но ||ΣR|| маленький
A: f_dc меньше, но ||ΣA|| огромный
```

Текущий verdict может сказать `RADIAL_DC`, хотя плато Δh несёт angular.

Ещё хуже: cancellation.

```text
share_rad = 1.8
share_ang = 1.6
share_cross = −2.4
sum = 1
```

Обе компоненты большие, но net displacement мал из-за отрицательного cross. Verdict `MIXED` будет misleading: это не “оба несут”, а “оба почти отменяются”.

**Фикс:**

Verdict должен иметь две оси:

1. **DC dominance внутри компоненты** — текущий `frac_dc / median f_dc`.
2. **Contribution to Δh net displacement** — `share_rad / share_ang / share_cross`.

Минимум:

```python
contrib_R = share_rad + 0.5 * share_cross
contrib_A = share_ang + 0.5 * share_cross
contrib_R + contrib_A = 1
```

Это half-cross attribution. Можно также report raw shares.

Итоговый carrier-verdict должен использовать contribution, а не только `f_dc`.

---

## C2. Не проверяется, есть ли DC у самого Δh в этом пуле

Код:

```python
fdc_dh, _, _ = sign_flip_fdc(dh, doc_idx)
out["fdc_dh"] = fdc_dh
```

`p_fdc` для Δh выбрасывается.

Но вопрос:

```text
кто несёт DC плато Δh
```

предполагает, что **Δh DC вообще есть** в данном пуле/слое.

Если в primary-пуле `fdc_dh` не значим или мал, то `RADIAL_DC / ANGULAR_DC` не должен выдаваться. Сейчас `_verdict` может сказать `RADIAL_DC`, даже если сам Δh не DC-present.

**Фикс:**

Сохранять:

```python
fdc_dh, p_dh, fdc_dh_perm_med = sign_flip_fdc(dh, doc_idx)
out["p_fdc_dh"] = p_dh
out["dc_present_dh"] = p_dh < ALPHA
```

В агрегате:

```python
dh_frac_dc
dh_fdc_med
dh_p_med
```

И gate:

```text
if dh_frac_dc <= 0.5 or dh_fdc_med below DELTA_MIN:
    verdict = NO_DH_DC / not_testable
```

Иначе carrier-verdict не интерпретируется.

---

## C3. `frac_dc` и множитель `1.1` — без uncertainty / n_present

Текущий verdict:

```text
frac_dc > 0.5
median f_dc > 1.1 × другой
```

Проблемы:

1. `frac_dc` — доля клеток с `p < 0.05`. Это зависит от:
    - N,
    - B,
    - effect size,
    - числа present components.
2. `1.1` — arbitrary threshold без CI.
3. Нет `n_present` в агрегате.
4. Median ratio может быть >1.1, но CI перекрывается.
5. Если component absent в части клеток, `frac_dc` считается по оставшимся, но denominator не виден.

В `_pool_stats`:

```python
vals = [c[nm][key] for c in cells
        if isinstance(c.get(nm), dict)
        and not c[nm].get("absent")
        and key in c[nm]]
```

Но `n_present` не сохраняется.

Следствие:

```text
radial_frac_dc = 1.0
```

может означать:

```text
1/1 present cell significant
```

а не 37/37.

**Фикс:**

В агрегат добавить:

```python
radial_n_present
angular_n_present
radial_n_dc_present
angular_n_dc_present
```

и CI для:

```python
median_fdc_R
median_fdc_A
median_fdc_R / median_fdc_A
median contrib_R / contrib_A
```

Verdict rule должен быть pre-registered с resolution / DELTA_MIN, а не только `1.1`.

---

## C4. Sign-flip null — это null среднего, не ACF-matched DC null

Это не баг, но граница интерпретации.

Sign-flip проверяет:

```text
нет систематического среднего направления при фиксированных ||x_t||
```

Он не сохраняет:

- autocorrelation,
- telescoping,
- bounded-path constraints,
- lag structure.

Поэтому:

```text
dc_present = p_signflip < 0.05
```

нельзя читать как:

```text
DC-мощность значима относительно stationary process с той же ACF
```

Это другое утверждение.

Для R/A-split это допустимо, потому что вопрос:

```text
есть ли направленный mean drift компоненты
```

Но в тексте надо явно:

```text
DC presence = sign-flip mean-direction test, not ACF-matched spectral null.
```

Иначе читатель свяжет это с T0.1/T0.2 spectral DC и получит неверный мост.

---

## C5. `n_docs` в `_pool_stats` может считать absent cells

```python
stat["n_docs"] = len(cells)
```

Если `cells` включает absent L1 или absent dh_zero, то `n_docs` завышен относительно present measurements.

Для primary это, вероятно, не проблема, потому что own_w0/4 degen исключены. Но для `loopy` / `all_96` L1 absent есть.

**Фикс:**

```python
stat["n_cells_total"] = len(cells)
stat["n_cells_present"] = sum(not absent)
stat["radial_n_present"] = ...
stat["angular_n_present"] = ...
```

И не называть это `n_docs`, если это cells per layer.

---

## C6. Provenance: decode probe skipped

Код:

```python
try:
    dec = a1.decode_probe(...)
except ImportError:
    dec = {"skipped": ...}
```

и note:

```text
G0-целостность держится на sha256; проба — content-sanity
```

Это честно, но по серии G0-a decode probe — не просто optional.

Особенно потому что:

- corpus без токенов;
- own_w0/own_w4 уже были подозрительные;
- gen tokens используются в primary.

Если decode probe skipped, primary должен быть:

```text
provenance_limited: content_sanity_skipped
```

а не просто “G0 sha ok”.

Это не флипает R/A математику, но ограничивает content claims.

---

## C7. Store / manifest self-hash semantics

Код:

```python
store["store_sha256"] = None
store_b = json.dumps(pf.jclean(store), sort_keys=True).encode()
store_sha = sha256(store_b)
store["store_sha256"] = store_sha
write(store)
```

То есть `store_sha256` — это hash payload **с полем None**, а не sha фактического файла.

То же для manifest:

```python
man_b = json.dumps(man, sort_keys=True).encode()
man["manifest_sha256"] = sha256(man_b)
write(man)
```

Это допустимая конвенция, но её надо явно записать:

```text
store_sha256 = canonical payload hash excluding self-field
file_sha256 = sha256(actual file)
```

Иначе verifier, который хеширует файл, получит mismatch.

**Фикс:** добавить `file_sha256` для store/claim/manifest или документировать canonical convention.

---

# 3. Что ещё стоит добавить, но это не blocker

## M1. Energy additivity gate

Из same-t orthogonality следует:

\[
c_0(\Delta h)=c_0(R)+c_0(A).
\]

Можно добавить дешёвый gate:

```python
abs(c0_dh - c0_R - c0_A) / c0_dh < 1e-12
```

Это поймает баг в `ra_split` или нормировке раньше, чем verdict.

## M2. `vt_plateau` anchor не используется в verdict

Код считает:

```python
vt_plateau = median_t cos(9..32)
```

Это хороший мост к T0.1, но пока descriptive.

Если клетка называется “кто несёт плато T0.1”, стоит хотя бы gate:

```text
vt_plateau present / above floor
```

или явно:

```text
anchor descriptive, not part of verdict.
```

## M3. Component degenerate fraction

`COMP_DEGEN_FRAC = 1e-6` — хорошо. Но надо агрегировать:

```python
radial_absent_frac
angular_absent_frac
```

Иначе `frac_dc` по present components может быть survivor-biased.

---

# 4. Минимальный патч перед финализацией verdict

Без новых форвардов, из уже посчитанных cells:

## 4.1. Сохранить / агрегировать Δh DC test

В `cell_ra`:

```python
fdc_dh, p_dh, fdc_dh_perm_med = sign_flip_fdc(dh, doc_idx)
out["fdc_dh"] = fdc_dh
out["p_fdc_dh"] = p_dh
out["fdc_dh_perm_median"] = fdc_dh_perm_med
out["dc_present_dh"] = p_dh < ALPHA
```

В `_pool_stats`:

```python
dh_frac_dc
dh_fdc_med
dh_p_med
```

## 4.2. Aggregates для shares

Уже есть per-cell:

```python
share_rad, share_ang, share_cross
```

Добавить:

```python
contrib_R = share_rad + 0.5 * share_cross
contrib_A = share_ang + 0.5 * share_cross
```

Агрегировать median + bootstrap CI.

## 4.3. Новый verdict rule

Разделить:

```text
dominance_verdict: по frac_dc / f_dc компонент
carrier_verdict: по contribution to Δh net displacement
```

Пример carrier rule:

```text
gate: dh_dc_present
if contrib_R_CI_low > 0.5 and contrib_R > contrib_A: RADIAL_CARRIER
elif contrib_A_CI_low > 0.5 and contrib_A > contrib_R: ANGULAR_CARRIER
elif share_cross < -0.5 and share_rad > 0.5 and share_ang > 0.5: CANCELLED
elif both contrib CI overlap 0.5: MIXED / INCONCLUSIVE
else: not_testable
```

Точные пороги — pre-reg, но принцип: **verdict по вкладу в Δh, не по f_dc компоненты**.

## 4.4. n_present / absent fractions

В каждый pool/layer aggregate:

```python
n_cells_total
n_cells_present
radial_n_present
angular_n_present
radial_absent_frac
angular_absent_frac
```

## 4.5. Provenance

```python
decode_probe_status: ok | skipped
provenance_limited: content_sanity_skipped if skipped
store_file_sha256
claim_file_sha256
manifest_file_sha256
```

---

# 5. Что можно принять уже сейчас

Принять как математически валидный decomposition:

```text
Δh = R + A exactly
R ⊥ A at same t
closure / additivity / orth / gram gates
sign-flip f_dc null with B=199
primary verified∩nondeg pool
L1-degrade handling
cross-lags descriptive
```

Принять как descriptive:

```text
radial/angular f_dc medians
radial/angular frac_dc
share_rad / share_ang / share_cross medians
cross_RA / cross_AR curves
vt_plateau anchor
```

Не принимать пока как carrier verdict:

```text
RADIAL_DC ⇒ плато Δh = рост нормы
ANGULAR_DC ⇒ плато Δh = когерентный поворот
MIXED ⇒ оба несут
NO_DC ⇒ остаток T0.5
```

Потому что текущий verdict не проверяет:

1. есть ли DC у Δh в этом пуле;
2. какая компонента вкладывает net displacement;
3. есть ли cancellation;
4. сколько present components реально стоит за `frac_dc`.

---

# 6. Итоговый вердикт

**AT0_2b как R/A decomposition: PASS.**  
**AT0_2b как “кто несёт DC плато Δh”: REVISE verdict rule.**

Главные правки:

1. **verdict должен использовать shares / contribution, а не только component f_dc;**
2. **Δh DC presence должен быть gate;**
3. **frac_dc требует n_present и CI;**
4. **sign-flip null назвать mean-direction null, не ACF-matched spectral null;**
5. **provenance/hash semantics дописать.**

После этого клетка станет действительно разводящей:

```text
RADIAL_CARRIER  → плато Δh = drift нормы
ANGULAR_CARRIER → плато Δh = когерентный поворот
CANCELLED       → обе большие, но net почти ноль
NO_DH_DC        → нечего разводить
INCONCLUSIVE    → не разрешено
```

Сейчас же она честно измеряет R/A, но verdict ещё может ответить не на тот вопрос.