Принял `AT2b.py`, `AT2b_prereg.md`, `TZ_add_0-3.md`. В контексте видны фрагменты кода и пре-рега; `TZ_add_0-3.md` пустой, артефакта/лога прогона нет. Поэтому это **ревью конструкции AT2b**, не научного вердикта.

**Итог: AT2b закрывает большинство дыр AT2 r1 по конструкции. Но есть несколько мест, которые надо проверить в полном коде до запуска/интерпретации — они способны перевернуть вердикт. Главные: ключ метрики в дереве вердиктов, односторонность k=0-контроля, исключение k=0 из scratchpad-доказательства, S7-счётчик и калибровка DELTA_MIN.**

---

## 1. Что закрыто по сравнению с AT2 r1

| прежняя проблема | AT2b |
|---|---|
| DC-пол нецентрированной метрики | primary = центрированная метрика; `R²_c` по центрированному TSS; S6b `|R²_c| < 1e-12` |
| Сильвестр вместо Haar | true Haar = QR(G), `rng_base = default_rng(0)`, сначала G, потом uniform; Сильвестр убран из null-армов |
| нет k=0 контроля | k=0 добавлен как функциональная проверка top/weak |
| нет DC-free вариантов | `*_dc`: `V_dc = QR(V − (V·dc)dcᵀ)`, `dc = ones/32` |
| нет uniform-floor | uniform-арм, одна глобальная Gaussian-прямая |
| ridge penalty на feature 0 | `A[-1, -1] -= ALPHA` |
| lk4 без n / вырожденный арм | n per k, non-informative статус |
| N8-интерпретация под DC-метрикой | report-only |
| P-zero 0/0 в centered cos | через `R²_c = 0`, не cos |
| детерминизм | smoke ×2 bit-identical det-подартефакт |

Это правильно.

---

## 2. Что может перевернуть вердикт — проверить в полном коде

### V1. Дерево вердиктов читает `cos_median` — точно ли это центрированная метрика? ⛔

В видимом фрагменте:

```python
mtop = max(abs(agg[(L, k, "top")]["cos_median"]) ...)
mweak = max(abs(agg[(L, k, "weak")]["cos_median"]) ...)
b_ok = (mtop <= DELTA_MIN) and (mweak <= DELTA_MIN)
```

Но primary по пре-регу — **центрированная** метрика. Если `cos_median` остался legacy-ключом сырого cos, то:

```text
cos_median ≈ 0.93
DELTA_MIN ≈ 0.01
b_deadweight никогда не сработает
```

И хуже: если `a_scratchpad` тоже смотрит на raw-CI, то разности 0.002 на DC-полу могут быть «значимыми» и дать ложный SCRATCHPAD.

Нужно одно из:

```python
agg[...]["cos_c_median"]
```

или явный assert:

```python
assert metric_key == "cos_c_median"
assert "cos_median" not in verdict_inputs
```

Если `cos_median` в AT2b уже означает centered — переименовать или зафиксировать в schema, что это centered. Иначе это самый быстрый путь к неверному вердикту.

### V2. k=0-контроль должен быть **односторонним**: `lo > 0`, не «CI исключает 0» ⛔

Пре-рег:

> гейт: CI top−weak исключает 0 на всех 4 слоях; провал → SUBSPACE_CONTROL_FAIL

Это та же формулировка, которая в AT2 r1 дала D2-артефакт: **отрицательный CI тоже исключает 0**.

Нужно:

```python
control_ok = all(ci_top_minus_weak[k=0][0] > 0 for L in LAYS)
```

Если top < weak, но CI не содержит 0, гейт по текущей формулировке пройдёт. Это инвертирует смысл контроля.

### V3. `a_scratchpad` должен использовать только k≥1 и DELTA_MIN ⛔

Видимый фрагмент:

```python
if n_a >= 2:
    verdict = "SCRATCHPAD"
```

Нужно проверить:

1. `a_per_k` считается по `k ∈ {1,2,4,8}`, **не по k=0**.  
   k=0 — контроль текущей видимости, не доказательство будущего скретчпада.

2._PASS_ по k требует не просто `CI excludes 0`, а:

```python
win_ci(weak - max(pca, haar))[0] > DELTA_MIN
```

или хотя бы `> 0` + отдельный effect-size gate. Если a считается по raw-CI или без DELTA_MIN, центрированная метрика всё равно может дать шумовые «победы».

3. `n_a >= 2` — пре-рег должен сказать, почему 2, а не 1/4. Иначе это forking threshold.

### V4. S7 exact-set обязан отражать k=0 и новые арм ⛔

AT2 r1:

```text
286272 = 32 × 497 × 18
497 = 127+126+124+120
18 = 4 слоя × 4 базиса + L23 × 2
```

AT2b добавляет:

```text
k=0 ⇒ 625 = 128+127+126+124+120
новые арм: uniform, *_dc, возможно ker-J
```

Ожидание:

```text
S7 = 32 × 625 × A_new
```

Если S7 останется **286272**, k=0 или новые арм не дошли до per-position цикла.

Нужно в артефакте явно:

```text
KS
ARM_MAIN
L23_control_arms
kerJ_in_per_position: yes/no
S7_expected
```

Без этого нельзя сверить «заявленные фиксы ↔ посчитанные клетки».

### V5. DELTA_MIN должен быть калиброван по centered-полосе, а не унаследован ⚠️

Пре-рег вводит `DELTA_MIN + METRIC_UNINFORMATIVE`, но в видимых фрагментах нет обоснования значения.

После центрирования полоса может быть, например:

```text
cos_c ~ 0.10
floor uniform ~ 0.00–0.02
pairwise spread ~ 0.004
```

Тогда `DELTA_MIN = 0.01` разумен. Но если:

```text
cos_c ~ 0.003
```

то `0.01` больше всего сигнала ⇒ всё станет METRIC_UNINFORMATIVE.

Нужно из smoke/main отчитать:

```text
centered band per arm
uniform floor CI
pairwise spread
DELTA_MIN vs band/spread
```

И правило:

```text
DELTA_MIN > max(pairwise_spread, floor_CI_width)
DELTA_MIN < expected_signal_scale
```

Иначе METRIC_UNINFORMATIVE либо недостижима, либо съедает всё.

---

## 3. Остальное — не блокеры, но проверить

### M1. QR-знаки и кросс-платформенный детерминизм

Пре-рег говорит, что S9 smoke bit-identical закрывает LAPACK-зависимость. На одной машине — да. Но QR может давать разные знаки на разных LAPACK.

Если нужна межмашинная воспроизводимость:

```python
Q, R = qr(G)
sign = np.sign(np.diag(R))
Q = Q * sign
```

или аналогичная фиксация. Иначе sha-детерминизм локальный.

### M2. uniform-floor должен иметь CI

Uniform-арм — floor-референс. Одно число без CI опасно: при r=1 и LOO ridge он может дать случайный cos_c ~0.01–0.03.

Нужно:

```text
uniform_cos_c_median
uniform_cos_c_CI
```

И METRIC_UNINFORMATIVE должен сравнивать readout-арм с floor CI, а не с точечным floor.

### M3. pca per-doc остаётся transductive

`pca` фитится на H того же дока, который потом LOO-исключается из ridge. Это объявлено, но всё ещё асимметрия:

```text
top/weak: per (l,t), J-cache
pca: per (l,doc), test-doc H
haar/uniform: global
```

Опционально: `pca_loo` — базис из других доков. Не обязательно для запуска, но если pca вдруг выигрывает, это первый подозреваемый.

### M4. ker-J control: индексация правильная, но статус должен быть явен

```python
Vh[128:128+r_final]
```

Для JR формы `(128, 1024)` rank ≤ 128, значит это null-space directions. Хорошо.

Но нужно явно:

```text
ker-J: descriptive/control или входит в verdict?
```

Если ker-J future-readout высокий, это интересно, но не должно автоматически становиться SCRATCHPAD без пре-рега.

### M5. lk4 k-ahead lookup

```python
nxt[key] = row[i+k]
```

Это k-ahead lookup, не стандартный next-token. Для k>1 он может быть разрежен и шумный. Пре-рег уже помечает lk4 non-informative — хорошо. Главное, чтобы он не участвовал в `a_scratchpad` как полноценный baseline.

---

## 4. Что прислать для закрытия ревью

Достаточно фрагментов/полей:

```python
KS
ARM_MAIN
L23_ARMS
DELTA_MIN
metric_key_in_verdict
a_per_k_definition
control_gate_code
S7_expected
gates
verdict
agg_sample_keys
smoke_centered_band
uniform_floor_CI
```

Или минимальный extract:

```python
c = json.load(open("results/AT2b/AT2b.json"))
print(c["meta"]["KS"], c["meta"]["ARM_MAIN"], c["meta"]["DELTA_MIN"])
print(c["gates"])
print(c["verdict"])
print({k: list(v.keys()) for k, v in c["agg_sample"].items()})
```

---

## 5. Вердикт

**AT2b по конструкции: ACCEPT conditional.**

Он правильно чинит главные болезни AT2 r1: DC-пол, Сильвестр, отсутствие k=0, P-zero 0/0, ridge intercept, lk4/N8.

Но до интерпретации надо закрыть пять проверок:

1. verdict tree использует **centered** metric key;
2. k=0 control — **lower CI > 0**;
3. scratchpad — только k≥1 и с DELTA_MIN;
4. S7 отражает k=0 и новые арм;
5. DELTA_MIN калиброван по centered-полосе и floor CI.

Если эти пять закрыты, AT2b можно запускать/интерпретировать. Если k=0 control провалится — это не «ещё одна метрика», а невалидность top/weak-разбиения: T2.1 останавливать, не переписывать readout.