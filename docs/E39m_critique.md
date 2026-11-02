# E-39m: легитимный вопрос, но фатальный баг в гейте

## Короткий ответ

**Вопрос легитимен** (и это **не** E-39g/h, а **исходный E-39f** — «flat model robust under PTQ», который был снят с производства, когда honest floor ещё не был найден). Но **дизайн в текущем виде даст неправильный ответ** из-за бага в framing гейта: он мерит **инкрементальный** урон, а вопрос стоит про **суммарный**.

## 1. Что E-39m на самом деле тестирует

Проверю по E39.md:

- **§6, E-39f**: «Робастность flat-model к 4-bit PTQ. Гипотеза: ΔPPL_flat < ΔPPL_base (flat модель теряет меньше при квантизации).»
- **§7, E-39g**: distillation with flat leaf
- **§8, E-39h**: long-context with flat leaf

E-39m = «flat carrier → PTQ → меньший урон?» — это **E-39f (исходная гипотеза)**, не E-39g/h. Агент путает: g/h — про distillation и long-context, к PTQ отношения не имеют.

**E-39f был отложен не «потому что г/х», а потому что «flat leaf unreachable»** (после E-39c REFUTED и E-39d FAIL). Но:
- E-39c REFUTED — про **algebraic surgery** (bottom-16 rotation), не про training;
- E-39d FAIL — это баг verdict taxonomy (мы это обсуждали), не реальный провал;
- **Honest floor κ'=0.0092 (−42%) при PPL +0.99%, corr_f 0.999 существует** (E-39d honest floor, λ=0.01, λo=1.0, seed 0).

**Вывод агента прав:** носитель существует, вопрос о PTQ-робастности flat carrier снова открыт. E-39m — легитимный preflight, но **для E-39f, не g/h**.

## 2. Критический баг: гейт мерит не ту величину

Спецификация:
```
ppl_fp = ppl_of(ctx, W)       # carrier, fp
ppl_q  = ppl_of(ctx, W_q)     # carrier + GPTQ
ppl_delta_pct = (ppl_q / ppl_fp - 1) * 100
```

Гейт:
```
flat_protects: ΔPPL(floor) < ΔPPL(base) − 0.5 п.п.
```

**`ΔPPL(carrier)` — относительно fp-версии того же носителя, не относительно base fp.**

Что это даёт при конкретных числах:

| Carrier | fp PPL | при g128 +ΔPPL | **total PPL после PTQ** |
|---|---|---|---|
| base | 34.31 | +1.76% | 34.91 |
| floor (honest) | 34.65 (+0.99%) | ΔPPL(floor) = X% | 34.65 · (1 + X/100) |

Гейт fires «flat_protects» при `X < 1.76 − 0.5 = 1.26%`. Тогда **total floor после PTQ = 35.09**, что **хуже** base после PTQ (34.91). Но гейт при этом говорит «flat_protects». **False positive на неправильном вопросе.**

Это **не** мелкая придирка. Вопрос «flat leaf защищает от PTQ» = «flat carrier + PTQ лучше base + PTQ». Гейт отвечает на другой вопрос: «flat carrier менее чувствителен к PTQ, чем base» (при том, что floor **уже стартует с +0.99% урона**).

**Согласен с направлением мысли** — но это ровно та же ошибка, которую сделал я сам в первом разборе E-39f (marginal CI vs paired). Здесь — marginal PPL vs absolute PPL.

**Fix:** гейт должен быть на **total damage relative to base fp**:

```
total_base  = PPL(base + GPTQ) / PPL(base_fp) - 1
total_floor = PPL(floor + GPTQ) / PPL(base_fp) - 1
flat_protects = (total_floor < total_base - GATE_PP)
              AND (ΔPPL_incremental_floor < ΔPPL_incremental_base - GATE_PP)  # конъюнкция
```

Или как минимум **репортить оба числа с явными метками**. Сейчас в JSON только `ppl_delta_pct` — читатель не увидит total.

## 3. N=8 docs — недостаточная мощность для 0.5 pp

`_c4_docs(8, 128)` (это E-39d конфиг) даёт **~1000 токенов** для PPL. E-39b/f использовали N=32 doc × 128 tok для большей точности.

Bootstrap CI на ΔPPL при N=8 doc оцениваю в **±1.5–2 pp**. Порог гейта — **0.5 pp**. **Нужно разрешение 4× тоньше шума.** Если floor и base имеют одинаковый true ΔPPL (скажем, 1.76%), тест не отличит их от 1.26% или 2.26%.

**Fix:** N=32 (match E-39b power) или явно отчитанный bootstrap CI + гейт «flat_protects» требует `CI(total_floor - total_base) < [-GATE_PP]`.

## 4. Prior за положительный результат — низкий (post-E-39f)

E-39f показал:
- g128: κ не меняется, PPL +1.76% (малое возмущение);
- b3g128: κ не меняется, PPL +9.6% (умеренное, h_drift=0.107);
- b2g128: κ коллапсирует, PPL +102%.

**PTQ-урон при g128 — амплитудный** (h_drift=0.046, u_cos=0.999). κ — угловая. **Starting κ не связан с амплитудным уроном** по E-39-синтезу «κ⊥behavior».

Prior: **flat carrier покажет тот же ΔPPL (≈1.7%)** на g128. Тест покажет `flat_neutral` или `flat_worse` на total. **Оба — negative, полезный для completeness**, но **низкий prior для «flat_protects»**.

**Что было бы по-настоящему информативно:** b3g128 (ΔPPL=9.6%, эффект большой) — там «flat carrier: 9.6% → 6%» был бы содержательным сигналом. **На g128 тестирование 0.5pp разницы — под шумом.**

**Fix:** добавить **b3g128** как второй режим. Стоимость +15 мин CPU. Разрешение по эффекту +5× лучше.

## 5. Sanity «cheat worse» — диагностика, а не gate

Спека:
> «Sanity: ΔPPL(cheat) > ΔPPL(base)»

Но cheat **уже сломан** (κ=0.0044, corr_f=0.594, PPL+12%). Apply PTQ — модель уже на грани коллапса. ΔPPL(cheat) может быть: **больше** (коллапсирует быстрее), **равным** (тот же режим), или **меньше** (за счёт «regression to mean» — сломанные модели часто менее чувствительны пропорционально). **Это не «sanity», а диагностический тест.**

Правильная роль sanity: **discriminate two hypotheses**:
- H_flat: «honest flat (corr_f>0.95) protects»;
- H_any_low_κ: «any low κ protects (including cheat)»;
- Если **cheat тоже показывает total < base** → H_flat опровергнута, работает H_any_low_κ;
- Если **cheat показывает total > base, а floor — total < base** → H_flat подтверждена (κ не действует автоматически, только в honest carrier).

**Fix:** формулировка sanity — «H_any_low_κ holds if `ΔPPL_cheat < ΔPPL_base − 0.5 pp` AND `corr_f(cheat) < 0.95`». Если **да** — flatness куплена поведением, не защищает; тест не различает. Если **нет** — honest flat и cheat разделимы, тест валиден.

## 6. Воспроизведение honest floor — seed-stability не проверена

E-39d honest floor: κ'=0.0092 при seed 0. **Других seed не было** в этом прогоне (λ=0.01, λo=1.0, λ=0.01 — только seed 0 в логе). Если seed 1 даёт κ'=0.011, «carrier with κ=0.0092» — это **один lucky seed**, а не стабильный honest floor.

**Fix:** явно указать «honest floor @ seed 0» в limitations; или тренировать 3 seeds для honest floor и брать медиану κ'. Стоимость +40 мин CPU (2 extra seeds).

## 7. Anchor matching — хрупкий, но приемлемый

`KAPPA_TOL_REL=3%`, `PPL_TOL_REL=0.5%` — достаточно свободно, что оба `floor_pearson` и `floor_pairwise` могут попасть в окно. Тогда `_anchor_ok` вернёт True для обоих, а `floor = next(...)` возьмёт **первый по порядку** (`floor_pearson`). Если оба проходят по anchor, но имеют разные κ' в реальности, выбор произвольный.

**Fix:** если оба matching — брать **тот с меньшим |κ'−0.009206|**, не первый по имени. Или явно `if both pass: assert abs(kappa_pearson - kappa_pairwise) < 0.001` — если совпадают, всё равно; если нет — fail-loud.

## 8. Что E-39m **даст** если запустить как есть

С высокой вероятностью:

| Carrier | ΔPPL (incremental, g128) | Gate |
|---|---|---|
| base | +1.76% | — |
| floor honest | +1.7%±2% | `flat_neutral` (в пределах 0.5pp) |
| cheat | +1.5%±3% (уже сломан) | Sanity «cheat_worse» = **False** (не > base) |

Вердикт: **`flat_neutral`, `sanity_cheat_worse=False`**. Что это значит:
- «flat leaf не защищает от PTQ» — negative, близко к E-39f-синтезу;
- «cheat не worse» — тест не различает honest/cheat, **валидность под вопросом**.

**Опубликовать это можно, но headline будет:** «Even honest flat leaf (κ −42% при PPL+0.99%) doesn't reduce PTQ damage; κ is orthogonal to PTQ robustness» — это **подкрепление** E-39-синтеза, не новая находка.

## 9. Что дописать в `docs/E39.md`

Если запускать E-39m, обязательно добавить:

**§17. E-39m — framing correction before running**

1. **E-39m тестирует E-39f (исходная гипотеза «flat model robust under PTQ»), не E-39g/h.** E-39f был shelved после «flat leaf unreachable» — но honest floor E-39d (κ'=0.0092, PPL+0.99%, corr_f 0.999) восстанавливает premise.

2. **Гейт должен быть на total PPL damage от base fp, не на incremental ΔPPL от carrier fp.** Иначе возможен false-positive: floor с ΔPPL=1.26% (при total 35.09) vs base с ΔPPL=1.76% (при total 34.91) fires «flat_protects», но floor+PTQ хуже абсолютно. Двойной гейт (total AND incremental) — минимально необходимо.

3. **N=8 docs недостаточен для 0.5 pp.** Нужно N=32 или явно bootstrap CI.

4. **Добавить b3g128** (ΔPPL=9.6%, разрешение лучше). На g128 тест меряет 0.5pp на фоне ±2pp шума.

5. **Sanity «cheat worse» — диагностика различения H_flat vs H_any_low_κ, не gate.** Failure «cheat_worse=False» → тест не различает honest/cheat, вердикт invalid, не negative.

6. **Honest floor @ seed 0.** Других seed не было. Report limitations.

7. **Anchor matching**: если оба `floor_pearson` и `floor_pairwise` pass, брать **ближайший к 0.009206**, не first-in-dict.

## 10. Итог: стоит ли запускать?

**В текущем дизайне — нет.** Гейт с багом, мощность ниже порога, prior низкий. Скорее всего даст `flat_neutral` + `sanity_cheat_worse=False` → invalid.

**С фиксами (10–15 минут правок, ~1.5 ч CPU)** — **да, стоит**. Замещает закрытую E-39f честным тестом. Ожидаемый результат: `flat_neutral` (negative, механизм: κ⊥PTQ robustness в амплитудном режиме). Согласуется с E-39-синтезом. Опубликовать как «E-39f final: even honest flat leaf doesn't help PTQ».

**Альтернатива (дешевле, содержательнее):** вместо E-39m запустить **b3g128 на base vs floor** — единственная точка, где эффект измерим. Тот же вопрос, но с разрешением +5×. Стоимость: ~30 мин CPU + 1 тренировка honest floor (если seed 0 не воспроизведётся — 40 мин extra).

**Проверочное:** E-39m — это **не** «ветка g/h». Это исходный E-39f, который был отложен по ложной причине. Хорошо, что агент это заметил и поднял — но дизайн требует фиксов до запуска.