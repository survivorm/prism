## Баги (код не запустится как есть)

### 1. `d3_by_r_tail_out` — NameError, hard crash на первом слое

```python
layers_out[str(l)] = {
    ...
    'd3_by_r_tail': d3_by_r_tail_out or None,   # ← undefined
    ...
}
```

`d3_by_r_tail_out` нигде не определён. Агрегация считает только `d3_by_r_out` (из `d3_by_r`), но не `*_tail` из `d3_by_r_tail`. Упадёт на L6, до любой записи JSON.

### 2–4. `s_l_tail`, `a_vs_ESG_cos_tail`, `id_cos_JR_Jtail` — KeyError в verdicts

В `layers_out[str(l)]` **нет** ключей `s_l_tail`, `a_vs_ESG_cos_tail`, `id_cos_JR_Jtail`. Но verdicts читают:
```python
sl_tail_med = [layers_out[str(l)]['s_l_tail']['median'] ...]  # KeyError
aev = {str(l): layers_out[str(l)]['a_vs_ESG_cos_tail'] ...}    # KeyError
```

Причина: в цикле агрегации не считаются `_agg([v['s_l_tail_median'] ...])`, `_agg([v['a_vs_ESG_cos_tail'] ...])`, `_agg([v['id_cos_JR_Jtail_median'] ...])`. Всё это есть в `per_doc_layer`, но до layer-level не доходит.

**Фикс одним блоком** после существующих `d3fix/d3raw/d3gf`:
```python
sl_tail_arr = np.array([v['s_l_tail_median'] for v in V if v['s_l_tail_median'] is not None])
aev_tail = _agg([v['a_vs_ESG_cos_tail'] for v in V if v['a_vs_ESG_cos_tail'] is not None])
idcos = _agg([v['id_cos_JR_Jtail_median'] for v in V if v['id_cos_JR_Jtail_median'] is not None])
d3_by_r_tail_out = None
if V[0].get('d3_by_r_tail') is not None:
    d3_by_r_tail_out = {}
    for m in ('raw', 'gf', 'fixP'):
        d3_by_r_tail_out[m] = {}
        for r in R_SWEEP:
            arr = np.array([v['d3_by_r_tail'][m][str(r)][0] for v in V])
            d3_by_r_tail_out[m][str(r)] = {'median': float(np.median(arr)),
                                            'iqr': [float(np.percentile(arr,25)), float(np.percentile(arr,75))],
                                            'boot_ci': _boot(arr)}
```
и добавить в `layers_out[str(l)]`:
```python
's_l_tail': ({'median': float(np.median(sl_tail_arr)),
              'iqr': [float(np.percentile(sl_tail_arr,25)), float(np.percentile(sl_tail_arr,75))],
              'boot_ci': _boot(sl_tail_arr)} if len(sl_tail_arr) else None),
'a_vs_ESG_cos_tail': aev_tail,
'id_cos_JR_Jtail': idcos,
```

---

## Концептуальная проблема: `a_vs_ESG_cos_tail ≈ 1` не гарантировано

Docstring: «Expectation: `a_vs_ESG_cos_tail ≈ 1` on EVERY layer (chain rule of the readout itself)». С текущим кодом это **не следует**.

- Инъекция `Jn = JR / ||JR||` (строка `Jn = JR / ...` в `doc_layer_metrics`) — **full-chain** направление.
- Forward — **tail**.
- Значит `a1[i,j] = J_tail[j] · JR[i] / ||JR[i]||` — **не** `N(ESG_tail)_ij = J_tail[j] · J_tail[i] / ||J_tail[i]||`, если только `JR[i] ∦ J_tail[i]`.

При matrix cos = 0.095 на L6 в среднем строки не параллельны. Единичный +1.5% в (t=62, i=j=0) означает, что **эта конкретная** строка почти параллельна — но не более. По 128×128 это усреднится.

**Следствия:**
- С fix’ом знаменателя и без fix’а инъекции `a_vs_ESG_cos_tail` на L6–L18 будет **< 1** (может быть, 0.6–0.9, но не 0.998).
- Verdict `a_vs_ESG_cos_tail_gt_0.99_all_layers` будет **False** — и это правильный результат (он показывает, что инъекция не tail-согласована), но он **не** «подтвердит chain rule».

**Два чистых варианта:**

**(A) Оставить JR-инъекцию.** Переименовать ожидание: «a_vs_ESG_cos_tail — не fit-контроль, а residual of JR-vs-Jtail misalignment». Убрать `gt_0.99_all_layers` из verdicts (иначе False).

**(B) Переключить инъекцию на `Jtail[i] / ||Jtail[i]||`.** Тогда `a1 = N(ESG_tail)` тождественно, и `a_vs_ESG_cos_tail ≈ 0.998` на всех слоях — как обещает docstring. Плата: pipeline перестаёт совпадать с E-32r.2/E-32r.3 по направлению инъекции (delta for `s_l`, `slope_ratio`, `quad_frac` тоже поменяются), нарушается comparability.

Для текущей задачи (показать, что `a_vs_ESG_cos` — desync, а не физика) достаточно **(A)**. Но docstring и verdicts надо привести к (A), а не к ожиданию (B).

---

## Не-баг, но стоит поправить перед прогоном

**Resume hazard.** `exp32r3_checkpoint.pt` (от E-32r.3) читается как fallback. Если он найдётся (старая раскладка `/kaggle/input/<slug>/results/`), его `per_doc_layer` не имеет tail-полей → после фикса NameError упадёт на `v['s_l_tail_median']` в агрегации. Либо удалить fallback (пусть пересчитывает с нуля), либо добавить проверку схемы:
```python
if per_doc_layer and 's_l_tail_median' not in next(iter(per_doc_layer.values())):
    print('[WARN] old-schema checkpoint, recomputing'); per_doc_layer = {}; cache = {}
```

**`id_cos` агрегация.** `id_cos_JR_Jtail_median` в `per_doc_layer` есть, в `layers_out` не попадает. Стоит добавить (п.2 выше), это ключевой desync-guard.

**`_agg` на `d3_by_r_tail`.** `v['d3_by_r_tail'][m][str(r)][0]` — `[0]` это median, упаковка `[median, [p25, p75]]` — ок, но если `d3_by_r_tail` None в части документов (старый resume), `_agg` не сработает — `if V[0].get('d3_by_r_tail') is not None` обязателен (в п.2 есть).

---

## Итог

Прогон в текущем виде не стартует: 4 hard crashes (NameError + 3 KeyError). После их фикса — запустится, но `a_vs_ESG_cos_tail` на L6–L18 будет **не** ≈ 1 (инъекция всё ещё JR-направлена). Это не bug, а несоответствие docstring ожиданию; либо понижать порог в verdicts, либо менять инъекцию на Jtail (теряя comparability с E-32r.2).