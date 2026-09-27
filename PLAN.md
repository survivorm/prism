# Prism — PLAN

## Запрос (фиксация)

Выполнить ТЗ `docs/Exp27.6.md`: P0-приложение к Exp 27.5 — domain isolation, block
structure и null-baselines для вердиктов 27.5. 8 задач T1–T8, постобработка
`results/exp27_jcache.pt` + `results/exp27_pair_matrix_S64K64p0.pt` +
`results/exp27_5_postprocess.json`. Без GPU, ≤30 мин CPU. Выход:
`results/exp27_6_isolation.json` + PNG `results/exp27_6_bock_4x4.png`.
Скрипт: `experiments/exp27_6_isolation.py`.

Входы (проверены на диске):
- `results/exp27_jcache.pt` (1.65GB, JR/P128/H, ключи (dom, si, l, S, p)) — K^G, T8, T7
- `results/exp27_pair_matrix_S64K64p0.pt` — 6× (32×32) **K^F** (JSON — только summary)
- `results/exp27_5_postprocess.json` — перекрёстная сверка
- dip-модуля нет в venv → T5 вариант B: свой Hartigan dip + bootstrap-null по
  unimodal-распределению (Beta, matched mean/std)

Домейн-порядок в pair_matrix: wiki=0-7, code=8-15, chat=16-23, math=24-31
(проверить по n_per_domain/меткам в скрипте).

## Exp 27.5 (COMPLETE, 09-23)

- [x] код: experiments/exp27_5_postprocess.py (по docs/Exp27.5.md, seed=42, B=1000)
- [x] review против ТЗ: 2 бага (gap-null вырождение: те же 32 ротации на 100 итераций;
      R без manual_seed) — починены; остальное ТЗ-совместимо
- [x] прогон CPU 2 мин → results/exp27_5_postprocess.json
- [x] findings: docs/exp27_findings.md (секция Exp 27.5)

## DONE / DOING / NEXT

- [x] ТЗ 27.6 прочитано, входы инвентаризированы
- [ ] **DOING** T1: T_domain (K^F и K^G) для {all, no_math, no_chat, no_code} × 6 слоёв
- [ ] T2: Bock 4×4 на K^G (диагональ per_domain + off-diag 6×6) + PNG 4 панели
- [ ] T6: eigengap спектра K^F и K^G (λ1..λ31, gap_ratios 2/3..5, r_eff, has_structure)
- [ ] T3: bootstrap CI на T_domain и T_ent (B=1000, ресэмплинг сэмплов, quartile-метки из P128)
- [ ] T4: permutation-null ARI (кластерные метки фиксированы: аггломеративная k=silhouette_argmax из 27.5; B=1000 перемешиваний доменов)
- [ ] T5: dip (Hartigan, свой) + bootstrap-null unimodal (B=500) для K^G off-diag
- [ ] T7: Spearman ρ(layer, C_a) pos-pk/same-pk-early/raw, bootstrap CI + perm-null (n=6 слоёв, B=1000)
- [ ] T8: per-sample ковариаты (E, logE, r_eff, r_stable, ‖h‖, g_uni, H; σ_z=not_cached)
- [ ] JSON + PNG + вердикты pass/fail/inconclusive по T1–T8
- [ ] findings: дописать в docs/exp27_findings.md, обновить вердикты 27.5 (п.13 ТЗ), коммит

## Границы (не менять)

- S=64, K=128, pos=0, seed=42, B=1000 — по ТЗ
- Не пересчитывать кластеризацию при бутстрепах (T4: метки фиксированы)
- Resonc: T3 ресэмплинг **сэмплов**, не пар
- T6 gate: has_structure = ∃i∈{2,3,4} g_i≥1.5; нет структуры → T3/T4/T5 всё равно
  считаем (дёшево), но помечаем silhouette-блок 27.5 как «спектр плоский»
- jcache грузим один раз, не копируем в repo (в .gitignore)
