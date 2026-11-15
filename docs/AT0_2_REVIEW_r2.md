# AT0_2_REVIEW_r2.md — ревью AT0_2 (T0.2 «Форма профиля при фиксированной сумме»), раунд 2

Дата: 13.11.2026. Ревьюер: субагент (pi -p, obleth/ngenix-llm:low, read-only, пакет
/tmp/at02_review_pkg.md: §A код дословно 820 строк + §A2 пре-рег + §A3 раунд 1 + §B факты
с sha256 + §C ответы + §P механика + §D формат). Маркер AT02_R2_DONE в ответе — сверен
grep'ом до сохранения (шаблон REVIEW_PROMPT_TEMPLATE.md, правило 1).

Первый запуск пакета умер на output-token-лимите (truncation) → кросс-бриф по правилу 3
(точечные пункты K1–K5, лимит 2000 символов) — ответил в лимите.

## Ответ ревьюера (дословно)

```
# AT0_2_REVIEW_r2.md
Ревьюер: внешний ревьюер исполнения/кода (раунд 2, точечные сверки §A↔§A2↔§B)
ВЕРДИКТ: PASS_WITH_MINORS
N1: закрыто. Блок `if smoke:` строки 718–726 (return 726) — ДО первого `open(..., "w")` (строка 758, store). До return единственный open — read-режим (599, gen_meta); store (meta.n_docs = n_docs = len(docs) = 96) строится только в полном прогоне, где len(doc_names)=96 — в smoke-режиме meta ни в какой артефакт не попадает.
N2: да, 3 копии совпадают: хедер §A (43–51) ↔ §A2 «## Вердикт» ↔ aggregate() (497–509): ΔMIN не_testable (D_real < пол-ширины CI, проверяется первым) → refuted (D_real > CI_null верх) → confirmed_anomaly (< CI_null низ) → confirmed. «p_agg» в §A: 0 вхождений.
N3: да. K1 — строка 493 `d_real, ci_real = boot_ci(Dc)` (Dc = per-doc mean по L, строка 491); K2 — строка 533 `mv, civ = boot_ci(doc_means)` (doc_means — per-doc mean, 530–532); D_null аналогично (494, Dns = per_doc_n means, 492).
N4: да. claim["aggregates"]=agg (784) — передаёт весь dict; финальный logline (809–811): D_real_mean/D_real_ci/D_null_mean/D_null_ci/frac_cells_p_lt_alpha/verdict — все есть в out-дикте (539–550); smoke-logline (723–724) — подмножество тех же ключей; manifest (789–802) ключи aggregate не трогает. «frac_docs_p» в §A: 0 вхождений.
N5: да. K5: 643 `expected = n_docs * len(LAYERS) * len(OBJECTS)` (=1440), 685 `n_absent == n_l1_absent_exec * len(OBJECTS)` (=6), assert 687. Сиды 2026+doc_idx·1009+s (323/354/276), fourier +1000 (414, B=3), boot seed 117/B=1000 (102–103). PC 1e-6/2e-4 (109–110), assert pc_ok (712) ДО aggregate (715). G7-inv — только perm-цикл (init 318, update 328), AR-цикл (376–404) r_inv_max не трогает. Fail-loud: assert g2_ok/gram/inv/pc + ValueError c0≤0 (191), AR σ²<−1e-8·r0 (265), N<25 (672), G1 (196) — тихих fallback'ов нет.
НОВЫЕ:
- [MINOR] G2 в smoke: строка 654 принудит l1_bad=False при smoke, но 684 считает n_l1_absent_exec без учёта smoke → если бы smoke-док был L1-degraded, assert g2_ok сработал бы (сейчас латентно: wiki_0/math_3 не degraded); чинить: `and (not smoke)` в 684 или считать по реально записанным absent-клеткам.
- [COSMETIC] Не закрываются file-хэндлы: 599 `json.load(open(...))`, 793 `open(OUT_JSON, "rb")`; плюс терминология «p_doc < 0.05» (хедер, строка 50) vs «p_cell < 0.05» (§A2) при ключах frac_cells (per-cell); чинить: with-менеджеры / единый термин (не блокирует).
VERDICT=PASS_WITH_MINORS
AT02_R2_DONE
```

## Ответы исполнителя на НОВЫЕ (раунд 2)

1. **[MINOR] G2-smoke: принято, исправлено.** Строка 684: `n_l1_absent_exec = sum(1 for
   nm in doc_names if (not smoke) and nm in l1_degraded and l1_degraded[nm])` — ровно
   предложенный ревьюером фикс (guard `(not smoke)`). В полном прогоне поведение
   идентично (smoke=False); в smoke — согласованно с l1_bad=False (строка 654).
   Повторный smoke после фикса: exit 0, G2 30/30 ok=True, «артефакты не пишутся»,
   mtime results/AT0/ не изменились.
2. **[COSMETIC] терминология: принято, исправлено.** docs/AT0_2_prereg.md: «p_cell < 0.05»
   → «p_doc < 0.05» (p_doc — имя переменной в коде, значение считается на клетку (doc, L);
   хедер был верен). File-хэндлы: **отклонено** — CPython закрывает по refcount сразу
   после выражения, риска утечки нет; with-менеджеры в однострочных json.load/read —
   косметика поверх косметики (ревьюер сам пометил «не блокирует»).

## Итог

Раунд 1 (BLOCK): N1–N5 закрыты (см. §C пакета и docs/AT0_2_selfreview_r2.md).
Раунд 2: PASS_WITH_MINORS, оба MINOR/COSMETIC обработаны (исправлены / отклонено с
аргументом). Само-ревью исполнителя: docs/AT0_2_selfreview_r2.md (VERDICT=PASS).
Дальше — полный прогон: `./run_one.sh at0_2_full .venv/bin/python experiments/AT0_2_shape.py`.
