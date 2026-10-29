# PLAN: доведение — E38-раунд (коммит) + paper [M-4] синхронизация

## Запрос (фиксация)

После E39_fixes (DONE: 354e64d, 2391fd6, e3e0ccd) и paper_add (DONE: e4993af,
3271991 — верифицировано по таблице ТЗ paper_add.md §1: 8/8 «выполнено») в дереве
остался незакоммиченный E38-раунд и stale [M-4]-вердикты в paper.md. Довести:
закоммитить, синхронизировать [M-4] (A.4/A.5 E38_paper.md), закрыть хвост D.2–D.5.

## DONE

- [x] paper_add верификация: G(l)/R_can/R_raw в §2.5, шаблон 7 полей + лестница
  каузальности в §3.2, §6 пункты 13–17, запрещённые термины — 0 вхождений
  (вхождения «gauge share» — ESG-метрика; «не причинный тест» — корректная
  отрицательная формулировка). PLAN stale «шаги 4–8 NEXT» снят
- [x] Коммит A (842afa9): E38-раунд — E38.md (+565, восстановление текста всех
  экспериментов — обязательство E38_fixes2), E38_fixes.md, E38_fixes2.md, E38+.md,
  E38_paper.md, ref-exp.md (карточки E-39a/E-39l + M-4-строка E-38c + D.2),
  exp38d/e/g/h (C-фиксы), results JSON ×3, kernel-exp38d; .gitignore
  (.hf_cache, logs, npz, archive/v2check)
- [x] Коммит B (bfacd27): docs/models/ (6-раундовый консенсус — source paper_add),
  docs/paper_add.md, experiments/exp39a_u_obs_rank.py (критично: закоммиченный
  exp39c_pre импортирует `_kappa_dryrun` из него — без коммита репо сломано)
- [x] Paper [M-4] синхронизация (A.4/A.5): §5 WZ-гипотеза, §6 п.8, proof map —
  «не прогонялся/мораторий» → «пройден (tail-only); мораторий снят для tail»
- [x] D.2 (уже был в рабочем diff, вошёл в A), D.3 (exp38i), D.5 (hi_common)
- [x] Коммит C: paper.md (3 строки [M-4]) + exp38i + hi_common + PLAN.md

## Границы (не менять)

- [ ] Paper §4 числа/вердикты — не трогать (граница paper_add; A.1/A.2/D.1 уже
  закрыты — соответствующих величин в paper.md v6.0 нет: 0 grep-хитов)
- [ ] A.3 (переименование `jlens_not_tail_modes` + перегенерация outcome) — НЕ делать:
  требует перегенерации JSON (Kaggle)
- [ ] C.13–C.14 (trivial_frac/−π в exp38d) — НЕ делать: по ТЗ E38_paper.md —
  после пере-генерации E-38d JSON (Kaggle); локально кодовые фиксы уже в diff
- [ ] Результаты E-38a–i не пересчитывать; E38.md — только уже восстановленный текст

## NEXT (открытые хвосты, не в этом раунде)

- **Параллельная сессия: E39l v3** (docs/E39l_fixes.md: anti-control →
  пермутационный тест на residual, NaN→None, fail-loud endpoint, monotone
  both directions, verdict_confidence по frac_measured) — В ПРОГРЕССЕ, не
  коммитить её файлы (exp39l_drift_behavior.py, E39l_fixes.md, scatter.png)
  до завершения; после — синхронизировать E39.md §12/§13/§15 и paper §6 п.17
  (ρ_res сейчас «+0.031, p=0.933» — сменится пермутационным критерием)
- E-39b (настоящий PTQ/GPTQ — Kaggle kernel-exp39b); E-39c (M_K-хирургия) —
  pre-gate FAIL (KL 4.986 > 0.05), серия заблокирована до решения по pre-stage
- E-39d/e — каузальный тест drift vs flatness (training)
- Tier 0.x/1.x/2.x (F_lt profile, κ_fixP, locking, master-curve) — по paper_add
