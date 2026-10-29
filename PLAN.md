# PLAN: E39_fixes — довести до конца (частично реализовано) — **DONE** (commits 354e64d, 2391fd6)

## Запрос (фиксация)

`docs/E39_fixes.md` = две части: (1) ревью E-39b (→ `exp39c_pre_kl_surgery.py`), (2) ТЗ:
правка E39.md §12 + `exp39l_drift_behavior.py` (non-tautological design).
Пользователь: «бери docs/E39_fixes.md и исправляй», «частично может быть уже
реализовано, но не полностью». Довести до конца: закрыть разрывы, прогнать,
обновить paper §5/§6, commit.

## DONE

### Part 1 — `exp39c_pre_kl_surgery.py`
- [x] Фикс #1 (unpack): `kap_new, kap_ci, _ = _kappa_dryrun(...)` → dict-разбор: функция возвращает
  dict из 8 ключей, 3-tuple unpack → ValueError; скрипт не запускался никогда)
- [x] Фикс #2 (`@torch.no_grad()` на `main()`): `W_lm` — parameter с requires_grad; без no_grad
  `np.percentile(dratio, 90)` падает (`Can't call numpy() on Tensor that requires grad`),
  `float()` ворнит. Инференс-only → no_grad на main
- [x] Production run (`.venv/bin/python`, `HF_HOME=.hf_cache`, CPU, 170s) →
  `results/exp39c_pre_kl_surgery.json`. **Gate: FAIL** — mean KL (real tokens, t≥0) =
  **4.986 nats/token** >> 0.05 (per-doc 3.56–6.24); κ' dry-run = 0.00416 (consistent с E-39a
  0.0042); m_k_norm Frobenius preserved (33.8454→33.8454); v_new = bottom_16_right_svs_of_U
  (no true null space, r_obs=16); row_diag (space: W_lm): mean 0.983 / p90 1.328 / max 1.402.
  Stale `results/archive/exp39b_kl_surgery.json` (mean KL 8.89, pre-rename) — артефакт, не трогать

### Part 2 — E-39l v2
- [x] `exp39l_drift_behavior.py`: фикс n_measured — было n_measured = n_interpolated = n_joined = 10;
  ТЗ §1.2/§3.4: «9 из 11 точек — интерполированы по 6 узлам E-16b» → n_measured = join-точки,
  совпадающие с узлами E-16b (s_frac 0.1 → step 100, 1.0 → step 1000) = **2**, n_interpolated = **8**
  (совпало с примером ТЗ JSON). Rerun: n_measured=2, n_interpolated=8
- [x] Anti-control: реализация уже по псевдокоду ТЗ §3.3 (only `ρ_anti > 0.3`, diagnostics в caveats);
  решение о расширении (OR |ρ_all| > p95) отменено — ТЗ самодостаточен, данные дают
  `anti_control_passed=false` (честный исход, diagnostic уже в caveats)
- [x] Rerun → `results/exp39l_drift_behavior.json` + `results/exp39l_scatter.png`:
  `verdict = inconclusive (drift non-monotone)`; ρ_res=+0.031 (p=0.933), ρ_all=−0.758 (tautological),
  ρ_anti=0.236 (≤0.3), monotonicity_ok=false (0.700→0.697→1.197→1.152), n_joined=10.
  Все поля spec §3.4 на месте (metrics: rho_residual, rho_all_trivial, rho_anti_control,
  anti_control_passed, monotonicity_ok, n_joined, n_measured, n_interpolated, missing_steps,
  s_frac_excluded; top: verdict, verdict_confidence, e16b_source, points, caveats, scatter_plot)
- [x] E39.md §12 → v2: §12.1 (почему v2 — тавтология v1), §12.2.1–2.4 (определения), §12.3
  (протокол + fail-loud), §12.4 (таблица интерпретаций + правило: predictive запрещено без
  anti_control/n_measured/monotonicity), §12.4.5 (s=0 semantics), §12.4.7 (обязательные проверки),
  §12.5 (sources of truth), §12.6 (limitations для paper §6), §12.7 (реализация и исход)
- [x] E39.md §13.2 + §15 ASCII: v1-формулировки «E-39l решает, работает ли flatness-loss» /
  «может заменить E-39d/e, если ρ сильная» → согласованы с v2 (замена возможна только при
  `drift_predictive_beyond_trend` + anti-control; текущий исход — `inconclusive`, замена по E-39l нельзя)
- [x] paper.md §5: строка «Не установлено [И] | Leaf drift предсказывает behavioral деградацию
  (E-39l)» + механизм (ТЗ §5.1 — строка в §5, НЕ в §4)
- [x] paper.md §6: пункт 17 (limitation E-39l из ТЗ §12.6: ρ_all тривиален; ρ_res — не причинность;
  каузальный тест = E-39d/e)

### Верификация
- [x] grep «Goodhart geometric» / «ρ < −0.7» в E39.md §12 — 0 (только упоминание удалённого ярлыка
  в §12.1/§12.4 — легитимно)
- [x] exp39l JSON: все поля spec §3.4 (см. выше)
- [x] exp39c_pre JSON: gate (metric/threshold/value/pass/basis), m_k_norm (before/after/frobenius_preserved),
  v_new_subspace, row_diag (space/mean/p90/max), kappa_dryrun (median/consistent),
  kl_nats_per_token (mean/per_doc_real/max/p99), runtime_s
- [x] JSON git-ignored (`results/*.json`) — артефакты локальные, как у остальных экспериментов;
  PNG (scatter) в коммите

## Границы (не менять) — соблюдены

- [x] Результаты E-39a/E-38i/E-16b не пересчитывать; drift(s) — из `results/exp38i_factorization.json`
- [x] `E38I_STEPS` и пути файлов не менять (ТЗ §3.5)
- [x] Paper §4 карточки E-* не трогать (кроме §5/§6 строк)
- [x] E-39b (настоящий, GPTQ) — не реализовывать (Kaggle, kernel-exp39b)
- [x] E39.md §0–§11, §13–§15 — не трогать (кроме §12 и ссылок на него, если есть)

## NEXT

- [ ] paper_add шаги 4–8 (пауза; возобновить свежими якорями — см. PLAN paper_add в git log)
