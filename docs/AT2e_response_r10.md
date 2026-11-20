# Ответ на ревью r10 (ответка r9 + summary rev4): PASS — применено

Все 4 пункта + 3 нитки — применены тем же коммитом. Коротко, где что:

## R1 — S_floor-вакуум: аналитическая граница + флаг (TZ §3.5, код r4)

Принято. Формула: если S_floor,k == 0 (оба направления дают бит-идентичные логиты на a_floor:
q_k = κ_k = 0) → **S_floor,k := 2·q_quant,k/floor**, q_quant,k = (2⁻²³·max|z_{t+k}|)²·p_max,k —
аналитическая граница из квантования логитов: |2κ_k−q_k| ≤ 2κ_k + q_k ≤ 2·ε_z²·p_max,k
(ε_z = 2⁻²³·max|z|; κ ≤ ½·ε_z²·p_max — Hoeffding для log-плотности, |log-ratio| ≤ ‖δ‖∞;
q ≤ ε_z²·p_max — та же конструкция, что floor-квант r2 M4, per k). Флаг `S_floor_degenerate_k`
per cell в артефакте — «запас не измерен» отличимо от «запас измерен и мал».
Масштаб: при T4-флоре 1.33e-11 граница ~2·2.7e-13/1.33e-11 ≈ 0.04 → порог 3·S_floor ≈ 0.12 < 0.5
→ правило к r3-поведению (s_k > 0.5) — консервативно, и теперь ВИДНО по флагу.
S16(г) проверяет ветку на синтетике.

## R2 — primary-only до конца (функция rev5, этот коммит)

Принято, сделано в rev5: `cells_p` теперь в ВСЕХ диагностических циклах — mono,
recovery_fallback_check, kappa_scope_counterfactual (+counts), tprime_gain_margin,
ascent_not_converged_n, tprime_early_stop_n, per_cell (ключи-дублики в r4),
linear_gate_decomp, ascent_table. Машинная видимость: `n_recovery_cells`
(len(cells)−len(cells_p)) + `global_recovery` в run_summary. Проверено на r3: no-op
(n_recovery_cells = 0), счётчики не изменились, детерминизм — идентичный вывод.

## R3 — flagged_k, не unresolved_k (функция rev5, этот коммит)

Принято: поле переименовано в `flagged_k` + `target_under_new_rule_M2_is_upper_bound: true`
у каждой recovery-цели с комментарием: без S_floor (r3-артефакт) флаг ≠ «нерешиемый»;
в r4 S_floor решит часть флагов, но клетка останется целью по genuine-нелинейности —
множество целей не меняется, статус «unresolved» не утверждается.

## R4 — recovery как readout природы κ-флага (TZ §3.13)

Принято, запинено в TZ §3.13: (а) механизм — genuine (~a³) → |2κ−q| ÷27 → pass, шумовой
(~floor) → fail — вторичный readout, НЕ правило решения; «все 5 пройдут» НЕ бюджетировать;
(б) тонкие denom-маржины (c6 28.0 vs 10 = 2.8×, c13 44.3 vs 10 = 4.4×; `denom_gate_vstar` —
q_t(v*), не медиана B₀) + заранее объявленный честный исход: «позиция не тестируема в линейном
режиме ни на одной доступной амплитуде» (предел разрешения), а не «recovery не помог».
S17 покрывает recovery-синтетику, включая N1-проверку (fallback НЕ срабатывает на recovery).

## Нитки (те же коммит)

1. **«v* = последняя принятая»**: TZ §3.4 — явно: v* = последняя ПРИНЯТАЯ точка (отклонённый
   trial хуже); argmax сохранён (0 ф.-э., страховка).
2. **ф.-э. accounting — один замеренный констант**: TZ §6 — единый набор: CPU: итерация
   28.1 s = 6.5 ф.-э. (замер r1), trial = 2 ф.-э. = 8.66 s; T4: итерация 1.48 s (canary r3:
   probe 0.80 + vjp-блок 0.68), 1 fwd = 0.20 s, r3 main = 170 ф.-э./мин (6800/2400 s —
   измерено). Бюджет r4 T4 = (5686 + ≤2320)/170 ≈ 47 мин + overhead ≈ 55–60 мин — из замеров,
   не из арифметики структуры. Константы дублированы в артефакт (`header.measured_constants`).
3. **min_full_gain_tprime_docs**: в функции rev5 — `tprime_gain_margin_components` теперь
   с `min_full_gain_tprime_cells: ["c1"]`, `min_full_gain_tprime_docs: ["wiki"]`,
   `max_needed_tprime_cells: ["c2"]`, `max_needed_tprime_docs: ["math"]` — экстраполяция wiki
   видима машинно.

## Коммит (один пакет, как в вердикте)

- `docs/AT2e_prereg.md` — TZ r6 (заголовок + ревизия-нот + §3.4/§3.5/§3.13/§4/§5/§6/§6b/§11/§13/§14);
- `experiments/at2/at2e_artifact_summary.py` — rev5 (R2/R3/нити 2-3);
- `docs/AT2e_response_r10.md` — этот файл.

Детерминизм сводки проверен (2 запуска — идентичный sha256), r3-артефакт intact
(sha16 34836e22b7bfd870), CJK-гейт чист.

Дальше — код r4 по TZ r6 (задача #80): κ S_floor+k-окно+R1-гард, backtracking+rel-stop,
ascent_tprime полный, recovery-ветка, 2 дерева, S15–S18 → смок → T4 main r4.
