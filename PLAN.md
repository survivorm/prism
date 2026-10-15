# PLAN — E-33r.2: диагностика «почему gf стабильнее raw»

## Запрос (фиксация)
docs/Exp33r2_errors.md (ревью 2026-10-08): A (fix-p0 вместо median), E ( TZ-формула с RAW-референсом),
F (cross-P, не self-overlap), B (разделить E_alpha и B_snr), CLI --diagnostics/--alpha-sweep.

## DONE
- [x] A: p0 = pair-mean (p_t + p_{t+Δ})/2 на пару — argmax-H и median обе деградативны (канонические p0 не существуют)
- [x] E: J_a = (1−α)J + α·PJ, референс v_raw(t) (α=0 ≡ raw, α=1 = angle(v_gf(t+1), v_raw(t)), НЕ dgf_1)
- [x] F: by_r — cross-P overlap Ω_raw/Ω_gf/Δ = ‖U_gf(t+Δ)ᵀU_raw(t+Δ)‖²_F/r, s ∈ {t−Δ..t+Δ}∖{t,t+Δ}, |s−t| ≤ 4
- [x] B: E_alpha отдельно; B_snr_raw/B_snr_gf (σ16/σ17) только в diag-блоке
- [x] CLI: --diagnostics (алиас --diag), --alpha-sweep
- [x] OOM-фикс: 1 слой в памяти (33GB → ~6GB), aggregate_layer вынесен, depth via depth_keep
- [x] selfcheck OK (2026-10-08): p_apply, P kills uniform, rank-1, Ω=r/D check, sentinel, fix-p0, E_alpha
- [x] full 32-doc run завершён (pair-mean, diag, α-sweep), JSON в results/exp33r_gauge_anatomy.json
- [x] meta: e33r2_refit_note (история median→argmax-H→pair-mean), A_F_note, p0_method='pair_mean'

## DOING
- [x] strict-refit (p0='fix', c diag) 10.14 10:18 → results/exp33r_gauge_anatomy_strict.json
      (E_alpha(fix): 0.289/0.289/0.309/0.849/1.565 — переход тоже α 0.5→0.75; D_omega/E_fixed_p на месте;
       A_fp НЕ агрегирован: batch_metrics пишет A только для pair_mean — не блокирует, добавить по требованию)
- [x] правки из docs/Exp33r2_errors.md: n = T−dt, docstring survive_frac, отступ 5b; selfcheck OK
- [x] pair-mean full run 10.14 20:26 → 10.15 00:16 → results/exp33r_gauge_anatomy.json (pair_mean, diag, α-sweep)

## ДОП (результаты, L23 medians)
- A_pair_mean = 0.256/0.248/0.251/0.256 (Δ=1/4/8/16) — структурированный, ≈ raw (0.289), далёк от случайного 1.5665 → P-as-filter отвергнут
- E_alpha (pair-mean): 0.2892/0.2890/0.3124/1.1895/1.5665 — резкий переход α ∈ (0.5, 0.75): структуру убивает только сильный фильтр
- C_gain(r16) ≈ −1.27 (все Δ); C_gf ≈ 1.5665 ≈ случайный (1.5673); B_snr raw/gf 1.035/1.13; B_rho 0.17–0.36
- verdict_gains: L6 −0.004 / L12 −0.009 / L18 −0.038 / L22 −0.091 / L23 −1.270 — мягкий градиент, скачок на L23
- корреляции gain↔gs_exact 0.017, kill↔gs_exact 0.236 — слабые
- Итог: лист L23 живёт в gauge-компоненте (α-зависимый), механизм = per-position вариация p_t (пороговый эффект)

## NEXT
- [x] отчёт пользователю (10.15)
- [ ] по требованию: A_fp (fix-p) агрегация в batch_metrics + дописать в docs/Exp33r2.md (pair-mean, TZ-формула E, cross-P F)
- E-33r.2 закрыт. Дальше по PLAN: E-32 (Hessian) — когда укажешь.

## Границы (не менять)
- exp28/exp34/exp30/exp27.5/exp35/exp31 скрипты, calibration.py, кэш exp27b — не трогать
- docs/Exp33.md — не менять (только Exp33r2.md)
- paper.md — только по явному указанию
- pair-mean JSON — не перезаписывать до завершения strict-refit
