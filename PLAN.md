

## GPU-аудит: какие эксперименты требуют re-run (запрошено пользователем, 2026-08-31)

- [x] `docs/gpu_restart_audit.md` (commit 8a4638a) — полный аудит по каждому эксперименту: замысел / код / результаты / вердикт + итоговая таблица.

### Результаты аудита (кратко)
- [x] **exp4_k_sensitivity — БРОКЕН**: `jacobian_bridge(J, J, E, E)` (self-bridge, 4B не загружен) → все значения 0.0. Не проверяет то, что должно. **Top GPU re-run** (загрузить 4B, `bridge(J_L,J_S,E_L,E_S)`, K до 256, n≥4).
- [x] **exp11 (defrag)**: `recon_fidelity` РАСТЕТ с r (19→31→48) — likely bug (defrag-basis не clean top-r SVD?); + exp11a K=32 vs exp11b K=256 mismatch (audit-«contradiction» partly artificial).
- [x] **exp16_readout_pull**: `delta_W` идентична при alpha=0.1/0.3/1.0 (14.276, 4.55 0.000000rac) — pull насыщается; investigate clamp/bug (CPU) → re-run.
- [x] **exp13/15/6**: `jacobian_bridge` **1.0-sentinel** (empty valid mask) — ранние пары = 1.0 (not data) → monotonicity −0.894/−0.949 identical = artifact rank-ties. **Paper-fix**: exclude sentinels from monotonicity/Spearman/bridge-mean. (overlap_hidden exp13 — good, no recompute.)
- [x] **exp12**: quantization destroys both variants (KL~15) — «comparative destruction», не «map preserves». Paper re-frame (или re-run с меньшим ε).
- [x] **exp17 power-up**: pending (500 steps, lr 3e-3, dual-cadence) — committed d3b69cf, blocked on Kaggle weekly 30h quota.
- [x] **OK (no re-run)**: exp1/3/5/6b/7/18/19, exp14 (add params to JSON), exp10a/d, exp2 (optional full 32×24 map).

### GPU-очередь (после сброса quota, ~8–10h total)
1. exp17 power-up (built, needs slot)
2. exp4 (fix + 4B, K→256, n≥4)
3. exp11 (debug 1 layer CPU → r-sweep 1..256 K=256 n=8)
4. exp16 (post-fix alpha-sweep, 0.8B only)
5. exp2 (full 32×24 map n=8/K=64)
6. optional: exp13/15 24-layer, exp6, exp10b n=8

**Paper-only fixes (no GPU, сделать сейчас)**: exclude 1.0 sentinels (exp13/15/6); exp11a vs 11b K-mismatch note; exp12 re-frame; exp14 add params.
