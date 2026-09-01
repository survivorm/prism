# GPU-аудит: какие эксперименты требуют re-run (запрошено пользователем, 2026-08-31)

## DONE

- [x] `docs/gpu_restart_audit.md` (commit 8a4638a) — полный аудит по каждому эксперименту: замысел / код / результаты / вердикт + итоговая таблица.
- [x] §L full audit — all 34 scripts re-read (exp2a/4_top_p/5_layer/8/9/10c/11c/12/17-helpers).

### Результаты аудита (кратко)
- [x] **exp4_k_sensitivity — БРОКЕН**: `jacobian_bridge(J, J, E, E)` (self-bridge, 4B не загружен) → все значения 0.0. **Top GPU re-run** (4B, `bridge(J_L,J_S,E_L,E_S)`, K→256, n≥4).
- [x] **exp11 (defrag)**: metric bug FOUND+FIXED (v1 projected raw J vs J_p); exp11a K=32 vs 11b K=256 = different measurements. 11c: defrag flattens bridge (negative result).
- [x] **exp16**: v1 = design bug (α = trust-region cap, not dose). v2 rebuilt: one unclamped run, dose curve, α* = knee (max step with Δbridge≥tol; null if none).
- [x] **exp13/15/6**: `jacobian_bridge` 1.0-sentinel (empty valid mask) → monotonicity −0.894/−0.949 identical = rank-tie artifact. Paper-fix: exclude sentinels.
- [x] **exp12**: both schemes destroyed (KL~15) — «comparative destruction». Paper re-frame.
- [x] **exp17**: OOM fixed (A4 chunked); SLOW (2nd-order HVP refresh, tail-grows). 7-pair full = 14 pair-runs ≈ 40–45h ≫ 12h kernel.
- [x] **OK (no re-run)**: exp1/3/5/6b/7/8/9/10a/d/10c/18, exp14 (add params to JSON), exp2a (code fixed + self-check PASSED; CPU run pending).

## DONE (cont.)

- [x] **Consolidate `docs/gpu_restart_audit.md`** (user 2026-09-01): 333 LOC, three verdict tables with contradictions (exp11/16/17), two timestamps (08-31 v1, 09-01 §L), stale exp19 «176 pairs» (stage0 cache = 7 pairs), exp17 verdict conflicts with `docs/HANDOFF_exp17_powerup.md` §4 (which says push full 7-pair; §L math says it can't fit one 12h kernel).
  - Merge into one doc: header → one verdict table (all 34) → per-exp sections → paper-only fixes → GPU queue.
  - §L (09-01) is the newer word; v1 (08-31) stale rows get replaced.

## NEXT

- [ ] `docs/HANDOFF_exp17_powerup.md` §4 stale: says "push full 7-pair, expect ~12h" — math in §L says 14 pair-runs ≈ 40–45h; push should be `--pairs 4_3` + single order (~9.2h). Update §4/§6 Q1 after user decision.

### GPU-очередь (после сброса weekly 30h quota; ~13–15h total, fits one window)
1. exp17 power-up: push `--pairs 4_3` + single order (~9.2h; 12h kernel cap) — built, quota-blocked.
2. exp4_k_sensitivity: fix + 4B + K→256 + n≥4 (~1–2h).
3. exp16 v2: one unclamped run (~5 min T4).
4. exp19 v2: 4 scenarios, dose = exp16 v2 knee (~10 min; depends on 3).
5. exp2: full 32×24 map n=8/K=64 (~1h).
6. Optional: exp13/15 24-layer, exp6, exp10b n=8, exp4_top_p (~1h), exp12 ε-sweep.

**Paper-only fixes (no GPU)**: exclude 1.0 sentinels (exp13/15/6); exp11a/11b K-mismatch note + 11c bridge-flattening (defrag = output-faithful but bridge-destroying); exp12 re-frame; exp14 add params; exp2a note (CKA null inflated when N≪D); exp17 topk_report part of v2 suite.
