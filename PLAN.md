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

## DONE (09-02, follow-up «продолжай»)

- [x] Re-verified consolidation already committed (c8c2b14/fd4d6f8); file is single coherent doc (217→218 lines, 22 main-table rows).
- [x] **Discovered `results/exp6_l18_snr.json` MISSING from repo** (git history + results/ + kaggle_kernel/ all checked) — the paper's headline exp6 (L18-only SNR, ρ=+2.94) data was never committed → **unverifiable**, needs GPU re-run. Distinguished from exp6_kl_baseline (separate experiment, −0.094).
  - Added: main-table row `exp6_l18_snr (paper headline)` + dedicated `### ⚠️ exp6_l18_snr — ДАННЫЕ ПОТЕРЯНЫ` section + paper-only item 8 + GPU queue item 6 (**mandatory**, not optional) + Total updated to ~9–10h.
- [x] **Paper-only item 5 (K/N-stale)** — corrected stale K/N: exp14 K=256→**K=64** (JSON n=8/K=64), exp11a K=256→**K=32** (N=4), exp12a/b N=4/K=32→**N=8/K=64**; exp11b/c/exp10c K values confirmed OK.
- [x] **GPU queue split** GPU (items 1–6, exp6_l18_snr now mandatory) from CPU/local-light (exp11 CPU-verify, exp2a run, **exp14 recompute** params+sentinel-stats, **exp13/15/6 sentinel-excluded recompute** → paper items 1,8).
- [x] «optional» rule honored: exp13/15/6 reruns, exp10b, exp4_top_p, exp12 ε-sweep, exp11c n=8 stay **optional**; only exp6_l18_snr promoted to mandatory (data lost, not just weak).

## DONE (09-02, paper re-correction — applied & committed)

- [x] **Scoping fact (verified 14-01):** `docs/paper.md` (v3.3) contains **exp 1–12c (§5–9) + 13/14/15 (§11 «Доменная специфичность, глубинная структура, фазовая робастность»)**. exp16/17/18/19 NOT in paper (audit + JSON only). FIRST PASS wrongly claimed 1–12c only — CORRECTED: §11 is in-paper, so 13/14/15 = fix numbers (done, c07c138), 16–19 = write sections (future).
- [x] **exp12a corrected:** header N=4→N=8, K=32→K=64; main table (uniformity 0.91/0.85/0.86 → 0.92/0.94/0.97; KL_g 7.04±0.37/7.46±0.47/6.42±0.97; KL_d 7.55/7.80/6.69; ratio 1.07/1.05/1.04×) + ε-sweep line from current JSON. Params note (script default N=4/K=64 out of sync).
- [x] **exp12b/12c params notes added** (tables = JSON N=4/K=32; script now K=64 → next run overwrites). exp12b/12c numbers themselves verified = match JSON, no change.
- [x] **exp11b v1-metric note added** (saved JSON = old metric `‖U_rJ_r−J_p‖/‖J_p‖` grows with r = artifact; fixed in c904db3 to `‖U_rU_rᵀJ_p−J_p‖/‖J_p‖` monotone-decreasing = 0 at r=K; table 14–38% = v1, re-run flips trend). Speedup column unaffected.
- [x] **exp6 (in-paper) sentinel note added** (L_bridge=1.00 early rows = sentinel from `prism/bridge.py:31`; ρ=-0.09 = with-sentinel; sentinel-excluded ρ → CPU item).
- [x] **exp7 CKA null-caveat added** (N≪D: indep 64×128 null ≈0.34; measured 0.0003–0.0007 far below → contrast stands).
- [x] **exp12 re-frame note added** (map = minimal-sensitivity direction; Δh 0.22–0.63 vs 0.82–2.52; both schemes destroy PPL 10⁶–10⁸; not «map preserves the model»). exp11c table verified = matches JSON (no change).
- [x] **Audit paper-only section:** scope fact fixed (13/14/15 in paper §11) + items 9/10 added (exp13/15 sentinel note; exp14 done: K=64, gauge_share non-monotone peak 0.48 L3, PR non-monotone, table sep) + CPU queue line (exp14 recompute, exp6 in-paper sentinel-excluded, exp11b recompute, exp2a run).
- [x] **§11 exp13/14/15 fixed against JSON (c07c138):** exp13 transition L12→L18 → gradual L12→L18 / jump L18→L23 + sentinel-cause note; exp14 K=256→K=64 + gauge_share «0.31→0.08» → NON-monotone (peak 0.48 L3) + PR non-monotone + broken table sep (3 dashes/6 cols); exp15 verdict-4 «scheme D holds higher bridge» → DEPTH CONFOUND (D 3rd anchor = 0.8B L13 shallower than A/B/C L15/16/14) + sentinel-cause note. §10: defrag 6–18×→2.7–18.9×; «map works for quantization»→«map = min-sensitivity direction (same KL, lower Δh)»; gauge-share row → «non-monotone (peak 0.48 L3)» (reattributed Exp13,14→Exp14). exp12b glued >Оговорка→blockquote; exp12c «preserves structure»→«Jacobian direction, not function»; hypotheses table 7-dash→5-dash sep.
- [x] **exp2a_cka_identity.py** pre-existing uncommitted fix: old `CKA(X,X)` same-tensor (=1.0 by construction, validated nothing) → disjoint sample groups (real consistency check) + `_selfcheck()` construction invariants (identity/scale/translation/perm). Self-check PASSED. n-samples 5→8.

## NEXT

- [x] **Committed** (e506bc0): paper.md + audit + exp2a script + PLAN.md (4 files, 122+/55-).
- [ ] **Blocked on GPU:** exp16/17/18/19 «paper sections» (write — not in paper yet); exp11b re-run (flip fidelity trend); exp12a N=4/K=32 rerun; exp12b/12c K=64 rerun; exp6_l18_snr data (mandatory, data lost).
- [ ] `docs/HANDOFF_exp17_powerup.md` §4 stale: says "push full 7-pair, expect ~12h" — math in §L says 14 pair-runs ≈ 40–45h; push should be `--pairs 4_3` + single order (~9.2h). Update §4/§6 Q1 after user decision.

### GPU-очередь — ВСЕ kernels wired in pipeline (commit a79466d; push after weekly-quota reset)

**All 16 kernels registered in `kaggle_kernel/push_all.py` + built to local `kernel-*/` (end-to-end build verified via `.venv`). `build_kernel.py` unchanged — the work was the `push_all` registry + 2 experiment scripts. Pushes no-op'd in this env (no Kaggle creds). `id`/`slug` PROVISIONAL for never-pushed kernels — kaggle derives canonical slug from title on first push; update `id`/`slug` after (re-push 409 otherwise, cf exp17).**

| # | kernel | script | status |
| --- | --- | --- | --- |
| 1 | exp17 (power-up) | exp17_lora_distill.py | **built** — pending push `pairs 4_3` + 1000 steps (~9.2h) |
| 2 | exp16 v2 (readout) | exp16_readout_pull.py | **built** (v2 code applied) — push to consume stage0 targets cache |
| 3 | exp19 v2 (destruction) | exp19_destruction.py | **built** — needs exp16 v2 JSON (knee) first |
| 4 | exp6_l18_snr (HEADLINE) | **exp6_l18_snr.py (NEW)** | **built** — data LOST, mandatory re-run; script written this session |
| 5 | exp4_k_sensitivity (FIXED) | exp4_k_sensitivity.py | **built** — was broken self-bridge; now cross-bridge 4B→0.8B + K→256 |
| 6 | exp2_heatmap | exp2_heatmap.py | **built** (n=8, K=64) |
| 7 | exp11c_defrag_bridge | exp11c_defrag_bridge.py | **built** (n=8) |
| 8 | exp10b_gauge_bridge | exp10b_gauge_bridge.py | **built** (n=8) |
| 9 | exp4_top_p_tokens | exp4_top_p_tokens.py | **built** (n=4) — optional |

**Still NOT in pipeline / paper:** exp11b re-run, exp12a N=4/K=32, exp12b/12c K=64, exp16/17/18/19 paper *sections* (write — not in paper yet). exp6_l18_snr is now in the pipeline (was the gap).
**Paper-only fixes (no GPU)**: exclude 1.0 sentinels (exp13/15/6); exp11a/11b K-mismatch note + 11c bridge-flattening (defrag = output-faithful but bridge-destroying); exp12 re-frame; exp14 add params; exp2a note (CKA null inflated when N≪D); exp17 topk_report part of v2 suite.
