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

- [x] **Paper: 4 new GPU results integrated** (commit 43e7ded): exp4 (N=4 K-sweep, 7 K-values × 6 par, converges K≥32), exp10b (N=8, centered<softmax<raw), exp11c (N=8, defrag destroys monotonicity at all r≤32: spearman 0.81/0.71/0.43), exp6 L18 SNR (new section: SNR=2.80 r-invariant, PR 2.1→16.9, gauge_share top-r 0.60→0.19).

- [x] **Committed** (e506bc0): paper.md + audit + exp2a script + PLAN.md (4 files, 122+/55-).
- [x] **Terminology unified** (committed 68b47ab): bridge = cos (higher=better alignment); gap = 1−cos (lower=better). `jacobian_bridge`→`jacobian_gap` (+deprecated alias); 1−cos reporters (exp1,4_u,6,7,10b,11c,12c,13,15) now report `gap`; cos reporters (exp16–19) keep `bridge`; 6 result JSONs migrated. **Paper done** (§1.6 renamed to gap, §1.7 added with direction table, L_bridge→gap in exps 1–15, future-work 16–18 direction-fixed): commit 43e7ded.
- [ ] **exp2 v4 PUSHED (Kaggle v4, 2026-09-05)** — v3 hit `NameError: __file__` (notebook has no __file__); removed the `sys.path` hack (notebook injects repo root, same as exp4/11c/10b/6 which import prism.* fine). Now: dense 4-layer grid {0,4,8,...,28} + adaptive valley search (stop on monotonicity break) + turning-point detection + per-(model,layer) J-cache; n=8, K=64, cap=6. Running.
- [ ] **exp4_top_p** — 2 duplicates were canceled; v3 then v4 to be re-pushed (K=32, n=4) when its slot frees (monitor auto-re-pushes).
- [ ] **TPU-only route** (GPU 30ч/нед ischerpana, user 2026-09-06; TPU ~20h/week v5e-8, separate).
  - **Canary v5**: GDN forward OK (~47s, torch-fallback); backward HUNG (transient XLA compile failure, NOT deterministic).
  - **ROOT CAUSE (transformers 5.15.1)**: Qwen3.5 GDN = `torch_chunk_gated_delta_rule` (chunk=64, prefill seq=34 → 1 chunk padded to 64). Backward = intra-chunk triangular loop + `last_recurrent_state` chain → large reverse graph (7 GB input / 6.5 GB output on v5e-8). v5 hang = XLA HLO compile failure — **transient** (iso-canary: same code ran fine).
  - **ISO-CANARY VERDICT (COMPLETE)**: BOTH backward (eval + train) compiled in 62s. **TPU v5e-8 IS viable for all prism causal experiments.** No XLA flags needed.
  - **`device.startswith` bug** (exp16/17/18): `torch.device` has no `.startswith` — fixed to `str(device).startswith(("cuda","xla"))` (committed).
  - **exp18-tpu v1** (first push): SyntaxError `from __future__` — build_tpu now strips all `__future__` imports + keeps one at top + py_compile guard (committed).
  - **exp16r-tpu**: XLA `SIGABRT` — `Cannot infer shape for dot: f32[1024,1024] <dot> f32[65536,65536]` (65536=K²=256²). NOT prism logic (CPU repro: J=[1024,1024], J_S=[256,1024] correct, no K² matmul anywhere) — XLA shape-inference bug in the 3-pass `compute_hidden_jacobian`. **exp16 (3 calls) + exp17 (10 calls) TPU-blocked.**
  - **exp18 + exp19 TPU-safe** (use per-token `compute_jacobian_for_tokens` + `softmax_jacobian`, NOT the 3-pass). exp18 v5 got TPU (958s) but crashed on `jacobian.py:125` reading `layer.number_of_states` (transformers 5.15.1 attr; TPU image = torch 2.8.0 + different transformers) = MY bug. FIXED: dynamic per-attr GDN-clone, version-agnostic. **GOTCHA: v6 push sent the STALE pre-fix kernel** (built before the fix) — re-diagnosed by grepping the built file (hardcoded `number_of_states` still present) + the 995s log. **Rebuilt kernel from fixed source → v7 pushed** (dynamic clone verified, 0 hardcoded). Same check applied to exp19 (fixed). **NEXT: monitor v7 → on [OK] push exp19-tpu (staged, dynamic-clone-built, targets dataset prism-exp16-targets, dead `_step_cm` uncalled → no xm.step bug).**
  - **NEXT**: exp19 TPU (after exp18), then exp16/17 on GPU (next week, 30h/wk; exp17=9.2h fits 12h slot) or TPU workaround (precompute J_norm on CPU → upload as dataset, avoids the 3-pass dot lowering).
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

## TPU verdict correction (2026-09-09) — IMPORTANT
**"exp16/19 TPU-safe" was WRONG.** All 4 causal exps share the `[D,D]=f32[1024,1024]` Jacobian (J_norm or per-token J) that XLA corrupts into a phantom `f32[65536,65536]` dot → SIGABRT.
- exp16-tpu v1: ERROR 98s, `Cannot infer shape for dot: f32[1024,1024]<dot>f32[65536,65536]` (analytic last-layer J_norm mm). **CPU repro: the op is [1024,1024]; [65536,65536] never formed by any prism op => XLA-side shape-inference bug, not prism logic.**
- exp18-tpu v6/v7: same phantom (per-token path, deeper). v10: pairs 1-2 OK, pair 3 hung (recompile).
- **Conclusion: TPU v5e-8 cannot compute the `[D,D]` Jacobian → exp16/17/18 (and exp19, same J) are TPU-blocked.** GPU after the 30h/wk reset (~2026-09-14). exp19 is the *least* blocked (per-token, no [D,D] J_norm) → last candidate on the remaining TPU slot, uncertain (exp18 per-token also hit it).
- **I was wrong to call exp16 "safe (no GDN bwd)" — never CPU-repro'd the analytic path's actual XLA mm shapes. Lesson (re-confirmed): CPU-repro the exact op chain before a hardware verdict.**

## exp19 TPU COMPLETE (2026-09-10) — gauge=null, target≈differential
- All 6 scenarios ran on TPU v5e-8 (~44min, per-token Jacobian + cosine, no [D,D]):
  no_edit(baseline bridge=0.6000), a_random(Δ=+0.00002), b_target_1x(+0.0017), d_gauge(-0.000006≈0),
  e_differential(+0.0017), c_target_10x(+0.0151). ‖ΔJ_gauge‖≈0 → target edit is purely differential;
  gauge is a null direction (no effect). Damage ~linear in dose (1x→0.0017, 10x→0.015).
- Bug: final json.dump crashed on device (torch.device not serializable) AFTER all data computed + partial saved;
  but json.dump clobbered the partial file → reconstructed the complete JSON from the log into results/exp19_destruction.json;
  source fixed (device→str(device)) at lines 435/439.
- Paper value: exp19 is the cleanest "gauge=benign / directional=causal" causal experiment. → needs a paper section.

## exp17 2nd-order TPU canary (2026-09-10, QUEUED)
- Pushed (version 1) after exp19 COMPLETE. 1 pair 15_11 (small L11, fixed start → no recompile fanout), 100 steps,
  real bridge_rows_backward (2nd-order HVP, create_graph=True). ~10-20min.
- Decides: 2nd-order compiles on v5e-8 → exp17 TPU-viable; else → dead-end, exp17 stays GPU.
- Monitor: _auto_chain/poll_canary.sh (polls to terminal, pulls log, greps 2nd-order verdict).
## exp19 v4 (7-pair, dose 1/10/100) + exp17 canary verdict (2026-09-10)
- **exp17 2nd-order canary: CANCELLED** (hung ~2h in XLA HLO compile on the 2nd-order HVP). Verdict: **2nd-order is a TPU dead-end → exp17 (and exp16) stay GPU.** No TPU 2nd-order expansion.
- **exp19 v4 pushed (v4, id prism-exp19-destruction-tpu): 7 pairs, dose 1/10/100, per-pair x per-scenario per-pair checkpoint.** All 7 pairs have targets. QUEUED (TPU pool long); runs when a slot frees; per-scenario checkpointing protects partial results. 1st-order (proven) → no XLA risk beyond the 2h wall-clock (7 pairs x 6 scenarios fit, per-pair partial save guards a kill).
- **DONE**: results pulled, **JSON-save bug fixed** (was `json.dump(..., device=xla)` → TypeError, file truncated; now `device=str(device)`), full 7-pair JSON recovered from the kernel log + completed scenarios from partials, integrated into docs/paper.md §11.13 (full per-layer bridge table + dose-response: dose-linear 0.17×/0.5×/0.9×, gauge=0.0000/0.0238/0.2382 at L23, gauge null-op, target=differential exactly, cross-layer gauge/diff ratio stable 1.6–1.9).
- **CORRECTION (v5 push 2026-09-11):** the v5 push was misconfigured as `pairs=[(31,23)]` → it re-ran only the **31_23 pair** (7 scenarios incl. doses 1x/10x/100x) in 44 min, NOT the 7-pair sweep. The pull therefore overwrote results/exp19_destruction.json with **31_23 single-pair** data (Δbridge +0.0017/+0.0151/+0.142 for 1x/10x/100x, gauge null). The **7-pair sweep is still pending-optional** (doses 1/10/100 × 7 target pairs) — ideal **box follow-up** (2-3h recompile < 9h cap; box keepalive defeats the idle-kill; GPU fallback after ~09-14 weekly reset). Paper §11.13 keeps the real 7-pair content (from the v4 log) — unchanged.

## kaggle-tpu-lab sandbox (2026-09-11, user reframe «интереса ради») — TPU-blocked exps via long-lived session

**Idea (user):** instead of competing for the TPU slot with 2h-capped kernels, run `ARahim3/kaggle-tpu-lab` — a **long-lived TPU session** (keepalive `keepalive_min:480`, Kaggle TPU hard-caps at 9h, beats the 2h per-kernel wall-clock that blocks exp16/17/18) — and inside the **same container** run our prism causal experiments as **4-6 concurrent threads** (one experiment = N parallel threads, NOT 4-6 agent instances).

**Why it unblocks what 2h-blocks:**
- Kaggle free-tier TPU = **v5e-8 = 8×32GB = 256GB HBM total**. Qwen3.8-27B (vLLM-TPU, TP=8, bf16 ~55-80GB) occupies only a fraction; ~150-200GB remains. **TPU is partitionable** (chip sub-units) → vLLM (JAX/Pallas) and prism (torch_xla) are two NON-MONOPOLISTIC clients coexisting (user: «vllm может и будет занимать TPU не монопольно → одновременно инференс и эксперимент»).
- 9h session ≫ 2h kernel → exp16 (~5-10min workload), exp17 (~9.2h but needs 2nd-order = dead-end, see below), exp18 (~1h 7-pair) all fit.

**Known TPU blockers that SURVIVE the sandbox (unchanged by it):**
- **exp16**: XLA phantom `[D,D]` dot SIGABRT — the 9h session doesn't fix a shape-inference bug. Needs the phantom-dot workaround (chunked/analytic split) OR a different image flag first.
- **exp17**: 2nd-order HVP = confirmed XLA HLO compile hang (canary cancelled). Sandbox doesn't fix 2nd-order. Stays GPU.
- **exp18**: 7 distinct l_S = 7 XLA recompiles ~4h total > 2h kernel, BUT **< 9h session** → sandbox is exactly what lets exp18 run. Highest-value target for the sandbox.
- **exp19 v4**: already running as kernel; if the 2h kill recurs, sandbox gives it 9h to finish.

**Implementation patterns (pick one):**
- **(A) concurrent coexist**: kaggle-tpu-lab `serve` runs vLLM (JAX) on part of the v5e-8; a second client (prism via torch_xla) on the remaining chips (TPU partition). Both alive simultaneously. Cleanest, needs TPU partitioning setup (vLLM on e.g. chips 0-3, prism on 4-7) — feasibility unverified.
- **(B) reuse-env, no vLLM**: skip the 27B server entirely; push a custom kernel that reuses kaggle-tpu-lab's prebuilt env (XLA compile cache from `rahim3/qwen38-tpu-env-v5e8` dataset, uv venv, internet on) but runs only prism. Avoids HBM waste + vLLM restart cost. The «agent harness» drives prism via a non-TPU model (the vLLM 27B is only needed if the agent itself runs on-TPU — it doesn't have to).
- **Thread fan-out (in either pattern):** one prism experiment = 4-6 Python threads/processes, each taking a different pair/scenario/shard of the token set, all in one session. Cuts wall-clock 4-6×.

**Open questions before launch (verify, don't assume):**
1. Does Kaggle free-tier allow 2 concurrent TPU clients on ONE v5e-8 (vLLM + torch_xla)? Or is 1-batch-session still enforced at the client level (observed: 'Maximum batch TPU session count of 1')? If the 1-session limit is per-PROCESS not per-hardware, partitioning may be blocked.
2. Does the 9h keepalive actually hold on our account (kaggle-tpu-lab's `keepalive_min:480` is a claim; canary showed the 2h is a KAGGLE enforcement — need to test if a non-kernel long-lived session also gets 2h-killed or gets the 9h).
3. TPU partitioning API: can torch_xla be pinned to a chip subset (PJRT devices 4-7 of v5e-8) while vLLM takes 0-3?

**Gate / next step:** build a 2-client coexist canary — kaggle-tpu-lab vLLM serve (chips 0-3) + a 10-line torch_xla forward on chips 4-7 — to answer Q1/Q3. If it coexists → wire exp18 (7-pair, 4-6 threads) as the first sandbox run. If 1-session enforced → fall back to pattern B (env reuse, single prism client, no vLLM), still 9h. **No GPU spend; uses TPU quota only.**

## Box: autonomous TPU experimenter (2026-09-11, READY — push pending user go)

`box/run_box.py` = pattern (A): KTL notebook (serve_qwen38.py verbatim: vLLM 27B bf16 from `rahim3/qwen3-8-27b-bf16` dataset, TP=4, env cache `rahim3/qwen38-tpu-env-v5e8`, keepalive 480 min) + **mini-swe-agent** (SWE-agent/mini-swe-agent: DefaultAgent + LiteLLMModel → local vLLM OpenAI endpoint) + **auto-compact at 85% of max context** (threshold from `/v1/models`) + `py-spy` tool + git-pull mailbox for two-way prompt correction + `box/context-pack.md` (log index + ranked XLA hypotheses) in the system prompt. Prism code in the box, experiments on chips 4-7 (`PJRT_DEVICE=4,5,6,7`), 1st-order only.
In-box agent = mini-swe-agent DefaultAgent (BoxAgent subclass: 85% auto-compact, pyspy, git cmd-channel, 9h keepalive) → local vLLM 27B. **First task = vLLM/torch_xla coexistence probe** (run a 0.8B torch_xla forward, report HBM split vs vLLM); then TPU-blocked exp16/17/18 fixes. Self-check PASS locally (compact 60714→4072 at 85%). Coexistence gate = the probe; if OOM → prism falls back to CPU in-session, TP=4 chip-partition is the upgrade path.
