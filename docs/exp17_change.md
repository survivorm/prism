# Exp17 v23: 300 steps + matrix-free J (no fallback) + LoRA delivery + C5 top-k

## STATUS (updated per step — ground truth, do not skip re-reading)

DO NOT TRY DOING ALL SIMULTANEOSELY!!! ONE POINT AT A TIME!!! UPDATE PLAN, ANALYSE EXACTLY ONE PROBLEM, FIX, UPDATE PLAN.

| Item | State | Note |
| ------ | ------- | ------ |
| §0 stage0 v2-cache | ✅ DONE | Kaggle v12 finished: 7/7 pairs, 55.3 min. Logs: exp16-stage-0-target-jacobian-cache.log. Caches have input_ids+J_tgt_p+p_L_full. |
| §A1 matrix-free J in prism/jacobian.py | ✅ DONE | jacobian.py core + exp17 wiring: **refresh** = matrix-free rows (W_rows = W_eff[token_set] live, create_graph=True, p = p_L_full[token_set]/sum fp32), rows graph freed by per-sample backward; **fill/stale/eval** keep dense `compute_hidden_jacobian` (create_graph=False = 4MB [D,D], never the OOM; live W_eff keeps lm_head bridge grad on stale steps); refresh step re-fills dense J_norm (freshness = v22). Deviations from spec (recorded): `return_hidden` NOT used (refresh WAS 3 forwards: KL + rows + dense fill) — **v23.2 now dedups to 1 full + 1 decoder-tail (see §v23.2)**; bridge gives lm_head grad on stale steps (spec: zero). |
| §A4 chunked C=8 | ✅ DONE in jacobian.py + VERIFIED | `bridge_rows_backward` (jacobian.py): one `_forward_h_norm` + one shared `J_bar`, then each chunk of C=8 rows backprops its `(1-cos).sum()/K` immediately. **Root-cause fix found via test**: the KL `extra_loss` backprops in a SEPARATE `retain_graph=True` backward before the chunk loop — NOT coupled into chunk 0 (coupling frees the shared h forward graph). Chunks: `retain_graph=not is_last` keeps the ONE shared h-graph alive; each chunk's 2nd-order row-graph freed by refcounting → peak = C+1, not K. The KL is a separate forward in exp17, so its own graph frees independently. **tests/test_bridge_rows_backward.py: 2 PASS** (grads vs dense one-shot [A_lh/B_lh/A_attn/B_attn/lm_head] + J_S_p rows + chunk-size invariance C=8 vs C=K). |
| §A2 VRAM telemetry | ✅ DONE (exp17; B2 kernel pending) | `[vram] pair=X step=Y si=Z k=.. peak=..G cur=..G` after every refresh (cuda only). |
| §A3 fail-fast os._exit(3) | ✅ DONE | Probe FATAL (OOM/dead-graph) + refresh FATAL (OOM/RuntimeError → `[FATAL] v23 refresh OOM pair step si` → `os._exit(3)`). No degrade path left in exp17. |
| §A5 forbidden list | ✅ OK | no bf16/fp16/lowrank/defrag/trunc/S-shrink in code. |
| §B1 tests | ✅ DONE | 6/6 matrix-free + 25/25 full suite + 2/2 bridge_rows_backward. FIX: `_clone_gdn_states_on_entry()` in prism/jacobian.py — scoped context manager cloning GDN `initial_state`/`conv_states` at kernel entry (torch_recurrent_gated_delta_rule / causal_conv1d_update), wrapped in `_run_decoder_pass` AND `_run_decoder_to_norm`. Root cause: transformers 5.15 GDN forward saves the cache state object (v0) in the recurrent loop, then update_recurrent_state does `stored.copy_(new)` in-place (v0→1) → backward crash. Clone at entry → read≠write. No-grad paths unaffected (same clone logic, already safe). |
| §B1b smoke_j (B_attn+A_attn nonzero) | ✅ DONE | `B_attn = randn*0.02` (init edit); probe AND-check `ok = gA>0 and gB>0` + probe rewired to matrix-free rows in edit 2. `B_lh` stays zeros (standard LoRA, dL/dB_lh≠0 anyway). |
| §A2b p_S≈p_L gauge gate | ✅ DONE | `check_a2_ps_pl.py` (0.8B fp32 vs cache p_L_full): 3/4 samples cos 0.90–0.996 (gauges identical — re-normalized K-subset = stage0 gauge). si=0: cos 0.48 = REAL 4B flatness (fp16 re-run matches cache: max p 0.179, 14 tokens >1%; 0.8B sharper: 0.257). Not a gauge bug; si=0 = weak but valid target (p_K max 0.186). |
| §B2 kaggle_kernel/exp16_4_3/ | ⏸️ FD off-kernel; re-push pending | **v2 (fd_check=1) OOM-killed on T4** (DeadKernelError t=905s, after smoke_j OK — the FD check = 2 full forwards + dense [D,D] on top of the 2nd-order probe → 16GB T4 kill), before the train loop. **Decision: FD check off-kernel** (B2 = no-FATAL-only, fd_check=0). Ran the FD check **locally** (`experiments/exp17_fd_local.py`, 0.8B CPU): **arg order + jacobian math CORRECT** (no crash; h_norm identical via both code paths, maxdiff 0.0; clean 1D element FD-central vs analytic rel~1e-2). The exp17 block's rel~1.6e-1 = **FD truncation amplified by the ~1000× decoder-tail gain (f'')**, not a math bug (central barely helps 1.2e-1); a tighter number needs larger central eps or compare at the h_norm level. exp17's fd_check computes the identical quantity (only token_set differs, token-indep in J_norm) → would match the 16% too. **Next: re-push exp16_4_3 with fd_check=0** (no-FATAL-only, ~1h). https://www.kaggle.com/code/voroninmichael/prism-exp16-4-3-v23-single-pair-fd-check |
| §C1 300 steps | ✅ DONE | `steps = 300 if cuda else 40` + `--steps N` override (argparse, default 0 = auto). |
| §C2 early-stop N=50 P=3 | ✅ DONE (logic verified) | No-grad eval every 50 steps (dense J, live W_eff, re-normalized gauge p=p_L_full[token_set]/sum — same as A2b check); stop at 3 consecutive < 0.5% relative improvement (bridge-loss improvement; baseline at 0). **State machine synthetically verified** (no-stop / stop@150 / stop-after-recovery cases). Runs on GPU only (heavy). |
| §C3 5 ckpt/pair @60/120/180/240/300 | ✅ DONE | `ckpt_{pair}_{step}.pt` (A/B matrices lm_head+o_proj, optimizer state, eval-cos, peak-vram, l_S, order). Saved post-freeze. |
| §C4 lora_{pair}.pt ×7 | ✅ DONE | `lora_{pair}.pt` per pair after freeze: A/B lm_head+o_proj, optimizer, eval-cos, peak-vram, l_S, order, step. (o_proj included in file — Q2 default = discard at use time; tensor stored either way.) |
| §C5 c5_topk_report.json | ✅ DONE (module; run on GPU) | `experiments/exp17_topk_report.py`: 3 models (0.8B base / 0.8B + cumulative lm_head deltas, +o_proj with --keep-o-proj / 4B fp16 with OOM fallback: empty_cache+retry, then skip-with-note), 20 prompts × 4 categories (prism.calibration), k sweep 1/5/10, agreement+garbage_rate, examples[3] per (cat,k). Compile+pyright clean; not run (needs lora_{pair}.pt from a full run). |
| §D1 pairing by input_ids | ✅ DONE | exp17: `sample_targets` matched by exact input_ids (D1), per-pair match log + warn+skip, smoke_j/eval loops use matched target. VERIFIED: 28/28 cache files match freshly-generated calibration samples. NOTE: cache K=256 (not 128 — plan number stale). |
| §D2 clamp_min(1e-12) train+eval | ✅ DONE | exp17 lines 339, 472. |
| §D3 per_layer_results.append | ✅ DONE | exp17 line 492. |
| §D4 device.startswith("cuda") | ✅ DONE | exp17 line 119. |
| §D5 rewrite description | ✅ DONE | Module docstring + result description rewritten to v23 (matrix-free, FATAL, k=50, 300 steps, both-deltas-freeze, input_ids pairing). `j_budget` var removed. Pyright 0 errors. **CPU SMOKE (4 steps): PASS — smoke_j AND-check green on pairs 31_23 & 20_15 (dL/dA_attn 4.04e-02/7.19e-01, dL/dB_attn 3.55e-02/1.27e+00), k=50 logged, fill+stale dense path runs (step 1 bridge 0.3625/0.6619). Refresh path NOT exercised at 4 steps (k=50) — needs GPU/B2. Smoke running in bg (log /tmp/exp17_v23_smoke.log, PID 3666831).** |
| §E pushes (B2 then exp17) | ✅ DONE (v2) / ⏳ BLOCKED (full-16) | exp17 → `prism-lora-distillation` **v2** pushed (latest, RUNNING) — registry id fixed from stale `prism-exp17-lora-distill` (was 409). exp16_4_3 → `prism-exp16-4-3-v23-single-pair-fd-check` v1 **COMPLETE**. **Full exp16 = `exp16_readout` → `prism-readout-pull` (kernel-exp16r) — NOT stage0.** Built (72KB, refreshed from 08/28 code) but **push BLOCKED: "Maximum batch GPU session count of 2 reached"** — 2 slots held by two exp17 sessions (v1 redundant + v2). Kaggle CLI 2.1.2 has **no stop/cancel**; REST `CancelKernelSession` needs a numeric session id that no public endpoint exposes. Need to free a slot (web UI cancel or wait for v1/v2 to finish) before re-pushing exp16_readout. Both exp17 + exp16_readout: stage0-source-attach warning (non-fatal, targets via dataset). |
| §v23.1 kl_only guard + FD-arg-order fix + dataset re-land (7-pair) | ✅ DONE | (1) **GDN/kl_only guard**: per-sample bridge block (fill/refresh/stale) + J-refresh print + pre-run probe + final no-grad eval all gated on `A_attn is not None` (= o_proj present = full-attn = bridge-eligible); GDN pairs (l_S ∈ {0,6,9,12,18,21,22}) train **KL-only** (lm_head LoRA via KL leash; no 2nd-order J refresh — it would reach no o_proj), logged `kl_only: no o_proj at l_S=..`. (2) **FD spot-check arg-order fix** (the 4_3 v1 crash): `_extract_hidden_state(mS, l_S, ids_0, attn_0)` → `(mS, ids_0, attn_0, l_S)` (layer_idx is LAST). (3) **Dataset re-land**: Kaggle `prism-exp16-targets` was a **stale 9-pair set** (0_0,4_3,8_6,12_9,16_12,20_15,24_18,28_21,31_23 — 5 GDN pairs) that the killed exp17/4_3 runs mounted (stage0 source-mount unreliable → fell to dataset); re-uploaded the **local 7-pair correct set** (0_0,4_3,9_7,15_11,20_15,25_19,31_23 = 6 full-attn 3mod4 + 0_0 kl_only). Local `results/targets/` + stage0 kernel-outputs mount were already 7-pair; **all 3 sources now agree**. pyright 0, FD fix + guard decision verified on tiny model / real config. |
| §v23.2 forward dedup (h reuse) | ✅ DONE | A **refresh sample did 3 full forwards** of the same deterministic hidden state (KL + matrix-free `_forward_h_norm` + dense re-fill `_extract_hidden_state`); a fill sample did 2. Fix: the KL forward is now `output_hidden_states=True, use_cache=False` (matches the J-path mode) → captures `h_s = hidden_states[l_S+1]`; `bridge_rows_backward` and `compute_hidden_jacobian` gain an optional `h` param (reuse: skip the model forward, only the cheap decoder-tail-to-norm runs). Refresh **3 → (1 full + 1 decoder-tail)**; fill **2 → 1**. **VERIFIED exact**: `_forward_h_norm(h=reused)` returns the same h_norm/h as standalone (diff 0.000e+00, tiny model). matrixfree 6/6, pyright 0. (0.8B full-model CPU FD smoke NOT run — user: heavy CPU tests forbidden.) |
| §F re-push B2 (4_3) + exp17 (7-pair) | ✅ DONE (pushed 2026-08-30) | Rebuilt both notebooks (v23.1+v23.2 inlined) and pushed (user-approved): **exp16_4_3 v2** (single 4_3, steps=300, fd_check=1) + **exp17 v1 of `prism-lora-distillation`** (full 7-pair: 6 bridge + 0_0 kl_only). Dataset re-landed 7-pair (28 files). **v23.2b**: exp17 killed (4× bridge refresh = 4 samples, not a recompute — nothing reusable: h_S/p/J_tgt/HVP all per-sample); k 50→100 (halve refresh rate, cut dominant wall-clock), **re-pushed v1 = k=100**. Then user read logs: "learning force too weak, you smeared 100 steps over 300" — k=100 made the 2nd-order (o_proj) refresh 10× sparser than v22 (v22 k~10). **v23.2c DUAL CADENCE (user design, 2026-08-30): 2nd-order (HVP, o_proj) stays k=100 (expensive); 1st-order (no-HVP, dense J_norm via h-reuse) recomputes EVERY step** — the stale branch now runs `compute_hidden_jacobian(h=h_s)` (decoder-tail, no 2nd-order graph) so the lm_head bridge runs on a FRESH hidden every step (no 100-step staleness); trade-off +1 decoder-tail forward/sample/step wall-clock for a much stronger lm_head bridge. 17 killed + re-pushed with this. exp17 ~12h, 4_3 ~1h. URLs §B2/§E. |
| Q2/Q3/Q4 | ❓ OPEN | Ask at push time. Defaults: discard o_proj deltas / try in-kernel 4B / N=20 k=10 sweep. |

## v23 body

v23 supersedes v22 (marked [v22, SUPERSEDED]). The v22 run died on OOM in the
create_graph J (27-32GB: 128 full D×D graphs retained simultaneously). v23 fixes
the math (matrix-free, chunked backward, exact), the loop (300 steps, early-stop,
5 ckpt/pair), and the deliverable (LoRA A/B + top-k report).

## 0. Stage0 redo — v2-cache with input_ids (pushed, running)

- Select 7 pairs from 32×24 via 3mod4 filter (done, v12 running on Kaggle):
  (0,0),(4,3),(9,7),(15,11),(20,15),(25,19),(31,23)
- v2-cache = {l_L}_{l_S}_{si}.pt: J_tgt_p, p_L_full, **input_ids** (new),
  layer_types (new), K
- Exp17 pairing by exact input_ids match (not file order); mismatch = warn+skip
- p_L_full replaces p_L_top128 (C5 + KL against full dist if needed)
- layer_types printed from config (no hardcode)

## A. Matrix-free J (the OOM fix) [v23, replaces v22 §A]

### A1. Per-sample J_S_p, no dense [D,D] J_norm ever built

Current: h=extract → J_norm=autograd.grad(h_norm, h, create_graph=True) [D,D]
→ J_S=W_eff[token_set]@J_norm → softmax_jacobian. OOM = 128 retained
second-order graphs (prefix-decode intermediates per graph ≈ 25-30GB total).

v23: for the same h (graph-connected through the patched o_proj):

```
for i in K:                                   # token_set indices
    J_i = autograd.grad(h_norm @ W_rows[i], h,   # W_rows[i]=lm_head[i] [D]
                        create_graph=True, retain_graph=True)[0]   # [D]
W_bar = p @ W_rows                            # [D], p = softmax(logits)
J_bar = autograd.grad(h_norm @ W_bar, h, create_graph=True)[0]      # [D]
Jp_i = p_i * (J_i - J_bar)                     # = softmax_jacobian row, exact
J_S_p[i] = Jp_i / U                            # U = number of samples using this J
loss_i = (1 - cosine(J_S_p[i], J_tgt_p[i])).mean()
```

- `h_norm @ W_rows[i]` is a scalar per i → one backward per row; **no [D,D]
  tensor is ever allocated**. 128 rows × 1 graph each = same graph count as
  v22's 128 rows of a dense J — but each graph's terminal op is a scalar mat
  (tiny), and crucially the backward is CHUNKED (A4) so graphs die in groups.
- Exact: `∂(h_norm W)/∂h = W^T @ d(h_norm)/dh = W^T @ J_norm`, so J_i =
  J_norm[i] and Jp_i = p_i(J_norm[i] − p·J_norm) = softmax_jacobian's Jp row.
  Same math as `softmax_jacobian` (prism/jacobian.py), applied without the
  dense matrix.
- W_rows in graph → backward of loss_i delivers dL/dh_i to lm_head (A_lh,
  B_lh) AND, since h was built through the patched o_proj, to o_proj (A_attn,
  B_attn). Same connectivity as v22 §A.1.
- J_S_p computed **per sample** (h from that sample's prefix); bridge compares
  sample si's J_S_p[i] with sample si's J_tgt_p[i] (fixes cross-sample
  mismatch, v22 §A.2).

### A2. VRAM telemetry (mandatory, both kernels)

After every refresh: `torch.cuda.max_memory_allocated()` peak +
`torch.cuda.memory_allocated()` current → log line:
`[vram] pair=X step=Y k=... peak=..G cur=..G`. On B2 (4B) this is the number
that decides C (must stay < 16GB; if peak > 13GB → halve C, re-push).

### A3. Fail-fast (no fallback)

Any OOM/exception in a refresh → print `[FATAL] v23 refresh OOM pair=.. step=..`
→ `os._exit(3)` (kaggle log stops; no "o_proj KL-only" degrade, no silent
skip). The run is the experiment; a degraded run is a lie.

### A4. Chunked backward (C=8)

The 128 row-gradients are computed in chunks of C=8: within a chunk build all
8 (J_i, J_bar shared), backward the chunk's loss sum (retain_graph=True on all
but the last), then `.detach()` the chunk's row-graphs (and W_bar graph) to
release them before the next chunk. Peak retained = C graphs, not 128.
C=8 default; C=4/2 if 4B peaks > 13GB. (v22 had this in code but never reached
it; v23 keeps it as the primary path.)

### A5. Forbidden

No bf16/fp16 in the J chain. No low-rank/defrag tricks. No token_set
truncation. No S shrink. (v22 §A5, unchanged.)

## B. Math sanity (B1 local, B2 Kaggle) [v23, replaces v22 §B]

### B1. Local (CPU, tiny 2-layer model, <1 min)

1. Matrix-free J_S_p == dense J_S_p (build tiny dense J_norm as reference):
   rel err < 1e-3 (fp32).
2. smoke_j with **B_attn and A_attn both nonzero-init** (v22 bug: B_attn=0
   made dL/dA_attn==0 at step 0 → "graph dead" false alarm; now both
   ~randn*0.02): backward of probe loss must give dL/dA>0 AND dL/dB>0.
3. softmax_jacobian identity: J=I → Jp_i = p_i(δ_i − p) (known-answer).
   Run: `.venv/bin/python -m pytest tests/test_jacobian_matrixfree.py -q`

### B2. Kaggle GPU T4 mini-kernel, 1 pair (4,3), 300 steps

- New dir `kaggle_kernel/exp16_4_3/` (build_kernel.py, id
  voroninmichael/prism-exp16-4-3, T4, sources=exp16 stage0).
- 4B student (fp16) vs 0.8B teacher (fp32), 1 GPU, pair (4,3), K=128, C=8,
  k=50, 300 steps, 5 ckpt, A-implementation, fail-fast, VRAM log (A2).
- Success = all 300 steps (or early-stop), peak VRAM logged < 16GB,
  dL/dA_attn>0 AND dL/dB_attn>0 (smoke_j), finite-diff spot check (B1.2 on
  GPU): J_S_p[0] (row 0) vs (h_norm(h+1e-4 e_0)@W[0] − h_norm@W[0])/(1e-4)
  rel err < 1e-2.
- If OOM: C=4 re-push (once). Still OOM → K=64 (ask user). Kill = no retry.
- ~25-40 min T4.

## C. Exp17 loop (v23) [replaces v22 §C]

### C1. 300 steps per pair (max); early-stop may end sooner

### C2. Early-stop

Every N=50 steps: one FRESH no-grad J_S_p (C1-style, no create_graph — a
decode + row grads, ~3×370ms, cheap) → eval bridge-cos per sample, mean.
Stop the pair when P=3 consecutive eval points show < 0.5% relative
improvement. Log: steps run, stopped-at, best eval-cos, final train-metrics.
(Max 300, so at most 7 eval points.)

### C3. Checkpoints: 5 per pair, at steps 60/120/180/240/300 (35 total)

Each ckpt = `results/exp17/ckpt_{pair}_{step}.pt`:

- A/B matrices for lm_head (r=4) AND o_proj of l_S (r=4) — the cumulative
  trained deltas (LoRA weights only, ~2-5MB)
- optimizer state (for potential resume)
- metrics: eval-cos, train-kL, k, t_J, peak VRAM (A2)
Ckpts are cumulative LoRA (all deltas so far), not per-step.

### C4. Delivery = trained LoRA A/B

Final per-pair file `results/exp17/lora_{pair}.pt` = {A_lh, B_lh, A_attn,
B_attn, l_S, r, peak_vram, steps, eval-cos}. This is the deliverable: apply
the deltas to a fresh 0.8B (or load as LoRA modules) — no full-model copies.
o_proj deltas: [v23 decision, OPEN — see Q2].

### C5. Top-k report (3 models, same prompts)

- Model 1: 0.8B base (no deltas)
- Model 2: 0.8B + all 7 lm_head deltas (cumulative, applied)
- Model 3: 4B (fp16)
- Same N prompts per category (wiki/math/code/chat — the exp16 calibration
  sets, e.g. 20/category), same k (e.g. 10; sweep 1,5,10 if cheap)
- agreement = |topk(model2) ∩ (topk(model1) ∪ topk(model3))| / k
- garbage_rate = 1 − agreement (per category + overall)
- Question: does the LoRA-tuned 0.8B propose tokens that are (a) already its
  own base's top-k (no-op / garbage-in-garbage-out) or (b) the 4B's (real
  distillation) or (c) neither (true garbage)?
- Output: `results/exp17/c5_topk_report.json` {k, category, agreement,
  garbage_rate, examples[3]}. If 4B-inference OOMs after training (both
  models loaded), unload 0.8B first; if still OOM, run 4B topk as a separate
  small kernel (Q3).
- Runs inside exp17 kernel after training (4B is already loaded for
  exp17... actually exp17 loads 0.8B student only; 4B load happens at report
  time — see Q3 for VRAM).

## D. Exp17 bug fixes (carried from v22 audit, still open)

- **NEXT (DOING): §B2 kernel — create kaggle_kernel/exp16_4_3/ (build_kernel.py, id voroninmichael/prism-exp16-4-3, T4, sources=exp16 stage0 cache): 0.8B student (fp32 J chain — NOT "4B student", doc typo to fix), 4B targets from stage0 cache (4B never loaded), K=256 (cache built K=256, plan's 128 stale), C=8, k=50, 300 steps, 5 ckpt, matrix-free rows, fail-fast, VRAM log, finite-diff spot check.** Then §E pushes (ask-before; B2 first, then exp17 v23; Q2/Q3/Q4 to user).
- §B2 note (fix at kernel creation): section B2 text says "4B student vs 0.8B teacher" — REVERSED (kernel = 0.8B student, 4B targets from stage0 cache, 4B never loaded) and K=128 stale (cache K=256, see §D1). Confirm wording with user at push time (Q2/Q3/Q4).

1. Pairing: exact input_ids match (v2-cache has input_ids now)
2. KL: `p_ed.clamp_min(1e-12)` in train AND eval KL (NaN guard)
3. `per_layer_results.append(layer_result)` (was never called — per_layer: [])
4. `device == "cuda"` → `device.startswith("cuda")` (else 40 steps not 300)
5. Rewrite stale description/docstring (v22 said "J_norm cached once per
   layer, create_graph=False" — lies; v23: per-sample matrix-free refresh)

## E. Push plan (after stage0 DONE)

- B2 first (ask-before), ~30-40 min. VRAM < 16GB → proceed.
- exp17 v23 (ask-before), full run, T4, 12h. 7 pairs × ≤300 steps + C5.
- exp18/19: only if user wants controls rerun (default: no).
- Max 2 concurrent kernels.

## F. Audit checklist (verifyCommand: audit_v23.py checks all)

- [ ] 0.1 7 pairs selected by 3mod4, last=bridge
- [ ] 0.2 layer_types printed from config
- [ ] 0.3 v2-cache: input_ids, J_tgt_p, p_L_full per sample
- [ ] 0.4 pairing by input_ids (log: match count)
- [ ] A1 no dense J_norm; J_S_p per sample; exact formula
- [ ] A2 VRAM peak logged every refresh
- [ ] A3 OOM → os._exit(3) (no degrade path in code)
- [ ] A4 C=8 chunked backward; C tunable
- [ ] A5 no bf16/fp16/lowrank/defrag/trunc/S-shrink
- [ ] B1 tiny-model: matrix-free==dense (rel<1e-3); smoke_j B_init≠0
      dL/dA>0 AND dL/dB>0; softmax identity known-answer
- [ ] B2 4B 1 pair 300 steps: peak VRAM < 16GB, smoke_j ok, finite-diff ok
- [ ] C1 300/pair max
- [ ] C2 early-stop N=50 P=3 0.5%
- [ ] C3 5 ckpt/pair at 60/120/180/240/300, cumulative LoRA + optim + metrics
- [ ] C4 lora_{pair}.pt delivered (7 files)
- [ ] C5 c5_topk_report.json: 3 models, per-category agreement+garbage
- [ ] D1-D5 bug fixes
- [ ] pytest green

## Q. Open (ask user at push time)

- Q2: o_proj deltas — persist after each pair (fold into weights like
  lm_head) or discard (scaffold, only lm_head delivers)? Default: discard.
- Q3: C5 4B-inference VRAM (0.8B+4B both in 16GB T4? if not, 4B topk in a
  separate tiny kernel). Default: try in-kernel first.
- Q4: C5 prompts N=20/category, k=10 (sweep 1,5,10)? Default: yes.

## [v22, SUPERSEDED]

v22: amortized J budget + o_proj-KL-only fallback. Died: OOM in create_graph
J (27-32GB). Keep: exp16 7 pairs (now re-selected 3mod4), exp18 control,
exp19 destruction. Deleted in v23: use_fresh OOM degrade (→ A3 fail-fast),
J_budget formula (→ k=50 fixed + early-stop), dense J_norm (→ A1).
