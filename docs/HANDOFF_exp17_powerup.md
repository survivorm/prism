# Handoff — exp17 v23.2c power-up (resume point)

> Written 2026-09-01 17:36 MSK. Ground truth = this file + `docs/exp17_change.md`
> (§F/§K rows) + `git log`. Read top-to-bottom; the ONE pending action is §4.

## 1. TL;DR

- **exp17 v23 code is stable and committed.** Current regime (v23.2c):
  dual J-cadence — **1st-order (dense no-HVP, h-reuse) recomputed EVERY step** +
  **2nd-order (HVP, o_proj) every k=100**; **lr=3e-3**, **steps=500** (cuda) /
  40 (cpu), kl_weight=0.1, r=4, n_samples=4 (gpu), fail-fast os._exit(3) on refresh OOM.
- **The ONE pending action:** push `prism-lora-distillation` (kernel-exp17, the
  **full 7-pair** power-up run). It is **built + committed** but **NOT pushed** —
  blocked by **Kaggle weekly 30h GPU quota** (exhausted; see §5). Status = **404**
  (killed/deleted; nothing live). **Push it after the Kaggle weekly quota resets.**
- **4_3 separate project is CLOSED** (was a tool: fast 1-pair iteration). Its
  completed run is a clean **vanilla single-pair baseline** (v23.2b single-cadence,
  lr=1e-3, 300 steps) — numbers in §3. Do NOT re-push a separate 4_3 unless we
  specifically need a **vanilla-base dual-cadence/500** control (§6 Q).
- The power-up 17 run will be the **first** result of the new regime (dual + lr3e-3
  + 500). Its 4_3 pair carries the real cumulative base (5 frozen pairs ahead of it).

## 2. What is DONE (committed)

| Item | State |
|------|-------|
| v23.2c dual-cadence wiring | ✅ committed (`8efec80`) |
| v23.2 h-reuse (1 full + 1 decoder-tail forward) | ✅ committed (`8efec80`) |
| power-up: lr 1e-3→3e-3, steps 300→500 | ✅ committed (`d3b69cf`) |
| kernel-exp17 notebook (full 7-pair) rebuilt with lr=3e-3/500/dual | ✅ committed (HEAD `ea20f07`) |
| GDN/kl_only guard + FD arg-order fix + 7-pair dataset re-land | ✅ committed (§v23.1) |
| kernel-exp16_4_3 notebook (single 4_3, fd_check=0) | ✅ committed; **kernel COMPLETE** (both orders, 300 steps) |
| Local 7-pair targets cache `results/targets/` (28 files, K=256) | ✅ valid |
| exp16 v2 (dose-curve) source + kernel-exp16r | ✅ committed; push **also quota-blocked** (§K) |

Commits (newest first): `ea20f07` (push tooling) → `d3b69cf` (power-up lr/steps) →
`8efec80` (dual-cadence + h-reuse). Working tree: only `docs/gpu_restart_audit.md`
and `experiments/exp2a_cka_identity.py` modified (unrelated to exp17; leave or
commit separately).

## 3. Reference numbers (the 4_3 COMPLETED run — v23.2b single-cadence, lr=1e-3, 300)

Clean **vanilla single-pair 4_3** baseline (base = pristine 0.8B, 4_3 only, both
orders; from the `prism-exp16-4-3-v23-single-pair-fd-check` kernel). Bridge loss
(start→finish) over 300 steps:

| order | step 0 | step 300 | Δ |
|-------|--------|----------|---|
| late_to_early | 0.9835 | 0.7987 | −0.185 |
| early_to_late | 0.9832 | 0.8124 | −0.171 |

eval-cos (=1−bridge): **0.201 / 0.188** (late/early). Late_to_early slightly better
for 4_3 (opposite of the 9-pair trend where late_to_early was much worse — the
early pairs confounded it there). Local artifacts:
`results/exp17/lora_4_3.pt`, `ckpt_4_3_{60,120,180,240,300}.pt`,
`results/exp17_lora_distill.json` (all from this run; eval-cos + adam state match
the eval history exactly).
Log: `prism-exp16-4-3-v23-single-pair-fd-check.log` (note: its print is the OLD
`J refresh: k=100 (v23 matrix-free rows)` = **single-cadence**, confirming this
run predates dual-cadence).

**Interpretation for the power-up:** the upcoming 17 (dual + lr3e-3 + 500) run —
compare its 4_3 (in-path, real base) and overall curve against this baseline to
see if (a) dual-cadence and (b) the power-up actually move the bridge more than
the old regime over more steps.

## 4. THE pending action (one thing)

**Push the exp17 power-up full run** once the Kaggle weekly quota has reset:

```
cd /home/survivor/work/prism/kaggle_kernel/kernel-exp17
kaggle kernels push -p .        # -> voroninmichael/prism-lora-distillation (new version)
```

- Notebook already correct (verified: `lr = 3e-3`, `steps = 500 if`, `k = 100`,
  `DUAL CADENCE`). No rebuild needed unless code changes.
- Expect ~12h. It's the **full 7-pair** run (6 bridge + 0_0 kl_only), both orders,
  500 steps, lr=3e-3, dual-cadence, 5 ckpt/pair, lora_{pair}.pt ×7, C5 top-k report.
- If push still hits "weekly quota", wait (Kaggle week = Mon–Sun; verify with
  `kaggle kernels push` until accepted, or ask the user to bump the Kaggle tier).

## 5. Kaggle CLI gotchas (recurring; 2.1.2 installed, 2.2.2 available)

- **No `stop`/`cancel`/`kill` subcommand.** Only: list, files, get, init, push,
  pull, output, status, logs, update, delete. To stop a run: `yes | kaggle kernels
  delete <slug>` (interactive prompt; crashes without a tty → pipe `yes`).
- After delete, `kaggle kernels status <slug>` → **404** (kernel gone). That is the
  current state of `prism-lora-distillation`.
- `kaggle kernels list --user <u> --search <slug>` often returns **empty** even for
  RUNNING/COMPLETE kernels (CLI quirk). Use `kaggle kernels status <slug>` for truth.
- `kaggle kernels output <slug>` sometimes returns only pip-install noise or an
  upgrade warning (no JSON). For the 4_3 run the JSON landed as
  `results/exp17_lora_distill.json` + logs pulled manually.
- **Weekly GPU quota = 30h** (hard, non-bypassable this session). Prior runs
  (~33h: 17 v1+v2 + 18) exhausted it. Re-push after reset.
- Owner slug for our kernels = `voroninmichael` (NOT `survivor-17` — that 404s /
  denies `kernels.get`).
- exp16 stage0 kernel-outputs **mount is unreliable**; targets come via the
  **dataset** `voroninmichael/prism-exp16-targets` (re-landed = 7-pair, 28 files).
  The 7-pair set is what both 17 and 4_3 consumed (all 3 sources agreed).

## 6. Decisions / open questions for the user

- **Q1 (main): just push 17 after quota, or also re-push 4_3?**
  - Default (my rec): **17 only.** 4_3 reference (single-cadence vanilla) is done;
    the 17 power-up produces 4_3 in its real path position under the new regime.
    A separate 4_3 dual-cadence/500 re-push is only worth it if we want to
    **isolate 4_3's intrinsic power on a vanilla base** under the new regime
    (vs. the 5-pair confound in 17). Costs 1 GPU slot.
- **Q2: o_proj deltas** — persist (fold into weights like lm_head) or discard
  (scaffold; only lm_head delivers)? Current code folds both in; `lora_{pair}.pt`
  stores both regardless. (Default: discard at use time.)
- **Q3: C5 4B-inference VRAM** — 0.8B+4B both in 16GB T4? If OOM, 4B topk in a
  separate tiny kernel. (C5 runs at the end of the 17 kernel.)
- **Q4: C5 prompts** N=20/category, k sweep 1/5/10? (Default: yes.)
- If after the 17 power-up the bridge still feels weak → next lever is lowering
  **k (100→10–50)** for more 2nd-order o_proj refreshes (costs wall-clock/VRAM).
  Do NOT raise it.

## 7. Key files

| Path | What |
|------|------|
| `docs/exp17_change.md` | live plan (STATUS table = ground truth; §F/§K rows) |
| `experiments/exp17_lora_distill.py` | the experiment (dual-cadence, power-up values) |
| `prism/jacobian.py` | `bridge_rows_backward`, `compute_hidden_jacobian` (h-reuse), matrix-free rows |
| `kaggle_kernel/kernel-exp17/kernel.ipynb` | **ready to push** (lr=3e-3/500/dual, 7-pair) |
| `kaggle_kernel/kernel-exp16_4_3/kernel.ipynb` | 4_3 single-pair (fd_check=0); kernel COMPLETE |
| `kaggle_kernel/push_all.py` | kernel registry (exp16_4_3 / exp17 entries; runner_extra pins pairs/steps/fd) |
| `results/targets/` | 7-pair stage0 cache (28 files, K=256) |
| `results/exp17/` + `results/exp17_lora_distill.json` | 4_3 reference artifacts |
| `prism-exp16-4-3-v23-single-pair-fd-check.log` | 4_3 run log |
| `docs/gpu_restart_audit.md` | which experiments need GPU re-run (modified, uncommitted) |
