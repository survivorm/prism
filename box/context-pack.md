# Context pack — TPU debugging session (box)

You are starting a debugging run on Kaggle TPU v5e-8 for the prism causal
experiments (exp16/17/18/19). This pack indexes prior run logs and states the
working hypotheses. **Read the logs before forming new theories** — most XLA
failures here were diagnosed already.

## 1. Log index (repo-relative)

| Log | What it is | Key content |
|---|---|---|
| `kaggle_kernel/prism-exp18-logit-kd-tpu.log` | exp18 TPU v10 (7 pairs, 100 steps) | pairs 1-2 completed bridge eval (~85s each), **pair 3 hung → 2h kernel kill**; per-pair checkpointing saved partial data |
| `kaggle_kernel/kernel-exp19-tpu/prism-exp19-destruction-tpu.log` | exp19 TPU v3 — **COMPLETE 6/6 scenarios, 44 min** | baseline bridge 0.6000; gauge=null (Δ≈-6e-06); target 1x=+0.0017, 10x=+0.0151; ‖ΔJ_gauge‖=0 at 31_23; final `json.dump` crashed on `torch.device` (fixed in source after) |
| `kaggle_kernel/kernel-exp19-tpu/results/exp19_destruction.json` | exp19 v3 result JSON (reconstructed from log after the dump crash) | structured numbers for the 6 scenarios; also mirrored at repo `results/exp19_destruction.json` (2.7K, partial fields) |
| `kaggle_kernel/kernel-exp17-2nd-canary/out/prism-exp17-2nd-order-canary-tpu.log` | exp17 2nd-order HVP canary (1 pair, 100 steps) | **HLO compile hang, 2h+ zero progress** — 48-line log ends mid-compile; py-spy would have shown the stuck compile thread |
| `results/exp17_logs.txt` + `results/exp17_lora_distill.json` | exp17 GPU-era outputs (pre-TPU, 1st-order path only) | prior LoRA runs; not TPU data, but the reference behavior to compare against |
| `kaggle_kernel/_push_queue.log` | full push history (times, versions) | timeline of which kernel version ran when |
| `PLAN.md` (repo root) | **the lab notebook** — sections "TPU verdict correction (2026-09-09)" and "exp19 TPU COMPLETE (2026-09-10)" | full verdict history incl. exp16-tpu v1 crash (log never saved to repo): ERROR 98s, `Cannot infer shape for dot: f32[1024,1024]<dot>f32[65536,65536]` in the **analytical last-layer RMSNorm+lm_head path** (not compute_hidden_jacobian) |

## 2. Confirmed facts (verified, not hypotheses)

1. **TPU v5e-8 = 8×32GB = 256GB HBM.** vLLM (JAX) + torch_xla (prism) can coexist as partitioned clients; vLLM's 27B bf16 takes ~55-80GB.
2. **1st-order backward through GDN works** (iso-canary: eval+train backward, 62s, no special flags).
3. **Per-token Jacobian at last_layer=23 works** (exp19 v3: 0 remaining decoder layers to backprop through).
4. **Kaggle kills kernels at ~2h wall-clock** (enforced; restarts the process from scratch). This session (a long-lived serve, not a batch kernel) is the workaround.
5. **XLA_FLAGS are process-start-only** — they must be in the environment before torch_xla initializes; flag variants = separate processes.
6. **The Kaggle TPU image's transformers ≠ our 5.15.1** — prism needs `transformers==5.15.1 safetensors==0.8.0` pinned (the build_tpu `--pip` pattern); `create_recurrent_attention_mask` is missing from the image's version.
7. **Every distinct loop-bound over layers = a new HLO graph = full recompile (~300-600s).** exp18's 7 target layers ≈ 4h of recompile alone.

## 3. Hypotheses (what may be wrong, ranked)

1. **Phantom dot `f32[1024,1024]<dot>f32[65536,65536]` (SIGABRT, shape-inference crash)** — confirmed XLA-side: CPU repro shows no prism op ever forms [65536,65536]. It hit exp16's analytical path and (flakily) the per-token path. **Fix candidates, in priority order (each = one fresh process, flags are process-start-only):** a) `XLA_FLAGS=--xla_optimize_hlo=false`, b) `XLA_FLAGS=--xla_use_stablehlo=True`, c) smaller K (128→64), d) chunked FD.
2. **HLO compile hangs** (exp17 2nd-order canary: 2h+ zero progress; exp18 pair 3) — **diagnose with py-spy on the hung pid** (that is the whole point of this session; prior runs died as dead logs). If the stack is in XLA lowering → it's a recompile/compile deadlock, not prism logic; mitigation = avoid the new graph (fix the layer loop bound, or precompile the shapes in a warm-up pass).
3. **Transient XLA glitches** (canary v5 failed, iso-canary passed on the *same* GDN backward) — expect one-off compile failures; **retry before drawing a verdict**. A single hang is evidence, not proof.
4. **exp17 2nd-order HVP** = confirmed dead-end (canary hung ~2h in HLO compile). Do not re-attempt 2nd-order on TPU; exp17 stays GPU. exp16's analytical loss path is 1st-order *if* lm_head is the only optimized param — verify that holds in code before running.

## 4. Box layout (this session)

- vLLM (KTL serve) holds TPU **chips 0-3** (TP=4, 27B bf16 ≈ 55-80GB).
- Run prism experiments with `PJRT_DEVICE=4,5,6,7` (chips 4-7, 128GB) — do **not** double-book the vLLM chips.
- `PYTORCH_XLA_DEBUG=0` by default; enable `=1` only when hunting a specific op.
- `pip install py-spy` is pre-done in the preamble; use `py-spy dump --pid N` on any stalled job.
- Box repo = /kaggle/working/box (prism code at /kaggle/working/prism, results land in its results/).
