# E-39b PTQ perturbation — fixes per docs/E39b_fixes.md

Task: apply the E39b_fixes audit to experiments/exp39b_ptq_perturbation.py,
rebuild the Kaggle kernel, and get a clean run with results/exp39b_ptq_perturbation.json.

## DONE / DOING / NEXT

- [x] Audit read (docs/E39b_fixes.md) — P0.1–P0.5, P1.1–P1.8, P2.13–15.
- [x] v4/v5 ran; v4 passed self-check (rel err max 7.77e-7 on Kaggle, matches local),
      v5 OOM'd in calibration `model(...)` forward (full logits [128,128,248320] ≈ 8 GB).
- [x] P0 fixes in worktree — `_c4_docs` no char-slice; Hessian from `model.model().last_hidden_state`
      (post-norm = lm_head input) masked by attn (kills OOM + P0.4 double-norm + P0.2 padding);
      PPL from last_hidden_state in doc-chunks, return real PPL=exp(NLL) (P0.3).
- [x] P0.4 verification: in-script check `lm_head(last_hidden_state) == model(...).logits` (4-doc batch).
- [x] P0.5: JSON meta `readout_only_ptq: true` (tied-weights limitation, E39 §14.1).
- [x] P1.4 seed + deterministic; P1.5 del J_rec/J_cache/H/M_K_base after self-check;
      P1.6 gate pass requires κ<0.02 AND ratio≤1.25.
- [x] P1.7/P2.12 rename level "int8" → "int8_rtn"; P2.14 drop `.abs()` on NLL; P2.15 log n_real_tokens.
- [ ] DOING: remove dead `_ppl_from_logits`, rebuild + push kernel v6, monitor to COMPLETE, download JSON.
- [ ] Notes (no code): P1.1 L<23 = readout-linearized convention (per spec §2.3), gate on L23 exact;
      P1.2 u_g = JJᵀ1/‖·‖ is the documented E-37 gauge axis (keep); P1.3 no sentinel filter
      (consistent with E-38d baseline).
- [ ] Rebuild + push kernel v6, monitor to COMPLETE, download JSON.
- [ ] Read results; report gate (κ(L23) gptq4 vs base), PPL Δ, J_rel_drift.

## Границы (не менять)
- Do NOT touch the readout-linearized J formula or the L23 self-check (verified 1.6e-7).
- Do NOT quantize activations or embed_tokens (readout-only PTQ per spec + E39 §14.1).
- Do NOT change C4 split skips (8192 cal / 24576 PPL fallback) or GPTQ params
  (4-bit, group 128, damp 0.01, desc_act) — all per spec §2.
- Do NOT recompute E-38a/i/E-16b results; base κ(L23)=0.0162 (E-38d) is the reference.
