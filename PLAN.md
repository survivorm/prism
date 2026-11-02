# E-39b PTQ perturbation — fixes per docs/E39b_fixes.md (round 2)

**DONE** — прогон v12 отдал `results/exp39b_ptq_perturbation.json`, гейт = **`pass`**
(κ(L23) base 0.016233 → gptq4 0.016043, ratio 0.988; J_rel_drift 0.1205; PPL +0.90%).

## DONE / DOING / NEXT

- [x] Audit read (docs/E39b_fixes.md) — P0.1–P0.5, P1.1–P1.8, N1–N6.
- [x] P0.1–P0.3 fixed (token-slice, padded Hessian, PPL=exp(NLL)); P0.4 check; P0.5 readout_only_ptq.
- [x] P1.4 seed+deterministic; P1.5 del; P1.6 relative gate; P1.7 int8_rtn rename.
- [x] Round-2 §6 minimal set applied to exp39b_ptq_perturbation.py:
      - N1: PPL NLL mask `real` moved to device (attn_ppl_dev).
      - N3: model_dtype fp32 on CPU / fp16 on CUDA.
      - N2: base self-checks assert κ(L23)≈0.0162, β/σ₁≈0.882 (E-38d/E-38e) before gate.
      - N4: single `W_lm32 = W_lm.float()` for PPL matmuls (no per-chunk cast).
      - N5: dead `invperm` removed.
      - P1.1: `synthetic_readout_j = L != 23` + `extra_layer = L not in L_GATE` in every metrics[L][name].
      - P1.8: L19 marked extra_layer (not in gate).
      - P0.4: `raise` (not warn) on ref_err > 2.0 — PPL would be INVALID.
      - P1.4: use_deterministic_algorithms best-effort on CPU too (outside cuda block).
      - einsum `bmm` bug in `_cos_ug_1k` fixed (v = JJᵀ1 via einsum, 1/√K cancels in norm; verified vs per-position ref to 1e-7).
      - P1.3 documented in script header (no sentinel filter, consistent with E-38d).
- [x] v10/v11 на Kaggle упали: v10 — старый bmm-код (пуш не долетел), v11 — CUDA OOM в
      `_recompute_J` (3 копии W_lm [151936,1024] fp64 = 3.7 ГБ держались на GPU + base/quant J
      по 2.15 ГБ). Причина 409 на пушах — id в metadata ≠ slug из title; канонический
      slug `e-39b-ptq-perturbation-gate-gptq-4-bit-lm-head`, id поправлен в metadata и в
      push_all.py (иначе каждый push = 409).
- [x] OOM-фикс: полные копии W_lm (fp64/fp32 + оба квантованных уровня) живут на CPU, в цикл
      по слоям уходит только M_K [128,1024] (`.to(dev)`); PPL поднимает по одному fp32-уровню
      и освобождает; `_recompute_J` пишет в срез выхода (без лишнего [nc,S,K,D] временного).
      Локальные самопроверки: `_recompute_J` vs цикл по позициям 7e-7; GPTQ vs RTN на
      ранговых активациях — act-metric в 200 раз лучше RTN (Frobenius больше — норма для GPTQ).
- [x] v12 (id slug + OOM-фикс) дошла до результата на T4 за 532 с; самопроверки:
      base-J vs кэш max 7.77e-07 (tol 1e-5), κ base 0.01623 ≡ E-38d, β/σ₁ 0.8821 ≡ E-38e.
      JSON → results/exp39b_ptq_perturbation.json.
- [x] Результат разложен по докам: docs/E39.md §2.7 (реализация + отклонения от ТЗ + таблица),
      docs/ref-exp.md (карточка E-39b между E-39a и E-39c_pre + строка индекса),
      docs/paper.md (§4.8 строка «Что получили» + канальная метка, §5 строка серии E-39,
      §6 п.18 — три границы: readout-only слайс vs tied-deploy, точен только L23,
      ветка «κ ×10 ⇒ связь с PPL» не тестировалась), docs/E39b_fixes.md §8 (закрытие §5/§6).

### NEXT (открытые нитки)
- [ ] E-39e (full backprop: PPL/bridge в loss) — единственный оставшийся путь серии E-39
      после E-39d 0/12 SUCCESS (связующее ограничение — PPL).
- [ ] E-39c заблокирован pre-gate (KL 4.986 > 0.05); E-39d/e каузальный тест drift↔flatness.
- [ ] paper_add-следы не требуются (серия paper_add закрыта, 8/8).
- [ ] Push ветки в origin (19+ коммитов впереди, push не запрашивался).

## Границы (не менять)
- Do NOT touch the readout-linearized J formula or the L23 self-check (verified 1.6e-7).
- Do NOT quantize activations or embed_tokens (readout-only PTQ per spec + E39 §14.1).
- Do NOT change C4 split skips (8192 cal / 24576 PPL fallback) or GPTQ params
  (4-bit, group 128, damp 0.01, desc_act) — all per spec §2.
- Do NOT recompute E-38a/i/E-16b results; base κ(L23)=0.0162 (E-38d) is the reference.
