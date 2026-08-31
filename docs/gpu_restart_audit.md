# GPU-перезапуск: какие эксперименты не выразительны в текущем формате

> **Статус:** Готово (2026-08-31) — все эксперименты пройдены; итоговая таблица + GPU-очередь в конце. commit 8a4638a (v1) → финальный commit.
> **Критерий «не выразительны»:** (a) стат. мощность недостаточна (N/S/K малы), (b) результат
> самопротиворечив, (c) метрика/формула неверна, (d) баслайн слишком слаб/отсутствует,
> (e) «minimal run» — данные не финальные.
> **Адекватность кода (по требованию):** для каждого — сверка КОДА с замыслом (spec в
> `docs/Exps.md` / `paper.md`): (i) параметры (n_samples, K, max_length, слои, пары) — тот
> диапазон, что заявлен? (ii) формулы метрик — считаем то, что написано в статье? (iii)
> фолбэки/тихие подмены (данные, модели, дефолты) — не изменили ли смысл эксперимента?
> Если код ≠ замысел → эксперимент «мертв» вне зависимости от результата.
> **Исход:** `docs/paper_audit.md` (критический разбор), `docs/deferred_items.md`.
> **GPU-квота:** weekly 30h исчерпана (08-31) → пересчёты только после сброса.
> Формат записи: **замысел → код (адекват?) → результат → проблема → что переделать → prio**.

---

## Итоговая таблица (final)

| Exp | Проблема | Тип | GPU re-run? | Prio |
|-----|----------|-----|-------------|------|
| **13** | overlap_hidden **good** (0.015→0.998); bridge 1.0-sentinel (3/5 pts), monotonicity −0.894 artifact | sentinel (paper) + N=8 thin | paper-fix; optional 24-layer | MEDIUM |
| **14** | adequate (n=8 real); JSON missing `params` | reproducibility | no (add params) | LOW |
| **11** | `recon_fidelity` GROWS with r; K=32 vs K=256 mismatch | likely bug + param mismatch | **YES** (debug 1 layer → r-sweep) | MEDIUM-HIGH |
| **4_k** | **BROKEN**: self-bridge `jacobian_bridge(J,J,E,E)`, no 4B loaded → all-zeros | code bug (wrong thing) | **YES** (load 4B, K→256, n≥4) | **HIGH** |
| **15** | 1.0-sentinel (2/4 pts); −0.949 identical across 4 schemes | sentinel (paper) | paper-fix; optional more anchors | MEDIUM |
| **2** | n=3, K=8, sparse (no full 32×24 map) | thin | YES (n=8, K=64, full map) | LOW-MED |
| **7** | CKA≈0 vs bridge 1→0.8 — contrast works as intended | — | no | none |
| **3** | clean ablation (corr 0 / no-corr 0.7 / random 1.0) | — | no | none |
| **1/5/6b** | self-bridge legitimate (identity/perm/condition-number) | — | no | none |
| **10a/b/d** | a,d OK; b n=1 thin (gauge invariance robust) | thin | optional (b n=8) | LOW |
| **6** | n=16 OK but bridge_means sentinel 1.0; spearman −0.09 partly artifact | sentinel (paper) | paper-fix; optional rerun | MEDIUM |
| **12a/b/c** | quantization destroys both (KL~15) — «comparative destruction» not preservation | framing | paper re-frame; optional smaller-ε | MEDIUM |
| **16** | **VOID**: α was a trust-region cap, not a dose — at α≥0.1 the cap never binds → bit-identical runs; "α\*=0.01" = first-slope artifact | design bug (v1 uninterpretable) | **YES** — v2 dose-curve rebuilt (one unclamped run, per-step (‖ΔW‖, bridge, KL), α\*=knee) **~5 min T4** | **HIGH (cheap)** |
| **18/19** | GDN pairs fine here (logit-KD/destruction, not o_proj) | — | no | LOW |
| **17** | dual-cadence «too weak» (pair 2 half at 300) | underpowered | **YES** (500, lr 3e-3 — built, quota) | **HIGH** (pending) |

---

## Exp 13 — доменная специфичность ESG/мостов

**Замысел (Exps.md):** 4 домена (wiki/math/code/chat), n=4, K=64, max_len=64; слои
[0, n/4, n/2, 3n/4, n−1]; топ-8 eigenvectors ESG на слой; overlap подпространств между
доменами; bridge 4B→0.8B по sampled-парам (step=len(mapping)//5). Ожидания: gauge (v1)
universal (overlap>0.9), дифф. подпространства доменно-специфичны, bridge монотонен (ρ≤−0.7).

**Код vs замысел (проверка адекватности):**
- `n_samples=8` (код) vs `n=4` (spec) — код лучше spec (8>4).
- TWO overlap-метрики (token + hidden) — spec упомянул один, extra = бонус.
- **paper_audit's «overlap = 1» — WRONG**: `compute_esg` returns eigenvectors **[8,K]** (orthonormal rows, `[:, idx].T`), `_subspace_overlap` делит на `shape[0]=8` = ‖U₁U₂ᵀ‖²_F/8 = Σcos²(principal angles) — **correct**, 1 iff same subspace, 0 if orthogonal. 0.30–0.42 plausible. Audit's identity holds only for square full bases.
- **token_set SHARED across domains** (line 31: sum softmax, top-K) → token-space overlap compares identical K-spaces = valid.
- ⚠️ `jacobian_bridge` returns **sentinel 1.0** when `_valid_mask.sum()==0` (`bridge.py`: `if mask.sum()==0: return 1.0`) — NOT a real cosine loss. Mask = rows with `J_S` or `J_S_target` norm < 1e-6.

**Результат (exp13_domain_esg.json, N=8, K=64):**
- **overlap_hidden (the good one)**: layer 0 ≈ 0.015–0.044 (domain-specific), monotonically ↑ to layer 23 ≈ 0.994–0.998 (universal). Clean domain-grad across all 6 cross-pairs. **Выразительно.**
- **overlap_token**: 0.30–0.42 (early) → 1.0 (layer 23). Also clean.
- **bridge: 3 of 5 points = exactly 1.0 in ALL domains** (layers 0→0, 6→4, 12→9) = the sentinel, not real loss. Only 2 of 5 points (18→13, 24→18) are genuine cosine. **monotonicity −0.8944271909999157 identical for all 4 domains** — artifact of the shared 1.0 plateau (rank ties in 5-point Spearman), NOT a data leak.

**Проблема:**
1. `jacobian_bridge`'s `1.0` sentinel (mask empty, early layers) reported in the paper as a real loss value + used in monotonicity → the "bridge decays monotonically" claim rests on 2 real points + 3 sentinels.
2. Root cause of empty early-layer mask: early-layer `J_S` or `J_S_target` (via `J_L@E_Lᵀ@E_S@G_inv`) has norm < 1e-6 — early layers have small-norm Jacobians (plausible) or the pinv(G) amplifies noise (bug). Need to distinguish.
3. N=8 power for Spearman: 5 points is thin (audit right on this, even if the "-0.95 impossible" was a misread of the code).

**Что переделать на GPU (после сброса weekly 30h):**
1. **Debug first (no GPU needed, but 0.8B on GPU is fast)**: for one early layer (0→0), dump `‖J_S‖` and `‖J_S_target‖` row-norms to confirm why the mask is empty. If early `J_S` is legitimately tiny → early layers truly have no decodable direction → document, exclude from monotonicity (use only 2 real points OR add intermediate layers so the curve has ≥4 real points).
2. **Fix the paper**: drop the 1.0 sentinel values from the monotonicity claim (they're not a cosine loss); or replace sentinel with `None` and compute Spearman on the non-None points only (N=2 minimum, ideally 4+).
3. **Optional GPU**: rerun bridge with `n_samples=16`, `K=128`, all 24 layers (not 5 anchors) so the curve has 24 real points and the early-layer pattern is a statement, not an artifact.
4. The **overlap_hidden** (0.015→0.998) is the real, beautiful result — promote it to the paper's main bridge figure; the current 5-anchor version under-sells it.

**Prio:** MEDIUM — the overlap_hidden result is already good and needs no recompute; the bridge-1.0-sentinel is a paper-fix (no recompute) + optional GPU for a 24-layer curve.

---

## Exp 14 — SNR и Participation Ratio per layer

**Замысел:** 0.8B, все 24 слоя, n_samples=8, K=64; PR, gauge_share, SNR per layer; PNG PR/SNR vs depth (GDN/full).

**Код vs замысел (адекватность):**
- ✅ `n_samples=8, K=64` (code) = spec. **НЕ minimal run** — `snr_per_sample` has **8 distinct values** → run real, deferred_items' "n=1 minimal" claim is **stale/wrong**.
- ⚠️ **JSON не записывает n_samples** (top keys = layer indices only) → can't tell from JSON which n produced the run. Trivial fix: record n_samples/K/max_length in JSON (like exp11a's `params`).

**Результат:** PR растёт с глубиной (6.4→8+), gauge_share падает, SNR_sample падает к поздним слоям — как ожидалось. Выразительно (24 слоя × 8 сэмплов).

**Проблема:** только репродуктивность (n/K не в JSON). Нет GPU-recompute.
**Что сделать:** добавить `params` в JSON (trivial, no GPU). Опц.: bootstrap-CI (N=8 thin). **Prio: LOW** (не re-run).

---

## Exp 11 — defrag (fidelity / cost / bridge)

**Замысел:** exp11a = defrag reconstruction of J (r=4/8/16/32) vs full vs raw (output: KL/TV/cos/top1); exp11b = cost (raw vs defrag time/FLOPs/storage) + recon_fidelity.

**Код vs замысел (адекватность) — KEY:**
- ⚠️ **exp11a и exp11b use РАЗНЫЕ params**: exp11a = **K=32, n=4** (params `{K:32, n:4, max_len:48}`); exp11b = **K=256, n=2** (code line 147-148). Audit's «contradiction» — **apples-to-oranges** (two differently-configured runs).
- exp11a: output KL **r-invariant** (dr4=dr8=dr16=dr32=gf=8.517) — clean «defrag preserves output at r=4».
- **exp11b: `recon_fidelity_rel = ‖J_tilde−J_p‖/‖J_p‖` GROWS with r** (19→31→48). **ROOT CAUSE (this session, code-level)**: v1 computed `J_tilde = U_r @ J_r` with J_r = U_r·J → J_tilde = U_r·U_rᵀ·**J** (projection of the RAW J) compared against **J_p** — two different matrices. At r=K the projector is the identity → metric = ‖J − J_p‖/‖J_p‖ = the **gauge gap, a constant independent of r** (10–48× — matches the observed plateau/growth); residual r-dependence = projecting J (not J_p) in G_p's eigenbasis. The intended truncation tail ‖U_r U_rᵀ·J_p − J_p‖ never entered. **Fix applied** (`experiments/exp11b_defrag_cost.py`): `J_tilde = U_r @ U_r.T @ J_p` — monotone decreasing, 0 at r=K by construction. The defrag *basis* (eigh of G_p = J_p·J_pᵀ) is sound; the *cost* measurements (times/speedups/storage) stand.

**Проблема:**
1. exp11b recon_fidelity — **metric bug found + fixed** (not a basis bug).
2. exp11a/exp11b inconsistent params (K=32 vs K=256) → audit's «contradiction» partly artificial.

**Что переделать:**
1. **CPU verify (cheap, no model-forward changes)**: recompute the fixed recon_fidelity for r = 1,4,8,16,32,64,128,256 on ONE layer/sample from an already-computed J_p — curve must decrease monotonically to 0 at r=K. (A few matmuls on [K,D]; minutes.)
2. **GPU re-run only if the paper needs fresh T4 cost numbers** (times/speedups) — otherwise the existing exp11b cost data + the fixed fidelity curve is enough.
3. **Paper fix**: state exp11a (output, K=32) vs exp11b (recon, K=256) are different measurements, not a contradiction; note the v1 fidelity metric was void and replaced.

**Prio: MEDIUM-HIGH** — likely a real bug in the defrag recon (the headline «defrag is cheap AND accurate» claim rests on exp11b's fidelity, which is currently non-sensical).

---

## Sweep: remaining experiments (params + red flags)

### exp4_k_sensitivity — **BROKEN (self-bridge bug, top priority)**
- **Code**: `loss = jacobian_bridge(J, J, E, E, probs)` (line 59) — **same J and E twice** (0.8B self). `main()` **never loads the 4B** (`from_pretrained(model)` only, `model="Qwen/Qwen3.5-0.8B"`).
- **Intent** (spec): measure how the bridge loss responds to K (token set size). **But a self-bridge is the identity case** (`J_L == J_S`, `E_L == E_S`) → `loss ≈ 0` by construction for every K.
- **Result**: **all values = 0.0** for K=4/8/16/32, all layers (avg ≈ 0 to 8 decimals). The "sensitivity" is measuring a constant zero.
- **Verdict**: **not testing what was intended**. Fix = load 4B + `bridge(J_L, J_S, E_L, E_S, probs)` (mirror `exp4_u_sensitivity` which does this correctly at line 92) + n_samples≥4 + K values up to 256. **GPU re-run** (4B+0.8B, K sweep). **Prio: HIGH** — this is the cleanest "we tested the wrong thing" in the repo.

### exp15_gdn_anchors — **sentinel bug (same as exp13) + thin N**
- **Params**: n_samples=4 (per scheme), K=256 (per spec), 4 anchor schemes (A=full, B=post-full, C=pre-full, D=full−2).
- **Result**: all 4 schemes show **identical monotonicity −0.949** and A_vs_B=C=D=1.0. **Cause**: pairs `3->3` and `11->7` have **mean=1.0 with values=[1.0,1.0,1.0,1.0]** — the `jacobian_bridge` **sentinel** (empty valid mask) again, just like exp13. Only the last 2 of 4 pairs are real. The identical −0.949 across all 4 schemes is a rank-tie artifact of the shared 1.0 plateau; the real difference between schemes is only in the 2 real points (max_abs_diff=0.121).
- **Verdict**: the "anchors don't matter" conclusion rests on 2 real points per scheme; the 1.0 plateau is not data. **Paper fix**: drop sentinels from the monotonicity + Spearman(A,B). **GPU re-run (optional)**: more layers (all 6 full-attention anchors) so the real curve has ≥4 points; also K=128 to reduce the empty-mask incidence (smaller K → fewer near-zero rows).

### exp2_heatmap — thin (n=3, K=8)
- **Params**: n=3, **K=8**, max_len=64. 32×24 full map never generated (only sparse windows per deferred_items #5). K=8 is very small — bridge on 8 tokens is noisy.
- **Verdict**: thin but directionally OK (losses ≈ 1.0 for far pairs, decay toward min_at). **GPU re-run** to produce the 32×24 full map at n=8, K=64 (deferred item). **Prio: LOW-MEDIUM** (visualization, not a claim-breaker).

### exp7_cka_baseline — OK
- n=5, K=8, 4B+0.8B. CKA ≈ 0 (0.001) across all pairs; bridge 1.03→0.796 decaying; spearman(cka,bridge)=−0.77. The **contrast** is the point: CKA (representation similarity) ≈ 0 while the bridge captures meaningful decay. Works as intended. **Prio: NONE** (adequate as a contrast baseline).

### exp3_ablation — OK
- n=4, K=32. with_correction ≈ 0 (1e-8), without_correction ≈ 0.7, random_orthogonal ≈ 1.0. Clean ablation showing the Gramian correction is necessary. **Prio: NONE**.

### exp1_identity / exp5_embedding_perm / exp6b_condition_number — OK (self-bridge legitimate)
- exp1: identity ~1e-8 (n=1, K=4 — tiny but it's a determinism/identity check, not a measurement). exp5: perm vs baseline vs inv_perm (self-bridge with permuted E — the permutation is the signal, not the cross-model gap). exp6b: condition number of EᵀE. All legitimately self-model. **Prio: NONE**.

### exp12a/b/c — **needs deeper look (audit flagged quantization)**
- Audit: «KL in both variants ≈ 15, PPL ~10⁶–10⁸ — complete destruction; comparing Δh_map≈0.24 vs Δh_uniform≈0.95 as evidence the map "directs quantization" is invalid — both are unusable.»
- **Verdict (pending deeper read)**: the experiment as run is **comparative destruction** (map-based perturbation destroys *less* than uniform at the same KL), not "the map preserves the model". Either re-frame in the paper ("map is the minimal-sensitivity direction") or re-run with smaller ε so the model isn't destroyed. **GPU re-run (optional)**: sweep ε smaller to find the regime where the map's advantage is visible without total destruction. **Prio: MEDIUM** (paper re-frame is the cheap fix; re-run is optional).

---

## Newer experiments (10 / 6 / 16 / 18 / 19)

### exp10a/b/d (gauge) — mostly OK, one thin
- **exp10a** (n=4, K=32): gauge_share varies by layer (0.11→0.62→0.17), raw vs centered vs softmax eigen-spectra differ wildly (raw trace 1.1e7 vs softmax 2438). Works.
- **exp10b** (n=**1**, K~32, 7 layers): raw/centered/softmax gauge bridge curves ≈ same (avg 0.787/0.755/0.774) → "gauge choice doesn't matter." **n=1 is thin** (single-sample bridge is noisy) but the 3-gauge agreement + 7-layer shape makes the conclusion robust. **Prio: LOW** (optional rerun n=8).
- **exp10d** (n=4): common-mode c vs hidden h per layer; spearman varies (−0.4..0.4). OK.

### exp6_kl_baseline — **sentinel artifact in the headline correlation**
- n=16, K=64, 11 layer pairs. **spearman_kl_vs_bridge = −0.094 ≈ 0** ("KL loss doesn't correlate with the bridge" — a null/interesting result). **BUT `bridge_means=[1.0, 1.0, ...]`** — the early pairs (0->0, 3->2, ...) are the **sentinel 1.0** again. The −0.09 spearman is partly computed over sentinel points. **Paper fix**: exclude sentinel pairs before computing the KL-vs-bridge correlation. **GPU re-run optional** (n=16 is OK; the issue is the sentinel, not N). **Prio: MEDIUM**.

### exp16_readout_pull — **delta_W saturates identically at alpha 0.1/0.3/1.0 (possible clamp/bug)**
## exp16_readout_pull — VERDICT: design bug (not a clamp) — re-run cheap

- **ROOT CAUSE (this session, code-level)**: `alpha` was never a dose knob. In v1 each pair ran a 10-step LR trajectory with a per-step trust-region cap on ‖ΔW‖ ≈ 0.01·‖W‖; at α≥0.1 the cap never binds, so the trajectories are **bit-identical** (bridge 0.7008 / ‖ΔW‖ 14.276 identical at 0.1/0.3/1.0 — not a clamp). The "α*=0.01" headline = first-slope artifact: at α=0.01 the cap binds on the FIRST step and ΔW ≈ 0.01·‖W‖. The v1 "dose curve" was never a dose curve.
- **Fix (applied, in place)**: v2 = one unclamped run (trust-region off) → per-step (‖ΔW‖, bridge gap, KL) → the real dose curve; α* = knee (first step where |slope| ≥ 0.9·max). Stage-0 targets cache consumed as-is (v2 format, 175/176 pairs; 1 null-target pair honestly flagged). v1 JSON → results/voided/. **GPU re-run: one unclamped run on 0.8B ≈ 5 min (v1 log: ~3 s/step).** exp19 consumes this α* (see exp19).

### exp16_readout_pull v2 / exp19_destruction v2 — rewritten this session


- **exp18**: logit-KD on 8 pairs (incl. GDN 24_18, 28_21), r=4, steps=100, bridge 0.606/0.492 — **GDN fine here** (KL on outputs, not o_proj). No re-run.
- **exp19 v2 (rewritten)**: 4 scenarios (full 176-pair set, full edit, no-edit baseline, α*=knee from the exp16 v2 JSON — refuses to run without it) on the LAST layer (l23) with correct per-target input_ids pairing. v1 consumed a VOID artifact (α*=0.01) and fell back to the null target 0_0 (input mismatch → garbage) — voided → results/voided/. GPU re-run: 4 short runs, **~10 min T4**.
---

## ИТОГО (summary table) — what to re-run on GPU vs paper-fix vs leave

| Exp | Problem | Type | GPU re-run? | Prio |
|-----|---------|------|-------------|------|
| **exp4_k_sensitivity** | **BROKEN**: self-bridge `jacobian_bridge(J,J,E,E)`, no 4B loaded → all-zeros | code bug (wrong thing tested) | **YES** (4B+0.8B, K sweep to 256, n≥4) | **HIGH** |
| **exp11 (defrag)** | `recon_fidelity` GROWS with r (19→31→48) — **metric bug found+fixed**: v1 compared U_r·U_rᵀ·J against J_p (at r=K = gauge gap, r-independent); now projects J_p (monotone ↓, 0 at r=K) | metric bug (basis itself fine) | CPU-verify fixed curve (no GPU); GPU re-run only if fresh cost numbers needed | **MEDIUM** |
| **exp17 (power-up)** | dual-cadence "too weak" (pair 2 half at 300) | underpowered | **YES** (500 steps, lr 3e-3 — built, blocked by weekly quota) | **HIGH** (pending) |
| **exp16_readout_pull** | delta_W saturates identically at alpha 0.1/0.3/1.0 | possible clamp/bug | investigate (CPU) then **YES** if bug | **MEDIUM-HIGH** |
| **exp13_domain_esg** | bridge 1.0 sentinel (3/5 pts); monotonicity −0.894 identical artifact | sentinel (paper) + N=8 thin | paper-fix now; **optional** 24-layer/K=128 rerun | **MEDIUM** |
| **exp15_gdn_anchors** | same 1.0 sentinel (2/4 pts); −0.949 identical across schemes | sentinel (paper) | paper-fix; **optional** more anchors | **MEDIUM** |
| **exp6_kl_baseline** | bridge_means sentinel 1.0; spearman −0.09 partly artifact | sentinel (paper) | paper-fix; optional rerun | **MEDIUM** |
| **exp12a/b/c** | quantization destroys both variants (KL~15) — "comparative destruction" not preservation | framing | paper re-frame; **optional** smaller-ε rerun | **MEDIUM** |
| **exp2_heatmap** | n=3, K=8, sparse (no full 32×24 map) | thin | **YES** (n=8, K=64, full map) | **LOW-MED** |
| **exp10b** | n=1 thin (gauge invariance) | thin | optional (n=8) | **LOW** |
| exp1 / exp3 / exp5 / exp6b / exp7 / exp18 / exp19 | OK (self-bridge legitimate / clean ablation / contrast works / logit-KD on GDN fine) | — | no | none |
| exp14 | adequate (n=8 real); JSON missing `params` | reproducibility | no (add params to JSON) | **LOW** |

### Paper-only fixes (no GPU, do first)
1. **exp13/15/6**: exclude the `1.0` sentinel points from every monotonicity / Spearman / bridge-mean computation (they're not cosine losses). Re-state the claims on the real points only.
2. **exp11a vs 11b**: state the two use different K (32 vs 256) — not a contradiction, different measurements; or re-run both at the same K.
3. **exp12**: re-frame as "map is the minimal-sensitivity direction under destructive perturbation," not "map preserves the model."
4. **exp14**: add `params` (n, K, max_length) to the JSON for reproducibility.

### GPU queue (after the weekly 30h quota resets) — in priority order
1. **exp17 power-up** (500 steps, lr 3e-3) — already built + committed; just needs the slot.
2. **exp4_k_sensitivity** — fix to `bridge(J_L, J_S, E_L, E_S)` + load 4B + K up to 256 + n≥4 (4B+0.8B, 1–2h).
3. **exp11 defrag** — CPU-verify the fixed recon_fidelity curve (r=1..256, eigh+matmuls only, no model); GPU re-run only if fresh T4 cost numbers are needed.
4. **exp16_readout_pull v2** — dose-curve re-run (one unclamped run, per-step ‖ΔW‖/bridge/KL, α*=knee). 0.8B only, **~5 min on T4** (v1 log: ~3 s/step). Run BEFORE exp19 (exp19 must consume a real knee, use pair 31_23, add a no-edit baseline — see exp16 section).
5. **exp2_heatmap** — full 32×24 map at n=8/K=64 (4B+0.8B, ~1h).
6. Optional: exp13/exp15 24-layer/6-anchor rerun; exp6 rerun; exp10b n=8.

**Total estimated GPU:** ~8–10h (fits one weekly 30h window with room for the exp17 power-up + retries).


