# GPU-аудит: какие эксперименты не выразительны в текущем формате

> **Статус (2026-09-01):** полная сверка ВСЕХ `experiments/*.py` (код + замысел + params;
> §L-аудит 09-01 влит, старая v1-таблица 08-31 заменена). Итоговая таблица — ниже,
> GPU-очередь — в конце.
> **Критерий «не выразительны»:** (a) стат. мощность недостаточна (N/S/K малы),
> (b) результат самопротиворечив, (c) метрика/формула неверна, (d) баслайн слишком
> слаб/отсутствует, (e) «minimal run» — данные не финальные.
> **Адекватность кода:** для каждого — (i) параметры (n, K, max_length, слои, пары) —
> тот диапазон, что в spec (`docs/Exps.md`)? (ii) формулы метрик — то, что в статье?
> (iii) фолбэки/тихие подмены (данные, модели, дефолты) — не изменили ли смысл?
> Код ≠ замысел → эксперимент «мертв» вне зависимости от результата.
> **Исход:** `docs/paper_audit.md`, `docs/deferred_items.md`; live-статус exp17 —
> `docs/HANDOFF_exp17_powerup.md`.
> **GPU-квота:** Kaggle weekly 30h (исчерпана 08-31, сброс = конец недели Mon–Sun).
> Формат записи: **замысел → код (адекват?) → результат → проблема → что переделать → prio**.

---

## Итоговая таблица (все экспериментные скрипты)

| Exp | Вердикт / проблема | Action | GPU? | Prio |
|-----|-------------------|--------|------|------|
| **exp4_k_sensitivity** | **BROKEN**: self-bridge `jacobian_bridge(J,J,E,E)`, 4B не загружен → все 0.0 (identity-кейс, не «sensitivity») | fix: `bridge(J_L,J_S,E_L,E_S)` (по образцу exp4_u:92) + load 4B + K→256 + n≥4 | **YES** | **HIGH** |
| **exp16 v1→v2** | v1 **VOID** (results/voided/): α = trust-region cap, не dose; cap не binds при α≥0.1 → 0.1/0.3/1.0 bit-identical; «α*=0.01» = first-slope artifact. v2 rewritten: one unclamped run, dose-curve, α*=knee/null | run v2 (0.8B, **~5 min T4**) | **YES** (cheap) | **HIGH** |
| **exp17_lora_distill** | не сломан, но **SLOW**: 2nd-order (HVP) refresh растёт с длиной decoder-tail; full 7-pair = 14 pair-runs ≈ 40–45h > 12h-лимита kernel | push `--pairs 4_3` + single order (≈4.6h; both orders ≈9.2h тоже влезает) | **YES** | **HIGH** |
| **exp11 a/b/c (defrag)** | 11b-v1: recon_fidelity **растёт** с r = **metric bug (found+fixed)**; 11a K=32 vs 11b K=256 — разные замеры, не contradiction; **11c: defrag выглаживает bridge (ρ≈0) — реальный negative result, в tension с 11a** | CPU-verify кривая 11b (r=1..256, без модели); paper: a/b/c = 3 разных измерения; 11c optional n=8 | CPU / opt | MEDIUM |
| **exp13_domain_esg** | overlap_hidden **good** (0.015→0.998); bridge: 3/5 pts = sentinel 1.0 → monotonicity −0.894 identical во всех доменах = rank-tie artifact | paper-fix (exclude sentinels; overlap_hidden в main figure); optional 24-layer/K=128 | opt | MEDIUM |
| **exp15_gdn_anchors** | то же sentinel (2/4 pts = 1.0); −0.949 identical во всех schemes = rank-tie artifact; реальное различие только в 2 real pts | paper-fix; optional 6 anchors + K=128 | opt | MEDIUM |
| **exp6_kl_baseline** | n=16 OK; bridge_means = sentinel 1.0 в ранних парах → spearman −0.09 partly artifact | paper-fix (exclude sentinel pairs); optional rerun | opt | MEDIUM |
| **exp6_l18_snr** | **ДАННЫЕ ВОССТАНОВЛЕНЫ** (Kaggle 05.09, 55s T4): L18 SNR=2.80, PR 2.1→16.9, gauge_share 0.60→0.19; `results/exp6_l18_snr.json` | done — data in repo; paper section optional (see §exp6_l18_snr note: «paper exp6»=KL, not this) | n/a | done |
| **exp12 a/b/c** | код correct; при выбранном ε/budget **оба** варианта разрушают модель (KL≈15, PPL 1e6–1e8) — «comparative destruction» (Δh_map 0.24 < Δh_uniform 0.95), не preservation | paper re-frame (free); optional ε/budget 10× sweep (~1h) | opt | MEDIUM |
| **exp4_top_p_tokens** | код CORRECT (re-verified 09-01: `probs` внутри per-set loop — v1-audit misread); **never ran** (4B on CPU) | run, если token-set-bias claim в paper (~1h) | opt | LOW-MED |
| **exp2_heatmap** | thin: n=3, K=8, sparse (no full 32×24 map); bridge on 8 tokens noisy | YES (n=8, K=64, full map, ~1h) | opt | LOW-MED |
| **exp14_snr_pr** | adequate (n=8 real, 8 distinct snr values); код пишет `params` (`0514675`), но сохранённый JSON — до fix (без `params`) | trivial CPU re-run для записи JSON | no | LOW |
| **exp10b_gauge_bridge** | n=1 thin; conclusion robust (3 gauges agree, 7-layer shape) | optional n=8 | opt | LOW |
| **exp2a_cka_identity** | v1 bug: H1≡H2 (same tensor twice) → vacuous CKA control; **FIXED** (disjoint groups + `--self-check` **PASSED**: CKA(X,X)=1, CKA(X,5X)=1, translation/perm ~1e-8); never ran | one CPU run (~10 min, 0.8B+4B) — within-model point | no (CPU) | LOW |
| exp1 / exp3 / exp5_perm / exp6b | self-bridge legitimate (identity ~1e-8 / clean ablation 0–0.7–1.0 / perm = signal / cond(EᵀE)) | none | no | none |
| exp4_u_sensitivity | correct cross-bridge (pattern для exp4k-fix) | none | no | none |
| exp5_layer_sensitivity | n=1/K=8, descriptive (‖J‖ 272→16.8), no claim | none (optional n=8/K=64 if в paper) | no | none |
| exp7_cka_baseline | contrast works: CKA≈0.001 vs bridge 1.03→0.796; null linear-CKA inflated при N≪D (indep 64×128 ≈0.34) → measured 0.001 **far below** → conclusion stands | paper note | no | none |
| exp8_svd_ablation | self-bridge control exactly right: truncated G_inv k=16→1024: 0.89→0.30→2.4e-8 (к≈512–1024 нужен) | none | no | none |
| exp9_esg | ran (data + 2 PNGs); nuance: token-set = top-K one sample (less rigorous cross-layer) | none | no | none |
| exp10a / 10c / 10d | OK (10c: top-K 90.6% @K=64 vs top-P K̄ 398/2152 **unbounded** — state the cost) | none | no | none |
| exp18_logit_kd | GDN pairs fine (logit-KD = KL on outputs, not o_proj): bridge 0.606/0.492 | `comparison_with_exp17: None` — slopes считать вручную из двух JSON | no | none |

---

## Sentinel 1.0 — shared root cause (exp13 / exp15 / exp6)

`prism/bridge.py:31` — `if mask.sum() == 0: return 1.0`. Mask = строки с
‖J_S‖ или ‖J_S_target‖ < 1e-6. Ранние слои → малые по норме Jacobians → маска
пустая → **1.0 в каждом sample** = sentinel, не cosine loss. Отчёт в paper как
реальное значение → monotonicity/Spearman считаются по rank-ties (−0.894 и −0.949
identical = artifact).
**Paper-fix (free):** исключить sentinel-точки из monotonicity / Spearman /
bridge-mean; переформулировать claims на real points.
**Debug (CPU, без forward):** для layer 0→0 выгрузить row-norms ‖J_S‖,
‖J_S_target‖ — ранний слой реально недекодируем vs pinv(G) amplifies noise (bug).
**Optional GPU:** больше anchors (all layers / all 6 full-attn) и K=128 (меньше
near-zero rows) → кривая из 4+ real points.

### exp13 (доменная специфичность ESG/мостов)
- **Замысел (Exps.md):** 4 домена (wiki/math/code/chat), n=4, K=64, max_len=64;
  слои [0, n/4, n/2, 3n/4, n−1]; топ-8 eigenvectors ESG на слой; overlap
  подпространств; bridge 4B→0.8B по sampled-парам. Ожидания: gauge universal
  (overlap>0.9), дифф. подпространства доменно-специфичны, bridge монотонен.
- **Код vs замысел:** n=8 (code) > 4 (spec) — better. Two overlap-metrics
  (token+hidden) — extra = бонус. **paper_audit's «overlap = 1» — WRONG**:
  `compute_esg` возвращает eigenvectors [8,K] (orthonormal rows);
  `_subspace_overlap` делит на shape[0]=8 → ‖U₁U₂ᵀ‖²_F/8 = Σcos²(principal
  angles) — correct (1 iff same subspace); 0.30–0.42 plausible. token-set
  SHARED across domains → token-overlap compares identical K-spaces = valid.
- **Результат (N=8, K=64):** **overlap_hidden (the good one)**: layer 0 ≈
  0.015–0.044 → layer 23 ≈ 0.994–0.998, clean domain-grad во всех 6 cross-pairs —
  **выразительно, promote в main figure** (current 5-anchor version under-sells).
  overlap_token: 0.30–0.42 → 1.0. Bridge: 3/5 pts sentinel, только 18→13/24→18 real.
- **Prio MEDIUM**: overlap_hidden не требует recompute; bridge = paper-fix +
  optional 24-layer rerun.

### exp15 (GDN anchors)
- n=4 per scheme, K=256, 4 schemes (A=full, B=post-full, C=pre-full, D=full−2).
  Pairs 3→3 и 11→7 = sentinel [1.0×4]; real только последние 2; max_abs_diff
  schemes = 0.121. «Anchors don't matter» стоит на 2 real pts per scheme.

### exp6 (KL baseline)
- n=16, K=64, 11 layer pairs. spearman(KL, bridge) = −0.094 ≈ 0 (null result) —
  но считается в т.ч. по sentinel-парам (0→0, 3→2, ...). Exclude → пересчитать.

### ✅ exp6_l18_snr — данные ВОССТАНОВЛЕНЫ (Kaggle, 05.09)
**Kaggle re-run УСПЕШЕН** (`voroninmichael/prism-exp6-l18-snr`, 55s T4) → `results/exp6_l18_snr.json`. **Л18-only SNR vs r**, 0.8B, N=8, K=256:
- **SNR = 2.80** (r-invariant; mean²/sd² = 27531.7/7238.7).
- **PR(r): 4→2.1, 8→3.5, 16→5.7, 32→9.9, 64→16.9** (monotone rising).
- **gauge_share(r): 0.602, 0.464, 0.354, 0.264, 0.198** (monotone falling) — deepens spectrum → gauge fraction drops.
- trace_top_r: 6343→19317. **Л18 = linear_attention, not last**.

**Коррекция аудит-мифа:** «paper Эксперимент 6 = L18 SNR (ρ=+2.94)» — **неверно**. Реальный paper §6.6 «Эксперимент 6 (N=16)» = **KL-дивергенция** (exp6_kl_baseline, −0.094) — другой эксперимент. ρ=+2.94 в body paper **нет** (только в утерянном JSON). exp6_l18_snr = standalone L18 SNR-vs-r; в paper section нет → можно добавить (SNR vs r + gauge_share) или держать как supplementary.

**Bug fixed in re-run:** first Kaggle push failed (shape `[256] vs [4]`) — my `_gauge_share_top_r` built `u` in r-dim, not K-dim; fixed `u=1/√K` (K=256, full dim), numerator top-r only. exp14-style, confirmed by run.

---

## exp11 — defrag (fidelity / cost / bridge)

- **11a** (K=32, n=4, max_len=48): output KL **r-invariant** (dr4=dr8=dr16=dr32=
  gf=8.517) — clean «defrag preserves output at r=4».
- **11b** (K=256, n=2): v1 `recon_fidelity_rel` **GROWS** with r (19→31→48).
  **ROOT CAUSE (code-level):** v1 `J_tilde = U_r @ J_r` (J_r = U_r·J) →
  J_tilde = U_rU_rᵀ·**J** (projection RAW J) vs **J_p** — разные матрицы;
  at r=K projector = identity → metric = **gauge gap** (константа,
  r-independent; 10–48× — matches plateau/growth). **FIX APPLIED**
  (`exp11b_defrag_cost.py:260`): `J_tilde = U_r @ U_r.T @ J_p` → monotone ↓,
  0 at r=K by construction. Basis (eigh of G_p = J_p·J_pᵀ) sound; cost
  measurements (times/speedups/storage) stand.
- **11c** (K=256, n=1, 7 pairs, r∈{8,16,32}): reconstruction CORRECT (projects
  J_p) — но **defrag_r FLAT ≈0.97–1.05 across all depth** (ρ vs raw
  0.19/0.50/0.86), no sentinel — flattening real: rank-r (r≤32) truncation
  256-row gauge-free J выбрасывает ровно тот detail, в котором живёт bridge
  signal. **Negative result в tension с 11a** (output-faithful vs
  bridge-destroying). Paper должен state: defrag = output-faithful,
  bridge-destroying — the bridge IS the high-r structure.
- **11a vs 11b «contradiction» (audit)** = apples-to-oranges: разные K
  (32/256) и разные метрики — не contradiction, different measurements.
- **Action:** CPU-verify: recompute fixed 11b fidelity для r=1,4,8,...,256 на
  ONE layer/sample из уже посчитанного J_p (matmuls [K,D], minutes; кривая
  monotone ↓ to 0). GPU re-run только если нужны свежие T4 cost numbers.
  **Prio MEDIUM-HIGH** (headline «defrag is cheap AND accurate» стоял на
  non-sensical v1 metric; после fix — coherent, но bridge-flattening нужно
  отразить).

---

## exp16 / exp19 — readout pull (dose) и destruction

**exp16 v1 — VOID** (results/voided/): α was a one-sided **trust-region cap**
(‖ΔW‖ ≤ α·‖W‖), не dose. При α≥0.1 cap never binds → trajectories **bit-identical**
(bridge 0.7008 / ‖ΔW‖ 14.276 at 0.1/0.3/1.0 — не clamp); «α*=0.01» = first-slope
artifact (at α=0.01 cap binds on FIRST step). v1 «dose curve» never была dose curve.

**exp16 v2 (applied):** one **unclamped** Adam run на lm_head (trust-region block
removed on purpose). Loss = mean per input_ids-paired sample (1−cos(J_S_p,
J_tgt)) + 0.1·KL(p_ed‖p_orig); target = **last-layer pair (31_23, asserted** —
shallow target = null target). Every `eval_every` steps: (dose ‖ΔW‖_F,
α=‖ΔW‖/‖W‖, bridge via fresh J_norm/h-reuse, true KL). **α* (knee) = largest step
with Δbridge ≥ tol**; pull that keeps buying bridge has no knee → **α* = null**
(honest, not first-slope). Stage-0 targets cache consumed as-is (7 pairs;
1 null-target pair flagged). **~5 min T4** (v1 log: ~3 s/step).

**exp19 v1 — VOID** (3 independent reasons): target fell back to 0_0 (GDN
kl_only = NULL target, cross-layer bridge ≈ 0 by construction) → «bridge» =
noise on noise; dose from voided α*=0.01; no no-edit baseline.

**exp19 v2 (applied):** 6 scenarios at the SAME dose ‖ΔW‖ (Frobenius); ~10–15 min T4:
no_edit (THE baseline) / a_random / b_target (ΔJ pull, 1×) / c_10x / d_gauge /
e_diff. Target = last layer (asserted); input_ids pairing (mismatch → skip term,
no silent position pairing); **dose = knee из exp16 v2 JSON — FATAL if missing**
(no silent default). Sanity: baseline bridge vs exp16 step-0 (±0.05) or abort.
Paper читает DELTAS от no_edit. **After exp16 v2.**

---

## exp17 — LoRA distillation (dual-cadence power-up)

- **OOM (killed v22/v23 early runs) — FIXED**: A4 chunked bridge backward
  (C=8) → peak = C+1 2nd-order row-graphs, not K; A3 fail-fast (os._exit(3))
  on refresh OOM. `t_J=OOM` markers в 9-pair k=100 log = **STALE** (pre-fix);
  valid post-fix data = completed JSON.
- **SLOW (not broken):** dominant cost = 2nd-order (HVP, o_proj) J refresh —
  единственный `create_graph=True` path; backward through decoder-tail retained
  graph → растёт с длиной tail от l_S (0_0 ≫ 31_23, ~9.5× spread in per-layer
  walls). Regime v23.2c: 2nd-order refresh every k=100 + 1st-order dense
  (no-HVP, h-reuse) every step; lr 3e-3; steps 500.
- **Measured anchor (post-fix):** pair 4_3, k=100, 300 steps, **BOTH orders**,
  elapsed 19724s = **32.9 s/step** (completed, no OOM; `results/exp17_lora_distill.json`).
- **Target cache = 7 pairs** (6 full-attn + 0_0 kl_only), NOT 176;
  `_layer_order` (cuda) = 2 orders → `--pairs ""` = **14 pair-runs ≈ 40–45h**
  (early-layer pairs slowest) — **не влезает в one 12h kernel**.
  `--pairs 4_3` + both orders = 1000 steps ≈ **9.2h** → fits; + single order
  ≈ **4.6h** → headroom (order-symmetry question = второй kernel при желании).
- **Pending action:** после сброса weekly quota push `kaggle_kernel/kernel-exp17`
  с `--pairs 4_3` + single order (build committed, `ea20f07`). topk_report (C5)
  runs with the v2 suite (4B fp16, OOM-skip). exp18's
  `comparison_with_exp17: None` → Δbridge/ΔKL slopes — вручную из двух JSON
  (data present, no code gap).
- Helpers: fd_local (FD spot check — central passes, forward 16% = O(eps)·1000×
  decoder-tail gain), grad_sanity (bridge backward reaches all 4 LoRA params) — OK.
- **Prio HIGH (cheap-ish)** — единственный pending push; blocked только на quota.

---

## Paper-only fixes (no GPU, do first)

**Скоуп-факт (проверено 14-01):** paper (docs/paper.md, v3.3) содержит
эксперименты **1–12c (§5–9) + 13/14/15 (§11 «Доменная специфичность,
глубинная структура, фазовая робастность»)**. exp16/17/18/19 **не в
paper** — только audit + JSON (дописать при v2-публикации). Значит:
«paper-fix» для 13/14/15 = **исправление чисел/выводов в §11** (DONE
14-01, см. items 5/9/10); для 16–19 = написание секций.

1. **exp6 (in-paper):** exclude 1.0-sentinel rows из Spearman / bridge-mean
   (таблица exp6 L_bridge=1.00 в ранних строках — sentinel, см. note в paper).
   → **DONE** (note добавлен 14-01); sentinel-excluded ρ → CPU item 7.
2. **exp11b:** v1-metric void (`‖U_r J_r−J_p‖/‖J_p‖` растёт с r — баг);
   fix в `experiments/exp11b_defrag_cost.py` (c904db3: `‖U_r U_rᵀJ_p−J_p‖/‖J_p‖`,
   monotone, 0 at r=K) но JSON — до fix → **note в paper добавлен**;
   re-run / CPU recompute после quota.
3. **exp11c:** state «defrag output-faithful but bridge-destroying — the
   bridge IS the high-r structure» (defrag_8/16/32 в JSON = flat ~1.0,
   contrast с 11a/11b — different measurements, не contradiction).
4. **exp12:** re-frame как «map = minimal-sensitivity direction; при равном
   destructive budget degrades less» (Δh 0.22–0.63 vs 0.82–2.52), не «map
   preserves the model». → **DONE** (reframe note в paper, 14-01).
5. **exp7/2a:** CKA(4B→0.8B)≈0.001 far below inflated linear-CKA null
   (N≪D: indep 64×128 ≈0.34) → contrast stands; **note в paper добавлен
   (14-01)**. 2a = separate CPU control (queue item 8).
6. **exp16/19:** v1 void; paper читает v2 dose curve; exp19 — deltas от no_edit.
   (exp16/19 не в paper — дописать при v2 publication.)
8. **exp6_l18_snr:** `results/exp6_l18_snr.json` — ✅ **ВОССТАНОВЛЕН** (Kaggle 05.09, 55s T4, в репо). L18 SNR=2.80, PR 2.1→16.9, gauge_share 0.60→0.19. Paper section опциональный.
9. **exp13/15 (in-paper, §11) sentinel note:** первые 3 якоря exp13 и
   первые 2 якоря exp15 = 1.0-sentinel (empty mask, `prism/bridge.py:31`).
   Цитируемые ρ=-0.89 / -0.95 / cross-Spearman 0.971 = **с sentinel**
   (совпадает с JSON «all points»). Paper уже говорит «saturated (1.000)»
   → пометить как sentinel; sentinel-excluded пере-расчёт → CPU.
10. **exp14 (in-paper, §11): DONE (14-01).** К=256→K=64; gauge_share
    «падает 0.31→0.08» → немонотонна (peak 0.48 L3); PR немонотонна
    (провал L0→L3); broken table separator (3 dashes/6 cols) fixed.
    `params` key absent in saved JSON (script 0514675 writes it) →
    тривиальный CPU re-run добавит.

---

## GPU queue (после сброса weekly 30h) — in priority order

1. **exp17**: push `--pairs 4_3` + single order (≈4.6h; both orders ≈9.2h тоже
   влезает) — built, quota-blocked.
2. **exp4_k**: fix + load 4B + K→256 + n≥4 (4B+0.8B, ~1–2h).
3. **exp16 v2**: one unclamped run (~5 min T4).
4. **exp19 v2**: 6 scenarios, dose = exp16 v2 knee (~10–15 min; **after 3**).
5. **exp2**: full 32×24 map, n=8/K=64 (4B+0.8B, ~1h).
6. ~~**exp6_l18_snr re-run**~~ ✅ **DONE (Kaggle 05.09, 55s T4)**: `results/exp6_l18_snr.json` в репо (L18 SNR=2.80, PR 2.1→16.9, gauge_share 0.60→0.19). Paper section опциональный (см. §exp6_l18_snr: «paper exp6»=KL, а не этот).
7. Optional: exp13/15 (24-layer / 6 anchors, K=128); exp10b n=8;
   exp4_top_p (~1h); exp12 ε/budget sweep; exp11c n=8 (~1h).
   CPU (no GPU, local light): **exp14 recompute** (params + sentinel-excluded stats); **exp6 (in-paper) sentinel-excluded recompute** (N=16/K=64 в JSON) → обновить paper (paper-only item 1); **exp11b recompute** (fixed metric, monotone in r) → обновить paper (paper-only item 2); exp2a CPU run (separate control).

**Total ≈ 9–10h обязательных (вкл. exp6_l18_snr ~2h) + ~3–4h optional + CPU** — fits one weekly 30h window
с запасом (12h kernel limit respected: each item ≤ ~9.2h).
