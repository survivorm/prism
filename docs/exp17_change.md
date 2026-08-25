# КОНТРАКТ EXP17 (v23) — ЗАКРЕПЛЁН 2026-08-25. ПЕРЕЧИТЫВАТЬ ПЕРЕД КАЖДЫМ ШАГОМ

> Источник: диктовка пользователя, 2026-08-25. Контракт = что exp17 ДОЛЖЕН сделать
> и какие гарантии/верификации обязательны. Всё ниже имеет приоритет над всеми
> остальными заметками сессии.
> **Каждый пункт — чекбокс `[ ]` → `[x]` по мере прохождения** (gate green = tick).
> Лишнего от старых запусков здесь НЕТ — только вперёд (v23).

## Контекст: почему v23 (2 строки фактов v22 — не план)

- v22 (9 пар, uniform step 3): 5/9 l_S = GDN (нет o_proj) → bridge мёртв по конструкции;
  create_graph J OOM на 7/9; чекпоинтов нет, deltas сгорели. → перевыбор + matrix-free + sanity + ckpt.

## Приоритеты (ранг; #2 = МЕТА-приоритет, #1 существует для #2)

1. **Победить OOM** при вычислении J (R=20, слои ближе к (0,0)).
   **Fallback (stale-J / o_proj-KL-only) ЗАПРЕЩЁН** — run с fallback = невалиден, лучше честный fail.
2. **Exp17 делает то, что задумано** — математически, программно, по духу. **САМОЕ ВАЖНОЕ.**
   (o_proj bridge с реальными градиентами, J correct, pairing correct, deltas apply-able.)
3. **300 шагов + early-stop + 5 чекпоинтов/пару** — от переобучения.
4. **Санити >1мин/4Гб на CPU → Kaggle GPU T4, ≤15 мин** (kill=FAIL).
   Дешёвый локальный чек (B1, 0.8B 1 слой <4Гб) остаётся local пока ≤1мин/≤4Гб.

## PLAN (порядок = приоритеты)

### 0. Re-selection + stage0 redo (ПРИОРИТЕТ 0) — `experiments/exp16_stage0_targets.py`

Правило (user): **l_S = full-attention (из `get_layer_types(mS)` runtime) + l_S=0 (0_0) + last-layer.**

- [ ] **0.1 Runtime filter** (НЕ хардкод — 3mod4-гипотеза {3,7,11,15,19,23} подтверждается из config).
  **Перевыбор по l_S** (mapping many-to-one — итерация по l_L даст dupes):

  ```python
  m      = {l: round(l*(n_S-1)/(n_L-1)) for l in range(n_L)}   # uniform_layer_mapping
  full_S = [i for i, t in enumerate(get_layer_types(mS)) if t == "full_attention"]
  sel    = {0} | set(full_S)                  # l_S=0 (0_0) + full-attn
  pairs  = [(min(l for l in range(n_L) if m[l]==j), j) for j in sorted(sel)]
  if (n_L-1, n_S-1) not in pairs: pairs.append((n_L-1, n_S-1))  # ensure last-layer
  ```

  Ожидаемо (3mod4): **{(4,3),(9,7),(15,11),(20,15),(25,19),(31,23),(0,0)} = 7 пар**.
- [ ] **0.2 Классификация пар** (по `layer_types[l_S]`, НЕ по "крайности"):
  - **l_S ∈ full-S** (3,7,11,15,19,23) → **o_proj bridge** (matrix-free refresh).
    R = n_S−l_S−1: **3→20 (самый OOM-рискованный), 7→16, 11→12, 15→8, 19→4, 23(last)→0 (самый дешёвый).**
  - **l_S = 0 (GDN, нет o_proj)** → **KL-only** (lm_head LoRA; o_proj LoRA и refresh — НЕТ).
  - **Point 6 (user, 2026-08-25): последняя пара 31_23 ИМЕЕТ o_proj bridge — ДА, возможно.**
    Layer 23 = full-attn (3mod4) → есть o_proj. Через **create_graph=True path**: full forward
    даёт h_23 connected к o_proj_23 → `h_norm = norm(h_23)` → J_norm (autograd, **R=0**, no decode)
    → bridge → **o_proj_23 + lm_head**. Это С А М Ы Й дешёвый refresh (R=0).
    **Analytical (no-grad) path — ТОЛЬКО для eval-J** (там o_proj reach не нужен).
    IF config 23 = GDN → тогда 31_23 = KL-only.
- [ ] **0.3 Stage0 recompute** (kernel-exp16, T4): 7 пар × samples → `{l_L}_{l_S}_{si}.pt` v2-кэш
  (J_tgt_p, J_S_p, p, p_L_full, **input_ids**, token_set). Pairing exp17 = **compare input_ids** 
  - Запускать на Kaggle СРАЗУ как сделано. Потом продолжать работу над остальным кодом. 
  (v2-кэш хранит их) — строже, чем parse_si.
- [ ] **0.4** exp17/exp16r/exp19: pair-list = те же 7 (тот же filter).

### A. OOM fix (приоритет 1) — `prism/jacobian.py` + exp17 refresh

**Ядро: matrix-free (из `docs/Exp17_OOM_optimizations.md`).**
v22 OOM = 1024 живых create_graph-графов [D,D]. Fix: [D,D] с живыми графами НЕ строится;
считаются только K=128 строк `J_S = W_rows @ J_norm`.

- [ ] **A1** Refresh per-row `J_i = autograd.grad(h_norm @ W_rows[i], h, create_graph=True)`
  (1 live graph, не 1024) + **A1a** `J_bar = autograd.grad(h_norm @ W_bar, h, create_graph=True)`,
  `W_bar = p @ W_rows` (1 вектор) → softmax coupling `Jp_i = p_i·(J_i − J_bar)`.
  Exact-cond (doc §3): W_rows в графе (grad→lm_head), h connected к o_proj (grad→o_proj),
  J_bar не detached, все K rows, масштаб 1/U.
- [ ] **A1b** `W_rows = lm_head.weight[token_set] + B_lh[token_set]@A_lh` (не full W_eff [V,D]).
- [ ] **A1c** `softmax_jacobian` safe formula `J_p = J − p[:,None]·(p@J)` / `p·(J−J_bar)`;
  НЕ dense `diag(p)−pp^T`.
- [ ] **A1d** **Chunked backward**: C=8 строк → `loss_chunk.backward()` (grads accumulate
  в shared A_lh/B_lh/A_attn/B_attn); live graphs = C (не 1024).
- [ ] **A1e** `torch.autograd.graph.save_on_cpu(pin_memory=True)` around create_graph rows
  ЕСЛИ B2 peak VRAM впритык. **Сначала: GDN-cache + create_graph compat check.**
- [ ] **A2** Peak-VRAM log `torch.cuda.max_memory_allocated()` per pair / refresh / backward.
- [ ] **A3** **Fail-fast, НЕ fallback** (priority 1): OOM после A1 → `[FATAL] OOM on <pair>` + kill.
  Путь use_fresh/degrade из v22 **удаляется**.
- [ ] **A4** Stale path (между refresh): detached [D,D] J_norm (~4MB, no live graphs, no OOM);
  per-step `J_S = W_rows @ J_norm_detached` (W live, J frozen) — exact values.
  **R = n_S − l_S − 1** = decoder-слоёв second-order прогон (jacobian `start=layer_idx+1…n_layers`);
  OOM растёт с R.
- [ ] **A5 НЕ используем** (ломают объект/смысл — doc §4): bf16/fp16 J-цепи (только после
  отдельной valid), low-rank/defrag, truncation token_set, shrink S, freeze o_proj в свежем
  refresh, `h.detach()` вместо живого o_proj пути.

### B. Math verification (приоритет 2)

- [ ] **B1 LOCAL CPU** `exp17_grad_sanity` (0.8B, 1 full-attn слой, <4Гб, ≤1мин → local по #4):
  (i) **smoke_j FIXED**: A_attn AND B_attn NONZERO init в probe → assert `dL/dA>0 AND dL/dB>0`
  (fix ложного OK v22: B=0 init ⇒ dL/dA≡0 on step 0);
  (ii) **finite-difference**: 1-2 строки J_S(create_graph) vs `h_norm(h+eps·e_i)` → rel err < 1e-3;
  (iii) `softmax_jacobian` gauge: identity (J=I) + known-small-matrix.
- [ ] **B2 KAGGLE GPU T4, ≤15 мин** (Q3): full-pair refresh **4_3** (самая ранняя, R=20) c matrix-free
  → assert A_attn/B_attn grads>0 + finite-diff PASS + peak VRAM < 16GB на РЕАЛЬНОЙ 4B.
  **Kill at 15 min = FAIL (no retry).**
- [ ] **B3** Last pair 31_23: create_graph J **R=0** (cheapest) → o_proj_23 reachable (point 6);
  no-grad eval = analytical. Оба pathes logged.

### C. Training loop (приоритет 3)

- [ ] **C1** **300 steps**/pair (GPU).
- [ ] **C2** **Early stop** (Q2=b): fresh **no-grad** J bridge-cos (eval) каждые **N=50**;
  stop after **P=3** consecutive eval-точки без improvement > 0.5%.
  Log: steps run, stopped-at, best eval-cos, N, P.
- [ ] **C3** **Checkpoints: 5/пару** (Q4) at **60/120/180/240/300** → total 5×7 = **35** (7 пар).
  Каждый = cumulative **LoRA A/B** (lm_head + o_proj) + optimizer state +
  running metrics (eval-cos, KL, k, t_J, peak VRAM). Цель: per-pair траектория + rollback.
- [ ] **C4** **Delivery = trained LoRA A/B** (rank-4, ~5MB, <<1GB Kaggle file limit) —
  apply-able объект для rebuild patched small model.
- [ ] **C5 VERIFY TOP-K** (зачем deltas, 2026-08-25). Apply deltas → **3 модели:
  (1) 0.8B base, (2) 0.8B + lora, (3) 4B**. На вопросах того же category (training domain)
  → **top-k для каждой**. Метрика = **как меняется top-k для lora**:
  **ОЖИДАНИЕ: top-k(lora) распределён МЕЖДУ top-k(base-0.8B) и top-k(4B)** — чем ближе к 4B, тем лучше.
  **Мусор = токен, НЕ входящий ни в один из двух base top-k (ни base-0.8B, ни 4B) — чем больше, тем хуже.**
  Quantify: `agreement = |lora_topk ∩ (base_topk ∪ 4b_topk)| / k`; `garbage_rate = 1 − agreement`.
  **Начать с "последних" (final) чекпоинтов; если плохо → смотреть промежуточные.**
  (Точные training-вопросы = **НЕ делать**.)

## GATES (не двигаемся пока green; каждый push — ТОЛЬКО по явной «GO»)

- [ ] **G0 Re-selection**: 7 пар (runtime filter); stage0 recompute → v2-кэш (input_ids) для всех 7.
- [ ] **G1 A1 matrix-free on 4_3** (R=20): no OOM (peak VRAM < 16GB) + o_proj bridge active (grads>0).
- [ ] **G2 Math green**: B1 local + B2 Kaggle GPU (≤15min).
- [ ] **G3 Checkpoint protocol**: local mini-run (2 пары, 40 шагов) → kill на ckpt → resume →
  continuation, deltas match.
- [ ] **G4 Push v23** (T4, 12h) → full run → per-pair JSON + LoRA deltas + **C5 top-k report (3 модели, final ckpt)**.

## SEQUENCE (порядок исполнения)

1. **0** — re-selection (code) + stage0 redo (kernel-exp16, T4) → **G0**
2. **A** — matrix-free rows + J_bar + chunked + peak-VRAM + fail-fast (убрать v22 fallback paths) → **G1**
3. **B1** local CPU → **B2** Kaggle GPU (4_3) → **G2**
4. **C** — 300 / early-stop / 5-ckpt / LoRA delivery / C5 top-k + local mini-run checkpoint check → **G3**
5. Build kernel-exp17 v23 → **user GO** → push → monitor → results (JSON + LoRA deltas + C5 top-k)

## ANSWERS (user, 2026-08-25)

- **Q1** → K=128 (как в stage0 cache; full-V bridge из кэша невозможен — p gauge bf16, no full logits).
- **Q2** → (b) eval-cos (fresh no-grad J) каждые N=50, patience P=3, threshold 0.5%.
- **Q3** → sanity на Kaggle GPU T4, ≤15 мин (kill=FAIL, no retry).
- **Q4** → 5 чекпоинтов/пару (60/120/180/240/300). Total = 5×7 = 35.
- **Q5** → LoRA A/B (apply-able, ~5MB) + **C5 top-k VERIFY: 3 модели (0.8B base, 0.8B+lora, 4B),
  final ckpt first, garbage = токен вне обоих base top-k**. Точные training-вопросы — НЕ делать.
- **Re-selection** → l_S = full-attn + 0_0; l_L = nearest preimage.
  **7 пар: {(4,3),(9,7),(15,11),(20,15),(25,19),(31,23),(0,0)}.**
  Классификация: **full-attn l_S (6, вкл. last 23 → R=0 o_proj bridge) ; l_S=0 (GDN) → KL-only.**
  Stage0 переделывается. l_L НЕ снейпится на full-attn 4B (frozen target, no LoRA on l_L).

## OPEN (default'ы — confirm/adjust, не block)

- [ ] **full_S из config**: 3mod4 {3,7,11,15,19,23} = 7 пар выше подтвердится в stage0
  (печатает layer_types); если нет — пары = runtime filter (без хардкода), классификация по config.
- [ ] **A1d** chunk C=8 (A1e save_on_cpu — on/off по B2 peak VRAM).
