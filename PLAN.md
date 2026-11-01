# PLAN: E-39b — PTQ perturbation gate (Kaggle)

## Запрос (фиксация)
docs/E39.md §2 (строки 92–140): «κ(L23) и M_K-структура под GPTQ-4bit — что ломает квантование».
PTQ квантует lm_head.weight → M_K → J_t; û_t заморожен (активации из 27b-jcache L23).
Уровни: fp16-базовый / 8-bit RTN / 4-bit GPTQ (group 128, desc_act, damp 0.01).
Гейт на κ(L23): <0.02 pass / 0.02–0.10 moderate / >0.10 catastrophic (base = 0.0162).
Скрипт-источник: `experiments/exp39b_ptq_perturbation.py`; kernel: `kaggle_kernel/kernel-exp39b/`;
результат: `results/exp39b_ptq_perturbation.json` (скачать с /kaggle/working после ранa).

## DONE
- [x] Рекон: ВСЕ контракты зафиксированы в FINDINGS ниже (2026-10-30). НЕ перечитывать — смотреть в FINDINGS.
- [x] PLAN.md как findings-ledger
- [x] `experiments/exp39b_ptq_perturbation.py` написан полностью (ruff чистый), локальный smoke пройден:
  self-check rebuild J @L23 vs 27b-cache: rel err max 7.8e-7 (tol 1e-5); κ(cached J) = 0.016233
  = exp38d L23 r16 (0.016233441) — pipeline верен; RTN/GPTQ finiteness + детерминированность GPTQ
  (max diff 0.0) проверены. Поправлены: форма c.unsqueeze(1)→[n,1,1,1] в _recompute_J;
  V_all [n,r,D] = Vh[:,:r,:].transpose(-1,-2) в _kappa_from_J (Vh — строки); main() → small_model.
- [x] push_all.py: entry exp39b (dir kernel-exp39b, title="Prism: exp39b-ptq-perturbation" — title
  должен резолвиться в id, иначе 400; sources 27b-dense-profile, datasets jca+jcb, extra_local _e38_cache).
- [x] Kernel собран и pushed (v1): https://www.kaggle.com/code/voroninmichael/prism-exp39b-ptq-perturbation
  (self-source warning при первом push — expected, kernel ещё не существовал; datasets смонтированы).

## DOING
- [ ] Мониторинг Kaggle-рана (T4, ~20–40 мин): status → log; затем скачать
  results/exp39b_ptq_perturbation.json с /kaggle/working.

## NEXT
- [ ] GATE по κ(L23) r=16: <0.02 pass / 0.02–0.10 moderate / >0.10 catastrophic (base 0.016233).
- [ ] E39.md: вписать результаты (новый §2.x / §15 строка) — только после получения JSON.
- [ ] Коммит: exp39b-скрипт + kernel-exp39b/ + push_all entry + E39.md (paper.md НЕ трогать).

## Границы (не менять)
- paper.md — E-39b результаты tier-pending, в paper не попадают (правило: ни одна строка paper не зависит от E-39)
- docs/E39.md §2 (спека), 27b-jcache (read-only, 6×2.1 GB)
- квантуются ТОЛЬКО веса (не активации); GPTQ параметры фиксированы спекой
- lm_head.weight НЕ заменять in-place для PPL (tied weights: lm_head = embed)

---

## FINDINGS (ledger — всё подтверждено, не перечитывать)

### 1. Формулы (E-38 серия, переиспользовать)
- `M_K = W_lm[U128] · diag(g)`, g = weight финального RMSNorm + 1 (`_lm_module(model).norm.weight + 1`).
- `J_t = c_t · M_K · (I − û_t û_tᵀ)`, c_t = √D/‖h_t‖, û_t = h_t/‖h_t‖; h_t — из jcache поля H [1024].
- Локально верифицировано (2026-10-30): rebuild J на L23 из jcache совпадает с кэшем JR:
  rel err median 1.62e-7 / p95 2.83e-7 / max 7.18e-7 (fp32-модель, fp64 арифметика, ~89 c).
  → self-check «rebuild base J vs cached J at L23» ОБЯЗАТЕЛЕН в ядре.
- κ(L) = ‖H_012 − I‖_F; H_012 = R_01·R_12·R_20, R — Procrustes-вращения по top-r SVD-базам (правые сингулярные
  векторы J_t, r = 16 headline); тройки (t,t+1,t+2), S−2 = 126 на док.
- Procrustes (из exp38d `_procrustes_batched`, строка 93): V_a,V_b [B,D,r]; M = V_bᵀV_a; SVD M=PΣQt;
  R = QtᵀPᵀ; residual_sq = 2r − 2Σs (clamp 0). fp64.
- β/σ₁: w = M_Kᵀ(1/√K) [D]... (K-вектор), α_t = û_tᵀw, β_t = ‖w − α_t û_t‖;
  baseline median β/σ₁(M_K) ≈ 0.9 ± 0.15; β/σ₁ = cos(u_g, 1/√K) (верхняя грань 1).
- u_g = J Jᵀ 1 / ‖J Jᵀ 1‖ — K-вектор на позицию.
- J_rel_drift = ‖J_q − J‖_F / ‖J‖_F per (doc,t); отчёт median/p95/max; базовый уровень → 0 по определению
  (J_base считается той же формулой из кэша H/U128).

### 2. jcache (experiments/_e38_cache.py)
- Файлы: `kaggle_kernel/results/exp27b_jcache_L{l}.pt`, l ∈ {6,12,18,19,22,23} (все есть локально, ~2.1 GB).
- Ключи (dom, si, t): JR [128,1024] raw J; P128 [128] (sum=1); P128_RAW; H [1024]; RANKS; U128 list[128] (общий).
- S=D=K=128/1024, 32 дока = 4 домена (wiki/math/code/chat) × 8.
- `load_layer(l, want=["J","P","H"])` → J [N,S,K,D] fp32 (стек JR), P (норм-гард sum=1), H [N,S,D]; U128
  ВСЕГДА возвращается (не want-поле). want принимает только 'J','P','H'.
- Kaggle: `jcache_path` резолвит через dataset-маунты `/kaggle/input/datasets/voroninmichael/prism-exp27b-jcache-{jca,jcb}/`
  (jca=L6/12/18, jcb=L19/22/23) — работает только если build_kernel встраивает `_e38_cache` (--extra-local).
- GDN-слои среди LAYERS: 6/12/18/22 (linear-attention), 19/23 full attention.

### 3. κ base (референс гейта)
- results/exp38d_holonomy_spectrum.json → metrics['23']['16']: kappa_median = 0.016233441,
  CI [0.0049, 0.0377], degenerate_frac 0, trivial_frac 0. (E-28 baseline 0.018, tol 5e-3.)
- κ сильно зависит от r (r=64 → ~3.69) → гейт СТРОГО при r=16.

### 4. GPTQ (каноника IST-DASLab/gptq, полные исходники получены — authority)
- `quantize(x, scale, zero, maxq)`: q = clamp(round(x/scale)+zero, 0, maxq); dequant = scale·(q−zero).
- find_params (perchannel, weight, sym): scale = (xmax−xmin)/maxq, zero = (maxq+1)/2;
  симметрия: xmax = max(|xmin|, xmax), xmin = −xmax где xmin<0; (0,0)→(−1,+1).
- fasterquant (blocksize=128, percdamp=0.01, groupsize=128, actorder=True, static_groups=False):
  1) H = (2/n)·Σ x xᵀ (n = число векторов, x — [n, D_in], т.е. Hessian [D_in, D_in] по входам lm_head);
  2) dead = diag(H)==0 → H[dead,dead]=1, W[:,dead]=0;
  3) perm = argsort(diag(H), desc); W,H переставить; invperm = argsort(perm);
  4) damp = 0.01·mean(diag(H)); H += damp·I; Hinv = chol(chol_inverse(chol(H))) — верхний треугольник,
     т.е. (Lᵀ)⁻¹ при H = LLᵀ;
  5) блоки i1..i2 по 128: W1 clone, Hinv1 = Hinv[i1:i2, i1:i2]; по колонке i:
     d = Hinv1[i,i]; если (i1+i)%128==0 → find_params(W[:, i1+i : i1+i+128]) (НА ТЕКУЩИХ
     error-compensated значениях); q = quantize(w); Q1[:,i]=q; err1 = (w−q)/d;
     W1[:, i:] −= err1 ⊗ Hinv1[i, i:]; Err1[:,i] = err1;
     ПОСЛЕ БЛОКА: W[:, i2:] −= Err1 @ Hinv[i1:i2, i2:] (cross-block компенсация!);
  6) Q[:, invperm] — unpermute; вернуть DEQUANTIZED Q (fp32).
- RNG отсутствует → детерминировано.
- 8-bit: plain RTN, per-group (128) min-max affine, round-to-nearest, без Hessian/компенсации.
- fp16 baseline: загруженный fp16-вектор → fp64, без квантизации.
- GPTQ по ЦЕЛОМУ W_lm [151936, 1024] (PPL нужен весь head; J_t — только строки token-set).
- .venv БЕЗ библиотек квантизации (gptqmodel/optimum/auto_gptq/quanto/bb) → порт вручную.

### 5. Hessian-калибровка и PPL (C4)
- Hessian: forward-hook на ВХОД model.lm_head, свежий C4: train skip(8192).take(128) доков × 128 токенов
  = 16384 вектора, один forward, fp16 X → fp32, H = (2/n) XXᵀ (fp64).
  (Свежий сплит ≠ 27b-cache (первые доки доменов) и ≠ exp39c (train skip 4096) — требование спеки
  «другой C4-сплит, чем E-16».)
- PPL holdout: C4 val take(32) доков (фолбэк train skip 32768 если val недоступен).
- PPL с tied weights: logits = (model.model.norm(last_hidden_state)).float() @ W_q.T в fp32,
  остальная модель fp16. lm_head.weight НИКОГДА не менять in-place.
- Ядра: модель fp16 (спека), J/κ — fp64, GPTQ-Hessian — fp32/fp64.

### 6. Runner/build (kaggle_kernel/build_kernel.py)
- Runner cell: `inspect.signature(main_<exp>)`; передаёт small_model=SMALL, device=device_S
  (одна T4 → device_S=cuda:0 fp32), results_dir=RES=/kaggle/working/results.
  → main(results_dir=None, device="auto") — паттерн exp38d.
- build_kernel встраивает: prism.{jacobian,bridge,esg,calibration,cache_io,layer_selection} +
  --extra-modules + --extra-local (имена базовых из experiments/, e.g. _e38_cache);
  вырезает `prism.*` импорты из ячейки эксперимента; `assert_no_bare_prism` УБИВАЕТ build,
  если в эксперименте остался bare `import prism`/`from prism` → Скрипт не может импортировать prism
  напрямую; нужные prism-функции — только через extra_modules.
- Метадата (паттерн kernel-exp38d): id voroninmichael/prism-exp39b-..., kernel_sources: self +
  voroninmichael/prism-exp27b-dense-profile; dataset_sources: prism-exp27b-jcache-jca + prism-exp27b-jcache-jcb.
- resolve_model: /kaggle/input slugs (listdir!) → /kaggle/working → HF download (Qwen/Qwen3.5-0.8B).
- pip-ячейка: torch 2.5.1+cu121, transformers 5.15.1 (pinned!), flash-linear-attention 0.4.2
  --no-build-isolation; causal_conv1d УБРАН (ломает vmap).
- T4 16 GB, 30 h/нед кота. OOM-план: doc-чанки по 8; SVD [N*S,128,1024] в fp32 (~2.15 GB);
  Procrustes-блок fp64 мелкий; пик ≤ ~8–9 GB (см. комменты exp38d: chunk-формула per_row).
- push: `kaggle kernels push -p kaggle_kernel/kernel-expNN/` (CLI 2.1.2, ~/.kaggle/access_token есть).
- push_all.py: KERNELS registry {dir,id,title,sources,datasets,extra_modules,extra_local} → build() → push().
  Slug выводится Kaggle из title — после первого push проверить `kaggle kernels list`.

### 7. Решения дизайна (зафиксированы)
- Считать κ для ВСЕХ 6 слоёв jcache (6,12,18,19,22,23) — супер-множество спеки {6,12,18,22,23};
  гейт авторитетный только на L23 (там readout-linearized J точен; для L<23 — контролируемая
  конвенция «M_K-only эффект», base и quantized строятся идентично).
- Метрики per level per layer: κ(r16) median+CI, β/σ₁ median, cos(u_g,1/√K) median,
  J_rel_drift (median/p95/max), PPL (только L23-relevant, модельный).
- В JSON: gate verdict (pass/moderate/catastrophic) + base κ + self-check rel err + все уровни.
