# GPU-аудит: какие эксперименты требуют re-run (запрошено пользователем, 2026-08-31)

## ACTIVE (2026-09-14) — live state

- **ACTIVE (2026-09-15):** exp01 Coherence (GPU kernel v1 RUNNING — см. блок Exp 0.1 ниже; RED done, GREEN on Kaggle).
- **DONE:** exp16 v4 (GPU, 1000 steps, bridge 0.603→0.921) + LORA behavioral check (top-p retention: anti-correlation confirmed, point-Jacobian insufficient → Goodhart). Paper section added. box v5 (TPU) COMPLETE. exp19 v6 (TPU v7) — status: ERROR (2026-09-15, проверить log — возможно partial + fix).
- Details in section «exp19 v6 dose-sweep + exp16 LORA behavioral» below.

## Exp 0.1: Coherence Length (docs/Exp0(1-3).md, user 2026-09-14 «начинай планировать 0.1»)

**Цель:** измерить когерентность поля Якобианов J(x) по x — гладкое ли поле, чтобы доменные грамианы G^d = (1/N)ΣJ·Jᵀ несли смысл. Kill-switch для ESG/Exp 0.2. Если поле фрактальное — стоп, ESG не интерпретируем.

**DONE / DOING / NEXT:**
- [x] Чтение замысла (docs/Exp0(1-3).md) + аудита существующих механизмов
- [x] План ниже (этот блок)
- **v4 RESULTS (09-16, kernel v4 COMPLETE, commit ebc6bc2):** split-гейты (per-position gauge, 1 forward/сэмпл): L6 C_b=0.033 C_a=0.077, L12 C_b=0.070 C_a=0.099, L18 C_b=0.109 C_a=0.059, L22 C_b=0.120 C_a=0.224. Вердикты: L6/L12 FRACTAL-RISK (оба <0.1), L18/L22 WEAK. **Важно:** C_a упал с 0.33–0.92 (v3) до 0.06–0.22 — v3-градиент «глубже = плавнее» был артефактом final-position gauge + pooling; при per-position gauge внутри-сэмпловая когерентность на σ_d не выше межсэмпловой. Поле слабокогерентное на доменном масштабе во всех слоях. ESG half-split вынесен в отдельный скрипт `experiments/exp01_esg_halfsplit.py` (24 слоя, K=256, wta 1.0→0.89), результат `results/exp01_esg_halfsplit.json`.
- **v4 RETHINK (коллегa, 09-16):** пул-гейт (a)+(b) не имеет физического смысла ((a)≈0.3–0.9, (b)≈0.005–0.05 при сопоставимых δ → 10–100×). Правильный объект — **расслоение (bundle):** гладкий лист внутри сэмпла, листы почти ортогональны. Фиксы: (1) раздельные кривые C_a(σ_d) и C_b(σ_d), канонический гейт = C_b (межсэмпловая); (2) pk (gauge) на той же позиции, что и J (logits[0, S-pos]); (3) один forward на сэмпл (кэш logits+hs, J pass — только backward); (4) **half-split reliability ESG** (4+4 сэмпла, главные углы top-16 подпространств G_A/G_B) — тест осмысленности доменных G^d; (5) вердикты: SMOOTH / WEAK / FIBERED / FRACTAL-STOP.
- **v5 (09-16, коллега-2, commit ac94053):** три gauge-конвенции (pos-pk/same-pk/raw) из кэша J_raw, per-pair cos. v5 kernel: device-mismatch (Jr.cpu() раньше softmax_jacobian) → fix, v6 push.
- **v6 (09-16, ТЗ colleague review 3, diagnostics):** gauge_share=||P(p)J||^2/||J||^2 per (sample,layer,pos); cos(p_t,p_{t+1}) per pair; L_mean/L_p90 per convention (v5 shared pos-pk L — баг); verdict vs 1/sqrt(D) noise floor (ORTHOGONAL/WEAK/COHERENT); L18 flag C_b>1.5*C_a; pk_hash+pk_top_token_ids (полные PK — в кэше, в JSON — hash+top8); JSON exp01_coherence_v6.json. Selfcheck OK. **Pre-registered:** P1 gauge_share>0.7 (raw smooth = gauge-мода); P2 cos_p падает с глубиной (L18 anomaly); P3 L_samepk/L_pospk>1.5 на L18-L22. Kill: P1 ложна → raw smooth функционально; P2 ложна (cos_p~1) → баг проектора, реран v5.
- [ ] **GREEN v6:** CPU smoke → push kernel v6 (заменит запущенный device-fix-бэилд) → анализ P1/P2/P3.
- [x] **RED:** `experiments/exp01_coherence.py` — каркас: загрузка 0.8B + 4B, домены, J-каш, скрытые состояния; тест на маленьком срезе (1 домен, 1 слой, пара пар) — C(δ) считает без ошибок. (DONE 09-15: selfcheck PASSED (синтетическое гладкое поле: C монотонно, ell найден) + CPU smoke (wiki, L6, n=2, K=16) end-to-end OK. Отклонение от плана: 4B не грузится — self-bridge 0.8B, token set global top-K 0.8B (зафиксировано в плане).)
- [x] **GREEN:** полный прогон: 4 домена × слои {6,12,18,22} × типы пар (a) соседние позиции (a), (b) внутри-домен, (c) междомен. Выход: `results/exp01_coherence.json` + PNG. **v1 COMPLETE 09-15 (~10 min, budget 3.83/15.00 GPU-h): C~0.005-0.051 на delta 2.5-11.6, gate n/a (кривая не покрывала sigma_d; тип (a) = trunc-8 = сдвиг 8 токенов — не «малый», C~0).** **v3 PUSHED 09-15 (RUNNING)** (v2 push потерялся — shell &: (a)=соседние позиции (S-1,S)&(S-2,S-1) — точно по плану, 384 J-каши; гейт C(σ_d) на pooled (a)+(b) кривой (σ_d=1.7-7.2 < min delta (b)=2.5 — без (a) n/a); σ_d=norm(std по размерам) по спеке; фикс бага hs-порядка. Монитор: `_auto_chain/poll_exp01.sh`. **v3 COMPLETE 09-16 (644s, budget 0.18 GPU-h):** gate C(σ_d) на pooled (a)+(b), clamped: L6=0.33 MODERATE, L12=0.45 MODERATE, L18=0.64 SMOOTH, L22=0.87 SMOOTH. **НЕ FRACTAL** — STOP (<0.1) не сработал ни на одном слое; поле J глубже — плавнее. Ключевой сигнал: (b) between-samples C≈0.005–0.05 (J разных семплов домена почти ортогональны), (a) intra-sample C≈0.3–0.9 → для gramians нужен sample-local подход (как exp16/19); пул семплов сигнал убивает. None-баг (σ_d за краем binning-диапазона) исправлен: clamp + 2% padding. v1-впечатление «фракталь» — артефакт: кривая (b) начиналась при delta≫σ_d.
- [x] **Анализ/вердикт:** gate C(σ_d) по слоям вычислен (таблица в GREEN выше); ℓ(δ при C=0.5) формально n/a для (b)-кривых (C не пересекает 0.5 в их диапазоне) — по сути ℓ < min delta (b); интерпретация: гладкое/умеренное, стоп не нужен → можно Exp 0.2.
- [x] Paper-секция в docs/paper.md (§12, вшита в §6/12a/итоговая картина/future work, commit 4444d8b)

**Дизайн-решения (зафиксированы):**
- Метрика: C(δ) = E[⟨J(x),J(x')⟩_F / (‖J(x)‖‖J(x')‖)] по парам с ‖h(x)−h(x')‖≈δ; ℓ = δ при C=0.5. L(x,x') = ‖J−J'‖_F/‖h−h'‖_2 — попутно, в JSON.
- J: gauge-free `softmax_jacobian(J, p)` (prism/jacobian.py:270, та же конвенция exp11a/16). K=64. Токен-сет: top-K финальных logits 0.8B на каждом сэмпле (self-bridge, без транспорта 4B). План пишет «P(p_L) из 4B» — упрощение: общий токен-сет 0.8B, 4B не грузится → дешевле и self-consistent. Отклонение от плана: 4B-проекция отброшена (YAGNI; gauge-free уже убивает common-mode).
- h: скрытые состояния слоя l из output_hidden_states (один forward на сэмпл, все 4 слоя), последняя позиция.
- Пары: (a) позиции t,t+1 внутри сэмпла (δ ~ маленький); (b) пары сэмплов одного домена; (c) пары разных доменов. δ-биннинг: лог-масштаб, ≥5 пар на бин, иначе бин выкинуть.
- Данные: `load_calibration_data` (prism/calibration.py:69), домены wiki/math/code/chat, n_samples=16/домен (32 пары = тип (b); (c) перекрёсток).
- Slo-бюджет: J-каш по (сэмпл, слой) = 4 слоя × 64 сэмпла = 256 вызовов compute_jacobian_for_tokens (K=64 → 64 backward). GPU ~30-60 мин. TPU: пер-токен path доказан (exp19), но 4 distinct слоя = 4 рекомпиляции — GPU предпочтительнее после квоты; TPU фолбэк.
- Скрипт: `experiments/exp01_coherence.py`, main(small_model, device, results_dir, layers, K, n_samples, domains). .pt-сейвов нет (метрический эксперимент). Self-check assert на синтетическом гладком поле (C(δ) монотонно падает) — один runnable check.

**Границы (не менять):**
- Не трогать prism/jacobian.py, prism/bridge.py, calibration.py — только новый файл experiments/exp01_coherence.py + результаты.
- Не пушить kernel до CPU self-check (правило 2026-09-08: CPU repro с assert на shape/значениях до железа).
- Не смешивать с exp19 v6 (TPU слот занят; GPU квота ~09-14).

**Вердикт-гейты (из плана):**
- C(σ_d) > 0.5 → гладкое, ESG осмыслен → Exp 0.2
- C(σ_d) ≈ 0.3 → умеренное, shrinkage ε≥0.1
- C(σ_d) < 0.1 → фрактальное, **стоп** ESG

## Exp 24 — Сканирование расслоения (docs/Exp24-16.md, user 09-16 «начинай планировать первый эксперимент»)

**Порядок (ТЗ §3):** пилотный кэш (64 узла, 0.8B, слои 6/22) → Exp 24.0 (мощность, CPU) → полный кэш (2048 J-вызовов, 2 модели) → Exp 24 (гейты). Первые два шага — ниже.

### Шаг 1: Пилотный кэш (GPU, Kaggle T4)
- [ ] **`experiments/exp24_pilot_cache.py`** — 32 сэмпла (8 × {wiki,math,code,chat}), S=64 фикс (max_length=min_length=64), узлы (x, last) и (x, mid=32): J по префиксу x[:q] (каузальность), слои {6,22}, 0.8B. Итого 32×2×2 = 128 J-вызовов.
- [ ] Токен-сет: глобальный топ-64 по частоте в топ-K финальных logits 0.8B на калибровочном пуле (один общий для обеих моделей — ТЗ §1.1/D3); `token_set_id` = sha1 отсортированных id.
- [ ] J gauge-free self-p: `P(p) = diag(p) - pp^T`, p = softmax logits той же позиции, перенормированный на U; применить ровно 1× (D2).
- [ ] Кэш-лейаут (ТЗ §1.10): `results/jcache/v1/{model_id}/{git_commit}/{token_set_id}/layer{L}_pos{last|mid}/J.pt` + `manifest.json` (модель, коммит, хеш токен-сета, K, S, конвенция, зерно, тип данных); лоадер ассертит манифест.
- [ ] Ядра K^G = cos(gramian) по ТЗ §1.3 (K^F, K^V — отчётные) → сохранить в кэше; PNG heatmap опционально.
- [ ] Kernel: реестр push_all.py (dir kernel-exp24pilot, id voroninmichael/prism-exp24-pilot-cache, title "Prism: exp24-pilot-cache"), ~30 мин на T4, monitor в /data/prism_kernels/exp24pilot.

### Шаг 2: Exp 24.0 — мощность пайплайна (CPU, без GPU)
- [ ] **`experiments/exp24_power.py`**: planted-сигнал `K = min(K_perm + s·1[ell_i=ell_j], 1)` на перестановочном нуле реального K^G пилота; сетка k_true {2,4,8} × s {0,0.02,0.05,0.1,0.2} × 50 семян, N=256 узлов.
- [ ] **Интерпретация-решение (разрыв ТЗ):** пилот = 64 узла, сетка 24.0 = N=256. K_perm 256×256 строится ресэмплингом внедиагональных значений реального K^G пилота (эмпирическое распределение, симметризация). Зафиксировано в отчёте.
- [ ] Пайплайн точно по ТЗ §1.4–1.6: kNN k=15, w=max(K,0), Leiden resolution=1.0 seed=42 (python-igraph, УСТАНОВЛЕН), вырожденность k=1 или k>64 → гейт n/a; гейт: ARI_half − ARI_perm_mean ≥ 0.15 ∧ CI95_boot(low) > ARI_perm_mean + 0.05; 50 реплик перестановочного нуля, 100 бутстрепов по узлам-парами.
- [ ] Выход: `results/exp24_power.json` — power(s,k), s_min(k), FPR(s=0), интерпретация по таблице ТЗ (s_min vs наблюдаемый внедиагональный уровень; FPR ≤ 0.05; порог 0.15 калибруем).
- [ ] RED: selfcheck на синтетическом нуле (iid N(0,σ²)) — при s=0 гейт ~0, при s=0.2 power≈1, Leiden восстанавливает planted k=4.
- [ ] Запуск: CPU, ~10–30 мин (250 матриц × (50+100) Leiden-прогонов на 256 узлах).

**Границы:** пороги предрегистрированы ТЗ (D4) — не менять без аддендума; J-вычисления только GPU float32 (D6); кластеризация/статистика CPU по кэшу; не смешивать конвенции проектора в одном кэше.

## DONE (historical audit)

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

- [x] **Scoping fact (verified 14-01):** `docs/paper.md` (v3.3) contains **exp 1–12c (§5–9) + 13/14/15 (§11 «Доменная специфичность, глубинная структура, фазовая робастность»)**. exp16/19 now in paper (§11.12/13); exp17/18 still NOT in paper (audit + JSON only). FIRST PASS wrongly claimed 1–12c only — CORRECTED: §11 is in-paper, so 13/14/15 = fix numbers (done, c07c138), 16–19 = write sections (future).
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
| 2 | exp16 v2 (readout) | exp16_readout_pull.py | **COMPLETE 2026-09-10** (GPU T4, 126s; phantom was my shape bug, fixed in jacobian.py 605b3b3) |
| 3 | exp19 (destruction) | exp19_destruction.py | **COMPLETE** (TPU v5e-8: v3 6-scenario 44 min + v4 7-pair; gauge=0 at L23, dose-linear) |
| 4 | exp6_l18_snr (HEADLINE) | **exp6_l18_snr.py (NEW)** | **built** — data LOST, mandatory re-run; script written this session |
| 5 | exp4_k_sensitivity (FIXED) | exp4_k_sensitivity.py | **built** — was broken self-bridge; now cross-bridge 4B→0.8B + K→256 |
| 6 | exp2_heatmap | exp2_heatmap.py | **built** (n=8, K=64) |
| 7 | exp11c_defrag_bridge | exp11c_defrag_bridge.py | **built** (n=8) |
| 8 | exp10b_gauge_bridge | exp10b_gauge_bridge.py | **built** (n=8) |
| 9 | exp4_top_p_tokens | exp4_top_p_tokens.py | **built** (n=4) — optional |

**Still NOT in pipeline / paper:** exp11b re-run, exp12a N=4/K=32, exp12b/12c K=64, exp17/18 paper *sections* (write — exp16/19 done: §11.12/13). exp6_l18_snr is now in the pipeline (was the gap).
**Paper-only fixes (no GPU)**: exclude 1.0 sentinels (exp13/15/6); exp11a/11b K-mismatch note + 11c bridge-flattening (defrag = output-faithful but bridge-destroying); exp12 re-frame; exp14 add params; exp2a note (CKA null inflated when N≪D); exp17 topk_report part of v2 suite.

## TPU verdict (2026-09-07/08) — **CORRECTED 2026-09-09, exp16 now COMPLETE on GPU**
TPU v5e-8 (8 chips × 16 GB = 128 GB HBM) verified viable for **1st-order** causal work (iso-canary, exp19). The "XLA phantom dot" was MY `_last_layer_jacobian` shape bug (now fixed, commit `605b3b3`) — not an XLA defect.
- **exp16 ✅ COMPLETE** (GPU T4, 126 s, commit `d8a9289`): lm_head edit 0.60→0.754, KL 0→0.24, α*=0.3207@step100. Paper §11.12.
- **exp17 ❌ TPU-blocked** — only 2nd-order HVP experiment; `create_graph=True` → XLA HLO-compile hang (canary 120+ min, CANCELED → 2nd-order = TPU dead-end; exp17 GPU-only).
- **exp18 ❌ 2h limit** — 7 distinct target layers = 7 XLA recompiles ≈ 4 h > Kaggle free-tier 2 h kernel cap.
- **exp19 ✅ COMPLETE** (TPU v5e-8, v3 6-scenario 44 min + v4 7-pair dose, both in paper §11.13).
- **All 4 causal exps now have data** (exp16/18/19 done; exp17 = the one remaining, GPU-only, 30h/wk reset ~2026-09-14). Box's ideal TPU task = exp18 7-pair (recompile-bound, fits the 9h sandbox, not a 2h kernel).
- Lesson: CPU-repro the exact op chain before a hardware verdict — and **assert output tensor shape** (the [65536,65536] OOM'd on T4 but allocates silently on 64 GB RAM, so exception-only repros miss it).

### ⚠️ MISDIAGNOSIS CORRECTION (2026-09-10) — the "XLA phantom" was MY bug

The `f32[65536,65536]` was **not** an XLA shape-inference bug. It was a **prism shape bug in `_last_layer_jacobian`**: exp16 passed `h=hidden_states[last+1]` which is `[1,S,D]` = `[1,64,1024]`; the function did `h.reshape(1,-1)` → `[1,65536]` → `torch.eye(65536)` = **16 GB** `[65536,65536]` matrix. My CPU repro only checked "does it run" (64 GB RAM hides the 16 GB alloc), not "what shape does it produce" — so I misattributed the OOM/compile-abort to XLA. **GPU OOM was the tell**: 16 GB allocation on a 14.5 GB T4 cannot be a phantom; it's a real tensor. **Fix (commit 605b3b3):** `_last_layer_jacobian` now collapses `[1,S,D]`→`h[:,-1,:]` (the correct final-position readout) before reshape. **Result:** exp16 runs on GPU in 126s (bridge 0.60→0.754, KL 0→0.24). exp17/18: the same fix applies to their last-layer calls; the remaining TPU blockers are **genuine** (2nd-order HVP hangs XLA [exp17]; per-layer recompile exceeds 2h limit [exp18]). **Lesson (re-re-confirmed, the exact class I warned about):** a CPU repro that only checks "runs" not "right shape/value" masks OOM-class bugs and leads to hardware blame. Before any TPU/XLA verdict: verify the **actual tensor shapes** in a CPU run with the same input dimensions, not just "no crash".

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
- **DONE**: results pulled, **JSON-save bug fixed** (was `json.dump(..., device=xla)` → TypeError, file truncated; now `device=str(device)`), full 7-pair JSON recovered from the kernel log + completed scenarios from partials, integrated into docs/paper.md §11.13
- **Data-state note:** results/exp19_destruction.json was last overwritten by the v5 single-pair (31_23) run (3 scenarios + doses 1x/10x/100x). The 7-pair sweep data lives in paper §11.13 (recovered from the v4 log); the 7-pair JSON file itself is not reconstructable — acceptable: the paper is the source of truth for that result. (full per-layer bridge table + dose-response: dose-linear 0.17×/0.5×/0.9×, gauge=0.0000/0.0238/0.2382 at L23, gauge null-op, target=differential exactly, cross-layer gauge/diff ratio stable 1.6–1.9).
- **CORRECTION (v5 push 2026-09-11):** the v5 push was misconfigured as `pairs=[(31,23)]` → it re-ran only the **31_23 pair** (7 scenarios incl. doses 1x/10x/100x) in 44 min, NOT the 7-pair sweep. The pull therefore overwrote results/exp19_destruction.json with **31_23 single-pair** data (Δbridge +0.0017/+0.0151/+0.142 for 1x/10x/100x, gauge null). The **7-pair sweep is still pending-optional** (doses 1/10/100 × 7 target pairs) — ideal **box follow-up** (2-3h recompile < 9h cap; box keepalive defeats the idle-kill; GPU fallback after ~09-14 weekly reset). Paper §11.13 keeps the real 7-pair content (from the v4 log) — unchanged.

## kaggle-tpu-lab sandbox (2026-09-11, user reframe «интереса ради») — TPU-blocked exps via long-lived session

**Idea (user):** instead of competing for the TPU slot with 2h-capped kernels, run `ARahim3/kaggle-tpu-lab` — a **long-lived TPU session** (keepalive `keepalive_min:480`, Kaggle TPU hard-caps at 9h, beats the 2h per-kernel wall-clock that blocks exp16/17/18) — and inside the **same container** run our prism causal experiments as **4-6 concurrent threads** (one experiment = N parallel threads, NOT 4-6 agent instances).

**Why it unblocks what 2h-blocks:**
- Kaggle free-tier TPU = **v5e-8 = 8×16 GB = 128 GB HBM total** (NOT 256 — 32 GB is per-chip-pair, the unit is 16 GB/chip). Qwen3.8-27B bf16 = **55 GB** across **TP=8** (hardcoded in KTL `serve_qwen38.py:332`) → **~7 GB/chip**, leaving **~9 GB/chip free**. **Co-residency is a MEASUREMENT, not an assumption:** prism's XLA backward HLO graph (~7 GB observed on the iso-canary single-chunk GDN) may or may not fit in the ~9 GB/chip headroom next to vLLM. The box's **first task is a coexistence probe** (boot vLLM → run one prism eval-mode backward on a separate torch_xla device → measure). If it OOMs, prism runs on CPU in the same session (agent still drives). Chip-partition (vLLM TP=4 on chips 0-3, prism PJRT 4-7) is the upgrade path only if CPU is too slow.
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
In-box agent = mini-swe-agent DefaultAgent (BoxAgent subclass: 85% auto-compact, pyspy, git cmd-channel, 9h keepalive) → local vLLM 27B. **First task = vLLM/torch_xla coexistence probe** (run a 0.8B torch_xla forward, report HBM split vs vLLM); then TPU-blocked **exp18** (7-pair, recompile-bound; exp16/19 already done, exp17 = 2nd-order TPU-dead). Self-check PASS locally (compact 60714→4072 at 85%). Coexistence gate = the probe; if OOM → prism falls back to CPU in-session, TP=4 chip-partition is the upgrade path.

**box v3 (exp17 2nd-order) — COMPLETE 2026-09-13 19:15** (TPU v5e-8, slot now free). box v4 pushed 19:24 with the 401 fix (read api_key + model from KTL's generated vllm.log READY banner; v3's hardcoded `not-needed` was rejected by vLLM's per-boot random key).
**exp18 v11 CANCELLED** (degenerate 0.8B self-targets anyway; re-push only if wanted).

## exp19 v6 dose-sweep + exp16 LORA behavioral (2026-09-13/14, user direction)

**exp16 v4 (GPU T4) — COMPLETE:** 1000 steps, lr=1e-3, eval_every=1, K=256, N=4, pair 31_23. Bridge 0.603→0.921 (‖ΔW‖_F 15.4→128.8, no knee, no collapse). Saves `exp16_lm_head_final.pt` + `exp16_lm_head_knee_info.pt`.

**LORA behavioral check (local, top-p retention) — COMPLETE:**
- Applied exp16 v4 ΔW to 0.8B lm_head, compared base vs +LORA vs 4B teacher on held-out queries.
- **Anti-correlation confirmed**: bridge rises 0.60→0.92 but top-p retention falls 0.837→0.416 (C4) / 0.841→0.016 (clean Qs).
- **Point-Jacobian metric insufficient**: single-point fit → Goodhart, no behavioral improvement.
- Primary metric = top-p retention (p=0.9, robust mass), NOT fragile top-10. Tests: `tests/test_lora_effect.py`, `tests/test_lora_dose.py`.
- .pt: `/data/prism_ckpt/exp16_lm_head_final.pt`, `exp16_lm_head_knee_info.pt`, `exp19_dW_*.pt` (30 files).
- Paper: new section «Эксперимент 16 (v4, повед. проверка)» in docs/paper.md.

**exp19 v6 (TPU v5e-8, v7) — QUEUED:** 6 doses (0.1/1/10/100/500/1000) × 4 directions (target/anti/gauge/random) = 24 + baseline, pair 31_23, 1st-order (ΔW on lm_head only), early-stop at bridge<0.05, ΔW.pt per scenario. Correct TPU script kernel (was v5 in old dir; rebuilt with build_tpu.py, v7 pushed 2026-09-14).

**exp17 (2nd-order HVP) + exp18 (logit-KD):**
- exp18 CANCELLED (degenerate: 0.8B self-targets → KD=0 by construction).
- exp17: box v5 COMPLETE (MSWEA_COST_TRACKING=ignore_errors fixed 401/litellm cost crash); agent ran but 2nd-order HVP XLA hang still unresolved → GPU after quota reset (~09-14) or next box iteration.

**Monitor:**
- exp19: `kaggle kernels status voroninmichael/prism-exp19-destruction-tpu` (v7; banner [OK]/[FAIL] in log = true verdict)
- box: `kaggle kernels status voroninmichael/prism-box-autonomer`
- GPU quota (30h/wk): resets ~2026-09-14; exp17 runs there when available.

**Paper status:** exp16 behavioral check integrated. exp17/exp18 sections still missing (exp18 cancelled; exp17 data not yet produced).
