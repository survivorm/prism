# PLAN — E-39f: полный tied-weight GPTQ-4bit и сдвиг û_t (ТЗ docs/E39f.md)

## Запрос (фиксация)

«делай пока ./docs/E39f.md» — реализовать и прогнать E-39f по ТЗ `docs/E39f.md`:
полное tied-weight квантование одной матрицы `lm_head.weight == embed_tokens.weight`
(Qwen3.5-0.8B, tied, vocab 248320, D=1024) и декомпозиция изменения κ(L23, r=16) на
readout-канал (`M_K`) и upstream-канал (сдвиг `û_t` через квантованные эмбеддинги).
H-39f: `share_upstream ≥ 0.5`. Нуль: `< 0.25`. Плюс harsh-режим (group=32), PPL full
tied, null-модель (Haar `û`).

Параллельно в полёте (не трогать, не коммитить чужое):
* E-39b v13 на Kaggle (`e-39b-ptq-perturbation-gate-gptq-4-bit-lm-head`) — проверка
  детерминизма JSON-уточнений;
* E-39j локально (`experiments/exp39j_sentinel_confidence.py`, лог
  `results/exp39j_run.log`) — ждём `results/exp39j_sentinel.json`.

## DONE

- [x] Разобрано ТЗ `docs/E39f.md`; сверено с фактами репозитория:
      `tie_word_embeddings: true`, vocab 248320, hidden 1024 (config из `.hf_cache`);
      GPU локально нет (`nvidia-smi` отсутствует) → прогон либо CPU fp32, либо Kaggle T4.
- [x] Вычитан `experiments/exp39b_ptq_perturbation.py` (772 строки) как база: хелперы
      `build_M_K`, `_recompute_J`, `_kappa_from_J`, `_quantize_gptq`, `_c4_docs`,
      residency-правило (копии `W_lm` на CPU, на GPU только `M_K [128,1024]`).
- [x] Проверен способ переиспользования: `build_kernel.py` инлайнит `--extra-local`
      **без** зачистки их собственных local-импортов → импорт хелперов из
      `exp39b_ptq_perturbation` на Kaggle упадёт ⇒ в E-39f хелперы копируются.
- [x] Выяснен источник `H` в кэше: `exp27b_dense._forward_dense` берёт
      `out.hidden_states[l+1]` ⇒ H(L23) = output слоя 23, ДО финальной RMSNorm.
      Значит `û` для E-39f обязано сниматься хуком на последнем decoder-слое,
      а не `model.model(...).last_hidden_state` (это уже post-norm).

## DOING

- [x] `experiments/exp39f_full_tied.py` + `experiments/test_exp39f_math.py` (проверка
      математики против эталонов E-39b: J — 0.0, κ — побитово, drift 6e-08, β/σ₁,
      `_M_K_rows` ≡ `build_M_K`). Найдено и исправлено тестом: `c[c0:c1].view(-1,1,1,1)`
      вместо `unsqueeze(2).unsqueeze(3)` (масштаб c по позиции, а не по документу).
- [x] Kaggle-ядро собрано и запушено: `kaggle_kernel/kernel-exp39f/` (id
      `e-39f-full-tied-ptq-kappa-decomposition`, jcb-кэш L23, `--extra-local _e38_cache`).

## NEXT

- [x] Забрать E-39b v13 (`kaggle kernels output`) → diff с v12-архивом: **побитово
      идентичны** (κ 0.016233/0.016245/0.016043, β/σ₁, cos_ug_1k_proxy, drift, PPL
      30.4753/30.7494/30.7498, gate=pass) + присутствуют все поля round-3
      (`p04_ref_err`, `p04_passed`, `warning_kappa_synthetic`, `kappa_semantics_note`,
      `levels_note`, `gate.readout_channel_only`, `cos_ug_1k_proxy`) ⇒ раунд 3 закрыт.
- [ ] E-39f на Kaggle: `voroninmichael/e-39f-full-tied-ptq-kappa-decomposition` v1 —
      дождаться COMPLETE, проверить лог на traceback'и и на маркеры версии
      (`κ_control`, `null/`, `full_tied`), забрать `results/exp39f_full_tied.json`.
- [ ] Проверить воспроизведение E-39b внутри E-39f: `κ_fp16 ≈ 0.01623`,
      `κ_readout ≈ 0.01604`, `drift_readout ≈ 0.1205` (assert'ы в скрипте) и
      `κ_control − κ_fp16` (шумовое поле) против `Δ_upstream`.
- [ ] Док-раунд E-39f: `docs/E39.md §6` (ссылка на новую редакцию ТЗ `docs/E39f.md`),
      карточка `E-39f` в `docs/ref-exp.md`, строки в `docs/paper.md` §4.8/§5/§6.
- [ ] Забрать `results/exp39j_sentinel.json` → вердикт → доки E-39j.

## Границы (не менять)

- Числа E-39b (κ 0.016233/0.016043, drift 0.12047, PPL 30.4753/30.7498) — референсы для
  assert'ов, не пересчитывать «улучшая»; GPTQ-детерминирован.
- Формула `J_t = c_t·M_K·(I − û_t û_tᵀ)`, `M_K = W_lm[U128]·diag(g)`, `g = 1 + w_norm`
  (RMSNorm не квантуется) — точность на L23 доказана 7.8e-07.
- C4-сплиты E-39b: калибровка train skip 8192 (128×128), PPL-holdout val→fallback train
  skip 24576 (32×128); пул документов — только код `_load_pool` (exp27b), сплиты E-27b
  не трогать.
- Только L23 (L6–L22 в этом протоколе синтетика, ТЗ §11).
- Файлы параллельной сессии не коммитить: `docs/E39d_fixes.md`,
  `experiments/exp39d_l23_training.py`, `results/exp39d_stage1_ordering.log`,
  а также E-39l-артефакты (`experiments/exp39l_drift_behavior.py`, `docs/E39l_fixes.md`).
