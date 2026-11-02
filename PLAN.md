# PLAN — E-39f: полный tied-weight GPTQ-4bit и сдвиг û_t (ТЗ docs/E39f.md)

## Запрос (фиксация)

«делай пока ./docs/E39f.md» — реализовать и прогнать E-39f по ТЗ `docs/E39f.md`:
полное tied-weight квантование одной матрицы `lm_head.weight == embed_tokens.weight`
(Qwen3.5-0.8B, tied, vocab 248320, D=1024) и декомпозиция изменения κ(L23, r=16) на
readout-канал (`M_K`) и upstream-канал (сдвиг `û_t` через квантованные эмбеддинги).
H-39f: `share_upstream ≥ 0.5`. Нуль: `< 0.25`. Плюс harsh-режим (group=32), PPL full
tied, null-модель (Haar `û`).

Параллельно в полёте (не трогать, не коммитить чужое):
* E-39d/E-39l — файлы параллельной сессии (список в «Границах»);
* E-39j — **готово** (см. DONE), ждёт только коммита вместе с доками.

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
- [x] Источник `H` в кэше exp27b — см. исправленный пункт в DOING: на L23 это **post-norm**
      (output финальной RMSNorm), а не output слоя 23. Первичная версия этого пункта
      («до финальной RMSNorm») была неверна и привела бы к ложному Δ_upstream ≈ 4.2°.

## DOING

- [x] `experiments/exp39f_full_tied.py` + `experiments/test_exp39f_math.py` (проверка
      математики против эталонов E-39b: J — 0.0, κ — побитово, drift 6e-08, β/σ₁,
      `_M_K_rows` ≡ `build_M_K`). Найдено и исправлено тестом: `c[c0:c1].view(-1,1,1,1)`
      вместо `unsqueeze(2).unsqueeze(3)` (масштаб c по позиции, а не по документу).
- [x] Kaggle-сборка собрана и запушена: `kaggle_kernel/kernel-exp39f/` (id
      `e-39f-full-tied-ptq-kappa-decomposition`, jcb-кэш L23, `--extra-local _e38_cache`).
- [x] Три краха на Kaggle разобраны и починены (v1, v2 → v3 в прогоне):
      1. `'CausalLMOutputWithPast' has no attribute 'last_hidden_state'` — post-norm теперь
         снимается вторым хуком (на финальной RMSNorm), а forward идёт через
         `_lm_module(model)(...)`, то есть lm_head вообще не считается (нет logits-аллокаций);
      2. `torch.stack` по чанкам давал 4-D `[4,8,128,1024]` — заменено на `torch.cat(dim=0)`
         (поймано локальной пробой на 2 документах, до пуша);
      3. конвенция H (см. ниже) вместо догадки выбирается численно.
- [x] **Конвенция H на L23**: у exp27b `hidden_states[l+1]` при l = последний слой — это
      output финальной RMSNorm (HF применяет norm после цикла слоёв). Замер: cos(свежий
      norm-hook, кэш H) = 1.0000000 (нормы до 1.2e-07), cos(output слоя, H) = 0.956,
      т.к. g = 1+w_norm не константа (mean 4.31, std/mean 0.155). Значит E-35-фактор с
      û = H/‖H‖ определён на post-norm-состоянии, и κ_full обязан брать его же, иначе
      Δ_upstream = артефакт конвенции (≈4.2°). Скрипт снимает оба состояния хуками и берёт
      совпавшее с кэшем (`selfchecks.h_convention`), при несовпадении обеих — abort.
- [x] Пул документов сверен с кэшем напрямую: `pool_fp` пула = `pool_fp` внутри
      `exp27b_jcache_L23.pt` = один и тот же для локального и Kaggle-стриминга
      (wiki 0cc16ec344e2 / math e71cc282c85d / code 209e83b1898b / chat 2afac288344a)
      ⇒ `_load_pool` воспроизводим между машинами; Kaggle-ядро переиспользует пул через
      self-source mount (`POOL_SELF_MOUNT`), экономя ~6 минут стриминга на перепрогоне.
- [x] E-39j доведён до конца и закрыт (3121 с, CPU fp32): самопроверка — f_sent,
      пересчитанная независимым свежим forward, совпала с E-32r.3 **побитово**
      (max |Δ| = 0.0; среднее 0.507080078125 = 2077/4096 = якорь E-34; 0 позиций ближе
      1e-4 к порогу; инвариантно к слою). ρ_all(f_sent, log PPL) = **+0.421** (p = 0.023),
      но непрерывные базлайны сильнее (entropy **+0.905**, max-prob −0.895, p10 mass +0.636),
      partial ρ = **−0.106** (p = 0.548), Δρ vs best baseline = **−0.484** CI [−0.778, −0.229]
      ⇒ вердикт «не метрика, а потеря версии базлайна»; с геометрией листа f_sent не связан
      (все |ρ| ≤ 0.33), при этом d3_gf↔PPL ρ = −0.569 (p = 0.002, exploratory).
      Доки: `docs/E39.md` §10.7, карточка + индексная строка `E-39j` в `docs/ref-exp.md`.
- [x] Исправлена логика вердикта в `exp39j_sentinel_confidence.py`: v1 сравнивала |Δρ| с
      нулём симметрично и печатала «промежуточный» там, где базлайн строго лучше. Теперь
      направление учтено (`_verdict`), а `--recheck <json>` пересчитывает строку вердикта из
      сохранённых чисел без перепрогона (числа не менялись, история в `verdict_history`).
- [x] Четвёртый крах на Kaggle (v3): `sub(): functions with out=... don't support autograd` —
      `g = norm.weight + 1` не был `.detach()`-нут ⇒ M_K требовал градиент. Починено:
      `g.detach()` + декоратор `@torch.no_grad()` на `_J_from_u`, `_drift_dMK`,
      `_kappa_per_position`, `_beta_sigma1`, `_hidden_to_u`, `_quantize_gptq`; мат-тест
      `test_exp39f_math.py` после правок проходит (κ 0.039539 ≡ E-39b).
- [x] Порог `UCOS_MIN` ослаблен 0.9999 → 0.999: измеренный fp16-фон для верной конвенции
      u_cos_min = 0.99997 (запас был 3e-05), неверная конвенция даёт 0.734 — разрыв
      по-прежнему решающий. Заодно `max_rel` fresh-vs-cache на fp16 = 8e-03 (assert 1e-5
      оставлен только для fp32, как и задокументировано в D1).

## NEXT

- [x] Забрать E-39b v13 (`kaggle kernels output`) → diff с v12-архивом: **побитово
      идентичны** (κ 0.016233/0.016245/0.016043, β/σ₁, cos_ug_1k_proxy, drift, PPL
      30.4753/30.7494/30.7498, gate=pass) + присутствуют все поля round-3
      (`p04_ref_err`, `p04_passed`, `warning_kappa_synthetic`, `kappa_semantics_note`,
      `levels_note`, `gate.readout_channel_only`, `cos_ug_1k_proxy`) ⇒ раунд 3 закрыт.
- [ ] E-39f на Kaggle: `voroninmichael/e-39f-full-tied-ptq-kappa-decomposition` **v4** —
      дождаться COMPLETE, проверить лог на traceback'и и маркеры (`конвенция`, `κ_control`,
      `null/`, `full_tied`), забрать `results/exp39f_full_tied.json`.
- [ ] Проверить воспроизведение E-39b внутри E-39f: `κ_fp16 ≈ 0.01623`,
      `κ_readout ≈ 0.01604`, `drift_readout ≈ 0.1205` (assert'ы в скрипте) и
      `κ_control − κ_fp16` (шумовое поле) против `Δ_upstream`.
- [ ] Док-раунд E-39f: `docs/E39.md §6` (ссылка на новую редакцию ТЗ `docs/E39f.md`),
      карточка `E-39f` в `docs/ref-exp.md`, строки в `docs/paper.md` §4.8/§5/§6.
- [ ] QA-downstream для E-39j (ТЗ §10.4 п.2) остаётся открытым: офлайн-датасетов нет,
      записано как ограничение, не как результат.

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
