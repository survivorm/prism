# AT3 (T3.1) — пре-рег «Margin-conditional dip» (ТЗ docs/TZ_add_0-3.md §4.5, v1)

Записан супервизором 16.11 (по прецеденту AT2 S2: воркер-писатель нежизнеспособен
на output-кап; факты — docs/dynamic/orch/AT3_scope.md, SUPERVISOR_FALLBACK).

## 1. Объект (G14-координаты)

- **Основной статистик:** `dip_V,med | Q_k(margin)` — per-position dip(1) = cos(Vo[t], Vo[t+1])
  (Vo — ортогонализованный тангенс, канон prism/field.py `vpath`, ok-маска ‖Vo‖>1e-6),
  медиана по позициям внутри квантильного бина margin, медиана по докум. Новая
  (условная) версия канона dip_V,med — зафиксировано здесь.
- **Стороны (раздельно, две независимые вердикты):**
  - **TF (чтение):** состояния P2_corpus (16 wiki-доков, T=163..255), токены из локального
    пина (data/calib_pins, tokenize → [:256]), margin = top1−top2 из
    z = (rmsnorm(h)·w)@Eᵀ (w = norm.weight, E = tied embed, eps 1e-6).
  - **Генерация:** состояния P1_own (16 доков × 384), margin = **saved mg** (TPU fp64,
    точный; пересчёт — только гейт).
- **Alignment (зафиксировано scope'ом, верифицировано end-to-end):** state[j] —
  состояние, предсказавшее tok[j]; переход state[j]→state[j+1] потребляет tok[j];
  **Vo[j] ↔ mg[j]** (обе стороны; TF: state[j] = после ids[j], z_j предсказывает
  ids[j+1] с margin mg_tf[j] = mg_tf-индекс j).
- Слой: **L23 — primary** (якорь dip_V,med corpus = −0.16714, TЗ §4.2). L17 —
  secondary описательный блок (вердиктов нет).

## 2. Фиксации (не менять после старта)

1. Квантили: **глобальные квартили** (Q1..Q4) margin по ВСЕМ позициям стороны
   (с обеих сторон, отдельно; ~2850 TF / 6144 gen позиций с lag-1).
2. Статистик: per-doc per-bin **медиана** dip[t] (семья dip_V,med); свёртка по докум —
   **медиана per-doc medians** (primary) + pooled-медиана (check, в артефакте).
3. Δ = dip|Q1 − dip|Q4 (per-doc → медиана по докум). CI95 — **paired bootstrap по
   докум**, B=1000, seed 0 (prism/nulls_and_stats.py `paired_bootstrap`).
4. Порог значимости (зафиксирован ДО прогона): **0.02** (≈13% масштаба dip ~0.15).
5. Стороны независимо: TF-вердикт и gen-вердикт (ТЗ: «раздельно»).
6. p1: secondary таблица (те же бинации по p1: gen — saved p1; TF — из z), вердиктов нет.
7. Монотонность Q1→Q4 — отчитывается (тренд), не гейтится.
8. Логит-шум fp16 на TF-стороне (maxdiff ~4 на логитах) — принят: типичный шум
   ~0.02 на margin << ширина бина (~0.45); в артефакте — описательное поле
   `bin_boundary_frac` (frac позиций в ±0.05 от ближайшей границы бина, на сторону),
   гейта нет. На gen-сторону не распространяется (saved fp64).
9. Sлой L17: только описательный (median dip|Q_k по сторонам), без CI/вердиктов.
10. Цена: ~10–20 мин CPU (TF matvec ~6 мин + dip + bootstrap). Trackio:
    `at3_margin_r1` (step = wall-с, config: device/model/log/md5/revision/script_sha/seed).
11. Артефакты: `results/AT3/AT3.json` (gitignored, sha в отчёте), `logs/AT3_run.log`
    (tracked), отчёт `docs/AT3_results.md`. Скрипт: `experiments/AT3.py`
    (на prism/explog.py + prism/nulls_and_stats.py — указание user 16.11).
12. Среда: /opt/miniconda3 base (py 3.12.9, torch 2.14.1 CPU), 4 тр. (OMP), seed 0;
    full-forward модели в прогоне НЕ нужен (только верификация scope'ом).

## 3. Вердикты (фиксированы ДО прогона, на сторону)

- **(a) boundary:** CI95(Δ) **целиком < −0.02** — «dip|низкая margin существенно
  глубже dip|высокая» (boundary-версия ТЗ подтверждена на этой стороне).
- **(b) suffix-карта (слабая/недоказанная зависимость):** ВСЁ, что не (a) и не (c)
  — CI содержит 0, или Δ ≥ −0.02, или CI целиком < 0 но не ниже −0.02
  (например [−0.07, −0.01]: эффект есть по знаку, но слабее порога) — margin не
  вентиль (карта отвечает на суффикс). Под-метки (weak / insufficient power) —
  описательно, в артефакте.
- **(c) сверх карты:** CI95(Δ) целиком > +0.02 — эффект в противоположную сторону:
  результат за пределами ТЗ-карты (отчитывается как таковой).
  (Партиция полная: (a), (c) — непересекающиеся; (b) — остаток.)
- Итог AT3: пара вердиктов (TF, gen). Оба (a) → T3.1 подтверждён; иначе — по тексту.

## 4. Гейты (fail-loud, в порядке)

- **S1** sha256 (полные, из scope §1):
  - P1_own.npz `783e16338cdcb652198722a33ecec8a6e8e98fa3c611c687b17db722b0d58198`
  - P2_corpus.npz `5f42a08442a5d3bcb8d7c4de5c9658340c0378a917a5e633b9b2aa589ceac125`
  - P1_own_meta.json `3cebe6f328083c39a7c0130555dde84a62c1ab4c745d48b1f28b5f5eb3538042`
  - P2_corpus_meta.json `5fb47a32f9402074cdc47284a5e26bc7996572563abfc7b46e2c8eb661581a2e`
  - REPLAY.json `5580d8f2c8341e9a149d18b20ba16bbe8ad43729dacf3441fd1a7905111e5e8f`
  - pin `083c732ff67c04f8585a5c942eadc9fea7fa2134f78cf869826f11ab3d35ce2f`
  - model.safetensors `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`
- **S2** anchor dip: corpus L23 median-of-docs dip_V,med ∈ [−0.16714 ± 0.001]
  И own L23 ∈ [−0.1111 ± 0.002] (воспроизведение scope'а).
- **S3** TF-здравый gate: top-1 accuracy (argmax z_j == ids[j+1]) ≥ 0.25 на КАЖДОМ
  TF-доке (измерено 0.277..0.529; при сбитых токенах/весах ≈ 0).
- **S4** alignment-гейт (gen): Spearman(recomputed mg из fp16, saved mg) ≥ 0.95
  на КАЖДОМ доке (измерено 0.958..0.996).
- **S5** bin-size: каждый (сторона, квантиль) — ≥ 50 позиций c lag-1; пересчёт
  Δ и CI после отброса позиций без lag-1 (n_lag1 из scope: 81..382/док).
- **S6** детерминизм: seed 0; verdicts поля артефакта восстанавливаемы из остальных
  полей (`--recheck` — без пересчёта).

## 5. Отклонения от ТЗ (зафиксированы)

- **D1:** ТЗ §4.2 «точный хэд gate 1.000» — **закрыто в r3 (ревью r1, F2)**:
  это self-consistency хэда (argmax z == свой токен), не TF-accuracy. На pre-norm
  состояниях e64 точный хэд = (h/rms)·w·Eᵀ (S4 0.9868, fidelity 0.9958 ≈ 1.000
  с fp16-шумом); ТЗ-формула «без norm.weight» калибрована на post-norm L23
  (jcache-конвенцию). TF-accuracy 0.277–0.529 — next-token-точность 0.8B на корпусе
  (другая величина). Закрыто в §6.
- **D2:** TЗ «p1-треки из e81b v3» — фактически совпадают с P1_own p1 (1e-4);
  используем saved p1 из P1_own (те же данные).
- **D3:** TЗ-«margin» на TF не был в npz — вычисляется z-формулой (scope §2,
  верифицировано); на gen — saved.

## 6. Правка r3 (ревью r1, 16.11; ответы — docs/AT3_review_answers_r1.md)

Записано ДО перепуска r2 (disk-only, dip не пересчитывается):

1. **Дегенеративный фильтр (F1):** `DOCS_EFF = 16\{0,4}` (период-1 лупы по
   AT0_4 token_provenance: max_share 0.9297/0.9401, min_period_90pct=1) для ВСЕХ
   научных агрегатов (бины, pooled, monotonic, Δ, p1, boundary). S2-якоря —
   воспроизведение ТЗ, остаются по 16. AT0_4_MIX.json — 8-й вход S1-гейта (sha).
2. **Хэд (F2):** основной margin — точный хэд pre-norm (h/rms)·w·Eᵀ (S4 0.9868,
   fidelity 0.9958); no-w — sensitivity в артефакте (S4_nw 0.8974, fidelity_nw
   0.9823, flip 2.07%, масштабы margin). Новый гейт **S3b**: min doc fidelity ≥ 0.98.
   TF-сторона — низкорешиционная (fp16-шум margin ~1.3 vs диапазон 0.27–1.38).
3. **Порог вердиктов (F4):** `threshold_used = max(0.02, max_b|pooled_b − mono_b|)`
   по стороне (L23, после фильтра); зафиксирован в артефакте, вердикты (a)/(b)/(c)
   с ним. (b) переименован **b_non_boundary** — без утверждения о суффикс-карте.
4. **per_doc_bin_n + min-n sensitivity {3,20,50} (F5).**
5. **tf_p1 (F6):** p1-квантильная таблица на обеих сторонах, L23 и L17.
6. **±1 margin-shift (F7):** dip[j] ↔ mg[j+1] — квартиры, Δ, CI.
7. **окна bin_boundary_frac из данных (мелкое):** gen — p95|recomputed − saved|;
   tf — p95|margin_w − margin_nw|.
8. **sign-test** (gen L23) в артефакте и отчёте.

## 7. Правка r3-прогона (интерпретационное ревью, 16.11; ответы —
   docs/AT3_review_answers_r1.md §6)

Записано ДО перепуска r3 (disk-only; dip/бининг/вердикты не меняются, только
поля артефакта + один новый matmul на док):

1. **N1:** `min_n_sensitivity[mn] = {delta, n_docs, docs}`.
2. **N2:** p1-блок → Δ + paired CI + n; δp1 = p1_med(1−p1_med)·noise (gen: noise =
   p95|mgw − saved|; tf: перенос gen-оценки с оговоркой); флаг `resolvable`
   (δp1 < мин. ширина средних бинов); неразрешимые оси — описательные.
3. **N3:** L17-local readout `(h_L17/rms)·w·Eᵀ` → L17-маржа; dip L17-gen по
   L17-марже (квартили + Δ + CI); corr(L17-local, final).
4. **N4:** headroom гейтов S3b/S4 (exact vs no-w min, threshold) в артефакт;
   **дискриминатор — S4** (разрыв 0.182), S3b — вспомогательный, порог 0.98
   (предложение ревью «0.99» отклонено: exact min 0.9895 < 0.99).
5. **N5:** `boundary_window` → `margin_windows`: `margin_noise_gen`,
   `convention_spread_tf`, `h_norm` по сторонам, `tf_noise_estimate` (перенос
   gen, оговорка).
6. **N6:** ok-pair-rate и ‖Vo‖-квантили (p10/p50/p90) по margin-бинам.
7. **N7:** corr(mg[j], mg[j+1]) per side (per-doc + медиана), L23.
8. **N8:** dropped per doc в S2-якорях; hard-assert нет (якорь — канон vpath);
   max dropped > 0 → warning.
9. **Отчёт:** строка чтения F7-асимметрии — как результат, не sensitivity
   («коррекция после неуверенного шага», T1-флэвор; корреляция + конфаунд
   сложности контента).

Не меняется: S1–S5, S3b=0.98, primary Δ (min-n=3), threshold_used = max(0.02,
spread), DOCS_EFF, партиция (a)/(b)/(c).
