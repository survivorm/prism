# AT3 (T3.1) — S1 scope-аудит: факты (16.11)

**Маркер: SUPERVISOR_FALLBACK** — 2 воркера (r1: 28 мин sleeping без TCP, r2: 25 мин S 0.0% CPU,
0 байт вывода; endpoint-повес, прецедент AT2) убиты, шаг выполнен супервизором.
Инфра-смоук (1-line prompt) между убийствами — ОК. Все факты ниже воспроизводимы
скриптом /tmp/at3_scope.py (disk-only, wall 381.7 с) + проверочными forward'ами.

## 1. Данные (инвентарь, sha16)

| файл | sha256 (16) | содержимое |
|---|---|---|
| results/e64/tpu_v8/P1_own.npz | 783e16338cdcb652 | 16 own-gen: `own_w{i}_{L{1,5,11,17,23}}` (384×1024 fp16), `_hn` (384,) f32, `_mg` (384,) f64, `_p1` (384,) f64, `_tok` (384,) i64 |
| results/e64/tpu_v8/P2_corpus.npz | 5f42a08442a5d3bc | 16 TF-доков: **только состояния** `cor_w{i}_L{1,5,11,17,23}` (T×1024 fp16), T = 163..255 |
| results/e64/tpu_v8/P1_own_meta.json | 3cebe6f328083c39 | per-doc: curves (dip/hump/plateau/curve k=1..48, raw-diff estimator, fp32 TPU), same1, lookup |
| results/e64/tpu_v8/P2_corpus_meta.json | 5fb47a32f9402074 | per-doc curves (то же) |
| results/e81/REPLAY.json | 5580d8f2c8341e9a | e81b v3: gen_ids/gen_p1 (w0..w7, 384), gate_pass 16/16, C_tail_vs_rep8 |
| data/calib_pins/wiki___…plain.json | 083c732ff67c04f8 | 16 wiki-текстов (fp 144c722df3046f2d), pin локальный |
| model.safetensors (0.8B) | 04b1c301231dd422 | Qwen3.5-0.8B, /data/Qwen |

P2_corpus **не содержит токенов/mg/p1** (e64_tpu.py P2-блок сохраняет только
`full[L][:len(ids)-1]`). Токены TF-стороны восстанавливаются: pins → tokenize →
`ids[:256]` (calibration max_length=256 — именно обрезка до 256 даёт точное
совпадение длин с npz: 16/16, `[:448]` в gen_corpus не действует).

## 2. Alignment (КРИТИЧНО, верифицировано реальным forward)

Модель: Qwen3_5ForCausalLM — **гибрид** (linear attention + full, 24 слоя, D=1024,
V=248320, tied embed, rms_norm_eps 1e-6). CPU-forward требует чисто-torch fallback
fla (triton на этой машине не работает; monkey-patch /tmp/qwen35_cpu.py).
T3.1 не нужен full-forward (только matvec), fallback — только для верификации.

- **npz[j] = состояние, предсказавшее tok[j]** (margin mg[j]): cos(npz[0], h_63) =
  0.9838 vs cos(npz[0], h_64) = 0.699; cos(npz[1], h_64) = 0.9827. Т.е. TPU-hook
  записал состояние НА Позиции 63+j (до токена), хотя код e64_tpu.py читает pos+1 —
  provenance-факт, не влияет на анализ (alignment верифицирован end-to-end).
- margin[j] = margin предсказания tok[j] (логит на позиции 63+j). Сид-проверка:
  мой forward на 64-сиде w7: margin 0.0182 == saved mg[0] 0.0182 (точно),
  p1 0.0637 vs 0.0635, greedy-токены совпадают (16576, 364, …).
- **Переход state[j] → state[j+1] потребляет tok[j], предсказанный с margin mg[j].
  ⇒ Vo[j] ↔ mg[j] (генерация).** TF-сторона симметрично: state[j] = после ids[j],
  z_j предсказывает ids[j+1] с margin mg_tf[j] ⇒ **Vo[j] ↔ mg_tf[j].**
  (Согласованно: margin предсказания, сделанного из state[j].)
- **z-формула: z = (rmsnorm(h) · w) @ Eᵀ** (w = norm.weight, E = tied embed):
  cos(z, real logits) = 0.99913..0.99927, argmax совпадает на всех проверенных
  позициях (maxdiff ~4 — fp16-шум состояний).
- REPLAY.json: gen_ids[w0..w7] == P1_own tok **384/384 побайтово**; gen_p1 совпадает
  с own_w*_p1 в пределах 1e-4. e81b v3 — полный реплей той же генерации.

## 3. Канон dip_V,med (vpath, prism/field.py) — воспроизведение якорей

- **corpus L23: медиана по 16 докум = −0.167138** (якорь ТЗ §4.2 −0.16714 ✓ до 5 знаков).
- own (генерация) L23: −0.11109 (meta raw-diff TPU: −0.1111 ✓).
- corpus L17: −0.10371; own L17: −0.1880.
- dropped (‖Vo‖≤1e-6): **0 во всех 32 (док, слой)**; n_lag1 ≈ T−2 (81..382) —
  per-position dip(1) определён практически на всех позициях.
- Per-doc corpus L23: −0.1199 … −0.2382 (медиана −0.1671, разброс ±0.05).
- **raw-diff estimator на fp16-состояниях = −0.377/−0.382 — артефакт fp16
  (катострофическая компенсация в разности), НЕ ИСПОЛЬЗОВАТЬ.** Meta-значения dip —
  TPU fp32 raw-diff, совпадают с vpath (на гладких траекториях оценщики равны
  с точностью O(θ²)).

## 4. Margin: распределения (квантили строятся, STOP-критерий пройден)

**Генерация (saved mg, 16×384, точный TPU fp64):**
- q25/med/q75 по всем: **6.57 / 8.92 / 10.35**; per-doc med 8.74..9.54;
  p10 3.69..4.39; min 0.018..0.078; frac(≤0) = 0 во всех 16 (динамика ~500×).
- p1: mean 0.940..0.970, p10 0.924..0.966, p90 0.9998..1.0, frac(==1.0) = 0.

**TF (вычислено z-формулой из fp16-состояний, 16 доков, ~2850 позиций):**
- L23: med 0.68..0.73; p10 0.11; p90 2.2..2.34; min 0.001..0.007; max 4.5..7.7;
  frac(≤0) = 0. (Динамика ~1000×, но уровень в ~12× ниже генерации.)
- L17: med 0.37..0.56.
- p1 (L23): mean 0.245..0.254, p10 0.046, p90 0.54..0.59.
- **gate TF (argmax z_j == ids[j+1]): 0.277..0.529 с w; 0.35..0.41 без w.**
  ТЗ §4.2 «точный хэд gate 1.000» НЕ воспроизводится на этих данных — отклонение D1
  (см. пре-рег); на margin-анализ не влияет (margin — margin истинных логитов).
  (L17 gate 0.02..0.05 — ожидаемо для промежуточного слоя.)

## 5. Прочее

- Среда: py 3.12.9, torch 2.14.1+cu130 (CPU), 12 ядер, **16 GB avail** (на момент S1;
  запуск полного прогона — при avail ≥10G, гейт как у AT2).
- Цена TF-margin matvec: L23+L17 = ~6 мин CPU (16 доков). Генерация — 0 (saved).
- vpath/dcurve — spec-лоад prism/field.py (минуя prism/__init__: matplotlib отсутствует
  в /opt/miniconda3 base; паттерн AT2.py:67).
- Стороны различны по уровню dip (corpus −0.167 vs gen −0.111) и по margin (0.68 vs 8.9)
  — «раздельно на TF и генерации» в ТЗ не формально, а по существу.

## 6. Стоп-критерии (все пройдены)

1. TF-сторона disk-only: состояния ✓, токены из локального пина ✓ (длина 16/16).
2. mg не вырожден: frac(≤0) = 0, динамика 500–1000×, обе стороны.
3. Канон anchor −0.16714 воспроизведён до 5 знаков.

SUPERVISOR_FALLBACK AT3_SCOPE_DONE
