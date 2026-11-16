# S1: scope-аудит AT3 (клетка T3.1) — read-only

Цель: собрать ФАКТЫ для пре-рега AT3 — клетка T3.1 «Margin-conditional dip»
(docs/TZ_add_0-3.md §4.5, disk-only, 1–2 ч по ТЗ). Ты не решаешь, ЧТО измерять —
только что ЕСТЬ на диске и что достижимо без новых GPU-прогонов. Ответ — ФАКТАМИ,
не гипотезами. Финальная строка вывода: AT3_SCOPE_DONE

CWD: /home/survivor/work/prism. Python: /opt/miniconda3/bin/python (py3.12.9, torch 2.14.1 CPU,
4 трёда). Кэши/npz — read-only, НЕ копировать в репо, НЕ менять.

## Канон-файлы (читать)
1. docs/TZ_add_0-3.md, строки 1137–1290 (§4 T3: постановка, математика, T3.1–T3.6,
   §4.6, §4.7) — объект и пре-рег гипотезы.
2. docs/TZ_add_0-3.md, строки 76–124 (G14-координаты: любое «dip» несёт dip_{space,stat,agg};
   dip_V,med — канон серии; dip_raw,mean — другой объект).
3. docs/addentum/3-2.md, строка ~38 (развёртка T3.1: «Условная Δ(1) по квартилям margin/p1
   (треки p1 из e81b v3, состояния из npz), раздельно на чтении (TF) и генерации»).
4. prism/field.py — `vpath(u, hn)` (канон dip_V: ортогонализованный тангенс; возвращает
   V (T−1−m, D) и dropped m) и `dcurve(P, kk, stat)`.
5. experiments/at0/AT0_1_accounting.py, строки 100–320 — канон пересчёта dip_V,med из npz
   (ANCHOR_LADDER = −0.0722, L1_DEGRADE_TOL, как читать npz, объекты h_raw/du_unit/v_tangent).
6. experiments/kaggle_e64/e64_tpu.py, строки 60–200 (run_doc, P1, P2) — КАК считались
   mg (margin), p1, hn, состояния (какой слой, пост-норм или нет, alignment по t).
7. experiments/at2/AT2.py (строки 1–60) — формат артефакта/логов этой серии (только формат).

## Данные (проверить фактом)
- results/e64/tpu_v8/P1_own.npz — own-greedy генерации (16 wiki-доков, 384 шага):
  ключи own_w{i}_{L{1,5,11,17,23},tok,hn,mg,p1}.
- results/e64/tpu_v8/P1_own_meta.json — per-doc кривые (dip по слоям) + same1 + lookup.
- results/e64/tpu_v8/P2_corpus.npz — TF-чтение корпуса (16 доков, T=163..255):
  ключи cor_w{i}_L{...}. Есть ли в нём mg/p1/hn/tok — ПРОВЕРИТЬ.
- results/e64/tpu_v8/P2_corpus_meta.json — per-doc кривые.
- results/e81/REPLAY.json — e81b v3: gen_p1 (треки p1, длина 384 × 16), gen_ids,
  gate/gate_pass, C_tail_vs_rep8.
- results/e64/tpu/P1_own.npz и results/e64/tpu/P1_own_meta.json (СТАРЫЕ копии — есть ли
  разница с tpu_v8; использовать только tpu_v8, но зафиксировать расхождение).
- Модель: /data/Qwen/Qwen3.5-0.8B (safetensors: model.language_model.embed_tokens.weight —
  tied E; model.language_model.norm.weight — gate-веса финального RMSNorm). Записать:
  vocab size D, shape E, min/max/mean norm.weight, frac(gate ≠ 1).

## Замеры (python, всё disk-only CPU; каждый факт — с числом)
1. P1_own.npz: ВСЕ ключи + shape/dtype; на w0 и w15: T (должно 384), mg: median/p10/p90/
   min/max, frac(mg ≤ 0), frac(mg < 0.01); p1: mean, frac(p1 == 1.0), p10/p90; hn: min/max.
2. P2_corpus.npz: ВСЕ ключи + shape/dtype (есть ли mg/p1/hn/tok — ДА/НЕТ с ключами).
   T на каждом доке (из shape).
3. REPLAY.json: структура, gate_pass, длина gen_p1[w0], mean gen_p1, frac(==1.0);
   совпадает ли gen_ids[w0] с own_w0_tok из P1_own.npz (первое 32: побайтово) — ДА/НЕТ.
4. Alignment (КРИТИЧНО, из кода e64_tpu.py, не гадать): mg[t] соответствует какому t —
   предсказанию tok[t+1] по состоянию после tok[t]? Состояние в npz — «после токена t» или
   «до»? Цитируй строки кода (run_doc + сохранение).
5. Канон dip_V,med: пересчитать из tpu_v8 (как AT0_1_accounting.py: vpath на unit-состояниях
   + dcurve(k=1, median)) для L23 на P2_corpus (16 доков) и P1_own (16 доков):
   (а) медиана по докукам — сравнить с −0.16714 (corpus, TЗ §4.2) и 0.5527 (L23 dip в
   P1_own_meta — почему различие: ортогонализованный тангенс vs сырой v = h_{t+1}−h_t?);
   (б) сколько dropped строк (m) на (док, слой) — диапазон;
   (в) ВНИМАНИЕ: vpath возвращает V БЕЗ позиции t (выброшенные строки) — зафиксировать,
   что per-position dip(1) = cos(V_t, V_{t+1}) требует собственного цикла с сохранением
   t-mapping; посчитать, сколько позиций останется после отброса на L23 (сумма по докум).
6. Margin на TF-стороне (P2_corpus): z = rmsnorm(H_L23) * diag(norm.weight) @ E^T,
   где rmsnorm(h) = h / sqrt(mean(h²)+eps) (как rmsn в exp64_A7_margin.py:20-22).
   Проверить на L23: (а) gate — frac(argmax(z[t]) == tok[t]) на всех позициях (ожидаем
   ~1.0 по ТЗ §4.7 «точный хэд gate 1.000»; НУЖЕН tok — если P2_corpus без tok, откуда
   взять: есть ли токены корпуса в данных/пинах — ПРОВЕРИТЬ); (б) margin = top1−top2:
   median/p10/p90/frac(≤0); (в) wall-секунды расчёта для 16 доков.
7. Слои: для L17 тоже пересчитать (5) и (6) кратко (только median dip + median margin) —
   для выбора слоя в пре-реге.
8. sha256 каждого входного файла (npz/json, 6 штук + safetensors).
9. Среда: python/torch версии, CPU avail RAM (free -g), nproc.

## Запрещено
- Менять/копировать npz/json в репо; запускать модель в GPU-режиме; писать в results/.
- Решать, что именно идти в пре-рег (это делает супервизор).

## STOP-критерии (сомнение → STOP + запись в конец вывода до маркера)
- P2_corpus не несёт ни состояний, ни возможности восстановить токены → TF-сторона
  не disk-only → STOP с фактами.
- mg в P1_own вырожден (frac(≤0) > 0.5 или std < 1e-3) → квантили margin не строятся →
  STOP с распределением.
- Канон dip_V,med L23 corpus НЕ совпадает с −0.16714 в пределах 0.01 → стоп, факт.
