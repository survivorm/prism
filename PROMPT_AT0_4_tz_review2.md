# КРОСС-БРИФ: AT0_4 TZ (docs/AT0_4_prereg.md v1) — 5 точечных сверок

Ты ревьюер **ТЗ** эксперимента AT0_4 (T0.4: клетка 3×2 {R0 trained, R1 random trunk +
trained tied embed+head, R2 random всё} × {real, mix}; CPU-форварды Qwen3.5-0.8B;
новое, не disk-only). Найди дефекты ДО прогона. Репозиторий НЕ читать — всё в пакете.

## ТЕХНИКА: нет bash/write — верни ревью как ПОСЛЕДНИЙ ОТВЕТ (≤2500 символов).
Один пункт — 2–3 предложения. НЕ пересчитывай: точечные сверки по §A/§P.

## §P. Факты (замеры исполнителя, не пересчитывать)
- Qwen3.5-0.8B: 24 слоя, hid=1024, hybrid (18 lin-attn + 6 full), **tie_word_embeddings=True**,
  V=248320, RMSNorm. torch 2.14 CPU fp32 4tr. **Бенч: 51.7 с/forward (384 ток.)** →
  112 fwd ≈ 2 ч. 320 unique-тензоров, 752M (tied embed+lm_head = 1 тензор 254.3M).
- P1_own.npz: 16 доков, T=384, **с токенами** (nuniq 16..66); P2_corpus: 0/16 с токенами
  (G16); gen_layers = модельные генерации (не «real corpus»).
- 1D-параметры (баг ndim≥2 серии, exp66_lesion:40/exp7177c:37 — только ndim≥2):
  115 тензоров: input_ln ×24 (mean≈0.41/std 0.13), post_ln ×24 (0.22/0.10),
  lin_attn.norm ×18 (0.94/0.09), model.norm (3.31/0.67), A_log ×18 (−1.24/0.78),
  dt_bias ×18 (−1.75/2.79).
- exp91 (существующий harness): half-random=trained embed+norm+layer-norm'ы+head;
  shuffled ОБЩАЯ для всех клеток (seed 7, M2); sanity finiteness; trained-пин.
- AT0_1: r/r_max/f_dc для own-доков НЕ считаются (только gen) → R0-пин через P1_own npz
  (per-doc median cosine vs TPU fp16, прецедент exp62: >0.999).

## §A. TZ (ключевое)
**Доки:** own_w0..w7 (первые 8 лекс.) — единственный источник с токенами. real = исходные
токены (TF); mix = full-permutation тех же токенов, ОДНА перестановка на док для всех
клеток (default_rng(7)). L=23 → hidden_states[24].
**Рандомизация:** std-matched по КАЖДОМУ тензору N(0, σ_t), σ_t=std(trained); **1D gain-ы
рандомизируются (фикс бага)**. R1 = всё кроме tied embed+head → random (вкл. model.norm и
все layer-norm'ы; НЕ exp91 half-random). R2 = всё random, tied сохранён. Сиды S∈{11,12,13}:
torch.manual_seed(S) перед from_config; стволы R1_S/R2_S побайто (один init-поток + trained-
пара поверх). S2: torch.equal(lm_head, embed) после рандомизации.
**Метрики (per-doc, du_unit L23, fp64):** r=‖ΣΔu‖²/Σ‖Δu‖², f_DC, r_max=4/Σ‖Δu‖² (замер
ДО оценки пре-рега), d_eff=exp(−Σλlnλ) (PCA u, 383×1024), ρ̄(k) moments-ACF (описательно).
**Гейты:** S1: R0×real per-doc median cosine > 0.999 vs P1_own npz (провал ЛЮБОГО дока →
STOP); S2 tied; S3 finiteness 1док×32ток R1_11/R2_11; S4 |r−N·f_DC|/(N·f_DC) ≤ 1e-9;
S5 telescope ≤ 1e-9; S6 R0 head-acc ≥ 0.99. G_nd (R1/R2 per inst×док, иначе not_testable):
cv(‖Δu‖)>0.05; d_eff≥8; H(softmax(logits))/log(V)>0.5.
**Пре-рег (per-doc, медиана по докам, per-seed, α=0.05):** P1: R2×real r≈1 := r∈[0.75,1.5]
ТОЛЬКО при r_max≥1 (иначе vacuous(r_max<1) — объявленный исход); P2: |median r(R1)−r(R2)|
≥0.1 per-seed ⇒ trained-пара вносит вклад (знак не регится); P3: R0 r(mix)−r(real)>0
(направленная, нуль допустим); P4: random |r real−r mix|<0.1 (провал = token-statistics
флаг); P5: f_DC(mix)−f_DC(real)>0 (R0). Σ — НЕ статистика (G7).

## §D. Пункты (2–3 предложения на каждый)
**N1 (доки)**: own_w0..w7 = «real corpus TF» — корректно? Конфаунд, который TZ не называет
(small-vocab: nuniq 16..66 на 384 ток.)?
**N2 (рандомизация)**: R1=«всё кроме tied embed+head» против T0.4-формулы «random trunk,
trained tied embed+head» — расхождение? N(0,σ) для model.norm (mean 3.31, std 0.67) —
обнуляет ли? Tied-R2: как именно (один тензор)?
**N3 (пороги)**: P1 r∈[0.75,1.5] — обоснованно (N=383)? P2/P4 0.1 — априори или подогнан?
«vacuous» — порок, уходящий от фальсификации?
**N4 (гейты)**: cv>0.05 / d_eff≥8 / ent>0.5 — false-gate на random-клетках? S1 0.999 vs
fp16-шум — жёстко/мягко? S6 ≥0.99 на own-доках — реалистично?
**N5 (mix/целостность)**: full-permutation (не block) — смешивает P3/P4 эффекты? Одна
перестановка для всех клеток — зависимость в P4? Цена 112×51.7с+загрузки — честна?

**НОВЫЕ**: <= 5, «- [BLOCK|MAJOR|MINOR|COSMETIC] <пункт> <что не так> <как чинить>».
Или «НОВЫЕ: нет».

## ФОРМАТ (строго, ≤2500 символов):
# AT0_4_TZ_REVIEW
Ревьюер: ...
N1: <закрыто/нет + <=2 строки>
N2: <то же>
N3: <то же>
N4: <то же>
N5: <то же>
НОВЫЕ: …
VERDICT=PASS|PASS_WITH_MINORS|BLOCK
AT0_4_TZ_DONE
