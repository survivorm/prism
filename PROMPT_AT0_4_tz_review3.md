# КРОСС-БРИФ: AT0_4 TZ v2 — 4 точечных сверки (≤1500 символов ответа)

Ты ревьюер ТЗ эксперимента AT0_4 (T0.4: 3×2 {R0 trained, R1 random trunk+trained
tied embed, R2 random всё} × {real, mix}; CPU-форварды Qwen3.5-0.8B; НЕ disk-only).
Ревьюер умер на length дважды — ответ КРАТКИЙ: 1–2 предложения на пункт. Репозиторий
НЕ читать. Найди дефекты ДО прогона.

## §P. Факты (не пересчитывать)
- Qwen3.5-0.8B: 24 слоя, hid=1024, hybrid (18 lin+6 full), tie_word_embeddings=True,
  V=248320, RMSNorm. torch 2.14 CPU fp32 4tr. **51.7 с/forward (384 ток.)** → 112 fwd ≈ 2ч.
- P1_own.npz: 16 доков, T=384, С ТОКЕНАМИ (авторегресс. генерации: prompt 64 wiki + 384 gen).
  P2_corpus: 0/16 с токенами. gen_layers = модельные генерации.
- 1D-параметры (баг ndim≥2 серии): 115 тензоров (input_ln ×24 mean≈0.41, post_ln ×24
  0.22, lin_attn.norm ×18 0.94, model.norm 3.31, A_log ×18 −1.24, dt_bias ×18 −1.75).
- **v2-смена (право по смоку):** вход = [tok] 384 БЕЗ prompt (TPU-косинус-пин
  недостижим: TPU-bf16 vs CPU-fp32 linear-attention state, greedy-каскад w0 93%/w1 5%);
  S1 = R0 liveness (head-acc≥0.90, измерено 0.945–0.977); G_nd not_testable только
  если ВСЕ доки вырождены (d_eff у LLM на тексте 5..21, порог 8).

## §A. TZ v2 (ключевое)
Доки own_w0..w7. real = [tok] 384 TF; mix = full-permutation тех же (ОДНА перестановка
на док для всех клеток, seed 7). L=23→hidden_states[24]. Рандомизация std-matched
N(0,σ_t) по ВСЕМ тензорам (1D gain ВКЛ, фикс бага); R1=всё кроме tied embed+head;
R2=всё random, tied сохранён; сиды S∈{11,12,13} (torch.manual_seed до from_config;
стволы R1_S/R2_S побайто). Метрики per-doc du_unit L23 fp64: r, f_DC, r_max=4/Σ‖Δu‖²
(замер ДО оценки пре-рега), d_eff=exp(−Σλlnλ), ρ̄ moments-ACF. Гейты: S1 liveness,
S2 tied, S3 finiteness, S4 |r−N·f_DC|/(N·f_DC)≤1e-9, S5 telescope≤1e-9. G_nd
(cv>0.05 ∧ d_eff≥8 ∧ ent>0.5). Пре-рег (per-doc, медиана, per-seed, α=0.05):
P1 R2×real r≈1[0.75,1.5] только при r_max≥1 (иначе vacuous); P2 |med r(R1)−r(R2)|≥0.1
per-seed; P3 R0 r(mix)−r(real)>0; P4 random |r real−mix|<0.1 (иначе token-stat-флаг);
P5 R0 f_DC(mix)−f_DC(real)>0. Σ — НЕ статистика (G7).

## §D. Пункты (1–2 предложения)
**N1**: вход [tok] без prompt — не теряет ли AT0_4 суть T0.4 («real corpus»)? Oбоснована
ли замена S1-пина liveness-чеком для относительного сравнения?
**N2**: std-matched N(0,σ_t) — обнуляет ли model.norm (mean 3.31, std 0.67) / A_log?
R1 «всё кроме tied embed+head» против T0.4 «random trunk, trained tied embed+head» — расхождение?
**N3**: пороги P1 [0.75,1.5] (N=383) и P2/P4 0.1 — априори/подогнаны? «vacuous(r_max<1)»
— порок, уходящий от фальсификации?
**N4**: G_nd d_eff≥8 — false-gate на random-клетках (измерено: real ~5, mix ~17)? Правило
«ВСЕ доки» — не слишком ли мягкое (1 живых док ⇒ testable)?

**НОВЫЕ**: <=4, «- [BLOCK|MAJOR|MINOR|COSMETIC] <пункт> <что не так> <как чинить>».
Или «НОВЫЕ: нет».

## ФОРМАТ (строго, ≤1500 символов):
# AT0_4_TZ_REVIEW
Ревьюер: ...
N1: <ок/нет + 1 строка>
N2: <то же>
N3: <то же>
N4: <то же>
НОВЫЕ: …
VERDICT=PASS|PASS_WITH_MINORS|BLOCK
AT0_4_TZ_DONE
