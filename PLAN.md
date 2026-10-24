# PLAN: E38_fixes.md — фиксы E-16 + E-38h + E-38i

## Запрос (фиксация)
Выполнить `docs/E38_fixes.md`. Перезапуск только E-16 на Kaggle — **только если
нужен**. Текущий E-16: steps=1000, eval_every=1, cuda, knee_step=84, снапшоты
100/100 в `results/exp16_deltas/exp16_dW_snapshots.pt` — **покрывает
требования E-38h/38i; P1-фиксы E-16 не инвалидируют текущий прогон
(detach/pairing не меняют градиент по lm_head, pairing 4/4 уже верифицирован)
=> перезапуск НЕ нужен.**

## DONE / DOING / NEXT
- [x] Прочитать E38_fixes.md, E-16/38h/38i, hi_common; проверить состояние
      (tied=True в config; снапшоты 100/100; баг-сигнатура в результатах:
      E-38i drift O(1))
- [x] E-38h P0: `dW = snaps` (убрать `− W_orig`) + meta-оговорка про tied
- [x] E-38i P0: `dW_shared = snaps[shared]` (убрать `− W_orig256`);
      переписать tautological verdict; meta: tied ⇒ «h invariant» ложно для
      реальной модели, кэшированный h; drift: max/median + shared-only
- [x] hi_common.load_snapshots: комментарий «snapshots = ΔW» + проверка
      ‖snaps[k]‖ ≤ dose_full(k) из exp16 curve
- [x] E-16: 8 фиксов (W_orig — отдельный readout-снапшот, не финал;
      eval_every=1 по умолчанию; meta steps; eval на 256 токенах;
      readout на 256 с shared_idx; W_orig в JSON + snapshots; top-1 по частоте;
      docstring tied-весов) — текущий прогон не тронут (валиден)
- [x] Перезапуск E-38h и E-38i локально → новые JSON (E-38i: residual
      5.48e-16 тавтология, J_rel_drift max 0.355→1.152 к s=1000, median 0.656,
      bridge 0.60→0.91 согласуется; E-38h: энергия top-16 ΔW 0.735→0.280,
      ΔJ 0.717→0.487, оба канала Abelian) — guard: cos(snap, W_orig) ≈ 0.07
      (double subtraction даёт ≈ −1); dose-guard по norm невалиден для 256-среза
      (2.658 vs full-dose 61.79 — разные срезы), заменён cos-проверкой
- [x] ref-exp карточки E-38h/38i + индекс → v2 (новые числа, оговорка tied)
- [x] E38_fixes.md: секция статуса исполнения
- [x] Коммит + пуш (паттерн ci.yml temp-remove)

## Границы (не менять)
- E-16 на Kaggle не перезапускать (текущий прогон валиден — см. Запрос)
- Не переименовывать ID E-*/L-*; не трогать science-часть ревью
- K_STEPS не менять (все 100 снапшотов есть)
