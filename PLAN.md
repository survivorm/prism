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
- [ ] E-38h P0: `dW = snaps` (убрать `− W_orig`) + meta-оговорка про tied
- [ ] E-38i P0: `dW_shared = snaps[shared]` (убрать `− W_orig256`);
      переписать tautological verdict; meta: tied ⇒ «h invariant» ложно для
      реальной модели, кэшированный h; drift: max/median + shared-only
- [ ] hi_common.load_snapshots: комментарий «snapshots = ΔW» + проверка
      ‖snaps[k]‖ ≤ dose_full(k) из exp16 curve (ловит двойное вычитание;
      ‖snaps‖ ≪ ‖W_orig‖ НЕ подходит — при dose 161.85 > ‖W_orig‖ 9.93)
- [ ] E-16: (1) schema-поле в snap_payload; (2) tied-оговорка в docstring/meta;
      (3) h_s.detach() в loop; (4) pairing_ok полная форма; (5) W_orig fp32;
      (6) снапшот через [token_set] без полного тензора; (7) meta: CPU steps=40
      не даёт снапшотов для 38h, knee=84 только при eval_every=1;
      (8) dose в pt_final — fp32 (унификация с final_dose)
- [ ] Перезапуск E-38h и E-38i локально → новые JSON
- [ ] E38_fixes.md: секция статуса исполнения
- [ ] Коммит + пуш (паттерн ci.yml temp-remove)

## Границы (не менять)
- E-16 на Kaggle не перезапускать (текущий прогон валиден — см. Запрос)
- Не переименовывать ID E-*/L-*; не трогать science-часть ревью
- K_STEPS не менять (все 100 снапшотов есть)
