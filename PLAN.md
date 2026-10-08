# E-33 (preflight) — gauge-голономия: транспорт vs случайный

## Запрос
«E33 следующий» — preflight E-33 по docs/Exp33.md. Код, selfcheck, полный прогон, отчёт.
Границы: exp27b кэш + calibration.py — не трогать (read-only); exp34/exp35/exp30/exp28 — не менять.

## DONE / DOING / NEXT
- [x] ТЗ прочитано (docs/Exp33.md)
- [ ] код exp33_gauge_holonomy.py (preflight)
- [ ] selfcheck
- [ ] полный прогон → results/exp33_gauge_holonomy.json
- [ ] отчёт пользователю

## Дизайн preflight (по ТЗ, с задокументированными отклонениями)
- u_g(t) = normalize(J Jᵀ 1) — left-компонента, ближайшая к uniform; v_g = Jᵀ u_g / ‖·‖.
- Позиционный транспорт: u' = norm(P(p_t) u_g(t)); сравнение с u_g(t+1) (1 шаг) и цепочка L=8 шагов;
  baseline — случайные единичные u в R^K, та же цепочка; один-сторонний permutation p.
- Глубинная нога: D_l = ∂h^{l+1}/∂h^l в кэше НЕТ (только H финальный per-layer). Замеряем
  когерентность cos(v_g^l(t), v_g^{l+1}(t)) vs случайные пары в R^1024 — прокси «gauge-направление
  сохраняется вглубь», НЕ транспорт через T^h. Флаг в JSON.
- Sentinel P128_RAW sum<0.5 — позиции исключаются (both t, t+1).
- Слои 6/12/18/22/23, N=32.
- Правило: transport ≈ random (p > 0.05) → gauge-связь тривиальна, E-33 закрывается как контроль.
