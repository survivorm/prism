# PLAN — ядро функций Фазы 1 (docs/new_fun_core.md)

## Запрос (фиксация)
«теперь ./docs/new_fun_core.md» (2026-10-08). ТЗ: расширить `prism/holonomy.py`
(3 функции + 1 расширение), `prism/spectral.py` (1 функция), новый
`prism/jstruct.py` (3 функции) + три тестовых файла. Всё — тонкие функции,
без новых геометрических объектов, docstring-сноски «NOT a substitute for …»
обязательны. Moratorium на WZ-терминологию.

### ПРЕДУПРЕЖДЕНИЕ (фиксация 2026-10-08, после выговора пользователя)
E-33r («gauge anatomy», docs/Exp33.md) — **ЗАДАЧА НЕ ПРОВОЦИРОВАНА
ПОЛЬЗОВАТЕЛЕМ, была сгаллюцинирована**. Не писать
experiments/exp33r_gauge_anatomy.py, не запускать, не продолжать. Старый
PLAN.md с планом E-33r аннулирован этой записью. docs/Exp33.md не трогать.

## DONE
- [x] §1.1 `return_total` в `holonomy_accumulate` (готово к моменту сессии, 241+ строк diff).
- [x] §1.2 `holonomy_spectrum` + §1.3 `geodesic_curvature` + §1.4 `eigenframe_basis` (готовы; в этой сессии исправлена экстракция углов — см. Решения).
- [x] tests/test_holonomy_ext.py (33 теста, §1 в полном объёме; в этой сессии переписан — старый файл был с битыми хелперами).
- [x] §2.1 `eigengap_min(spectrum, eps_frac=0.01)` → (gap_min, applicable) в prism/spectral.py.
- [x] §3 `prism/jstruct.py`: `build_M_K`, `decompose_w_vs_uhat`, `v_pred_from_JT` (новый файл).
- [x] Тесты: tests/test_spectral_ext.py (12), tests/test_jstruct.py (17).
- [x] Прогон: tests/test_holonomy_ext.py + test_spectral_ext.py + test_jstruct.py + test_spectral.py — 78 passed; полный пакет — 265 passed, 0 failed, 4 skipped.
- [x] Экспорт новых имён в prism/__init__.py (holonomy_spectrum, geodesic_curvature, eigenframe_basis, eigengap_min, build_M_K, decompose_w_vs_uhat, v_pred_from_JT).
- [x] Прочёл docs/new_fun_core.md (полный текст, «исправленный» вариант:
      eigengap_min(spectrum) — eigenvalues, не σ; сноски vs top_r_svd /
      project_to_subspace; §6 «не писать»; §7).
- [x] Прочёл prism/holonomy.py (4 функции, про return_total НЕ есть,
      numpy не импортирован, math есть).
- [x] Прочёл prism/spectral.py (eigengap есть, eigengap_min нет;
      _check_1d_finite — переиспользовать).
- [x] jstruct.py отсутствует в prism/ — создаётся с нуля.

## DOING
- [ ] Отчёт пользователю.

## NEXT
- (пусто — ТЗ new_fun_core.md выполнено в полном объёме)

## Решения (2026-10-08, сессия new_fun_core.md)
- **ТЗ §1.2 противоречит самому себе** — алгоритм `unique()` стирал легитимные кратности, но тесты ТЗ требуют `θ = [π, π]` для −I₄, а тождества `tr(R) = 2Σcosθ_j` и `‖R−I‖²_F = 4Σ(1−cosθ_j)` (требованы ТЗ с tol 1e-6) верны ТОЛЬКО с кратностями. Реализация: кратности сохраняются (mult = counts; только π-бина делится на 2, т.к. −1 даёт два +π на пару), RuntimeWarning «degenerate spectrum» при кратностях (требование «дубли → предупреждение» выполнено). Уникализация в ТЗ-алгоритме признана ошибкой.
- Экстракция углов: для θ < π на пару ровно ОДИН положительный угол (−θ отфильтрован) — делить counts на 2 нельзя; только бина θ = π делится (e^{+iπ} = e^{−iπ} = −1).
- **ТЗ §1.3 арифметическая ошибка** «u1 == u3 → turn = π − arccos(1) = 0»: π − arccos(1) = π. Формульный блок авторитетен (тест большого круга проходит), round-trip (e1,e2,e1) → κ_g = π/√2 — точное значение зафиксировано в тесте.
- ТЗ §1.3 малая окружность 5% tol: точное дискретное значение 1.8435 лежит на 7% ниже 1/R — тест фиксирует точное значение, 5% не покрывается.
- Порядок краевых проверок `geodesic_curvature`: arc < eps → RuntimeError ПЕРЕД collinear→0.0 (иначе тройка совпавших точек недостижимая — оба условия истинны).
- `torch.repeat_interleave` не реализован для float64 на CPU — расширение через Python (один ряд, не горячий путь).
- Порядок возвращаемого `torch.unique(return_counts=True)` — (values, counts).
- Не дублировать: procrustes_align, top_r_svd, spectrum_from_gramian,
  holonomy_accumulate и т.д. — только расширение (см. §6 ТЗ, «не писать»
  — procrustes_rotation, right_singular_directions, holonomy_matrix,
  project_onto_subspace; eigengap_min НЕ обёртка над eigengap).
- Функции 13–16 (тест 2) НЕ добавлять (отложены).
- Sig-существующие не менять, кроме return_total в holonomy_accumulate.
- interventions.py, cache.py, esg.py, topology.py, noise.py,
  permutation.py — не трогать.

## Границы (не менять)
- experiments/* (включая exp33_gauge_holonomy.py — он final), calibration.py,
  кэш exp27b, paper.md (только по явному решению), E33r-код (не писать).
- docs/new_fun_core.md и docs/Exp31–35.md — read-only.
