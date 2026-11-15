# AT0_2b — ответка на ревью round 2 (docs/dynamic/AT0_2b_REVIEW_r2.md)

**Статус: PASS_WITH_MINORS → все 4 MINOR + 3 NIT закрыты (v2.1, store 15ad06bdae5c14c5,
claim file sha 014bf176246540aa).**
Дата: 2026-11-15. Скрипт: `experiments/AT0_2b_ra_split.py` (v2.1). Пре-рег:
`docs/AT0_2b_prereg.md` (v2 + правка MINOR-4).

## MINOR-фиксы (все — только текст/полевые метки; вычисления не тронуты)

| пт | Фикс | Доказательство |
|---|---|---|
| **MINOR-1** (gram = инициализация) | `n_gate["gram"]` обновляется из per-cell gram (main, в том же цикле, где GRAM_TOL-гейт) | claim/store `gate_max.gram = 6.439293542825908e-15` = независимый пересчёт ревьюера (max по 956 компонент-значений) |
| **MINOR-2** (tracked лог без хвоста) | `RUN_LOG` пишется ПОСЛЕ всех logline (store:/written:/ALL PASS) | tracked `AT0_2b_run.log` заканчивается строкой `AT0_2b gates: ALL PASS` (ранее — до 3 строк) |
| **MINOR-3** (location-зависимый canonical sha) | ключи `data_sha256` = **basenames** файлов (location-invariant) + правило записано в `meta.hash_convention` | новые ключи: `gen_layers_qwen.npz / P1_own.npz / P2_corpus.npz`; canonical sha v2→v2.1 изменился ИСКЛЮЧИТЕЛЬНО из-за basename-ключей + meta (cells payload bit-eq, см. ниже) |
| **MINOR-4** (пре-рег «corpus 123..254») | → «corpus 82..254» (cor_w8 N=82, cor_w11 N=94) | пре-рег §2; store: длины rho96 {96, 81, 93} |

NIT: (1) мёртвая `D` в `sign_flip_fdc` — удалена; (2) `_boot_ci_joint` — fail-loud
`assert len(r) == len(a)` (desync joint-сэмплинга); (3) decode-проба: except расширен
(ImportError, ModuleNotFoundError, OSError, ValueError) → «skipped», не краш.

## Чего не изменилось (инвариант v2 → v2.1)

- **cells payload bit-eq**: `json.dumps(cells, sort_keys=True)` — идентичен
  (store b7e90c93… → 15ad06bd…); вычисления детерминированы сидами, повторный прогон дал те же
  значения (primary L23: cR −0.0015142104413634894, cA 1.0015142104413635 — как в v2).
- **Все 15 вердиктов = NO_DH_DC** (dh_frac_dc = 0.0 везде) — не тронуто; dominance-канал
  (descriptive) — не тронуто.
- meta-диф v2→v2.1 = ровно 4 ключа: `data_sha256` (ключи), `gate_max` (gram),
  `hash_convention` (v2.1-заметка), `script_sha256` (новый скрипт).
- Store v2 (`b7e90c9366d9bbad.json`) сохранён на диске (superseded; провенанс round-2-ревью).

## Основные результаты (незменённые с round 2, из claim)

- **Δh DC-присутствия нет ни в одном пуле/слое** (dh_frac_dc = 0.0; p_fdc_dh_med = 1.0;
  fdc_dh_med 1.07e-5…5.6e-5 — подтверждает v1-самокоррекцию «f_dc(Δh) ≈ 1.5e-5»).
- **Net displacement Δh несёт angular-компонента** (primary L23: contrib_A med 1.0015,
  CI [0.9755, 1.0686]; contrib_R med −0.0015, CI [−0.0686, 0.0245]) — h блуждает вокруг
  опорной точки (когерентный поворот направлений), без дрейфа нормы.
- **Descriptive (dominance): radial-компонента DC-доминантнее** (med f_dc R/A ratio CI
  [138.6, 258.5]; frac_R до 0.93 в loopy L1) — но это DC внутри компоненты, не DC Δh (C1
  round 1: «RADIAL_DC при ‖ΣR‖ маленьком» — именно этот ловушки-сценарий закрыт gate C2).
- Итог T0.2b: **NO_DH_DC** — «плато T0.1» (v_tangent, vt_plateau 0.046–0.112 per слой) не DC
  приращений Δh; территория T0.5/VAR (персистентность направлений), как предсказано
  пре-регом v1 (самокоррекция §1).

## 5. Round 2b (addendum r2b: PASS)

- Addendum к r2: **VERDICT: PASS** (старый FAIL был чтением старой версии файла; все 4 MINOR
  закрыты, 956 численных листьев bit-eq, 202/202 строк лога дословно).
- **NIT-пометка (MINOR-2) — принята без реран:** manifest-note «run_log = stdout (tracked)»
  не буквальна (1 строка field.py-selfcheck — print вне logline(); RUN_RC — строка run_one.sh,
  не скрипта). Данные/вердикты/воспроизводимость не затронуты (ревьюер: «уровень NIT, не
  блокирует»). Фикс в скрипте (текст note) — на следующий артефакт; v2.1 закрывается как есть.
- **Итог AT0_2b: PASS.** Данные: Δh (full − null) не содержит DC-компоненту ни в одном
  пуле/слое (carrier не проходит floor-gate, floor = |Δh_null| ≈ 3e-17); угловая
  составляющая ≈ 100% смещения; L23 — ядро (99.85% / 103.87%); P2 (corpus) ≠ P1 (own);
  L11 — единственная явная анти-корреляция.
