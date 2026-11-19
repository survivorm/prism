# Ответ на ревью кода AT2e r2 (docs/AT2e_response.md, sha 2d91e175)

Код r3: `experiments/at2/AT2e.py` (sha256 `f040dbde7ab1...`), TZ r5.2: `docs/AT2e_prereg.md`.
Все пункты ревью сверены построчно с кодом на диске (не с версией в памяти — ревью пришло,
когда r2 уже был на диске и смок r2 уже прогнан). Вердикт ревью: REVISE — принимаю; C1 —
не воспроизводится (evidence ниже).

---

## C1. `assert pool_fp_ok` «вне ветки» — **НЕ ВОСПРОИЗВОДИТСЯ**

В коде на диске (r2, который ревьюировался) assert **уже внутри** `if not SMOKE:`:

```
experiments/at2/AT2e.py:1380:    pool_fp_ok = None
experiments/at2/AT2e.py:1381:    if not SMOKE:
experiments/at2/AT2e.py:1382:        pool_fp_ok = (pool["meta"]["pool_fp"] == jcache["pool_fp"])
experiments/at2e.py:1385:        assert pool_fp_ok, ("S1b FAIL: ...")
```

Подтверждено живым прогоном: смок r2 (тот же код на диске) **прошёл** эту зону и завершился
`AT2E_SMOKE_DONE`, RC=0 (results/AT2e/at2e_smoke_r2.log, 13:11:24; артефакт
results/AT2e/at2e_smoke_r2.json — S12 PASS, все S1–S13 PASS). Если бы assert был вне ветки,
смок упал бы через ~20 с после загрузки модели, как в ревью — он прожил 19 минут и дошёл
до артефакта. Вероятно, ревью читало промежуточную версию файла (в сессии была краткая
промежуточная редакция main-блока). Повторных правок по C1 не делал; контроль в r3 тот же.

## C2. Последний чанк `init_null_weights` — **принято, исправлено, проверено**

Было: `ch = shape[0] if st + per >= shape[0] else per` (последний чанк — полная ось →
shape-mismatch в `copy_`). Стало: `ch = min(per, shape[0] - st)`.

Проверка (dry-run реинита, §6 п.2 ревью): embedding 248320×1024, per=3906, 64 чанка,
последний = 2242 строки; `copy_` прошёл; двойной реинит → `np.array_equal` (детерминизм);
`probe` на null-модели — полный численный путь, q конечен.

## C3. CUDA-путь мёртв — **принято, исправлено**

- Дефолт `DEVICE = "cpu"`; cuda — только явное `AT2E_DEVICE=cuda` (+ assert «CUDA задана,
  но недоступна»). Автоподхват GPU удалён (ловушка, о которой ревью — закрыта).
- 11 мест: `.cpu()` перед `.numpy()` (make_pwin, probe×2, b0_construct, vjp_leaf,
  measure_floor, g_zero×3, nonlin_probe_1c×4, run_stage1b); `device=DEVICE` во всех
  `torch.as_tensor` (probe, g_zero, nonlin_probe_1c); `sp` в probe — на CPU вместе с `lp`
  (device-mismatch снят); input-тензоры `ids`/`mask` — `.to(DEVICE)` (main + run_stage1b).
- `md5_tensor` уже был корректен (.detach().cpu().numpy()).
- **Ограничение:** на этой машине CUDA нет (`torch.cuda.is_available() = False`) — тест §6 п.4
  (`AT2E_DEVICE=cuda` на 1 клетке) здесь не исполним. Числовой путь CPU не меняется
  (все добавления — no-опы на cpu: `.cpu()` на cpu-тензоре, `.to("cpu")`); верификация на
  реальном GPU — T4-canary (§6b, если T4-путь выбран) или cuda-машина.

## C4. `:.4f` на None в pos-spec logline — **принято, исправлено**

Хелпер `_f(x) = "None" if x is None else f"{x:.4f}"` во всех трёх полях logline.
Проверено forced-degenerate тестом (ниже).

## V1. `rec_min` без `random_ref` → KeyError — **принято, исправлено**

- main-цикл и resume-ветка: `rr = rec.get("random_ref") or {}; sib = {"refs": rr.get("per_dir", []), ...}`.
- `pos_spec` с пустым sibling: `n_ref = 0` → `ratio_ref = []` → `p10 = None` → `screen_ok = False`
  (провал, не краш); `rstar = None` (guard на пустые finite_idx).
- `det_main`: c0 без `ascent` → дет-блок пропускается с пометкой `{"skipped": ..., "all_ok": None}`.
- Проверено: run_cell на DeadFwd (все dz=0 → eps0=0 → rec_min) — не краш; `cell_verdict(rec_min)`
  и `global_verdict([rec_min]*16)` = DEAD_WEIGHT (семантика верна: 16 мёртвых клеток).

## V2. rstar-разъезд + eps0_tprime_diag — **принято, исправлено**

- `rstar` — argmax по **конечным** направлениям, индекс ratio берётся через фильтр
  (`sum(1 for j in range(rstar_j) if ref_lams_t16[j] > 0)`) — выравнено; все мёртвы → `None`.
- Оба сайта `eps0_tprime_diag` (pos_spec t=96 и main N4 post-pass): guard `lam_v_t > 0`,
  иначе diag не пишется.
- Проверено: частичная смертность (4 из 16 живых) — `n_ref_finite=4`, `rstar idx 0` —
  индекс совпадает с ratio_ref; полная смертность — все None, screen_ok=False.

## V3. det_main теряется на resume — **принято, исправлено, проверено**

`partial["det_main"] = det_main` перед `save_partial()`; на resume — чтение оттуда
(`det_main = partial.get("det_main")`). **Resume-тест (§6 п.5):** partial с 16 клетками +
`stage_done` + det_main → `--resume`: 16× «skip (resume)», floor из partial (0 пересчёта),
`det_main: skip (resume, stage_done) — all_ok=True`, артефакт содержит det_main (не null),
RC=0, `AT2E_MAIN_DONE`.

## V4. Null не «идентичный» (force_pos_spec=True) — **принято, исправлено**

`run_stage1b`: `force_pos_spec=False` — weak-скрин и 1d на null-клетках запускаются по тем
же условиям, что в trained main (companion + rank-1 K1 ≥ 2/3). Поверхность None-крашей (C4/V2)
на мёртвом null-отклике сокращена: rec_min early-exit до pos_spec. TZ §4 — обновлён.

## V5. Объект линейного гейта ≠ объект вердикта — **решено: гейт на K1, запинено**

- normal: `L1 = |λ^{K1}(a_lo) − λ^{K1}(a_hi)|/λ^{K1}` (раньше — полный λ);
  `L2` — на q_t (общий знаменатель, без окна — не меняется).
- denom_below_floor: `L1_num/L2_num` — на q̄_{K1} (раньше — q̄_fut; companion тоже по K1).
- Полные величины — отчёт: `lambda_full_report`, `qbar_fut_report`.
- Запинено в TZ §3.5 (r3) и в режиме гейта в `verify.linear.mode`.

## V6. Отсутствующие отчёты — **принято, добавлено**

| поле | где |
|---|---|
| `denom_eps_share_tau[3]` | `verify.per_amp[a].denom_eps_share_tau` (a²·τ·ε₀/(q_t + a²·τ·ε₀) per τ) |
| `transformers_version` | header |
| `eps_ascent` | `ascent.eps_ascent` (= 1e-2·eps0) |
| `b0 per_dir` | `b0.per_dir[32]`: {j, label, K1/K2/full: lambda_tau[3]} (из b0_q_hi, 0 forward) |
| `canary` | header: `None` (CPU-run; T4-canary — в T4-обёртке, §6b) |
| `forward_s / vjp_s` | header: 4.33 / 8.41 (провенанс pre-study r1) |

S12 расширен до проверки этих ключей (transformers_version, canary, forward_s,
denom_eps_share_tau, eps_ascent, b0.per_dir[32]) + gates/cells/verdicts.

## Минорные — **все приняты, исправлены**

- Докстринг градиента (шапка + ascent): `g = D·Ov_K1 − 2·q̄_K1·(Cv + eps_reg·v)` — формула
  r1/r2 (потеряны множитель 2 и ε-член) исправлена; код всегда был верен (cos(g, grad) = 1.0).
- `invariants["18_transfer_gap"] = True` (в смоке 1d гоняется — force_1d/S13).
- DEAD_WEIGHT: `n_found_sho == 0` добавлено (приоритет §5: FOUND-ветви первее).
- `header.status = "ok_det_warn"` при `det_main.all_ok == False`.
- floor pre-pass кэшируется в partial (`floor_map`/`floor_global`) — resume не пересчитывает
  32 fwd (проверено в resume-тесте).

## TZ r5.2 (docs/AT2e_prereg.md)

Правки по ревью: ref-семь shared per (док, слой) (§3.6, B4); floor 16 замеров × 2 forward = 32
(§3.9 + смета §6: стадия 1 5670→5686, worst 9894→9910 ≈ 6.85–11.9 ч); linear-гейты на K1
(§3.5, V5); fallback-сетка объявлена нереализованной (правило → NOT_TESTABLE, §3.5); CPU —
основной путь, T4 — тонкая обёртка с явным AT2E_DEVICE=cuda (§6b, M1/C3); canary — только
если T4; смок 1d безусловно (S13, M2); gates-блок/поля схемы (§13, M3/V6); null — идентичный
поиск (§4, V4); DEAD_WEIGHT-guard (§5); коммит-план (§14: код r3, main r3).

## Валидационные тесты (§6 ревью)

1. **Смок** — r2 прогнан до артефакта (RC=0, S1–S13 PASS); **r3 запущен** (at2e_smoke_r3) —
   результат в логе/артефакте.
2. **Dry-run реинита** — PASS (C2; детерминизм; null probe).
3. **Forced-degenerate** — PASS: rec_min + sibling (V1), pos_spec все-мёртвые (None-ратии,
   rstar=None, diag отсутствует, screen_ok=False — C4/V2), частичная смертность (индексы
   выравнены).
4. **AT2E_DEVICE=cuda** — не исполним на этой машине (CUDA отсутствует); ловушка снята
   (дефолт cpu + assert), верификация на GPU — при выборе T4-пути (canary §6b).
5. **--resume** — PASS (V3: det_main не теряется; клетки/стадии не повторяются; floor из
   partial).

## Самонайденные дефекты (само-аудит r3)

1. **UnboundLocalError в floor-кэше** (перенос кэша в partial): `partial` читался до
   объявления — порядок в main. Пойман resume-тестом (RC=1), исправлено: инфраструктура
   чекпойнта поднята ДО floor pre-pass. Повторный resume-тест — PASS.
2. **S12-проверка `canary is not None`** (моё же r3-расширение): `canary` в CPU-ранге
   легитимно None (схема §13: `canary: null`) — чек должен проверять НАЛИЧИЕ ключа, не
   ненулевость. Поймано первым прогоном смока r3 (S12 FAIL, RC=1 — остальные 12 гейтов
   PASS, численно идентичны r2). Исправлено: `"canary" in back["header"]`.
3. **rec_min-гарды (3)** — designed guard-сценарий «мёртвая клетка» (eps0≤0) не был
   покрыт в отчётных блоках: S11 `c_sm["det_md5"]` и S7 `v7["denom_gate_cell_value"]`
   (KeyError), secondary/s1c/cext (main-only) — `lam_k2` при b0q=0, Dden=0 → 0/0 →
   ZeroDivisionError. Исправлено: S11/S7 — presence-guard'ы; secondary/s1c/cext —
   SKIP мёртвой клетки с пометкой в лог.

Провенанс: смок r3 прогоняется на финальном коде (все выше правки ДО старта; два
предшествующих смока r3 — S12-проверка и rec_min-гарды — убиты после старта как
устаревшие по script_sha, AGENTS p.7: правки после старта — только если ветки не
взяты; здесь S11/S7-правки в taken-ветках смока, поэтому перепуск).

## Вердикт

PRINATO: C2, C3, C4, V1–V6, все минорные. C1 — не воспроизводится (evidence: строки кода +
живой смок r2). TZ r5.2 — правлен по всем затронутым пунктам. Код r3 + TZ r5.2 — готовы к
внешнему ревью-лупу после PASS смока r3.

EXEC_DONE
