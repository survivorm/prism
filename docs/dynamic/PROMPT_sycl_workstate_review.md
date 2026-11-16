# Промпт: внешнее ревью КОДА/АРТЕФАКТА SYCL-WORKSTATE

Ты — независимый ревьюер. Ревьюируются ИЗМЕНЕНИЯ (код + артефакт), а не ТЗ
(ТЗ уже прошло 3 раунда ревью, PASS).

**ТЗ (контекст):** `/home/survivor/work/prism/docs/TZ_sycl_workstate.md`
**Ревью ТЗ (для справки):** `/home/survivor/work/prism/docs/dynamic/TZ_sycl_workstate_REVIEW.md`

## Что изменено (ревью-цель)

1. `sycl_prism/src/prism_sycl.cpp` — 3 D2H-копира получили `.wait()`:
   `fetch_f32_as_f64` (~строка 215) и два memcpy в конце `vpath_impl` (~строки 288–289).
   Фикс гонок (CODE-A/B из ТЗ).
2. `sycl_prism/src/prism_sycl.h` — `PRISM_SYCL_SRC_REV` → `v17-20261117-waitfix`.
3. `sycl_prism/probe/probe_kernel_name.cpp` — D2H + wait() перенесён ПЕРЕД check(),
   check() теперь сверяет хост-буфер (было: дереф USM-указателя на хосте → SEGV).
4. `sycl_prism/sycl_env.sh` (новый) — окружение: PATH(icpx), LD_LIBRARY_PATH(UMF),
   LD_PRELOAD(системная libstdc++, append + warning при чужом).
5. `sycl_prism/run_local.sh` — в начале ВСЕГДА `source sycl_env.sh`.
6. `sycl_prism/py/make_workstate_artifact.py` (новый) — producer артефакта.
7. Гигиена: удалены `sycl_prism/{np,torch,argparse,importlib.util}` (PostScript-мусор);
   новый `sycl_prism/.gitignore`.

## Артефакт и логи

- Артефакт: `results/sycl/sycl_workstate_20261117.json`
- Лог прогона (G1, чистый шелл через run_one.sh): `/tmp/sycl_all.log`
- Stdout пробы (G4): `results/sycl/probe_sycl_min_stdout.txt`

## Что проверять

1. **Полнота фикса гонок**: полный инвентарь `q.memcpy` в `src/prism_sycl.cpp` —
   нет ли ДРУГОГО D2H-копира без wait (H2D-копиры допустимы без wait, если за ними
   следует q.wait() — проверить, что так и есть).
2. **Корректность wait()**: нет ли избыточных/лишних wait, ломающих что-то;
   нет ли изменения порядка операций (только добавление .wait()).
3. **sycl_env.sh**: set -u безопасность (все переменные с :-(гвардом); append-семантика
   (не перезапись); идемпотентность; warning-ветка; поведение при отсутствующем icpx
   (fail-fast, не тихий фолбэк).
4. **probe_kernel_name**: check() получает ХОСТ-буфер; D2H до check; rc-логика
   (a_ok2 && (b_ok2 || c_ok2)) сохранена; компилируется (проверь, если есть icpx).
5. **make_workstate_artifact.py**: sanitize (None-сентинел vs NaN/inf-ошибка),
   точность regex-ов против РЕАЛЬНЫХ строк лога (сверь несколько чисел артефакта
   с /tmp/sycl_all.log), гейты src_rev/ИТОГ/красные маркеры.
6. **Артефакт**: числа совпадают с логом (выбери минимум 5 чисел по разным операциям
   из разных форм), src_rev, artifact_kind, sycl_ls строки — реальные (L0+OpenCL).
7. **Гейты ТЗ (G1–G6)**: сверь по логам/файлам — все ли выполнены по формулировкам ТЗ.
8. **AGENTS.md**: guard'ы, каскад, json, артефакт≠stdout (артефакт = парсинг лога),
   «случайные мелочи» в новом коде (мёртвые параметры, дубли констант).

## Формат

Файл: `/home/survivor/work/prism/docs/sycl_workstate_REVIEW.md`
- заголовок: дата, sha256 артефакта, список ревьюируемых файлов с их sha256;
- по пунктам 1–8: находки MAJOR/MINOR/NIT с указанием файла/строки;
- `VERDICT: PASS|REVISE|FAIL`;
- финальная строка `REVIEW_DONE`.

НЕ редактируй ничего, кроме ревью-файла.
