# TZ: sycl_prism — допинг до рабочего состояния

**Дата:** 17.11.2026 · **Код:** SYCL-WORKSTATE · **Цель:** `sycl_prism/` (SYCL-прибор
для iGPU Iris Xe) — от «собирается, но падает» до «полный прогон зелёный, окружение
воспроизводимо».

## 1. Контекст и текущее состояние (проверено на машине 17.11)

`prism/field.py`-операции (vpath, dcurve mean/median, acf_moments, R/A-split closure,
perm-ядра) перенесены на iGPU через C ABI + ctypes. Состояние на момент ТЗ:

| что | статус 17.11 |
|---|---|
| `probe_sycl_min` (проба: карта видна? fp32 GFLOP/s?) | **зелёная** при правильном env: DEVICE_OK, тайлованное fp32 57–69 GFLOP/s (×2 к CPU-базе 28) |
| сборка `libprism_sycl.so` (icpx -fsycl) | **собирается**, грузится python'ом, ревизия v16 читается |
| **паритет** (`py/test_parity.py`) | **КРАСНЫЙ**: `malloc(): unaligned tcache chunk detected` → abort (core dump) |
| `probe_kernel_name` | **SEGV** в main |
| окружение | воспроизводится только «по памяти»: два неочевидных env-фикса, не записанных ни в одном скрипте |

## 2. Найденные дефекты (каждый — с доказательством, не гипотеза)

### ENV-A. L0-адаптер не видит карту без `libumf.so.1` в пути
`sycl-ls --verbose` (чистый env): `failed to load adapter 'libur_adapter_level_zero.so.0'
… libumf.so.1: cannot open shared object file`. UMF лежит в
`/opt/intel/oneapi/umf/1.1/lib` — в PATH/LD_LIBRARY_PATH по умолчанию его нет.
С этим путём: карта видна (2 записи GPU = L0 + OpenCL бэкенды одной карты).

### ENV-B. Python-процесс не видит устройство: GLIBCXX-тень conda
Факт (17.11, воспроизводимо): из C-хоста (dlopen той же библиотеки) `get_devices()` = 2;
из python (ctypes) = **0**. Диагноз по `LD_DEBUG=libs` + `ZE_ENABLE_LOADER_DEBUG_TRACE`:
- цепочка L0 требует GLIBCXX до **3.4.35**: сам загрузчик `/usr/lib/libze_loader.so.1`
  — до **3.4.32**, драйвер `libze_intel_gpu.so.1.17.39758` — до **3.4.35**
  (обе цифры переизмерены 17.11 `objdump -T`);
- conda-python тянет в процесс **свой** `libstdc++.so.6` (RPATH
  `/opt/miniconda3/bin/../lib`, максимум **GLIBCXX_3.4.29**) раньше любой dlopen;
- при dlopen `libze_loader.so.1` динамический линкер ищет версию GLIBCXX в уже
  загруженной libstdc++ ⇒ `error: version lookup error` ⇒ L0-ICD не грузится ⇒
  устройств 0. OpenCL-бэкенд тоже не даёт устройств в этом состоянии.
- Фикс (проверен): `LD_PRELOAD=/usr/lib/libstdc++.so.6` (системная 6.0.36,
  GLIBCXX до 3.4.36) — грузится первой, conda-копия дедуплицируется по soname.
  После фикса python видит оба устройства.
  (3.4.32/3.4.35 > 3.4.29 ⇒ version lookup error — механизм не зависит от выбора цифры.)
- ВНИМАНИЕ: старый «фикс» из PROTOCOL.md («положить libstdc++ компилятора в
  LD_LIBRARY_PATH») не работает: у oneAPI 2026.0 в `compiler/*/lib` libstdc++ НЕТ,
  а в python RPATH имеет приоритет над LD_LIBRARY_PATH.

### CODE-A. Гонка D2H без `wait()` в `fetch_f32_as_f64` (корень краша)
`src/prism_sycl.cpp` (~строка 215):
```cpp
static void fetch_f32_as_f64(sycl::queue& q, const float* dev, std::size_t n, double* out) {
    std::vector<float> tmp(n);
    q.memcpy(tmp.data(), dev, n * sizeof(float));   // АСИНХРОННО, event не дождался
    for (...) out[i] = (double)tmp[i];              // читает, пока DMA летит
}                                                    // tmp уничтожен → DMA пишет в свободный heap
```
Последствие: драйвер завершает DMA уже в переразделённый tcache-чанк glibc
⇒ `malloc(): unaligned tcache chunk detected` ⇒ abort. Затрагивают все операции
через fetch: dcurve (mean+median), acf_moments, perm_row_sq, perm_lags.

### CODE-B. Та же гонка в `vpath_impl` (мусор/NaN в выводе)
`src/prism_sycl.cpp` (~288–290): два `q.memcpy` D2H (V, keep) без `.wait()`,
затем `return PS_OK` — python читает numpy-буферы в полёте. Наблюдение 17.11:
V по числам корректен (перехватывался после завершения), keep — мусор
(269 ноль / 18 единиц / 96 случайных байт при эталонных 383 единицах), в
test_parity — NaN в V.

### CODE-C. `probe/diagnostic probe_kernel_name.cpp` — SEGV
`check(tag, ran, dout, n)` получает **USM-указатель устройства** и дереференсирует
его на хосте (`out[i]`, где out = `dout`) ⇒ SEGV в main (gdb подтвердил). Кроме
того, D2H-копия делается ПОСЛЕ check. Зонд диагностический (эпохи v15, вопрос
«No kernel named» решён путём AOT), но в run_local.sh он в прогоне и падает.

## 3. Изменения (что делаем)

1. **`sycl_prism/sycl_env.sh` (новый)** — единственный источник окружения, ВСЕ переменные
   с явным `export` (дети: python/icpx наследуют):
   - `PATH += /opt/intel/oneapi/2026.0/bin` (icpx, sycl-ls);
   - `LD_LIBRARY_PATH += /opt/intel/oneapi/umf/1.1/lib` (ENV-A);
   - `LD_PRELOAD="/usr/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"` (ENV-B) —
     ДОБАВЛЯЕТСЯ к существующему (не «только если не задан»): чужой LD_PRELOAD
     не молча пропускает фикс ENV-B; повтор — безвреден (дедупликация по soname);
     если LD_PRELOAD уже не пуст — печатает warning с текущим значением;
   - печатает что поставил + быстрый чек `sycl-ls` (число GPU-устройств);
   - идемпотентен (source twice — безопасно); двойной source проверяется смоук-прогоном
     в шаге имплементации (`source sycl_env.sh && source sycl_env.sh` в чистом шелле —
     без ошибок, переменные не задублированы); отдельным гейтом не вынесен.
2. **`run_local.sh`** — в начале: ВСЕГДА `source sycl_env.sh` (идемпотентный; не
   «только если icpx не в PATH» — иначе полувготовое env с icpx в PATH но без
   LD_PRELOAD молча пропускает фикс ENV-B и G1 падает с tcache-abort). Не ломать
   ручной env: sycl_env.sh добавляет, не перезаписывает.
3. **`src/prism_sycl.cpp`** — `.wait()` после каждого D2H-копира, за которым следует
   чтение хостом: (a) `fetch_f32_as_f64`; (b) оба memcpy в `vpath_impl`.
   H2D-копиры не трогаем (после каждого уже `q.wait()`).
4. **`src/prism_sycl.h`** — `PRISM_SYCL_SRC_REV` → `v17-20261117-waitfix`.
5. **`probe/probe_kernel_name.cpp`** — переставить D2H + `wait()` ПЕРЕД `check()`,
   check() сверяет хост-буфер (как задумано в комментарии).
6. **Гигиена:** удалить мусор в корне `sycl_prism/`: `np` (36 MB), `torch`,
   `argparse`, `importlib.util` — PostScript-документы, не код (проверено `file`).
   Новый `sycl_prism/.gitignore`: `build/`, `.venv-xpu/`, `__pycache__/`, `*.pyc`,
   `*.o`, `CMakeFiles/`, `*.so` (.so живут только в build/). Каталог `sycl_prism/`
   пока вне git (`?? sycl_prism/`) — коммит каталога входит в работу (иначе
   .gitignore не начинает действовать).
7. **Артефакт** `results/sycl/sycl_workstate_20261117.json` — создаётся СКРИПТОМ
   (`sycl_prism/py/make_workstate_artifact.py`, закоммичен; принимает путь к логам
   run_one.sh и stdout пробы как аргументы), НЕ вручную: loadavg на момент прогона,
   device_info (имя, driver, max_workgroup, local_mem, fp16, barrier_form, src_rev
   И СТРОКИ `sycl-ls` — реальный бэкенд level_zero/opencl, а не placeholder-маркер
   `unavailable-in-this-sycl` из кода), результаты паритета по всем формам (maxdiff/rel
   на каждую операцию), вердикты шагов run_local.sh, путь лога. Все числа — из
   лога/stdout (парсинг строк); json.dump через sanitize (только float()/int(),
   NaN/inf не допускаются — при парсинге NaN ⇒ ошибка, а не запись).
   Числа артефакта сверяются со stdout прогона строка в строку (G6).
   **Отношение к `protocol/check_artifact.py`:** артефакт bring-up, не транспортный;
   чеker неприменим (требует speedup_vs_cpu/gemm_dtype/reduce_dtype — у bring-up
   по определению нет). Это зафиксировано в артефакте (поле `artifact_kind: bringup`)
   и в этом ТЗ — чтобы будущий автоматический контроль не уперся молча.
   **trackio (AGENTS.md, раздел trackio, правило 7.2):** SYCL-WORKSTATE — bring-up, НЕ
   эксперимент; trackio-запись не создаётся, тайм-метки — из лога run_one.sh
   (RUN_RC/время — в артефакте).

## 4. Гейты приёмки (все обязательны)

| # | гейт | критерий |
|---|---|---|
| G1 | `bash run_local.sh` через `./run_one.sh` | ЧИСТЫЙ шелл (icpx НЕ в PATH, UMF НЕ в LD_LIBRARY_PATH, LD_PRELOAD не задан — т.е. ветка «sycl_env.sh подхватывает всё» реально проверяется): `./run_one.sh sycl_all bash sycl_prism/run_local.sh`. Критерий: шаг 3 `ИТОГ: ПАРИТЕТ OK` на всех трёх формах (384×1024, 96×128, 64×32); в логе НЕТ: строк `код возврата:` с кодом ≠ 0, строк `!!!`, маркеров `Aborted`/`Segmentation fault`/`core dumped` |
| G2 | `PRISM_SYCL_BACKEND=reference python3 py/test_parity.py` | `ИТОГ: ПАРИТЕТ OK` (самопроверка обвязки, без GPU) |
| G3 | `pytest sycl_prism/tests/` | зелёные либо skips с обоснованием (torch-зависимые) |
| G4 | `source sycl_prism/sycl_env.sh && ./sycl_prism/build/probe_sycl_min` | stdout пробы сохраняется в `results/sycl/probe_sycl_min_stdout.txt` (вход для артефакта-скрипта, G6). Строка `ВЕРДИКТ: DEVICE_OK` (любой из двух вариантов — карта видна и замер выполнен; решение по порогу 50 GFLOP/s out-of-scope по §5). ПЕРЕПУСК: если строки `НЕСТАБИЛЬНЫЙ ЗАМЕР`/`ЗАМЕР ПОД НАГРУЗКОЙ`/`BENCH_FAILED` — повтор в тихом окне (loadavg ≤ 4), loadavg всех попыток — в артефакт; гейт засчитывается по DEVICE_OK в любом из прогонов |
| G5 | детерминизм после фикса | test_parity 3 прогона подряд — все зелёные (до фикса: краш) |
| G6 | артефакт + гигиена | (a) JSON валиден, числа совпадают со stdout/логом прогона, `src_rev` = ТОЧНО `v17-20261117-waitfix`, поле `artifact_kind: bringup` присутствует; (b) в корне `sycl_prism/` НЕТ `np`/`torch`/`argparse`/`importlib.util`; (c) существует `sycl_prism/.gitignore`; (d) каталог закоммичен (git status чист по sycl_prism/) |

## 5. Out-of-scope (не в этом ТЗ)

- Бенчмарки dense-пути E-45 на GPU (отдельное ТЗ; сейчас — только паритет).
- torch.xpu (маршрут 0 закрыт по ROUTES.md).
- fp64 на устройстве (аспекта нет — физически невозможно на Iris Xe).
- Маршруты 2–6 из ROUTES.md (OpenCL/L0/MKL/T4/статистика).
- Изменение протокола точности PROTOCOL.md (допуски не трогаем).

## 6. Риски

- **Loadavg во время прогона** (другие сессии, AT0-линия) ⇒ GFLOP/s и время
  будут ниже потенциала: в артефакт пишем loadavg, интерпретация «карта слабая»
  запрещена при loadavg > 4.
- `.wait()` после D2H — сериализация, но данные всё равно потребляются на хосте
  ⇒ выигрыш от асинхронности нулевой; корректность важнее.
- LD_PRELOAD системной libstdc++ действует на ВЕСЬ shell-сеанс (run_local.sh
  ВСЕГДА source'ит sycl_env.sh в начале, см. §3.2 ⇒ все subprocess'ы, включая
  icpx/cmake-сборку). На этой машине безвредно: рантайм компилятора — та же
  системная libstdc++ 6.0.36 (дедупликация по soname); проверено сборкой в этом
  env (шаг 2 зелёный). Для future-пакетов: 3.4.36 ⊃ 3.4.29 ⇒ conda-зависимости
  не сломаются; риск обратный (потребность >3.4.36) на этой системе маловероятен.
