#!/usr/bin/env bash
# run_local.sh — сборка и проверка прибора переноса на машине пользователя.
#
# Порядок (каждый шаг можно запустить отдельно):
#   0. ОКРУЖЕНИЕ      — какой python, какая сборка torch, есть ли torch.xpu
#   1. ТЕСТЫ          — pytest (эталон формул E-45 + самопроверка зонда)
#   1a. СИНТАКС-ГЕЙТ  — компиляция пробы и библиотеки против шима SYCL API
#                      (без icpx и без GPU) + сверка C ABI ↔ ctypes
#   2. СБОРКА SYCL    — icpx -fsycl ⇒ build/libprism_sycl.so  (нужен для C-ABI пути)
#   3. ПАРИТЕТ ПРИБОРА— numpy-эталон vs SYCL (если библиотека собралась)
#   4. БЕНЧМАРКИ      — стоимость прибора и нулевого слоя (numpy-база)
#   5. ЗОНД XPU       — dense-путь E-45 на iGPU + аналитический контроль E-45c
#
# ВАЖНО (сборка torch): модуль `torch.xpu` существует ТОЛЬКО в XPU-сборке.
# Обычная CUDA-сборка (например 2.13.0+cu130) его не содержит — тогда шаг 5
# отработает как CPU-база. Установка XPU-сборки:
#     pip install torch --index-url https://download.pytorch.org/whl/xpu
# Альтернатива, не требующая torch.xpu: собственный SYCL-прибор (шаги 2-4),
# он работает с ЛЮБЫМ torch (C ABI + ctypes).
set -u

cd "$(dirname "$0")"
ROOT="$(pwd)"

# ОКРУЖЕНИЕ: ВСЕГДА source'им sycl_env.sh (идемпотентен: append + дедупликация).
# Не «только если icpx не в PATH»: полувготовое env (icpx в PATH, но без
# LD_PRELOAD) молча пропускает фикс ENV-B (GLIBCXX-тень conda) ⇒ python видит
# 0 устройств и паритет падает с tcache-abort. sycl_env.sh добавляет,
# не перезаписывает ручной env. || exit 1: сбой env — фатален, а не «шаги
# пропущены» (ревью NIT-5).
source "$ROOT/sycl_env.sh" || exit 1
# 1) ОТДЕЛЬНЫЙ venv для зонда (sycl_prism/.venv-xpu) — в нём живёт XPU-сборка torch.
#    Причина: XPU-сборок torch новее 2.9.1 НЕ существует, а в venv репозитория
#    стоит 2.13.0+cu130 ⇒ установка XPU-wheel в venv репозитория = понижение
#    версии torch в окружении серии. Окружение серии не трогаем.
# 2) иначе — venv репозитория (как ../.venv из sycl_prism/).
# Интерпретатор: приоритет — тот, где есть numpy И pytest (иначе шаг «тесты»
# молча пропускается, как это случилось с conda-python без pytest).
# Тот же порядок, что в гейте probe/check_cpp_syntax.sh.
if [ -z "${PY:-}" ]; then
    for cand in ".venv-xpu/bin/python" "../.venv/bin/python" \
                "../../.venv/bin/python" ".venv/bin/python" "python3"; do
        path="$cand"
        case "$cand" in /*) ;; *) path="$ROOT/$cand" ;; esac
        [ -x "$path" ] || command -v "$cand" >/dev/null 2>&1 || continue
        if "$path" -c "import numpy, pytest" >/dev/null 2>&1; then
            PY="$path"; break
        fi
        [ -n "${PY:-}" ] || PY="$path"
    done
fi
PY="${PY:-python3}"

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

say "0. ОКРУЖЕНИЕ"
echo "python: $($PY -V 2>&1)  ($PY)"
"$PY" - <<'PYEOF'
import importlib.util
import sys

print("sys.executable:", sys.executable)
if importlib.util.find_spec("torch") is None:
    print("torch: НЕТ — шаги 4-5 (torch-часть) пропускаются; "
          "прибор на numpy работает и без него")
    raise SystemExit(0)
import torch

v = torch.__version__
print("torch:", v)
if "xpu" in v.lower():
    print("  ⇒ XPU-сборка")
else:
    print("  ⇒ это НЕ XPU-сборка ⇒ модуля torch.xpu в ней нет")
    print("  ⇒ установить XPU-сборку: "
          "pip install torch --index-url https://download.pytorch.org/whl/xpu")
if hasattr(torch, "xpu"):
    try:
        ok = torch.xpu.is_available()
        print("  torch.xpu.is_available():", ok)
        if ok:
            print("  устройство:", torch.xpu.get_device_name(0))
    except Exception as e:
        print("  torch.xpu: ошибка:", repr(e))
try:
    import numpy
    print("numpy:", numpy.__version__)
except Exception as e:
    print("numpy: НЕТ", repr(e))
PYEOF

say "1. ТЕСТЫ (pytest)"
if "$PY" -c "import pytest" 2>/dev/null; then
    "$PY" -m pytest tests/ -q || echo "!!! тесты упали — не идти дальше"
else
    echo "pytest не установлен: pip install pytest"
fi

say "1a. СИНТАКСИЧЕСКИЙ ГЕЙТ (шим SYCL API + сверка C ABI)"
# Ловит ошибки, которые иначе всплывают только на машине с icpx:
# шаблоны внутри extern "C", несуществующие трейты info::*, enum вместо строки
# в get_info<>, расхождение числа аргументов C-прототипа и ctypes-argtypes.
bash probe/check_cpp_syntax.sh || echo "!!! синтаксический гейт красный — сначала починить"

say "1b. МИНИМАЛЬНЫЙ SYCL-ЗОНД (решающий вопрос: видно ли карту моему коду?)"
if command -v icpx >/dev/null 2>&1; then
    # БЕЗ ЭТОЙ СТРОКИ линкер падает с «cannot open output file build/...:
    # No such file or directory» — icpx НЕ создаёт выходной каталог.
    mkdir -p build
    icpx -fsycl -O2 -std=c++20 probe/probe_sycl_min.cpp -o build/probe_sycl_min \
        2>&1 | tail -5 || echo "!!! компиляция не удалась"
    if [ -x build/probe_sycl_min ]; then
        echo "--- запуск (с таймаутом 120 с) ---"
        timeout 120 ./build/probe_sycl_min; rc=$?
        echo "код возврата: $rc"
        if [ $rc -eq 0 ]; then
            echo "ЗОНД ЗЕЛЁНЫЙ: мой SYCL-код видит устройство и меряет скорость."
            echo "Дальше — сборка прибора и паритет (шаги 2-3)."
        elif [ $rc -eq 1 ]; then
            echo "ЗОНД КРАСНЫЙ: устройство НЕ ВИДНО (см. подсказки в выводе)."
            echo "⇒ Дальше идти бессмысленно: сначала починить видимость (см. §ЧТО ОСТАЛОСЬ)."
        else
            echo "ЗОНД УПАЛ (код $rc) — это НЕ «устройство не видно»: см. вывод выше."
            echo "⇒ Если в выводе есть «РЕЗЮМЕ» с GFLOP/s — устройство ВИДНО, упал уже замер."
            echo "⇒ Пришлите последние строки вывода пробы."
        fi
    else
        echo "!!! бинарник не собран"
    fi
else
    echo "icpx не найден — шаг пропущен"
fi

say "1c. ФОРМЫ БАРЬЕРА НА ЭТОМ КОМПИЛЯТОРЕ (какой sycl::group_barrier доступен)"
# Ключевой вопрос, который НЕЛЬЗЯ угадывать: какую форму group_barrier
# предлагает ВАШ SYCL, и какую выбирает наш helper. Барьер БЕЗ fence_space не
# упорядочивает локальную память (на карте это дало нули и порчу кучи) ⇒ форма
# без fence теперь ЗАПРЕЩЕНА в helper'е, и если её нет — библиотека не соберётся.
# Этот шаг компилирует крошечную программу (без запуска на GPU) и печатает формы.
if command -v icpx >/dev/null 2>&1; then
    mkdir -p build
    echo "--- (1c-1) какая форма барьера выбрана на ВАШЕМ компиляторе ---"
    icpx -fsycl -std=c++20 probe/probe_barrier_forms.cpp -o build/probe_barrier_forms 2>&1 | tail -12
    if [ -x build/probe_barrier_forms ]; then
        timeout 60 ./build/probe_barrier_forms || true
    else
        echo "!!! probe_barrier_forms не собрался — см. вывод компилятора выше"
    fi
    echo "--- (1c-2) СПИСОК ПЕРЕГРУЗОК group_barrier (компилятор печатает сам) ---"
    echo "    (этот файл НАМЕРЕННО сломан: нужен список кандидатов от компилятора)"
    icpx -fsycl -std=c++20 probe/probe_barrier_candidates.cpp -o /tmp/_prism_cand 2>&1 | head -40
    rm -f /tmp/_prism_cand
else
    echo "icpx не найден — шаг пропущен"
fi

say "1c-3. ПОИСК ЯДРА ПО ИМЕНИ (решает вопрос «No kernel named»)"
# ВОПРОС, который нельзя угадать: почему рантайм не находит ядро по имени.
# Зонд запускает ТРИ варианта ядра (обычная функция / шаблонная функция /
# явное имя) и печатает, КАКОЙ отработал ⇒ одна гипотеза проверяется за раз.
# Если A (обычная) работает, а B (шаблон) — нет: причина в шаблоне.
# Если работают оба: причина в СБОРКЕ (путь SPIR-V вместо AOT).
if command -v icpx >/dev/null 2>&1; then
    mkdir -p build
    if icpx -fsycl -std=c++20 probe/probe_kernel_name.cpp -o build/probe_kernel_name 2>&1 | tail -8; then
        if [ -x build/probe_kernel_name ]; then
            timeout 180 ./build/probe_kernel_name; echo "код возврата: $?"
        else
            echo "!!! бинарник не собран"
        fi
    else
        echo "!!! probe_kernel_name не собрался — см. вывод компилятора выше"
    fi
else
    echo "icpx не найден — шаг пропущен"
fi

say "1d. МИНИМАЛЬНЫЙ ВОСПРОИЗВОДИТЕЛЬ: локальная память и барьер (на УСТРОЙСТВЕ)"
# Диагностический прибор: 6 шагов механизма (USM, local_accessor+барьер, аксессор
# по значению, две группы, RMW, bitonic sort) ⇒ печатает «ожидаемо / получено».
# На CPU-шиме все шаги проходят (проверено), значит любое расхождение на карте —
# это свойство устройства/драйвера, и сразу видно, WHICH шаг сломан.
if command -v icpx >/dev/null 2>&1; then
    mkdir -p build
    if icpx -fsycl -std=c++20 probe/probe_local_mem.cpp -o build/probe_local_mem 2>&1 | tail -6; then
        if [ -x build/probe_local_mem ]; then
            timeout 120 ./build/probe_local_mem; echo "код возврата: $?"
        else
            echo "!!! бинарник не собран"
        fi
    else
        echo "!!! не удалось собрать probe_local_mem"
    fi
else
    echo "icpx не найден — шаг пропущен"
fi

say "2. СБОРКА SYCL-БИБЛИОТЕКИ"
if [ -f build/CMakeCache.txt ] && grep -q "PRISM_SYCL_STATIC_STDCXX:BOOL=ON" build/CMakeCache.txt; then
    echo "ВНИМАНИЕ: в build/CMakeCache.txt закэширована опция "
    echo "PRISM_SYCL_STATIC_STDCXX=ON ⇒ icpx её не примет (-static-libstdc++ с -fsycl)."
    echo "  Исправляется: rm -rf build   (или переконфигурация ниже с -D...=OFF)"
fi
if [ -f build/CMakeCache.txt ] && grep -qE "PRISM_SYCL_TARGETS:STRING=..+" build/CMakeCache.txt; then
    echo "ВНИМАНИЕ: в build/CMakeCache.txt закэширован непустой PRISM_SYCL_TARGETS"
    echo "  ⇒ -fsycl-targets=... ⇒ рантайм НЕ найдёт ядро по имени. Исправляется:"
    echo "  rm -rf build   (ниже передаётся явный -DPRISM_SYCL_TARGETS=)"
fi
mkdir -p build
if command -v icpx >/dev/null 2>&1; then
    icpx --version | head -2
    if command -v cmake >/dev/null 2>&1; then
        # -DPRISM_SYCL_STATIC_STDCXX=OFF — ЯВНО: иначе закэшированное в
        # build/CMakeCache.txt значение ON переживёт смену умолчания, и icpx
        # упадёт: "invalid argument '-static-libstdc++' not allowed with '-fsycl'".
        # Если у вас уже собран build/ со старой опцией — проще rm -rf build.
        # -DPRISM_SYCL_TARGETS= (ПУСТО) — ЯВНО: при -fsycl-targets=spir64_x86_64
        # рантайм не находит ядро по имени ("No kernel named _ZTS... was found"),
        # потому что ищет по имени ТИПА, а в SPIR-V-модуле имя точки входа —
        # манглированное operator(). Без флага (путь AOT) — работает, это доказано
        # зондом. Явное значение перекрывает закэшированное в build/CMakeCache.txt.
        cmake -B build -DCMAKE_CXX_COMPILER=icpx -DPRISM_SYCL_STATIC_STDCXX=OFF \
              -DPRISM_SYCL_TARGETS= -S . >/dev/null 2>&1 \
            && cmake --build build -j 2>&1 | tail -5
        ls -la build/libprism_sycl.so 2>/dev/null \
            && echo "OK: build/libprism_sycl.so" \
            || echo "!!! сборка не дала библиотеку"
        # ПРОВЕРКА ЗАГРУЗКИ: ловим несовместимость libstdc++ на месте, а не в паритете.
        if [ -f build/libprism_sycl.so ]; then
            if "$PY" -c "import ctypes; ctypes.CDLL('build/libprism_sycl.so')" 2>/tmp/_load.err; then
                echo "OK: библиотека загружается этим python'ом ($($PY -V 2>&1))"
            else
                echo "!!! библиотека НЕ загружается этим python'ом:"
                head -3 /tmp/_load.err
                echo "    ⇒ лечение: грузить под тем же python'ом, которым собрано, либо"
                echo "      положить libstdc++ компилятора в LD_LIBRARY_PATH:"
                echo "      export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.0/lib:\$LD_LIBRARY_PATH"
                echo "      (в системном oneAPI 2026.0 libstdc++ новее, чем в conda-python)."
                echo "      ВНИМАНИЕ: '-static-libstdc++' icpx с '-fsycl' ЗАПРЕЩАЕТ."
            fi
        fi
    else
        echo "cmake не найден"
    fi
else
    echo "icpx не найден (oneAPI не установлен или не в PATH)."
    echo "Без него C-ABI/SYCL-путь не собирается; torch-путь (шаг 5) работает."
    echo "Установка: https://www.intel.com/content/www/us/en/developer/tools/oneapi.html"
fi

say "2b. РЕВИЗИЯ СОБРАННОЙ БИБЛИОТЕКИ (защита от «собралась старая копия»)"
if [ -f build/libprism_sycl.so ]; then
    # PYTHONPATH=py — модуль лежит в py/, а не в корне (было: ModuleNotFoundError
    # и шаг молча печатал traceback — «тихая» поломка проверки ревизии).
    # PYTHONPATH НЕ должен молча перехватывать ошибку: если импорт не удался —
    # это КРАСНЫЙ шаг, а не информация.
    # ВАЖНО: статус берём ДО пайплайна (cmd | tail = статус tail, всегда 0 —
    # ветка else была мертва, ревью NIT-NEW-1). Скрипт по дизайну продолжает
    # работать после сбоев шага (красный маркер «!!!» — сигнал для G1).
    out2b="$(PRISM_SYCL_LIB=build/libprism_sycl.so PYTHONPATH="$PWD/py" "$PY" -c \
        "import prism_sycl; print('собрана ревизия:', prism_sycl.device_info().get('src_rev','?'))" \
        2>&1)"; rc2b=$?
    printf '%s\n' "$out2b" | tail -2
    if [ "$rc2b" -ne 0 ]; then
        echo "!!! не удалось прочитать ревизию собранной библиотеки (шаг 2b)"
    fi
fi

say "3. ПАРИТЕТ ПРИБОРА (numpy-эталон ↔ SYCL)"
if [ -f build/libprism_sycl.so ]; then
    "$PY" py/test_parity.py || echo "!!! паритет не пройден"
else
    echo "нет build/libprism_sycl.so — пропуск (соберите на шаге 2)"
fi

say "4. БЕНЧМАРКИ ПРИБОРА (numpy-база) — ПО УМОЛЧАНИЮ ПРОПУЩЕНЫ"
echo "  CPU-база уже измерена: dcurve mean 104-220 мс, acf 56-96 мс, null-слой ~1-2 с/сид;"
echo "  типовой прогон 1440 клеток ≈ 9-18 мин CPU. Замер нужен только при переносе:"
echo "  ⇒ bash run_local.sh --with-bench   (или WITH_BENCH=1)"
if [ "${WITH_BENCH:-0}" = "1" ] || [ "${1:-}" = "--with-bench" ]; then
    "$PY" py/bench.py 2>/dev/null | tail -12 || echo "bench.py не отработал"
    "$PY" py/bench_null.py --N 383 --D 1024 2>/dev/null | tail -8 \
        || echo "bench_null.py не отработал"
fi

say "5. ЗОНД XPU (torch.xpu) — ПО УМОЛЧАНИЮ ПРОПУЩЕН"
cat <<'EOF'
  Маршрут 0 (torch.xpu) ЗАКРЫТ: device_count == 0 (конфликт рантаймов oneAPI),
  XPU-сборок torch новее 2.9.1 не существует, а собственный SYCL-прибор уже
  закрывает вопрос «видит ли мой код карту» (шаги 1b/2/3) и НЕ зависит от torch.
  Зонд probe/probe_xpu_pipeline.py отвечает на ДРУГОЙ вопрос (перенёс ли dense-путь
  E-45 через torch) и в дефолтном прогоне даёт только: повтор selftest-эталона
  (он уже в pytest) + «torch.xpu недоступен» + CPU-базу dense-пути (самая долгая
  часть прогона, и она уже измерена: base_pre 73.7 с, dense_metrics 125.8 с).
  ⇒ ВКЛЮЧИТЬ ТОЛЬКО ЯВНО:  bash run_local.sh --with-xpu
EOF
if [ "${WITH_XPU:-0}" = "1" ] || [ "${1:-}" = "--with-xpu" ]; then
    echo "запускаю XPU-зонд (явный запрос)"
    # Если есть отдельный venv с XPU-сборкой torch — поднимаем её окружение:
    # pip кладёт runtime-библиотеки oneAPI в <venv>/lib, а этот каталог не на пути
    # линкера ⇒ без этого загрузится СИСТЕМНЫЙ libur_loader (старой версии).
    if [ -x "$ROOT/.venv-xpu/bin/python" ] \
       && "$ROOT/.venv-xpu/bin/python" -c "import numpy" >/dev/null 2>&1; then
        # ВАЖНО: зонд — это PYTHON-файл ⇒ запускать интерпретатором, а НЕ через bash.
        bash -c "set -a; source '$ROOT/probe/env_xpu.sh' >/dev/null 2>&1 || true; \
                 '$ROOT/.venv-xpu/bin/python' '$ROOT/probe/probe_xpu_pipeline.py' \
                      --V 248000 --N 4096 --v-chunk 32768 --n 1 --skip-forward"
    elif [ -x "$ROOT/.venv-xpu/bin/python" ]; then
        echo ".venv-xpu есть, но без numpy — пропускаю (или поставь: "
        echo "  ./.venv-xpu/bin/pip install numpy)"
    else
        echo "нет .venv-xpu — см. инструкцию в конце (source probe/env_xpu.sh)"
    fi
    echo "целевая форма E-45: N=4096 позиций, D=1024, V≈248k словарь"
    echo "(v-chunk ограничивает пик памяти: W сам по себе ≈1 ГБ)"
    "$PY" probe/probe_xpu_pipeline.py --V 248000 --N 4096 --v-chunk 32768 \
        --n 1 --skip-forward
fi

say "ЧТО ОСТАЛОСЬ ОТ ВАС"
cat <<'EOF'
1. ГЛАВНОЕ — шаг 3 (паритет на КАРТЕ): вывод «ИТОГ: ПАРИТЕТ OK» означает, что
   прибор на iGPU даёт те же числа, что и numpy-эталон, в пределах шума fp32.
   Если паритет красный — пришлите строки (vpath/dcurve/acf/G1-G3/null).

2. Шаг 1b (проба) печатает вердикт по стоп-условиям:
       ТАЙЛОВАННОЕ fp32 >= 50 GFLOP/s  ⇒ dense-путь E-45 окупается;
       < 50                              ⇒ не окупается, остаются прибор и T4;
   и отношение fp64/fp32 (на Iris Xe fp64 отсутствует ⇒ fp64-счита на хосте).

3. torch.xpu НЕ НУЖЕН: маршрут 0 закрыт (device_count == 0, XPU-сборок torch
   новее 2.9.1 нет), а собственный SYCL-прибор работает с ЛЮБЫМ torch. Если всё
   же хотите перепроверить маршрут 0 — отдельный venv, окружение серии не трогаем:
       cd sycl_prism
       python3 -m venv .venv-xpu
       ./.venv-xpu/bin/pip install -i https://download.pytorch.org/whl/xpu \
               "torch==2.9.1+xpu" numpy
   и только потом: bash run_local.sh --with-xpu

4. Следующий шаг по существу (не про iGPU): разбивка E-45 по этапам —
       python3 probe/profile_stages.py --script experiments/exp45_damage_vs_structure.py \
              --timeout 900
   ⇒ сколько часов стоит каждый этап и что из этого переносимо, а что — нет.

5. Если iGPU-трансфер закрылся по стоп-условиям, цель «ускорить эксперименты»
   решается T4 (форвард 51.7 с ⇒ x10-x30) и статистической экономией
   (пре-регистрация v4: <=12 доков, burst — отдельный estimand).
EOF
