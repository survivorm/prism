#!/usr/bin/env bash
# Проверка C++-части прибора БЕЗ icpx и БЕЗ GPU.
#
# Четыре слоя (каждый ловит свой класс ошибок):
#   1. СИНТАКСИС  — компиляция пробы и библиотеки против шима SYCL API (g++ -fsyntax-only).
#                   Ловит: шаблоны внутри extern "C", несуществующие трейты info::*,
#                   enum вместо строки в get_info<>, опечатки в типах.
#   2. СОВМЕСТИМОСТЬ ЗАГОЛОВКА — src/prism_sycl.h собирается C-компилятором (C11).
#   3. СБОРКА + СИМВОЛЫ — библиотека собирается против ИСПОЛНЯЕМОГО CPU-шима
#                   (probe/sycl_cpu) и все функции заголовка присутствуют в .so
#                   как НЕМАНГЛИРОВАННЫЕ символы. Ловит: забытый extern "C"
#                   (символ не экспортируется ⇒ падение при загрузке на чужой машине).
#   4. ИСПОЛНЕНИЕ — паритет ядер с numpy-эталоном (py/test_parity.py) на CPU-шиме.
#                   Ловит: ошибки АРИФМЕТИКИ и семантики ядер — без GPU.
#   + сверка C ABI ↔ ctypes (py/check_abi.py): число аргументов прототипа и argtypes.
#
# ЧЕГО ГЕЙТ НЕ ДАЁТ: семантику планировщика GPU, раскладку памяти устройства,
# реальную скорость и поведение icpx. Настоящая проверка GPU — сборка icpx +
# тот же паритет на карте (run_local.sh, шаги 2-3).
#
# Использование:  bash probe/check_cpp_syntax.sh
set -u
cd "$(dirname "$0")/.." || exit 2
SHIM="probe/sycl_shim"
CPU_SHIM="probe/sycl_cpu"
CXX=${CXX:-g++}
FLAGS="-std=c++20 -fsyntax-only -I${SHIM}"
rc=0

# Python для паритета: нужен тот, где есть numpy. Берём тот же порядок, что и
# run_local.sh: venv репозитория/проекта → системный python3.
if [ -z "${PY:-}" ]; then
    for cand in ".venv-xpu/bin/python" "../.venv/bin/python" "../../.venv/bin/python" \
                ".venv/bin/python" "python3"; do
        if [ -x "$cand" ] || command -v "$cand" >/dev/null 2>&1; then
            if "$cand" -c "import numpy" >/dev/null 2>&1; then PY="$cand"; break; fi
            [ -n "${PY:-}" ] || PY="$cand"
        fi
    done
fi
PY=${PY:-python3}

say() { printf '\n\033[1m--- %s ---\033[0m\n' "$*"; }

say "0. КАКАЯ ВЕРСИЯ (штамп ревизии + sha256 исходников)"
# Штамп и хэши: по выводу видно, что собрано ИМЕННО ЭТО, а не старая копия.
grep -h "define PRISM_SYCL_SRC_REV" src/prism_sycl.h || true
grep -h "define PROBE_REV" probe/probe_sycl_min.cpp || true
if command -v sha256sum >/dev/null 2>&1; then
    sha256sum src/prism_sycl.cpp src/prism_sycl_compat.h probe/probe_sycl_min.cpp
fi

say "1. СИНТАКСИС (шим SYCL API, без icpx)"
for f in probe/probe_sycl_min.cpp src/prism_sycl.cpp \
         probe/probe_barrier_forms.cpp \
         probe/probe_local_mem.cpp probe/probe_kernel_name.cpp \
         probe/check_barrier_probe_consistency.cpp; do
  if [ ! -f "$f" ]; then
    echo "НЕТ ФАЙЛА: $f"; rc=1; continue
  fi
  out=$($CXX $FLAGS "$f" 2>&1)
  if [ -n "$out" ]; then
    echo "=== ОШИБКИ в $f ==="; echo "$out" | head -20; rc=1
  else
    echo "OK: $f"
  fi
done
warn=$($CXX $FLAGS -Wall -Wextra src/prism_sycl.cpp 2>&1 | grep -c "warning:")
echo "предупреждений в библиотеке (-Wall -Wextra): $warn"

say "2. СОВМЕСТИМОСТЬ ЗАГОЛОВКА (C11)"
if command -v gcc >/dev/null 2>&1; then
  printf '#include "prism_sycl.h"\nint main(void){return prism_sycl_abi_version();}\n' > /tmp/_abi_c.c
  if gcc -std=c11 -I src -fsyntax-only /tmp/_abi_c.c 2>/tmp/_abi_c.err; then
    echo "OK: src/prism_sycl.h собирается C-компилятором"
  else
    echo "=== ОШИБКИ C-совместимости заголовка ==="; head -10 /tmp/_abi_c.err; rc=1
  fi
fi

say "3. СБОРКА ПРОТИВ ИСПОЛНЯЕМОГО CPU-ШИМА + ПРОВЕРКА ЭКСПОРТИРУЕМЫХ СИМВОЛОВ"
mkdir -p build
# Статическая линковка libstdc++/libgcc — обязательна: иначе .so требует
# GLIBCXX новее, чем у libstdc++ из состава python'а (conda-python на машине
# пользователя) ⇒ падает при загрузке с "version `GLIBCXX_3.4.31' not found".
STATIC_LINK="-static-libstdc++ -static-libgcc"
if ! $CXX -std=c++20 -O2 -fPIC -shared $STATIC_LINK -I"${CPU_SHIM}" src/prism_sycl.cpp \
       -o build/libprism_sycl_cpu.so 2>/tmp/_cpu_shim.err; then
    echo "(статическая линковка не удалась — пробуем обычную)"
    $CXX -std=c++20 -O2 -fPIC -shared -I"${CPU_SHIM}" src/prism_sycl.cpp \
       -o build/libprism_sycl_cpu.so 2>>/tmp/_cpu_shim.err
fi
if [ -f build/libprism_sycl_cpu.so ]; then
  echo "OK: build/libprism_sycl_cpu.so собран (исполняемый CPU-шим)"
  # ПРОВЕРКА ЗАГРУЗКИ: ловим несовместимость libstdc++ на месте, а не в паритете.
  if "$PY" -c "import ctypes,sys; ctypes.CDLL('build/libprism_sycl_cpu.so')" 2>/tmp/_load.err; then
    echo "OK: библиотека загружается этим python'ом ($($PY -V 2>&1))"
  else
    echo "!!! библиотека НЕ загружается этим python'ом:"
    head -3 /tmp/_load.err
    echo "    ⇒ либо соберите со статической линковкой (-static-libstdc++),"
    echo "      либо запускайте паритет тем же python'ом, которым собрано."
    rc=1
  fi
  if command -v nm >/dev/null 2>&1; then
    # Имена — только из КОДА заголовка (комментарии упоминают «prism_sycl_compat.h»
    # ⇒ без фильтра это ложное «нет символа»).
    declared=$("$PY" - <<'PYEOF'
import re, pathlib
h = pathlib.Path("src/prism_sycl.h").read_text(encoding="utf-8")
h = re.sub(r"/\*.*?\*/", "", h, flags=re.S)
h = re.sub(r"//[^\n]*", "", h)
print("\n".join(sorted(set(re.findall(r"prism_sycl_[a-z0-9_]+", h)))))
PYEOF
)
    exported=$(nm -D --defined-only build/libprism_sycl_cpu.so 2>/dev/null \
               | awk '$2=="T"{print $3}' | sort -u)
    missing=""
    for s in $declared; do
      echo "$exported" | grep -qx "$s" || missing="$missing $s"
    done
    if [ -n "$missing" ]; then
      echo "!!! НЕ ЭКСПОРТИРУЮТСЯ (падут при загрузке на чужой машине):$missing"
      rc=1
    else
      echo "OK: все $(echo "$declared" | wc -w) функции заголовка экспортируются без манглинга"
    fi
  else
    echo "nm не найден — проверка символов пропущена"
  fi
else
  echo "=== ОШИБКИ СБОРКИ ПРОТИВ CPU-ШИМА ==="; head -20 /tmp/_cpu_shim.err; rc=1
fi

say "4. ИСПОЛНЕНИЕ: паритет ядер с numpy-эталоном (без GPU)"
echo "python для паритета: $PY ($($PY -V 2>&1))"
if ! "$PY" -c "import numpy" >/dev/null 2>&1; then
    echo "!!! у этого python'а нет numpy — паритет не выполним."
    echo "    ⇒ укажите python с numpy:  PY=/path/to/python bash probe/check_cpp_syntax.sh"
    echo "    (например venv репозитория: ../.venv/bin/python)"
    rc=1
elif [ -f build/libprism_sycl_cpu.so ]; then
  PRISM_SYCL_LIB=build/libprism_sycl_cpu.so "$PY" py/test_parity.py > /tmp/_parity.log 2>&1
  tail -3 /tmp/_parity.log
  if grep -q "ПАРИТЕТ OK" /tmp/_parity.log; then
    echo "OK: ПАРИТЕТ OK"
  else
    echo "!!! паритет не пройден (см. /tmp/_parity.log)"
    rc=1
  fi
else
  echo "нет build/libprism_sycl_cpu.so — пропуск"
fi

say "4a. СОГЛАСОВАННОСТЬ SFINAE-ПРОБ БАРЬЕРА С ШИМОМ (ловит опечатки в именах)"
# КЛАСС ДЕФЕКТА (16.11.2026): проба формы C-value обращалась к
# group_type_of<It>::fence_scope, а член называется fence_space ⇒ SFINAE честно
# отвечал «формы нет», helper выбирал NONE, библиотека не собиралась
# ("static_assert(sizeof(It) == 0)"). Ни один изолированный тест это не видел:
# рядом лежала дословно та же проба с правильным именем — и она отвечала true.
#
# Ловим ТАК: шим ПРЕДОСТАВЛЯЕТ формы DPC++ 2026 ⇒ проба обязана их находить.
# «Шим не слабее реальности»: если шим даёт форму, а проба её не находит —
# промах в имени (или расхождение шима с DPC++), и это ловится БЕЗ карты.
if ! $CXX -std=c++20 -O1 -I"${CPU_SHIM}" probe/check_barrier_probe_consistency.cpp \
        -o build/check_barrier_probe_consistency 2>/tmp/_probe_consistency.err; then
    echo "=== ОШИБКИ СБОРКИ GUARD'А ПРОБ ==="; head -10 /tmp/_probe_consistency.err; rc=1
elif ! ./build/check_barrier_probe_consistency > /tmp/_probe_consistency.log 2>&1; then
    cat /tmp/_probe_consistency.log
    echo "!!! пробы барьера не согласованы с шимом (см. выше)"
    rc=1
else
    head -2 /tmp/_probe_consistency.log
    echo "OK: пробы барьера согласованы с шимом (D и C-value найдены, C-type отсутствует, ловушка «без fence» есть)"
fi

say "5. СВЕРКА C ABI ↔ CTYPES"
# (и заодно: python-файлы не должны запускаться через shell — баг был)
if "$PY" py/check_abi.py > /tmp/_abi.log 2>&1; then
  echo "OK: $(tail -1 /tmp/_abi.log)"
else
  echo "=== РАСХОЖДЕНИЯ ABI ==="; cat /tmp/_abi.log; rc=1
fi

say "6. СКРИПТЫ: НИКАКОЙ .py НЕ ЗАПУСКАЕТСЯ ЧЕРЕЗ SHELL"
# ГИГИЕНА DEVICE-КОДА: анонимное пространство / namespace-scope static
# = внутренняя связность = "No kernel named ... was found" в рантайме.
# Этот класс ошибок CPU-шим НЕ ВИДИТ ⇒ ловим статически.
if grep -nE "(^|[^:[:alnum:]_])namespace[[:space:]]*\{" src/prism_sycl.cpp src/prism_sycl_compat.h probe/probe_sycl_min.cpp; then
    echo "!!! анонимное пространство в исходнике SYCL: ядро не найдётся по имени (внутренняя связность)"; rc=1
else
    echo "OK: анонимных namespace в исходниках SYCL нет"
fi
# ПОРЯДОК ВАРИАНТОВ БАРЬЕРА: вариант С FENCE должен проверяться ПЕРВЫМ.
# Форма group_barrier(group) без fence НЕ упорядочивает локальную память ⇒
# на карте это дало нули в результате и порчу кучи. Шим теперь содержит обе
# перегрузки и ловит выбор runtime-ловушкой, но порядок лучше проверять и
# статически.
if "$PY" - <<'PYEOF'
import re, sys, pathlib
txt = pathlib.Path("src/prism_sycl_compat.h").read_text(encoding="utf-8")
m = re.search(r"inline void barrier\(It& it\)\s*\{(.*?)\n\}", txt, re.S)
if not m:
    sys.exit(1)
body = m.group(1)
# ПРАВИЛО: используется форма С ЯВНОЙ областью видимости; форма БЕЗ fence — НИКОГДА.
fenced_names = ["can_barrier_memory_scope", "can_barrier_scope_value",
                "can_barrier_group_fence", "can_barrier_access_fence"]
nofence = body.find("can_barrier_group_only")
fenced = min([body.find(n) for n in fenced_names if body.find(n) >= 0] or [-1])
# Плюс: SFINAE-пробы обязаны брать группу с НЕ-CONST lvalue — в DPC++ 2026
# get_group() не-const, и проба на const& отвергала ВСЕ формы (16.11.2026).
# комментарии вырезаем: сам файл ДОКУМЕНТИРУЕТ этот прокол (значит, строка
# "declval<const It&>().get_group()" в комментарии — это не ошибка кода)
code = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
code = re.sub(r"//[^\n]*", "", code)
probe_const = "declval<const It&>().get_group()" in code
sys.exit(0 if (fenced >= 0 and nofence < 0 and not probe_const) else 1)
PYEOF
then
    echo "OK: барьер использует форму С ЯВНОЙ областью видимости (без fence — запрещена); SFINAE-пробы на не-CONST lvalue"
else
    echo "!!! барьер: нет формы с fence, либо проба на const& (get_group() в DPC++ 2026 не-const)"; rc=1
fi
# ХОСТОВЫЕ УКАЗАТЕЛИ В ЯДРАХ — класс, который CPU-шим НЕ ВИДИТ (один адрес.
# пространство), а реальная GPU ловит: ядро читает host-память ⇒ мусор ⇒ OOB ⇒
# порча кучи ("malloc(): unaligned tcache chunk detected").
if "$PY" probe/check_device_pointers.py; then
    :
else
    rc=1
fi
# ЦЕЛЬ SYCL по умолчанию должна быть ПУСТОЙ: при -fsycl-targets=spir64_x86_64
# рантайм не находит ядро по имени (ищет по имени ТИПА _ZTS..., а в SPIR-V-модуле
# имя точки входа — манглированное operator() _ZZ...). Без флага (путь AOT) —
# работает; это доказано зондом, который исполняется на карте.
if "$PY" - <<'PYEOF'
import re, sys, pathlib
txt = pathlib.Path("CMakeLists.txt").read_text(encoding="utf-8")
m = re.search(r'set\(PRISM_SYCL_TARGETS\s+("(?:[^"]*)")', txt)
sys.exit(0 if (m and m.group(1) == '""') else 1)
PYEOF
then
    echo "OK: PRISM_SYCL_TARGETS по умолчанию пуст (доказанный путь AOT)"
else
    echo "!!! PRISM_SYCL_TARGETS по умолчанию непустой ⇒ ядра не найдутся по имени"; rc=1
fi
# ВНИМАНИЕ: namespace-scope static-функции РАЗРЕШЕНЫ — проверено фактом:
# ядра пробы определены внутри static-функций и успешно исполняются на карте.
# Запрещено только АНОНИМНОЕ пространство (см. выше): именно оно ломает
# поиск ядра по имени ("No kernel named ... was found").
# CMake-опция статической линковки по умолчанию OFF: icpx запрещает
# '-static-libstdc++' вместе с '-fsycl' (проверено на машине).
if grep -A3 "option(PRISM_SYCL_STATIC_STDCXX" CMakeLists.txt | grep -q "OFF)"; then
    echo "OK: PRISM_SYCL_STATIC_STDCXX по умолчанию OFF (icpx не примет -static-libstdc++ с -fsycl)"
else
    echo "!!! PRISM_SYCL_STATIC_STDCXX по умолчанию не OFF — сборка icpx упадёт на линковке"; rc=1
fi
if grep -nE "^[^#]*\b(bash|sh|zsh)\b[^|;&]*\.py" run_local.sh > /tmp/_shellpy.txt 2>/dev/null; then
    echo "!!! python-файл запускается через shell:"; cat /tmp/_shellpy.txt; rc=1
else
    echo "OK: .py запускаются только интерпретатором"
fi

[ $rc -eq 0 ] && echo "=== ВСЁ ЗЕЛЁНОЕ (синтаксис, символы, исполнение, ABI, скрипты) ===" \
              || echo "=== ЕСТЬ ОШИБКИ ==="
exit $rc
