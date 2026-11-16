cmake_minimum_required(VERSION 3.24)
project(prism_sycl LANGUAGES CXX)

# Сборка ТОЛЬКО icpx (oneAPI). Обычная сборка:
#   source /opt/intel/oneapi/setvars.sh
#   CXX=icpx cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
# Итог: build/libprism_sycl.so

if(NOT CMAKE_CXX_COMPILER MATCHES "icpx")
  message(WARNING "Ожидался компилятор icpx (oneAPI), а стоит '${CMAKE_CXX_COMPILER}'.")
  message(WARNING "Соберите так: CXX=icpx cmake -B build ...")
endif()

# C++20 — та же конфигурация, что проверялась гейтом и исполняемым CPU-шимом.
# Принцип: конфигурация, которую проверяли, должна быть той, что собирается.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Цели SYCL.
#
# ВАЖНО (эмпирически, 16.11.2026): целевой триплет по умолчанию ПУСТОЙ, т.е.
# флаг -fsycl-targets НЕ передаётся. Причина: при -fsycl-targets=spir64_x86_64
# (SPIR-V-путь) рантайм не находит ядро по имени:
#     "No kernel named _ZTSZZN17prism_sycl_detail10vpath_implIfEE... was found"
# — он ищет по имени ТИПА замыкания (_ZTS...), а в SPIR-V-модуле имя точки входа
# — манглированное имя operator() (_ZZ...) ⇒ не совпадают.
# Тот же самый код БЕЗ -fsycl-targets (путь AOT/нативный) работает: это доказано
# зондом probe_sycl_min, который на карте исполняется и печатает GFLOP/s.
# ⇒ Умолчание = ровно то, что доказанно работает. Не «на всякий случай spir64».
set(PRISM_SYCL_TARGETS "" CACHE STRING
    "SYCL device targets; пусто = не передавать -fsycl-targets (проверенный путь)")

add_library(prism_sycl SHARED src/prism_sycl.cpp)

target_compile_options(prism_sycl PRIVATE
  -fsycl
  -O3
  -fno-math-errno
)
if(PRISM_SYCL_TARGETS)
  target_compile_options(prism_sycl PRIVATE -fsycl-targets=${PRISM_SYCL_TARGETS})
endif()

target_link_options(prism_sycl PRIVATE -fsycl)

# Статическая линковка libstdc++/libgcc.
#
# ВАЖНО: icpx ЗАПРЕЩАЕТ '-static-libstdc++' вместе с '-fsycl'
#   ("icpx: error: invalid argument '-static-libstdc++' not allowed with '-fsycl'")
# ⇒ для реальной сборки опция по умолчанию ВЫКЛЮЧЕНА.
# Ловушка GLIBCXX (conda-python старше системного g++) лечится не флагом, а
# окружением: библиотека грузится тем же python'ом, что и собиралась, либо
# в LD_LIBRARY_PATH стоит libstdc++ компилятора (oneAPI setvars.sh это делает).
# Для СБОРКИ ПРОТИВ CPU-шима (обычный g++) статическая линковка нужна и включена
# в probe/check_cpp_syntax.sh.
option(PRISM_SYCL_STATIC_STDCXX
       "Слинковать libstdc++/libgcc статически (icpx с -fsycl это запрещает)" OFF)
if(PRISM_SYCL_STATIC_STDCXX)
  target_link_options(prism_sycl PRIVATE -static-libstdc++ -static-libgcc)
endif()

# Целевой триплет сообщаем и линковщику тоже — иначе icpx ругается
# "linked binaries do not contain expected 'spir64-unknown-unknown' target".
# Передаём ТОЛЬКО если он задан (по умолчанию — нет: см. комментарий выше).
if(PRISM_SYCL_TARGETS)
  target_link_options(prism_sycl PRIVATE -fsycl-targets=${PRISM_SYCL_TARGETS})
endif()

# Диагностика: печатать список устройств при первом probe.
option(PRISM_SYCL_DEBUG "Print device list on probe" OFF)
if(PRISM_SYCL_DEBUG)
  target_compile_definitions(prism_sycl PRIVATE PRISM_SYCL_DEBUG=1)
endif()

install(TARGETS prism_sycl LIBRARY DESTINATION lib)
