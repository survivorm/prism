#!/usr/bin/env bash
# env_xpu.sh — окружение для XPU-сборки torch (2.9.1+xpu) на Intel GPU.
#
# Две независимые проблемы, две разные причины — не смешивать:
#
# (1) ИМПОРТ ПАДАЕТ: libsycl.so.8: undefined symbol: urEnqueueCooperativeKernelLaunchExp,
#     version LIBUR_LOADER_0.12.
#     Причина: pip кладёт runtime-библиотеки oneAPI в <venv>/lib, а этот каталог НЕ
#     на пути линкера ⇒ грузится СИСТЕМНЫЙ libur_loader старой версии.
#     Фикс: <venv>/lib в начало LD_LIBRARY_PATH.
#
# (2) УСТРОЙСТВО НЕ ВИДНО: torch.xpu.device_count() == 0 ("XPU device count is zero").
#     Причина (другая!): несовместимость РАНТАЙМА и L0-ДРАЙВЕРА. Пип-рантайм
#     2025.2.1 против системного L0-драйвера ⇒ перечисление молча не даёт
#     устройств. Системный oneAPI при этом работает (в llama.cpp-SYCL карта видна).
#     ⇒ нужно уметь выбирать, ЧЕЙ рантайм использовать:
#         XPU_RUNTIME=auto   (по умолчанию) — если системный loader новее пип-овского
#                             (символы LIBUR_LOADER_x.y >= 0.12), берём СИСТЕМНЫЙ;
#                             иначе — пип-овский (<venv>/lib).
#         XPU_RUNTIME=system — только системный (нужен oneAPI >= 2025.2)
#         XPU_RUNTIME=pip    — только пип-овский из venv
#
# Использование:
#   cd <репозиторий>/sycl_prism
#   source probe/env_xpu.sh
#   python -c "import torch; print(torch.__version__, torch.xpu.device_count())"
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
VENV="${XPU_VENV:-$ROOT/.venv-xpu}"
PY="$VENV/bin/python"
MODE="${XPU_RUNTIME:-auto}"

if [ ! -x "$PY" ]; then
    echo "НЕТ venv: $VENV" >&2
    echo "  python3 -m venv $VENV" >&2
    echo "  $PY -m pip install -i https://download.pytorch.org/whl/xpu 'torch==2.9.1+xpu' numpy" >&2
    return 1 2>/dev/null || exit 1
fi

VENV_LIB="$(cd "$(dirname "$PY")/.." && pwd)/lib"
SYS_LIB="/opt/intel/oneapi/lib"

# максимальная версия символов у кандидата
symver() {   # $1 = файл libur_loader
    [ -e "$1" ] || return 1
    strings "$1" 2>/dev/null | grep -oE 'LIBUR_LOADER_[0-9]+\.[0-9]+' \
        | sed 's/LIBUR_LOADER_//' | sort -V | tail -1
}
PIP_VER="$(symver "$VENV_LIB/libur_loader.so.0" || true)"
SYS_VER="$(symver "$SYS_LIB/libur_loader.so.0" 2>/dev/null || true)"

echo "XPU-venv     : $VENV"
echo "venv/lib     : $VENV_LIB"
echo "пип loader   : ${PIP_VER:-—}"
echo "систем loader (симв. версии): ${SYS_VER:-—}"

# ---- удаление пути из LD_LIBRARY_PATH (нужно, а не «не добавлять») ----
strip_ld() {   # $1 = подстрока, которую убрать
    local keep="" p
    local IFS=:
    for p in $LD_LIBRARY_PATH; do
        [ -n "$p" ] || continue
        case "$p" in *"$1"*) continue ;; esac
        keep="${keep:+$keep:}$p"
    done
    LD_LIBRARY_PATH="$keep"
}
has_ld() { case ":${LD_LIBRARY_PATH:-}:" in *"$1"*) return 0 ;; *) return 1 ;; esac; }

# ---- ГДЕ ЖИВЁТ СИСТЕМНЫЙ РАНТАЙМ ----
# ВНИМАНИЕ: не /opt/intel/oneapi/lib (его обычно НЕТ), а
#   /opt/intel/oneapi/compiler/<версия>/lib  (там libsycl.so.9 и libur_loader)
# плюс прочие версионные каталоги из setvars.sh (в LD_LIBRARY_PATH).
# Рекурсивный поиск по всему /opt/intel/oneapi ЗАВИСАЕТ ⇒ только точечные пути.
SYS_LOADER=""
for cand in /opt/intel/oneapi/lib/libur_loader.so.0 \
             /opt/intel/oneapi/compiler/*/lib/libur_loader.so.0 \
             /opt/intel/oneapi/compiler/latest/lib/libur_loader.so.0; do
    if [ -e "$cand" ]; then SYS_LOADER="$cand"; break; fi
done
SYS_LIB="$(dirname "$SYS_LOADER" 2>/dev/null)"
SYS_ONEAPI="$(ls -1 /opt/intel/oneapi/compiler 2>/dev/null | grep -E '^[0-9]{4}\.' | sort -V | tail -1)"
SYS_VER_GE_252=0
if [ -n "$SYS_ONEAPI" ]; then
    if [ "$(printf '%s\n2025.2\n' "$SYS_ONEAPI" | sort -V | tail -1)" = "$SYS_ONEAPI" ]; then
        SYS_VER_GE_252=1
    fi
fi
echo "системный oneAPI : ${SYS_ONEAPI:-не найден}"
echo "системный loader : ${SYS_LOADER:-не найден}"

use_pip=1
case "$MODE" in
    system) use_pip=0 ;;
    pip)    use_pip=1 ;;
    auto)
        # СИСТЕМНЫЙ рантайм приоритетнее, если он новее 2025.2 И ЕСЛИ ЕГО РЕАЛЬНО НАЙДЕНО:
        # у нас 2026.0, и именно пип-овский 2025.2 (в пути) не видит карту и ломает sycl-ls.
        # Если системный loader не найден — остаёмся на пип-овском (иначе импорт сломается).
        if [ "$SYS_VER_GE_252" = "1" ] && [ -n "$SYS_LOADER" ]; then use_pip=0; fi
        ;;
    clean)
        # убрать и пип-овский, и оставить только системный
        use_pip=0 ;;
    *) echo "неизвестный XPU_RUNTIME=$MODE (auto|system|pip|clean)" >&2; return 1 2>/dev/null || exit 1 ;;
esac

if [ "$use_pip" = "1" ]; then
    export LD_LIBRARY_PATH="$VENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "режим рантайма: ПИП-овский (<venv>/lib) — в начало LD_LIBRARY_PATH"
else
    # ВАЖНО: не просто «не добавлять», а УБРАТЬ venv/lib из пути — иначе он
    # продолжает затенять системный loader (именно это ломало sycl-ls).
    strip_ld "$VENV_LIB"
    if [ -n "$SYS_LOADER" ]; then
        has_ld "$SYS_LIB" || export LD_LIBRARY_PATH="$SYS_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "режим рантайма: СИСТЕМНЫЙ oneAPI ($SYS_ONEAPI) — venv/lib УБРАН, $SYS_LIB добавлен"
    else
        # страховка: системный loader не найден ⇒ возвращаем пип-овский, иначе импорт сломается
        export LD_LIBRARY_PATH="$VENV_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "режим рантайма: ПИП-овский (системный loader не найден — снят риск сломать импорт)"
    fi
fi
export XPU_VENV="$VENV" XPU_VENV_LIB="$VENV_LIB" XPU_RUNTIME_CHOSEN="$MODE"

echo
if "$PY" - <<'EOF'
import sys
try:
    import torch
except Exception as e:
    print("ИМПОРТ torch НЕ ПРОШЁЛ:", repr(e))
    sys.exit(1)
print("torch:", torch.__version__)
if not hasattr(torch, "xpu"):
    print("модуля torch.xpu нет ⇒ это не XPU-сборка")
    sys.exit(1)
try:
    n = torch.xpu.device_count()
    print("torch.xpu.device_count():", n)
except Exception as e:
    print("перечисление упало:", repr(e))
    sys.exit(1)
if not n:
    print("УСТРОЙСТВО НЕ ВИДНО ⇒ это слой «рантайм vs L0-драйвер», а не импорт.")
    print("Диагностика по слоям: python3 " + sys.argv[0] if len(sys.argv) else "")
    sys.exit(2)
for i in range(n):
    print("устройство:", torch.xpu.get_device_name(i))
x = torch.ones(1024, 1024, device="xpu")
print("контрольный matmul на XPU:", (x @ x).sum().item())
EOF
then
    echo "OK: XPU-путь работает."
else
    echo "Диагностика по слоям (драйвер / L0 / OpenCL / совместимость):"
    echo "  python3 $HERE/check_xpu_device.py"
    echo "Попробовать другой рантайм:"
    echo "  XPU_RUNTIME=system source $HERE/env_xpu.sh   # убирает venv/lib из пути"
    echo "  XPU_RUNTIME=pip    source $HERE/env_xpu.sh"
    echo "  XPU_RUNTIME=clean  source $HERE/env_xpu.sh   # только системный oneAPI"
fi
