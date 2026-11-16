#!/usr/bin/env bash
# install_xpu_torch.sh — поставить torch+xpu так, чтобы РАНТАЙМ ОДИН, и он совпадал
# с L0-драйвером. Это устраняет найденную тень: pip-овский oneAPI 2025.2.1 в
# <venv>/lib затенял системный oneAPI 2026.0 ⇒ карта не перечислялась, а sycl-ls падал.
#
# Маршрут A (по умолчанию, рекомендуется): torch БЕЗ зависимостей oneAPI
#     pip install --no-deps "torch==<вер>+xpu" numpy
#   ⇒ пип не ставит свой runtime ⇒ работает СИСТЕМНЫЙ oneAPI (2026.0) ⇒ он соответствует
#     L0-драйверу. Тени нет.
#   Требование: torch должен быть собран под тот же runtime, что и система.
#   Версию берём ТУ ЖЕ, что в venv репозитория (напр. 2.13.0) — тогда API совпадает с серией.
#
# Маршрут B (запасной): torch С зависимостями oneAPI
#     pip install "torch==<вер>+xpu" numpy
#   ⇒ пип ставит свой runtime (2025.x/2026.x) ⇒ он и должен быть в LD_LIBRARY_PATH.
#   Применять, если маршрут A даёт device_count == 0.
#
# Использование:
#   bash probe/install_xpu_torch.sh --recreate --runtime system
#   bash probe/install_xpu_torch.sh --recreate --runtime pip
#   bash probe/install_xpu_torch.sh --dry-run            # только план, ничего не ставит
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
REPO="${PRISM_REPO:-$ROOT/..}"
VENV="$ROOT/.venv-xpu"
RUNTIME="system"
RECREATE=0
DRY=0
TORCH_VER=""

while [ $# -gt 0 ]; do
    case "$1" in
        --recreate) RECREATE=1 ;;
        --dry-run)  DRY=1 ;;
        --runtime)  RUNTIME="${2:-system}"; shift ;;
        --torch-version) TORCH_VER="${2:-}"; shift ;;
        --venv)     VENV="${2:-}"; shift ;;
        -h|--help)  sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
    esac
    shift
done

PY_VENV="$VENV/bin/python"

# версия torch: та же, что в окружении репозитория (если venv есть)
if [ -z "$TORCH_VER" ]; then
    for cand in "$REPO/.venv/bin/python" "$REPO/../.venv/bin/python"; do
        if [ -x "$cand" ]; then
            TORCH_VER="$("$cand" -c 'import torch;print(torch.__version__.split("+")[0])' 2>/dev/null || true)"
            [ -n "$TORCH_VER" ] && break
        fi
    done
fi
[ -z "$TORCH_VER" ] && TORCH_VER="2.13.0"

SYS_ONEAPI="$(ls -1 /opt/intel/oneapi/compiler 2>/dev/null | grep -E '^[0-9]{4}\.' | sort -V | tail -1)"
SYS_LOADER=""
for cand in /opt/intel/oneapi/lib/libur_loader.so.0 \
             /opt/intel/oneapi/compiler/*/lib/libur_loader.so.0; do
    [ -e "$cand" ] && { SYS_LOADER="$cand"; break; }
done
L0_DRV="$(ls -1 /usr/lib64/libze_intel_gpu.so* /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so* 2>/dev/null | head -1)"

echo "ПЛАН УСТАНОВКИ"
echo "  venv            : $VENV"
echo "  версия torch    : $TORCH_VER (та же, что в окружении репозитория)"
echo "  маршрут рантайма: $RUNTIME"
echo "  системный oneAPI: ${SYS_ONEAPI:-не найден}"
echo "  системный loader: ${SYS_LOADER:-не найден}"
echo "  L0-драйвер      : ${L0_DRV:-не найден}"
echo
case "$RUNTIME" in
    system)
        echo "  ⇒ pip install --no-deps torch==${TORCH_VER}+xpu  (СИСТЕМНЫЙ рантайм, без тени)"
        ;;
    pip)
        echo "  ⇒ pip install torch==${TORCH_VER}+xpu  (пип-овский oneAPI; нужен в LD_LIBRARY_PATH)"
        ;;
    *) echo "неизвестный --runtime=$RUNTIME (system|pip)" >&2; exit 2 ;;
esac

if [ "$DRY" = "1" ]; then
    echo; echo "--dry-run: ничего не ставлю."
    exit 0
fi

if [ "$RECREATE" = "1" ] && [ -d "$VENV" ]; then
    echo; echo "удаляю старый venv: $VENV"
    rm -rf "$VENV"
fi
if [ ! -x "$PY_VENV" ]; then
    echo; echo "создаю venv…"
    python3 -m venv "$VENV" || { echo "не удалось создать venv" >&2; exit 1; }
fi

echo; echo "ставлю torch ${TORCH_VER}+xpu (маршрут $RUNTIME)…"
if [ "$RUNTIME" = "system" ]; then
    "$PY_VENV" -m pip install -q --upgrade pip
    "$PY_VENV" -m pip install -q --no-deps "torch==${TORCH_VER}+xpu" \
        || { echo "не удалось поставить torch ${TORCH_VER}+xpu" >&2; exit 1; }
    "$PY_VENV" -m pip install -q numpy
else
    "$PY_VENV" -m pip install -q "torch==${TORCH_VER}+xpu" numpy \
        || { echo "не удалось поставить torch ${TORCH_VER}+xpu" >&2; exit 1; }
fi
echo "готово."

echo; echo "ПРОВЕРКА"
# маршрут system ⇒ venv/lib НЕ поднимаем над системой; маршрут pip ⇒ поднимаем
if [ "$RUNTIME" = "system" ]; then
    echo "(системный рантайм: venv/lib не добавляем; при необходимости — source /opt/intel/oneapi/setvars.sh)"
    export LD_LIBRARY_PATH="$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v "$VENV/lib" | paste -sd:)"
    if [ -n "${SYS_LOADER:-}" ]; then
        export LD_LIBRARY_PATH="$(dirname "$SYS_LOADER")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
else
    export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

"$PY_VENV" - <<'EOF'
import sys
try:
    import torch
except Exception as e:
    print("ИМПОРТ НЕ ПРОШЁЛ:", repr(e))
    sys.exit(1)
print("torch:", torch.__version__)
n = torch.xpu.device_count()
print("torch.xpu.device_count():", n)
if n:
    for i in range(n):
        print("устройство:", torch.xpu.get_device_name(i))
    x = torch.ones(1024, 1024, device="xpu")
    print("контрольный matmul на XPU:", (x @ x).sum().item())
    print("ГОТОВО: карта видна под torch.")
else:
    print("КАРТА НЕ ВИДНА. Следующий шаг:")
    print("  1) python3 probe/check_xpu_device.py   (слои + UR-трейс)")
    print("  2) bash probe/install_xpu_torch.sh --recreate --runtime pip")
    print("     (маршрут B: пип-овский oneAPI, согласованный с torch)")
EOF
