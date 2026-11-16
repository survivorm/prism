#!/usr/bin/env bash
# sycl_env.sh — ЕДИНСТВЕННЫЙ источник окружения для SYCL-прибора (sycl_prism).
#
# Два неочевидных фикса, без которых карта НЕ видна (диагностика 17.11, см.
# docs/TZ_sycl_workstate.md §2):
#   ENV-A: L0-адаптер oneAPI требует libumf.so.1 — лежит в
#          /opt/intel/oneapi/umf/1.1/lib, по умолчанию в пути НЕТ.
#          (sycl-ls --verbose: "failed to load adapter ... libumf.so.1")
#   ENV-B: python-процесс (conda) тянет в себя СВОЙ libstdc++ (RPATH,
#          GLIBCXX ≤ 3.4.29) РАНЬШЕ системного L0-загрузчика, которому нужны
#          3.4.32/3.4.35 ⇒ "version lookup error" ⇒ 0 устройств.
#          Фикс: LD_PRELOAD системной libstdc++ (6.0.36, ≤ 3.4.36).
#          Старый «фикс» (libstdc++ компилятора в LD_LIBRARY_PATH) НЕ работает:
#          у oneAPI 2026.0 в compiler/*/lib libstdc++ нет, а RPATH сильнее
#          LD_LIBRARY_PATH.
#
# Использование:
#   source sycl_prism/sycl_env.sh
# Идемпотентен: повторный source безопасен (append + дедупликация по soname).
set -u

ONEAPI_BIN="${ONEAPI_BIN:-/opt/intel/oneapi/2026.0/bin}"
UMF_LIB="${UMF_LIB:-/opt/intel/oneapi/umf/1.1/lib}"
SYS_STDCXX="${SYS_STDCXX:-/usr/lib/libstdc++.so.6}"

if [ ! -x "$ONEAPI_BIN/icpx" ]; then
    echo "sycl_env.sh: icpx не найден ($ONEAPI_BIN/icpx) — установите oneAPI 2026.0 или укажите ONEAPI_BIN" >&2
    return 1 2>/dev/null || exit 1
fi

# ---- PATH (icpx, sycl-ls) ----
case ":${PATH}:" in
    *":$ONEAPI_BIN:"*) : ;;
    *) export PATH="$ONEAPI_BIN:$PATH" ;;
esac

# ---- ENV-A: UMF в LD_LIBRARY_PATH ----
case ":${LD_LIBRARY_PATH:-}:" in
    *":$UMF_LIB:"*) : ;;
    *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$UMF_LIB" ;;
esac

# ---- ENV-B: LD_PRELOAD системной libstdc++ (APPEND, не skip) ----
case ":${LD_PRELOAD:-}:" in
    *":$SYS_STDCXX:"*) : ;;
    *)
        if [ -n "${LD_PRELOAD:-}" ]; then
            echo "sycl_env.sh: ВНИМАНИЕ: LD_PRELOAD уже задан: $LD_PRELOAD — добавляю $SYS_STDCXX в начало"
        fi
        export LD_PRELOAD="$SYS_STDCXX${LD_PRELOAD:+:$LD_PRELOAD}"
        ;;
esac

echo "sycl_env.sh: PATH += $ONEAPI_BIN"
echo "sycl_env.sh: LD_LIBRARY_PATH += $UMF_LIB"
echo "sycl_env.sh: LD_PRELOAD = $LD_PRELOAD"

# ---- быстрый чек: видит ли рантайм GPU ----
if command -v sycl-ls >/dev/null 2>&1; then
    n_gpu=$(sycl-ls 2>/dev/null | grep -c "level_zero:gpu\|opencl:gpu" || true)
    if [ "${n_gpu:-0}" -ge 1 ]; then
        echo "sycl_env.sh: OK: GPU устройств видно: $n_gpu"
    else
        echo "sycl_env.sh: !!! GPU НЕ ВИДНО (sycl-ls). Диагностика:" >&2
        echo "  sycl-ls --verbose   # адаптеры и причины" >&2
        echo "  ZE_ENABLE_LOADER_DEBUG_TRACE=1 <prog>   # L0-лоадер" >&2
    fi
fi
