#!/usr/bin/env bash
# Зонд окружения: что нужно для SYCL на Intel iGPU (Linux).
# Ничего не меняет, только читает. Запуск: bash probe/probe_env.sh
set -u

echo "=== ОС / ядро ==="
uname -a
echo
echo "=== GPU (PCI) ==="
(lspci | grep -iE "vga|display|3d" || echo "lspci нет")
echo
echo "=== /dev/dri ==="
ls -l /dev/dri 2>/dev/null || echo "нет /dev/dri (нужен mesa/intel-media-driver)"
echo
echo "=== oneAPI ==="
for v in ONEAPI_ROOT ONEAPI_DEVICE_SELECTOR SYCL_CACHE_PERSISTENT; do
  echo "$v=${!v:-<не задан>}"
done
if command -v icpx >/dev/null 2>&1; then
  echo "icpx: $(command -v icpx)"
  icpx --version | head -2
else
  echo "icpx: НЕТ (нужен oneAPI Base Toolkit; либо только runtime для sycl-ls)"
fi
if command -v sycl-ls >/dev/null 2>&1; then
  echo "--- sycl-ls ---"
  sycl-ls 2>&1 | head -30
else
  echo "sycl-ls: НЕТ"
fi
echo
echo "=== Level Zero / OpenCL ICD ==="
ldconfig -p 2>/dev/null | grep -iE "libze_loader|libze_intel|libOpenCL" | head -10 || echo "ничего не найдено"
ls -l /etc/OpenCL/vendors 2>/dev/null || echo "нет /etc/OpenCL/vendors"
echo
echo "=== Vulkan ==="
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary 2>/dev/null | head -40
else
  echo "vulkaninfo нет (нужен vulkan-tools)"
fi
echo
echo "=== Python ==="
python3 -c "import sys; print(sys.version)"
python3 - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("  xpu:", hasattr(torch, "xpu"))
except Exception as e:
    print("torch: нет/ошибка:", e)
PY
