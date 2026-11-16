"""Диагностика уровня УСТРОЙСТВА для XPU: почему `torch.xpu.device_count() == 0`.

Импорт torch уже работает (конфликт загрузчика решён через `probe/env_xpu.sh`).
Дальше падает другое: рантайм не видит GPU. Тут четыре независимые причины, и
различать их надо по слоям, а не угадывать:

  слой 1  ДРАЙВЕР      — i915/xe в ядре, /dev/dri/renderD*, libze_intel_gpu.so (Level Zero)
  слой 2  L0-ЗАГРУЗЧИК — libze_loader.so.1 (часть драйвера) + адаптер UR libur_adapter_level_zero.so
  слой 3  OpenCL-ICD   — /etc/OpenCL/vendors/*.icd + libur_adapter_opencl.so (запасной путь)
  слой 4  ЗАТЕНЕНИЕ    — В КОМПЛЕКТЕ с «рантайм vs L0-драйвер»: если в LD_LIBRARY_PATH
                         лежит <venv>/lib с runtime oneAPI 2025.2.1, он ПЕРЕКРЫВАЕТ
                         системный (2026.0) ⇒ и sycl-ls падает, и устройство не видно

Скрипт работает и когда torch не импортируется: часть проверок — чистые
файлы/подпроцессы, torch трогается последним и в try/except.

ВАЖНО: скрипт сам перезапускается под python'ом XPU-venv, если запущен не он
(иначе он смотрел бы в чужое окружение — conda-python — и показывал бы чужие данные).

Запуск:
    cd <репозиторий>/sycl_prism
    source probe/env_xpu.sh
    python3 probe/check_xpu_device.py
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys


def ensure_venv_python() -> None:
    """Перезапуск под python'ом XPU-venv, если запущен не он."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    venv_py = (os.environ.get("XPU_VENV_PY")
               or os.path.join(root, ".venv-xpu", "bin", "python"))
    if not os.path.exists(venv_py):
        return
    try:
        same = os.path.realpath(sys.executable) == os.path.realpath(venv_py)
    except OSError:
        same = False
    if not same and not os.environ.get("XPU_ENV_REEXEC"):
        os.environ["XPU_ENV_REEXEC"] = "1"
        os.execv(venv_py, [venv_py, os.path.abspath(__file__), *sys.argv[1:]])


def run(cmd: str, timeout: int = 90) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return ((r.stdout or "") + (r.stderr or "")).strip()
    except (OSError, subprocess.SubprocessError) as e:
        return f"(ошибка запуска: {e})"


def hdr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sym_versions(path: str) -> list[str]:
    out = run(f"strings '{path}' | grep -oE 'LIBUR_LOADER_[0-9.]+' | sort -u")
    return [x for x in out.split("\n") if x.startswith("LIBUR_LOADER_")]


def bounded_find(root: str, pattern: str, maxdepth: int = 4,
                 timeout: int = 30) -> list[str]:
    """find с ограничением глубины и времени (рекурсивный glob здесь зависает)."""
    out = run(f"timeout {timeout} find -P {root!r} -maxdepth {maxdepth} "
              f"-name {pattern!r} 2>/dev/null", timeout=timeout + 10)
    return [x for x in out.split("\n") if x.strip() and x.startswith("/")]


def venv_lib_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")


def main() -> int:
    import time
    t_start = time.time()
    ensure_venv_python()
    site = sys.prefix
    vlib = venv_lib_dir()
    print("python    :", sys.executable)
    print("sys.prefix:", site)
    print("venv/lib  :", vlib)
    print("LD_LIBRARY_PATH:")
    for p in (os.environ.get("LD_LIBRARY_PATH") or "").split(":"):
        if p:
            print("   ", p)

    # ── слой 1: драйвер ядра ──
    hdr("СЛОЙ 1 — драйвер ядра и устройство")
    print("модуль i915:", run("ls /sys/module/i915 >/dev/null 2>&1 && echo ДА || echo НЕТ"))
    print("модуль xe   :", run("ls /sys/module/xe >/dev/null 2>&1 && echo ДА || echo НЕТ"))
    print("устройства DRM:", run("ls -l /dev/dri/ 2>&1 | head -8"))
    print("PCI (vga/display):",
          run("lspci -nn 2>/dev/null | grep -iE 'vga|display|3d' | head -5"))
    print("intel_gpu_top:", "да" if shutil.which("intel_gpu_top") else "нет")

    # ── слой 2: Level Zero ──
    hdr("СЛОЙ 2 — Level Zero (драйвер + адаптер UR)")
    l0_loader = (glob.glob("/usr/lib64/libze_loader.so*")
                 + glob.glob("/usr/lib/x86_64-linux-gnu/libze_loader.so*")
                 + glob.glob("/opt/intel/oneapi/lib/libze_loader.so*"))
    l0_gpu = (glob.glob("/usr/lib64/libze_intel_gpu.so*")
              + glob.glob("/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so*")
              + glob.glob("/opt/intel/oneapi/lib/libze_intel_gpu.so*"))
    print("libze_loader.so (L0-загрузчик):", l0_loader or "НЕ НАЙДЕН")
    print("libze_intel_gpu.so (L0-драйвер):", l0_gpu or "НЕ НАЙДЕН")
    ur_l0 = (glob.glob(os.path.join(vlib, "libur_adapter_level_zero*.so*"))
             + bounded_find("/opt/intel/oneapi", "libur_adapter_level_zero*.so*"))
    ur_ocl = (glob.glob(os.path.join(vlib, "libur_adapter_opencl*.so*"))
              + bounded_find("/opt/intel/oneapi", "libur_adapter_opencl*.so*"))
    print("UR-адаптер Level Zero (venv):", [os.path.basename(x) for x in ur_l0] or "НЕ НАЙДЕН")
    print("UR-адаптер OpenCL    (venv):", [os.path.basename(x) for x in ur_ocl] or "НЕ НАЙДЕН")

    oneapi_dirs = run("ls -1 /opt/intel/oneapi/compiler 2>/dev/null | sort -V | tail -3")
    print("версии oneAPI (каталоги компилятора):", oneapi_dirs.replace("\n", " "))
    sys_oneapi = run("ls -1 /opt/intel/oneapi/compiler 2>/dev/null | sort -V | tail -1").strip()
    print("системная oneAPI:", sys_oneapi or "—")

    # ВНИМАНИЕ: рекурсивный glob по /opt/intel/oneapi/** ЗАВИСАЕТ (симлинки +
    # огромное дерево) ⇒ используем ограниченный find без следования симлинкам.
    sys_loaders = (glob.glob("/opt/intel/oneapi/lib/libur_loader.so*")
                   + bounded_find("/opt/intel/oneapi", "libur_loader.so*", maxdepth=4))
    pip_loaders = glob.glob(os.path.join(vlib, "libur_loader.so*"))
    print("libur_loader (системный):",
          {os.path.basename(p): sym_versions(p) for p in sys_loaders[:2]})
    print("libur_loader (пип, venv):",
          {os.path.basename(p): sym_versions(p) for p in pip_loaders[:2]})

    # ── слой 3: OpenCL ICD ──
    hdr("СЛОЙ 3 — OpenCL ICD (запасной путь)")
    print("/etc/OpenCL/vendors:", run("ls /etc/OpenCL/vendors/ 2>&1 | head -5"))
    print("libOpenCL:", glob.glob("/usr/lib64/libOpenCL.so*")
          + glob.glob("/usr/lib/x86_64-linux-gnu/libOpenCL.so*") or "НЕ НАЙДЕН")

    # ── слой 4: системный рантайм ──
    hdr("СЛОЙ 4 — что видит СИСТЕМНЫЙ рантайм oneAPI")
    if shutil.which("sycl-ls"):
        print("sycl-ls (С ТЕКУЩИМ LD_LIBRARY_PATH):")
        print(run("timeout 60 sycl-ls 2>&1 | head -10", timeout=75))
        clean = run(f"LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\\n' "
                    f"| grep -v '{vlib}' | paste -sd:) timeout 60 sycl-ls 2>&1 "
                    f"| head -10", timeout=75)
        print("sycl-ls (С ОЧИЩЕННЫМ путём, venv/lib убран) — РЕШАЮЩИЙ ТЕСТ:")
        print(clean)
    else:
        print("sycl-ls не найден в PATH")

    # ── слой 5: перечисление через torch ──
    hdr("СЛОЙ 5 — перечисление через torch")
    try:
        import torch
        print("torch:", torch.__version__, "| executable:", sys.executable)
        n = torch.xpu.device_count()
        print("torch.xpu.device_count():", n)
        for i in range(n):
            print("  устройство:", torch.xpu.get_device_name(i))
    except Exception as e:  # noqa: BLE001
        print("torch не дал device_count:", repr(e))

    print(f"\n[затрачено {time.time() - t_start:.1f} с] "
          f"осталось до бюджета {max(0.0, 240.0 - (time.time() - t_start)):.0f} с")
    print("\n-- UR-трейс перечисления --")
    print(run(f"UR_LOG_TRACING=level:info '{sys.executable}' -c \"import torch;"
              " print(torch.xpu.device_count())\" 2>&1 | head -20"))

    # ── решение ──
    hdr("ЧТО ДЕЛАТЬ")
    has_l0 = bool(l0_gpu)
    has_ocl = bool(glob.glob("/etc/OpenCL/vendors/*.icd"))
    pip_ver = ""
    for p in pip_loaders[:1]:
        v = sym_versions(p)
        pip_ver = v[0] if v else ""
    sys_ver = ""
    for p in sys_loaders[:1]:
        v = sym_versions(p)
        sys_ver = v[0] if v else ""
    shadowed = bool(pip_loaders) and vlib in (os.environ.get("LD_LIBRARY_PATH") or "")
    print(f"L0-драйвер Intel GPU : {'есть' if has_l0 else 'НЕТ'}")
    print(f"OpenCL ICD            : {'есть' if has_ocl else 'нет'}")
    print(f"системная oneAPI      : {sys_oneapi or '—'}")
    print(f"системный UR loader   : {sys_ver or '—'}")
    print(f"пип-UR loader (venv)  : {pip_ver or '—'}")
    print(f"venv/lib В ПУТИ       : {'ДА (значит, он затеняет системный)' if shadowed else 'нет'}")
    print()
    if shadowed and sys_oneapi and sys_oneapi >= "2025.2":
        print("ВЕРОЯТНАЯ ПРИЧИНА (главная гипотеза):")
        print("  пип-овский runtime oneAPI 2025.2.1 лежит в LD_LIBRARY_PATH и ЗАТЕНЯЕТ")
        print("  системный oneAPI (2026.0). Рантайм 2025.2 не перечисляет карту с вашим")
        print("  L0-драйвером 1.17.x ⇒ device_count = 0; и тот же loader ломает sycl-ls")
        print("  (undefined symbol ... LIBUR_LOADER_0.12).")
        print()
        print("  РЕШЕНИЕ (одна команда, окружение репозитория не трогаем):")
        print("    cd <репозиторий>/sycl_prism")
        print("    source probe/env_xpu.sh        # auto: система ≥2025.2 ⇒ venv/lib УБРАН")
        print("    python3 probe/check_xpu_device.py")
        print("  Если auto не сработал — принудительно:")
        print("    XPU_RUNTIME=system source probe/env_xpu.sh")
        print("  Ручной эквивалент (одна строка):")
        print(f"    export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\\n' "
              f"| grep -v '{vlib}' | paste -sd:)")
        print("  И после этого проверьте, что sycl-ls больше не падает и видит карту.")
    else:
        if not has_l0 and not has_ocl:
            print("A. В системе нет ни L0-драйвера, ни OpenCL-ICD ⇒ нужен драйвер Intel GPU")
            print("   (intel-level-zero-gpu + intel-opencl-icd, либо compute runtime с")
            print("   сайта Intel).")
        elif has_ocl:
            print("A. Есть OpenCL-ICD ⇒ запасной путь (без L0):")
            print("     ONEAPI_DEVICE_SELECTOR=opencl:gpu")
            print("   (или SYCL_DEVICE_FILTER=opencl:gpu)")
        else:
            print("A. L0-драйвер есть, OpenCL-ICD нет ⇒ должно работать через Level Zero.")
        print("B. Сравните с системным рантаймом: sycl-ls — если там карта есть,")
        print("   а в torch нет, это несовместимость версий (см. PROTOCOL.md §0.1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
