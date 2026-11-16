"""Диагностика XPU-окружения: почему `import torch` падает на libsycl/libur_loader.

Ситуация, которую ловит этот скрипт:
    ImportError: libsycl.so.8: undefined symbol: urEnqueueCooperativeKernelLaunchExp,
                 version LIBUR_LOADER_0.12
Причина: pip-сборка torch 2.9.1+xpu требует **пип-овский** `libur_loader`
(2025.2.1, символы LIBUR_LOADER_0.12), а загружается **системный** от oneAPI,
если `/opt/intel/oneapi/lib` стоит в LD_LIBRARY_PATH раньше site-packages.

Скрипт работает БЕЗ импорта torch (чистый ctypes/файлы) ⇒ его можно запускать
именно тогда, когда `import torch` уже падает.

Запуск:
    python3 check_xpu_env.py            # из sycl_prism/probe или sycl_prism/
"""
from __future__ import annotations

import ctypes
import glob
import os
import re
import subprocess
import sys


def venv_site() -> str:
    """site-packages ближайшего .venv-xpu (или текущего окружения)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for cand in (os.environ.get("XPU_SP"),
                 os.path.join(root, ".venv-xpu", "lib"),
                 sys.prefix):
        if not cand:
            continue
        hits = glob.glob(os.path.join(cand, "python3*", "site-packages"))
        if hits:
            return hits[0]
    return sys.prefix


def lib_versions(path: str) -> list[str]:
    """Версии символьных версий (LIBUR_LOADER_0.12) в .so — через strings."""
    try:
        out = subprocess.run(["strings", path], capture_output=True, text=True,
                             timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted(set(re.findall(r"LIBUR_LOADER_[0-9.]+", out)))


def needed_versions(path: str) -> list[str]:
    """Требуемые версии символов из .so (objdump -p | grep 'Version needed')."""
    try:
        out = subprocess.run(["objdump", "-p", path], capture_output=True,
                             text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted(set(re.findall(r"LIBUR_LOADER_[0-9.]+", out)))


def resolved_path() -> str:
    """Какой libur_loader реально загрузит динамический линкер сейчас."""
    try:
        ctypes.CDLL("libur_loader.so.0", mode=ctypes.RTLD_GLOBAL)
    except OSError as e:
        return f"НЕ ЗАГРУЖАЕТСЯ ({e})"
    try:
        with open("/proc/self/maps", encoding="utf-8") as f:
            for line in f:
                if "libur_loader" in line:
                    return line.split()[-1]
    except OSError:
        pass
    return "неизвестно (нет /proc/self/maps)"


def main() -> int:
    site = venv_site()
    print("=" * 78)
    print("ДИАГНОСТИКА XPU-ОКРУЖЕНИЯ")
    print("=" * 78)
    print("python  :", sys.executable)
    print("site    :", site)
    print("LD_LIBRARY_PATH:")
    for p in (os.environ.get("LD_LIBRARY_PATH") or "").split(":"):
        if p:
            print("   ", p)

    print("\n--- venv/lib (куда pip положил runtime oneAPI) ---")
    vlib = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
    if os.path.isdir(vlib):
        libs = sorted(os.path.basename(x) for x in glob.glob(os.path.join(vlib, "*.so*")))
        print(f"  {vlib}")
        print("   ", ", ".join(libs[:14]) if libs else "— пусто —")
    else:
        print(f"  {vlib} — нет")

    print("\n--- кандидаты libur_loader ---")
    # pip кладёт runtime-библиотеки oneAPI в <venv>/lib (НЕ в site-packages)
    venv_lib = os.path.dirname(os.path.dirname(sys.executable))
    venv_lib = os.path.join(venv_lib, "lib")
    cands = []
    cands += sorted(glob.glob(os.path.join(venv_lib, "libur_loader.so*")))
    cands += sorted(glob.glob(os.path.join(
        site, "intel_cmplr_lib_ur-*.data/data/lib/libur_loader.so*")))
    cands += sorted(glob.glob("/opt/intel/oneapi/lib/libur_loader.so*"))
    cands += sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libur_loader.so*"))
    cands += sorted(glob.glob("/usr/local/lib/libur_loader.so*"))
    if not cands:
        print("не найдено ни одного libur_loader")
    for c in cands:
        v = lib_versions(c)
        is_venv = c.startswith(os.path.dirname(os.path.dirname(sys.executable)))
        mark = "  ← pip-овский (нужен 0.12)" if is_venv else ""
        print(f"  {c}\n      версии символов: {v if v else '—'}{mark}")

    print("\n--- что требует libsycl.so.8 (torch) ---")
    vlib2 = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "lib")
    sy = sorted(glob.glob(os.path.join(site, "libsycl.so*"))
                + glob.glob(os.path.join(vlib2, "libsycl.so*")))
    for s in sy:
        print(f"  {s}: требует {needed_versions(s) or '—'}")
    if not sy:
        print("  libsycl.so.* не найден (нет pip-пакета intel-sycl-rt?)")

    print("\n--- что загрузится СЕЙЧАС ---")
    print("  libur_loader:", resolved_path())

    print("\n--- импорт torch ---")
    try:
        import torch
        print("  torch:", torch.__version__, "— импорт прошёл")
        if hasattr(torch, "xpu"):
            try:
                ok = torch.xpu.is_available()
                print("  torch.xpu.is_available():", ok)
                if ok:
                    print("  устройство:", torch.xpu.get_device_name(0))
            except Exception as e:  # noqa: BLE001
                print("  torch.xpu:", repr(e))
        else:
            print("  модуля torch.xpu нет ⇒ это НЕ XPU-сборка")
        return 0
    except Exception as e:  # noqa: BLE001
        print("  ИМПОРТ УПАЛ:", repr(e))

    print("\n" + "=" * 78)
    print("ЧТО ДЕЛАТЬ")
    print("=" * 78)
    print("1) Не трогайте venv репозитория и установленный oneAPI.")
    print("2) Поднимите pip-библиотеки XPU выше системных:")
    print("     cd <репозиторий>/sycl_prism && source probe/env_xpu.sh")
    print("   (скрипт ставит <venv>/lib в начало LD_LIBRARY_PATH — там pip держит")
    print("    libur_loader.so.0 с LIBUR_LOADER_0.12 и libsycl.so.8)")
    print("3) Если не помогло — принудительно:")
    ur = next((c for c in cands if "intel_cmplr_lib_ur" in c), None)
    if ur:
        print(f"     export LD_PRELOAD={ur}")
    print("4) Если и это не помогло — система новее пип-версии: тогда")
    print("   либо обновите oneAPI, либо поставьте torch попроще:")
    print("     pip install -i https://download.pytorch.org/whl/xpu 'torch==2.7.1+xpu'")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
