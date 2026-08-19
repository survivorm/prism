#!/usr/bin/env python3
"""Build kaggle_kernel/kernel.ipynb — single notebook, all prism inlined, no internal imports."""

import importlib
import inspect
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def get_source(module_path):
    """Get source of a module via importlib + inspect."""
    mod = importlib.import_module(module_path)
    return inspect.getsource(mod)


def strip_docstring(code: str) -> str:
    m = re.match(r'^\s*(""".*?"""|\'\'\'.*?\'\'\')\s*\n', code, re.DOTALL)
    return code[m.end() :] if m else code


def strip_main_block(code: str) -> str:
    """Remove argparse import and if __name__ block."""
    code = re.sub(r"^import argparse\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"^if __name__ == .*$[\s\S]*$", "", code, flags=re.MULTILINE)
    return code


def strip_prism_imports(code: str) -> str:
    """Remove all 'from prism.X import ...' — multi-line first, then single-line."""
    code = re.sub(r"\n?from prism\.\w+ import \([\s\S]*?\)\n?", "", code)
    code = re.sub(r"\n?from prism\.\w+ import [^\n]*\n?", "", code)
    return code


def rename_main(code: str, new_name: str) -> str:
    code = re.sub(r"\bdef main\b", f"def {new_name}", code)
    code = re.sub(r"\bmain\s*\(", f"{new_name}(", code)
    return code


def code_cell(src: str) -> dict:
    lines = src.split("\n")
    result = [line + "\n" if line else "" for line in lines]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": result,
    }


# ---- Build ----

# Read deps from pyproject.toml
try:
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        pyproject = f.read()
except FileNotFoundError:
    pyproject = ""

if not pyproject:
    print("No pyproject.toml found", file=sys.stderr)
    sys.exit(1)

deps_match = re.search(r"dependencies\s*=\s*\[([^\]]+)\]", pyproject, re.DOTALL)
deps = re.findall(r'"([^"]+)"', deps_match.group(1)) if deps_match else []
pip_cell = (
    "!pip uninstall -y -q torch torchvision torchaudio cudf easyocr numba-cuda dask-cuda pylibraft-cu12 raft-dask-cu12 pylibcudf-cu12 rmm-cu12 quantecon cuvs-cu12 ucxx-cu12 cuml-cu12 cudf-cu12 libcuvs-cu12 libraft-cu12 libcuml-cu12 cuml-cu12 cuda-python cuml libcuml libcuvs libraft numba ydata-profiling google-colab\n"
    "!pip install -q torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121\n"
    "!pip install -q " + " ".join(deps) + "\n"
    "!pip install -q -U transformers\n"
)

cells = [
    # Cell 0: pip install
    code_cell(f"# Install dependencies\n{pip_cell}\n"),
    # Cell 1: imports + GPU setup
    code_cell(
        "import os\n"
        "import sys\n"
        "import time\n"
        "import traceback\n"
        "import torch\n"
        "\n"
        'WORKING = "/kaggle/working"\n'
        "sys.path.insert(0, WORKING)\n"
        "\n"
        "num_gpus = torch.cuda.device_count()\n"
        "if num_gpus >= 2:\n"
        '    device_L, device_S = "cuda:0", "cuda:1"\n'
        "    dtype_L = dtype_S = torch.float32\n"
        "elif num_gpus == 1:\n"
        '    device_L = device_S = "cuda:0"\n'
        "    dtype_L = dtype_S = torch.float32\n"
        "else:\n"
        '    device_L = device_S = "cpu"\n'
        "    dtype_L = dtype_S = torch.float32\n"
        "\n"
        "if num_gpus > 0:\n"
        '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n'
        'print(f"torch {torch.__version__} | L={device_L}, S={device_S}")\n'
        "\n"
        'RES = os.path.join(WORKING, "results")\n'
        "os.makedirs(RES, exist_ok=True)\n"
        "\n"
        "\n"
        "def resolve(local_dir, hf_id):\n"
        "    return local_dir if os.path.isdir(local_dir) else hf_id\n"
        "\n"
        "\n"
        'SMALL = resolve("/kaggle/input/qwen35-08b", "Qwen/Qwen3.5-0.8B")\n'
        'LARGE = resolve("/kaggle/input/qwen35-4b", "Qwen/Qwen3.5-4B")\n'
    ),
]

# Prism modules via importlib + inspect.getsource
for mod in ["prism.jacobian", "prism.bridge", "prism.esg", "prism.calibration"]:
    src = get_source(mod)
    src = strip_docstring(src)
    cells.append(code_cell(src))

# Exp files — strip prism imports, __main__, rename main()
for exp_path, new_main in [
    ("kaggle_kernel/exp13_domain_esg.py", "main_13"),
    ("kaggle_kernel/exp15_gdn_anchors.py", "main_15"),
]:
    try:
        with open(os.path.join(ROOT, exp_path)) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"Missing: {exp_path}", file=sys.stderr)
        sys.exit(1)
    src = strip_docstring(src)
    src = strip_prism_imports(src)
    src = strip_main_block(src)
    src = rename_main(src, new_main)
    cells.append(code_cell(src))

# Runner cell
cells.append(
    code_cell(
        "# ---------- run ----------\n"
        "JOBS = [\n"
        '    ("exp13", lambda: main_13(\n'
        "        large_model=LARGE, small_model=SMALL, n_samples=8,\n"
        "        device_L=device_L, device_S=device_S, results_dir=RES,\n"
        "    )),\n"
        '    ("exp15", lambda: main_15(\n'
        "        model=SMALL, large_model=LARGE, n_samples=4,\n"
        "        device_L=device_L, device_S=device_S, results_dir=RES,\n"
        "    )),\n"
        "]\n"
        "\n"
        "for name, fn in JOBS:\n"
        "    t0 = time.time()\n"
        "    try:\n"
        "        fn()\n"
        "        elapsed = time.time() - t0\n"
        '        print(f"[OK] {name} {elapsed:.0f}s")\n'
        "        if elapsed > 36000:\n"
        '            print(f"  WARNING: exceeded 10h cap")\n'
        "    except Exception as e:\n"
        '        print(f"[FAIL] {name}: {e}")\n'
        "        traceback.print_exc()\n"
        "\n"
        'print("\\nResults:")\n'
        "if os.path.isdir(RES):\n"
        "    try:\n"
        "        for fn in sorted(os.listdir(RES)):\n"
        '            print(f"  {fn} ({os.path.getsize(os.path.join(RES, fn)):,} bytes)")\n'
        "    except OSError:\n"
        "        pass\n"
    )
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(ROOT, "kaggle_kernel/kernel.ipynb")
try:
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=None, separators=(",", ":"))
except OSError as e:
    print(f"Write failed: {e}", file=sys.stderr)
    sys.exit(1)

print(f"Written {out_path} ({os.path.getsize(out_path):,} bytes, {len(cells)} cells)")
print(f"Deps: {', '.join(deps)}")
