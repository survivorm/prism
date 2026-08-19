#!/usr/bin/env python3
"""Build a single kernel.ipynb for one experiment.

Usage:
    uv run python3 build_kernel.py --exp exp13 --out kaggle_kernel/kernel-exp13/
"""

import argparse
import importlib
import inspect
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ---- Template metadata ----
METADATA_TEMPLATE = {
    "kernel_type": "notebook",
    "is_gpu_enabled": True,
    "machine_shape": "NvidiaTeslaT4",
    "accelerator": "gpu",
    "language": "python",
    "code_file": "kernel.ipynb",
}


def get_source(module_path):
    mod = importlib.import_module(module_path)
    return inspect.getsource(mod)


def strip_docstring(code: str) -> str:
    m = re.match(r'^\s*(""".*?"""|\'\'\'.*?\'\'\')\s*\n', code, re.DOTALL)
    return code[m.end() :] if m else code


def strip_main_block(code: str) -> str:
    code = re.sub(r"^import argparse\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"^if __name__ == .*$[\s\S]*$", "", code, flags=re.MULTILINE)
    return code


def strip_prism_imports(code: str) -> str:
    code = re.sub(r"\n?\s*from prism\.\w+ import \([\s\S]*?\)\n?", "", code)
    code = re.sub(r"\n?\s*from prism\.\w+ import [^\n]*\n?", "", code)
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


def build_kernel(
    exp_file: str,
    out_dir: str,
    kernel_id: str = "",
    kernel_title: str = "Prism Experiment",
    kernel_sources: list[str] | None = None,
):
    """Build kernel.ipynb and kernel-metadata.json for one experiment."""
    # Read deps from pyproject.toml
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        pyproject = f.read()

    deps_match = re.search(r"dependencies\s*=\s*\[([^\]]+)\]", pyproject, re.DOTALL)
    deps = re.findall(r'"([^"]+)"', deps_match.group(1)) if deps_match else []

    pip_cell = (
        "!pip uninstall -y -q torch torchvision torchaudio cudf easyocr numba-cuda dask-cuda pylibraft-cu12 raft-dask-cu12 pylibcudf-cu12 rmm-cu12 quantecon cuvs-cu12 ucxx-cu12 cuml-cu12 cuda-python cuml libcuml libcuvs libraft numba ydata-profiling google-colab\n"
        "!pip install -q torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121\n"
        "!pip install -q " + " ".join(deps) + "\n"
        "!pip install -q -U transformers\n"
    )

    # Read exp file
    with open(exp_file) as f:
        exp_src = f.read()
    exp_name = os.path.basename(exp_file).replace(".py", "")
    main_name = f"main_{exp_name.replace('exp', '').replace('_', '')}"

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
            "# KAGGLEHUB_CACHE: persist models across sessions\n"
            'os.environ["KAGGLEHUB_CACHE"] = "/kaggle/working/.model_cache"\n'
            'os.makedirs("/kaggle/working/.model_cache", exist_ok=True)\n'
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
            "    dtype_L = torch.float16\n"
            "    dtype_S = torch.float32\n"
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
            "# Resolve models: kernel_sources -> working -> HF, with save_pretrained fallback\n"
            "def resolve_model(hf_id):\n"
            "    # 1. Check ../input/ from kernel_sources\n"
            '    for d in ["qwen35-08b", "qwen35-4b", "prism-exp16-stage0-targets"]:\n'
            '        p = os.path.join("/kaggle/input", d, hf_id.split("/")[-1])\n'
            "        if os.path.isdir(p):\n"
            '            print(f"Found {hf_id} at {p}")\n'
            "            return p\n"
            "    # 2. Check WORKING\n"
            '    wf = os.path.join(WORKING, "qwen35-08b" if "0.8B" in hf_id else "qwen35-4b")\n'
            "    if os.path.isdir(wf):\n"
            '        print(f"Found {hf_id} at {wf}")\n'
            "        return wf\n"
            "    # 3. Return HF ID — from_pretrained will download, save_pretrained will persist\n"
            '    print(f"Loading {hf_id} from HF...")\n'
            "    return hf_id\n"
            "\n"
            "# Load + save for persistence (calls from_pretrained, then save_pretrained if from HF)\n"
            "def load_model(cls, hf_id, **kw):\n"
            "    path = resolve_model(hf_id)\n"
            "    model = cls.from_pretrained(path, **kw)\n"
            "    # If path was HF id (not local), persist it\n"
            "    if path == hf_id:\n"
            '        dst = os.path.join(WORKING, "qwen35-08b" if "0.8B" in hf_id else "qwen35-4b")\n'
            "        if not os.path.isdir(dst):\n"
            '            print(f"Saving {hf_id} to {dst}...")\n'
            "            model.save_pretrained(dst)\n"
            '            print(f"Saved")\n'
            "    return model\n"
            "\n"
            'SMALL = resolve_model("Qwen/Qwen3.5-0.8B")\n'
            'LARGE = resolve_model("Qwen/Qwen3.5-4B")\n'
        ),
    ]

    # Prism modules
    for mod in ["prism.jacobian", "prism.bridge", "prism.esg", "prism.calibration"]:
        src = get_source(mod)
        src = strip_docstring(src)
        cells.append(code_cell(src))

    # Experiment file
    src = strip_docstring(exp_src)
    src = strip_prism_imports(src)
    src = strip_main_block(src)
    src = rename_main(src, main_name)
    cells.append(code_cell(src))

    # Runner cell — inspect main() signature, pass appropriate kwargs
    runner_code = (
        f"# ---------- run {exp_name} ----------\n"
        f"import inspect\n"
        f"sig = inspect.signature({main_name})\n"
        f"kwargs = {{}}\n"
        f"params = list(sig.parameters)\n"
        f"if 'small_model' in params: kwargs['small_model'] = SMALL\n"
        f"if 'large_model' in params: kwargs['large_model'] = LARGE\n"
        f"if 'model' in params: kwargs['model'] = SMALL\n"
        f"if 'device_L' in params: kwargs['device_L'] = device_L\n"
        f"if 'device_S' in params: kwargs['device_S'] = device_S\n"
        f"if 'device' in params: kwargs['device'] = device_S\n"
        f"if 'results_dir' in params: kwargs['results_dir'] = RES\n"
        f"if 'targets_dir' in params: kwargs['targets_dir'] = os.path.join(RES, 'targets')\n"
        f"# Remove conflicting keys: if both model and small_model, prefer specific\n"
        f"if 'small_model' in kwargs and 'model' in kwargs: del kwargs['model']\n"
        f"# Ensure models exist in WORKING before running\n"
        f"import transformers\n"
        f"if not os.path.isdir('/kaggle/working/qwen35-08b'):\n"
        f"    print('Loading small model (first run)...')\n"
        f"    _m = load_model(transformers.AutoModelForCausalLM, 'Qwen/Qwen3.5-0.8B', torch_dtype=torch.float32, attn_implementation='eager')\n"
        f"    del _m\n"
        f"    if torch.cuda.is_available(): torch.cuda.empty_cache()\n"
        f"if not os.path.isdir('/kaggle/working/qwen35-4b'):\n"
        f"    print('Loading large model (first run)...')\n"
        f"    _m = load_model(transformers.AutoModelForCausalLM, 'Qwen/Qwen3.5-4B', torch_dtype=torch.float16, attn_implementation='eager')\n"
        f"    del _m\n"
        f"    if torch.cuda.is_available(): torch.cuda.empty_cache()\n"
        f"t0 = time.time()\n"
        f"try:\n"
        f"    {main_name}(**kwargs)\n"
        f"    elapsed = time.time() - t0\n"
        f'    print(f"[OK] {exp_name} {{elapsed:.0f}}s")\n'
        f"    # Persist models (in case experiment loaded from HF but didn't save)\n"
        f"    try:\n"
        f'        dst_s = os.path.join(WORKING, "qwen35-08b")\n'
        f'        dst_l = os.path.join(WORKING, "qwen35-4b")\n'
        f"        if os.path.isdir(dst_s):\n"
        f'            print(f"Small model persisted at {{dst_s}}")\n'
        f"        if os.path.isdir(dst_l):\n"
        f'            print(f"Large model persisted at {{dst_l}}")\n'
        f"    except Exception:\n"
        f"        pass\n"
        f"except Exception as e:\n"
        f'    print(f"[FAIL] {exp_name}: {{e}}")\n'
        f"    traceback.print_exc()\n"
        f"\n"
        f'print("\\nResults:")\n'
        f"if os.path.isdir(RES):\n"
        f"    try:\n"
        f"        for fn in sorted(os.listdir(RES)):\n"
        f'            print(f"  {{fn}} {{os.path.getsize(os.path.join(RES, fn)):,}} bytes")\n'
        f"    except OSError:\n"
        f"        pass\n"
    )
    cells.append(code_cell(runner_code))

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

    out_path = os.path.join(out_dir, "kernel.ipynb")
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=None, separators=(",", ":"))

    meta_path = os.path.join(out_dir, "kernel-metadata.json")
    meta = dict(METADATA_TEMPLATE)
    if kernel_id:
        meta["id"] = kernel_id
    if kernel_title:
        meta["title"] = kernel_title
    if "/" in kernel_id:
        meta["slug"] = kernel_id.split("/")[-1]
    if kernel_sources:
        meta["kernel_sources"] = kernel_sources
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Written {out_path} ({os.path.getsize(out_path):,} bytes, {len(cells)} cells)"
    )
    print(f"Written {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        required=True,
        help="Path to exp file, e.g. experiments/exp13_domain_esg.py",
    )
    parser.add_argument(
        "--out", required=True, help="Output dir, e.g. kaggle_kernel/kernel-exp13/"
    )
    parser.add_argument(
        "--id", default="", help="Kaggle kernel id, e.g. voroninmichael/prism-exp13"
    )
    parser.add_argument("--title", default="Prism Experiment", help="Kernel title")
    parser.add_argument(
        "--sources", default=[], nargs="*", help="kernel_sources for persistence"
    )
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    build_kernel(args.exp, args.out, args.id, args.title, args.sources)
