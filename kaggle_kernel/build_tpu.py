#!/usr/bin/env python3
"""Build a TPU *script* kernel for a prism causal experiment.

Reuses build_kernel.py's prism-module inlining but emits a single .py +
script-kernel metadata (no notebook, no pip — the Kaggle v5e-8 image already
ships torch / torch_xla / transformers, as the iso-canary proved).

Usage:
    python3 build_tpu.py --exp ../../experiments/exp18_logit_kd.py \
        --out kaggle_kernel/kernel-exp18-tpu --name exp18_logit_kd \
        --id voroninmichael/prism-exp18-logit-kd-tpu --title "Prism: exp18-logit-kd-tpu"
"""
import argparse
import importlib
import inspect
import json
import os
import sys

# Reuse build_kernel's inlining helpers (module import + strip).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_kernel import (  # noqa: E402
    get_source,
    strip_docstring,
    strip_prism_imports,
    strip_main_block,
    rename_main,
    assert_no_bare_prism,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRISM_DEPS = [
    "prism.jacobian",
    "prism.bridge",
    "prism.esg",
    "prism.calibration",
    "prism.cache_io",
    "prism.layer_selection",
]

def _strip_future(s: str) -> str:
    """Remove 'from __future__' lines (a single one is placed at file top)."""
    return "\n".join(l for l in s.split("\n") if not l.strip().startswith("from __future__"))


# A script is ONE module: 'from __future__' must be the very first statement,
# so we emit exactly one here and strip the rest from the inlined sources.
HEADER = '''from __future__ import annotations
# ---------- TPU runtime ----------
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ.setdefault("HF_HOME", "/kaggle/working/.hf_cache")
import time, sys, traceback, inspect, gc
import torch
torch.set_num_threads(1)
import torch_xla.core.xla_model as xm
xdev = xm.xla_device()
device = "xla"
WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.expanduser("~/.kaggle_test")
RES = os.path.join(WORKING, "results")
os.makedirs(RES, exist_ok=True)
print(f"[TPU] device={xdev} torch={torch.__version__}")
'''


def build_tpu(exp_file, out_dir, kernel_id, title, runner_extra="", name=None, datasets=None, pip_extra=""):
    with open(exp_file) as f:
        exp_src = f.read()
    exp_name = name or os.path.basename(exp_file).replace(".py", "")
    main_name = f"main_{exp_name.replace('exp', '').replace('_', '')}"

    # Optional pip override, inserted AFTER the mandatory first line
    # ('from __future__ import annotations') but BEFORE any other import,
    # so prism's inlined sources import transformers==5.15.1 for this run.
    # --no-deps keeps the image's torch/torch_xla untouched. A version assert
    # makes a bad install fail fast on the [OK]/[FAIL] banner instead of
    # burning a long TPU queue wait on a mid-run ImportError.
    header = HEADER
    if pip_extra:
        first, rest = HEADER.split("\n", 1)
        pip_block = (
            "import subprocess, sys\n"
            f"subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps', {repr(pip_extra)}])\n"
            "import transformers as _tf\n"
            "_v = _tf.__version__\n"
            "print(f'  [pip] transformers={_v}')\n"
            "if _v != '5.15.1':\n"
            "    print(f'[FAIL] transformers pinned to 5.15.1 but got {_v}; aborting before model load')\n"
            "    sys.exit(0)\n"
        )
        header = first + "\n" + pip_block + "\n" + rest
    parts = [header]
    for mod in PRISM_DEPS:
        parts.append(_strip_future(strip_docstring(get_source(mod))))
    parts.append("\n# ---------- " + exp_name + " ----------\n")
    exp_inl = _strip_future(strip_docstring(exp_src))
    exp_inl = strip_prism_imports(exp_inl)
    exp_inl = strip_main_block(exp_inl)
    exp_inl = rename_main(exp_inl, main_name)
    parts.append(exp_inl)

    runner = (
        f"# ---------- run {exp_name} ----------\n"
        f"gc.collect()\n"
        f"sig = inspect.signature({main_name})\n"
        f"params = list(sig.parameters)\n"
        f"kwargs = {{}}\n"
        f"if 'small_model' in params: kwargs['small_model'] = 'Qwen/Qwen3.5-0.8B'\n"
        f"if 'large_model' in params: kwargs['large_model'] = 'Qwen/Qwen3.5-4B'\n"
        f"if 'model' in params: kwargs['model'] = 'Qwen/Qwen3.5-0.8B'\n"
        f"if 'device' in params: kwargs['device'] = 'xla'\n"
        f"if 'results_dir' in params: kwargs['results_dir'] = RES\n"
        f"if 'targets_dir' in params: kwargs['targets_dir'] = os.path.join(RES, 'targets')\n"
        f"if 'small_model' in kwargs and 'model' in kwargs: del kwargs['model']\n"
        + (runner_extra or "")
        + f"t0 = time.time()\n"
        f"try:\n"
        f"    {main_name}(**kwargs)\n"
        f"    print(f'[OK] {exp_name} {{time.time()-t0:.0f}}s')\n"
        f"except Exception as e:\n"
        f"    print(f'[FAIL] {exp_name}: {{e}}')\n"
        f"    traceback.print_exc()\n"
        f'print("\\nResults:")\n'
        f"if os.path.isdir(RES):\n"
        f"    for fn in sorted(os.listdir(RES)):\n"
        f"        print(f'  {{fn}} {{os.path.getsize(os.path.join(RES, fn)):,}} bytes')\n"
    )
    parts.append(runner)
    code = "\n".join(parts)

    # Guard: no bare prism import may survive (would crash on Kaggle).
    cells = [{"cell_type": "code", "source": code.split("\n")}]
    assert_no_bare_prism(cells)

    py_name = f"{exp_name}.py"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, py_name), "w") as f:
        f.write(code)
    # Local compile check: catches 'from __future__ mid-file' / any SyntaxError
    # before burning a TPU slot (Kaggle would otherwise fail on the first cell).
    import py_compile
    py_compile.compile(os.path.join(out_dir, py_name), doraise=True)
    print(f"compile OK: {py_name}")
    meta = {
        "id": kernel_id,
        "title": title,
        "code_file": py_name,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "true",
        "enable_internet": "true",
    }
    if datasets:
        meta["dataset_sources"] = datasets
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Written {out_dir}/{py_name} ({os.path.getsize(os.path.join(out_dir, py_name)):,} bytes)")
    print(f"Written {out_dir}/kernel-metadata.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--id", default="")
    p.add_argument("--title", default="Prism TPU")
    p.add_argument("--runner-extra", default="")
    p.add_argument("--datasets", default=[], nargs="*", help="dataset_sources (e.g. prism-exp16-targets)")
    p.add_argument("--pip", default="transformers==5.15.1 safetensors==0.8.0",
                   help="pip pkgs to install before run (default pins transformers==5.15.1 + safetensors==0.8.0 to match dev env / kill API drift)")
    a = p.parse_args()
    build_tpu(a.exp, a.out, a.id, a.title, a.runner_extra, a.name or None, a.datasets or None, a.pip)
