#!/usr/bin/env python3
"""build_kernel.py — T4-кейл для AT2e (kaggle_run.sh pattern, TZ §6b).

Читает experiments/at2/AT2e.py, инжектит T4-bootstrap (pip-pin, модель, pool из
dataset, env), пишет experiments/kaggle_at2e/at2e_t4_<mode>.py для mode ∈ {canary, main}.

Прогон:
  kaggle datasets create voroninmichael/prism-at2e-pool -p pool_ds/ -m pool_ds/meta.json  (одноразово)
  python3 experiments/kaggle_at2e/build_kernel.py canary
  ./kaggle_run.sh experiments/kaggle_at2e/at2e_t4_canary.py at2e-t4-canary gpu voroninmichael/prism-at2e-pool
  # после canary PASS:
  python3 experiments/kaggle_at2e/build_kernel.py main
  ./kaggle_run.sh experiments/kaggle_at2e/at2e_t4_main.py at2e-t4-main gpu voroninmichael/prism-at2e-pool

Провенанс: bootstrap несёт AT2E_SCRIPT_SHA = sha256 локального AT2e.py на момент билда;
артефакт (в kernel'е без __file__) использует его; builder ASSERT, что sha встроенного
текста (за вычетом bootstrap) равен локальному файлу.
"""
import hashlib, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "experiments", "at2", "AT2e.py")

BOOTSTRAP = '''# ==== T4 bootstrap (builder-injected, AT2e r3; TZ §6b) ====
import os as _t4os, sys as _t4sys, subprocess as _t4sp, time as _t4time, shutil as _t4sh
_T4_T0 = _t4time.time()
print(f"[t4-bootstrap] start py={_t4sys.version.split()[0]}", flush=True)
_t4sp.check_call([_t4sys.executable, "-m", "pip", "install", "-q",
                  "transformers==5.15.1", "tokenizers==0.22.2", "huggingface_hub"])
_t4os.makedirs("/data/Qwen", exist_ok=True)
_MODEL_DIR = "/data/Qwen/Qwen3.5-0.8B"
if not _t4os.path.exists(_MODEL_DIR):
    from huggingface_hub import snapshot_download
    _t4os.symlink(snapshot_download("Qwen/Qwen3.5-0.8B"), _MODEL_DIR)
def _locate(fname):
    for _r, _d, _f in _t4os.walk("/kaggle/input"):
        if fname in _f:
            return _t4os.path.join(_r, fname)
    raise FileNotFoundError(fname)
_POOL_SRC = _locate("exp27b_pool_ids_L23.pt")
_t4os.makedirs("/data/prism_kernels/exp27b/results", exist_ok=True)
_t4sh.copy(_POOL_SRC, "/data/prism_kernels/exp27b/results/exp27b_pool_ids_L23.pt")
# envVars metadata (kaggle_run.sh) — НЕ проверенный механизм; bootstrap — надёжный
# единственный путь (at16 паттерн: хардкод /kaggle/working)
_t4os.environ.setdefault("PRISM_KAGGLE_ROOT", "/kaggle/working")
_t4os.environ["AT2E_MODEL_DIR"] = _MODEL_DIR
_t4os.environ["AT2E_POOL"] = "/data/prism_kernels/exp27b/results/exp27b_pool_ids_L23.pt"
_t4os.environ["AT2E_JCACHE"] = "/data/prism_kernels/exp27b/results/exp27b_jcache_L12.pt"
_t4os.environ["AT2E_DEVICE"] = "cuda"
_t4os.environ["AT2E_TRACKIO"] = "0"
_t4os.environ["AT2E_NO_JCACHE"] = "1"
_t4os.environ["AT2E_SCRIPT_SHA"] = "__AT2E_SHA__"
_t4sys.argv = ["at2e_t4", "__T4_MODE_ARG__"]
print(f"[t4-bootstrap] ready in {_t4time.time()-_T4_T0:.1f}s; mode={'canary' if '--canary' in _t4sys.argv else 'main'}", flush=True)
# ==== конец T4 bootstrap ====

'''

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "canary"
    assert mode in ("canary", "main"), f"mode {mode}?"
    src = open(SRC, encoding="utf-8").read()
    sha = hashlib.sha256(src.encode()).hexdigest()
    bootstrap = BOOTSTRAP.replace("__AT2E_SHA__", sha).replace(
        "__T4_MODE_ARG__", "--canary" if mode == "canary" else "main")
    out = os.path.join(HERE, f"at2e_t4_{mode}.py")
    with open(out, "w", encoding="utf-8") as f:
        f.write(bootstrap + src)
    # guard: встроенный текст (за вычетом bootstrap) = локальный файл
    back = open(out, encoding="utf-8").read()
    assert back[len(bootstrap):] == src, "embedded text != local AT2e.py"
    # guard: bootstrap-константы на месте
    assert f'"AT2E_SCRIPT_SHA"] = "{sha}"' in back
    print(f"OK {os.path.basename(out)}: sha256(AT2e.py)={sha}, mode={mode}")
    print(f"run: ./kaggle_run.sh experiments/kaggle_at2e/at2e_t4_{mode}.py "
          f"at2e-t4-{mode} gpu voroninmichael/prism-at2e-pool")

if __name__ == "__main__":
    main()
