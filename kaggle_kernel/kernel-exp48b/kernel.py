import os, sys, time, glob, traceback, subprocess, importlib.util

t0 = time.time()
os.environ.setdefault("HF_HOME", "/kaggle/working/.model_cache")
os.makedirs("/kaggle/working/.model_cache", exist_ok=True)
try:
    subprocess.run(["pip", "uninstall", "-y", "-q", "torch", "torchvision", "torchaudio",
                    "cudf", "dask-cuda", "cuml", "cuml-cu12", "libcuml", "numba-cuda"],
                   check=False, capture_output=True)
    subprocess.run(["pip", "install", "-q", "torch==2.6.0+cu124",
                    "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
    subprocess.run(["pip", "install", "-q", "-U", "transformers", "datasets"], check=True)

    p = glob.glob("/kaggle/input/**/exp48b_lfm3_gen_kinematics.py", recursive=True)
    print("bundle:", p, flush=True)
    code = os.path.dirname(os.path.dirname(p[0]))
    sys.path.insert(0, code)

    from huggingface_hub import snapshot_download
    model = snapshot_download("LiquidAI/LFM3-1.7B", local_dir="/kaggle/working/lfm3-17b")
    print("model at", model, flush=True)

    spec = importlib.util.spec_from_file_location("exp48b", p[0])
    X = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(X)

    out = X.main(model=model, n_docs=16, max_new=384, prompt_len=64,
                 device="cuda", results_dir="/kaggle/working")
    nd = {d0: (sum(x["n_loops"] for x in out["docs"] if x["dom"] == d0),
               sum(1 for x in out["docs"] if x["dom"] == d0 and x["n_loops"] > 0))
          for d0 in ("wiki", "math", "code", "chat")}
    print("loops (episodes, docs):", nd, flush=True)
except Exception:
    traceback.print_exc()
print("DONE %.1fs" % (time.time() - t0))
