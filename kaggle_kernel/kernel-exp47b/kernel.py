import os, sys, time, json, glob, zipfile, traceback, subprocess

t0 = time.time()
os.environ.setdefault("HF_HOME", "/kaggle/working/.model_cache")
os.makedirs("/kaggle/working/.model_cache", exist_ok=True)
try:
    # env: same recipe as exp12a/exp27b kernels (strip stale CUDA stack, fresh
    # torch cu121 + transformers >= 5.15 for Qwen3.5)
    subprocess.run(["pip", "uninstall", "-y", "-q", "torch", "torchvision", "torchaudio",
                    "cudf", "dask-cuda", "cuml", "cuml-cu12", "libcuml", "numba-cuda"],
                   check=False, capture_output=True)
    subprocess.run(["pip", "install", "-q", "torch==2.6.0+cu124",
                    "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
    subprocess.run(["pip", "install", "-q", "-U", "transformers", "datasets"], check=True)

    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)

    zpath = glob.glob("/kaggle/input/**/exp47b_gen_kinematics.py", recursive=True)
    print("bundle:", zpath, flush=True)
    code = os.path.dirname(os.path.dirname(zpath[0]))   # .../experiments/... -> корень
    sys.path.insert(0, code)

    from huggingface_hub import snapshot_download
    model = snapshot_download("Qwen/Qwen3.5-0.8B", local_dir="/kaggle/working/qwen35-08b")
    print("model at", model, flush=True)

    exp = os.path.join(code, "experiments", "exp47b_gen_kinematics.py")
    cmd = [sys.executable, "-u", exp, "--model", model, "--device", "cuda",
           "--n-docs", "16", "--max-new", "384", "--prompt-len", "64",
           "--results-dir", "/kaggle/working"]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    out = "/kaggle/working/exp47b_gen_kinematics.json"
    d = json.load(open(out))
    for k, v in sorted(d.get("verdicts", {}).items()):
        print("VERDICT %s -> %s" % (k, v))
    for dom, st in d.get("descriptive", {}).get("per_domain", {}).items():
        print("  %-8s a=%.3f s=%.3f p=%.3f loops=%d" % (
            dom, st.get("a_med", -1), st.get("s_med", -1), st.get("p_med", -1),
            st.get("n_loops", -1)))
    for f in os.listdir("/kaggle/working"):
        if f.endswith((".json", ".npz")):
            print("OK", f, os.path.getsize("/kaggle/working/" + f) // 1024, "KB")
except Exception:
    traceback.print_exc()
print("DONE %.1fs" % (time.time() - t0))
