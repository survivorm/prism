import os, sys, time, glob, json, shutil, zipfile, traceback, importlib.util
t0 = time.time()
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    import subprocess
    subprocess.run(["pip", "install", "-q", "-U", "transformers"], check=False)
    from huggingface_hub import snapshot_download
    model = snapshot_download("Qwen/Qwen3.5-0.8B", local_dir="/kaggle/working/qwen35-08b")
    print("model at", model, flush=True)

    print("/kaggle/input:", os.listdir("/kaggle/input"), flush=True)
    p = glob.glob("/kaggle/input/**/exp50v3_curvature.py", recursive=True)
    if not p:                                  # dir-mode zip: распаковать
        for z in glob.glob("/kaggle/input/prism-code-48b/*.zip"):
            print("unzip", z, flush=True)
            with zipfile.ZipFile(z) as zf: zf.extractall("/kaggle/working/bundle")
        p = glob.glob("/kaggle/working/bundle/**/exp50v3_curvature.py", recursive=True)
        bundle = None                          # zip'ы лежали плоско: bundle == working/bundle
    else:
        bundle = os.path.dirname(os.path.dirname(p[0]))
    print("bundle root:", bundle, "code:", p[:1], flush=True)
    code_dir = os.path.dirname(p[0])
    res_dir = os.path.join(bundle, "results") if bundle else "/kaggle/working/bundle/results"
    data_dir = os.path.join(bundle, "data") if bundle else "/kaggle/working/bundle/data"
    for env_k, path in (("PRISM_QWEN_DIR", model),
                        ("PRISM_JCACHE", data_dir + "/U128.pt"),
                        ("PRISM_NPZ", data_dir + "/exp47b_gen_kinematics.npz"),
                        ("PRISM_DOCS_JSON", res_dir + "/exp47b_gen_kinematics.json")):
        os.environ[env_k] = path
    work = "/kaggle/working/bundle"
    if os.path.dirname(code_dir) != work:      # если смонтировано не zip'ом — копируем (readonly)
        shutil.copytree(code_dir, work + "/experiments", dirs_exist_ok=True)
        if not os.path.isdir(work + "/results"): shutil.copytree(res_dir, work + "/results")
        os.environ["PRISM_DOCS_JSON"] = work + "/results/exp47b_gen_kinematics.json"
    sys.path.insert(0, work + "/experiments")
    spec = importlib.util.spec_from_file_location("exp50v3", work + "/experiments/exp50v3_curvature.py")
    X = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(X)
    sys.argv = ["exp50v3", "--model", "qwen"]
    X.main()
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
print("DONE %.1fs" % (time.time() - t0))
