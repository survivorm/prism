# ---------- driver ----------
# E-52k driver (DRIVER MODE, инлайн-бандл, без датасетов).
# Кривая τ(size) на GPU bf16: 0.8B → 2B → 4B (+повтор 0.8B как якорь протокола)
# и честный PTQ с HF-весов: bnb-int8 на 4B и 9B (7B в семье Qwen3.5 нет — 9B
# и есть «третий-крупный»; bf16-9B в T4 не влезает, int8 влезает).
import os, sys, time, glob, json, shutil, traceback, gc, subprocess
t0 = time.time()
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["PRISM_SKIP_NORM_TEST"] = "1"  # семантика g проверена локально (E-47b)
os.environ["HF_HOME"] = "/kaggle/working/.hf"          # staging под local_dir — в working, не в home
os.environ["HF_HUB_CACHE"] = "/kaggle/working/.hf/hub"
CODE = "/kaggle/working/code"
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "experiments"))
try:
    subprocess.run(["pip", "install", "-q", "-U", "transformers", "bitsandbytes"], check=False)
    from huggingface_hub import snapshot_download
    import importlib.util, torch

    def complete(p):
        import re
        ss = [f for f in os.listdir(p) if f.endswith(".safetensors")]
        if not ss:
            return False
        mt = [re.search(r"-(\d+)-of-(\d+)\.safetensors$", f) for f in ss]
        if any(x is None for x in mt):
            return True
        tot = {int(x.group(2)) for x in mt}
        have = {int(x.group(1)) for x in mt}
        return len(tot) == 1 and have == set(range(1, tot.pop() + 1))

    def load(mod, fn):
        spec = importlib.util.spec_from_file_location(mod, os.path.join(CODE, "experiments", fn))
        X = importlib.util.module_from_spec(spec)
        sys.modules[mod] = X
        spec.loader.exec_module(X)
        return X
    E47 = load("e47", "exp47b_gen_kinematics.py")
    # PRISM-каналы: jcache синтезируем из литерала U128 (бинарник не нужен)
    d08 = "/kaggle/working/m_Qwen3.5-0.8B"
    if not glob.glob(d08 + "/*.safetensors"):
        prev = [p for p in glob.glob("/kaggle/input/**/m_Qwen3.5-0.8B", recursive=True) if complete(p)]
        if prev:
            d08_src = prev[0]
        else:
            d08_src = snapshot_download("Qwen/Qwen3.5-0.8B", local_dir=d08)
    os.environ["PRISM_QWEN_DIR"] = d08 if glob.glob(d08 + "/*.safetensors") else d08_src
    jc = "/kaggle/working/U128_synth.pt"
    torch.save({"U128": list(E47.U128)}, jc)
    os.environ["PRISM_JCACHE"] = jc
    print("[env] QWEN_DIR:", os.environ["PRISM_QWEN_DIR"], flush=True)
    E52 = load("e52", "exp52a_perturb_axis.py")   # после env (load_model читает при вызове)
    JOBS = [  # (tag, hf_repo, dtype, bnb-int8, последний заход за этой моделью)
        ("q08_bf16", "Qwen/Qwen3.5-0.8B", "bfloat16", False, True),
        ("q2_bf16",  "Qwen/Qwen3.5-2B",   "bfloat16", False, True),
        ("q4_bf16",  "Qwen/Qwen3.5-4B",   "bfloat16", False, False),
        ("q4_int8",  "Qwen/Qwen3.5-4B",   "bfloat16", True,  True),
        ("q9_int8",  "Qwen/Qwen3.5-9B",   "bfloat16", True,  True),
    ]
    for tag, repo, dt, q8, last in JOBS:
        try:
            name = repo.split("/")[-1]
            md = f"/kaggle/working/m_{name}"
            if not glob.glob(md + "/*.safetensors"):
                prev = [p for p in glob.glob(f"/kaggle/input/**/m_{name}", recursive=True) if complete(p)]
                md = prev[0] if prev else snapshot_download(repo, local_dir=md)
            rd = f"/kaggle/working/r52k_{tag}"; os.makedirs(rd, exist_ok=True)
            print(f"\n##### {tag} ({repo}, {dt}{', bnb-int8' if q8 else ''})", flush=True)
            E47.main(model=md, n_docs=4, max_new=160, device="cuda:0", results_dir=rd,
                     do_verify_basis=False, dtype=dt, load_in_8bit=q8)
            sys.argv = ["e52a", "--model", "qwen", "--n-docs", "16", "--model-dir", md,
                        "--families", "rtn",
                        "--npz", rd + "/exp47b_gen_kinematics.npz",
                        "--docs-json", rd + "/exp47b_gen_kinematics.json",
                        "--out", f"/kaggle/working/exp52k_tau_{tag}.json"]
            E52.main()
            if last and name != "Qwen3.5-0.8B" and md.startswith("/kaggle/working"):
                shutil.rmtree(md, ignore_errors=True)   # освобождаем диск под следующий размер
            shutil.rmtree("/kaggle/working/.hf", ignore_errors=True)  # staging-дубли снапшотов
            free = shutil.disk_usage("/kaggle/working").free / 2**30
            print(f"[disk] free {free:.1f} GiB", flush=True)
            gc.collect(); torch.cuda.empty_cache()
        except Exception:
            print(f"##### {tag} FAILED", flush=True)
            traceback.print_exc()
    print("\n===== SUMMARIES =====", flush=True)
    for f in sorted(glob.glob("/kaggle/working/exp52k_tau_*.json")):
        s = json.load(open(f))["summary"]
        print(f.split("exp52k_tau_")[1], "base:", s["base"]["tau"], flush=True)
except Exception:
    traceback.print_exc()
print("DONE %.1fs" % (time.time() - t0))
