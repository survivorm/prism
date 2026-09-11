"""gen_box.py — build the autonomous prism-agent box as a Kaggle TPU script kernel.

The box materializes the prism repo on disk (inlined as base64, so no private-gitlab
clone needed), installs miniswe-agent + py-spy, clones KTL, boots the Qwen3.8-27B
vLLM-TPU head, and runs the BoxAgent (run_box.py run) which runs+debugs the
TPU-blocked causal experiments with 85% auto-compact.

Usage (from prism repo root):
    .venv/bin/python kaggle_kernel/gen_box.py \
        --out kaggle_kernel/kernel-box --name box_autonomer
    # then: kaggle kernels push -p kaggle_kernel/kernel-box
"""
import base64, os, sys, argparse, json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def embed(relpath):
    p = os.path.join(REPO, relpath)
    return base64.b64encode(open(p, "rb").read()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "kernel-box"))
    ap.add_argument("--name", default="box_autonomer")
    ap.add_argument("--datasets", default="rahim3/qwen3-8-27b-bf16,rahim3/qwen38-tpu-env-v5e8")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)

    # files to materialize into /kaggle/working/prism/ : ALL prism modules (avoids
    # missing transitive imports) + the TPU-blocked exps + the box harness.
    import glob as _g
    files = {}
    for fp in sorted(_g.glob(os.path.join(REPO, "prism", "*.py"))):
        files["prism/" + os.path.basename(fp)] = os.path.relpath(fp, REPO)
    for e in ("exp16_readout_pull", "exp17_lora_distill", "exp18_logit_kd"):
        rel = "experiments/" + e + ".py"
        files[rel] = rel
    files["box/run_box.py"] = "box/run_box.py"
    files["box/context-pack.md"] = "box/context-pack.md"
    embedded = {dst: embed(src) for dst, src in files.items()}

    emb_json = json.dumps(embedded, indent=0)  # relpath -> b64

    # the inlined kernel .py
    kernel = '''"""autonomer-box — persistent autonomous prism experiment agent on Kaggle TPU.
Head: Qwen3.8-27B vLLM-TPU (KTL). Loop: mini-swe-agent (BoxAgent, 85% auto-compact).
Diag: py-spy. Channel: git cmd/->outbox. Session: 9h keepalive (defeats 2h idle-kill).
"""
import base64, json, os, sys, time, subprocess, threading

PRISM = "/kaggle/working/prism"
KAGGLE_WORK = "/kaggle/working"

# ---- 1. materialize the prism repo from the inlined payload ----
EMBEDDED = __EMBEDDED__

def materialize():
    os.makedirs(PRISM, exist_ok=True)
    for rel, b64 in EMBEDDED.items():
        dst = os.path.join(PRISM, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(base64.b64decode(b64))
    print("[box] materialized " + str(len(EMBEDDED)) + " prism files into " + PRISM, flush=True)

# ---- 2. install agent deps ----
def install():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "miniswe-agent==2.4.6", "py-spy"], check=False)
    print("[box] installed miniswe-agent + py-spy", flush=True)

# ---- 3. clone KTL (vLLM head) ----
def clone_ktl():
    ktl = os.path.join(PRISM, "ktl")
    if not os.path.isdir(os.path.join(ktl, "kernel")):
        subprocess.run(["git", "clone", "--depth", "1",
                        os.environ.get("KTL_REPO", "https://github.com/ARahim3/kaggle-tpu-lab"),
                        ktl], check=False)
    print("[box] KTL ready at " + ktl, flush=True)
    return ktl

# ---- 4. keepalive (defeats the 2h idle-kill; 9h guard) ----
def keepalive(max_hours=9):
    stop = {"v": False}
    def loop():
        t0 = time.time(); i = 0
        while not stop["v"] and time.time() - t0 < max_hours * 3600:
            i += 1
            print("[box-keepalive] t=%ds alive=%d" % (int(time.time()-t0), i), flush=True)
            time.sleep(5)
    threading.Thread(target=loop, daemon=True).start()
    return stop

def main():
    t0 = time.time()
    print("=== PRISM AUTONOMER BOX: start ===", flush=True)
    stop = keepalive()
    materialize()
    install()
    clone_ktl()
    # run_box.py boots vLLM (serve) and then runs the BoxAgent
    env = dict(os.environ, VLLM_MODEL="Qwen3.8-27B",
               TP=os.environ.get("TP", "8"))
    if os.environ.get("TP", "8") == "4":
        # chip-partition coexistence: vLLM on chips 0-3, prism on 4-7
        env["VLLM_CHIPS"] = "0,1,2,3"
    r = subprocess.run([sys.executable, os.path.join(PRISM, "box", "run_box.py"),
                        "serve"], env=env, cwd=PRISM)
    stop["v"] = True
    print("=== BOX done in %.0fs (agent exit rc=%d) ===" % (time.time()-t0, r.returncode),
          flush=True)

if __name__ == "__main__":
    try:
        main()
        print("OK")
    except Exception as e:
        import traceback; traceback.print_exc()
        print("[FAIL] " + str(e))
'''
    kernel = kernel.replace("__EMBEDDED__", emb_json)

    with open(os.path.join(a.out, a.name + ".py"), "w") as f:
        f.write(kernel)

    meta = {
        "id": "voroninmichael/prism-" + a.name,
        "title": "Prism: " + a.name,
        "code_file": a.name + ".py",   # script kernels need the source-file name
        "code_only": False,
        "kernel_type": "script",
        "language": "python",
        "enable_gpu": "false",
        "enable_tpu": "true",
        "enable_internet": "true",
        "is_private": "false",
        "dataset_sources": a.datasets.split(","),
    }
    with open(os.path.join(a.out, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)

    sz = os.path.getsize(os.path.join(a.out, a.name + ".py"))
    print("built %s" % os.path.join(a.out, a.name + ".py"))
    print("  size: %d bytes (%d KB)" % (sz, sz // 1024))
    print("  datasets: %s" % a.datasets)
    print("  metadata: TPU script kernel, internet on")
    if sz > 512 * 1024:
        print("  !! > 512KB — Kaggle script-kernel limit; drop files from EMBEDDED")
    print("\npush:  kaggle kernels push -p " + a.out)


if __name__ == "__main__":
    main()
