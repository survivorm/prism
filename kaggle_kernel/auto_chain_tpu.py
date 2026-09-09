#!/usr/bin/env python3
"""Auto-chain TPU causal experiments on the single-slot Kaggle free-tier TPU.

Kaggle allows only 1 concurrent batch TPU session, so kernels cannot be
queued back-to-back — each must be pushed the moment the previous terminates.
This chain: poll exp18-tpu until terminal, wait for the slot to release,
push exp19-tpu, poll it until terminal, print a combined verdict.

Usage: python3 auto_chain_tpu.py [poll_sec]
  (default poll_sec=45). Logs to auto_chain_tpu.log next to itself.
"""
import subprocess, time, re, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
E18 = "voroninmichael/prism-exp18-logit-kd-tpu"
E19 = "voroninmichael/prism-exp19-destruction-tpu"
POLL = int(sys.argv[1]) if len(sys.argv) > 1 else 45

def kaggle(*args):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=60)
    return (r.stdout or "") + (r.stderr or "")

def status(slug):
    out = kaggle("kernels", "status", slug)
    m = re.search(r'(KernelWorkerStatus\.\w+)', out)
    return m.group(1) if m else "UNKNOWN"

def is_terminal(st):
    return st in ("KernelWorkerStatus.COMPLETE", "KernelWorkerStatus.ERROR",
                  "KernelWorkerStatus.CANCEL")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)

def wait_terminal(slug, label):
    log(f"polling {label} ({slug})")
    for i in range(600):  # ~7.5h max at 45s
        st = status(slug)
        if is_terminal(st):
            log(f"{label} TERMINAL: {st}")
            return st
        if i % 8 == 0:  # log every ~6 min to avoid spam
            log(f"{label} still {st}")
        time.sleep(POLL)
    log(f"{label} still not terminal after 600 polls; giving up")
    return "UNKNOWN"

def wait_slot_open(label, tries=12):
    """Kaggle holds the TPU slot ~3-9 min after a kernel completes. Retry push
    until it's accepted (no 'session count of 1' / no 404)."""
    for t in range(tries):
        out = kaggle("kernels", "push", "-p", "kernel-exp19-tpu")
        if "Maximum batch TPU session count of 1" in out:
            log(f"{label}: slot still busy (try {t+1}/{tries}), waiting 90s")
            time.sleep(90)
            continue
        if "pushed" in out or "successfully" in out.lower():
            log(f"{label}: pushed OK")
            return True
        # 404 / other: the kernel may exist; treat as pushed and verify by status
        log(f"{label}: push output: {out.strip()[:200]}")
        return True
    log(f"{label}: could not push after {tries} tries")
    return False

def main():
    log("=== TPU auto-chain start ===")
    # 1) wait for exp18-tpu to finish
    st18 = wait_terminal(E18, "exp18-tpu")
    if st18 != "KernelWorkerStatus.COMPLETE":
        # Still push exp19? No — if exp18 errored, exp19 likely shares the
        # same root cause. Stop and let the human decide.
        log(f"exp18-tpu did not COMPLETE ({st18}). NOT auto-pushing exp19 "
            f"(likely shared root cause). Investigate before chaining.")
        log("=== chain ABORTED (exp18 not COMPLETE) ===")
        return
    # 2) slot release, then push exp19
    log("waiting for TPU slot to release, then pushing exp19-tpu...")
    if not wait_slot_open("exp19-tpu push"):
        log("=== chain ABORTED (exp19 push failed) ===")
        return
    # 3) wait for exp19 to finish
    st19 = wait_terminal(E19, "exp19-tpu")
    log("=== chain DONE ===")
    log(f"  exp18-tpu: {st18}")
    log(f"  exp19-tpu: {st19}")
    log("Next: fetch results")
    log("  kaggle kernels output voroninmichael/prism-exp18-logit-kd-tpu -w -o --file-pattern results/")
    log("  kaggle kernels output voroninmichael/prism-exp19-destruction-tpu -w -o --file-pattern results/")

if __name__ == "__main__":
    main()
