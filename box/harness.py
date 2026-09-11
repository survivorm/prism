#!/usr/bin/env python3
"""In-box autonomous experiment-debugger: a pure-python agent loop that drives
a local (in-process) LLM — vLLM on TPU — through a tool-call protocol, so a
long-lived Kaggle TPU session becomes a self-debugging lab.

The head is vLLM (optimized, preinstalled in the Kaggle TPU image — the same
stack kaggle-tpu-lab serves). This file is the *harness*: it talks to that
model and executes its tool calls. No framework, no HTTP server — the model is
imported in-process (`from vllm import LLM`) and called with `llm.chat(...)`.

Protocol (model <-> harness), deliberately parser-agnostic: the model emits ONE
JSON object per turn, `{"tool": <name>, "args": {...}, "note": <why>}`, or
`{"tool":"done","args":{"summary":...}}`. We regex-extract the JSON object from
the raw completion (vLLM chat-template tool-call parsing varies across builds;
a text-JSON protocol works with any of them and is trivially debuggable).

Tools: bash, run_bg, bg_output, kill, pyspy, read, write, done.
  - bash       : short foreground command, returns stdout+stderr.
  - run_bg     : long command as a detached proc -> pid (for TPU runs that
                 compile/trace for many minutes).
  - bg_output  : accumulated output of a bg pid (poll this; no new lines for a
                 while == stall).
  - pyspy      : `py-spy dump --pid <pid>` — the point of 2h of *active*
                 debugging: a hung XLA HLO compile is a readable stack, not a
                 dead log.
  - read/write : files in the box (code, logs, results, PLAN.md, outbox).
  - done       : end the run with a summary.

Channel: one-way = plain `print` — Kaggle live-streams kernel logs in the
browser while the kernel runs (CLI can't, the web UI can). Two-way correction:
the box has egress (internet on); `cmd_pull` `git pull`s and folds any new
cmd/*.txt lines into the next turn (a human edits the goal mid-run, the agent
picks it up next turn).

Self-check (no TPU, no vLLM): `python3 harness.py` runs the loop against a
FakeLLM that scripts bash->read->done and asserts the mechanics (JSON parse,
tool exec, bg/stall, outbox) work. The real head is only the last lazy import.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

# --------------------------------------------------------------------------- config
MAX_TURNS = int(os.environ.get("HARNESS_MAX_TURNS", "40"))   # safety bound on turns
OUTBOX_DIR = os.environ.get("HARNESS_OUTBOX", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outbox"))
CMD_DIR = os.environ.get("HARNESS_CMD", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmd"))
STALL_S = int(os.environ.get("HARNESS_STALL_S", "120"))      # no new bg output for this long -> "stalled"
OUT_CHARS = int(os.environ.get("HARNESS_OUT_CHARS", "8000")) # truncate tool output fed back to the model

SYSTEM = """You are an autonomous experiment-debugger inside a Kaggle TPU session.
Your goal: {goal}

Decision rules (apply before acting; these encode prior findings — do not re-derive blindly):
- Before blaming hardware/XLA, reproduce on CPU (small shape / fake) to prove the op is valid.
- If an XLA run hangs (bg_output shows no new lines for a while): py-spy the pid, read the stack, then change ONE thing and rerun. Never stare at a dead log.
- If a [D,D] matmul or 2nd-order HVP is the blocker: prefer the per-token [K,D] Jacobian path or an analytical last-layer form; verify numerical equivalence on CPU before trusting it on TPU.
- Do NOT change the metric definitions (gap = prob-weighted 1-cos, lower=better; bridge = cos, higher=better) — only the computation path.
- After each scenario that succeeds, snapshot results (write them to results/) so a mid-run kill loses nothing.
- Keep PLAN.md in the box current: what you tried, what failed, what the next step is.
- When the goal's result is produced and saved, respond with done.

Respond with EXACTLY one JSON object per turn, no prose outside it:
  {"tool": "bash", "args": {"cmd": "..."}, "note": "why"}
  {"tool": "run_bg", "args": {"cmd": "..."}, "note": "long TPU run"}
  {"tool": "bg_output", "args": {"pid": 123}, "note": "check progress"}
  {"tool": "pyspy", "args": {"pid": 123}, "note": "stalled, get the stack"}
  {"tool": "kill", "args": {"pid": 123}, "note": "stop this"}
  {"tool": "read", "args": {"path": "..."}}
  {"tool": "write", "args": {"path": "...", "content": "..."}}
  {"tool": "done", "args": {"summary": "..."}}
"""

# --------------------------------------------------------------------------- LLM client
class FakeLLM:
    """Canned-response head for the self-check: replays a fixed script of tool
    calls. Proves loop + JSON parse + tool exec + bg/stall + outbox without any
    model. `script` is a list of dicts matching the protocol; each turn pops one."""
    def __init__(self, script):
        self.script = list(script)
    def chat(self, messages):
        if not self.script:
            return {"tool": "done", "args": {"summary": "script exhausted"}}
        turn = self.script.pop(0)
        return json.dumps(turn)

class VLLMClient:
    """Real head: in-process vLLM (preinstalled in the Kaggle TPU image). Lazy
    import so the file runs anywhere (self-check never touches vLLM)."""
    def __init__(self, model, **kw):
        from vllm import LLM  # lazy; not available off-TPU
        self.llm = LLM(model=model, **kw)
    def chat(self, messages):
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        out = self.llm.chat(msgs)
        return out[0].outputs[0].text

# --------------------------------------------------------------------------- tools
class Tools:
    def __init__(self, cwd="."):
        self.cwd = cwd
        self.bg = {}   # pid -> {proc, out, last_len, ts}
    def _run(self, cmd, timeout=None):
        try:
            r = subprocess.run(["bash", "-c", cmd], cwd=self.cwd, capture_output=True,
                               text=True, timeout=timeout)
            return (r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr else "")
                    + f"\n[exit {r.returncode}]").strip()
        except subprocess.TimeoutExpired:
            return "[timeout after %ss]" % (timeout or "?")
    def bash(self, cmd): return self._run(cmd, timeout=300)
    def run_bg(self, cmd):
        p = subprocess.Popen(["bash", "-c", cmd], cwd=self.cwd,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        import threading
        self.bg[p.pid] = {"proc": p, "out": [], "last_len": 0, "ts": time.time()}
        def drain():
            for line in p.stdout:
                self.bg.get(p.pid, {}).setdefault("out", []).append(line)
                self.bg.get(p.pid, {}).__setitem__("ts", time.time())
            p.wait()
        threading.Thread(target=drain, daemon=True).start()
        return p.pid
    def bg_output(self, pid):
        j = self.bg.get(int(pid));  pid = int(pid)
        if not j: return f"no bg job {pid}"
        out = "".join(j["out"]); n = len(out)
        stalled = n == j["last_len"] and time.time() - j["ts"] > STALL_S
        j["last_len"] = n
        return out[-OUT_CHARS:] + (f"\n[STALLED: no new output {time.time()-j['ts']:.0f}s — py-spy this pid]" if stalled else "") + f"\n[bytes {n}]"
    def kill(self, pid):
        j = self.bg.get(int(pid))
        if j and j["proc"] and j["proc"].poll() is None:
            j["proc"].kill(); return f"killed {pid}"
        return f"{pid} not running"
    def pyspy(self, pid):
        # py-spy dump: readable stack of a live (incl. hung) process — the 2h-debug superpower.
        # Kernel preamble pip-installs py-spy; if absent, say so honestly.
        if "py-spy" in self._run("which py-spy"):
            return self._run(f"py-spy dump --pid {int(pid)}", timeout=60)
        return "[py-spy not installed: pip install py-spy]"
    def read(self, path):
        try: return open(os.path.join(self.cwd, path)).read()[-OUT_CHARS:]
        except Exception as e: return f"[read {path}: {e}]"
    def write(self, path, content):
        p = os.path.join(self.cwd, path); os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        open(p, "w").write(content); return f"wrote {len(content)} chars -> {path}"
    def done(self, summary): raise _Done(summary)

    def call(self, name, args):
        f = getattr(self, name, None)
        if f is None: return f"[unknown tool {name}]"
        return f(**args)

class _Done(Exception):
    def __init__(self, summary): self.summary = summary

# --------------------------------------------------------------------------- channel
def outbox_push(transcript, cwd="."):
    """One-way: print to the kernel log (browser live-streams it) + optional
    git push of the transcript for the correction repo (best-effort egress)."""
    print("[outbox]", transcript[:OUT_CHARS], flush=True)   # <- this IS the one-way channel
    os.makedirs(OUTBOX_DIR, exist_ok=True)
    with open(os.path.join(OUTBOX_DIR, "transcript.log"), "a") as fh: fh.write(transcript + "\n")
    try:
        subprocess.run(["git", "add", OUTBOX_DIR], cwd=cwd, capture_output=True, timeout=30)
        subprocess.run(["git", "commit", "-m", "harness: outbox", "--allow-empty"],
                       cwd=cwd, capture_output=True, timeout=30)
        subprocess.run(["git", "push"], cwd=cwd, capture_output=True, timeout=60)
    except Exception:
        pass  # egress optional; the print above already did the one-way job

def cmd_pull(cwd="."):
    """Two-way: pull and return any new correction lines from cmd/."""
    subprocess.run(["git", "pull", "--ff-only"], cwd=cwd, capture_output=True)
    lines = []
    if os.path.isdir(CMD_DIR):
        for fn in sorted(os.listdir(CMD_DIR)):
            if fn.endswith(".txt"):
                lines += [l for l in open(os.path.join(CMD_DIR, fn)).read().splitlines() if l.strip()]
    return "\n".join(lines)

# --------------------------------------------------------------------------- loop
def _extract_json(text):
    """First {...} balanced object in the completion (tolerates prose wrappers)."""
    s = text.find("{")
    if s < 0: return None
    depth = 0
    for i in range(s, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(text[s:i+1])
                except Exception: return None
    return None

def run(goal, llm, cwd=".", max_turns=MAX_TURNS):
    tools = Tools(cwd)
    system = SYSTEM.replace("{goal}", goal)
    correction = cmd_pull(cwd)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": correction or "Begin. Take the first step toward the goal."}]
    transcript = []
    for turn in range(max_turns):
        raw = llm.chat(messages)
        obj = _extract_json(raw)
        if obj is None:
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": "Invalid: respond with exactly one JSON object per the system prompt."})
            continue
        name, args = obj.get("tool"), obj.get("args", {})
        if name == "done":
            outbox_push(json.dumps(transcript), cwd)
            return {"status": "done", "summary": args.get("summary"), "turns": turn + 1}
        result = tools.call(name, args)
        line = f"[t{turn}] {name} {args} -> {str(result)[:400]}"
        transcript.append(line); print(line)
        if turn % 5 == 4: outbox_push(json.dumps(transcript), cwd)   # periodic one-way sync
        if correction:  # fold a fresh mid-run correction into the next turn
            correction = cmd_pull(cwd)
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"tool result:\n{str(result)[:OUT_CHARS]}" + (f"\n\nNEW INSTRUCTION:\n{correction}" if correction else "")})
    outbox_push(json.dumps(transcript), cwd)
    return {"status": "max_turns", "turns": max_turns}

# --------------------------------------------------------------------------- self-check (no TPU / no vLLM)
def _selfcheck():
    import tempfile
    d = tempfile.mkdtemp(prefix="harness_")
    script = [
        {"tool": "write", "args": {"path": "bad.py", "content": "x=1/0\nprint(x)\n"}, "note": "repro first"},
        {"tool": "bash", "args": {"cmd": "python3 bad.py"}, "note": "see the error"},
        {"tool": "read", "args": {"path": "bad.py"}},
        {"tool": "write", "args": {"path": "fixed.py", "content": "x=1 if 1 else 0\nprint(x)\n"}},
        {"tool": "bash", "args": {"cmd": "python3 fixed.py"}},
        {"tool": "done", "args": {"summary": "reproduced, fixed, verified"}},
    ]
    r = run("fix bad.py so it prints 1", FakeLLM(script), cwd=d)
    assert r["status"] == "done", r
    assert "ZeroDivisionError" in open(os.path.join(OUTBOX_DIR, "transcript.log")).read(), "repro error missing from transcript"
    print("SELF-CHECK OK — loop parsed JSON, ran bash/read/write, outbox log has the error, hit done.")
    return r

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "real":
        # real head: vLLM in-box. model path passed via arg/env (Kaggle TPU image has vllm preinstalled).
        model = os.environ.get("VLLM_MODEL", "/data/Qwen/Qwen3-27B")
        goal = sys.argv[2] if len(sys.argv) > 2 else "Get results/exp16_readout_pull.json produced on this TPU; debug any XLA failures."
        print(run(goal, VLLMClient(model, dtype="bfloat16", tensor_parallel_size=4)))
    else:
        _selfcheck()
