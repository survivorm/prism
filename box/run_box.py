#!/usr/bin/env python3
"""The box: KTL's vLLM-TPU serve (their code, their datasets) + mini-swe-agent
(their loop, their model client) + thin glue that makes a long-lived Kaggle TPU
session a self-debugging lab.

Deliberate reuse, not reinvention:
  - vLLM serve, keepalive (480min loop + heartbeats), model loading from Kaggle
    datasets (rahim3/qwen3-8-27b-bf16 weights, rahim3/qwen38-tpu-env-v5e8 XLA
    cache): ARahim3/kaggle-tpu-lab kernel/serve_qwen38.py, fetched via git
    (kernel has internet).
  - agent loop / tool protocol / limits / trajectory JSON:
    SWE-agent/mini-swe-agent (pip install mini-swe-agent).
  - model client: minisweagent LitellmModel -> OpenAI-compatible
    http://127.0.0.1:8000/v1 (the local vLLM server). No cloud.

Glue THIS file adds (the only net-new code):
  1. Auto-compact at 85% of the model's max context (max read from vLLM
     /models; threshold ~ tokens, estimated chars/4). On cross: the model
     compresses history into a STATE block, history resets to [system, STATE].
     Trajectory JSON stays complete — it is the durable record.
  2. Context pack — the goal preamble points the agent at
     box/context-pack.md: index of prior-run logs (repo-relative paths) +
     hypotheses for what may be wrong. The agent reads it first, like a
     researcher getting the lab notebook on arrival.
  3. Two-way channel — periodic git-pull of cmd/*.txt folded in as a
     NEW-INSTRUCTION turn (mid-run correction); one-way = print (Kaggle
     live-log in the browser) + trajectory JSON.
  4. pyspy — pip-installed in the box preamble; the agent uses it through the
     mini-swe bash tool (`py-spy dump --pid N`) to read the stack of a hung
     XLA process. No custom tool registration needed.
  5. Keepalive — after the agent exits, main waits out the vLLM keepalive so
     the session (XLA cache + results) stays alive.

Self-check (no TPU/vLLM): python3 run_box.py --selfcheck drives compact()
against a stub model.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

REPO = os.environ.get("BOX_REPO", "/kaggle/working/box")
KTL = f"{REPO}/kaggle-tpu-lab"
VLLM_URL = "http://127.0.0.1:8000/v1"
MODEL_NAME = "qwen3.8-27b"
KEEPALIVE_MIN = int(os.environ.get("BOX_KEEPALIVE_MIN", "480"))
COMPACTION_RATIO = float(os.environ.get("BOX_COMPACT_RATIO", "0.85"))
CONTEXT_PACK = f"{REPO}/context-pack.md"

DECISION_RULES = """
Decision rules (pre-encoded from prior sessions — follow, do not re-derive):
- Read box/context-pack.md FIRST: prior-run log index + hypotheses. Verify against the logs before forming new theories.
- Run experiments as long jobs via the background tool (or `nohup ... &`) and poll; for a job that stops producing output for >2 min, use the pyspy tool on its pid — an XLA HLO compile hang is a readable stack, not a dead log.
- Before blaming hardware/XLA, reproduce the failing op on CPU at tiny shape (the phantom-shape bugs so far were XLA-side, not prism logic).
- If [D,D] matmul or 2nd-order HVP is blocked on XLA: use the per-token [K,D] Jacobian path or the analytical last-layer form; verify numerical equivalence on CPU before trusting it on TPU.
- XLA_FLAGS must be set in the environment before the process starts torch_xla — they cannot be changed mid-process; flag experiments = separate processes.
- Consider other compute resources in the box (vLLM is a model server, not a compute pool) — but TPU chips are already partitioned: vLLM takes its share, experiments take the rest; do not double-book.
- Metric definitions are FROZEN: gap = prob-weighted (1-cos), lower=better; bridge = cos, higher=better. Change computation paths, never definitions.
- After each successful scenario: save the results/ JSON + update PLAN.md (tried / failed / next).
"""


# --------------------------------------------------------------------------- 1: KTL serve
def start_serve():
    """Fetch kaggle-tpu-lab, run its serve_qwen38.py (their code, their
    datasets mounted at /kaggle/input). Returns the Popen; it keeps vLLM alive
    for its own keepalive and heartbeats to the kernel log."""
    if not os.path.isdir(KTL):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ARahim3/kaggle-tpu-lab", KTL], check=True)
    p = subprocess.Popen([sys.executable, f"{KTL}/kernel/serve_qwen38.py"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def tail():
        for line in p.stdout:
            print(f"[vllm] {line}", end="", flush=True)
    threading.Thread(target=tail, daemon=True).start()
    return p


def wait_healthy(timeout_s=3600):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{VLLM_URL}/models", timeout=5) as r:
                if r.status == 200:
                    return json.load(r)
        except Exception:
            time.sleep(20)
    raise SystemExit("vLLM never became healthy")


def max_context(models_payload):
    for m in models_payload.get("data", []):
        if m.get("max_model_len"):
            return int(m["max_model_len"])
    return 262144  # KTL default


# --------------------------------------------------------------------------- 2: compact at 85%
def compact(agent, client=None):
    """Auto-compact when estimated history crosses COMPACTION_RATIO of the
    model's max context. The model compresses history into a STATE block;
    messages reset to [system, STATE]. `client` = the raw (unwrapped) model
    query, to avoid recursing through the glue wrapper. Returns True if
    it compacted."""
    est_tokens = sum(len(m.get("content") or "") for m in agent.messages) // 4
    if est_tokens < agent._compact_threshold:
        return False
    print(f"[compact] ~{est_tokens} tok >= {agent._compact_threshold} (85% of "
          f"{agent._compact_threshold // COMPACTION_RATIO}) — auto-compacting", flush=True)
    hist = [{"role": m.get("role"), "content": (m.get("content") or "")[:3000]}
            for m in agent.messages]
    out = (client or agent.model).query([{"role": "user", "content":
        "Compress this debugging session into a STATE block. Sections: "
        "## Tried (one line each: action -> outcome), ## Proven (verified facts, "
        "numbers kept), ## Open (hypotheses, next experiments in priority order), "
        "## State (files changed, running jobs with pids, current target). "
        "Concrete: names, values, commands, log paths. No commentary.\n\n"
        + json.dumps(hist, ensure_ascii=False)}])
    text = out.get("content")
    if isinstance(text, list):
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    text = text or str(out)
    keep = [m for m in agent.messages if m.get("role") == "system"]
    agent.messages = keep + [{"role": "user",
        "content": f"STATE (session so far, compacted):\n{text}\n\nContinue the task."}]
    print("[compact] done — history reset to system + STATE", flush=True)
    return True


# --------------------------------------------------------------------------- 3: correction channel
def maybe_correct(agent):
    """Two-way: git-pull and fold new operator corrections into the conversation."""
    try:
        subprocess.run(["git", "pull", "--ff-only"], cwd=REPO,
                       capture_output=True, timeout=60)
        lines = []
        cmd = f"{REPO}/cmd"
        if os.path.isdir(cmd):
            for fn in sorted(os.listdir(cmd)):
                if fn.endswith(".txt"):
                    lines += [l for l in open(os.path.join(cmd, fn)).read().splitlines() if l.strip()]
        if lines and agent.messages:
            agent.messages.append({"role": "user",
                                   "content": "NEW INSTRUCTION from operator:\n" + "\n".join(lines)})
            print("[correct]", lines, flush=True)
    except Exception as e:
        print(f"[correct] skip: {e}", flush=True)


# --------------------------------------------------------------------------- main
def run_box(goal):
    serve = start_serve()
    models = wait_healthy()
    thr = int(max_context(models) * COMPACTION_RATIO)
    print(f"[box] vLLM healthy (max_model_len={max_context(models)}, compact at {thr}) — starting agent", flush=True)

    from minisweagent.agents import get_agent
    from minisweagent.environments import get_environment
    from minisweagent.models import get_model

    model = get_model("litellm", model=f"openai/{MODEL_NAME}",
                      model_kwargs={"api_base": VLLM_URL, "api_key": "kaggle-local"})
    env = get_environment("local")
    agent = get_agent("default", model=model, environment=env,
                      mode="yolo", step_limit=0)
    agent.config.output_path = f"{REPO}/traj.json"
    agent._compact_threshold = thr

    # Wrap the model call: after each turn, auto-compact (85%) + check for
    # operator corrections. Mini-swe has no per-step hook; wrapping query is
    # the least-invasive hook point (their loop, our glue).
    inner_query = model.query

    def query_with_glue(messages, **kw):
        r = inner_query(messages, **kw)
        try:
            compact(agent, inner_query)
            maybe_correct(agent)
        except Exception as e:
            print(f"[glue] {e}", flush=True)
        return r

    model.query = query_with_glue

    preamble = (goal
                + f"\n\nFirst action: `read` {CONTEXT_PACK} and follow its log index."
                + "\nWork maximally efficiently; debug via pyspy; results + PLAN.md after each success.")
    try:
        agent.run(task=preamble)
        print("[box] agent finished — keeping the session alive for the rest of keepalive", flush=True)
    finally:
        t0 = time.time()
        while serve.poll() is None and time.time() - t0 < KEEPALIVE_MIN * 60:
            time.sleep(120)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        # No TPU/vLLM: prove compact() fires at threshold and resets history.
        import types

        class CModel:
            def get_template_vars(self):
                return {}

            def query(self, messages, **kw):
                return {"content": "## Proven: stub fact\n## Next: stub step"}

        agent = types.SimpleNamespace(
            messages=[{"role": "system", "content": "You are an agent"},
                      {"role": "user", "content": "x" * 2000},
                      {"role": "assistant", "content": "y" * 2000}],
            model=CModel(),
        )
        agent._compact_threshold = 600  # (2000+2000)/4 = 1000 est tokens > 600 => fires
        assert compact(agent) is True
        assert len(agent.messages) == 2
        assert "STATE (session so far, compacted)" in agent.messages[1]["content"]
        assert compact(agent) is False  # below threshold now
        print("SELF-CHECK OK — auto-compact fires at 85% threshold, resets to [system, STATE], stays quiet below.")
    else:
        goal = sys.argv[1] if len(sys.argv) > 1 else \
            "Produce results/exp16_readout_pull.json on this TPU (see context pack for the XLA blockers and workarounds)."
        sys.exit(run_box(goal))
