"""run_box.py — persistent autonomous prism-experiment agent on Kaggle TPU.

Thin glue over EXISTING tools (no reinvention):
  * KTL (kaggle-tpu-lab) serve_qwen38.py  -> Qwen3.8-27B vLLM-TPU head (OpenAI endpoint)
  * mini-swe-agent (SWE-agent)            -> agent loop (DefaultAgent + LocalEnvironment)
  * py-spy                                -> TPU/XLA hang diagnostics
  * prism git repo (cloned here)          -> the experiments the agent edits & runs

The box is a Kaggle TPU session (9h hard cap; keepalive below defeats the 2h
idle-kill). vLLM serves on the TPU; this script runs in the Kaggle system python
and drives the agent against vLLM over HTTP.

MODES:
  selfcheck  (default; NO vLLM/TPU needed) — pure test of the 85% auto-compact
              + mini-swe import. Run this first, locally, to prove the harness.
  serve      — clone KTL, boot vLLM-TPU head, health-check, then run the agent.
  run        — assume vLLM already up at VLLM_URL; just run the agent.

  python run_box.py selfcheck
  python run_box.py serve  [--task "..."]
  python run_box.py run    --task "..."

Env:
  VLLM_URL     (default http://127.0.0.1:8000/v1)
  VLLM_MODEL   (default Qwen3.8-27B)
  VLLM_API_KEY (default not-needed)
  TP           (default 8; set 4 for chip-partition coexistence with prism)
  KTL_REPO     (default https://github.com/ARahim3/kaggle-tpu-lab)
  GOAL_PROMPT  (default: run+debug the TPU-blocked prism experiments)
"""
import os, sys, time, subprocess, argparse, urllib.request, json

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

try:
    from minisweagent.agents.default import DefaultAgent as _DA
    _HAS_MSWE = True
except Exception:
    _DA = object
    _HAS_MSWE = False


# ----------------------------------------------------------------------------
# 85% auto-compaction — pure core (testable without any model/TPU)
# ----------------------------------------------------------------------------
def _msg_tokens(m):
    return sum(len(str(m.get(k, ""))) // 4 + 1 for k in ("role", "content", "name"))


def _msg_text(m):
    c = m.get("content", "")
    if isinstance(c, list):
        c = " ".join(str(x) for x in c)
    return str(c)


def compact(messages, max_tokens, threshold=0.85, recent=8, summarize=None):
    """Compact `messages` when prompt_tokens exceeds threshold*max_tokens.
    summarize(text) -> str (default: structural marker, no model call).
    Preserves messages[0] (system). Returns (new_messages, compacted:bool)."""
    total = sum(_msg_tokens(m) for m in messages)
    if total < threshold * max_tokens:
        return messages, False
    system = messages[:1]
    recent_n = min(recent, max(0, len(messages) - 1))
    head, recent = (messages[1:-recent_n], messages[-recent_n:]) if recent_n else (messages[1:], [])
    if summarize is None:
        summarize = lambda _t: "[context compacted: earlier turns summarized/omitted]"
    summary = summarize(_msg_text(m) for m in head)
    new = system + [{"role": "assistant", "content": f"[summary] {summary}",
                     "extra": {"compaction": True}}] + recent
    return new, True


class BoxAgent(_DA):
    """mini-swe DefaultAgent + 85% auto-compact + git command channel."""
    def __init__(self, model, env, max_tokens, **cfg):
        if not _HAS_MSWE:
            raise RuntimeError("mini-swe-agent not installed (pip install mini-swe-agent==2.4.6)")
        super().__init__(model, env, **cfg)
        self.box_max_tokens = int(max_tokens)

    def _maybe_compact(self):
        last = self.messages[-1]
        pt = (last.get("extra") or {}).get("usage", {}).get("prompt_tokens")
        if pt and pt > self.box_max_tokens * 0.85:
            def summarize(gen):
                try:
                    out = self.model.query([
                        {"role": "system",
                         "content": "Summarize an ML-experiment agent's work-so-far into a tight "
                                     "self-contained recap it can continue from. Keep file paths, "
                                     "commands, decisions, errors, open questions. <=1200 chars."},
                        {"role": "user", "content": "\n".join(gen)},
                    ])
                    return _msg_text(out)
                except Exception:
                    return None
            new, did = compact(self.messages, self.box_max_tokens,
                               summarize=(lambda g: summarize(g)))
            if did:
                self.messages = new
                print(f"[box] compacted: {pt} -> {sum(_msg_tokens(m) for m in new)}", flush=True)

    def _pull_commands(self):
        cmd = os.path.join(REPO, "box", "cmd")
        out = []
        if os.path.isdir(cmd):
            for f in sorted(os.listdir(cmd)):
                if f.endswith(".txt"):
                    out.append(f"NEW-INSTRUCTION (operator): {open(os.path.join(cmd, f)).read().strip()}")
        return out

    def run(self, task="", **kw):
        self.extra_template_vars |= {"task": task, **kw}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system",
                                      content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user",
                                      content=self._render_template(self.config.instance_template)),
        )
        while True:
            try:
                self.step()
                self._maybe_compact()
            except Exception:
                self.handle_uncaught_exception(sys.exc_info()[1])
                break
            if self.messages[-1].get("role") == "exit":
                break
            for note in self._pull_commands():
                self.add_messages({"role": "user", "content": note, "extra": {"command": True}})
        return self.messages[-1].get("extra", {})


# ----------------------------------------------------------------------------
# selfcheck — prove compaction + harness locally, no vLLM/TPU
# ----------------------------------------------------------------------------
def selfcheck():
    print("== selfcheck: 85% auto-compact (pure, no model) ==")
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(60):
        msgs.append({"role": "user", "content": f"turn {i} " + "x" * 2000})
        msgs.append({"role": "assistant", "content": f"ok {i} " + "y" * 2000})
    max_t = 3000
    before = sum(_msg_tokens(m) for m in msgs)
    calls = []
    def mock_sum(gen):
        calls.append(1)
        return "SUMMARY: did turns 0..N; current focus: X; open: Y"
    new, did = compact(msgs, max_t, summarize=mock_sum)
    assert did, "should have compacted (total over threshold)"
    assert len(calls) == 1, "summarizer called exactly once"
    assert new[0]["role"] == "system", "system preserved"
    assert any(m.get("extra", {}).get("compaction") for m in new), "summary marker present"
    assert new[-1]["content"].startswith("ok 59"), "recent tail preserved"
    assert sum(_msg_tokens(m) for m in new) < before * 0.9
    small = compact([{"role": "system", "content": "sys"},
                     {"role": "user", "content": "hi"}], max_t, summarize=mock_sum)
    assert small[1] is False, "small context should NOT compact"
    print(f"   before={before}  after={sum(_msg_tokens(m) for m in new)}  (compacted x{len(calls)})")
    print("== selfcheck: mini-swe import ==")
    if _HAS_MSWE:
        import minisweagent
        print(f"   mini-swe {getattr(minisweagent, '__version__', '?')} OK; BoxAgent(DefaultAgent) ready")
        print("   (real agent needs a vLLM endpoint; serve/run mode wires it)")
    else:
        print("   mini-swe NOT installed here; pure compaction still validated above")
    print("== selfcheck: PASS ==")


# ----------------------------------------------------------------------------
# vLLM head (KTL) + health
# ----------------------------------------------------------------------------
def wait_health(url, timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/health", timeout=10) as r:
                if r.status == 200:
                    print(f"[box] vLLM healthy after {int(time.time()-t0)}s", flush=True)
                    return True
        except Exception:
            pass
        time.sleep(10)
    return False


def boot_vllm(kaggle_dir="/kaggle/working/ktl", tp=8):
    url = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")
    if wait_health(url.replace("/v1", ""), 10):
        return url
    ktl = os.path.join(REPO, "ktl")
    if not os.path.isdir(os.path.join(ktl, "kernel")):
        print(f"[box] cloning KTL {os.environ.get('KTL_REPO','https://github.com/ARahim3/kaggle-tpu-lab')}", flush=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        os.environ.get("KTL_REPO", "https://github.com/ARahim3/kaggle-tpu-lab"), ktl], check=True)
    # Merged over KTL's DEFAULTS: keepalive 540 (~9h, matches the box keepalive);
    # text_only True skips the vision tower (~8 min fewer TPU graphs); ntfy_topic
    # stays empty (in-box is local, no public tunnel needed); tool_call_parser
    # stays qwen3_coder (agent tool-calling); mtp_tokens stays 3 (KTL's patch fixes
    # the GDN-state corruption). weights_dataset already set above in the source.
    cfg = {"keepalive_min": 540, "text_only": True, "tpu": {"tp": tp}}
    if os.environ.get("VLLM_WEIGHTS"):
        cfg["weights"] = {"source": os.environ["VLLM_WEIGHTS"]}
    script = os.path.join(kaggle_dir, "ktl_cfg.py")
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(script, "w") as f:
        f.write("# injected config\nCFG = " + repr(cfg) + "\n")
    serve = os.path.join(ktl, "kernel", "serve_qwen38.py")
    env = dict(os.environ, TP=str(tp), TEXT_ONLY="1", NPROC=str(tp))
    print(f"[box] boot vLLM-TPU (TP={tp}) ...", flush=True)
    p = subprocess.Popen([sys.executable, serve, "--config", script], env=env)
    if not wait_health(url.replace("/v1", "")):
        p.terminate()
        raise SystemExit("[box] vLLM did not become healthy in time")
    return url


# ----------------------------------------------------------------------------
# agent wiring (mini-swe -> vLLM)
# ----------------------------------------------------------------------------
def _mini_templates():
    """Load mini-swe's own system/instance templates (they define the bash-tool format)."""
    import yaml, importlib.resources
    p = importlib.resources.files("minisweagent").joinpath("config/mini.yaml")
    cfg = yaml.safe_load(p.read_text())
    return cfg["agent"]["system_template"], cfg["agent"]["instance_template"]


def box_task(goal):
    cp = os.path.join(HERE, "context-pack.md")
    body = open(cp).read() if os.path.isfile(cp) else "(no context-pack.md)"
    return "Goal: " + goal + "\n\n=== context-pack.md ===\n" + body


def build_agent(vllm_url, vllm_model, max_tokens, cwd=REPO):
    from minisweagent.models import get_model
    from minisweagent.environments import get_environment
    api_key = os.environ.get("VLLM_API_KEY", "not-needed")
    model = get_model(None, {
        "model_name": "openai/" + vllm_model,
        "model_class": "litellm",
        "model_kwargs": {"api_base": vllm_url, "api_key": api_key, "max_tokens": max_tokens},
    })
    env = get_environment({"environment_class": "local", "cwd": cwd})
    system, instance = _mini_templates()
    system += (
        "\n\nYou are an autonomous ML-experiment engineer on a Kaggle TPU box working in the "
        "prism repo (torch_xla). On a process hang run `py-spy dump --pid <PID>` to read the "
        "stuck thread (XLA HLO compile shows torch_xla::XRT::Run / XlaOp::Build); fix root "
        "causes, do not just retry. Before blaming hardware, reproduce on CPU.\n\n"
        "## Context pack (prior runs + hypotheses)\n"
    )
    cp = os.path.join(HERE, "context-pack.md")
    if os.path.isfile(cp):
        system += open(cp).read()
    goal = os.environ.get("GOAL_PROMPT", (
        "PRISM causal status: exp16 (readout, DONE — lm_head edit moved the analytical "
        "bridge 0.600->0.754, confirming the bridge as a causal lever) and exp18 "
        "(logit-KD, DONE but DEGENERATE — the 0.8B self-targets gave KD loss ~0) are "
        "complete. The ONE remaining prize is exp17_lora_distill: the only experiment "
        "using 2nd-order HVP (create_graph=True in prism/jacobian.py bridge_rows_backward "
        "through o_proj). A dedicated canary HUNG 120 min on XLA HLO compile before the "
        "2h idle-kill. YOUR JOB: get exp17's 2nd-order HVP to COMPILE and RUN on TPU v5e-8 "
        "in this 9h session (long enough to outlive the compile), and produce a correct "
        "exp17 result JSON. "
        "FIRST (coexistence probe): vLLM Qwen3.8-27B serves the TPU; run a tiny torch_xla "
        "probe (load Qwen3.5-0.8B, one forward) and report the observed HBM split "
        "(TP=8 uses all 8 chips). "
        "THEN: run the exp17 2nd-order canary (1 pair 15_11, ~100 steps) with py-spy on "
        "any hang to capture the stack of the stuck compile thread; apply a real fix "
        "(XLA flag --xla_optimize_hlo=false or --xla_use_stablehlo=True, or batched-HVP, "
        "or CPU-precompute-J_norm+dataset), and verify against the CPU reference. "
        "Read box/context-pack.md for the full diagnostic history."
    ))
    return BoxAgent(model, env, max_tokens=max_tokens,
                    system_template=system, instance_template=instance,
                    output_path=os.path.join(HERE, "agent_trace.json"),
                    cost_limit=1e9, max_consecutive_format_errors=0)


# ----------------------------------------------------------------------------
# keepalive (defeats the 2h idle-kill; 9h process-lifetime guard)
# ----------------------------------------------------------------------------
def keepalive(max_hours=9):
    from threading import Thread
    stop = {"v": False}
    def loop():
        t0 = time.time(); i = 0
        while not stop["v"] and time.time() - t0 < max_hours * 3600:
            i += 1
            print(f"[box-keepalive] t={int(time.time()-t0)}s alive={i}", flush=True)
            time.sleep(5)
    th = Thread(target=loop, daemon=True)
    th.start()
    return stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["selfcheck", "serve", "run"], nargs="?", default="selfcheck")
    ap.add_argument("--task", default=None)
    ap.add_argument("--tp", type=int, default=int(os.environ.get("TP", "8")))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("VLLM_MAX_TOKENS", "32768")))
    a = ap.parse_args()

    if a.mode == "selfcheck":
        selfcheck()
        return

    if not _HAS_MSWE:
        raise SystemExit("serve/run needs mini-swe-agent: pip install mini-swe-agent==2.4.6")

    vllm_url = os.environ.get("VLLM_URL", "http://127.0.0.1:8000/v1")
    vllm_model = os.environ.get("VLLM_MODEL", "Qwen3.8-27B")
    stop = keepalive()
    if a.mode == "serve":
        vllm_url = boot_vllm(tp=a.tp)
    agent = build_agent(vllm_url, vllm_model, a.max_tokens)
    task = a.task or os.environ.get("GOAL_PROMPT") or (
        "exp16/18 are done. The prize is exp17 (2nd-order HVP): get it to compile and "
        "run on TPU v5e-8 in this 9h session and produce a correct result JSON. "
        "Run the 1-pair canary first with py-spy on any hang. Read box/context-pack.md.")
    out = agent.run(task)
    stop["v"] = True
    print(f"[box] agent exited: {json.dumps(out, default=str)[:500]}", flush=True)


if __name__ == "__main__":
    main()
