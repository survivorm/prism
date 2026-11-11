"""subagent — оркестраторское расширение: фоновые headless-субагенты pig.

Инварианты (уроки раунда 9):
- субагент = `pig -p --approve --no-extensions --no-session --model <m> --tools <t> <prompt>`
  (список аргументов, НЕ shell: кавычки/доллары в промпте безопасны);
- detached (start_new_session) — переживает перезапуск родителя;
- журнал jobs в <cwd>/.pig/jobs/<id>.json, вывод — <id>.out; статус/wait читают их;
- обязательный финальный маркер в выводе — не молчание RC=0: ждём текстовый DONE-маркер.
"""
import json, os, subprocess, time, uuid

from pig_sdk import Extension

JOBS = ".pig/jobs"


def _job_paths(jid):
    return f"{JOBS}/{jid}.json", f"{JOBS}/{jid}.out"


def _load(jid):
    p, _ = _job_paths(jid)
    with open(p) as f:
        return json.load(f)


def _save(jid, meta):
    p, _ = _job_paths(jid)
    with open(p, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def _poll(meta):
    """Обновлённый статус: running -> done_rcX (или marker_done), подмечает трупов."""
    jid = meta["id"]
    _, out = _job_paths(jid)
    if meta["state"] in ("done_marker", "done_rc0", "done_rcX"):
        return meta
    alive = False
    try:
        os.kill(meta["pid"], 0)
        alive = True
    except OSError:
        alive = False
    txt = ""
    if os.path.exists(out):
        with open(out, errors="replace") as f:
            txt = f.read()
    if meta.get("marker") and meta["marker"] in txt:
        meta["state"] = "done_marker"
    elif not alive:
        rc = None
        try:
            pid, status = os.waitpid(meta["pid"], os.WNOHANG)
            if pid == meta["pid"]:
                rc = os.waitstatus_to_exitcode(status)
        except ChildProcessError:
            rc = None  # не наш ребёнок (после рестарта) — считаем по RC-строке в выводе
        if rc is None and "SUBAGENT_RC=" in txt:
            rc = int(txt.rsplit("SUBAGENT_RC=", 1)[1].split("\n")[0])
        meta["rc"] = rc
        meta["state"] = "done_rc0" if rc == 0 else "done_rcX" if rc is not None else "dead"
    meta["bytes"] = len(txt)
    meta["updated"] = time.strftime("%H:%M:%S")
    _save(jid, meta)
    return meta


def new_extension() -> Extension:
    ext = Extension("subagent")

    def spawn(ctx, args):
        os.makedirs(JOBS, exist_ok=True)
        if args.get("prompt_file"):
            with open(args["prompt_file"]) as f:
                prompt = f.read()
        elif args.get("prompt"):
            prompt = args["prompt"]
        else:
            return {"error": "нужен prompt_file или prompt"}
        if prompt.lstrip().startswith("-"):
            prompt = " " + prompt  # защита: pig съедает аргументы с дефиса
        model = args.get("model", "obleth/ngenix-llm:low")
        tools = args.get("tools", "read,bash,write")
        cwd = args.get("cwd") or os.getcwd()
        jid = time.strftime("%H%M%S") + "-" + uuid.uuid4().hex[:6]
        _, out = _job_paths(jid)
        # обёртка пишет RC в тот же out — переживая смерть самого pig
        cmd = [
            "bash", "-c",
            'pig -p --approve --no-extensions --no-session --model "$1" --tools "$2" "$3" '
            '> "$4" 2>&1; echo "SUBAGENT_RC=$?" >> "$4"',
            "x", model, tools, prompt, os.path.abspath(out),
        ]
        p = subprocess.Popen(
            cmd, cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        meta = {"id": jid, "pid": p.pid, "model": model, "tools": tools, "cwd": cwd,
                "marker": args.get("marker"), "state": "running", "spawned": time.strftime("%H:%M:%S")}
        _save(jid, meta)
        return {"id": jid, "pid": p.pid, "out": out}

    ext.tool("subagent_spawn",
        "Launch detached headless pig subagent; returns job id. SPAWN ONCE per task (перед повтором — subagent_status!). prompt via FILE path "
        "(preferred) or inline text. model like 'obleth/ngenix-llm[:low]'.",
        {"type": "object", "properties": {
            "prompt_file": {"type": "string", "description": "файл с промптом (рекомендуется)"},
            "prompt": {"type": "string", "description": "инлайн-промпт (если нет файла)"},
            "model": {"type": "string", "description": "по умолчанию obleth/ngenix-llm:low"},
            "tools": {"type": "string", "description": "allowlist, по умолчанию read,bash,write"},
            "cwd": {"type": "string", "description": "рабочая директория (по умолчанию текущая)"},
            "marker": {"type": "string", "description": "DONE-маркер в выводе (для статуса done_marker)"},
        }}, spawn)

    def status(ctx, args):
        if not os.path.isdir(JOBS):
            return {"jobs": []}
        ids = [args["id"]] if args.get("id") else \
            sorted(f[:-5] for f in os.listdir(JOBS) if f.endswith(".json"))[-8:]
        res = []
        for jid in ids:
            try:
                meta = _poll(_load(jid))
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            _, out = _job_paths(jid)
            tail = ""
            if os.path.exists(out):
                with open(out, errors="replace") as f:
                    f.seek(max(0, os.path.getsize(out) - 4000))
                    tail = f.read()[-1500:]
            res.append({**meta, "tail": tail})
        return {"jobs": res}

    ext.tool("subagent_status", "Status+tail of subagent jobs (no id -> all recent).",
        {"type": "object", "properties": {"id": {"type": "string"}}}, status)

    def wait(ctx, args):
        meta = _load(args["id"])
        t0 = time.time()
        lim = args.get("timeout_s", 900)
        while time.time() - t0 < lim:
            meta = _poll(meta)
            if meta["state"].startswith("done") or meta["state"] == "dead":
                return meta
            time.sleep(10)
        return meta

    ext.tool("subagent_wait", "Block until job finishes (marker/dead) or timeout.",
        {"type": "object", "properties": {"id": {"type": "string"},
                                          "timeout_s": {"type": "number"}}}, wait)

    return ext
