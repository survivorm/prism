#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 common: llama-server client, gates, stats.
Всё, что здесь сравнивается, проверяется grep'ом в ~/work/llama.cpp (0b8cfb8fe);
показания — в results/e94/NOTES.md (секция grep-доказательства).
"""
import json, os, socket, subprocess, time
import urllib.request
import numpy as np

LLAMA_BIN = os.path.expanduser("~/work/llama.cpp/build/bin/llama-server")
GGUF = "/data/Qwen/qwen3.5-0.8B-f16.gguf"
# --slots: включён ПО УМОЛЧАНИЮ (common/common.h:671 endpoint_slots = true) — флаг оставлен
# явным ради прозрачности; без /slots кэш-гейт (r3a-B3) невозможен.
SERVER_ARGS = ["-m", GGUF, "-sm", "none", "-mg", "0", "-ngl", "99", "-fa", "on",
               "-c", "2048", "-np", "1", "--slots",
               "--host", "127.0.0.1"]
# нейтральные штрафы, сланные явно (дефолты сервера — из GGUF-метаданных, common.cpp:1267):
# repeat_penalty=1.0 = мультипликатор «без влияния» (0.0 выкидывал бы токены, НЕ нейтрал);
# presence/frequency=0, repeat_last_n=0 (окно DRY пустое), dry_multiplier/base=1.0,
# xtc_probability=0, mirostat=0. Зафиксировано в NOTES (интерпретация «нулевые штрафы»).
NEUTRAL = {"repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
           "repeat_last_n": 0, "dry_multiplier": 1.0, "dry_base": 1.0,
           "xtc_probability": 0.0, "mirostat": 0}

def san(obj):
    """json.dump sanitize: numpy→python, NaN/inf→None (AGENTS.md #4)."""
    if isinstance(obj, dict):
        return {k: san(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [san(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and (obj != obj or obj in (float("inf"), float("-inf"))):
        return None
    return obj

def rankdata_avg(x):
    """ранги со средними по тайам (тай-коррекция, требование r1b-2)."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return ranks

def spear(a, b):
    """Spearman c avg-ranks; None если вырожден (обеспечивает каскад None-потребителей)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(rankdata_avg(a), rankdata_avg(b))[0, 1])

def perm_null(a, b, n=2000, seed=8181):
    """(obs, p two-sided, null CI 2.5/97.5) — ОДНА процедура на обеих сторонах (AGENTS.md #1)."""
    obs = spear(a, b)
    if obs is None:
        return None, None, None
    rng = np.random.default_rng(seed)
    nl = [spear(rng.permutation(a), b) for _ in range(n)]
    nl = np.array([v for v in nl if v is not None])
    p = float((np.abs(nl) >= abs(obs)).mean())
    return obs, p, [float(np.percentile(nl, 2.5)), float(np.percentile(nl, 97.5))]

def bootstrap_ci_median(x, n=10000, seed=8181):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    meds = [float(np.median(rng.choice(x, size=len(x), replace=True))) for _ in range(n)]
    return float(np.median(x)), [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]

def rep_flags(ids, n):
    """дословно из experiments/exp81b_replay.py (не переизобретать)."""
    seen = set()
    out = []
    for t in range(len(ids)):
        g = tuple(ids[max(0, t - n + 1):t + 1])
        if len(g) < n:
            out.append(False)
            continue
        out.append(g in seen)
        seen.add(g)
    return np.array(out)

def rep8_segments(ids, off=64, seg=32, nseg=12):
    """12 сегментов по 32 по gen-региону; rep8 := mean(segments['8']) (C-секция exp81b)."""
    fl = rep_flags(ids, 8)
    segs = [float(fl[off + a:off + a + seg].mean()) for a in range(0, nseg * seg, seg)]
    return segs, float(np.mean(segs))

def prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n

def pick_port(cands):
    for p in cands:
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            return p
        except OSError:
            s.close()
    raise RuntimeError("no free port in %s" % cands)

def base_body(ids, n_predict, cache, **overrides):
    """тело /completion: только whitelist-имена (server-schema.cpp, grep в NOTES);
    stream:false ВСЕГДА (completion_probabilities печатается только при !stream,
    server-task.cpp:358); logprobs+top_logprobs только парой (server-common.cpp:1381-1389)."""
    body = {"prompt": list(ids), "n_predict": int(n_predict), "temperature": 0,
            "stream": False, "return_tokens": True, "logprobs": True,
            "top_logprobs": 2, "cache_prompt": bool(cache), "seed": 8181}
    body.update(NEUTRAL)
    body.update(overrides)
    return body

def check_schema(resp):
    """СТРУКТУРНЫЙ pre-чек (r5c B1): cp — непустой list, cp[0].logprob есть,
    tokens — непустой int-массив и == [cp.id]. Требование len(top_logprobs)==2 ТЗ
    вынесено в finding (steering оркестратора: поле top_logprobs в /completion
    игнорируется — свип фиксирует доступность top-k и контракт Q2, не hard-гейт)."""
    cp = resp.get("completion_probabilities")
    if not isinstance(cp, list) or len(cp) == 0:
        return False, "no completion_probabilities"
    p0 = cp[0]
    if not isinstance(p0, dict) or "logprob" not in p0:
        return False, "cp[0] keys=%s" % list(p0 or {})
    toks = resp.get("tokens")
    if not isinstance(toks, list) or len(toks) == 0 or not all(isinstance(t, int) for t in toks):
        return False, "tokens not a non-empty int array"
    if [p.get("id") for p in cp] != toks:
        return False, "tokens != [p.id for p in completion_probabilities]"
    return True, "cp_len=%d tokens_len=%d" % (len(cp), len(toks))

def top2_of(p0):
    """top-2 id: top_logprobs включает топ-1 (get_token_probabilities, server-common.cpp:1502)
    → top2 = tl[1]; страховка от инверсии — по id топ-1."""
    tl = p0["top_logprobs"]
    if tl[0].get("id") == p0.get("id"):
        return tl[0]["id"], tl[1]["id"]
    return p0.get("id"), tl[0]["id"]

def check_prompt_count(resp, sent):
    te = resp.get("tokens_evaluated")
    return te == sent, "tokens_evaluated=%r sent=%d" % (te, sent)

def slot_cache_ok(slots, id_slot, k):
    """гейт кэша на /slots (r3a-B3, r5c): n_prompt_tokens==k,
    n_prompt_tokens_processed<=2, n_prompt_tokens_cache>=k-2."""
    if isinstance(slots, dict):  # error-ответ эндпоинта (например, без --slots)
        return False, {"error": slots.get("error", "unknown slots error")}
    for s in slots:
        if s.get("id") == id_slot:
            info = {kk: s.get(kk) for kk in ("n_prompt_tokens", "n_prompt_tokens_processed",
                                             "n_prompt_tokens_cache")}
            ok = (info["n_prompt_tokens"] == k
                  and info["n_prompt_tokens_processed"] is not None
                  and info["n_prompt_tokens_processed"] <= 2
                  and info["n_prompt_tokens_cache"] is not None
                  and info["n_prompt_tokens_cache"] >= k - 2)
            return ok, info
    return False, {"error": "slot %r not found in /slots" % id_slot}

class LlamaServer:
    """один сервер-процесс, ребёнок скрипта; stdout/stderr → файл, никогда не в pipe
    (риск взаимоблокировки reader/writer, r1b-8)."""

    def __init__(self, port, log_path):
        self.port, self.log_path, self.proc = int(port), str(log_path), None

    @property
    def pid(self):
        return self.proc.pid if self.proc else None

    def foreign_running(self):
        r = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True, text=True)
        return [l.strip() for l in r.stdout.split() if l.strip()]

    def start(self):
        foreign = self.foreign_running()
        if foreign:
            raise RuntimeError("foreign llama-server running: %s (STOP, NOTES)" % foreign)
        cmd = ("source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1; export ZES_ENABLE_SYSMAN=1; "
               "exec %s %s --port %d --log-disable"
               % (LLAMA_BIN, " ".join(SERVER_ARGS), self.port))
        with open(self.log_path, "ab") as lf:
            self.proc = subprocess.Popen(["bash", "-c", cmd], stdout=lf, stderr=lf,
                                         stdin=subprocess.DEVNULL, start_new_session=True)

    def wait_health(self, timeout=240):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("server died during load, rc=%s" % self.proc.returncode)
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % self.port, timeout=3) as r:
                    if r.status == 200:
                        return True
            except Exception:
                time.sleep(2)
        raise TimeoutError("server not healthy in %ds" % timeout)

    def device_fd(self):
        """device-гейт: fd-доказательство /proc/<pid>/fd → /dev/dri/* (r3a-B1)."""
        try:
            fds = os.listdir("/proc/%d/fd" % self.pid)
        except FileNotFoundError:
            return []
        dri = []
        for f in fds:
            try:
                t = os.readlink("/proc/%d/fd/%s" % (self.pid, f))
            except OSError:
                continue
            if t.startswith("/dev/dri/"):
                dri.append(t)
        return dri

    def _post(self, path, body, timeout):
        req = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (self.port, path),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read().decode())
        return resp, time.time() - t0

    def completion(self, body, timeout=1800):
        return self._post("/completion", body, timeout)

    def tokenize(self, content, timeout=60):
        return self._post("/tokenize", {"content": content}, timeout)

    def slots(self):
        with urllib.request.urlopen("http://127.0.0.1:%d/slots" % self.port, timeout=10) as r:
            return json.loads(r.read().decode())

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def kill(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill()
                self.proc.wait()
        self.proc = None
