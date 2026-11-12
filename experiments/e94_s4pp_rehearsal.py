#!/usr/bin/env python3
"""РЕПЕТИЦИЯ E94_S4pp на mock-сервере: 7 миров (M1 / wide / undec / drift / invalid / crash /
смерть на 7-м запросе) плюс пути гейта модели (S4PP_WRONG_MODEL, S4PP_NO_PROPS).

GPU НЕ трогает: подменяет LlamaServer и fetch_props. Нужен потому, что глаз не поймал
дефекты, которые поймала эта репетиция (UnboundLocalError в halt/first_err, KeyError 'w',
halt_reason писавшийся до вычисления halt, мёртвая ветка «сервер умер», подмена «зависимости
от ширины» условием «emit != вершина»).

Запуск: S4PP_MODES=M1 python3 experiments/e94_s4pp_rehearsal.py   (без S4PP_MODES — все миры)
Артефакты миров: /tmp/s4pp_t/S4pp_<mode>.json, их проверяет experiments/e94_s4pp_audit.py.
"""
import json, os, sys, tarfile
sys.path.insert(0, 'experiments')
import exp94_s4pp as M

M.OUT, M.ART, M.JL = "/tmp/s4pp_t", "/tmp/s4pp_t/S4pp.json", "/tmp/s4pp_t/reqs.jsonl"
os.makedirs(M.OUT, exist_ok=True)

with tarfile.open(M.RAW_Q2) as tf:
    BASE = {k: M.load_addr(tf, k) for k in M.ADDR}
    TOPS = {k: [(t["id"], t["logprob"]) for t in json.load(
        tf.extractfile(b["mem"]))["response"]["completion_probabilities"][0]["top_logprobs"]]
        for k, b in BASE.items()}

TAG2ADDR = {"S4pp-ref": "probe", "S4pp-1": "probe", "S4pp-2": "probe2",
            "S4pp-ctrlA": "ctrlA", "S4pp-ctrlB": "ctrlB", "S4pp-ctrlTieA": "tieA",
            "S4pp-ctrlTieB": "tieB", "S4pp-0": "probe", "S4pp-repro": "repro",
            "S2a": "probe", "S2b": "probe"}
SEQ = ["S4pp-ref", "S4pp-1", "S4pp-2", "S4pp-ctrlA", "S4pp-ctrlB", "S4pp-ctrlTieA",
       "S4pp-ctrlTieB", "S4pp-0", "S4pp-repro", "S2a", "S2b"]
for kk in (1, 2, 3, 4, 8, 400):
    SEQ += ["S3-k%d-probe" % kk, "S3-k%d-ctrlA" % kk]
assert len(SEQ) == 23, len(SEQ)

SC = {"mode": "M1", "i": 0, "die_at": None}


def respond(body):
    tag = SEQ[SC["i"]]
    SC["i"] += 1
    bk = TAG2ADDR.get(tag, "probe" if tag.endswith("probe") else "ctrlA")
    b, k = BASE[bk], len(body["prompt"])
    ban = (body.get("logit_bias") or [[None, None]])[0][0]
    mode = SC["mode"]
    if mode == "crash":
        raise OSError("connection reset by peer")
    if mode == "invalid":
        emit = ban if ban is not None else b["emit"]
    elif mode == "M1":
        # выбор не вершины отчёта и не забаненного (probe -> 326, probe2 -> 3240)
        third = [i for i, _ in TOPS[bk] if i not in (b["emit"], b["top1"])][0]
        emit = third if ban is not None else b["emit"]
    elif mode == "wide":
        # зависимость от ширины на пробе (k=1 даёт другой токен), на контроле — нет
        emit = (326 if (tag == "S3-k1-probe") else b["emit"])
    elif mode == "drift":
        # новые ветки r4: репро-адрес дрейфнул, строки S2 различаются
        emit = 999 if tag == "S4pp-repro" else (777 if tag == "S2a" else b["emit"])
    elif mode == "undec":
        emit = b["top1"] if ban is not None else b["emit"]
    else:
        emit = b["emit"]
    return dict(tokens=[emit], content=str(emit), tokens_evaluated=k, tokens_predicted=1,
                stop_type="stop",
                completion_probabilities=[dict(id=emit, logprob=-1.0, top_logprobs=[
                    dict(id=i, logprob=lp, token=str(i)) for i, lp in TOPS[bk]])],
                generation_settings=dict(
                    temperature=0.0, top_k=body.get("top_k", 400), top_p=1.0, min_p=0.0,
                    typical_p=1.0, xtc_probability=0.0, seed=8181, n_probs=20,
                    cache_prompt=body["cache_prompt"],
                    backend_sampling=body.get("backend_sampling"),
                    post_sampling_probs=body.get("post_sampling_probs"),
                    logit_bias=[dict(bias=-100.0, token=int(ban))] if ban is not None else [],
                    samplers=["top_k"], **{"speculative.types": "none"}))


class Fake:
    def __init__(self, port, log):
        self.pid, self.n = 12345, 0

    def start(self):
        pass

    def wait_health(self, timeout=0):
        return True

    def device_fd(self):
        return ["/dev/dri/renderD128"]

    def alive(self):
        return self.mode_alive

    mode_alive = True

    def completion(self, body, timeout=0):
        self.n += 1
        if SC["die_at"] == self.n:
            self.mode_alive = False
            raise OSError("server died")
        return respond(body), 0.01

    def kill(self):
        pass


M.fetch_props = lambda port, timeout=5: {
    "model_path": M.MODEL, "webui": True,
    "default_generation_settings": {"n_ctx": 2048}}
if os.environ.get("S4PP_WRONG_MODEL"):
    M.fetch_props = lambda port, timeout=5: {"model_path": "/data/Other/foreign.gguf"}
if os.environ.get("S4PP_NO_PROPS"):
    def _boom(port, timeout=5):
        raise OSError("props 404")
    M.fetch_props = _boom
M.LlamaServer = Fake
M.pick_port = lambda c: 8999

EXPECT = {}
_sel = os.environ.get("S4PP_MODES", "M1,wide,undec,drift,invalid,crash").split(",")
_all = [("M1", None), ("wide", None), ("undec", None), ("drift", None), ("invalid", None),
        ("crash", None), ("M1", 7)]
for mode, die in [x for x in _all if x[0] in _sel]:
    SC.update(mode=mode, i=0, die_at=die)
    Fake.mode_alive = True
    print("\n============= MODE=%s die_at=%s =============" % (mode, die))
    rc = M.main()
    import shutil
    shutil.copy(M.ART, "/tmp/s4pp_t/S4pp_%s%s.json" % (mode, "_die%s" % die if die else ""))
    A = json.load(open(M.ART))
    print("rc=%s halt=%s n=%s ok=%s" % (rc, A.get("halt_reason"),
                                        A.get("requests_n"), A.get("requests_ok")))
    print("VERDICT:", A["summary"]["verdict"])
    print("ref:", A["summary"]["ref"]["verdict"], "| probes:",
          [(p["tag"], p["emit"], p["verdict"]) for p in A["summary"]["probes"]])
    print("controls:", [(c["tag"], c["emit"], c["verdict"], c["manip_ok"])
                        for c in A["summary"]["controls"]])
    print("S3 ctrl moves:", A["summary"]["S3_control_expectation"], "| k1:",
          A["summary"]["k1_note"])
    print("S3 probe emits:", [(r["tag"], r["emit"], r["verdict"])
                              for r in A["summary"]["S3_topk"] if r["tag"].endswith("probe")])
    print("repro:", A["summary"]["repro"], "| S4pp_0:", A["summary"]["S4pp_0"]["verdict"])
    print("S2 pair:", {k2: (v2["emit"], v2["top1"], v2["n_tl"]) for k2, v2 in
                       A["summary"]["S2_pair"].items()},
          "| diff:", A["summary"]["S2_pair_diff"])
    print("gate2:", A["summary"]["gate2"])
    assert not [r for r in A["requests"] if r.get("verdict") == "unknown"], \
        "остались неизвестные чтения"
    g = A["summary"]["gate2"]
    exp = {"M1": (False, False, False), "wide": (True, False, True)}\
        .get((mode, die)[0], (None, None, None))
    if exp[0] is not None and die is None:
        assert (g["probe_dependence"], g["control_dependence"], g["gate2_met"]) == exp, \
            ("гейт прочитался не так, как предрегистрировано", mode, g)
    print("gate2_met=", g["gate2_met"], "probe_dep=", g["probe_dependence"],
          "ctrl_dep=", g["control_dependence"], "probe_stays=", g.get("probe_stays_at_archive_emit"))
    for r in A["requests"]:
        if r.get("status") == "ok":
            assert r.get("verdict"), ("пустой verdict", r["tag"])
