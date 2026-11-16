#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94/S3b — кусается ли top_k на этом форке (микро-рука, T=1.0, 7 запросов,
ОДИН запуск сервера). ТЗ: docs/dynamic/0_BRIEF_E94_S3b_TZ.md (r19), пакет
docs/dynamic/0_PACKET_E94_S3b_TZ_r19.md. Артефакт: results/e94/S3b.json
(S4pp.json не перезаписывается).

ОДИН вопрос: применяется ли top_k к пост-сэмплированному набору кандидатов
(post_sampling_probs=true) на этом форке. Порядок решения — ДО запуска:
4 → 3 → B3 → 1a → 1b → 2 → 5 (первая применимая).

Само-аудит до запуска (AGENTS.md): (1) ассерт числа запросов 7 ДО лонча;
(2) функция decide() ПРОГОНЯЕТСЯ на 50 mock-мирах ТЗ — расхождение = fail-fast
до сервера; (3) синтаксис/ветки — само-аудит исполнителя (в логе лонча).
"""
import json
import math
import os
import sys
import time
import tarfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp94_sycl_common import LlamaServer, pick_port, san, SERVER_ARGS, LLAMA_BIN

OUT = "results/e94"
ART = OUT + "/S3b.json"
RAW_Q2 = "results/e94/raw_q2.json.tar.gz"
N_PROBS = 20
# адреса и модель — как в S4pp (проба: Q2 w4 k=70; контроль: Q2 w13 k=101)
ADDR = {"probe": dict(w=4, k=70), "ctrl": dict(w=13, k=101)}
# порядок запросов — ровно как в таблице ТЗ; None = top_k-поле ОПУЩЕНО (off)
PLAN = [
    ("S3b-k1-probe",  "probe", 1),
    ("S3b-k2-probe",  "probe", 2),
    ("S3b-k4-probe",  "probe", 4),
    ("S3b-off-probe", "probe", None),
    ("S3b-k1-ctrl",   "ctrl", 1),
    ("S3b-k2-ctrl",   "ctrl", 2),
    ("S3b-k4-ctrl",   "ctrl", 4),
]
N_PLAN = 7
LADDER_ROWS = (1, 2, 3, 5, 6, 7)  # 4b: лестничные (off-строка — 4)

# ---------------------------------------------------------------------------
# decide() — правило решения ТЗ (записано ДО запуска; иного чтения не будет).
# Порядок: 4 → 3 → B3 → 1a → 1b → 2 → 5 (первая применимая).
# ---------------------------------------------------------------------------
SETS = {(1, 2, 4), (2, 3, 5)}
FORMS = {0: {(1, 1, 1), (1, 2, 2), (1, 2, 3), (1, 2, 4)},
         1: {(1, 1, 1), (2, 2, 2), (2, 3, 3), (2, 3, 4), (2, 3, 5)}}
VOID = {"echo_no_topk", "echo_no_psp"}  # → mech_void, остальной halt → mech_invalid


def flat(v):
    return len(set(v)) == 1


def decide(n_p, n_c, n_off, halts):
    """n_p/n_c — (N(k1),N(k2),N(k4)); n_off — строка 4; halts — список
    (row, halt_reason) по строкам, у которых halt (внутри строки halt_reason
    уже первого сработавшего ассерта). Возвращает (rule, kind, verdict,
    halt_reason, reading)."""
    # 4. любой halt ассертов 1–4 → рука невалидна
    if halts:
        row, kind = min(halts, key=lambda h: h[0])  # первый halt в порядке строк
        v = "mech_void" if kind in VOID else "mech_invalid"
        return (4, "mech", v, kind,
                "рука невалидна: halt_reason=%s (строка %d) — механизм не установлен"
                % (kind, row))
    # 3. N == 0 на ЛЮБОЙ строке → рука невалидна, разбирательство по исходнику
    if 0 in n_p or 0 in n_c or n_off == 0:
        return (3, "mech", "mech_invalid", "report_empty",
                "рука невалидна: пустой пост-сэмплированный отчёт (N=0) — "
                "разбирательство по исходнику, выводов нет")
    # B3 (инвариант, глобальный): компонент ЛЮБОГО вектора > 20 ИЛИ N_off > 20
    # — отчёт не может быть длиннее n_probs=20 → наблюдение вне матрицы → 5
    if any(x > 20 for x in n_p) or any(x > 20 for x in n_c) or n_off > 20:
        return (5, "mech", "mech_void", "no_preregistered_reading",
                "инвариант B3: компонент > 20 либо N_off > 20 — отчёт не может "
                "быть длиннее n_probs=20; наблюдение вне матрицы отчёта — "
                "не вычислено, не выдумывается")
    # 1a. лестница на обоих адресах, ОДИН набор, N_probe(4) <= N_off, N_off <= 20
    if (n_p in SETS and n_c in SETS and n_p == n_c
            and n_p[2] <= n_off and n_off <= 20):
        return (1, "mech", "mech_ok", None,
                "top_k применяется к пост-сэмплированному набору (механика "
                "установлена); gate2 T=0 не меняется; осмысленным становится "
                "повтор условия 2 при T>0")
    # 1b. лестница + потолок B4: единое c, форма min(k+c, N_off), c-формы контроля
    for cc in (0, 1):
        form = (min(1 + cc, n_off), min(2 + cc, n_off), min(4 + cc, n_off))
        if (n_p == form and not flat(n_p) and n_off <= 20 and n_c in FORMS[cc]):
            cap_note = ("потолок на пробе измерен (cap_p = N_off)" if n_off < 20
                        else "N_off = 20 — отчёт насыщен top-20, cap_p >= N_off, НЕ измерен")
            return (1, "mech", "mech_ok", None,
                    "top_k применяется и B4 держит потолок (c=%d): %s; на контроле "
                    "потолок не измерен (off-запроса нет) — высота плато либо >= "
                    "N_ctrl(4) при отсутствии плато" % (cc, cap_note))
    # 2. не применяется: ОБА вектора плоские, max(v) >= 3, v_p == N_off, гейты B3
    vp, vc = n_p[0], n_c[0]
    if (flat(n_p) and flat(n_c) and max(vp, vc) >= 3 and vp == n_off
            and vp <= 20 and vc <= 20):
        cap_note = ("потолок на пробе измерен (cap = N_off)" if n_off < 20
                    else "N_off == n_probs — отчёт насыщен top-20, cap >= N_off, не измерен")
        return (2, "mech", "mech_flat", None,
                "top_k не применяется к набору (механика отсутствует); повтор "
                "условия 2 при T>0 бессмысленен; рука B не запускается по этой "
                "причине (предрегистрированное следствие); %s" % cap_note)
    # 5. catch-all
    return (5, "mech", "mech_void", "no_preregistered_reading",
            "вне предрегистрированных чтений (смешанная конвенция / не-min-"
            "лестница / плоские пары max(v) <= 2 / v_p != N_off и пр.) — "
            "не вычислено, не выдумывается")


# ---------------------------------------------------------------------------
# Само-аудит до запуска: 50 mock-миров ТЗ — decide() обязана дать вердикты ТЗ.
# (Миры и EXPECT — дословно из ТЗ r19 / пакета §C.)
# ---------------------------------------------------------------------------
W = [
 ("W1",  (1,2,4),  (1,2,4), 20, []),
 ("W2",  (2,3,5),  (2,3,5), 20, []),
 ("W3",  (20,20,20),(20,20,20), 20, []),
 ("W4",  (1,1,1),  (1,1,1), 1, []),
 ("W5",  (1,2,4),  (20,20,20), 20, []),
 ("W6",  (1,2,4),  (1,2,4), 20, [(1,'no_field')]),
 ("W7",  (1,2,4),  (1,2,4), 20, [(1,'http_500')]),
 ("W8",  (0,2,4),  (1,2,4), 20, []),
 ("W9",  (1,2,4),  (1,2,4), 4, []),
 ("W10", (1,2,4),  (1,2,4), 4, [(4,'off_default_low')]),
 ("W11", (2,4,8),  (2,4,8), 20, []),
 ("W12", (1,2,4),  (1,2,4), 2, []),
 ("W13", (1,1,1),  (1,2,4), 1, []),
 ("W14", (1,2,4),  (1,2,4), 20, [(1,'echo_mismatch')]),
 ("W15", (1,2,4),  (1,2,4), 20, [(1,'echo_no_topk')]),
 ("W16", (3,3,3),  (3,3,3), 3, []),
 ("W17", (1,2,3),  (1,2,3), 3, []),
 ("W18", (2,2,2),  (2,2,2), 2, []),
 ("W19", (3,3,3),  (5,5,5), 3, []),
 ("W20", (1,1,1),  (5,5,5), 1, []),
 ("W21", (1,2,3),  (1,2,2), 3, []),
 ("W22", (1,2,4),  (1,2,4), 20, [(1,'echo_mismatch')]),
 ("W23", (1,2,4),  (1,2,4), 20, [(5,'server_died')]),
 ("W24", (1,1,1),  (2,2,2), 1, []),
 ("W25", (1,2,4),  (1,1,1), 4, []),
 ("W26", (2,3,3),  (2,3,3), 3, []),
 ("W27", (1,2,4),  (2,3,5), 20, []),
 ("W28", (1,2,3),  (1,2,3), 20, []),
 ("W29", (2,3,4),  (2,3,5), 4, []),
 ("W30", (0,2,4),  (1,2,4), 20, []),
 ("W31", (1,2,4),  (1,2,4), 0, []),
 ("W32", (1,2,4),  (0,2,4), 20, []),
 ("W33", (1,2,4),  (1,2,4), 20, [(1,'echo_no_psp')]),
 ("W34", (1,2,4),  (1,2,2), 20, []),
 ("W35", (20,20,20),(20,20,20), 5, []),
 ("W36", (1,3,4),  (1,3,4), 20, []),
 ("W37", (1,2,4),  (1,2,4), 20, [(4,'echo_no_topk')]),
 ("W38", (1,2,4),  (1,2,4), 20, [(2,'echo_mismatch'), (3,'echo_no_topk')]),
 ("W39", (1,2,4),  (1,2,4), 20, [(1,'http_500'), (5,'echo_no_psp')]),
 ("W40", (3,4,5),  (3,4,5), 20, []),
 ("W41", (1,2,4),  (1,2,4), 20, [(5,'server_died')]),
 ("W42", (1,2,4),  (1,2,4), 20, [(1,'echo_mismatch'), (1,'echo_no_topk')]),
 ("W43", (2,3,5),  (2,2,2), 21, []),
 ("W44", (1,2,4),  (1,2,4), 20, [(4,'echo_mismatch')]),
 ("W45", (2,3,5),  (2,2,2), 20, []),
 ("W46", (1,2,4),  (1,2,4), 20, [(1,'echo_no_psp')]),
 ("W47", (3,3,3),  (1,2,4), 3, []),
 ("W48", (2,3,5),  (2,3,5), 3, []),
 ("W49", (5,5,5),  (21,21,21), 5, []),
 ("W50", (1,2,4),  (1,2,4), 21, [(1,'echo_mismatch')]),
]
EXPECT = {"W1":"1a","W2":"1a","W3":"2","W4":"5","W5":"5","W6":"4","W7":"4",
 "W8":"3","W9":"1a","W10":"4","W11":"5","W12":"5","W13":"5","W14":"4",
 "W15":"4","W16":"2","W17":"1b","W18":"5","W19":"2","W20":"2","W21":"1b",
 "W22":"4","W23":"4","W24":"5","W25":"1b","W26":"1b","W27":"5","W28":"5",
 "W29":"1b","W30":"3","W31":"3","W32":"3","W33":"4","W34":"1b","W35":"5",
 "W36":"5","W37":"4","W38":"4","W39":"4","W40":"5","W41":"4","W42":"4",
 "W43":"5","W44":"4","W45":"1b","W46":"4","W47":"5","W48":"5","W49":"5",
 "W50":"4"}


def selftest():
    """До сервера: decide() на всех 50 мирах ТЗ. Расхождение = fail-fast (sys.exit(3))."""
    mism = {}
    counts = {}
    for name, p, c, off, halts in W:
        rule, _k, _v, _h, _r = decide(p, c, off, halts)
        got = ("1b" if "B4" in _r else "1a") if rule == 1 else str(rule)
        counts[got] = counts.get(got, 0) + 1
        if got != EXPECT[name]:
            mism[name] = (got, EXPECT[name])
    if mism:
        print("[selftest] FAIL: decide() != ТЗ на %r" % mism, flush=True)
        sys.exit(3)
    print("[selftest] 50/50 миров ТЗ: decide() совпадает, counts=%r" % counts, flush=True)
    return {"worlds": len(W), "mismatches": [], "counts": counts}


# ---------------------------------------------------------------------------
# запросы
# ---------------------------------------------------------------------------
def load_prompts():
    """Тела адресов — из закоммиченного архива Q2 (канон S4pp)."""
    prompts = {}
    with tarfile.open(RAW_Q2) as tf:
        for key, a in ADDR.items():
            mem = "e94raw/q2_w%d_k%d.json" % (a["w"], a["k"])
            d = json.load(tf.extractfile(mem))
            prompts[key] = list(d["request"]["prompt"])
    return prompts


def body(prompt, top_k):
    """Общая форма ТЗ: T=1.0, seed=8181, n_predict=1, cache_prompt=false,
    post_sampling_probs=true, n_probs=20, logprobs=true, top_logprobs=20;
    top_k=off => поле ОПУЩЕНО. Неутрал. штрафы — как в S4pp (дефолты-ловушки
    GGUF выключены явно)."""
    b = {"prompt": list(prompt), "n_predict": 1, "temperature": 1.0, "stream": False,
         "return_tokens": True, "logprobs": True, "top_logprobs": N_PROBS,
         "n_probs": N_PROBS, "post_sampling_probs": True, "cache_prompt": False,
         "seed": 8181, "repeat_penalty": 1.0, "presence_penalty": 0.0,
         "frequency_penalty": 0.0, "repeat_last_n": 0, "dry_multiplier": 1.0,
         "dry_base": 1.0, "xtc_probability": 0.0, "mirostat": 0,
         "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0}
    if top_k is not None:
        b["top_k"] = int(top_k)
    return b


def send(port, b, timeout=600):
    """Ассерт 1 (транспорт). Возврат: (status, halt1, resp|None, wall).
    - ответа вовсе нет (reset/timeout/смерть сервера) → sent + server_died;
    - HTTP != 200 → received + http_<code>;
    - HTTP 200, тело НЕ парсится → received + server_died (приравнено к
      transport-ошибке, r10-MINOR);
    - иначе → received + None."""
    req = urllib.request.Request(
        "http://127.0.0.1:%d/completion" % port,
        data=json.dumps(b).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
        wall = time.time() - t0
    except urllib.error.HTTPError as e:
        code = e.code
        try:
            e.read()
        except Exception:
            pass
        return "received", "http_%d" % code, None, None
    except Exception:
        return "sent", "server_died", None, None
    try:
        resp = json.loads(raw)
    except Exception:
        return "received", "server_died", None, wall
    return "received", None, resp, wall


def report_len(resp):
    """N(x) = длина набора кандидатов item[0] (ТЗ: len(completion_probabilities[0]);
    на этом форке cp[0] — dict). ВАЖНО: поле набора Зависит от режима:
    - post_sampling_probs=true: `top_probs` (итер {id, token, bytes, prob});
    - post_sampling_probs=false (pre-sampling, факт B5: n_tl=20): `top_logprobs`
      (итер {id, token, bytes, logprob}). Диагностический прогон 13.11 (ошибка v1:
    v1 читал только top_logprobs → ложный N=0 → ложный rule 3; пере-прогон).
    cp == [] или [0] отсутствует → 0 (→ правило 3)."""
    cp = resp.get("completion_probabilities")
    if not isinstance(cp, list) or len(cp) == 0:
        return 0
    item0 = cp[0]
    if isinstance(item0, dict):
        tp = item0.get("top_probs")
        if isinstance(tp, list):
            return len(tp)
        tl = item0.get("top_logprobs")
        return len(tl) if isinstance(tl, list) else 0
    if isinstance(item0, list):
        return len(item0)
    return 0


def report_field(resp):
    """Какое поле набора использовалось (для аудита артефакта)."""
    cp = resp.get("completion_probabilities")
    if not isinstance(cp, list) or len(cp) == 0:
        return None
    item0 = cp[0]
    if isinstance(item0, dict):
        if isinstance(item0.get("top_probs"), list):
            return "top_probs"
        if isinstance(item0.get("top_logprobs"), list):
            return "top_logprobs"
    return None


def sum_p0(resp):
    """Диагностика (НЕ критерий): Σ вероятностей по кандидатам item[0].
    post-sampling: `prob` — прямая вероятность (Σ ≈ 1.0 при насыщенном отчёте);
    pre-sampling: `logprob` — Σ exp(logprob) (Б7: ≤ 1.0 ВСЕГДА)."""
    cp = resp.get("completion_probabilities")
    if not isinstance(cp, list) or len(cp) == 0:
        return None
    item0 = cp[0]
    if not isinstance(item0, dict):
        return None
    tp = item0.get("top_probs")
    if isinstance(tp, list) and tp:
        try:
            return float(sum(x["prob"] for x in tp))
        except (KeyError, TypeError, ValueError):
            return None
    tl = item0.get("top_logprobs")
    if isinstance(tl, list) and tl:
        try:
            return float(sum(math.exp(t["logprob"]) for t in tl))
        except (KeyError, TypeError, ValueError):
            return None
    return None


def row_halt(row, status, halt1, resp, top_k_sent):
    """Ассерты 2–4 на ОДНОЙ строке (только при received + парсимом теле).
    Порядок внутри строки — строго по ТЗ: 1 → 2 → 3 → 4a → 4b.
    Возврат: (halt_reason|None, echo|None, N|None, sp0|None)."""
    if status == "sent" or halt1 is not None or resp is None:
        return halt1, None, None, None
    # ассерт 2: поле completion_probabilities ЕСТЬ
    if "completion_probabilities" not in resp:
        return "no_field", None, None, None
    echo = resp.get("generation_settings")
    has_echo = isinstance(echo, dict)
    # ассерт 3: только off-строка (row 4); top_k «НЕТ в эхе» включает случай,
    # когда объекта echo нет целиком (поля нет — объект пуст)
    if row == 4:
        if (not has_echo) or ("top_k" not in echo) or echo["top_k"] is None:
            return "echo_no_topk", echo, None, None
        v = echo["top_k"]
        if v <= 20:
            return "off_default_low", echo, None, None
    # ассерт 4a(i): echo/generation_settings отсутствует ЦЕЛИКОМ (лестничные
    # и off-строка, прошедшая ассерт 3)
    if not has_echo:
        return "echo_no_psp", None, None, None
    # ассерт 4a(ii): post_sampling_probs на КАЖДОЙ полученной строке
    if "post_sampling_probs" not in echo:
        return "echo_no_psp", None, None, None
    if echo["post_sampling_probs"] is not True:
        return "echo_mismatch", None, None, None
    # ассерт 4b: лестничные — echo.top_k == k
    if row in LADDER_ROWS:
        if "top_k" not in echo:
            return "echo_no_topk", echo, None, None
        if echo["top_k"] != top_k_sent:
            return "echo_mismatch", echo, None, None
    n = report_len(resp)
    return None, echo, n, sum_p0(resp)


def fetch_props(port, timeout=5):
    with urllib.request.urlopen("http://127.0.0.1:%d/props" % port, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    # ассерты ДО лонча (ТЗ: «число (7) — ассерт ДО лонча сервера»)
    assert len(PLAN) == N_PLAN == 7, "план не 7 запросов"
    assert [p[0] for p in PLAN] == ["S3b-k1-probe", "S3b-k2-probe", "S3b-k4-probe",
                                    "S3b-off-probe", "S3b-k1-ctrl", "S3b-k2-ctrl",
                                    "S3b-k4-ctrl"], "порядок планa != таблице ТЗ"
    st = selftest()  # до сервера: 50/50 миров ТЗ
    prompts = load_prompts()

    port = pick_port(range(8120, 8160))
    srv = LlamaServer(port, "/tmp/e94_s3b_server.log")
    notes = []
    art = None
    t_start = time.time()
    srv.start()
    try:
        srv.wait_health(timeout=300)
        props = fetch_props(port)
    except Exception as e:
        notes.append("server: %r" % (e,))
        art = dict(exp="E94_S3b", halt_reason="server_failed", notes=notes,
                   requests=[], selftest=st)
        srv.kill()
        with open(ART, "w") as f:
            json.dump(san(art), f, ensure_ascii=False, indent=1)
        print("[abort] server: %r" % (e,), flush=True)
        return

    server = dict(port=port, pid=srv.pid, dri=srv.device_fd(),
                  args=["-m", "/data/Qwen/qwen3.5-0.8B-f16.gguf"] + SERVER_ARGS
                      + ["--port", str(port)],
                  model="/data/Qwen/qwen3.5-0.8B-f16.gguf",
                  model_reported=None, model_matches=False, n_ctx=None)
    mp = (props or {}).get("model_path") or (props or {}).get("model_alias") or ""
    server["model_reported"] = mp
    server["model_matches"] = bool(props is not None and os.path.basename(
        "/data/Qwen/qwen3.5-0.8B-f16.gguf") == os.path.basename(str(mp)))
    server["n_ctx"] = (props or {}).get("default_generation_settings", {}).get("n_ctx")

    recs = []
    for row, (tag, role, k) in enumerate(PLAN, start=1):
        b = body(prompts[role], k)
        status, halt1, resp, wall = send(port, b)
        halt, echo, n, sp0 = row_halt(row, status, halt1, resp, k)
        emit = te = None
        if resp is not None:
            toks = resp.get("tokens") or []
            emit = toks[0] if toks else None
            te = resp.get("tokens_evaluated")
        recs.append(dict(
            row=row, tag=tag, role=role,
            addr="Q2 w%d k=%d" % (ADDR[role]["w"], ADDR[role]["k"]),
            k=ADDR[role]["k"], top_k_sent=k,
            payload=b,  # дословно: prompt (токены Q2-канона) + все поля
            status=status, per_row_halt_reason=halt, wall_s=wall,
            emit=emit, te=te,
            echo=echo, N=n, report_field=report_field(resp), sum_p0=sp0,
        ))
        print("[req] row=%d %s status=%s halt=%s N=%s"
              % (row, tag, status, halt, n), flush=True)

    # решение: halt — по порядку строк 1→7 (внутри строки — первый ассерт)
    halts = [(r["row"], r["per_row_halt_reason"]) for r in recs
             if r["per_row_halt_reason"]]
    n_p = tuple(r["N"] for r in recs[:3])
    n_c = tuple(r["N"] for r in recs[4:7])
    n_off = recs[3]["N"]
    rule, kind, verdict, halt_reason, reading = decide(
        n_p, n_c, n_off, halts)

    # per-row reading: короткое, по строке
    for r in recs:
        if r["per_row_halt_reason"]:
            r["reading"] = "halt: %s (ассерты 1–4)" % r["per_row_halt_reason"]
        elif r["status"] == "received" and r["N"] is not None:
            r["reading"] = ("N=%s, sum_p0=%s (эмиссия T=1.0 не читается как довод)"
                            % (r["N"], ("%.4f" % r["sum_p0"]) if r["sum_p0"] is not None else None))
        else:
            r["reading"] = "ответ не получен (status=%s)" % r["status"]

    # r15-ремедиация (2_REVIEW_E94_S3b_exec_r1, COSMETIC): «жадная инвариантность» —
    # свойство T=0, не T=1.0; реальное основание не-чтения эмиссий — вырожденная
    # верша отчёта (p0, вычислено из ЭТОГО артефакта — динамически, SYCL-шум)
    _p0s = [r["sum_p0"] for r in recs if r["sum_p0"] is not None]
    _p0min = ("%.4f" % min(_p0s)) if _p0s else "н/д"
    summary = dict(
        N_probe=list(n_p), N_ctrl=list(n_c), N_off=n_off,
        sum_p0={r["tag"]: r["sum_p0"] for r in recs},
        rule=rule, kind=kind, verdict=verdict,
        halt_reason=halt_reason if (rule in (3, 4, 5) and halt_reason) else None,
        reading=reading,
        note=("эмиссии 7 запросов не читаются как довод о механизме (T=1.0: эмиссия = "
              "вырожденная вершина отчёта, p0 ≥ %s на каждой строке этого артефакта; "
              "предрегистрация «эмиссии — диагностика»; жадная инвариантность — свойство "
              "T=0, не T=1.0); ни один вывод S3b не цитируется как про M1/M2 и не правит "
              "gate2 в S4pp.json" % _p0min),
    )

    art = dict(
        exp="E94_S3b",
        spec=("docs/dynamic/0_BRIEF_E94_S3b_TZ.md (r19) + пакет "
              "docs/dynamic/0_PACKET_E94_S3b_TZ_r19.md; конвенция A/B — СВОЙСТВО "
              "СЕРВЕРА, одна на оба адреса (факт S4pp: n_tl=20 при n_probs=20, без +1)"),
        started_local=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        wall_s=round(time.time() - t_start, 1),
        halt_reason=halt_reason,
        server=server,
        selftest=st,
        prereg=dict(
            order="4 → 3 → B3 → 1a → 1b → 2 → 5 (первая применимая)",
            invariant_B3=("отчёт не может быть длиннее n_probs=20 (исходник: n_probs = "
                          "min(max_probs, n_probs_request), цикл i < n_probs); любой КОМПОНЕНТ = "
                          "длина отчёта (N) на строке (эхо-поля, напр. off-строки top_k=40, НЕ "
                          "считаются компонентами) ЛЮБОГО вектора (v_p И v_c) > 20 ИЛИ N_off > 20 "
                          "→ правило 5"),
            readings=dict(
                r4="рука невалидна (mech_invalid; echo_no_topk/echo_no_psp — mech_void)",
                r3="рука невалидна: пустой отчёт (mech_invalid, report_empty)",
                r1a="top_k применяется к пост-сэмплированному набору (механика установлена)",
                r1b="top_k применяется и B4 держит потолок",
                r2="top_k не применяется к набору (механика отсутствует)",
                r5="catch-all: no_preregistered_reading (mech_void)"),
            n_definition=("N(x) = длина набора кандидатов item[0] (ТЗ: len(completion_probabilities[0]); "
                          "на этом форке cp[0] — dict). ВАЖНО: поле набора зависит от режима — "
                          "post_sampling_probs=true: `top_probs` (итер с `prob`); "
                          "post_sampling_probs=false (pre-sampling, факт B5: n_tl=20): `top_logprobs` "
                          "(итер с `logprob`). v1-прогон (заменён) читал только top_logprobs → ложный "
                          "N=0 → ложный rule 3; v2 — оба поля, выбор зафиксирован в "
                          "requests[i].report_field; cp == [] / [0] нет → N=0"),
            noninformative=["эмиссии (T=1.0: при насыщенном отчёте эмиссия = вырожденная вершина отчёта; жадная инвариантность — свойство T=0)",
                            "sum_p0 (чистая диагностика, Σ <= 1.0, порогов НЕТ)"]),
        requests_n=len(recs),
        requests_received=sum(1 for r in recs if r["status"] == "received"),
        notes=notes,
        summary=san(summary),
        requests=san(recs),
    )
    if srv.pid:
        try:
            srv.kill()
        except Exception as e:
            notes.append("kill: %r" % (e,))
    with open(ART, "w") as f:
        json.dump(san(art), f, ensure_ascii=False, indent=1)
    print("[done] %s rule=%s verdict=%s halt=%s wall=%.1fs"
          % (ART, rule, verdict, halt_reason, time.time() - t_start), flush=True)


if __name__ == "__main__":
    main()
