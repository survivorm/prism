#!/usr/bin/env python3
"""
E94/S5 — строгая реконструкция эмиссента из logprobs + DRY. 0 GPU.
TZ: docs/dynamic/0_PACKET_E94_S5_TZ.md

Порт llama_sampler_dry_apply из /home/survivor/work/llama.cpp/src/llama-sampler.cpp
(git 0b8cfb8fe) literatim, включая обратную индексацию ring_buffer::rat (:95).
Реализация DRY ровно одна (dry_apply): unit-тест и прогон идут через неё, копий
алгоритма в файле нет (AGENTS.md §2).

Смоук:  ./run_one.sh exp94_s5_dryrecon -- --selftest
Полный: ./run_one.sh exp94_s5_dryrecon
"""
import argparse
import hashlib
import json
import math
import mmap
import os
import struct
import subprocess
import tarfile
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAR = os.path.join(REPO, "results/e94/raw_q1.json.tar.gz")
GGUF = "/data/Qwen/qwen3.5-0.8B-f16.gguf"
OUT = os.path.join(REPO, "results/e94/S5.json")
SRC = "/home/survivor/work/llama.cpp/src/llama-sampler.cpp"

# Значения, которые сервер ЭХОМ вернул в generation_settings каждого записанного
# ответа (проверяется в run_rep, при расхождении — падение, а не «допустим»).
DRY_MULT, DRY_BASE, DRY_ALLOWED, DRY_LASTN = 1.0, 1.0, 2, 64
BREAKERS = [b"\n", b":", b'"', b"*"]     # байты: окно DRY оперирует байтами
TIE_TOL = 1e-9
# include/llama.h:92-98: UNDEFINED0 NORMAL1 UNKNOWN2 CONTROL3 USER_DEFINED4 UNUSED5 BYTE6
# ВНИМАНИЕ: BYTE(6) здесь НЕ исключается - у qwen35 байтовые куски лежат в тексте
# токена и декодируются decode_text(); спец-токены ChatML-подобные дают "".
NO_TEXT_TYPES = {0, 2, 3, 4, 5}


# --------------------------------------------------------------- GGUF vocab
def gguf_meta(path):
    """Метаданные GGUF через mmap (не читает 1.5 ГБ в RAM)."""
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        if mm[:4] != b"GGUF":
            raise SystemExit("not GGUF")
        off = [4]

        def u(fmt):
            v = struct.unpack_from(fmt, mm, off[0])[0]
            off[0] += struct.calcsize(fmt)
            return v

        def s():
            n = u("<Q")
            v = mm[off[0]:off[0] + n].decode("utf-8", "replace")
            off[0] += n
            return v

        SCAL = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q"}

        def val(t):
            if t == 8:
                return s()
            if t == 9:
                et, cnt = u("<I"), u("<Q")
                return [val(et) for _ in range(cnt)]
            if t == 7:
                return u("<B") != 0
            return u(SCAL[t])

        kv = {}
        ver = u("<I")
        u("<Q")                      # n_tensors
        for _ in range(u("<Q")):
            k = s()
            kv[k] = val(u("<I"))
        if "tokenizer.ggml.tokens" not in kv:
            raise SystemExit("no tokenizer.ggml.tokens")
        return kv, ver


# GPT-2/llama.cpp byte decoder: порт unicode_utf8_to_byte для BPE-ветки
# (llama-vocab.cpp:3649 `llama_decode_text`, escape_whitespaces=false у pre=qwen35;
# таблица = обратная к byte_encoder: диапазоны 33..126, 161..172, 174..255 остаются
# собой, остальные байты получают cpt 256+n). Строка с 'Ġ' -> ' ', 'Ċ' -> '\n'.
def _byte_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) \
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {c: bytes([b]) for c, b in zip(cs, bs)}


BYTE_DEC = _byte_decoder()


def decode_text(piece):
    """
    literatim llama_decode_text: на выходе БАЙТЫ (в llama.cpp всё есть std::string
    байтов). Кодпоинт -> байт по BYTE_DEC; неизвестный -> литерал '[UNK_BYTE_0x..]'
    (это ровно ветка out_of_range в llama-vocab.cpp:3358-3365, включая её же баг
    `decoded_text += text`). Разборка на UTF-8 - только для отчётов.
    """
    out = bytearray()
    for ch in piece:
        cpt = ord(ch)
        if cpt in BYTE_DEC:
            out += BYTE_DEC[cpt]
        else:
            u = ch.encode("utf-8")
            out += b"[UNK_BYTE_0x" + "".join(f"{c:02x}" for c in u).encode() \
                + piece.encode("utf-8", "replace") + b"]"
    return bytes(out)


def build_texts(kv):
    """text(id) так, как его видит DRY: detokenize(..., remove_special=true)."""
    toks = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type") or [1] * len(toks)
    texts = []
    for i, t in enumerate(toks):
        # token_to_piece (:3579-3581): спец-атрибуты при remove_special дают ""
        texts.append(b"" if types[i] in NO_TEXT_TYPES else decode_text(t))
    # get_overlapping_token_sequences (:3330-3335): head = каждый токен, чей текст
    # СОДЕРЖИТ breaker; все наши breaker'ы односимвольные => хвост пуст => это и
    # single-token breaker, освобождённый от штрафа (:3568-3581).
    brk = frozenset(i for i, t in enumerate(texts) if any(b in t for b in BREAKERS))
    return texts, brk


# ------------------------------------------------------------------ DRY port
def dry_apply(window, breakers, mult=DRY_MULT, base=DRY_BASE, allowed=DRY_ALLOWED,
              variant="real"):
    """
    window — хронологический список последних <=DRY_LASTN принятых токенов.
    Возвращает (penalty_map, repeat_count). Один порт, используется и тестом, и
    прогоном.

    variant != "real" — намеренно сломанные реализации, ТОЛЬКО для negative control
    (замечание ревьюера r1: «контроль penalty:=0 не прогоняет код DRY, любой баг
    переноса его проходит»). Отличия:
      rat_forward — rat(i) читает i-й СТАРЫЙ токен вместо i-го от нового (:95);
      penalize_head — штраф навешивается на первый токен повтора, а не на следующий;
      no_breakers — шаг 1 (rep_limit) пропущен, breaker'ы не обрезают окно.
    Все три — правдоподобные ошибки переноса; negative control обязан их ловить,
    иначе тест не имеет различающей силы.
    """
    if variant not in ("real", "rat_forward", "penalize_head", "no_breakers"):
        raise SystemExit(f"unknown variant {variant!r}")
    n = len(window)                              # last_n_repeat
    if n <= allowed:                             # :3393 (min(|tokens|, last_n))
        return {}, [0] * n

    def rat(i):                                  # :95 data[(first+sz-i-1)%cap]
        return window[i] if variant == "rat_forward" else window[n - 1 - i]

    # шаг 1 (:3424-3458): rep_limit по sequence breakers
    rep_limit = n
    for i in range(n):
        if variant == "no_breakers":
            break
        if rat(i) in breakers:                   # seq_len == 0 для 1-символьных
            rep_limit = i
            break
    if rep_limit < allowed:
        return {}, [0] * n

    # шаг 2 (:3486-3527): обратный Z-алгоритм
    rc = [0] * n
    last = n - 1
    rt = lt = 0
    for k in range(1, n):
        if k > rt:
            m = 0
            while m + k < n and rat(m) == rat(m + k):
                m += 1
            rc[last - k] = min(m, rep_limit)
            if m > 0:
                lt, rt = k, k + m - 1
        else:
            p = k - lt
            right_part_len = rt - k + 1
            if rc[last - p] < right_part_len:
                rc[last - k] = min(rc[last - p], rep_limit)
            else:
                i2 = rt + 1
                while i2 < n and rat(i2) == rat(i2 - k):
                    i2 += 1
                m = min(i2 - k, rep_limit)
                rc[last - k] = m
                lt, rt = k, i2 - 1

    # шаг 3 (:3545-3556): штраф на токен, идущий ПОСЛЕ найденного повтора
    maxrep = {}
    for i in range(n - 1):
        r = rc[i]
        if r >= allowed:
            tok = rat(i if variant == "penalize_head" else n - 2 - i)
            if maxrep.get(tok, -1) < r:
                maxrep[tok] = r

    # шаг 4 (:3565-3586): сам штраф; breaker-токены пропускаются
    pen = {}
    max_exponent = int(88.7228391 / math.log(base)) if base > 1.000001 else 0
    for tok, r in maxrep.items():
        if tok in breakers:
            continue
        exp = r - allowed
        if max_exponent > 0 and exp > max_exponent:
            exp = max_exponent
        pen[tok] = mult * (base ** exp)
    return pen, rc


def unit_test(breakers):
    """
    Доковый пример исходника (:3475-3479):
      Last N tokens: a b c c b c y a b c
      Repeat counts: 0 0 3 1 0 2 0 0 0 0
    Плюс граничание breaker'а и «окно короче allowed».
    """
    wd = {c: 1000 + i for i, c in enumerate("abcy")}
    win = [wd[c] for c in "abccbcyabc"]          # a b c c b c y a b c
    pen, rc = dry_apply(win, frozenset())
    want = [0, 0, 3, 1, 0, 2, 0, 0, 0, 0]
    assert rc == want, f"unit repeat_count {rc} != {want}"
    assert pen == {wd["c"]: 1.0, wd["y"]: 1.0}, f"unit penalty {pen}"
    assert set(pen.values()) == {1.0}

    # окно короче allowed => нет штрафа вообще
    assert dry_apply(win[:2], frozenset())[0] == {}
    # (а) breaker-токен освобождается от штрафа, но чужой штраф остаётся: y стоит на
    #     дистанции 3 от нового => rep_limit=3 >= allowed=2, штраф считается (:3578)
    assert dry_apply(win, frozenset([wd["y"]]))[0] == {wd["c"]: 1.0}, \
        "breaker-освобождение работает неверно"
    # (б) breaker на дистанции 1 => rep_limit=1 < allowed => штрафа нет вообще (:3459)
    assert dry_apply(win, frozenset([wd["b"]]))[0] == {}, "rep_limit не обрезал штраф"
    # penalty масштабируется множителем, при base=1.0 не зависит от длины повтора
    assert dry_apply(win, frozenset(), mult=0.5)[0] == {wd["c"]: 0.5, wd["y"]: 0.5}
    return {"repeat_count": rc, "expected": want, "penalized": sorted(pen), "ok": True}


# ----------------------------------------------------------- реконструкция
def classify(cands, emitted, lp_emitted, pen):
    """Корзина TZ §4 + двусторонняя проверка (предсказан vs наблюдался флип)."""
    lp = dict(cands)
    if emitted not in lp:
        lp[emitted] = lp_emitted
    ordered = sorted(lp.values(), reverse=True)
    lp_floor = ordered[-1]                      # худший из показанных (порог хвоста)
    adj = {cid: lg - pen.get(cid, 0.0) for cid, lg in lp.items()}
    best = max(adj.values())
    winners = [cid for cid, v in adj.items() if v > best - TIE_TOL]
    vertex = max(lp, key=lambda c: lp[c])
    if best < lp_floor - TIE_TOL:
        bucket = "INCONCLUSIVE"                 # победитель мог лежать за top-20
    elif winners == [emitted]:
        bucket = "EXPLAINED"
    elif emitted in winners:
        bucket = "TIE"
    else:
        bucket = "UNEXPLAINED"
    obs_flip = emitted != vertex
    return {
        "bucket": bucket, "vertex": vertex, "emitted": emitted,
        "obs_flip": obs_flip,
        "pred_flip": bucket != "INCONCLUSIVE" and vertex not in winners,
        "g12": round(ordered[0] - ordered[1], 6) if len(ordered) > 1 else None,
        "g_need": round(lp[vertex] - lp[emitted], 6) if obs_flip else 0.0,
        "pen_emitted": pen.get(emitted, 0.0),
        "pen_vertex": pen.get(vertex, 0.0),
        "n_pen": len(pen),
    }


def rep_rows(tf, nm, breakers, apply_dry, text_sink, variant="real"):
    d = json.load(tf.extractfile(nm))
    prompt = d["request"]["prompt"]
    gen = d["response"]["tokens"]
    cps = d["response"]["completion_probabilities"]
    gs = d["response"]["generation_settings"]
    if not (isinstance(prompt, list) and prompt and all(isinstance(x, int) for x in prompt)):
        raise SystemExit(f"{nm}: request.prompt не список id — окно DRY не восстановить")
    if len(gen) != len(cps):
        raise SystemExit(f"{nm}: tokens {len(gen)} != completion_probabilities {len(cps)}")
    for key, want in (("dry_multiplier", DRY_MULT), ("dry_base", DRY_BASE),
                      ("dry_allowed_length", DRY_ALLOWED),
                      ("dry_penalty_last_n", DRY_LASTN)):
        if abs(float(gs.get(key, -1)) - want) > 1e-9:
            raise SystemExit(f"{nm}: {key}={gs.get(key)} != {want} — канон не тот")
    if float(gs.get("temperature", -1)) != 0.0:
        raise SystemExit(f"{nm}: temperature != 0, greedy не тот")
    rows = []
    full = list(prompt) + list(gen)
    npg = len(prompt)
    for t, cp in enumerate(cps):
        for c in [cp] + list(cp.get("top_logprobs", [])):
            if "token" in c:
                text_sink[c["id"]] = c["token"]
        end = npg + t
        win = full[max(0, end - DRY_LASTN):end]
        pen = dry_apply(win, breakers, variant=variant)[0] if apply_dry else {}
        r = classify([(c["id"], c["logprob"]) for c in cp["top_logprobs"]],
                     cp["id"], cp["logprob"], pen)
        r["doc"] = os.path.basename(nm)
        r["pos"] = t
        rows.append(r)
    return rows


def summarize(tf, names, breakers, apply_dry, text_sink, variant="real"):
    agg, conf = Counter(), Counter()
    flips, incon, gn_ok, gn_bad = [], [], 0, []
    per_doc, ctl_bad, nonexpl = {}, 0, []
    n_pen_pos = 0
    for nm in names:
        rows = rep_rows(tf, nm, breakers, apply_dry, text_sink, variant)
        dsum_raw = Counter(r["bucket"] for r in rows)
        dsum = {k: dsum_raw.get(k, 0) for k in ("EXPLAINED", "TIE", "UNEXPLAINED", "INCONCLUSIVE")}
        per_doc[os.path.basename(nm)] = dict(dsum)
        nonexpl += [f'{r["doc"]}:{r["pos"]}' for r in rows if r["bucket"] != "EXPLAINED"]
        if not apply_dry:
            ctl_bad += sum(1 for r in rows if r["bucket"] != "EXPLAINED")
            for r in rows:
                conf[("obs1" if r["obs_flip"] else "obs0",
                      "pred1" if r["pred_flip"] else "pred0")] += 1
                agg[r["bucket"]] += 1
            continue
        for r in rows:
            agg[r["bucket"]] += 1
            conf[("obs1" if r["obs_flip"] else "obs0",
                  "pred1" if r["pred_flip"] else "pred0")] += 1
            if r["n_pen"] > 0:
                n_pen_pos += 1
            if r["obs_flip"]:
                flips.append(r)
                if r["g_need"] < 1.0 - 1e-9:
                    gn_ok += 1
                else:
                    gn_bad.append({"doc": r["doc"], "pos": r["pos"], "g_need": r["g_need"]})
            if r["bucket"] == "INCONCLUSIVE":
                incon.append(f'{r["doc"]}:{r["pos"]}')
    total = sum(agg.values())
    # явные нули по всем корзинам (r2-ревью: TIE=0 обязан быть виден в buckets,
    # а не только в verdict.tie_zero)
    buckets = {k: agg.get(k, 0) for k in ("EXPLAINED", "TIE", "UNEXPLAINED", "INCONCLUSIVE")}
    tp, fp, fn = conf["obs1", "pred1"], conf["obs0", "pred1"], conf["obs1", "pred0"]
    # раскладка по механизму (TZ §4 «побочно»): флип = DRY-штраф на ВЕРШИНЕ
    # (pen_vertex>0) сдвигает её ниже эмиссента при g_need < штрафа
    mech = {
        "vertex_penalized_emitted_unpenalized":
            sum(1 for f in flips if f["pen_vertex"] > 0.0 and f["pen_emitted"] == 0.0),
        "pen_vertex_covers_g_need":
            sum(1 for f in flips if f["pen_vertex"] >= f["g_need"] - 1e-9),
        "g_need_lt_1nat":
            sum(1 for f in flips if f["g_need"] < 1.0 - 1e-9),
        "emitted_also_penalized":
            sum(1 for f in flips if f["pen_emitted"] > 0.0),
        "no_penalty_at_all_on_position":
            sum(1 for f in flips if f["n_pen"] == 0),
    }
    return {
        "docs": len(names), "positions": total, "buckets": buckets,
        "confusion": {"TP_obs1_pred1": tp, "FP_obs0_pred1": fp,
                      "FN_obs1_pred0": fn, "TN_obs0_pred0": conf["obs0", "pred0"]},
        "obs_flip_rate": round((tp + fn) / max(total, 1), 6),
        "pred_flip_rate": round((tp + fp) / max(total, 1), 6),
        "positions_with_dry_penalty": n_pen_pos,
        "positions_with_dry_penalty_rate": round(n_pen_pos / max(total, 1), 6),
        "flip_mechanism": mech,
        "flips_total": tp + fn,
        # r1-ревью FINDING 1: «объяснено DRY» = flip ∧ корзина EXPLAINED (не TP
        # конфьюжна — при гипотетическом UNEXPLAINED>0 сдвиг на третий токен был бы
        # зачтён TP и завысил бы счётчик)
        "flips_explained_by_dry": sum(1 for f in flips if f["bucket"] == "EXPLAINED"),
        # r1-ревью FINDING 2: набор позиций (doc:pos) вне EXPLAINED — для сверки
        # control_no_dry по МНОЖЕСТВАМ, а не по счетам
        "non_explained": sorted(set(nonexpl)),
        "g_need_lt_1nat": gn_ok, "g_need_ge_1nat": gn_bad[:20],
        "g_need_max": round(max([f["g_need"] for f in flips]), 6) if flips else None,
        "inconclusive_examples": incon[:20], "per_doc": per_doc,
        "flips_detail": [{"doc": f["doc"], "pos": f["pos"], "vertex": f["vertex"],
                          "emitted": f["emitted"], "g12": f["g12"], "g_need": f["g_need"],
                          "pen_emitted": f["pen_emitted"], "pen_vertex": f["pen_vertex"],
                          "pred_flip": f["pred_flip"], "bucket": f["bucket"]}
                         for f in flips],
    }, ctl_bad


def negative_controls(tf, names, breakers):
    """Negative control (r1-ревью п.2): тест обязан ловить правдоподобные баги переноса.
    Для каждого сломанной реализации считаем те же корзины; «ловится» = хотя бы одна
    позиция стала не-EXPLAINED, либо набор штрафов разошёлся с эталонным. Если
    какой-то вариант проходит — наш тест не различает реализации, и «UNEXPLAINED==0»
    само по себе ничего не стоит (это ровно то, в чём упрекнул контроль penalty:=0).
    r1-ревью FINDING 5: прогон по ВСЕМ r1-документам + scope записан в JSON."""
    out = {}
    sink_ref = {}
    ref, _ = summarize(tf, names, breakers, True, sink_ref)
    for v in ("rat_forward", "penalize_head", "no_breakers"):
        sink = {}
        s, _ = summarize(tf, names, breakers, True, sink, variant=v)
        differs = (s["buckets"] != ref["buckets"]
                   or s["confusion"] != ref["confusion"])
        out[v] = {
            "caught": differs,
            "buckets": s["buckets"],
            "ref_buckets": ref["buckets"],
            "extra_unexplained": s["buckets"].get("UNEXPLAINED", 0)
            - ref["buckets"].get("UNEXPLAINED", 0),
            "fp": s["confusion"]["FP_obs0_pred1"], "ref_fp": ref["confusion"]["FP_obs0_pred1"],
            "fn": s["confusion"]["FN_obs1_pred0"], "ref_fn": ref["confusion"]["FN_obs1_pred0"],
        }
    return {"variants": out,
            "scope": {"rep": "r1", "docs": len(names),
                      "positions": ref["positions"]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    kv, ver = gguf_meta(GGUF)
    texts, brk = build_texts(kv)
    ut = unit_test(brk)
    git = subprocess.run(["git", "-C", os.path.dirname(SRC), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"[vocab] n={len(texts)} pre={kv.get('tokenizer.ggml.pre')!r} "
          f"breaker_ids={len(brk)} gguf_v={ver}", flush=True)
    print(f"[unit] {ut}", flush=True)

    with tarfile.open(TAR) as tf:
        names_all = sorted(tf.getnames())
        if a.selftest:
            for rep in ("r1", "r2"):
                for nm in [m for m in names_all if m.endswith(f"_{rep}.json")][:2]:
                    sink = {}
                    rows = rep_rows(tf, nm, brk, True, sink)
                    bad = [(i, texts[i].decode("utf-8", "replace"), s)
                           for i, s in sink.items() if texts[i].decode("utf-8", "replace") != s]
                    b = Counter(r["bucket"] for r in rows)
                    print(f"[smoke] {nm} pos={len(rows)} buckets={dict(b)} "
                          f"vocab_mismatch={len(bad)}", bad[:3], flush=True)
                    c = Counter((r["obs_flip"], r["pred_flip"]) for r in rows)
                    print(f"[smoke]   conf obs/pred {dict(c)}", flush=True)
            print("SELFTEST_OK")
            return

        payload = {
            "tz": "docs/dynamic/0_PACKET_E94_S5_TZ.md",
            "unit_test": ut,
            "src": {"file": SRC, "git_head": git,
                    "sha1": hashlib.sha1(open(SRC, "rb").read()).hexdigest(),
                    "lines_cited": [95, 3330, 3393, 3424, 3459, 3486, 3545, 3565, 3645]},
            "input": {"tar": os.path.relpath(TAR, REPO),
                      "md5": hashlib.md5(open(TAR, "rb").read()).hexdigest()},
            "dry": {"multiplier": DRY_MULT, "base": DRY_BASE, "allowed_length": DRY_ALLOWED,
                    "penalty_last_n": DRY_LASTN,
                    # bytes -> str: артефакт обязан быть JSON-сериализуем (AGENTS.md #4)
                    "breakers": [b.decode() for b in BREAKERS],
                    "breaker_token_ids": len(brk)},
            "vocab": {"n_tokens": len(texts), "pre": kv.get("tokenizer.ggml.pre")},
            "runs": {}}

        r1_names = []
        for rep in ("r1", "r2"):
            names = [m for m in names_all if m.endswith(f"_{rep}.json")]
            if a.limit_docs:
                names = names[:a.limit_docs]
            if rep == "r1":
                r1_names = names
            sink = {}
            summ, _ = summarize(tf, names, brk, True, sink)
            if rep == "r1":
                sink_r1 = sink   # сервер-тексты r1 — для сверки промпт-идов (FINDING 3)
            ctrl, _ = summarize(tf, names, brk, False, sink)
            # r1-ревью FINDING 2: контроль «ошибиться ровно на флипах» — равенство
            # НАБОРОВ (doc:pos), а не счетов (равенство счетов не исключает «один
            # лишний флип + один ложный»)
            flips_set = {f"{f['doc']}:{f['pos']}" for f in summ["flips_detail"]}
            ctrl_mism = set(ctrl["non_explained"])
            summ["control_no_dry"] = {
                "mismatches": len(ctrl_mism),
                "flips_total": summ["flips_total"],
                "expected_eq_obs_flips": ctrl_mism == flips_set,
                "mismatch_positions": sorted(ctrl_mism),
                "note": "без штрафа argmax==emitted обязан ломаться ровно на флипах (сверка по множествам doc:pos)"}
            mism = [(i, texts[i].decode("utf-8", "replace"), s)
                    for i, s in sink.items() if texts[i].decode("utf-8", "replace") != s]
            impact = []
            for i, o, s in mism:
                server_breaker = any(b in s.encode("utf-8", "replace") for b in BREAKERS)
                impact.append({"id": i, "ours": o, "server": s,
                               "our_breaker": i in brk, "server_breaker": server_breaker,
                               "penalty_affected": (i in brk) != server_breaker})
            summ["vocab_text_check"] = {
                "ids_checked": len(sink), "mismatches": len(mism),
                "scope": ("серверный текст виден только для id, попавших в top_logprobs "
                          "ответа (n=ids_checked из 248320); влияние ограничено breaker-"
                          "набором — проверено impact[] и control_no_dry"),
                "examples": [{"id": i, "ours": o, "server": s}
                             for i, o, s in mism[:10]],
                "impact": impact,
                "any_penalty_affected": any(x["penalty_affected"] for x in impact)}
            payload["runs"][rep] = summ

        # r1-ревью FINDING 3: breaker-принадлежность промпт-идов (влияет на rep_limit)
        # зафиксировать + сверить с сервером там, где это возможно из raw (текст сервера
        # виден только для id, попавших в top_logprobs ответа)
        prompt_ids = set()
        for nm in r1_names:
            prompt_ids.update(json.load(tf.extractfile(nm))["request"]["prompt"])
        pb = sorted(i for i in prompt_ids if i in brk)
        pb_verified, pb_unverified = [], []
        for i in pb:
            if i in sink_r1:
                srv = sink_r1[i]
                srv_brk = any(b in srv.encode("utf-8", "replace") for b in BREAKERS)
                pb_verified.append({"id": i,
                                    "ours": texts[i].decode("utf-8", "replace"),
                                    "server": srv,
                                    "affected": (i in brk) != srv_brk})
            else:
                pb_unverified.append(i)
        r1vt = payload["runs"]["r1"]["vocab_text_check"]
        r1vt["prompt_brk_ids"] = pb
        r1vt["prompt_brk_ids_verified"] = pb_verified
        r1vt["prompt_brk_ids_unverifiable_from_raw"] = pb_unverified
        # any_penalty_affected — по ответным И промпт-токенам (сверенным)
        r1vt["any_penalty_affected"] = (any(x["penalty_affected"] for x in r1vt["impact"])
                                       or any(x["affected"] for x in pb_verified))

        r1 = payload["runs"]["r1"]
        b, c = r1["buckets"], r1["confusion"]
        unexp, tie, inc = b.get("UNEXPLAINED", 0), b.get("TIE", 0), b.get("INCONCLUSIVE", 0)
        controls_ok = all(v["control_no_dry"]["expected_eq_obs_flips"]
                          for v in payload["runs"].values())
        vocab_ok = all(v["vocab_text_check"]["mismatches"] == 0
                       for v in payload["runs"].values())
        vocab_affected = any(v["vocab_text_check"]["any_penalty_affected"]
                             for v in payload["runs"].values())
        # TZ §4.1 (предрегистр, ДО запуска): H0 = UNEXPLAINED==0 && TIE==0 &&
        # INCONCLUSIVE<=1% && контроль чист. vocab-совпадение — отдельный integrity-чек,
        # в §4 его НЕТ: он не гейтит H0 сам по себе, но если бы несоответствие изменило
        # breaker-принадлежность (vocab_affected), пересчёт штрафа был бы невалиден.
        payload["verdict"] = {
            "rule": ("TZ §4.1: UNEXPLAINED==0 && TIE==0 && INCONCLUSIVE<=1% && контроль чист "
                     "(vocab-совпадение — integrity-чек, в §4 отсутствует)"),
            "unexplained": unexp, "unexplained_zero": unexp == 0,
            "tie_zero": tie == 0,
            "inconclusive": inc,
            "inconclusive_le_1pct": inc <= 0.01 * r1["positions"],
            "inconclusive_pct": round(100.0 * inc / max(r1["positions"], 1), 4),
            "controls_ok": controls_ok,
            "flips_total": r1["flips_total"],
            "flips_explained_by_dry": r1["flips_explained_by_dry"],
            "confusion": c,
            "vocab_ok": vocab_ok, "vocab_penalty_affected": vocab_affected,
            "H0_supported": False, "H0_supported_and_valid": False}
        payload["verdict"]["H0_supported"] = bool(
            unexp == 0 and tie == 0 and inc <= 0.01 * r1["positions"] and controls_ok)
        payload["verdict"]["H0_supported_and_valid"] = bool(
            payload["verdict"]["H0_supported"] and not vocab_affected)
        # negative control ДО json.dump (AGENTS.md #4: артефакт, а не stdout);
        # r1-ревью FINDING 5: все r1-документы, scope в JSON
        payload["negative_controls"] = negative_controls(tf, r1_names, brk)
        with open(a.out, "w") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1, allow_nan=False)
        for rep, v in payload["runs"].items():
            print(rep, v["confusion"], v["buckets"],
                  "ctl", v["control_no_dry"], "vocab", v["vocab_text_check"]["mismatches"])
        print("verdict", json.dumps(payload["verdict"], ensure_ascii=False))
        print("S5_DONE", a.out)


if __name__ == "__main__":
    main()
