#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E94 Q3 (ТЗ §2.3, версия «без GPU»): margin_t = p1−p2 по top_logprobs[0..1]
против rep8-флагов (дословный rep_flags exp81b, окно n=8, конкатенация
prompt(64)+gen, off=64). ТОЛЬКО один движок (llama f16/SYCL, собственные id);
кросс-движковое смешение позиций запрещено, CPU-аналог N/A (REPLAY.json без
вторых кандидатов). Данные: /tmp/e94raw/q1_w<i>_r<1,2>.json (из
results/e94/raw_q1.json.tar.gz), r1 — основной, r2 — контроль воспроизводимости.
Метрики: (a) spearman(margin, flag8) по t>=64; (b) то же с флагом на лаге
L ∈ {1,4,8,16} (margin_t vs flag_{t+L}); (c) медиана margin при flag=1 vs
flag=0 + бутстрап-CI разности (позиции — пер-доковый, документы — общий).
Null: perm_null n=2000 seed=8181 (two-sided |rho|, MC-доля без +1, p_mc_se).
Артефакт results/e94/Q3.json пишется при ЛЮБОМ исходе (halt_reason).
RUN_RC: 0 ok · 1 данные/исключение · 11 <8 доков с непустыми метриками."""
import json, os, sys, time, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp94_sycl_common import san, rep_flags, spear, perm_null, prefix_len, bootstrap_ci_median
import numpy as np

RAW = "/tmp/e94raw"
OUT = "results/e94/Q3.json"
LAGS = (1, 4, 8, 16)
N_BOOT, SEED, N_PERM = 10000, 8181, 2000
OFF = 64


def load_docs():
    """16 доков × r1/r2 из raw; (docs, errors). docs: {key: {r: dict(request,
    response)}}. Ошибки — по (key, r), не исключение: артефакт пишется и при
    частичной потере (rc решается ниже)."""
    docs, errs = {}, {}
    for i in range(16):
        key = "w%d" % i
        docs[key] = {}
        for r in (1, 2):
            path = os.path.join(RAW, "q1_%s_r%d.json" % (key, r))
            try:
                d = json.load(open(path))
                req, resp = d["request"], d["response"]
                ids64, toks, cp = req["prompt"], resp["tokens"], resp["completion_probabilities"]
                assert len(ids64) == 64, "prompt len %d != 64" % len(ids64)
                assert isinstance(toks, list) and toks and all(isinstance(t, int) for t in toks)
                assert isinstance(cp, list) and len(cp) == len(toks), "cp len != tokens len"
                assert [p.get("id") for p in cp] == toks, "tokens != [p.id]"
                docs[key][r] = {"ids64": list(ids64), "tokens": list(toks), "cp": cp}
            except Exception as e:
                errs["%s r%d" % (key, r)] = "%s: %s" % (path, e)
    return docs, errs


def doc_metrics(key, rec):
    """один док (rec = {ids64, tokens, cp}): конкатенация prompt(64)+gen, off=64;
    margin_t = exp(lp_top1) − exp(lp_top2), позиции с len(top_logprobs)<2
    исключаются (отдельный счётчик). (a) spear(margin, flag8); (b) то же с
    flag_{t+L}, L ∈ LAGS; (c) медианы по flag=1/0 + бутстрап-CI разности,
    ресемплинг ПОЗИЦИЙ (n=N_BOOT, seed)."""
    ids = rec["ids64"] + rec["tokens"]
    flag = rep_flags(ids, 8)  # дословно exp81b (через common), окно 8
    n_gen = len(rec["tokens"])
    assert len(rec["cp"]) == n_gen, "cp len %d != tokens len %d" % (len(rec["cp"]), n_gen)
    pos, m, fl, n_short = [], [], [], 0
    # cp — только gen-токены (координата j: 0..n_gen-1); конкат-индекс t = OFF + j
    for j, p in enumerate(rec["cp"]):
        tl = p.get("top_logprobs", [])
        if len(tl) < 2:
            n_short += 1
            continue
        pos.append(OFF + j)
        m.append(float(np.exp(tl[0]["logprob"]) - np.exp(tl[1]["logprob"])))
        fl.append(bool(flag[OFF + j]))
    m, fl = np.array(m), np.array(fl)
    e = {"doc": key, "n_gen": int(n_gen), "n_used": int(len(m)),
         "n_excluded_short_toplogprobs": int(n_short),
         "margin": [round(x, 8) for x in m], "flag": [int(x) for x in fl],
         "pos": [int(p) for p in pos], "ids_len": int(len(ids)),
         "flag_full": [int(x) for x in flag]}  # пер-позиционные массивы для аудита
    # (a)
    e["spear_a"] = spear(m, fl)
    # (b): margin_t vs flag_{t+L}; позиции, где t+L >= len(ids), отбрасываются
    b, dropped = {}, {}
    for L in LAGS:
        pair = [(mv, flag[p + L]) for p, mv in zip(pos, m) if p + L < len(ids)]
        b[L], dropped[L] = spear(np.array([x[0] for x in pair]), np.array([x[1] for x in pair])), \
            int(len(m) - len(pair))
    e["spear_b"], e["n_dropped_lag"] = b, dropped
    # (c)
    f1, f0 = m[fl == 1], m[fl == 0]
    e["n_flag1"], e["n_flag0"] = int(len(f1)), int(len(f0))
    e["med_flag1"] = float(np.median(f1)) if len(f1) else None
    e["med_flag0"] = float(np.median(f0)) if len(f0) else None
    if len(f1) and len(f0):
        diff = float(np.median(f1) - np.median(f0))
        rng = np.random.default_rng(SEED)
        ds = []
        for _ in range(N_BOOT):
            ix = rng.choice(len(m), size=len(m), replace=True)
            a, bb = m[ix][fl[ix] == 1], m[ix][fl[ix] == 0]
            if len(a) and len(bb):
                ds.append(float(np.median(a) - np.median(bb)))
        e["delta_median"], e["delta_ci_2.5_97.5"] = diff, \
            [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
        e["delta_n_boot_used"] = len(ds)
    else:
        e["delta_median"], e["delta_ci_2.5_97.5"] = None, None
    return e


def _perm_block(a, b):
    obs, p, nullci = perm_null(a, b, n=N_PERM, seed=SEED)
    if obs is None:
        return {"rho": None, "p_perm": None, "null_ci": None, "n_positions": int(len(a))}
    return {"rho": float(obs), "p_perm": float(p),
            "null_ci": [float(nullci[0]), float(nullci[1])],
            "p_mc_se": float(np.sqrt(p * (1.0 - p) / N_PERM)),
            "n_positions": int(len(a)), "n_perm": N_PERM, "seed": SEED,
            "p_definition": ("two-sided on |rho|: MC fraction of %d permutations with "
                             "|rho_perm| >= |rho_obs| (no +1 correction), margin permuted "
                             "against fixed rep8 flag, avg-ranks on ties, n_perm=%d, seed=%d")
            % (N_PERM, N_PERM, SEED)}


def overall(per):
    """(a)/(b): пул позиций всех доков (один движок — смешение разрешено),
    ОДНА процедура (perm_null common). (c): медиана пер-доковых дельт;
    бутстрап по ДОКУМЕНТАМ (единица ресемплинга — док, явно)."""
    ma, fa = [], []
    for e in per:
        ma += e["margin"]; fa += e["flag"]
    ma, fa = np.array(ma, float), np.array(fa, bool)
    O = {"a": _perm_block(ma, fa), "b": {}}
    # (b): пары (margin_t, flag_{t+L}) по выжившим позициям в пул
    for L in LAGS:
        xs, ys = [], []
        for e in per:
            for p, mv in zip(e["pos"], e["margin"]):
                if p + L < e["ids_len"]:
                    xs.append(mv); ys.append(e["flag_full"][p + L])
        O["b"][str(L)] = _perm_block(np.array(xs, float), np.array(ys, bool))
    # (c)
    diffs = [e["delta_median"] for e in per if e.get("delta_median") is not None]
    c = {"per_doc_diffs": {e["doc"]: e["delta_median"] for e in per},
         "resample_unit": "document (общее число), position (пер-доковый) — явно",
         "n_bootstrap": N_BOOT, "seed": SEED}
    if diffs:
        d = np.array(diffs)
        rng = np.random.default_rng(SEED)
        meds = [float(np.median(rng.choice(d, size=len(d), replace=True))) for _ in range(N_BOOT)]
        c["median_diff"] = float(np.median(d))
        c["median_diff_ci_2.5_97.5"] = [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]
    else:
        c["median_diff"], c["median_diff_ci_2.5_97.5"] = None, None
    O["c"] = c
    # (d) per-doc rho — первичная статистика (MAJOR-1): единица ресемплинга = ДОК
    rhos = [e["spear_a"] for e in per if e.get("spear_a") is not None]
    d = {"resample_unit": "document (единица ресемплинга — док, НЕ позиция)",
         "n_bootstrap": N_BOOT, "seed": SEED}
    if rhos:
        med, ci = bootstrap_ci_median(rhos, n=N_BOOT, seed=SEED)
        d["median"], d["ci_2.5_97.5"] = med, ci
        d["n_docs"], d["n_positive"] = len(rhos), int(sum(1 for r in rhos if r > 0))
    else:
        d["median"], d["ci_2.5_97.5"], d["n_docs"], d["n_positive"] = None, None, 0, 0
    O["d_per_doc_rho"] = d
    O["p_note"] = ("p_perm на пуле позиций оптимистичен (автокорреляция внутри дока); "
                   "первичная статистика — d_per_doc_rho")
    return O


def r2_control(per2, per1):
    """r2 — контроль воспроизводимости: тот же (a) per-doc + общий пул (тот же
    perm_null), плюс prefix_len r1/r2 (детерминизм жадного потока)."""
    by1 = {e["doc"]: e for e in per1}
    rows = []
    for e in per2:
        r1 = by1.get(e["doc"])
        rows.append({"doc": e["doc"], "spear_a_r2": e["spear_a"],
                     "n_used_r2": e["n_used"],
                     "n_excluded_short_r2": e["n_excluded_short_toplogprobs"],
                     "spear_a_r1": r1["spear_a"] if r1 else None,
                     "prefix_len_r1_r2": e.get("prefix_len_r1_r2")})
    ma, fa = [], []
    for e in per2:
        ma += e["margin"]; fa += e["flag"]
    return {"per_doc": rows,
            "overall_a": _perm_block(np.array(ma, float), np.array(fa, bool)) if ma else None,
            "mean_abs_delta_spear_a": (float(np.mean(np.abs(
                [r["spear_a_r2"] - r["spear_a_r1"] for r in rows
                 if r["spear_a_r2"] is not None and r["spear_a_r1"] is not None])))
                if any(r["spear_a_r2"] is not None and r["spear_a_r1"] is not None for r in rows) else None),
            "median_prefix_len_r1_r2": (float(np.median(
                [r["prefix_len_r1_r2"] for r in rows if r["prefix_len_r1_r2"] is not None]))
                if any(r["prefix_len_r1_r2"] is not None for r in rows) else None)}


def finish(R, rc, halt, t0):
    R["rc"], R["halt_reason"] = rc, halt
    R["t_total_s"] = round(time.time() - t0, 2)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(san(R), f, ensure_ascii=False, indent=1)
    print("E94 q3 rc=%d%s" % (rc, (" halt=" + halt) if halt else ""), flush=True)
    print("RUN_RC=%d" % rc, flush=True)
    return rc


def main():
    t0 = time.time()
    R = {"phase": "q3", "started_unix": int(t0),
         "engine": "llama-server f16/SYCL (одна сторона; позиции — собственные id движка)",
         "data_source": "results/e94/raw_q1.json.tar.gz → /tmp/e94raw/q1_w<i>_r<1,2>.json",
         "convention": ("margin_t=exp(lp1)-exp(lp2) по top_logprobs[0..1]; rep8 — дословный "
                        "rep_flags(exp81b), окно 8, конкатенация prompt(64)+gen, off=64; "
                        "лаги: margin_t vs flag_{t+L} (низкий отрыв → ПОСЛЕДУЮЩИЕ повторы)"),
         "interpretability": "correlational, single engine, no mechanism claim",
         "limitations": [
             "CPU-аналог недоступен: в results/e81/REPLAY.json нет вторых кандидатов "
             "(docs.<key> = segments/onset/p1_med, gen_p1 — скаляр p1) → сравнение "
             "«corr против corr» между движками сегодня не делается и объявляется N/A",
             "rep8 насыщен (91% p-значений > 0.9) — полоса насыщена, null неинформативен без эффекта",
             "16 доков — малая выборка, CI широкие"],
         "per_doc": [], "summary": {}}
    try:
        docs, errs = load_docs()
        R["load_errors"] = errs
        per1, per2 = [], []
        for i in range(16):
            key = "w%d" % i
            if 1 in docs.get(key, {}):
                per1.append(doc_metrics(key, docs[key][1]))
            if 2 in docs.get(key, {}):
                e2 = doc_metrics(key, docs[key][2])
                if 1 in docs[key]:
                    e2["prefix_len_r1_r2"] = int(prefix_len(docs[key][1]["tokens"], docs[key][2]["tokens"]))
                per2.append(e2)
        R["per_doc"] = per1
        R["r2_control"] = r2_control(per2, per1)
        R["n_docs_with_metrics"] = len(per1)
        ok_docs = [e for e in per1 if e["n_used"] > 0]
        if len(ok_docs) < 8:
            return finish(R, 11, "только %d доков с непустыми метриками (<8)" % len(ok_docs), t0)
        R["summary"] = overall(ok_docs)
        return finish(R, 0, None, t0)
    except Exception:
        traceback.print_exc()
        return finish(R, 1, "данные/исключение (см. traceback выше)", t0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        R = {"phase": "q3", "rc": 1, "halt_reason": "exception (см. traceback)"}
        try:
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            json.dump(san(R), open(OUT, "w"), ensure_ascii=False, indent=1)
        except Exception:
            pass
        print("RUN_RC=1", flush=True)
        sys.exit(1)
