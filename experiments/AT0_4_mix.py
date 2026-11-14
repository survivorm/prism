#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AT0_4 — T0.4 «Разводящая клетка 3×2: кто несёт профиль» (CPU-форварды, волна 2).

Прегр: docs/AT0_4_prereg.md (v3; TZ-ревью — луп ДО прогона).
Серия AT (ТЗ v1.1): AT0_1–AT0_3 disk-only закрыты; AT0_4 — новые CPU-форварды
(вход НЕ вычисляется из TPU-npz — P10).

Факторизация 3×2: {R0 trained, R1 random trunk + trained tied embed+head,
R2 random всё} × {real (TF на исходных токенах), mix (full-permutation тех же
токенов; одна общая перестановка на док для всех клеток)}.

Различающие переменные (не тождества): r, f_DC, r_max, d_eff, профиль ρ̄(k)
(описательно). G1 (r=N·f_DC) — sanity; Σ — НЕ тестовая статистика (G7).

Клетки: R0×{real,mix} (1 инстанс — trained детерминирован) + R1×{real,mix}×S +
R2×{real,mix}×S, S ∈ {11,12,13} = 14 инстансов × 8 доков = 112 forward.

Слой: L=23 (0-based) → hidden_states[L+1]=hidden_states[24] (hook-семантика серии).
Доки: own_w0..own_w7 из P1_own.npz (единственный источник с токенами; G16:
P2_corpus 0/16 с токенами). T=384, N=383, K=96.

Гейты: S1 (R0 liveness: per-doc head-accuracy ≥0.90 на собственных генерациях —
TPU-cosine-пин заменён: TPU-bf16 vs CPU-fp32 расхождение в linear-attention state
не воспроизводимо без TPU-аудита, а AT0_4 — относительное сравнение R0 vs R1/R2
на одном CPU-пайплайне), S2 (tied), S3 (finiteness 1док×32ток R1_11/R2_11),
S4 (G1-тождество ≤1e-9), S5 (telescope ≤1e-9), G_nd1..3 (невырожденность
random-клетки, иначе not_testable).

Прегр (per-doc, медиана по докам, per-seed отчёт, α=0.05): P1 (R2×real r≈1
[0.75,1.5] только при r_max≥1, иначе vacuous), P2 (|r R1−R2|≥0.1 per-seed),
P3 (r(mix)−r(real)>0 для R0), P4 (random |r real−r mix|<0.1, иначе token-
statistics флаг), P5 (f_DC(mix)−f_DC(real)>0 для R0).
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

# CPU-политика серии: 4 тред (run_one.sh экспортирует 6 — переопределяем ДО
# импорта torch; бенч 51.7 с/forward снят на 4 треде).
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "4"
# transformers.use_kernel_func_from_hub_with_fallback импортирует fla, а fla-l2norm
# поднимает triton-драйвер и крашится "0 active drivers" (эксп91-ноут) — чистый
# torch-fallback (референсный путь CPU-прогонов серии).
sys.modules["fla"] = False
sys.modules["fla.ops"] = False

import numpy as np
import torch
import transformers
torch.set_num_threads(4)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_fspec = importlib.util.spec_from_file_location(
    "prism_field_pinned", os.path.join(ROOT, "prism/field.py"))
pf = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(pf)
sys.modules["prism_field_pinned"] = pf
acf_moments, jclean, sha256_file = pf.acf_moments, pf.jclean, pf.sha256_file

from transformers import AutoModelForCausalLM, AutoConfig

# ── константы ──────────────────────────────────────────────────────────────────
BASE = "/data/Qwen/Qwen3.5-0.8B"
OWN_NPZ = os.path.join(ROOT, "results/e64/tpu_v8/P1_own.npz")
OUT_DIR = os.path.join(ROOT, "results/AT0")
STORE_DIR = os.path.join(OUT_DIR, "store")
OUT_JSON = os.path.join(OUT_DIR, "AT0_4_MIX.json")
MAN_JSON = os.path.join(OUT_DIR, "AT0_4_manifest.json")
RUN_LOG = os.path.join(OUT_DIR, "AT0_4_run.log")
CKPT = os.path.join(OUT_DIR, "AT0_4_checkpoint.json")

L = 23                      # 0-based; hidden_states[L+1] = hidden_states[24]
K = 96                      # канон серии
DOCS = ["own_w%d" % i for i in range(8)]   # own_w0..own_w7 (первые 8 лекс. порядка)
SEEDS = (11, 12, 13)
SHUFFLE_SEED = 7            # mix: ОБЩАЯ перестановка на док для всех клеток (M2 exp91)
TIED_KEY = "model.embed_tokens.weight"     # tied-пара: lm_head — view того же storage
HEAD_ACC_MIN = 0.90         # S1: R0 liveness (head-acc на собственных генерациях)
CV_MIN = 0.05               # G_nd1
DEFF_MIN = 3.0              # G_nd2 (v3: было 8 — убивало random×real: d_eff≈5 —
                            #   облако u у доков с nuniq 16..66 ограничено словарём,
                            #   это не коллапс; ≥3 ловит только точку/линию (1/2))
ENT_MIN = 0.5               # G_nd3 (H(softmax)/log(V))
R_APPROX_LO, R_APPROX_HI = 0.75, 1.5   # P1: r≈1
P2_THR, P4_THR = 0.1, 0.1   # P2/P4 пороги
METRICS_VERSION = "v1"      # ключ ckpt: версия cell-генерации (forward+du_metrics+
                            # logits_stats+build_random/build_r1). Bump при изменении
                            # метрик; правки агрегации/вердикта клетки НЕ инвалидируют.
LOG = []


def logline(s=""):
    print(s)
    LOG.append(s)


def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, np.floating):
        o = float(o)
    elif isinstance(o, np.integer):
        o = int(o)
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


# ── вход ──────────────────────────────────────────────────────────────────────
def load_docs():
    z = np.load(OWN_NPZ, allow_pickle=True)
    docs = {}
    for nm in DOCS:
        toks = np.asarray(z[f"{nm}_tok"]).ravel().astype(np.int64)
        if toks.shape[0] != 384:
            logline(f"FATAL: {nm} T={toks.shape[0]} ≠ 384 — стоп (RC=5)")
            raise SystemExit(5)
        # mix: full-permutation тех же токенов; ОДНА общая для всех клеток (seed 7)
        sh = np.random.default_rng(SHUFFLE_SEED).permutation(toks)
        docs[nm] = {"toks": toks.tolist(), "mix": sh.tolist()}
    return docs


# ── forward + метрики ─────────────────────────────────────────────────────────
def forward(model, toks):
    """TF-forward: h на L (hidden_states[24]) + logits (384, V)."""
    with torch.no_grad():
        o = model(input_ids=torch.tensor([toks]), output_hidden_states=True)
    h = o.hidden_states[L + 1][0].double().numpy()          # (384, 1024) fp64
    logits = o.logits[0].float()                            # (384, V)
    return h, logits


def du_metrics(h):
    """u_t = h_t/‖h_t‖; Δu_t = u_t − u_{t−1}; r, f_DC, r_max, d_eff, профиль."""
    hn = np.linalg.norm(h, axis=1, keepdims=True)
    u = h / np.maximum(hn, 1e-12)
    du = u[1:] - u[:-1]                                     # (N=383, 1024)
    N = du.shape[0]
    c0 = float(np.einsum("ij,ij->i", du, du).sum())
    if c0 <= 0:
        raise ValueError("c(0)=0: вырожденный Δu")
    r = float(np.linalg.norm(du.sum(axis=0)) ** 2 / c0)
    f_dc = float(np.linalg.norm(du.mean(axis=0)) ** 2 / (c0 / N))
    r_max = 4.0 / c0
    # S4: G1-тождество r = N·f_DC (fp64)
    g1_dev = abs(r - N * f_dc) / max(N * f_dc, 1e-300)
    # S5: telescope ‖ΣΔu‖ = ‖u_N − u_0‖ (точно в fp64)
    telescope = abs(np.linalg.norm(du.sum(axis=0)) - np.linalg.norm(u[-1] - u[0]))
    # d_eff: PCA облака u (по данным, не по проектору)
    cov = np.cov(u, rowvar=False)
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 0.0, None)
    lam = ev / ev.sum()
    lam = lam[lam > 0]
    d_eff = float(math.exp(-(lam * np.log(lam)).sum()))
    # профиль: moments ACF (канон field), T−k-нормировка (ρ̄ = mean_t cos для unit)
    ac = acf_moments(du, K)
    # G_nd1: cv(‖Δu‖)
    dn = np.linalg.norm(du, axis=1)
    cv = float(dn.std() / dn.mean())
    return {
        "r": r, "f_dc": f_dc, "r_max": r_max, "d_eff": d_eff,
        "c0": c0, "N": int(N), "g1_dev": float(g1_dev), "telescope": float(telescope),
        "profile": [float(x) for x in ac["rho_Tk"]],
        "profile_k": list(range(1, K + 1)),
        "gram_resid": float(ac["gram_resid"]),
        "cv_du": cv, "sum_du2": c0,
        "hn_med": float(np.median(hn.ravel())),
        "hn_iqr": [float(np.percentile(hn.ravel(), 25)),
                   float(np.percentile(hn.ravel(), 75))],
    }


def logits_stats(logits, toks, V):
    """S1 liveness (head-accuracy) + G_nd3 (энтропия выходного распределения)."""
    with torch.no_grad():
        acc = float((logits.argmax(dim=1)[:len(toks) - 1]
                     == torch.tensor(toks[1:])).float().mean().item())
        p = torch.softmax(logits, dim=-1)
        ent = float((-(p * torch.log(p + 1e-30)).sum(dim=-1) / math.log(V)).mean().item())
    return acc, ent


# ── рандомизация (std-matched, incl. 1D gain — фикс бага ndim≥2) ─────────────
def sigma_table(model):
    """σ_t = std(trained-тензор) по unique-тензорам (tied — один)."""
    tab = {}
    for n, p in model.named_parameters(remove_duplicate=True):
        tab[n] = {"shape": list(p.shape), "std": float(p.float().std().item()),
                  "mean": float(p.float().mean().item())}
    return tab


def build_random(cfg, S, sigma):
    """R2_S: from_config(S) + ВСЕ unique-тензоры → N(0, σ_t) (фиксированный
    generator — воспроизводимо). Tied-семантика сохраняется (один unique-тензор)."""
    torch.manual_seed(S)
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    m.requires_grad_(False)
    m.eval()
    g = torch.Generator().manual_seed(S)
    with torch.no_grad():
        for n, p in m.named_parameters(remove_duplicate=True):
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32)
                    * sigma[n]["std"])
    return m


def build_r1(cfg, S, sigma, trained, m2):
    """R1_S = ствол m2 (R2_S) побайто + trained tied-пара. m2 передаётся
    готовым (не дублируется — пик памяти: trained+m2+m1, не 4 модели)."""
    torch.manual_seed(S)
    m1 = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    m1.requires_grad_(False)
    m1.eval()
    with torch.no_grad():
        for n, p in m1.named_parameters(remove_duplicate=True):
            if n == TIED_KEY:
                p.copy_(trained.state_dict()[TIED_KEY].float())
            else:
                p2 = m2.state_dict().get(n)
                if p2 is None:
                    logline(f"FATAL: R1_S{S}: ключ {n} отсутствует в R2-стволе — стоп (RC=4)")
                    raise SystemExit(4)
                p.copy_(p2)
    # ствол побайто = R2 (проверка по всем ключам кроме tied)
    bad = [n for n, p in m1.named_parameters(remove_duplicate=True)
           if n != TIED_KEY and not torch.equal(p, m2.state_dict()[n])]
    if bad:
        logline(f"FATAL: R1_S{S} ствол ≠ R2_S{S} по {bad} — стоп (RC=4)")
        raise SystemExit(4)
    return m1


def tied_check(model):
    return model.state_dict()["lm_head.weight"].data_ptr() == \
        model.state_dict()[TIED_KEY].data_ptr()


def med(xs):
    return float(np.median(xs)) if xs else None


def build_p4(cells, docs_run, seeds_run, inst_not_testable, divergent):
    """P4 (exec-review P2: per_doc + per-doc outlier note; тест — медиана по TZ)."""
    p4 = {}
    for S in seeds_run:
        inst = f"S{S}"
        for cell in ("R1", "R2"):
            key = f"{cell}_{inst}"
            if key in inst_not_testable and inst_not_testable[key]:
                p4[f"{cell} {inst}"] = {"verdict": "not_testable"}
                logline(f"P4 {cell} {inst}: not_testable (гейт невырожденности)")
                continue
            if divergent.get(key):
                p4[f"{cell} {inst}"] = {"verdict": "not_testable", "note": "divergent (G_nd4)"}
                logline(f"P4 {cell} {inst}: not_testable (divergent, G_nd4)")
                continue
            docs_ok = [nm for nm in docs_run
                       if not cells[(cell, inst, "real", nm)].get("degenerate")
                       and not cells[(cell, inst, "mix", nm)].get("degenerate")]
            if not docs_ok:
                p4[f"{cell} {inst}"] = {"verdict": "not_testable", "note": "все доки degenerate"}
                logline(f"P4 {cell} {inst}: not_testable (все доки degenerate)")
                continue
            per_doc = {nm: abs(cells[(cell, inst, "real", nm)]["r"]
                               - cells[(cell, inst, "mix", nm)]["r"]) for nm in docs_ok}
            md = med([per_doc[nm] for nm in docs_ok])
            flag = md >= P4_THR
            p4[f"{cell} {inst}"] = {"median_abs_diff": md, "threshold": P4_THR,
                                    "per_doc": per_doc,
                                    "verdict": "token_statistics_flag" if flag else "null",
                                    "note": ("≥0.1: случайные веса «видят» порядок токенов"
                                             if flag else "random-модель нечувствительна к порядку (медиана)")}
            logline(f"P4 {cell} {inst}: median|r real−mix| = {md:.4f} → {p4[f'{cell} {inst}']['verdict']}")
    p4["n_token_statistics_flags"] = sum(1 for v in p4.values()
                                         if isinstance(v, dict) and v.get("verdict") == "token_statistics_flag")
    # per-doc исключения по инстансам (exec-review P2: «null по медиане» ≠ «нет исключений»)
    live = [k for k, v in p4.items() if isinstance(v, dict) and "per_doc" in v]
    out_cnt = {}
    for k in live:
        for nm, dv in p4[k]["per_doc"].items():
            if dv >= P4_THR:
                out_cnt[nm] = out_cnt.get(nm, 0) + 1
    if out_cnt:
        p4["per_doc_outliers"] = {nm: f"{c}/{len(live)}" for nm, c in sorted(out_cnt.items())}
        p4["outlier_note"] = ("null по медиане, но per-doc исключения ≥" + f"{P4_THR}: "
                              + ", ".join(f"{nm} {c}/{len(live)}" for nm, c in sorted(out_cnt.items())))
    return p4


def build_summary(p1, p2_cell, p3, p5, p4):
    """Summary (exec-review P2/P3: P5 — mirror P3 по G3; P4 — per-doc исключения)."""
    n_live = sum(1 for v in p4.values() if isinstance(v, dict) and "per_doc" in v)
    n_null = sum(1 for v in p4.values() if isinstance(v, dict) and v.get("verdict") == "null")
    n_nt = sum(1 for v in p4.values() if isinstance(v, dict) and v.get("verdict") == "not_testable")
    n_flag = p4.get("n_token_statistics_flags", 0)
    out = ""
    if "per_doc_outliers" in p4:
        out = "; per-doc исключения: " + ", ".join(
            f"{nm} ({c})" for nm, c in sorted(p4["per_doc_outliers"].items()))
    return ("AT0_4 (3×2): R2×real — "
            + ", ".join(f"{k} {v['verdict']}" for k, v in p1.items())
            + f"; R1 vs R2 (real): {p2_cell}; R0 real-vs-mix: r {p3['verdict']}"
            f" (med {p3['median_diff'] if p3['median_diff'] is None else format(p3['median_diff'], '.3f')}), "
            f"f_DC {p5['verdict']} (med {p5['median_diff'] if p5['median_diff'] is None else format(p5['median_diff'], '.3f')}, "
            f"mirror P3 по G3 — не независимый сигнал); "
            f"random order-invariant: {n_null}/{n_live} null (медианный тест{out}), "
            f"{n_flag} token-statistics флаг(ов), {n_nt} not_testable")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(STORE_DIR, exist_ok=True)  # артефакты: не опираться на существующие каталоги

    logline(f"AT0_4 T0.4 «Разводящая клетка 3×2» (CPU-форварды)"
            + (" [SMOKE]" if args.smoke else ""))
    logline(f"py{sys.version.split()[0]} torch{torch.__version__} "
            f"np{np.__version__} transformers{transformers.__version__} "
            f"threads={torch.get_num_threads()}/{os.environ.get('OMP_NUM_THREADS')}")

    docs = load_docs()
    logline(f"docs={len(docs)} (P1_own, T=384, L={L}, K={K}, N=383)")

    # G0: провенанс
    g0_sha = {"P1_own": sha256_file(OWN_NPZ),
              "config": sha256_file(os.path.join(BASE, "config.json")),
              "field": sha256_file(os.path.join(ROOT, "prism/field.py"))}
    logline(f"G0: P1_own sha {g0_sha['P1_own'][:16]} config {g0_sha['config'][:16]} "
            f"field {g0_sha['field'][:16]}")

    cfg = AutoConfig.from_pretrained(BASE)
    V = cfg.text_config.vocab_size  # Qwen3_5TextConfig — атрибуты, не dict

    t0 = time.time()
    trained = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    trained.requires_grad_(False)
    trained.eval()
    logline(f"trained load: {time.time() - t0:.1f}s")
    sigma = sigma_table(trained)
    sig_sha = hashlib.sha256(json.dumps(sigma, sort_keys=True).encode()).hexdigest()
    n1d = sum(1 for v in sigma.values() if len(v["shape"]) == 1)
    logline(f"σ_t-таблица: {len(sigma)} unique-тензоров (1D: {n1d} — вкл. RMSNorm gain, "
            f"баг ndim≥2 исправлен), sha {sig_sha[:16]}")

    # S2 (trained tied)
    s2_trained = tied_check(trained)
    logline(f"S2 (trained tied): {'PASS' if s2_trained else 'FAIL — стоп (RC=4)'}")
    if not s2_trained:
        raise SystemExit(4)

    # cells: (cell, inst, mode, doc) — mode ∈ {real, mix}; ключ не коллидирует
    cells = {}

    # ── checkpoint (анти-OOM: краш теряет ≤1 forward; resume по script+sigma sha)
    ckpt_script_sha = sha256_file(os.path.abspath(__file__))

    def _ck_key(c, i, mo, nm):
        return f"{c}|{i}|{mo}|{nm}"

    def save_ckpt():
        if args.smoke:
            return
        j = {"metrics_version": METRICS_VERSION,
             "script_sha256": ckpt_script_sha, "sigma_table_sha256": sig_sha,
             "cells": {_ck_key(c, i, mo, nm): m for (c, i, mo, nm), m in cells.items()}}
        tmp = CKPT + ".tmp"
        with open(tmp, "w") as f:
            json.dump(jclean(j), f, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, CKPT)

    def load_ckpt():
        if args.smoke or not os.path.exists(CKPT):
            return False
        try:
            j = json.load(open(CKPT))
        except Exception as e:
            logline(f"CKPT: нечитаемый ({e}) — полный прогон")
            return False
        if j.get("metrics_version") != METRICS_VERSION or j.get("sigma_table_sha256") != sig_sha:
            logline(f"CKPT: mismatch (metrics_version {j.get('metrics_version')} vs "
                    f"{METRICS_VERSION} / σ-таблица) — полный прогон")
            return False
        for k, m in j.get("cells", {}).items():
            c, i, mo, nm = k.split("|")
            cells[(c, i, mo, nm)] = m
        if cells:
            logline(f"CKPT: resume — {len(cells)} клеток уже посчитаны (скип)")
        return True

    def run_cell(cell, inst, model, use_mix, doc_subset):
        for nm in doc_subset:
            mode = "mix" if use_mix else "real"
            if (cell, inst, mode, nm) in cells:
                m = cells[(cell, inst, mode, nm)]
                if m.get("degenerate"):
                    logline(f"  {cell}/{inst}/{mode}/{nm}: DEGENERATE (c0=0) — метрики не определены [resume]")
                else:
                    logline(f"  {cell}/{inst}/{mode}/{nm}: "
                            f"r={m['r']:.4f} f_dc={m['f_dc']:.4f} r_max={m['r_max']:.4f} "
                            f"d_eff={m['d_eff']:.2f} acc={m['head_acc']:.3f} [resume]")
                continue
            toks = docs[nm]["mix" if use_mix else "toks"]
            h, logits = forward(model, toks)
            try:
                m = du_metrics(h)
            except ValueError:
                # точечный коллапс (Δu≡0, c0=0) — объявленный исход G_nd, не краш
                # (р1-ревью: неловленный ValueError убивал прогон ДО метки not_testable)
                logline(f"  {cell}/{inst}/{mode}/{nm}: DEGENERATE (c0=0) — метрики не определены")
                cells[(cell, inst, mode, nm)] = {"degenerate": True}
                save_ckpt()
                continue
            m["head_acc"], m["ent_ratio"] = logits_stats(logits, toks, V)
            cells[(cell, inst, mode, nm)] = m
            save_ckpt()
            logline(f"  {cell}/{inst}/{mode}/{nm}: "
                    f"r={m['r']:.4f} f_dc={m['f_dc']:.4f} r_max={m['r_max']:.4f} "
                    f"d_eff={m['d_eff']:.2f} acc={m['head_acc']:.3f}")

    n_docs = 1 if args.smoke else len(DOCS)
    docs_run = DOCS[:n_docs]
    seeds_run = (11,) if args.smoke else SEEDS

    # ── S1: R0×real (эти форварды = клетка R0×real) + R0 liveness (head-acc) ──
    load_ckpt()
    logline(f"S1: R0×real ({n_docs} доков) — R0 liveness: head-acc на собственных генерациях")
    run_cell("R0", "trained", trained, use_mix=False, doc_subset=docs_run)
    s1 = {}
    for nm in docs_run:
        m = cells[("R0", "trained", "real", nm)]
        if m.get("degenerate"):
            s1[nm] = None  # trained-модель «мертва» на доке (c0=0)
        else:
            s1[nm] = m["head_acc"]
    s1_ok = all(v is not None and v >= HEAD_ACC_MIN for v in s1.values())
    logline("S1: " + ", ".join(
        f"{k}={'degenerate' if v is None else format(v, '.4f')}" for k, v in sorted(s1.items()))
            + (f" | PASS (≥{HEAD_ACC_MIN})" if s1_ok
               else f" | FAIL — STOP (RC=2): trained-модель не 'жива' на собственных генерациях"))
    if not s1_ok:
        raise SystemExit(2)

    # ── S3: finiteness (ВСЕ сида; 1 док × 32 ток) ДО полного прогона ─────────
    # (р1-ревью: только seed 11 — расхождение S12/S13 убивало ~1ч посреди прогона)
    s3_res = {}
    for S in seeds_run:
        m2 = build_random(cfg, S, sigma)
        m1 = build_r1(cfg, S, sigma, trained, m2)  # ствол = m2 побайто + trained tied
        for tag, mm in ((f"R2_S{S}", m2), (f"R1_S{S}", m1)):
            with torch.no_grad():
                oo = mm(input_ids=torch.tensor([docs[DOCS[0]]["toks"][:32]]),
                        output_hidden_states=True)
            hs = oo.hidden_states[L + 1][0, :32]
            fin = bool(torch.isfinite(hs).all())
            s3_res[tag] = {"finite": fin, "min": float(hs.min()), "max": float(hs.max())}
            logline(f"S3 {tag}: finite={'PASS' if fin else 'FAIL'} "
                    f"min={float(hs.min()):.4g} max={float(hs.max()):.4g}")
            if not fin:
                raise SystemExit(3)
        del m1, m2
        gc.collect()

    # ── полный прогон ─────────────────────────────────────────────────────────
    s2_results = {}
    logline(f"R0×mix ({n_docs} доков)")
    run_cell("R0", "trained", trained, use_mix=True, doc_subset=docs_run)

    for S in seeds_run:
        inst = f"S{S}"
        logline(f"R2×{{real,mix}} seed={S}")
        m2 = build_random(cfg, S, sigma)
        s2_ok = tied_check(m2)
        s2_results[f"R2_{inst}"] = s2_ok
        if not s2_ok:
            logline(f"FATAL: R2_{inst} tied сломан — стоп (RC=4)")
            raise SystemExit(4)
        run_cell("R2", inst, m2, use_mix=False, doc_subset=docs_run)
        run_cell("R2", inst, m2, use_mix=True, doc_subset=docs_run)
        logline(f"R1×{{real,mix}} seed={S} (ствол = R2_{inst} + trained tied-пара)")
        m1 = build_r1(cfg, S, sigma, trained, m2)  # ствол = m2 побайто + trained tied
        s2_ok = tied_check(m1)
        s2_results[f"R1_{inst}"] = s2_ok
        if not s2_ok:
            logline(f"FATAL: R1_{inst} tied сломан — стоп (RC=4)")
            raise SystemExit(4)
        # дизайн-асерт: ствол R1_S побайто = ствол R2_S (тот же init-поток);
        # build_r1 проверял против своего m2 — здесь явно против модели R2-клетки
        bad = [n for n, p in m1.named_parameters(remove_duplicate=True)
               if n != TIED_KEY and not torch.equal(p, m2.state_dict()[n])]
        if bad:
            logline(f"FATAL: R1_{inst} ствол ≠ R2_{inst} ствол по {bad} — стоп (RC=4)")
            raise SystemExit(4)
        run_cell("R1", inst, m1, use_mix=False, doc_subset=docs_run)
        run_cell("R1", inst, m1, use_mix=True, doc_subset=docs_run)
        del m1, m2
        gc.collect()

    # ── S4/S5 (по не-вырожденным клеткам; degenerate — метрик нет) ─────────
    live = [m for m in cells.values() if not m.get("degenerate")]
    if not live:
        logline("FATAL: все клетки вырождены (c0=0) — стоп (RC=6)")
        raise SystemExit(6)
    g1_max = max(m["g1_dev"] for m in live)
    tele_max = max(m["telescope"] for m in live)
    logline(f"S4: max |r−N·f_DC|/(N·f_DC) = {g1_max:.3e} "
            f"{'PASS' if g1_max <= 1e-9 else 'FAIL — стоп (RC=6)'}")
    logline(f"S5: max telescope = {tele_max:.3e} {'PASS' if tele_max <= 1e-9 else 'FAIL'}")
    if not (g1_max <= 1e-9 and tele_max <= 1e-9):
        raise SystemExit(6)

    # ── G_nd: невырожденность random-клеток (per inst×doc на real-режиме;
    #     not_testable только если ВСЕ доки вырождены — «any-провал» убивал
    #     инстанс на одном вырожденном доке) ──────────────────────────────────
    gnd = {}
    for (cell, inst, mode, nm), m in sorted(cells.items()):
        if cell not in ("R1", "R2") or mode != "real":
            continue
        if m.get("degenerate"):
            # точечный коллапс — вырожденный док по определению (ok=False)
            gnd.setdefault(f"{cell}_{inst}", {})[nm] = {
                "cv_du": None, "d_eff": None, "ent_ratio": None,
                "hn_med": None, "ok": False, "degenerate": True}
            continue
        ok = bool(m["cv_du"] > CV_MIN and m["d_eff"] >= DEFF_MIN and m["ent_ratio"] > ENT_MIN)
        gnd.setdefault(f"{cell}_{inst}", {})[nm] = {
            "cv_du": m["cv_du"], "d_eff": m["d_eff"], "ent_ratio": m["ent_ratio"],
            "hn_med": m["hn_med"], "ok": ok}
    inst_not_testable = {}
    for k, v in sorted(gnd.items()):
        n_ok = sum(1 for x in v.values() if x["ok"])
        inst_not_testable[k] = (n_ok == 0)
        logline(f"G_nd {k}: {n_ok}/{len(v)} не-вырожденных доков (real) "
                f"(cv>{CV_MIN} ∧ d_eff≥{DEFF_MIN} ∧ ent>{ENT_MIN})"
                + (" → NOT_TESTABLE (все доки вырождены)" if inst_not_testable[k] else ""))
    # G_nd4: нормы random vs trained — per-instance median/IQR ‖h‖ + жёсткий
    # порог «divergent» (р1-ревью: только отчёт — finite-but-huge state дал бы
    # «валидные» метрики дивергировавшей модели; S3 ловит только inf)
    HNR_DIV_MULT = 100.0
    r0_hn = [m["hn_med"] for (c, i, mo, nm), m in cells.items()
             if c == "R0" and mo == "real" and not m.get("degenerate")]
    r0_hn_med = float(np.median(r0_hn)) if r0_hn else None
    hnorm_rep = {}
    divergent = {}
    for k, v in sorted(gnd.items()):
        hn = [x["hn_med"] for x in v.values() if x["hn_med"] is not None]
        med_hn = float(np.median(hn)) if hn else None
        iqr_hn = float(np.subtract(*np.percentile(hn, [75, 25]))) if hn else None
        div = bool(med_hn is not None and r0_hn_med is not None
                   and med_hn > HNR_DIV_MULT * r0_hn_med)
        divergent[k] = div
        hnorm_rep[k] = {"median": med_hn, "iqr": iqr_hn, "divergent": div,
                        "ratio_vs_R0": (med_hn / r0_hn_med) if (med_hn is not None
                                                                and r0_hn_med) else None}
        r0_disp = "n/a" if r0_hn_med is None else format(r0_hn_med, ".2f")
        ratio = hnorm_rep[k]["ratio_vs_R0"]
        ratio_disp = "n/a" if ratio is None else format(ratio, ".2g")
        logline(f"G_nd4 {k}: median ‖h‖ = {med_hn if med_hn is None else format(med_hn, '.2f')} "
                f"(R0 real: {r0_disp}, ratio {ratio_disp})"
                + (" → DIVERGENT (>{:.0f}×R0)".format(HNR_DIV_MULT) if div else ""))

    # ── пре-рег (per-doc → медиана по докам; per-seed) ───────────────────────
    prereg = {}

    # P1: R2×real — r≈1 [0.75,1.5] только при r_max≥1 (r_max замерен ДО оценки)
    p1 = {}
    for S in seeds_run:
        inst = f"S{S}"
        if f"R2_{inst}" in inst_not_testable and inst_not_testable[f"R2_{inst}"]:
            p1[inst] = {"verdict": "not_testable",
                        "note": "инстанс не прошёл гейт невырожденности"}
            logline(f"P1 R2×real {inst}: NOT_TESTABLE (гейт невырожденности)")
            continue
        if divergent.get(f"R2_{inst}"):
            p1[inst] = {"verdict": "not_testable",
                        "note": "инстанс дивергировал (G_nd4: ‖h‖ > 100×R0)"}
            logline(f"P1 R2×real {inst}: NOT_TESTABLE (divergent, G_nd4)")
            continue
        rows = []
        for nm in docs_run:
            m = cells[("R2", inst, "real", nm)]
            if m.get("degenerate"):
                rows.append({"doc": nm, "r": None, "r_max": None, "status": "degenerate"})
                continue
            if m["r_max"] < 1.0:
                status = "vacuous(r_max<1)"
            elif R_APPROX_LO <= m["r"] <= R_APPROX_HI:
                status = "ok_orthogonal"
            else:
                status = "not_approx"
            rows.append({"doc": nm, "r": m["r"], "r_max": m["r_max"], "status": status})
        reach = [x for x in rows
                 if x["status"] not in ("vacuous(r_max<1)", "degenerate")]
        med_r = med([x["r"] for x in reach])
        verdict = ("vacuous" if not reach
                   else ("confirmed" if med_r is not None and R_APPROX_LO <= med_r <= R_APPROX_HI
                         else "refuted"))
        p1[inst] = {"per_doc": rows, "n_reachable": len(reach),
                    "median_r_reachable": med_r, "verdict": verdict}
        logline(f"P1 R2×real {inst}: reachable {len(reach)}/{len(rows)}, "
                f"median_r={med_r if med_r is None else format(med_r, '.4f')} → {verdict}")
    prereg["P1_R2_real_orthogonal_walk"] = p1

    # P2: |median r(R1)−r(R2)| ≥ 0.1 per-seed (real) — trained-пара вносит вклад
    p2 = {}
    for S in seeds_run:
        inst = f"S{S}"
        nt1 = inst_not_testable.get(f"R1_{inst}", False)
        nt2 = inst_not_testable.get(f"R2_{inst}", False)
        dv1 = divergent.get(f"R1_{inst}", False)
        dv2 = divergent.get(f"R2_{inst}", False)
        if nt1 or nt2 or dv1 or dv2:
            p2[inst] = {"verdict": "not_testable",
                        "note": "инстанс(ы) не прошли гейт" + (" / дивергенция" if (dv1 or dv2) else "")}
            logline(f"P2 {inst}: not_testable (гейт/divergent)")
            continue
        # только доки, живые в ОБЕИХ клетках (degenerate → метрик нет)
        docs_ok = [nm for nm in docs_run
                   if not cells[("R1", inst, "real", nm)].get("degenerate")
                   and not cells[("R2", inst, "real", nm)].get("degenerate")]
        if not docs_ok:
            p2[inst] = {"verdict": "not_testable", "note": "все доки degenerate"}
            logline(f"P2 {inst}: not_testable (все доки degenerate)")
            continue
        diffs = [abs(cells[("R1", inst, "real", nm)]["r"] - cells[("R2", inst, "real", nm)]["r"])
                 for nm in docs_ok]
        md = med(diffs)
        p2[inst] = {"median_abs_diff_r": md, "per_doc": {nm: abs(cells[('R1', inst, 'real', nm)]['r'] - cells[('R2', inst, 'real', nm)]['r']) for nm in docs_ok},
                    "threshold": P2_THR, "verdict": "different" if md >= P2_THR else "same"}
        logline(f"P2 {inst}: median|r R1−R2| = {md:.4f} → {p2[inst]['verdict']}")
    verdicts = [v["verdict"] for v in p2.values() if v["verdict"] in ("different", "same")]
    p2_cell = ("embed_head_contributes" if verdicts and all(v == "different" for v in verdicts)
               else ("trunk_carries" if verdicts and all(v == "same" for v in verdicts)
                     else ("not_evaluable" if not verdicts else "mixed")))
    p2["cell_verdict"] = p2_cell
    prereg["P2_R1_R2_trunk_vs_embed"] = p2

    # P3: R0 — median per-doc (r(mix) − r(real)) > 0
    p3_docs = [nm for nm in docs_run
               if not cells[("R0", "trained", "real", nm)].get("degenerate")
               and not cells[("R0", "trained", "mix", nm)].get("degenerate")]
    p3_per = [cells[("R0", "trained", "mix", nm)]["r"] - cells[("R0", "trained", "real", nm)]["r"]
              for nm in p3_docs]
    p3_md = med(p3_per)
    p3 = {"per_doc": {nm: cells[("R0", "trained", "mix", nm)]["r"]
                      - cells[("R0", "trained", "real", nm)]["r"] for nm in p3_docs},
          "median_diff": p3_md, "threshold": 0.0,
          "verdict": ("confirmed" if p3_md is not None and p3_md > 0
                      else ("not_testable" if not p3_docs else "null"))}
    logline(f"P3 R0: median r(mix)−r(real) = {p3_md if p3_md is None else format(p3_md, '.4f')} → {p3['verdict']}")
    prereg["P3_R0_real_vs_mix"] = p3

    # P4: random — median per-doc |r real − r mix| < 0.1 per-seed (нулевая)
    p4 = build_p4(cells, docs_run, seeds_run, inst_not_testable, divergent)
    n_flag = p4["n_token_statistics_flags"]
    prereg["P4_random_order_insensitive"] = p4

    # P5: R0 — median per-doc (f_DC(mix) − f_DC(real)) > 0
    p5_docs = [nm for nm in docs_run
               if not cells[("R0", "trained", "real", nm)].get("degenerate")
               and not cells[("R0", "trained", "mix", nm)].get("degenerate")]
    p5_per = [cells[("R0", "trained", "mix", nm)]["f_dc"] - cells[("R0", "trained", "real", nm)]["f_dc"]
              for nm in p5_docs]
    p5_md = med(p5_per)
    p5 = {"per_doc": {nm: cells[("R0", "trained", "mix", nm)]["f_dc"]
                      - cells[("R0", "trained", "real", nm)]["f_dc"] for nm in p5_docs},
          "median_diff": p5_md, "threshold": 0.0,
          "verdict": ("confirmed" if p5_md is not None and p5_md > 0
                      else ("not_testable" if not p5_docs else "null"))}
    logline(f"P5 R0: median f_DC(mix)−f_DC(real) = {p5_md if p5_md is None else format(p5_md, '.4f')} → {p5['verdict']}")
    prereg["P5_R0_real_vs_mix_fdc"] = p5

    if args.smoke:
        logline("AT0_4 SMOKE OK (артефакты не пишутся)")
        return

    # ── вердикт + артефакт ────────────────────────────────────────────────────
    verdict = {
        "pretest_results": {
            "P1": {k: v["verdict"] for k, v in p1.items()},
            "P2": {"cell_verdict": p2_cell, **{k: v["verdict"] for k, v in p2.items()
                                              if k != "cell_verdict"}},
            "P3": p3["verdict"], "P5": p5["verdict"],
            "P4": {k: v["verdict"] for k, v in p4.items()
                   if isinstance(v, dict) and "verdict" in v},
            "P4_token_statistics_flags": n_flag,
        },
        "summary": build_summary(p1, p2_cell, p3, p5, p4),
    }

    meta = {
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "field_sha256": g0_sha["field"],
        "model": BASE,
        "config_sha256": g0_sha["config"],
        "own_npz_sha256": g0_sha["P1_own"],
        "sigma_table_sha256": sig_sha,
        "sigma_table": sigma,
        "sigma_note": "std-matched N(0,σ_t) по unique-тензорам; 1D gain-ы ВКЛЮЧЕНЫ "
                      "(фикс бага ndim≥2 серии exp66_lesion:40/exp7177c:37)",
        "layer": L,
        "hidden_states_index": L + 1,
        "K": K,
        "docs": DOCS,
        "seeds": list(SEEDS),
        "shuffle_seed": SHUFFLE_SEED,
        "tied_key": TIED_KEY,
        "tied_note": "lm_head.weight — view model.embed_tokens.weight (data_ptr совпадает, "
                     "S2); R1 = trained tied-пара поверх random-ствола; R2 = random-пара",
        "env": {"python": sys.version.split()[0], "torch": torch.__version__,
                "numpy": np.__version__, "transformers": transformers.__version__,
                "OMP": os.environ.get("OMP_NUM_THREADS")},
    }
    gates = {
        "S1_R0_liveness": {"per_doc": s1, "threshold": HEAD_ACC_MIN, "ok": bool(s1_ok),
                           "degenerate_docs": [k for k, v in s1.items() if v is None],
                           "note": "head-acc на собственных генерациях (TPU-cosine-пин "
                                  "заменён: TPU-bf16 vs CPU-fp32 linear-attention state не "
                                  "воспроизводимо без TPU-аудита; AT0_4 — относительное сравнение)"},
        "S2_tied": {"trained": bool(s2_trained),
                    **s2_results,
                    "ok": bool(s2_trained) and all(s2_results.values())},
        "S3_finite": {"ok": True, "seeds": list(seeds_run), "per_instance": s3_res,
                      "note": "1 док×32 ток, ВСЕ сида (р1-ревью: был только seed 11)"},
        "S4_G1_identity": {"max_rel_dev": float(g1_max), "threshold": 1e-9,
                           "ok": bool(g1_max <= 1e-9)},
        "S5_telescope": {"max_abs_dev": float(tele_max), "threshold": 1e-9,
                         "ok": bool(tele_max <= 1e-9)},
        "G_nd": {"thresholds": {"cv": CV_MIN, "d_eff": DEFF_MIN, "ent": ENT_MIN},
                 "per_instance": gnd, "not_testable_instances": inst_not_testable,
                 "hnorm_report_G_nd4": hnorm_rep,
                 "divergent_instances": divergent,
                 "hnr_div_mult": HNR_DIV_MULT,
                 "hnorm_R0_real_median": r0_hn_med,
                 "ok": not any(divergent.values())},
    }
    per_doc = {}
    for (cell, inst, mode, nm), m in cells.items():
        if m.get("degenerate"):
            per_doc.setdefault(f"{cell}_{inst}_{mode}", {})[nm] = {"degenerate": True}
            continue
        per_doc.setdefault(f"{cell}_{inst}_{mode}", {})[nm] = {
            k: m[k] for k in ("r", "f_dc", "r_max", "d_eff", "c0", "g1_dev", "telescope",
                              "cv_du", "hn_med", "hn_iqr", "head_acc", "ent_ratio",
                              "gram_resid")}
    # profile — отдельно (96 чисел × 112 — большой, но в store)
    profiles = {f"{c}_{i}_{mo}": {nm: cells[(c, i, mo, nm)]["profile"]
                                   for nm in docs_run
                                   if not cells[(c, i, mo, nm)].get("degenerate")}
                for (c, i, mo) in sorted({(c, i, mo) for (c, i, mo, nm) in cells})}

    store = {"meta": meta, "prereg_text": "docs/AT0_4_prereg.md v3",
             "gates": gates, "per_doc": per_doc, "profiles": profiles,
             "prereg": prereg, "verdict": verdict, "provenance": {"g0_sha": g0_sha}}
    store_j = jclean(store)
    blob = json.dumps(store_j, ensure_ascii=False, sort_keys=True)
    store_sha = hashlib.sha256(blob.encode()).hexdigest()
    store_path = os.path.join(STORE_DIR, store_sha[:16] + ".json")
    with open(store_path, "w") as f:
        f.write(blob)
    logline(f"store: {store_path} (sha {store_sha[:16]})")
    if os.path.exists(CKPT):
        os.remove(CKPT)
        logline("CKPT: удалён (прогон завершён успешно)")

    claim = {"meta": meta, "gates": gates, "per_doc": per_doc, "prereg": prereg,
             "verdict": verdict, "store": {"path": os.path.relpath(store_path, OUT_DIR),
                                            "sha256": store_sha}}
    claim_j = jclean(claim)
    claim_blob = json.dumps(claim_j, ensure_ascii=False, sort_keys=True)
    with open(OUT_JSON, "w") as f:
        f.write(claim_blob)
    logline(f"written: {OUT_JSON}")

    man = {"manifest": {"claim_sha256": hashlib.sha256(claim_blob.encode()).hexdigest(),
                        "store_sha256": store_sha, "date": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    with open(MAN_JSON, "w") as f:
        json.dump(sanitize(man), f, indent=1)
    logline(f"written: {MAN_JSON}")

    logline("AT0_4 gates: " + ("ALL PASS" if all(g.get("ok", True) for g in gates.values()
                                                  if isinstance(g, dict)) else "CHECK LOG"))
    logline("AT0_4 DONE")
    # run.log пишется ПОСЛЕ финальных logline (баг v3.0: была дописана строка,
    # а «ALL PASS»/«DONE» терялись) — run.log = полный stdout
    with open(RUN_LOG, "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
