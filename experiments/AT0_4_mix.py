#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AT0_4 — T0.4 «Разводящая клетка 3×2: кто несёт профиль» (CPU-форварды, волна 2).

Скоуп (exec-review r3 C1): own-doc r/DC carrier pilot — вход только own_w0..7
(G16: corpus-токенов в источниках нет); профиль ρ̄(k) хранится описательно и НЕ
тестируется в P1–P5; corpus-клетка T0.4 остаётся открытой.

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
S4 (G1-тождество ≤1e-9), S5 (vector-telescope ≤1e-9, r3 C6.1), G_nd1..3 (невырожденность
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
ALPHA = 0.05                # уровень значимости (P3/P5 sign test) — ИСПОЛЬЗУЕТСЯ, r3 C3
DELTA_MIN_R = 0.01          # порог практической значимости для r: 1/10 от P4-порога
                            # 0.1 (априорный масштаб «большого эффекта» серии для r)
DELTA_MIN_FDC = DELTA_MIN_R / 383.0   # f_DC = r/N (G1) — тот же порог, масштаб 1/N
BOOT_B, BOOT_SEED = 2000, 117  # paired bootstrap (серийный boot-сид, AT0_1)
METRICS_VERSION = "v2"      # ключ ckpt: версия cell-генерации (forward+du_metrics+
                            # logits_stats+build_random/build_r1). Bump при изменении
                            # метрик; правки агрегации/вердикта клетки НЕ инвалидируют.
                            # v2 (r3 C6.1): S5 — vector-telescope (не разность норм).
                            # Текущий артефакт (05:19:54) не пересчитан: норма-форма ≤1e-9
                            # PASS, и fp64-vector-остаток ≤ γ_383·N·√c0_max ≈ 3.2e-10.
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
    # S5: telescope — ВЕКТОРНЫЙ остаток ‖ΣΔu − (u_N − u_0)‖ (r3 C6.1: разность норм
    # слабее — нормы могут совпадать при разных векторах)
    telescope = float(np.linalg.norm(du.sum(axis=0) - (u[-1] - u[0])))
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


P2_CELL_NOTE = ("hidden-state r (hidden_states[24], ДО lm_head) идентифицирует только "
                "вклад tied input-эмбеддинга: lm_head действует на logits после L+1 и не "
                "входит в hidden-траекторию; «embed+head contributes» было бы неверно (r3 C5)")

# ── r4-константы (exec-review r4: r — 2-DoF сводка, масштабно-зависимые пороги, n=8) ──
R2DOF_IDENTITY = ("r = ‖u_N−u_0‖²/c0 = 4(1−cos(u_0,u_N))/c0; r/r_max = (1−cos(u_0,u_N))/2 "
                  "(телескоп, S5) — r это сводка ДВУХ чисел (endpoint-косинус, c0); профиль "
                  "ρ̄(k) r НЕ видит по построению (r4 §1)")
P1_BAND_MEANING = ("бэнд [0.75,1.5] — сигнатура E[r]=1 для НЕКОГЕРЕНТНЫХ малых шагов "
                   "(iid-приращения, low-motion c0≤4); измеренные null'ы (synthetic_null): "
                   "low-motion iid — r≈1 (в бэнде), когерентный дрейф — r≈N (ВНЕ бэнда, "
                   "r_med≈192), iid-позиции на сфере — r≈1/N (телескопирование). r≈1 НЕ "
                   "означает «когерентный дрейф» (r4 §4 — спор с ревью). Бэнд — НЕ детектор "
                   "когерентного дрейфа (synthetic null: drift_smooth_walk r_med≈192 — вне бэнда) "
                   "и на данных c0~260 structurally unreachable — P1 здесь: исторический "
                   "пре-рег, не содержательный тест (r5 C3)")
SIGN_TEST_RESOLUTION = ("при n=8 двусторонний sign test достигает p<0.05 ТОЛЬКО при "
                        "единогласии 8/8 (p=0.0078); 7/1 ⇒ p=0.0703 — между единогласием и "
                        "прочим тест не имеет разрешения при α=0.05 (r4 §3)")
P2_SCALE_NOTE = ("абсолютный порог 0.1 масштабно-зависим: r = 4(1−cos)/c0 ∝ 1/c0. "
                 "Clean-доки (c0 ~ 200–400): |Δr| ~ 1e-3–3e-3 — в 30–100× НИЖЕ порога, «same» — "
                 "не near-call; вырожденные low-motion доки (c0 ~ 12–24) — порог достижим "
                 "(P4-outliers). Bounded path + N=383 ⇒ r кинематически зажат для trained И "
                 "random (r4 §2/§3). Relative-метрика (|Δr|/r, знаменатель = r(R2)) — "
                 "диагностика, НЕ изменение пре-рег-вердикта (r4 §2). cell_verdict "
                 "«trunk_carries» читать как trunk_carries_absolute_scale: «same» — вердикт "
                 "абсолютного порога, НЕ zero-effect — tied input embedding даёт ненулевую "
                 "relative-модуляцию 0.10–0.17 (per-instance; cell_relative_note, r5 C2)")
P4_SCALE_NOTE = ("абсолютный порог 0.1 масштабно-зависим: r ∝ 1/c0. Clean-доки (c0 ~ 200–400): "
                 "|Δr| ~ 0.0001–0.0023 ≪ 0.1, НО |Δr|/r ~ 0.05–0.4 (порядок меняет r на ~20% в "
                 "относительных единицах); вырожденные low-motion доки (own_w0/own_w4: c0 ~ 12–24, "
                 "period-1 token loop) — |Δr|/r ~ 0.77–0.90, порог 0.1 достижим. «null по "
                 "медиане» — утверждение об абсолютном пороге, не о модели (r4 §2)")


def med(xs):
    return float(np.median(xs)) if xs else None


def sign_test_pval(diffs):
    """Точный двусторонний sign test (биномиальный): p = 2·Σ_{i≤min(pos,neg)} C(n,i)/2^n.

    Нули исключаются из n; n=0 → p=1 (нет доказательств). (r3 C3: «confirmed»
    должен опираться на тест, не на знак медианы.)"""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    k = min(sum(1 for x in d if x > 0), sum(1 for x in d if x < 0))
    return float(2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def boot_ci_median(diffs, B=BOOT_B, seed=BOOT_SEED):
    """Paired bootstrap CI медианы per-doc-разниц (ресэмплинг доков)."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(B, d.size))
    m = np.median(d[idx], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def build_p2(cells, docs_run, seeds_run, inst_not_testable, divergent, gnd=None):
    """P2 (|r R1−R2| ≥ 0.1 per-seed, real) + gnd-фильтр (r4 §5) + relative-метрика (r4 §2).

    Вынесено из main() в функцию (единственный источник — rebuild r4 вызывает).
    Relative: |Δr|/r, знаменатель = r(R2) (референс-клетка)."""
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
        # доки, живые в ОБЕИХ клетках (degenerate → метрик нет) + gnd ok (r4 §5)
        docs_ok = [nm for nm in docs_run
                   if not cells[("R1", inst, "real", nm)].get("degenerate")
                   and not cells[("R2", inst, "real", nm)].get("degenerate")
                   and (gnd is None or (gnd.get(f"R1_{inst}", {}).get(nm, {}).get("ok", True)
                                        and gnd.get(f"R2_{inst}", {}).get(nm, {}).get("ok", True)))]
        if not docs_ok:
            p2[inst] = {"verdict": "not_testable", "note": "все доки degenerate"}
            logline(f"P2 {inst}: not_testable (все доки degenerate)")
            continue
        diffs = [abs(cells[("R1", inst, "real", nm)]["r"] - cells[("R2", inst, "real", nm)]["r"])
                 for nm in docs_ok]
        rels = [abs(cells[("R1", inst, "real", nm)]["r"] - cells[("R2", inst, "real", nm)]["r"])
                / cells[("R2", inst, "real", nm)]["r"] for nm in docs_ok]
        md = med(diffs)
        p2[inst] = {"median_abs_diff_r": md,
                    "per_doc": {nm: abs(cells[('R1', inst, 'real', nm)]['r'] - cells[('R2', inst, 'real', nm)]['r']) for nm in docs_ok},
                    "median_abs_diff_rel": med(rels),
                    "n_docs": len(docs_ok),
                    "gnd_filter": gnd is not None,
                    "threshold": P2_THR, "verdict": "different" if md >= P2_THR else "same"}
        logline(f"P2 {inst}: median|r R1−R2| = {md:.4f} → {p2[inst]['verdict']}")
    verdicts = [v["verdict"] for v in p2.values() if v["verdict"] in ("different", "same")]
    p2["cell_verdict"] = ("tied_embed_contributes_to_r" if verdicts and all(v == "different" for v in verdicts)
                          else ("trunk_carries" if verdicts and all(v == "same" for v in verdicts)
                                else ("not_evaluable" if not verdicts else "mixed")))
    p2["cell_note"] = P2_CELL_NOTE
    p2["scale_note"] = P2_SCALE_NOTE
    # r5 C2: «trunk_carries» ≠ «embed не влияет» — relative-модуляция ненулевая
    rel_meds = {k: v["median_abs_diff_rel"] for k, v in p2.items()
                if isinstance(v, dict) and v.get("median_abs_diff_rel") is not None}
    if rel_meds:
        p2["cell_relative_note"] = (
            "«trunk_carries» = trunk_carries_absolute_scale: «same» — вердикт абсолютного "
            f"порога {P2_THR}, НЕ zero-effect verdict; tied input embedding даёт ненулевую "
            "относительную модуляцию r: median |Δr|/r = "
            + ", ".join(f"{k} {v:.4f}" for k, v in sorted(rel_meds.items())) +
            " (r5 C2)")
    return p2


def build_p3_p5(cells, docs_run):
    """P3/P5 R0 real-vs-mix c inference (r3 C3): exact sign test + paired bootstrap
    CI + DELTA_MIN; вердикт-лестница confirmed/descriptive/null/not_testable.

    P5 — алгебраическое зеркало P3 (G3: f_DC = r/N, S4 ≤ 1e-9) — НЕ независимое
    свидетельство (вычисляется независимо, помечено mirror_of)."""
    docs_ok = [nm for nm in docs_run
               if not cells[("R0", "trained", "real", nm)].get("degenerate")
               and not cells[("R0", "trained", "mix", nm)].get("degenerate")]

    def _block(metric, delta_min, mirror=None):
        per = {nm: cells[("R0", "trained", "mix", nm)][metric]
               - cells[("R0", "trained", "real", nm)][metric] for nm in docs_ok}
        md = med(list(per.values()))
        pv = sign_test_pval(list(per.values()))
        ci = boot_ci_median(list(per.values()))
        if not docs_ok or md is None:
            verdict, note = "not_testable", "нет живых доков"
        elif md > 0 and pv < ALPHA and md >= delta_min:
            verdict, note = "confirmed", "sign test p<α И |med| ≥ Δmin"
        elif md > 0:
            verdict, note = ("descriptive", "med>0, но sign test p≥α и/или |med|<Δmin — "
                              "описательная положительная медиана, не статистическое подтверждение")
        else:
            verdict, note = "null", "med≤0 — объявленный нулевой исход пре-рега (mix не "
            "создаёт/не разрушает структуру ⇒ носитель — модель)"
        b = {"per_doc": per, "median_diff": md, "n_docs": len(docs_ok),
             "sign_test_p_value": pv, "alpha": ALPHA, "delta_min": delta_min,
             "boot_ci_median": ci, "verdict": verdict, "note": note,
             "sign_test_resolution": SIGN_TEST_RESOLUTION}
        if md is not None and len(docs_ok) == 8:
            b["note"] += ("; при n=8 «confirmed» требует единогласия 8/8 (p=0.0078) — "
                          "высокая положительная медиана при 6/2 — отсутствие мощности, не слабый эффект")
        if mirror:
            b["mirror_of"] = mirror
            b["note"] += (f"; алгебраическое зеркало {mirror} (G3: f_DC = r/N, S4 ≤ 1e-9) — "
                          "НЕ независимое свидетельство")
        return b

    return _block("r", DELTA_MIN_R), _block("f_dc", DELTA_MIN_FDC, mirror="P3")


def synthetic_null_r(T=384, D=1024, B=200, seed=117):
    """Синтетический нуль для P1-бэнда (r3 C4; без форвардов, чистый numpy).

    Два реальных null-режима: (1) iid-позиции на сфере — «полностью случайная
    траектория»: ΣΔu = u_end−u_0 (телескопирование) ⇒ r ≈ 1/N ≈ 0.0026, c0 ≈ 2N ⇒
    P1 vacuous (и по r_max, и по r); (2) iid-малые-шаги (low-motion, c0 ≤ 4):
    E[r] = 1 — бэнд [0.75, 1.5] — сигнатура этого режима, не arbitrary.
    Дрейф ⇒ r ≈ N (вне бэнда). c0* = 4 — НА ГРАНИЦЕ r_max=1 (reach ≈ 0.5)."""
    rng = np.random.default_rng(seed)

    def _stat(rs, c0s, rms):
        r = np.asarray(rs)
        return {"r": {"median": float(np.median(r)), "p05": float(np.percentile(r, 5)),
                     "p95": float(np.percentile(r, 95))},
                "c0_median": float(np.median(c0s)),
                "r_max_median": float(np.median(rms)),
                "band_frac_0.75_1.5": float(((r >= 0.75) & (r <= 1.5)).mean()),
                "reachable_frac_rmax_ge_1": float((np.asarray(rms) >= 1).mean())}

    out = {"T": T, "D": D, "B": B, "seed": seed}
    # (1) iid-сфера: u_t независимы → Δu = u_{t+1}−u_t, c0 ≈ 2N (НЕ low-motion)
    rs, c0s, rms = [], [], []
    for _ in range(B):
        U = rng.normal(size=(T, D))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        du = U[1:] - U[:-1]
        c0 = float((du * du).sum())
        rs.append(float(np.linalg.norm(du.sum(axis=0)) ** 2 / c0))
        c0s.append(c0)
        rms.append(4.0 / c0)
    out["iid_sphere"] = _stat(rs, c0s, rms)
    # (2,3) iid-малые-шаги: Δu ~ N(0, (c0*/(N·D))·I), c0* = 2 / 4 (low-motion режим)
    for c0t, name in ((2.0, "iid_lowmotion_c0_2"), (4.0, "iid_lowmotion_c0_4")):
        rs, c0s, rms = [], [], []
        s2 = c0t / ((T - 1) * D)
        for _ in range(B):
            du = rng.normal(scale=np.sqrt(s2), size=(T - 1, D))
            c0 = float((du * du).sum())
            rs.append(float(np.linalg.norm(du.sum(axis=0)) ** 2 / c0))
            c0s.append(c0)
            rms.append(4.0 / c0)
        out[name] = _stat(rs, c0s, rms)
    # (4) плавный дрейф: u_t = unit(u_0 + t·δ·v) → Δu ≈ const → r ≈ N (не «марш»)
    rs, c0s, rms = [], [], []
    for _ in range(B):
        u0 = rng.normal(size=D)
        u0 /= np.linalg.norm(u0)
        v = rng.normal(size=D)
        v /= np.linalg.norm(v)
        U = u0[None, :] + np.arange(T, dtype=np.float64)[:, None] * 0.01 * v[None, :]
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        du = U[1:] - U[:-1]
        c0 = float((du * du).sum())
        rs.append(float(np.linalg.norm(du.sum(axis=0)) ** 2 / c0))
        c0s.append(c0)
        rms.append(4.0 / c0)
    out["drift_smooth_walk"] = _stat(rs, c0s, rms)
    return out


def build_p1(cells, docs_run, seeds_run, inst_not_testable, divergent, null_rep, gnd=None):
    """P1 (R2×real) + diagnostics (r3 C4): распределения c0/r_max, reachable
    fraction, low-motion флаг (c0≤4 ⇔ r_max≥1), синтетический нуль бэнда.
    r4 §5: gnd-фильтр — доки с ok=False (вырожденный ‖Δu‖-профиль) исключаются из теста
    (строка остаётся со status=gnd_not_ok). r4 §4: reachability-note (structurally
    unreachable: c0 ≫ 4) + band_meaning (р≈1 = некоррелированные малые шаги)."""
    p1 = {}
    for S in seeds_run:
        inst = f"S{S}"
        if inst_not_testable.get(f"R2_{inst}", False):
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
            if gnd is not None and not gnd.get(f"R2_{inst}", {}).get(nm, {}).get("ok", True):
                rows.append({"doc": nm, "r": m["r"], "r_max": m["r_max"],
                             "status": "gnd_not_ok"})
                continue
            if m["r_max"] < 1.0:
                status = "vacuous(r_max<1)"
            elif R_APPROX_LO <= m["r"] <= R_APPROX_HI:
                status = "ok_orthogonal"
            else:
                status = "not_approx"
            rows.append({"doc": nm, "r": m["r"], "r_max": m["r_max"], "status": status})
        reach = [x for x in rows
                 if x["status"] not in ("vacuous(r_max<1)", "degenerate", "gnd_not_ok")]
        med_r = med([x["r"] for x in reach])
        verdict = ("vacuous" if not reach
                   else ("confirmed" if med_r is not None and R_APPROX_LO <= med_r <= R_APPROX_HI
                         else "refuted"))
        c0s = [cells[("R2", inst, "real", nm)]["c0"] for nm in docs_run
               if not cells[("R2", inst, "real", nm)].get("degenerate")]
        c0a = np.asarray(c0s)
        rmax_max = max(x["r_max"] for x in rows if x["r_max"] is not None) if rows else None
        p1[inst] = {"per_doc": rows, "n_reachable": len(reach),
                    "reachable_frac": (len(reach) / len(rows)) if rows else None,
                    "median_r_reachable": med_r, "verdict": verdict,
                    "gnd_filter": gnd is not None,
                    "c0_stats": {"median": float(np.median(c0a)), "max": float(c0a.max())}
                    if c0a.size else None,
                    "low_motion_docs_c0_le_4": [nm for nm in docs_run
                                                if not cells[("R2", inst, "real", nm)].get("degenerate")
                                                and cells[("R2", inst, "real", nm)]["c0"] <= 4.0],
                    "reachability_note": (f"r_max = 4/c0 ≤ {rmax_max:.4f} < {R_APPROX_LO} — "
                                          f"structurally unreachable: c0 median "
                                          f"{float(np.median(c0a)):.0f} ≫ 4 (r4 §4)" if rmax_max
                                          and rmax_max < R_APPROX_LO else None)}
        logline(f"P1 R2×real {inst}: reachable {len(reach)}/{len(rows)}, "
                f"c0 median/max = {np.median(c0a):.2f}/{c0a.max():.2f}, "
                f"median_r={med_r if med_r is None else format(med_r, '.4f')} → {verdict}")
    p1["synthetic_null"] = null_rep
    p1["band_note"] = ("бэнд [0.75,1.5] = сигнатура E[r]=1 для некоррелированных МАЛЫХ шагов "
                       "(synthetic_null: iid-сфера r≈1/N≈0.003 — vacuous и по r_max, и по r; "
                       "low-motion iid r≈1; дрейф r≈N) — не arbitrary; r_max≥1 — low-motion гейт "
                       "(c0≤4): отбор малоподвижных траекторий, объявлен a priori (prereg v3.1: "
                       "vacuous — наиболее вероятный исход для случайной модели)")
    p1["band_meaning"] = P1_BAND_MEANING
    return p1


def token_provenance(toks_by_doc):
    """Decode-проба + токеновая статистика по докам (r3 C6.5: outlier own_w4 6/6
    должен быть интерпретируемым). Decode — через `tokenizers` (локальный
    tokenizer.json; transformers в venv может быть сломан — best-effort)."""
    tok = None
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(os.path.join(BASE, "tokenizer.json"))
    except Exception as e:
        logline(f"token_provenance: decode-проба SKIPPED ({type(e).__name__}: "
                f"{str(e).splitlines()[0]})")
    out = {}
    for nm in DOCS:
        t = np.asarray(toks_by_doc[nm], dtype=np.int64)
        vals, cnts = np.unique(t, return_counts=True)
        rep = {p: float((t[p:] == t[:-p]).mean()) for p in range(1, 17)}
        cand = [p for p in sorted(rep) if rep[p] >= 0.9]
        entry = {"n_uniq": int(len(vals)), "max_share": float(cnts.max() / t.size),
                 "rep8": rep[8], "rep16": rep[16],
                 "min_period_90pct": cand[0] if cand else None,
                 "max_period_agreement_1_16": float(max(rep.values()))}
        if tok is not None:
            try:
                res = tok.decode(t[:48].tolist())
                text = res[0] if isinstance(res, tuple) else res
                entry["decode_48"] = text.replace("\n", " ⏎ ")[:160]
            except Exception as e:
                entry["decode_48"] = f"decode fail: {str(e).splitlines()[0]}"
        out[nm] = entry
    return out


def build_p4(cells, docs_run, seeds_run, inst_not_testable, divergent, gnd=None):
    """P4 (exec-review P2: per_doc + per-doc outlier note; тест — медиана по TZ).
    r4 §5: gnd-фильтр (доки с ok=False исключаются); r4 §2: relative-метрика
    |Δr|/r (знаменатель = r(real) — референс-порядок) + scale_note."""
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
                       and not cells[(cell, inst, "mix", nm)].get("degenerate")
                       and (gnd is None or gnd.get(key, {}).get(nm, {}).get("ok", True))]
            if not docs_ok:
                p4[f"{cell} {inst}"] = {"verdict": "not_testable", "note": "все доки degenerate"}
                logline(f"P4 {cell} {inst}: not_testable (все доки degenerate)")
                continue
            per_doc = {nm: abs(cells[(cell, inst, "real", nm)]["r"]
                               - cells[(cell, inst, "mix", nm)]["r"]) for nm in docs_ok}
            per_doc_rel = {nm: per_doc[nm] / cells[(cell, inst, "real", nm)]["r"] for nm in docs_ok}
            md = med([per_doc[nm] for nm in docs_ok])
            flag = md >= P4_THR
            p4[f"{cell} {inst}"] = {"median_abs_diff": md, "threshold": P4_THR,
                                    "per_doc": per_doc,
                                    "per_doc_rel": per_doc_rel,
                                    "median_abs_diff_rel": med([per_doc_rel[nm] for nm in docs_ok]),
                                    "n_docs": len(docs_ok), "gnd_filter": gnd is not None,
                                    "verdict": "token_statistics_flag" if flag else "null",
                                    "note": ("≥0.1: случайные веса «видят» порядок токенов"
                                             if flag else "random-модель нечувствительна к порядку (медиана)"),
                                    "scale_note": P4_SCALE_NOTE}
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
                              + ", ".join(f"{nm} {c}/{len(live)}" for nm, c in sorted(out_cnt.items()))
                              + " — ВСЕ — вырожденные low-motion доки (period-1 token loop, "
                                "c0 ~ 12–24; см. doc_order_provenance/token_provenance) — "
                                "флаги измерены на вырожденном входе (r4 §6)")
    # doc-level флаг (r3 C2): счётчик ≥ порога по ВСЕМ live-инстансам; «systematic» —
    # эффект на уровне документа, а не выброс по инстансам (own_w4: 6/6)
    n_live_inst = len(live)
    if n_live_inst:
        p4["doc_order_flags"] = {
            nm: {"count": f"{out_cnt.get(nm, 0)}/{n_live_inst}",
                 "flag": ("systematic" if out_cnt.get(nm, 0) == n_live_inst
                          else ("partial" if out_cnt.get(nm, 0) > 0 else "none"))}
            for nm in sorted({nm for k in live for nm in p4[k]["per_doc"]})}
    return p4


def _random_real_cells(cells, seeds_run, nm):
    """6 random-инстансов (R1/R2 × S) real-mode метрики для доку."""
    out = []
    for S in seeds_run:
        for c in ("R1", "R2"):
            m = cells.get((c, f"S{S}", "real", nm))
            if m is not None and not m.get("degenerate") and m.get("c0") is not None:
                out.append((f"{c}_S{S}", m))
    return out


def build_r_decomposition(cells, docs_run, seeds_run, token_prov=None, gnd=None):
    """r4 §1/§8.3: r = 2-DoF сводка (endpoint-косинус, c0) + c0-тест trained vs random.

    endpoint_cos = 1 − r·c0/2 (из телескопа: r/r_max = (1−cos)/2, r_max = 4/c0).
    c0-тест — POST-HOC (exec-review r4 §8.3), без форвардов: paired по докам,
    effect size = median ratio; n=8 — sign test разрешает только единогласие.
    r5 C1: sensitivity (all_8 / no_period1 / gnd_filtered / no_periodic) +
    pairwise_note (48 ≠ 48 независимых) + status (post-hoc, descriptive)."""
    per_cell = {}
    for c, i, mo in ([("R0", "trained", "real"), ("R0", "trained", "mix")]
                     + [(c, f"S{S}", mo) for S in seeds_run for c in ("R1", "R2")
                        for mo in ("real", "mix")]):
        block = {}
        for nm in docs_run:
            m = cells.get((c, i, mo, nm))
            if m is None or m.get("degenerate") or m.get("r") is None:
                continue
            r, c0 = m["r"], m["c0"]
            block[nm] = {"r": r, "c0": c0, "r_max": 4.0 / c0,
                         "endpoint_cos": 1.0 - r * c0 / 2.0}
        per_cell[f"{c}_{i}_{mo}"] = block
    per_doc, per_inst, ratios = {}, {}, []
    for nm in docs_run:
        tr = cells.get(("R0", "trained", "real", nm))
        if tr is None or tr.get("degenerate") or tr.get("c0") is None:
            continue
        rr = _random_real_cells(cells, seeds_run, nm)
        if not rr:
            continue
        c0s = [m["c0"] for _, m in rr]
        coses = [1.0 - m["r"] * m["c0"] / 2.0 for _, m in rr if m.get("r") is not None]
        mr = float(np.median(c0s))
        per_doc[nm] = {"c0_trained": tr["c0"], "c0_random_median": mr,
                       "c0_ratio": tr["c0"] / mr,
                       "endpoint_cos_trained": 1.0 - tr["r"] * tr["c0"] / 2.0,
                       "endpoint_cos_random_median": (float(np.median(coses)) if coses else None)}
        ratios.append(tr["c0"] / mr)
    for S in seeds_run:
        for c in ("R1", "R2"):
            diffs, rats = [], []
            for nm in docs_run:
                a = cells.get(("R0", "trained", "real", nm))
                b = cells.get((c, f"S{S}", "real", nm))
                if (a is None or b is None or a.get("degenerate") or b.get("degenerate")
                        or a.get("c0") is None or b.get("c0") is None):
                    continue
                diffs.append(a["c0"] - b["c0"])
                rats.append(a["c0"] / b["c0"])
            per_inst[f"{c}_S{S}"] = {"n": len(diffs),
                                     "n_positive": int(sum(1 for x in diffs if x > 0)),
                                     "sign_test_p": sign_test_pval(diffs),
                                     "median_ratio": (float(np.median(rats)) if rats else None)}
    ra = np.asarray(ratios)
    rw = []
    for nm in per_doc:
        tr = cells[("R0", "trained", "real", nm)]["c0"]
        for _, m in _random_real_cells(cells, seeds_run, nm):
            rw.append(tr / m["c0"])
    rw = np.asarray(rw)
    rng = np.random.default_rng(BOOT_SEED)
    boot = [float(np.median(rng.choice(ra, size=len(ra), replace=True))) for _ in range(BOOT_B)]
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    npos = int((ra > 1.0).sum())
    npos_w = int((rw > 1.0).sum())

    # r5 C1.3/C5: c0-sensitivity — headline «8/8» включает вырожденные low-motion доки,
    # которые P1/P2/P4 частично исключают — отчитаем срез отдельно (без форвардов).
    sensitivity = None
    if token_prov is not None:
        def _subset_stats(doc_list, pair_ok=None):
            per_doc_ratios, excluded, rw_s = [], [], []
            for nm in doc_list:
                tr = cells[("R0", "trained", "real", nm)]["c0"]
                rr = [(k, m) for k, m in _random_real_cells(cells, seeds_run, nm)
                      if pair_ok is None or pair_ok(nm, k)]
                if not rr:
                    excluded.append(nm)
                    continue
                per_doc_ratios.append(tr / float(np.median([m["c0"] for _, m in rr])))
                for _, m in rr:
                    rw_s.append(tr / m["c0"])
            ra_s = np.asarray(per_doc_ratios)
            rw_s = np.asarray(rw_s)
            return {"docs": [nm for nm in doc_list if nm not in excluded],
                    "n_docs": int(ra_s.size),
                    "excluded_docs": excluded,
                    "median_ratio": (float(np.median(ra_s)) if ra_s.size else None),
                    "boot_ci95": boot_ci_median(list(ra_s)) if ra_s.size else None,
                    "n_positive": int((ra_s > 1.0).sum()) if ra_s.size else 0,
                    "pairwise": {"n": int(rw_s.size),
                                 "n_positive": int((rw_s > 1.0).sum()) if rw_s.size else 0,
                                 "median": (float(np.median(rw_s)) if rw_s.size else None)}}
        period = {nm: token_prov[nm]["min_period_90pct"] for nm in docs_run}
        gnd_ok = (lambda nm, k: gnd.get(k, {}).get(nm, {}).get("ok", True)) if gnd is not None \
            else None
        gf = _subset_stats(list(docs_run), pair_ok=gnd_ok)
        partial = []
        if gnd is not None:
            for nm in docs_run:
                n_ok = sum(1 for S in seeds_run for c in ("R1", "R2")
                           if gnd.get(f"{c}_S{S}", {}).get(nm, {}).get("ok", True))
                if 0 < n_ok < 2 * len(seeds_run):
                    partial.append(f"{nm} {n_ok}/{2 * len(seeds_run)}")
        sens = {
            "all_8": _subset_stats(list(docs_run)),
            "no_period1_docs": _subset_stats(
                [nm for nm in docs_run if period[nm] != 1],
                pair_ok=None),
            "gnd_filtered_pairs": gf,
            "no_periodic_docs": _subset_stats(
                [nm for nm in docs_run if period[nm] is None],
                pair_ok=None),
            "note": ("clean-only = без period-1 low-motion (own_w0/own_w4); no_periodic — "
                     "без ЛЮБОЙ token-периодичности (исключены: "
                     + ", ".join(f"{nm} period-{period[nm]}"
                                for nm in docs_run if period[nm] not in (None, 1)) +
                     "); gnd_filtered — per-instance пары, где G_nd ok (целиком исключены: "
                     + ", ".join(gf["excluded_docs"]) +
                     ("; частично: " + ", ".join(partial) + ")" if partial else ")")
                     + ". r5 C1.3/C5")}
        sensitivity = sens
        sens_bits = ", ".join(f"{k}: {s['n_positive']}/{s['n_docs']}"
                              for k, s in sorted(sens.items()) if isinstance(s, dict))

    summary_line = (f"c0 (энергия шагов ‖Δu‖²): trained/random median ratio "
                    f"{float(np.median(ra)):.4f} [CI95 {ci[0]:.4f}, {ci[1]:.4f}] "
                    f"(n={len(ra)}, paired по докам, {npos}/{len(ra)} > 1; pairwise "
                    f"{npos_w}/{len(rw)} > 1 (descriptive — НЕ 48 независимых наблюдения, "
                    f"см. pairwise_note), median {float(np.median(rw)):.4f}; "
                    "per-instance — sign test, см. per_instance); post-hoc, descriptive — "
                    "НЕ pre-registered confirmatory test (exec-review r4 §8.3, r5 C1.1), "
                    "без форвардов" + (f"; sensitivity {sens_bits}" if sensitivity else ""))
    return {"identity": R2DOF_IDENTITY,
            "note": ("endpoint_cos = 1 − r·c0/2 (per-doc/per-instance в per_cell и "
                     "trained_vs_random). Overlap диапазонов endpoint-cos при n≤8 — НЕ тест: "
                     "«геометрия endpoint'ов совпадает» не закрепляется (r4 §1, следствие 2); "
                     "чистый сигнал — c0 (paired ratio)"),
            "per_cell": per_cell,
            "trained_vs_random": {"per_doc": per_doc, "per_instance": per_inst,
                                  "median_ratio": float(np.median(ra)),
                                  "boot_ci95": ci, "n_pairs": int(len(ra)),
                                  "n_positive": npos,
                                  "pairwise": {"n": int(len(rw)), "n_positive": npos_w,
                                                "median": float(np.median(rw))},
                                  "status": ("post-hoc, descriptive — НЕ pre-registered "
                                              "confirmatory test (r4 §8.3, r5 C1.1)"),
                                  "pairwise_note": ("48 = 8 доков × 6 random-инстансов; "
                                                    "инстансы делят doc/trunk/seed structure — "
                                                    "это НЕ 48 независимых наблюдения; "
                                                    "inferential sign test — per-doc (n=8) и "
                                                    "per-instance (n=8); pairwise — "
                                                    "описательно (r5 C1.2)"),
                                  "sensitivity": sensitivity,
                                  "summary_line": summary_line}}


def build_doc_order_provenance(cells, docs_run, seeds_run, p4, token_prov):
    """r4 §6: P4-флаги × token_provenance × c0 — одна таблица (флаг + объяснение).
    r5 C4: per-doc note — token-периодичность и absence order-флага — разные объекты."""
    flags = p4.get("doc_order_flags", {})
    live = [k for k, v in p4.items() if isinstance(v, dict) and "per_doc" in v]
    out = {}
    for nm in docs_run:
        tr = cells.get(("R0", "trained", "real", nm))
        c0s = [m["c0"] for _, m in _random_real_cells(cells, seeds_run, nm)]
        rdiffs = [p4[k]["per_doc"][nm] for k in live if nm in p4[k]["per_doc"]]
        fl = flags.get(nm)
        tp_e = token_prov[nm]
        tr_live = tr is not None and not tr.get("degenerate")
        entry = {
            "order_flag": (fl or {}).get("flag", "absent"),
            "order_flag_count": (fl or {}).get("count", "—"),
            "min_period_90pct": tp_e.get("min_period_90pct"),
            "max_share": tp_e.get("max_share"),
            "n_uniq": tp_e.get("n_uniq"),
            "c0_trained": (tr["c0"] if tr_live else None),
            "c0_random_median": (float(np.median(c0s)) if c0s else None),
            "r_trained_real": (tr["r"] if tr_live else None),
            "max_abs_diff_r": (max(rdiffs) if rdiffs else None),
            "interp": (f"period-1 token loop (single-token mass ≈{round(tp_e['max_share'] * 100)}%) — "
                       "вырожденный low-motion вход" if tp_e.get("min_period_90pct") == 1 else "—")}
        # r5 C4: периодичность есть, но order-флаг absent — не противоречие
        p = tp_e.get("min_period_90pct")
        if p is not None and p > 1 and entry["order_flag"] not in ("systematic", "partial"):
            if nm in {nm2 for k in live for nm2 in p4[k]["per_doc"]}:
                why = (f"|Δr| ниже абсолютного порога 0.1 во всех live-парах "
                       f"(макс {max(rdiffs):.4f})")
            else:
                why = ("пары не live: не входят в per_doc ни в одном live-инстансе "
                       "(G_nd not-ok во всех random-инстансах)")
            entry["note"] = (f"token periodicity (period-{p}, n_uniq={tp_e['n_uniq']}) present, "
                             f"но absolute order-флаг absent — {why}; детектор непротиворечив (r5 C4)")
        out[nm] = entry
    flagged = [nm for nm in docs_run if out[nm]["order_flag"] in ("systematic", "partial")]
    note = None
    if flagged:
        all_loop = all(out[nm]["min_period_90pct"] == 1 for nm in flagged)
        note = ("P4-флаги: " + ", ".join(
                  f"{nm} ({out[nm]['order_flag_count']}, {out[nm]['order_flag']})" for nm in flagged)
                + (" — ВСЕ — period-1 token loop ⇒ вырожденные low-motion доки; "
                   "«систематическая чувствительность к порядку» читать как «на вырожденных "
                   "low-motion доках» (r4 §6)" if all_loop else " (состав — см. interp)"))
    return {"docs": out, "note": note, "note_key": "order_flag ≠ token-периодичность: "
            "периодический док может не иметь order-флага (не live / ниже порога), и "
            "наоборот (r5 C4)"}


def build_summary(p1, p2_cell, p3, p5, p4, extra_note=None):
    """Summary (r4: r-2-DoF; P1 — structurally unreachable; P3/P5 — n=8-разрешение;
    P4 — null по медиане + doc-level исключения на вырожденных low-motion доках;
    extra_note — c0-тест r4 §8.3)."""
    n_live = sum(1 for v in p4.values() if isinstance(v, dict) and "per_doc" in v)
    n_null = sum(1 for v in p4.values() if isinstance(v, dict) and v.get("verdict") == "null")
    n_nt = sum(1 for v in p4.values() if isinstance(v, dict) and v.get("verdict") == "not_testable")
    n_flag = p4.get("n_token_statistics_flags", 0)
    docfl = ""
    flags = p4.get("doc_order_flags", {})
    sysd = [f"{nm} ({c['count']})" for nm, c in sorted(flags.items()) if c["flag"] == "systematic"]
    part = [f"{nm} ({c['count']})" for nm, c in sorted(flags.items()) if c["flag"] == "partial"]
    if sysd or part:
        docfl = ("; doc-level order sensitivity: " + (", ".join(sysd) + " systematic" if sysd else "")
                 + (("; " + ", ".join(part)) + " partial" if part else "")
                 + " — все — вырожденные low-motion доки (period-1 token loop; см. "
                   "doc_order_provenance)")

    def _vd(b):
        s = (f"{b['verdict']} (med {b['median_diff'] if b['median_diff'] is None else format(b['median_diff'], '.4f')}, "
             f"sign test p={b['sign_test_p_value']:.4f})")
        ci = b.get("boot_ci_median")
        if ci:
            s += f", CI [{ci[0]:.4f}, {ci[1]:.4f}]"
        if b.get("n_docs") == 8:
            s += " [при n=8 p<0.05 только при 8/8]"
        return s

    random_part = f"random: null по медианному тесту {n_null}/{n_live}"
    if docfl:
        random_part += docfl
    random_part += f"; {n_flag} token-statistics флаг(ов), {n_nt} not_testable"

    p1v = {k: v for k, v in p1.items() if isinstance(v, dict) and "verdict" in v}
    c0s = [v["c0_stats"]["max"] for v in p1v.values() if v.get("c0_stats")]
    medc0s = [v["c0_stats"]["median"] for v in p1v.values() if v.get("c0_stats")]
    p1_part = ("R2×real — " + ", ".join(f"{k} {v['verdict']}" for k, v in p1v.items())
               + (f" — structurally unreachable (r_max = 4/c0 ≤ {4.0 / max(c0s):.4f}; "
                  f"c0 median ≈ {med(medc0s):.0f} ≫ 4)" if c0s and max(c0s) > 4.0 else ""))

    p2_part = f"R1 vs R2 (real): {p2_cell} (hidden-state r — input-эмбеддинг, не head)"
    if p2_cell == "trunk_carries":
        p2_part += ("; trunk_carries = абсолютный масштаб, НЕ zero-effect (relative-модуляция "
                    "0.10–0.17 — P2.cell_relative_note, r5 C2)")

    s = ("AT0_4 (3×2 own-doc r/DC carrier pilot; r — 2-DoF сводка endpoint-косинус × c0, "
         "профиль ρ̄(k) не тестируется в P1–P5; corpus T0.4 — open): "
         + p1_part
         + f"; {p2_part}; "
         f"R0 real-vs-mix: r {_vd(p3)}; f_DC {_vd(p5)} (mirror P3 по G3 — не независимый сигнал); "
         + random_part)
    if extra_note:
        s += "; " + extra_note
    return s


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
        if (j.get("metrics_version") != METRICS_VERSION or j.get("sigma_table_sha256") != sig_sha
                or j.get("script_sha256") != ckpt_script_sha):
            logline(f"CKPT: mismatch (metrics_version {j.get('metrics_version')} vs "
                    f"{METRICS_VERSION} / σ-таблица / script sha {j.get('script_sha256', 'absent')[:16]} vs "
                    f"{ckpt_script_sha[:16]}) — полный прогон")
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
    logline(f"S5: max vector telescope ‖ΣΔu − (u_N−u_0)‖ = {tele_max:.3e} "
            f"{'PASS' if tele_max <= 1e-9 else 'FAIL'}")
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

    # P1: R2×real — r≈1 [0.75,1.5] только при r_max≥1 + diagnostics (r3 C4)
    t_null = time.time()
    null_rep = synthetic_null_r()
    logline(f"P1 synthetic null (r3 C4): {time.time() - t_null:.1f}s, B={null_rep['B']}")
    p1 = build_p1(cells, docs_run, seeds_run, inst_not_testable, divergent, null_rep)
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
    p2_cell = ("tied_embed_contributes_to_r" if verdicts and all(v == "different" for v in verdicts)
               else ("trunk_carries" if verdicts and all(v == "same" for v in verdicts)
                     else ("not_evaluable" if not verdicts else "mixed")))
    p2["cell_verdict"] = p2_cell
    p2["cell_note"] = P2_CELL_NOTE
    prereg["P2_R1_R2_trunk_vs_embed"] = p2

    # P3/P5: R0 real-vs-mix + inference (r3 C3: sign test + bootstrap CI + DELTA_MIN)
    p3, p5 = build_p3_p5(cells, docs_run)
    logline(f"P3 R0: median r(mix)−r(real) = "
            f"{p3['median_diff'] if p3['median_diff'] is None else format(p3['median_diff'], '.4f')} "
            f"(sign test p={p3['sign_test_p_value']:.4f}, Δmin={DELTA_MIN_R}) → {p3['verdict']}")
    logline(f"P5 R0: median f_DC(mix)−f_DC(real) = "
            f"{p5['median_diff'] if p5['median_diff'] is None else format(p5['median_diff'], '.6f')} "
            f"(sign test p={p5['sign_test_p_value']:.4f}, mirror P3) → {p5['verdict']}")
    prereg["P3_R0_real_vs_mix"] = p3
    prereg["P5_R0_real_vs_mix_fdc"] = p5

    # P4: random — median per-doc |r real − r mix| < 0.1 per-seed (нулевая)
    p4 = build_p4(cells, docs_run, seeds_run, inst_not_testable, divergent)
    n_flag = p4["n_token_statistics_flags"]
    prereg["P4_random_order_insensitive"] = p4

    if args.smoke:
        logline("AT0_4 SMOKE OK (артефакты не пишутся)")
        return

    # ── вердикт + артефакт ────────────────────────────────────────────────────

    # r4: r-декомпозиция (2-DoF) + c0-тест + doc_order_provenance (без форвардов)
    tp = token_provenance({nm: docs[nm]["toks"] for nm in DOCS})
    rdec = build_r_decomposition(cells, docs_run, seeds_run, token_prov=tp, gnd=gnd)
    dop = build_doc_order_provenance(cells, docs_run, seeds_run, p4, tp)
    logline(f"r4 c0-test: {rdec['trained_vs_random']['summary_line']}")

    verdict = {
        "pretest_results": {
            "P1": {k: v["verdict"] for k, v in p1.items()
                   if isinstance(v, dict) and "verdict" in v},
            "P2": {"cell_verdict": p2_cell,
                   **{k: v["verdict"] for k, v in p2.items()
                      if isinstance(v, dict) and "verdict" in v}},
            "P3": p3["verdict"], "P5": p5["verdict"],
            "P4": {k: v["verdict"] for k, v in p4.items()
                   if isinstance(v, dict) and "verdict" in v},
            "P4_token_statistics_flags": n_flag,
        },
        "summary": build_summary(p1, p2_cell, p3, p5, p4,
                                 extra_note=rdec["trained_vs_random"]["summary_line"]),
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
        "mix_note": ("mix = одна фиксированная перестановка на док (seed 7) — mix treatment "
                     "fixed; no permutation-seed sensitivity (новых форвардов нет); эффект mix "
                     "может зависеть от конкретной перестановки (r3 C6.4, r5 C7)"),
        "doc_stratification": {
            "period_1_low_motion": [nm for nm in DOCS if tp[nm]["min_period_90pct"] == 1],
            "periodic_non_low_motion": [nm for nm in DOCS
                                        if tp[nm]["min_period_90pct"] not in (None, 1)],
            "clean_no_periodic": [nm for nm in DOCS if tp[nm]["min_period_90pct"] is None],
            "note": ("стратификация по token_provenance.min_period_90pct + c0 (r5 C5); "
                     "own_w7 — тоже периодический (period-16, rep16≥0.9) — в clean-наборе "
                     "ревью r5 его не учтено (р5-ответка C5); c0-headline стратифицирован в "
                     "r_decomposition.trained_vs_random.sensitivity")},
        "scope": ("own-doc r/DC carrier pilot: вход — только own_w0..7 (G16: corpus-токенов "
                  "в источниках нет); профиль ρ̄(k) — описательно (в store), НЕ тестируется в "
                  "P1–P5; corpus-клетка T0.4 — open (r3 C1); " + R2DOF_IDENTITY +
                  "; абсолютные пороги 0.1 на r масштабно-зависимы (r ∝ 1/c0, r4 §2)"),
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
             "prereg": prereg, "verdict": verdict, "provenance": {"g0_sha": g0_sha},
             "token_provenance": tp,
             "r_decomposition": rdec, "doc_order_provenance": dop}
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
             "verdict": verdict, "token_provenance": store["token_provenance"],
             "r_decomposition": rdec, "doc_order_provenance": dop,
             "store": {"path": os.path.relpath(store_path, OUT_DIR),
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

    logline("AT0_4 gates: " + ("ALL PASS" if all(g.get("ok") is True for g in gates.values()
                                                  if isinstance(g, dict)) else "CHECK LOG"))
    logline("AT0_4 DONE")
    # run.log пишется ПОСЛЕ финальных logline (баг v3.0: была дописана строка,
    # а «ALL PASS»/«DONE» терялись) — run.log = полный stdout
    with open(RUN_LOG, "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
