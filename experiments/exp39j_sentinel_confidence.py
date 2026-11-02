"""E-39j — Sentinel-позиции как cheap uncertainty metric (docs/E39.md §10).

Гипотеза ТЗ: per-doc доля sentinel-позиций (масса глобального U128 в softmax < 0.5,
порог E-34) коррелирует с downstream confidence (PPL документа).

Сторона A (sentinel) берётся из уже посчитанного пайплайна E-32r.3
(`results/exp32r3_layer_profile.json`, per_doc[L|dom/si].sentinel_fraction =
`RAW.sum(-1) < 0.5`, все S=128 позиций) — тот же код-путь, что дал якорь E-34
50.7% (2077/4096). Сторона B (downstream) — per-doc NLL/PPL того же 128-префикса,
fresh forward fp32/eager (как в производителе кэша exp27b).

Пререгистрация (paper §3.2), до прогона:
  * Гипотеза: rho_S(f_sent, PPL) > 0.5 (ТЗ §10.6, «cheap confidence»).
  * Объект и канал: mass_t = sum(softmax(logits_t)[U128]) — ЕСТЕСТВЕННЫЙ срез,
    raw (без gauge-операторов), top-масса глобального множества, ЛЕВАЯ сторона
    (распределение токенов); downstream = mean NLL тех же позиций.
  * Null/базлайн: тривиальный предиктор той же природы — mean mass / mean max-prob /
    mean entropy. Гипотеза проверяется только через **избыточность**: partial Spearman
    f_sent vs PPL | (mean mass, mean entropy) и сравнение ρ с лучшим базлайном.
  * Что умирает при провале: «sentinel fraction — отдельная дешёвая метрика
    уверенности»; остаётся только «пороговая версия средней массы» (это не потеря,
    но и не новый объект).
  * Критерий: ρ_partial значим (перестановочный p < 0.05) И boot-CI разности
    Δρ = ρ(f_sent) − ρ(best baseline) исключает 0 → «добавляет». Иначе — «не добавляет».
  * Ярус: observational, N=32 документов (скрининг, не каузально).
  * При успехе остаётся: sentinel fraction как дешёвый proxy (top-k logprobs без
    полного логита истинного токена) — только в пределах C4/0.8B/S=128.

ТАВТОЛОГИЯ (урок E-39l): f_sent и PPL — функции одного и того же softmax на тех же
позициях, поэтому rho_all заведомо сильный и НЕ независимый: он не выносит вердикт
по порогам ТЗ. Первична именно избыточная статистика.

Confounds:
  * длина документа из ТЗ §10.5 НЕПрименима: все документы — ровно один и тот же
    128-токенный префикс окна 256 (assert), длина константа; вместо него — партинг
    по домену (ранги внутри домена).
  * позиционное затухание: NLL на p=0..2 огромен (контекст 1-3 токена) и там же масса
    минимальна → режем срез p>=32 (ppl_late / f_sent_late) как чувствительность.

Fail-loud (до любых корреляций):
  1. U128 воспроизводится процедурой exp27b (32 documents, top-128 по частоте на
     последней позиции 256-токенного окна), |U128| = 128.
  2. f_sent, пересчитанный независимо от кэша, совпадает с E-32r.3 per-doc
     (max |Δ| < 0.02) И среднее = 0.5071 (якорь E-34) с допуском 0.02. Иначе
     документы/сплит не те — корреляция считалась бы по разным окнам.

Ограничение (честно): accuracy на HellaSwag/ARC (ТЗ §10.4 п.3) не считается —
датасетов QA в офлайне нет; downstream = только PPL/NLL.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import torch
from scipy.stats import spearmanr

MODEL = "Qwen/Qwen3.5-0.8B"
DOMAINS = ("wiki", "math", "code", "chat")
N_PER_DOMAIN = 8
S_PREFIX = 128          # то же окно, что у exp27b / E-32r.3
L_POOL = 256            # длина пула exp27b (_load_pool)
SENT_THRESHOLD = 0.5    # E-34
E34_ANCHOR = 2077 / 4096
LATE_FROM = 32          # чувствительность: позиции p >= 32
R_GEOM = "1"            # d3 по r=1
BOOT = 1000
PERM = 1000
SEED = 0
GEOM_FIELDS = ("s_l_median", "nb_median", "quad_frac_median", "slope_ratio_median")


# ---------------------------------------------------------------- statistics
def _rho(x, y):
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _rho_p(x, y, nperm, rng):
    """Spearman + перестановочный p (двусторонний, точный по (k+1)/(B+1))."""
    r = _rho(x, y)
    null = np.array([_rho(x[rng.permutation(len(x))], y) for _ in range(nperm)])
    p = (np.sum(np.abs(null) >= abs(r)) + 1) / (nperm + 1)
    return r, float(p), null


def _boot_ci(x, y, b, rng):
    vals = []
    n = len(x)
    for _ in range(b):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 3 or len(np.unique(y[idx])) < 3:
            continue
        vals.append(_rho(x[idx], y[idx]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _resid(y, X):
    """Остатки линейной регрессии y ~ [1, X] (ранги подаются заранее)."""
    A = np.column_stack([np.ones(len(y))] + [X[:, j] for j in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def _rank(a):
    """Нормированные ранги (для partial-корреляций в духе Spearman)."""
    order = np.argsort(np.argsort(a))
    return order.astype(float) / (len(a) - 1)


def _demean_by_group(v, groups):
    out = np.array(v, dtype=float)
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean()
    return out


# ---------------------------------------------------------------- data side
def build_pool():
    """Тот же код-путь отбора документов, что у производителя кэша exp27b."""
    from exp27b_dense import _load_pool

    pool = _load_pool(MODEL, list(DOMAINS), N_PER_DOMAIN, L_POOL)
    for dom in DOMAINS:
        assert len(pool[dom]) == N_PER_DOMAIN, f"pool {dom}: {len(pool[dom])}"
        for s in pool[dom]:
            assert s["input_ids"].shape[1] >= L_POOL, "doc короче 256"
    return pool


def pool_fp(pool):
    """sha256 пула — та же схема, что exp27b_dense (сверка сплита)."""
    fp = {}
    for dom in DOMAINS:
        blob = b"".join(x["input_ids"].flatten().numpy().tobytes() for x in pool[dom])
        fp[dom] = hashlib.sha256(blob).hexdigest()[:12]
    return fp


def build_U128(m, pool):
    """U128 = top-128 по частоте появления в top-128 последней позиции 256-префикса.

    Вербатим процедуры exp27b_dense (тот же порядок обхода — важен для tie-break).
    """
    counts: dict[int, int] = {}
    with torch.no_grad():
        for dom in DOMAINS:
            for s in pool[dom]:
                out = m(input_ids=s["input_ids"], attention_mask=s["attention_mask"])
                p = torch.softmax(out.logits[0, -1].float(), dim=-1)
                for t in p.topk(128).indices.tolist():
                    counts[t] = counts.get(t, 0) + 1
                del out
    return sorted([t for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:128]])


def doc_position_stats(m, ids_row, U128):
    """mass / nll / entropy / max-prob по позициям 128-токенного префикса."""
    ids = ids_row[:S_PREFIX].unsqueeze(0)
    with torch.no_grad():
        logits = m(input_ids=ids).logits[0].float()          # [S, V]
    logp = torch.log_softmax(logits, dim=-1)                  # [S, V]
    u = torch.as_tensor(U128, dtype=torch.long)
    mass = logp[:, u].exp().sum(-1).numpy()                   # [S] = p(U128)
    maxp = logp.max(-1).values.exp().numpy()                  # [S]
    ent = -(logp.exp() * logp).sum(-1).numpy()                # [S]
    # NLL: логит позиции p предсказывает токен p+1 → p = 0..S-2
    tgt = ids_row[1:S_PREFIX]
    nll = (-logp[: S_PREFIX - 1].gather(1, tgt.unsqueeze(1))[:, 0]).numpy()
    del logits, logp
    return mass, nll, ent, maxp


def load_e32r3():
    """Сторона A: sentinel_fraction + геометрия per doc из E-32r.3 (тот же кэш)."""
    d = json.load(open(os.path.join("results", "exp32r3_layer_profile.json")))
    out = {}
    for dom in DOMAINS:
        for si in range(N_PER_DOMAIN):
            rec = {"dom": dom, "si": si}
            for L in (6, 12, 18, 19, 22, 23):
                pd = d["per_doc"][f"{L}|{dom}/{si}"]
                rec[f"sent_frac_L{L}"] = pd["sentinel_fraction"]
            pd = d["per_doc"][f"23|{dom}/{si}"]
            for f in GEOM_FIELDS:
                rec[f"geom_{f}"] = float(pd[f])
            for m in ("raw", "gf"):
                rec[f"geom_d3_{m}_r1"] = float(pd["d3_by_r"][m][R_GEOM][0])
            out[(dom, si)] = rec
    return out


# ---------------------------------------------------------------- main
def _verdict(rho_all, rho_partial, p_partial, delta, ci):
    """Вердикт по пререгистрированной схеме. Направление ВАЖНО: симметричное
    «CI исключает 0» читается как «бьёт базлайн» и печатало «промежуточный» там, где
    базлайн строго лучше (реальный случай E-39j: Δρ = −0.48)."""
    sig = p_partial < 0.05
    if sig and ci[0] > 0:
        return ("incremental: f_sent даёт сигнал сверх mean-mass/entropy и лучше "
                "лучшего непрерывного базлайна")
    if ci[1] < 0:
        return ("не метрика, а потеря версии базлайна: f_sent предсказывает downstream "
                f"СТРОГО хуже непрерывного базлайна (Δρ = {delta:+.3f}, CI исключает 0), "
                f"сверх базлайна сигнала нет (partial ρ = {rho_partial:+.3f}, "
                f"p = {p_partial:.3f}) → как отдельная дешёвая метрика уверенности не "
                "используется")
    if not sig and abs(delta) < 0.1:
        return ("не добавляет ничего к тривиальному базлайну: sentinel fraction — "
                "пороговая (грубая) версия средней массы U128")
    return ("промежуточный: часть связи снимается базлайном, остаток незначим; "
            "сильного вердикта не выносим")


def recheck_verdict(path):
    """Пересчёт поля verdict из уже сохранённых чисел (числа прогона не трогаются).

    Нужен для прогонов, сделанных до починки направления в `_verdict`: статистика в
    JSON полная и корректная, ошибочна была только строка вердикта.
    """
    with open(path) as f:
        d = json.load(f)
    inc, pr, bs = d["incremental"], d["primary"], d["baselines"]
    old = d.get("verdict")
    d["verdict"] = _verdict(pr[0]["rho"], inc["rho_partial"], inc["p_perm"],
                            inc["delta_rho_vs_best_baseline"], inc["delta_boot_ci95"])
    inc["better_than_baseline"] = bool(inc["delta_boot_ci95"][0] > 0)
    inc["worse_than_baseline"] = bool(inc["delta_boot_ci95"][1] < 0)
    if "beats_baseline" in inc:
        # v1 писал один симметричный флаг `beats_baseline` (|Δρ| ≠ 0 → true), то есть
        # «прогон отличается от базлайна». После исправления направления он читался бы
        # как «лучше базлайна» и противоречил вердикту, поэтому поле удаляется.
        inc["field_history"] = {"removed_beats_baseline_v1": True,
                                "replaced_by": ["better_than_baseline",
                                                "worse_than_baseline"]}
        inc.pop("beats_baseline")
    d["beats_baseline_direction_fixed"] = True
    # идемпотентность: повторный вызов не должен затирать v1 текущим verdict
    v1 = (d.get("verdict_history") or {}).get("v1") or old
    d["verdict_history"] = {"v1": v1, "v2": d["verdict"],
                            "why": "v1 сравнивал |Δρ| с нулём симметрично и не различал "
                                   "направление (базлайн лучше → писалось «промежуточный»); "
                                   "числа не менялись"}
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"verdict (v1): {old}\nverdict (v2): {d['verdict']}")
    return d


def main(device="cpu", results_dir="results", **kw):
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)

    e32 = load_e32r3()
    print("Loading pool (exp27b code path)...", flush=True)
    pool = build_pool()
    fp = pool_fp(pool)
    print(f"pool_fp = {fp}", flush=True)

    from transformers import AutoModelForCausalLM

    print(f"Loading {MODEL} (fp32, eager — как в exp27b_dense)...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()

    U128 = build_U128(m, pool)
    assert len(U128) == 128, f"U128: {len(U128)} != 128"

    rows = []
    for dom in DOMAINS:
        for si in range(N_PER_DOMAIN):
            ids = pool[dom][si]["input_ids"][0]
            mass, nll, ent, maxp = doc_position_stats(m, ids, U128)
            sent = mass < SENT_THRESHOLD
            n_nll = len(nll)                              # S-1
            late_n = n_nll - LATE_FROM                    # p >= 32
            rec = dict(e32[(dom, si)])
            rec.update({
                "sent_frac_mine": float(sent.mean()),
                "sent_frac_mine_late": float(sent[LATE_FROM:].mean()),
                "mass_mean": float(mass.mean()),
                "mass_median": float(np.median(mass)),
                "mass_p10": float(np.percentile(mass, 10)),
                "maxprob_mean": float(maxp.mean()),
                "entropy_mean": float(ent[:n_nll].mean()),
                "nll_mean": float(nll.mean()),
                "ppl": float(np.exp(nll.mean())),
                "nll_mean_late": float(nll[LATE_FROM:].mean()),
                "ppl_late": float(np.exp(nll[LATE_FROM:].mean())),
                "n_positions": int(len(mass)),
                "n_nll_positions": int(n_nll),
                "near_boundary": int(np.sum(np.abs(mass - SENT_THRESHOLD) < 1e-4)),
                "n_sent": int(sent.sum()),
            })
            rows.append(rec)
            print(f"  {dom}/{si}: sent={rec['sent_frac_mine']:.3f} "
                  f"(e32r3 {rec['sent_frac_L23']:.3f}) ppl={rec['ppl']:.2f} "
                  f"ppl_late={rec['ppl_late']:.2f}", flush=True)
    del m
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- fail-loud self-checks: документы должны быть ТЕ ЖЕ, что у кэша
    mine = np.array([r["sent_frac_mine"] for r in rows])
    ref = np.array([r["sent_frac_L23"] for r in rows])
    dmax = float(np.max(np.abs(mine - ref)))
    assert dmax < 0.02, (
        f"sentinel fraction не совпал с E-32r.3 (max |Δ| = {dmax:.4f}): сплит/документы "
        "не те — корреляция считалась бы по разным окнам")
    assert abs(mine.mean() - E34_ANCHOR) < 0.02, (
        f"средняя доля sentinel {mine.mean():.4f} не равна якорю E-34 {E34_ANCHOR:.4f}")
    # инвариантность по слоям (raw sum не зависит от слоя — E-34)
    per_layer = {L: float(np.mean([r[f"sent_frac_L{L}"] for r in rows]))
                 for L in (6, 12, 18, 19, 22, 23)}
    assert max(per_layer.values()) - min(per_layer.values()) < 1e-3, per_layer
    # длина константа → confound длины из ТЗ §10.5 неприменим
    assert len({r["n_positions"] for r in rows}) == 1 and \
        rows[0]["n_positions"] == S_PREFIX

    # ---- статистика
    ppl = np.array([r["ppl"] for r in rows])
    ppl_late = np.array([r["ppl_late"] for r in rows])
    doms = np.array([r["dom"] for r in rows])
    f_all, f_late = mine, np.array([r["sent_frac_mine_late"] for r in rows])
    mass_mean = np.array([r["mass_mean"] for r in rows])
    mass_med = np.array([r["mass_median"] for r in rows])
    mass_p10 = np.array([r["mass_p10"] for r in rows])
    maxprob = np.array([r["maxprob_mean"] for r in rows])
    ent = np.array([r["entropy_mean"] for r in rows])

    def stat(x, y, label):
        r, p, null = _rho_p(x, y, PERM, rng)
        return {"pair": label, "rho": r, "p_perm": p,
                "boot_ci95": _boot_ci(x, y, BOOT, rng),
                "null_abs_p95": float(np.percentile(np.abs(null), 95)),
                "n": int(len(x))}

    primary = [
        stat(f_all, ppl, "f_sent(all) ~ PPL(all)  [ТЗ-формулировка, НЕ независима]"),
        stat(f_late, ppl_late, "f_sent(p>=32) ~ PPL(p>=32)  [чувствительность к позициям]"),
    ]
    baselines = [
        stat(mass_mean, ppl, "baseline mean mass ~ PPL"),
        stat(mass_med, ppl, "baseline median mass ~ PPL"),
        stat(mass_p10, ppl, "baseline p10 mass ~ PPL"),
        stat(maxprob, ppl, "baseline mean max-prob ~ PPL"),
        stat(ent, ppl, "baseline mean entropy ~ PPL"),
        stat(f_late, ppl, "f_sent(p>=32) ~ PPL(all)"),
    ]

    # Избыточность: partial Spearman по рангам, контроль mean mass + mean entropy.
    rx, ry = _rank(f_all), _rank(np.log(ppl))
    Z = np.column_stack([_rank(mass_mean), _rank(ent)])
    partial = _rho(_resid(rx, Z), _resid(ry, Z))
    # перестановочный null для partial: перемешиваем x-документы целиком
    null_p = []
    for _ in range(PERM):
        pi = rng.permutation(len(rx))
        null_p.append(_rho(_resid(rx[pi], Z), _resid(ry, Z)))
    p_partial = (np.sum(np.abs(null_p) >= abs(partial)) + 1) / (PERM + 1)
    # контроль домена (ранги внутри домена)
    xd, yd = _demean_by_group(rx, doms), _demean_by_group(ry, doms)
    r_dom, p_dom, _ = _rho_p(xd, yd, PERM, rng)

    # Δρ против лучшего базлайна (парный bootstrap)
    base_cands = {"mean mass": mass_mean, "median mass": mass_med,
                  "p10 mass": mass_p10, "mean max-prob": maxprob,
                  "mean entropy": ent}
    base_rho = {k: _rho(v, ppl) for k, v in base_cands.items()}
    best_k = max(base_rho, key=lambda k: abs(base_rho[k]))
    xb = base_cands[best_k]
    diff_boot = []
    n = len(rx)
    for _ in range(BOOT):
        idx = rng.integers(0, n, n)
        a, b = _rho(f_all[idx], ppl[idx]), _rho(xb[idx], ppl[idx])
        if np.isfinite(a) and np.isfinite(b):
            diff_boot.append(a - b)
    d_obs = _rho(f_all, ppl) - base_rho[best_k]

    # вторичная ось: геометрия (не общий softmax) — exploratory
    geom = {}
    for k in ("geom_s_l_median", "geom_nb_median", "geom_quad_frac_median",
              "geom_slope_ratio_median", "geom_d3_raw_r1", "geom_d3_gf_r1"):
        v = np.array([r[k] for r in rows])
        geom[k] = {"vs_f_sent": stat(v, f_all, f"{k} ~ f_sent"),
                   "vs_ppl": stat(v, ppl, f"{k} ~ PPL")}

    # ---- вердикт (направление важно, см. _verdict) ----
    tz_rho = primary[0]["rho"]
    ci = [float(np.percentile(diff_boot, 2.5)), float(np.percentile(diff_boot, 97.5))]
    verdict = _verdict(tz_rho, partial, p_partial, d_obs, ci)

    out = {
        "experiment": "E-39j sentinel as confidence",
        "model": MODEL, "dtype": "fp32", "attn": "eager", "device": device,
        "s_prefix": S_PREFIX, "sent_threshold": SENT_THRESHOLD,
        "late_from": LATE_FROM, "n_docs": len(rows), "domains": list(DOMAINS),
        "downstream": "per-doc mean NLL / PPL на позициях p=0..S-2 того же префикса",
        "selfchecks": {
            "sent_frac_vs_e32r3_max_abs_diff": dmax, "tol": 0.02,
            "sent_frac_mean": float(mine.mean()), "e34_anchor": E34_ANCHOR,
            "sent_frac_by_layer": per_layer,
            "n_positions_const": True,
            "near_boundary_positions_total": int(sum(r["near_boundary"] for r in rows)),
            "pool_fp": fp, "U128_sha12": hashlib.sha256(
                str(U128).encode()).hexdigest()[:12],
        },
        "tautology_note": (
            "f_sent и PPL — функции одного softmax на тех же позициях, поэтому rho_all "
            "заведомо силён и не является независимым свидетельством: пороги ТЗ §10.6 "
            "к нему не применяются. Первична избыточная статистика (partial vs базлайн)."
        ),
        "confounds": {
            "length": "неприменим: все документы — один и тот же 128-токенный префикс",
            "position": f"срез p>={LATE_FROM} как чувствительность",
            "domain": "ранги внутри домена (r_dom)",
        },
        "primary": primary,
        "baselines": baselines,
        "baseline_rho_vs_ppl": base_rho,
        "baseline_best": best_k,
        "incremental": {
            "controls": ["rank(mean mass)", "rank(mean entropy)"],
            "rho_partial": float(partial), "p_perm": float(p_partial),
            "significant": bool(p_partial < 0.05),
            "rho_within_domain": r_dom, "p_within_domain": p_dom,
            "delta_rho_vs_best_baseline": float(d_obs),
            "delta_boot_ci95": ci, "better_than_baseline": bool(ci[0] > 0),
            "worse_than_baseline": bool(ci[1] < 0),
        },
        "secondary_geometry_exploratory": geom,
        "verdict_tz_thresholds": {
            "rho_all": tz_rho,
            "tz_reading": ("rho > 0.5 → «sentinel — cheap confidence»; rho < 0.3 → "
                           "«не использовать»") ,
            "applied": False,
            "why_not_applied": "rho_all не независима (общий softmax) — см. tautology_note",
        },
        "verdict": verdict,
        "limitations": [
            "downstream = только PPL/NLL; QA accuracy (HellaSwag/ARC) не считался — "
            "датасетов нет в офлайне (ТЗ §10.4 п.3)",
            "N=32 документа, 4 домена × 8 — скрининг, не каузально (paper §3.2)",
            "один чекпойнт (0.8B) и один датасет-микс (C4/math/code/chat)",
            "позиции p=0..2 дают почти всю вариацию NLL (контекст 1-3 токена) — "
            "срез p>=32 приведён рядом",
        ],
        "per_doc": rows,
        "wall_sec": time.time() - t0,
    }
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp39j_sentinel.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nverdict: {verdict}")
    print(f"rho_all = {tz_rho:+.3f} | partial | mean-mass,entropy = {partial:+.3f} "
          f"(p_perm = {p_partial:.4f}) | best baseline {best_k} rho = {base_rho[best_k]:+.3f}")
    print(f"Δρ vs best baseline = {d_obs:+.3f} CI {ci}  "
          f"better={ci[0] > 0} worse={ci[1] < 0}")
    print(f"output: {out_path}  wall: {time.time() - t0:.0f}s")
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 2 and sys.argv[1] == "--recheck":
        recheck_verdict(sys.argv[2])
    else:
        main()
