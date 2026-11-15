#!/usr/bin/env python3
"""AT0_7 gates + tests — G0..G2, G1 identity, G7-калибровка, POSITIVE, H1–H5, вердикт
(ТЗ T0.7 v3, closed r3: PASS). Импорт: experiments/AT0_7_loop.py.
Константы и logline — at0_7_core; Q/null-формулы — AT0_6_spectrum (v2.2, импорт);
токено-эстиманды и (токен,фаза)-lookup — at0_7_core. Документация — в docstring'ах (пре-рег ТЗ).
"""
import numpy as np

import AT0_6_spectrum as a6
import at0_7_core as core

from at0_7_core import (  # noqa: F401  (константы ТЗ + logline — shared)
    ALPHA, B, D_DIM, G1D_SEED, G1A_P, G1A_T, G1B_D, G1B_SEED, G7_K_MAX, G7_N,
    G7_RATIO_THR, G7_SEED, G7_TOK_T, H1_FLOOR, H5_TOL, P_GRID, POS_A, POS_D,
    POS_P0, POS_PPF_THR, POS_QD, POS_SEED, R0_MIN, REPMIN, T_PRIME, V_SYNTH,
    logline, unique_root,
)
# ------------------------------------------------------------------- гейты

def smoke_edges(docs, pools):
    """Смоук/деградация: doc без токенов (corpus), doc с len < p, p из T0.3."""
    cor = docs["cor_w0"]["toks"]
    assert len(cor) == 0
    assert core.ppf(cor, 13) is None and core.maxrun(cor, 13) is None
    assert core.chance(cor) is None and core.pstar(cor) == (None, None)
    assert core.token_row(cor, None)["rep_frac"] is None
    short = np.arange(10)                                  # len < p
    assert core.ppf(short, 13) is None and core.maxrun(short, 13) is None
    assert core.ratio(short, 13) is None
    tr = core.token_row(docs["code_14"]["toks"], pools["p_map"]["code_14"])   # p из T0.3
    assert tr["p_rep"] == pools["p_map"]["code_14"] and 0.0 < tr["rep_frac"] <= 1.0
    logline(f"SMOKE edges: corpus-док (0 токенов) → None; len<p → None; "
            f"p_T0.3 (code_14, p={tr['p_rep']}) → rep_frac = {tr['rep_frac']:.4f}")


def g1a_identity():
    """G1a: синтетика tok = repeat([v_i], v_i = i, 13 РАЗЛИЧНЫХ), p=13, T=383 ⇒
    rep_frac(13) = 1.0 ТОЧНО ∧ |RATIO(13) − 1/C| < 1e-12 (C = 11287/146689; RATIO = 12.996279
    ≠ 13 — конечный T: 383 = 13·29+6) ∧ maxrun = T−p = 370."""
    t = core.repeat_tokens(np.arange(13), G1A_T)
    r13 = core.ppf(t, G1A_P)
    C = core.chance(t)
    C_exact = 11287 / 146689
    ratio13 = core.ratio(t, G1A_P)
    mr = core.maxrun(t, G1A_P)
    ok = bool(r13 == 1.0 and abs(ratio13 - 1.0 / C) < 1e-12
              and abs(C - C_exact) < 1e-12 and mr == G1A_T - G1A_P)
    logline(f"G1a: rep_frac(13) = {r13} (exact 1.0); C = {C:.10f} (11287/146689, dev "
            f"{abs(C - C_exact):.1e}); RATIO = {ratio13:.6f} (≠13); maxrun = {mr} (= T−p)")
    return {"ok": ok, "rep_frac_13": r13, "C": C, "C_exact": C_exact,
            "C_dev": float(abs(C - C_exact)), "RATIO": float(ratio13), "maxrun": mr,
            "tol": 1e-12, "T": G1A_T, "p": G1A_P}


def g1b_q_identity():
    """G1b: Q-identity на чистой p-синусоиде (a=1, σ=1, p₀=13, D=32) ⇒ Q(p₀)/D ≥ 50
    (как T0.6 G1b, замер 66.5 — тот же сид 77700). G1c УБРАН (R≡0 тождество, тривиален, r2)."""
    x = a6.synth_line(POS_P0, 1.0, G1B_SEED, T=T_PRIME, D=G1B_D)
    q0 = a6.q_single(x, POS_P0)
    ok = bool(q0 / G1B_D >= 50.0)
    logline(f"G1b: Q(13)/32 = {q0 / G1B_D:.1f} (≥50; T0.6 G1b замер 66.5, seed {G1B_SEED})")
    return {"ok": ok, "q0_over_D": float(q0 / G1B_D), "D": G1B_D, "seed": G1B_SEED,
            "a": 1.0, "sigma": 1.0, "p0": POS_P0,
            "note": "G1c УБРАН (R≡0 — алгебраическое тождество, тривиален, r2)"}


def g1d_iid(toks777):
    """G1d: iid-токены (V=4096, сид 777 (= G7 k=0), T=383) ⇒ max_p RATIO(p) ≤ 1.1
    (детерминированно: замер 1.0118 @p37; V=1000 дал бы 1.59 — словарь ЗАДАН, r1)."""
    rs = {p: core.ratio(toks777, p) for p in P_GRID}
    mp = max(rs, key=rs.get)
    ok = bool(rs[mp] <= 1.1)
    logline(f"G1d: iid токены (V=4096, seed 777, T=383): max_p RATIO = {rs[mp]:.4f} @p{mp} "
            f"(≤1.1; замер 1.0118 @p37)")
    return {"ok": ok, "max_ratio": float(rs[mp]), "p_at_max": int(mp), "tol": 1.1,
            "seed": G1D_SEED, "V": V_SYNTH, "T": G1A_T, "grid": [P_GRID[0], P_GRID[-1]]}


def g2_cells(docs, pools):
    """G2: 96 доков; T' = T−1 per-doc (L23); per-dim var > 0; n_tok: 30 (loopy) + 39 (H2-пул)
    + 11 (loopy-undetected) = 80, corpus 0/16 (явный assert в main)."""
    bad = []
    for nm in sorted(docs):
        u, hn = docs[nm]["layers"][23]
        x = a6.h_cell(docs[nm], 23)
        if x.shape[0] != u.shape[0] - 1:
            bad.append((nm, "T'"))
        if not np.all(np.var(x, axis=0) > 0):
            bad.append((nm, "var"))
    nt = {nm: int(len(docs[nm]["toks"])) for nm in sorted(docs)}
    n_l = sum(1 for nm in pools["loopy30"] if nt[nm] > 0)
    n_h2 = sum(1 for nm in pools["h2"] if nt[nm] > 0)
    n_lu = sum(1 for nm in pools["lu11"] if nt[nm] > 0)
    n_cor0 = sum(1 for nm in pools["cor16"] if nt[nm] == 0)
    ok = bool(len(docs) == 96 and not bad and n_l == 30 and n_h2 == 39
              and n_lu == 11 and n_cor0 == 16
              and all(nt[nm] == 384 for nm in pools["tokenizable80"]))
    logline(f"G2: 96 доков; T'=T−1 + per-dim var>0 (L23) {'OK' if not bad else bad[:6]}; "
            f"n_tok: loopy {n_l}/30 + H2-пул {n_h2}/39 + LU {n_lu}/11 = {n_l + n_h2 + n_lu}/80; "
            f"corpus {n_cor0}/16 без токенов (G0-a not_testable)")
    return {"ok": ok, "n_docs": int(len(docs)), "n_tok_loopy": n_l, "n_tok_h2": n_h2,
            "n_tok_lu11": n_lu, "n_tok_total": int(n_l + n_h2 + n_lu), "n_corpus_no_tok": n_cor0,
            "t_prime": "T−1 per-doc (L23)", "bad": bad}


# ------------------------------------------------------------------- G7 / POSITIVE

def run_g7(N=G7_N, Bn=B):
    """G7 (калибровка): 20 iid-синтетиков (D=1024, T'=383, h=N(0,1), сид 777+k; токены
    rng(777+k).integers(0,4096,383)): (a) count-гейт: ≤1 из 20 с max_p RATIO(p) ≥ 3 (r2
    MAJOR-A6: «≤1.1» падало на seed 784=1.98/785=1.96; MC 5000 сидов: P(сид ≥3) ≈ 1.4e-3);
    (b) Q(p) null one-sided k ≤ 3 (как T0.6 G7; P(ложный фейл) = P(k≥4|Bin(20,0.05)) = 0.016).
    doc_idx-сдвиг 9000+k — конвенция T0.6 G7."""
    rows = []
    n_ratio3 = 0
    for k in range(N):
        toks = np.random.default_rng(G7_SEED + k).integers(0, V_SYNTH, G7_TOK_T)
        x = np.random.default_rng(G7_SEED + k).standard_normal((T_PRIME, D_DIM))
        x = x - x.mean(axis=0, keepdims=True)
        rs = {p: core.ratio(toks, p) for p in P_GRID}
        mp = max(rs, key=rs.get)
        n_ratio3 += int(rs[mp] >= G7_RATIO_THR)
        _, qmax, pval = a6.perm_null_grid(x, 9000 + k, Bn=Bn)
        rows.append({"k": k, "seed": G7_SEED + k, "max_ratio": float(rs[mp]),
                     "p_at_max": int(mp), "q_max_obs": float(qmax), "p": float(pval),
                     "q_hit": bool(pval <= ALPHA)})
        logline(f"G7 k={k}: max_p RATIO = {rs[mp]:.3f} @p{mp}; Q_max = {qmax:.1f}, p = {pval:.4f}")
    k_hits = sum(r["q_hit"] for r in rows)
    ok = bool(n_ratio3 <= 1 and k_hits <= G7_K_MAX)
    logline(f"G7: (a) count-гейт max_p RATIO ≥ 3: {n_ratio3}/{N} (≤1); "
            f"(b) Q-null k = {k_hits}/{N} (≤{G7_K_MAX}; B={Bn})")
    return {"ok": ok, "n": N, "B": Bn,
            "a": {"gate": f"≤1 из {N} с max_p RATIO ≥ {G7_RATIO_THR}", "n_gt3": int(n_ratio3)},
            "b": {"gate": f"k ≤ {G7_K_MAX}", "k": int(k_hits),
                  "false_fail_p": float(a6.binom_sf(G7_K_MAX + 1, 20, ALPHA))},
            "rows": rows}


def run_positive():
    """POSITIVE: 2 синтетика (сиды 7770+k, a=1, σ=1, словарь 4096, D=32, T'=383):
    (1) чистый p-repeat токенов (p=13, v_i = rng(7771).integers(0,4096,13), assert 13
        различных) ⇒ rep_frac(13) = 1.0 точно (гейт проверяет p=13, НЕ кратные);
    (2) чистая h-осцилляция на p=13 (iid-токены rng(7772)) ⇒ max_p PPF < 0.2 ∧ Q(13)/32 ≥ 50
        (линия есть, repeat'а нет — разделяет две сигнатуры; замер max_p PPF ≈ 0.003-0.012)."""
    v = np.random.default_rng(POS_SEED + 1).integers(0, V_SYNTH, POS_P0)
    assert len(np.unique(v)) == POS_P0, "POS1: 13 РАЗЛИЧНЫХ v_i"
    toks1 = core.repeat_tokens(v, T_PRIME)
    ppf1 = core.ppf(toks1, POS_P0)
    h1 = a6.synth_line(POS_P0, POS_A, POS_SEED, T=T_PRIME, D=POS_D)
    q1 = a6.q_single(h1, POS_P0)
    pos1 = {"ok": bool(ppf1 == 1.0), "gate": "rep_frac(13) = 1.0 точно (p=13, не кратные)",
            "rep_frac_13": ppf1, "v": [int(x) for x in v], "seed_v": POS_SEED + 1,
            "seed_h": POS_SEED, "q_over_D_descriptive": float(q1 / POS_D)}
    toks2 = np.random.default_rng(POS_SEED + 2).integers(0, V_SYNTH, T_PRIME)
    ppfs2 = {p: core.ppf(toks2, p) for p in P_GRID}
    mx2 = max(ppfs2.values())
    h2 = a6.synth_line(POS_P0, POS_A, POS_SEED + 1, T=T_PRIME, D=POS_D)
    q2 = a6.q_single(h2, POS_P0)
    ok2 = bool(mx2 < POS_PPF_THR and q2 / POS_D >= POS_QD)
    pos2 = {"ok": ok2, "gate": f"max_p PPF < {POS_PPF_THR} ∧ Q(13)/32 ≥ {POS_QD}",
            "max_ppf": float(mx2), "ppf_13": float(ppfs2[POS_P0]),
            "q_over_D": float(q2 / POS_D), "seed_h": POS_SEED + 1, "seed_toks": POS_SEED + 2}
    logline(f"POS1: rep_frac(13) = {ppf1} (exact 1.0), Q/D (descr) = {q1 / POS_D:.1f}; "
            f"POS2: max_p PPF = {mx2:.4f} (<0.2), Q(13)/32 = {q2 / POS_D:.1f} (≥50)")
    return {"ok": bool(pos1["ok"] and pos2["ok"]), "pos1": pos1, "pos2": pos2}


# ------------------------------------------------------------------- тесты

def run_h1_h4(docs, pools, doc_idx, rows, names, Bn=B):
    """H1 (primary, ретроспективный r0 + валидация): f1 = #{rep_frac(p_T0.3) ≥ 0.8}/30,
    PASS ≥ 28 (r0: 30/30, min 0.888 (code_14), med 0.9973 — воспроизведение в 1e-12).
    H4 (descriptive): Spearman(Q(p_T0.3), maxrun(p_T0.3)) + Spearman(Q, rep_frac);
    Q(p_T0.3) + p_perm — первый полный committed (B=199, сид 2026+doc_idx·1009+s,
    doc_idx = индекс в sorted(96), как T0.6)."""
    h1_rows, q_rows = [], []
    for nm in names:
        tr = rows[nm]
        x = a6.h_cell(docs[nm], 23)
        q_obs, pval = a6.perm_null_single(x, tr["p_rep"], doc_idx[nm], Bn=Bn)
        h1_rows.append({"doc": nm, "p": int(tr["p_rep"]), "rep_frac": tr["rep_frac"],
                        "maxrun": tr["maxrun"], "RATIO": tr["RATIO"]})
        q_rows.append({"doc": nm, "p": int(tr["p_rep"]), "q_obs": float(q_obs),
                       "p_perm": float(pval), "maxrun": tr["maxrun"],
                       "rep_frac": tr["rep_frac"]})
        logline(f"H1/H4 {nm} p={tr['p_rep']}: rep_frac = {tr['rep_frac']:.4f}, "
                f"maxrun = {tr['maxrun']}, Q = {q_obs:.1f}, p_perm = {pval:.4f}")
    rep = [r["rep_frac"] for r in h1_rows]
    f1 = int(sum(1 for r in rep if r >= REPMIN))
    r0_ok = bool(f1 == 30 and abs(min(rep) - R0_MIN) < 1e-12)
    h1 = {"n": len(h1_rows), "f1": f1, "gate": f"f1 ≥ {H1_FLOOR}", "ok": bool(f1 >= H1_FLOOR),
          "min_rep_frac": float(min(rep)), "median_rep_frac": float(np.median(rep)),
          "r0_anchor": {"f1": 30, "min": R0_MIN, "min_doc": "code_14", "med": 0.9973,
                        "tol": "1e-12"}, "r0_reproduced": r0_ok, "rows": h1_rows}
    h4 = {"n": len(q_rows),
          "spearman_q_maxrun": core.spearman([r["q_obs"] for r in q_rows],
                                             [r["maxrun"] for r in q_rows]),
          "spearman_q_repfrac": core.spearman([r["q_obs"] for r in q_rows],
                                              [r["rep_frac"] for r in q_rows]),
          "gate": "ρ_s(Q, maxrun) ≥ 0.4 → h-линия следует за длиной repeat'а",
          "rows": q_rows}
    logline(f"H1: f1 = {f1}/{len(h1_rows)} (PASS ≥ {H1_FLOOR}); min rep_frac = {min(rep):.4f} "
            f"(r0: {R0_MIN}); r0_reproduced(1e-12) = {r0_ok}")
    logline(f"H4: Spearman(Q, maxrun) = {h4['spearman_q_maxrun']:.4f}; "
            f"Spearman(Q, rep_frac) = {h4['spearman_q_repfrac']:.4f}")
    return h1, h4, r0_ok


def run_h2(rows, pools, pairing):
    """H2 (descriptive, НЕ бинарный): per-subpool #{rep_frac ≥ 0.8}: loopy-undetected 6/11
    (по докам) = 5/8 (по уникальным текстам), clean-37 0/37, remainder 2/2 (оба — токено-дубля
    wiki_0/wiki_4 = 0 новых уникальных) + per-unique-text (H2-пул 39 = 38 уникальных: пара
    own_w13≡wiki_13)."""
    out = {}
    for lab, names in (("loopy_undetected", pools["lu11"]), ("clean37", pools["clean37"]),
                       ("remainder", pools["rem2"])):
        stab = [nm for nm in names
                if rows[nm]["rep_frac"] is not None and rows[nm]["rep_frac"] >= REPMIN]
        roots = {unique_root(nm, pairing) for nm in names}
        stab_u = {unique_root(nm, pairing) for nm in stab}
        out[lab] = {"n_docs": len(names), "n_unique_texts": len(roots),
                    "n_stable_docs": len(stab), "n_stable_unique": len(stab_u),
                    "stable_docs": sorted(stab),
                    "docs": [{"doc": nm, "p_rep": rows[nm]["p_rep"],
                              "rep_frac": rows[nm]["rep_frac"],
                              "maxrun": rows[nm]["maxrun"], "max_ratio": rows[nm]["max_ratio"]}
                             for nm in sorted(names)]}
        logline(f"H2 {lab}: {len(stab)}/{len(names)} (по докам) = {len(stab_u)}/{len(roots)} "
                f"(по уникальным текстам): {sorted(stab)}")
    out["r0_expected"] = {"loopy_undetected": "6/11 (по докам) = 5/8 (уникальные)",
                          "clean37": "0/37", "remainder": "2/2 (оба — дубли wiki_0/wiki_4)"}
    return out


def run_h3(rows, pools):
    """H3 (descriptive, граница necessary): кросстаб по ВСЕМ 80 токенизируемых
    (стабильный repeat rep_frac ≥ 0.8 × T0.3-detected) → доля loopy среди стабильных-repeat
    доков (repeat necessary, не sufficient — 6/11 undetected со стабильным repeat без
    T0.3-детекции). Corpus-16 — без токенов, отдельная not_testable-строка (не в 80)."""
    names = pools["tokenizable80"]
    tab = {"detected_stable": [], "detected_unstable": [],
           "undetected_stable": [], "undetected_unstable": []}
    for nm in sorted(names):
        det = nm in pools["p_map"]
        st = rows[nm]["rep_frac"] is not None and rows[nm]["rep_frac"] >= REPMIN
        tab[("detected" if det else "undetected") + ("_stable" if st else "_unstable")].append(nm)
    n_stab = len(tab["detected_stable"]) + len(tab["undetected_stable"])
    share = float(len(tab["detected_stable"]) / n_stab) if n_stab else None
    out = {"n_total": len(names),
           "table": {k: {"n": len(v), "docs": v} for k, v in tab.items()},
           "loopy_share_among_stable": share,
           "corpus_not_testable": {"n": len(pools["cor16"]), "note": "0 токенов — G0-a not_testable"},
           "note": ("repeat necessary, не sufficient: 6/11 undetected со стабильным repeat без "
                    "T0.3-детекции + 2 remainder-дубля (wiki_0/wiki_4)")}
    logline(f"H3: стабильный repeat = {n_stab}/80; доля loopy (T0.3-detected) среди стабильных "
            f"= {len(tab['detected_stable'])}/{n_stab} = {share:.4f}")
    return out


def run_h5(docs, rows, pools):
    """H5 (h-уровневая ГРАНИЦА, descriptive, НЕ тест): R(p) ≡ 0 — алгебраическое тождество
    ((токен,фаза)-lookup, p-bin-preserving); численно: все 30 loopy R ∈ [2.5e-32, 3.5e-30]
    (r0); clean: max ~1e-28. Пороги 0.1/0.3 УБРАНЫ (недостижимы — R≡0). Rtok — per-token без
    фазы (вырожден при V≫T — словарезависим). Выровнение: x_t = h_{t+1} − mean ⇒ x_t ↔ tok[t+1].
    По всем 96: corpus-16 — not_testable-строка (ветка реального исполнителя)."""
    # все 96 доков: corpus-16 — отдельная not_testable-строка (ветка — реальный исполнитель)
    per_doc = {}
    for nm in sorted(docs):
        tr = rows[nm]
        if tr["p_rep"] is None or tr["T"] < 2:
            per_doc[nm] = {"p": None, "R": None, "Rtok": None, "V": tr["V"],
                           "note": "not_testable (нет токенов)"}
            continue
        x = a6.h_cell(docs[nm], 23)
        tk = docs[nm]["toks"][1 : 1 + x.shape[0]]
        R, Rtok = core.lookup_resid(x, tk, tr["p_rep"])
        per_doc[nm] = {"p": tr["p_rep"], "R": R, "Rtok": Rtok, "V": tr["V"]}
    bad = [nm for nm in pools["tokenizable80"]
           if per_doc[nm]["R"] is None or per_doc[nm]["R"] > H5_TOL]
    r_loopy = [per_doc[nm]["R"] for nm in pools["loopy30"]]
    r_clean = [per_doc[nm]["R"] for nm in pools["clean37"]]
    rtok_loopy = [per_doc[nm]["Rtok"] for nm in pools["loopy30"]]
    out = {"per_doc": per_doc,
           "R_loopy30": {"min": float(min(r_loopy)), "max": float(max(r_loopy))},
           "R_clean37_max": float(max(r_clean)),
           "Rtok_loopy30": {"min": float(min(rtok_loopy)), "max": float(max(rtok_loopy))},
           "identity": "R(p) ≡ 0 (алгебраическое тождество, не измерение; пороги 0.1/0.3 УБРАНЫ)",
           "fp64_tol": H5_TOL, "ok": bool(not bad), "bad": bad,
           "r2_ref": {"R_loopy30_range": [2.5e-32, 3.5e-30], "R_clean37_max": "~1e-28",
                      "note": ("r2-замер (порядок суммирования fp64); r3-ревью и текущий прогон: "
                               "[2.88e-31, 3.66e-30] — тот же порядок; гейт <1e-20 не затронут (К3 r1-exec)")},
           "alignment": "x_t = h_{t+1} − mean ⇒ x_t ↔ tok[t+1] (tk = toks[1:T'])",
           "note": ("вывод ТЗ: p-структура h = необходимое feedforward-последствие token-copy-paste; "
                    "«динамика сверх токенов» per-doc disk-only НЕ ИЗМЕРИМА (контрфактуальный "
                    "forward — вне scope)")}
    logline(f"H5: R loopy30 ∈ [{min(r_loopy):.3e}, {max(r_loopy):.3e}] (r0: 2.5e-32..3.5e-30); "
            f"R clean37 max = {max(r_clean):.3e} (r0: ~1e-28); Rtok loopy30 ∈ "
            f"[{min(rtok_loopy):.3e}, {max(rtok_loopy):.3e}] (вырожден при V≫T); "
            f"identity R<1e-20: {'OK' if not bad else bad}")
    return out


# ------------------------------------------------------------------- вердикт

def make_verdict(gates_ok, h1, r0_ok, h4, h3, h5):
    """ВЕРДИКТ (primary = H1 ретроспективный + валидация; FWER не применяется — объявлено)."""
    if not gates_ok:
        return {"status": "GATES_FAIL",
                "detail": "контроли (§0) не пройдены — числа непубликуемы; разобрать и перепустить"}
    if h1["ok"] and r0_ok:
        return {"status": "CONFIRMED",
                "detail": (f"H1: петля ⇒ стабильный repeat (f1 = {h1['f1']}/30 ≥ {H1_FLOOR}; r0 "
                           f"воспроизведена в 1e-12: 30/30, min {R0_MIN} code_14). H3: repeat "
                           f"necessary, не sufficient (доля loopy среди стабильных-repeat = "
                           f"{h3['loopy_share_among_stable']:.4f}; 6/11 undetected + 2 дубля). "
                           f"H4: h-линия на p следует за repeat'ом (ρ_s = "
                           f"{h4['spearman_q_maxrun']:.3f}, порог 0.4). H5: R(p) ≡ 0 — граница. "
                           f"Итог: петля = стабильный copy-paste ∧ сильная Δu-структура; период-p "
                           f"структура T0.3 и спектральная линия T0.6 — feedforward-ПОСЛЕДСТВИЯ "
                           f"token-repeat'а. K4: T0.7 НЕ влечёт T0.6 (линия в h ≠ линия в du; "
                           f"носители разные — r1); T0.6-результат (smoke) НЕ переписывается T0.7.")}
    if h1["ok"]:
        return {"status": "CONFIRMED_WITH_FLAG",
                "detail": (f"H1 формально PASS (f1 = {h1['f1']}/30 ≥ {H1_FLOOR}), но ретроспектив "
                           f"r0 (30/30, min {R0_MIN}) НЕ воспроизведён в 1e-12 — расхождение "
                           f"прогона с r0: разбор (r0 верифицирована r1/r2-пересчётом).")}
    return {"status": "DIVERGENCE_FROM_R0",
            "detail": (f"¬H1: f1 = {h1['f1']}/30 < {H1_FLOOR} — расхождение прогона с r0 "
                       f"(30/30, min {R0_MIN}) — разбор (r0 верифицирована r1/r2-пересчётом). "
                       f"K4: T0.7 НЕ влечёт T0.6.")}

