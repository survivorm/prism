#!/usr/bin/env python
"""Самоаудит серии E-40+: числа в docs против артефактов.

Зачем. Раунд аудита 40+ (`docs/audit_40plus.md`) нашёл четыре класса дефектов, которые
возникают молча:

1. doc цитирует числа ПРЕЖНЕГО (или УПАВШЕГО) прогона: артефакт пересоздан, doc нет;
2. знак величины переписан «по аналогии» с соседней строкой (ΔAUC LFM3 = +0.017 стал «−0.017»);
3. вывод об отсутствии признака («eq_prev_rate = 0 даже в петлевых документах») не сверен
   с колонкой артефакта (у двух wiki-документов коллапса там 0.938 и 0.927);
4. doc ссылается на артефакт, которого нет в git.

Каждый класс закрыт проверкой, а не наблюдением: скрипт падает, если число в docs не совпало
с артефактом, или если артифакт из ссылки не закоммичен.

Запуск:  cd experiments && ../.venv/bin/python exp40plus_doc_check.py     (CPU, ~15 с)

Договор. Для E-45b источником считаем закоммиченный лог `exp45b_analysis.log` (он пересоздаётся
детерминированно, seed 0, 40 с): doc обязан ЦИТИРОВАТЬ его строки. Для E-46 — `exp46_verdicts.log`.
Для остального — JSON-артефакты. Числа сравниваются по значению с допуском в один печатный
разряд, а не по «кто что округлил».
"""
from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
R = os.path.join(ROOT, "results")
D = os.path.join(ROOT, "docs")

FAIL: list[str] = []
NOK = 0


def doc(name: str) -> str:
    with open(os.path.join(D, name), encoding="utf-8") as f:
        return f.read()


def art_json(name: str):
    p = os.path.join(R, name)
    if name.endswith(".gz"):
        with gzip.open(p, "rt") as f:
            return json.load(f)
    with open(p) as f:
        return json.load(f)


def art_log(name: str) -> str:
    with open(os.path.join(R, name), encoding="utf-8") as f:
        return f.read()


def chk(cond: bool, what: str) -> None:
    global NOK
    if cond:
        NOK += 1
    else:
        FAIL.append(what)


def need(hay: str, needle: str, where: str) -> None:
    chk(needle in hay, f"{where}: нет строки {needle!r}")


def medf(rows, key, pred=None) -> float:
    vals = []
    for r in rows:
        if pred is not None and not pred(r):
            continue
        v = r.get(key)
        if v is not None and np.isfinite(float(v)):
            vals.append(float(v))
    return float(np.median(vals)) if vals else float("nan")


# ------------------------------------------------------- E-45b: §1 и §3 обязаны цитировать лог
def check_e45b_quotes_log() -> None:
    d = doc("E45b_leaf_memory.md")
    log = art_log("exp45b_analysis.log")
    w = "E45b ↔ exp45b_analysis.log"
    chk("Traceback" not in log and "rc(" not in log,
        f"{w}: закоммиченный лог содержит падение — doc не может называть его прогоном")

    # таблица §1: колонки, которые doc обязан цитировать (порядок колонок в логе фиксирован
    # заголовком «домен d1_r16 k2_r16 pr_sigma g_uni j_norm h_norm nll ent copy16 align_*»),
    # и формат, которым они напечатаны в docs
    allrow = re.search(r"^\s*ВСЕ\s+(.*)$", log, re.M)
    chk(allrow is not None, f"{w}: в логе нет строки «ВСЕ»")
    if allrow:
        vals = [float(x) for x in re.findall(r"-?\d+\.\d+", allrow.group(1))]
        chk(len(vals) == 12, f"{w}: в строке «ВСЕ» {len(vals)} колонок, ждали 12")
        for idx, fmt, lbl in ((0, ".4f", "d1_r16"), (1, ".4f", "k2_r16"), (2, ".2f", "PR(σ²)"),
                              (3, ".4f", "g_uni"), (4, ".2f", "‖J‖"), (5, ".1f", "‖h‖"),
                              (10, ".4f", "align_rand")):
            if idx < len(vals):
                need(d, format(vals[idx], fmt), f"{w}: {lbl} = {format(vals[idx], fmt)}")
        need(d, "0.0155", f"{w}: k2_r16 — docs обязан иметь медиану: np.median от колонки с 32 NaN давал nan")

    # §3: нули из строк лога (теория / измеренный Haar / align_rand / align_pca_rand)
    for pat, lbl in ((r"измеренный align_haar = ([\d.]+)", "align_haar"),
                     (r"align_pca_rand = ([\d.]+)", "align_pca_rand"),
                     (r"= ×(\d\.\d+) от Haar", "отношение align_rand/Haar")):
        m = re.search(pat, log)
        chk(m is not None, f"{w}: в логе нет величины {lbl}")
        if m:
            val = m.group(1)
            need(d, val if "×" not in lbl else "×" + val, f"{w}: {lbl} = {val} нет в docs")

    for pair in ("d1_r16 → ent", "align_next → dnll", "h_norm → copy16",
                 "j_norm → copy16", "k2_r16 → ent", "d1_r16 → copy16"):
        m = re.search(re.escape(pair) + r"\s*: ρ = (\S+)\s+z =\s*(\S+)", log)
        chk(m is not None, f"{w}: пары {pair} нет в логе")
        if not m:
            continue
        rho, z = m.group(1), m.group(2)
        # в docs знак печатается как «−», а величины вроде 0.118 стоят в тексте без знака:
        # принимаем либо точную форму со знаком, либо модуль
        def present(v: str) -> bool:
            forms = {v, v.replace("-", "−"), v.lstrip("+-")}
            return any(f in d for f in forms)

        chk(present(rho), f"{w}: {pair}: ρ={rho} нет в docs")
        if pair != "d1_r16 → copy16":   # эта пара в docs только в тексте, без z
            chk(present(z), f"{w}: {pair}: z={z} нет в docs")
    need(log, "|z| > 3: 26", f"{w}: 26 пар из 54")
    need(d, "26", f"{w}: число «26» пар должно быть в docs")
    m = re.search(r"Spearman\(κ_локальная, ‖h‖\) = (\S+) \(блочный z = (\S+)", log)
    if m:
        need(d, m.group(1).replace("-", "−"), f"{w}: ρ(κ,‖h‖) в docs")

    # §4 (E-45b-часть): score'ы и CI
    need(log, "score(лист) медианой по позициям = 1.134, score(PCA) = 2.040", w)
    need(d, "1.134", f"{w}: score(лист)")
    need(d, "2.040", f"{w}: score(PCA)")
    m = re.search(r"разница по документам = (\S+), bootstrap-CI \[([^,]+), ([^\]]+)\]", log)
    chk(m is not None, f"{w}: нет строки разницы по документам")
    if m:
        for val in (m.group(1), m.group(2).strip(), m.group(3).strip()):
            need(d, val.replace("-", "−"), f"{w}: {val} (из лога) нет в docs")


# ------------------------------------------------------- E-45b §1/§2: артефакт-таблица
def check_e45b_artifact() -> None:
    d = doc("E45b_leaf_memory.md")
    art = art_json("exp45b_leaf_table.json.gz")
    rows, meta = art["rows"], art["meta"]
    w = "E45b ↔ exp45b_leaf_table.json.gz"
    chk(meta["n_rows"] == len(rows) == 4096, f"{w}: число строк не 4096")
    need(d, str(meta["n_with_target"]), f"{w}: позиций с известной целью")
    need(d, f"{meta['align_haar_theory']:.4f}", f"{w}: теория Haar √(r/D)")
    gap = np.array([float(r["gap_r16"]) for r in rows])
    med_gap = float(np.median(gap))
    need(d, f"{med_gap:.4f}", f"{w}: медиана зазора")
    need(d, f"{np.mean(gap < 0.01) * 100:.1f} %", f"{w}: доля позиций с зазором < 1 %")
    chk(abs(float(np.percentile(gap, 10)) - 0.0085) < 5e-5, f"{w}: p10 зазора не 0.0085")
    chk(abs(float(np.percentile(gap, 90)) - 0.0091) < 5e-5, f"{w}: p90 зазора не 0.0091")
    # «×8722» — отношение медианы зазора к шуму 1e-6 (GAP_NOISE_FLOOR в exp45b_analysis.py)
    need(d, f"×{round(med_gap / 1e-6)}", f"{w}: отношение медианы зазора к полу 1e-6")
    chk("под численным полом" not in d,
        f"{w}: в docs остался перевёрнутый смысл отношения («×… под численным полом»)")


# ------------------------------------------------------- E-46 §4 против verdicts-лога
E46_ROWS = ("leaf_lag0", "leaf_lag1", "leaf_lag8", "haar", "pca_doc", "pca_doc_raw",
            "pca_win", "mean_dir", "leaf_perm")


def check_e46() -> None:
    d = doc("E45b_leaf_memory.md")
    log = art_log("exp46_verdicts.log")
    w = "E46 ↔ exp46_verdicts.log"
    tab = {}
    for ln in log.splitlines():
        parts = ln.split()
        if len(parts) == 8 and parts[0] in E46_ROWS:
            tab[parts[0]] = [float(x) for x in parts[1:]]
    chk(len(tab) == len(E46_ROWS), f"{w}: в логе {len(tab)} строк таблицы из {len(E46_ROWS)}")
    for name in E46_ROWS:
        if name not in tab:
            continue
        kl, _sc, kl_nr, _snr, energy, _top5, dnll = tab[name]
        m = re.search(rf"^\|\s*`?{re.escape(name)}`?[^|]*\|(.*)$", d, re.M)
        chk(m is not None, f"{w}: в таблице docs нет строки {name}")
        if not m:
            continue
        cells = []
        for c in m.group(1).split("|"):
            c = c.strip().strip("*").strip()
            if re.fullmatch(r"[\d.]+", c):
                cells.append(float(c))
        for want, label in ((kl, "KL"), (kl_nr, "KL_nr"), (energy, "энергия"), (dnll, "ΔNLL")):
            # docs печатает 3 знака, лог — 4 ⇒ погрешность округления до 5e-4
            chk(any(abs(c - want) <= 6e-4 for c in cells),
                f"{w}: {name}/{label} — в docs нет {want}, строка {cells}")
    for label, a in (("pca_doc", "leaf_lag1 − pca_doc"), ("pca_doc_raw", "leaf_lag1 − pca_doc_raw"),
                     ("pca_win", "leaf_lag1 − pca_win"), ("haar", "leaf_lag1 − haar")):
        m = re.search(re.escape(a) + r"\s*: Δ = (\S+) CI \[([^,]+), ([^\]]+)\]", log)
        chk(m is not None, f"{w}: нет bootstrap-строки {a}")
        if not m:
            continue
        for val in (m.group(1), m.group(2).strip(), m.group(3).strip()):
            need(d, val.replace("-", "−"), f"{w}: {label}: {val} из лога нет в docs")


# ------------------------------------------------------- E-47 / E-48 против json
def check_e47_e48() -> None:
    d = doc("leaf_dynamics.md")
    w = "E-47/E-48 ↔ leaf_dynamics.md"

    c = art_json("exp47c_p2_loop_prediction.json")
    for model, name in (("qwen", "Qwen"), ("lfm3", "LFM3")):
        v = c[model]["verdict"]
        m = re.search(rf"^\|\s*{name}\s*\|(.*)$", d, re.M)
        chk(m is not None, f"{w}: нет строки {name} в таблице P2")
        if not m:
            continue
        cells = [x.strip().replace("−", "-").strip("*") for x in m.group(1).split("|") if x.strip()]
        want = [v['AUC_leaf_d16'], v['AUC_triv_d16'], v['AUC_all_d16'], v['delta_auc_all_vs_triv']]
        for i, lbl in enumerate(("AUC leaf", "AUC triv", "AUC all", "ΔAUC")):
            got = cells[i] if i < len(cells) else "<нет ячейки>"
            try:
                ok = abs(float(got.replace("−", "-")) - want[i]) < 1e-9
            except ValueError:
                ok = False
            chk(ok, f"{w}: {name}/{lbl} — docs {got!r}, артефакт {want[i]:+.3f}")
        # обратного знака нигде в docs быть не должно (именно так и появился прежний «−0.017»:
        # знак у LFM3 переписали «по аналогии» с Qwen, где он и правда отрицательный)
        d_auc = v["delta_auc_all_vs_triv"]
        if abs(d_auc) > 1e-9:
            opp = f"−{abs(d_auc):.3f}" if d_auc > 0 else f"+{abs(d_auc):.3f}"
            chk(re.search(r"(?<![\d.])" + re.escape(opp) + r"(?![\d])", d) is None,
                f"{w}: {name}: в docs встречается противоположный знак ΔAUC ({opp})")

    b = art_json("exp47b_gen_kinematics.json")
    brows = b["docs"]
    loop = [r for r in brows if r["in_loop"] and r["in_loop"]["a"] is not None]
    zero = [r for r in loop if r["trivial"]["eq_prev_rate"] == 0]
    need(d, f"{len(zero)} из {len(loop)}", f"{w}: число петлевых документов с eq_prev=0 ({len(zero)}/{len(loop)})")
    # обязательны к упоминанию те петлевые документы, где признак НЕ ноль и велик (коллапс в
    # соседние дубли) — именно про них раньше было написано «= 0 даже в петлевых документах»
    for r in loop:
        e = r["trivial"]["eq_prev_rate"]
        if e >= 0.05:
            need(d, f"{e:.3f}", f"{w}: eq_prev {r['dom']} s{r['si']} = {e:.3f} не упомянут в docs")
    need(d, f"{medf(loop, 'a_med'):.3f}", f"{w}: a по петлевым документам целиком")
    need(d, f"{medf(loop, 's_med'):.3f}", f"{w}: s по петлевым документам целиком")
    for dom in ("wiki", "math", "code", "chat"):
        n = sum(1 for r in brows if r["dom"] == dom and r["n_loops"] > 0)
        need(d, f"{n}/16", f"{w}: петлевых документов {dom}")
    for key, val in (("median_r_sa", b["verdicts"]["K1.1"]["median_r_sa"]),
                     ("p_in", b["verdicts"]["K1.3"]["p_in"]),
                     ("p_out", b["verdicts"]["K1.3"]["p_out"])):
        need(d, f"{val:.3f}", f"{w}: K1 {key}")
    need(d, str(b["verdicts"]["K1.2"]["loop_docs"]), f"{w}: K1.2 loop_docs")
    need(d, str(b["verdicts"]["K1.2"]["acf_ge3"]), f"{w}: K1.2 acf_ge3")

    a = art_json("exp48a_lfm3_kinematics.json")
    v = a["verdicts"]
    pb = a["descriptive"]["per_doc"]
    for val, fmt, lbl in ((v["T1"]["a0"], ".3f", "T1 a0"), (v["T1"]["A"], ".2f", "T1 A"),
                          (v["T2"]["p_med"], ".4f", "T2 p_med"), (v["T3"]["ratio"], ".2f", "T3 ratio")):
        need(d, format(val, fmt), f"E-48a: {lbl}")
    need(d, format(v["T0.2"]["floor"], ".1e").replace("e-0", "e-"), "E-48a: T0.2 floor")
    for dom in ("wiki", "math", "code", "chat"):
        need(d, f"{medf(pb, 'a_med', lambda r, dom=dom: r['dom'] == dom):.2f}", f"E-48a: a({dom})")

    g = art_json("exp48b_lfm3_gen_kinematics.json")
    gd = g["docs"]
    # §9.5: таблица доменов есть и в §8.1 (prompt-режим, CI), поэтому ищем только после заголовка 9.5
    m95 = re.search(r"^### 9\.5\..*?(?=^### |^## )", d, re.M | re.S)
    chk(m95 is not None, "E-48b: в leaf_dynamics.md нет секции 9.5")
    sec95 = m95.group(0) if m95 else d
    for dom in ("wiki", "math", "code", "chat"):
        sel = [r for r in gd if r["dom"] == dom]
        m = re.search(rf"^\|\s*{dom}\s*\|(.*)$", sec95, re.M)
        chk(m is not None, f"E-48b: нет строки {dom} в §9.5")
        if not m:
            continue
        cells = [x.strip() for x in m.group(1).split("|") if x.strip()]
        row_txt = m.group(1)
        ent = medf([{"e": r["trivial"]["ent_med"]} for r in sel], "e")
        want = [f"{medf(sel, 'a_med'):.3f}", f"{medf(sel, 's_med'):.3f}", f"{medf(sel, 'p_med'):.3f}",
                f"{medf(sel, 'g_med'):.1f}", f"{ent:.2f}"]
        for val, lbl in zip(want, ("a", "s", "p", "g", "ent")):
            # сравнение числами, не строками: «1.0» и «1.000» — одно и то же число,
            # а строковое сравнение ломалось на формате ячейки g
            ok = any(c.replace(",", ".").strip("≈~ ").replace("−", "-") == val
                     or _close(c, val) for c in cells)
            chk(ok, f"E-48b: {dom}/{lbl} — в docs нет {val}, строка {cells}")
        # петли в docs стоят одной ячейкой: «15.2/док, 11/16»
        loops_mean = f"{float(np.mean([r['n_loops'] for r in sel])):.1f}/док"
        n_loopdocs = f"{sum(1 for r in sel if r['n_loops'] > 0)}/16"
        need(row_txt, loops_mean, f"E-48b: {dom} петель/док")
        need(row_txt, n_loopdocs, f"E-48b: {dom} документов с петлями")


def _close(cell: str, want: str) -> bool:
    """Ячейка таблицы ≡ числу `want` с точностью до половины последнего печатного знака."""
    try:
        c, w = float(cell), float(want)
    except ValueError:
        return False
    dec = len(want.split(".")[1]) if "." in want else 0
    return abs(c - w) <= 0.5 * 10 ** (-dec) + 1e-12


# ------------------------------------------------------- трассируемость артефактов серии
def check_artifacts_tracked() -> None:
    where = "трассируемость"
    manifest = ""
    mp = os.path.join(R, "exp40plus_artifacts.sha256")
    if os.path.exists(mp):
        manifest = open(mp, encoding="utf-8").read()
    refs = set()
    for f in sorted(os.listdir(D)):
        if f.endswith(".md"):
            with open(os.path.join(D, f), encoding="utf-8") as h:
                refs |= set(re.findall(r"results/(exp4[0-9][\w./-]*?\.(?:json\.gz|json|npz|pt|log))", h.read()))
    chk(len(refs) > 10, f"{where}: ссылок на артефакты 40+ найдено {len(refs)} — подозрительно мало")
    for rel in sorted(refs):
        p = os.path.join(R, rel)
        if not os.path.exists(p):
            FAIL.append(f"{where}: docs ссылается на results/{rel}, файла нет")
            continue
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", f"results/{rel}"],
                                 cwd=ROOT, capture_output=True).returncode == 0
        chk(tracked or rel in manifest,
            f"{where}: results/{rel} — на него есть ссылки, его нет ни в git, ни в манифесте")


def main() -> int:
    check_e45b_quotes_log()
    check_e45b_artifact()
    check_e46()
    check_e47_e48()
    check_artifacts_tracked()
    print(f"проверок пройдено: {NOK}, провалено: {len(FAIL)}")
    for f in FAIL:
        print("  ✗", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
