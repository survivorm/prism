#!/usr/bin/env python3
"""Сверка чисел `docs/paper.md` с артефактами (раунд «гигиена статьи», 2026-11-05).

Зачем. Статья цитирует числа из трёх источников истины: JSON-артефакты, закоммиченные
логи прогонов и манифесты. Этот скрипт берёт каждое такое число из источника, форматирует
его так, как оно напечатано в статье, и требует присутствия этой строки в `docs/paper.md`.
Отдельным блоком — трассируемость: каждое `results/...`, упомянутое в статье, обязано
существовать и быть в git или в `results/exp40plus_artifacts.sha256`.

Запуск: `cd experiments && ../.venv/bin/python exp_paper_check.py`
Негативная половина проверена подсаживанием ошибок (см. docs/audit_40plus.md §4 для
однотипного чекера серии 40+).
"""
from __future__ import annotations

import json  # noqa: E402
import re  # noqa: E402
import subprocess
import sys
from pathlib import Path

import statistics as st  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAPER = (ROOT / "docs" / "paper.md").read_text(encoding="utf-8")

ok = 0
fails: list[str] = []


def chk(cond: bool, msg: str) -> None:
    global ok
    if cond:
        ok += 1
    else:
        fails.append(msg)


def need(value: str, where: str) -> None:
    """Строка `value` обязана встречаться в статье (знак минус допускаем и ASCII, и «−»)."""
    forms = {value, value.replace("-", "−")}
    chk(any(f in PAPER for f in forms), f"{where}: в статье нет «{value}»")


def art(name: str):
    return json.loads((ROOT / "results" / f"{name}.json").read_text(encoding="utf-8"))


def log(name: str) -> str:
    return (ROOT / "results" / name).read_text(encoding="utf-8")


def f3(x: float) -> str:
    return f"{x:.3f}"


def f4(x: float) -> str:
    return f"{x:.4f}"


SECTIONS = {}
_now = "_преамбула"
for _ln in PAPER.splitlines():
    if _ln.startswith("## ") or _ln.startswith("### "):
        _now = _ln.lstrip("#").strip()
        SECTIONS.setdefault(_now, [])
    SECTIONS.setdefault(_now, []).append(_ln)
SEC_TXT = {k: "\n".join(v) for k, v in SECTIONS.items()}


def need_in(sec: str, s: str, label: str) -> None:
    """Наличие числа внутри раздела, заголовок которого содержит `sec`.

    need() ищет по всей статье, поэтому подмена числа в нужном месте не ловится, если то же
    число упоминается рядом как снятая историческая версия (ровно случай R9/R10: в §4.11 есть
    «до перегенерации было +0.017»).
    """
    hits = [v for k, v in SEC_TXT.items() if sec in k]
    chk(len(hits) >= 1, f"{label}: раздел «{sec}» не найден")
    if hits:
        chk(any(f in hits[0] for f in (s, s.replace("-", "−"))),
            f"{label}: в разделе «{sec}» нет «{s}» — вне раздела есть, а в самом разделе нет")


# ---------------------------------------------------------------- E-40.1: знак Δ_gain
a1 = art("E40_1_audit")
a2 = a1["A2_delta_gain_signs"]
chk(a2["all_negative_all_layers_all_D_4of4"] is True, "E-40.1: знаки Δ_gain не «все отрицательны»")
chk(a2["card_reproducible"] is False, "E-40.1: карточка вдруг стала воспроизводимой")
for lay in ("6", "22", "23"):   # именно эти три процитированы в статье
    need(f"{a2['artifact_sign_r16_d1'][lay]:.4f}", f"E-40.1 Δ_gain(L{lay})")
for lay, v in a2["card_values"].items():
    if lay != "23":  # карточное число в статье упоминается только как несуществующее
        need(f"{v:.3f}".rstrip("0").rstrip("."), f"E-40.1 значение карточки (L{lay})")

# ---------------------------------------------------------------- E-40.2: unstable
a2f = art("E40_2_double_fit")
chk(a2f["outcome"] == "unstable", "E-40.2: outcome поехал")
need("1.19e-07", "E-40.2 якорь E-31")
chk(abs(a2f["raw_anchor_worst_dev"] - 1.19e-07) < 1e-9, "E-40.2: отклонение якоря не 1.19e-07")
need("17.10", "E-40.2 ΔAICc L18 C_top16")
chk("stretched" not in json.dumps(a2f.get("stretched_beta_C_F", {})), "E-40.2: stretched вдруг выбрана")

# ---------------------------------------------------------------- E-40.2c: срез позиции
ps = art("E40_2c_pos_slice")
bp = ps["per_layer"]["6"]["by_position"]
rng = [v["K128"]["r_eff_KF"] for t, v in bp.items() if t != "127"]   # t=127 — нулевой контекст, отдельно
need(f"L6 {min(rng):.1f}–{max(rng):.1f}", "E-40.2c размах r_eff L6 (без t=127)")
t127 = [ps["per_layer"][L]["by_position"]["127"]["K128"]["r_eff_KF"] for L in ("6", "12", "18", "22", "23")]
need("/".join(f"{x:.1f}" for x in t127), "E-40.2c r_eff при t=127")

# ---------------------------------------------------------------- E-40.3: ковариата
a3 = art("E40_3_fft")
chk(a3["n_pass_3sigma"] == 1 and a3["n_linear_pass_3sigma"] == 3, "E-40.3: счётчики пиков поехали")
chk(a3["verdict"] == "local_filter", "E-40.3: вердикт поехал")
blob3 = json.dumps(a3)
for s in ("0.29", "0.74"):
    chk(s in blob3 or s in PAPER, f"E-40.3: ковариата {s} не прослеживается")
need("0.41…0.98", "E-40.3 p в log-сетке")

# ---------------------------------------------------------------- E-40.4: r_cross
a4 = art("E40_4_intersection")
for lay, want in (("18", 0.41), ("22", 0.569)):
    v = a4["per_layer"][lay]
    chk(abs(v["frac_boot_no_crossing"] - want) < 5e-3, f"E-40.4 L{lay}: frac_boot_no_crossing поехал")
    need(str(int(v["r_cross"])), f"E-40.4 r_cross L{lay}")
    need(f"{v['delta_peak']:.3f}", f"E-40.4 пик L{lay}")
    need(f"{v['null_peak_p95']:.3f}", f"E-40.4 null p95 L{lay}")
chk(a4["prediction_check"]["verdict"] == "miss", "E-40.4: ставка вдруг стала попаданием")

# ---------------------------------------------------------------- E-40.5: кинетика
a5 = art("E40_5_kinetic")
need(f"{max(a5['per_layer'][L]['gram_closure_worst_rel'] for L in a5['per_layer']):.1e}".replace("-15", "-15"),
     "E-40.5 замыкание Gram")
offs = "/".join(f"{a5['per_layer'][L]['E_off_over_dJ']:.2f}" for L in ("6", "12", "18", "22", "23"))
need(offs, "E-40.5 доли E_off_manifold")
need("0.0003/0.002/0.044/0.234", "E-40.5 доли linearization_valid")
rb = a5["per_layer"]["23"]["rank_block_energy_5x5"]
chk(rb["rank"] == 2 and abs(rb["PR"] - 2.38) < 0.01 and abs(rb["rho_hat"] - 0.42) < 0.01,
    "E-40.5: блок 5×5 на L23 поехал")
chk(abs(a5["per_layer"]["23"]["eps3_over_dJ_valid"] + 3e-4) < 5e-5, "E-40.5: eps3 на L23 поехал")
need("−3e-4", "E-40.5 eps3 на L23")
need("1.0 нестабильных пар", "E-40.5 левый канал")
chk(a5["per_layer"]["23"]["left_channel_threshold"]["unstable_left_frac_sampled"] == 1.0,
    "E-40.5: доля нестабильных левых пар поехала")

# ---------------------------------------------------------------- E-40.6: R_can/R_raw
a6 = art("E40_6_R_profile")
lay = ("6", "12", "18", "22", "23")
need("/".join(f4(a6["per_layer"][L]["R_can_corpus"]) for L in lay), "E-40.6 R_can")
need("/".join(f4(a6["per_layer"][L]["R_raw_corpus"]) for L in lay), "E-40.6 R_raw")
diff = {L: abs(a6["per_layer"][L]["R_can_corpus"] - a6["per_layer"][L]["R_can_nogauge"]) for L in lay}
chk(max(diff[L] for L in ("6", "12", "18", "22")) <= 5e-5, f"E-40.6: расхождение на L6–L22 поехало: {diff}")
chk(abs(diff["23"] - 1.8e-3) < 1e-4, f"E-40.6: расхождение на L23 не 1.8e-3: {diff['23']}")
need("до 5e-5 на L6–L22", "E-40.6 оговорка про R_can_nogauge")

# ---------------------------------------------------------------- E-40.7: G(l)
a7 = art("E40_7_G_fixation")
need("/".join(f4(a7["per_layer"][L]["G"]) for L in lay), "E-40.7 профиль G")
gv = [a7["per_layer"][L]["G"] for L in lay]
chk(all(x < y for x, y in zip(gv, gv[1:])), "E-40.7: монотонность G по глубине нарушена")

# ---------------------------------------------------------------- E-40.8: 6×6
a8 = art("E40_8_synthesis")
need("/".join(f3(x) for x in a8["eigenvalues"]), "E-40.8 собственные значения")
need("/".join(f3(x) for x in a8["permutation_null_thresholds_p95"]), "E-40.8 перцентили нуля")
r8 = a8["rank_estimate"]
chk(r8["rank"] == 1 and r8["bootstrap_CI_rank"] == [1.0, 2.0], "E-40.8: ранг поехал")
need(f"PR {r8['PR']:.2f}", "E-40.8 PR")
need(f"CI [{r8['bootstrap_CI_PR'][0]:.2f}, {r8['bootstrap_CI_PR'][1]:.2f}]", "E-40.8 CI по PR")
chk(a8["R1_R2_R3_outcome"] == "R3", "E-40.8: вердикт R3 поехал")
need(f3(a8["inter_cluster_mean_abs_rho"]), "E-40.8 межкластерный |ρ|")

# ---------------------------------------------------------------- E-3: тождество и null'ы
e3 = art("exp3_ablation")
import statistics as _st  # noqa: E402

for kind in ("with_correction", "without_correction", "random_orthogonal"):
    v = list(e3[kind].values())
    chk(len(v) == 24, f"E-3 {kind}: ожидалось 24 слоя, получено {len(v)}")
need(f"{min(e3['without_correction'].values()):.3f}–{max(e3['without_correction'].values()):.3f}",
     "E-3 разброс без коррекции")
_ro = list(e3["random_orthogonal"].values())
need(f"{_st.mean(_ro):.3f} ± {_st.pstdev(_ro):.3f}", "E-3 null «ортогональная R»")
chk(min(_ro) > 0.95 and max(_ro) < 1.05, "E-3: null ортогонального отображения ушёл от единицы")
chk(max(abs(x) for x in e3["with_correction"].values()) < 1e-6, "E-3: тождественный проход не схлопнулся")

# ---------------------------------------------------------------- E-45: ущерб
d45 = art("exp45_damage_vs_structure")
need(f"{d45['anchors']['kappa_base']:.6f}", "E-45 κ₀")
ed = {x["name"]: x for x in d45["edits"]}
dense_rand = [ed[k]["ratio_A"] for k in ("gptq4", "rtn8", "rand_full_d0.06", "rand_full_d0.12", "rand_full_d0.25")]
sn = [v["ratio_A"] for k, v in ed.items() if k.startswith("seednull")]
need(f"[{min(dense_rand):.4f}, {max(dense_rand):.4f}]", "E-45 полоса квантизации/случайных правок")
need(f"[{min(sn):.4f}, {max(sn):.4f}]", "E-45 iid-полоса одного замера")
klb = sorted(x["kl_be_B"] for x in d45["edits"] if x.get("kl_be_B"))
need(f"{klb[0]:.1e}", "E-45 нижняя KL_B")
need(f"{klb[-1]:.2f}", "E-45 верхняя KL_B")
need(f"{max(d45['ppl_delta_pct'].values()):+.0f} %", "E-45 верхний ΔPPL")
for k, want in (("flat_l0.01", 0.325), ("flat_l0.1", 0.211), ("flat_noleash", 0.129)):
    need(f"{ed[k]['ratio_A']:.3f}", f"E-45 уплощение {k}")
    chk(abs(ed[k]["ratio_A"] - want) < 5e-4, f"E-45: {k} не {want}")

# ---------------------------------------------------------------- E-45b / E-46: память
l45b = log("exp45b_analysis.log")
row_vse = [l for l in l45b.splitlines() if l.strip().startswith("ВСЕ")][0].split()
need(row_vse[4], "E-45b g_uni на свежем тексте")           # 0.1409
chk("0.1403" in PAPER, "E-45b: перенос g_uni на кэш exp27b (0.1403) не в статье")
l46 = log("exp46_verdicts.log")
rows46: dict[str, list[str]] = {}
for w in l46.splitlines():
    parts = w.split()
    if parts and parts[0] in ("leaf_lag1", "pca_doc", "pca_doc_raw") and parts[0] not in rows46:
        rows46[parts[0]] = parts   # берём ПЕРВОЕ вхождение: таблица медиан, а не лестница лагов
need(rows46["leaf_lag1"][2], "E-46 score(лист)")
need(rows46["pca_doc"][2], "E-46 score(pca_doc)")
need(rows46["pca_doc_raw"][2], "E-46 score(pca_doc_raw)")
for pat, printed, where in ((r"leaf_lag1 − haar\s*: Δ = [-−](0\.344)", None, "E-46 Δ против Haar (KL)"),
                            (r"leaf_lag1 − haar\s*: Δ = \+(0\.266)", None, "E-46 Δ против Haar (KL_nr)"),
                            (r"ИТОГ: (носителя нет: как минимум один null не хуже листа)",
                             "носителя нет: как минимум один null не хуже листа", "E-46 итог")):
    m = re.search(pat, l46)
    chk(m is not None, f"{where}: нет в логе-источнике")
    if m:
        need(printed or m.group(1), where)

# ---------------------------------------------------------------- E-47/E-48: носитель
v47 = art("exp47a_kinematics_cache")
need(f"{v47['verdicts']['K0.1']['r_s_a_median']:.2f}", "E-47 K0.1")
chk(v47["verdicts"]["K0.2"]["max_rel_J_vs_JR"] < 3e-7, "E-47: route-дрейф стал больше 2.7e-07")
need("2.7e-07", "E-47 route-дрейф")
chk(v47["verdicts"]["K0.2"]["scale_floor_max_rad"] < 1.5e-6, "E-47: пол поехал")
need("1.4e-06", "E-47 численный пол")
# a0 и A статьи — фит по бинованным средним (docs/leaf_dynamics.md §8.1); в артефакте лежат
# сами бинованные средние, поэтому фит пересчитывается здесь, а не верится на слово
import numpy as np  # noqa: E402  (уже в зависимостях репозитория)

ae = np.asarray(v47["descriptive"]["a_edges"], dtype=float)
sb = np.asarray(v47["descriptive"]["mean_s_by_abin"], dtype=float)
x = 0.5 * (ae[1:] + ae[:-1])
grid = np.linspace(0.3, 1.5, 2401)
best = None
for a0 in grid:
    t = np.tanh(x / a0)
    A = float((sb * t).sum() / (t * t).sum())
    r2 = 1.0 - float(((sb - A * t) ** 2).sum() / ((sb - sb.mean()) ** 2).sum())
    if best is None or r2 > best[2]:
        best = (a0, A, r2)
chk(0.5 <= best[0] <= 1.0, f"E-47: пересчитанное a0 = {best[0]:.3f} вне здравого диапазона")
chk(abs(best[1] - 0.383) < 0.05, f"E-47: пересчитанное A = {best[1]:.3f}, статья говорит 0.383")
print(f"  [diag] E-47 a0 по бинам пересчитано как {best[0]:.3f} (A = {best[1]:.3f}, R² = {best[2]:.3f}): "
      "статья цитирует фит по бинованным средним (0.715), точного совпадения не ожидается")
need("0.383·tanh(a/0.715)", "E-47 фит насыщения")
v47b = art("exp47b_gen_kinematics")["verdicts"]
need(f3(v47b["K1.1"]["median_r_sa"]), "E-47 K1.1")
chk(v47b["K1.2"]["frac"] == 0.89, "E-47: доля документов с AC≥3 поехала")
v47c = art("exp47c_p2_loop_prediction")
for model in ("qwen", "lfm3"):
    vd = v47c[model]["verdict"]
    need(f"{vd['delta_auc_all_vs_triv']:+.3f}", f"E-47c ΔAUC {model}")
    if model == "lfm3":  # статья печатает тройку AUC только для LFM3 (там она и есть предмет)
        for key in ("AUC_leaf_d16", "AUC_triv_d16", "AUC_all_d16"):
            need_in("4.11", f"{vd[key]:.3f}", f"E-47c {key} {model}")
        need_in("4.11", f"{vd['delta_auc_all_vs_triv']:+.3f}", "E-47c ΔAUC lfm3 (в §4.11)")
        # Первичная строка сравнения двух носителей — с контекстом: одно и то же число
        # встречается в разделе дважды (вывод + «до перегенерации было»), и без контекста
        # подмена в первичной строке не ловится (проверено подменой +0.088 → +0.017).
        pair = (f"{v47c['qwen']['verdict']['delta_auc_all_vs_triv']:+.3f}** (Qwen) и "
                f"**{vd['delta_auc_all_vs_triv']:+.3f}")
        need_in("4.11", pair, "E-47c первичная строка ΔAUC (Qwen и LFM3 рядом)")
    # вердикт обязан совпадать с тремя ногами, иначе статья перескажет его неверно
    legs = vd["legs"]
    chk(vd["verdict"].startswith("ЛИСТ") == all(legs.values()),
        f"E-47c {model}: вердикт не соответствует ногам критерия {legs}")
    if model == "lfm3":  # R10: triv-фичи из того же npz, внешний пересчёт не использован
        chk(vd["triv_feature_source"] == ["full"], f"E-47c lfm3: triv-фичи не из npz: {vd['triv_feature_source']}")
        chk(legs["delta_auc_ge_0.05"] and not legs["AUC_leaf_ge_0.8"],
            "E-47c lfm3: статья объясняет отрицательный вердикт ногой AUC_leaf, а ноги поменялись")
b48 = art("exp48a_lfm3_kinematics")
need(f3(b48["verdicts"]["T1"]["a0"]), "E-48a a0")
need(f"{st.median([r['p_med'] for r in b48['descriptive']['per_doc']]):.4f}", "E-48a p_med")
chk(b48["verdicts"]["T0.2"]["gap16"] == 1.0, "E-48a: gap16 поехал")
gen48 = art("exp48b_lfm3_gen_kinematics")
rows = gen48["docs"]

for dom in ("wiki", "math", "code", "chat"):
    sel = [r for r in rows if r["dom"] == dom]
    need(f"{float(np.mean([r['n_loops'] for r in sel])):.1f}", f"E-48b петель/док {dom}")

f = lambda xs: [x for x in xs if isinstance(x, (int, float))]
loop = [r for r in rows if r["n_loops"] > 0]
pk = lambda r: f(r["in_loop"].get("acf_peak_by_loop") or [])
n_acf = sum(1 for r in loop if pk(r) and max(pk(r)) >= 3)
need_in("4.11", f"{n_acf}/{len(loop)}", f"E-48b петлевые документы с AC(s) ≥ 3 ({n_acf}/{len(loop)})")
chk(n_acf / len(loop) < 0.89, "E-48b: периодичность петли у LFM3 перестала быть слабее Qwen (24/27)")
need(f"{round(100 * n_acf / len(loop))} %", f"E-48b доля AC(s) ≥ 3 в процентах ({round(100 * n_acf / len(loop))} %)")
chk("decoding_policy_note" in gen48["params"] and "gain w" in gen48["params"]["decoding_policy_note"],
    "E-48b: в артефакте нет decoding_policy_note про гейн w — это прежняя генерация, R9 не применён")

# ---------------------------------------------------------------- E-39-дополнение (§2.5)
def floats(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from floats(v)
    elif isinstance(o, list):
        for v in o:
            yield from floats(v)
    elif isinstance(o, (int, float)):
        yield float(o)


pr = {}
for name in ("exp39_supp_pscale_probe.json", "exp39_supp_sentinel_probe.json"):
    pr.update(json.loads((ROOT / "results" / name).read_text(encoding="utf-8")))
nums = list(floats(pr))
# статья печатает эти величины с округлением до 3 знаков, поэтому проверяем пару: артефакт
# (4 знака, допуск 5e-5) и строка статьи (то округление, которым оно в ней напечатано)
for src, printed in (("0.4896", "0.490"), ("0.4759", "0.476"), ("0.0050", "0.005"),
                     ("0.9952", "0.995"), ("0.4686", "0.4686"), ("0.2380", "0.2380"),
                     ("0.443", "0.443"), ("0.5398", "0.540"), ("0.5193", "0.519"),
                     ("0.4563", "0.456")):
    want = float(src)
    chk(any(abs(x - want) <= 5e-5 for x in nums), f"источник E-39-дополнения: нет {src} в двух probe-артефактах")
    need(printed, f"E-39-дополнение {src} → напечатано как {printed}")
supp = art("exp39_supp_three_objects")
rd = [supp["layers"][str(L)]["nulls_same_slice"]["real_over_direction"] for L in (6, 12, 18, 22, 23)]
need(f"{min(rd):.3f}–{max(rd):.3f}", "E-39-дополнение real/dir")

# ---------------------------------------------------------------- трассируемость ссылок
tracked = set(subprocess.run(["git", "ls-files", "results"], cwd=ROOT, capture_output=True, text=True)
              .stdout.split())
def _is_hex(x: str) -> bool:
    return len(x) == 64 and all(c in "0123456789abcdef" for c in x)


manifest = {ln.split()[1].split("/")[-1]
            for ln in (ROOT / "results/exp40plus_artifacts.sha256").read_text(encoding="utf-8").splitlines()
            if len(ln.split()) == 2 and _is_hex(ln.split()[0])}
for ref in sorted(set(re.findall(r"results/([0-9A-Za-z_.\-]+\.(?:json|json\.gz|npz|log|pt|sha256))", PAPER))):
    p = ROOT / "results" / ref
    chk(p.exists(), f"статья ссылается на отсутствующий results/{ref}")
    chk(f"results/{ref}" in tracked or ref in manifest,
        f"results/{ref} не в git и не в манифесте → читатель не сможет сверить")

# ---------------------------------------------------------------- итог
print(f"проверок пройдено: {ok}, провалено: {len(fails)}")
for f in fails[:25]:
    print("  ✗", f)
sys.exit(1 if fails else 0)
