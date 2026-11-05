"""Приёмочный тест E-40: каждое заявленное число обязано лежать в артефакте.

Зачем. В раунде 2026-11-03 в `docs/E40_tier0.md` оказались блоки про `s_same`,
чувствительность к p-срезу и метрику убийства `P(p)` — объектов, которых в скрипте и в
`E40_2_double_fit.json` нет; отдельно цитировались коммиты, которых нет среди объектов git.
Правило яруса (§ «Правила яруса», п. 4) требует, чтобы число попадало в отчёт только из
свежего чтения артефакта. Этот тест и есть механическая версия того правила: он сверяет
строки чек-листа `docs/e40_todo.md` §12 с `results/E40_*.json`.

Артефакты в git не лежат (`results/*.json` в .gitignore) — при отсутствии файла тест
пропускается, а не падает.
"""
from __future__ import annotations

import json
import pathlib

import pytest

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
LAYERS = ("6", "12", "18", "22", "23")


def art(name: str) -> dict:
    path = RESULTS / f"E40_{name}.json"
    if not path.exists():
        pytest.skip(f"нет артефакта {path.name} (перепрогнать experiments/exp40_*.py)")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["0_gate", "1_T4a", "1_audit", "1_core_manifest", "2_double_fit",
                                  "3_fft", "4_intersection", "5_kinetic", "6_R_profile",
                                  "7_G_fixation", "8_synthesis"])
def test_shaпка_общая(name):
    d = art(name)
    assert d["observational"] is True
    assert d["new_forwards"] == 0
    assert d.get("core_version", "v1.0.0") == "v1.0.0"


def test_0_gate_решает_код_аудит():
    d = art("0_gate")
    assert d["gate_pass"] is True and d["E40_2_can_start"] is True
    assert d["E27_5_6_p_slice"] == "p_t" and d["p_position"] == 0 and d["T5_p_position"] == 32
    assert all(a["found"] for a in d["code_audit"]), "строка продюсера съехала — гейт протух"
    # p_checksum удалён как неработоспособный, а не спрятан
    assert "p_checksum" not in d and "p_checksum_note" in d
    inv = d["cache_invariants"]
    assert inv["keysets_equal"] and inv["p_normalized"] and inv["j_shape_mismatch"] == 0
    vec = d["vector_reconcile"]
    assert vec["matches_exp27b"] and vec["reversal_confirmed"]
    assert vec["max_abs_dev_reversed"] < 1e-5 < vec["max_abs_dev_wrong_pairing"]


def test_2_исход_и_якоря():
    d = art("2_double_fit")
    assert d["outcome"] == "unstable"
    assert d["stretched_beta_C_F"] == {}
    assert d["corr_f_applicable"] is False, "corr_f вне ΔW-канала запрещён (правило §6 п. 21)"
    assert d["raw_anchor_worst_dev"] < 1e-6
    flags = [d["per_layer"][L][w]["resolution_flag"] for L in LAYERS for w in ("C_F", "C_top16")]
    assert flags.count("not_distinguishable_at_Δ_grid") >= 9, flags
    for L in LAYERS:
        for w in ("C_F", "C_top16"):
            c = d["per_layer"][L][w]
            assert c["aicc_gap_to_second"] >= 0.0
    # чего в этом артефакте нет — зафиксировано, чтобы никто не вернул числа по памяти
    for absent in ("s_same", "sensitivity_p0", "norm_kill", "energy_kill"):
        assert absent not in json.dumps(d), f"{absent} появился в артефакте — обновить документ"


def test_3_нет_пика_и_нет_dyadic():
    d = art("3_fft")
    assert d["grid_is_dyadic"] is False
    assert d["n_close_w0"] == 0, "пик на ω₀ появился — пересматривать вердикт"
    assert d["n_pass_3sigma"] <= 1


def test_4_ставка_и_H4():
    d = art("4_intersection")
    L22 = d["per_layer"]["22"]
    # CI вырожден: все реплики с пересечением дают тот же r_cross (артефакт v2, 37b9045).
    # docs/E40_tier0.md до 2026-11-05 обещал [127, 129] — это было число из более раннего
    # прогона; свёрстано по артефакту, ширина здесь неинформативна, см. frac_boot_no_crossing
    assert L22["r_cross"] == 127.0 and L22["bootstrap_CI95"] == [127.0, 127.0]
    assert abs(d["per_layer"]["18"]["frac_boot_no_crossing"] - 0.41) < 5e-3
    assert abs(L22["frac_boot_no_crossing"] - 0.569) < 5e-3
    assert d["prediction_check"]["verdict"] == "miss" and d["prediction_check"]["predicted"] == 60
    assert d["H4_strong_status"] == "alive"
    for L in ("18", "22"):  # мёртвый столбец S04 притворяется независимостью (E40_8, раунд 1)
        assert len(d["per_doc_S04_tail_gap"][L]) == 32


def test_5_тождество_и_флаги():
    d = art("5_kinetic")
    for L in LAYERS:
        v = d["per_layer"][L]
        assert v["gram_closure_worst_rel"] <= 1e-10, (L, v["gram_closure_worst_rel"])
    assert d["per_layer"]["23"]["linearization_valid_all"] is True
    for L in ("6", "12", "18", "22"):
        assert d["per_layer"][L]["linearization_valid_all"] is False
    for L in LAYERS:  # единственный линейный слой — L23, и там eps3 в пороге 1e-2
        assert d["per_layer"][L]["flags_frac"]["rotation_unique"] == 0.0
    assert abs(d["per_layer"]["23"]["eps3_over_dJ_valid"]) <= 1e-2
    offs = [d["per_layer"][L]["E_off_over_dJ"] for L in LAYERS]
    assert min(offs) >= 0.3 and max(offs) < 0.9, offs


def test_2c_sрез_по_позиции_не_объясняет_диапазон_r_eff():
    d = art("2c_pos_slice")
    assert d["new_forwards"] == 0
    inner_t = ("0", "8", "16", "32", "48", "64", "96")
    for L in LAYERS:
        v = d["per_layer"][L]
        inner = [v["by_position"][t]["K128"]["r_eff_KF"] for t in inner_t]
        assert all(1.0 <= x <= 32.0 for x in inner), (L, inner)  # PR матрицы 32 документов
        assert v["by_position"]["127"]["K128"]["r_eff_KF"] < min(inner), L
        ent = v["p_entropy_by_t"]
        assert ent["127"] == max(ent.values()), ent  # t=127 = первый токен (нулевой контекст)
        span = (max(inner) - min(inner)) / (sum(inner) / len(inner))
        if L == "6":
            assert span < 0.02, span  # якорь: на мелком слое срез стоит


def test_6_критерий_и_насыщение():
    d = art("6_R_profile")
    assert sum(1 for v in d["criteria"].values() if v is True) == 2
    assert d["verdict"] == "monotone_or_flat__one_sided_no_L0"
    assert d["corr_Rcan_kappa"]["spearman"] < -0.99  # насыщение при n=5, не аргумент


def test_7_G_монотонен_и_бутстрап_накрывает():
    d = art("7_G_fixation")
    g = [d["per_layer"][L]["G"] for L in LAYERS]
    assert all(x <= y for x, y in zip(g, g[1:])), g
    for L in LAYERS:
        v = d["per_layer"][L]
        assert abs(v["G"] - (1.0 - v["delta_fixP"] / v["delta_raw"])) < 1e-12, L
        assert v["delta_fixP"] <= v["delta_raw"], L
        b = v["G_doc_bootstrap"]
        assert b["ci_lo"] <= v["G"] <= b["ci_hi"], (L, b, v["G"])
        assert b["ci_hi"] - b["ci_lo"] < 0.1, (L, b)  # узкое место — число слоёв, не документы
    assert d["max_G"] < 0.5 and d["verdict"] in ("ambiguous", "G_small_monotone")


def test_8_ранг_внутри_своего_CI():
    d = art("8_synthesis")
    r = d["rank_estimate"]
    assert r["rank"] == 1 and r["bootstrap_CI_rank"][0] <= 1 <= r["bootstrap_CI_rank"][1]
    assert r["bootstrap_CI_PR"][0] <= r["PR"] <= r["bootstrap_CI_PR"][1]
    assert d["framework_single_object"] == "indeterminate"
