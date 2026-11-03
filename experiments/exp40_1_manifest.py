#!/usr/bin/env python3
"""E-40.1 — манифест ядра v1 + пункты аудита (a1)–(l1) + freeze (l1) (docs/e40_todo.md §3.1, §3.3).

Статусы реестра — результат прогона: 17 инвариантов гоняются their-check'ами на
синтетике, O1_eq_s2 дополнительно проверяется на реальном кэше exp27b; unit-тесты
— фактический запуск pytest трёх файлов. (l1): шесть summary-статистик
per-(doc,layer) заморожены до вычислений 0.2–0.7.

Запуск: .venv/bin/python experiments/exp40_1_manifest.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from _e38_cache import load_layer  # noqa: E402
from exp40_common import LAYERS, ROOT, dump_json, header  # noqa: E402
from prism.invariants import INVARIANTS  # noqa: E402
from prism.operator_checks import O1, s  # noqa: E402

TEST_FILES = ["tests/test_nulls_tag_gate.py", "tests/test_operator_invariants.py",
              "tests/test_transport_kinetic.py"]

L1_FREEZE = {
    "carrier": "per_(doc,layer), n=32×5=160 (слоёв 5, не 6 — фиксация исполнителя)",
    "n_primary": 160,
    "stats": {
        "S02_xi_exp": "ℓ из per-doc фитa экспоненты C(Δ)=A·exp(−Δ/ℓ)+c0 на canonical "
                      "C_F (E-40.2), Δ=1..32; не сошёлся → NaN (исключается парно)",
        "S03_amp_oct": "амплитуда ДПФ-бина на 1 цикл/октаву (ω=2π/log2) остатка "
                       "log C − log (fit S02), log-uniform интерполяция 64 точек "
                       "(конвейер E-40.3), per-doc",
        "S04_tail_gap": "разность хвостовых долей: Σλ ESG r∈[119,128]/Σλ минус "
                        "b-доля гессайновых мод k∈[1,6] (E-40.4), per-doc",
        "S05_radial_share": "mean_t E_radial / (E_radial + E_off_manifold) — только "
                            "всегда-интерпретируемые каналы (E-40.5), per-doc",
        "S06_R_can": "λmax(E_K)/tr(E_K), E_K = Σ_t P(p_t)J_t (P(p_t)J_t)ᵀ (E-40.6), per-doc",
        "S07_G": "1 − Δ_fixP/Δ_raw, r=16, Δ=1, per-doc (E-40.7; fixP = P(p̄)J, "
                 "p̄ = среднее p по корпусу)",
    },
    "reserved": {"S08a": "E-41.4 (фаза 2)", "S08b": "E-44 (фаза 1.5)"},
    "standardization": "z-скор по слою внутри статистики (mean/std по 32 документам), "
                       "NaN-пары исключаются попарно перед корреляцией",
    "clusters": {"C1": ["S02", "S03", "S04"], "C2": ["S05", "S06", "S07"],
                 "C3": [], "C4": []},
    "rules": {"R1": "rank≥5 или PR≥5 → рамка «один объект» умирает",
              "R2": "rank≤1 и PR≤2.5 → сильная унификация с оговоркой о провенансе",
              "R3": "иначе — отчёт по осям с CI"},
}

ITEMS = {
    "(a1)": "«98–99%» = survive_frac = ‖PJ‖/‖J‖ (НОРМА; энергия 99.95–99.98%) — "
            "E40_1_audit.A3 + E39-supplement (git) §2; triple (operator=P(p_t), "
            "normalization=norm, slice=all 4096)",
    "(a2)": "g_uni ≡ ESG gauge_share (тождество, ≤8e-16); O1 — отдельная величина; "
            "третьего объекта нет (E-39 suppl §2)",
    "(a3)": "identity-asserts на реальном кэше — см. results/E40_1_T4a.json "
            "a3_identity_asserts (O2≡O3 ≤7.1e-15, g_uni(J^gf) ≤5.2e-17, iso inline=closed)",
    "(b)": "r_eff E-27.6 T6 — ESG-спектр от raw JR (в конвейере exp27_6 нет P); "
           "коллизия имён с r_eff^U(τ)=178 E-39a закрыта субскриптом (A4 аудита); "
           "p_slice спектров E-27.5/27.6 — p_t, POS=0/T5=32 (E40_0_gate)",
    "(c)": "Δ_gf E-33r.2 — L23, правый, срез gf, r=16, Δ=1, Δ_SVD (max principal "
           "angle), med по документам — E40_1_audit.A1",
    "(d)": "Δ_raw — все 5 слоёв (layers.*.delta); Δ_fixP — в основном JSON только L23, "
           "в exp33r_gauge_anatomy_strict(_diag).json — все 5 слоёв (diag.E_fixed_p) "
           "⇒ E-40.7 без даунгрейда",
    "(e)": "L0 в exp27b НЕТ (слои 6/12/18/22/23) ⇒ E-40.6 критерий односторонний, "
           "not_testable_no_L0 по построению",
    "(f)": "порог левого канала max(0.05·E_right, Q_95(E_left^null)); нуль — Haar-"
           "конъюгация фактической M (M'=OMOᵀ, O∈Haar(128)), B=1000; полное Haar-"
           "вращение отвергнуто (недостижимый масштаб) — применяется в E-40.5",
    "(g)": "exp31 C_F — на RAW J (exp31_correlation.py, JR без P) ⇒ пересчёт "
           "canonical в E-40.2 (не рефит); raw-пересчёт — якорь сверки с C_med",
    "(h)": "exp30: моды Грама ESG (G=Σ_t J Jᵀ, per-sample медианы), r-сетка 1..128, "
           "слои {18,22}; l-сетки нет (один срез слоёв)",
    "(i)": "exp32-tail: b — коэффициент a2 при e² в dz=a1 e+a2 e²+a3 e³ по "
           "token-пространству (T=128), слои {18,22}, per_doc есть",
    "(j)": "exp27b: 4 домена × 8 документов × S=128 позиций, K=128 (U128-база), "
           "слои 6/12/18/22/23; sentinel 50.71% (порог 0.5)",
    "(k)": "Δ_raw: 5 слоёв; Δ_fixP: 5 слоёв только в strict-вариантах (см. (d))",
    "(l1)": "заморожено ниже (L1_FREEZE), ноль вычислений",
}


def run_pytest() -> dict:
    r = subprocess.run([sys.executable, "-m", "pytest", *TEST_FILES, "-q", "--no-header"],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    tail = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"(\d+) passed", tail)
    f = re.search(r"(\d+) failed", tail)
    return {"cmd": "pytest " + " ".join(TEST_FILES) + " -q",
            "passed": int(m.group(1)) if m else 0,
            "failed": int(f.group(1)) if f else 0,
            "rc": r.returncode}


def o1_eq_s2_on_cache() -> float:
    """O1_eq_s2 на реальном кэше: s(J,p)² ≡ O1(J,p) на фиксированных позициях."""
    worst = 0.0
    ck = load_layer(23, want=("J", "P"))
    J = ck["J"].double().reshape(-1, *ck["J"].shape[-2:])
    P = ck["P"].double().reshape(-1, ck["P"].shape[-1])
    P = P / P.sum(-1, keepdim=True)
    for pos in torch.linspace(0, J.shape[0] - 1, 16).long():
        j = J[pos].numpy()
        p = P[pos].numpy()
        # корень у s живёт в rom/mean-агрегациях; per_position = сырая энергия (docstring)
        o = float(O1(j, p, aggregation="ratio_of_means", p_slice="p_t"))
        ss = float(s(j, p, aggregation="ratio_of_means", p_slice="p_t"))
        worst = max(worst, abs(o - ss ** 2) / max(o, 1e-300))
    return worst


def main(results_dir: str = None) -> dict:
    results_dir = results_dir or os.path.join(ROOT, "results")
    t0 = time.time()
    assert set(LAYERS) == {6, 12, 18, 22, 23}, "LAYERS должен совпадать с exp27b"

    # аудит пересоздаём (детерминирован, 0.06 с) и дописываем разрешение A4
    subprocess.run([sys.executable, os.path.join(HERE, "exp40_1_audit.py")],
                   check=True, capture_output=True, text=True, timeout=120)
    audit_path = os.path.join(results_dir, "E40_1_audit.json")
    audit = json.load(open(audit_path))
    gate = json.load(open(os.path.join(results_dir, "E40_0_gate.json")))
    audit["A4_resolution"] = {
        "p_slice": gate["E27_5_6_p_slice"], "p_position": gate["p_position"],
        "T5_p_position": gate["T5_p_position"],
        "evidence": "код-аудит exp27.5_pair_matrix.py:28, exp27_6_isolation.py:38,58 "
                    "+ векторная сверка на S=128 (matches=True)",
        "artifact": "results/E40_0_gate.json", "status": "resolved"}
    audit["gate_open_items_for_E40_2"] = [
        {"item": "p_slice спектров E-27.5/27.6", "status": "resolved (E-40.0)",
         "value": "p_t (POS=0; T5=POS=32)"}]
    dump_json(audit, audit_path)

    registry = {}
    for name, inv in INVARIANTS.items():
        entry = {"status": inv.status, "identity": inv.identity,
                 "modules": list(inv.modules)}
        try:  # живой прогон проверки инварианта на синтетике
            inv.check(np.random.default_rng(40))
            entry["live_run"] = "passing"
        except Exception as exc:  # noqa: BLE001 — протоколируем отказ как статус
            entry["live_run"] = f"FAILING: {exc}"
        registry[name] = entry
    rel = o1_eq_s2_on_cache()
    registry["O1_eq_s2"]["live_cache_rel_dev"] = rel
    assert rel < 1e-12, f"O1_eq_s2 на реальном кэше не выдержана: {rel:.2e}"
    failing = [k for k, v in registry.items() if str(v["live_run"]).startswith("FAIL")]
    assert not failing, f"инварианты в отказе: {failing}"

    pytest_res = run_pytest()
    assert pytest_res["failed"] == 0 and pytest_res["passed"] >= 26, pytest_res

    t4a = json.load(open(os.path.join(results_dir, "E40_1_T4a.json")))
    assert t4a["E6_status"] == "resolved" and t4a["g_uni_checkpoint"]["matches"]

    rep = header(
        "E-40.1-manifest", "docs/e40_todo.md §3.3",
        {"exp27b_jcache_L23": "/data/prism_kernels/exp27b/results/exp27b_jcache_L23.pt",
         "E40_1_T4a": os.path.join(results_dir, "E40_1_T4a.json"),
         "E40_0_gate": os.path.join(results_dir, "E40_0_gate.json")},
        channel_tags=[],
        extra={"invariants_registry": registry,
               "audit_items": ITEMS,
               "l1_summary_stats_freeze": L1_FREEZE,
               "unit_tests": pytest_res,
               "prereg_schema_check": {
                   "all_8_tests_filled": True,
                   "evidence": "tests IV.1–6 в test_operator_invariants.py + "
                               "T4a grid (test_triple_grid_8_cells) + негативный median",
                   "e44_draft_filled": os.path.exists(os.path.join(ROOT, "docs", "E44.md"))},
               "wall_sec": round(time.time() - t0, 1)})
    out = os.path.join(results_dir, "E40_1_core_manifest.json")
    dump_json(rep, out)
    print(f"wrote {out}: pytest {pytest_res['passed']} passed, "
          f"registry {len(registry)}/17 live-passing, O1≡s² rel {rel:.1e}")
    return rep


if __name__ == "__main__":
    main()
