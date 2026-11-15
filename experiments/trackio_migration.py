#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Одноразовая не-деструктивная миграция всех прогонов prism в trackio.

ТЗ: docs/TZ_trackio_migration.md (v1.8). Реестр: 115 run'ов / 130 .log (снапшот 15.11 04:00).
Читает: логи (read-only). Пишет: {TRACKIO_DIR}/prism.db (SQLite, gitignored) +
results/trackio_migration_report.json (sanitized).

Протокол (ТЗ §8): quiescence (§5.1) → md5 до → trackio.init/log/finish по каждому run →
md5 после → отчёт. Fail-loud: find-аффт, md5-расхождение, live writer, DB без --fresh.
Цифры только из строк логов (ТЗ §4/§5.4): ничего не выдумывать; счётчики экстракторов
сверяются с ТЗ-заявлениями (расхождение = error в отчёте, RC≠0).
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKIO_DIR = Path(os.environ.get("TRACKIO_DIR", str(ROOT / ".trackio")))
DB_PATH = TRACKIO_DIR / "prism.db"
LOCK_PATH = TRACKIO_DIR / "prism.lock"
REPORT_PATH = ROOT / "results" / "trackio_migration_report.json"
TZ_PATH = ROOT / "docs" / "TZ_trackio_migration.md"
TZ_VERSION = "v1.8"

# ---------------------------------------------------------------------------
# Реестр (ТЗ §3, v1.7). name: snake_case; log: путь от ROOT; duration_s: None =
# вычисляется (T4: макс поля time; e27b_early: макс колонки Time) или нет (null).
# ---------------------------------------------------------------------------
T4_RUNS = [
    ("e17_lora_distill", "prism-exp17-lora-distill.log", ""),
    ("e16_4_3_v23_fd_check", "prism-exp16-4-3-v23-single-pair-fd-check.log", ""),
    ("e32r3_layer_profile", "prism-exp32r3-layer-profile.log", ""),
    ("e50v3_loop_curvature", "results/e50v3_out/prism-exp50v3-loop-curvature-qwen.log", ""),
    ("e52k_scale_curve_ptq", "logs/e52k_v7/prism-exp52k-scale-curve-real-ptq.log",
     "голова лога обрезана Kaggle (пометка ТЗ)"),
    ("e27_noise_floor", "kaggle_kernel/kernel-exp27/out/prism-exp27-noise-floor.log", ""),
    ("e27_noise_floor_v9", "_auto_chain/exp27_v9/prism-exp27-noise-floor.log", ""),
    ("e27b_dense_profile", "kaggle_kernel/prism-exp27b-dense-profile.log", ""),
    ("e39m_flat_ptq", "results/exp39m/run.log", ""),
    ("e16_readout_pull", "prism-readout-pull.log", ""),
    ("e16_stage0_targets", "prism-exp16-stage0-targets.log", "канон md5-дублей §3.5"),
    ("e4_k_sensitivity", "prism-exp4-k-sensitivity.log", ""),
    ("e19_destruction_tpu", "kaggle_kernel/kernel-exp19-tpu/prism-exp19-destruction-tpu.log", ""),
    ("e18_logit_kd", "prism-exp18-logit-kd.log", ""),
    ("e11c_defrag_bridge", "prism-exp11c-defrag-bridge.log", ""),
    ("e64_tpu_v8", "results/e64/tpu_v8/prism-e64-gpu.log", ""),
    ("e19_destruction_early", "prism-exp19-destruction-tpu.log", ""),
    ("e16_stage0_targets_short", "results/prism-exp16-stage0-targets.log", ""),
    ("e10b_gauge_bridge", "prism-exp10b-gauge-bridge.log", ""),
    ("e01_coherence", "prism-exp01-coherence.log", "канон md5-дублей §3.5"),
    ("e18_logit_kd_tpu", "kaggle_kernel/prism-exp18-logit-kd-tpu.log", ""),
    ("at16_ceiling_ceil4", "results/ceil4_out/prism-at16-ceiling.log", ""),
    ("at16_ceiling", "results/ceil_out/prism-at16-ceiling.log", ""),
    ("e16_greedy_proto", "results/proto_out/prism-greedy-proto.log", ""),
    ("rleaf_random", "results/rleaf_out/prism-r-leaf-random.log", ""),
    ("at16_ceiling_ceil3", "results/ceil3_out/prism-at16-ceiling.log", ""),
    ("e32r3_1_tail_recompute", "kaggle_kernel/kernel-exp32r3/output_v7/prism-exp32r31-tail-recompute.log", ""),
    ("e64_b4_temperature_pool", "results/e64/kag_out/prism-b4-temperature-pool.log", ""),
    ("e2_heatmap", "prism-exp2-heatmap.log", ""),
    ("e6_l18_snr", "prism-exp6-l18-snr.log", ""),
    ("e19_destruction_short", "prism-exp19-destruction.log", ""),
    ("e38d_holonomy", "prism-exp38d-holonomy-spectrum.log", ""),
    ("e16_readout_pull_tpu", "prism-exp16-readout-pull-tpu.log", ""),
    ("e64_gpu_old", "results/e64gpu_out_old/prism-e64-gpu.log", ""),
    ("e16_readout_pull_tpu2", "prism-readout-pull-tpu.log", ""),
    ("e17_2nd_canary_tpu", "kaggle_kernel/kernel-exp17-2nd-canary/out/prism-exp17-2nd-order-canary-tpu.log", ""),
    ("lfm3_textgen", "results/lfm3gen_out/prism-lfm3-textgen.log", ""),
    ("probe_input", "results/probe_out/prism-probe-input.log", ""),
]
TABLE_RUNS = [
    ("e27b_early", "prismexp27b.log", ""),
]
# (name, log, duration_s, notes) — duration из реестра ТЗ §3.3 (источник в скобках там)
CPU_RUNS = [
    ("e45_damage_vs_structure", "results/exp45_damage_vs_structure.log", None,
     "нижняя граница 15505s из queue_round.log (config: duration_lb_s)"),
    ("e45b_leaf_table", "results/exp45b_leaf_table.log", 6079, "queue: 04:18:49→06:00:08"),
    ("e45c_gauge_null", "results/exp45c_gauge_null.log", 3583, "queue: 03:19:06→04:18:49"),
    ("e46_frame_as_memory", "results/exp46_frame_as_memory.log", 8, "queue: 06:00:09→06:00:17"),
    ("e45c_rerun", "results/exp45c_gauge_null.log", 1203,
     "второй run на том же файле (файл содержит ТОЛЬКО вывод rerun)"),
    ("e45_seednull", "results/exp45_seednull.log", 255, "queue: 06:00:18→06:04:33"),
    ("e45_verdicts", "results/exp45_verdicts.log", None, ""),
    ("e45_gap_scan", "results/exp45_gap_scan.log", 1007.9, "wall 1007.9 с"),
    ("e45_kappa_seed_null", "results/exp45_kappa_seed_null.log", None, ""),
    ("e45b_analysis", "results/exp45b_analysis.log", None, ""),
    ("e45f_gauge_decomp", "results/exp45f_gauge_decomp.log", 2373, "wall 2373 с"),
    ("e45g_cut_stability", "results/exp45g_cut_stability.log", 779, "wall 779 с"),
    ("e46_verdicts", "results/exp46_verdicts.log", None, ""),
    ("e47c_p2_loop_prediction", "results/exp47c_p2_loop_prediction.log", None, ""),
    ("e7_cka_baseline", "results/exp7_cka_baseline_n1024.log", 26620, "[OK] … 26620s"),
    ("e_r11_basis_tie", "results/exp_r11_basis_tie.log", None, ""),
    ("e32r2_hessian", "results/exp32r2_hessian.log", 6142, "total 6142s"),
    ("e32r2_smoke", "results/exp32r2_smoke.log", 170, "total 170s"),
    ("e32r3_1", "logs/exp32r3_1.log", None, ""),
    ("e32tail_pilot", "logs_e32tail_pilot.log", None, "Triton→CPU fallback"),
    ("e35_preflight", "results/exp35_preflight.log", None, ""),
    ("e37_run", "results/exp37_run.log", None, ""),
    ("e37_run2", "results/exp37_run2.log", None, ""),
    ("e37_run3", "results/exp37_run3.log", None, ""),
    ("e37_run4", "results/exp37_run4.log", None, ""),
    ("e37r_run", "results/exp37r_run.log", None, ""),
    ("e37r_agg", "results/exp37r_agg.log", None, ""),
    ("e38a_rerun", "results/exp38a_rerun.log", None, "«in 7.7s» — время записи JSON, НЕ duration"),
    ("e38b_rerun", "results/exp38b_rerun.log", None, ""),
    ("e38c_rerun", "results/exp38c_rerun.log", None, ""),
    ("e38d_rerun", "results/exp38d_rerun.log", None, ""),
    ("e38e_rerun", "results/exp38e_rerun.log", None, ""),
    ("e38f_rerun", "results/exp38f_rerun.log", None, ""),
    ("e38g_rerun", "results/exp38g_rerun.log", None, ""),
    ("e38h_rerun", "results/exp38h_rerun.log", None, ""),
    ("e38i_rerun", "results/exp38i_rerun.log", None, ""),
    ("e38e_true_j_l23", "results/E38e_true_J_L23.log", 1374.6, "wall 1374.6 с; канон дубля §3.5"),
    ("e39a_v2", "results/archive/exp39a_v2.log", None, ""),
    ("e39a_v3", "results/archive/exp39a_v3.log", None, ""),
    ("e39a_v4", "results/archive/exp39a_v4.log", None, ""),
    ("e39a_v5", "results/archive/exp39a_v5.log", None, ""),
    ("e39a_v6", "results/archive/exp39a_v6.log", None, ""),
    ("e39a_v7", "results/archive/exp39a_v7.log", None, ""),
    ("e39a_v8", "results/archive/exp39a_v8.log", None, ""),
    ("e39a_v9", "results/archive/exp39a_v9.log", None, ""),
    ("e39a_v10", "results/archive/exp39a_v10.log", None, ""),
    ("e39a_v11", "results/archive/exp39a_v11.log", None, ""),
    ("e39c_pre_rerun_fix", "logs/exp39c_pre_rerun_fix.log", None, ""),
    ("e39d_head_route_probe", "results/E39d_head_route_probe.log", 489.7, "wall 489.7 с"),
    ("e39d_control_lam0_order1", "results/exp39d_control_lam0_order1.log", None, ""),
    ("e39d_floor_seeds12", "results/exp39d_floor_seeds12.log", None, ""),
    ("e39d_stage1_ordering", "results/exp39d_stage1_ordering.log", None, ""),
    ("e39j_run", "results/exp39j_run.log", None, ""),
    ("e39k_run", "results/exp39k_run.log", 22.6, "wall 22.6 с"),
    ("e39m_smoke", "results/exp39m_smoke.log", None, ""),
    ("e39_supp_pscale_probe", "results/exp39_supp_pscale_probe.log", None, ""),
    ("e39_supp_three_objects", "results/exp39_supp_three_objects.log", 356.6, "wall 356.6 с"),
    ("e01_coherence_v6_draw2", "results/exp01_coherence_v6.draw2.log", None, ""),
    ("e01_coherence_v6_pinned", "results/exp01_coherence_v6.pinned.log", 213, "[OK] … 213s"),
    ("e51_lfm3", "logs/e51_lfm3.log", None, ""),
    ("e51_qwen", "logs/e51_qwen.log", None, ""),
    ("at0_1", "results/AT0/AT0_1_run.log", None, ""),
    ("at0_2", "results/AT0/AT0_2_run.log", None, ""),
    ("at0_2_r6_stdout", "results/AT0/AT0_2_r6_stdout.log", None,
     "r6-свип (96 доков) — возможно живой писатель: quiescence §5.1"),
    ("at0_3", "results/AT0/AT0_3_run.log", None, ""),
    ("at0_4", "results/AT0/AT0_4_run.log", None, ""),
    ("at0_4_rebuild", "results/AT0/AT0_4_rebuild.log", None, ""),
    ("at0_4_rebuild_r3", "results/AT0/AT0_4_rebuild_r3.log", None, "rebuild r3, 14.11; маркеров wall нет"),
    ("at0_4_rebuild_r4", "results/AT0/AT0_4_rebuild_r4.log", None, "rebuild r4, 14.11; маркеров wall нет"),
    ("at0_4_rebuild_r5", "results/AT0/AT0_4_rebuild_r5.log", None, "rebuild r5, 15.11; маркеров wall нет"),
    ("at0_5", "results/at0_5_run.log", None, ""),
    ("chain_e45b", "results/chain_e45b.log", None, ""),
    ("chain_e46", "results/chain_e46.log", None, ""),
    ("stage0", "results/stage0.log", None, "Device: cpu, n_samples=2, K=128 — smoke-прогон"),
    ("at0_2b", "results/AT0/AT0_2b_run.log", None, "R/A-split r2, 15.11; маркеров wall нет"),
]
IGPU_RUNS = [
    ("e93_sycl_bench", "results/e93/sycl/bench_all.txt", None,
     "bench-таблица t/s по model×precision×fa (4-я секция пуста — бенч оборван; 6 точек)"),
]

# md5-дубли (ТЗ §3.5): путь -> канон-ран (НЕ отдельные run)
MD5_DUPES = {
    "kaggle_kernel/kernel-exp01/out/prism-exp01-coherence.log": "e01_coherence",
    "exp16-stage-0-target-jacobian-cache.log": "e16_stage0_targets",
    "exp16-stage0-target-jacobian-cache.log": "e16_stage0_targets",
    "results/E38e_true_J_L23_carrier_cheat.log": "e38e_true_j_l23",
    "results/AT0/store/runlog_r5_pre_rerun.log": "at0_1",
}
# Исключённые (ТЗ §3.6): путь -> причина. XET_GLOB — два xet-лога (пути с меткой времени).
EXCLUDED = {
    "kaggle_kernel/auto_chain_tpu.log": "статусы-очереди Kaggle",
    "kaggle_kernel/_push_queue.log": "статусы-очереди Kaggle",
    "kaggle_kernel/_push_wait.log": "статусы-очереди Kaggle",
    "logs/kaggle_daemon.log": "статусы",
    "logs/exp32r3_1_status.log": "статусы",
    "results/queue_round.log": "wall-часы очереди (источник duration, не run)",
    "results/queue_tail.log": "wall-часы очереди (источник duration, не run)",
    "results/tail_round.log": "wall-часы очереди (источник duration, не run)",
    "results/e93/sycl/bench_err.log": "stderr бенча",
    "results/e93/sycl/smoke_err.log": "stderr бенча",
}
XET_GLOBS = [
    "kaggle_kernel/kernel-exp17-2nd-canary/out/.hf_cache/xet/logs/xet_*.log",
    "kaggle_kernel/kernel-exp19-tpu/.hf_cache/xet/logs/xet_*.log",
]

FIND_EXCLUDES = [".git", ".venv", ".hf_cache", "box"]

RUN_FAILED_MARKERS = ["[FAIL]", "Aborted (core dumped)", "AssertionError"]
# прогресс-маркеры (не экспериментальные метрики, ТЗ §7.5 v1.8 N18): n_lines + step_idx
PROGRESS_METRICS = {"n_lines", "step_idx"}
EMPTY_METRICS_EXPECTED = {
    "e16_stage0_targets", "e16_stage0_targets_short", "e2_heatmap", "e6_l18_snr",
    "e38d_holonomy", "e16_readout_pull_tpu", "e16_readout_pull_tpu2",
    "e17_2nd_canary_tpu", "lfm3_textgen", "probe_input", "e18_logit_kd_tpu", "e27b_early",
}
RUN_FAILED_EXPECTED = {
    "e27_noise_floor", "e27_noise_floor_v9", "e19_destruction_tpu", "e19_destruction_early",
    "e18_logit_kd_tpu", "e2_heatmap", "e38d_holonomy", "e16_readout_pull_tpu",
    "e16_readout_pull_tpu2", "lfm3_textgen",
}

# Счётчики ТЗ (r3-пересчёт, проверено ревьюером) — сверка фактических raw-hit'ов.
TZ_EXPECTED_HITS = {
    "bridge": 1549, "kl": 1551, "dose": 1011, "jdone_e27": 384, "jdone_e27b": 48,
    "jnorm_e17": 64, "layer_done": 20, "step_idx": 799, "loss_e39m": 45, "chat": 1,
    "eval": 1014, "dip": 130, "k_avg": 7, "e11c_rawsoft": 8, "rho": 4, "e10b": 7,
    "cell_s": 192, "s_l": 12, "e01_jdone": 32, "c_b": 4, "at16": 286,
    "rand_seed": 3, "trained_done": 1, "doc_done": 64, "p_same": 1,
}

# ---------------------------------------------------------------------------
# Паттерны (ТЗ §4, v1.7). Каждый — (id, regex, обработчик(match, rec, step))
# ---------------------------------------------------------------------------
RE_BRIDGE = re.compile(r"bridge=([\d.]+)")
RE_KL = re.compile(r"kl=([-+]?[\d.]+)", re.I)
RE_DOSE = re.compile(r"‖ΔW‖=([\d.]+)")
RE_JDONE_E27 = re.compile(r"L(\d+): (\d+) J done \((\d+)/(\d+)\)")
RE_JDONE_E27B = re.compile(r"L(\d+) J (\d+)/(\d+)")
RE_JNORM = re.compile(r"J_norm: (\d+)/(\d+)")
RE_LAYER_DONE = re.compile(r"Layer \d+_\d+ done in (\d+)s")
RE_STEP_IDX = re.compile(r"step (\d+)/(\d+):")
RE_LOSS = re.compile(r"step (\d+): loss=([\d.]+)")
RE_CHAT = re.compile(r"chat s\d+: loops=\d+ a=([\d.-]+) s=([\d.-]+) p=([\d.-]+) g=([\d.-]+) ent=([\d.-]+)")
RE_EVAL = re.compile(r"\[eval\] step=?\s*(\d+)(?=[:\s])")
RE_DIP = re.compile(r"P([0-5][a-z]?)(?: ([\w.]+))? dip (-?[\d.]+)")
RE_SCEN_A = re.compile(r"^--- Scenario \((\w+)\) ---\s*$")
RE_SCEN_B = re.compile(r"^--- Scenario: (\w+) \(baseline\) ---\s*$")
RE_SCEN_C = re.compile(r"^--- Scenario \(([a-z])\): .+ ---\s*$")
RE_K_AVG = re.compile(r"K=([\d.]+): avg=([\d.]+)")
RE_E11C = re.compile(r"(\d+)->(\d+): raw=([\d.]+), soft=([\d.]+), dr8=([\d.]+), dr16=([\d.]+), dr32=([\d.]+)")
RE_RHO = re.compile(r"raw vs (\w+): ρ = ([\d.-]+)")
RE_E10B = re.compile(r"(\d+)->(\d+): raw=([\d.]+), cent=([\d.]+), soft=([\d.]+)")
RE_CELL_S = re.compile(r"L(\d+) T=128 \('(\w+)', (\d+)\): ([\d.]+)s")
RE_S_L = re.compile(r"L(\d+): s_l=([\d.eE+-]+) slope=([\d.eE+-]+) quad=([\d.eE+-]+) d3r1 raw/gf/fixP = ([\d.eE+-]+)/([\d.eE+-]+)/([\d.eE+-]+) d_b=([\d.eE+-]+)")
RE_E01_JDONE = re.compile(r"(\w+) s\d+: \d+L x \d+pos J done \((\d+)/(\d+)\)")
RE_C_B = re.compile(r"L(\d+): C_b=([\d.]+)")
RE_AT16 = re.compile(r"(\S+) (\S+)\|(bare|nudge) (hit|garbage) (\d+)s")
RE_RAND_SEED = re.compile(r"rand seed (\d+) (\d+)s")
RE_TRAINED_DONE = re.compile(r"trained done (\d+)s")
RE_DOC_DONE = re.compile(r"^\w+_\d+ done$")
RE_P_SAME = re.compile(r"(\w+): \{'p_same': ([\d.]+), 'sha': '(\w+)', 'uniq': ([\d.]+)")
RE_MARK_OK = re.compile(r"\[OK\] \S+ (\d+)s")
RE_MARK_TOTAL = re.compile(r"Total time: (\d+)s")
RE_MARK_ALLDONE = re.compile(r"ALL DONE (\d+)s")
RE_MARK_DONE = re.compile(r"DONE ([\d.]+)s")
RE_WALL = re.compile(r"wall ([\d.]+) с")
RE_MARK_TOTAL_S = re.compile(r"\btotal (\d+)s")
RE_TIME_COL = re.compile(r"^(\d+\.\d+)s\t(\d+)\t")
RE_BENCH_SEC = re.compile(r"^== (.+?) fa=(on|off) ==$")


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_json_log(path):
    """JSON-Kaggle-лог: первая непустая строка (после ','-обрезки) начинается с '{'."""
    with open(path, errors="replace") as f:
        for line in f:
            s = line.strip().lstrip(",").rstrip(",")
            if s:
                return s.startswith("{")
    return False


def parse_json_log(path):
    """-> (events, max_time, n_lines): events = [(time, text)], text может быть multiline."""
    events, max_time, n_lines = [], 0.0, 0
    with open(path, errors="replace") as f:
        for line in f:
            n_lines += 1
            s = line.strip().lstrip(",").rstrip(",")
            if not s.startswith("{"):
                continue
            try:
                o = json.loads(s)
            except Exception:
                continue
            t = o.get("time")
            if not isinstance(t, (int, float)):
                continue
            max_time = max(max_time, float(t))
            events.append((float(t), o.get("data", "")))
    return events, max_time, n_lines


def extract_metrics(path, events):
    """Применяет все экстракторы ТЗ §4 к JSON-логу. -> (metrics, raw_hits, warnings).

    metrics: dict step -> {metric: value} (первый выигрыш при коллизии key+step).
    raw_hits: Counter по id-паттернам (до scenario-tagging — для сверки с ТЗ).
    """
    metrics = defaultdict(dict)
    raw_hits = Counter()
    dropped = 0
    _cur_scen = "no_edit"  # начальное состояние (ТЗ §4 п.12)

    # scenario-tagging: pre-scan (есть ли заголовки)
    has_scen = any((RE_SCEN_A.match(l) or RE_SCEN_B.match(l) or RE_SCEN_C.match(l))
                   for _, d in events for l in d.splitlines())

    for t, data in events:
        step = int(t)
        for line in data.splitlines():
            ls = line.strip()

            # --- scenario headers (только если заголовки есть в логе)
            if has_scen:
                m = RE_SCEN_A.match(ls) or RE_SCEN_B.match(ls)
                if m:
                    _cur_scen = m.group(1)
                    continue
                m = RE_SCEN_C.match(ls)
                if m:
                    _cur_scen = m.group(1)
                    continue

            # 1-3: bridge / kl / dose (с тегом сценария если has_scen)
            m = RE_BRIDGE.search(line)
            if m:
                raw_hits["bridge"] += 1
                name = f"bridge_{_cur_scen}" if has_scen else "bridge"
                _put(metrics, step, name, float(m.group(1)), dropped)
            m = RE_KL.search(line)
            if m:
                raw_hits["kl"] += 1
                name = f"kl_{_cur_scen}" if has_scen else "kl"
                _put(metrics, step, name, float(m.group(1)), dropped)
            m = RE_DOSE.search(line)
            if m:
                raw_hits["dose"] += 1
                name = f"dose_{_cur_scen}" if has_scen else "dose"
                _put(metrics, step, name, float(m.group(1)), dropped)

            # 4-5b: J-прогресс
            m = RE_JDONE_E27.search(line)
            if m:
                raw_hits["jdone_e27"] += 1
                _put(metrics, step, f"j_progress_L{m.group(1)}", int(m.group(2)) / int(m.group(4)), dropped)
            else:
                m = RE_JDONE_E27B.search(line)
                if m:
                    raw_hits["jdone_e27b"] += 1
                    _put(metrics, step, f"j_progress_L{m.group(1)}", int(m.group(2)) / int(m.group(3)), dropped)
            m = RE_JNORM.search(line)
            if m:
                raw_hits["jnorm_e17"] += 1
                _put(metrics, step, "j_progress", int(m.group(1)) / int(m.group(2)), dropped)

            # 6: layer done
            m = RE_LAYER_DONE.search(line)
            if m:
                raw_hits["layer_done"] += 1
                _put(metrics, step, "layer_done_s", float(m.group(1)), dropped)

            # 7: step idx
            m = RE_STEP_IDX.search(line)
            if m:
                raw_hits["step_idx"] += 1
                _put(metrics, step, "step_idx", float(m.group(1)), dropped)

            # 8: loss
            m = RE_LOSS.search(line)
            if m:
                raw_hits["loss_e39m"] += 1
                _put(metrics, step, "loss", float(m.group(2)), dropped)

            # 9: chat
            m = RE_CHAT.search(line)
            if m:
                raw_hits["chat"] += 1
                for k, v in zip(("a", "s", "p", "g", "ent"), m.groups()):
                    _put(metrics, step, f"chat_{k}", float(v), dropped)

            # 10: eval
            m = RE_EVAL.search(line)
            if m:
                raw_hits["eval"] += 1
                _put(metrics, step, "eval_step_idx", float(m.group(1)), dropped)

            # 11: dip
            m = RE_DIP.search(line)
            if m:
                raw_hits["dip"] += 1
                n, label, val = m.group(1), m.group(2), m.group(3)
                name = f"p{n}_{label}_dip" if label else f"p{n}_dip"
                _put(metrics, step, name, float(val), dropped)

            # 13: k_avg
            m = RE_K_AVG.search(line)
            if m:
                raw_hits["k_avg"] += 1
                _put(metrics, step, "k_avg", float(m.group(2)), dropped)

            # 14: e11c
            m = RE_E11C.search(line)
            if m:
                raw_hits["e11c_rawsoft"] += 1
                for k, v in zip(("raw", "soft", "dr8", "dr16", "dr32"), m.groups()[2:]):
                    _put(metrics, step, k, float(v), dropped)

            # 15: rho
            m = RE_RHO.search(line)
            if m:
                raw_hits["rho"] += 1
                _put(metrics, step, f"rho_{m.group(1)}", float(m.group(2)), dropped)

            # 16: e10b
            m = RE_E10B.search(line)
            if m:
                raw_hits["e10b"] += 1
                for k, v in zip(("raw", "cent", "soft"), m.groups()[2:]):
                    _put(metrics, step, k, float(v), dropped)

            # 17: cell_s (step = eps-idx — особое правило ТЗ v1.7 N10)
            m = RE_CELL_S.search(line)
            if m:
                raw_hits["cell_s"] += 1
                L, cell, idx, val = m.groups()
                _put(metrics, int(idx), f"cell_s_L{L}_{cell}", float(val), dropped)

            # 18: s_l (имена с L — несколько слоёв в одном логе)
            m = RE_S_L.search(line)
            if m:
                raw_hits["s_l"] += 1
                L = m.group(1)
                vals = [float(x) for x in m.groups()[2:]]
                for k, v in zip(("s_l", "slope", "quad", "d3r1_raw", "d3r1_gf",
                                 "d3r1_fixp", "d_b"), vals):
                    _put(metrics, step, f"{k}_L{L}", v, dropped)

            # 19: e01 j done
            m = RE_E01_JDONE.search(line)
            if m:
                raw_hits["e01_jdone"] += 1
                _put(metrics, step, "j_progress", int(m.group(2)) / int(m.group(3)), dropped)

            # 20: C_b
            m = RE_C_B.search(line)
            if m:
                raw_hits["c_b"] += 1
                _put(metrics, step, f"C_b_L{m.group(1)}", float(m.group(2)), dropped)

            # 21: at16
            m = RE_AT16.search(line)
            if m:
                raw_hits["at16"] += 1
                variant, outcome, sec = m.group(3), m.group(4), m.group(5)
                _put(metrics, step, f"at16_{outcome}_{variant}", float(sec), dropped)

            # 22: rleaf
            m = RE_RAND_SEED.search(line)
            if m:
                raw_hits["rand_seed"] += 1
                _put(metrics, step, "rand_seed_s", float(m.group(2)), dropped)
            m = RE_TRAINED_DONE.search(line)
            if m:
                raw_hits["trained_done"] += 1
                _put(metrics, step, "trained_done_s", float(m.group(1)), dropped)

            # 23: doc done
            if RE_DOC_DONE.match(ls):
                raw_hits["doc_done"] += 1
                _put(metrics, step, "doc_done", 1.0, dropped)

            # 24: p_same (только «hf:» — со двоеточием)
            m = RE_P_SAME.search(line)
            if m:
                raw_hits["p_same"] += 1
                src = m.group(1)
                _put(metrics, step, f"p_same_{src}", float(m.group(2)), dropped)
                _put(metrics, step, f"uniq_{src}", float(m.group(4)), dropped)

            # 25: rleaf trained JSON
            if ls.startswith("{") and '"trained":' in ls:
                try:
                    obj = json.loads(ls)
                    tr = obj.get("trained")
                    if isinstance(tr, dict):
                        for k, v in tr.items():
                            if isinstance(v, (int, float)):
                                _put(metrics, step, f"trained_{k}", float(v), dropped)
                except Exception:
                    pass
    return metrics, raw_hits


def _put(metrics, step, name, value, dropped):
    """Первый выигрыш при коллизии (step, name); value — float."""
    if name in metrics[step]:
        return
    metrics[step][name] = value


def find_logs():
    args = ["find", "."]
    for ex in FIND_EXCLUDES:
        args += ["-not", "-path", f"./{ex}/*"]
    args += ["-name", "*.log"]
    out = subprocess.run(args, capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        raise RuntimeError(f"find failed: {out.stderr}")
    return sorted(os.path.relpath(p, ".") for p in out.stdout.splitlines() if p.strip())


def check_registry(find_list):
    """Аффт реестра против find (ТЗ §3): каждый .log — run, дубль или исключён."""
    reg_paths = {r[1] for r in T4_RUNS} | {r[1] for r in TABLE_RUNS} | \
                {r[1] for r in CPU_RUNS} | {IGPU_RUNS[0][1]}
    xet = []
    for pat in XET_GLOBS:
        xet.extend(os.path.relpath(p, ".") for p in glob.glob(pat, root_dir=ROOT))
    excluded = set(EXCLUDED) | set(xet)
    errors = []
    for p in find_list:
        if p in reg_paths or p in MD5_DUPES or p in excluded:
            continue
        errors.append(f"UNCOVERED .log (не run, не дубль, не исключён): {p}")
    for p in reg_paths | set(MD5_DUPES) | excluded:
        if p.endswith(".log") and p not in find_list:
            errors.append(f"registry/dup/excl-файл НЕ найден find'ом: {p}")
    n_uniq_reg_log = len([p for p in reg_paths if p.endswith(".log")])
    stats = {
        "find_count": len(find_list),
        "registry_runs": len(T4_RUNS) + len(TABLE_RUNS) + len(CPU_RUNS) + len(IGPU_RUNS),
        "registry_uniq_log_files": n_uniq_reg_log,
        "md5_dupes": len(MD5_DUPES),
        "excluded": len(excluded),
        "sum_check": n_uniq_reg_log + len(MD5_DUPES) + len(excluded),
        "xet_logs": xet,
    }
    # 130 = 113 уник .log в run'ах + 5 дублей + 12 исключённых (ТЗ §3)
    if stats["find_count"] != 130 or stats["sum_check"] != 130 or stats["registry_runs"] != 115:
        errors.append(f"арифметика 115/130 нарушена: {stats}")
    return errors, stats, xet


def find_writers(paths):
    """/proc-scan: кто держит открытые fd в файлы paths (Linux; надёжнее lsof —
    lsof на этой машине даёт неполный вывод из-за docker overlay). -> {path: {pids}}"""
    writers = {}
    targets = {str(ROOT / p) for p in paths}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        fddir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fddir)
        except (PermissionError, FileNotFoundError, OSError):
            continue
        for fd in fds:
            try:
                link = os.readlink(f"{fddir}/{fd}")
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if link in targets:
                writers.setdefault(link, set()).add(pid)
    return writers


def quiescence(find_list, md5_before):
    """ТЗ §5.1 (N15): нет живого писателя в регистровые логи + mtime стабилен >= 10 мин."""
    now = time.time()
    fresh = []
    for p in find_list:
        fp = ROOT / p
        if now - fp.stat().st_mtime > 600:
            continue
        fresh.append(p)
    writers_map = find_writers(find_list)
    return {"writers": {p: sorted(v) for p, v in writers_map.items()},
            "fresh_mtimes_10min": fresh}


def sanitize(obj):
    """float-касты, no NaN/Inf (ТЗ §5)."""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):
            return None
        return obj
    if isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    return str(obj)


def bench_points(path):
    """e93: t_s_{f16|f32}_fa{on|off}_{pp512|tg64} — 6 точек (4-я секция пуста)."""
    prec = None
    fa = None
    pts = {}
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_BENCH_SEC.match(line.strip())
            if m:
                prec = "f16" if "f16" in m.group(1) else "f32"
                fa = m.group(2)
                continue
            if line.startswith("|") and prec and fa and "t/s" not in line and "---" not in line:
                cols = [c.strip() for c in line.strip().strip("|").split("|")]
                # тест = предпоследняя колонка (pp512/tg64), t/s = последняя
                if len(cols) >= 2:
                    test, ts = cols[-2], cols[-1]
                    if test in ("pp512", "tg64"):
                        vm = re.match(r"([\d.]+)", ts)
                        if vm:
                            pts[f"t_s_{prec}_fa{fa}_{test}"] = float(vm.group(1))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="удалить только prism.db (ТЗ §5.2)")
    ap.add_argument("--smoke", action="store_true",
                    help="только find-аффт + quiescence + md5, без trackio (демо-режим)")
    args = ap.parse_args()

    report = {"tz": str(TZ_PATH), "tz_version": TZ_VERSION, "rc": 0,
              "errors": [], "warnings": [], "runs": []}

    # --- idempotency (ТЗ §5.2)
    if DB_PATH.exists() and not args.fresh:
        print(f"FATAL: {DB_PATH} существует; пересоздание только с --fresh (ТЗ §5.2)", file=sys.stderr)
        return 2
    if args.fresh and DB_PATH.exists():
        DB_PATH.unlink()
    if LOCK_PATH.exists():
        LOCK_PATH.unlink()
    TRACKIO_DIR.mkdir(parents=True, exist_ok=True)

    # --- find + аффт
    find_list = find_logs()
    reg_errors, reg_stats, xet = check_registry(find_list)
    report["registry"] = reg_stats
    report["errors"].extend(reg_errors)
    if reg_errors:
        _finish_report(report)
        return 1

    # --- md5 до + quiescence
    md5_before = {p: md5_of(ROOT / p) for p in find_list}
    report["md5_before"] = md5_before
    q = quiescence(find_list, md5_before)
    report["quiescence"] = q
    if q["writers"] or q["fresh_mtimes_10min"]:
        report["errors"].append(
            f"quiescence нарушена (ТЗ §5.1/N15): writers={q['writers']} fresh_mtimes={q['fresh_mtimes_10min']} — "
            f"дождитесь завершения живых прогонов (ат0_2 r6) и повторите")
        _finish_report(report)
        return 3

    if args.smoke:
        print(f"SMOK: find={len(find_list)} quiescence OK, trackio не запускался")
        _finish_report(report)
        return 0

    # --- trackio
    import trackio  # лениво: смоук не требует установленного trackio

    t4_names = {r[0] for r in T4_RUNS}
    n_metrics_total = 0
    raw_hits_all = Counter()

    def process_run(name, log, device, group, duration_s, notes):
        nonlocal n_metrics_total
        fp = ROOT / log
        is_json = is_json_log(fp)
        max_time = 0.0
        n_lines = 0
        metrics = {}
        run_failed = False
        dur_marker = None
        dur_marker_src = None
        text_all = ""

        if is_json:
            events, max_time, n_lines = parse_json_log(fp)
            metrics, raw_hits = extract_metrics(fp, events)
            raw_hits_all.update(raw_hits)
            text_all = "".join(d for _, d in events)
            if duration_s is None:
                duration_s = max_time
        else:
            with open(fp, errors="replace") as f:
                text_all = f.read()
            n_lines = text_all.count("\n") + (0 if text_all.endswith("\n") else 1)

        # duration-маркеры (T4: из JSON-текста; CPU: из текста; ALL DONE до DONE —
        # «ALL DONE Ns» содержит подстроку «DONE Ns»)
        for rx, src in ((RE_MARK_OK, "[OK]"), (RE_MARK_TOTAL, "Total time"),
                        (RE_MARK_ALLDONE, "ALL DONE"), (RE_MARK_DONE, "DONE"),
                        (RE_WALL, "wall"), (RE_MARK_TOTAL_S, "total")):
            m = rx.search(text_all)
            if m:
                dur_marker = float(m.group(1))
                dur_marker_src = src
                break

        # run_failed (T4)
        if device == "T4":
            run_failed = any(mk in text_all for mk in RUN_FAILED_MARKERS)

        # e27b_early: таблица — duration = макс колонки Time, метрик нет (ТЗ §3.2/§4)
        if name == "e27b_early":
            times = [float(m.group(1)) for m in
                     (RE_TIME_COL.match(l.lstrip()) for l in text_all.splitlines()) if m]
            if not times:
                raise RuntimeError(f"{name}: табличный лог без Time-строк")
            duration_s = max(times)
            metrics = {}

        # e93: bench-таблица (n_lines нет — не .log; 6 точек — гарантированные)
        if name == "e93_sycl_bench":
            b = bench_points(fp)
            if not b:
                raise RuntimeError(f"{name}: bench-парсинг дал 0 точек")
            metrics = {0: b}
            n_lines = None

        # гарантированная точка n_lines (ТЗ §4); empty_metrics — без экспериментальных
        # метрик (прогресс-маркеры n_lines/step_idx не считаются — ТЗ §7.5 v1.8 N18)
        exp_names = set()
        for st, mm in metrics.items():
            exp_names.update(mm)
        exp_names -= PROGRESS_METRICS
        empty = len(exp_names) == 0
        if n_lines is not None:
            metrics.setdefault(0, {})["n_lines"] = float(n_lines)

        cfg = {
            "name": name, "device": device, "log": log, "md5": md5_before.get(log) or (md5_of(fp) if fp.exists() else None),
            "duration_s": duration_s, "duration_marker_s": dur_marker,
            "duration_marker_src": dur_marker_src, "n_lines": n_lines,
            "empty_metrics": empty, "run_failed": run_failed, "notes": notes,
            "tz": TZ_VERSION,
        }
        if name == "e45_damage_vs_structure":
            cfg["duration_lb_s"] = 15505

        run = trackio.init(project="prism", name=name, group=group, config=sanitize(cfg))
        for st in sorted(metrics):
            if metrics[st]:
                run.log({k: float(v) for k, v in metrics[st].items()}, step=st)
        run.finish()
        n_metrics_total += sum(len(v) for v in metrics.values())
        report["runs"].append({"name": name, "log": log, "device": device,
                               "duration_s": duration_s, "duration_marker_s": dur_marker,
                               "n_lines": n_lines, "empty_metrics": empty,
                               "run_failed": run_failed,
                               "n_metrics": sum(len(v) for v in metrics.values())})

    try:
        for name, log, notes in T4_RUNS:
            process_run(name, log, "T4", "T4", None, notes)
        for name, log, notes in TABLE_RUNS:
            process_run(name, log, "table", "table", None, notes)
        for name, log, dur, notes in CPU_RUNS:
            process_run(name, log, "CPU", "CPU", dur, notes)
        name, log, dur, notes = IGPU_RUNS[0]
        process_run(name, log, "iGPU", "iGPU", dur, notes)
    except Exception as e:
        report["errors"].append(f"trackio-прогон прерван: {type(e).__name__}: {e}")
        _finish_report(report)
        return 1

    # --- md5 после (ТЗ §5.1)
    md5_after = {p: md5_of(ROOT / p) for p in find_list}
    mismatch = [p for p in find_list if md5_before[p] != md5_after[p]]
    report["md5_mismatch"] = mismatch
    if mismatch:
        report["errors"].append(f"md5-расхождение после миграции (нарушение не-деструктивности): {mismatch}")

    # --- сверка счётчиков с ТЗ
    report["raw_hits"] = dict(raw_hits_all)
    for k, expected in TZ_EXPECTED_HITS.items():
        got = raw_hits_all.get(k, 0)
        if got != expected:
            report["errors"].append(f"счётчик «{k}»: факт {got} ≠ ТЗ {expected} (ТЗ §4/r3)")
    # --- expected-списки
    empty_actual = {r["name"] for r in report["runs"] if r["empty_metrics"]}
    failed_actual = {r["name"] for r in report["runs"] if r["run_failed"]}
    if empty_actual != EMPTY_METRICS_EXPECTED:
        report["errors"].append(f"empty_metrics: факт {sorted(empty_actual)} ≠ ТЗ §7.5 "
                                f"{sorted(EMPTY_METRICS_EXPECTED)}")
    if failed_actual != RUN_FAILED_EXPECTED:
        report["errors"].append(f"run_failed: факт {sorted(failed_actual)} ≠ ТЗ §4 "
                                f"{sorted(RUN_FAILED_EXPECTED)}")
    # --- duration marker vs max_time (T4: dev > 30s = warning)
    for r in report["runs"]:
        if r["device"] == "T4" and r["duration_marker_s"] is not None and r["duration_s"] is not None:
            dev = abs(r["duration_s"] - r["duration_marker_s"])
            if dev > 30:
                report["warnings"].append(f"{r['name']}: marker {r['duration_marker_s']} vs "
                                          f"max_time {r['duration_s']} dev {dev:.1f}s > 30s "
                                          f"(NbConvert-хвост, ТЗ §3.1)")

    report["n_runs"] = len(report["runs"])
    report["n_metrics_total"] = n_metrics_total

    _finish_report(report)
    return 1 if report["errors"] else 0


def _finish_report(report):
    report = sanitize(report)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    if report["errors"]:
        print(f"REPORT: {len(report['errors'])} errors -> {REPORT_PATH}", file=sys.stderr)
        for e in report["errors"]:
            print("  ERROR:", e, file=sys.stderr)
    if report["warnings"]:
        print(f"REPORT: {len(report['warnings'])} warnings -> {REPORT_PATH}")
    print(f"REPORT: {report.get('n_runs', 0)} runs, {report.get('n_metrics_total', 0)} metrics -> {REPORT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
