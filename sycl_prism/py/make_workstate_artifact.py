"""make_workstate_artifact.py — производитель артефакта SYCL-WORKSTATE (TZ §3.7).

Собирает `results/sycl/sycl_workstate_20261117.json` ТОЛЬКО из лог-строк:
  - лог run_one.sh (stdout+stderr всего прогона run_local.sh, включая паритет);
  - stdout пробы probe_sycl_min (отдельный файл — G4);
  - строки sycl-ls (выполняются самим скриптом: окружение, а не расчёт).
Ничего «из памяти»: каждое число — парсинг строки лога/stdout.

sanitize: только float()/int()/str/bool; NaN/inf — ОШИБКА (не запись).
src_rev обязан равняться TOЧНО v17-20261117-waitfix (G6(a)) — иначе RC=1.

Запуск:
    python3 py/make_workstate_artifact.py \
        --log /tmp/sycl_all.log \
        --probe-stdout results/sycl/probe_sycl_min_stdout.txt \
        --out results/sycl/sycl_workstate_20261117.json
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_SRC_REV = "v17-20261117-waitfix"

# ---------- парсинг строк (каждая — из лога/stdout) ----------

RE_LOADAVG = re.compile(r"загрузка машины \(loadavg, 1 мин\): ([0-9.]+)")
RE_DEVICE = re.compile(r"^устройство: (\{.*\})\s*$", re.M)
RE_VPATH = re.compile(r"\[(\d+)x(\d+)\] vpath\s+maxdiff=([0-9.eE+-]+) dropped=(\d+)")
RE_DCURVE_MEAN = re.compile(r"\[(\d+)x(\d+)\] dcurve mean rel=([0-9.eE+-]+)")
RE_DCURVE_MED = re.compile(r"\[(\d+)x(\d+)\] dcurve med\s+rel=([0-9.eE+-]+)")
RE_ACF = re.compile(
    r"\[(\d+)x(\d+)\] acf c rel=([0-9.eE+-]+) c0 rel=([0-9.eE+-]+) r rel=([0-9.eE+-]+)"
    r" gram_resid Δ=([0-9.eE+-]+)")
RE_G1G3_REF = re.compile(r"\[(\d+)x(\d+)\] G1\(ref\)=([0-9.eE+-]+) G3\(ref\)=([0-9.eE+-]+)")
RE_G1G3_DEV = re.compile(
    r"\[(\d+)x(\d+)\] G1\(устройство\)=([0-9.eE+-]+) G3\(устройство\)=([0-9.eE+-]+)")
RE_RA = re.compile(r"\[(\d+)x(\d+)\] ra closure Δ/scale=([0-9.eE+-]+) cross=([0-9.eE+-]+)")
RE_FP16 = re.compile(r"\[fp16\] dcurve mean rel=([0-9.eE+-]+)")
RE_NULL = re.compile(r"\[null\] (\w+)\s+rel=([0-9.eE+-]+)")
RE_ITOG = re.compile(r"ИТОГ: (ПАРИТЕТ OK|РАСХОЖДЕНИЕ)")
RE_STEP3 = re.compile(r"^=== 3\. ПАРИТЕТ ПРИБОРА", re.M)
RE_STEP = re.compile(r"^=== (\d+[a-z0-9-]*)\. (.+?) ===\s*$", re.M)
RE_PROBE_VERDICT = re.compile(r"ВЕРДИКТ: ([A-ZА-Я _]+)")
RE_PROBE_GFLOPS = re.compile(r"iGPU, тайлованное fp32:\s+([0-9.]+) GFLOP/s")

RED_MARKERS = ("!!!", "Aborted", "Segmentation fault", "core dumped")
RE_RC_NONZERO = re.compile(r"код возврата: ([1-9]\d*)")


def _f(s: str) -> float:
    """Число из строки: только конечные (NaN/inf — ошибка, TZ §3.7)."""
    v = float(s)
    if not math.isfinite(v):
        raise ValueError(f"NaN/inf в лог-строке: {s!r} — запись в артефакт запрещена")
    return v


def _sanitize(o):
    """Рекурсивный каст: float/int/str/bool/None/list/dict.
    None — явный сентинел «не найдено в логе» (JSON null); NaN/inf — исключение
    (число найдено, но неконечное — запись запрещена, TZ §3.7)."""
    if o is None:
        return None
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, int):
        return int(o)
    if isinstance(o, float):
        if not math.isfinite(o):
            raise ValueError(f"NaN/inf в артефакте: {o!r}")
        return float(o)
    if isinstance(o, str):
        return o
    raise TypeError(f"тип {type(o)!r} не записывается в артефакт")


def _strip_ansi(s: str) -> str:
    """ANSI-коды из заголовков шагов (say() в run_local.sh) — вне чисел."""
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def parse_log(text: str) -> dict:
    d = {}
    text = _strip_ansi(text)
    m = RE_LOADAVG.search(text)
    d["loadavg_at_run"] = _f(m.group(1)) if m else None

    m = RE_DEVICE.search(text)
    if not m:
        raise ValueError("лог: нет строки 'устройство: {...}' (паритет не запускался?)")
    d["device"] = ast.literal_eval(m.group(1))

    forms = {}
    for m in RE_VPATH.finditer(text):
        forms.setdefault((int(m.group(1)), int(m.group(2))), {})["vpath_maxdiff"] = _f(m.group(3))
        forms[(int(m.group(1)), int(m.group(2)))]["vpath_dropped"] = int(m.group(4))
    for m in RE_DCURVE_MEAN.finditer(text):
        forms.setdefault((int(m.group(1)), int(m.group(2))), {})["dcurve_mean_rel"] = _f(m.group(3))
    for m in RE_DCURVE_MED.finditer(text):
        forms.setdefault((int(m.group(1)), int(m.group(2))), {})["dcurve_med_rel"] = _f(m.group(3))
    for m in RE_ACF.finditer(text):
        forms.setdefault((int(m.group(1)), int(m.group(2))), {}).update(
            {"acf_c_rel": _f(m.group(3)), "acf_c0_rel": _f(m.group(4)),
             "acf_r_rel": _f(m.group(5)), "acf_gram_resid_delta": _f(m.group(6))})
    for m in RE_G1G3_REF.finditer(text):
        forms.setdefault((int(m.group(1)), int(m.group(2))), {}).update(
            {"G1_ref": _f(m.group(3)), "G3_ref": _f(m.group(4))})
    for m in RE_G1G3_DEV.finditer(text):
        forms.setdefault((int(m.group(1)), int(m.group(2))), {}).update(
            {"G1_device": _f(m.group(3)), "G3_device": _f(m.group(4))})
    for m in RE_RA.finditer(text):
        forms.setdefault((int(m.group(1)), int(m.group(2))), {}).update(
            {"ra_closure_rel": _f(m.group(3)), "ra_cross": _f(m.group(4))})
    d["parity_forms"] = [dict(f, T=t, D=dd) for (t, dd), f in sorted(forms.items())]

    m = RE_FP16.search(text)
    d["parity_fp16"] = {"dcurve_mean_rel": _f(m.group(1))} if m else None
    d["parity_null"] = {m.group(1): _f(m.group(2)) for m in RE_NULL.finditer(text)}

    # ИТОГ берётся ТОЛЬКО из секции шага 3 (реальный GPU-паритет):
    # в логе раньше встречается шим-ИТОГ синтаксис-гейта (шаг 1a, CPU), а в
    # финальной секции — упоминание «ИТОГ: ПАРИТЕТ OK» в тексте ⇒ ни первое,
    # ни последнее совпадение не годится (ревью MINOR-1).
    m3 = RE_STEP3.search(text)
    if m3 is None:
        raise ValueError("лог: нет секции шага 3 (ПАРИТЕТ) — паритет не запускался")
    itog_m = RE_ITOG.search(text, m3.end())
    d["parity_total"] = itog_m.group(1) if itog_m else None
    d["parity_total_ok"] = (d["parity_total"] == "ПАРИТЕТ OK")

    steps = [f"{n}. {name}" for n, name in RE_STEP.findall(text)]
    d["steps_in_log"] = steps
    d["red_markers_in_log"] = [mk for mk in RED_MARKERS if mk in text]
    mrc = RE_RC_NONZERO.search(text)
    d["nonzero_rc_probe"] = int(mrc.group(1)) if mrc else None
    return d


def parse_probe_stdout(text: str) -> dict:
    d = {}
    text = _strip_ansi(text)
    m = RE_PROBE_VERDICT.search(text)
    d["verdict"] = m.group(1).strip() if m else None
    m = RE_PROBE_GFLOPS.search(text)
    d["tiled_fp32_gflops"] = _f(m.group(1)) if m else None
    m = RE_LOADAVG.search(text)
    d["loadavg_at_probe"] = _f(m.group(1)) if m else None
    return d


def sycl_ls_lines(out_file=None) -> list:
    """Строки sycl-ls (окружение, не расчёт — ТЗ §3.7); копирование в файл
    даёт провенанс (ревью NIT-2)."""
    try:
        r = subprocess.run(["sycl-ls"], capture_output=True, text=True, timeout=60)
    except Exception as e:  # окружение без sycl-ls — не крашим артефакт
        return [f"sycl-ls недоступен: {e!r}"]
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if out_file is not None and lines:
        out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="лог run_one.sh прогона run_local.sh")
    ap.add_argument("--probe-stdout", required=True, help="stdout пробы probe_sycl_min")
    ap.add_argument("--out", required=True, help="путь к итоговому JSON")
    args = ap.parse_args()

    log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    probe_text = Path(args.probe_stdout).read_text(encoding="utf-8", errors="replace")

    log = parse_log(log_text)
    probe = parse_probe_stdout(probe_text)
    log["sycl_ls"] = sycl_ls_lines(Path(args.out).parent / "sycl_ls.txt")

    # гейт src_rev — ТОЧНО (G6(a))
    src_rev = log["device"].get("src_rev")
    if src_rev != EXPECTED_SRC_REV:
        print(f"ОШИБКА: src_rev = {src_rev!r} ≠ {EXPECTED_SRC_REV!r} — "
              f"артефакт не создан (пересоберите библиотеку)", file=sys.stderr)
        return 1
    if not log["parity_total_ok"]:
        print(f"ОШИБКА: паритет не пройден (ИТОГ = {log['parity_total']!r}) — "
              f"артефакт не создан", file=sys.stderr)
        return 1
    if log["red_markers_in_log"] or log["nonzero_rc_probe"] is not None:
        print(f"ОШИБКА: красные маркеры в логе: {log['red_markers_in_log']} "
              f"rc={log['nonzero_rc_probe']} — артефакт не создан", file=sys.stderr)
        return 1

    artifact = {
        "artifact_kind": "bringup",
        "tz": "docs/TZ_sycl_workstate.md",
        "note": ("bring-up, НЕ транспортный артефакт: protocol/check_artifact.py "
                 "неприменим (нет speedup_vs_cpu/gemm_dtype/reduce_dtype); "
                 "trackio-запись не создаётся (AGENTS.md: bring-up, не эксперимент)"),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "loadavg": {"at_run": log["loadavg_at_run"], "at_probe": probe["loadavg_at_probe"]},
        "device": log["device"],
        "sycl_ls": log["sycl_ls"],
        "sycl_ls_file": str(Path(args.out).parent / "sycl_ls.txt"),
        "probe": probe,
        "parity": {
            "forms": log["parity_forms"],
            "fp16": log["parity_fp16"],
            "null": log["parity_null"],
            "total": log["parity_total"],
        },
        "steps_in_log": log["steps_in_log"],
        "log": str(args.log),
        "probe_stdout": str(args.probe_stdout),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_sanitize(artifact), f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"OK: артефакт {args.out} (src_rev={src_rev}, ИТОГ={log['parity_total']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
