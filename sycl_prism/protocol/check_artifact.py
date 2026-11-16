"""Проверка артефакта на соответствие протоколу переноса (protocol_v1.json).

Запуск:
    python3 check_artifact.py results/AT0/AT0_4_MIX.json                 # одно поле
    python3 check_artifact.py --stages base_pre,dense_metrics <арт>     # проверка стадий
    python3 check_artifact.py --pair cpu.json gpu.json                  # паритет CPU↔GPU

Что проверяется:
  1. наличие обязательных полей (backend, device, protocol_version, gemm_dtype,
     reduce_dtype, control_passed, parity_max_abs, speedup_vs_cpu);
  2. запреты: TF32/bf16 в GEMM, молчаливый фолбэк, смешение форматов без пометки;
  3. что стадия из `never_transfer` не заявлена как перенесённая;
  4. (--pair) что значения общих полей CPU↔GPU совпадают в допуске протокола.

Код возврата 0 = PASS, 1 = FAIL. Печать — человекочитаемая.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# Протокол лежит как Python-модуль (в нём комментарии и docstring) ⇒ импортируем его.
_spec = _ilu.spec_from_file_location("prism_gpu_protocol", os.path.join(HERE, "protocol_v1.py"))
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PROTOCOL = _mod.PROTOCOL

OK = "  \033[32mOK\033[0m"
BAD = "  \033[31mFAIL\033[0m"
WARN = "  \033[33mWARN\033[0m"


def _walk(obj, path=""):
    """Плоский список (путь, значение) по JSON."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _walk(v, f"{path}[{i}]")
    else:
        out.append((path, obj))
    return out


def find_value(obj, key):
    """Первое значение по имени ключа в любом месте JSON (путь, значение)."""
    hits = [(p, v) for p, v in _walk(obj) if p.split(".")[-1].split("[")[0] == key]
    return hits[0] if hits else (None, None)


def check_one(path: str, stages: list[str] | None = None) -> int:
    with open(path, encoding="utf-8") as f:
        art = json.load(f)
    fails, warns = [], []

    print(f"артефакт: {path}")
    print(f"протокол: {PROTOCOL['protocol']} v{PROTOCOL['version']}")

    # 1) обязательные поля
    for field in PROTOCOL["required_artifact_fields"]:
        p, v = find_value(art, field)
        if p is None:
            fails.append(f"нет обязательного поля «{field}» (R1: число без backend/device не публикуется)")
        elif field in ("backend", "device", "gemm_dtype", "reduce_dtype", "protocol_version"):
            print(f"{OK} {field} = {v}  [{p}]")
        elif field == "control_passed":
            if v is not True:
                fails.append(f"control_passed = {v} (R7: перенос без зелёных контролей не принимается)")
            else:
                print(f"{OK} control_passed = True")
        elif field == "parity_max_abs":
            tol = PROTOCOL["controls"]["parity"]["tolerances"]["fp32"]
            if isinstance(v, (int, float)) and v > tol:
                fails.append(f"parity_max_abs = {v} > допуска {tol}")
            else:
                print(f"{OK} parity_max_abs = {v} (допуск {tol})")
        elif field == "speedup_vs_cpu":
            if v is None:
                warns.append("speedup_vs_cpu = None: выигрыш не измерен на целевой форме (R8)")
            elif isinstance(v, (int, float)) and v < PROTOCOL["acceptance"]["min_speedup"]:
                fails.append(f"speedup_vs_cpu = {v} < min_speedup {PROTOCOL['acceptance']['min_speedup']}")
            else:
                print(f"{OK} speedup_vs_cpu = {v}")

    # 2) запреты
    _, gemm = find_value(art, "gemm_dtype")
    if isinstance(gemm, str) and gemm.lower() in ("tf32", "bf16", "bfloat16", "fp16", "float16"):
        fails.append(f"gemm_dtype = {gemm}: для метрик запрещено ускоренное GEMM (R4)")
    _, backend = find_value(art, "backend")
    _, fb = find_value(art, "fallback_reason")
    if backend == "cpu" and fb is not None:
        warns.append("backend=cpu и заполнен fallback_reason — если вычисление шло на GPU, "
                     "отметьте это явно (иначе артефакт вводит в заблуждение)")
    if backend not in (None, "cpu") and fb is not None:
        print(f"{WARN} fallback_reason = {fb}")
    if backend not in (None, "cpu"):
        # смешение форматов
        _, red = find_value(art, "reduce_dtype")
        _, note = find_value(art, "systematic_note")
        if red == "fp64" and note is None:
            warns.append("reduce_dtype=fp64 без systematic_note (R3): если null считался в fp32, "
                         "это надо объявить")

    # 3) never_transfer
    for name in PROTOCOL["never_transfer"]:
        p, v = find_value(art, name)
        if v is not None and isinstance(v, str) and v.lower() in ("gpu", "sycl", "xpu", "torch_xpu"):
            fails.append(f"стадия {name} помечена как перенесённая на GPU, но она в never_transfer")

    # 4) стадии
    if stages:
        for st in stages:
            spec = PROTOCOL["stages"].get(st)
            if spec is None:
                fails.append(f"стадия «{st}» не описана в протоколе")
                continue
            print(f"{OK} стадия {st}: {spec['label']} → {spec['device']} "
                  f"(контроли: {', '.join(spec['controls'])})")

    print()
    for w in warns:
        print(f"{WARN} {w}")
    for e in fails:
        print(f"{BAD} {e}")
    if fails:
        print(f"\nИТОГ: FAIL ({len(fails)} нарушений, {len(warns)} предупреждений)")
        return 1
    print(f"\nИТОГ: PASS ({len(warns)} предупреждений)")
    return 0


def check_pair(cpu_path: str, gpu_path: str) -> int:
    with open(cpu_path, encoding="utf-8") as f:
        a = json.load(f)
    with open(gpu_path, encoding="utf-8") as f:
        b = json.load(f)
    fa = dict((p, v) for p, v in _walk(a) if isinstance(v, (int, float)) and not isinstance(v, bool))
    fb = dict((p, v) for p, v in _walk(b) if isinstance(v, (int, float)) and not isinstance(v, bool))
    common = sorted(set(fa) & set(fb))
    print(f"общих числовых полей: {len(common)}")
    if not common:
        print(f"{BAD} нет общих полей — нечего сравнивать")
        return 1
    tol = PROTOCOL["controls"]["parity"]["tolerances"]["fp32"]
    worst, worst_p = 0.0, None
    for p in common:
        x, y = fa[p], fb[p]
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        d = abs(x - y) / max(abs(x), 1e-30)
        if d > worst:
            worst, worst_p = d, p
    print(f"{'OK' if worst <= tol else 'FAIL'} максимальное относительное расхождение = "
          f"{worst:.3e} (допуск {tol:.0e}) на поле {worst_p}")
    return 0 if worst <= tol else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", nargs="?")
    ap.add_argument("--stages", default="")
    ap.add_argument("--pair", nargs=2, default=None, metavar=("CPU", "GPU"))
    args = ap.parse_args()
    if args.pair:
        return check_pair(args.pair[0], args.pair[1])
    if not args.artifact:
        ap.error("нужен путь к артефакту или --pair CPU GPU")
    stages = [s for s in args.stages.split(",") if s] or None
    return check_one(args.artifact, stages)


if __name__ == "__main__":
    raise SystemExit(main())
