"""Профилировщик с РАЗБИВКОЙ ПО ЭТАПАМ — то, чего у E-45 нет (6 ч 18 м без атрибуции).

Идея. cProfile даёт плоский список функций. Нам нужен ответ на вопрос уровня стадии:
«сколько времени ушло на GEMM по словарю, сколько на fp64-eigh (κ), сколько на форвард».
Поэтому функциям сопоставляется стадия, а время суммируется **по tottime** (время В САМОЙ
функции) — это не даёт двойного счёта при вложенности. cumtime печатается рядом для контекста.

Почему это решает задачу: переносить на iGPU имеет смысл только для стадий из
`protocol/protocol_v1.json` со строкой device=«gpu_ok». Скрипт печатает эту пометку
рядом с долей времени ⇒ сразу видно, какой процент работы вообще переносим, а какой
навсегда остаётся на CPU (fp64 eigh, autograd).

Запуск:
    python3 profile_stages.py --script experiments/exp45_damage_vs_structure.py \
        --script-args "--limit 2" --timeout 900
    python3 profile_stages.py --script ... --map my_map.json --out /tmp/e45.prof

Ограничения (честно): cProfile добавляет 1.5–3× накладных ⇒ абсолютные времена завышены,
но ДОЛИ и ранжирование верны. Для абсолютных чисел — таймеры по этапам в самом коде
(шаг 5.6 плана).
"""
from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import runpy
import signal
import sys
import time
from collections import defaultdict

# --- карта стадий по умолчанию: (подстрока имени функции) -> (стадия, вердикт) ---
DEFAULT_MAP = {
    # ---- поведение / E-45 ----
    "base_pre":      ("GEMM по словарю + logsumexp", "GPU: да (fp32)"),
    "dense_metrics": ("метрики правки (KL/TV/flip)", "GPU: да (fp32-часть)"),
    "sparse_metrics":("разреженные метрики правки", "GPU: частично"),
    "kappa_of":      ("κ: голономия топ-r (fp64 eigh)", "CPU: НЕ переносить"),
    "drift_med":     ("‖ΔJ‖/‖J‖ (fp64)", "CPU: НЕ переносить"),
    "spectral_flags":("спектральные флаги (eigh fp64)", "CPU: НЕ переносить"),
    "make_j_fn":     ("Якобиан J (зависит от J-функции)", "T4: не переносить"),
    # ---- прибор / AT0 ----
    "dcurve":        ("кривая прибора", "GPU: да (SYCL)"),
    "vpath":         ("касательные прибора", "GPU: да (SYCL)"),
    "acf_moments":   ("ACF прибора", "GPU: да (SYCL)"),
    "_perm_stat":    ("перестановочный null-слой", "GPU: да (SYCL)"),
    "null_permutations": ("нулевой слой целиком", "GPU: да (SYCL)"),
    "paired_bootstrap":  ("бутстрап по документам", "GPU: да"),
    "resolution_floor":  ("resolution floor (ресемплирование)", "GPU: да"),
    # ---- модель ----
    "forward":       ("форвард модели", "GPU: да (prefill)"),
    "hidden_states": ("извлечение hidden states", "GPU: да"),
}


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _Timeout()


def load_map(path: str | None):
    if not path:
        return dict(DEFAULT_MAP)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: (v[0], v[1]) if isinstance(v, (list, tuple)) else (v, "")
            for k, v in raw.items()}


def classify(funcname: str, stage_map):
    base = os.path.basename(funcname)
    for key, val in stage_map.items():
        if key in base or key in funcname:
            return key, val
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--script-args", default="")
    ap.add_argument("--map", default="", help="JSON: {имя функции: [стадия, вердикт]}")
    ap.add_argument("--timeout", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    script = os.path.abspath(args.script)
    sys.argv = [script] + (args.script_args.split() if args.script_args else [])
    sys.path.insert(0, os.path.dirname(script))

    stage_map = load_map(args.map)
    print(f"скрипт : {script}")
    print(f"args   : {' '.join(sys.argv[1:]) or '<нет>'}")
    print(f"стадий в карте: {len(stage_map)}")

    if args.timeout:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(args.timeout)

    t0 = time.perf_counter()
    pr = cProfile.Profile()
    try:
        pr.enable()
        runpy.run_path(script, run_name="__main__")
    except _Timeout:
        print(f"\n[таймаут {args.timeout} с — профиль частичный]")
    except SystemExit as e:
        print(f"\n[скрипт завершился с кодом {e.code}]")
    except Exception as e:  # noqa: BLE001
        print(f"\n[скрипт упал: {e!r}]")
    finally:
        pr.disable()
        if args.timeout:
            signal.alarm(0)
    dt = time.perf_counter() - t0
    print(f"\nвремя под профилем: {dt/60:.1f} мин")

    if args.out:
        pstats.Stats(pr).dump_stats(args.out)
        print(f"профиль сохранён: {args.out}")

    st = pstats.Stats(pr)
    stage_tt = defaultdict(float)
    stage_ct = defaultdict(float)
    stage_calls = defaultdict(int)
    unattributed = 0.0
    for (fn, lineno, name), (cc, nc, tt, ct, callers) in st.stats.items():
        key, _val = classify(name, stage_map)
        if key is None:
            unattributed += tt
            continue
        stage_tt[key] += tt
        stage_ct[key] += ct
        stage_calls[key] += nc

    total = sum(stage_tt.values()) + unattributed
    print("\n" + "=" * 100)
    print("РАЗБИВКА ПО ЭТАПАМ (tottime = эксклюзивное время; без двойного счёта)")
    print("=" * 100)
    print(f"{'стадия':<34}{'функция':<20}{'эксклюзив, с':>12}{'доля':>8}   переносимость")
    print("-" * 100)
    for key, tt in sorted(stage_tt.items(), key=lambda kv: -kv[1]):
        label, verdict = stage_map.get(key, (key, ""))
        share = 100.0 * tt / total if total else 0.0
        print(f"{label[:33]:<34}{key[:19]:<20}{tt:12.1f}{share:7.1f}%   {verdict}")
    print("-" * 100)
    if unattributed > 0:
        print(f"{'(не отнесено к стадиям)':<34}{'':<20}{unattributed:12.1f}"
              f"{100.0*unattributed/total if total else 0:7.1f}%")
    print(f"{'ИТОГО':<54}{total:12.1f}{'100.0%':>8}")

    print("\n" + "=" * 100)
    print("TOP-N функций по cumtime (для контекста; вложенность ⇒ суммы больше 100%)")
    print("=" * 100)
    st.sort_stats("cumulative")
    st.print_stats(args.top)
    print("\nКак читать: доля в таблице этапов — это доля РАБОТЫ, а не вызовов. "
          "Если доля переносимых стадий < 20–30 %, перенос на iGPU не окупится даже "
          "при нулевом ускорении этой части.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
