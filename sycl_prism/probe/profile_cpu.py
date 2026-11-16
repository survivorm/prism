"""Неинтрузивный профилировщик длинных CPU-прогонов (cProfile + top-N).

Зачем. Из `0_EXPER_DURATIONS.md`: E-45 (≥6 ч 18 м, CPU) идёт **без разбивки по этапам**
(«собственный лог прогона не сохранён»), E-32r2 (1 ч 42 м) — тоже. Прежде чем что-то
переносить на iGPU, надо знать, **какой именно этап** жрёт часы. Этот скрипт даёт ответ
без правки исходников: оборачивает запуск в cProfile и печатает top-N по кумулятивному
времени (функции) и по числу вызовов.

Использование:
    python3 profile_cpu.py --script experiments/exp45_damage_vs_structure.py \
        --script-args "--limit 2" --top 25

    # полный профиль в файл (для pstats / snakeviz):
    python3 profile_cpu.py --script ... --out /tmp/e45.prof

    # ограничить по времени (для длинных прогонов):
    python3 profile_cpu.py --script ... --timeout 900

Замечание. cProfile даёт накладные расходы (обычно 1.5–3× на уровне вызовов Python) ⇒
абсолютные времена завышены, но ** ранжирование этапов остаётся верным** — именно оно и нужно.
Для точных абсолютных чисел используйте второй режим: `--no-profile` не трогает код, но
в скрипте полезно добавить таймеры по этапам (см. записку, §5.6).
"""
from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import runpy
import signal
import sys
import time


class _Timeout(Exception):
    pass


def _alarm(signum, frame):  # noqa: ARG001
    raise _Timeout()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True)
    ap.add_argument("--script-args", default="")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=int, default=0, help="секунд; 0 = без лимита")
    ap.add_argument("--cwd", default="")
    args = ap.parse_args()

    script = os.path.abspath(args.script)
    if args.cwd:
        os.chdir(args.cwd)
    elif os.path.isdir(os.path.join(os.path.dirname(script), "experiments")):
        os.chdir(os.path.join(os.path.dirname(script), "experiments"))

    sys.argv = [script] + (args.script_args.split() if args.script_args else [])
    sys.path.insert(0, os.path.dirname(script))

    if args.timeout:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(args.timeout)

    print(f"запуск: {script}")
    print(f"args  : {' '.join(sys.argv[1:]) or '<нет>'}")
    print(f"cwd   : {os.getcwd()}")
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

    st = pstats.Stats(pr)
    if args.out:
        st.dump_stats(args.out)
        print(f"профиль сохранён: {args.out}")

    print("\n=== TOP-N по кумулятивному времени (cumulative) ===")
    st.sort_stats("cumulative")
    st.print_stats(args.top)
    print("=== TOP-N по числу вызовов (ncalls) ===")
    st.sort_stats("ncalls")
    st.print_stats(min(args.top, 15))
    print("\nПодсказка: 'tottime' — время В САМОЙ функции; 'cumtime' — включая вложенные вызовы.")
    print("Ищите строки с большим cumtime, у которых tottime тоже велик — это и есть горячий этап.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
