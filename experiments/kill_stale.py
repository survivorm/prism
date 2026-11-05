#!/usr/bin/env python3
"""kill_stale.py — убить зависшие прогоны exp51 (python-only), без самоубийств."""
import os, signal, sys
pat = sys.argv[1] if len(sys.argv) > 1 else "exp51"
me = os.getpid()
ppid = os.getppid()
for p in os.listdir("/proc"):
    if not p.isdigit() or int(p) in (me, ppid):
        continue
    try:
        c = open(f"/proc/{p}/cmdline", "rb").read().decode(errors="ignore").replace("\0", " ")
    except Exception:
        continue
    if pat in c and "/.venv/bin/python" in c:
        print("kill", p, c[:80])
        try:
            os.kill(int(p), signal.SIGTERM)
        except ProcessLookupError:
            pass
