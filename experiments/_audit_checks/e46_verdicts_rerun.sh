#!/usr/bin/env bash
# Аудит E-46: перепуск exp46_verdicts.py на ЕГО артефакте (путь чинится в памяти, файл не правится).
set -eu
cd "$(dirname "$0")/.."
exec ../.venv/bin/python - <<'PY'
import os, sys
sys.path.insert(0, "e41_50")
import exp46_verdicts as m
m.RES = os.path.join("..", "results", "exp46_frame_as_memory.json.gz")
m.main()
PY
