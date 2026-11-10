#!/bin/bash
# скачать e64-gpu артефакты и прогнать вердикты
set -e
mkdir -p results/e64/tpu
timeout 120 kaggle kernels output voroninmichael/prism-e64-gpu -p results/e64/tpu >/dev/null
.venv/bin/python experiments/exp64_tpu_report.py results/e64/tpu | tail -40
.venv/bin/python experiments/exp64_readmatrix.py results/e64/tpu | tail -8
