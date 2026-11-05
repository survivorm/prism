#!/usr/bin/env bash
# Хвост цепочки: артефакт E-45b уже на диске, остались два его потребителя + вердикты.
set -u
cd "$(dirname "$0")"
R=../results
run() {
  echo "[$(date +%H:%M:%S)] $*"
  OMP_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 taskset -c 0-7 \
    ../.venv/bin/python "$@"
  echo "rc($1)=$? at $(date +%H:%M:%S)"
}
run exp45b_analysis.py
run exp46_frame_as_memory.py
run exp46_verdicts.py
echo "[$(date +%H:%M:%S)] цепочка пуста"
