#!/usr/bin/env bash
# Цепочка хвоста раунда E-45b → E-46: долгая таблица листа идёт первой, за ней два
# потребителя её артефакта. Каждая ступень пишет rc в лог, чтобы обрыв был виден без
# разбора лога самой ступени (в прошлый раз «rc=0 за 8 секунд» означал FileNotFoundError
# первой же строки, и это выглядело как успех).
set -u
cd "$(dirname "$0")"
R=../results
run() {
  echo "[$(date +%H:%M:%S)] $*"
  OMP_NUM_THREADS=8 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 taskset -c 0-7 \
    ../.venv/bin/python "$@"
  echo "rc($1)=$? at $(date +%H:%M:%S)"
}
run exp45b_leaf_table.py
if [ -f "$R/exp45b_leaf_table.json.gz" ]; then
  run exp45b_analysis.py
  run exp46_frame_as_memory.py
  run exp46_verdicts.py
else
  echo "нет exp45b_leaf_table.json.gz — цепочка остановлена (смотреть meta.checks артефакта)"
fi
echo "[$(date +%H:%M:%S)] цепочка пуста"
