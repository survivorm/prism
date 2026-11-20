#!/usr/bin/env bash
# Агрегатор гейтов G1/G2/G6 (итерация 1). Бюджет по ТЗ §3.4: G1 <= 60 c, G2 <= 30 c, G6 <= 20 c,
# вместе <= 120 c. Каждый гейт печатает своё фактическое время и возвращает свой RC; агрегатор
# возвращает 1, если упал хотя бы один. Ни сети, ни GPU, ни весов.
set -u
cd "$(dirname "$0")/.." || exit 2
PY_BIN="${PY_BIN:-.venv/bin/python}"
[ -x "$PY_BIN" ] || PY_BIN=python3
start=$(date +%s)
rc=0
for gate in collect paths artifacts; do
  echo "--- gate_${gate}"
  "$PY_BIN" "tools/gate_${gate}.py" || rc=1
done
echo "gate_all: total=$(( $(date +%s) - start ))s rc=$rc"
exit $rc
