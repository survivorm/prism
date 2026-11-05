#!/usr/bin/env bash
# Ждёт свободный GPU-слот Kaggle (лимит 2 одновременные сессии) и пуляет ядро.
# Нужен потому, что `kaggle kernels push` при полном лимите падает с
# "Maximum batch GPU session count of 2 reached", а `cancel` в kaggle 2.1.2 нет.
#
# Usage: _push_when_slot_free.sh <push_all-эксп> [макс. минут ожидания]
set -uo pipefail
EXP="${1:?usage: $0 <exp> [max_min]}"
MAX_MIN="${2:-240}"
cd "$(dirname "$0")"
PY=/home/survivor/work/prism/.venv/bin/python
end=$(( $(date +%s) + MAX_MIN * 60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  out=$("$PY" push_all.py --exp "$EXP" 2>&1)
  if grep -q "successfully pushed" <<<"$out"; then
    echo "[push-when-free] $EXP: $(grep -o 'Kernel version [0-9]* successfully pushed' <<<"$out" | head -1)"
    exit 0
  fi
  if grep -q "Maximum batch GPU session count" <<<"$out"; then
    echo "[push-when-free] $EXP: слотов нет ($(date -u +%H:%M:%S)), повтор через 120 с"
  else
    echo "[push-when-free] $EXP: другая ошибка:"; grep -viE "outdated|consider upgrading" <<<"$out" | tail -6
  fi
  sleep 120
done
echo "[push-when-free] $EXP: так и не дождались слота за ${MAX_MIN} мин"
exit 1
