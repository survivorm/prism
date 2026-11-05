#!/usr/bin/env bash
# Хвост раунда E-45: дожидается основного прогона и доделывает то, что в нём не помещалось.
#   1) E45_SEEDNULL=8 --resume — 8 draw'ов того же семейства при дрейфе 0.12: это полоса
#      «разброс одного draw'а», без неё знак Δκ и величина gap16 у rand_* нечитаемы
#      (строк seednull_* в текущем JSON нет). Строки kind=sparse, кадровой дырой не задеты.
#   2) сводный вердикт exp45_verdicts.py по смёрженному результату.
# Повторы flat-правок со сдвигом seed в хвост НЕ включены: seed зашит в код (manual_seed(0)),
# а кэш flat-носителя общий на все seed'ы, так что сначала надо разнести кэш по seed'ам.
# Шаг 1 обязан проверять по метке в логе, а не по наличию процесса: pgrep ловит этот же скрипт.
set -uo pipefail
cd "$(dirname "$0")"
LOG=../results/exp45_damage_vs_structure.log
MAIN=../results/exp45_damage_vs_structure.json
FIN=../results/exp45_finish.log

while ! grep -q "сохранено: $MAIN" "$LOG" 2>/dev/null; do
  if ! pgrep -f "exp45_damage_vs_structure.py" > /dev/null; then
    echo "[tail] $(date +%H:%M:%S) основного прогона нет, метки завершения нет — выходить, см. $LOG"
    exit 1
  fi
  sleep 120
done
echo "[tail] $(date +%H:%M:%S) основной прогон завершён"

if grep -q "seednull_s0" "$FIN" 2>/dev/null; then
  echo "[tail] iid-семейство уже посчитано — пропускаю"
else
  echo "[tail] $(date +%H:%M:%S) E45_SEEDNULL=8 --resume (8 draw'ей при дрейфе 0.12)"
  E45_SEEDNULL=8 OMP_NUM_THREADS=4 ../.venv/bin/python exp45_damage_vs_structure.py --resume \
    >> "$FIN" 2>&1 || echo "[tail] iid-проход упал, см. $FIN"
fi

echo "[tail] $(date +%H:%M:%S) сводный вердикт"
OMP_NUM_THREADS=4 ../.venv/bin/python exp45_verdicts.py > ../results/exp45_verdicts.log 2>&1 \
  || echo "[tail] exp45_verdicts.py упал, см. ../results/exp45_verdicts.log"
echo "[tail] $(date +%H:%M:%S) готово"
