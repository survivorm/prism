#!/bin/bash
# Очередь раунда E-45/E-46: ждать окончания E-45, затем прогнать остаток по очереди.
# Каждый шаг независим (ошибка одного не останавливает остальные), логи — в results/.
cd /home/survivor/work/prism/experiments || exit 1
R=/home/survivor/work/prism/results

echo "[queue] $(date +%H:%M:%S) ждём окончания exp45_damage_vs_structure"
# шаблон подбираем так, чтобы не поймать сам скрипт очереди
while pgrep -f "venv/bin/python exp45_damage_vs_structure.py" >/dev/null 2>&1; do sleep 30; done
echo "[queue] $(date +%H:%M:%S) E-45 завершён"

echo "[queue] $(date +%H:%M:%S) E-45c (калибровочный нуль)"
../.venv/bin/python exp45c_gauge_null.py > "$R/exp45c_gauge_null.log" 2>&1
echo "[queue] $(date +%H:%M:%S) rc(E-45c)=$?"

echo "[queue] $(date +%H:%M:%S) E-45b (таблица лист↔поведение)"
../.venv/bin/python exp45b_leaf_table.py > "$R/exp45b_leaf_table.log" 2>&1
echo "[queue] $(date +%H:%M:%S) rc(E-45b)=$?"

echo "[queue] $(date +%H:%M:%S) вердикты E-45"
../.venv/bin/python exp45_verdicts.py > "$R/exp45_verdicts.log" 2>&1
echo "[queue] $(date +%H:%M:%S) rc(verdicts)=$?"

echo "[queue] $(date +%H:%M:%S) анализ E-45b"
../.venv/bin/python exp45b_analysis.py > "$R/exp45b_analysis.log" 2>&1
echo "[queue] $(date +%H:%M:%S) rc(analysis)=$?"

echo "[queue] $(date +%H:%M:%S) E-46 (лист как носитель)"
../.venv/bin/python exp46_frame_as_memory.py > "$R/exp46_frame_as_memory.log" 2>&1
echo "[queue] $(date +%H:%M:%S) rc(E-46)=$?"

echo "[queue] $(date +%H:%M:%S) вердикты E-46"
../.venv/bin/python exp46_verdicts.py > "$R/exp46_verdicts.log" 2>&1
echo "[queue] $(date +%H:%M:%S) rc(E-46 verdicts)=$?"
echo "[queue] $(date +%H:%M:%S) очередь пуста"
