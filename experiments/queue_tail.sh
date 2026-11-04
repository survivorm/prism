#!/bin/bash
# Сторож хвоста раунда: когда очередь освободится, добить iid-семейство и пересобрать вердикты.
# Нужен потому, что queue_round.sh был отредактирован НА ХОДУ (append перед последней строкой) —
# bash мог закинуть старый хвост в буфер и новые шаги не увидеть. Проверка по метке в логе
# очереди делает повторный запуск безопасным: если очередь их всё-таки выполнила, сторож молчит.
cd /home/survivor/work/prism/experiments || exit 1
R=/home/survivor/work/prism/results

while pgrep -f "queue_roun[d].sh" >/dev/null 2>&1 \
      || pgrep -f "venv/bin/python exp4[56]" >/dev/null 2>&1; do sleep 60; done

if grep -q "rc(seednull)" "$R/queue_round.log" 2>/dev/null; then
    echo "[watch] $(date +%H:%M:%S) очередь сама сделала seednull — выхожу"
    exit 0
fi
echo "[watch] $(date +%H:%M:%S) iid-семейство (8 draw'ов) + повторные вердикты E-45"
E45_SEEDNULL=8 ../.venv/bin/python exp45_damage_vs_structure.py --resume > "$R/exp45_seednull.log" 2>&1
echo "[watch] $(date +%H:%M:%S) rc(seednull)=$?"
../.venv/bin/python exp45_verdicts.py > "$R/exp45_verdicts.log" 2>&1
echo "[watch] $(date +%H:%M:%S) rc(verdicts)=$?"
