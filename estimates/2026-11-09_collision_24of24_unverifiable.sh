#!/bin/bash
# ПРИКИДКА→РЕШЕНИЕ (inconclusive→решение): утверждение «первая полная карта столкновений
# 24/24» НЕ воспроизводится: commit недостижим, artifact пуст, в логах нет строк прогона.
# Решение: из ТЗ/статей утверждение убрано; тест столкновений — только после
# Triton-ядра E-64f на T4. Воспроизводимость: три greps ниже.
cd "$(dirname "$0")/.."
echo "-- коммит с claim (недостижим):"; git log --all --oneline | grep -iE "collision|24/24" || echo "   (пусто — коммита в истории нет)"
echo "-- артефакт:"; cat results/e64/SOLITON_COLLISION.json
echo "-- строки прогона в живых логах:"; grep -rl "collision" /tmp/*.log 2>/dev/null || echo "   (нет)"
