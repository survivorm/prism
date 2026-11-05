#!/usr/bin/env bash
# Хвост раунда E-45 после починки двух баг'ов:
#  1) exp45c — дописать измеряемый κ(J) на калибровке (первый вариант писал в строку базовое κ)
#     и поднять допуск κ(QJ) до float32-пола document_profile (замер 2.7e-05 вместо жидкого 1e-6);
#  2) iid-полоса — σ draw'ей калибровался от J самого шума вместо правки W+G, из-за чего фактический
#     дрейф вышел 0.0855 вместо заказанных 0.12, то есть полоса была не на том дрейфе, что rand_*.
# Порядок жёсткий: всё по очереди и по 4 ядрам — два процесса по 12 потоков MKL живлят друг друга
# (замер: правка шла 46 с на 4 ядрах против 75 минут под 12+12).
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0
CPU="taskset -c 4-7"
PY=../.venv/bin/python

echo "[tail] $(date +%H:%M:%S) exp45c (gauge-null, правка допусков + κ(J) по правке)"
$CPU $PY exp45c_gauge_null.py > ../results/exp45c_gauge_null.log 2>&1
echo "rc(exp45c)=$?"

echo "[tail] $(date +%H:%M:%S) вырезать seednull_* из partial"
$PY exp45_strip_seednull.py
echo "rc(strip)=$?"

echo "[tail] $(date +%H:%M:%S) iid-полоса заново (E45_SEEDNULL=8, σ от правки W+G)"
echo "=== перезапуск с исправленной калибровкой σ, $(date +%H:%M) ===" >> ../results/exp45_seednull.log
E45_SEEDNULL=8 $CPU $PY exp45_damage_vs_structure.py --resume >> ../results/exp45_seednull.log 2>&1
echo "rc(seednull)=$?"

echo "[tail] $(date +%H:%M:%S) вердикты"
$CPU $PY exp45_verdicts.py > ../results/exp45_verdicts.log 2>&1
echo "rc(verdicts)=$?"
echo "[tail] $(date +%H:%M:%S) готово"
