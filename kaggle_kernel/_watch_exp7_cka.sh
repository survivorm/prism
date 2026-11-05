#!/usr/bin/env bash
# Мониторинг R14 (E-7 на настоящем CKA): ждём финиша ядра и сразу забираем вывод.
# COMPLETE у этих кернелов успехом не считать — лог читает человек (правило PLAN).
set -u
SLUG="voroninmichael/prism-exp7-cka-baseline"
OUT="/tmp/e7_done"
LOG="/tmp/watch_e7.log"
cd "$(dirname "$0")/.." || exit 1   # пуш ищет kaggle_kernel/ относительно корня репо
# единственный экземпляр: два сторожа = два пуша exp27c
exec 9>/tmp/watch_e7.lock
flock -n 9 || { echo "[watch] уже запущен, выхожу" >>"$LOG"; exit 0; }
mkdir -p "$OUT"
echo "[watch] старт $(date -u +%H:%M:%S)" >>"$LOG"
while :; do
  st=$(timeout 90 kaggle kernels status "$SLUG" 2>&1 | tail -1)
  case "$st" in
    *RUNNING*|*QUEUED*|*NO_STATUS*|*404*|*Error*) sleep 300 ;;
    *COMPLETE*|*CANCEL*|*ERROR*|*STANDARD_ERROR*)
      echo "[watch] финиш $(date -u +%H:%M:%S): $st" >>"$LOG"
      timeout 600 kaggle kernels output "$SLUG" -p "$OUT" -o >>"$LOG" 2>&1
      echo "[watch] вывод в $OUT:" >>"$LOG"
      find "$OUT" -type f -printf '  %s\t%p\n' >>"$LOG" 2>&1
      # Решение пользователя 05.11: перезaпуск exp27c (R15) — СРАЗУ ОСВОБОДИВШИЙСЯ слот, то есть
      # после этого финиша. Пушим только если вывод exp7 забран (иначе мы не узнаем R14).
      if [ -n "$(find "$OUT" -type f -name '*.log' -print -quit)" ]; then
        echo "[watch] пуш exp27c (R15) $(date -u +%H:%M:%S)" >>"$LOG"
        timeout 600 kaggle kernels push -p kaggle_kernel/kernel-exp27c >>"$LOG" 2>&1
      else
        echo "[watch] вывода exp7 нет, exp27c НЕ пушаю — разобраться вручную" >>"$LOG"
      fi
      break
      ;;
    *) sleep 300 ;;
  esac
done
