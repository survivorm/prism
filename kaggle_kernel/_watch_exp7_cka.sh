#!/usr/bin/env bash
# Мониторинг R14 (E-7 на настоящем CKA): ждём финиша ядра и сразу забираем вывод.
# COMPLETE у этих кернелов успехом не считать — лог читает человек (правило PLAN).
set -u
SLUG="voroninmichael/prism-exp7-cka-baseline"
OUT="/tmp/e7_done"
LOG="/tmp/watch_e7.log"
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
      break
      ;;
    *) sleep 300 ;;
  esac
done
