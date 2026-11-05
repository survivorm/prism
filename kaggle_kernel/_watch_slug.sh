#!/usr/bin/env bash
# Общий сторож Kaggle-ядра: ждёт ухода из RUNNING, забирает вывод в каталог.
# COMPLETE у этих кернелов успехом не считать — лог читает человек (правило PLAN).
# Usage: _watch_slug.sh <slug> <outdir> [interval_s]
set -u
SLUG="${1:?slug}"; OUT="${2:?outdir}"; INT="${3:-600}"
NAME="$(basename "$SLUG")"
LOG="/tmp/watch_${NAME}.log"
PIDF="/tmp/watch_${NAME}.pid"
# pgrep -f матчит собственную командную строку, поэтому страж — только pid-file.
if [[ -f $PIDF ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "[watch] уже запущен (pid $(cat "$PIDF"))" >> "$LOG"; exit 0
fi
echo $$ > "$PIDF"
trap 'rm -f "$PIDF"' EXIT
mkdir -p "$OUT"
echo "[watch] старт $(date +%H:%M:%S) $SLUG -> $OUT" >> "$LOG"
while :; do
  ST="$(timeout 60 kaggle kernels status "$SLUG" 2>&1 | tail -1)"
  case "$ST" in
    *RUNNING*|*QUEUED*|*PENDING*|*STARTING*) sleep "$INT" ;;
    *)
      echo "[watch] $(date +%H:%M:%S) финиш: $ST" >> "$LOG"
      timeout 900 kaggle kernels output "$SLUG" -p "$OUT" >> "$LOG" 2>&1
      echo "[watch] вывод в $OUT: $(find "$OUT" -type f | wc -l) файлов" >> "$LOG"
      break
      ;;
  esac
done
