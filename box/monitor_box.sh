#!/bin/bash
# Poll the box kernel; log transitions to box/box_monitor.log. Run in background:
#   setsid nohup bash box/monitor_box.sh >/dev/null 2>&1 &   (or foreground)
cd /home/survivor/work/prism
K=voroninmichael/prism-box-autonomer
LOG=box/box_monitor.log
echo "[$(date '+%F %T')] start monitoring $K" > "$LOG"
last=""
for i in $(seq 1 1200); do
  st=$(/usr/bin/kaggle kernels status "$K" 2>&1 | grep -oE "KernelWorkerStatus\.[A-Z]+" | head -1)
  if [ -z "$st" ]; then st="UNKNOWN"; fi
  if [ "$st" != "$last" ]; then
    echo "[$(date '+%F %T')] $i: $st" >> "$LOG"
    # on terminal state, grab the tail of the log (the [OK]/[FAIL] verdict + coexistence probe result)
    case "$st" in
      KernelWorkerStatus.COMPLETE|KernelWorkerStatus.ERROR|KernelWorkerStatus.CANCELACKNOWLEDGED)
        echo "[$(date '+%F %T')] TERMINAL=$st — grabbing log" >> "$LOG"
        /usr/bin/kaggle kernels output "$K" -o -p box/box_output --file-pattern "\.log$" >> "$LOG" 2>&1
        echo "[$(date '+%F %T')] log saved to box/box_output/" >> "$LOG"
        break ;;
    esac
    last="$st"
  fi
  sleep 300
done
echo "[$(date '+%F %T')] monitor exit" >> "$LOG"
