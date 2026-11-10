#!/bin/bash
# Смотритель Kaggle: долбит push v7 до сброса недельной GPU-квоты, затем следит
# за сессией, забирает tau-JSON, при FAILED/CANCEL перезаливает. Логи вне /tmp.
K=voroninmichael/prism-exp52k-scale-curve-real-ptq
D=/home/survivor/work/prism/logs
LOG=$D/kaggle_daemon.log
ST=$D/kaggle_daemon.state
[ -f "$ST" ] || echo "PUSHED=0" > "$ST"
say(){ echo "$(date '+%m-%d %H:%M') $*" >> "$LOG"; }
while true; do
  . "$ST"
  if [ "$PUSHED" = "0" ]; then
    out=$(cd /home/survivor/work/prism/kaggle_kernel && kaggle kernels push -p kernel-exp52k 2>&1 | grep -v outdated)
    say "push: ${out:0:90}"
    case "$out" in
      *"successfully pushed"*) echo "PUSHED=1" > "$ST";;
    esac
  elif [ "$PUSHED" = "1" ]; then
    st=$(kaggle kernels status $K 2>/dev/null | grep -oE "COMPLETE|RUNNING|FAILED|CANCEL|QUEUED|PENDING" | tail -1)
    say "status=$st"
    case "$st" in
      COMPLETE)
        rm -rf $D/e52k_v7; mkdir -p $D/e52k_v7
        kaggle kernels output $K -o -p $D/e52k_v7 --file-pattern ".*exp52k_tau_.*\.json$" >/dev/null 2>&1
        n=$(ls $D/e52k_v7/exp52k_tau_*.json 2>/dev/null | wc -l)
        say "забрано tau json: $n"
        if [ "$n" -ge 5 ]; then say "v7 полная — DONE"; echo "PUSHED=DONE" > "$ST"; exit 0; fi
        echo "PUSHED=0" > "$ST";;
      FAILED|CANCEL) echo "PUSHED=0" > "$ST";;
    esac
  fi
  # e51: если процесс модели умер и кэш неполный — один реюзер на модель за цикл
  for m in qwen lfm3; do
    n=$(wc -l < /home/survivor/work/prism/results/exp51_cache_$m.jsonl 2>/dev/null || echo 0)
    if [ "$n" -lt 64 ] && ! pgrep -f "exp51_higher_order.py --model $m" >/dev/null; then
      say "e51 $m мёртв (кэш $n/64) — реюз"
      (cd /home/survivor/work/prism/experiments && setsid nohup nice -n 10 ../.venv/bin/python exp51_higher_order.py --model $m >> /home/survivor/work/prism/logs/e51_$m.log 2>&1 &)
    fi
  done
  sleep 2700
done
