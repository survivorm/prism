#!/usr/bin/env bash
# run_one.sh <name> <cmd...> — ОДИН запуск одного эксперимента: pidfile-flock,
# ожидание памяти (avail>=10G), тред-лимит, лог/RC в /tmp/<name>.log. Выход 0 если
# уже запущен (BUSY) — повторный лонч физически невозможен.
set -u
name="$1"; shift
lockf="/tmp/run_one_${name}.lock"
exec 9>"$lockf"
if ! flock -n 9; then echo "BUSY: $name уже запущен (pid $(cat /tmp/run_one_${name}.pid 2>/dev/null))"; exit 0; fi
for i in $(seq 1 240); do
  A=$(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$A" -ge 10 ] && break
  echo "WAIT mem: avail=${A}G <10G (poll 60s)"; sleep 60
done
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
: > "/tmp/${name}.log"
setsid nohup bash -c "cd /home/survivor/work/prism && nice -n 10 $* >> /tmp/${name}.log 2>&1; echo RUN_RC=\$? >> /tmp/${name}.log; rm -f /tmp/run_one_${name}.pid" </dev/null >/dev/null 2>&1 &
echo $! > /tmp/run_one_${name}.pid
echo "LAUNCHED $name wrapper=$(cat /tmp/run_one_${name}.pid) log=/tmp/${name}.log"
