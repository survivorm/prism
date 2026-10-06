#!/usr/bin/env bash
# Wait for P128_RAW injection to finish cleanly, then run full E-34.
set -u
cd /home/survivor/work/prism
INJ_LOG=/tmp/inject_p128_raw.log
E34_LOG=/tmp/exp34_rerun.log

echo "[chain] waiting for injection (log: $INJ_LOG) ..." 
while :; do
  if grep -q "\[OK\] P128_RAW injected" "$INJ_LOG" 2>/dev/null; then
    echo "[chain] injection OK -> launching E-34"
    break
  fi
  # detect a crash: process gone and no OK marker
  if ! pgrep -f _round3_inject_p128_raw > /dev/null 2>&1; then
    echo "[chain] injection process gone; tail:"
    tail -5 "$INJ_LOG"
    echo "[chain] INJECTION_FAILED"
    exit 1
  fi
  sleep 60
done

echo "[chain] running E-34 full ..."
.venv/bin/python experiments/exp34_p1_coupling.py --out results/exp34_p1_coupling.json 2>&1 | tee "$E34_LOG"
rc=${PIPESTATUS[0]}
echo "[chain] E-34 exit=$rc"
exit $rc
