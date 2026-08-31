#!/bin/bash
# Wait for stage0 v11 to COMPLETE (guarantees 17/18 mount the v2 cache with
# input_ids + p_L_full + 31_23 pair), then push 17, then 18 (18 waits for a
# free GPU slot: max 2 concurrent batch GPU sessions on free tier).
cd /home/survivor/work/prism/kaggle_kernel
S0=voroninmichael/exp16-stage-0-target-jacobian-cache
S16=voroninmichael/prism-readout-pull
S17=voroninmichael/prism-exp17-lora-distill

status_of() { timeout 40 kaggle kernels status "$1" 2>&1 | tail -1 | grep -oE 'KernelWorkerStatus\.[A-Z]+'; }
is_active() { case "$1" in *QUEUED*|*RUNNING*) return 0;; *) return 1;; esac; }

# 1) Wait for stage0 v11 to reach a terminal state (COMPLETE or ERROR).
s0=$(status_of "$S0")
while is_active "$s0"; do
  echo "[$(date +%H:%M)] stage0=$s0, waiting..."
  sleep 120
  s0=$(status_of "$S0")
done
echo "[$(date +%H:%M)] stage0 terminal: $s0"
if [[ "$s0" != *"COMPLETE"* ]]; then
  echo "stage0 did not COMPLETE ($s0) — not pushing 17/18. Investigate manually."
  exit 1
fi

# 2) Push 17 (a slot is free: stage0 just completed).
echo "[$(date +%H:%M)] stage0 done — pushing exp17"
uv run python3 push_all.py --exp exp17 2>&1 | grep -E "successfully pushed|push error"

# 3) Push 18 once a slot frees (active among 16r/17 < 2).
sleep 60
while true; do
  active=0
  for id in "$S16" "$S17"; do
    s=$(status_of "$id")
    is_active "$s" && active=$((active+1))
  done
  [ "$active" -lt 2 ] && break
  echo "[$(date +%H:%M)] active=$active, waiting for a slot for exp18..."
  sleep 120
done
echo "[$(date +%H:%M)] slot free — pushing exp18"
uv run python3 push_all.py --exp exp18 2>&1 | grep -E "successfully pushed|push error"
echo "[$(date +%H:%M)] watcher done: 17 + 18 pushed (both after stage0 v11 complete)"
