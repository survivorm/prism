#!/bin/bash
# Waits for a free batch-GPU slot (exp17/exp18 RUNNING), then pushes probe -> exp19 -> exp16r.
cd /home/survivor/work/prism/kaggle_kernel || exit 1
L=voroninmichael/prism-exp17-lora-distill
S=voroninmichael/prism-exp18-logit-kd
log(){ echo "[$(date +%H:%M:%S)] $*"; }
push(){ # $1=dir $2=slug
  log "push $1"
  kaggle kernels push -p "$1" 2>&1 | grep -vE 'it/s|100%|Warning|outdated' | tail -2
}
# Wait until one of the two frees a slot (max 3h)
for i in $(seq 1 180); do
  a=$(kaggle kernels status $L 2>/dev/null | tail -1 | grep -oE 'KernelWorkerStatus\.[A-Z]+')
  b=$(kaggle kernels status $S 2>/dev/null | tail -1 | grep -oE 'KernelWorkerStatus\.[A-Z]+')
  log "exp17=$a exp18=$b"
  if [ "$a" != "KernelWorkerStatus.RUNNING" ] || [ "$b" != "KernelWorkerStatus.RUNNING" ]; then
    log "slot free (exp17=$a exp18=$b)"
    break
  fi
  sleep 60
done
# Push in priority order: probe (answers user's question), then exp19, then exp16r
push kernel-probe-fullvocab prism-full-vocab-jacobian-probe
push kernel-exp19 prism-exp19-destruction
push kernel-exp16r prism-readout-pull
log "DONE — all three pushed (or see errors above)"
