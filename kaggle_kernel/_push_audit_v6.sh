#!/bin/bash
# Аудит v6: очередь GPU-перегенераций. Ждём слот (E-39m + exp01), потом exp3 -> exp2a.
cd /home/survivor/work/prism/kaggle_kernel || exit 1
A=voroninmichael/e-39m-flat-carrier-vs-ptq
B=voroninmichael/prism-exp01-coherence
log(){ echo "[$(date +%H:%M:%S)] $*"; }
st(){ kaggle kernels status "$1" 2>/dev/null | tail -1 | grep -oE 'KernelWorkerStatus\.[A-Z]+'; }
push(){ log "push $1"; kaggle kernels push -p "$1" 2>&1 | grep -viE 'it/s|100%|outdated' | tail -2; sleep 10; }

for i in $(seq 1 240); do
  a=$(st $A); b=$(st $B)
  log "e39m=$a exp01=$b"
  if [ "$a" != "KernelWorkerStatus.RUNNING" ] && [ "$b" != "KernelWorkerStatus.RUNNING" ]; then
    log "оба слота свободны"
    break
  fi
  sleep 60
done
push kernel-exp3
sleep 30
push kernel-exp2a
log "DONE: exp3 + exp2a запушены"
