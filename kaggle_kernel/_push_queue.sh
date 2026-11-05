#!/bin/bash
# Очередь GPU-перегенераций (docs/regen_queue.md). Слот: Kaggle держит 2 одновременные
# GPU-сессии на аккаунт, поэтому пушим по одному, как только освобождается слот.
# Убитый ранее сторож: exp3 -> exp2a; здесь добавлен exp48b (аудит E46+ A1).
cd /home/survivor/work/prism/kaggle_kernel || exit 1
QUEUE=(kernel-exp3 kernel-exp2a kernel-exp48b)
WATCH=(voroninmichael/e-39m-flat-carrier-vs-ptq voroninmichael/prism-exp3-ablation \
       voroninmichael/prism-exp2a-cka-identity voroninmichael/prism-exp48b-lfm3-gen-kinematics)
log() { echo "[$(date +%H:%M:%S)] $*"; }
st() { kaggle kernels status "$1" 2>/dev/null | tail -1 | grep -oE 'KernelWorkerStatus\.[A-Z]+'; }
push() { log "push $1"; kaggle kernels push -p "$1" 2>&1 | grep -viE 'it/s|100%|outdated' | tail -2; sleep 15; }

for i in $(seq 1 480); do
  running=0
  state=""
  for k in "${WATCH[@]}"; do
    s=$(st "$k"); state="$state ${k##*/}=${s#KernelWorkerStatus.}"
    [ "$s" = "KernelWorkerStatus.RUNNING" ] && running=$((running + 1))
  done
  log "running=$running$state"
  if [ ${#QUEUE[@]} -gt 0 ] && [ "$running" -lt 2 ]; then
    nxt=${QUEUE[0]}
    QUEUE=("${QUEUE[@]:1}")
    push "$nxt"
    continue
  fi
  [ ${#QUEUE[@]} -eq 0 ] && { log "очередь пуста"; break; }
  sleep 60
done
log "watcher exit"
