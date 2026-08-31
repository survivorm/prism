#!/bin/bash
# Push the v2 kernels that the user approved: 16r (exp16 readout-pull v2, ~5 min)
# + 17 (exp17 LoRA distillation, 500 steps / lr 3e-3, ~9h). Batch free tier = 2
# concurrent, both slots free now (old versions COMPLETE). exp19 is NOT pushed
# here (not approved yet; it also reads 16r's knee, so it can't run until 16r done).
cd /home/survivor/work/prism/kaggle_kernel || exit 1
log(){ echo "[$(date +%H:%M:%S)] $*"; }
status_of(){ timeout 40 kaggle kernels status "$1" 2>&1 | tail -1 | grep -oE 'KernelWorkerStatus\.[A-Z]+'; }
push(){
  log "pushing $1"
  out=$(kaggle kernels push -p "$1" 2>&1 | grep -viE 'it/s|100%|warning|outdated|deprecat')
  echo "$out" | tail -3
  s=$(status_of "$2"); log "  status: $s"
}
push kernel-exp16r voroninmichael/prism-readout-pull
push kernel-exp17  voroninmichael/prism-lora-distillation
log "DONE: 16r + 17 pushed. (19 held pending approval.)"
