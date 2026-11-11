#!/bin/bash
# kaggle_run.sh <script.py> <kernel-slug> <accelerator: none|gpu|tpu> [dataset-ref ...]
# Упаковывает py-скрипт в notebook (1 cell), пушит kernel, ждёт завершения, тянет output
# в results/kaggle_out/<slug>/. После успеха — ОБЯЗАТЕЛЬНО: sha16 в results/ARTIFACTS_MANIFEST.json.
# СОГЛАШЕНИЕ О ПУТЯХ: скрипт читает env PRISM_KAGGLE_ROOT (в kernel'е =/kaggle/working);
# локально — корень репо. Артефакты пишет в <ROOT>/results/... — пути едины на обеих сторонах.
set -euo pipefail
SRC=$1; SLUG=$2; ACC=${3:-none}; shift 3 || true
CODE=${KAGGLE_CODE:-voroninmichael/$SLUG}
KDIR=$(mktemp -d); trap 'rm -rf "$KDIR"' EXIT
python3 - "$SRC" "$KDIR" "$ACC" "$CODE" "$SLUG" "$@" <<'PY'
import json,sys
src,kdir,acc,code,slug,*ds=sys.argv[1:]
nb={"cells":[{"cell_type":"code","metadata":{},"execution_count":None,
     "source":open(src).read().splitlines(keepends=True)}],
 "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
   "accelerator":acc if acc!="none" else "None",
   "dataSources":[{"datasetRef":d} for d in ds],
   "envVars":[{"key":"PRISM_KAGGLE_ROOT","value":"/kaggle/working"}]},
 "nbformat":4,"nbformat_minor":5}
json.dump(nb,open(kdir+"/kernel.ipynb","w"))
json.dump({"id":code,"title":"Prism: "+slug,"code_file":"kernel.ipynb","language":"python",
 "kernel_type":"notebook","is_private":True,
 "enable_gpu":acc=="gpu","enable_tpu":acc=="tpu","enable_internet":True,
 "dataset_sources":ds,"competition_sources":[],"kernel_sources":[],"model_sources":[]},
 open(kdir+"/kernel-metadata.json","w"))
PY
kaggle kernels push -p "$KDIR"
echo "pushed $CODE — polling…"
for i in $(seq 1 480); do
  st=$(kaggle kernels status "$CODE" || true)
  case "$st" in *ompleted*|*COMPLETE*) break;; *Failed*|*CANCEL*|*Stopped*) echo "KERNEL_BAD: $st"; exit 2;; esac
  sleep 30
done
OUT=results/kaggle_out/$SLUG; mkdir -p "$OUT"
kaggle kernels output "$CODE" -p "$OUT" >/dev/null
find "$OUT" -type f | sed 's/^/artifact: /'
echo "KAGGLE_RUN_DONE $CODE — не забыть sha16 в results/ARTIFACTS_MANIFEST.json"
