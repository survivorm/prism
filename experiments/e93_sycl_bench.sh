#!/bin/bash
# e93 SYCL: сравнительный бенч qwen3.5-0.8B (f16/f32, FA on/off) на Iris Xe.
# Сравнивать с прежними числами llama-Vulkan на этой же карте: CPU tg0.92, VK f32 tg5.1, VK f16 tg10.5.
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ZES_ENABLE_SYSMAN=1
B=~/work/llama.cpp/build/bin
OUT=~/work/prism/results/e93/sycl
mkdir -p "$OUT"
for M in /data/Qwen/qwen3.5-0.8B-f16.gguf /data/Qwen/qwen3.5-0.8B-f32.gguf; do
  for FA in on off; do
    echo "== $M fa=$FA =="
    "$B/llama-bench" -m "$M" -ngl 99 -fa "$FA" -sm none -mg 0 -p 512 -n 64 -r 2 -mmp 0 2>/dev/null
  done
done
echo BENCH_DONE
