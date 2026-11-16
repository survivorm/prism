#!/bin/bash
# e93 SYCL smoke: qwen3.5-0.8B f16 на Iris Xe через level-zero бэкенд llama.
# Проверяем: (1) все слои на GPU, (2) /dev/dri fd у процесса во время работы,
# (3) tg/pp цифры против прежних Vulkan-замеров (5.1/10.5 t/s tg).
# set-флагов нет: setvars.sh живёт только в чистом окружении (проверено руками)
source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export ZES_ENABLE_SYSMAN=1
B=~/work/llama.cpp/build/bin
M=/data/Qwen/qwen3.5-0.8B-f16.gguf
OUT=~/work/prism/results/e93/sycl
mkdir -p "$OUT"
echo "== llama-completion -ngl 99 -sm none -mg 0, greedy, 256 токенов =="
"$B/llama-completion" -m "$M" -p "Q: Name the three largest planets in the solar system. A: 1." \
  -n 256 -e -ngl 99 -sm none -mg 0 --load-mode auto -fa on --temp 0 -s 1 -t 4 \
  >"$OUT/smoke_gen.txt" 2>"$OUT/smoke_err.log" &
CP=$!
sleep 20
echo "-- fds /dev/dri у pid=$CP --" | tee "$OUT/fd_check.txt"
ls -l /proc/$CP/fd 2>/dev/null | grep -m3 dri >>"$OUT/fd_check.txt" || echo "NO_DRI_FD (возможно ze через sysfs; см. dmesg/device idx)" >>"$OUT/fd_check.txt"
wait $CP; echo "cli_rc=$?"
grep -aE "SYCL|offloaded|load time|eval time|prompt eval" "$OUT/smoke_err.log" | tail -8
echo "== mini-bench pp512/tg64 =="
"$B/llama-bench" -m "$M" -ngl 99 -fa on -sm none -mg 0 -p 512 -n 64 -r 2 -mmp 0 \
  >"$OUT/bench_sycl.txt" 2>"$OUT/bench_err.log"
echo SMOKE_DONE
