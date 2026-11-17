#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
using Clock = std::chrono::steady_clock;

/* 1) FMA-потолок: только регистры. Чек стабильности: 2 разные ITERS. */
template <int NACC, int ITERS>
static double bench_fma(sycl::queue& q, size_t nthreads) {
  float* out = sycl::malloc_device<float>(nthreads, q);
  auto kern = [&]() {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(nthreads, [=](sycl::id<1> i) {
        float a[NACC];
        for (int j = 0; j < NACC; ++j) a[j] = 1.0f + 1e-6f * float(i[0] + j);
        float b[NACC];
        for (int j = 0; j < NACC; ++j) b[j] = 1.0000001f + 1e-7f * float(j);
        for (int it = 0; it < ITERS; ++it) {
#pragma unroll
          for (int j = 0; j < NACC; ++j)
            a[j] = a[j] * b[j] + 1e-9f;
        }
        float s = 0.f;
        for (int j = 0; j < NACC; ++j) s += a[j];
        out[i[0]] = s;
      });
    }).wait_and_throw();
  };
  kern();  // warmup
  auto t0 = Clock::now();
  const int reps = 5;
  for (int r = 0; r < reps; ++r) kern();
  q.wait_and_throw();
  auto t1 = Clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count() / reps;
  double flops = 2.0 * double(NACC) * double(ITERS) * double(nthreads);
  sycl::free(out, q);
  return flops / sec / 1e9;
}

/* 2) SLM-пропускная: local buffer <= 64KB, все нити WG читают его в цикле. */
static double bench_slm(sycl::queue& q) {
  constexpr size_t WG = 512;
  constexpr size_t LB = 4096;     // float'ов в shared local (16KB) — в пределах 64KB
  constexpr int ITERS = 4096;
  constexpr size_t NWG = 64;      // рабочих групп
  float* out = sycl::malloc_device<float>(NWG * WG, q);
  auto kern = [&]() {
    q.submit([&](sycl::handler& h) {
      auto la = sycl::local_accessor<float, 1>(sycl::range<1>(LB), h);
      h.parallel_for(sycl::nd_range<1>(NWG * WG, WG), [=](sycl::nd_item<1> it) {
        size_t wg = it.get_group(0);
        size_t lid = it.get_local_id(0);
        // инициализация local (один раз на WG)
        if (lid < LB) la[lid] = 1.0f + 1e-6f * float(lid);
        sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
        float s = 0.f;
        for (int it2 = 0; it2 < ITERS; ++it2) {
          size_t base = (lid * 7 + it2 * 13) % LB;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            s += la[(base + i) % LB];
        }
        out[wg * WG + lid] = s;
      });
    }).wait_and_throw();
  };
  kern();
  auto t0 = Clock::now();
  const int reps = 5;
  for (int r = 0; r < reps; ++r) kern();
  q.wait_and_throw();
  auto t1 = Clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count() / reps;
  double bytes = double(NWG) * double(WG) * double(ITERS) * 8.0 * 4.0;  // 8 reads x 4B
  sycl::free(out, q);
  return bytes / sec / 1e9;
}

/* 3) DRAM-пропускная: чтение из USM, без race (каждая нить пишет в свой слот). */
static double bench_dram(sycl::queue& q) {
  constexpr size_t N = 128 * 1024 * 1024;  // 512MB
  constexpr size_t WT = 65536;
  float* buf = sycl::malloc_device<float>(N, q);
  float* out = sycl::malloc_device<float>(WT, q);
  const size_t PER = N / WT;
  auto kern = [&]() {
    q.submit([&](sycl::handler& h) {
      h.parallel_for(WT, [=](sycl::id<1> i) {
        size_t base = i[0] * PER;
        float s = 0.f;
        for (size_t j = 0; j < PER; ++j)
          s += buf[base + j];
        out[i[0]] = s;
      });
    }).wait_and_throw();
  };
  kern();
  auto t0 = Clock::now();
  const int reps = 3;
  for (int r = 0; r < reps; ++r) kern();
  q.wait_and_throw();
  auto t1 = Clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count() / reps;
  double bytes = double(N) * 4.0;
  sycl::free(buf, q); sycl::free(out, q);
  return bytes / sec / 1e9;
}

int main(){
  setvbuf(stdout,NULL,_IOLBF,0);
  sycl::device gpu; bool have=false;
  for(auto&d:sycl::device::get_devices()) if(d.is_gpu()){gpu=d;have=true;break;}
  if(!have){std::printf("NO GPU\n");return 1;}
  sycl::queue q{gpu, sycl::property::queue::in_order()};
  std::printf("device: %s\n", gpu.get_info<sycl::info::device::name>().c_str());
  double fma1 = bench_fma<8, 100000>(q, 65536);
  double fma2 = bench_fma<8, 200000>(q, 65536);
  std::printf("FMA peak (reg-only):  %.1f / %.1f GF/s (2 ITERS, стабильность)\n", fma1, fma2);
  double slm = bench_slm(q);
  std::printf("SLM read bandwidth:   %.1f GB/s\n", slm);
  double dram = bench_dram(q);
  std::printf("DRAM read bandwidth:  %.1f GB/s\n", dram);
  std::printf("\n== Ceiling анализ для GEMM ==\n");
  std::printf("  FMA-потолок:        %.1f GF/s\n", fma2);
  std::printf("  GEMM лучший замер:  140 GF/s = %.1f%% от FMA-пика\n", 100.0*140.0/fma2);
  std::printf("  => GEMM НЕ compute-bound; упор в SLM/DRAM-кормлении\n");
  std::printf("  SLM-limited (1 read/FMA):  %.1f GF/s\n", slm / 4.0);
  std::printf("  DRAM-limited (TILE=64):    %.1f GF/s\n", dram / (4.0/64.0));
  std::printf("MICROBENCH_DONE\n");
  return 0;
}
