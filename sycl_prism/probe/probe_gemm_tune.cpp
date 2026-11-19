// probe_gemm_tune.cpp — подбор тайлинга для SYCL GEMM (Intel iGPU).
//
// ЦЕЛЬ: найти конфигурацию (CTM,CTN, WGM,WGN, KC, DB), дающую максимум GFLOP/s
// на реалистичных размерах. Сравниваем с CPU OpenBLAS-4т (~83 GF/s) и с
// текущим 16x16 (6-28 GF/s).
//
// ВАЖНО (найдено диагностикой 17.11): форма S1 = 4096x1024x65536 (N=65536)
// ЗАВИШАЕТ/крайне медленна на этом драйвере (Iris Xe, Level-Zero 1.17).
// Плоское ядро с тем же гридом (4096,65536) работает => это не лимит грида,
// а дефект драйвера/рантайма для больших 2D-гридов в nd_range с local_mem.
// Поэтому здесь безопасные формы: N <= 4096.
//
// СБОРКА:  icpx -fsycl -std=c++20 -O3 probe/probe_gemm_tune.cpp -o build/probe_gemm_tune
// ЗАПУСК:  source sycl_env.sh && ./build/probe_gemm_tune
#include "../src/prism_sycl_compat.h"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <vector>

using Clock = std::chrono::steady_clock;

/* Тайлованный GEMM, USM-указатели, compat-барьер.
   A[M,K] @ B[K,N] -> C[M,N], row-major fp32. */
template <size_t CTM, size_t CTN, size_t WGM, size_t WGN, size_t KC, bool DB>
void gemm_tiled(sycl::queue& q, const float* A, const float* B, float* C,
                size_t M, size_t K, size_t N) {
  constexpr size_t TM = CTM / WGM;  // элементов C по строкам на нить
  constexpr size_t TN = CTN / WGN;  // элементов C по колонкам на нить
  static_assert(TM >= 1 && TN >= 1, "WG больше тайла");
  const size_t numM = (M + CTM - 1) / CTM;
  const size_t numN = (N + CTN - 1) / CTN;
  q.submit([&](sycl::handler& h) {
    auto la0 = sycl::local_accessor<float, 1>(sycl::range<1>(CTM * KC), h);
    auto lb0 = sycl::local_accessor<float, 1>(sycl::range<1>(KC * CTN), h);
    // DB: второй буфер; !DB: алиас la0 (не используется, безопасный биндинг)
    auto la1 = DB ? sycl::local_accessor<float, 1>(sycl::range<1>(CTM * KC), h)
                  : la0;
    auto lb1 = DB ? sycl::local_accessor<float, 1>(sycl::range<1>(KC * CTN), h)
                  : lb0;
    h.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(numM * WGM, numN * WGN),
                          sycl::range<2>(WGM, WGN)),
        [=](sycl::nd_item<2> it) {
          const size_t gid_r = it.get_global_id(0);
          const size_t gid_c = it.get_global_id(1);
          const size_t lid_r = it.get_local_id(0);
          const size_t lid_c = it.get_local_id(1);
          const size_t row0 = (gid_r / WGM) * CTM;
          const size_t col0 = (gid_c / WGN) * CTN;

          float acc[TM][TN];
#pragma unroll
          for (size_t i = 0; i < TM; ++i)
#pragma unroll
            for (size_t j = 0; j < TN; ++j) acc[i][j] = 0.0f;

          auto load_chunk = [&](sycl::local_accessor<float, 1> LA,
                                sycl::local_accessor<float, 1> LB,
                                size_t k0) {
            for (size_t kk = 0; kk < KC; ++kk) {
              const size_t gk = k0 + kk;
              for (size_t i = 0; i < TM; ++i)
                LA[i * KC + kk] = (row0 + lid_r + i < M && gk < K)
                                       ? A[(row0 + lid_r + i) * K + gk]
                                       : 0.0f;
              for (size_t j = 0; j < TN; ++j)
                LB[kk * CTN + j] = (gk < K && col0 + lid_c + j < N)
                                       ? B[gk * N + (col0 + lid_c + j)]
                                       : 0.0f;
            }
          };

          load_chunk(la0, lb0, 0);
          for (size_t k0 = 0; k0 < K; k0 += KC) {
            const size_t bi = (k0 / KC) % 2;
            auto& C_LA = bi ? la1 : la0;
            auto& C_LB = bi ? lb1 : lb0;
            auto& N_LA = bi ? la0 : la1;
            auto& N_LB = bi ? lb0 : lb1;
            prism_sycl::barrier(it);  // текущий чанк видим + прошлый compute завершён
            if (DB && k0 + KC < K) load_chunk(N_LA, N_LB, k0 + KC);  // prefetch
#pragma unroll
            for (size_t kk = 0; kk < KC; ++kk) {
#pragma unroll
              for (size_t i = 0; i < TM; ++i) {
                const float a = C_LA[i * KC + kk];
#pragma unroll
                for (size_t j = 0; j < TN; ++j) acc[i][j] += a * C_LB[kk * CTN + j];
              }
            }
            if (!DB && k0 + KC < K) {
              prism_sycl::barrier(it);  // compute прочитал буфер — можно перезаписывать
              load_chunk(C_LA, C_LB, k0 + KC);
            }
          }
          for (size_t i = 0; i < TM; ++i) {
            const size_t r = row0 + lid_r + i;
            if (r >= M) continue;
#pragma unroll
            for (size_t j = 0; j < TN; ++j) {
              const size_t c = col0 + lid_c + j;
              if (c < N) C[r * N + c] = acc[i][j];
            }
          }
        });
  }).wait_and_throw();
}

/* Хост-референс (f64): контрольная сумма C. */
static double ref_gemm_checksum(const std::vector<float>& a,
                                const std::vector<float>& b, size_t M,
                                size_t K, size_t N) {
  double s = 0.0;
  std::vector<double> c(M * N, 0.0);
  for (size_t i = 0; i < M; ++i)
    for (size_t k = 0; k < K; ++k)
      if (a[i * K + k] != 0.0)
        for (size_t j = 0; j < N; ++j)
          c[i * N + j] += (double)a[i * K + k] * (double)b[k * N + j];
  for (size_t i = 0; i < c.size(); i += c.size() / 7 + 1) s += c[i];
  return s;
}

struct Cfg {
  const char* name;
  double gf1 = 0.0;   // GF/s на S1 (4096x1024x4096)
  double gf2 = 0.0;   // GF/s на S2 (1024x1024x1024)
  bool ok = false;
};

template <auto kern>
static double time_shape(sycl::queue& q, const float* A, const float* B,
                         float* C, size_t M, size_t K, size_t N, int iters) {
  kern(q, A, B, C, M, K, N);  // прогрев
  q.wait_and_throw();
  std::vector<double> t;
  for (int r = 0; r < iters; ++r) {
    auto t0 = Clock::now();
    kern(q, A, B, C, M, K, N);
    q.wait_and_throw();
    auto t1 = Clock::now();
    t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(t.begin(), t.end());
  double med = t[t.size() / 2];
  return 2.0 * double(M) * double(K) * double(N) / (med / 1000.0) / 1e9;
}

int main() {
  // Лог через nohup -> block-buffered; без line-buffering не видно ни строки.
  setvbuf(stdout, NULL, _IOLBF, 0);
  std::printf("[probe_gemm_tune] start pid=%d\n", (int)getpid());
  std::vector<sycl::device> devs = sycl::device::get_devices();
  sycl::device chosen;
  bool have = false;
  for (const sycl::device& d : devs) {
    if (d.is_gpu()) { chosen = d; have = true; break; }
  }
  if (!have) {
    std::printf("GPU не найдена среди %zu устройств\n", devs.size());
    return 1;
  }
  sycl::queue q{chosen, sycl::property::queue::in_order()};
  auto dev = q.get_device();
  std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
  std::printf("local_mem (device info): %zu KB\n",
              dev.get_info<sycl::info::device::local_mem_size>() / 1024);

  // S0 — корректность (1024^3)
  const size_t M0 = 1024, K0 = 1024, N0 = 1024;
  std::vector<float> a0(M0 * K0), b0(K0 * N0), c0(M0 * N0);
  for (size_t i = 0; i < a0.size(); ++i) a0[i] = 0.01f * float(i % 97);
  for (size_t i = 0; i < b0.size(); ++i) b0[i] = 0.01f * float(i % 31);
  const double ref0 = ref_gemm_checksum(a0, b0, M0, K0, N0);

  // S1 — основной замер: 4096x1024x4096 (34.4 GFLOP, БЕЗОПАСНЫЙ грид)
  const size_t M1 = 4096, K1 = 1024, N1 = 4096;
  std::vector<float> a1(M1 * K1), b1(K1 * N1), c1(M1 * N1);
  for (size_t i = 0; i < a1.size(); ++i) a1[i] = 0.01f * float(i % 97);
  for (size_t i = 0; i < b1.size(); ++i) b1[i] = 0.01f * float(i % 31);

  // S2 — средний размер: 1024x1024x1024 (1 GFLOP)
  const size_t M2 = 1024, K2 = 1024, N2 = 1024;
  std::vector<float> a2(M2 * K2), b2(K2 * N2), c2(M2 * N2);
  for (size_t i = 0; i < a2.size(); ++i) a2[i] = 0.01f * float(i % 97);
  for (size_t i = 0; i < b2.size(); ++i) b2[i] = 0.01f * float(i % 31);

  auto dmalloc = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
  float *A0 = dmalloc(a0.size()), *B0 = dmalloc(b0.size()), *C0 = dmalloc(c0.size());
  float *A1 = dmalloc(a1.size()), *B1 = dmalloc(b1.size()), *C1 = dmalloc(c1.size());
  float *A2 = dmalloc(a2.size()), *B2 = dmalloc(b2.size()), *C2 = dmalloc(c2.size());
  q.submit([&](sycl::handler& h) { h.copy(a0.data(), A0, a0.size()); });
  q.submit([&](sycl::handler& h) { h.copy(b0.data(), B0, b0.size()); });
  q.submit([&](sycl::handler& h) { h.copy(a1.data(), A1, a1.size()); });
  q.submit([&](sycl::handler& h) { h.copy(b1.data(), B1, b1.size()); });
  q.submit([&](sycl::handler& h) { h.copy(a2.data(), A2, a2.size()); });
  q.submit([&](sycl::handler& h) { h.copy(b2.data(), B2, b2.size()); });
  q.wait_and_throw();

  auto bench = [&](const char* name, auto kern) {
    Cfg c{name};
    std::printf(">> config %s\n", name);
    try {
      // корректность на S0
      kern(q, A0, B0, C0, M0, K0, N0);
      q.wait_and_throw();
      q.submit([&](sycl::handler& h) { h.copy(C0, c0.data(), c0.size()); });
      q.wait_and_throw();
      double got = 0.0;
      for (size_t i = 0; i < c0.size(); i += c0.size() / 7 + 1) got += c0[i];
      c.ok = std::fabs(got - ref0) < 1e-2 * std::fabs(ref0) + 1e-3;
      // замер на S1 (5 итер) и S2 (5 итер)
      c.gf1 = time_shape<kern>(q, A1, B1, C1, M1, K1, N1, 5);
      c.gf2 = time_shape<kern>(q, A2, B2, C2, M2, K2, N2, 5);
    } catch (const sycl::exception& e) {
      std::printf("CONFIG %s: SYCL error: %s\n", name,
                  std::string(e.what()).substr(0, 120).c_str());
      c.ok = false;
    }
    std::printf("   %-24s  S1(4096^2): %7.1f GF/s   S2(1024^3): %7.1f GF/s   S0: %s\n",
                c.name, c.gf1, c.gf2, c.ok ? "OK" : "FAIL");
    return c;
  };

  std::printf("\n=== S1 = 4096x1024x4096 (34.4 GFLOP), S2 = 1024^3, S0 = 1024^3 (чек) ===\n");
  std::vector<Cfg> results;
  #define BENCH(NAME, CTM, CTN, WGM, WGN, KC, DB) \
    results.push_back(bench(NAME, [](auto& qq, auto A, auto B, auto C, auto M, auto K, auto N) { \
      gemm_tiled<CTM, CTN, WGM, WGN, KC, DB>(qq, A, B, C, M, K, N); }));

  BENCH("16x16  wg16x16 kc32", 16, 16, 16, 16, 32, false);
  BENCH("32x32  wg16x16 kc32", 32, 32, 16, 16, 32, false);
  BENCH("32x32  wg16x8  kc32", 32, 32, 16, 8, 32, false);
  BENCH("64x32  wg16x16 kc32", 64, 32, 16, 16, 32, false);
  BENCH("32x64  wg16x16 kc32", 32, 64, 16, 16, 32, false);
  BENCH("64x64  wg16x16 kc32", 64, 64, 16, 16, 32, false);
  BENCH("128x32 wg32x16 kc32", 128, 32, 32, 16, 32, false);
  BENCH("32x32  wg8x8   kc32", 32, 32, 8, 8, 32, false);
  BENCH("32x128 wg16x16 kc32", 32, 128, 16, 16, 32, false);
  BENCH("32x32  wg16x16 kc64", 32, 32, 16, 16, 64, false);
  BENCH("64x64  wg16x16 kc64", 64, 64, 16, 16, 64, false);
  BENCH("32x64  wg16x16 kc64", 32, 64, 16, 16, 64, false);
  BENCH("32x32  wg16x16 kc32 DB", 32, 32, 16, 16, 32, true);
  BENCH("32x64  wg16x16 kc32 DB", 32, 64, 16, 16, 32, true);
  BENCH("64x64  wg16x16 kc32 DB", 64, 64, 16, 16, 32, true);
  BENCH("32x64  wg16x16 kc64 DB", 32, 64, 16, 16, 64, true);
  BENCH("32x128 wg16x16 kc32 DB", 32, 128, 16, 16, 32, true);
  #undef BENCH

  sycl::free(A0, q); sycl::free(B0, q); sycl::free(C0, q);
  sycl::free(A1, q); sycl::free(B1, q); sycl::free(C1, q);
  sycl::free(A2, q); sycl::free(B2, q); sycl::free(C2, q);

  // Итоговая таблица + лучший
  std::printf("\n=== ИТОГ (GF/s, fp32, Iris Xe) ===\n");
  std::printf("%-26s %12s %12s   %s\n", "config", "S1 4096^2", "S2 1024^3", "S0");
  const Cfg* best = nullptr;
  for (const auto& c : results) {
    if (!c.ok) continue;
    std::printf("%-26s %12.1f %12.1f   %s\n", c.name, c.gf1, c.gf2, "OK");
    if (!best || c.gf1 > best->gf1) best = &c;
  }
  if (best)
    std::printf("\nЛУЧШИЙ (по S1): %s  —  %.1f GF/s (CPU OpenBLAS-4т ~83 GF/s, коэффициент %.2fx)\n",
                best->name, best->gf1, best->gf1 / 83.0);
  else
    std::printf("\nНИ ОДНА конфигурация не прошла S0-чек\n");
  std::printf("PROBE_GEMM_TUNE_DONE\n");
  return 0;
}
