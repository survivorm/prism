#include "../src/prism_sycl_compat.h"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <cmath>
/* Тайлованный GEMM, USM-указатели, compat-барьер.
   A[M,K] @ B[K,N] -> C[M,N], row-major fp32. */
template <size_t CTM, size_t CTN, size_t WGM, size_t WGN, size_t KC, bool DB>
void gemm_tiled(sycl::queue& q, const float* A, const float* B, float* C,
                size_t M, size_t K, size_t N, float* SCR = nullptr) {
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
            if (SCR != nullptr && lid_r == 0 && lid_c == 0)
              SCR[(k0 / KC)] = (bi ? lb1 : lb0)[0];
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

int main(){
  setvbuf(stdout,NULL,_IOLBF,0);
  sycl::device gpu; bool have=false;
  for(auto&d:sycl::device::get_devices()) if(d.is_gpu()){gpu=d;have=true;break;}
  if(!have) return 1;
  sycl::queue q{gpu, sycl::property::queue::in_order()};
  const size_t M=128,K=128,N=128;
  std::vector<float> ah(M*K,1.0f), bh(K*N,0.0f);
  for (size_t k=0;k<K;k++) for (size_t j=0;j<N;j++) bh[k*N+j] = float(k/32 + 1);
  float *A=sycl::malloc_device<float>(M*K,q), *B=sycl::malloc_device<float>(K*N,q),
        *C=sycl::malloc_device<float>(M*N,q), *SC=sycl::malloc_device<float>(K/32+8,q);
  q.submit([&](sycl::handler&h){h.copy(ah.data(),A,M*K);});
  q.submit([&](sycl::handler&h){h.copy(bh.data(),B,K*N);});
  q.wait_and_throw();
  {
    std::printf("=== !DB 16x16 kc32 (K=128, 4 chunks) ===\n");
    gemm_tiled<16,16,16,16,32,false>(q,A,B,C,M,K,N,SC);
    std::vector<float> scr(K/32+8), c(4);
    q.submit([&](sycl::handler&h){h.copy(SC,scr.data(),K/32+8);}).wait_and_throw();
    q.submit([&](sycl::handler&h){h.copy(C,c.data(),4);}).wait_and_throw();
    std::printf("  buffer-chunk-id per iteration:");
    for (size_t t=0;t<K/32;t++) std::printf(" %zu", (size_t)scr[t]);
    std::printf("   (expect 0 1 2 3)\n");
    std::printf("  C[0..3]=%.1f %.1f %.1f %.1f (expect 240)\n",c[0],c[1],c[2],c[3]);
  }
  {
    std::printf("=== DB  16x16 kc32 (K=128, 4 chunks) ===\n");
    gemm_tiled<16,16,16,16,32,true>(q,A,B,C,M,K,N,SC);
    std::vector<float> scr(K/32+8), c(4);
    q.submit([&](sycl::handler&h){h.copy(SC,scr.data(),K/32+8);}).wait_and_throw();
    q.submit([&](sycl::handler&h){h.copy(C,c.data(),4);}).wait_and_throw();
    std::printf("  buffer-chunk-id per iteration:");
    for (size_t t=0;t<K/32;t++) std::printf(" %zu", (size_t)scr[t]);
    std::printf("   (expect 0 1 2 3)\n");
    std::printf("  C[0..3]=%.1f %.1f %.1f %.1f (expect 240)\n",c[0],c[1],c[2],c[3]);
  }
  std::printf("TRACE_DONE\n");
  return 0;
}
