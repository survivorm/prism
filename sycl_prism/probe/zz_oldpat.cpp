#include "../src/prism_sycl_compat.h"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
using Clock = std::chrono::steady_clock;

template <size_t TILE, size_t KC>
void gemm_oldpat(sycl::queue& q, const float* A, const float* B, float* C,
                 size_t M, size_t K, size_t N) {
  const size_t numM = (M + TILE - 1) / TILE;
  const size_t numN = (N + TILE - 1) / TILE;
  q.submit([&](sycl::handler& h) {
    auto ta = sycl::local_accessor<float, 2>(sycl::range<2>(TILE, KC), h);
    auto tb = sycl::local_accessor<float, 2>(sycl::range<2>(KC, TILE), h);
    h.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(numM * TILE, numN * TILE),
                          sycl::range<2>(TILE, TILE)),
        [=](sycl::nd_item<2> it) {
          const size_t row0 = (it.get_global_id(0) / TILE) * TILE;
          const size_t col0 = (it.get_global_id(1) / TILE) * TILE;
          const size_t lr = it.get_local_id(0);
          const size_t lc = it.get_local_id(1);
          float acc[TILE / TILE];  // 1 element C per thread (TILE threads cover TILE rows)
          acc[0] = 0.f;
          for (size_t k0 = 0; k0 < K; k0 += KC) {
            // load: thread (lr,lc) loads A-tile row lr, cols k0..k0+KC-1 (KC elems)
            for (size_t kk = 0; kk < KC; ++kk)
              ta[lr][kk] = (row0 + lr < M && k0 + kk < K)
                               ? A[(row0 + lr) * K + k0 + kk] : 0.f;
            for (size_t kk = 0; kk < KC; ++kk)
              tb[kk][lc] = (k0 + kk < K && col0 + lc < N)
                               ? B[(k0 + kk) * N + col0 + lc] : 0.f;
            prism_sycl::barrier(it);
#pragma unroll
            for (size_t kk = 0; kk < KC; ++kk)
              acc[0] += ta[lr][kk] * tb[kk][lc];
            prism_sycl::barrier(it);
          }
          if (row0 + lr < M && col0 + lc < N) C[(row0 + lr) * N + col0 + lc] = acc[0];
        });
  }).wait_and_throw();
}

int main(){
  setvbuf(stdout,NULL,_IOLBF,0);
  sycl::device gpu; bool have=false;
  for(auto&d:sycl::device::get_devices()) if(d.is_gpu()){gpu=d;have=true;break;}
  if(!have) return 1;
  sycl::queue q{gpu, sycl::property::queue::in_order()};
  for (auto shape : std::vector<std::array<size_t,3>>{{1024,1024,1024},{4096,1024,4096}}) {
    const size_t M=shape[0],K=shape[1],N=shape[2];
    std::vector<float> a(M*K,0.01f), b(K*N,0.01f), c(M*N,0.f);
    float *A=sycl::malloc_device<float>(M*K,q), *B=sycl::malloc_device<float>(K*N,q), *C=sycl::malloc_device<float>(M*N,q);
    q.submit([&](sycl::handler&h){h.copy(a.data(),A,M*K);});
    q.submit([&](sycl::handler&h){h.copy(b.data(),B,K*N);});
    q.wait_and_throw();
    auto bench1=[&](const char* nm, auto kern){
      kern();
      std::vector<double> t;
      for (int r=0;r<5;r++){
        auto t0=Clock::now(); kern(); auto t1=Clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
      }
      std::sort(t.begin(),t.end());
      double gf=2.0*M*K*N/(t[t.size()/2]/1000.0)/1e9;
      std::printf("  %-14s %.1f GF/s (med %.1f ms)\n", nm, gf, t[t.size()/2]);
    };
    std::printf("shape %zux%zux%zu:\n", M,K,N);
    bench1("old KC=16", [&]{ gemm_oldpat<16,16>(q,A,B,C,M,K,N); });
    bench1("KC=32",     [&]{ gemm_oldpat<16,32>(q,A,B,C,M,K,N); });
    bench1("KC=64",     [&]{ gemm_oldpat<16,64>(q,A,B,C,M,K,N); });
    bench1("T32 KC=32", [&]{ gemm_oldpat<32,32>(q,A,B,C,M,K,N); });
    sycl::free(A,q); sycl::free(B,q); sycl::free(C,q);
  }
  std::printf("OLDPAT_DONE\n");
  return 0;
}
