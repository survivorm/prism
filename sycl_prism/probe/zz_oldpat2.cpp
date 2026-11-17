#include "../src/prism_sycl_compat.h"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
using Clock = std::chrono::steady_clock;

// DOSLOVNO старый паттерн (probe_sycl_min): buffer<2> + 2D local + 1 elem C/нить
static void gemm_buf(sycl::queue& q, std::vector<float>& a, std::vector<float>& b,
                     std::vector<float>& c, size_t M, size_t K, size_t N) {
  const size_t TILE = 16;
  sycl::buffer<float, 2> ba(a.data(), sycl::range<2>(M, K));
  sycl::buffer<float, 2> bb(b.data(), sycl::range<2>(K, N));
  sycl::buffer<float, 2> bc(c.data(), sycl::range<2>(M, N));
  q.submit([&](sycl::handler &h) {
      auto aa = ba.get_access<sycl::access::mode::read>(h);
      auto ab = bb.get_access<sycl::access::mode::read>(h);
      auto ac = bc.get_access<sycl::access::mode::write>(h);
      auto ta = sycl::local_accessor<float, 2>(sycl::range<2>(TILE, TILE), h);
      auto tb = sycl::local_accessor<float, 2>(sycl::range<2>(TILE, TILE), h);
      h.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(M, N), sycl::range<2>(TILE, TILE)),
          [=](sycl::nd_item<2> it) {
              const size_t row = it.get_global_id(0);
              const size_t col = it.get_global_id(1);
              const size_t lr = it.get_local_id(0);
              const size_t lc = it.get_local_id(1);
              float acc = 0.f;
              for (size_t k0 = 0; k0 < K; k0 += TILE) {
                  ta[lr][lc] = (row < M && k0 + lc < K) ? aa[row][k0 + lc] : 0.f;
                  tb[lr][lc] = (k0 + lr < K && col < N) ? ab[k0 + lr][col] : 0.f;
                  prism_sycl::barrier(it);
                  for (size_t kk = 0; kk < TILE; ++kk)
                      acc += ta[lr][kk] * tb[kk][lc];
                  prism_sycl::barrier(it);
              }
              if (row < M && col < N) ac[row][col] = acc;
          });
  }).wait_and_throw();
}

int main(){
  setvbuf(stdout,NULL,_IOLBF,0);
  sycl::device gpu; bool have=false;
  for(auto&d:sycl::device::get_devices()) if(d.is_gpu()){gpu=d;have=true;break;}
  if(!have) return 1;
  sycl::queue q{gpu};
  for (auto shape : std::vector<std::array<size_t,3>>{{1024,1024,1024},{4096,1024,4096}}) {
    const size_t M=shape[0],K=shape[1],N=shape[2];
    std::vector<float> a(M*K,0.01f), b(K*N,0.01f), c(M*N,0.f);
    auto bench=[&](const char* nm){
      gemm_buf(q,a,b,c,M,K,N);
      std::vector<double> t;
      for (int r=0;r<5;r++){
        auto t0=Clock::now(); gemm_buf(q,a,b,c,M,K,N); auto t1=Clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
      }
      std::sort(t.begin(),t.end());
      double gf=2.0*M*K*N/(t[t.size()/2]/1000.0)/1e9;
      std::printf("  %-14s %.1f GF/s (med %.1f ms)\n", nm, gf, t[t.size()/2]);
    };
    std::printf("shape %zux%zux%zu (buffer<2>, TILE=16, KC=16):\n", M,K,N);
    bench("1x");
    bench("2x");
  }
  std::printf("OLDPAT2_DONE\n");
  return 0;
}
