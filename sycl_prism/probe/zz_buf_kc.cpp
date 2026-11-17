#include "../src/prism_sycl_compat.h"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
using Clock = std::chrono::steady_clock;

// buffer<2> pattern, параметризованный KC (TILE=16, 1 elem C/нить)
template <size_t TILE, size_t KC>
static void gemm_buf(sycl::queue& q, std::vector<float>& a, std::vector<float>& b,
                     std::vector<float>& c, size_t M, size_t K, size_t N) {
  sycl::buffer<float, 2> ba(a.data(), sycl::range<2>(M, K));
  sycl::buffer<float, 2> bb(b.data(), sycl::range<2>(K, N));
  sycl::buffer<float, 2> bc(c.data(), sycl::range<2>(M, N));
  q.submit([&](sycl::handler &h) {
      auto aa = ba.get_access<sycl::access::mode::read>(h);
      auto ab = bb.get_access<sycl::access::mode::read>(h);
      auto ac = bc.get_access<sycl::access::mode::write>(h);
      auto ta = sycl::local_accessor<float, 2>(sycl::range<2>(TILE, KC), h);
      auto tb = sycl::local_accessor<float, 2>(sycl::range<2>(KC, TILE), h);
      h.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(M, N), sycl::range<2>(TILE, TILE)),
          [=](sycl::nd_item<2> it) {
              const size_t row = it.get_global_id(0);
              const size_t col = it.get_global_id(1);
              const size_t lr = it.get_local_id(0);
              const size_t lc = it.get_local_id(1);
              float acc = 0.f;
              for (size_t k0 = 0; k0 < K; k0 += KC) {
#pragma unroll
                  for (size_t kk = 0; kk < KC; ++kk)
                      ta[lr][kk] = (row < M && k0 + kk < K) ? aa[row][k0 + kk] : 0.f;
#pragma unroll
                  for (size_t kk = 0; kk < KC; ++kk)
                      tb[kk][lc] = (k0 + kk < K && col < N) ? ab[k0 + kk][col] : 0.f;
                  prism_sycl::barrier(it);
#pragma unroll
                  for (size_t kk = 0; kk < KC; ++kk)
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
  const size_t M=4096,K=1024,N=4096;
  std::vector<float> a(M*K,0.01f), b(K*N,0.01f), c(M*N,0.f);
  auto bench=[&](const char* nm, auto kern){
    kern();
    std::vector<double> t;
    for (int r=0;r<5;r++){
      auto t0=Clock::now(); kern(); auto t1=Clock::now();
      t.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
    }
    std::sort(t.begin(),t.end());
    double gf=2.0*M*K*N/(t[t.size()/2]/1000.0)/1e9;
    std::printf("  %-12s %.1f GF/s (med %.1f ms)\n", nm, gf, t[t.size()/2]);
  };
  std::printf("4096x1024x4096, buffer<2>, TILE=16:\n");
  bench("KC=16", [&]{ gemm_buf<16,16>(q,a,b,c,M,K,N); });
  bench("KC=32", [&]{ gemm_buf<16,32>(q,a,b,c,M,K,N); });
  bench("KC=64", [&]{ gemm_buf<16,64>(q,a,b,c,M,K,N); });
  std::printf("BUF_KC_DONE\n");
  return 0;
}
