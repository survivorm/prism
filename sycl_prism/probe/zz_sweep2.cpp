#include "../src/prism_sycl_compat.h"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
using Clock = std::chrono::steady_clock;

/* Коалесценсный тайловый GEMM, USM.
   TILE x TILE C-тайл, WG = (TILE/CPT) x TILE, CPT строк C на нить.
   Загрузка: блок-маппинг t + NT*i -> коалесценсно (t%stride консекутивен).
   B хранится в local транспонированным [col][k] -> вычисление: обе строки. */
template <size_t TILE, size_t CPT, size_t KC, bool DB>
void gemm_usm2(sycl::queue& q, const float* A, const float* B, float* C,
               size_t M, size_t K, size_t N) {
  constexpr size_t WGR = TILE / CPT;
  constexpr size_t WGC = TILE;
  constexpr size_t NT  = WGR * WGC;
  constexpr size_t RA  = (TILE * KC + NT - 1) / NT;   // A-элементов на нить (ceil)
  constexpr size_t RB  = (KC * TILE + NT - 1) / NT;   // B-элементов на нить (ceil)
  static_assert(RA >= 1 && RB >= 1, "слишком мало элементов на нить");
  const size_t numM = (M + TILE - 1) / TILE;
  const size_t numN = (N + TILE - 1) / TILE;
  q.submit([&](sycl::handler& h) {
    auto ta0 = sycl::local_accessor<float, 2>(sycl::range<2>(TILE, KC), h);
    auto tb0 = sycl::local_accessor<float, 2>(sycl::range<2>(TILE, KC), h); // [col][k]
    auto ta1 = DB ? sycl::local_accessor<float, 2>(sycl::range<2>(TILE, KC), h)
                  : sycl::local_accessor<float, 2>(sycl::range<2>(0, 1), h);
    auto tb1 = DB ? sycl::local_accessor<float, 2>(sycl::range<2>(TILE, KC), h)
                  : sycl::local_accessor<float, 2>(sycl::range<2>(0, 1), h);
    h.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(numM * WGR, numN * WGC),
                          sycl::range<2>(WGR, WGC)),
        [=](sycl::nd_item<2> it) {
          const size_t lr = it.get_local_id(0);
          const size_t lc = it.get_local_id(1);
          const size_t t  = lr * WGC + lc;
          const size_t row0 = (it.get_global_id(0) / WGR) * TILE;
          const size_t col0 = (it.get_global_id(1) / WGC) * TILE;

          auto load = [&](auto ta, auto tb, size_t k0) {
#pragma unroll
            for (size_t i = 0; i < RA; ++i) {
              size_t e = t + NT * i;
              if (e < TILE * KC) {
                size_t ra = e / KC, ca = e % KC;
                ta[ra][ca] = (row0 + ra < M && k0 + ca < K)
                                 ? A[(row0 + ra) * K + k0 + ca] : 0.f;
              }
            }
#pragma unroll
            for (size_t i = 0; i < RB; ++i) {
              size_t e = t + NT * i;
              if (e < KC * TILE) {
                size_t rb = e / TILE, cb = e % TILE;
                tb[cb][rb] = (k0 + rb < K && col0 + cb < N)
                                 ? B[(k0 + rb) * N + col0 + cb] : 0.f;
              }
            }
          };

          load(ta0, tb0, 0);
          float acc[CPT];
#pragma unroll
          for (size_t r = 0; r < CPT; ++r) acc[r] = 0.f;

          if constexpr (DB) {
            for (size_t k0 = 0; k0 < K; k0 += KC) {
              const size_t bi = (k0 / KC) % 2;
              auto& Cta = bi ? ta1 : ta0;
              auto& Ctb = bi ? tb1 : tb0;
              auto& Nta = bi ? ta0 : ta1;
              auto& Ntb = bi ? tb0 : tb1;
              prism_sycl::barrier(it);
              if (k0 + KC < K) load(Nta, Ntb, k0 + KC);
#pragma unroll
              for (size_t kk = 0; kk < KC; ++kk)
#pragma unroll
                for (size_t r = 0; r < CPT; ++r)
                  acc[r] += Cta[lr + r * (TILE / CPT)][kk] * Ctb[lc][kk];
            }
          } else {
            for (size_t k0 = 0; k0 < K; k0 += KC) {
              prism_sycl::barrier(it);
#pragma unroll
              for (size_t kk = 0; kk < KC; ++kk)
#pragma unroll
                for (size_t r = 0; r < CPT; ++r)
                  acc[r] += ta0[lr + r * (TILE / CPT)][kk] * tb0[lc][kk];
              if (k0 + KC < K) {
                prism_sycl::barrier(it);
                load(ta0, tb0, k0 + KC);
              }
            }
          }
          for (size_t r = 0; r < CPT; ++r) {
            const size_t row = row0 + lr + r * (TILE / CPT);
            if (row < M && col0 + lc < N) C[row * N + col0 + lc] = acc[r];
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
  const size_t M=4096,K=1024,N=4096;
  std::vector<float> a(M*K), b(K*N);
  for(size_t i=0;i<a.size();++i) a[i]=0.01f*float(i%97);
  for(size_t i=0;i<b.size();++i) b[i]=0.01f*float(i%31);
  float *A=sycl::malloc_device<float>(M*K,q), *B=sycl::malloc_device<float>(K*N,q),
        *Cm=sycl::malloc_device<float>(M*N,q);
  q.submit([&](sycl::handler&h){h.copy(a.data(),A,M*K);});
  q.submit([&](sycl::handler&h){h.copy(b.data(),B,K*N);});
  q.wait_and_throw();

  double best=0; const char* bestname="";
  auto bench=[&](const char* nm, auto kern){
    try {
      kern();
      std::vector<double> t;
      for (int r=0;r<5;r++){
        auto t0=Clock::now(); kern(); auto t1=Clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
      }
      std::sort(t.begin(),t.end());
      double gf=2.0*M*K*N/(t[t.size()/2]/1000.0)/1e9;
      std::printf("%-28s %8.1f GF/s (med %7.1f ms)\n", nm, gf, t[t.size()/2]);
      if (gf>best){best=gf; bestname=nm;}
    } catch (const sycl::exception& e) {
      std::printf("%-28s ERROR: %s\n", nm, std::string(e.what()).substr(0,80).c_str());
    }
  };
  std::printf("SWEEP 4096x1024x4096 USM coalesced (TILE,CPT,KC,DB):\n");
  #define SW(T,P,K,D) bench(#T "x" #P " kc" #K " db" #D, [&]{ gemm_usm2<T,P,K,D>(q,A,B,Cm,M,K,N); });
  SW(16,1,16,false) SW(16,1,16,true)  SW(16,1,32,false) SW(16,1,32,true)
  SW(16,1,64,false) SW(16,1,64,true)  SW(16,1,128,false) SW(16,1,128,true)
  SW(16,2,16,false) SW(16,2,16,true)  SW(16,2,32,false) SW(16,2,32,true)
  SW(16,2,64,false) SW(16,2,64,true)  SW(16,2,128,false) SW(16,2,128,true)
  SW(32,1,16,false) SW(32,1,16,true)  SW(32,1,32,false) SW(32,1,32,true)
  SW(32,1,64,false) SW(32,1,64,true)  SW(32,1,128,false)
  SW(32,2,16,false) SW(32,2,16,true)  SW(32,2,32,false) SW(32,2,32,true)
  SW(32,2,64,false) SW(32,2,64,true)  SW(32,2,128,false)
  #undef SW
  std::printf("\nBEST: %s = %.1f GF/s (пик ~1460, доля %.1f%%)\n",
              bestname, best, 100.0*best/1460.0);
  std::printf("SWEEP2_DONE\n");
  return 0;
}
