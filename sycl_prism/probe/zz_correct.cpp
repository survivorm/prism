#include "../src/prism_sycl_compat.h"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <cmath>
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
  const size_t M=1024,K=1024,N=1024;
  std::vector<float> a(M*K), b(K*N);
  for(size_t i=0;i<a.size();++i) a[i]=0.01f*float(i%97);
  for(size_t i=0;i<b.size();++i) b[i]=0.01f*float(i%31);
  std::vector<double> cref(M*N,0.0);
  for(size_t i=0;i<M;++i)
    for(size_t k=0;k<K;++k) if(a[i*K+k]!=0.0)
      for(size_t j=0;j<N;++j) cref[i*N+j]+=(double)a[i*K+k]*(double)b[k*N+j];
  float *A=sycl::malloc_device<float>(M*K,q), *B=sycl::malloc_device<float>(K*N,q),
        *Cm=sycl::malloc_device<float>(M*N,q);
  q.submit([&](sycl::handler&h){h.copy(a.data(),A,M*K);});
  q.submit([&](sycl::handler&h){h.copy(b.data(),B,K*N);});
  q.wait_and_throw();
  auto chk=[&](const char* nm, auto kern){
    std::vector<float> z(M*N,0.f);
    q.submit([&](sycl::handler&h){h.copy(z.data(),Cm,M*N);});
    q.wait_and_throw();
    kern();
    std::vector<float> c(M*N);
    q.submit([&](sycl::handler&h){h.copy(Cm,c.data(),M*N);}).wait_and_throw();
    double maxrel=0.0; size_t mi=0;
    for(size_t i=0;i<M*N;i++){
      double r=std::fabs(c[i]-cref[i])/(std::fabs(cref[i])+1e-30);
      if(r>maxrel){maxrel=r; mi=i;}
    }
    std::printf("%-24s maxrel=%.3e (эталон f64, 1024^3) %s\n",
                nm, maxrel, maxrel<1e-4?"OK":"FAIL");
  };
  chk("16x1 kc16 dbfalse", [&]{ gemm_usm2<16,1,16,false>(q,A,B,Cm,M,K,N); });
  chk("16x1 kc16 dbtrue ", [&]{ gemm_usm2<16,1,16,true >(q,A,B,Cm,M,K,N); });
  chk("16x2 kc16 dbtrue ", [&]{ gemm_usm2<16,2,16,true >(q,A,B,Cm,M,K,N); });
  std::printf("CORRECT_DONE\n");
  return 0;
}
