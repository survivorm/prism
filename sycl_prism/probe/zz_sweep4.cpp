#include "../src/prism_sycl_compat.h"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <cmath>
using Clock = std::chrono::steady_clock;

/* gemm_usm3: как usm2, но B в local [k][col] (bank-conflict-free),
   A [row][k]. WG = (TILE/CPT) x TILE, ограничение WG <= 512. */
template <size_t TILE, size_t CPT, size_t KC, bool DB>
void gemm_usm3(sycl::queue& q, const float* A, const float* B, float* C,
               size_t M, size_t K, size_t N) {
  constexpr size_t WGR = TILE / CPT;
  constexpr size_t WGC = TILE;
  constexpr size_t NT  = WGR * WGC;
  static_assert(NT <= 512, "WG > max 512");
  constexpr size_t RA  = (TILE * KC + NT - 1) / NT;
  constexpr size_t RB  = (KC * TILE + NT - 1) / NT;
  static_assert(RA >= 1 && RB >= 1);
  const size_t numM = (M + TILE - 1) / TILE;
  const size_t numN = (N + TILE - 1) / TILE;
  q.submit([&](sycl::handler& h) {
    auto ta0 = sycl::local_accessor<float, 2>(sycl::range<2>(TILE, KC), h); // [row][k]
    auto tb0 = sycl::local_accessor<float, 2>(sycl::range<2>(KC, TILE), h); // [k][col]
    auto ta1 = DB ? sycl::local_accessor<float, 2>(sycl::range<2>(TILE, KC), h)
                  : sycl::local_accessor<float, 2>(sycl::range<2>(0, 1), h);
    auto tb1 = DB ? sycl::local_accessor<float, 2>(sycl::range<2>(KC, TILE), h)
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
                tb[rb][cb] = (k0 + rb < K && col0 + cb < N)
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
                  acc[r] += Cta[lr + r * (TILE / CPT)][kk] * Ctb[kk][lc];
            }
          } else {
            for (size_t k0 = 0; k0 < K; k0 += KC) {
              prism_sycl::barrier(it);
#pragma unroll
              for (size_t kk = 0; kk < KC; ++kk)
#pragma unroll
                for (size_t r = 0; r < CPT; ++r)
                  acc[r] += ta0[lr + r * (TILE / CPT)][kk] * tb0[kk][lc];
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

static double ref_gemmd(const std::vector<float>& a, const std::vector<float>& b,
                        size_t M, size_t K, size_t N) {
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


int main(){
  setvbuf(stdout,NULL,_IOLBF,0);
  std::printf("[sweep4] start pid=%d\n",(int)getpid());
  sycl::device gpu; bool have=false;
  for(auto&d:sycl::device::get_devices()) if(d.is_gpu()){gpu=d;have=true;break;}
  if(!have){std::printf("NO GPU\n");return 1;}
  sycl::queue q{gpu, sycl::property::queue::in_order()};

  const size_t M0=1024,K0=1024,N0=1024;
  std::vector<float> a0(M0*K0), b0(K0*N0);
  for(size_t i=0;i<a0.size();++i) a0[i]=0.01f*float(i%97);
  for(size_t i=0;i<b0.size();++i) b0[i]=0.01f*float(i%31);
  const double ref0 = ref_gemmd(a0,b0,M0,K0,N0);

  const size_t M1=4096,K1=1024,N1=4096;
  std::vector<float> a1(M1*K1), b1(K1*N1);
  for(size_t i=0;i<a1.size();++i) a1[i]=0.01f*float(i%97);
  for(size_t i=0;i<b1.size();++i) b1[i]=0.01f*float(i%31);

  float *A0=sycl::malloc_device<float>(a0.size(),q), *B0=sycl::malloc_device<float>(b0.size(),q),
        *C0=sycl::malloc_device<float>(a0.size(),q);
  float *A1=sycl::malloc_device<float>(a1.size(),q), *B1=sycl::malloc_device<float>(b1.size(),q),
        *C1=sycl::malloc_device<float>(a1.size(),q);
  q.submit([&](sycl::handler&h){h.copy(a0.data(),A0,a0.size());});
  q.submit([&](sycl::handler&h){h.copy(b0.data(),B0,b0.size());});
  q.submit([&](sycl::handler&h){h.copy(a1.data(),A1,a1.size());});
  q.submit([&](sycl::handler&h){h.copy(b1.data(),B1,b1.size());});
  q.wait_and_throw();

  double best=0; const char* bestnm="";
  auto bench=[&](const char* nm, auto kernS0, auto kernS1){
    bool ok=false; double gf=0.0;
    try {
      std::vector<float> z(a0.size(),0.f);
      q.submit([&](sycl::handler&h){h.copy(z.data(),C0,a0.size());});
      q.wait_and_throw();
      kernS0();
      std::vector<float> c0(a0.size());
      q.submit([&](sycl::handler&h){h.copy(C0,c0.data(),a0.size());}).wait_and_throw();
      double got=0.0;
      for(size_t i=0;i<c0.size();i+=c0.size()/7+1) got+=c0[i];
      ok = std::fabs(got-ref0) < 1e-2*std::fabs(ref0)+1e-3;
      std::vector<double> t;
      for(int rep=0;rep<5;rep++){
        auto t0=Clock::now(); kernS1(); auto t1=Clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
      }
      std::sort(t.begin(),t.end());
      gf=2.0*M1*K1*N1/(t[t.size()/2]/1000.0)/1e9;
    } catch (const sycl::exception& e) {
      std::printf("  %-24s SYCL ERROR: %s\n", nm, std::string(e.what()).substr(0,80).c_str());
      return;
    }
    std::printf("  %-24s %s  %8.1f GF/s\n", nm, ok?"OK":"FAIL", gf);
    if (ok && gf>best){best=gf; bestnm=nm;}
  };

  std::printf("== SWEEP4a: fine KC around 64x8 spike + CPT variants ==\n");
  #define SW(T,P,K,D) bench(#T "x" #P " kc" #K " db" #D, \
    [&]{ gemm_usm3<T,P,K,D>(q,A0,B0,C0,M0,K0,N0); }, \
    [&]{ gemm_usm3<T,P,K,D>(q,A1,B1,C1,M1,K1,N1); });
  SW(64,8,4,true) SW(64,8,8,true) SW(64,8,12,true) SW(64,8,16,true) SW(64,8,24,true) SW(64,8,32,true)
  SW(64,16,4,true) SW(64,16,8,true) SW(64,16,12,true) SW(64,16,16,true) SW(64,16,24,true) SW(64,16,32,true)
  SW(32,8,16,true) SW(32,8,32,true) SW(32,8,64,true)
  SW(32,4,16,true) SW(32,4,32,true) SW(32,4,64,true)
  SW(64,8,8,false) SW(64,8,12,false) SW(64,8,16,false)
  SW(16,8,32,true) SW(16,8,64,true)
  #undef SW

  std::printf("\n== SWEEP4b: REPEAT top-3 (independent pass, median of 5 again) ==\n");
  SW(64,8,8,true) SW(32,4,32,true) SW(64,16,32,true)

  std::printf("\nBEST: %s = %.1f GF/s (CPU ~83, coeff %.2fx)\n", bestnm, best, best/83.0);
  std::printf("SWEEP4_DONE\n");
  return 0;
}
