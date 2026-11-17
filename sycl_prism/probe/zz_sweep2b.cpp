#include "../src/prism_sycl_compat.h"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <cmath>
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

struct Res { const char* nm; bool ok; double gf; };

int main(){
  setvbuf(stdout,NULL,_IOLBF,0);
  std::printf("[sweep2] start pid=%d\n",(int)getpid());
  sycl::device gpu; bool have=false;
  for(auto&d:sycl::device::get_devices()) if(d.is_gpu()){gpu=d;have=true;break;}
  if(!have){std::printf("NO GPU\n");return 1;}
  sycl::queue q{gpu, sycl::property::queue::in_order()};

  // S0 (корректность, 1024^3)
  const size_t M0=1024,K0=1024,N0=1024;
  std::vector<float> a0(M0*K0), b0(K0*N0);
  for(size_t i=0;i<a0.size();++i) a0[i]=0.01f*float(i%97);
  for(size_t i=0;i<b0.size();++i) b0[i]=0.01f*float(i%31);
  const double ref0 = ref_gemmd(a0,b0,M0,K0,N0);

  // S1 (замер, 4096x1024x4096 = 34.4 GFLOP)
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

  std::vector<Res> res;
  double best=0; const char* bestnm="";
  auto bench=[&](const char* nm, auto kern){
    Res r{nm,false,0.0};
    try {
      // S0 check
      std::vector<float> z(a0.size(),0.f);
      q.submit([&](sycl::handler&h){h.copy(z.data(),C0,a0.size());});
      q.wait_and_throw();
      kern();
      std::vector<float> c0(a0.size());
      q.submit([&](sycl::handler&h){h.copy(C0,c0.data(),a0.size());}).wait_and_throw();
      double got=0.0;
      for(size_t i=0;i<c0.size();i+=c0.size()/7+1) got+=c0[i];
      r.ok = std::fabs(got-ref0) < 1e-2*std::fabs(ref0)+1e-3;
      // timing: median of 5
      std::vector<double> t;
      for(int rep=0;rep<5;rep++){
        auto t0=Clock::now(); kern(); auto t1=Clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
      }
      std::sort(t.begin(),t.end());
      double med=t[t.size()/2];
      r.gf=2.0*M1*K1*N1/(med/1000.0)/1e9;
    } catch (const sycl::exception& e) {
      std::printf("  %-24s SYCL ERROR: %s\n", nm, std::string(e.what()).substr(0,80).c_str());
    }
    std::printf("  %-24s %s  %8.1f GF/s\n", nm, r.ok?"OK":"FAIL", r.gf);
    res.push_back(r);
    if (r.ok && r.gf>best){best=r.gf; bestnm=nm;}
  };

  std::printf("== SWEEP2: USM coalesced, S0=1024^3 (чек), S1=4096x1024x4096 (медиана 5) ==\n");
  #define SW(T,P,K,D) bench(#T "x" #P " kc" #K " db" #D, [&]{ gemm_usm2<T,P,K,D>(q,A1,B1,C1,M1,K1,N1); });
  // NOTE: S0-check uses same kern but on S0 buffers — need separate. Fix below.
  #undef SW

  // Correct approach: each bench runs kern on S1; S0 check uses a lambda that
  // calls the same template on S0 buffers.
  auto bench2=[&](const char* nm, auto kernS0, auto kernS1){
    Res r{nm,false,0.0};
    try {
      std::vector<float> z(a0.size(),0.f);
      q.submit([&](sycl::handler&h){h.copy(z.data(),C0,a0.size());});
      q.wait_and_throw();
      kernS0();
      std::vector<float> c0(a0.size());
      q.submit([&](sycl::handler&h){h.copy(C0,c0.data(),a0.size());}).wait_and_throw();
      double got=0.0;
      for(size_t i=0;i<c0.size();i+=c0.size()/7+1) got+=c0[i];
      r.ok = std::fabs(got-ref0) < 1e-2*std::fabs(ref0)+1e-3;
      std::vector<double> t;
      for(int rep=0;rep<5;rep++){
        auto t0=Clock::now(); kernS1(); auto t1=Clock::now();
        t.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
      }
      std::sort(t.begin(),t.end());
      double med=t[t.size()/2];
      r.gf=2.0*M1*K1*N1/(med/1000.0)/1e9;
    } catch (const sycl::exception& e) {
      std::printf("  %-24s SYCL ERROR: %s\n", nm, std::string(e.what()).substr(0,80).c_str());
    }
    std::printf("  %-24s %s  %8.1f GF/s\n", nm, r.ok?"OK":"FAIL", r.gf);
    res.push_back(r);
    if (r.ok && r.gf>best){best=r.gf; bestnm=nm;}
  };
  #define SW(T,P,K,D) bench2(#T "x" #P " kc" #K " db" #D, \
    [&]{ gemm_usm2<T,P,K,D>(q,A0,B0,C0,M0,K0,N0); }, \
    [&]{ gemm_usm2<T,P,K,D>(q,A1,B1,C1,M1,K1,N1); });
  SW(16,1,16,false) SW(16,1,16,true)  SW(16,1,32,false) SW(16,1,32,true)
  SW(16,1,64,false) SW(16,1,64,true)  SW(16,1,128,false) SW(16,1,128,true)
  SW(16,2,16,false) SW(16,2,16,true)  SW(16,2,32,false) SW(16,2,32,true)
  SW(16,2,64,false) SW(16,2,64,true)  SW(16,2,128,false) SW(16,2,128,true)
  SW(32,1,16,false) SW(32,1,16,true)  SW(32,1,32,false) SW(32,1,32,true)
  SW(32,1,64,false) SW(32,1,64,true)  SW(32,1,128,false)
  SW(32,2,16,false) SW(32,2,16,true)  SW(32,2,32,false) SW(32,2,32,true)
  SW(32,2,64,false) SW(32,2,64,true)  SW(32,2,128,false)
  #undef SW

  std::printf("\n== ITGO (GF/s, 4096x1024x4096, медиана 5; пик ~2 TF/s по ревью) ==\n");
  std::printf("%-24s %-6s %8s\n","config","S0","GF/s");
  for (const auto& r : res)
    std::printf("%-24s %-6s %8.1f\n", r.nm, r.ok?"OK":"FAIL", r.gf);
  std::printf("\nBEST (OK): %s = %.1f GF/s (%.0f%% от ~2000 GF/s)\n", bestnm, best, 100.0*best/2000.0);
  std::printf("SWEEP2_DONE\n");
  return 0;
}
