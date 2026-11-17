#include "../src/prism_sycl_compat.h"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <unistd.h>
#include <chrono>
#include <thread>
#include <atomic>
#include <string>
#include <cstdlib>
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


/* gemm_usm4: обобщённый C-блок TR x TC, WG = WGR x WGC <= 512, CPT = TR/WGR строк
   на нить, B в local [k][col]. Коалесценсные блоковые загрузки. */
template <size_t TR, size_t TC, size_t WGR, size_t WGC, size_t KC, bool DB>
void gemm_usm4(sycl::queue& q, const float* A, const float* B, float* C,
               size_t M, size_t K, size_t N) {
  constexpr size_t CPT = TR / WGR;
  static_assert(TR % WGR == 0, "TR kратно WGR");
  static_assert(WGR * WGC <= 512, "WG > 512");
  static_assert(TC <= 512, "TC > 512");
  constexpr size_t NT  = WGR * WGC;
  constexpr size_t RA  = (TR * KC + NT - 1) / NT;
  constexpr size_t RB  = (KC * TC + NT - 1) / NT;
  static_assert(RA >= 1 && RB >= 1);
  const size_t numM = (M + TR - 1) / TR;
  const size_t numN = (N + TC - 1) / TC;
  q.submit([&](sycl::handler& h) {
    auto ta0 = sycl::local_accessor<float, 2>(sycl::range<2>(TR, KC), h);
    auto tb0 = sycl::local_accessor<float, 2>(sycl::range<2>(KC, TC), h);
    auto ta1 = DB ? sycl::local_accessor<float, 2>(sycl::range<2>(TR, KC), h)
                  : sycl::local_accessor<float, 2>(sycl::range<2>(0, 1), h);
    auto tb1 = DB ? sycl::local_accessor<float, 2>(sycl::range<2>(KC, TC), h)
                  : sycl::local_accessor<float, 2>(sycl::range<2>(0, 1), h);
    h.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(numM * WGR, numN * WGC),
                          sycl::range<2>(WGR, WGC)),
        [=](sycl::nd_item<2> it) {
          const size_t lr = it.get_local_id(0);
          const size_t lc = it.get_local_id(1);
          const size_t t  = lr * WGC + lc;
          const size_t row0 = (it.get_global_id(0) / WGR) * TR;
          const size_t col0 = (it.get_global_id(1) / WGC) * TC;

          auto load = [&](auto ta, auto tb, size_t k0) {
#pragma unroll
            for (size_t i = 0; i < RA; ++i) {
              size_t e = t + NT * i;
              if (e < TR * KC) {
                size_t ra = e / KC, ca = e % KC;
                ta[ra][ca] = (row0 + ra < M && k0 + ca < K)
                                 ? A[(row0 + ra) * K + k0 + ca] : 0.f;
              }
            }
#pragma unroll
            for (size_t i = 0; i < RB; ++i) {
              size_t e = t + NT * i;
              if (e < KC * TC) {
                size_t rb = e / TC, cb = e % TC;
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
                  acc[r] += Cta[lr + r * WGR][kk] * Ctb[kk][lc];
            }
          } else {
            for (size_t k0 = 0; k0 < K; k0 += KC) {
              prism_sycl::barrier(it);
#pragma unroll
              for (size_t kk = 0; kk < KC; ++kk)
#pragma unroll
                for (size_t r = 0; r < CPT; ++r)
                  acc[r] += ta0[lr + r * WGR][kk] * tb0[kk][lc];
              if (k0 + KC < K) {
                prism_sycl::barrier(it);
                load(ta0, tb0, k0 + KC);
              }
            }
          }
          for (size_t r = 0; r < CPT; ++r) {
            const size_t row = row0 + lr + r * WGR;
            if (row < M && col0 + lc < N) C[row * N + col0 + lc] = acc[r];
          }
        });
  }).wait_and_throw();
}


int main(){
  setvbuf(stdout,NULL,_IOLBF,0);
  std::printf("[repro] start pid=%d\n",(int)getpid());
  sycl::device gpu; bool have=false;
  for(auto&d:sycl::device::get_devices()) if(d.is_gpu()){gpu=d;have=true;break;}
  if(!have){std::printf("NO GPU\n");return 1;}
  sycl::queue q{gpu, sycl::property::queue::in_order()};

  // Sampler GPU-частоты (для clock-нормировки)
  std::vector<std::pair<double,int>> freqlog;  // (t_s, MHz)
  std::atomic<bool> sampling{true};
  double tstart = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  std::thread freqthr([&]{
    while (sampling.load()) {
      int mhz = 0;
      if (auto* p = popen("intel_gpu_frequency 2>/dev/null | grep '^cur:' | head -1", "r")) {
        char buf[64];
        while (fgets(buf, sizeof buf, p)) {
          int v; char tail[32];
          if (sscanf(buf, "%d %31s", &v, tail) == 2) { mhz = v; break; }
        }
        pclose(p);
      }
      if (mhz > 0) freqlog.push_back({std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - tstart, mhz});
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
  });
  auto avg_mhz = [&](double t0, double t1) {
    int s=0; int n=0;
    for (auto& [t,f] : freqlog) if (t>=t0 && t<=t1) { s+=f; n++; }
    return n? s/(double)n : 300;
  };

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

  struct Cfg { const char* nm; void(*k)(sycl::queue&,const float*,const float*,float*,size_t,size_t,size_t); };
  auto run1=[&](const char* nm, auto kernS0, auto kernS1){
    bool ok=false; double gf=0.0;
    std::vector<float> z(a0.size(),0.f);
    q.submit([&](sycl::handler&h){h.copy(z.data(),C0,a0.size());});
    q.wait_and_throw();
    kernS0();
    std::vector<float> c0(a0.size());
    q.submit([&](sycl::handler&h){h.copy(C0,c0.data(),a0.size());}).wait_and_throw();
    double got=0.0;
    for(size_t i=0;i<c0.size();i+=c0.size()/7+1) got+=c0[i];
    ok = std::fabs(got-ref0) < 1e-2*std::fabs(ref0)+1e-3;
    auto now_s=[&](){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - tstart; };
    double tw0 = now_s();
    std::vector<double> t;
    for(int rep=0;rep<5;rep++){
      auto t0=Clock::now(); kernS1(); auto t1=Clock::now();
      t.push_back(std::chrono::duration<double,std::milli>(t1-t0).count());
    }
    double tw1 = now_s();
    std::sort(t.begin(),t.end());
    gf=2.0*M1*K1*N1/(t[t.size()/2]/1000.0)/1e9;
    double fmhz = avg_mhz(tw0, tw1);
    std::printf("  %-22s %s %8.1f GF/s (clock %.0f MHz, norm@1GHz %8.1f)\n",
                nm, ok?"OK":"FAIL", gf, fmhz, gf*fmhz/1000.0);
  };

  // топ-кандидаты из sweep5 + окрестности 128x64
  run1("128x64 kc16 db", [&]{gemm_usm4<128,64,4,64,16,true >(q,A0,B0,C0,M0,K0,N0);}, [&]{gemm_usm4<128,64,4,64,16,true >(q,A1,B1,C1,M1,K1,N1);});
  run1("128x64 kc8  db", [&]{gemm_usm4<128,64,4,64,8 ,true >(q,A0,B0,C0,M0,K0,N0);}, [&]{gemm_usm4<128,64,4,64,8 ,true >(q,A1,B1,C1,M1,K1,N1);});
  run1("128x64 kc32 db", [&]{gemm_usm4<128,64,4,64,32,true >(q,A0,B0,C0,M0,K0,N0);}, [&]{gemm_usm4<128,64,4,64,32,true >(q,A1,B1,C1,M1,K1,N1);});
  run1("128x64 kc16 noDB",[&]{gemm_usm4<128,64,4,64,16,false>(q,A0,B0,C0,M0,K0,N0);}, [&]{gemm_usm4<128,64,4,64,16,false>(q,A1,B1,C1,M1,K1,N1);});
  run1("128x128 kc8 noDB",[&]{gemm_usm4<128,128,4,128,8,false>(q,A0,B0,C0,M0,K0,N0);}, [&]{gemm_usm4<128,128,4,128,8,false>(q,A1,B1,C1,M1,K1,N1);});
  run1("64x32 kc8 db",   [&]{gemm_usm4<64,32,4,32,8,true  >(q,A0,B0,C0,M0,K0,N0);}, [&]{gemm_usm4<64,32,4,32,8,true  >(q,A1,B1,C1,M1,K1,N1);});
  run1("64x32 kc16 db",  [&]{gemm_usm4<64,32,4,32,16,true >(q,A0,B0,C0,M0,K0,N0);}, [&]{gemm_usm4<64,32,4,32,16,true >(q,A1,B1,C1,M1,K1,N1);});
  run1("64x64 kc8 db",   [&]{gemm_usm4<64,64,4,64,8,true  >(q,A0,B0,C0,M0,K0,N0);}, [&]{gemm_usm4<64,64,4,64,8,true  >(q,A1,B1,C1,M1,K1,N1);});
  run1("32x32 kc32 db",  [&]{gemm_usm4<32,32,4,32,32,true >(q,A0,B0,C0,M0,K0,N0);}, [&]{gemm_usm4<32,32,4,32,32,true >(q,A1,B1,C1,M1,K1,N1);});
  sampling.store(false);
  freqthr.join();
  std::printf("REPRO_DONE\n");
  return 0;
}
