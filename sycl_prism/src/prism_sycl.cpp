/* prism_sycl.cpp — реализация прибора prism на SYCL (Intel iGPU).
 *
 * Что здесь и почему именно это:
 *   - vpath, dcurve (mean|median), acf_moments, R/A-split closure — ровно те
 *     величины, которые prism считает тысячи раз (AT0-реестр: 2880 клеток × 96 лагов).
 *   - вход fp32/fp16, аккумуляция fp32; эталон — float64 (numpy) ⇒ допуски в тесте.
 *   - USM (malloc_device) + явные копирования: без сырых host-указателей в
 *     accessor'ах (спека SYCL 2020 для конструктора accessor'а от указателя строгая).
 *   - median — честная: локальная память + bitonic sort, numpy-семантика
 *     (для чётного N — среднее двух центральных).
 *
 * Геометрия запусков: плоская 1-D, global = n_lags * WG, local = WG (WG=64).
 *   lag_index = id / WG, lane = id % WG. Так не нужен nd_range с кратностью,
 *   которой не кратно число лагов (K=96 не кратно 64).
 *
 * В лямбды ядра захватываются ТОЛЬКО сырые указатели (объекты-обёртки некопируемы).
 */
#include "prism_sycl.h"

#include <sycl/sycl.hpp>

#include "prism_sycl_compat.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace prism_sycl_detail {

constexpr int PS_OK          = 0;
constexpr int PS_ERR_NODEV   = -1;
constexpr int PS_ERR_EXC     = -2;
constexpr int PS_ERR_ARG     = -3;
constexpr int PS_ERR_LOCAL   = -5;

constexpr unsigned WG = 64;   /* размер work-group для всех редукций */

/* PRISM_UNROLL — только для icpx (g++: -Wunknown-pragmas, warning нарушает
   «0 предупреждений»-гейт tests/test_cpp_syntax.py). */
#if defined(__INTEL_LLVM_COMPILER) || defined(__SYCL_DEVICE_ONLY__)
#define PRISM_UNROLL _Pragma("unroll")
#else
#define PRISM_UNROLL
#endif

thread_local std::string g_err;

void set_err(const std::string& s) { g_err = s; }

/* IEEE half -> float (битовая конверсия; без зависимости от sycl::half). */
inline float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t out;
    if (exp == 0u) {
        if (man == 0u) {
            out = sign;
        } else {
            int s = 0;
            while ((man & 0x400u) == 0u) { man <<= 1; ++s; }
            man &= 0x3FFu;
            out = sign | ((uint32_t)(103 + s) << 23) | (man << 13);
        }
    } else if (exp == 31u) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

template <typename T> struct Loader;
template <> struct Loader<float> {
    static inline float load(const float* p) { return *p; }
};
template <> struct Loader<uint16_t> {
    static inline float load(const uint16_t* p) { return h2f(*p); }
};

struct QueueHolder {
    sycl::queue q;
    bool ok = false;
    /* Очередь создаётся через gpu_selector: перечислять бэкенды вручную нельзя —
       в oneAPI 2026 имена sycl::backend::level_zero/cuda/hip отсутствуют, а сам
       бэкенд выбирается рантаймом. in_order — чтобы порядок суммирования был
       детерминированным (важно для воспроизводимости чисел). */
    QueueHolder() {
        try {
            sycl::queue qq{sycl::gpu_selector_v, sycl::property::queue::in_order()};
            if (!qq.get_device().is_gpu()) {
                set_err("sycl: gpu_selector выбрал не GPU");
                ok = false;
                return;
            }
            q = qq;
            ok = true;
        } catch (const std::exception& e) {
            set_err(std::string("no sycl gpu: ") + e.what());
            ok = false;
        }
    }
};

QueueHolder& holder() {
    static QueueHolder h;
    return h;
}

/* RAII-буфер в USM (некопируемый — в ядра не попадает). */
template <typename T>
struct Dev {
    sycl::queue& q;
    T* p = nullptr;
    size_t n = 0;
    Dev(sycl::queue& q_, size_t n_) : q(q_), n(n_) {
        if (n_ > 0) {
            p = sycl::malloc_device<T>(n_, q_);
            if (p == nullptr) { set_err("malloc_device failed"); n = 0; }
        }
    }
    ~Dev() { if (p) sycl::free(p, q); }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
};

/* Сумма по work-group: partials -> sh[0..WG-1], редукция на lane 0,
 * результат в ОТДЕЛЬНОМ слоте sh[WG].
 * ВАЖНО (race-фикс 18.11): результат НЕ в sh[0] — sh[0] это partial lane 0.
 * Между чтением результата (после 2-го барьера) и первым write следующего
 * wg_sum (sh[lid]=v) барьера НЕТ: быстрый lane 0 успевал перезаписать sh[0]
 * partial-суммой следующего вызова, пока медленный lane читал результат
 * предыдущего — расхождение rel между ланеями ⇒ рассинхронный break ⇒
 * deadlock. Отдельный слот sh[WG] перезаписывается только после первого
 * барьера следующего вызова (64 arrival'а — все уже прочитали) ⇒ race нет.
 * Буфер обязан быть размера WG+1. */
template <class It>
inline float wg_sum(It& it, sycl::local_accessor<float, 1> sh, float v, unsigned lid,
                    unsigned range) {
    sh[lid] = v;
    prism_sycl::barrier(it);
    if (lid == 0) {
        float s = 0.f;
        const unsigned lim = (range < WG) ? range : WG;
        for (unsigned i = 0; i < lim; ++i) s += sh[i];
        sh[WG] = s;
    }
    prism_sycl::barrier(it);
    return sh[WG];
}

template <class It>
inline float wg_max(It& it, sycl::local_accessor<float, 1> sh, float v, unsigned lid,
                    unsigned range) {
    sh[lid] = v;
    prism_sycl::barrier(it);
    if (lid == 0) {
        float m = sh[0];
        const unsigned lim = (range < WG) ? range : WG;
        for (unsigned i = 1; i < lim; ++i) { if (sh[i] > m) m = sh[i]; }
        sh[WG] = m;
    }
    prism_sycl::barrier(it);
    return sh[WG];
}

} // namespace prism_sycl_detail

/* ============================== C ABI ================================= */

using namespace prism_sycl_detail;

extern "C" {

int prism_sycl_abi_version(void) { return PRISM_SYCL_ABI_VERSION; }

const char* prism_sycl_last_error(void) { return g_err.c_str(); }

const char* prism_sycl_barrier_form(void) {
    return prism_sycl::barrier_form_name();
}

int prism_sycl_probe(PrismSyclDeviceInfo* info) {
    try {
        QueueHolder& h = holder();
        if (!h.ok) return PS_ERR_NODEV;
        sycl::device d = h.q.get_device();
        if (info != nullptr) {
            std::memset(info, 0, sizeof(*info));
            const std::string nm = d.get_info<sycl::info::device::name>();
            /* Имя бэкенда: в oneAPI 2026 нет ни sycl::info::backend, ни
               sycl::info::device::backend, а перечисление sycl::backend не
               содержит level_zero/cuda/hip. Поэтому бэкенд НЕ опрашивается:
               поле заполняется стабильным маркером, а реальный бэкенд смотрится
               через sycl-ls (см. ROUTES.md). */
            const std::string bk = "unavailable-in-this-sycl";
            const std::string dr = d.get_info<sycl::info::device::driver_version>();
            std::snprintf(info->name, sizeof(info->name), "%s", nm.c_str());
            std::snprintf(info->backend, sizeof(info->backend), "%s", bk.c_str());
            std::snprintf(info->driver, sizeof(info->driver), "%s", dr.c_str());
            info->device_type[0] = d.is_gpu() ? 'g' : (d.is_cpu() ? 'c' : 'a');
            info->max_workgroup_size = (int64_t)d.get_info<sycl::info::device::max_work_group_size>();
            info->global_mem_bytes = (int64_t)d.get_info<sycl::info::device::global_mem_size>();
            info->local_mem_bytes = (int64_t)d.get_info<sycl::info::device::local_mem_size>();
            info->has_fp16 = (int)d.has(sycl::aspect::fp16);
            /* bf16 / sub_group / sub_group_size в oneAPI 2026 отсутствуют
               (sub_group_size переехал в sycl::ext::oneapi::experimental) ⇒
               сообщаем 0 = «не измерено», а не «нет». */
            info->has_bf16 = 0;
            info->has_subgroups = 0;
            info->sub_group_size = 0;
            std::snprintf(info->src_rev, sizeof(info->src_rev), "%s",
                          PRISM_SYCL_SRC_REV);
        }
        return PS_OK;
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

/* Хост-конвертация: буферы УСТРОЙСТВА — fp32 (на Iris Xe аспекта fp64 нет ⇒
   DPC++ не соберёт ни одного ядра с double), а наружу (ABI) отдаём fp64.
   Приведение float→double точное; погрешность вносит только fp32-суммирование
   на устройстве (шумовой пол ~ sqrt(n)·eps_fp32), и он учитывается в допусках. */
static void fetch_f32_as_f64(sycl::queue& q, const float* dev, std::size_t n, double* out) {
    std::vector<float> tmp(n);
    /* wait() ОБЯЗАТЕЛЕН: memcpy D2H асинхронен; без него tmp читается (и
       уничтожается) до завершения DMA, и драйвер дописывает в уже
       переразделённый heap-чанк glibc ⇒ «malloc(): unaligned tcache chunk
       detected» ⇒ abort. (fix v17, docs/TZ_sycl_workstate.md §2 CODE-A) */
    q.memcpy(tmp.data(), dev, n * sizeof(float)).wait();
    for (std::size_t i = 0; i < n; ++i) out[i] = (double)tmp[i];
}

/* ---------------- vpath ---------------- */

} // extern "C"  /* внутренние шаблоны: C linkage недопустим */
namespace prism_sycl_detail {

template <typename InT>
int vpath_impl(sycl::queue& q, const InT* u, const double* hn, int64_t T, int64_t D,
               float* v, uint8_t* keep) {
    Dev<InT> du(q, (size_t)(T * D));
    Dev<float> dhn(q, (size_t)T);
    Dev<float> dv(q, (size_t)((T - 1) * D));
    Dev<uint8_t> dk(q, (size_t)(T - 1));
    if (du.p == nullptr || dhn.p == nullptr || dv.p == nullptr || dk.p == nullptr) return PS_ERR_EXC;

    q.memcpy(du.p, u, (size_t)(T * D) * sizeof(InT));
    std::vector<float> hnf((size_t)T);
    for (int64_t i = 0; i < T; ++i) hnf[(size_t)i] = (float)hn[i];
    q.memcpy(dhn.p, hnf.data(), (size_t)T * sizeof(float));
    q.wait();

    const InT*   pu  = du.p;
    const float* phn = dhn.p;
    float*        pv  = dv.p;
    uint8_t*      pk  = dk.p;

    q.submit([&](sycl::handler& h) {
        /* USM-указатели используются в ядре напрямую: конструктора
           accessor<T,1>(ptr, handler) в SYCL 2020 (oneAPI 2026) НЕТ. */
        h.parallel_for(sycl::range<1>((size_t)(T - 1)), [=](sycl::id<1> id) {
            const int64_t t = id[0];
            /* U[t] = normalize(u[t]*hn[t]) */
            float n0 = 0.f, n1 = 0.f;
            for (int64_t d = 0; d < D; ++d) {
                float a = Loader<InT>::load(pu + t * D + d) * (float)phn[t];
                float b = Loader<InT>::load(pu + (t + 1) * D + d) * (float)phn[t + 1];
                n0 += a * a; n1 += b * b;
            }
            n0 = sycl::sqrt(n0); n1 = sycl::sqrt(n1);
            const float i0 = (n0 > 0.f) ? (1.f / n0) : 0.f;
            const float i1 = (n1 > 0.f) ? (1.f / n1) : 0.f;
            float dot = 0.f;
            for (int64_t d = 0; d < D; ++d) {
                float a = Loader<InT>::load(pu + (t + 1) * D + d) * (float)phn[t + 1] * i1;
                float b = Loader<InT>::load(pu + t * D + d) * (float)phn[t] * i0;
                dot += a * b;
            }
            float vv = 0.f;
            for (int64_t d = 0; d < D; ++d) {
                float a = Loader<InT>::load(pu + (t + 1) * D + d) * (float)phn[t + 1] * i1;
                float b = Loader<InT>::load(pu + t * D + d) * (float)phn[t] * i0;
                float w = a - dot * b;
                pv[t * D + d] = w;
                vv += w * w;
            }
            const float nv = sycl::sqrt(vv);
            if (nv > 1e-6f) {
                const float iv = 1.f / nv;
                for (int64_t d = 0; d < D; ++d) pv[t * D + d] *= iv;
                pk[t] = 1;
            } else {
                for (int64_t d = 0; d < D; ++d) pv[t * D + d] = 0.f;
                pk[t] = 0;
            }
        });
    }).wait();

    /* wait() ОБЯЗАТЕЛЕН: python читает numpy-буферы v/keep сразу после возврата;
       асинхронный D2H давал мусор/NaN (fix v17, §2 CODE-B). */
    q.memcpy(v, dv.p, (size_t)((T - 1) * D) * sizeof(float)).wait();
    q.memcpy(keep, dk.p, (size_t)(T - 1) * sizeof(uint8_t)).wait();
    return PS_OK;
}

/* ---------------- dcurve ---------------- */

template <typename InT>
int dcurve_impl(sycl::queue& q, const InT* P, int64_t T, int64_t D,
                const int64_t* ks, int64_t nk, int stat, double* out) {
    Dev<InT> dP(q, (size_t)(T * D));
    Dev<float> dout(q, (size_t)nk);
    Dev<float> dn(q, (size_t)T);
    /* СПИСОК ЛАГОВ — ХОСТОВЫЙ массив ⇒ копируем на устройство.
       Без этого ядро читало бы хост-память: k = мусор ⇒ T-k = мусор ⇒ запись
       ЗА ПРЕДЕЛЫ буфера ⇒ "malloc(): unaligned tcache chunk detected" и нули
       в результате (rel=1.0). На CPU-шиме это не видно (один адрес. пространство). */
    Dev<int64_t> dks(q, (size_t)nk);
    if (dP.p == nullptr || dout.p == nullptr || dn.p == nullptr || dks.p == nullptr)
        return PS_ERR_EXC;

    /* ВАЛИДАЦИЯ ЛАГОВ НА ХОСТЕ: недопустимый лаг ⇒ громкая ошибка с точным
       значением, а не нули и порча кучи. Именно так проявлялась неисправность:
       k = мусор ⇒ T-k <= 0 ⇒ цикл не выполняется ⇒ нули, а при отрицательном
       T-k — запись мимо буфера.
       Допустим: 1 <= k <= T-1 (k=0 — вырожденный случай, k>=T — нет пары). */
    for (int64_t i = 0; i < nk; ++i) {
        if (ks[i] <= 0 || ks[i] >= T) {
            set_err("dcurve: недопустимый лаг ks[" + std::to_string(i) + "]=" +
                    std::to_string(ks[i]) + " при T=" + std::to_string(T) +
                    " (допустимо 1..T-1)");
            return PS_ERR_ARG;
        }
    }
    q.memcpy(dP.p, P, (size_t)(T * D) * sizeof(InT));
    q.memcpy(dks.p, ks, (size_t)nk * sizeof(int64_t));
    q.wait();

    const InT* pP = dP.p;

    /* нормы строк */
    q.submit([&](sycl::handler& h) {
        float* dn_p = dn.p;
        h.parallel_for(sycl::range<1>((size_t)T), [=](sycl::id<1> id) {
            const int64_t t = id[0];
            float s = 0.f;
            for (int64_t d = 0; d < D; ++d) {
                float v = Loader<InT>::load(pP + t * D + d);
                s += v * v;
            }
            dn_p[t] = sycl::sqrt(s);
        });
    }).wait();

    const float* pn = dn.p;
    const int64_t* pks = dks.p;   /* указатель УСТРОЙСТВА, не хоста */

    if (stat == 0) {
        /* mean по t: work-group на лаг.
           local_accessor создаётся В ОБЛАСТИ command-group, а не внутри ядра:
           его конструктор требует handler& (неконстантный), а внутри [=]-лямбды
           handler константен. По смыслу аксессор принадлежит work-group'у. */
        q.submit([&](sycl::handler& h) {
            float* dout_p = dout.p;
            auto sh = sycl::local_accessor<float, 1>(sycl::range<1>((size_t)WG + 1), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)(nk * WG)),
                                             sycl::range<1>((size_t)WG)),
                           [=](sycl::nd_item<1> it) {
                const size_t gid = it.get_global_id(0);
                const int64_t i = (int64_t)(gid / WG);
                const unsigned lane = (unsigned)(gid % WG);
                const int64_t k = pks[i];
                /* ЗАЩИТА ОТ OOB: даже если k окажется мусором, цикл не выйдет
                   за пределы буфера. (На хосте k уже проверен; это страховка.) */
                if (k <= 0 || k >= T) { if (lane == 0) dout_p[i] = 0.f; return; }
                float part = 0.f;
                for (int64_t t = lane; t < T - k; t += WG) {
                    float dot = 0.f;
                    for (int64_t d = 0; d < D; ++d) {
                        float a = Loader<InT>::load(pP + t * D + d);
                        float b = Loader<InT>::load(pP + (t + k) * D + d);
                        dot += a * b;
                    }
                    const float na = pn[t], nb = pn[t + k];
                    const float den = (na > 0.f && nb > 0.f) ? (na * nb) : 1.f;
                    part += (den > 0.f) ? (dot / den) : 0.f;
                }
                const float tot = wg_sum(it, sh, part, lane, (unsigned)(T - k));
                if (lane == 0) dout_p[i] = tot / (float)(T - k);
            });
        }).wait();
    } else {
        /* median по t: локальная память + bitonic sort */
        int64_t need = 1;
        while (need < T) need <<= 1;
        const size_t local_bytes = (size_t)need * sizeof(float);
        const size_t limit = q.get_device().get_info<sycl::info::device::local_mem_size>();
        if (local_bytes > limit) {
            set_err("median: local_mem too small (need " + std::to_string(local_bytes) +
                    " B, limit " + std::to_string(limit) + " B) — увеличьте T-профиль или " +
                    "считайте медиану на CPU");
            return PS_ERR_LOCAL;
        }
        q.submit([&](sycl::handler& h) {
            float* dout_p = dout.p;
            auto sh = sycl::local_accessor<float, 1>(sycl::range<1>((size_t)need), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)(nk * WG)),
                                             sycl::range<1>((size_t)WG)),
                           [=](sycl::nd_item<1> it) {
                const size_t gid = it.get_global_id(0);
                const int64_t i = (int64_t)(gid / WG);
                const unsigned lane = (unsigned)(gid % WG);
                const int64_t k = pks[i];
                if (k <= 0 || k >= T) { if (lane == 0) dout_p[i] = 0.f; return; }
                const int64_t n = T - k;
                for (int64_t q2 = lane; q2 < need; q2 += WG) sh[q2] = INFINITY;
                prism_sycl::barrier(it);
                for (int64_t t = lane; t < n; t += WG) {
                    float dot = 0.f;
                    for (int64_t d = 0; d < D; ++d) {
                        float a = Loader<InT>::load(pP + t * D + d);
                        float b = Loader<InT>::load(pP + (t + k) * D + d);
                        dot += a * b;
                    }
                    const float na = pn[t], nb = pn[t + k];
                    const float den = (na > 0.f && nb > 0.f) ? (na * nb) : 1.f;
                    sh[t] = (den > 0.f) ? (dot / den) : 0.f;
                }
                prism_sycl::barrier(it);
                for (int64_t kk = 2; kk <= need; kk <<= 1) {
                    for (int64_t jj = kk >> 1; jj > 0; jj >>= 1) {
                        for (int64_t q2 = lane; q2 < need; q2 += WG) {
                            const int64_t ixj = q2 ^ jj;
                            if (ixj > q2) {
                                const bool up = ((q2 & kk) == 0);
                                const float a1 = sh[q2], a2 = sh[ixj];
                                if ((a1 > a2) == up) { sh[q2] = a2; sh[ixj] = a1; }
                            }
                        }
                        prism_sycl::barrier(it);
                    }
                }
                if (lane == 0) {
                    float med;
                    if (n % 2 == 0) med = 0.5f * (sh[n / 2 - 1] + sh[n / 2]);
                    else med = sh[n / 2];
                    dout_p[i] = med;
                }
            });
        }).wait();
    }

    fetch_f32_as_f64(q, dout.p, (size_t)nk, out);
    return PS_OK;
}

/* ---------------- acf ---------------- */

template <typename InT>
void acf_lags(sycl::queue& q, const InT* x, int64_t N, int64_t D, double* c_out) {
    const int64_t NL = N - 1;
    Dev<InT> dx(q, (size_t)(N * D));
    Dev<float> dc(q, (size_t)NL);
    if (dx.p == nullptr || dc.p == nullptr) return;
    q.memcpy(dx.p, x, (size_t)(N * D) * sizeof(InT));
    q.wait();
    const InT* px = dx.p;
    q.submit([&](sycl::handler& h) {
        float* dc_p = dc.p;
        auto sh = sycl::local_accessor<float, 1>(sycl::range<1>((size_t)WG + 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)(NL * WG)),
                                         sycl::range<1>((size_t)WG)),
                       [=](sycl::nd_item<1> it) {
            const size_t gid = it.get_global_id(0);
            const int64_t i = (int64_t)(gid / WG);
            const unsigned lane = (unsigned)(gid % WG);
            const int64_t k = i + 1;
            float part = 0.f;
            for (int64_t t = lane; t < N - k; t += WG) {
                float s = 0.f;
                for (int64_t d = 0; d < D; ++d) {
                    const float a = Loader<InT>::load(px + t * D + d);
                    const float b = Loader<InT>::load(px + (t + k) * D + d);
                    s += a * b;
                }
                part += s;
            }
            const float tot = wg_sum(it, sh, part, lane, (unsigned)(N - k));
            if (lane == 0) dc_p[i] = tot;
        });
    }).wait();
    fetch_f32_as_f64(q, dc.p, (size_t)NL, c_out);
}

template <typename InT>
void rowsumsq(sycl::queue& q, const InT* x, int64_t N, int64_t D, double* e_out) {
    Dev<InT> dx(q, (size_t)(N * D));
    Dev<float> de(q, (size_t)N);
    if (dx.p == nullptr || de.p == nullptr) return;
    q.memcpy(dx.p, x, (size_t)(N * D) * sizeof(InT));
    q.wait();
    const InT* px = dx.p;
    q.submit([&](sycl::handler& h) {
        float* de_p = de.p;
        h.parallel_for(sycl::range<1>((size_t)N), [=](sycl::id<1> id) {
            const int64_t t = id[0];
            float s = 0.f;
            for (int64_t d = 0; d < D; ++d) {
                const float v = Loader<InT>::load(px + t * D + d);
                s += v * v;
            }
            de_p[t] = s;
        });
    }).wait();
    fetch_f32_as_f64(q, de.p, (size_t)N, e_out);
}

template <typename InT>
void rowsum_vec(sycl::queue& q, const InT* x, int64_t N, int64_t D, double* sum_out) {
    Dev<InT> dx(q, (size_t)(N * D));
    Dev<float> ds(q, (size_t)D);
    if (dx.p == nullptr || ds.p == nullptr) return;
    q.memcpy(dx.p, x, (size_t)(N * D) * sizeof(InT));
    q.wait();
    const InT* px = dx.p;
    q.submit([&](sycl::handler& h) {
        float* ds_p = ds.p;
        h.parallel_for(sycl::range<1>((size_t)D), [=](sycl::id<1> id) {
            const int64_t d = id[0];
            float s = 0.f;
            for (int64_t t = 0; t < N; ++t) s += Loader<InT>::load(px + t * D + d);
            ds_p[d] = s;
        });
    }).wait();
    fetch_f32_as_f64(q, ds.p, (size_t)D, sum_out);
}

template <typename InT>
int acf_impl(sycl::queue& q, const InT* x, int64_t N, int64_t D, int64_t K,
             double* c, double* c0, double* r, double* f_dc,
             double* sum_rho_full, double* gram_resid, double* row_norm_sq) {
    const int64_t NL = N - 1;
    std::vector<double> cf((size_t)NL, 0.0);
    std::vector<double> e((size_t)N, 0.0);
    std::vector<double> svec((size_t)D, 0.0);
    acf_lags<InT>(q, x, N, D, cf.data());
    rowsumsq<InT>(q, x, N, D, e.data());
    rowsum_vec<InT>(q, x, N, D, svec.data());

    double c0v = 0.0;
    for (int64_t t = 0; t < N; ++t) c0v += e[t];
    if (c0v <= 0.0) { set_err("acf: c(0)=0 (degenerate object)"); return PS_ERR_ARG; }
    double nrm_sum = 0.0;
    for (int64_t d = 0; d < D; ++d) nrm_sum += svec[d] * svec[d];
    const double rv = nrm_sum / c0v;
    const double fdcv = rv / (double)N;              /* G3: r = N·f_DC */
    double full = 0.0;
    for (int64_t k = 0; k < NL; ++k) full += cf[k];
    const double sumrho = full / c0v;
    const double resid = std::fabs(1.0 + 2.0 * sumrho - rv) / std::fmax(1.0, rv);

    for (int64_t k = 0; k < K && k < NL; ++k) c[k] = cf[k];
    *c0 = c0v;
    *r = rv;
    *f_dc = fdcv;
    *sum_rho_full = sumrho;
    *gram_resid = resid;
    for (int64_t t = 0; t < N; ++t) row_norm_sq[t] = e[t];
    return PS_OK;
}

/* ---------------- R/A-split closure ---------------- */

/* R/A-split closure — СЧИТАЕТСЯ НА ХОСТЕ В fp64, а не на устройстве.
   Причина (установлена исполнением): closure_max — величина порядка ошибки
   ФП (~1e-13), т.е. это остаток тождества G6, а не число. На устройстве в fp32
   шум ~1e-6·scale ⇒ величина НЕРАЗРЕШИМА (во сто раз крупнее самой closure).
   На Iris Xe аспекта fp64 нет вовсе ⇒ DPC++ не соберёт такое ядро.
   Стоимость на хосте: T·D (384×1024) fp64 ⇒ пренебрежимо.
   Формулировка та же, что в эталоне prism/field.py, но в fp64 на хосте. */
template <typename InT>
int ra_impl(sycl::queue& /*q*/, const InT* u, const double* hn, int64_t T, int64_t D,
            double* closure_max, double* du_u_cross_max) {
    if (u == nullptr || hn == nullptr || closure_max == nullptr ||
        du_u_cross_max == nullptr || T < 2 || D < 1) {
        set_err("ra_closure: неверные аргументы");
        return PS_ERR_ARG;
    }
    std::vector<double> nrm((size_t)T, 0.0);
    for (int64_t t = 0; t < T; ++t) {
        double s = 0.0;
        for (int64_t d = 0; d < D; ++d) {
            const double v = (double)Loader<InT>::load(u + t * D + d);
            s += v * v;
        }
        nrm[(size_t)t] = std::sqrt(s);
    }
    double cl = 0.0, cr = 0.0;
    for (int64_t t = 0; t < T - 1; ++t) {
        const double h0 = hn[t], h1 = hn[t + 1];
        const double n0 = nrm[(size_t)t], n1 = nrm[(size_t)t + 1];
        const double i0 = (n0 > 0.0) ? (1.0 / n0) : 0.0;
        const double i1 = (n1 > 0.0) ? (1.0 / n1) : 0.0;
        const double rbar = 0.5 * (h0 + h1);
        const double dr = h1 - h0;
        double dhn_ = 0.0, cdu = 0.0, cang = 0.0, crad = 0.0;
        for (int64_t d = 0; d < D; ++d) {
            const double u0 = (double)Loader<InT>::load(u + t * D + d) * i0;
            const double u1 = (double)Loader<InT>::load(u + (t + 1) * D + d) * i1;
            const double dhu = (u1 * h1) - (u0 * h0);
            const double duu = u1 - u0;
            const double umid = 0.5 * (u1 + u0);
            const double rad = dr * umid;
            const double ang = rbar * duu;
            dhn_ += dhu * dhu;
            crad += rad * rad;
            cang += ang * ang;
            cdu += duu * umid;
        }
        const double a1 = std::fabs(dhn_ - crad - cang);
        const double a2 = std::fabs(cdu);
        if (a1 > cl) cl = a1;
        if (a2 > cr) cr = a2;
    }
    *closure_max = cl;
    *du_u_cross_max = cr;
    return PS_OK;
}

} // namespace prism_sycl_detail
using namespace prism_sycl_detail;

extern "C" {

int prism_sycl_vpath_f32(const float* u, const double* hn, int64_t T, int64_t D,
                         float* v, uint8_t* keep) {
    try {
        QueueHolder& h = holder();
        if (!h.ok) return PS_ERR_NODEV;
        if (T < 2 || D <= 0) return PS_ERR_ARG;
        return vpath_impl<float>(h.q, u, hn, T, D, v, keep);
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

int prism_sycl_vpath_f16(const uint16_t* u, const double* hn, int64_t T, int64_t D,
                         float* v, uint8_t* keep) {
    try {
        std::vector<float> tmp((size_t)(T * D));
        for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = h2f(u[i]);
        return prism_sycl_vpath_f32(tmp.data(), hn, T, D, v, keep);
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

int prism_sycl_dcurve_f32(const float* P, int64_t T, int64_t D,
                           const int64_t* ks, int64_t nk, int stat, double* out) {
    try {
        QueueHolder& h = holder();
        if (!h.ok) return PS_ERR_NODEV;
        if (T < 2 || D <= 0 || nk <= 0) return PS_ERR_ARG;
        return dcurve_impl<float>(h.q, P, T, D, ks, nk, stat, out);
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

int prism_sycl_dcurve_f16(const uint16_t* P, int64_t T, int64_t D,
                           const int64_t* ks, int64_t nk, int stat, double* out) {
    try {
        std::vector<float> tmp((size_t)(T * D));
        for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = h2f(P[i]);
        return prism_sycl_dcurve_f32(tmp.data(), T, D, ks, nk, stat, out);
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

int prism_sycl_acf_f32(const float* x, int64_t N, int64_t D, int64_t K,
                       double* c, double* c0, double* r, double* f_dc,
                       double* sum_rho_full, double* gram_resid, double* row_norm_sq) {
    try {
        QueueHolder& h = holder();
        if (!h.ok) return PS_ERR_NODEV;
        if (N < 2 || D <= 0 || K < 1 || K > N - 1) return PS_ERR_ARG;
        return acf_impl<float>(h.q, x, N, D, K, c, c0, r, f_dc, sum_rho_full,
                               gram_resid, row_norm_sq);
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

int prism_sycl_acf_f16(const uint16_t* x, int64_t N, int64_t D, int64_t K,
                       double* c, double* c0, double* r, double* f_dc,
                       double* sum_rho_full, double* gram_resid, double* row_norm_sq) {
    try {
        std::vector<float> tmp((size_t)(N * D));
        for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = h2f(x[i]);
        return prism_sycl_acf_f32(tmp.data(), N, D, K, c, c0, r, f_dc, sum_rho_full,
                                  gram_resid, row_norm_sq);
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

int prism_sycl_ra_closure_f32(const float* u, const double* hn, int64_t T, int64_t D,
                               double* closure_max, double* du_u_cross_max) {
    try {
        QueueHolder& h = holder();
        if (!h.ok) return PS_ERR_NODEV;
        if (T < 2 || D <= 0) return PS_ERR_ARG;
        return ra_impl<float>(h.q, u, hn, T, D, closure_max, du_u_cross_max);
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

int prism_sycl_ra_closure_f16(const uint16_t* u, const double* hn, int64_t T, int64_t D,
                               double* closure_max, double* du_u_cross_max) {
    try {
        std::vector<float> tmp((size_t)(T * D));
        for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = h2f(u[i]);
        return prism_sycl_ra_closure_f32(tmp.data(), hn, T, D, closure_max, du_u_cross_max);
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

/* ---------------- перестановочный null-слой ---------------- */

} // extern "C"  /* внутренние шаблоны: C linkage недопустим */
namespace prism_sycl_detail {

template <typename InT>
void perm_rowsq(sycl::queue& q, const InT* x, int64_t N, int64_t D,
                const int32_t* idx, int64_t n_perm, double* out) {
    Dev<InT> dx(q, (size_t)(N * D));
    Dev<int32_t> di(q, (size_t)(n_perm * N));
    Dev<float> de(q, (size_t)(n_perm * N));
    if (dx.p == nullptr || di.p == nullptr || de.p == nullptr) return;
    q.memcpy(dx.p, x, (size_t)(N * D) * sizeof(InT));
    q.memcpy(di.p, idx, (size_t)(n_perm * N) * sizeof(int32_t));
    q.wait();
    const InT* px = dx.p;
    const int32_t* pi = di.p;
    q.submit([&](sycl::handler& h) {
        float* de_p = de.p;
        h.parallel_for(sycl::range<1>((size_t)(n_perm * N)), [=](sycl::id<1> id) {
            const size_t idl = id[0];
            const int64_t p = (int64_t)(idl / N);
            const int64_t t = (int64_t)(idl % N);
            const int64_t src = pi[p * N + t];
            float s = 0.f;
            for (int64_t d = 0; d < D; ++d) {
                const float v = Loader<InT>::load(px + src * D + d);
                s += v * v;
            }
            de_p[idl] = s;
        });
    }).wait();
    fetch_f32_as_f64(q, de.p, (size_t)(n_perm * N), out);
}

template <typename InT>
void perm_lags(sycl::queue& q, const InT* x, int64_t N, int64_t D,
               const int32_t* idx, int64_t n_perm, const int64_t* ks, int64_t nk,
               double* c_out) {
    Dev<InT> dx(q, (size_t)(N * D));
    Dev<int32_t> di(q, (size_t)(n_perm * N));
    Dev<float> dc(q, (size_t)(n_perm * nk));
    /* список лагов — тоже хостовый массив ⇒ копируем на устройство (см. dcurve) */
    Dev<int64_t> dks(q, (size_t)nk);
    if (dx.p == nullptr || di.p == nullptr || dc.p == nullptr || dks.p == nullptr) return;
    q.memcpy(dx.p, x, (size_t)(N * D) * sizeof(InT));
    q.memcpy(di.p, idx, (size_t)(n_perm * N) * sizeof(int32_t));
    q.memcpy(dks.p, ks, (size_t)nk * sizeof(int64_t));
    q.wait();
    const InT* px = dx.p;
    const int32_t* pi = di.p;
    const int64_t* pks = dks.p;   /* указатель УСТРОЙСТВА, не хоста */
    q.submit([&](sycl::handler& h) {
        float* dc_p = dc.p;
        auto sh = sycl::local_accessor<float, 1>(sycl::range<1>((size_t)WG + 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)(n_perm * nk * WG)),
                                         sycl::range<1>((size_t)WG)),
                       [=](sycl::nd_item<1> it) {
            const size_t gid = it.get_global_id(0);
            const int64_t g = (int64_t)(gid / WG);
            const unsigned lane = (unsigned)(gid % WG);
            const int64_t p = g / nk;
            const int64_t ki = g % nk;
            const int64_t k = pks[ki];
            float part = 0.f;
            for (int64_t t = lane; t < N - k; t += WG) {
                const int64_t s0 = pi[p * N + t];
                const int64_t s1 = pi[p * N + t + k];
                float s = 0.f;
                for (int64_t d = 0; d < D; ++d) {
                    const float a = Loader<InT>::load(px + s0 * D + d);
                    const float b = Loader<InT>::load(px + s1 * D + d);
                    s += a * b;
                }
                part += s;
            }
            const float tot = wg_sum(it, sh, part, lane, (unsigned)(N - k));
            if (lane == 0) dc_p[p * nk + ki] = tot;
        });
    }).wait();
    fetch_f32_as_f64(q, dc.p, (size_t)(n_perm * nk), c_out);
}

} // namespace prism_sycl_detail
using namespace prism_sycl_detail;

extern "C" {

int prism_sycl_perm_row_sq_f32(const float* x, int64_t N, int64_t D,
                                const int32_t* perm_idx, int64_t n_perm, double* row_sq) {
    try {
        QueueHolder& h = holder();
        if (!h.ok) return PS_ERR_NODEV;
        if (N < 2 || D <= 0 || n_perm <= 0) return PS_ERR_ARG;
        perm_rowsq<float>(h.q, x, N, D, perm_idx, n_perm, row_sq);
        return PS_OK;
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

int prism_sycl_perm_row_sq_f16(const uint16_t* x, int64_t N, int64_t D,
                                const int32_t* perm_idx, int64_t n_perm, double* row_sq) {
    try {
        std::vector<float> tmp((size_t)(N * D));
        for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = h2f(x[i]);
        return prism_sycl_perm_row_sq_f32(tmp.data(), N, D, perm_idx, n_perm, row_sq);
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

int prism_sycl_perm_lags_f32(const float* x, int64_t N, int64_t D,
                              const int32_t* perm_idx, int64_t n_perm,
                              const int64_t* ks, int64_t nk, double* c_out) {
    try {
        QueueHolder& h = holder();
        if (!h.ok) return PS_ERR_NODEV;
        if (N < 2 || D <= 0 || n_perm <= 0 || nk <= 0) return PS_ERR_ARG;
        perm_lags<float>(h.q, x, N, D, perm_idx, n_perm, ks, nk, c_out);
        return PS_OK;
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

int prism_sycl_perm_lags_f16(const uint16_t* x, int64_t N, int64_t D,
                              const int32_t* perm_idx, int64_t n_perm,
                              const int64_t* ks, int64_t nk, double* c_out) {
    try {
        std::vector<float> tmp((size_t)(N * D));
        for (size_t i = 0; i < tmp.size(); ++i) tmp[i] = h2f(x[i]);
        return prism_sycl_perm_lags_f32(tmp.data(), N, D, perm_idx, n_perm, ks, nk, c_out);
    } catch (const std::exception& e) {
        set_err(e.what());
        return PS_ERR_EXC;
    }
}

} // extern "C"

/* =====================================================================
 * AT0_3 (TZ_sycl_new_ops.md v5, 2026-11-17): gemm / eigtop / eigbot / eigjac
 *
 * Паттерны библиотеки (как существующие операции):
 *   - USM (malloc_device) + явные memcpy; wait() после D2H (n-21);
 *   - in_order-очередь ⇒ детерминизм (порядок чанков/суммирования фиксирован);
 *   - сырые USM-указатели в ядрах (не accessor'ы);
 *   - fp32-арифметика на устройстве (Iris Xe без fp64-асpekта);
 *   - наружу — fp64 через точное float→double (fetch_f32_as_f64 / memcpy+каст).
 *
 * GEMM — ядро gemm_usm4 в конфиге, зафиксированном тюнингом
 *   (probe/RESULTS_tuning.md: TR=128, TC=64, WGR=4, WGC=64, KC=16,
 *   double-buffer, SLM=24KB); N-чанкинг с бюджетом 32 GFLOP на launch
 *   (hang-check драйвера, TZ §2.1; сигнатура не меняется — чанкинг внутри).
 * EIGTOP/EIGBOT — power/inverse iteration + МГС (двойной, n-14): один WG=64
 *   на матрицу; V/W — local memory (явный хост-бюджет, PS_ERR_LOCAL);
 *   A/L — GLOBAL USM; Rayleigh и подпространственный остаток считаются на
 *   ХОСТЕ в f64 (A уже в хосте, V — малый D2H): точнее, чем ядерный f32;
 *   выход по контракту TZ (vals f64, vecs f32, resid f64 — ЧЕСТНЫЙ отчёт,
 *   НЕ критерий остановки; стоп — max_iter).
 * EIGJAC — cyclic Jacobi, полный спектр симметричного батча (n ≤ 256):
 *   A/V — GLOBAL USM; local memory — c/s/rt1/rt2 по парам + wg_sum-буфер.
 *   Механизм (race-фикс 18.11) — как у XLA eigh (openxla/xla, Apache 2.0,
 *   hlo/transforms/expanders/eigh_expander.cc): sweep = n-1 раундов
 *   perfect-matching КРЕСТ-КВАДРАНТНЫХ пар (l, k+l), k = n/2; в раунде row-pass
 *   + col-pass, где лань держит целую пару (каждая ячейка — ровно один writer
 *   ⇒ гонки нет ПО СТРОЕНИЮ) + interleave-перестановка (Brent/Luk round-robin,
 *   single (n-1)-cycle) — за n-1 раундов каждая пара индексов покрыта ровно
 *   один раз. Результат раунда = J_all^T·A·J_all (disjойнтные ротации
 *   коммутируют). Параллельность — WG=64 ланей на матрицу × batch.
 *   Почему не «наивный» multi-lane (реплика /tmp/jac4_replica.cpp): поддержка
 *   ротации = целые строки+колонки {a,c}, поэтому ЛЮБЫЕ две ротации в одном
 *   параллельном шаге пишут кросс-ячейки {a,c}×{a',c'} разными частичными
 *   формулами ⇒ data-race ДАЖЕ на «disjoint»-парах (1,2) и (3,4) гоняются на
 *   A[1,3]; бит-расхождение par-прогонов — с шага 1. 1 work-item/матрицу —
 *   детерминирован, но слишком медленный на iGPU (2.95 c/sweep, driver
 *   watchdog на ~8 sweep'ах) ⇒ отвергнут.
 */

namespace prism_sycl_detail {

/* --- GEMM: тейлованное ядро (gemm_usm4 из probe/zz_sweep5.cpp) -----------
 * C-блок TR x TC, WG = WGR x WGC (<= 512), CPT = TR/WGR строк на нить, B в
 * local [k][col]. Nfull — ширина С (для записи), Nc — ширина чанка B
 * (граны/обход OOB). Запись C ТОЛЬКО при (m<M ∧ n<N) — ревью m-15. */
template <size_t TR, size_t TC, size_t WGR, size_t WGC, size_t KC, bool DB>
static void gemm_tile(sycl::queue& q, const Dev<float>& dA, const Dev<float>& dB,
                      Dev<float>& dC, size_t M, size_t K, size_t Nfull, size_t Nc,
                      size_t B_off) {
  constexpr size_t CPT = TR / WGR;
  static_assert(TR % WGR == 0, "TR кратно WGR");
  static_assert(WGR * WGC <= 512, "WG > max 512");
  static_assert(TC <= 512, "TC > 512");
  constexpr size_t NT  = WGR * WGC;
  constexpr size_t RA  = (TR * KC + NT - 1) / NT;
  constexpr size_t RB  = (KC * TC + NT - 1) / NT;
  static_assert(RA >= 1 && RB >= 1);
  const float* A = dA.p;
  const float* B = dB.p + B_off;   // старт N-чанка (ширина Nc)
  float* C = dC.p;
  const size_t numM = (M + TR - 1) / TR;
  const size_t numN = (Nc + TC - 1) / TC;
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
PRISM_UNROLL
            for (size_t i = 0; i < RA; ++i) {
              size_t e = t + NT * i;
              if (e < TR * KC) {
                size_t ra = e / KC, ca = e % KC;
                ta[ra][ca] = (row0 + ra < M && k0 + ca < K)
                                 ? A[(row0 + ra) * K + k0 + ca] : 0.f;
              }
            }
PRISM_UNROLL
            for (size_t i = 0; i < RB; ++i) {
              size_t e = t + NT * i;
              if (e < KC * TC) {
                size_t rb = e / TC, cb = e % TC;
                tb[rb][cb] = (k0 + rb < K && col0 + cb < Nc)
                                 ? B[(k0 + rb) * Nc + col0 + cb] : 0.f;
              }
            }
          };

          load(ta0, tb0, 0);
          float acc[CPT];
PRISM_UNROLL
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
PRISM_UNROLL
              for (size_t kk = 0; kk < KC; ++kk)
PRISM_UNROLL
                for (size_t r = 0; r < CPT; ++r)
                  acc[r] += Cta[lr + r * WGR][kk] * Ctb[kk][lc];
            }
          } else {
            for (size_t k0 = 0; k0 < K; k0 += KC) {
              prism_sycl::barrier(it);
PRISM_UNROLL
              for (size_t kk = 0; kk < KC; ++kk)
PRISM_UNROLL
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
            if (row < M && col0 + lc < Nfull) C[row * Nfull + col0 + lc] = acc[r];
          }
        });
  }).wait_and_throw();
}

/* Конфиг, зафиксированный тюнингом (probe/RESULTS_tuning.md, 17.11): 168–170
   GF/s sustained; SLM = (128*16 + 16*64)*4*2 = 24KB. Большие тайлы
   (128x128, 256x128, 512x*) тюнингом НЕ подтверждены — не использовать. */
constexpr size_t GEMM_TR = 128, GEMM_TC = 64, GEMM_WGR = 4, GEMM_WGC = 64, GEMM_KC = 16;
/* Бюджет на один launch, TZ §2.1: <= 32 GFLOP (hang-check драйвера; запас
   ~3-5x от худшего окна; на 6.5 GF/s — <= 5 с на чанк). */
constexpr double GEMM_FLOP_BUDGET = 32e9;

static int gemm_impl(sycl::queue& q, const float* hA, const float* hB,
                     int64_t M, int64_t K, int64_t N, float* hC) {
  if (M < 1 || M > 65536 || K < 1 || K > 4096 || N < 1 || N > 262144) {
    set_err("gemm: ограничения размеров (1<=M<=65536, 1<=K<=4096, 1<=N<=262144), "
            "получено M=" + std::to_string(M) + " K=" + std::to_string(K) +
            " N=" + std::to_string(N));
    return PS_ERR_ARG;
  }
  /* Явный хост-бюджет local memory (double-buffer): при нарушении — PS_ERR_LOCAL.
     (Константный конфиг — проверка формальна, но паттерн обязателен по TZ.) */
  const size_t slm_bytes = (size_t)(GEMM_TR * GEMM_KC + GEMM_KC * GEMM_TC) * 4 * 2;
  if (slm_bytes > 65536) {
    set_err("gemm: local memory budget " + std::to_string(slm_bytes) + " > 65536");
    return PS_ERR_LOCAL;
  }
  Dev<float> dA(q, (size_t)M * (size_t)K);
  Dev<float> dB(q, (size_t)K * (size_t)N);
  Dev<float> dC(q, (size_t)M * (size_t)N);
  if (!dA.p || !dB.p || !dC.p) {
    set_err("gemm: malloc_device failed (M=" + std::to_string(M) +
            " K=" + std::to_string(K) + " N=" + std::to_string(N) + ")");
    return PS_ERR_EXC;
  }
  q.memcpy(dA.p, hA, (size_t)M * (size_t)K * 4).wait();
  q.memcpy(dB.p, hB, (size_t)K * (size_t)N * 4).wait();
  /* Чанкинг по N (TZ §2.1): N_chunk = floor_to_tile(32e9 / (2*M*K), TC=64);
     детерминизм: in_order-очередь, порядок чанков фиксирован, каждый чанк
     пишет в несомещаемую часть C. Крайний случай (M*K огромен): N_chunk = TC,
     чанк может на ~8% превысить бюджет (65536x4096x64 = 34.4 GFLOP) — в окне
     запаса. */
  const double nch_f = GEMM_FLOP_BUDGET / (2.0 * (double)M * (double)K);
  size_t nch = ((size_t)std::floor(nch_f) / GEMM_TC) * GEMM_TC;
  if (nch < GEMM_TC) nch = GEMM_TC;
  for (size_t c0 = 0; c0 < (size_t)N; c0 += nch) {
    const size_t Nc = std::min(nch, (size_t)N - c0);
    gemm_tile<GEMM_TR, GEMM_TC, GEMM_WGR, GEMM_WGC, GEMM_KC, true>(
        q, dA, dB, dC, (size_t)M, (size_t)K, (size_t)N, Nc, c0 * (size_t)K);
  }
  q.wait();
  q.memcpy(hC, dC.p, (size_t)M * (size_t)N * 4).wait();
  return PS_OK;
}

/* --- LCG (TZ m-13): a=1103515245, c=12345, m=2^31 ------------------------
 * c ≠ 0; сид = batch*7919 + col*104729 + 12345 (при batch=0/col=0 сид = 12345
 * ≠ 0). Перезапуск вырожденной колонки — сид + 7919. */
inline uint32_t lcg_state(uint32_t st) {
  return (uint32_t)(((uint64_t)st * 1103515245ull + 12345ull) & 0x7FFFFFFFull);
}

static inline uint32_t eig_seed(int64_t b, int64_t j, int64_t extra) {
  return (uint32_t)(b * 7919 + (uint64_t)j * 104729 + 12345 + (uint64_t)extra);
}

/* --- eigtop/eigbot: power/inverse iteration + МГС -------------------------
 * Один WG=64 на матрицу. local: lv/lw — [n][r] (1D, индексы i*r+j), sh — 64.
 * INV=false (eigtop): шаг W = A·V (A в GLOBAL, построчно).
 * INV=true  (eigbot): шаг = два triangular-сolves: L·W = V (forward),
 *   затем (Lᵀ)·W = W (backward, in-place, i от n-1 вниз: строка i читается
 *   как старая ДО затирания, W[k>i] — уже новые — корректно). Lower-tri
 *   solve последователен по строкам ⇒ параллельность только по колонкам
 *   (тред t < r — колонка t; TZ «n=128: 2 строки/нить» — это про шаг AV,
 *   для solves параллельны колонки, не строки — см. само-аудит).
 * МГС двойной (n-14): один проход даёт ортогональность ~√n·eps ≈ 1e-5 при
 *   n=128 (на грани требования 1e-5) — второй проход доводит до ~eps.
 * После цикла — V в global; Rayleigh/resid — на хосте (f64). */
template <bool INV>
static void eig_iterate(sycl::queue& q, const Dev<float>& dA, const Dev<float>& dL,
                        Dev<float>& dV, int64_t batch, int64_t n, int64_t r,
                        int64_t max_iter) {
  const int64_t n2 = n * n;
  const float* A = dA.p;
  const float* L = dL.p;
  float* V = dV.p;
  q.submit([&](sycl::handler& h) {
    const float* pA = A;
    const float* pL = L;
    float* pV = V;
    auto lv = sycl::local_accessor<float, 1>(sycl::range<1>((size_t)n * (size_t)r), h);
    auto lw = sycl::local_accessor<float, 1>(sycl::range<1>((size_t)n * (size_t)r), h);
    auto sh = sycl::local_accessor<float, 1>(sycl::range<1>((size_t)WG + 1), h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t)batch * WG),
                                     sycl::range<1>((size_t)WG)),
                   [=](sycl::nd_item<1> it) {
      const size_t gid = it.get_global_id(0);
      const int64_t b = (int64_t)(gid / WG);
      const int lane = (int)(gid % WG);
      const float* A = pA + (size_t)b * n2;
      const float* L = pL + (size_t)b * n2;
      const int64_t nr = n * r;

      // (1) V0 — LCG (тред 0; компонент k — k-е значение последовательности)
      if (lane == 0) {
        for (int64_t j = 0; j < r; ++j) {
          uint32_t st = eig_seed(b, j, 0);
          for (int64_t k = 0; k < n; ++k) {
            st = lcg_state(st);
            lv[k * r + j] = (float)st / 2147483648.0f;
          }
        }
      }
      prism_sycl::barrier(it);

      for (int64_t iter = 0; iter < max_iter; ++iter) {
        if (!INV) {
          // (2) W = A·V, построчно (тред t — строки t, t+64, …)
          for (int64_t i = lane; i < n; i += WG) {
            const float* ar = A + i * n;
            for (int64_t j = 0; j < r; ++j) {
              float s = 0.f;
              for (int64_t k = 0; k < n; ++k) s += ar[k] * lv[k * r + j];
              lw[i * r + j] = s;
            }
          }
        } else {
          if (lane < r) {
            for (int64_t i = 0; i < n; ++i) {
              float s = lv[i * r + lane];
              const float* lr_ = L + i * n;
              for (int64_t k = 0; k < i; ++k) s -= lr_[k] * lw[k * r + lane];
              lw[i * r + lane] = s / lr_[i];
            }
          }
          prism_sycl::barrier(it);
          if (lane < r) {
            for (int64_t i = n - 1; i >= 0; --i) {
              float s = lw[i * r + lane];   // старое значение строки i (ещё не затёрто)
              for (int64_t k = i + 1; k < n; ++k) s -= L[k * n + i] * lw[k * r + lane];
              lw[i * r + lane] = s / L[i * n + i];
            }
          }
          prism_sycl::barrier(it);
        }
        // (3a) МГС + нормировка: lw (W) → lv (V), колонка за колонкой.
        // Критерий остановки — max_iter (TZ); λ/resid — только финальные.
        for (int64_t j = 0; j < r; ++j) {
          for (int64_t l2 = 0; l2 < j; ++l2) {
            float dot = 0.f;
            for (int64_t i = lane; i < n; i += WG) dot += lw[i * r + j] * lv[i * r + l2];
            const float d = wg_sum(it, sh, dot, (unsigned)lane, (unsigned)n);
            for (int64_t i = lane; i < n; i += WG) lw[i * r + j] -= d * lv[i * r + l2];
            prism_sycl::barrier(it);
          }
          float nrm2 = 0.f;
          for (int64_t i = lane; i < n; i += WG) nrm2 += lw[i * r + j] * lw[i * r + j];
          float n2v = wg_sum(it, sh, nrm2, (unsigned)lane, (unsigned)n);
          if (n2v < 1e-24f) {   // ‖w‖ < 1e-12 — вырожденная колонка (TZ)
            if (lane == 0) {
              uint32_t st = eig_seed(b, j, 7919);   // перезапуск — сид + 7919
              for (int64_t k = 0; k < n; ++k) {
                st = lcg_state(st);
                lw[k * r + j] = (float)st / 2147483648.0f;
              }
            }
            prism_sycl::barrier(it);
            nrm2 = 0.f;
            for (int64_t i = lane; i < n; i += WG) nrm2 += lw[i * r + j] * lw[i * r + j];
            n2v = wg_sum(it, sh, nrm2, (unsigned)lane, (unsigned)n);
          }
          const float inv = 1.f / sqrtf(n2v);
          for (int64_t i = lane; i < n; i += WG) lv[i * r + j] = lw[i * r + j] * inv;
          prism_sycl::barrier(it);
        }
        // (3b) двойной МГС (n-14): повторная ортогонализация по обновлённой основе
        for (int64_t j = 0; j < r; ++j) {
          for (int64_t l2 = 0; l2 < j; ++l2) {
            float dot = 0.f;
            for (int64_t i = lane; i < n; i += WG) dot += lv[i * r + j] * lv[i * r + l2];
            const float d = wg_sum(it, sh, dot, (unsigned)lane, (unsigned)n);
            for (int64_t i = lane; i < n; i += WG) lv[i * r + j] -= d * lv[i * r + l2];
            prism_sycl::barrier(it);
          }
          float nrm2 = 0.f;
          for (int64_t i = lane; i < n; i += WG) nrm2 += lv[i * r + j] * lv[i * r + j];
          const float n2v = wg_sum(it, sh, nrm2, (unsigned)lane, (unsigned)n);
          const float inv = 1.f / sqrtf(n2v);
          for (int64_t i = lane; i < n; i += WG) lv[i * r + j] = lv[i * r + j] * inv;
          prism_sycl::barrier(it);
        }
      }
      // (5) V в global
      for (int64_t i = lane; i < n; i += WG)
        for (int64_t j = 0; j < r; ++j) pV[(size_t)b * nr + (size_t)i * r + (size_t)j] = lv[i * r + j];
    });
  }).wait_and_throw();
}

/* Хост-постобработка (f64): Rayleigh λ_j = ⟨v_j, A v_j⟩ и подпространственный
 * остаток resid = ‖AV − V(VᵀAV)‖_F/‖A‖_F на последнем V. A — входной хостовый
 * массив (f32 → double точно); V — малый D2H. Точнее ядерного f32; выход по
 * контракту TZ. NaN в A → NaN в resid (честный сигнал, не тихий мусор). */
static void eig_postprocess(const float* hA, const float* hV, int64_t batch,
                            int64_t n, int64_t r, double* hVals, double* hResid) {
  for (int64_t b = 0; b < batch; ++b) {
    const float* Ab = hA + (size_t)b * n * n;
    const float* Vb = hV + (size_t)b * n * r;
    std::vector<double> W((size_t)n * r);
    double nrmA2 = 0.0;
    for (int64_t i = 0; i < n; ++i) {
      for (int64_t k = 0; k < n; ++k) {
        const double a = Ab[i * n + k];
        nrmA2 += a * a;
      }
      for (int64_t j = 0; j < r; ++j) {
        double s = 0.0;
        for (int64_t k = 0; k < n; ++k) s += (double)Ab[i * n + k] * (double)Vb[k * r + j];
        W[i * r + j] = s;
      }
    }
    const double nrmA = std::sqrt(nrmA2);
    // P = VᵀW (r×r)
    std::vector<double> P((size_t)r * r, 0.0);
    for (int64_t i = 0; i < n; ++i)
      for (int64_t i2 = 0; i2 < r; ++i2)
        for (int64_t j = 0; j < r; ++j)
          P[i2 * r + j] += (double)Vb[i * r + i2] * W[i * r + j];
    // R = W − V·P; resid = ‖R‖_F / ‖A‖_F
    double res2 = 0.0;
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < r; ++j) {
        double s = W[i * r + j];
        for (int64_t i2 = 0; i2 < r; ++i2) s -= (double)Vb[i * r + i2] * P[i2 * r + j];
        res2 += s * s;
      }
    hResid[b] = (nrmA > 0.0) ? std::sqrt(res2) / nrmA : 0.0;
    for (int64_t j = 0; j < r; ++j) {
      double s = 0.0;
      for (int64_t i = 0; i < n; ++i) s += (double)Vb[i * r + j] * W[i * r + j];
      hVals[(size_t)b * r + (size_t)j] = s;
    }
  }
}

static int eig_impl(sycl::queue& q, const float* hA, const float* hL,
                    int64_t batch, int64_t n, int64_t r, int64_t max_iter,
                    double* hVals, float* hVecs, double* hResid, bool inverse) {
  const char* op = inverse ? "eigbot" : "eigtop";
  if (batch < 1 || batch > 65536 || n < 1 || n > 1024 || r < 1 || r > 32 ||
      r >= n || max_iter < 1 || max_iter > 512) {
    set_err(std::string(op) + ": ограничения (1<=batch<=65536, 1<=n<=1024, 1<=r<=32, r<n, "
                            "1<=max_iter<=512), получено batch=" +
            std::to_string(batch) + " n=" + std::to_string(n) + " r=" + std::to_string(r) +
            " max_iter=" + std::to_string(max_iter));
    return PS_ERR_ARG;
  }
  /* Явный хост-бюджет local memory (TZ: 2·n·r·4 + r²·4 + 64·4 ≤ 65536;
     n=1024: r ≤ 7, r=8 ⇒ 66048 > 65536 — m-12b). */
  const size_t local_bytes = (size_t)2 * n * r * 4 + (size_t)r * r * 4 + 64 * 4;
  if (local_bytes > 65536) {
    set_err(std::string(op) + ": local memory budget 2*n*r*4 + r*r*4 + 64*4 = " +
            std::to_string(local_bytes) + " > 65536 (n=1024: r<=7)");
    return PS_ERR_LOCAL;
  }
  Dev<float> dA(q, (size_t)batch * n * n);
  Dev<float> dL(q, inverse ? (size_t)batch * n * n : 0);
  Dev<float> dV(q, (size_t)batch * n * r);
  if (!dA.p || (inverse && !dL.p) || !dV.p) {
    set_err(std::string(op) + ": malloc_device failed (batch=" + std::to_string(batch) +
            " n=" + std::to_string(n) + " r=" + std::to_string(r) + ")");
    return PS_ERR_EXC;
  }
  q.memcpy(dA.p, hA, (size_t)batch * n * n * 4).wait();
  if (inverse) q.memcpy(dL.p, hL, (size_t)batch * n * n * 4).wait();
  if (!inverse) {
    eig_iterate<false>(q, dA, dL, dV, batch, n, r, max_iter);
  } else {
    eig_iterate<true>(q, dA, dL, dV, batch, n, r, max_iter);
  }
  std::vector<float> Vh((size_t)batch * n * r);
  q.memcpy(Vh.data(), dV.p, Vh.size() * 4).wait();
  eig_postprocess(hA, Vh.data(), batch, n, r, hVals, hResid);
  memcpy(hVecs, Vh.data(), Vh.size() * 4);
  return PS_OK;
}

// ============================================================================
// EIGJAC v5.3 — sequential cyclic Jacobi, 1 work-item/матрицу, batch-чанкинг,
// sweep-луп на хосте. Заменяет eigjac_run (XLA parallel matching) в
// src/prism_sycl.cpp. TZ_sycl_new_ops §2.3 (v5.3; дельта-ревью v5.2 F1–F7).
//
// Изменения v5.3 vs драфт v5.2:
//  (1) F1: batch-чанкинг — каждый launch ≤ EIGJAC_CHUNK матриц (hang-check
//      ~20с; t_sweep(C) ≤ 4с — твёрдый тест-гейт §2.7, C фиксируется тюнингом
//      AT0_3, начальное 1024);
//  (2) fp64-аспекта на iGPU НЕТ ⇒ device-буферы offdiag/diag — f32
//      (4·batch B); f64 — только на хосте (TZ v5.2 «8·batch B» — несовместимо);
//  (3) curve: явный контракт F3 — curve[s−1] = off[b=0] после sweep s (s=1..k);
//      curve[k..curve_len) = offdiag[0] (явный fill до конца буфера).
// ============================================================================

/* --- eigjac v5.3: последовательный циклический Jacobi ---------------------
 * Почему НЕ параллельный матчинг (XLA eigh_expander, отклонён замером v5.2):
 * simultaneous-приложение disjойнтных ротаций раунда = Jacobi-итерация по
 * парам: линейное схождение ratio 0.997–0.9999/sweep; на 6 реальных jcache
 * 128×128 (конд. ~1e8–1e9) 64 sweep'а → offdiag_rel 0.069–0.59, НИ ОДНА не
 * достигла 1e-4. Любая intra-матричная параллельность либо race
 * (кросс-ячейки — реплика /tmp/jac4_replica.cpp), либо этот механизм.
 * Циклический порядок: sweep = для d = 1..n-1: для i = 0..n-1-d: rot(i, i+d)
 * — ровно n(n-1)/2 ротаций, каждая пара один раз. Квадратичное схождение:
 * 4–5 sweep'ов до 1e-4, 4–6 до 1e-5 на всех 6 реальных jcache (замер v5.2,
 * f64/f32 идентично).
 * Детерминизм: один work-item на матрицу, фиксированный порядок ротаций,
 * локальной памяти НЕТ, редукции — только offdiag (фикс. порядок суммирования
 * в одном work-item) и host-MGS (фикс. порядок) ⇒ бит-в-бит повторный прогон.
 * Watchdog (F1): 1 launch = 1 чанк ≤ C матриц; t_sweep(C) ≤ 4с (тест §2.7,
 * 5×-запас к hang-check ~20с); в v5.2 «1 sweep ≈ 3с» был замером на
 * одиночных матрицах (batch=1) — без чанкинга batch=16384 (AT2c) / 65536
 * выходил за hang-check. */

/* C — фиксация тюнинга AT0_3 (замер t_sweep по чанкам {1..4096}, n=128, iGPU;
   правило: C = наибольшее чанк с t_sweep(C) ≤ 4с). Начальное: 1024.
   n > 128 (F9): чанк = C/4 — стоимость sweep на матрицу ∝ n(n−1)/2: для
   n=256 это ×4.02 к n=128 — тот же 5×-запас к hang-check ~20с. */
constexpr int64_t EIGJAC_CHUNK = 1024;

static void eigjac_v_init(sycl::queue& q, Dev<float>& dV, int64_t batch, int64_t n) {
  float* dV_p = dV.p;
  q.submit([&](sycl::handler& h) {
    float* pV = dV_p;
    h.parallel_for(sycl::range<1>((size_t)batch * n * n),
                   [=](sycl::id<1> id) {
                     const size_t g = (size_t)id[0];
                     const int64_t b = (int64_t)(g / ((size_t)n * n));
                     const size_t r = g % ((size_t)n * n);
                     const int64_t i = (int64_t)(r / (size_t)n);
                     const int64_t j = (int64_t)(r % (size_t)n);
                     pV[(size_t)b * (size_t)n * (size_t)n + r] = (i == j) ? 1.f : 0.f;
                   });
  });
}

// 1 launch = ЧАНК [b0, b0+chunk) матриц; work-item — своя матрица.
static void eigjac_sweep_chunk(sycl::queue& q, Dev<float>& dA, Dev<float>& dV,
                               Dev<float>& dOff, Dev<float>& dDiag,
                               int64_t b0, int64_t chunk, int64_t n) {
  float* dA_p = dA.p;
  float* dV_p = dV.p;
  float* dOff_p = dOff.p;
  float* dDiag_p = dDiag.p;
  q.submit([&](sycl::handler& h) {
    float* pA = dA_p; float* pV = dV_p; float* pOff = dOff_p; float* pDiag = dDiag_p;
    h.parallel_for(sycl::range<1>((size_t)chunk), [=](sycl::id<1> id) {
      const int64_t b = b0 + (int64_t)id[0];
      const int64_t n64 = n;
      float* A_ = pA + (size_t)b * n64 * n64;
      float* V_ = pV + (size_t)b * n64 * n64;
      // Циклический порядок: d = 1..n-1, i = 0..n-1-d, j = i+d.
      for (int64_t d = 1; d < n64; ++d) {
        for (int64_t i = 0; i + d < n64; ++i) {
          const int64_t j = i + d;
          const float a = A_[i * n64 + i];
          const float bq = A_[i * n64 + j];
          const float dd = A_[j * n64 + j];
          // fudge-guard XLA (F5: eps_f32 = 1.1920929e-7):
          // |A_ij| <= 0.1*eps_f32*min(|A_ii|,|A_jj|) — ротация не применяется.
          const float tiny = 0.1f * 1.1920929e-7f * fminf(fabsf(a), fabsf(dd));
          if (bq == 0.0f || fabsf(bq) <= tiny) continue;
          // Малый корень: t = (A_jj−A_ii)/(2·A_ij), th = sign(t)/(|t|+sqrt(1+t^2)),
          // c = 1/sqrt(1+th^2), s = th*c (|th| <= 1 — нулирует A[i,j]).
          const float tt = (dd - a) / (2.0f * bq);
          const float th = (tt == 0.0f) ? 1.0f
              : copysignf(1.0f, tt) / (fabsf(tt) + sqrtf(1.0f + tt * tt));
          const float c = 1.0f / sqrtf(1.0f + th * th);
          const float s = th * c;
          // A: кросс-строки/колонки i,j (симметрия: строка = колонка).
          for (int64_t k = 0; k < n64; ++k) {
            if (k == i || k == j) continue;
            const float ai = A_[i * n64 + k];
            const float aj = A_[j * n64 + k];
            const float ni = c * ai - s * aj;
            const float nj = s * ai + c * aj;
            A_[i * n64 + k] = ni;
            A_[j * n64 + k] = nj;
            A_[k * n64 + i] = ni;
            A_[k * n64 + j] = nj;
          }
          A_[i * n64 + i] = a - th * bq;   // rt1 (диагональ)
          A_[j * n64 + j] = dd + th * bq;  // rt2 (диагональ)
          A_[i * n64 + j] = 0.0f;
          A_[j * n64 + i] = 0.0f;
          // V: колонки i,j (V' = V·M) — ВКЛЮЧАЯ блок {i,j}×{i,j}
          // (баг-фикс драфта v5.2: пропуск k==i/j в блоке V ломал V).
          for (int64_t k = 0; k < n64; ++k) {
            const float vi = V_[k * n64 + i];
            const float vj = V_[k * n64 + j];
            V_[k * n64 + i] = c * vi - s * vj;
            V_[k * n64 + j] = s * vi + c * vj;
          }
        }
      }
      // offdiag_rel (f32 на устройстве — iGPU без fp64-aspect; фиксированный
      // порядок суммирования в одном work-item — детерминизм) + диагональ.
      float tot = 0.f, off2 = 0.f;
      for (int64_t i = 0; i < n64; ++i) {
        for (int64_t k = 0; k < n64; ++k) {
          const float x = A_[i * n64 + k];
          tot += x * x;
          if (i != k) off2 += x * x;
        }
        pDiag[(size_t)b * n64 + (size_t)i] = A_[i * n64 + i];
      }
      pOff[b] = (tot > 0.f) ? sqrtf(off2 / tot) : 0.f;
    });
  });
}

// host-MGS (f64): V — [batch,n,n] f32, столбцы = собственные векторы;
// modified Gram-Schmidt, 1 проход, на месте → f32 (F4: write-back в vecs f32).
// Ортогональность f32-Jacobi при n=128 ~2.5e-4 > допуска 1e-4 (замер v5.2) —
// MGS даёт ~5e-15 в f64-расчёте; после f32-cast замер: ‖VᵀV−I‖ ≈ 3e-7
// (n=128, 6 реальных jcache) ≤ допуска 1e-4; subdist УЛУЧШАЕТСЯ
// (1.4e-4 → 8.4e-5, замер L6 d0t0).
static void eigjac_mgs_host(float* hV, int64_t batch, int64_t n) {
  std::vector<double> W((size_t)n * n);
  for (int64_t b = 0; b < batch; ++b) {
    float* Vb = hV + (size_t)b * n * n;
    for (int64_t col = 0; col < n; ++col)
      for (int64_t i = 0; i < n; ++i) W[i * n + col] = (double)Vb[i * n + col];
    for (int64_t jj = 0; jj < n; ++jj) {
      for (int64_t i = 0; i < jj; ++i) {
        double dot = 0.0;
        for (int64_t k = 0; k < n; ++k) dot += W[k * n + i] * W[k * n + jj];
        for (int64_t k = 0; k < n; ++k) W[k * n + jj] -= dot * W[k * n + i];
      }
      double nrm = 0.0;
      for (int64_t k = 0; k < n; ++k) nrm += W[k * n + jj] * W[k * n + jj];
      const double r = std::sqrt(nrm);
      if (r > 0.0)
        for (int64_t k = 0; k < n; ++k) W[k * n + jj] /= r;
    }
    for (int64_t i = 0; i < n; ++i)
      for (int64_t col = 0; col < n; ++col) Vb[i * n + col] = (float)W[i * n + col];
  }
}

// Сортировка пар (λ f64, вектор f32) по возрастанию λ, столбцы vecs — вместе
// с λ (stable; порядок torch).
static void sort_eigpairs_f64_host(const double* hVals64, float* hVecs32,
                                   int64_t batch, int64_t n,
                                   double* hOutVals, float* hOutVecs) {
  for (int64_t b = 0; b < batch; ++b) {
    std::vector<int64_t> idx(n);
    for (int64_t i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t i, int64_t j) {
      return hVals64[(size_t)b * n + (size_t)i] < hVals64[(size_t)b * n + (size_t)j];
    });
    for (int64_t i = 0; i < n; ++i) {
      hOutVals[(size_t)b * n + (size_t)i] = hVals64[(size_t)b * n + (size_t)idx[i]];
      for (int64_t row = 0; row < n; ++row)
        hOutVecs[((size_t)b * n + (size_t)row) * n + (size_t)i] =
            hVecs32[((size_t)b * n + (size_t)row) * n + (size_t)idx[i]];
    }
  }
}

static int eigjac_impl(sycl::queue& q, const float* hA, int64_t batch, int64_t n,
                       int64_t max_sweeps, double tol, double* hVals, float* hVecs,
                       double* hOff, double* hCurve, int64_t curve_len) {
  if (batch < 1 || batch > 65536 || n < 1 || n > 256 || max_sweeps < 1 ||
      max_sweeps > 64 || tol < 1e-6 || tol > 1e-3) {
    set_err("eigjac: ограничения (1<=batch<=65536, 1<=n<=256, 1<=max_sweeps<=64, "
            "1e-6<=tol<=1e-3), получено batch=" + std::to_string(batch) +
            " n=" + std::to_string(n) + " max_sweeps=" + std::to_string(max_sweeps) +
            " tol=" + std::to_string(tol));
    return PS_ERR_ARG;
  }
  if (hCurve != nullptr && (curve_len < 0 || curve_len < max_sweeps)) {
    set_err("eigjac: curve_len=" + std::to_string(curve_len) +
            " < max_sweeps=" + std::to_string(max_sweeps) +
            " (буфер кривой должен держать все sweep'ы)");
    return PS_ERR_ARG;
  }
  /* v5.3: локальной памяти НЕТ (один work-item на матрицу); форма проверки
     сохранена для единообразия с eigtop/eigbot (TZ: 0 ≤ 65536). */
  const size_t local_bytes = 0;
  if (local_bytes > 65536) {
    set_err("eigjac: local memory budget " + std::to_string(local_bytes) + " > 65536");
    return PS_ERR_LOCAL;
  }
  Dev<float> dA(q, (size_t)batch * n * n);
  Dev<float> dV(q, (size_t)batch * n * n);
  Dev<float> dOff(q, (size_t)batch);      /* f32 (iGPU без fp64-aspect) */
  Dev<float> dDiag(q, (size_t)batch * n); /* f32 (λ — f64 только на хосте) */
  if (!dA.p || !dV.p || !dOff.p || !dDiag.p) {
    set_err("eigjac: malloc_device failed (batch=" + std::to_string(batch) +
            " n=" + std::to_string(n) + ")");
    return PS_ERR_EXC;
  }
  q.memcpy(dA.p, hA, (size_t)batch * n * n * 4).wait();
  eigjac_v_init(q, dV, batch, n);
  /* Sweep-луп на хосте (v5.3 F1: 1 sweep = ⌈batch/C⌉ запусков чанков; между
     sweep'ами хост читает offdiag (4·batch B, f32) и останавливается, когда
     ВСЕ матрицы < tol (иначе — max_sweeps). curve (F3): curve[s−1] =
     offdiag_rel матрицы 0 после sweep s; curve[k..curve_len) = offdiag[0]. */
  std::vector<float> off32((size_t)batch);
  std::vector<double> off64((size_t)batch);
  // F9: n > 128 — чанк C/4 (стоимость sweep на матрицу ×4.02 к n=128).
  const int64_t chunk_max = (n > 128) ? EIGJAC_CHUNK / 4 : EIGJAC_CHUNK;
  int64_t stop_sw = 0;
  for (int64_t sw = 1; sw <= max_sweeps; ++sw) {
    for (int64_t b0 = 0; b0 < batch; b0 += chunk_max) {
      const int64_t chunk = std::min<int64_t>(chunk_max, batch - b0);
      eigjac_sweep_chunk(q, dA, dV, dOff, dDiag, b0, chunk, n);
    }
    q.memcpy(off32.data(), dOff.p, off32.size() * 4).wait();
    for (int64_t b = 0; b < batch; ++b) off64[b] = (double)off32[b];
    if (hCurve != nullptr) hCurve[sw - 1] = off64[0];
    bool all_ok = true;
    for (int64_t b = 0; b < batch; ++b)
      if (!(off64[b] < tol)) { all_ok = false; break; }
    if (all_ok) { stop_sw = sw; break; }
  }
  if (hCurve != nullptr && stop_sw > 0) {
    // F3: финальное значение = offdiag[0]; fill от индекса stop_sw до конца
    // буфера (при стопе по max_sweeps цикл выше записал [0..max_sweeps),
    // остаток [max_sweeps..curve_len) — тоже финальное).
    const int64_t last = std::min<int64_t>(stop_sw, max_sweeps);
    for (int64_t sw = last; sw < curve_len; ++sw) hCurve[sw] = off64[0];
  }
  /* Д2Х: диагональ (f32→f64) и V (f32). offdiag[batch] — значение на
     остановке (f64, хост). */
  std::vector<float> diag32((size_t)batch * n);
  q.memcpy(diag32.data(), dDiag.p, diag32.size() * 4).wait();
  std::vector<double> diag64(diag32.size());
  for (size_t i = 0; i < diag32.size(); ++i) diag64[i] = (double)diag32[i];
  std::vector<float> vecs32((size_t)batch * n * n);
  q.memcpy(vecs32.data(), dV.p, vecs32.size() * 4).wait();
  std::vector<float> out_vecs(vecs32.size());
  sort_eigpairs_f64_host(diag64.data(), vecs32.data(), batch, n, hVals, out_vecs.data());
  eigjac_mgs_host(out_vecs.data(), batch, n);
  memcpy(hVecs, out_vecs.data(), out_vecs.size() * 4);
  for (int64_t b = 0; b < batch; ++b) hOff[b] = off64[b];
  return PS_OK;
}

} // namespace prism_sycl_detail
using namespace prism_sycl_detail;

extern "C" {

int prism_sycl_gemm_f32(const float* A, const float* B,
                        int64_t M, int64_t K, int64_t N, float* C) {
  try {
    QueueHolder& h = holder();
    if (!h.ok) return PS_ERR_NODEV;
    return gemm_impl(h.q, A, B, M, K, N, C);
  } catch (const std::exception& e) {
    set_err(e.what());
    return PS_ERR_EXC;
  }
}

int prism_sycl_eigtop(const float* A, int64_t batch, int64_t n, int64_t r,
                      int64_t max_iter, double* vals, float* vecs, double* resid) {
  try {
    QueueHolder& h = holder();
    if (!h.ok) return PS_ERR_NODEV;
    return eig_impl(h.q, A, nullptr, batch, n, r, max_iter, vals, vecs, resid, false);
  } catch (const std::exception& e) {
    set_err(e.what());
    return PS_ERR_EXC;
  }
}

int prism_sycl_eigbot(const float* A, const float* Lfac, int64_t batch, int64_t n,
                      int64_t r, int64_t max_iter, double* vals, float* vecs,
                      double* resid) {
  try {
    QueueHolder& h = holder();
    if (!h.ok) return PS_ERR_NODEV;
    return eig_impl(h.q, A, Lfac, batch, n, r, max_iter, vals, vecs, resid, true);
  } catch (const std::exception& e) {
    set_err(e.what());
    return PS_ERR_EXC;
  }
}

int prism_sycl_eigjac(const float* A, int64_t batch, int64_t n,
                      int64_t max_sweeps, double tol,
                      double* vals, float* vecs, double* offdiag,
                      double* offdiag_curve, int64_t curve_len) {
  try {
    QueueHolder& h = holder();
    if (!h.ok) return PS_ERR_NODEV;
    return eigjac_impl(h.q, A, batch, n, max_sweeps, tol, vals, vecs, offdiag,
                       offdiag_curve, curve_len);
  } catch (const std::exception& e) {
    set_err(e.what());
    return PS_ERR_EXC;
  }
}

} // extern "C"
