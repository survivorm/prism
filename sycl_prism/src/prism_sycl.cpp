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
constexpr int PS_ERR_NOFP16  = -4;
constexpr int PS_ERR_LOCAL   = -5;

constexpr unsigned WG = 64;   /* размер work-group для всех редукций */

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

/* Сумма по work-group: partials -> sh, редукция на lane 0, результат в sh[0]. */
template <class It>
inline float wg_sum(It& it, sycl::local_accessor<float, 1> sh, float v, unsigned lid,
                    unsigned range) {
    sh[lid] = v;
    prism_sycl::barrier(it);
    if (lid == 0) {
        float s = 0.f;
        const unsigned lim = (range < WG) ? range : WG;
        for (unsigned i = 0; i < lim; ++i) s += sh[i];
        sh[0] = s;
    }
    prism_sycl::barrier(it);
    return sh[0];
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
        sh[0] = m;
    }
    prism_sycl::barrier(it);
    return sh[0];
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
            auto sh = sycl::local_accessor<float, 1>(sycl::range<1>((size_t)WG), h);
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
        auto sh = sycl::local_accessor<float, 1>(sycl::range<1>((size_t)WG), h);
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
        auto sh = sycl::local_accessor<float, 1>(sycl::range<1>((size_t)WG), h);
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
