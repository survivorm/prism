/* probe_local_mem.cpp — МИНИМАЛЬНЫЙ ВОСПРОИЗВОДИТЕЛЬ механизма локальной памяти.
 *
 * ЗАЧЕМ (16.11.2026): dcurve на карте даёт НУЛИ + порчу кучи, а vpath (без
 * локальной памяти и барьеров) сошёлся с шумом 8.9e-08. Барьер при этом
 * оказался УПОРЯДОЧЕННЫМ: в DPC++ 2026 сигнатура
 *     group_barrier(Group G, memory_scope FenceScope = Group::fence_space)
 * ⇒ вызов group_barrier(g) по умолчанию подставляет Group::fence_space ⇒
 * барьер упорядочивает память ⇒ причина в чём-то другом, и я её не знаю.
 *
 * ⇒ Вместо ещё одной гипотезы — ТАБЛИЦА: каждый шаг механизма проверяется
 * отдельно, печатается «ожидаемо / получено». Сломанный шаг виден сразу.
 *
 * Шаги:
 *   T1  USM: запись ядром → чтение хостом                    (baseline, как vpath)
 *   T2  local_accessor + барьер + чтение (nd_range)           (видимость после барьера)
 *   T3  аксессор передаётся в функцию ПО ЗНАЧЕНИЮ            (паттерн wg_sum)
 *   T4  два work-group'а по 32 (проверка области синхронизации)
 *   T5  read-modify-write в локальной памяти                  (паттерн bitonic sort)
 *   T6  полная сортировка 64 элементов + чтение медианы       (паттерн median dcurve)
 *
 * Запуск: icpx -fsycl -std=c++20 probe/probe_local_mem.cpp -o build/probe_local_mem
 *         ./build/probe_local_mem
 */
#include <sycl/sycl.hpp>

#include "../src/prism_sycl_compat.h"

#include <cstdio>
#include <vector>

static int failures = 0;

static void check(const char* what, double expect, double got) {
    const bool ok = (got == got) && (expect == expect) &&
                    (got - expect < 1e-3) && (expect - got < 1e-3);
    if (!ok) ++failures;
    std::printf("  %-54s ожидаемо %9.1f | получено %9.1f  %s\n",
                what, expect, got, ok ? "OK" : "<<< РАСХОЖДЕНИЕ");
}

/* Суммирование через ОТДЕЛЬНУЮ функцию с аксессором ПО ЗНАЧЕНИЮ —
   ровно тот паттерн, который использует wg_sum в приборе. */
template <class It>
float sum_by_value(It& it, sycl::local_accessor<float, 1> sh, unsigned n) {
    float s = 0.f;
    for (unsigned i = 0; i < n; ++i) s += sh[i];
    return s;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    constexpr size_t N = 64;

    std::printf("=== минимальный воспроизводитель: локальная память и барьер ===\n");
    std::printf("устройство: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("форма барьера, выбранная helper'ом: %s\n",
                prism_sycl::barrier_form_name());

    /* --- T1: USM baseline --- */
    {
        std::vector<float> src(N);
        for (size_t i = 0; i < N; ++i) src[i] = (float)(i + 1);
        float* dev = sycl::malloc_device<float>(N, q);
        q.memcpy(dev, src.data(), N * sizeof(float)).wait();
        std::vector<float> host(N);
        q.memcpy(host.data(), dev, N * sizeof(float)).wait();
        double s = 0;
        for (size_t i = 0; i < N; ++i) s += host[i];
        check("T1 USM: запись ядром → чтение хостом", 2080.0, s);
        sycl::free(dev, q);
    }

    /* --- T2: local_accessor + барьер + чтение --- */
    {
        float* out = sycl::malloc_device<float>(1, q);
        q.submit([&](sycl::handler& h) {
            auto sh = sycl::local_accessor<float, 1>(sycl::range<1>(N), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
                           [=](sycl::nd_item<1> it) {
                const size_t lid = it.get_local_id(0);
                sh[lid] = (float)(lid + 1);
                prism_sycl::barrier(it);
                float s = 0.f;
                for (size_t i = 0; i < N; ++i) s += sh[i];
                if (lid == 0) out[0] = s;
            });
        }).wait();
        float v = 0;
        q.memcpy(&v, out, sizeof(float)).wait();
        check("T2 local_accessor + барьер: сумма 1..64", 2080.0, v);
        sycl::free(out, q);
    }

    /* --- T3: аксессор по значению в функцию --- */
    {
        float* out = sycl::malloc_device<float>(1, q);
        q.submit([&](sycl::handler& h) {
            auto sh = sycl::local_accessor<float, 1>(sycl::range<1>(N), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
                           [=](sycl::nd_item<1> it) {
                const size_t lid = it.get_local_id(0);
                sh[lid] = (float)(lid + 1);
                prism_sycl::barrier(it);
                const float s = sum_by_value(it, sh, (unsigned)N);
                if (lid == 0) out[0] = s;
            });
        }).wait();
        float v = 0;
        q.memcpy(&v, out, sizeof(float)).wait();
        check("T3 аксессор по ЗНАЧЕНИЮ в функцию: сумма 1..64", 2080.0, v);
        sycl::free(out, q);
    }

    /* --- T4: два work-group'а по 32: область синхронизации --- */
    {
        constexpr size_t WG = 32, GROUPS = 2;
        float* out = sycl::malloc_device<float>(GROUPS, q);
        q.submit([&](sycl::handler& h) {
            auto sh = sycl::local_accessor<float, 1>(sycl::range<1>(WG), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(WG * GROUPS),
                                             sycl::range<1>(WG)),
                           [=](sycl::nd_item<1> it) {
                const size_t lid = it.get_local_id(0);
                const size_t gid = it.get_global_id(0);
                sh[lid] = (float)(lid + 1);
                prism_sycl::barrier(it);
                float s = 0.f;
                for (size_t i = 0; i < WG; ++i) s += sh[i];
                if (lid == 0) out[gid / WG] = s;
            });
        }).wait();
        std::vector<float> host(GROUPS);
        q.memcpy(host.data(), out, GROUPS * sizeof(float)).wait();
        for (size_t g = 0; g < GROUPS; ++g) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "T4 группа %zu: сумма 1..32 (своя группа)", g);
            check(buf, 528.0, host[g]);
        }
        sycl::free(out, q);
    }

    /* --- T5: read-modify-write в локальной памяти --- */
    {
        float* out = sycl::malloc_device<float>(1, q);
        q.submit([&](sycl::handler& h) {
            auto sh = sycl::local_accessor<float, 1>(sycl::range<1>(N), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
                           [=](sycl::nd_item<1> it) {
                const size_t lid = it.get_local_id(0);
                for (size_t i = lid; i < N; i += N) sh[i] = (float)(N - i);
                prism_sycl::barrier(it);
                for (size_t i = lid; i < N; i += N) {
                    const size_t ixj = i ^ 1u;
                    if (ixj > i) {
                        const float a1 = sh[i], a2 = sh[ixj];
                        sh[i] = a2; sh[ixj] = a1;
                    }
                }
                prism_sycl::barrier(it);
                float s = 0.f;
                for (size_t i = 0; i < N; ++i) s += sh[i];
                if (lid == 0) out[0] = s;
            });
        }).wait();
        float v = 0;
        q.memcpy(&v, out, sizeof(float)).wait();
        check("T5 RMW (обмен пар) в локальной памяти: сумма сохраняется", 2080.0, v);
        sycl::free(out, q);
    }

    /* --- T6: полная сортировка (паттерн median dcurve) --- */
    {
        float* out = sycl::malloc_device<float>(1, q);
        q.submit([&](sycl::handler& h) {
            auto sh = sycl::local_accessor<float, 1>(sycl::range<1>(N), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(N), sycl::range<1>(N)),
                           [=](sycl::nd_item<1> it) {
                const size_t lid = it.get_local_id(0);
                for (size_t i = lid; i < N; i += N) sh[i] = (float)(N - i);
                prism_sycl::barrier(it);
                for (size_t kk = 2; kk <= N; kk <<= 1) {
                    for (size_t jj = kk >> 1; jj > 0; jj >>= 1) {
                        for (size_t i = lid; i < N; i += N) {
                            const size_t ixj = i ^ jj;
                            if (ixj > i) {
                                const bool up = ((i & kk) == 0);
                                const float a1 = sh[i], a2 = sh[ixj];
                                if ((a1 > a2) == up) { sh[i] = a2; sh[ixj] = a1; }
                            }
                        }
                        prism_sycl::barrier(it);
                    }
                }
                /* РОВНО как в ядре median-dcurve: для чётного n — среднее двух
                   центральных (numpy-семантика). */
                if (lid == 0) out[0] = 0.5f * (sh[N / 2 - 1] + sh[N / 2]);
            });
        }).wait();
        float v = 0;
        q.memcpy(&v, out, sizeof(float)).wait();
        /* после сортировки по возрастанию 1..64: (32+33)/2 = 32.5 */
        check("T6 bitonic sort 1..64: медиана (среднее двух центральных)", 32.5, v);
        sycl::free(out, q);
    }

    std::printf("\n=== ИТОГ: %s ===\n",
                failures == 0 ? "все шаги сошлись"
                              : "ЕСТЬ РАСХОЖДЕНИЕ — сломанный шаг указан выше");
    return failures == 0 ? 0 : 1;
}
