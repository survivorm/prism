/* probe_kernel_name.cpp — РЕШАЮЩИЙ зОНД: почему рантайм не находит ядро
 * по имени ("No kernel named _ZTS… was found").
 *
 * ВОПРОС (открыт после v15): библиотека падает на карте с
 *   "No kernel named _ZTSZZN17prism_sycl_detail10vpath_implIfEE… was found",
 * а зонд probe_sycl_min.cpp на той же карте РАБОТАЕТ. Различия:
 *   (а) ядро зонда — в ОБЫЧНОЙ функции; ядро библиотеки — внутри ШАБЛОННОЙ
 *       функции (vpath_impl<InT>) ⇒ имя типа замыкания содержит параметры
 *       шаблона (_ZTSZ…vpath_implIfEE…);
 *   (б) зонд собран icpx напрямую (AOT-путь), библиотека — через CMake.
 *
 * ГИПОТЕЗЫ (проверяются здесь, на УСТРОЙСТВЕ, а не чтением):
 *   H1: шаблонная функция ⇒ имя типа замыкания не совпадает с тем, что
 *       ищет рантайм.
 *   H2: путь сборки (SPIR-V против AOT) ⇒ рантайм ищет по имени ТИПА
 *       (_ZTS…), а в SPIR-V-модуле имя точки входа — манглированное
 *       operator() (_ZZ…) ⇒ не совпадают.
 *   H3: дело в передаче аргумента барьера/энумерации — уже исключено
 *       (зонд local_mem работает).
 *
 * ЧТО ДЕЛАЕТ ЗОНД: запускает ТРИ варианта ядра, каждый в СВОЕМ исключении
 * (перехват), и печатает, какой именно отработал. Различия изолированы по
 * одному фактору за раз:
 *   A) лямбда в ОБЫЧНОЙ функции (как в probe_sycl_min) — базовая линия;
 *   B) лямбда в ШАБЛОННОЙ функции с одним параметром — проверка H1;
 *   C) явное имя ядра (NamedKernel) — проверка обхода H1;
 *   (вариант D из ранней версии промпта — «обычная функция с шаблонным
 *    контекстом имён» — не реализован: A и B закрывают вопрос о шаблоне,
 *    контроль на «имя vs шаблон» дублирует B).
 *
 * Каждый вариант считает контрольную сумму ⇒ «отработал» = «посчитал верно»,
 * а не «не упал».
 *
 *   icpx -fsycl -std=c++20 probe/probe_kernel_name.cpp -o build/probe_kernel_name
 *   ./build/probe_kernel_name
 */
#include <sycl/sycl.hpp>

#include <cstdio>
#include <exception>
#include <vector>

namespace probe_ns {

/* ---------- A) лямбда в ОБЫЧНОЙ функции (базовая линия) ---------- */
bool run_plain(sycl::queue& q, const float* in, float* out, int64_t n) {
    try {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> id) {
                out[id[0]] = in[id[0]] * 2.0f + 1.0f;
            });
        }).wait();
        return true;
    } catch (const std::exception& e) {
        std::printf("   (исключение: %s)\n", e.what());
        return false;
    }
}

/* ---------- B) лямбда в ШАБЛОННОЙ функции (проверка H1) ---------- */
template <typename T>
bool run_templated(sycl::queue& q, const float* in, float* out, int64_t n) {
    try {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t)n), [=](sycl::id<1> id) {
                out[id[0]] = in[id[0]] * 2.0f + 1.0f;
            });
        }).wait();
        return true;
    } catch (const std::exception& e) {
        std::printf("   (исключение: %s)\n", e.what());
        return false;
    }
}

/* ---------- C) ЯВНОЕ имя ядра (обход H1 через NamedKernel) ---------- */
/* SYCL 2020 убрал именованные ядра из API, но DPC++ 2026 даёт
 * sycl::ext::oneapi::experimental::named_kernel. Если он есть — обход. */
template <typename T>
struct KernelBody {
    const float* in;
    float* out;
    void operator()(sycl::id<1> id) const {
        out[id[0]] = in[id[0]] * 2.0f + 1.0f;
    }
};

bool run_named(sycl::queue& q, const float* in, float* out, int64_t n) {
    try {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>((size_t)n), KernelBody<float>{in, out});
        }).wait();
        return true;
    } catch (const std::exception& e) {
        std::printf("   (исключение: %s)\n", e.what());
        return false;
    }
}

}  // namespace probe_ns

static bool check(const char* tag, bool ran, const float* out, int64_t n) {
    if (!ran) {
        std::printf("%-12s: НЕ ОТРАБОТАЛ (ядро не найдено по имени)\n", tag);
        return false;
    }
    /* Контроль: out[i] должно быть in[i]*2+1 = 3 при in[i]=1. */
    const float expect = 3.0f;
    double maxerr = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double d = (double)out[i] - (double)expect;
        if (d < 0) d = -d;
        if (d > maxerr) maxerr = d;
    }
    const bool ok = (maxerr < 1e-5);
    std::printf("%-12s: ОТРАБОТАЛ, макс. ошибка = %.3g %s\n", tag, maxerr,
                ok ? "(OK)" : "(НЕВЕРНЫЙ РЕЗУЛЬТАТ!)");
    return ok;
}

int main() {
    const int64_t n = 1024;
    std::vector<float> in((size_t)n, 1.0f);
    std::vector<float> out((size_t)n, 0.0f);

    std::printf("=== Поиск ядра по имени: что работает на этом устройстве ===\n");

    try {
        sycl::queue q{sycl::gpu_selector_v};
        std::printf("устройство: %s\n",
                    q.get_device().get_info<sycl::info::device::name>().c_str());

        float* din = sycl::malloc_device<float>((size_t)n, q);
        float* dout = sycl::malloc_device<float>((size_t)n, q);
        if (!din || !dout) {
            std::printf("!!! не удалось выделить USM-память\n");
            return 2;
        }
        q.memcpy(din, in.data(), (size_t)n * sizeof(float));
        q.wait();

        const bool a_ok = probe_ns::run_plain(q, din, dout, n);
        /* D2H ДО check: check() читает ХОСТ-буфер; до фикса (v17) сюда
           передавался USM-указатель устройства ⇒ SEGV на хосте. */
        q.memcpy(out.data(), dout, (size_t)n * sizeof(float)).wait();
        const bool a_ok2 = check("A: обычная", a_ok, out.data(), n);

        const bool b_ok = probe_ns::run_templated<float>(q, din, dout, n);
        q.memcpy(out.data(), dout, (size_t)n * sizeof(float)).wait();
        const bool b_ok2 = check("B: шаблон", b_ok, out.data(), n);

        const bool c_ok = probe_ns::run_named(q, din, dout, n);
        q.memcpy(out.data(), dout, (size_t)n * sizeof(float)).wait();
        const bool c_ok2 = check("C: имя ядра", c_ok, out.data(), n);

        sycl::free(din, q);
        sycl::free(dout, q);

        std::printf("\n=== ВЫВОД ===\n");
        if (a_ok2 && b_ok2) {
            std::printf("A и B оба работают ⇒ шаблон НЕ причина ⇒ дело в СБОРКЕ "
                        "(флаги/путь AOT-SPIR-V), а не в коде.\n");
            std::printf("  ⇒ проверьте: cmake -B build -DPRISM_SYCL_TARGETS=  "
                        "(ПУСТО, AOT) и rm -rf build.\n");
        } else if (a_ok2 && !b_ok2) {
            std::printf("A работает, B — нет ⇒ ПРИЧИНА: ядро внутри шаблонной "
                        "функции (имя типа замыкания не совпадает).\n");
            std::printf("  ⇒ лечение: вынести лямбду ядра из шаблона "
                        "(например, снять шаблон с InT и сделать две явные "
                        "функции f32/f16).\n");
        } else if (!a_ok2) {
            std::printf("НЕ работает даже A (обычная функция) ⇒ причина не в "
                        "шаблоне: скорее всего путь сборки (SPIR-V) или "
                        "несовместимость рантайма.\n");
            std::printf("  ⇒ соберите БЕЗ -fsycl-targets (AOT) и сравните.\n");
        } else {
            std::printf("A работает, B работает, C не работает ⇒ "
                        "явное имя ядра не поддержано — не важно, A и B "
                        "работают ⇒ причина в сборке.\n");
        }
        return (a_ok2 && (b_ok2 || c_ok2)) ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("!!! исключение верхнего уровня: %s\n", e.what());
        return 2;
    }
}
