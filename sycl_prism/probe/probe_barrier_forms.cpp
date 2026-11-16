/* probe_barrier_forms.cpp — какая форма sycl::group_barrier доступна на ВАШЕМ
 * компиляторе, и какую выбирает наш переносимый helper.
 *
 * Компилируется БЕЗ запуска на GPU: проверка форм — чисто компиляционная
 * (SFINAE). Проверки построены на ЗАВИСИМЫХ именах (G::fence_space), потому
 * что SFINAE не работает по несуществующим именам (sycl::fence_space даёт
 * hard error).
 *
 * ИСТОРИЯ ПРАВКИ (16.11.2026): прежняя версия этого файла проверяла ТОЛЬКО
 * форму C (SYCL 2020: G::fence_space::local_space) и объявляла «формы с fence
 * нет ⇒ библиотека не соберётся». На DPC++ 2026 это было НЕВЕРНО: там есть
 * group_barrier(Group G, memory_scope FenceScope = Group::fence_space), и
 * достаточно передать scope ЯВНО (форма D). Зонд врал ⇒ пользователь был
 * уверен, что сборка невозможна, хотя правильный API рядом.
 * ⇒ Теперь проверяются ВСЕ формы, и «невозможно» выводится только если
 * действительно не найдено ни одной формы С fence.
 *
 *   icpx -fsycl -std=c++20 probe/probe_barrier_forms.cpp -o build/probe_barrier_forms
 *   ./build/probe_barrier_forms
 */
#include <sycl/sycl.hpp>

#include "../src/prism_sycl_compat.h"

#include <cstdio>
#include <string>
#include <typeinfo>

#if defined(__GNUC__)
#include <cxxabi.h>
#include <cstdlib>
static std::string demangle(const char* n) {
    int st = 0;
    char* d = abi::__cxa_demangle(n, nullptr, nullptr, &st);
    std::string out = (st == 0 && d) ? d : n;
    if (d) std::free(d);
    return out;
}
#else
static std::string demangle(const char* n) { return std::string(n); }
#endif

int main() {
    using G = sycl::group<1>;
    using It = sycl::nd_item<1>;

    std::printf("=== формы sycl::group_barrier на этом компиляторе ===\n");
    std::printf("тип get_group() у nd_item<1>: %s\n",
                demangle(typeid(G).name()).c_str());

    const bool d_mem   = prism_sycl::can_barrier_memory_scope<It>::value;
    const bool c_value = prism_sycl::can_barrier_scope_value<It>::value;
    const bool c_type  = prism_sycl::can_barrier_group_fence<It>::value;
    const bool a_nofence = prism_sycl::can_barrier_group_only<G>::value;

    std::printf("D ) group_barrier(g, memory_scope::work_group): %s  "
                "<= форма SYCL 2020, scope передаётся ЯВНО\n",
                d_mem ? "есть" : "нет");
    std::printf("Cv) group_barrier(g, G::fence_space): %s  "
                "<= DPC++ 2026: G::fence_space — constexpr-ЗНАЧЕНИЕ типа memory_scope\n",
                c_value ? "есть" : "нет");
    std::printf("C ) group_barrier(g, G::fence_space::local_space): %s  "
                "<= SYCL 2020: G::fence_space — ИМЯ ТИПА\n",
                c_type ? "есть" : "нет");
    std::printf("A ) group_barrier(g) (БЕЗ fence): %s  "
                "(если есть — ИМЕННО ЕЁ выбирал старый helper ⇒ нули и порчу кучи)\n",
                a_nofence ? "есть" : "нет");
    std::printf("выбрано helper'ом: %s\n", prism_sycl::barrier_form_name());

    const bool any_fenced = d_mem || c_value || c_type;
    if (!any_fenced) {
        std::printf("\nВНИМАНИЕ: НИ ОДНОЙ формы С fence компилятор не нашёл ⇒ "
                    "библиотека прибора НЕ СОБЕРЁТСЯ\n(намеренно: такая сборка "
                    "даёт нули и порчу куши). Чтобы узнать точный API — "
                    "скомпилируйте probe/probe_barrier_candidates.cpp и пришлите "
                    "вывод компилятора (он печатает список перегрузок "
                    "group_barrier).\n");
        return 2;
    }
    std::printf("\nOK: выбрана форма С fence ⇒ барьер упорядочивает локальную "
                "память.\n");
    return 0;
}
