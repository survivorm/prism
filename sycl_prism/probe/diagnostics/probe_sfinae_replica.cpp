/* probe_sfinae_replica.cpp — локализация расхождения в SFINAE-пробах барьера.
 *
 * Наблюдение: can_barrier_scope_value<It>::value == 0, но дословно
 * идентичная проба, объявленная ниже в том же namespace, даёт 1.
 * ⇒ две семантически одинаковые пробы с разными ответами = дефект.
 *
 * Печатает результат КАЖДОГО написания той же идеи, чтобы увидеть, чем
 * именно отличается работающее от неработающего.
 */
#include <sycl/sycl.hpp>

#include "../src/prism_sycl_compat.h"

#include <cstdio>
#include <type_traits>

using It = sycl::nd_item<1>;
using G = sycl::group<1>;

/* (1) ровно как в header: алиас group_type_of + ::fence_space */
template <class I, class = void>
struct probe1 : std::false_type {};
template <class I>
struct probe1<I, std::void_t<decltype(sycl::group_barrier(
        std::declval<I&>().get_group(),
        prism_sycl::group_type_of<I>::fence_space))>> : std::true_type {};

/* (2) тот же доступ, но без алиаса group_type_of */
template <class I, class = void>
struct probe2 : std::false_type {};
template <class I>
struct probe2<I, std::void_t<decltype(sycl::group_barrier(
        std::declval<I&>().get_group(),
        decltype(std::declval<I&>().get_group())::fence_space))>> : std::true_type {};

/* (3) конкретный (не зависимый) тип группы */
template <class I, class = void>
struct probe3 : std::false_type {};
template <class I>
struct probe3<I, std::void_t<decltype(sycl::group_barrier(
        std::declval<I&>().get_group(),
        G::fence_space))>> : std::true_type {};

/* (4) fence_space как ТИП (SYCL 2020) — ровно как в header, БЕЗ typename */
template <class I, class = void>
struct probe4 : std::false_type {};
template <class I>
struct probe4<I, std::void_t<decltype(sycl::group_barrier(
        std::declval<I&>().get_group(),
        prism_sycl::group_type_of<I>::fence_space::local_space))>> : std::true_type {};

/* (5) требует ли тип FENCE-аргумента ровно memory_scope? */
template <class I, class = void>
struct probe5 : std::false_type {};
template <class I>
struct probe5<I, std::void_t<decltype(sycl::group_barrier(
        std::declval<I&>().get_group(),
        std::declval<sycl::memory_scope>()))>> : std::true_type {};

/* (6) requires */
template <class I>
concept can_scope_value = requires(I& it) {
    sycl::group_barrier(it.get_group(), prism_sycl::group_type_of<I>::fence_space);
};

template <class I>
concept can_mem_scope = requires(I& it) {
    sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
};

int main() {
    std::printf("header can_barrier_memory_scope = %d\n",
                (int)prism_sycl::can_barrier_memory_scope<It>::value);
    std::printf("header can_barrier_scope_value  = %d\n",
                (int)prism_sycl::can_barrier_scope_value<It>::value);
    std::printf("header can_barrier_group_fence  = %d\n",
                (int)prism_sycl::can_barrier_group_fence<It>::value);
    std::printf("-- реплики --\n");
    std::printf("(1) group_type_of<It>::fence_space      = %d\n", (int)probe1<It>::value);
    std::printf("(2) decltype(get_group())::fence_space  = %d\n", (int)probe2<It>::value);
    std::printf("(3) G::fence_space (конкретный тип)    = %d\n", (int)probe3<It>::value);
    std::printf("(4) fence_space как ТИП (typename)      = %d\n", (int)probe4<It>::value);
    std::printf("(5) memory_scope-аргумент               = %d\n", (int)probe5<It>::value);
    std::printf("(6) requires can_scope_value            = %d\n", (int)can_scope_value<It>);
    std::printf("(6) requires can_mem_scope             = %d\n", (int)can_mem_scope<It>);
    std::printf("тип fence_space: %s\n",
                typeid(decltype(prism_sycl::group_type_of<It>::fence_space)).name());
    std::printf("header barrier_form_name() = %s\n", prism_sycl::barrier_form_name());
    return 0;
}
