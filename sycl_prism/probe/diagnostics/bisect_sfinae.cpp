/* bisect.cpp — БИСЕКЦИЯ расхождения can_barrier_scope_value.
 *
 * Факт: в header проба даёт 0, идентичная реплика — 1.
 * Гипотеза: дело в ПОРЯДКЕ объявления (частичная специализация объявлена
 * ДО первого использования, но ПОСЛЕ того, как шаблон уже был «закрыт»
 * другой частичной специализацией), либо в области видимости имени.
 *
 * V1 — всё в одном namespace, реплика добавлена В КОНЕЦ (как было раньше).
 * V2 — реплика объявлена ДО специализации can_barrier_scope_value.
 * V3 — реплика с квалификатором namespace.
 * V4 — реплика, где group_type_of заменён на decltype(...).
 */
#include <sycl/sycl.hpp>

#include <cstdio>
#include <type_traits>
#include <utility>

namespace v1 {

template <class It>
using group_type_of = std::remove_cv_t<std::remove_reference_t<
    decltype(std::declval<It&>().get_group())>>;

/* Форма D */
template <class It, class = void>
struct can_D : std::false_type {};
template <class It>
struct can_D<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           sycl::memory_scope::work_group))>> : std::true_type {};

/* Форма C-value (ИДЕНТИЧНА header) */
template <class It, class = void>
struct can_C : std::false_type {};
template <class It>
struct can_C<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           group_type_of<It>::fence_space))>> : std::true_type {};

/* РЕПЛИКА — в конец namespace */
template <class It, class = void>
struct replica : std::false_type {};
template <class It>
struct replica<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           group_type_of<It>::fence_space))>> : std::true_type {};

}  // namespace v1

namespace v2 {

template <class It>
using group_type_of = std::remove_cv_t<std::remove_reference_t<
    decltype(std::declval<It&>().get_group())>>;

/* РЕПЛИКА — В НАЧАЛЕ */
template <class It, class = void>
struct replica : std::false_type {};
template <class It>
struct replica<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           group_type_of<It>::fence_space))>> : std::true_type {};

/* Форма D — ПОСЛЕ реплики */
template <class It, class = void>
struct can_D : std::false_type {};
template <class It>
struct can_D<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           sycl::memory_scope::work_group))>> : std::true_type {};

/* Форма C-value — ИДЕНТИЧНА header */
template <class It, class = void>
struct can_C : std::false_type {};
template <class It>
struct can_C<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           group_type_of<It>::fence_space))>> : std::true_type {};

}  // namespace v2

namespace v3 {

template <class It>
using group_type_of = std::remove_cv_t<std::remove_reference_t<
    decltype(std::declval<It&>().get_group())>>;

/* РЕПЛИКА с квалификатором namespace (как probe1 в probe_sfinae_replica.cpp) */
template <class I, class = void>
struct replica_q : std::false_type {};
template <class I>
struct replica_q<I, std::void_t<decltype(sycl::group_barrier(
        std::declval<I&>().get_group(),
        v3::group_type_of<I>::fence_space))>> : std::true_type {};

}  // namespace v3

namespace v4 {

template <class It>
using group_type_of = std::remove_cv_t<std::remove_reference_t<
    decltype(std::declval<It&>().get_group())>>;

/* РЕПЛИКА с decltype вместо алиаса */
template <class It, class = void>
struct replica_dt : std::false_type {};
template <class It>
struct replica_dt<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           decltype(std::declval<It&>().get_group())::fence_space))>> : std::true_type {};

}  // namespace v4

int main() {
    using It = sycl::nd_item<1>;
    std::printf("v1: can_D=%d can_C=%d replica=%d\n",
                (int)v1::can_D<It>::value, (int)v1::can_C<It>::value,
                (int)v1::replica<It>::value);
    std::printf("v2: can_D=%d can_C=%d replica=%d\n",
                (int)v2::can_D<It>::value, (int)v2::can_C<It>::value,
                (int)v2::replica<It>::value);
    std::printf("v3: replica_q=%d\n", (int)v3::replica_q<It>::value);
    std::printf("v4: replica_dt=%d\n", (int)v4::replica_dt<It>::value);
    return 0;
}
