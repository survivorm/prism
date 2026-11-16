/* probe_shim_api.cpp — какие формы sycl::group_barrier видит компилятор
 * и что выбирает наш переносимый helper.
 *
 * Компилируется и ЗАПУСКАЕТСЯ против каждого из двух шимов
 * (probe/sycl_shim — только компиляция, probe/sycl_cpu — исполнение).
 * Если ответы двух шимов РАЗНЫЕ — шимы разошлись ⇒ SFINAE-выбор в
 * prism_sycl_compat.h на реальной карте может отличаться от проверяемого.
 * Это ровно тот класс прокола, который 16.11.2026 дал «NONE ⇒ не собирается».
 */
#include <sycl/sycl.hpp>

#include "../src/prism_sycl_compat.h"

#include <cstdio>

int main() {
    using It = sycl::nd_item<1>;
    using G = sycl::group<1>;
    std::printf("D=%d Cvalue=%d Ctype=%d A=%d chosen=%s\n",
                (int)prism_sycl::can_barrier_memory_scope<It>::value,
                (int)prism_sycl::can_barrier_scope_value<It>::value,
                (int)prism_sycl::can_barrier_group_fence<It>::value,
                (int)prism_sycl::can_barrier_group_only<G>::value,
                prism_sycl::barrier_form_name());
    return 0;
}
