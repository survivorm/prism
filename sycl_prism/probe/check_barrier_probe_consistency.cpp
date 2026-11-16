/* check_barrier_probe_consistency.cpp — GUARD на класс дефектов «SFINAE-проба
 * называет несуществующий член ⇒ молча отвечает false».
 *
 * ЧТО БЫЛО (16.11.2026, найдено БИСЕКЦИЕЙ):
 *   в src/prism_sycl_compat.h проба формы C-value обращалась к
 *   group_type_of<It>::fence_scope, а член называется fence_space.
 *   ⇒ проба честно отвечала «формы нет» (SFINAE), helper выбирал NONE,
 *   библиотека не собиралась: "static_assert(sizeof(It) == 0)".
 *   Рядом лежала ДОСЛОВНО та же проба, но с правильным именем, — и она
 *   отвечала true. Две семантически одинаковые пробы с разными ответами =
 *   дефект, который не видно ни в одном изолированном тесте.
 *
 * КАК ЛОВИМ: шим (probe/sycl_cpu) ПРЕДОСТАВЛЯЕТ ровно формы DPC++ 2026:
 *   group_barrier(Group G, memory_scope FenceScope = Group::fence_space)
 *   и group<D>::fence_space — constexpr-ЗНАЧЕНИЕ типа memory_scope.
 * Значит проба формы C-value ОБЯЗАНА находить эту форму. Если она отвечает
 * false — либо проба сломана (опечатка в имени), либо шим разошёлся с
 * реальностью. Оба случая — дефект артефакта.
 *
 * ПРАВИЛО (из опыта 16.11.2026): «шим не слабее реальности». Если шим
 * предоставляет форму, проверка обязана её находить — иначе шим проверяет не
 * тот код, и мы узнаём об этом только на карте.
 *
 * Компилируется и ЗАПУСКАЕТСЯ против исполняемого CPU-шима:
 *   g++ -std=c++20 -Iprobe/sycl_cpu probe/check_barrier_probe_consistency.cpp -o build/...
 */
#include <sycl/sycl.hpp>

#include "../src/prism_sycl_compat.h"

#include <cstdio>

int main() {
    using It = sycl::nd_item<1>;
    using G = sycl::group<1>;

    const bool d_mem     = prism_sycl::can_barrier_memory_scope<It>::value;
    const bool c_value   = prism_sycl::can_barrier_scope_value<It>::value;
    const bool c_type    = prism_sycl::can_barrier_group_fence<It>::value;
    const bool a_nofence = prism_sycl::can_barrier_group_only<G>::value;

    std::printf("D=%d Cvalue=%d Ctype=%d A=%d\n", (int)d_mem, (int)c_value,
                (int)c_type, (int)a_nofence);

    int rc = 0;

    /* 1. Шим предоставляет D (memory_scope::work_group) ⇒ проба обязана найти. */
    if (!d_mem) {
        std::printf("!!! проба формы D (memory_scope) не находит форму, которую "
                    "шим ПРЕДОСТАВЛЯЕТ ⇒ проба сломана (например, опечатка в "
                    "имени члена) либо шим разошёлся с DPC++ 2026\n");
        rc = 1;
    }

    /* 2. Шим предоставляет C-value (G::fence_space — constexpr memory_scope)
     *    ⇒ проба обязана найти. Именно эту пробу сломала опечатка
     *      fence_scope / fence_space. */
    if (!c_value) {
        std::printf("!!! проба формы C-value (G::fence_space) не находит форму, "
                    "которую шим ПРЕДОСТАВЛЯЕТ ⇒ опечатка в имени члена группы "
                    "(fence_space, НЕ fence_scope) либо расхождение шима с DPC++\n");
        rc = 1;
    }

    /* 3. C-type (fence_space как ИМЯ ТИПА) в DPC++ 2026 НЕ существует:
     *    fence_space — значение. ⇒ проба обязана давать false. */
    if (c_type) {
        std::printf("!!! проба формы C-type дала true, но в DPC++ 2026 "
                    "G::fence_space — constexpr-ЗНАЧЕНИЕ, а не имя типа ⇒ "
                    "проба на неверном API\n");
        rc = 1;
    }

    /* 4. Форма БЕЗ fence есть в DPC++ 2026 — но выбирать её нельзя.
     *    Если её нет — шим и реальность разошлись (наоборот опаснее). */
    if (!a_nofence) {
        std::printf("!!! форма group_barrier(g) без fence НЕ найдена, а в "
                    "DPC++ 2026 она есть ⇒ шим слабее реальности: проба "
                    "«без fence» не сработает как ловушка на карте\n");
        rc = 1;
    }

    /* 5. helper обязан выбрать форму С fence (не «NONE»). */
    const char* chosen = prism_sycl::barrier_form_name();
    std::printf("выбрано helper'ом: %s\n", chosen);
    if (std::string(chosen).rfind("NONE", 0) == 0) {
        std::printf("!!! helper не выбрал ни одной формы С fence ⇒ библиотека "
                    "не соберётся (static_assert в helper'е)\n");
        rc = 1;
    }

    if (rc == 0) {
        std::printf("OK: пробы барьера согласованы с шимом (D и C-value "
                    "найдены, C-type отсутствует, ловушка «без fence» есть)\n");
    }
    return rc;
}
