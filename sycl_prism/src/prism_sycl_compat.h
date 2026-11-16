// Совместимость вызова sycl::group_barrier между версиями SYCL.
//
// УСТАНОВЛЕНО (oneAPI DPC++ 2026.0, 16.11.2026, компиляцией):
//
// (1) СИГНАТУРА в 2026.0 (её показал сам компилятор, group_barrier.hpp:24):
//         group_barrier(Group G, memory_scope FenceScope = Group::fence_scope)
//     ⇒ второй параметр — `memory_scope` (стандарт SYCL 2020), а не
//       `access::fence_space`.
//
// (2) ФОРМА БЕЗ FENCE ЗАПРЕЩЕНА: sycl::group_barrier(group) без явной области
//     видимости не даёт гарантии порядка обращений к локальной памяти. Ядро,
//     читающее local_accessor после такого барьера, получало НЕСВЕЖИЕ значения
//     ⇒ нули в результате и запись мимо границ ⇒ "malloc(): unaligned tcache
//     chunk detected". Если подходящей формы с fence нет — сборка ПАДАЕТ на
//     этапе компиляции. Лучше не собраться, чем считать мусор.
//
// (3) ГЛАВНЫЙ ПРОКОЛ ЭТОГО ФАЙЛА (16.11.2026): `nd_item::get_group()` в DPC++
//     2026.0 — НЕ-CONST. Первые SFINAE-пробы брали группу так:
//         std::declval<const It&>().get_group()
//     ⇒ на карте это ill-formed ⇒ ВСЕ формы отверглись ⇒ helper выдал NONE ⇒
//     "static_assert(sizeof(It) == 0)" ⇒ библиотека не собиралась.
//     Ловушка: CPU-шим объявлял get_group() КОНСТ ⇒ дома всё было зелено.
//     ⇒ ПРАВИЛО: SFINAE-проба обязана использовать ТОТ ЖЕ тип выражения, что и
//       реальный вызов (barrier(It& it) → it.get_group() на НЕ-CONST lvalue).
//       Шим обязан моделировать API реальности, даже если это неудобно.
//
// (4) ОПЕЧАТКА В ИМЕНИ ЧЛЕНА — «ТИХИЙ» ДЕФЕКТ (16.11.2026, найдено БИСЕКЦИЕЙ).
//     Проба формы C-value обращалась к group_type_of<It>::fence_scope, а член
//     группы называется fence_space. Разница в одной букве:
//         fence_scope   (нет такого члена)  → SFINAE честно отвечает FALSE
//         fence_space   (есть, значение)    → SFINAE отвечает TRUE
//     ⇒ проба молча признавала форму недоступной; helper выбирал NONE;
//       сборка падала: "static_assert(sizeof(It) == 0)". Рядом лежала
//       ДОСЛОВНО та же проба с правильным именем — и она отвечала true.
//       Две семантически одинаковые пробы с РАЗНЫМИ ответами = дефект,
//       невидимый ни в одном изолированном тесте.
//     ⇒ ПРАВИЛО: если шим ПРЕДОСТАВЛЯЕТ форму, проба обязана её находить.
//       «Шим не слабее реальности» — иначе проверяем не тот код.
//       Guard: probe/check_barrier_probe_consistency.cpp (слой 4a гейта +
//       тест test_barrier_probes_agree_with_the_shim) — собрать и ЗАПУСТИТЬ
//       против исполняемого CPU-шима; он падает на этой самой опечатке.
//
// ТЕХНИКА (чтобы не повторить старый прокол):
//   SFINAE НЕ РАБОТАЕТ по несуществующим ИМЕНАМ: если написать
//       sycl::fence_space::local_space
//   и такого имени в SYCL нет, получается HARD ERROR (ошибка вне immediate
//   context), а не подстановочная ошибка ⇒ компиляция ломается целиком.
//   Поэтому проверки построены на ЗАВИСИМОМ имени (группа сама знает свой
//   перечислитель — G::fence_space, G — шаблонный параметр).
//   Исключение — форма D (sycl::memory_scope::work_group): имя берётся из
//   самого стандарта SYCL 2020 (memory_scope есть в любом SYCL 2020+), и её
//   явное использование НЕ зависит от того, что именно DPC++ назвал
//   G::fence_space (значение memory_scope? имя типа?).
#pragma once

#include <sycl/sycl.hpp>

#include <type_traits>
#include <utility>

namespace prism_sycl {

/* Тип группы, который возвращает it.get_group() (зависимое имя). */
template <class It>
using group_type_of = std::remove_cv_t<std::remove_reference_t<
    decltype(std::declval<It&>().get_group())>>;

/* РАЗЛИЧЕНИЕ API: в DPC++ 2026 `G::fence_space` — constexpr-ЗНАЧЕНИЕ типа
   memory_scope (сигнатура: group_barrier(Group G, memory_scope FenceScope =
   Group::fence_space)); в SYCL 2020 — ИМЯ ТИПА (enum class fence_space).
   ⇒ Различаем через `typename G::fence_space`: для значения это
   некорректно ⇒ SFINAE-отказ (false), для типа — корректно (true).
   Именно эта ошибка стоила нам круга: мы писали `G::fence_space::local_space`
   там, где нужно было передать ЗНАЧЕНИЕ `G::fence_space`. */
template <class G, class = void>
struct fence_scope_is_type : std::false_type {};
template <class G>
struct fence_scope_is_type<G, std::void_t<typename G::fence_space>>
    : std::true_type {};

/* Есть ли у группы член fence_space вообще (как значение). */
template <class G, class = void>
struct has_fence_scope_value : std::false_type {};
template <class G>
struct has_fence_scope_value<G, std::void_t<decltype(G::fence_space)>>
    : std::true_type {};

/* Форма D (СТАНДАРТ SYCL 2020): group_barrier(group, memory_scope::work_group).
   Проверяется ПЕРВОЙ: не зависит от того, что DPC++ назвал G::fence_space
   (в 2026.0 это constexpr-ЗНАЧЕНИЕ типа memory_scope, в SYCL 2020 — имя типа
   enum class fence_space). Второй параметр сигнатуры 2026.0 — memory_scope,
   поэтому передаём scope ЯВНО. */
template <class It, class = void>
struct can_barrier_memory_scope : std::false_type {};
template <class It>
struct can_barrier_memory_scope<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           sycl::memory_scope::work_group))>> : std::true_type {};

/* Форма C-value (DPC++ 2026): group_barrier(g, G::fence_space), где
   G::fence_space — constexpr-ЗНАЧЕНИЕ типа memory_scope. */
template <class It, class = void>
struct can_barrier_scope_value : std::false_type {};
template <class It>
struct can_barrier_scope_value<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           group_type_of<It>::fence_space))>> : std::true_type {};

/* Форма C-type (SYCL 2020): group_barrier(g, G::fence_space::local_space),
   где G::fence_space — ИМЯ ТИПА enum class fence_space. */
template <class It, class = void>
struct can_barrier_group_fence : std::false_type {};
template <class It>
struct can_barrier_group_fence<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           group_type_of<It>::fence_scope::local_space))>> : std::true_type {};

/* Диагностика: доступна ли форма БЕЗ fence (её использовать нельзя). */
template <class G, class = void>
struct can_barrier_group_only : std::false_type {};
template <class G>
struct can_barrier_group_only<G, std::void_t<decltype(sycl::group_barrier(
                                std::declval<const G&>()))>> : std::true_type {};

/* Переносимый барьер внутри ядра (it — sycl::nd_item<D>).
   Выбирается ТОЛЬКО форма С ЯВНОЙ областью видимости; форма без fence
   запрещена намеренно (см. (2) в шапке файла).
   ВАЖНО: проба использует НЕ-CONST lvalue (It&) — ровно как реальный вызов
   it.get_group(): в DPC++ 2026.0 get_group() не-const, и проба на const&
   отвергала ВСЕ формы (см. (3) в шапке файла). */
template <class It>
inline void barrier(It& it) {
    if constexpr (can_barrier_memory_scope<It>::value) {
        /* D: стандарт SYCL 2020 — scope передаётся ЯВНО. */
        sycl::group_barrier(it.get_group(), sycl::memory_scope::work_group);
    } else if constexpr (can_barrier_scope_value<It>::value) {
        /* DPC++ 2026: G::fence_space — constexpr-ЗНАЧЕНИЕ типа memory_scope. */
        sycl::group_barrier(it.get_group(), group_type_of<It>::fence_space);
    } else if constexpr (can_barrier_group_fence<It>::value) {
        /* SYCL 2020: G::fence_space — имя типа enum class fence_space. */
        sycl::group_barrier(it.get_group(),
                            group_type_of<It>::fence_space::local_space);
    } else {
        static_assert(sizeof(It) == 0,
                      "prism_sycl: нет sycl::group_barrier С явной областью "
                      "видимости — барьер без fence не упорядочивает локальную "
                      "память (даёт нули и порчу кучи). Форма без fence "
                      "ЗАПРЕЩЕНА намеренно. Соберите "
                      "probe/probe_barrier_forms.cpp — он печатает, какие формы "
                      "есть на вашем компиляторе и что выбрал helper.");
    }
}

/* Какая форма барьера выбрана — для самопроверки артефакта.
   Печатается в device_info(), чтобы по логу было видно, ЧТО именно собрано. */
inline const char* barrier_form_name() {
    if constexpr (can_barrier_memory_scope<sycl::nd_item<1>>::value) {
        return "memory_scope (D: group_barrier(g, memory_scope::work_group))";
    } else if constexpr (can_barrier_scope_value<sycl::nd_item<1>>::value) {
        return "scope-value (G::fence_space = memory_scope)";
    } else if constexpr (can_barrier_group_fence<sycl::nd_item<1>>::value) {
        return "group-fence (G::fence_space::local_space)";
    } else {
        return "NONE (сборка невозможна: барьер без fence запрещён)";
    }
}

}  // namespace prism_sycl
