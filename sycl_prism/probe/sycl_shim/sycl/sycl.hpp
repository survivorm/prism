// Минимальный ШИМ SYCL API — ТОЛЬКО для проверки компилируемости C++ без icpx.
//
// ВАЖНО: шим повторяет API-поверхность **oneAPI 2026.0 (SYCL 2020)**, а не
// «что-то примерно похожее». Это сделано после реальной ошибки компилятора на
// машине пользователя: шим допускал вещи, которых в 2026.0 нет, и тем самым
// прятал дефекты. Теперь шим ОТКАЗЫВАЕТ на том, на чём отказывает icpx:
//
//   * sycl::group — ШАБЛОННЫЙ класс (sycl::group<1>::fence_space::local_space);
//   * sycl::local_accessor — конструктор требует НЕКОНСТАНТНЫЙ handler&
//     (иначе нельзя создать аксессор внутри [=]-лямбды ядра — это ошибка
//     семантики, а не синтаксиса);
//   * конструктора sycl::accessor<T,1>(указатель, handler) НЕТ (в 2026.0
//     аксессор строится только из buffer) ⇒ в ядре используются USM-указатели;
//   * sycl::aspect не содержит bf16 и sub_group;
//   * нет ни sycl::info::backend, ни sycl::info::device::backend;
//   * перечисление sycl::backend не содержит level_zero/cuda/hip.
//
// ЧЕГО ШИМ НЕ ДЕЛАЕТ: не вычисляет, не проверяет семантику SYCL, не заменяет icpx.
// Это фильтр грубых ошибок, а не верификация. Проверка «по-настоящему» — icpx.
//
// Использование:
//     g++ -std=c++20 -fsyntax-only -Iprobe/sycl_shim probe/probe_sycl_min.cpp
//     g++ -std=c++20 -fsyntax-only -Iprobe/sycl_shim src/prism_sycl.cpp
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace sycl {

class queue;  // вперёд: USM-функции принимают queue&

namespace access { enum class mode { read, write, read_write, discard_write }; }

namespace info {
namespace device {
struct name { using value_type = std::string; };
struct driver_version { using value_type = std::string; };
struct max_work_group_size { using value_type = std::size_t; };
struct global_mem_size { using value_type = std::size_t; };
struct local_mem_size { using value_type = std::size_t; };
}  // namespace device
}  // namespace info

enum class aspect { fp64, fp32, fp16 };  // bf16/sub_group в 2026.0 нет

enum class backend { opencl };  // level_zero/cuda/hip в 2026.0 нет

enum class device_selector { host, cpu, gpu, accelerator, custom };
inline constexpr device_selector gpu_selector_v = device_selector::gpu;
inline constexpr device_selector cpu_selector_v = device_selector::cpu;
inline constexpr device_selector host_selector_v = device_selector::host;

namespace property {
namespace queue { struct default_selector {}; struct in_order {}; }
namespace device { struct host_device {}; }
}  // namespace property

class handler;

/* МОДЕЛЬ DPC++ 2026.0 (установлено компиляцией 16.11.2026, group_barrier.hpp:24):
 *     group_barrier(Group G, memory_scope FenceScope = Group::fence_scope)
 * ⇒ второй параметр — `memory_scope` (стандарт SYCL 2020), а `Group::fence_space`
 *   — constexpr-ЗНАЧЕНИЕ этого типа (НЕ имя типа!).
 * namespace access::fence_space в 2026.0 НЕТ ⇒ не используем.
 */
enum class memory_scope { work_item, sub_group, work_group, device, system };

template <int Dimensions = 1> class group {
 public:
  /* DPC++ 2026: fence_scope — constexpr-ЗНАЧЕНИЕ типа memory_scope. */
  static constexpr memory_scope fence_space = memory_scope::work_group;
};

template <int D> class id {
 public:
  std::size_t operator[](int) const { return 0; }
};

template <int D> class range {
 public:
  range(std::size_t, std::size_t = 1) {}
  std::size_t size() const { return 0; }
};

template <int D> class nd_range {
 public:
  nd_range(range<D>, range<D>) {}
};

template <int D> class nd_item {
 public:
  std::size_t get_global_id(int) const { return 0; }
  std::size_t get_local_id(int) const { return 0; }
  /* DPC++ 2026.0: get_group() — НЕ-CONST (установлено компиляцией 16.11.2026). */
  group<D> get_group() { return group<D>(); }  /* в SYCL 2020 — без аргумента */
};

class handler {
 public:
  template <int D, class F> void parallel_for(range<D>, F) {}
  template <int D, class F> void parallel_for(nd_range<D>, F) {}
};

/* В SYCL 2020 перегрузки group_barrier принимают ОБЪЕКТ ГРУППЫ, а не только
   fence_space (в 1.2.1 было наоборот). Шим повторяет 2026.0. */
/* В 2026.0 перегрузка с одним fence_space НЕТ: нужен объект группы. */
template <int D>
void group_barrier(group<D>, memory_scope) {}

/* Однопараметрическая перегрузка group_barrier(group) В DPC++ 2026 ЕСТЬ.
   Выбирать её нельзя (нет явной области видимости ⇒ нет гарантии порядка
   обращений к памяти), но она должна ПРИСУТСТВОВАТЬ в шиме: иначе набор
   перегрузок в шиме ≠ реальном ⇒ SFINAE-выбор в prism_sycl_compat.h разойдётся
   между двумя шимами (ровно тот класс прокола, что мы уже ловили). */
template <int D>
void group_barrier(group<D>) {}

template <class T, int D> class accessor;

template <class T> class accessor<T, 1> {
 public:
  accessor() = default;
  T& operator[](std::size_t) const { static T t{}; return t; }
  std::size_t size() const { return 0; }
};

template <class T> class accessor<T, 2> {
 public:
  struct row {
    T& operator[](std::size_t) const { static T t{}; return t; }
  };
  accessor() = default;
  row operator[](std::size_t) const { return row{}; }
  row operator[](id<1>) const { return row{}; }
  T& operator[](id<2>) const { static T t{}; return t; }
};

// local_accessor: конструктор требует НЕКОНСТАНТНЫЙ handler& (как в 2026.0) —
// иначе шим не поймал бы ошибку «аксессор создан внутри лямбды ядра».
template <class T, int D> class local_accessor;

template <class T> class local_accessor<T, 1> {
 public:
  local_accessor(range<1>, handler&) {}
  T& operator[](std::size_t) const { static T t{}; return t; }
};

template <class T> class local_accessor<T, 2> {
 public:
  struct row {
    T& operator[](std::size_t) const { static T t{}; return t; }
  };
  local_accessor(range<2>, handler&) {}
  row operator[](std::size_t) const { return row{}; }
  row operator[](id<1>) const { return row{}; }
};

template <class T, int D> class buffer {
 public:
  buffer(T*, range<D>) {}
  template <access::mode M> accessor<T, D> get_access(handler&) {
    return accessor<T, D>();
  }
};

class event {
 public:
  void wait() {}
  void wait_and_throw() {}
};

// USM (в SYCL 2020 эти функции есть — ошибок icpx по ним не было).
template <class T = void> T* malloc_device(std::size_t, queue&) { return nullptr; }
template <class T = void> void free(T*, queue&) {}

// Математика SYCL 2020.
template <class T> T sqrt(T x) { return x; }
template <class T> T fmax(T a, T b) { return a > b ? a : b; }
template <class T> T abs(T x) { return x < 0 ? -x : x; }
struct half { float v; };

class device {
 public:
  static std::vector<device> get_devices() { return {device()}; }
  template <class T> typename T::value_type get_info() const {
    return typename T::value_type();
  }
  bool has(aspect) const { return true; }
  bool is_gpu() const { return true; }
  bool is_cpu() const { return false; }
};

class queue {
 public:
  queue() {}
  queue(device) {}
  queue(device_selector) {}
  queue(device_selector, property::queue::default_selector) {}
  queue(device_selector, property::queue::in_order) {}
  queue(property::device::host_device) {}
  device get_device() const { return device(); }
  /* DPC++ 2020: queue::memcpy возвращает event& (не void!) — иначе код,
     написанный для реального SYCL, не компилируется против шима
     (очередной класс «шим ≠ реальность»: проба probe_local_mem.cpp
     использует q.memcpy(...).wait() — на карте это законно, против
     старого шима — нет). */
  event& memcpy(void*, const void*, std::size_t) { return ev_; }
  void wait() {}
  template <class F> event& submit(F) { return ev_; }

 private:
  event ev_;
};

}  // namespace sycl
