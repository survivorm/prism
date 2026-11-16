// Мини-реализация SYCL для CPU: ВЫПОЛНЯЕТСЯ, а не только имитирует формы.
//
// Зачем: в песочнице нет icpx и нет GPU, но код прибора (ядра vpath/dcurve/acf/
// perm) должен быть проверен ДО машины пользователя. Форм-шим (probe/sycl_shim)
// ловит ошибки компиляции, но не проверяет АРИФМЕТИКУ.
//
// Этот шим исполняет parallel_for на CPU и повторяет API-поверхность oneAPI
// 2026.0 (SYCL 2020):
//   * range<D>  — последовательный перебор индексов;
//   * nd_range<D> — эмуляция work-group'ов на потоках (std::barrier ⇒ честная
//     семантика group_barrier);
//   * local_accessor — РЕАЛЬНАЯ разделяемая память группы. Аксессор создаётся
//     один раз в области command-group (как требует 2026.0: конструктор берёт
//     неконстантный handler&), а слот задаётся обработчиком ⇒ у всех членов
//     группы слот одинаков, у разных групп — разная память;
//   * USM malloc_device/free — обычный malloc/free;
//   * sycl::group — шаблонный класс (sycl::group<1>::fence_space).
//
// ЧЕГО ЭТО НЕ ДАЁТ: это не проверка планировщика GPU, не проверка раскладки
// памяти устройства и не измерение скорости. Это проверка ЛОГИКИ ядер.
//
// Сборка:  g++ -std=c++20 -O2 -fPIC -shared -Iprobe/sycl_cpu src/prism_sycl.cpp \
//              -o build/libprism_sycl_cpu.so
#pragma once
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace sycl {

namespace access { enum class mode { read, write, read_write, discard_write }; }

enum class aspect { fp64, fp32, fp16 };
enum class backend { opencl };
enum class device_selector { host, cpu, gpu, accelerator, custom };

inline constexpr device_selector gpu_selector_v = device_selector::gpu;
inline constexpr device_selector cpu_selector_v = device_selector::cpu;
inline constexpr device_selector host_selector_v = device_selector::host;

namespace property {
namespace queue { struct default_selector {}; struct in_order {}; }
namespace device { struct host_device {}; }
}  // namespace property

namespace info {
namespace device {
struct name { using value_type = std::string; };
struct driver_version { using value_type = std::string; };
struct max_work_group_size { using value_type = std::size_t; };
struct global_mem_size { using value_type = std::size_t; };
struct local_mem_size { using value_type = std::size_t; };
}  // namespace device
}  // namespace info

namespace access {
/* Старый (SYCL 2020) перечислитель — оставлен для совместимости проверок. */
enum class fence_space { local_space, global_space };
}  // namespace access

/* API DPC++ 2026 (установлено компилятором 16.11.2026):
 *     group_barrier(Group G, memory_scope FenceScope = Group::fence_space)
 * ⇒ в 2026 второй параметр — это `memory_scope`, а `Group::fence_space` —
 * constexpr-ЗНАЧЕНИЕ этого типа (НЕ имя типа!). Шим повторяет именно этот API:
 * иначе SFINAE-выбор в helper'е расходился бы с картой (ровно тот класс
 * проколов, который мы уже ловили).
 */
enum class memory_scope { work_item, sub_group, work_group, device, system };

template <int Dimensions = 1> class group {
 public:
  /* DPC++ 2026: fence_space — constexpr-ЗНАЧЕНИЕ типа memory_scope. */
  static constexpr memory_scope fence_space = memory_scope::work_group;
};

class handler;  // вперёд: accessor/local_accessor принимают handler&

/* ---------------- id / range / nd_range / nd_item ---------------- */

template <int D> class id {
 public:
  std::size_t v_[D];
  id() { for (int i = 0; i < D; ++i) v_[i] = 0; }
  explicit id(const std::size_t* p) { for (int i = 0; i < D; ++i) v_[i] = p[i]; }
  std::size_t operator[](int i) const { return v_[i]; }
};

template <int D> class range {
 public:
  std::size_t s_[D];
  range() { for (int i = 0; i < D; ++i) s_[i] = 1; }
  range(std::size_t a) { for (int i = 0; i < D; ++i) s_[i] = (i == 0) ? a : 1; }
  range(std::size_t a, std::size_t b) {
    for (int i = 0; i < D; ++i) s_[i] = (i == 0) ? a : (i == 1 ? b : 1);
  }
  std::size_t size() const {
    std::size_t n = 1;
    for (int i = 0; i < D; ++i) n *= s_[i];
    return n;
  }
};

template <int D> class nd_range {
 public:
  range<D> g_, l_;
  nd_range(range<D> g, range<D> l) : g_(g), l_(l) {}
};

template <int D> class nd_item {
 public:
  id<D> g_, l_;
  nd_item() = default;
  nd_item(const id<D>& g, const id<D>& l) : g_(g), l_(l) {}
  std::size_t get_global_id(int i) const { return g_[i]; }
  std::size_t get_local_id(int i) const { return l_[i]; }
  /* DPC++ 2026.0: get_group() — НЕ-CONST (установлено компиляцией 16.11.2026:
     SFINAE-пробы на const lvalue отвергали ВСЕ формы барьера). Шим обязан
     моделировать реальность, даже если это неудобно: иначе класс ошибок
     «проба ≠ реальный вызов» невидим дома. */
  group<D> get_group() { return group<D>(); }  /* в SYCL 2020 — без аргумента */
};

/* ---------------- контекст work-group'а ---------------- */

namespace detail {

struct GroupCtx {
  std::unique_ptr<std::barrier<>> bar;  // std::barrier не копируется ⇒ через ptr
  std::mutex mtx;
  std::map<int, std::vector<char>> blocks;  // слот ⇒ память группы
};

inline thread_local GroupCtx* tls_ctx = nullptr;
inline thread_local GroupCtx* tcache_ctx = nullptr;
inline thread_local int tcache_slot = -1;
inline thread_local void* tcache_ptr = nullptr;

/* Память слота в ТЕКУЩЕЙ work-group. Аксессор создаётся один раз на
   command-group, но память у каждой группы своя (как в SYCL 2020). */
inline void* resolve_slot(int slot, std::size_t bytes) {
  GroupCtx* ctx = tls_ctx;
  if (ctx == nullptr) {  // вырожденный случай: вне work-group
    thread_local std::vector<char> fb;
    fb.assign(bytes ? bytes : 1, 0);
    return fb.data();
  }
  if (tcache_ctx == ctx && tcache_slot == slot) return tcache_ptr;
  std::lock_guard<std::mutex> lk(ctx->mtx);
  auto it = ctx->blocks.find(slot);
  if (it == ctx->blocks.end())
    it = ctx->blocks.emplace(slot, std::vector<char>(bytes ? bytes : 1, 0)).first;
  void* p = it->second.data();
  tcache_ctx = ctx;
  tcache_slot = slot;
  tcache_ptr = p;
  return p;
}

}  // namespace detail

/* Барьер work-group: в SYCL 2020 вызов принимает объект группы. */
/* ОДНОПАРАМЕТРИЧЕСКАЯ перегрузка group_barrier(group) — в настоящем DPC++ 2020
 * она ЕСТЬ, и она НЕ даёт гарантий порядка обращений к памяти.
 *
 * Зачем она в шиме: набор перегрузок в шиме должен СОВПАДАТЬ с реальным, иначе
 * SFINAE-выбор в prism_sycl_compat.h различается, и шим проверяет не тот код.
 * Именно так и вышло (16.11.2026): в шиме была только форма с fence ⇒ шим всегда
 * выбирал её ⇒ барьер с гарантией памяти; а на карте DPC++ предлагал ещё и
 * форму без fence, и наш хелпер выбирал её первую ⇒ локальная память читалась
 * без синхронизации ⇒ нули и порча кучи.
 *
 * Здесь перегрузка реализована честно (тот же барьер) — но выбирать её нельзя:
 * см. порядок вариантов в prism_sycl_compat.h.
 */
template <int D> void group_barrier(group<D>) {
    /* ЛОВУШКА: если эта перегрузка будет ВЫБРАНА (т.е. helper в
       prism_sycl_compat.h отдаст предпочтение форме без fence) — сборка шума
       упадёт с внятным текстом. Реальный DPC++ 2026 эту форму ПРЕДОСТАВЛЯЕТ,
       но выбирать её нельзя: она не гарантирует порядок обращений к памяти.
       Форма с fence (group, access::fence_space) — единственно допустимая. */
    /* ЛОВУШКА (runtime): если эта перегрузка реально ВЫЗВАНА — падаем с
       внятным сообщением. Важно: ловушка именно runtime, а не static_assert
       внутри тела: static_assert сработал бы при SFINAE-проверке (ошибка вне
       immediate context ⇒ hard error) и сломал бы сам механизм выбора.
       Шим исполняет ядра по-настоящему ⇒ вызов поймается на этапе паритета. */
    std::fprintf(stderr,
                 "prism_sycl: ЛОВУШКА — вызван group_barrier(group) БЕЗ явной "
                 "области видимости!\n"
                 "  В DPC++ 2026 значение по умолчанию упорядочивает память, но "
                 "полагаться\n  на него нельзя: helper обязан передавать область "
                 "явно (G::fence_space).\n");
    std::fflush(stderr);
    std::abort();
}

/* DPC++ 2026: group_barrier(Group G, memory_scope FenceScope = Group::fence_space) */
template <class Group>
void group_barrier(Group, memory_scope = Group::fence_space) {
  detail::GroupCtx* ctx = detail::tls_ctx;
  if (ctx && ctx->bar) ctx->bar->arrive_and_wait();
}

/* ---------------- accessor / local_accessor ---------------- */

template <class T, int D> class accessor;

template <class T> class accessor<T, 1> {
 public:
  T* p_ = nullptr;
  accessor() = default;
  accessor(T* p, range<1>) : p_(p) {}
  accessor(T* p, range<1>, const handler&) : p_(p) {}
  T& operator[](std::size_t i) const { return p_[i]; }
  std::size_t size() const { return 0; }
};

template <class T> class accessor<T, 2> {
 public:
  struct row {
    T* p_;
    std::size_t n_;
    T& operator[](std::size_t j) const { return p_[j]; }
  };
  T* p_ = nullptr;
  std::size_t n_ = 0;
  accessor() = default;
  accessor(T* p, range<2> r) : p_(p), n_(r.s_[0] * r.s_[1]) {}
  accessor(T* p, range<2> r, const handler&) : p_(p), n_(r.s_[0] * r.s_[1]) {}
  row operator[](std::size_t i) const { return row{p_ + i * n_, n_}; }
  row operator[](id<1>) const { return row{p_, n_}; }
  T& operator[](id<2> i) const { return p_[i[0] * n_ + i[1]]; }
};

template <class T, int D> class local_accessor;

template <class T> class local_accessor<T, 1> {
 public:
  int slot_ = -1;
  std::size_t bytes_ = 0;
  local_accessor(range<1> r, handler& h) {
    slot_ = h.next_local_slot();
    bytes_ = r.size() * sizeof(T);
  }
  T& operator[](std::size_t i) const {
    return ((T*)detail::resolve_slot(slot_, bytes_))[i];
  }
};

template <class T> class local_accessor<T, 2> {
 public:
  struct row {
    T* p_;
    std::size_t n_;
    T& operator[](std::size_t j) const { return p_[j]; }
  };
  int slot_ = -1;
  std::size_t n_ = 0;
  std::size_t bytes_ = 0;
  local_accessor(range<2> r, handler& h) {
    slot_ = h.next_local_slot();
    n_ = r.s_[0] * r.s_[1];
    bytes_ = n_ * sizeof(T);
  }
  row operator[](std::size_t i) const {
    return row{(T*)detail::resolve_slot(slot_, bytes_) + i * n_, n_};
  }
  row operator[](id<1>) const {
    return row{(T*)detail::resolve_slot(slot_, bytes_), n_};
  }
};

/* ---------------- buffer ---------------- */

template <class T, int D> class buffer {
 public:
  T* p_ = nullptr;
  range<D> r_;
  std::size_t n_ = 0;
  buffer() = default;
  buffer(T* p, range<D> r) : p_(p), r_(r), n_(r.size()) {}
  template <access::mode M> accessor<T, D> get_access(handler&) {
    return accessor<T, D>(p_, r_);
  }
};

/* ---------------- handler ---------------- */

class handler {
 public:
  std::shared_ptr<std::atomic<int>> slot_counter =
      std::make_shared<std::atomic<int>>(0);
  int next_local_slot() { return (*slot_counter)++; }

  template <int D, class F> void parallel_for(range<D> r, F f) {
    const std::size_t total = r.size();
    std::size_t idx[D];
    for (int d = 0; d < D; ++d) idx[d] = 0;
    for (std::size_t n = 0; n < total; ++n) {
      std::size_t rem = n;
      for (int d = D - 1; d >= 0; --d) {
        idx[d] = rem % r.s_[d];
        rem /= r.s_[d];
      }
      id<D> i;
      for (int d = 0; d < D; ++d) i.v_[d] = idx[d];
      f(i);
    }
  }

  template <int D, class F> void parallel_for(nd_range<D> nr, F f) {
    const std::size_t lsz = nr.l_.size();
    const std::size_t gsz = nr.g_.size();
    const std::size_t groups = lsz ? gsz / lsz : 0;
    for (std::size_t gi = 0; gi < groups; ++gi) {
      detail::GroupCtx ctx;
      ctx.bar = std::make_unique<std::barrier<>>(lsz);
      std::vector<std::thread> th;
      th.reserve(lsz);
      for (std::size_t l = 0; l < lsz; ++l) {
        th.emplace_back([&, gi, l, lsz]() {
          detail::tls_ctx = &ctx;
          detail::tcache_ctx = nullptr;
          detail::tcache_slot = -1;
          id<D> gid, lid;
          std::size_t rem = gi * lsz + l;
          for (int d = D - 1; d >= 0; --d) {
            gid.v_[d] = rem % nr.g_.s_[d];
            rem /= nr.g_.s_[d];
          }
          for (int d = 0; d < D; ++d) lid.v_[d] = (d == 0) ? l : 0;
          nd_item<D> it(gid, lid);
          f(it);
          detail::tls_ctx = nullptr;
        });
      }
      for (auto& t : th) t.join();
    }
  }
};

/* ---------------- device / queue / event ---------------- */

class device {
 public:
  static std::vector<device> get_devices() { return std::vector<device>(1); }
  template <class T> typename T::value_type get_info() const {
    if constexpr (std::is_same_v<typename T::value_type, std::string>) {
      if constexpr (std::is_same_v<T, info::device::name>)
        return std::string("CPU-SHIM (эмуляция SYCL)");
      return std::string("0.0-cpu-shim");
    } else {
      if constexpr (std::is_same_v<T, info::device::local_mem_size>)
        return std::size_t(65536);
      if constexpr (std::is_same_v<T, info::device::global_mem_size>)
        return std::size_t(1ull << 34);
      if constexpr (std::is_same_v<T, info::device::max_work_group_size>)
        return std::size_t(256);
      return typename T::value_type();
    }
  }
  // CPU эмулирует GPU: is_gpu()=true, иначе прибор сам откажется работать.
  bool has(aspect) const { return true; }
  bool is_gpu() const { return true; }
  bool is_cpu() const { return false; }
};

class event {
 public:
  void wait() {}
  void wait_and_throw() {}
};

class queue {
 public:
  std::shared_ptr<std::atomic<int>> slot_counter =
      std::make_shared<std::atomic<int>>(0);
  queue() {}
  queue(device) {}
  queue(device_selector) {}
  queue(device_selector, property::queue::default_selector) {}
  queue(device_selector, property::queue::in_order) {}
  queue(property::device::host_device) {}
  device get_device() const { return device(); }
  /* DPC++: memcpy возвращает событие (event), у которого есть wait().
     В шиме возвращаем заглушку с тем же интерфейсом — иначе код, написанный
     для реального SYCL, не компилируется против шима (очередной класс
     расхождения API: «шим ≠ реальность»). */
  event& memcpy(void* dst, const void* src, std::size_t n) {
    if (n) std::memcpy(dst, src, n);
    return ev_;
  }
  void wait() {}
  template <class F> event& submit(F f) {
    handler h;
    h.slot_counter = slot_counter;
    f(h);
    return ev_;
  }

 private:
  event ev_;
};

template <class T = void> T* malloc_device(std::size_t n, queue&) {
  return (T*)std::malloc(n * sizeof(T));
}
template <class T = void> void free(T* p, queue&) { std::free(p); }

/* ---------------- математика ---------------- */
template <class T> T sqrt(T x) { return (T)std::sqrt((double)x); }
template <class T> T fmax(T a, T b) { return a > b ? a : b; }
template <class T> T abs(T x) { return x < 0 ? -x : x; }
struct half { float v; };

}  // namespace sycl
