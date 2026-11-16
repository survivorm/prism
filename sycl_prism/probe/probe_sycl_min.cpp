// probe_sycl_min.cpp — МИНИМАЛЬНЫЙ решающий зонд: «видит ли мой собственный
// SYCL-код карту и с какой скоростью?».
//
// Почему отдельный файл, а не библиотека прибора: ответ нужен ДО того, как
// затевать прибор. Он проверяет ровно то, что ломалось в torch.xpu, но без torch:
//   (1) компилятор icpx -fsycl работает и собирает код;
//   (2) SYCL-перечисление видит Intel Iris Xe (тот же стек, на котором у вас
//       работает llama.cpp-SYCL ⇒ это не «новая технология», а проверенная);
//   (3) fp32 GEMM даёт разумную скорость (нужна для dense-пути E-45);
//   (4) fp64 на Xe-LP = 1/64 от fp32 ⇒ подтверждает требование «частичные
//       суммы в fp32, объединение в fp64» (PROTOCOL.md §0.1 п.5).
//
// ВНИМАНИЕ: этот файл НИ РАЗУ не компилировался (в песочнице нет icpx).
// Чтобы ошибок было меньше, здесь НЕТ шаблонов: fp32 и fp64 — две явные
// функции (в шаблонной функции пришлось бы писать `buf.template get_access<…>`).
//
// ВАЖНО про метрику: скорость наивного ядра — это НЕ пик GPU. Решение о переносе
// принимается по ТАЙЛОВАННОМУ ядру (16x16, локальная память) — именно оно сравнимо
// с CPU-базой (28 GFLOPS fp32, измерено). Наивное ядро печатается рядом, чтобы
// увидеть разрыв «плохое ядро» vs «нормальное ядро», и как базу для отношения
// fp64/fp32 (одно и то же ядро ⇒ отношение корректно).
//
// Сборка и запуск:
//     mkdir -p build     # без этого линкер падает: icpx НЕ создаёт выходной каталог
//     icpx -fsycl -O2 -std=c++20 probe/probe_sycl_min.cpp -o build/probe_sycl_min
//     ./build/probe_sycl_min
//
// Ожидаемый вывод (Iris Xe, замерено 16.11.2026):
//     probe rev: v8-20261116-link
//     SYCL devices найдено: 3   (Iris Xe / CPU / Iris Xe — две записи GPU)
//     fp32 GEMM (НАИВНОЕ):      62.5 GFLOP/s
//     fp32 GEMM (ТАЙЛОВАННОЕ):  64.7 GFLOP/s   ⇒ x2.3 к CPU-базе (28)
//     fp64 НЕ ПОДДЕРЖИВАЕТСЯ — замеры fp64 пропускаются (на хосте)
//     ВЕРДИКТ: DEVICE_OK — fp32-путь окупается (>=50 GFLOP/s)
//
// МЕТРИКА: best-of-N (минимум времени из 5 повторов). Наивное среднее давало
// 18-34 GFLOP/s из-за разброса между прогонами (x4!) и переворачивало ВЕРДИКТ.
#include <sycl/sycl.hpp>

#include "../src/prism_sycl_compat.h"

/* ШТАМП РЕВИЗИИ ПРОБЫ: печатается при запуске, чтобы по выводу было видно,
   какая версия пробы собрана (защита от «собралась старая копия»). */
#define PROBE_REV "v16-20261116-barrierprobe"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using Clock = std::chrono::high_resolution_clock;

/* Статистика одного замера throughput.
 *
 * УСТАНОВЛЕНО (16.11.2026): на этой машине разброс между прогонами ОДНОГО И
 * ТОГО ЖЕ ядра достигает ×4 (17.4 / 37.0 / 58.3 / 62.5 / 64.7 GFLOP/s на
 * наивном ядре) — потому что машина общая: параллельно идут прогоны AT и
 * сборки. ⇒ «best-of-N» (минимум) НЕ делает метрику устойчивой: минимум тоже
 * плавает, потому что даже самый быстрый прогон может попасть в конкуренцию.
 * ⇒ ПРАВИЛО (общее для всех метрик решений): замер обязан нести ОЦЕНКУ
 * НЕОПРЕДЕЛЕННОСТИ. Если разброс max/min > 2 — вердикт НЕ ПРИНИМАЕТСЯ
 * («UNDECIDED»), а не «красный» или «зелёный».
 */
/* Загрузка машины (Linux /proc/loadavg) — контекст для замера.
 *
 * ЗАЧЕМ: машина общая (параллельно идут прогоны AT, видеозвонок, сборка) ⇒
 * абсолютное значение GFLOP/s зависит от загрузки, а ВЕРДИКТ — сравнение с
 * порогом 50 ⇒ решение может перевернуться из-за посторонней нагрузки.
 * ⇒ Замер под нагрузкой надо ПОМЕЧАТЬ, а не выдавать как характеристику карты.
 */
static double load_avg_1min() {
    std::ifstream f("/proc/loadavg");
    if (!f) return -1.0;
    double v = -1.0;
    f >> v;
    return v;
}

struct Bench {
    double gflops_min = 0.0;  /* наименее нагруженный прогон = оценка возможности */
    double gflops_med = 0.0;  /* медиана */
    double gflops_max = 0.0;  /* наиболее нагруженный */
    double spread    = 1.0;  /* max/min */
    bool   stable    = true; /* spread <= 2 */
};

static Bench summarize(std::vector<double> gflops) {
    Bench b;
    if (gflops.empty()) return b;
    std::sort(gflops.begin(), gflops.end());
    b.gflops_min = gflops.front();
    b.gflops_max = gflops.back();
    b.gflops_med = gflops[gflops.size() / 2];
    b.spread = (b.gflops_min > 0) ? (b.gflops_max / b.gflops_min) : 1.0;
    b.stable = (b.spread <= 2.0);
    return b;
}

static void print_bench(const char* label, const Bench& b) {
    std::printf("%s: min %.1f | медиана %.1f | max %.1f GFLOP/s   "
                "(разброс x%.1f)%s\n",
                label, b.gflops_min, b.gflops_med, b.gflops_max, b.spread,
                b.stable ? "" : "  <-- НЕСТАБИЛЬНО");
}

static Bench bench_gemm_f32(sycl::queue &q, size_t M, size_t K, size_t N,
                            int reps) {
    std::vector<float> a(M * K), b(K * N), c(M * N);
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.01f * float(i % 97);
    for (size_t i = 0; i < b.size(); ++i) b[i] = 0.01f * float(i % 31);

    auto once = [&]() {
        sycl::buffer<float, 2> ba(a.data(), sycl::range<2>(M, K));
        sycl::buffer<float, 2> bb(b.data(), sycl::range<2>(K, N));
        sycl::buffer<float, 2> bc(c.data(), sycl::range<2>(M, N));
        q.submit([&](sycl::handler &h) {
            auto aa = ba.get_access<sycl::access::mode::read>(h);
            auto ab = bb.get_access<sycl::access::mode::read>(h);
            auto ac = bc.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> id) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) sum += aa[id[0]][k] * ab[k][id[1]];
                ac[id] = sum;
            });
        }).wait_and_throw();
    };

    once();  // прогрев
    /* Собираем ВСЕ прогоны, а не только лучший: разброс — это информация о
       занятости машины, а не шум, который надо спрятать (см. struct Bench). */
    std::vector<double> per_rep;
    per_rep.reserve((size_t)reps);
    const double flops = 2.0 * double(M) * double(K) * double(N);
    for (int r = 0; r < reps; ++r) {
        auto a = Clock::now();
        once();
        auto b = Clock::now();
        double s = std::chrono::duration<double>(b - a).count();
        per_rep.push_back(s > 0 ? flops / s / 1e9 : 0.0);
    }
    double checksum = 0.0;
    for (size_t i = 0; i < c.size(); i += c.size() / 8 + 1) checksum += c[i];
    std::printf("   (контроль суммы fp32: %.6f)\n", checksum);
    return summarize(per_rep);
}

/* Тайлованное ядро 16x16 с локальной памятью — релевантная метрика для решения
   о переносе (сравнимо с CPU-базой 28 GFLOPS fp32), в отличие от наивного. */
static Bench bench_gemm_f32_tiled(sycl::queue &q, size_t M, size_t K, size_t N,
                                  int reps) {
    const size_t TILE = 16;
    std::vector<float> a(M * K), b(K * N), c(M * N);
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.01f * float(i % 97);
    for (size_t i = 0; i < b.size(); ++i) b[i] = 0.01f * float(i % 31);

    auto once = [&]() {
        sycl::buffer<float, 2> ba(a.data(), sycl::range<2>(M, K));
        sycl::buffer<float, 2> bb(b.data(), sycl::range<2>(K, N));
        sycl::buffer<float, 2> bc(c.data(), sycl::range<2>(M, N));
        q.submit([&](sycl::handler &h) {
            auto aa = ba.get_access<sycl::access::mode::read>(h);
            auto ab = bb.get_access<sycl::access::mode::read>(h);
            auto ac = bc.get_access<sycl::access::mode::write>(h);
            // local_accessor создаётся В ОБЛАСТИ command-group: его конструктор
            // требует неконстантный handler&, а внутри [=]-лямбды handler константен.
            auto ta = sycl::local_accessor<float, 2>(sycl::range<2>(TILE, TILE), h);
            auto tb = sycl::local_accessor<float, 2>(sycl::range<2>(TILE, TILE), h);
            h.parallel_for(
                sycl::nd_range<2>(sycl::range<2>(M, N),
                                  sycl::range<2>(TILE, TILE)),
                [=](sycl::nd_item<2> it) {
                    const size_t row = it.get_global_id(0);
                    const size_t col = it.get_global_id(1);
                    const size_t lr = it.get_local_id(0);
                    const size_t lc = it.get_local_id(1);
                    float acc = 0.f;
                    for (size_t k0 = 0; k0 < K; k0 += TILE) {
                        ta[lr][lc] =
                            (row < M && k0 + lc < K) ? aa[row][k0 + lc] : 0.f;
                        tb[lr][lc] =
                            (k0 + lr < K && col < N) ? ab[k0 + lr][col] : 0.f;
                        prism_sycl::barrier(it);
                        for (size_t kk = 0; kk < TILE; ++kk)
                            acc += ta[lr][kk] * tb[kk][lc];
                        prism_sycl::barrier(it);
                    }
                    if (row < M && col < N) ac[row][col] = acc;
                });
        }).wait_and_throw();
    };

    once();  // прогрев
    const double flops = 2.0 * double(M) * double(K) * double(N);
    std::vector<double> per_rep;
    per_rep.reserve((size_t)reps);
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        once();
        auto t1 = Clock::now();
        double sec = std::chrono::duration<double>(t1 - t0).count();
        per_rep.push_back(sec > 0 ? flops / sec / 1e9 : 0.0);
    }
    double checksum = 0.0;
    for (size_t i = 0; i < c.size(); i += c.size() / 8 + 1) checksum += c[i];
    std::printf("   (контроль суммы fp32, тайлованное: %.6f)\n", checksum);
    return summarize(per_rep);
}

static double bench_gemm_f64(sycl::queue &q, size_t M, size_t K, size_t N,
                             int reps) {
    std::vector<double> a(M * K), b(K * N), c(M * N);
    for (size_t i = 0; i < a.size(); ++i) a[i] = 0.01 * double(i % 97);
    for (size_t i = 0; i < b.size(); ++i) b[i] = 0.01 * double(i % 31);

    auto once = [&]() {
        sycl::buffer<double, 2> ba(a.data(), sycl::range<2>(M, K));
        sycl::buffer<double, 2> bb(b.data(), sycl::range<2>(K, N));
        sycl::buffer<double, 2> bc(c.data(), sycl::range<2>(M, N));
        q.submit([&](sycl::handler &h) {
            auto aa = ba.get_access<sycl::access::mode::read>(h);
            auto ab = bb.get_access<sycl::access::mode::read>(h);
            auto ac = bc.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<2>(M, N), [=](sycl::id<2> id) {
                double sum = 0.0;
                for (size_t k = 0; k < K; ++k) sum += aa[id[0]][k] * ab[k][id[1]];
                ac[id] = sum;
            });
        }).wait_and_throw();
    };

    once();
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        auto a = Clock::now();
        once();
        auto b = Clock::now();
        double s = std::chrono::duration<double>(b - a).count();
        if (s < best) best = s;
    }
    double flops = 2.0 * double(M) * double(K) * double(N);
    double checksum = 0.0;
    for (size_t i = 0; i < c.size(); i += c.size() / 8 + 1) checksum += c[i];
    std::printf("   (контроль суммы fp64: %.6f)\n", checksum);
    return best > 0 ? flops / best / 1e9 : 0.0;
}

int main() {
    std::printf("probe rev: %s\n", PROBE_REV);
    std::vector<sycl::device> devs = sycl::device::get_devices();
    std::printf("SYCL devices найдено: %zu\n", devs.size());
    if (devs.empty()) {
        std::printf("VERDICT: DEVICE_MISSING\n");
        std::printf(
            "Подсказки:\n"
            "  * ONEAPI_DEVICE_SELECTOR=opencl:gpu   (запасной бэкенд)\n"
            "  * в LD_LIBRARY_PATH не должно быть ДВУХ рантаймов oneAPI\n"
            "    (пип-овский <venv>/lib затеняет системный — снимите его: env_xpu.sh)\n"
            "  * карта видна системному рантайму? sycl-ls\n");
        return 1;
    }
    for (size_t i = 0; i < devs.size(); ++i) {
        const sycl::device &d = devs[i];
        // is_gpu()/is_cpu() — члены SYCL 2020; трейты info::device::device_type
        // в разных реализациях называются по-разному ⇒ их не используем.
        const char *type = d.is_gpu() ? "gpu" : (d.is_cpu() ? "cpu" : "other");
        std::printf("device %zu: %s | type=%s | fp64=%s\n", i,
                    d.get_info<sycl::info::device::name>().c_str(), type,
                    d.has(sycl::aspect::fp64) ? "да" : "нет");
    }
    /* Одна и та же карта может перечислиться НЕСКОЛЬКО раз (разные бэкенды:
       Level Zero и OpenCL). Для прибора это не «две карты» ⇒ выбираем РОВНО одну
       GPU и не считаем дубли за разные устройства. */
    size_t n_gpu = 0;
    for (const sycl::device &d : devs) if (d.is_gpu()) ++n_gpu;
    if (n_gpu > 1)
        std::printf("ВНИМАНИЕ: GPU перечислена %zu раз (вероятно, разные бэкенды "
                    "для одной карты) — прибор берёт первую.\n", n_gpu);
    if (n_gpu == 0)
        std::printf("ВНИМАНИЕ: GPU не найдена среди %zu устройств.\n", devs.size());

    sycl::device chosen = devs[0];
    for (const sycl::device &d : devs) {
        if (d.is_gpu()) {
            chosen = d;
            break;
        }
    }
    sycl::queue q(chosen);

    const size_t n = 1024;
    Bench b_naive, b_tiled;
    try {
    b_naive = bench_gemm_f32(q, n, n, n, 9);
    print_bench("fp32 GEMM (НАИВНОЕ)", b_naive);
    b_tiled = bench_gemm_f32_tiled(q, n, n, n, 9);
    print_bench("fp32 GEMM (ТАЙЛОВАННОЕ 16x16)", b_tiled);
    } catch (const std::exception &e) {
        std::printf("ОШИБКА fp32-замера: %s\n", e.what());
        std::printf("ВЕРДИКТ: BENCH_FAILED (устройство видно, но замер не выполнен)\n");
        return 2;
    }
    /* Оценка возможности = наименее нагруженный прогон (минимум).
       Медиана и максимум печатаются рядом — по ним видно конкуренцию. */
    const double g32 = b_tiled.gflops_min;
    const double g32_naive = b_naive.gflops_min;
    const double spread = b_tiled.spread > b_naive.spread ? b_tiled.spread : b_naive.spread;
    const bool stable = b_tiled.stable && b_naive.stable;
    /* fp64: на Iris Xe аспекта fp64 НЕТ ⇒ замер невозможен, и попытка его сделать
       роняет программу (uncaught exception) — а это, в свою очередь, УБИВАЕТ
       вердикт. Поэтому: проверяем аспект и ПРОПУСКАЕМ замер с явной пометкой. */
    bool has64 = false;
    try {
        has64 = chosen.has(sycl::aspect::fp64);
    } catch (...) { has64 = false; }
    double g64 = 0.0;
    if (has64) {
        try {
            g64 = bench_gemm_f64(q, n, n, n, 3);
            std::printf("fp64 GEMM %zux%zux%zu (наивное, для отношения): %.1f GFLOP/s\n",
                        n, n, n, g64);
            if (g32_naive > 0 && g64 > 0)
                std::printf("отношение fp64/fp32 (одно ядро): 1/%.0f\n",
                            g32_naive / g64);
        } catch (const std::exception& e) {
            std::printf("fp64-замер пропущен (исключение): %s\n", e.what());
        }
    } else {
        std::printf("fp64 на устройстве НЕ ПОДДЕРЖИВАЕТСЯ (aspect::fp64 = нет) "
                    "— замер пропущен.\n");
        std::printf("  => Это не дефект: на Iris Xe fp64 отсутствует как аспект.\n"
                    "  => Следствие для прибора: ВСЕ fp64-величины (closure G6, "
                    "комбинация частичных lse) считаются НА ХОСТЕ в fp64; "
                    "устройство работает в fp32.\n");
    }
    std::printf("разрыв наивное/тайлованное: x%.1f\n",
                g32_naive > 0 ? g32 / g32_naive : 0.0);

    if (has64) {   // fp64-редукция — только если аспект есть (иначе DPC++ бросит
                   // исключение из деструктора буфера ⇒ abort БЕЗ вердикта)
        const size_t m = 1u << 20;
        std::vector<double> v(m, 1.0);
        double result = 0.0;
        sycl::buffer<double, 1> b(v.data(), sycl::range<1>(m));
        auto t0 = Clock::now();
        q.submit([&](sycl::handler &h) {
            auto acc = b.get_access<sycl::access::mode::read>(h);
            sycl::buffer<double, 1> out(&result, sycl::range<1>(1));
            // аксессор создаётся В handler-контексте (внутри parallel_or h нет)
            auto o = out.get_access<sycl::access::mode::write>(h);
            h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                double tmp = 0.0;
                for (size_t i = 0; i < acc.size(); ++i) tmp += acc[i];
                o[0] = tmp;
            });
        }).wait_and_throw();
        auto t1 = Clock::now();
        std::printf("fp64-редукция %zu элементов: %.3f мс  (частичные суммы "
                    "на GPU держим в fp32)\n", m,
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
    } else {
        std::printf("fp64-редукция пропущена (аспекта fp64 нет) — на хосте.\n");
    }

    /* --- РЕЗЮМЕ ПЕРЕД ОПЦИОНАЛЬНЫМИ БЛОКАМИ ---------------------------------
       Чтобы вердикт был виден даже если последующий блок уронит программу. */
    {
        const double cpu_base_gflops = 28.0;
        std::printf("\n--- РЕЗЮМЕ (сразу после измерений) ---\n");
        std::printf("CPU-база (dense_metrics, 1 поток): %.0f GFLOP/s\n", cpu_base_gflops);
        std::printf("iGPU, тайлованное fp32:            %.1f GFLOP/s  (x%.1f к CPU)\n",
                    g32, g32 / cpu_base_gflops);
        if (!stable)
            std::printf("ВЕРДИКТ: НЕСТАБИЛЬНЫЙ ЗАМЕР (разброс x%.1f) — РЕШЕНИЕ НЕ ПРИНИМАЕТСЯ.\n"
                        "  => Машина занята (параллельные прогоны AT / сборка): повторить в покое.\n"
                        "  => Ориентир (наименее нагруженный прогон): %.1f GFLOP/s\n",
                        spread, g32);
        else if (g32 >= 50.0)
            std::printf("ВЕРДИКТ: DEVICE_OK — fp32-путь окупается (>=50 GFLOP/s).\n");
        else
            std::printf("ВЕРДИКТ: DEVICE_OK, но НАШ fp32-GEMM не окупается (<50 GFLOP/s).\n"
                        "  => Это НЕ предел GPU: это предел НАШЕГО простого ядра (тайлинг 16x16,\n"
                        "     без векторизации и без oneMKL). Для плотного GEMM нужен библиотечный\n"
                        "     (oneMKL GPU) — либо dense-путь остаётся на CPU.\n"
                        "  => Приборные ядра (нулевой слой, ACF) — memory-bound, им скорость GEMM\n"
                        "     не нужна: их перенос решается отдельно и по факту измерения.\n");
    }

    /* Решение по стоп-условиям (ROUTES.md): CPU-база dense_metrics = 28 GFLOPS fp32.
       Требуется >= 50 GFLOP/s, чтобы dense-путь E-45 окупался (x1.8 минимум,
       целевой перенос x10+). Метрика — ТАЙЛОВАННАЯ. */
    const double cpu_base_gflops = 28.0;
    const double load1 = load_avg_1min();
    std::printf("\n--- РЕШЕНИЕ ---\n");
    if (load1 >= 0.0) {
        std::printf("загрузка машины (loadavg, 1 мин): %.2f%s\n", load1,
                    (load1 >= 4.0)
                        ? "  <-- ВЫСОКАЯ: замер идёт под нагрузкой, значение — "
                          "НИЖНЯЯ ОЦЕНКА возможностей карты"
                        : "");
    }
    std::printf("CPU-база (dense_metrics, 1 поток): %.0f GFLOP/s\n", cpu_base_gflops);
    std::printf("iGPU, тайлованное fp32:            %.1f GFLOP/s  (x%.1f к CPU)\n",
                g32, g32 / cpu_base_gflops);
    if (!stable) {
        std::printf("ВЕРДИКТ: НЕСТАБИЛЬНЫЙ ЗАМЕР (разброс x%.1f) — РЕШЕНИЕ НЕ ПРИНИМАЕТСЯ.\n",
                    spread);
        std::printf("  Машина занята: повторить пробу в покое (без параллельных прогонов AT).\n");
        std::printf("  Ориентир (наименее нагруженный прогон): %.1f GFLOP/s\n", g32);
    } else if (load1 >= 4.0 && g32 < 50.0) {
        std::printf("ВЕРДИКТ: ЗАМЕР ПОД НАГРУЗКОЙ (loadavg %.2f) — порог 50 НЕ достигнут,\n"
                    "  но это НЕ оценка возможностей карты (под нагрузкой любая цифра "
                    "занижена).\n", load1);
        std::printf("  Ориентир (наименее нагруженный прогон): %.1f GFLOP/s — повторить в покое.\n",
                    g32);
    } else if (g32 >= 50.0) {
        std::printf("ВЕРДИКТ: DEVICE_OK — fp32-путь окупается (>=50 GFLOP/s).\n");
        std::printf("Дальше: сборка прибора и паритет с numpy (run_local.sh, шаги 2-3).\n");
    } else {
        std::printf("ВЕРДИКТ: DEVICE_OK, но fp32-плотный путь НЕ окупается (<50 GFLOP/s).\n");
        std::printf("Остаются: прибор на приборных ядрах (нулевой слой) и/или T4.\n");
    }
    if (g32_naive > 0 && g64 > 0 && (g32_naive / g64) > 40.0) {
        std::printf("ВНИМАНИЕ: fp64 НЕ в 1/64 от fp32 (получилось 1/%.0f) — "
                    "проверьте требование 'частичные lse в fp32'.\n",
                    g32_naive / g64);
    }
    return 0;
}
