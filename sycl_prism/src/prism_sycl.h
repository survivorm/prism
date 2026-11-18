/* prism_sycl.h — C ABI прибора prism на SYCL (Intel iGPU: Level Zero / OpenCL).
 *
 * Почему C ABI, а не torch.utils.cpp_extension:
 *   icpx тянет свою libstdc++/oneAPI-runtime; torch-extension требует совпадения
 *   C++ ABI с torch. Конфликт версий GCC/libstdc++ — обычное дело. C ABI +
 *   ctypes снимает класс проблем целиком: .so собирается отдельно, грузится
 *   через dlopen, данные передаются указателями.
 *
 * Все функции потокобезопасны по состоянию (очередь создаётся лениво, ошибки
 * пишутся в thread-local буфер).
 *
 * Соглашения:
 *   - вход  float* (fp32) или uint16_t* (fp16, битовый IEEE half, как torch.float16);
 *   - накопление всегда в float (в эталоне field.py — float64: см. допуски в test_parity.py);
 *   - выход double* для скаляров и кривых (наружу — как в field.py);
 *   - матрицы row-major [T, D].
 *
 * ТОЧНОСТЬ (установлено на реальной карте 16.11.2026):
 *   На Intel Iris Xe аспекта fp64 НЕТ ⇒ DPC++ не собирает ни одного ядра с
 *   double. Поэтому ВСЕ буферы УСТРОЙСТВА — fp32, а наружу отдаётся fp64
 *   (приведение float→double точное; погрешность вносит только fp32-суммирование
 *   на устройстве: шум ~ sqrt(n)·eps_fp32, т.е. ~1e-7 при n=1024 и ~1.5e-5 при
 *   n=65535 ⇒ допуски паритета выводятся ИЗ АРИФМЕТИКИ, а не подгоняются).
 *   ИСКЛЮЧЕНИЕ: R/A-split closure (G6) считается ЦЕЛИКОМ НА ХОСТЕ в fp64.
 *   Причина: closure_max — величина порядка ошибки ФП (~1e-13), т.е. остаток
 *   тождества, а не число; в fp32 шум ~1e-6·scale в 5000 раз крупнее самой
 *   closure ⇒ на устройстве величина неразрешима. На хосте (T·D fp64) — тривиально.
 */
#ifndef PRISM_SYCL_H
#define PRISM_SYCL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ШТАМП РЕВИЗИИ: меняется при каждом содержательном изменении прибора.
 * Печатается при загрузке (поле src_rev) и позволяет ОДНОЗНАЧНО установить,
 * какая именно версия собрана — это закрывает ошибку «собралась старая копия».
 * Правило: при любом изменении кода прибора эту строку поднимать. */
#define PRISM_SYCL_SRC_REV "v18-20261117-newops"

#define PRISM_SYCL_ABI_VERSION 2

typedef struct {
    char     name[256];
    char     backend[32];        /* "level_zero" | "opencl" | "ext_oneapi" | ... */
    char     driver[256];
    char     device_type[32];    /* "gpu" | "cpu" */
    int64_t  max_workgroup_size;
    int64_t  global_mem_bytes;
    int64_t  local_mem_bytes;
    int      has_fp16;           /* нативная поддержка half */
    int      has_bf16;
    int      has_subgroups;
    int      sub_group_size;
    char     src_rev[32];        /* PRISM_SYCL_SRC_REV собранной библиотеки */
} PrismSyclDeviceInfo;

/* 0 — ok, <0 — ошибка (код и текст в prism_sycl_last_error). */
int  prism_sycl_abi_version(void);

/* Зонд устройства. Возвращает 0, если SYCL-устройство найдено, иначе <0.
 * info может быть NULL — тогда только факт доступности. */
int  prism_sycl_probe(PrismSyclDeviceInfo* info);

/* Какая форма sycl::group_barrier выбрана при сборке ("access-fence" /
   "group-fence"). Нужна для самопроверки артефакта: по логу видно, каким
   барьером собрана библиотека (барьер БЕЗ fence недопустим — см.
   prism_sycl_compat.h). */
const char* prism_sycl_barrier_form(void);

/* Текст последней ошибки в текущем потоке (буфер внутри библиотеки). */
const char* prism_sycl_last_error(void);

/* --- vpath: ортогонализованный тангенс (канон dip_V) -------------------
 * u [T,D], hn [T] -> v [T-1,D] (fp32), keep [T-1] (0/1).
 * Семантика 1:1 с prism.field.vpath: V[t] = norm(U[t+1] - <U[t+1],U[t]> U[t]),
 * U = normalize(u*hn); строки с ||V|| <= 1e-6 помечаются keep=0.
 */
int  prism_sycl_vpath_f32(const float* u, const double* hn, int64_t T, int64_t D,
                           float* v, uint8_t* keep);
int  prism_sycl_vpath_f16(const uint16_t* u, const double* hn, int64_t T, int64_t D,
                           float* v, uint8_t* keep);

/* --- dcurve: кривая per-t косинусов -----------------------------------
 * P [T,D] -> out[nk]; stat: 0 = mean, 1 = median (как в field.dcurve).
 * Медиана — точная (numpy-семантика: для чётного T — среднее двух центральных).
 */
int  prism_sycl_dcurve_f32(const float* P, int64_t T, int64_t D,
                           const int64_t* ks, int64_t nk, int stat, double* out);
int  prism_sycl_dcurve_f16(const uint16_t* P, int64_t T, int64_t D,
                           const int64_t* ks, int64_t nk, int stat, double* out);

/* --- acf_moments -------------------------------------------------------
 * x [N,D] -> c0, c[1..K], r, f_dc, sum_rho_full, gram_resid, row_norm_sq[N].
 * c считается для ВСЕХ лагов 1..N-1 (как в эталоне), наружу отдаются первые K.
 * rho/rho_Tk считаются на стороне Python (нужен префикс-сумм, дешёвый).
 */
int  prism_sycl_acf_f32(const float* x, int64_t N, int64_t D, int64_t K,
                        double* c, double* c0, double* r, double* f_dc,
                        double* sum_rho_full, double* gram_resid, double* row_norm_sq);
int  prism_sycl_acf_f16(const uint16_t* x, int64_t N, int64_t D, int64_t K,
                        double* c, double* c0, double* r, double* f_dc,
                        double* sum_rho_full, double* gram_resid, double* row_norm_sq);

/* --- R/A-split: closure и cross санити (G6) ---------------------------
 * u [T,D], hn [T] -> closure_max, du_u_cross_max (в эталоне — максимумы по t).
 */
int  prism_sycl_ra_closure_f32(const float* u, const double* hn, int64_t T, int64_t D,
                               double* closure_max, double* du_u_cross_max);
int  prism_sycl_ra_closure_f16(const uint16_t* u, const double* hn, int64_t T, int64_t D,
                               double* closure_max, double* du_u_cross_max);

/* --- перестановочный null-слой (главный CPU-горб серии) -----------------
 * Протокол AT0_1._perm_stat: перестановка блоков blk=2 (сохраняет лаг-1),
 * 60 сидов × 53 доков × 3 объекта. RNG ОСТАЁТСЯ НА ХОСТЕ (numpy, те же сиды ⇒
 * побитово тот же поток перестановок) ⇒ на GPU уезжает только арифметика.
 *
 * perm_idx — int32[n_perm * N]: индекс исходной строки для каждой (перестановка, позиция).
 *
 * perm_row_sq: e[p][t] = ||x[perm_idx[p*N+t]]||^2  (нужно для знаменателя ρ̄_Tk).
 * perm_lags:   c[p][k] = Σ_{t<N-k} x[idx[p,t]] · x[idx[p,t+k]]  для ks[nk].
 */
int  prism_sycl_perm_row_sq_f32(const float* x, int64_t N, int64_t D,
                                 const int32_t* perm_idx, int64_t n_perm, double* row_sq);
int  prism_sycl_perm_row_sq_f16(const uint16_t* x, int64_t N, int64_t D,
                                 const int32_t* perm_idx, int64_t n_perm, double* row_sq);
int  prism_sycl_perm_lags_f32(const float* x, int64_t N, int64_t D,
                               const int32_t* perm_idx, int64_t n_perm,
                               const int64_t* ks, int64_t nk, double* c_out);
int  prism_sycl_perm_lags_f16(const uint16_t* x, int64_t N, int64_t D,
                               const int32_t* perm_idx, int64_t n_perm,
                               const int64_t* ks, int64_t nk, double* c_out);

/* --- gemm: C[M,N] = A[M,K] @ B[K,N], row-major, ТОЛЬКО fp32 --------------
 * AT0_3 (TZ_sycl_new_ops §2.1): тейлованное ядро, конфиг зафиксирован тюнингом
 * (probe/RESULTS_tuning.md: 128x64, WG=256, KC=16, double-buffer, SLM=24KB).
 * Чанкинг по hang-check драйвера: один launch ≤ 32 GFLOP (N режется на блоки,
 * сигнатура НЕ меняется — чанкинг внутри). Ограничения: 1≤M≤65536, 1≤K≤4096,
 * 1≤N≤262144, иначе PS_ERR_ARG. Local-бюджет — PS_ERR_LOCAL.
 */
int  prism_sycl_gemm_f32(const float* A, const float* B,
                         int64_t M, int64_t K, int64_t N, float* C);

/* --- eigtop / eigbot: r пар собственных (n ≤ 1024) ------------------------
 * A [batch,n,n] f32, row-major. eigtop — power iteration + MGS (верхние r);
 * eigbot — inverse iteration с Cholesky (нижние r): L — фактор Холецкого
 * (A + μI) = L·Lᵀ, считается НА ХОСТЕ (numpy) и приходит аргументом.
 * Выход: vals[batch,r] f64 (Rayleigh по A), vecs[batch,n,r] f32 (столбцы —
 * собственные векторы), resid[batch] f64 (подпространственный остаток
 * ‖AV − V(VᵀAV)‖_F/‖A‖_F на последней итерации — ЧЕСТНЫЙ отчёт, не критерий
 * остановки). Ограничения: 1≤batch≤65536, 1≤n≤1024, 1≤r≤32 (r<n),
 * 1≤max_iter≤512, иначе PS_ERR_ARG. Local-бюджет (2·n·r·4 + r·r·4 + 64·4)
 * ≤ 65536, иначе PS_ERR_LOCAL.
 */
int  prism_sycl_eigtop(const float* A, int64_t batch, int64_t n, int64_t r,
                       int64_t max_iter, double* vals, float* vecs, double* resid);
int  prism_sycl_eigbot(const float* A, const float* Lfac, int64_t batch, int64_t n,
                       int64_t r, int64_t max_iter, double* vals, float* vecs,
                       double* resid);

/* --- eigjac: полный спектр симметричного батча, cyclic Jacobi (n ≤ 256) ---
 * A [batch,n,n] f32, симметричный. Выход: vals[batch,n] f64 (ПО ВОЗРАСТАНИЮ —
 * сортировка на хосте, порядок torch), vecs[batch,n,n] f32 (столбцы,
 * host-MGS f64 — write-back в f32), offdiag[batch] f64 (относительное
 * ‖offdiag(A)‖_F/‖A‖_F на остановке — ЧЕСТНЫЙ отчёт сходимости).
 * Ограничения: 1≤batch≤65536, 1≤n≤256, 1≤max_sweeps≤64, 1e-6≤tol≤1e-3,
 * иначе PS_ERR_ARG. Local-бюджет — PS_ERR_LOCAL (v5.3: локальной памяти НЕТ,
 * проверка — форма единообразия с eigtop/eigbot).
 * offdiag_curve/curve_len (v5.2, контракт v5.3 F3): опциональны (NULL/0 —
 * без кривой). curve[s−1] = offdiag_rel матрицы 0 после sweep s (s=1..k,
 * k = число реально выполненных sweep'ов); если k < curve_len —
 * curve[k..curve_len) = offdiag[0] (явный fill до конца буфера).
 * Batch-чанкинг (v5.3 F1): 1 sweep = ⌈batch/C⌉ запусков, каждый ≤ C матриц
 * (C = EIGJAC_CHUNK — фиксация тюнинга AT0_3, начальное 1024; t_sweep(C) ≤ 4с
 * — твёрдый тест-гейт §2.7, 5×-запас к hang-check ~20с).
 * Типы (v5.3 F7): max_sweeps/curve_len — int64_t, tol — double. */
int  prism_sycl_eigjac(const float* A, int64_t batch, int64_t n,
                       int64_t max_sweeps, double tol,
                       double* vals, float* vecs, double* offdiag,
                       double* offdiag_curve, int64_t curve_len);

#ifdef __cplusplus
}
#endif
#endif /* PRISM_SYCL_H */
