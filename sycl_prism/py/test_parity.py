"""Паритет SYCL ↔ numpy-эталон (py/prism_sycl_ref.py, побитово совпал с prism/field.py).

Запуск:
    python3 test_parity.py              # против эталона
    PRISM_SYCL_BACKEND=reference python3 test_parity.py   # самопроверка обвязки

Что проверяем и почему такие допуски:
  - вход fp32, аккумуляция fp32, эталон — float64 ⇒ расхождение только от порядка
    суммирования: для косинусов берём |Δ| ≤ 1e-4 (значения ~1), для c(k) — rel ≤ 1e-4;
  - вход fp16: сперва округляем вход до fp16, эталон считаем по округлённым значениям
    (cast в float64) ⇒ любое расхождение — чисто арифметика ядра, не вход;
  - тождества G1 (r = 1 + 2Σρ̂) и G3 (r = N·f_DC) проверяются на ВЫВОДЕ SYCL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prism_sycl as ps
import prism_sycl_ref as ref


def make_data(T: int, D: int, seed: int = 11):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((T, D))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    hn = np.cumsum(rng.standard_normal(T) * 0.01) + 10.0
    x = rng.standard_normal((T, D))          # приращения (для acf)
    return u, hn, x


def nan_report(a, name):
    """Диагностика NaN/inf в выходе прибора.

    НУЛИ и NaN — разные вещи: нули означают «ядро записало нули», NaN — «в
    выходе нечисловые значения» (часто: буфер не записан вовсе ⇒ в нём
    мусор/неинициализированная память). Различать их обязательно: это разные
    диагнозы с разными причинами.
    """
    a = np.asarray(a)
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    fin = a[np.isfinite(a)]
    lo = float(fin.min()) if fin.size else float("nan")
    hi = float(fin.max()) if fin.size else float("nan")
    print(f"    {name}: nan={n_nan} inf={n_inf} из {a.size} | "
          f"finite[{lo:.6g}, {hi:.6g}] | первые: "
          f"{np.array2string(a.ravel()[:4], precision=5)}")
    return n_nan, n_inf


def rel(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = max(float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b)) / denom)


def main() -> int:
    print("backend:", ps.backend())
    if not ps.available():
        print("SYCL недоступен:", getattr(ps, "_load_error", "?"))
        return 2
    if not ps._is_reference():
        print("устройство:", ps.device_info())

    ok = True
    shapes = [(384, 1024), (96, 128), (64, 32)]
    ks = list(range(1, 97))

    for (T, D) in shapes:
        u, hn, x = make_data(T, D)
        K = min(96, T - 1)

        # --- vpath ---
        V_ref, drop_ref = ref.vpath(u, hn)
        V, keep = ps.vpath(u, hn)
        Vr = V[keep.astype(bool)] if keep is not None else V
        d = float(np.max(np.abs(Vr - V_ref))) if Vr.shape == V_ref.shape else float("inf")
        print(f"[{T}x{D}] vpath      maxdiff={d:.3e} dropped={drop_ref}")
        nan_report(V, "vpath V")
        print(f"    vpath keep: оставлено {int(np.sum(keep))} из {keep.size} "
              f"(эталон оставил {V_ref.shape[0]}, отброшено {drop_ref})")
        ok &= d < 1e-5

        # --- dcurve mean ---
        cm_ref = ref.dcurve(u, ks[:K], "mean")
        cm = ps.dcurve(u, ks[:K], "mean")
        d = rel(cm, cm_ref)
        print(f"[{T}x{D}] dcurve mean rel={d:.3e}")
        nan_report(cm, "dcurve mean")
        ok &= d < 1e-4

        # --- dcurve median ---
        cd_ref = ref.dcurve(u, ks[:K], "median")
        cd = ps.dcurve(u, ks[:K], "median")
        d = rel(cd, cd_ref)
        print(f"[{T}x{D}] dcurve med  rel={d:.3e}")
        nan_report(cd, "dcurve med")
        ok &= d < 1e-4

        # --- acf_moments ---
        a_ref = ref.acf_moments(x, K)
        a = ps.acf_moments(x, K)
        d_c = rel(a["c"], a_ref["c"])
        d_c0 = rel([a["c0"]], [a_ref["c0"]])
        d_r = rel([a["r"]], [a_ref["r"]])
        d_gr = abs(a["gram_resid"] - a_ref["gram_resid"])
        print(f"[{T}x{D}] acf c rel={d_c:.3e} c0 rel={d_c0:.3e} r rel={d_r:.3e} "
              f"gram_resid Δ={d_gr:.3e}")
        ok &= (d_c < 1e-4 and d_c0 < 1e-4 and d_r < 1e-4 and d_gr < 1e-6)

        # --- тождества G1/G3 ---
        # ВНИМАНИЕ: r = N·f_DC, где N = ЧИСЛО СТРОК (T), а не N-1 (см. field._selfcheck).
        # (a) на ЭТАЛОНЕ: проверяет сам эталон (должно быть ~1e-16);
        full = ref.acf_moments(x, T - 1)
        g1 = abs(1.0 + 2.0 * full["rho"].sum() - full["r"])
        g3 = abs(full["r"] - T * ref.f_dc(x))
        print(f"[{T}x{D}] G1(ref)={g1:.3e} G3(ref)={g3:.3e}")
        ok &= (g1 < 1e-9 and g3 < 1e-9)

        # (b) на ВЫХОДЕ УСТРОЙСТВА — это и есть настоящая проверка прибора.
        # Устройство (Iris Xe) не имеет fp64 ⇒ суммы считаются в fp32, и тождество
        # держится не до 1e-9, а до ШУМА fp32: абсолютная ошибка суммы ~ sqrt(n)·eps,
        # n = N-k слагаемых ⇒ eps_fp32/2 = 6e-8 ⇒ допуск ~ 1e-5 (запас ×3).
        # Допуск ВЫВОДИТСЯ из арифметики, а не подгоняется под результат.
        full_d = ps.acf_moments(x, T - 1)
        rho_d = np.asarray(full_d["c"], dtype=np.float64) / full_d["c0"]
        g1d = abs(1.0 + 2.0 * rho_d.sum() - full_d["r"])
        g3d = abs(full_d["r"] - T * ref.f_dc(x))
        eps32 = float(np.finfo(np.float32).eps)  # 1.19e-7
        n_terms = max(1, T - 1)
        tol_id = max(1e-6, 4.0 * eps32 * np.sqrt(n_terms))
        print(f"[{T}x{D}] G1(устройство)={g1d:.3e} G3(устройство)={g3d:.3e} "
              f"(допуск шума fp32 = {tol_id:.2e})")
        ok &= (g1d < tol_id and g3d < tol_id)

        # --- R/A-split closure ---
        # closure_max — величина порядка ошибки ФП (~1e-13), т.е. это не «число»,
        # а остаток тождества G6 ⇒ сравниваем по МАСШТАБУ слагаемых, а не относительно него.
        s_ref = ref.ra_split(u, hn)
        scale = float(np.mean(np.einsum("ij,ij->i", s_ref["dh"], s_ref["dh"]))) or 1.0
        cl, cr = ps.ra_closure(u, hn)
        d_cl = abs(cl - s_ref["closure_max"]) / scale
        print(f"[{T}x{D}] ra closure Δ/scale={d_cl:.3e} cross={cr:.3e} "
              f"(ref {s_ref['closure_max']:.3e}/{s_ref['du_u_cross_max']:.3e})")
        ok &= (d_cl < 1e-9 and cr < 1e-9)

    # --- fp16-вход: эталон по округлённым значениям ---
    u, hn, x = make_data(384, 256, seed=5)
    u16 = u.astype(np.float16)
    K = 96
    cm_ref = ref.dcurve(u16.astype(np.float64), list(range(1, K + 1)), "mean")
    cm = ps.dcurve(u16, list(range(1, K + 1)), "mean")
    d = rel(cm, cm_ref)
    print(f"[fp16] dcurve mean rel={d:.3e}")
    ok &= d < 5e-3

    # --- перестановочный null-слой: паритет с протоколом AT0_1 ---
    # Эталон ref.perm_stat побитово идентичен AT0_1_accounting._perm_stat (проверено отдельно).
    x = make_data(383, 128, seed=9)[2]
    f = ref.f_dc(x)
    seeds = [1009 + i for i in range(8)]
    got = ps.permutation_null(x, f, seeds)
    want = np.array([ref.perm_stat(x, f, s) for s in seeds])
    for name, col in (("band", 0), ("e3", 1), ("e4", 2)):
        a = np.asarray(got[name], dtype=np.float64)
        b = want[:, col]
        scale = max(float(np.max(np.abs(b))), 1e-12)
        d = float(np.max(np.abs(a - b)) / scale)
        print(f"[null] {name:5s} rel={d:.3e} (band/e3/e4 ~ {b[0]:.3e})")
        ok &= d < 1e-4

    print("\nИТОГ:", "ПАРИТЕТ OK" if ok else "РАСХОЖДЕНИЕ")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
