"""Загрузчик prism_sycl: тонкая обёртка над C ABI (ctypes) + numpy-эталон как фолбэк.

Использование:
    import prism_sycl as ps
    print(ps.device_info())
    curve = ps.dcurve(P, ks=list(range(1, 97)), stat="median")   # numpy-вход
    # либо torch-тензор — конвертируется прозрачно

Если .so не найдена или устройства нет ⇒ ps.available() == False, и вызовы
явно падают с RuntimeError (молчаливого перехода на CPU нет: молчаливый
фолбэк в измерительном приборе — источник тихих расхождений).

Фолбэк-режим (эталон на numpy) включается явно: ps.use_reference() / env
PRISM_SYCL_BACKEND=reference.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

import prism_sycl_ref as ref

__all__ = [
    "available",
    "device_info",
    "vpath",
    "dcurve",
    "acf_moments",
    "f_dc",
    "ra_closure",
    "permutation_indices",
    "permutation_null",
    "use_reference",
    "backend",
]

_HERE = Path(__file__).resolve().parent
_LIB_CANDIDATES = [
    os.environ.get("PRISM_SYCL_LIB", ""),
    str(_HERE.parent / "build" / "libprism_sycl.so"),
    str(_HERE / "libprism_sycl.so"),
    str(_HERE.parent / "build" / "prism_sycl.so"),
]

_ERR = {
    -1: "нет SYCL-устройства",
    -2: "исключение в SYCL",
    -3: "неверные аргументы",
    -4: "нет поддержки fp16",
    -5: "не хватило local memory (медиана при большом T)",
}


class _DeviceInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("backend", ctypes.c_char * 32),
        ("driver", ctypes.c_char * 256),
        ("device_type", ctypes.c_char * 32),
        ("max_workgroup_size", ctypes.c_int64),
        ("global_mem_bytes", ctypes.c_int64),
        ("local_mem_bytes", ctypes.c_int64),
        ("has_fp16", ctypes.c_int),
        ("has_bf16", ctypes.c_int),
        ("has_subgroups", ctypes.c_int),
        ("sub_group_size", ctypes.c_int),
        ("src_rev", ctypes.c_char * 32),
    ]


_lib = None
_load_error = None


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    for cand in _LIB_CANDIDATES:
        if not cand or not Path(cand).exists():
            continue
        try:
            lib = ctypes.CDLL(cand)
            lib.prism_sycl_abi_version.restype = ctypes.c_int
            lib.prism_sycl_abi_version.argtypes = []
            if lib.prism_sycl_abi_version() != 2:
                _load_error = f"{cand}: ABI mismatch"
                continue
            lib.prism_sycl_probe.restype = ctypes.c_int
            lib.prism_sycl_probe.argtypes = [ctypes.POINTER(_DeviceInfo)]
            lib.prism_sycl_barrier_form.argtypes = []
            lib.prism_sycl_barrier_form.restype = ctypes.c_char_p
            lib.prism_sycl_last_error.restype = ctypes.c_char_p
            lib.prism_sycl_last_error.argtypes = []

            # ВАЖНО: без явных argtypes ctypes передаёт целые как 32-бит int
            # ⇒ T/D/N обрезались бы. Все размеры — int64.
            c_p = ctypes.POINTER(ctypes.c_float)
            c_d = ctypes.POINTER(ctypes.c_double)
            c_i64 = ctypes.POINTER(ctypes.c_int64)
            c_i32 = ctypes.POINTER(ctypes.c_int32)
            c_u8 = ctypes.POINTER(ctypes.c_uint8)

            def bind(name, argtypes):
                fn = getattr(lib, "prism_sycl_" + name)
                fn.restype = ctypes.c_int
                fn.argtypes = argtypes
                return fn

            bind("vpath_f32", [c_p, c_d, ctypes.c_int64, ctypes.c_int64, c_p, c_u8])
            bind("vpath_f16", [ctypes.POINTER(ctypes.c_uint16), c_d,
                               ctypes.c_int64, ctypes.c_int64, c_p, c_u8])
            bind("dcurve_f32", [c_p, ctypes.c_int64, ctypes.c_int64, c_i64,
                                ctypes.c_int64, ctypes.c_int, c_d])
            bind("dcurve_f16", [ctypes.POINTER(ctypes.c_uint16), ctypes.c_int64,
                                ctypes.c_int64, c_i64, ctypes.c_int64,
                                ctypes.c_int, c_d])
            acf_args = [None, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                        c_d, c_d, c_d, c_d, c_d, c_d, c_d]
            bind("acf_f32", [c_p] + acf_args[1:])
            bind("acf_f16", [ctypes.POINTER(ctypes.c_uint16)] + acf_args[1:])
            ra_args = [None, c_d, ctypes.c_int64, ctypes.c_int64, c_d, c_d]
            bind("ra_closure_f32", [c_p] + ra_args[1:])
            bind("ra_closure_f16", [ctypes.POINTER(ctypes.c_uint16)] + ra_args[1:])
            perm_args = [None, ctypes.c_int64, ctypes.c_int64, c_i32,
                         ctypes.c_int64, c_d]
            bind("perm_row_sq_f32", [c_p] + perm_args[1:])
            bind("perm_row_sq_f16", [ctypes.POINTER(ctypes.c_uint16)] + perm_args[1:])
            perm_lags_args = [None, ctypes.c_int64, ctypes.c_int64, c_i32,
                              ctypes.c_int64, c_i64, ctypes.c_int64, c_d]
            bind("perm_lags_f32", [c_p] + perm_lags_args[1:])
            bind("perm_lags_f16", [ctypes.POINTER(ctypes.c_uint16)] + perm_lags_args[1:])
            _lib = lib
            return _lib
        except OSError as e:  # pragma: no cover
            _load_error = f"{cand}: {e}"
            # Частая причина на машине с conda-python: .so собрана более новым
            # g++/icpx, чем libstdc++ из состава python'а ⇒
            #   "version `GLIBCXX_3.4.31' not found (required by ...so)"
            # ⇒ лечится либо статической линковкой (-static-libstdc++), либо тем,
            #    что библиотека грузится под тем же python'ом, под которым собрана.
            if "GLIBCXX" in str(e) or "CXXABI" in str(e):
                _load_error += (
                    "\n    ПОДСКАЗКА: несовместимая libstdc++ (сборка новее, чем у "
                    "python'а).\n    Лечение 1: пересобрать .so со статической "
                    "линковкой (-static-libstdc++ -static-libgcc).\n"
                    "    Лечение 2: грузить библиотеку тем же python'ом, которым "
                    "собрана, либо добавить её libstdc++ в LD_LIBRARY_PATH.")
    if _load_error is None:
        _load_error = "libprism_sycl.so не найдена (соберите: CXX=icpx cmake -B build && cmake --build build)"
    return None


def _check(rc: int) -> None:
    if rc != 0:
        msg = _ERR.get(rc, f"код {rc}")
        lib = _load()
        detail = ""
        if lib is not None:
            detail = lib.prism_sycl_last_error().decode("utf-8", "replace")
        hint = ""
        if "No kernel named" in detail:
            hint = (
                "\n  ПОДСКАЗКА (проверено на карте 16.11.2026): причина в СБОРКЕ, "
                "не в коде.\n  Если библиотека собрана с -fsycl-targets=spir64_x86_64 "
                "(SPIR-V-путь),\n  рантайм ищет ядро по имени ТИПА (_ZTS...), а в "
                "SPIR-V-модуле имя\n  точки входа — манглированное operator() "
                "(_ZZ...) ⇒ не совпадают.\n  РЕШЕНИЕ: собирать БЕЗ "
                "-fsycl-targets (путь AOT) — это доказано работающим:\n      "
                "rm -rf build && cmake -B build -DPRISM_SYCL_TARGETS= ...\n  "
                "Второй кандидат (тоже проверен): ядро в АНОНИМНОМ пространстве ⇒ "
                "в исходнике\n  не должно быть `namespace {`.")
        elif "undefined symbol" in detail or "version LIBUR_LOADER" in detail:
            hint = (
                "\n  ПОДСКАЗКА: несовместимость oneAPI-рантаймов (LIBUR_LOADER). "
                "В LD_LIBRARY_PATH\n  должен быть ровно один набор oneAPI-библиотек.")
        raise RuntimeError(f"prism_sycl: {msg}" + (f" — {detail}" if detail else "")
                           + hint)


def _np(x):
    """torch-тензор → numpy (CPU), numpy → numpy. Без копий, где возможно."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


def _f32(x) -> np.ndarray:
    a = _np(x)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    return np.ascontiguousarray(a)


def _f16(x) -> np.ndarray:
    a = _np(x)
    if a.dtype != np.float16:
        a = a.astype(np.float16)
    return np.ascontiguousarray(a)


def _c_f64(x) -> np.ndarray:
    return np.ascontiguousarray(_np(x).astype(np.float64))


def available() -> bool:
    if os.environ.get("PRISM_SYCL_BACKEND", "").lower() == "reference":
        return True
    lib = _load()
    if lib is None:
        return False
    info = _DeviceInfo()
    return lib.prism_sycl_probe(ctypes.byref(info)) == 0


def device_info() -> dict:
    lib = _load()
    if lib is None:
        raise RuntimeError(f"prism_sycl: {_load_error}")
    info = _DeviceInfo()
    rc = lib.prism_sycl_probe(ctypes.byref(info))
    _check(rc)
    return {
        "src_rev": info.src_rev.decode("utf-8", "replace"),
        "name": info.name.decode("utf-8", "replace"),
        "backend": info.backend.decode("utf-8", "replace"),
        "driver": info.driver.decode("utf-8", "replace"),
        "device_type": info.device_type.decode("utf-8", "replace"),
        "max_workgroup_size": int(info.max_workgroup_size),
        "global_mem_bytes": int(info.global_mem_bytes),
        "local_mem_bytes": int(info.local_mem_bytes),
        "has_fp16": bool(info.has_fp16),
        "has_bf16": bool(info.has_bf16),
        "has_subgroups": bool(info.has_subgroups),
        "sub_group_size": int(info.sub_group_size),
        "barrier_form": lib.prism_sycl_barrier_form().decode("utf-8", "replace"),
    }


def use_reference() -> None:
    """Явно переключиться на numpy-эталон (для сверки и для CI без GPU)."""
    os.environ["PRISM_SYCL_BACKEND"] = "reference"


def _is_reference() -> bool:
    return os.environ.get("PRISM_SYCL_BACKEND", "").lower() == "reference"


def backend() -> str:
    if _is_reference():
        return "reference"
    return "sycl" if _load() is not None else "unavailable"


# ------------------------------ операции ------------------------------

def vpath(u, hn):
    """Ортогонализованный тангенс: (V [T-1,D] float32, keep [T-1] uint8)."""
    u_a = _f32(u)
    hn_a = _c_f64(hn)
    if _is_reference():
        U = u_a.astype(np.float64) * hn_a[:, None]
        U /= np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        Vo = U[1:] - (np.einsum("ij,ij->i", U[1:], U[:-1])[:, None] * U[:-1])
        nv = np.linalg.norm(Vo, axis=1)
        keep = (nv > 1e-6).astype(np.uint8)
        Vfull = np.zeros_like(Vo)
        m = keep.astype(bool)
        Vfull[m] = Vo[m] / nv[m][:, None]
        return Vfull.astype(np.float32), keep
    lib = _load()
    if lib is None:
        raise RuntimeError(f"prism_sycl: {_load_error}")
    T, D = u_a.shape
    V = np.empty((T - 1, D), dtype=np.float32)
    keep = np.empty(T - 1, dtype=np.uint8)
    rc = lib.prism_sycl_vpath_f32(
        u_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        hn_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(T), ctypes.c_int64(D),
        V.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        keep.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    _check(rc)
    return V, keep


def dcurve(P, ks, stat="median"):
    """Кривая per-t косинусов: out[nk] (float64)."""
    P_a = _f32(P)
    ks_a = np.ascontiguousarray(np.asarray(ks, dtype=np.int64))
    stat_i = 1 if stat == "median" else 0
    if _is_reference():
        return ref.dcurve(P_a.astype(np.float64), ks_a.tolist(), stat)
    lib = _load()
    if lib is None:
        raise RuntimeError(f"prism_sycl: {_load_error}")
    T, D = P_a.shape
    out = np.empty(len(ks_a), dtype=np.float64)
    rc = lib.prism_sycl_dcurve_f32(
        P_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(T), ctypes.c_int64(D),
        ks_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(len(ks_a)), ctypes.c_int(stat_i),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    _check(rc)
    return out


def acf_moments(x, K):
    """Моментный ACF: словарь как в prism.field.acf_moments (rho/rho_Tk — на хосте)."""
    x_a = _f32(x)
    N, D = x_a.shape
    if _is_reference():
        return ref.acf_moments(x_a.astype(np.float64), K)
    lib = _load()
    if lib is None:
        raise RuntimeError(f"prism_sycl: {_load_error}")
    NL = N - 1
    c = np.empty(NL, dtype=np.float64)
    e = np.empty(N, dtype=np.float64)
    c0 = np.zeros(1, dtype=np.float64)
    r = np.zeros(1, dtype=np.float64)
    fdc = np.zeros(1, dtype=np.float64)
    srf = np.zeros(1, dtype=np.float64)
    gr = np.zeros(1, dtype=np.float64)
    rc = lib.prism_sycl_acf_f32(
        x_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(N), ctypes.c_int64(D), ctypes.c_int64(K),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c0.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        fdc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        srf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        gr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        e.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    _check(rc)
    # rho/rho_Tk — префикс-суммы норм (дёшево, и сохраняет семантику эталона).
    S = np.cumsum(e)
    d = S[N - 2 - np.arange(K)]
    return {
        "c0": float(c0[0]),
        "c": c[:K],
        "rho": c[:K] / c0[0],
        "rho_Tk": c[:K] / d,
        "r": float(r[0]),
        "sum_rho_full": float(srf[0]),
        "gram_resid": float(gr[0]),
    }


def f_dc(x) -> float:
    a = acf_moments(x, 1)
    return float(a["r"]) / float(x.shape[0])


def ra_closure(u, hn):
    """(G6) closure_max и du_u_cross_max."""
    u_a = _f32(u)
    hn_a = _c_f64(hn)
    if _is_reference():
        s = ref.ra_split(u_a.astype(np.float64), hn_a)
        return s["closure_max"], s["du_u_cross_max"]
    lib = _load()
    if lib is None:
        raise RuntimeError(f"prism_sycl: {_load_error}")
    T, D = u_a.shape
    cl = np.zeros(1, dtype=np.float64)
    cr = np.zeros(1, dtype=np.float64)
    rc = lib.prism_sycl_ra_closure_f32(
        u_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        hn_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int64(T), ctypes.c_int64(D),
        cl.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    _check(rc)
    return float(cl[0]), float(cr[0])


# ------------------------- перестановочный null-слой -------------------------

def permutation_indices(n: int, seed: int, blk: int = 2) -> np.ndarray:
    """Индексы block-blk перестановки (1:1 с AT0_1_accounting._perm_stat)."""
    return ref.permutation_indices(n, seed, blk)


def permutation_null(x, fdc, seeds, blk: int = 2, K: int = 96):
    """Перестановочный null: (band, e3, e4) для каждого сида.

    Протокол AT0_1_accounting._perm_stat: band = mean_{k=9..K}(ρ̄_Tk(k) − f_dc),
    e3 = ρ̄_Tk(3) − f_dc, e4 = ρ̄_Tk(4) − f_dc.

    x     — ряд приращений [N, D];
    fdc   — f_dc(x) (инвариантна к перестановке, пересчёту не подлежит);
    seeds — список целых сидов (в AT0_1: (s+1)*1009 + i).

    Возвращает dict с массивами band/e3/e4 длины len(seeds).
    """
    x_a = _f32(x)
    N, D = x_a.shape
    seeds = list(seeds)
    if not seeds:
        return {"band": np.zeros(0), "e3": np.zeros(0), "e4": np.zeros(0)}
    f = float(fdc)

    if _is_reference():
        res = [ref.perm_stat(x_a.astype(np.float64), f, s, blk=blk, K=K) for s in seeds]
        return {"band": np.array([r[0] for r in res]),
                "e3": np.array([r[1] for r in res]),
                "e4": np.array([r[2] for r in res])}

    lib = _load()
    if lib is None:
        raise RuntimeError(f"prism_sycl: {_load_error}")

    idx = np.stack([ref.permutation_indices(N, s, blk) for s in seeds]).astype(np.int32)
    idx_c = np.ascontiguousarray(idx)
    K = min(K, N - 1)
    ks = np.array([3, 4] + list(range(9, K + 1)), dtype=np.int64)
    ks_c = np.ascontiguousarray(ks)
    nk = len(ks)

    c = np.empty((len(seeds), nk), dtype=np.float64)
    row_sq = np.empty((len(seeds), N), dtype=np.float64)
    rc = lib.prism_sycl_perm_lags_f32(
        x_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(N), ctypes.c_int64(D),
        idx_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(len(seeds)),
        ks_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(nk),
        c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    _check(rc)
    rc = lib.prism_sycl_perm_row_sq_f32(
        x_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int64(N), ctypes.c_int64(D),
        idx_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(len(seeds)),
        row_sq.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    _check(rc)

    S = np.cumsum(row_sq, axis=1)                     # S[p][i] = Σ_{t<=i} ‖xp_t‖²
    den = np.empty((len(seeds), nk), dtype=np.float64)
    for j, k in enumerate(ks):
        den[:, j] = S[:, N - 1 - k]
    rho = c / den
    e3 = rho[:, 0] - f
    e4 = rho[:, 1] - f
    band = rho[:, 2:].mean(axis=1) - f
    return {"band": band, "e3": e3, "e4": e4}


if __name__ == "__main__":  # быстрый смоук
    print("backend:", backend())
    if available() and not _is_reference():
        print(device_info())
    else:
        print("SYCL-устройство недоступно:", _load_error)
