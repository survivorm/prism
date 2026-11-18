"""py/test_new_ops.py — AT0_3: тесты новых операций (TZ_sycl_new_ops §2.7).

Запуск:  python -m pytest py/test_new_ops.py -q   (run_local.sh — отдельный шаг)
DEVICE_OK-guard: без библиотеки/устройства — skip всех (референс-режим
PRISM_SYCL_BACKEND=reference — тесты идут по numpy, CI без GPU).

Допуски — из TZ §2.7 (выведены, не подогнаны):
  - gemm: max|Δ|/max|C| ≤ 1e-4 (fp32-аккумуляция по K=1024); детерминизм 2× бит-в-бит;
  - eigjac jcache: rel(λ) ≤ 1e-3 (рассеянные) / ≤ 1e-2 (кластерные, n-33);
    подпространства top/bottom-32 ≤ 5e-2; ортогональность ≤ 1e-4; offdiag < 1e-4
    (ЯВНО max_sweeps=64, m-28);
  - svd_top H-доки: σ rel ≤ 1e-3, подпространства U/V ≤ 1e-3;
  - Gagg-композиция: подпространства ≤ 1e-3, S5b-аналог ≤ 1e-5 (f64, хост);
  - eigtop/eigbot random-SPD: relλ ≤ 1e-3, subdist ≤ 1e-3; jcache top/bottom-7:
    подпространство ≤ 5e-2 + Ritz-max ≥/≤ λ_r(1∓1e-3); ортогональность ≤ 1e-5.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

import prism_sycl as ps
import prism_sycl_ref as ref

JCACHE_DIR = "/data/prism_kernels/exp27b/results"
LAYERS = (6, 23)
DT_SET = ((0, 0), (0, 1), (1, 0))   # 3 (d,t) на слой (TZ §2.7)
# jcache exp27b: ключи JR/P128/H — (domain, d, t); фиксированный выбор домена
# (все 4 домена структурно идентичны — тест на 3 (d,t) в пределах одного).
DOMAIN = "wiki"
R = 7
RESULTS = ROOT / "results" / "AT0_3"

# ----------------------------- DEVICE_OK -----------------------------

def _device_ok() -> bool:
    try:
        return ps.available()
    except Exception:
        return False


def _is_real_gpu() -> bool:
    return _device_ok() and not ps._is_reference()


def _skip_nogpu():
    if not _device_ok():
        pytest.skip("DEVICE_OK: sycl-библиотека/устройство недоступны")


# ----------------------------- данные -----------------------------

_jcache_cache: dict = {}


def _jcache(L: int, want=("H", "JR", "P128")):
    if L not in _jcache_cache:
        ck = torch.load(os.path.join(JCACHE_DIR, f"exp27b_jcache_L{L}.pt"),
                        map_location="cpu", weights_only=False)
        _jcache_cache[L] = {k: ck[k] for k in want}
        del ck
        gc.collect()
    return _jcache_cache[L]


def _gram64(jr: torch.Tensor, p: torch.Tensor) -> np.ndarray:
    """P-взвешенный Граимиан (f64) — ТОЧНО как AT2c L386-393."""
    jr64 = jr.double()
    sp = torch.sqrt(p.double())
    g = (sp.unsqueeze(1) * (jr64 @ jr64.T)) * sp.unsqueeze(0)
    return g.cpu().numpy()


def _subspace_dist(v: np.ndarray, v_ref: np.ndarray) -> float:
    """‖(I − V Vᵀ) V_ref‖_F (v [n,k], v_ref [n,k], k одинаковы)."""
    return float(np.linalg.norm(v_ref - v @ (v.T @ v_ref)))


def _rel_eig(lam: np.ndarray, lam_ref: np.ndarray) -> float:
    return float(np.max(np.abs(lam - lam_ref)) / np.max(np.abs(lam_ref)))


# ----------------------------- GEMM -----------------------------

def _gemm_check(A: np.ndarray, B: np.ndarray, tag: str) -> float:
    C = ps.gemm(A, B)
    Cref = (A.astype(np.float64) @ B.astype(np.float64))
    rel = float(np.max(np.abs(C - Cref)) / np.max(np.abs(Cref)))
    print(f"[gemm {tag}] {A.shape} @ {B.shape}: max|Δ|/max|C| = {rel:.3e}")
    assert rel <= 1e-4, f"gemm {tag}: rel={rel} > 1e-4"
    del Cref
    gc.collect()
    return rel


def test_gemm_small():
    _skip_nogpu()
    rng = np.random.default_rng(11)
    A = rng.standard_normal((64, 128)).astype(np.float32)
    B = rng.standard_normal((128, 256)).astype(np.float32)
    _gemm_check(A, B, "64x128@128x256")


def test_gemm_big():
    _skip_nogpu()
    rng = np.random.default_rng(12)
    A = rng.standard_normal((4096, 1024)).astype(np.float32)
    B = rng.standard_normal((1024, 4096)).astype(np.float32)
    _gemm_check(A, B, "4096x1024@1024x4096")
    del A, B
    gc.collect()


def test_gemm_wide():
    """512×1024 @ 1024×248320 (TZ: память ≈ 3GB, n-17)."""
    _skip_nogpu()
    rng = np.random.default_rng(13)
    A = rng.standard_normal((512, 1024)).astype(np.float32)
    B = rng.standard_normal((1024, 248320)).astype(np.float32)
    _gemm_check(A, B, "512x1024@1024x248320")
    del A, B
    gc.collect()


def test_gemm_oob():
    """OOB-формы: M не кратно 128 / N не кратно 64 (m-16, designed-сценарий guard'а)."""
    _skip_nogpu()
    rng = np.random.default_rng(14)
    A = rng.standard_normal((33, 64)).astype(np.float32)
    B = rng.standard_normal((64, 97)).astype(np.float32)
    _gemm_check(A, B, "33x64@64x97")
    del A, B
    gc.collect()
    if not _is_real_gpu():
        pytest.skip("wide OOB на reference-режиме — слишком долго (CPU f64)")
    A = rng.standard_normal((4096, 1024)).astype(np.float32)
    B = rng.standard_normal((1024, 248321)).astype(np.float32)
    _gemm_check(A, B, "4096x1024@1024x248321")
    del A, B
    gc.collect()


def test_gemm_determinism():
    """Детерминизм: 2 запуска — бит-в-бит (in_order + фиксированный порядок чанков)."""
    _skip_nogpu()
    rng = np.random.default_rng(15)
    A = rng.standard_normal((512, 1024)).astype(np.float32)
    B = rng.standard_normal((1024, 4096)).astype(np.float32)
    C1 = ps.gemm(A, B)
    C2 = ps.gemm(A, B)
    assert np.array_equal(C1, C2), "gemm: 2 запуска НЕ бит-в-бит"
    del A, B, C1, C2
    gc.collect()


# ----------------------------- EIGJAC -----------------------------

def _eigjac_jcache_layer(L: int) -> dict:
    """Сверка eigjac с torch.linalg.eigh (f64) на 3 реальных (d,t) слоя L."""
    dc = _jcache(L)
    JRt, Pt = dc["JR"], dc["P128"]
    out = {}
    for (d, t) in DT_SET:
        g64 = _gram64(JRt[(DOMAIN, d, t)], Pt[(DOMAIN, d, t)])
        lam_ref, vecs_ref = torch.linalg.eigh(torch.from_numpy(g64))
        lam_ref, vecs_ref = lam_ref.cpu().numpy(), vecs_ref.cpu().numpy()
        lam, vecs, off = ps.eigjac(g64.astype(np.float32)[None, :, :],
                                   max_sweeps=64, tol=1e-5)
        lam, vecs, off = lam[0], vecs[0], off[0]
        # rel(λ): рассеянные ≤ 1e-3, кластерные ≤ 1e-2 (n-33):
        # кластерный λ_i — если сосед (по спектру) в пределах 5% от |λ_i|
        order = np.argsort(lam_ref)
        worst = 0.0
        for i in range(len(lam_ref)):
            j = np.argsort(np.abs(lam_ref - lam_ref[i]))[1]
            clustered = abs(lam_ref[j]) / max(abs(lam_ref[i]), 1e-300) < 0.05
            tol = 1e-2 if clustered else 1e-3
            d_i = abs(lam[i] - lam_ref[i]) / max(abs(lam_ref[i]), 1e-300)
            worst = max(worst, d_i)
            assert d_i <= tol, f"eigjac L{L} (d={d},t={t}): λ[{i}] rel={d_i} > {tol}"
        # подпространства top-32 / bottom-32
        top_mine = vecs[:, -32:]
        top_ref = vecs_ref[:, -32:]
        d_top = _subspace_dist(top_mine, top_ref)
        bot_mine = vecs[:, :32]
        bot_ref = vecs_ref[:, :32]
        d_bot = _subspace_dist(bot_mine, bot_ref)
        ortho = float(np.linalg.norm(vecs.T @ vecs - np.eye(128)))
        assert d_top <= 5e-2, f"eigjac L{L} (d={d},t={t}): top-32 subdist={d_top}"
        assert d_bot <= 5e-2, f"eigjac L{L} (d={d},t={t}): bottom-32 subdist={d_bot}"
        assert ortho <= 1e-4, f"eigjac L{L} (d={d},t={t}): ortho={ortho}"
        assert off < 1e-4, f"eigjac L{L} (d={d},t={t}): offdiag={off} >= 1e-4"
        out[f"L{L}_{d}_{t}"] = {"rel_lambda_max": worst, "top32_subdist": d_top,
                                "bottom32_subdist": d_bot, "ortho": ortho,
                                "offdiag": off}
        print(f"[eigjac L{L} d{d}t{t}] relλmax={worst:.2e} top32={d_top:.2e} "
              f"bot32={d_bot:.2e} ortho={ortho:.2e} offdiag={off:.2e}")
    return out


def test_eigjac_jcache():
    _skip_nogpu()
    res = {}
    for L in LAYERS:
        res.update(_eigjac_jcache_layer(L))
    _jcache_cache.clear()
    gc.collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eigjac_jcache.json").write_text(json.dumps(res, indent=2))


def test_eigjac_determinism():
    """TZ §0.5: бит-в-бит повторный запуск. v5.3: один work-item на матрицу,
    фиксированный порядок ротаций, локальной памяти нет (в v5.1 multi-lane
    режим гонялся на кросс-ячейках — реплика /tmp/jac4_replica.cpp); батч >
    EIGJAC_CHUNK — чанки (детерминизм не зависит от чанкирования: матрицы
    независимы)."""
    _skip_nogpu()
    rng = np.random.default_rng(31)
    M = rng.standard_normal((4, 64, 64))
    A = (M @ M.transpose(0, 2, 1) + 64 * np.eye(64)).astype(np.float32)
    r1 = ps.eigjac(A, max_sweeps=8, tol=1e-5)
    r2 = ps.eigjac(A, max_sweeps=8, tol=1e-5)
    for k in range(4):
        assert np.array_equal(r1[0][k], r2[0][k]), \
            f"eigjac det: матрица {k}: vals НЕ бит-в-бит"
        assert np.array_equal(r1[1][k], r2[1][k]), \
            f"eigjac det: матрица {k}: vecs НЕ бит-в-бит"
    assert np.array_equal(r1[2], r2[2]), "eigjac det: offdiag НЕ бит-в-бит"
    print("[eigjac det] 2 запуска бит-в-бит (4×64×64, 8 sweep)")


def test_eigjac_curve():
    """TZ §2.3/§2.7 (v5.3 F3): curve — per-sweep offdiag матрицы 0, явный
    контракт: curve[s−1] = offdiag после sweep s (s=1..k), curve[k..size) =
    offdiag[0] (явный fill)."""
    _skip_nogpu()
    rng = np.random.default_rng(41)
    M = rng.standard_normal((1, 128, 128))
    A = (M @ M.transpose(0, 2, 1) + 128 * np.eye(128)).astype(np.float32)
    tol = 1e-5
    curve = np.empty(64)
    vals, vecs, off = ps.eigjac(A, max_sweeps=64, tol=tol, curve=curve)
    assert curve[0] > 0.0, f"eigjac curve: curve[0]={curve[0]} — должен быть > 0"
    # неубывание (f64)
    for i in range(1, 64):
        assert curve[i] <= curve[i - 1] + 1e-15, \
            f"eigjac curve: рост на позиции {i}: {curve[i-1]} → {curve[i]}"
    # k = первый индекс с curve < tol (стоп после sweep k+1)
    k = int(np.argmax(curve < tol)) if (curve < tol).any() else 64
    # offdiag[0] — финальное; curve[k-1] == offdiag[0]; curve[k..64) == offdiag[0]
    assert abs(curve[k - 1] - off[0]) < 1e-15, \
        f"eigjac curve: curve[{k-1}]={curve[k-1]} != offdiag[0]={off[0]}"
    assert np.all(curve[k:] == off[0]), \
        f"eigjac curve: fill [k..64) не финальным значением: {curve[k:k+4]}"
    # без curve — 3-тupl, как прежде
    r0 = ps.eigjac(A, max_sweeps=64, tol=tol)
    assert r0[0].shape == (1, 128) and r0[1].shape == (1, 128, 128)
    print(f"[eigjac curve] k={k}, curve[0]={curve[0]:.2e}, финал={off[0]:.2e}, fill OK")


def test_eigjac_sweep_time_guard():
    """TZ §2.7 (v5.3 F1/F9): sweep-time guard — ТЮНИНГ-шаг (таблица
    (n, chunk)→t_sweep в артефакт) + твёрдые гейты: 1 launch (n=128, чанк
    C=1024) ≤ 4с И 1 launch (n=256, чанк C/4=256) ≤ 4с (5×-запас к hang-check
    ~20с; n=256 — стоимость ×4.02) И C ≥ 256 (иначе громкий fail — не тихое
    замедление)."""
    _skip_nogpu()
    CHUNKS = (1, 16, 64, 256, 1024, 4096)
    C_FIXED = 1024  # = EIGJAC_CHUNK в src/prism_sycl.cpp (фиксация тюнинга)
    rng = np.random.default_rng(51)
    table = {}
    for chunk in CHUNKS:
        M = rng.standard_normal((chunk, 128, 128))
        A = (M @ M.transpose(0, 2, 1) + 128 * np.eye(128)).astype(np.float32)
        # прогрев + 3 замера (медиана; max_sweeps=1 — ровно 1 sweep)
        ps.eigjac(A, max_sweeps=1, tol=1e-6)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            ps.eigjac(A, max_sweeps=1, tol=1e-6)
            ts.append(time.perf_counter() - t0)
        med = float(np.median(ts))
        table[f"n128_c{chunk}"] = med
        print(f"[eigjac tuning] n=128 chunk={chunk:>5}: t_sweep(med3)={med:.3f}с")
        del A, M
        gc.collect()
    # F9: n=256, чанк C/4 (стоимость sweep на матрицу ×4.02 к n=128)
    C256 = C_FIXED // 4
    M = rng.standard_normal((C256, 256, 256))
    A = (M @ M.transpose(0, 2, 1) + 256 * np.eye(256)).astype(np.float32)
    ps.eigjac(A, max_sweeps=1, tol=1e-6)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        ps.eigjac(A, max_sweeps=1, tol=1e-6)
        ts.append(time.perf_counter() - t0)
    med256 = float(np.median(ts))
    table[f"n256_c{C256}"] = med256
    print(f"[eigjac tuning] n=256 chunk={C256:>5}: t_sweep(med3)={med256:.3f}с")
    del A, M
    gc.collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eigjac_tuning.json").write_text(json.dumps(table, indent=2))
    # твёрдые гейты (TZ §2.7 v5.3):
    assert table[f"n128_c{C_FIXED}"] <= 4.0, \
        f"eigjac F1: t_sweep(n=128, чанк {C_FIXED})={table[f'n128_c{C_FIXED}']:.2f}с > 4с — " \
        f"hang-check-риск; понизить EIGJAC_CHUNK и пересобрать"
    assert table[f"n256_c{C256}"] <= 4.0, \
        f"eigjac F9: t_sweep(n=256, чанк {C256})={table[f'n256_c{C256}']:.2f}с > 4с — " \
        f"hang-check-риск для n=256; понизить EIGJAC_CHUNK и пересобрать"
    assert table["n128_c256"] <= 4.0, \
        f"eigjac F1: t_sweep(n=128, чанк 256)={table['n128_c256']:.2f}с > 4с — GPU " \
        f"медленнее расчёта (C < 256 недостижим) — громкий fail, не тихое замедление"
    print(f"[eigjac F1/F9] гейты пройдены: n=128 c{C_FIXED}="
          f"{table[f'n128_c{C_FIXED}']:.3f}с; n=256 c{C256}={med256:.3f}с; c256={table['n128_c256']:.3f}с — все ≤ 4с")


def test_eigjac_fudge_guard():
    """TZ §2.7 (v5.3 F12): fudge-guard XLA — designed-тест-сценарий
    (AGENTS.md: guard обязан быть протестирован).
    (a) диагональная матрица — все ротации skip (|A_ij| = 0 ≤ guard),
        off = 0 со 2-го sweep, vals = диагональ (сортирована), без NaN;
    (b) пара с |A_ij| ≤ 0.1·eps_f32·min(|A_ii|,|A_jj|): A_ii=A_jj=1e10,
        A_ij=0.05 (guard: 0.1·1.1920929e-7·1e10 = 0.119 ≥ 0.05) — ротация
        НЕ применяется; off_rel = 0.05/(1e10·√2) ≈ 3.5e-12 < tol ⇒ стоп, vals =
        диагональ; без NaN."""
    _skip_nogpu()
    # (a) диагональная
    A = np.zeros((1, 8, 8), dtype=np.float32)
    diag = np.array([5.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0], dtype=np.float32)
    np.fill_diagonal(A[0], diag)
    vals, vecs, off = ps.eigjac(A, max_sweeps=4, tol=1e-5)
    assert not np.any(np.isnan(vals)) and not np.any(np.isnan(vecs)), \
        "fudge-guard (a): NaN"
    expect = np.sort(diag)
    assert np.allclose(vals[0], expect, rtol=0, atol=1e-6), \
        f"fudge-guard (a): vals={vals[0]} != diag={expect}"
    assert off[0] < 1e-12, f"fudge-guard (a): off={off[0]}"
    # (b) guard-skip пара (2×2, batch=1)
    B = np.array([[1e10, 0.05], [0.05, 1e10]], dtype=np.float32)[None, :, :]
    vals2, vecs2, off2 = ps.eigjac(B, max_sweeps=4, tol=1e-5)
    assert not np.any(np.isnan(vals2)) and not np.any(np.isnan(vecs2)), \
        "fudge-guard (b): NaN"
    assert abs(vals2[0, 0] - 1e10) < 1e4 and abs(vals2[0, 1] - 1e10) < 1e4, \
        f"fudge-guard (b): vals={vals2[0]} — ротация должна не применяться"
    assert off2[0] < 1e-10, f"fudge-guard (b): off={off2[0]}"
    print("[eigjac fudge-guard] (a) diagonal + (b) guard-skip: OK, без NaN")


def test_eigjac_random_spd():
    _skip_nogpu()
    rng = np.random.default_rng(21)
    M = rng.standard_normal((8, 128, 128))
    A = M @ M.transpose(0, 2, 1) + 128 * np.eye(128)
    vals, vecs, off = ps.eigjac(A.astype(np.float32), max_sweeps=64, tol=1e-5)
    lam_ref, vecs_ref = np.linalg.eigh(A)
    for b in range(8):
        rel = _rel_eig(vals[b], lam_ref[b])
        assert rel <= 1e-3, f"eigjac random-SPD batch {b}: relλ={rel}"
        d_top = _subspace_dist(vecs[b, :, -32:], vecs_ref[b, :, -32:])
        d_bot = _subspace_dist(vecs[b, :, :32], vecs_ref[b, :, :32])
        assert d_top <= 1e-3, f"eigjac random-SPD batch {b}: top subdist={d_top}"
        assert d_bot <= 1e-3, f"eigjac random-SPD batch {b}: bot subdist={d_bot}"
        ortho = float(np.linalg.norm(vecs[b].T @ vecs[b] - np.eye(128)))
        assert ortho <= 1e-4, f"eigjac random-SPD batch {b}: ortho={ortho}"
    assert np.all(off < 1e-4), f"eigjac random-SPD: offdiag={off}"


def test_eigjac_convergence_jcache():
    """m-28: ЧИСЛО sweep'ов до tol на 2–3 реальных jcache (→ артефакт)."""
    _skip_nogpu()
    res = {}
    for L in (6, 23):
        dc = _jcache(L)
        JRt, Pt = dc["JR"], dc["P128"]
        for (d, t) in DT_SET[:2]:
            g64 = _gram64(JRt[(DOMAIN, d, t)], Pt[(DOMAIN, d, t)])
            g32 = g64.astype(np.float32)[None, :, :]
            curve = {}
            for k in (1, 2, 4, 8, 16, 32, 64):
                t0 = time.time()
                _, _, off = ps.eigjac(g32, max_sweeps=k, tol=1e-6)
                curve[k] = {"offdiag": float(off[0]), "wall_s": round(time.time() - t0, 3)}
            sweeps_to_tol = next((k for k in (1, 2, 4, 8, 16, 32, 64)
                                  if curve[k]["offdiag"] < 1e-5), None)
            res[f"L{L}_{d}_{t}"] = {"curve": curve, "sweeps_to_1e-5": sweeps_to_tol}
            print(f"[eigjac L{L} d{d}t{t}] sweeps до 1e-5: {sweeps_to_tol}; "
                  f"curve: " + " ".join(f"{k}:{v['offdiag']:.1e}" for k, v in curve.items()))
    _jcache_cache.clear()
    gc.collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eigjac_convergence.json").write_text(json.dumps(res, indent=2))
    # честная проверка: хотя бы один из замеров обязан сойтись в 64 sweep'ах (TZ: < 1e-4)
    for key, v in res.items():
        assert v["curve"][64]["offdiag"] < 1e-4, f"{key}: 64 sweep'а не дали < 1e-4"


# ----------------------------- SVD_TOP -----------------------------

def test_svd_top_hdocs():
    _skip_nogpu()
    res = {}
    for L in LAYERS:
        dc = _jcache(L)
        Ht = dc["H"]
        # H-доки [128,1024]: 128 документов t=0..127 при (DOMAIN, d=0)
        Hdoc = torch.stack([Ht[(DOMAIN, 0, t)] for t in range(128)]).float().numpy()
        Hc = Hdoc - Hdoc.mean(0, keepdims=True)
        Hc32 = Hc.astype(np.float32)
        U, S, Vh = ps.svd_top(Hc32, 7)
        Uref, Sref, Vhref = np.linalg.svd(Hc.astype(np.float64), full_matrices=False)
        rel_S = float(np.max(np.abs(S - Sref[:7])) / np.max(np.abs(Sref[:7])))
        assert rel_S <= 1e-3, f"svd_top L{L}: σ rel={rel_S}"
        V_mine = Vh.T
        V_ref = Vhref.T
        d_U = _subspace_dist(U, Uref[:, :7])
        d_V = _subspace_dist(V_mine, V_ref[:, :7])
        assert d_U <= 1e-3, f"svd_top L{L}: U subdist={d_U}"
        assert d_V <= 1e-3, f"svd_top L{L}: V subdist={d_V}"
        # m > n (1024×128) — зеркальный случай
        U2, S2, Vh2 = ps.svd_top(Hc32.T, 7)
        rel_S2 = float(np.max(np.abs(S2 - Sref[:7])) / np.max(np.abs(Sref[:7])))
        assert rel_S2 <= 1e-3, f"svd_top L{L} m>n: σ rel={rel_S2}"
        d_U2 = _subspace_dist(U2, V_ref[:, :7])      # U(Aᵀ) = V(A)
        d_V2 = _subspace_dist(Vh2.T, Uref[:, :7])    # V(Aᵀ) = U(A)
        assert d_U2 <= 1e-3, f"svd_top L{L} m>n: U subdist={d_U2}"
        assert d_V2 <= 1e-3, f"svd_top L{L} m>n: V subdist={d_V2}"
        res[f"L{L}"] = {"sigma_rel": rel_S, "U_subdist": d_U, "V_subdist": d_V,
                        "mt_n_sigma_rel": rel_S2, "mt_n_U_subdist": d_U2,
                        "mt_n_V_subdist": d_V2}
        print(f"[svd_top L{L}] σrel={rel_S:.2e} U={d_U:.2e} V={d_V:.2e} "
              f"(m>n: σrel={rel_S2:.2e} U={d_U2:.2e} V={d_V2:.2e})")
    _jcache_cache.clear()
    gc.collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "svd_top_hdocs.json").write_text(json.dumps(res, indent=2))


def test_svd_top_degenerate():
    """Вырожденный вход (ранг < r): σ-нуль + V_i = 0-вектор (явный, не мусор)."""
    _skip_nogpu()
    A = np.zeros((8, 8), dtype=np.float32)
    A[:3, :3] = np.eye(3)
    U, S, Vh = ps.svd_top(A, 5)
    assert np.all(S[:3] > 0), f"svd_top degen: ненулевые σ потеряны: {S}"
    assert np.all(S[3:] == 0.0), f"svd_top degen: σ-нули не нули: {S}"
    if _is_real_gpu():
        assert np.all(Vh[3:, :] == 0.0), "svd_top degen: V_i ≠ 0-вектор для σ=0"
        assert np.all(U[:, 3:] == 0.0), "svd_top degen: U_i ≠ 0-вектор для σ=0"


# ----------------------------- GAGG-КОМПОЗИЦИЯ -----------------------------

def _gagg_check(L: int) -> dict:
    """TZ §2.4: Gsmall = Jagg·Jaggᵀ (128×128) + eigjac + lift — vs eigh(Gagg, f64)."""
    dc = _jcache(L, want=("JR",))
    JRt = dc["JR"]
    # Jagg [128,1024] — фиксированный блок (DOMAIN, d=0, t=0)
    Jagg = JRt[(DOMAIN, 0, 0)].double()              # [128,1024]
    Jagg32 = Jagg.float().numpy()                    # f64→f32 как в AT2c
    Jagg64 = Jagg.cpu().numpy()
    Gagg = Jagg64.T @ Jagg64                          # [1024,1024] rank ≤ 128 (f64)
    lam, w = torch.linalg.eigh(torch.from_numpy(Gagg))
    lam_np = lam.cpu().numpy()                        # по возрастанию
    w_np = w.cpu().numpy()
    order = torch.argsort(lam, descending=True).cpu().numpy()
    lammax = float(lam_np.max())
    nz = np.where(lam_np > 1e-12 * lammax)[0]
    top_idx = order[:R]                               # сильнейшие (убыв.)
    weak_idx = nz[:min(R, len(nz))]                   # слабейшие НЕ-нулевые
    weakmid_idx = order[R:2 * R]                      # позиции r..2r (C1c)
    # sycl-путь: Gsmall + eigjac + lift
    Gsmall = ps.gemm(Jagg32, Jagg32.T)                # [128,128] f32
    lam_s, U_s, _ = ps.eigjac(Gsmall[None, :, :], max_sweeps=64, tol=1e-6)
    lam_s, U_s = lam_s[0], U_s[0]                     # по возрастанию
    res = {}
    for name, idx in (("top", top_idx), ("weak", weak_idx), ("weakmid", weakmid_idx)):
        lam_sel = lam_np[idx]
        # подбор столбцов Gsmall по совпадению λ (в порядке убывания выбранной группы)
        u_cols = []
        for li in idx:
            d = np.abs(lam_s - lam_np[li]) / max(abs(lam_np[li]), 1e-300)
            j = int(np.argmin(d))
            assert d < 1e-3, f"gagg L{L} {name}: λ не совпал (rel={d})"
            u_cols.append(U_s[:, j])
        u_sel = np.stack(u_cols, axis=1)
        # lift v = Jaggᵀ u / √λ (f64, на хосте — аналог _lift_arr AT2c)
        sig = np.sqrt(np.maximum(lam_sel, 0.0))
        sig = np.where(sig <= 1e-12 * lammax, 0.0, sig)
        v = (Jagg64.T @ u_sel) / sig
        vref = w_np[:, idx]
        d_sp = _subspace_dist(v, vref)
        assert d_sp <= 1e-3, f"gagg L{L} {name}: subdist={d_sp}"
        # S5b-аналог: max_j ‖Gagg v_j − λ_j v_j‖/|λ_j| (f64, хост)
        s5b = 0.0
        for j in range(len(idx)):
            rj = np.linalg.norm(Gagg @ v[:, j] - lam_sel[j] * v[:, j]) / abs(lam_sel[j])
            s5b = max(s5b, float(rj))
        assert s5b <= 1e-5, f"gagg L{L} {name}: S5b={s5b} > 1e-5"
        res[name] = {"subdist": d_sp, "s5b_max": s5b}
        print(f"[gagg L{L} {name}] subdist={d_sp:.2e} s5b={s5b:.2e}")
    return res


def test_gagg_composition():
    _skip_nogpu()
    res = {}
    for L in LAYERS:
        res[f"L{L}"] = _gagg_check(L)
    _jcache_cache.clear()
    gc.collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "gagg_composition.json").write_text(json.dumps(res, indent=2))


# ----------------------------- EIGTOP / EIGBOT -----------------------------

def test_eigtop_eigbot_random_spd():
    _skip_nogpu()
    rng = np.random.default_rng(31)
    M = rng.standard_normal((6, 128, 128))
    A = M @ M.transpose(0, 2, 1) + 128 * np.eye(128)   # рассеянный спектр
    A32 = A.astype(np.float32)
    vals, vecs, resid = ps.eigtop(A32, R, max_iter=512)
    lam_ref, vecs_ref = np.linalg.eigh(A)
    for b in range(6):
        rel = _rel_eig(vals[b], lam_ref[b, -R:])
        assert rel <= 1e-3, f"eigtop random batch {b}: relλ={rel}"
        d = _subspace_dist(vecs[b], vecs_ref[b, :, -R:])
        assert d <= 1e-3, f"eigtop random batch {b}: subdist={d}"
        ortho = float(np.linalg.norm(vecs[b].T @ vecs[b] - np.eye(R)))
        assert ortho <= 1e-5, f"eigtop random batch {b}: ortho={ortho} (n-14)"
    # eigbot — ТОЛЬКО на full-rank SPD
    vals_b, vecs_b, resid_b = ps.eigbot(A32, R, max_iter=512)
    for b in range(6):
        rel = _rel_eig(vals_b[b], lam_ref[b, :R])
        assert rel <= 1e-3, f"eigbot random batch {b}: relλ={rel}"
        d = _subspace_dist(vecs_b[b], vecs_ref[b, :, :R])
        assert d <= 1e-3, f"eigbot random batch {b}: subdist={d}"
        ortho = float(np.linalg.norm(vecs_b[b].T @ vecs_b[b] - np.eye(R)))
        assert ortho <= 1e-5, f"eigbot random batch {b}: ortho={ortho} (n-14)"


def test_eigtop_eigbot_jcache():
    _skip_nogpu()
    res = {}
    for L in LAYERS:
        dc = _jcache(L)
        JRt, Pt = dc["JR"], dc["P128"]
        for (d, t) in DT_SET:
            g64 = _gram64(JRt[(DOMAIN, d, t)], Pt[(DOMAIN, d, t)])
            lam_ref, vecs_ref = np.linalg.eigh(g64)
            g32 = g64.astype(np.float32)[None, :, :]
            vals, vecs, resid = ps.eigtop(g32, R, max_iter=512)
            d_top = _subspace_dist(vecs[0], vecs_ref[:, -R:])
            ritz_max_top = float(vals[0].max())
            lam_r_top = float(lam_ref[-R])
            assert d_top <= 5e-2, f"eigtop L{L} d{d}t{t}: subdist={d_top}"
            assert ritz_max_top >= lam_r_top * (1 - 1e-3), \
                f"eigtop L{L} d{d}t{t}: Ritz-max={ritz_max_top} < λ_r(1-1e-3)"
            ortho = float(np.linalg.norm(vecs[0].T @ vecs[0] - np.eye(R)))
            assert ortho <= 1e-5, f"eigtop L{L} d{d}t{t}: ortho={ortho} (n-14)"
            vals_b, vecs_b, resid_b = ps.eigbot(g32, R, max_iter=512)
            d_bot = _subspace_dist(vecs_b[0], vecs_ref[:, :R])
            ritz_max_bot = float(vals_b[0].max())
            lam_r_bot = float(lam_ref[R - 1])
            assert d_bot <= 5e-2, f"eigbot L{L} d{d}t{t}: subdist={d_bot}"
            assert ritz_max_bot <= lam_r_bot * (1 + 1e-3), \
                f"eigbot L{L} d{d}t{t}: Ritz-max={ritz_max_bot} > λ_r(1+1e-3)"
            ortho_b = float(np.linalg.norm(vecs_b[0].T @ vecs_b[0] - np.eye(R)))
            assert ortho_b <= 1e-5, f"eigbot L{L} d{d}t{t}: ortho={ortho_b} (n-14)"
            res[f"L{L}_{d}_{t}"] = {"top_subdist": d_top, "bot_subdist": d_bot,
                                    "ritz_max_top": ritz_max_top, "lam_r_top": lam_r_top,
                                    "ritz_max_bot": ritz_max_bot, "lam_r_bot": lam_r_bot,
                                    "resid_top": float(resid[0]), "resid_bot": float(resid_b[0])}
            print(f"[eig L{L} d{d}t{t}] top: subdist={d_top:.2e} ritz={ritz_max_top:.3e}"
                  f"(λ_r={lam_r_top:.3e}) | bot: subdist={d_bot:.2e} ritz={ritz_max_bot:.3e}"
                  f"(λ_r={lam_r_bot:.3e}) | resid {resid[0]:.1e}/{resid_b[0]:.1e}")
    _jcache_cache.clear()
    gc.collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eigtop_eigbot_jcache.json").write_text(json.dumps(res, indent=2))


# ----------------------------- ДЕГРАДАЦИЯ -----------------------------

def test_degradation():
    _skip_nogpu()
    rng = np.random.default_rng(41)
    # NaN в A → NaN в resid/offdiag (не тихий мусор)
    A = rng.standard_normal((2, 32, 32))
    A = A @ A.transpose(0, 2, 1) + 32 * np.eye(32)
    A[0, 0, 0] = np.nan
    A32 = A.astype(np.float32)
    _, _, resid = ps.eigtop(A32, 3)
    assert np.isnan(resid[0]), f"eigtop NaN: resid={resid[0]} не NaN"
    _, _, off = ps.eigjac(A32, max_sweeps=4)
    assert np.isnan(off[0]), f"eigjac NaN: offdiag={off[0]} не NaN"
    # r ≥ n → PS_ERR_ARG
    with pytest.raises(RuntimeError, match="PS_ERR_ARG"):
        ps.eigtop(rng.standard_normal((1, 8, 8)).astype(np.float32), 8)
    with pytest.raises(RuntimeError, match="PS_ERR_ARG"):
        ps.eigbot(rng.standard_normal((1, 8, 8)).astype(np.float32), 8)
    # local-бюджет: n=1024, r=8 ⇒ 66048 > 65536 → PS_ERR_LOCAL (m-12b)
    with pytest.raises(RuntimeError, match="PS_ERR_LOCAL"):
        ps.eigtop(rng.standard_normal((1, 1024, 1024)).astype(np.float32), 8)
    # eigjac: n > 256 → PS_ERR_ARG
    with pytest.raises(RuntimeError, match="PS_ERR_ARG"):
        ps.eigjac(rng.standard_normal((1, 300, 300)).astype(np.float32))
    # gemm: не-fp32 → PS_ERR_ARG
    with pytest.raises(RuntimeError, match="fp32"):
        ps.gemm(np.ones((8, 8)), np.ones((8, 8)))   # f64 по умолчанию
    # gemm: размеры вне ограничений → PS_ERR_ARG
    with pytest.raises(RuntimeError, match="PS_ERR_ARG"):
        ps.gemm(np.ones((1, 4097), dtype=np.float32), np.ones((4097, 8), dtype=np.float32))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
