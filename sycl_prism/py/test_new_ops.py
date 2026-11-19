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
  - Gagg-композиция: подпространства ≤ 1e-3, S5b-аналог (норм. ‖G‖·‖v‖) ≤ 1e-4 (f64, хост);
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
    """Сверка с f64-эталоном. Эталон — ПОСТРОЧНЫМИ БЛОКАМИ колонок (f64 GEMM
    column-parallel: точка-по-точечное произведение по K идентично полному;
    память ~M*K*8 + K*BLK*8 + M*BLK*8 вместо ~M*N*8 — TZ n-17: wide-формы
    не влезают в память при целом Cref)."""
    C = ps.gemm(A, B)
    A64 = A.astype(np.float64)
    BLK = 8192
    err_max = 0.0
    ref_max = 0.0
    for c0 in range(0, B.shape[1], BLK):
        Cref_b = A64 @ B[:, c0:c0 + BLK].astype(np.float64)
        err_max = max(err_max, float(np.max(np.abs(C[:, c0:c0 + BLK] - Cref_b))))
        ref_max = max(ref_max, float(np.max(np.abs(Cref_b))))
        del Cref_b
        gc.collect()
    rel = err_max / ref_max
    print(f"[gemm {tag}] {A.shape} @ {B.shape}: max|Δ|/max|C| = {rel:.3e}")
    assert rel <= 1e-4, f"gemm {tag}: rel={rel} > 1e-4"
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
        # кластерный λ_i — если сосед (по спектру) в пределах 10% от |λ_i|
        # (TZ §2.7; порог 10% — замер: пара L23 d1t0 λ[46]/λ[47] = 1.15e-5/
        # 1.22e-5 (раздел 5.9%) даёт f32-пол 1.12e-3 rel независимо от tol
        # (почти-вырожденная пара: абсолютное возмущение ~1e-9 ⇒ относительное
        # ~1e-3) — 5%-порог ложно классифицировал её «рассеянной» → fail по
        # 1e-3. Баг классификации код-аудита r2: было |λ_j|/|λ_i| < 0.05 —
        # отношение амплитуд, а не расстояние по спектру (кластеры вне нуля
        # классифицировались как рассеянные).
        worst = 0.0
        for i in range(len(lam_ref)):
            j = np.argsort(np.abs(lam_ref - lam_ref[i]))[1]
            clustered = abs(lam_ref[j] - lam_ref[i]) / max(abs(lam_ref[i]), 1e-300) < 0.10
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
    # k = число sweep'ов до стопа: первый ИНДЕКС i с curve[i] < tol — это
    # offdiag после sweep i+1 ⇒ стоп после sweep k = i+1 (curve[k-1] == off[0],
    # fill [k..64)). (Оффс-один код-аудита r1.2: было k = i — проверяло
    # curve[i-1] против финала.)
    k = int(np.argmax(curve < tol)) + 1 if (curve < tol).any() else 64
    # offdiag[0] — финальное; curve[k-1] == offdiag[0]; curve[k..64) == offdiag[0]
    assert abs(curve[k - 1] - off[0]) < 1e-15, \
        f"eigjac curve: curve[{k-1}]={curve[k-1]} != offdiag[0]={off[0]}"
    assert np.all(curve[k:] == off[0]), \
        f"eigjac curve: fill [k..64) не финальным значением: {curve[k:k+4]}"
    # без curve — 3-тupl, как прежде
    r0 = ps.eigjac(A, max_sweeps=64, tol=tol)
    assert r0[0].shape == (1, 128) and r0[1].shape == (1, 128, 128)
    # регрессия (код-аудит r1 + ревью r2 C5/C6): fill [k..size) = offdiag[0]
    # (НЕ мусор/не перепись) в ОБАХ режимах стопа:
    # (1) стоп по tol: A*50 с tol=1e-6 сходится ровно на sweep 8 — k=8,
    #     curve[0..7] per-sweep, curve[8..64) fill;
    # (2) ИСТИННЫЙ стоп по max_sweeps (stop_sw=0, cpp:1559-1560): max_sweeps=4 —
    #     off после 4 sweep'ов ≈ 3.7e-4 >> tol=1e-6 ⇒ стоп по max_sweeps, fill
    #     [4..64). Именно эта комбинация (curve != nullptr + stop_sw==0) рождала
    #     мусорный хвост в HEAD (нет fill вовсе) — designed-ассерт off >= tol
    #     делает тест немёртвым (если матрица сойдёт быстрее — громкий fail).
    A_big_off = (A[0] * 50.0)[None, :, :].astype(np.float32)
    # (1) стоп по tol (sweep 8 == max_sweeps)
    curve2 = np.full(64, -1.0)
    vals2, vecs2, off2 = ps.eigjac(A_big_off, max_sweeps=8, tol=1e-6, curve=curve2)
    assert np.all(curve2[8:] == off2[0]), \
        f"curve fill [8..64) не финальным: {curve2[8:12]} vs off={off2[0]}"
    assert curve2[0] > 0 and curve2[7] >= curve2[8] - 1e-15
    # код-ревью r1 C2 (MINOR): тест обязан видеть баг «fill [0..curve_len) = off[0]»
    # (перепись per-sweep позиций финальным значением): off — ОТНОСИТЕЛЬНОЕ
    # ‖offdiag‖_F/‖A‖_F (cpp:1444); после sweep 1 off ≈ O(0.1), после sweep 8 —
    # O(1e-7) ⇒ curve[0] СТРОГО больше финального. При переписи curve2[0]==off2[0]
    # → fail.
    assert curve2[0] > off2[0], \
        (f"curve-регрессия (1): curve[0]={curve2[0]} <= offdiag[0]={off2[0]} — per-sweep "
         f"позиции [0..8) затёрты финальным значением (fill-баг) или сходимость "
         f"за 1 sweep (нереально для A*50)")
    # (2) истинный стоп по max_sweeps: max_sweeps=4 — off НЕ ниже tol (≈3.7e-4)
    curve3 = np.full(64, -1.0)
    vals3, vecs3, off3 = ps.eigjac(A_big_off, max_sweeps=4, tol=1e-6, curve=curve3)
    assert off3[0] >= 1e-6, \
        (f"curve-регрессия (2): off={off3[0]} < tol=1e-6 — матрица сошлась быстрее "
         f"max_sweeps=4, designed-сценарий «стоп по max_sweeps» не реализован: "
         f"усилить A (или поднять max_sweeps), тест немёртвый")
    assert np.all(curve3[4:] == off3[0]), \
        f"curve fill [4..64) (стоп по max_sweeps) не финальным: {curve3[4:8]} vs off={off3[0]}"
    assert curve3[0] > off3[0], \
        (f"curve-регрессия (2): curve[0]={curve3[0]} <= offdiag[0]={off3[0]} — per-sweep "
         f"позиции [0..4) затёрты финальным значением (fill-баг в ветке stop_sw=0)")
    print(f"[eigjac curve] k={k}, curve[0]={curve[0]:.2e}, финал={off[0]:.2e}, "
          f"fill OK (стоп по tol: curve[8..64)=off; стоп по max_sweeps: curve[4..64)=off, "
          f"curve[0]>off в обоих)")


def test_eigjac_sweep_time_guard():
    """TZ §2.7 (v5.3 F1/F9, дельта 19.11 + перетюнинг 19.11-день): sweep-time guard —
    ТЮНИНГ-шаг (таблица (n, chunk)→t_sweep в артефакт) + твёрдые гейты: 1 launch
    (n=128, чанк C=64) ≤ 4с И 1 launch (n=256, чанк C/16=4) ≤ 4с (запас к hang-check
    ~20с) И C_measured ≥ C_FIXED (консервативная фиксация, v5.4: iGPU-состояние
    нестабильно — деградация/восстановление 1.5–1.9× зафиксированы 19.11; C_FIXED
    выбрано под ХУЖОЕ из измеренных состояний. GPU быстрее — C_measured > C_FIXED —
    остаёмся на C_FIXED (стоимость на матрицу при c64/c256 идентична в замерах,
    14.8/15.0 мс); GPU медленнее — C_measured < C_FIXED — громкий fail, C_FIXED
    больше не безопасно). Перетюнинг: состояние
    iGPU 19.11-день тунинга — t_sweep(n=128, c256) ≈ 4.3–6.9с > 4с (мин. 4.32) —
    C=256 недостижим; c64 ≈ 1.0–1.9с ≤ 4с; n=256: c1 ≈ 2.4с (фикс. оверхед),
    c4 ≈ 2.7–3.4с ≤ 4с (замеры: /tmp/AT0_3_tsweep_reps.json). 19.11-день
    восстановления: c256 = 3.843с ≤ 4с → C_measured=256 — консервативно остаёмся на 64.

    ТЮНИНГ-луп — hang-check-гард: следующий (больший) чанк НЕ запускается, если
    монотонная оценка 4·t(prev)−0.3с > 4с — больший медленнее по определению
    (C выше не поднимется) и ближе к hang-окну драйвера: замер 19.11 —
    launch chunk=1024 (t≈11с+) — драйвер убил compute-команду
    (UR_RESULT_ERROR_DEVICE_LOST, error 20), состояние устройства ломается на
    весь процесс (все последующие malloc'и падают)."""
    _skip_nogpu()
    CHUNKS = (1, 16, 64, 256, 1024, 4096)
    C_FIXED = 64  # = EIGJAC_CHUNK в src/prism_sycl.cpp (перетюнинг 19.11-день: c256 ≥ 4.3с > 4с)
    rng = np.random.default_rng(51)
    table = {}
    C_measured = 1
    prev_med = None
    prev_chunk = None
    for chunk in CHUNKS:
        if prev_med is not None and 4.0 * prev_med - 0.3 > 4.0:
            print(f"[eigjac tuning] n=128 chunk={chunk:>5} — ПРОПУСК: монотонность "
                  f"(4·t(c{prev_chunk}) − 0.3с = {4.0 * prev_med - 0.3:.2f}с > 4с — C выше не "
                  f"поднимется; запуск — близко к hang-окну драйвера (DEVICE_LOST 19.11))")
            break
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
        if med <= 4.0:
            C_measured = chunk
        else:
            break   # C = предыдущий чанк (наибольшее с t ≤ 4с)
        prev_med = med
        prev_chunk = chunk
        del A, M
        gc.collect()
    # F9 (дельта 19.11 + перетюнинг): n=256, чанк C/16 (замер 19.11-день:
    # n=256 c1 ≈ 2.4с — фикс. оверхед, c4 ≈ 2.7–3.4с ≤ 4с; /tmp/AT0_3_tsweep_reps.json)
    C256 = C_FIXED // 16
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
    # твёрдые гейты (TZ §2.7 v5.3 + дельта 19.11):
    assert "n128_c64" in table, \
        "eigjac F1: t_sweep(n=128, чанк 64) > 4с ещё до замера (монотонный пропуск) — " \
        "GPU медленнее расчёта (C < 64 недостижим) — громкий fail, не тихое замедление"
    assert table["n128_c64"] <= 4.0, \
        f"eigjac F1: t_sweep(n=128, чанк {C_FIXED})={table['n128_c64']:.2f}с > 4с — " \
        f"hang-check-риск; понизить EIGJAC_CHUNK и пересобрать"
    assert table[f"n256_c{C256}"] <= 4.0, \
        f"eigjac F9: t_sweep(n=256, чанк {C256})={med256:.2f}с > 4с — " \
        f"hang-check-риск для n=256; понизить EIGJAC_CHUNK и пересобрать"
    assert C_measured >= C_FIXED, \
        f"eigjac F1: тюнинг дал C={C_measured} < EIGJAC_CHUNK={C_FIXED} — GPU медленнее " \
        f"худшего зафиксированного состояния: C_FIXED больше не безопасно, " \
        f"понизить константу и пересобрать (громкий fail, не тихое замедление)"
    if C_measured > C_FIXED:
        print(f"[eigjac F1] консервативная фиксация: тюнинг дал C={C_measured} (GPU быстрее), "
              f"остаёмся на EIGJAC_CHUNK={C_FIXED} (v5.4)")
    print(f"[eigjac F1/F9] гейты пройдены: n=128 c{C_FIXED}="
          f"{table['n128_c64']:.3f}с; n=256 c{C256}={med256:.3f}с — все ≤ 4с")


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


def test_eigjac_zero_and_curve_size():
    """Код-ревью r1 C3 (NIT) — designed-тесты guard'ов без покрытия:
    (a) нулевая матрица — tot = 0: sweep 1 даёт off = 0 (offdiag — относительное
        sqrt(off2/tot) с guard (tot > 0.f), cpp:1444; при нулевой матрице guard
        срабатывает ⇒ ровно 0.0, NaN невозможно) < tol ⇒ стоп после 1 sweep,
        vals = 0, vecs — базис;
    (b) curve.size < max_sweeps → ValueError на python-стороне (до C)."""
    _skip_nogpu()
    A = np.zeros((1, 16, 16), dtype=np.float32)
    vals, vecs, off = ps.eigjac(A, max_sweeps=16, tol=1e-5)
    assert not np.any(np.isnan(vals)) and not np.any(np.isnan(vecs)), \
        "eigjac zero: NaN в выходe"
    assert off[0] == 0.0, f"eigjac zero: off={off[0]} (ожидалось 0)"
    assert np.allclose(vals[0], 0.0, atol=0.0), f"eigjac zero: vals={vals[0]}"
    assert np.allclose(vecs[0], np.eye(16), atol=1e-6), \
        f"eigjac zero: vecs не базис: ortho={np.linalg.norm(vecs[0].T @ vecs[0] - np.eye(16))}"
    with pytest.raises(ValueError, match="curve"):
        ps.eigjac(A, max_sweeps=16, tol=1e-5, curve=np.empty(3))
    print("[eigjac zero/curve-size] (a) zero-matrix off=0 без NaN + (b) ValueError: OK")


def test_eigjac_convergence_jcache():
    """TZ §2.7 m-28 (v5.2): замер сходимости — ОДИН прогон max_sweeps=16 с
    буфером offdiag_curve (§2.3), 2 реальных jcache (L6 d0t0, L23 d0t0) — в
    артефакт (валидирует оценку §2.3: 4–6 sweep'ов до 1e-5); assert:
    sweeps_to(1e-5) ≤ 16 на обоих (сходится в бюджете, не «< 1e-4 на 64»)."""
    _skip_nogpu()
    res = {}
    for L in (6, 23):
        dc = _jcache(L)
        JRt, Pt = dc["JR"], dc["P128"]
        g64 = _gram64(JRt[(DOMAIN, 0, 0)], Pt[(DOMAIN, 0, 0)])
        g32 = g64.astype(np.float32)[None, :, :]
        curve = np.empty(16)
        t0 = time.time()
        _, _, off = ps.eigjac(g32, max_sweeps=16, tol=1e-5, curve=curve)
        wall = time.time() - t0
        conv = curve < 1e-5
        sweeps_to = int(np.argmax(conv)) + 1 if conv.any() else None
        assert sweeps_to is not None and sweeps_to <= 16, \
            (f"m-28: L{L} d0t0: сходимости до 1e-5 за 16 sweep'ов нет "
             f"(curve: {' '.join(f'{v:.1e}' for v in curve[:8])} ...)")
        res[f"L{L}_d0t0"] = {
            "sweeps_to_1e-5": sweeps_to,
            "wall_s": round(wall, 3),
            "offdiag_final": float(off[0]),
            "curve_per_sweep": [float(v) for v in curve[:sweeps_to]],
        }
        print(f"[eigjac m-28 L{L} d0t0] sweeps_to(1e-5)={sweeps_to} ≤ 16; "
              f"off_final={off[0]:.2e}; wall={wall:.2f}s; "
              f"curve: {' '.join(f'{v:.1e}' for v in curve[:sweeps_to])}")
    _jcache_cache.clear()
    gc.collect()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eigjac_convergence.json").write_text(json.dumps(res, indent=2))
    # TZ §2.3 оценка: 4–6 sweep'ов до 1e-5 на всех 6 реальных jcache —
    # артефакт валидирует её (отчёт; твёрдый assert — только ≤ 16 по m-28)


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
    Gnrm = float(np.linalg.norm(Gagg))   # для масштабно-инвариантного S5b-остатка
    nz = np.where(lam_np > 1e-12 * lammax)[0]
    top_idx = order[:R]                               # сильнейшие (убыв.)
    weak_idx = nz[:min(R, len(nz))]                   # слабейшие НЕ-нулевые
    weakmid_idx = order[R:2 * R]                      # позиции r..2r (C1c)
    # sycl-путь: Gsmall + eigjac + lift
    Gsmall = ps.gemm(Jagg32, np.ascontiguousarray(Jagg32.T))  # [128,128] f32 (Jagg32.T — view, gemm требует C-contiguous)
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
            assert d[j] < 1e-3, f"gagg L{L} {name}: λ не совпал (rel={d[j]:.2e})"
            u_cols.append(U_s[:, j])
        u_sel = np.stack(u_cols, axis=1)
        # lift v = Jaggᵀ u / √λ (f64, на хосте — аналог _lift_arr AT2c)
        sig = np.sqrt(np.maximum(lam_sel, 0.0))
        sig = np.where(sig <= 1e-12 * lammax, 0.0, sig)
        v = (Jagg64.T @ u_sel) / sig
        vref = w_np[:, idx]
        d_sp = _subspace_dist(v, vref)
        assert d_sp <= 1e-3, f"gagg L{L} {name}: subdist={d_sp}"
        # S5b-аналог (TZ §2.7): макс. j ‖Gagg v_j − λ_j v_j‖/(‖G‖·‖v_j‖) (f64, хост) —
        # масштабно-инвариантный собственный остаток. ВАЖНО: нормировка /|λ_j| здесь
        # НЕГОДНА (замер 19.11): квант f32-данных Jagg32 доминирует на слабых λ —
        # эталон «f64-математика на f32-данных» (eig точный) уже даёт s5b(|λ|-норм.) =
        # 1.35e-2 на L6 weak (λ_мин ~ 0.4, λ_max ~ 2.8e3), т.е. раздувание 1/λ^1.5 —
        # свойства данных, не ядра. Геометрический контракт — subdist (≤ 1e-3).
        s5b = 0.0
        for j in range(len(idx)):
            rj = (np.linalg.norm(Gagg @ v[:, j] - lam_sel[j] * v[:, j])
                  / (Gnrm * np.linalg.norm(v[:, j])))
            s5b = max(s5b, float(rj))
        assert s5b <= 1e-4, f"gagg L{L} {name}: S5b_rel={s5b} > 1e-4"
        res[name] = {"subdist": d_sp, "s5b_rel_max": s5b}
        print(f"[gagg L{L} {name}] subdist={d_sp:.2e} s5b_rel={s5b:.2e}")
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
    """TZ §2.7: random-SPD, РАССЕЯННЫЙ спектр (relλ ≤ 1e-3, subdist ≤ 1e-3).
    Рассеянность по конструкту: крайние 8 собственных значений — геометрическая
    прогрессия с ratio 0.8. Выведение: сходимость (λ_{r+1}/λ_r)^max_iter ≤ 1e-3
    ⇔ ratio ≤ 0.987; случайное wishart-спектр (M@Mᵀ+128I) даёт ratio 0.987–0.994
    для top-7 — 512 итераций недостаточно ДАЖЕ В F64 (замер: subdist 2.6e-3…
    8.1e-3). Случайный ортогональный Q (QR гаусса) × фиксированные λ ⇒ случайная
    SPD-матрица с отделёнными крайними. Спектры сравниваются ПОСЛЕ сортировки:
    контракт TZ §2.2 — λ_j = Rayleigh колонки j (порядок колонок = динамика MGS,
    не гарантируется)."""
    _skip_nogpu()
    rng = np.random.default_rng(31)
    G = rng.standard_normal((6, 128, 128))
    Q, _ = np.linalg.qr(G)                            # случайные ортогональные
    top = 128.0 * 0.8 ** np.arange(8)                # top-8 (убыв.): 128…26.8
    mid = 20.0 * (18.0 / 20.0) ** (np.arange(112) / 111.0)  # 112 средних: 20…18
    bot = 16.0 * 0.8 ** np.arange(8)[::-1]           # bottom-8 (убыв.): 16…3.03
    lam = np.concatenate([top, mid, bot])            # 128 убывающих, [3.03, 128]
    A = (Q * lam[None, :, None]) @ Q.transpose(0, 2, 1)   # f64, случайная SPD
    A32 = A.astype(np.float32)
    lam_ref, vecs_ref = np.linalg.eigh(A)
    vals, vecs, resid = ps.eigtop(A32, R, max_iter=512)
    for b in range(6):
        rel = _rel_eig(np.sort(vals[b]), np.sort(lam_ref[b, -R:]))
        assert rel <= 1e-3, f"eigtop random batch {b}: relλ={rel}"
        d = _subspace_dist(vecs[b], vecs_ref[b, :, -R:])
        assert d <= 1e-3, f"eigtop random batch {b}: subdist={d}"
        ortho = float(np.linalg.norm(vecs[b].T @ vecs[b] - np.eye(R)))
        assert ortho <= 1e-5, f"eigtop random batch {b}: ortho={ortho} (n-14)"
    # eigbot — ТОЛЬКО на full-rank SPD
    vals_b, vecs_b, resid_b = ps.eigbot(A32, R, max_iter=512)
    for b in range(6):
        rel = _rel_eig(np.sort(vals_b[b]), np.sort(lam_ref[b, :R]))
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
    # local-бюджет: n=1024, r=8 ⇒ 2*n*r*4 + (WG+1)*4 = 65796 > 65536 → PS_ERR_LOCAL
    # (m-12b; m3 r4: формула = фактический состав ядра, члена r²*4 в ядре нет)
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


def test_svd_top_output_contract():
    """M1/M2 (само-аудит r4): выходы БЕЗ отрицательных шагов (потребитель
    Backend.svd_top → torch.from_numpy не принимает отриц. шаги; канарейка в
    tests/test_backend.py::test_sycl_svd_top была КРАСНОЙ) + лимит min(m,n)<=256.

    Отрицательные шаги ловятся по .strides, не по C_CONTIGUOUS: у вырожденных
    осей (shape (1,1), (1,r)) numpy считает view «contiguous» и
    ascontiguousarray не копирует ⇒ флаговая проверка баг ПРОПУСКАЕТ.
    """
    _skip_nogpu()
    rng = np.random.default_rng(101)
    cases = (((8, 12), 4), ((12, 8), 4), ((1, 1), 1), ((1, 6), 1), ((6, 1), 1),
             ((16, 16), 16), ((16, 16), 3))
    for shape, r in cases:
        A = rng.standard_normal(shape).astype(np.float32)
        U, S, Vh = ps.svd_top(A, r)
        for nm, arr in (("U", U), ("S", S), ("Vh", Vh)):
            assert all(s >= 0 for s in arr.strides), \
                f"svd_top {shape} r={r}: {nm} отрицательные шаги {arr.strides}"
            assert arr.flags["C_CONTIGUOUS"], \
                f"svd_top {shape} r={r}: {nm} не C-contiguous ({arr.strides})"
        # документированный потребитель: без копии
        torch.from_numpy(U); torch.from_numpy(S); torch.from_numpy(Vh)
        Sr = np.linalg.svd(A.astype(np.float64), full_matrices=False)[1][:r]
        rel = float(np.abs(S - Sr).max() / max(Sr.max(), 1e-30))
        assert rel <= 1e-3, f"svd_top {shape} r={r}: σ rel={rel}"
        assert np.all(np.diff(S) <= 0.0), f"svd_top {shape} r={r}: S не descending: {S}"
        if r == min(shape):   # полный набор ⇒ реконструкция обязана сойтись
            recon = float(np.abs(A - (U @ np.diag(S) @ Vh)).max() / np.abs(A).max())
            assert recon <= 1e-3, f"svd_top {shape}: recon rel={recon}"
    # M2: граница по min(m,n): 256 — работает; 257 — PS_ERR_ARG НА УРОВНЕ svd_top.
    # Важно: большая сторона >256 ДОПУСТИМА (257x40: малая 40 ⇒ eigjac(n=40) OK).
    U, S, Vh = ps.svd_top(rng.standard_normal((256, 40)).astype(np.float32), 3)
    assert S.shape == (3,) and np.all(S > 0)
    U, S, Vh = ps.svd_top(rng.standard_normal((257, 40)).astype(np.float32), 3)
    assert S.shape == (3,), "svd_top 257x40: законен (малая сторона 40 <= 256)"
    with pytest.raises(RuntimeError, match=r"svd_top.*min\(m,n\)=257"):
        ps.svd_top(rng.standard_normal((257, 300)).astype(np.float32), 3)


def test_eigjac_curve_onplace_contract():
    """m1 (само-аудит r4): curve заполняется НА МЕСТЕ ⇒ только ndarray(f64).
    f32-буфер/список раньше молча заполняли КОПИЮ (no-op для пользователя)."""
    _skip_nogpu()
    rng = np.random.default_rng(202)
    A = rng.standard_normal((2, 16, 16)).astype(np.float32)
    A = (A + np.transpose(A, (0, 2, 1))) / 2
    for bad in (np.zeros(16, dtype=np.float32), [0.0] * 16):
        with pytest.raises(ValueError, match=r"ndarray\(float64\)"):
            ps.eigjac(A, max_sweeps=16, tol=1e-6, curve=bad)
    c = np.zeros(16, dtype=np.float64)
    ps.eigjac(A, max_sweeps=16, tol=1e-6, curve=c)
    assert c[0] > 0.0, "eigjac curve: буфер не заполнен (контракт «на месте»)"


def test_guards_small_n_and_mem():
    """n6/n7/m2 (само-аудит r4): явные ошибки вместо тихого NaN (band при K<9)
    и вместо поз-днего молчаливого OOM (комбинированный объём буферов)."""
    _skip_nogpu()
    rng = np.random.default_rng(303)
    # n6: band = mean(лаги 9..K) не определён — БЫЛ band=[nan nan] + RuntimeWarning
    with pytest.raises(ValueError, match=r"K=3.*< 9"):
        ps.permutation_null(rng.standard_normal((4, 2)).astype(np.float32),
                            0.1, [1], blk=1, K=96)
    # n7: тот же тип/смысл ошибки, что в reference (был RuntimeError из C)
    with pytest.raises(ValueError, match="вне диапазона"):
        ps.acf_moments(np.zeros((5, 2), dtype=np.float32), 8)
    # m2: guard комбинированного объёма (per-dim лимиты его не ловят)
    budget = ps._mem_budget_bytes()
    assert budget > 0, "бюджет памяти не определён (device_info)"
    with pytest.raises(RuntimeError, match="PS_ERR_MEM"):
        ps._check_mem("gemm", budget + 1, "тест")
    ps._check_mem("gemm", budget // 4, "тест")       # в бюджете — молча


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
