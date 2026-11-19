"""tests/test_backend.py — core-диспетчер prism.backend (TZ_sycl_new_ops §2.6).

cpu-режим: == torch (бит-в-бит на мелких) — всегда (нужен только torch).
sycl-режим: == torch ≤ 1e-3 — ТОЛЬКО если DEVICE_OK (иначе skip).

Импорт prism/backend.py — по файлу (importlib), НЕ `import prism`
(тяжёлый __init__).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _load_backend():
    spec = importlib.util.spec_from_file_location(
        "prism_backend_test", str(ROOT / "prism" / "backend.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pb = _load_backend()


def _device_ok() -> bool:
    """DEVICE_OK: sycl-библиотека найдена И устройство видно."""
    try:
        sys.path.insert(0, str(ROOT / "sycl_prism" / "py"))
        import prism_sycl as ps
    except Exception:
        return False
    try:
        return ps.available() and not ps._is_reference()
    except Exception:
        return False


torch.manual_seed(0)
A2 = torch.randn(64, 128)
B2 = torch.randn(128, 256)
A3 = torch.randn(4, 64, 128)
M = torch.randn(32, 32)
M = M @ M.T + 32 * torch.eye(32)
M3 = torch.stack([M, M + torch.eye(32)])
S = torch.randn(128, 256)

# =====================================================================
# CPU-режим: бит-в-бит с torch
# =====================================================================

@pytest.fixture(scope="module")
def cpu_b():
    return pb.Backend("cpu")


def test_cpu_name(cpu_b):
    assert cpu_b.name == "cpu"


def test_cpu_matmul_2d_bitexact(cpu_b):
    assert torch.equal(cpu_b.matmul(A2, B2), A2 @ B2)


def test_cpu_matmul_3d_bitexact(cpu_b):
    assert torch.equal(cpu_b.matmul(A3, B2), A3 @ B2)


def test_cpu_eigh_2d_bitexact(cpu_b):
    w, V = cpu_b.eigh(M)
    wref, Vref = torch.linalg.eigh(M)
    assert torch.equal(w, wref) and torch.equal(V, Vref)


def test_cpu_eigh_3d_bitexact(cpu_b):
    w, V = cpu_b.eigh(M3)
    wref, Vref = torch.linalg.eigh(M3)
    assert torch.equal(w, wref) and torch.equal(V, Vref)


def test_cpu_svd_top_bitexact(cpu_b):
    U, Ss, Vh = cpu_b.svd_top(S, 7)
    Uref, Sref, Vhref = torch.linalg.svd(S, full_matrices=False)
    assert torch.equal(U, Uref[:, :7])
    assert torch.equal(Ss, Sref[:7])
    assert torch.equal(Vh, Vhref[:7, :])


def test_cpu_errors():
    b = pb.Backend("cpu")
    with pytest.raises(pb.BackendError):
        b.matmul(A2, A2)              # K не сошёлся (128 != 64)
    with pytest.raises(pb.BackendError):
        b.eigh(torch.randn(32))       # не [n,n]
    with pytest.raises(pb.BackendError):
        b.svd_top(S, 300)             # r > min(m,n)
    with pytest.raises(pb.BackendError):
        pb.Backend("cuda")            # неизвестный режим


# =====================================================================
# SYCL-режим: <= 1e-3 к torch (только DEVICE_OK)
# =====================================================================

@pytest.fixture(scope="module")
def sycl_b():
    if not _device_ok():
        pytest.skip("DEVICE_OK: sycl-устройство/библиотека недоступны")
    return pb.Backend("sycl")


def test_sycl_matmul_2d(sycl_b):
    C = sycl_b.matmul(A2, B2)
    ref = A2 @ B2
    assert C.dtype == torch.float32
    rel = float((C - ref).abs().max() / ref.abs().max())
    assert rel <= 1e-3, f"matmul 2D rel={rel}"


def test_sycl_matmul_3d(sycl_b):
    C = sycl_b.matmul(A3, B2)
    ref = A3 @ B2
    assert C.shape == ref.shape
    rel = float((C - ref).abs().max() / ref.abs().max())
    assert rel <= 1e-3, f"matmul 3D rel={rel}"


def test_sycl_eigh_2d(sycl_b):
    w, V = sycl_b.eigh(M)
    wref, Vref = torch.linalg.eigh(M)
    assert w.dtype == torch.float64   # задокументированное отличие (TZ §2.6)
    rel = float((w - wref).abs().max() / wref.abs().max())
    assert rel <= 1e-3, f"eigh rel={rel}"
    # подпространства: ортогональность
    ortho = float((V.T @ V - torch.eye(32)).norm())
    assert ortho <= 1e-4, f"eigh ortho={ortho}"


def test_sycl_eigh_3d(sycl_b):
    w, V = sycl_b.eigh(M3)
    wref, _ = torch.linalg.eigh(M3)
    rel = float((w - wref).abs().max() / wref.abs().max())
    assert rel <= 1e-3, f"eigh 3D rel={rel}"


def test_sycl_svd_top(sycl_b):
    U, Ss, Vh = sycl_b.svd_top(S, 7)
    Uref, Sref, Vhref = torch.linalg.svd(S, full_matrices=False)
    rel = float((Ss - Sref[:7]).abs().max() / Sref[:7].abs().max())
    assert rel <= 1e-3, f"svd S rel={rel}"
    # подпространство U
    uu = U.T @ U
    u_orth = float((uu - torch.eye(7)).norm())
    assert u_orth <= 1e-3, f"svd U ortho={u_orth}"
    # Vh·Vhᵀ ≈ I (7)
    vh_orth = float((Vh @ Vh.T - torch.eye(7)).norm())
    assert vh_orth <= 1e-3, f"svd Vh ortho={vh_orth}"


def test_sycl_eigh_n_too_big(sycl_b):
    with pytest.raises(pb.BackendError):
        sycl_b.eigh(torch.eye(300))   # n > 256 → ошибка с подсказкой (не fallback)
