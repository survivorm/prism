"""Selfcheck for prism/topology.py (TZ part 3, §7.6).

giotto/dionysus tests are skipped when the backend is not installed.
"""
import importlib.util

import numpy as np
import pytest

from prism.topology import (
    project_to_pca_subspace,
    project_to_subspace,
    zigzag_persistence,
)

HAS_GIOTTO = importlib.util.find_spec("gtda") is not None
HAS_DIONYSUS = importlib.util.find_spec("dionysus") is not None

requires_giotto = pytest.mark.skipif(not HAS_GIOTTO, reason="giotto-tda not installed")
requires_dionysus = pytest.mark.skipif(not HAS_DIONYSUS, reason="dionysus not installed")


# --- 7.6.1: rows basis ----------------------------------------------------------

def test_project_rows():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(10, 8))
    basis = np.eye(8, dtype=np.float32)[:3]
    h_proj = project_to_subspace(h, basis)
    assert h_proj.shape == (10, 3)
    assert np.allclose(h_proj, h[:, :3], atol=1e-6)


# --- 7.6.2: cols basis ------------------------------------------------------------

def test_project_cols():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(10, 8))
    basis = np.eye(8, dtype=np.float32)[:, :3]  # [8, 3]: columns
    h_proj = project_to_subspace(h, basis)
    assert h_proj.shape == (10, 3)
    assert np.allclose(h_proj, h[:, :3], atol=1e-6)


# --- 7.6.3: 1-D h -------------------------------------------------------------------

def test_project_1d_h():
    rng = np.random.default_rng(0)
    h = rng.normal(size=8)
    basis = np.eye(8, dtype=np.float32)[:3]
    h_proj = project_to_subspace(h, basis)
    assert h_proj.shape == (3,)
    assert np.allclose(h_proj, h[:3], atol=1e-6)


# --- 7.6.4: ambiguous square basis -----------------------------------------------------

def test_project_ambiguous():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(8, 8))
    with pytest.raises(ValueError, match="square"):
        project_to_subspace(h, np.eye(8, dtype=np.float32))


# --- 7.6.5: invalid basis ----------------------------------------------------------------

def test_project_invalid_basis():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(10, 8))
    with pytest.raises(ValueError, match="2-D"):
        project_to_subspace(h, rng.normal(size=8))
    with pytest.raises(ValueError, match="incompatible"):
        project_to_subspace(h, rng.normal(size=(3, 5)))


# --- 7.6.6: PCA shapes + orthonormal components ------------------------------------------

def test_pca_shapes():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(100, 20))
    h_proj, comps = project_to_pca_subspace(h, n_components=5)
    assert h_proj.shape == (100, 5)
    assert comps.shape == (5, 20)
    assert np.allclose(comps @ comps.T, np.eye(5), atol=1e-5)


# --- 7.6.7: PCA captures variance -----------------------------------------------------------

def test_pca_variance():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(200, 20))
    h_proj, _ = project_to_pca_subspace(h, n_components=5)
    frac = h_proj.var(axis=0).sum() / h.var(axis=0).sum()
    assert 0.15 < frac < 0.6  # 5/20 components of iid data: ~0.25 ± noise


# --- 7.6.8: n_components too large -------------------------------------------------------------

def test_pca_too_many_components():
    with pytest.raises(ValueError, match="n_components"):
        project_to_pca_subspace(np.ones((10, 4)), n_components=5)


# --- 7.6.9: N < 2 --------------------------------------------------------------------------------

def test_pca_n_too_small():
    with pytest.raises(ValueError, match="N = 1 < 2"):
        project_to_pca_subspace(np.ones((1, 4)))


# --- 7.6.10-12: zigzag under giotto ---------------------------------------------------------------

@requires_giotto
def test_zigzag_circle():
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    X = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    res = zigzag_persistence([X, X, X], maxdim=1, backend="giotto")
    assert res["n_layers"] == 3
    assert res["n_points"] == 30
    assert len(res["dgms"]) == 2
    assert res["dgms"][1].shape[0] >= 1  # H1: the circle


@requires_giotto
def test_zigzag_trivial():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2)).astype(np.float32)
    res = zigzag_persistence([X, X, X], maxdim=1, backend="giotto")
    # H1 of 30 iid points in R2: none (a point cloud is contractible)
    assert res["dgms"][1].shape[0] == 0


@requires_giotto
def test_zigzag_two_clouds():
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=(30, 2)).astype(np.float32)
    X2 = (X1 + 0.1).astype(np.float32)
    res = zigzag_persistence([X1, X2], maxdim=1, backend="giotto")
    assert res["n_layers"] == 2
    for dgm in res["dgms"]:
            assert dgm.shape[1] == 2


# --- 7.6.13-14: validation (backend-independent) -----------------------------------------------------

def test_zigzag_n_points_too_small():
    X = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="n_points"):
        zigzag_persistence([X, X], backend="giotto")


def test_zigzag_maxdim():
    X = np.zeros((6, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="maxdim"):
        zigzag_persistence([X, X], maxdim=2, backend="giotto")


def test_zigzag_bad_metric():
    X = np.zeros((6, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="metric"):
        zigzag_persistence([X, X], metric="manhattan", backend="giotto")


def test_zigzag_nan():
    X = np.zeros((6, 2), dtype=np.float32)
    X[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        zigzag_persistence([X, X], backend="giotto")


def test_zigzag_mixed_n_points():
    X1 = np.zeros((6, 2), dtype=np.float32)
    X2 = np.zeros((7, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="n = 7"):
        zigzag_persistence([X1, X2], backend="giotto")


def test_zigzag_single_cloud():
    X = np.zeros((6, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=">= 2"):
        zigzag_persistence([X], backend="giotto")


# --- 7.6.15: missing backend -> ImportError with pip instruction -------------------------------------

def test_zigzag_missing_backend():
    if HAS_GIOTTO:
        pytest.skip("giotto installed; cannot test the missing-backend path")
    X = np.zeros((6, 2), dtype=np.float32)
    with pytest.raises(ImportError, match="pip install giotto-tda"):
        zigzag_persistence([X, X], backend="giotto")


# --- 7.6.16: dionysus path (skipped unless installed) -----------------------------------------------------

@requires_dionysus
def test_zigzag_dionysus():
    theta = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    X = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    res = zigzag_persistence([X, X, X], maxdim=1, backend="dionysus")
    assert len(res["dgms"]) == 2
    assert res["dgms"][1].shape[0] >= 1


# --- module hygiene: importable without backends -----------------------------------------------------------

def test_import_without_backends():
    import prism.topology  # noqa: F401  (works even if gtda/dionysus absent)
