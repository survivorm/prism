"""Tests for prism/cache.py (docs/new_code_1.md §2.6 selfcheck)."""

import math
import os

import pytest
import torch

from prism.cache import (
    SVDCache,
    MetricCache,
    bootstrap_ci,
    bootstrap_ci_paired,
    deterministic_mode,
    iter_position_pairs,
)


def _make_svd(K=8, D=16, r=4, dtype=torch.float32):
    """Random orthonormal (U, S, V) triple satisfying all put() invariants."""
    U0 = torch.linalg.qr(torch.randn(K, D)).Q[:, :r]
    S0 = torch.rand(r, dtype=dtype).sort(descending=True).values + 0.5
    V0 = torch.linalg.qr(torch.randn(r, D)).Q
    return U0.to(dtype), S0, V0.to(dtype)


# ---------------------------------------------------------------------------
# SVDCache
# ---------------------------------------------------------------------------


def test_svd_roundtrip(tmp_path):
    """TZ §2.6.1: put → get → allclose for U, S, V."""
    c = SVDCache(str(tmp_path), r=4)
    U, S, V = _make_svd()
    c.put("base", 6, 0, 0, U, S, V)
    U2, S2, V2 = c.get("base", 6, 0, 0)
    assert torch.allclose(U, U2) and torch.allclose(S, S2) and torch.allclose(V, V2)
    assert c.has("base", 6, 0, 0)
    assert c.get("base", 6, 1, 0) is None


def test_svd_key_collision(tmp_path):
    """TZ §2.6.2: same (edit, layer, doc) at different pos → separate files."""
    c = SVDCache(str(tmp_path), r=4)
    U, S, V = _make_svd()
    c.put("base", 6, 0, 0, U, S, V)
    c.put("base", 6, 0, 1, U, S, V)
    assert c.has("base", 6, 0, 0) and c.has("base", 6, 0, 1)
    files = list((tmp_path / "base" / "L06").glob("*.pt"))
    assert len(files) == 2
    # LRU key differs
    assert c.key("base", 6, 0, 0) != c.key("base", 6, 0, 1)


def test_svd_version_mismatch(tmp_path):
    """TZ §2.6.3: version v2 → has == False for a v1 entry."""
    c1 = SVDCache(str(tmp_path), r=4, version="v1")
    c2 = SVDCache(str(tmp_path), r=4, version="v2")
    U, S, V = _make_svd()
    c1.put("base", 6, 0, 0, U, S, V)
    assert c1.has("base", 6, 0, 0)
    assert not c2.has("base", 6, 0, 0)
    assert c2.get("base", 6, 0, 0) is None


def test_svd_r_mismatch(tmp_path):
    """TZ §2.6.4: r=4 and r=8 → disjoint keys/files."""
    c4 = SVDCache(str(tmp_path), r=4)
    c8 = SVDCache(str(tmp_path), r=8)
    c4.put("base", 6, 0, 0, *_make_svd(r=4))
    assert c4.has("base", 6, 0, 0)
    assert not c8.has("base", 6, 0, 0)  # r=8 file absent


def test_svd_validation(tmp_path):
    """TZ §2.6.5: unsorted S → ValueError; non-orthonormal U → ValueError."""
    c = SVDCache(str(tmp_path), r=4)
    U, S, V = _make_svd()
    with pytest.raises(ValueError, match="not descending"):
        c.put("base", 6, 0, 0, U, S.flip(0).contiguous(), V)
    U_bad = U.clone()
    U_bad[:, 0] *= 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        c.put("base", 6, 0, 0, U_bad, S, V)
    with pytest.raises(ValueError, match="dtype"):
        c.put("base", 6, 0, 0, U.double(), S, V)
    with pytest.raises(ValueError, match="S shape"):
        c.put("base", 6, 0, 0, U, S[:3], V)


def test_svd_atomicity(tmp_path, monkeypatch):
    """TZ §2.6.6: crash mid-save → no final file (only tmp), next put works."""
    c = SVDCache(str(tmp_path), r=4)
    U, S, V = _make_svd()

    def _boom(_payload, path):
        with open(path, "wb"):
            pass  # simulate a partial write
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(torch, "save", _boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        c.put("base", 6, 0, 0, U, S, V)
    monkeypatch.undo()  # restore torch.save BEFORE the recovery put
    d = tmp_path / "base" / "L06"
    assert not (d / "d0000_p0000_r4_v1.pt").exists()
    # tmp cleaned up, no torn final file
    assert not list(d.glob("*.tmp.*"))
    # next put works
    c.put("base", 6, 0, 0, U, S, V)
    assert c.get("base", 6, 0, 0) is not None


def test_svd_corrupt_file_raises(tmp_path):
    """get on a corrupt file → RuntimeError (never silent None)."""
    c = SVDCache(str(tmp_path), r=4)
    path = c._path("base", 6, 0, 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("not a torch file")
    with pytest.raises(RuntimeError, match="failed to load"):
        c.get("base", 6, 0, 0)


def test_svd_lru_eviction(tmp_path):
    c = SVDCache(str(tmp_path), r=4, cache_size=2)
    U, S, V = _make_svd()
    for p in range(3):
        c.put("base", 6, 0, p, U, S, V)
    assert len(c._lru) == 2
    assert c.key("base", 6, 0, 0) not in c._lru  # evicted (LRU)
    assert c.get("base", 6, 0, 2) is not None  # still on disk


# ---------------------------------------------------------------------------
# MetricCache
# ---------------------------------------------------------------------------


def test_metric_roundtrip(tmp_path):
    """TZ §2.6.7."""
    m = MetricCache(str(tmp_path))
    m.put("eigengap", "base", 6, 0, 2.5)
    assert m.get("eigengap", "base", 6, 0) == 2.5
    assert m.has("eigengap", "base", 6, 0)
    assert m.bulk_get("eigengap", "base", 6) == {0: 2.5}
    assert m.get("eigengap", "base", 6, 99) is None
    # bulk_put
    m.bulk_put("eigengap", "base", 6, {1: 1.0, 2: 3.0})
    assert m.bulk_get("eigengap", "base", 6) == {0: 2.5, 1: 1.0, 2: 3.0}


def test_metric_nan_inf(tmp_path):
    """TZ §2.6.8: nan/inf round-trip via string encoding."""
    m = MetricCache(str(tmp_path))
    m.put("snr", "base", 6, 0, float("nan"))
    m.put("snr", "base", 6, 1, float("inf"))
    assert math.isnan(m.get("snr", "base", 6, 0))
    assert math.isinf(m.get("snr", "base", 6, 1))
    # raw JSON shows the strings (human-readable, no NaN literals)
    raw = open(m._path("snr", "base", 6)).read()
    assert '"nan"' in raw and '"inf"' in raw
    assert not raw.replace('"nan"', "").replace('"inf"', "").replace('"-inf"', "").lower().count("nan")


def test_metric_doc_none(tmp_path):
    """TZ §2.6.9: doc=None aggregate is a separate key, excluded from bulk_get."""
    m = MetricCache(str(tmp_path))
    m.put("eigengap", "base", 6, 0, 1.0)
    m.put("eigengap", "base", 6, None, 2.5)
    assert m.get("eigengap", "base", 6, None) == 2.5
    assert m.has("eigengap", "base", 6, None)
    assert m.bulk_get("eigengap", "base", 6) == {0: 1.0}  # no aggregate
    # non-float value → ValueError (no silent conversion)
    with pytest.raises(ValueError, match="float"):
        m.put("eigengap", "base", 6, 1, torch.tensor(1.0))
    with pytest.raises(ValueError, match="float"):
        m.put("eigengap", "base", 6, 1, 1)  # int, not float


def test_metric_corrupt_json_raises(tmp_path):
    m = MetricCache(str(tmp_path))
    path = m._path("eigengap", "base", 6)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("{broken json")
    with pytest.raises(RuntimeError, match="failed to load"):
        m.get("eigengap", "base", 6, 0)


# ---------------------------------------------------------------------------
# bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_determinism():
    """TZ §2.6.10: same seed → same output."""
    a = bootstrap_ci([1, 2, 3, 4, 5], B=100, seed=0)
    b = bootstrap_ci([1, 2, 3, 4, 5], B=100, seed=0)
    assert a == b
    # on a larger dataset different seeds DO differ
    vals = list(range(1, 101))
    assert bootstrap_ci(vals, B=200, seed=0) != bootstrap_ci(vals, B=200, seed=1)


def test_bootstrap_constant():
    """TZ §2.6.11: constant values → (v, v)."""
    assert bootstrap_ci([5.0] * 3, B=100) == (5.0, 5.0)


def test_bootstrap_validation():
    with pytest.raises(ValueError):
        bootstrap_ci([1.0], B=100)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], B=10)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, float("nan")], B=100)
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 2.0], B=100, alpha=0.0)


def test_bootstrap_paired_narrower_than_independent():
    """TZ §2.6.12: paired CI is narrower than independent, on synthetic data."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 40
    # strongly correlated pairs: a = x + small, b = x + large
    x = rng.normal(size=n)
    a = x + rng.normal(scale=0.05, size=n)
    b = x + 2.0 + rng.normal(scale=0.05, size=n)
    lo_p, hi_p = bootstrap_ci_paired(a.tolist(), b.tolist(), B=500, seed=0)
    lo_i, hi_i = bootstrap_ci(a.tolist(), B=500, seed=0)
    lo_j, hi_j = bootstrap_ci(b.tolist(), B=500, seed=0)
    # independent difference CI (crude: width sum)
    w_paired = hi_p - lo_p
    w_indep = (hi_i - lo_i) + (hi_j - lo_j)
    assert w_paired < w_indep
    # the paired difference is ~ -2.0: its CI must exclude 0 (significant)
    assert hi_p < 0


# ---------------------------------------------------------------------------
# deterministic_mode
# ---------------------------------------------------------------------------


def test_deterministic_mode_restore():
    """TZ §2.6.13: RNG state fully restored on exit."""
    s_before = torch.random.get_rng_state()
    import numpy as np

    n_before = np.random.get_state()[1].copy()
    with deterministic_mode(seed=42):
        torch.randn(1)
        np.random.rand()
    assert torch.equal(s_before, torch.random.get_rng_state())
    assert (n_before == np.random.get_state()[1]).all()
    assert not torch.are_deterministic_algorithms_enabled()


def test_deterministic_mode_nested():
    """TZ §2.6.14: inner seed applies inside; outer state restored outside."""
    with deterministic_mode(seed=1):
        with deterministic_mode(seed=2):
            a = torch.randn(2).clone()
        b = torch.randn(2).clone()  # outer seed=1 stream continues
    # determinism: same nested sequence reproduces
    with deterministic_mode(seed=1):
        with deterministic_mode(seed=2):
            a2 = torch.randn(2).clone()
        b2 = torch.randn(2).clone()
    assert torch.equal(a, a2) and torch.equal(b, b2)


def test_deterministic_mode_strict_restores():
    with deterministic_mode(seed=0, strict=True):
        assert torch.are_deterministic_algorithms_enabled()
    assert not torch.are_deterministic_algorithms_enabled()


# ---------------------------------------------------------------------------
# iter_position_pairs
# ---------------------------------------------------------------------------


def test_iter_position_pairs():
    """TZ §2.6.15."""
    assert list(iter_position_pairs(4, "adjacent", 1)) == [(0, 1), (1, 2), (2, 3)]
    assert list(iter_position_pairs(4, "all")) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    assert list(iter_position_pairs(5, "strided", 2)) == [(0, 2), (1, 3), (2, 4)]
    assert list(iter_position_pairs(5, "adjacent", 2)) == [(0, 2), (2, 4)]
    # all is O(S^2): S=4 → 6 pairs
    assert len(list(iter_position_pairs(4, "all"))) == 6


def test_iter_position_pairs_errors():
    """TZ §2.6.16."""
    with pytest.raises(ValueError):
        list(iter_position_pairs(1, "adjacent"))
    with pytest.raises(ValueError):
        list(iter_position_pairs(4, "adjacent", 0))
    with pytest.raises(ValueError):
        list(iter_position_pairs(4, "bogus"))
    # stride > S: no valid pairs (empty iterator, not an error)
    assert list(iter_position_pairs(5, "adjacent", 10)) == []
