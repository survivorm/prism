"""Selfcheck for prism/permutation.py (TZ part 3, §6.6)."""
import numpy as np
import pytest

from prism.permutation import (
    cluster_based_permutation,
    generate_null,
    permutation_max_statistic,
)


def _median_diff(b, e):
    return np.median(e - b, axis=0)


# --- 6.6.1: trivial observed ---------------------------------------------------

def test_max_statistic_trivial():
    rng = np.random.default_rng(0)
    observed = np.zeros(6)
    null = rng.normal(0, 1, (100, 6))
    res = permutation_max_statistic(observed, null)
    assert res["p_value"] == 1.0
    assert not res["significant"]
    assert res["B"] == 100
    assert res["n_layers"] == 6


# --- 6.6.2: extreme observed ---------------------------------------------------

def test_max_statistic_extreme():
    rng = np.random.default_rng(0)
    observed = np.zeros(6)
    observed[2] = 100.0
    null = rng.normal(0, 1, (100, 6))
    res = permutation_max_statistic(observed, null)
    assert res["p_value"] <= 1.0 / 101
    assert res["significant"]


# --- 6.6.3: exact p formula ----------------------------------------------------

def test_max_statistic_p_formula():
    observed = np.array([1.0, 0.0, 0.0])
    null = np.zeros((100, 3))
    res = permutation_max_statistic(observed, null)
    # no null max reaches 1.0 -> k = 0 -> p = 1/101
    assert res["p_value"] == pytest.approx(1.0 / 101)
    # now all null maxes reach 1.0 -> k = 100 -> p = 1.0
    null2 = np.ones((100, 3))
    assert permutation_max_statistic(observed, null2)["p_value"] == pytest.approx(1.0)


# --- 6.6.4: correlation absorbed by the max ------------------------------------

def test_max_statistic_correlation():
    rng = np.random.default_rng(1)
    observed = np.ones(6)
    null = rng.normal(0, 1, (100, 6))
    res = permutation_max_statistic(observed, null)
    # max over 6 correlated (here iid) normals: P(max >= 1) >> P(one >= 1)
    single = 1 - 0.6826894921370859 / 2  # P(|N(0,1)| >= 1) ~ 0.317
    assert res["p_value"] > single  # max absorbs correlation/multiplicity
    assert 0.0 < res["p_value"] < 1.0


# --- 6.6.5: generate_null determinism ------------------------------------------

def test_generate_null_determinism():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 10)
    edit = base + 0.5
    fn = lambda b, e: np.array([np.median(e - b)])
    a = generate_null(base, edit, fn, n_permutations=100, seed=0)
    b2 = generate_null(base, edit, fn, n_permutations=100, seed=0)
    assert np.array_equal(a, b2)
    assert a.shape == (100, 1)
    c = generate_null(base, edit, fn, n_permutations=100, seed=1)
    assert not np.array_equal(a, c)


# --- 6.6.6: paired structure ----------------------------------------------------

def test_generate_null_paired_structure():
    base = np.array([1.0, 1.0, 1.0])
    edit = np.array([2.0, 2.0, 2.0])
    fn = lambda b, e: np.array([np.median(e - b)])
    null = generate_null(base, edit, fn, n_permutations=1000, seed=0)
    # each doc contributes +1 (unswapped) or -1 (swapped); median of 3
    # such values is in {-1, +1} (odd count -> never 0)
    vals = np.unique(null[:, 0])
    assert set(vals.tolist()).issubset({-1.0, 1.0})
    assert len(vals) == 2  # both signs occur
    # symmetry: nearly half +1, half -1
    frac_pos = float(np.mean(null[:, 0] == 1.0))
    assert 0.3 < frac_pos < 0.7


# --- 6.6.7: shape mismatch -------------------------------------------------------

def test_generate_null_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        generate_null(np.ones(5), np.ones(6),
                      lambda b, e: np.array([np.median(e - b)]),
                      n_permutations=100)


# --- 2-D input [n_docs, n_layers] (TZ §6.3 documented use case) --------------------
# Regression: np.where(signs, base, edit) with 1-D signs broadcast left-padded
# to (1, n_docs) — errors when n_docs != n_layers, and SILENTLY swaps columns
# (not document rows) when n_docs == n_layers. The per-doc coin flip must
# swap whole rows so that a document's value flips identically across layers.

def test_generate_null_2d_non_square():
    # n_docs=10, n_layers=3 — the broadcasting error case.
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (10, 3))
    edit = base + 0.5
    fn = lambda b, e: np.median(e - b, axis=0)  # [n_layers], TZ §6.3 example
    null = generate_null(base, edit, fn, n_permutations=100, seed=0)
    assert null.shape == (100, 3)
    # each doc contributes +0.5 (kept) or -0.5 (swapped); median of 10 such
    # values is in {-0.5, 0, +0.5} (even count -> middle two averaged), and
    # IDENTICAL across layers (row-level flips)
    assert set(np.unique(null).round(10).tolist()).issubset({-0.5, 0.0, 0.5})
    assert np.allclose(null, null[:, :1]), "all layers must share the row-flipped median"


def test_generate_null_2d_square_rows_not_columns():
    # n_docs == n_layers == 4: the silent column-swap case.
    # doc i has value (i+1) in every layer; base = 0.
    base = np.zeros((4, 4))
    edit = np.arange(1.0, 5.0)[:, None] * np.ones((1, 4))
    fn = lambda b, e: np.median(e - b, axis=0)  # [4] per-layer medians
    null = generate_null(base, edit, fn, n_permutations=200, seed=0)
    # correct (row) semantics: a document flip changes ALL its layers by the
    # same sign -> per-layer medians of one null row are always equal.
    # The old column-swap bug let per-layer medians differ within a row.
    assert np.allclose(null, null[:, :1])
    # and both flip states must occur across the run (median can be + or -)
    assert np.any(null[:, 0] > 0) and np.any(null[:, 0] < 0)


# --- 6.6.8: n_docs < 2 ------------------------------------------------------------

def test_generate_null_too_few_docs():
    with pytest.raises(ValueError, match="n_docs"):
        generate_null(np.ones(1), np.ones(1),
                      lambda b, e: np.array([np.median(e - b)]),
                      n_permutations=100)


# --- 6.6.9: metric_fn empty output -------------------------------------------------

def test_generate_null_empty_metric_output():
    with pytest.raises(ValueError, match="empty|1-D"):
        generate_null(np.ones(3), np.ones(3),
                      lambda b, e: np.array([]),
                      n_permutations=100)
    with pytest.raises(ValueError, match="NaN/Inf"):
        generate_null(np.ones(3), np.ones(3),
                      lambda b, e: np.array([float("nan")]),
                      n_permutations=100)


# --- 6.6.10: single cluster --------------------------------------------------------

def test_cluster_single():
    rng = np.random.default_rng(0)
    observed = np.array([0.0, 3.0, 4.0, 0.0, 0.0, 0.0])
    null = rng.normal(0, 1, (100, 6))
    res = cluster_based_permutation(observed, null, threshold=2.0)
    assert len(res["clusters"]) == 1
    assert res["clusters"][0]["layers"] == [1, 2]
    assert res["clusters"][0]["statistic"] == pytest.approx(7.0)


# --- 6.6.11: two clusters -----------------------------------------------------------

def test_cluster_two():
    rng = np.random.default_rng(0)
    observed = np.array([3.0, 4.0, 0.0, 0.0, 5.0, 6.0])
    null = rng.normal(0, 1, (100, 6))
    res = cluster_based_permutation(observed, null, threshold=2.0)
    assert len(res["clusters"]) == 2
    assert res["clusters"][0]["layers"] == [0, 1]
    assert res["clusters"][1]["layers"] == [4, 5]


# --- 6.6.12: no clusters -------------------------------------------------------------

def test_cluster_none():
    rng = np.random.default_rng(0)
    observed = np.zeros(6)
    null = rng.normal(0, 1, (100, 6))
    res = cluster_based_permutation(observed, null, threshold=2.0)
    assert res["clusters"] == []


# --- 6.6.13: strong observed cluster ---------------------------------------------------

def test_cluster_strong():
    rng = np.random.default_rng(0)
    observed = np.array([0.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    null = rng.normal(0, 1, (100, 6))
    res = cluster_based_permutation(observed, null, threshold=2.0)
    assert len(res["clusters"]) == 1
    assert res["clusters"][0]["p_value"] <= 1.0 / 101
    assert res["clusters"][0]["significant"]


# --- 6.6.14: threshold / B / alpha validation -------------------------------------------

def test_cluster_validation():
    rng = np.random.default_rng(0)
    observed = np.array([0.0, 3.0, 4.0, 0.0, 0.0, 0.0])
    null = rng.normal(0, 1, (100, 6))
    with pytest.raises(ValueError, match="threshold"):
        cluster_based_permutation(observed, null, threshold=0.0)
    with pytest.raises(ValueError, match="B = 50 < 100"):
        cluster_based_permutation(observed, null[:50], threshold=2.0)
    with pytest.raises(ValueError, match="alpha"):
        cluster_based_permutation(observed, null, threshold=2.0, alpha=1.5)
    with pytest.raises(ValueError, match="columns"):
        permutation_max_statistic(observed, rng.normal(0, 1, (100, 3)))
    with pytest.raises(ValueError, match="NaN/Inf"):
        bad = null.copy()
        bad[0, 0] = float("nan")
        permutation_max_statistic(observed, bad)
    # 2-D observed is rejected
    with pytest.raises(ValueError, match="flat"):
        permutation_max_statistic(observed[None, :], null)
    # list inputs are converted, not a TypeError from fancy indexing
    # (docs/changed_review.md #2: _check_common must propagate arrays)
    res = cluster_based_permutation(observed.tolist(), null, threshold=2.0)
    assert len(res["clusters"]) == 1
    res2 = permutation_max_statistic(observed.tolist(), null)
    assert res2["n_layers"] == 6
