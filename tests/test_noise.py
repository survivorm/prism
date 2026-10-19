"""Selfcheck for prism/noise.py (TZ part 3, §5.6)."""
import math

import numpy as np
import pytest
import torch

from prism.noise import calibrate_noise_floor, validate_signal


# --- 5.6.1: deterministic metric_fn ---------------------------------------

def test_deterministic_metric_fn():
    floor = calibrate_noise_floor(lambda: 42.0, n_repeats=5, reset_between=False)
    assert floor["std"] < 1e-10
    assert floor["is_deterministic"]
    assert len(floor["values"]) == 5
    assert floor["mean"] == 42.0
    assert floor["n_repeats"] == 5
    assert floor["reset_between"] is False


# --- 5.6.2: noisy metric_fn (no seeds set by the calibrator) ---------------

def test_noisy_metric_fn():
    rng = np.random.default_rng(0)
    floor = calibrate_noise_floor(lambda: float(rng.normal(0, 1)), n_repeats=100,
                                  reset_between=False)
    assert 0.5 < floor["std"] < 1.5
    assert not floor["is_deterministic"]


# --- 5.6.4: significant -----------------------------------------------------

def test_validate_signal_significant():
    floor = {"mean": 0.0, "std": 1.0, "is_deterministic": False}
    v = validate_signal(10.0, floor, k=5.0)
    assert v["verdict"] == "significant"
    assert v["ratio"] == pytest.approx(10.0)
    assert v["deviation"] == pytest.approx(10.0)
    assert v["threshold"] == pytest.approx(5.0)
    assert v["is_deterministic"] is False


# --- 5.6.5: noise -----------------------------------------------------------

def test_validate_signal_noise():
    floor = {"mean": 0.0, "std": 1.0, "is_deterministic": False}
    v = validate_signal(2.0, floor, k=5.0)
    assert v["verdict"] == "noise"
    assert v["ratio"] == pytest.approx(2.0)


# --- 5.6.6: deterministic floor ---------------------------------------------

def test_validate_signal_deterministic():
    floor = {"mean": 42.0, "std": 0.0, "is_deterministic": True}
    # TZ §5.6.6 uses 42.0 + 1e-9 and expects "no_floor", which contradicts
    # the explicit rule (|dev| > 1e-10 → "significant"); 1e-9 > 1e-10.
    # We follow the explicit rule; the selfcheck value is 1e-12 here.
    v = validate_signal(42.0 + 1e-12, floor)
    assert v["verdict"] == "no_floor"
    assert validate_signal(42.0 + 1e-9, floor)["verdict"] == "significant"
    v2 = validate_signal(50.0, floor)
    assert v2["verdict"] == "significant"
    assert v2["ratio"] == float("inf")


# --- 5.6.7: n_repeats < 2 ----------------------------------------------------

def test_n_repeats_too_small():
    with pytest.raises(ValueError, match="n_repeats"):
        calibrate_noise_floor(lambda: 1.0, n_repeats=1)


# --- 5.6.8: NaN in values -----------------------------------------------------

def test_nan_value_raises():
    with pytest.raises(ValueError, match="nan"):
        calibrate_noise_floor(lambda: float("nan"), n_repeats=2, reset_between=False)
    with pytest.raises(ValueError, match="inf"):
        calibrate_noise_floor(lambda: float("inf"), n_repeats=2, reset_between=False)


# --- 5.6.9: non-scalar return -------------------------------------------------

def test_non_scalar_raises():
    with pytest.raises(TypeError, match="scalar"):
        calibrate_noise_floor(lambda: [1.0, 2.0], n_repeats=2, reset_between=False)
    with pytest.raises(TypeError, match="scalar"):
        calibrate_noise_floor(lambda: torch.tensor([1.0, 2.0]), n_repeats=2,
                              reset_between=False)
    # 0-d / 1-element tensor is accepted
    floor = calibrate_noise_floor(lambda: torch.tensor(3.0), n_repeats=2,
                                  reset_between=False)
    assert floor["mean"] == 3.0


# --- 5.6.10: reset_between ----------------------------------------------------

def test_reset_between_runs():
    counter = {"n": 0}

    def metric():
        counter["n"] += 1
        return float(counter["n"])

    floor = calibrate_noise_floor(metric, n_repeats=4, reset_between=True)
    assert counter["n"] == 4
    assert floor["std"] >= 0
    # with distinct values the floor is not deterministic
    assert not floor["is_deterministic"]


# --- 5.6.11: k <= 0 -----------------------------------------------------------

def test_k_nonpositive():
    floor = {"mean": 0.0, "std": 1.0, "is_deterministic": False}
    with pytest.raises(ValueError, match="k must be > 0"):
        validate_signal(1.0, floor, k=0.0)
    with pytest.raises(ValueError, match="k must be > 0"):
        validate_signal(1.0, floor, k=-1.0)


# --- extra: floor without keys -> KeyError ------------------------------------

def test_floor_missing_key():
    with pytest.raises(KeyError, match="is_deterministic"):
        validate_signal(1.0, {"mean": 0.0, "std": 1.0})


# --- extra: inconsistent floor (std=0, not deterministic) ----------------------

def test_inconsistent_floor_raises():
    with pytest.raises(RuntimeError, match="inconsistent"):
        validate_signal(1.0, {"mean": 0.0, "std": 0.0, "is_deterministic": False})
