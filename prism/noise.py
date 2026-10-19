"""Noise-floor calibration and signal validation (TZ part 3, §5).

Generic: no model, no architecture specifics, no experiment logic. A
"noise floor" is the natural variance of REPEATED COMPUTATION of a
metric callable — it reflects forward/backward non-determinism
(recurrent-ordering effects, CUDA atomic adds, thread scheduling) when
the callable includes model passes, and is ≈ 0 when the callable reads
from a deterministic cache (correct: the cache IS deterministic).

paper.md §8.8 — noise floor and signal validation.

NOT covered here (TZ §5.8): architecture-specific policy
(experiments/exp_T0.py), ``deterministic_mode`` (cache.py — orthogonal:
it ELIMINATES variance), cross-version checks (exp_T0.py), paired
comparison of two floors (caller applies validate_signal twice), floor
caching (MetricCache).

Do NOT call ``calibrate_noise_floor`` inside
``deterministic_mode(strict=True)`` — the floor would be 0 (correct, but
not the "natural" floor).
"""

from __future__ import annotations

import gc
import math
from collections.abc import Callable
from typing import Any

import torch

__all__ = ["calibrate_noise_floor", "validate_signal"]


def _reset_caches() -> None:
    """Drop dynamo/CUDA caches and collect GC. Never raises.

    Increases the natural variance of repeated metric_fn calls (warm
    caches give std ≈ 0 even when real non-determinism exists).
    """
    try:
        torch._dynamo.reset()
    except Exception:  # dynamo not imported / not available
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:  # no CUDA
        pass
    gc.collect()


def _require_scalar(value: Any, i: int) -> float:
    """Accept Python scalars and 0-d/1-element tensors; reject the rest."""
    if isinstance(value, (bool,)):
        raise TypeError(f"metric_fn() returned bool at repeat {i}; need a float")
    if isinstance(value, float):
        out = value
    elif isinstance(value, int):
        out = float(value)
    elif isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise TypeError(
                f"metric_fn() returned a {value.numel()}-element tensor at "
                f"repeat {i}; need a scalar (use .item())"
            )
        out = value.item()
    else:
        raise TypeError(
            f"metric_fn() returned {type(value).__name__} at repeat {i}; "
            f"need a scalar float (use .item())"
        )
    if math.isnan(out) or math.isinf(out):
        raise ValueError(
            f"metric_fn() returned {'nan' if math.isnan(out) else 'inf'} at "
            f"repeat {i}; a zero-division or invalid computation is hiding "
            f"in the metric"
        )
    return out


def calibrate_noise_floor(
    metric_fn: Callable[[], float],
    n_repeats: int = 10,
    reset_between: bool = True,
    verbose: bool = False,
) -> dict:
    """Measure the natural variance of repeated metric_fn() calls.

    Seeds are NOT set (that would force the floor to 0). ``metric_fn``
    must be pure (no persistent mutation) — the caller's responsibility.

    Args:
        metric_fn: Callable[[], float]; must not install seeds.
        n_repeats: ≥ 2 (else std is undefined); 10 recommended.
        reset_between: True — reset dynamo/CUDA caches + gc between
            calls (realistic natural floor); False — back-to-back calls
            (warm cache may give std ≈ 0).
        verbose: print ``[i/N] value`` per repeat.

    Returns:
        dict with ``mean``, ``std`` (ddof=0), ``values`` (list of
        floats, len = n_repeats), ``is_deterministic`` (std < 1e-10),
        ``n_repeats``, ``reset_between``.

    ``is_deterministic=True`` means the computation is deterministic in
    this configuration: any non-zero signal is a real effect (not
    "no signal" — see validate_signal).

    Errors:
        ValueError: n_repeats < 2; NaN/Inf in a value (with repeat index).
        TypeError: metric_fn() returns a non-scalar (with repeat index).
    """
    if n_repeats < 2:
        raise ValueError(f"n_repeats must be >= 2, got {n_repeats}")

    values: list[float] = []
    for i in range(n_repeats):
        if reset_between and i > 0:
            _reset_caches()
        v = _require_scalar(metric_fn(), i)
        values.append(v)
        if verbose:
            print(f"[{i + 1}/{n_repeats}] {v}")

    mean = float(sum(values) / len(values))
    var = sum((x - mean) ** 2 for x in values) / len(values)  # ddof=0
    std = float(math.sqrt(var))
    return {
        "mean": mean,
        "std": std,
        "values": values,
        "is_deterministic": std < 1e-10,
        "n_repeats": n_repeats,
        "reset_between": reset_between,
    }


def _validate_floor(floor: dict) -> None:
    missing = [k for k in ("mean", "std", "is_deterministic") if k not in floor]
    if missing:
        raise KeyError(f"floor dict is missing key(s): {missing}")


def validate_signal(
    signal: float,
    floor: dict,
    k: float = 5.0,
) -> dict:
    """Decide whether ``signal`` exceeds the noise floor by k sigmas.

    paper.md §8.8. Verdicts:
        "significant" — |signal − mean| > k·std (or > 1e-10 when
            is_deterministic=True): a real effect may be claimed.
        "noise" — |signal − mean| ≤ k·std (and not deterministic):
            NO effect may be claimed.
        "no_floor" — deterministic floor and |signal − mean| ≤ 1e-10:
            nothing happens (signal ~ 0 in a deterministic system; not
            a bug, but not a "significant effect" either).

    Args:
        signal: observed scalar (e.g. κ_U or ε_gap).
        floor: dict from calibrate_noise_floor.
        k: number of sigmas; must be > 0 (default 5, the 5-sigma rule).

    Returns:
        dict with ``verdict``, ``ratio`` (|signal−mean|/std),
        ``deviation`` (signal−mean), ``threshold`` (k·std),
        ``is_deterministic``.

    Errors:
        ValueError: k ≤ 0; NaN/Inf in signal.
        KeyError: floor dict without the required keys.
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    if math.isnan(signal) or math.isinf(signal):
        raise ValueError(f"signal must be finite, got {signal}")
    _validate_floor(floor)

    mean = float(floor["mean"])
    std = float(floor["std"])
    is_det = bool(floor["is_deterministic"])
    deviation = float(signal) - mean
    abs_dev = abs(deviation)

    if is_det:
        verdict = "significant" if abs_dev > 1e-10 else "no_floor"
        ratio = float("inf") if abs_dev > 0.0 else 0.0
    else:
        if std == 0.0:
            raise RuntimeError(
                "floor has std == 0 but is_deterministic=False "
                "(inconsistent; re-run calibrate_noise_floor)"
            )
        verdict = "significant" if abs_dev > k * std else "noise"
        ratio = abs_dev / std

    return {
        "verdict": verdict,
        "ratio": ratio,
        "deviation": deviation,
        "threshold": k * std,
        "is_deterministic": is_det,
    }
