"""Strict RFC-8259 JSON serialization for experiment outputs.

``json.dump``'s ``default=`` hook is NOT called for Python floats:
``float('nan')`` / ``float('inf')`` are serialized natively as the
non-standard ``NaN`` / ``Infinity`` literals (and ``np.float64`` is a
``float`` subclass, so it has the same property; only ``np.float32``
reaches ``default=``). A ``default=`` function that stringifies
non-finite floats is therefore a dead guard for the common case.

Fix = recursive pre-sanitization + ``allow_nan=False`` (defense in
depth: any NaN that slips through raises ``ValueError`` loudly).

Convention (same as ``prism.cache._json_dump_value``): non-finite
floats become the strings ``"nan"`` / ``"inf"`` / ``"-inf"`` — valid
JSON, round-trippable with a matching loader.

Usage (replaces per-script ``_json_default``):

    from prism.json_utils import dump as json_dump
    with open(OUT, "w") as f:
        json_dump(out, f, indent=1)
"""
from __future__ import annotations

import json
import math

import numpy as np

__all__ = ["sanitize", "dump"]


def sanitize(o):
    """Recursively convert a JSON tree to strict-RFC-8259 form.

    - ``np.bool_`` -> TypeError (unwrapped numpy bool; fail loud)
    - ``np.integer`` -> ``int``
    - ``np.floating`` -> ``float`` (then non-finite check)
    - ``float`` NaN/Inf -> ``"nan"`` / ``"inf"`` / ``"-inf"``
    - dict/list/tuple -> elementwise
    """
    if isinstance(o, np.bool_):
        raise TypeError(f"unwrapped numpy bool_ in JSON: {o!r}")
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, float):
        if math.isnan(o):
            return "nan"
        if math.isinf(o):
            return "inf" if o > 0 else "-inf"
        return o
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


def dump(o, f, **kwargs):
    """``json.dump(sanitize(o), f, allow_nan=False, **kwargs)``."""
    kwargs["allow_nan"] = False
    json.dump(sanitize(o), f, **kwargs)
