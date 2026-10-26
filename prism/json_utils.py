"""Strict RFC-8259 JSON serialization for experiment outputs.

``json.dump``'s ``default=`` hook is NOT called for Python floats:
``float('nan')`` / ``float('inf')`` are serialized natively as the
non-standard ``NaN`` / ``Infinity`` literals (and ``np.float64`` is a
``float`` subclass, so it has the same property; only ``np.float32``
reaches ``default=``). A ``default=`` function that stringifies
non-finite floats is therefore a dead guard for the common case.

Fix = recursive pre-sanitization + ``allow_nan=False`` (defense in
depth: any NaN that slips through raises ``ValueError`` loudly) +
``default=_raise`` (a caller-passed ``default=str`` cannot swallow an
unserializable object into a useless string).

Convention (same as ``prism.cache._json_dump_value``): non-finite
floats become the strings ``"nan"`` / ``"inf"`` / ``"-inf"`` — valid
JSON. Writer-only: the matching reader is external
(``prism.cache._json_load_value`` handles these markers for
``MetricCache`` fields, non-recursively); a recursive loader is NOT
provided here.

Fail-loud boundary: ``np.ndarray`` / ``torch.Tensor`` fall through
``sanitize`` and are rejected by ``json.dump`` (default=_raise) — call
``.tolist()`` / ``.detach().cpu().tolist()`` at the caller if intended.

Caller pattern (atomic write — so a sanitize/default raise does not
leave a 0-byte OUT that downstream may read as "the experiment ran"):

    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json_dump(out, f, indent=1)
    os.replace(tmp, OUT)
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
    - dict keys AND values -> elementwise (a non-sanitizable key, e.g.
      ``np.bool_`` / ``np.int64``, raises here with the friendly message
      instead of a raw TypeError from ``json.dump``)
    - list/tuple -> elementwise (tuple -> list)
    - ``np.ndarray`` / ``torch.Tensor`` -> pass through; ``json.dump``
      rejects them (fail loud; see module docstring)
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
        return {sanitize(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    return o


def _raise(o):
    raise TypeError(
        f"unserializable object in JSON: {type(o).__name__!r} "
        f"(convert at the caller: .tolist() / .item() / float() / int())")


def dump(o, f, **kwargs):
    """``json.dump(sanitize(o), f, allow_nan=False, default=_raise, **kwargs)``.

    ``allow_nan`` and ``default`` are forced — a caller cannot bypass
    the strict contract.
    """
    kwargs["allow_nan"] = False
    kwargs["default"] = _raise
    json.dump(sanitize(o), f, **kwargs)
