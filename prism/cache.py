"""Caching infrastructure for the new experiment series (docs/new_code_1.md).

Two disk caches + bootstrap CI + determinism + position-pair iterators.

Division of labor with ``cache_io.py``: ``cache_io.find_targets_dir`` is
read-only path resolution for EXISTING producer caches (Kaggle FUSE
mounts); ``cache.py`` is read+WRITE with keys, atomic writes and
versioning. ``find_targets_dir`` is reused (imported), not duplicated.

Concurrency: ONE process per ``MetricCache`` (metric, edit, layer) triple
— atomic ``os.replace`` prevents torn files, not last-writer-lost across
threads/processes. No parallel writes are supported.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import uuid
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from .cache_io import find_targets_dir

__all__ = [
    "SVDCache",
    "MetricCache",
    "bootstrap_ci",
    "bootstrap_ci_paired",
    "deterministic_mode",
    "iter_position_pairs",
]

_LAYER_AGG_KEY = "__layer_aggregate__"


def _atomic_write(path: str, write_fn: Callable[[str], None]) -> None:
    """Write via tmp file + os.replace (no torn files on crash)."""
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


def _ensure_root(root: str) -> None:
    """Create the cache root if absent; reuse an existing producer cache.

    ``find_targets_dir`` (cache_io, read-only listdir walk) is consulted
    first so an existing Kaggle FUSE layout is picked up; if nothing is
    found the root is created locally. A read-only FUSE root passes the
    isdir check — writes there fail at ``put`` with a RuntimeError.
    """
    if os.path.isdir(root):
        return
    try:
        find_targets_dir(root)  # raises if no *.pt under root / mounts
    except FileNotFoundError:
        os.makedirs(root, exist_ok=True)


def _is_writable(path: str) -> bool:
    try:
        probe = os.path.join(path, f".probe.{uuid.uuid4().hex[:8]}")
        with open(probe, "w"):
            pass
        os.remove(probe)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# SVD cache
# ---------------------------------------------------------------------------


class SVDCache:
    """Disk cache of truncated SVD ``(U, S, V)`` keyed by
    ``(edit, layer, doc, pos)`` (plus ``r``, ``dtype``, ``version``).

    File format: ``torch.save({"U", "S", "V", "meta"})``, loaded with
    ``weights_only=True``. Writes are atomic (tmp + ``os.replace``).
    The cache is device-independent (CPU files); move results explicitly.
    """

    def __init__(
        self,
        root: str,
        r: int,
        dtype: torch.dtype = torch.float32,
        version: str = "v1",
        cache_size: int = 256,
    ):
        if not isinstance(r, int) or r < 1:
            raise ValueError(f"SVDCache: r must be a positive int, got {r!r}")
        self.root = root
        self.r = r
        self.dtype = dtype
        self.version = version
        self._lru: OrderedDict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            OrderedDict()
        )
        self._cache_size = cache_size
        _ensure_root(root)

    # -- keys --------------------------------------------------------------

    def key(self, edit: str, layer: int, doc: int, pos: int) -> str:
        """SHA-256 of the request identifier (used for the in-memory LRU)."""
        s = f"{edit}|L{layer}|d{doc}|p{pos}|r{self.r}|{self.dtype}|v{self.version}"
        return hashlib.sha256(s.encode()).hexdigest()

    def _path(self, edit: str, layer: int, doc: int, pos: int) -> str:
        # dtype is part of the path: TZ §2.3 lists it in the key, so two
        # caches with the same (r, version) but different dtypes must not
        # share a file (review new_code_1_review.md P2.4 — TZ typo resolved
        # in favor of the key).
        return os.path.join(
            self.root, edit, f"L{layer:02d}",
            f"d{doc:04d}_p{pos:04d}_r{self.r}_{self.dtype}_{self.version}.pt",
        )

    # -- API ---------------------------------------------------------------

    def has(self, edit: str, layer: int, doc: int, pos: int) -> bool:
        return os.path.isfile(self._path(edit, layer, doc, pos))

    def get(
        self, edit: str, layer: int, doc: int, pos: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Load (U, S, V) from memory LRU or disk.

        Returns None if the file for THIS (r, dtype, version) does not
        exist. A corrupted file raises RuntimeError (never silent None —
        that would hide cache corruption). A torch-version mismatch in the
        meta is a warning only (the format is compatible).
        """
        k = self.key(edit, layer, doc, pos)
        if k in self._lru:
            self._lru.move_to_end(k)
            return self._lru[k]
        path = self._path(edit, layer, doc, pos)
        if not os.path.isfile(path):
            return None
        try:
            payload = torch.load(path, weights_only=True)
            U, S, V = payload["U"], payload["S"], payload["V"]
            meta = payload["meta"]
        except Exception as e:  # corrupted / unexpected file
            raise RuntimeError(f"SVDCache: failed to load {path}: {e}") from e
        if meta.get("r") != self.r or meta.get("version") != self.version:
            return None  # key mismatch (different cache)
        if meta.get("dtype") != str(self.dtype):
            return None
        if meta.get("torch") not in (None, torch.__version__):
            warnings.warn(
                f"SVDCache: {path} was written with torch {meta.get('torch')}, "
                f"current is {torch.__version__} (format-compatible, proceeding)",
                RuntimeWarning,
            )
        U, S, V = U.to(self.dtype), S.to(self.dtype), V.to(self.dtype)
        self._lru[k] = (U, S, V)
        if len(self._lru) > self._cache_size:
            self._lru.popitem(last=False)
        return U, S, V

    def put(
        self,
        edit: str,
        layer: int,
        doc: int,
        pos: int,
        U: torch.Tensor,
        S: torch.Tensor,
        V: torch.Tensor,
    ) -> None:
        """Store (U, S, V) with full validation; atomic on-disk write."""
        if U.dtype != self.dtype or S.dtype != self.dtype or V.dtype != self.dtype:
            raise ValueError(
                f"SVDCache.put: dtype mismatch — expected {self.dtype}, got "
                f"U={U.dtype}, S={S.dtype}, V={V.dtype}"
            )
        if U.dim() != 2:
            raise ValueError(f"SVDCache.put: U must be 2-D, got {U.dim()}-D")
        K, r = U.shape
        if S.shape != (r,):
            raise ValueError(f"SVDCache.put: S shape {tuple(S.shape)} != ({r},)")
        if V.shape != (r, U.shape[1]):
            raise ValueError(f"SVDCache.put: V shape {tuple(V.shape)} != ({r}, {U.shape[1]})")
        if r > min(K, V.shape[1]):
            raise ValueError(f"SVDCache.put: r={r} > min(K, D)={min(K, V.shape[1])}")
        if r > 1 and not torch.all(S[:-1] >= S[1:]):
            raise ValueError("SVDCache.put: S is not descending")
        if not torch.allclose(U.T @ U, torch.eye(r, device=U.device, dtype=U.dtype), atol=1e-5):
            raise ValueError("SVDCache.put: U is not orthonormal (‖UᵀU − I‖_F ≥ 1e-5)")
        if not torch.allclose(V @ V.T, torch.eye(r, device=V.device, dtype=V.dtype), atol=1e-5):
            raise ValueError("SVDCache.put: V is not orthonormal (‖VVᵀ − I‖_F ≥ 1e-5)")

        path = self._path(edit, layer, doc, pos)
        if not _is_writable(self.root):
            raise RuntimeError(
                f"SVDCache.put: root is not writable (read-only Kaggle FUSE "
                f"mount?): {self.root}"
            )
        meta = {
            "edit": edit, "layer": layer, "doc": doc, "pos": pos,
            "r": r, "dtype": str(self.dtype), "version": self.version,
            "torch": torch.__version__,
        }
        payload = {"U": U.cpu(), "S": S.cpu(), "V": V.cpu(), "meta": meta}

        def _save(tmp: str) -> None:
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            torch.save(payload, tmp)

        _atomic_write(path, _save)
        k = self.key(edit, layer, doc, pos)
        self._lru[k] = (U, S, V)
        self._lru.move_to_end(k)
        if len(self._lru) > self._cache_size:
            self._lru.popitem(last=False)


# ---------------------------------------------------------------------------
# Scalar metric cache
# ---------------------------------------------------------------------------


def _json_dump_value(v: float) -> object:
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return v


def _json_load_value(v: object) -> float:
    if isinstance(v, str):
        if v == "nan":
            return float("nan")
        if v == "inf":
            return float("inf")
        if v == "-inf":
            return float("-inf")
    return float(v)


class MetricCache:
    """Disk cache of scalar metrics keyed by (metric, edit, layer, doc).

    JSON files ``{root}/{metric}/{version}/{edit}/L{layer:02d}.json``
    (human-readable, git-mergeable; scalars are small — .pt would be
    overengineering). ``nan``/``inf`` are stored as the strings
    "nan"/"inf"/"-inf". Document keys are zero-padded (``doc_{N:04d}``) so
    that git diffs stay stable. ``doc=None`` addresses the per-layer
    aggregate (key ``__layer_aggregate__``), which bulk_get/bulk_put do
    NOT include. No in-memory cache (files are small; a cache would
    complicate consistency under parallel writes).

    Version semantics: a change of ``version`` produces a NEW file
    (``{version}/`` in the path); old files remain untouched. A given
    root is therefore monoversion by construction.
    """

    def __init__(self, root: str, version: str = "v1"):
        self.root = root
        self.version = version
        _ensure_root(root)

    def _path(self, metric: str, edit: str, layer: int) -> str:
        # version is part of the path: TZ §2.3 says "version in the key;
        # version change -> new file" (review new_code_1_review.md P2.4 —
        # TZ typo resolved in favor of the key: without this, a version
        # change would mix v1/v2 documents under a single meta). A root
        # must therefore be used by exactly one version; see class docstring.
        return os.path.join(self.root, metric, self.version, edit, f"L{layer:02d}.json")

    def _load(self, metric: str, edit: str, layer: int) -> dict:
        path = self._path(metric, edit, layer)
        if not os.path.isfile(path):
            return {}
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise RuntimeError(f"MetricCache: failed to load {path}: {e}") from e
        return data

    def _save(self, metric: str, edit: str, layer: int, data: dict) -> None:
        path = self._path(metric, edit, layer)
        if not _is_writable(self.root):
            raise RuntimeError(
                f"MetricCache.put: root is not writable (read-only Kaggle FUSE "
                f"mount?): {self.root}"
            )
        data = {**data, "meta": {"version": self.version}}

        def _w(tmp: str) -> None:
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            with open(tmp, "w") as f:
                json.dump(data, f, indent=1, sort_keys=True)

        _atomic_write(path, _w)

    @staticmethod
    def _check_value(value: object) -> float:
        if not isinstance(value, float):
            raise ValueError(
                f"MetricCache.put: value must be a Python float "
                f"(got {type(value).__name__}); call .item() explicitly"
            )
        return value

    def has(self, metric: str, edit: str, layer: int, doc: int | None = None) -> bool:
        data = self._load(metric, edit, layer)
        key = "__layer_aggregate__" if doc is None else f"doc_{int(doc):04d}"
        return key in data

    def get(self, metric: str, edit: str, layer: int, doc: int | None = None) -> float | None:
        data = self._load(metric, edit, layer)
        key = "__layer_aggregate__" if doc is None else f"doc_{int(doc):04d}"
        if key not in data:
            return None
        return _json_load_value(data[key])

    def put(
        self,
        metric: str,
        edit: str,
        layer: int,
        doc: int | None,
        value: float,
    ) -> None:
        value = self._check_value(value)
        key = "__layer_aggregate__" if doc is None else f"doc_{int(doc):04d}"
        data = self._load(metric, edit, layer)
        data[key] = _json_dump_value(value)
        self._save(metric, edit, layer, data)

    def bulk_get(self, metric: str, edit: str, layer: int) -> dict[int, float]:
        """{doc: value} for all present documents (aggregate excluded)."""
        data = self._load(metric, edit, layer)
        return {
            int(k[4:]): _json_load_value(v)
            for k, v in data.items()
            if k.startswith("doc_")
        }

    def bulk_put(
        self,
        metric: str,
        edit: str,
        layer: int,
        values: dict[int, float],
    ) -> None:
        data = self._load(metric, edit, layer)
        for doc, value in values.items():
            data[f"doc_{int(doc):04d}"] = _json_dump_value(self._check_value(value))
        self._save(metric, edit, layer, data)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    B: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    statistic: Callable[[np.ndarray], float] = np.median,
) -> tuple[float, float]:
    """Percentile bootstrap CI of a statistic (default: median) over values.

    Resampling WITH replacement, ``len(values)`` draws per iteration,
    ``np.random.default_rng(seed)`` (deterministic). Paper §13: B=1000,
    percentile CI. NaN in values → ValueError (silently dropping them
    would hide bugs). For heavy-tailed metrics median (default) is the
    robust choice; ``mean`` is available via the ``statistic`` argument.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"bootstrap_ci: expected 1-D values, got {arr.ndim}-D")
    n = arr.size
    if n < 2:
        raise ValueError(f"bootstrap_ci: need >= 2 values, got {n}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bootstrap_ci: NaN/Inf in values")
    if B < 100:
        raise ValueError(f"bootstrap_ci: B must be >= 100, got {B}")
    if not (0 < alpha < 1):
        raise ValueError(f"bootstrap_ci: alpha must be in (0, 1), got {alpha}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    samples = arr[idx]
    stats = np.array([statistic(s) for s in samples])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def bootstrap_ci_paired(
    values_a: Sequence[float],
    values_b: Sequence[float],
    B: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Paired bootstrap CI for ``median(a) − median(b)`` on the SAME docs.

    Pair indices are resampled together, so the CI of the difference is
    (correctly) narrower than the one from two independent bootstraps —
    use it for base-vs-edit comparisons on identical document sets
    (ε, T2').
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"bootstrap_ci_paired: length mismatch {a.size} != {b.size}"
        )
    n = a.size
    if n < 2:
        raise ValueError(f"bootstrap_ci_paired: need >= 2 pairs, got {n}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("bootstrap_ci_paired: NaN/Inf in values")
    if B < 100:
        raise ValueError(f"bootstrap_ci_paired: B must be >= 100, got {B}")
    if not (0 < alpha < 1):
        raise ValueError(f"bootstrap_ci_paired: alpha must be in (0, 1), got {alpha}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    diffs = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@contextmanager
def deterministic_mode(seed: int = 0, strict: bool = False):
    """Seed torch/numpy/random; restore ALL global state on exit.

    Nesting is supported: each level restores the state captured at ITS
    entry. ``strict=True`` additionally enables
    ``torch.use_deterministic_algorithms(True)`` + cudnn.deterministic —
    caveat: some GDN kernels have no deterministic implementation and will
    raise RuntimeError in strict mode (use strict only for non-GDN paths).
    """
    torch_state = torch.random.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    np_state = np.random.get_state()
    random_state = random.getstate()
    torch_det = torch.are_deterministic_algorithms_enabled()
    cudnn_det = torch.backends.cudnn.deterministic
    if strict and torch_det:
        warnings.warn("deterministic_mode: strict=True but deterministic algorithms already enabled (redundant)", RuntimeWarning)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if strict:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        if strict:
            torch.use_deterministic_algorithms(torch_det)
            torch.backends.cudnn.deterministic = cudnn_det
        random.setstate(random_state)  # seed(tuple) is invalid — setstate restores
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


# ---------------------------------------------------------------------------
# Position pairs
# ---------------------------------------------------------------------------


def iter_position_pairs(
    S: int, mode: str = "adjacent", stride: int = 1
) -> Iterable[tuple[int, int]]:
    """Pairs (t, t') with 0 ≤ t < t' < S, in a fixed order.

    mode:
      "adjacent": (t, t+stride) for t in 0, stride, 2·stride, … (< S−stride)
      "strided":  (t, t+stride) for t = 0, 1, …, S−stride−1
      "all":      every (t, t') with t < t' — O(S²); for S=128 that is
                  8128 pairs, expensive; prefer adjacent/strided.

    Used by ε (adjacent), T2' (adjacent baseline), T1' (strided).
    """
    if not isinstance(S, int) or S < 2:
        raise ValueError(f"iter_position_pairs: need S >= 2, got {S}")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError(f"iter_position_pairs: stride must be >= 1, got {stride}")
    if mode == "adjacent":
        return ((t, t + stride) for t in range(0, S - stride, stride))
    if mode == "strided":
        return ((t, t + stride) for t in range(0, S - stride))
    if mode == "all":
        return ((t, tp) for t in range(S) for tp in range(t + 1, S))
    raise ValueError(f"iter_position_pairs: unknown mode {mode!r} (adjacent/strided/all)")
