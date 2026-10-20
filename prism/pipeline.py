"""Experiment specification + thin driver (TZ part 4).

``ExperimentSpec`` freezes every parameter of a run; ``run_spec``
orchestrates validation, environment capture, seeds, output paths,
meta.json/results.json with atomic writes, exception diagnostics and
idempotent reuse. ``run_spec`` does NOT load a model (that is
``exp_fn``'s job) and does NOT import any experiment-specific module
(spectral/holonomy/noise/permutation/topology) — if it did, that would
be an architectural error.

paper.md §19.A — spec structure.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from .cache import deterministic_mode
from .interventions import EditSpec

__all__ = [
    "ExperimentSpec",
    "OutputArtifacts",
    "run_spec",
    "run_many",
    "validate_spec",
    "load_artifacts",
    "default_output_subdir",
]

_VALID_EXPERIMENTS = ("T0", "eps", "T2", "T1", "delta")


def _json_safe(obj: Any) -> Any:
    """Recursive JSON conversion (TZ §3.3).

    np.ndarray/torch.Tensor -> lists; NaN/Inf -> "nan"/"inf"/"-inf";
    np scalars -> Python scalars; dict keys -> str; anything else ->
    "<unserializable: Type>" (diagnostics, not strict validation).
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if np.isnan(obj):
            return "nan"
        if np.isposinf(obj):
            return "inf"
        if np.isneginf(obj):
            return "-inf"
        return obj
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return _json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return f"<unserializable: {type(obj).__name__}>"


def _full_hash(spec: "ExperimentSpec") -> str:
    payload = json.dumps(spec.to_meta_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExperimentSpec:
    """Frozen run specification (TZ §2).

    ``config`` is a Mapping (typically MappingProxyType) with
    JSON-serializable values; ``strict=True`` in config enables
    ``deterministic_mode(strict=True)`` (do NOT default to strict —
    some GDN kernels have no deterministic implementation).
    """

    experiment: str
    edit: EditSpec
    layers: tuple[int, ...]
    token_set: tuple[int, ...]
    n_docs: int
    S: int
    r: int = 16
    n_theta: int = 16
    n_steps: int = 20
    epsilon_ratio: float = 0.1
    B_perm: int = 1000
    B_boot: int = 1000
    k_sigma: float = 5.0
    alpha: float = 0.05
    seed: int = 0
    cache_root: str = "results/.cache"
    output_root: str = "results"
    model_name: str = "Qwen/Qwen3.5-0.8B"
    config: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def to_meta_dict(self) -> dict:
        """JSON-serializable dict, EXCLUDING cache_root/output_root
        (paths do not affect results)."""
        d = {
            "experiment": self.experiment,
            "edit": {
                "name": self.edit.name,
                "norm_frobenius": _json_safe(self.edit.norm_frobenius),
                "projection_onto_top_r": _json_safe(self.edit.projection_onto_top_r),
            },
            "layers": list(self.layers),
            "token_set": list(self.token_set),
            "n_docs": self.n_docs,
            "S": self.S,
            "r": self.r,
            "n_theta": self.n_theta,
            "n_steps": self.n_steps,
            "epsilon_ratio": self.epsilon_ratio,
            "B_perm": self.B_perm,
            "B_boot": self.B_boot,
            "k_sigma": self.k_sigma,
            "alpha": self.alpha,
            "seed": self.seed,
            "model_name": self.model_name,
            "config": {
                str(k): _json_safe(v) for k, v in sorted(self.config.items())
            },
        }
        return d

    def to_key(self) -> str:
        """Stable 12-hex identifier (SHA-256 prefix) of the spec.

        Path-insensitive (cache_root/output_root excluded); sensitive to
        every result-affecting field including config and the edit
        identity (name/norm/token_set).
        """
        return _full_hash(self)[:12]


@dataclass(frozen=True)
class OutputArtifacts:
    """Handle to a saved (or reused) run: results, meta, path, key."""

    spec: ExperimentSpec
    results: dict
    meta: dict
    path: str
    spec_key: str
    reused: bool


def validate_spec(spec: ExperimentSpec) -> None:
    """TZ §2.1 field checks.

    Note: ``pipeline`` does NOT know ``n_layers`` (the model is loaded
    only inside ``exp_fn``); it checks ``l >= 0`` and ``len(layers) >= 1``.
    Full ``l < n_layers`` validation is the exp_fn's responsibility.

    Raises:
        ValueError: any field outside its contract.
    """
    if spec.experiment not in _VALID_EXPERIMENTS:
        raise ValueError(
            f"experiment must be one of {_VALID_EXPERIMENTS}, "
            f"got {spec.experiment!r}"
        )
    if tuple(spec.edit.token_set) != tuple(spec.token_set):
        raise ValueError(
            f"edit.token_set {tuple(spec.edit.token_set)} != "
            f"spec.token_set {tuple(spec.token_set)}"
        )
    if spec.edit.delta_W is not None and \
            spec.edit.delta_W.shape[0] != len(spec.token_set):
        raise ValueError(
            f"edit.delta_W rows {spec.edit.delta_W.shape[0]} != "
            f"len(token_set) {len(spec.token_set)}"
        )
    if len(spec.token_set) < 2:
        raise ValueError(f"len(token_set) must be >= 2, got {len(spec.token_set)}")
    if len(spec.layers) < 1:
        raise ValueError("layers must be non-empty")
    for l in spec.layers:
        if not (isinstance(l, int) and l >= 0):
            raise ValueError(f"layer must be a non-negative int, got {l!r}")
    if spec.n_docs < 1:
        raise ValueError(f"n_docs must be >= 1, got {spec.n_docs}")
    if spec.S < 2:
        raise ValueError(f"S must be >= 2, got {spec.S}")
    if spec.r < 1:
        raise ValueError(f"r must be >= 1, got {spec.r}")
    if spec.r > len(spec.token_set):
        raise ValueError(f"r = {spec.r} > len(token_set) = {len(spec.token_set)}")
    if spec.n_theta < 4:
        raise ValueError(f"n_theta must be >= 4, got {spec.n_theta}")
    if spec.n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {spec.n_steps}")
    if spec.epsilon_ratio <= 0:
        raise ValueError(f"epsilon_ratio must be > 0, got {spec.epsilon_ratio}")
    if spec.B_perm < 100:
        raise ValueError(f"B_perm must be >= 100, got {spec.B_perm}")
    if spec.B_boot < 100:
        raise ValueError(f"B_boot must be >= 100, got {spec.B_boot}")
    if spec.k_sigma <= 0:
        raise ValueError(f"k_sigma must be > 0, got {spec.k_sigma}")
    if not (0.0 < spec.alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {spec.alpha}")
    if spec.seed < 0:
        raise ValueError(f"seed must be >= 0, got {spec.seed}")
    if not spec.cache_root:
        raise ValueError("cache_root must be non-empty")
    if not spec.output_root:
        raise ValueError("output_root must be non-empty")
    if not spec.model_name:
        raise ValueError("model_name must be non-empty")


def default_output_subdir(spec: ExperimentSpec) -> str:
    """``{edit.name}/L{min:02d}-{max:02d}_r{r}_seed{seed}`` (TZ §4.1)."""
    return (
        f"{spec.edit.name}/"
        f"L{min(spec.layers):02d}-{max(spec.layers):02d}"
        f"_r{spec.r}_seed{spec.seed}"
    )


def _expected_output_path(spec: ExperimentSpec, output_subdir: str | None) -> str:
    return os.path.join(
        spec.output_root, spec.experiment,
        output_subdir or default_output_subdir(spec),
    )


def _ensure_output_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except PermissionError as e:
        raise RuntimeError(
            f"cannot create output dir {path!r} (read-only / Kaggle FUSE?): {e}"
        ) from e


def _write_json_atomic(path: str, obj: Any) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(f"cannot read/parse JSON at {path!r}: {e}") from e


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _detect_env() -> dict:
    env = {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "transformers": "unknown",
        "device": "cpu",
        "git_commit": "unknown",
    }
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        env["device"] = f"cuda:{torch.cuda.current_device()}"
    elif getattr(torch.backends, "mps", None) is not None and \
            torch.backends.mps.is_available():
        env["device"] = "mps"
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, check=True,
        )
        env["git_commit"] = out.stdout.strip()
    except Exception:
        pass
    return env


def _reconstruct_edit(meta_spec: dict) -> EditSpec:
    """EditSpec from meta metadata; delta_W is ALWAYS None here (it is
    not stored — size). Re-applying the edit requires the caller's own
    EditSpec (TZ §7)."""
    e = meta_spec.get("edit", {})
    token_set = tuple(meta_spec.get("token_set", ()))
    return EditSpec(
        name=str(e.get("name", "unknown")),
        delta_W=None,
        token_set=token_set,
        target_layer="lm_head",
        norm_frobenius=float(e.get("norm_frobenius", 0.0)),
        projection_onto_top_r=float(e.get("projection_onto_top_r", 0.0)),
        seed=None,
        meta={"reconstructed": True,
              "note": "delta_W is not stored in meta (size); "
                      "re-apply with the caller's own EditSpec"},
    )


def _load_or_reuse(meta_path: str, spec_key: str) -> OutputArtifacts | None:
    """Reuse a complete run whose spec_key matches; None otherwise."""
    if not os.path.exists(meta_path):
        return None
    try:
        meta = _read_json(meta_path)
    except RuntimeError:
        return None  # corrupt meta: re-run
    if meta.get("spec_key") != spec_key or not meta.get("complete"):
        return None
    out_dir = os.path.dirname(meta_path)
    results_path = os.path.join(out_dir, "results.json")
    results = _read_json(results_path) if os.path.exists(results_path) else {}
    if not os.path.exists(results_path):
        meta["results_available"] = False
    spec = _spec_from_meta(meta)
    return OutputArtifacts(
        spec=spec, results=results, meta=meta, path=meta_path,
        spec_key=spec_key, reused=True,
    )


def _spec_from_meta(meta: dict) -> ExperimentSpec:
    ms = meta.get("spec", {})
    edit = _reconstruct_edit(ms)
    return ExperimentSpec(
        experiment=ms.get("experiment", ""),
        edit=edit,
        layers=tuple(ms.get("layers", ())),
        token_set=tuple(ms.get("token_set", ())),
        n_docs=int(ms.get("n_docs", 0)),
        S=int(ms.get("S", 0)),
        r=int(ms.get("r", 16)),
        n_theta=int(ms.get("n_theta", 16)),
        n_steps=int(ms.get("n_steps", 20)),
        epsilon_ratio=float(ms.get("epsilon_ratio", 0.1)),
        B_perm=int(ms.get("B_perm", 1000)),
        B_boot=int(ms.get("B_boot", 1000)),
        k_sigma=float(ms.get("k_sigma", 5.0)),
        alpha=float(ms.get("alpha", 0.05)),
        seed=int(ms.get("seed", 0)),
        model_name=str(ms.get("model_name", "")),
        config=MappingProxyType({k: v for k, v in ms.get("config", {}).items()}),
    )


def run_spec(
    spec: ExperimentSpec,
    exp_fn: Callable[[ExperimentSpec], dict],
    output_subdir: str | None = None,
    skip_if_exists: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
) -> OutputArtifacts:
    """Validate, seed, run exp_fn, save results.json + meta.json (TZ §3).

    meta is written BEFORE exp_fn (complete=False) so a crash/Ctrl-C
    leaves on-disk diagnostics; written again after success. Atomic
    (tmp + os.replace). exp_fn must return a JSON-convertible dict and
    must NOT write to the output dir itself.

    Seeds: deterministic_mode(seed, strict=config.get("strict", False)) —
    strict is NEVER the default (GDN compatibility).

    Args:
        spec: the frozen spec.
        exp_fn: Callable[[ExperimentSpec], dict] — all experiment logic.
        output_subdir: override default_output_subdir(spec).
        skip_if_exists: reuse a complete meta with matching spec_key.
        dry_run: validate + print plan, do not call exp_fn.
        verbose: progress prints.

    Returns:
        OutputArtifacts (reused=True when loaded from disk).

    Raises:
        ValueError: spec invalid.
        TypeError: exp_fn returned a non-dict (meta marked incomplete).
        Any exception from exp_fn: re-raised AFTER meta diagnostics.
    """
    validate_spec(spec)
    spec_key = spec.to_key()
    output_path = _expected_output_path(spec, output_subdir)
    meta_path = os.path.join(output_path, "meta.json")

    if dry_run:
        if verbose:
            print(f"[run_spec] DRY RUN experiment={spec.experiment} "
                  f"key={spec_key} path={output_path}")
            print(json.dumps(spec.to_meta_dict(), indent=2, ensure_ascii=False,
                             default=str))
        return OutputArtifacts(spec=spec, results={}, meta={}, path="",
                               spec_key=spec_key, reused=False)

    _ensure_output_dir(output_path)
    if skip_if_exists:
        existing = _load_or_reuse(meta_path, spec_key)
        if existing is not None and verbose:
            print(f"[run_spec] reused experiment={spec.experiment} "
                  f"key={spec_key} path={meta_path}")
        if existing is not None:
            return existing

    meta = {
        "spec": spec.to_meta_dict(),
        "spec_key": spec_key,
        "spec_key_full": _full_hash(spec),
        "experiment": spec.experiment,
        "edit_name": spec.edit.name,
        "env": _detect_env() | {"timestamp_start": _now_iso()},
        "complete": False,
        "error": None,
        "reused": False,
    }
    if verbose:
        print(f"[run_spec] experiment={spec.experiment} key={spec_key} "
              f"path={output_path}")

    t0 = time.perf_counter()
    try:
        with deterministic_mode(seed=spec.seed,
                                strict=bool(spec.config.get("strict", False))):
            results = exp_fn(spec)
    except BaseException as e:  # KeyboardInterrupt included
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["env"]["timestamp_end"] = _now_iso()
        meta["env"]["duration_sec"] = time.perf_counter() - t0
        try:
            _write_json_atomic(meta_path, meta)
        except OSError:
            pass  # diagnostics best-effort; re-raise the original
        raise

    if not isinstance(results, dict):
        meta["error"] = (
            f"expected dict from exp_fn, got {type(results).__name__}"
        )
        meta["env"]["timestamp_end"] = _now_iso()
        meta["env"]["duration_sec"] = time.perf_counter() - t0
        _write_json_atomic(meta_path, meta)
        raise TypeError(meta["error"])

    meta["complete"] = True
    meta["env"]["timestamp_end"] = _now_iso()
    meta["env"]["duration_sec"] = time.perf_counter() - t0
    _write_json_atomic(os.path.join(output_path, "results.json"), results)
    _write_json_atomic(meta_path, meta)
    if verbose:
        print(f"[run_spec] done in {meta['env']['duration_sec']:.1f}s")

    return OutputArtifacts(spec=spec, results=results, meta=meta,
                           path=meta_path, spec_key=spec_key, reused=False)


def run_many(
    specs: Sequence[ExperimentSpec],
    exp_fn: Callable[[ExperimentSpec], dict],
    output_subdir_fn: Callable[[ExperimentSpec], str] | None = None,
    skip_if_exists: bool = True,
    verbose: bool = True,
) -> list[OutputArtifacts]:
    """Sequential batch of run_spec (TZ §3.5).

    Convenience batch wrapper — it adds NO new logic: each spec goes
    through the identical run_spec path (validate, cache-key, meta,
    seeds). The only added behavior is that an exception in one spec
    ABORTS the remaining batch (not swallowed); wrap in try/except for
    partial batches.
    """
    out = []
    for spec in specs:
        out.append(run_spec(
            spec, exp_fn,
            output_subdir=output_subdir_fn(spec) if output_subdir_fn else None,
            skip_if_exists=skip_if_exists,
            verbose=verbose,
        ))
    return out


def load_artifacts(path: str) -> OutputArtifacts:
    """Inverse of run_spec's save (TZ §3.4).

    ``path`` — meta.json or the directory containing it. The spec is
    reconstructed from meta (delta_W is None — it is not stored). A
    meta with complete=False raises RuntimeError (incomplete run).
    results.json missing → results={} and meta["results_available"]=False.

    Raises:
        RuntimeError: broken JSON, missing meta, incomplete run.
    """
    meta_path = (os.path.join(path, "meta.json")
                 if os.path.isdir(path) else path)
    if not os.path.exists(meta_path):
        raise RuntimeError(f"no meta.json at {meta_path!r}")
    meta = _read_json(meta_path)
    if not meta.get("complete"):
        raise RuntimeError(
            f"meta at {meta_path!r} is incomplete "
            f"(error: {meta.get('error')!r}); re-run the experiment"
        )
    results_path = os.path.join(os.path.dirname(meta_path), "results.json")
    results = _read_json(results_path) if os.path.exists(results_path) else {}
    if not os.path.exists(results_path):
        meta["results_available"] = False
    spec = _spec_from_meta(meta)
    return OutputArtifacts(
        spec=spec, results=results, meta=meta, path=meta_path,
        spec_key=str(meta.get("spec_key", spec.to_key())), reused=True,
    )
