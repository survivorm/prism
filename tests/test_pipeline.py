"""Selfcheck for prism/pipeline.py (TZ part 4, §6, 25 items)."""
import json
import os

import numpy as np
import pytest
import torch
from types import MappingProxyType

from prism.interventions import make_base
from prism.pipeline import (
    ExperimentSpec,
    OutputArtifacts,
    default_output_subdir,
    load_artifacts,
    run_many,
    run_spec,
    validate_spec,
)

TS = (101, 102, 103, 104)


def _spec(**over) -> ExperimentSpec:
    base = dict(
        experiment="T0",
        edit=make_base(TS),
        layers=(18, 19, 20, 21, 22, 23),
        token_set=TS,
        n_docs=8,
        S=32,
        r=4,
    )
    base.update(over)
    return ExperimentSpec(**base)


# --- 6.1: valid spec -----------------------------------------------------------

def test_validate_valid():
    validate_spec(_spec())  # must not raise


# --- 6.2: invalid experiment ----------------------------------------------------

def test_validate_bad_experiment():
    with pytest.raises(ValueError, match="experiment"):
        validate_spec(_spec(experiment="unknown"))


# --- 6.3: edit/token_set mismatch ------------------------------------------------

def test_validate_token_set_mismatch():
    with pytest.raises(ValueError, match="token_set"):
        validate_spec(_spec(token_set=(101, 102, 103)))  # edit has 4


# --- 6.4: layers ------------------------------------------------------------------

def test_validate_layers():
    with pytest.raises(ValueError, match="non-negative"):
        validate_spec(_spec(layers=(-1,)))
    with pytest.raises(ValueError, match="non-empty"):
        validate_spec(_spec(layers=()))


# --- 6.5: r > K ---------------------------------------------------------------------

def test_validate_r_too_big():
    with pytest.raises(ValueError, match="r = 5"):
        validate_spec(_spec(r=5))


# --- 6.6: alpha -----------------------------------------------------------------------

def test_validate_alpha():
    with pytest.raises(ValueError, match="alpha"):
        validate_spec(_spec(alpha=0.0))
    with pytest.raises(ValueError, match="alpha"):
        validate_spec(_spec(alpha=1.0))
    with pytest.raises(ValueError, match="B_perm"):
        validate_spec(_spec(B_perm=50))
    with pytest.raises(ValueError, match="n_docs"):
        validate_spec(_spec(n_docs=0))
    with pytest.raises(ValueError, match="S must be"):
        validate_spec(_spec(S=1))
    with pytest.raises(ValueError, match="seed"):
        validate_spec(_spec(seed=-1))


# --- 6.7-6.10: to_key -------------------------------------------------------------------

def test_to_key_determinism():
    a = _spec()
    b = _spec()
    assert a.to_key() == b.to_key()
    assert len(a.to_key()) == 12
    int(a.to_key(), 16)  # hex


def test_to_key_seed_sensitive():
    assert _spec(seed=0).to_key() != _spec(seed=1).to_key()


def test_to_key_path_insensitive():
    assert _spec(cache_root="a", output_root="x").to_key() == \
        _spec(cache_root="b", output_root="y").to_key()


def test_to_key_config_sensitive():
    assert _spec(config=MappingProxyType({"x": 1})).to_key() != \
        _spec(config=MappingProxyType({"x": 2})).to_key()


# --- 6.11: run_spec basic ---------------------------------------------------------------

def test_run_spec_basic(tmp_path):
    spec = _spec(output_root=str(tmp_path))
    calls = {"n": 0}

    def exp_fn(s):
        calls["n"] += 1
        return {"x": 42}

    art = run_spec(spec, exp_fn, verbose=False)
    assert art.results == {"x": 42}
    assert art.reused is False
    meta_path = os.path.join(
        str(tmp_path), "T0", default_output_subdir(spec), "meta.json"
    )
    assert os.path.exists(meta_path)
    assert art.meta["complete"] is True
    assert calls["n"] == 1
    # meta.json on disk is valid JSON with the env block
    meta = json.load(open(meta_path))
    assert meta["spec_key"] == art.spec_key
    assert meta["env"]["torch"] == torch.__version__
    assert meta["complete"] is True
    results = json.load(open(os.path.join(os.path.dirname(meta_path),
                                          "results.json")))
    assert results == {"x": 42}


# --- 6.12: skip_if_exists ---------------------------------------------------------------

def test_run_spec_reuse(tmp_path):
    spec = _spec(output_root=str(tmp_path))
    calls = {"n": 0}

    def exp_fn(s):
        calls["n"] += 1
        return {"x": 1}

    run_spec(spec, exp_fn, verbose=False)
    art2 = run_spec(spec, exp_fn, verbose=False)
    assert art2.reused is True
    assert calls["n"] == 1
    assert art2.results == {"x": 1}


# --- 6.13: skip_if_exists=False -----------------------------------------------------------

def test_run_spec_no_skip(tmp_path):
    spec = _spec(output_root=str(tmp_path))
    calls = {"n": 0}

    def exp_fn(s):
        calls["n"] += 1
        return {"x": 1}

    run_spec(spec, exp_fn, verbose=False)
    art2 = run_spec(spec, exp_fn, skip_if_exists=False, verbose=False)
    assert art2.reused is False
    assert calls["n"] == 2


# --- 6.14: dry_run ---------------------------------------------------------------------------

def test_run_spec_dry_run(tmp_path):
    spec = _spec(output_root=str(tmp_path))
    calls = {"n": 0}
    art = run_spec(spec, lambda s: calls.__setitem__("n", 1),
                   dry_run=True, verbose=False)
    assert art.results == {}
    assert art.path == ""
    assert art.reused is False
    assert calls["n"] == 0
    assert not os.path.exists(os.path.join(str(tmp_path), "T0"))  # nothing written


# --- 6.15: non-dict results ----------------------------------------------------------------------

def test_run_spec_non_dict(tmp_path):
    spec = _spec(output_root=str(tmp_path))
    with pytest.raises(TypeError, match="expected dict"):
        run_spec(spec, lambda s: 42, verbose=False)
    meta_path = os.path.join(str(tmp_path), "T0",
                             default_output_subdir(spec), "meta.json")
    meta = json.load(open(meta_path))
    assert meta["complete"] is False
    assert "expected dict" in meta["error"]


# --- 6.16: exp_fn raises ---------------------------------------------------------------------------

def test_run_spec_exception(tmp_path):
    spec = _spec(output_root=str(tmp_path))

    def boom(s):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_spec(spec, boom, verbose=False)
    meta_path = os.path.join(str(tmp_path), "T0",
                             default_output_subdir(spec), "meta.json")
    assert os.path.exists(meta_path)
    meta = json.load(open(meta_path))
    assert meta["complete"] is False
    assert "boom" in meta["error"]
    # a subsequent run must NOT reuse the incomplete meta
    art = run_spec(spec, lambda s: {"ok": True}, verbose=False)
    assert art.reused is False


# --- 6.17: NaN/Inf in results ---------------------------------------------------------------------------

def test_run_spec_nan_inf(tmp_path):
    spec = _spec(output_root=str(tmp_path))
    art = run_spec(spec, lambda s: {"x": float("nan"), "y": float("inf"),
                                    "z": float("-inf")}, verbose=False)
    # on-disk JSON is VALID (no bare NaN tokens)
    results_path = os.path.join(os.path.dirname(art.path), "results.json")
    loaded = json.load(open(results_path))
    assert loaded["x"] == "nan"
    assert loaded["y"] == "inf"
    assert loaded["z"] == "-inf"


# --- 6.18: ndarray/tensor in results -----------------------------------------------------------------------

def test_run_spec_arrays(tmp_path):
    spec = _spec(output_root=str(tmp_path))
    art = run_spec(spec, lambda s: {"arr": np.array([1, 2]),
                                    "t": torch.tensor([3.0, 4.0])},
                   verbose=False)
    results_path = os.path.join(os.path.dirname(art.path), "results.json")
    loaded = json.load(open(results_path))
    assert loaded["arr"] == [1, 2]
    assert loaded["t"] == [3.0, 4.0]


# --- 6.19: seeds determinism ------------------------------------------------------------------------------

def test_run_spec_seeds(tmp_path):
    calls = {"n": 0}

    def exp_fn(s):
        calls["n"] += 1
        return {"r": float(torch.randn(1))}

    spec0 = _spec(output_root=str(tmp_path), seed=0)
    r0a = run_spec(spec0, exp_fn, skip_if_exists=False, verbose=False).results["r"]
    r0b = run_spec(spec0, exp_fn, skip_if_exists=False, verbose=False).results["r"]
    assert r0a == r0b
    spec1 = _spec(output_root=str(tmp_path), seed=1)
    r1 = run_spec(spec1, exp_fn, skip_if_exists=False, verbose=False).results["r"]
    assert r1 != r0a


# --- 6.20: strict=True ------------------------------------------------------------------------------------

def test_run_spec_strict(tmp_path):
    spec = _spec(output_root=str(tmp_path),
                 config=MappingProxyType({"strict": True}))
    art = run_spec(spec, lambda s: {"r": float(torch.randn(1))}, verbose=False)
    assert art.meta["complete"] is True


# --- 6.21: load_artifacts round-trip --------------------------------------------------------------------------

def test_load_artifacts_roundtrip(tmp_path):
    spec = _spec(output_root=str(tmp_path))
    art = run_spec(spec, lambda s: {"x": 42, "arr": np.array([1.0])},
                   verbose=False)
    loaded = load_artifacts(art.path)
    assert loaded.reused is True
    assert loaded.spec_key == art.spec_key
    assert loaded.results["x"] == 42
    assert loaded.results["arr"] == [1.0]
    assert loaded.spec.experiment == spec.experiment
    assert loaded.spec.seed == spec.seed
    assert loaded.spec.edit.delta_W is None  # not reconstructed
    # also by directory
    loaded2 = load_artifacts(os.path.dirname(art.path))
    assert loaded2.spec_key == art.spec_key


def test_load_artifacts_incomplete(tmp_path):
    spec = _spec(output_root=str(tmp_path))
    with pytest.raises(ValueError):
        run_spec(spec, lambda s: (_ for _ in ()).throw(ValueError("x")),
                 verbose=False)
    meta_dir = os.path.join(str(tmp_path), "T0", default_output_subdir(spec))
    with pytest.raises(RuntimeError, match="incomplete"):
        load_artifacts(meta_dir)


# --- 6.22-6.23: run_many --------------------------------------------------------------------------------------

def test_run_many(tmp_path):
    specs = [_spec(output_root=str(tmp_path), seed=i) for i in range(3)]
    arts = run_many(specs, lambda s: {"seed": s.seed}, verbose=False)
    assert len(arts) == 3
    for art, spec in zip(arts, specs):
        assert art.results["seed"] == spec.seed
        assert os.path.exists(art.path)


def test_run_many_subdir_fn(tmp_path):
    specs = [_spec(output_root=str(tmp_path), seed=i) for i in range(2)]
    arts = run_many(specs, lambda s: {"ok": True},
                    output_subdir_fn=lambda s: f"custom_{s.seed}",
                    verbose=False)
    for art, spec in zip(arts, specs):
        assert f"custom_{spec.seed}" in art.path
        assert os.path.exists(art.path)


def test_run_many_aborts(tmp_path):
    specs = [_spec(output_root=str(tmp_path), seed=i) for i in range(3)]
    with pytest.raises(ZeroDivisionError):
        run_many(specs, lambda s: 1 / 0 if s.seed == 1 else {"ok": True},
                 verbose=False)


# --- 6.24: _json_safe recursive -----------------------------------------------------------------------------------

def test_json_safe_recursive():
    from prism.pipeline import _json_safe
    out = _json_safe({"a": [np.array([1, float("nan")])],
                      "b": {"c": torch.tensor([2.0])}})
    assert out == {"a": [[1, "nan"]], "b": {"c": [2.0]}}
    # np scalars
    assert _json_safe({"i": np.int64(5), "f": np.float32(1.5)}) == \
        {"i": 5, "f": 1.5}
    # unserializable
    assert _json_safe(object()).startswith("<unserializable:")
    # tuple / set
    assert _json_safe((1, 2)) == [1, 2]


# --- 6.25: default_output_subdir ---------------------------------------------------------------------------------------

def test_default_output_subdir():
    spec = _spec(layers=(6, 12, 18))
    assert default_output_subdir(spec) == "base/L06-18_r4_seed0"
