"""nulls_and_stats / tag / profiles / behavior-gate / M0 hub driver.
Covers ТЗ §IV.6 (MP Monte-Carlo, plaintiff rule) and the §II.1/§II.10 contracts."""

import re
from pathlib import Path

import numpy as np
import pytest

from prism.invariants import INVARIANTS, run_invariants
from prism.nulls_and_stats import (
    mp_lambda_max_mc, mp_upper_edge, operator_null, paired_bootstrap,
    permutation_test, reduction_control, resolution_floor,
)
from prism.profiles import G_l, R_can, R_raw, r_cross
from prism.tag import IDENTITY_FIELDS, make_tag


def test_tag_identity_fields_enforced():
    kw = dict(protocol="natural", slice="raw", window="full", side="right",
              p_mode="truncated_U128", p_slice="p_t", layer=23, rank=16,
              norm_type="energy", source="cache", status="measured")
    t = make_tag(**kw)
    assert t.norm_type == "energy"
    for f in IDENTITY_FIELDS:
        bad = dict(kw, **{f: None})
        with pytest.raises(ValueError, match="identity-relevant"):
            make_tag(**bad)
    with pytest.raises(ValueError, match="norm_type"):
        make_tag(**dict(kw, norm_type="magnitude"))
    with pytest.raises(ValueError, match="must be int"):
        make_tag(**dict(kw, layer="23"))


def test_mp_brackets_monte_carlo_is_plaintiff():
    """§IV.6: MP edge (1+√(p/n))² at n=32 reproduces the ТЗ's λ+ (2.05/2.25);
    MC brackets the finite-n λmax. MC is the plaintiff (user ruling)."""
    assert mp_upper_edge(6, 32) == pytest.approx(2.05, abs=0.005)
    assert mp_upper_edge(8, 32) == pytest.approx(2.25, abs=1e-3)
    mc = mp_lambda_max_mc(6, 32, B=300, rng=np.random.default_rng(0))
    assert mc["q05"] < mc["median"] < mc["mean"] < mc["q95"] + 1e-9
    assert mc["mean"] < mc["mp_edge"]                # finite-n mean below edge
    assert abs(mc["mean"] - 2.05) < 0.5              # the ТЗ's n=32 edge
    # Патч v2 §2.1: вторые числа ранних версий — это n=192 (1.3848 для p=6),
    # а 1.45 отозвано (истинное n=192 для p=8 — 1.4499); MC — калибровка,
    # docs/E40_math_core_answers.md §4.
    assert abs(mc["mean"] - 1.38) > 0.1


def test_operator_nulls():
    rng = np.random.default_rng(3)
    O = np.eye(8) * 0.5
    X = rng.standard_normal((4, 8, 6)) * 3
    iid = operator_null(O, X, kind="iid", rng=rng)
    assert iid.shape == X.shape
    assert abs(float((iid ** 2).sum()) - float((X ** 2).sum())) < 1e-6 * (X ** 2).sum()
    cm = operator_null(O, X, kind="content_matched", rng=rng)
    assert np.allclose(np.linalg.norm(X, axis=-1), np.linalg.norm(cm, axis=-1))
    with pytest.raises(ValueError):
        operator_null(O, X, kind="bogus")


def test_reduction_control_and_bootstrap_and_floor():
    rng = np.random.default_rng(4)
    x = np.arange(60.0)
    y = 3 * x + rng.standard_normal(60) * 0.5
    rc = reduction_control(y, x)
    assert rc["r2_trivial"] > 0.99
    assert abs(rc["residual"]).max() < 5
    docs = rng.standard_normal((20,)) * 2 + 1.0
    bs = paired_bootstrap(docs, lambda d: float(d.mean()), B=400, seed=1)
    assert bs["ci_low"] <= bs["point"] <= bs["ci_high"]
    p = permutation_test(0.9, rng.standard_normal(500))
    assert 0 < p <= 1
    fl = resolution_floor(lambda s: float(s.mean()), rng.standard_normal((32,)),
                          subsamples=(8, 16), reps=30, seed=0)
    assert fl["floor_by_size"][8] > fl["floor_by_size"][16]  # floor shrinks with size


def test_profiles():
    rng = np.random.default_rng(5)
    J = rng.standard_normal((24, 12))
    assert 0 < R_raw(J) <= 1.0
    x = rng.standard_normal(24) * 2
    p = np.exp(x) / np.exp(x).sum()
    assert 0 < R_can(J, p) <= 1.0
    assert G_l(4.0, 1.0) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        G_l(0.0, 1.0)
    assert r_cross([2.0, 1.0, 0.0], [0.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert r_cross([2.0, 1.5], [0.0, 0.5]) is None


def test_all_invariants_pass():
    """M0 hub driver: EVERY registered identity passes on synthetic fixtures;
    a status change in the registry is a manifest event (ТЗ §II.2)."""
    res = run_invariants(seed=0)
    failed = {k: v["detail"] for k, v in res.items() if not v["ok"]}
    assert not failed, failed
    assert {"passing", "failing", "blocked"} >= {i.status for i in INVARIANTS.values()}


def test_behavior_gate_no_import_in_kernel_and_tier0():
    """ТЗ §II.10: tier-0 (E-40.*) must not import prism.behavior; CI is
    pytest-only, so the CI import-check is this source scan. Module-level
    (column-0) imports are the kernel violation; the hub may lazily EXECUTE
    the gated module inside a check function (indented, documented)."""
    root = Path(__file__).resolve().parent.parent
    pat = re.compile(r"^(from\s+\.?behavior\b|from\s+prism\.behavior\b"
                     r"|import\s+prism\.behavior\b|from\s+prism\s+import[^#]*\bbehavior\b)")
    offenders = []
    for py in (root / "prism").glob("*.py"):
        if py.name == "behavior.py":
            continue
        for i, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if pat.match(line):
                offenders.append(f"prism/{py.name}:{i}")
    for py in sorted(set(list((root / "experiments").glob("*[Ee]40*.py"))
                         + list(root.glob("*[Ee]40*.py")))):
        if re.search(r"from\s+\.?behavior\b|from\s+prism\.behavior\b|import\s+prism\.behavior\b",
                     py.read_text(encoding="utf-8")):
            offenders.append(str(py.relative_to(root)))
    assert not offenders, f"behavior imported in tier-0/kernel: {offenders}"
