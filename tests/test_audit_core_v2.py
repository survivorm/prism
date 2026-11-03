"""Regression checks for the core-code audits (docs/audit3_math.md, docs/audit4.pdf).

One test per accepted finding, cheapest thing that fails if the fix is reverted:

* B5  ``procrustes_gap`` must be the Frobenius MINIMUM over O (the transposed
      optimum gives a non-zero "gap" for one subspace in two bases).
* Q1  ``effective_rank`` of a zero spectrum is 0, not the full dimension.
* Q3  ``energy_weighted`` is removed everywhere: the tag domain and the
      implemented aggregation set must agree exactly (the alias is now a loud
      error, and ``per_position`` became taggable).
* svds  a caller-supplied SVD must be bit-identical, and a stale/swapped one
      must raise (the fast path of E-40.5).
* B2  per-position Jacobians must use the prefix ``ids[:, :t+1]`` (no future
      tokens, RoPE position = t).

Tiny model only — no caches, no data files.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tests.helpers.tiny_model import make_tiny_qwen  # noqa: E402

from prism.exp38_hi_common import procrustes_gap  # noqa: E402
from prism.kinetic import channel_energy_matrix, kinetic_decomposition  # noqa: E402
from prism.operator_checks import O1, s  # noqa: E402
from prism.svd_bundle import effective_rank  # noqa: E402
from prism.tag import TAG_DOMAINS, make_tag  # noqa: E402

K, D_S, TS_N = 8, 64, 8


# --- B5: procrustes_gap = min over orthogonal O ---------------------------


def test_procrustes_gap_is_the_minimum_and_zero_for_one_subspace():
    rng = np.random.default_rng(0)
    r, d = 4, 12
    Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
    R, _ = np.linalg.qr(rng.standard_normal((r, r)))          # change of basis
    Q1 = torch.tensor(Q)
    Q2 = torch.tensor(Q @ R)                                  # SAME subspace
    assert procrustes_gap(Q1, Q2) < 1e-9, "one subspace in two bases must give 0"

    Qa = torch.tensor(np.linalg.qr(rng.standard_normal((d, r)))[0])
    Qb = torch.tensor(np.linalg.qr(rng.standard_normal((d, r)))[0])
    got = procrustes_gap(Qa, Qb)
    best_random = min(float((Qa - Qb @ torch.tensor(O)).norm() / r)
                      for O in (np.linalg.qr(rng.standard_normal((r, r)))[0]
                                for _ in range(200)))
    assert got <= best_random + 1e-9, f"not the minimum: {got} vs {best_random}"


# --- Q1: effective_rank on a zero spectrum --------------------------------


def test_effective_rank_zero_spectrum_is_zero():
    assert effective_rank(np.zeros(8)) == 0
    assert effective_rank(np.array([2.0, 1.0, 1e-3])) == 2   # threshold 1e-2 · σ₁
    assert effective_rank(np.array([2.0, 1.9, 1e-3])) == 2


# --- Q3 (strict): energy_weighted is GONE from both the tag domain and the
# aggregator; the vocabulary and the implemented set must agree exactly ---


def test_energy_weighted_removed_and_tag_domain_equals_aggregator():
    rng = np.random.default_rng(1)
    p = rng.random((3, 16))
    p /= p.sum(1, keepdims=True)
    J = rng.standard_normal((3, 16, 24))
    assert "energy_weighted" not in TAG_DOMAINS["aggregation"]
    with pytest.raises(ValueError, match="removed"):
        O1(J, p, "energy_weighted")                # loud, never silently aliased
    with pytest.raises(ValueError, match="removed"):
        s(J, p, "energy_weighted")
    with pytest.raises(ValueError, match="aggregation"):
        make_tag(aggregation="energy_weighted", slice="raw", p_mode="full_softmax",
                 norm_type="energy")
    # per_position really is another number and must be taggable — it was
    # implemented but absent from the domain, the mirror image of the alias bug
    assert "per_position" in TAG_DOMAINS["aggregation"]
    make_tag(aggregation="per_position", slice="raw", p_mode="full_softmax",
             norm_type="energy")
    per = O1(J, p, "per_position")
    assert per.shape == (3,)
    assert abs(float(per.mean()) - O1(J, p, "ratio_of_means")) > 0.0


# --- svds fast path of the kinetic core ----------------------------------


def test_supplied_svds_are_bit_identical_and_validated():
    rng = np.random.default_rng(2)
    J0 = rng.standard_normal((24, 40))
    J1 = J0 + 0.4 * rng.standard_normal((24, 40))
    sv = (np.linalg.svd(J0, full_matrices=False), np.linalg.svd(J1, full_matrices=False))
    assert str(kinetic_decomposition(J0, J1, r=8)) == str(
        kinetic_decomposition(J0, J1, r=8, svds=sv))
    assert np.array_equal(channel_energy_matrix(J0, J1, r=8),
                          channel_energy_matrix(J0, J1, r=8, svds=sv))
    with pytest.raises(ValueError, match="svds"):            # swapped triples
        kinetic_decomposition(J0, J1, r=8, svds=(sv[1], sv[0]))
    with pytest.raises(ValueError, match="svds"):            # wrong shapes
        U, S, W = sv[0]
        kinetic_decomposition(J0, J1, r=8, svds=((U[:2], S, W), sv[1]))


# --- B2: per-position Jacobians must not see future tokens ----------------


def _tiny():
    torch.manual_seed(0)
    return make_tiny_qwen(n_layers=4, hidden=D_S)


def test_doc_js_linearizes_at_its_own_position():
    """``doc_js`` must linearize position ``t`` with the prefix ``0..t``.

    Feeding the full sequence with ``h[t]`` puts tokens ``t..S-2`` in the KV
    cache and stamps RoPE position ``S-1`` — the reconstructed J is then the
    Jacobian of another point (audit4 B2: 18.7 % off already at t=S-2)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "experiments"))
    from _exp_common import doc_js  # noqa: PLC0415

    model = _tiny()
    S, L, ts = 6, 2, list(range(TS_N))
    ids = torch.randint(0, 40, (1, S))
    sample = {"input_ids": ids, "attention_mask": torch.ones(1, S, dtype=torch.long)}
    with torch.no_grad():
        out = model(**sample, output_hidden_states=True, use_cache=False)
    h = out.hidden_states[L + 1][0].float().cpu()
    p = torch.softmax(out.logits[0].float()[:, ts], dim=-1).cpu()

    js = doc_js(model, L, sample, ts, h, p, n_pos=2)
    assert len(js) == 2
    # the last position is the only one a full-sequence call may serve
    from _exp_common import gauge_free_J  # noqa: PLC0415
    last_full = gauge_free_J(model, L, ids, sample["attention_mask"], ts,
                             h[S - 1], p[S - 1])
    assert torch.allclose(js[-1], last_full, atol=1e-5, rtol=1e-4)
    # the second-to-last must NOT be the full-sequence answer (it was the bug)
    leak = gauge_free_J(model, L, ids, sample["attention_mask"], ts,
                        h[S - 2], p[S - 2])
    rel = float((js[0] - leak).norm() / leak.norm())
    assert rel > 0.02, f"doc_js looks like the future-leaking variant ({rel:.1e})"
    # and it must be the prefix answer
    pref = gauge_free_J(model, L, ids[:, :S - 1], sample["attention_mask"][:, :S - 1],
                        ts, h[S - 2], p[S - 2])
    assert torch.allclose(js[0], pref, atol=1e-5, rtol=1e-4)
