"""Column conventions of the dense J-cache producer (`experiments/exp27b_dense.py`).

exp27b (published) gathers the SAME global frequency columns at every position;
exp27c (audit4 Q10, strict) gives every position its OWN top-K. Both live in
`_forward_dense`, and this is the check that neither drifted:

* positional: stored columns/values ARE the full-softmax top-K of that position,
  and a row is a TRUNCATION (mass ≤ 1), not a renormalized distribution;
* freq: columns are gathered BY ID (regression on the published path — a silent
  change here would invalidate the whole exp27b cache line).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "experiments"))
from exp27b_dense import _forward_dense  # noqa: E402

from tests.helpers.tiny_model import make_tiny_qwen  # noqa: E402

S, K, D = 8, 5, 64          # tiny model: vocab 64, hidden 64


def _sample():
    g = torch.Generator().manual_seed(0)
    return {"input_ids": torch.randint(1, 60, (1, S), generator=g),
            "attention_mask": torch.ones(1, S, dtype=torch.long)}


def _full_softmax(m, s):
    with torch.no_grad():
        out = m(input_ids=s["input_ids"], attention_mask=s["attention_mask"])
    return torch.softmax(out.logits[0].float(), dim=-1)


def test_positional_columns_are_the_positions_own_topk():
    torch.manual_seed(0)
    m = make_tiny_qwen(n_layers=4, attn_implementation="eager").eval()
    s = _sample()
    full = _full_softmax(m, s)
    f = _forward_dense(m, s, S, [0, 1], None, K)
    p, u = f["p128"], f["utop"]
    assert p.shape == u.shape == (S, K)
    ref = full.topk(K, dim=-1)
    assert torch.equal(u, ref.indices), "columns must be that position's top-K ids"
    assert torch.allclose(p, ref.values, atol=1e-6), "values must be the softmax probs"
    assert bool((p.sum(-1) <= 1 + 1e-6).all()), "raw rows are truncations, mass <= 1"
    assert bool((p.sum(-1) < 0.999).any()), "K << vocab: some mass must fall outside"
    assert f["hs"][1].shape == (S, D)


def test_freq_mode_still_gathers_by_id():
    torch.manual_seed(0)
    m = make_tiny_qwen(n_layers=4, attn_implementation="eager").eval()
    s = _sample()
    full = _full_softmax(m, s)
    cols = [3, 7, 11, 15, 19]
    f = _forward_dense(m, s, S, [0, 1], cols, K)
    assert f["utop"] is None, "freq mode has no per-position basis"
    assert torch.allclose(f["p128"], full[:, cols], atol=1e-6), (
        "freq columns must be gathered by token id, same as the published cache")
