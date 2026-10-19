"""Selfcheck for prism/interventions.py (TZ part 2, §4.6).

20 checks on a tiny Qwen3.5 model (tests/helpers/tiny_model.py):
EditSpec (base), make_random_* (norm/projection/determinism),
make_tail_only, make_readout_pull (shape, Adam moved it, restoration,
requires_grad), apply_edit (no-op, exception restore, visibility,
nesting, wrong layer), apply_edit_and_cache (round-trip, restoration,
overwrite=False skip, meta).
"""

import json
import os

import pytest
import torch

from tests.helpers.tiny_model import make_tiny_qwen

from prism.interventions import (
    EditSpec,
    make_base,
    make_readout_pull,
    make_random_ortho,
    make_random_matched,
    make_tail_only,
    apply_edit,
    apply_edit_and_cache,
)

K, D_S, R = 8, 64, 3


def _basis():
    Q, _ = torch.linalg.qr(torch.randn(K, R))
    return Q


def _ref():
    return torch.randn(K, D_S)


def _tiny_model():
    torch.manual_seed(0)
    m = make_tiny_qwen(n_layers=4, hidden=D_S)
    # Untie lm_head (helper defaults to tie_word_embeddings=True). Production
    # 0.8B is untied; tied, Adam in make_readout_pull would also update the
    # embedding rows of the input tokens (J_S_p's graph flows through h →
    # embed_tokens), breaking the "only token_set rows change" guarantee.
    import torch.nn as nn
    with torch.no_grad():
        vocab = m.get_output_embeddings().weight.shape[0]
        m.lm_head = nn.Linear(D_S, vocab, bias=False)
        m.lm_head.weight.copy_(m.get_output_embeddings().weight)
    return m


def _sample(model, S=6):
    ids = torch.randint(0, 40, (1, S))
    return {"input_ids": ids, "attention_mask": torch.ones(1, S, dtype=torch.long)}


TS = list(range(K))


# --- 1: make_base -------------------------------------------------------------


def test_make_base():
    edit = make_base(TS)
    assert edit.delta_W is None
    assert edit.norm_frobenius == 0.0
    assert edit.token_set == tuple(TS)
    assert edit.target_layer == "lm_head"
    assert edit.seed is None
    with pytest.raises(Exception):  # frozen
        edit.name = "x"  # type: ignore[misc]


# --- 2-4: make_random_ortho -----------------------------------------------------


def test_random_ortho_norm():
    ref = _ref()
    edit = make_random_ortho(ref, _basis(), TS, seed=0)
    assert abs(edit.norm_frobenius - ref.norm().item()) < 1e-5


def test_random_ortho_projection():
    ref = _ref()
    basis = _basis()
    edit = make_random_ortho(ref, basis, TS, seed=0)
    P = basis @ basis.T
    assert (P @ edit.delta_W).norm().item() / edit.delta_W.norm().item() < 1e-4


def test_random_ortho_determinism():
    ref = _ref()
    b = _basis()
    e1 = make_random_ortho(ref, b, TS, seed=0)
    e2 = make_random_ortho(ref, b, TS, seed=0)
    assert torch.equal(e1.delta_W, e2.delta_W)
    e3 = make_random_ortho(ref, b, TS, seed=1)
    assert not torch.equal(e1.delta_W, e3.delta_W)


def test_random_ortho_errors():
    ref = _ref()
    bad = torch.randn(7, D_S)
    with pytest.raises(ValueError, match="rows = len"):
        make_random_ortho(bad, _basis(), TS, seed=0)
    non_orth = torch.randn(K, R)
    with pytest.raises(ValueError, match="orthonormal"):
        make_random_ortho(ref, non_orth, TS, seed=0)


# --- 5-6: make_random_matched ---------------------------------------------------


def test_random_matched_projection_preserved():
    ref = _ref()
    basis = _basis()
    edit = make_random_matched(ref, basis, TS, seed=0)
    P = basis @ basis.T
    assert (P @ (edit.delta_W - ref)).norm().item() < 1e-4 * ref.norm().item()


def test_random_matched_norm():
    ref = _ref()
    edit = make_random_matched(ref, _basis(), TS, seed=0)
    assert abs(edit.norm_frobenius - ref.norm().item()) < 1e-5


def test_random_matched_all_in_topr_raises():
    basis = _basis()
    ref = basis @ torch.randn(R, D_S)  # entirely in top-r span
    with pytest.raises(RuntimeError, match="entirely in the top-r"):
        make_random_matched(ref, basis, TS, seed=0)


# --- 7: make_tail_only -----------------------------------------------------------


def test_tail_only():
    ref = _ref()
    basis = _basis()
    edit = make_tail_only(ref, basis, TS)
    P = basis @ basis.T
    assert (P @ edit.delta_W).norm().item() / edit.delta_W.norm().item() < 1e-4
    assert edit.delta_W.norm().item() < ref.norm().item()  # strict (ref has top-r part)
    assert torch.allclose(edit.delta_W, ref - basis @ (basis.T @ ref), atol=1e-6)


def test_tail_only_all_in_topr_raises():
    basis = _basis()
    ref = basis @ torch.randn(R, D_S)
    with pytest.raises(RuntimeError, match="entirely in the top-r"):
        make_tail_only(ref, basis, TS)


# --- 8-11: make_readout_pull ------------------------------------------------------


def test_readout_pull_shape_and_restoration():
    model = _tiny_model()
    W_before = model.lm_head.weight.clone()
    sample = _sample(model)
    edit = make_readout_pull(
        model, sample["input_ids"], sample["attention_mask"],
        layer_idx=3, token_set=TS,
        target_J_p=torch.randn(K, D_S), n_steps=3, lr=1e-2, seed=0,
    )
    assert edit.delta_W is not None
    assert edit.delta_W.shape == (K, D_S)
    assert torch.equal(model.lm_head.weight, W_before)  # bitwise restoration
    assert edit.name == "readout_pull"
    assert edit.token_set == tuple(TS)


def test_readout_pull_adam_moved():
    model = _tiny_model()
    sample = _sample(model)
    edit = make_readout_pull(
        model, sample["input_ids"], sample["attention_mask"],
        layer_idx=3, token_set=TS,
        target_J_p=torch.randn(K, D_S), n_steps=3, lr=1e-2, seed=0,
    )
    assert edit.norm_frobenius > 0.0


def test_readout_pull_requires_grad_restoration():
    model = _tiny_model()
    model.lm_head.weight.requires_grad_(False)
    sample = _sample(model)
    make_readout_pull(
        model, sample["input_ids"], sample["attention_mask"],
        layer_idx=3, token_set=TS,
        target_J_p=torch.randn(K, D_S), n_steps=2, lr=1e-2, seed=0,
    )
    assert not model.lm_head.weight.requires_grad


def test_readout_pull_with_top_r_basis_fills_projection():
    model = _tiny_model()
    sample = _sample(model)
    edit = make_readout_pull(
        model, sample["input_ids"], sample["attention_mask"],
        layer_idx=3, token_set=TS,
        target_J_p=torch.randn(K, D_S), n_steps=2, lr=1e-2, seed=0,
        top_r_basis=_basis(),
    )
    assert 0.0 <= edit.projection_onto_top_r <= 1.0 + 1e-6
    assert "target_J_p_hash" in edit.meta and edit.meta["lr"] == 1e-2


def test_readout_pull_errors():
    model = _tiny_model()
    sample = _sample(model)
    ids, mask = sample["input_ids"], sample["attention_mask"]
    with pytest.raises(ValueError, match="target_J_p"):
        make_readout_pull(model, ids, mask, 1, TS, torch.randn(K, D_S - 1), n_steps=1)
    with pytest.raises(ValueError, match="n_steps"):
        make_readout_pull(model, ids, mask, 1, TS, torch.randn(K, D_S), n_steps=0)
    with pytest.raises(ValueError, match="lr"):
        make_readout_pull(model, ids, mask, 1, TS, torch.randn(K, D_S), n_steps=1, lr=0.0)


# --- 12-16: apply_edit -------------------------------------------------------------


def test_apply_edit_base_noop():
    model = _tiny_model()
    W_before = model.lm_head.weight.clone()
    with apply_edit(model, make_base(TS)):
        assert torch.equal(model.lm_head.weight, W_before)
    assert torch.equal(model.lm_head.weight, W_before)


def test_apply_edit_restores_on_exception():
    model = _tiny_model()
    W_before = model.lm_head.weight.clone()
    edit = make_tail_only(_ref(), _basis(), TS)
    with pytest.raises(RuntimeError, match="boom"):
        with apply_edit(model, edit):
            raise RuntimeError("boom")
    assert torch.equal(model.lm_head.weight, W_before)


def test_apply_edit_visible_inside():
    model = _tiny_model()
    W_before = model.lm_head.weight.clone()
    edit = make_tail_only(_ref(), _basis(), TS)
    with apply_edit(model, edit):
        W_now = model.lm_head.weight
        assert not torch.equal(W_now[TS], W_before[TS])
        assert torch.allclose(W_now[TS], W_before[TS] + edit.delta_W)
    assert torch.equal(model.lm_head.weight, W_before)


def test_apply_edit_nesting_raises():
    model = _tiny_model()
    edit = make_tail_only(_ref(), _basis(), TS)
    with pytest.raises(RuntimeError, match="not re-entrant"):
        with apply_edit(model, edit):
            with apply_edit(model, edit):
                pass


def test_apply_edit_wrong_layer_raises():
    model = _tiny_model()
    edit = make_base(TS)
    bad = EditSpec(
        name="x", delta_W=None, token_set=tuple(TS), target_layer="layer.0",
        norm_frobenius=0.0, projection_onto_top_r=0.0, seed=None, meta={},
    )
    assert edit is not None
    with pytest.raises(NotImplementedError, match="lm_head"):
        with apply_edit(model, bad):
            pass


def test_apply_edit_shape_check():
    model = _tiny_model()
    bad = EditSpec(
        name="bad", delta_W=torch.randn(4, D_S), token_set=tuple(TS),
        target_layer="lm_head", norm_frobenius=1.0, projection_onto_top_r=0.0,
        seed=None, meta={},
    )
    with pytest.raises(ValueError, match="delta_W shape"):
        with apply_edit(model, bad):
            pass


# --- 17-20: apply_edit_and_cache ------------------------------------------------------


def test_apply_edit_and_cache_roundtrip(tmp_path):
    model = _tiny_model()
    edit = make_tail_only(_ref(), _basis(), TS)
    samples = [_sample(model, S=6), _sample(model, S=7)]
    info = apply_edit_and_cache(model, edit, samples, layers=[3],
                                cache_dir=str(tmp_path))
    h_path = str(tmp_path / "tail_only" / "L03" / "doc0000_h.pt")
    p_path = str(tmp_path / "tail_only" / "L03" / "doc0000_p.pt")
    assert os.path.exists(h_path)
    h = torch.load(h_path, weights_only=False)["h"]
    p = torch.load(p_path, weights_only=False)["p"]
    assert h.shape == (6, D_S) and h.dtype == torch.float32
    assert p.shape == (6, K) and p.dtype == torch.float32
    assert info["S"] == 7 and info["K"] == K and info["n_docs"] == 2


def test_apply_edit_and_cache_restores_model(tmp_path):
    model = _tiny_model()
    W_before = model.lm_head.weight.clone()
    edit = make_tail_only(_ref(), _basis(), TS)
    apply_edit_and_cache(model, edit, [_sample(model)], layers=[0, 3],
                         cache_dir=str(tmp_path))
    assert torch.equal(model.lm_head.weight, W_before)


def test_apply_edit_and_cache_overwrite_false_skips(tmp_path):
    model = _tiny_model()
    edit = make_tail_only(_ref(), _basis(), TS)
    samples = [_sample(model)]
    apply_edit_and_cache(model, edit, samples, layers=[3], cache_dir=str(tmp_path))
    h_path = tmp_path / "tail_only" / "L03" / "doc0000_h.pt"
    mtime1 = os.path.getmtime(h_path)
    # second call: consistent cache → skipped, mtime unchanged
    apply_edit_and_cache(model, edit, samples, layers=[3], cache_dir=str(tmp_path),
                         overwrite=False)
    assert os.path.getmtime(h_path) == mtime1


def test_apply_edit_and_cache_meta_consistency(tmp_path):
    model = _tiny_model()
    edit = make_tail_only(_ref(), _basis(), TS)
    apply_edit_and_cache(model, edit, [_sample(model)], layers=[3],
                         cache_dir=str(tmp_path))
    meta_path = tmp_path / "tail_only" / "L03" / "doc0000_meta.json"
    meta = json.loads(meta_path.read_text())
    for key in ("edit", "layer", "doc", "S", "D_S", "K", "torch", "transformers"):
        assert key in meta, key
    assert meta["edit"] == "tail_only" and meta["layer"] == 3 and meta["doc"] == 0

    # mismatch → RuntimeError on re-run with different S
    bad_sample = _sample(model, S=11)
    with pytest.raises(RuntimeError, match="meta mismatch"):
        apply_edit_and_cache(model, edit, [bad_sample], layers=[3],
                             cache_dir=str(tmp_path))


def test_apply_edit_and_cache_errors(tmp_path):
    model = _tiny_model()
    edit = make_base(TS)
    with pytest.raises(ValueError, match="empty"):
        apply_edit_and_cache(model, edit, [], layers=[3], cache_dir=str(tmp_path))
    with pytest.raises(ValueError, match="layer"):
        apply_edit_and_cache(model, edit, [_sample(model)], layers=[99],
                             cache_dir=str(tmp_path))
