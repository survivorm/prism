"""Module 4: interventions — EditSpec and its application to the model (TZ part 2).

Five T0 edit types: base, readout_pull, random_ortho, random_matched,
tail_only. Plus `apply_edit` (context manager) and `apply_edit_and_cache`
(→ h_t, p_t on disk).

Key decisions (docs/new_functions_2.md §4.1):
1. **delta_W is [K, D_S], applied only to `lm_head.weight[token_set]`**
   — not [V, D_S]. ~1200× memory savings (storage of the DELTA only);
   semantics: J depends only on the token_set rows. Note: the Adam state
   in `edit_lm_head` is allocated on the FULL [V, D_S] weight (~2 GB
   extra for 0.8B) — the savings apply to the persisted delta, not to
   optimizer state (docs/E38_fixes.md §6).
2. **EditSpec knows its token_set** — never recomputed at apply time.
3. **apply_edit restores the model in `finally`** — even on exception.
4. **apply_edit_and_cache caches only h_t and p_t, not J** — J is
   reconstructed on the fly via compute_jacobian_for_tokens(h_override=
   h_t[l][t]) + SVDCache (in the experiment code, NOT here): ~30× less
   disk and any r without recompute.

Not used here (deliberate): bridge.jacobian_gap (needs ready J; the loss
needs create_graph rows), compute_jacobian_for_tokens (downstream only),
cache_io.find_targets_dir (read-only discovery; interventions write new
caches), _clone_gdn_states_on_entry around apply_edit_and_cache
(no_grad, no backward — in-place mutation is safe).

Related: paper.md §E-16a (readout_pull protocol), §19.A (EditSpec).
"""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F

from .jacobian import compute_hidden_jacobian_rows

__all__ = [
    "EditSpec",
    "make_base",
    "make_readout_pull",
    "make_random_ortho",
    "make_random_matched",
    "make_tail_only",
    "apply_edit",
    "apply_edit_and_cache",
]

# Module-level nesting guard for apply_edit (one edit at a time; TZ §4.3).
_APPLY_EDIT_ACTIVE = False


@dataclass(frozen=True)
class EditSpec:
    """Immutable edit specification (TZ §4.3).

    delta_W is [K, D_S] and refers to `lm_head.weight[token_set]` ONLY —
    applying it to the full [V, D_S] weight is a caller bug (apply_edit
    applies it to the token_set rows and checks shapes).

    `projection_onto_top_r` may be filled in after the fact via
    `dataclasses.replace(edit, projection_onto_top_r=...)` — frozen
    dataclasses support replace (TZ §4.3, make_readout_pull).

    `meta` is a dict; treated as immutable after creation (callers copy
    it at creation; no field-level enforcement).
    """

    name: str
    delta_W: torch.Tensor | None                    # [K, D_S], None for base
    token_set: tuple[int, ...]                      # frozen tuple, hashable
    target_layer: str                               # "lm_head" (only supported)
    norm_frobenius: float
    projection_onto_top_r: float
    seed: int | None
    meta: dict = field(default_factory=dict)


def _resolve_lm_head_weight(model) -> torch.Tensor:
    """model.lm_head.weight, with a get_output_embeddings() fallback."""
    lm = getattr(model, "lm_head", None)
    if lm is not None and getattr(lm, "weight", None) is not None:
        return lm.weight
    out = model.get_output_embeddings()
    if out is None or getattr(out, "weight", None) is None:
        raise AttributeError("model has no lm_head.weight / output embeddings")
    return out.weight


def _validate_orthonormal_columns(basis: torch.Tensor) -> None:
    K, r = basis.shape
    gram = basis.T @ basis
    err = torch.norm(gram - torch.eye(r, device=gram.device, dtype=gram.dtype), p="fro").item()
    if err > 1e-4:
        raise ValueError(f"top_r_basis not orthonormal: ‖U^T U − I‖_F = {err:.3e}")


def _projection_fraction(delta_W: torch.Tensor, top_r_basis: torch.Tensor) -> float:
    """‖P_topr ΔW‖_F / ‖ΔW‖_F, P_topr = U U^T ([K,K] projector on the
    token-space span of top_r_basis; each of the D_S columns is a [K]
    vector)."""
    P = top_r_basis @ top_r_basis.T
    nrm = delta_W.norm()
    if nrm.item() < 1e-12:
        return 0.0
    return (P @ delta_W).norm().item() / nrm.item()


def _scale_to_norm(delta_W: torch.Tensor, target_norm: float) -> torch.Tensor:
    nrm = delta_W.norm()
    if nrm.item() < 1e-12:
        raise RuntimeError("_scale_to_norm: zero-norm input")
    return delta_W * (target_norm / nrm)


def make_base(token_set: Sequence[int]) -> EditSpec:
    """No-op edit: uniform signature with the other edit types
    (apply_edit(model, base) enters and leaves the model untouched)."""
    return EditSpec(
        name="base",
        delta_W=None,
        token_set=tuple(token_set),
        target_layer="lm_head",
        norm_frobenius=0.0,
        projection_onto_top_r=0.0,
        seed=None,
        meta={},
    )


def make_readout_pull(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer_idx: int,
    token_set: list[int],
    target_J_p: torch.Tensor,
    n_steps: int = 100,
    lr: float = 1e-3,
    kl_leash_weight: float = 0.1,
    seed: int = 0,
    verbose: bool = False,
    top_r_basis: torch.Tensor | None = None,
) -> EditSpec:
    """E-16a protocol: Adam on lm_head.weight[token_set] rows only (TZ §4.3).

    Loss per step: (1 − cos(J_S_p, target_J_p)).mean() +
    kl_leash_weight · KL(p_base ‖ p_edit), where
      - J_S_p [K, D_S] via compute_hidden_jacobian_rows(create_graph=True)
        (reused from jacobian.py — the only create_graph site; its inner
        _clone_gdn_states_on_entry handles the GDN state),
      - cos — per-row F.cosine_similarity,
      - KL — over the token_set distributions (p_base computed ONCE up
        front, never re-normalized per step),
      - p_edit = softmax(lm_head(h_t)[:, token_set]) at the last position
        (E-16a protocol: last-position token_set distribution).

    Determinism: torch.manual_seed(seed); torch.use_deterministic_
    algorithms is NOT enabled (Adam on GDN models may raise).

    Memory note: ``params = [weight]`` — the Adam optimizer allocates full
    [V, D_S] state (m + v), not just the [K, D_S] delta: gradient masking
    keeps non-token_set rows frozen, but the state is still ~2 GB for
    0.8B. The "~1200× savings" claim covers the persisted delta only
    (docs/E38_fixes.md §6).

    Restoration: W0 = lm_head.weight[token_set].detach().clone() up
    front; copy_(W0) in finally; requires_grad saved/restored too.

    Returns:
        EditSpec(name="readout_pull", delta_W = W_final[token_set] − W0).
        projection_onto_top_r: 0.0 if top_r_basis is None, else computed
        (TZ §4.3: pass top_r_basis to fill it in-place; the experiment
        can also use dataclasses.replace later).

    The model is left exactly as found (weights, requires_grad, training
    mode).

    Errors:
        ValueError: target_J_p.shape != (K, D_S); n_steps < 1; lr ≤ 0.
    """
    K = len(token_set)
    if n_steps < 1:
        raise ValueError(f"n_steps must be ≥ 1, got {n_steps}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")

    weight = _resolve_lm_head_weight(model)
    D_S = weight.shape[1]
    ts = torch.tensor(token_set, device=weight.device)
    if target_J_p.shape != (K, D_S):
        raise ValueError(f"target_J_p must be ({K}, {D_S}), got {tuple(target_J_p.shape)}")
    target_J_p = target_J_p.detach().to(dtype=weight.dtype)

    global _APPLY_EDIT_ACTIVE
    if _APPLY_EDIT_ACTIVE:
        raise RuntimeError("make_readout_pull: nested model edit in progress")

    torch.manual_seed(seed)
    was_training = model.training
    model.eval()

    W0 = weight[ts].detach().clone()
    rg_was = weight.requires_grad
    weight.requires_grad_(True)
    _APPLY_EDIT_ACTIVE = True  # AFTER the setup that can raise below
    delta_W: torch.Tensor | None = None

    try:
        params = [weight]  # gradient masked below: only token_set rows update
        ts_mask = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
        ts_mask[ts] = True

        # p_base: once, under no_grad, over the K-subset (E-16 gauge).
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            p_base = F.softmax(out.logits[0, -1, ts].float(), dim=-1)

        opt = torch.optim.Adam(params, lr=lr)
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)

            # (1) Jacobian rows through the (trainable) lm_head rows.
            W_rows = weight[ts]
            J_S_p = compute_hidden_jacobian_rows(
                model, layer_idx, input_ids, attention_mask, token_set,
                p=p_base, W_rows=W_rows, create_graph=True,
            )
            cos = F.cosine_similarity(J_S_p, target_J_p, dim=1, eps=1e-12)
            j_loss = (1.0 - cos).mean()

            # (2) KL leash on the token_set distribution at the last pos.
            # h_t is recomputed every step so the leash tracks the edit.
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            p_edit = F.softmax(out.logits[0, -1, ts].float(), dim=-1)
            kl = F.kl_div(p_edit.log(), p_base, reduction="batchmean")

            loss = j_loss + kl_leash_weight * kl
            loss.backward()
            # TZ: "Adam state: only lm_head.weight[token_set]" — the KL term
            # (logits = h @ W^T) gradients the FULL [V, D_S] weight; mask
            # everything outside the token_set slice so Adam moves only the
            # K rows (zero grad → zero Adam update for the other rows).
            if weight.grad is not None:
                weight.grad.data[~ts_mask].zero_()
            opt.step()

            if verbose:
                print(f"    readout_pull step {step + 1}/{n_steps}: "
                      f"loss={loss.item():.6f} (cos={(cos.mean()).item():.4f})")
            # Capture the delta BEFORE finally restores the weights.
            delta_W = weight[ts].detach().clone() - W0
    finally:
        # index_copy_ on .data — in-place; a leaf with requires_grad=True
        # rejects in-place ops even under no_grad (graph corruption), and
        # weight[ts].copy_ would be a no-op (advanced indexing → copy).
        weight.data.index_copy_(0, ts, W0)
        weight.requires_grad_(rg_was)
        model.train(was_training)
        _APPLY_EDIT_ACTIVE = False

    assert delta_W is not None  # n_steps ≥ 1 guarantees one capture
    return EditSpec(
        name="readout_pull",
        delta_W=delta_W,
        token_set=tuple(token_set),
        target_layer="lm_head",
        norm_frobenius=delta_W.norm().item(),
        projection_onto_top_r=(
            _projection_fraction(delta_W, top_r_basis) if top_r_basis is not None else 0.0
        ),
        seed=seed,
        meta={
            "target_J_p_hash": _tensor_fingerprint(target_J_p),
            "lr": lr,
            "n_steps": n_steps,
            "kl_leash_weight": kl_leash_weight,
            "layer_idx": layer_idx,
        },
    )


def _tensor_fingerprint(t: torch.Tensor) -> str:
    """Deterministic 16-hex fingerprint (sum + abs-max) for meta."""
    t = t.detach().double()
    s = int(t.sum().item() * 1e9) & 0xFFFFFFFF
    m = int(t.abs().max().item() * 1e9) & 0xFFFFFFFF
    return f"{s:08x}{m:08x}"


def make_random_ortho(
    delta_W_ref: torch.Tensor,
    top_r_basis: torch.Tensor,
    token_set: Sequence[int],
    seed: int,
) -> EditSpec:
    """Random edit with ‖·‖_F = ‖ref‖_F and P_topr ΔW ≈ 0 (TZ §4.3).

    Algorithm: G ~ randn(K, D_S); G_proj = G − P G (orthogonal
    complement in token-space, P = U U^T [K, K]); scale to ref norm.
    G is the only source of randomness (seeded → deterministic).

    Post-checks (RuntimeError with diagnostics on failure):
        ‖delta_W‖_F ≈ ‖ref‖_F (1e-6 rel); ‖P ΔW‖_F / ‖ΔW‖_F < 1e-3.

    Errors:
        ValueError: delta_W_ref not 2-D / K mismatch vs token_set or
            basis; top_r_basis not orthonormal.
        RuntimeError: ref norm ≈ 0; G nearly entirely in top-r
            (‖G_proj‖ < 1e-6·‖G‖ — caller must reduce r or change basis).
    """
    _check_ref_and_basis(delta_W_ref, top_r_basis, token_set)
    K, D_S = delta_W_ref.shape
    torch.manual_seed(seed)
    G = torch.randn(K, D_S, dtype=delta_W_ref.dtype)
    P = top_r_basis @ top_r_basis.T
    G_proj = G - P @ G
    if G_proj.norm().item() < 1e-6 * G.norm().item():
        raise RuntimeError(
            "random_ortho: G is nearly entirely in the top-r span; "
            "reduce r or choose a different basis"
        )
    delta_W = _scale_to_norm(G_proj, delta_W_ref.norm().item())

    proj = _projection_fraction(delta_W, top_r_basis)
    if proj > 1e-3:
        raise RuntimeError(
            f"random_ortho post-check failed: top-r projection fraction "
            f"{proj:.3e} > 1e-3"
        )
    nrm = delta_W.norm().item()
    if abs(nrm - delta_W_ref.norm().item()) > 1e-6 * delta_W_ref.norm().item():
        raise RuntimeError(
            f"random_ortho post-check failed: norm {nrm:.9f} vs ref "
            f"{delta_W_ref.norm().item():.9f}"
        )
    return EditSpec(
        name="random_ortho",
        delta_W=delta_W,
        token_set=tuple(token_set),
        target_layer="lm_head",
        norm_frobenius=nrm,
        projection_onto_top_r=proj,
        seed=seed,
        meta={"ref_norm": delta_W_ref.norm().item(), "r": top_r_basis.shape[1]},
    )


def make_random_matched(
    delta_W_ref: torch.Tensor,
    top_r_basis: torch.Tensor,
    token_set: Sequence[int],
    seed: int,
) -> EditSpec:
    """Random edit preserving the top-r projection of ref (TZ §4.3).

    delta_W = ref_proj + G_residual · (‖ref_residual‖ / ‖G_residual‖):
    the top-r component is EXACTLY ref's; only the orthogonal complement
    is randomized. Same total norm as ref.

    Contrast with random_ortho (zero top-r projection): the two controls
    separate "effect of the norm" from "effect of landing in the signal".

    Errors: ValueError as in random_ortho; RuntimeError if ref is
    entirely in top-r (nothing to randomize).
    """
    _check_ref_and_basis(delta_W_ref, top_r_basis, token_set)
    K, D_S = delta_W_ref.shape
    torch.manual_seed(seed)
    P = top_r_basis @ top_r_basis.T
    ref_proj = top_r_basis @ (top_r_basis.T @ delta_W_ref)
    ref_residual = delta_W_ref - ref_proj
    if ref_residual.norm().item() < 1e-6 * delta_W_ref.norm().item():
        raise RuntimeError(
            "random_matched: reference is entirely in the top-r span; "
            "nothing to randomize (reduce r or change reference)"
        )
    G = torch.randn(K, D_S, dtype=delta_W_ref.dtype)
    G_residual = G - P @ G
    if G_residual.norm().item() < 1e-9:
        raise RuntimeError("random_matched: degenerate random residual")
    delta_W = ref_proj + _scale_to_norm(G_residual, ref_residual.norm().item())

    diff_proj = (P @ (delta_W - delta_W_ref)).norm().item()
    if diff_proj > 1e-4 * delta_W_ref.norm().item():
        raise RuntimeError(
            f"random_matched post-check failed: top-r projection changed by "
            f"{diff_proj:.3e} (rel to ref norm)"
        )
    nrm = delta_W.norm().item()
    if abs(nrm - delta_W_ref.norm().item()) > 1e-5 * delta_W_ref.norm().item():
        raise RuntimeError(
            f"random_matched post-check failed: norm {nrm:.9f} vs ref "
            f"{delta_W_ref.norm().item():.9f}"
        )
    return EditSpec(
        name="random_matched",
        delta_W=delta_W,
        token_set=tuple(token_set),
        target_layer="lm_head",
        norm_frobenius=nrm,
        projection_onto_top_r=_projection_fraction(delta_W, top_r_basis),
        seed=seed,
        meta={"ref_norm": delta_W_ref.norm().item(), "r": top_r_basis.shape[1]},
    )


def make_tail_only(
    delta_W_ref: torch.Tensor,
    top_r_basis: torch.Tensor,
    token_set: Sequence[int],
) -> EditSpec:
    """Reference minus its top-r projection — the tail ESG modes only
    (TZ §4.3).

    NOT renormalized: tail-only is exactly "what was left", not
    "as if the full norm". ‖delta_W‖_F = ‖ref_residual‖_F ≤
    ‖ref‖_F (strict if ref has a top-r component).

    Control semantics: if the Goodhart effect appears in both
    readout_pull and tail_only (but not random_ortho), the reference's
    STRUCTURE matters, not just tail landing; if only in readout_pull —
    the top-r component matters.

    Errors: ValueError as in random_ortho; RuntimeError if ref is
    entirely in top-r (nothing to keep).
    """
    _check_ref_and_basis(delta_W_ref, top_r_basis, token_set)
    ref_proj = top_r_basis @ (top_r_basis.T @ delta_W_ref)
    delta_W = delta_W_ref - ref_proj
    if delta_W.norm().item() < 1e-6 * delta_W_ref.norm().item():
        raise RuntimeError(
            "tail_only: reference is entirely in the top-r span; "
            "nothing to keep (reduce r or change reference)"
        )
    return EditSpec(
        name="tail_only",
        delta_W=delta_W,
        token_set=tuple(token_set),
        target_layer="lm_head",
        norm_frobenius=delta_W.norm().item(),
        projection_onto_top_r=_projection_fraction(delta_W, top_r_basis),
        seed=None,
        meta={"ref_norm": delta_W_ref.norm().item(), "r": top_r_basis.shape[1]},
    )


def _check_ref_and_basis(delta_W_ref: torch.Tensor, top_r_basis: torch.Tensor,
                         token_set: Sequence[int]) -> None:
    K = len(token_set)
    if delta_W_ref.dim() != 2 or delta_W_ref.shape[0] != K:
        raise ValueError(
            f"delta_W_ref must be 2-D with rows = len(token_set) = {K}, "
            f"got {tuple(delta_W_ref.shape)}"
        )
    if top_r_basis.shape[0] != K:
        raise ValueError(
            f"top_r_basis rows ({top_r_basis.shape[0]}) != len(token_set) ({K})"
        )
    _validate_orthonormal_columns(top_r_basis)


def _validate_edit_spec(edit: EditSpec, D_S: int) -> None:
    """TZ §4.4: check target_layer and delta_W.shape against the model."""
    if edit.target_layer != "lm_head":
        raise NotImplementedError(
            f"only target_layer='lm_head' supported, got {edit.target_layer!r}"
        )
    if edit.delta_W is not None and \
            edit.delta_W.shape != (len(edit.token_set), D_S):
        raise ValueError(
            f"delta_W shape {tuple(edit.delta_W.shape)} != "
            f"({len(edit.token_set)}, {D_S})"
        )


@contextlib.contextmanager
def apply_edit(model, edit: EditSpec) -> Iterator[None]:
    """Apply edit in-place, restore in finally (TZ §4.3).

    lm_head.weight[token_set] += delta_W under torch.no_grad(); after
    yield — copy_ the saved slice back. requires_grad of the full weight
    is saved/restored. `base` (delta_W=None) is a no-op (uniform API).

    NOT re-entrant: nested apply_edit → RuntimeError (one edit at a
    time). target_layer other than "lm_head" → NotImplementedError.

    Errors:
        ValueError: delta_W shape != (len(token_set), D_S).
    """
    global _APPLY_EDIT_ACTIVE
    if _APPLY_EDIT_ACTIVE:
        raise RuntimeError(
            "apply_edit is not re-entrant: one edit at a time "
            "(nested apply_edit detected)"
        )

    weight = _resolve_lm_head_weight(model)
    _validate_edit_spec(edit, weight.shape[1])
    ts = torch.tensor(edit.token_set, device=weight.device)
    delta = edit.delta_W.to(weight) if edit.delta_W is not None else None

    W_orig = weight[ts].detach().clone()
    rg_was = weight.requires_grad
    _APPLY_EDIT_ACTIVE = True
    try:
        if delta is not None:
            with torch.no_grad():
                # .data.index_add_: in-place. (weight[ts].add_(delta) is a
                # no-op — advanced indexing returns a copy; plain weight.
                # index_add_ fails on a grad-requiring leaf even under
                # no_grad.)
                weight.data.index_add_(0, ts, delta)
        yield
    finally:
        with torch.no_grad():
            weight.data.index_copy_(0, ts, W_orig)
        weight.requires_grad_(rg_was)
        _APPLY_EDIT_ACTIVE = False


def _atomic_torch_save(obj: Any, final_path: str) -> None:
    d = os.path.dirname(final_path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{final_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    torch.save(obj, tmp)
    os.replace(tmp, final_path)


def apply_edit_and_cache(
    model,
    edit: EditSpec,
    samples: list[dict],
    layers: list[int],
    cache_dir: str,
    overwrite: bool = False,
) -> dict:
    """Forward under the edit; cache h_t [S, D_S] and p_t [S, K] (TZ §4.3).

    Layout: cache_dir/{edit.name}/L{layer:02d}/doc{doc:04d}_{h,p}.pt
    (+ meta.json per (edit, layer, doc)). h stored in float32 (downstream
    convert to model dtype; fp32 on disk is precise and uniform), p in
    float32 (softmax precision for gauge-free P(p)).

    J is NOT cached (decision #4): reconstruct via
    compute_jacobian_for_tokens(h_override=h_t[l][t]) + SVDCache in the
    experiment — ~30× less disk, any r without recompute.

    No _clone_gdn_states_on_entry here: no_grad, no backward — in-place
    lm_head mutation is safe for the GDN recurrent state.

    overwrite=False: existing files are kept, but their meta is checked
    for consistency (edit/layer/doc/S/D_S/K) — mismatch → RuntimeError.
    Partial cache (some of h/p/meta missing): existing files are kept,
    missing ones are written, meta is checked if present.
    overwrite=True: atomic (tmp + os.replace) rewrite.

    Returns:
        {"root", "edit", "n_docs", "layers", "S", "D_S", "K", "files"}.

    Errors:
        ValueError: empty samples; layer index out of [0, n_layers);
            samples with differing S.
        RuntimeError: read-only cache_dir with overwrite=True (Kaggle
            FUSE) — explicit, not silent.
    """
    if not samples:
        raise ValueError("samples is empty")
    S = samples[0]["input_ids"].shape[1]
    for sample in samples:
        if sample["input_ids"].shape[1] != S:
            raise ValueError(
                f"all samples must have the same S, expected {S}, got "
                f"{sample['input_ids'].shape[1]}"
            )
    n_layers = _n_layers(model)
    for L in layers:
        if not (isinstance(L, int) and 0 <= L < n_layers):
            raise ValueError(f"layer {L} out of range [0, {n_layers})")

    was_training = model.training
    model.eval()
    weight = _resolve_lm_head_weight(model)
    K = len(edit.token_set)
    D_S = weight.shape[1]
    ts = torch.tensor(edit.token_set, device=weight.device)
    edit_root = os.path.join(cache_dir, edit.name)

    files: list[str] = []
    try:
        _cache_edit(model, edit, samples, layers, ts, S, D_S, K, edit_root,
                    overwrite, files, cache_dir)
    finally:
        model.train(was_training)

    return {
        "root": cache_dir,
        "edit": edit.name,
        "n_docs": len(samples),
        "layers": list(layers),
        "S": S,
        "D_S": D_S,
        "K": K,
        "files": files,
    }


def _cache_edit(model, edit, samples, layers, ts, S, D_S, K, edit_root,
                overwrite, files, cache_dir) -> None:
    with apply_edit(model, edit):
        for doc_idx, sample in enumerate(samples):
            with torch.no_grad():
                out = model(**sample, output_hidden_states=True, use_cache=False)
            h_states = out.hidden_states
            # p_t = softmax over the token_set (re-normalized on K) — the
            # p_S distribution of paper.md §4; required for gauge-free
            # P(p) = diag(p) − pp^T (P(p)·1 = 0 iff Σp = 1).
            p_t = F.softmax(out.logits[0][:, ts].float(), dim=-1)  # [S, K]
            for L in layers:
                h_t = h_states[L][0].float()                # [S, D_S] float32
                layer_dir = os.path.join(edit_root, f"L{L:02d}")
                meta = {
                    "edit": edit.name,
                    "layer": L,
                    "doc": doc_idx,
                    "S": S,
                    "D_S": D_S,
                    "K": K,
                    "model_dtype": str(model.dtype),
                    "torch": torch.__version__,
                    "transformers": _transformers_version(),
                    "seed": edit.seed,
                }
                h_path = os.path.join(layer_dir, f"doc{doc_idx:04d}_h.pt")
                p_path = os.path.join(layer_dir, f"doc{doc_idx:04d}_p.pt")
                meta_path = os.path.join(layer_dir, f"doc{doc_idx:04d}_meta.json")

                if not overwrite:
                    if os.path.exists(meta_path):
                        _check_meta_consistency(meta_path, meta)
                    if os.path.exists(h_path) and os.path.exists(p_path) \
                            and os.path.exists(meta_path):
                        files.extend([h_path, p_path, meta_path])
                        continue
                try:
                    # overwrite=False: keep existing files, write missing ones.
                    if overwrite or not os.path.exists(h_path):
                        _atomic_torch_save({"h": h_t, "meta": meta}, h_path)
                    if overwrite or not os.path.exists(p_path):
                        _atomic_torch_save({"p": p_t, "meta": meta}, p_path)
                    if overwrite or not os.path.exists(meta_path):
                        _atomic_torch_save_text(
                            json.dumps(meta, indent=2, sort_keys=True), meta_path
                        )
                except OSError as e:
                    raise RuntimeError(
                        f"cannot write cache to {cache_dir!r} (read-only / Kaggle "
                        f"FUSE?): {e}. The existing cache is incomplete "
                        f"(overwrite={overwrite}); use a writable dir."
                    ) from e
                files.extend([h_path, p_path, meta_path])


def _atomic_torch_save_text(text: str, final_path: str) -> None:
    d = os.path.dirname(final_path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = f"{final_path}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, final_path)


def _check_meta_consistency(meta_path: str, expected: dict) -> None:
    with open(meta_path) as f:
        existing = json.load(f)
    # dtype/torch/transformers are checked too (docs/E38_fixes.md §2): a
    # silent dtype switch between runs (e.g. fp16 -> fp32) would otherwise
    # mix stale and fresh cache files without any error, since
    # overwrite=False only writes missing files.
    for key in ("edit", "layer", "doc", "S", "D_S", "K",
                "model_dtype", "torch", "transformers"):
        if existing.get(key) != expected[key]:
            raise RuntimeError(
                f"cache meta mismatch for {meta_path}: {key} = "
                f"{existing.get(key)!r} != expected {expected[key]!r} "
                f"(pass overwrite=True to rewrite)"
            )


def _n_layers(model) -> int:
    cfg = getattr(model, "config", None)
    tc = getattr(cfg, "text_config", cfg)
    return int(tc.num_hidden_layers)


def _transformers_version() -> str:
    import transformers
    return transformers.__version__
