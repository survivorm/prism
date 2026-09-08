from __future__ import annotations
# ---------- TPU runtime ----------
import os
os.environ["PJRT_DEVICE"] = "TPU"
os.environ.setdefault("HF_HOME", "/kaggle/working/.hf_cache")
import time, sys, traceback, inspect, gc
import torch
torch.set_num_threads(1)
import torch_xla.core.xla_model as xm
xdev = xm.xla_device()
device = "xla"
WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.expanduser("~/.kaggle_test")
RES = os.path.join(WORKING, "results")
os.makedirs(RES, exist_ok=True)
print(f"[TPU] device={xdev} torch={torch.__version__}")

import contextlib

import torch
from transformers import DynamicCache
from transformers.cache_utils import (  # type: ignore[import-not-found]
    DynamicLayer,
    LinearAttentionLayer,
)


def compute_jacobian_for_tokens(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_set: list[int],
) -> torch.Tensor:
    """Compute Jacobian of logits w.r.t. hidden state at layer_idx.

    Args:
        model: Qwen3_5ForCausalLM instance.
        layer_idx: Layer index (0-based) to compute Jacobian at.
        input_ids: [1, S] token IDs.
        attention_mask: [1, S] attention mask.
        token_set: list of token IDs to compute gradients for.

    Returns:
        J: [|U|, D] Jacobian matrix (unnormalized).
    """
    model.eval()
    S = input_ids.shape[1]
    prefix_cache = _run_prefix_pass(model, input_ids, attention_mask)
    h = _extract_hidden_state(model, input_ids, attention_mask, layer_idx)
    # Match model dtype to avoid mismatch in decoder (fp16 models on GPU)
    model_dtype = next(model.parameters()).dtype
    h = h.to(model_dtype).detach().requires_grad_(True)
    # Re-enable grad: the decoder forward must build a graph over the leaf
    # h AND the row loop below must run autograd.grad with grad mode ON,
    # even when the caller is inside torch.no_grad() (e.g. eval loops in
    # exp18/19). A no_grad caller otherwise gets "element 0 ... does not
    # have a grad_fn" — the rows loop is part of the same enable_grad block.
    with torch.enable_grad():
        logits = _run_decoder_pass(model, prefix_cache, h, S, layer_idx)
        J = _compute_jacobian_rows(logits, h, token_set)
    return J


def _run_prefix_pass(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> DynamicCache:
    """Prefix pass: all tokens except last -> cache."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            use_cache=True,
            output_hidden_states=False,
        )
    return outputs.past_key_values


def _extract_hidden_state(model, input_ids, attention_mask, layer_idx) -> torch.Tensor:
    """Full pass: extract hidden state at layer_idx for last token."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    # hidden_states[0] = embeddings, [l+1] = output of layer l
    h = outputs.hidden_states[layer_idx + 1][:, -1:, :]
    return h  # [1, 1, D]


def _deep_clone_cache(cache):
    """Deep clone cache, ensuring all tensors are independent copies.

    Must copy every per-layer state field, not just the obvious tensors. Both
    layer types swallow unknown constructor kwargs, so passing
    ``keys=`` / ``conv_states=`` / ``recurrent_states=`` silently discards the
    state and yields an empty cache. We therefore assign attributes directly:

    * DynamicLayer (full attention): keys / values / is_initialized / dtype /
      device. Without this, ``update()`` lazy-initialises an empty cache and
      each layer attends only to the current token.
    * LinearAttentionLayer (GDN): conv_states / recurrent_states plus the flag
      dicts (``has_previous_state`` etc.). Without these, GDN layers treat the
      new token as the first in the sequence and drop all history.

    All cloned state tensors are .detach()-ed: the prefix cache is pure data
    (built under no_grad). Detaching drops pointless CopyBackwards graph nodes
    in the enable_grad decode. The transformers 5.15 in-place cache updates
    (update_recurrent_state's copy_, causal_conv1d_update's state update —
    cudagraph static-address optimisation) are NOT fixed by the detach —
    the in-place copy_ lands on the kernel's saved input regardless. Those
    are handled by _clone_gdn_states_on_entry (clone at kernel entry so the
    read object and the in-place-written object are different objects).

    Extra unknown attributes are deep-copied as a forward-compat safety net.
    """
    import copy as _copy

    def _cl(t):
        return t.clone().detach() if t is not None else None

    new_cache = DynamicCache()
    for idx in range(len(cache.layers)):
        layer = cache.layers[idx]
        if isinstance(layer, DynamicLayer):
            nl = DynamicLayer()
            nl.keys = _cl(layer.keys)
            nl.values = _cl(layer.values)
            nl.is_initialized = layer.is_initialized
            nl.dtype = layer.dtype
            nl.device = layer.device
            for k, v in vars(layer).items():
                if k not in vars(nl):
                    setattr(nl, k, _copy.deepcopy(v))
            new_cache.layers.append(nl)
        else:
            # GDN (LinearAttentionLayer). Clone every state field dynamically
            # rather than hardcoding transformers-5.15.1 attribute names
            # (number_of_states, conv_states, ...): other transformers versions
            # on other backends (e.g. the Kaggle TPU torch 2.8 image) name some
            # differently, and a hardcoded name crashes with AttributeError.
            # Tensors (or dicts of tensors) are clone+detached; scalars and
            # flag-dicts are deep-copied. This is the generic form of the old
            # field-by-field copy and behaves identically on 5.15.1.
            nl = LinearAttentionLayer()
            for k, v in vars(layer).items():
                if isinstance(v, torch.Tensor):
                    setattr(nl, k, _cl(v))
                elif isinstance(v, dict) and v and all(
                    isinstance(x, torch.Tensor) for x in v.values()
                ):
                    setattr(nl, k, {kk: _cl(xv) for kk, xv in v.items()})
                else:
                    setattr(nl, k, _copy.deepcopy(v))
            new_cache.layers.append(nl)
    return new_cache


@contextlib.contextmanager
def _clone_gdn_states_on_entry():
    """Clone GDN kernel state inputs at entry: read object != written object.

    transformers 5.15 Qwen3_5GatedDeltaNet.forward reads the cache's
    recurrent_states[0] / conv_states and passes them to
    torch_recurrent_gated_delta_rule / causal_conv1d_update. The recurrent
    loop saves that read object (version 0) for backward. Afterwards
    update_recurrent_state does a stored.copy_(new) IN PLACE on the very
    same object (5.15 cudagraph static-address optimisation) → version
    0→1 → "modified by an inplace operation: ... output 0 of CopyBackwards"
    in any create_graph decode. Cloning at kernel entry makes the loop's
    saved input the fresh clone; the in-place write lands on the cache's
    original (a different object) → no in-place on a saved node.
    Scoped: restores the original kernels on exit.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
    except Exception:
        yield  # non-Qwen3.5 build: nothing to patch
        return

    orig_rec = mq.torch_recurrent_gated_delta_rule
    orig_conv = mq.causal_conv1d_update

    def rec(q, k, v, g, beta, initial_state, output_final_state,
            use_qk_l2norm_in_kernel=False, **kw):
        if initial_state is not None:
            initial_state = initial_state.clone()
        return orig_rec(q, k, v, g, beta, initial_state, output_final_state,
                        use_qk_l2norm_in_kernel, **kw)

    def conv(h, cs, w, b, a):
        if cs is not None:
            cs = cs.clone()
        return orig_conv(h, cs, w, b, a)

    mq.torch_recurrent_gated_delta_rule = rec
    mq.causal_conv1d_update = conv
    try:
        yield
    finally:
        mq.torch_recurrent_gated_delta_rule = orig_rec
        mq.causal_conv1d_update = orig_conv


def _run_decoder_pass(
    model, prefix_cache, h: torch.Tensor, S: int, layer_idx: int
) -> torch.Tensor:
    """Decode h through remaining layers using prefix cache."""
    # Clone cache tensors independently to avoid in-place mutation
    cache_clone = _deep_clone_cache(prefix_cache)
    text_model = model.model  # Qwen3_5TextModel

    pos = torch.full((4, 1, 1), S - 1, dtype=torch.long, device=h.device)
    text_position_ids = pos[0]
    rope_position_ids = pos[1:]
    position_embeddings = text_model.rotary_emb(h, rope_position_ids)

    x = h
    n_layers = model.config.num_hidden_layers
    start = layer_idx + 1

    causal_mask_mapping = _build_causal_masks(
        text_model, text_position_ids, cache_clone, x
    )

    with _clone_gdn_states_on_entry():
        for i in range(start, n_layers):
            layer_type = model.config.layer_types[i]
            attn_mask = causal_mask_mapping[layer_type]
            x = text_model.layers[i](
                x,
                position_embeddings=position_embeddings,
                attention_mask=attn_mask,
                position_ids=text_position_ids,
                past_key_values=cache_clone,
            )

    x = text_model.norm(x)
    logits = model.lm_head(x[:, -1, :])
    return logits


def _build_causal_masks(text_model, text_pos, cache, x):
    """Build causal mask mapping for a single token with cache."""
    from transformers.masking_utils import (
        create_causal_mask,
        create_recurrent_attention_mask,
    )

    dummy_embeds = x.new_zeros(1, 1, text_model.config.hidden_size)
    mask_kwargs = {
        "config": text_model.config,
        "inputs_embeds": dummy_embeds,
        "attention_mask": None,
        "past_key_values": cache,
        "position_ids": text_pos,
    }
    return {
        "full_attention": create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
    }


def _compute_jacobian_rows(
    logits: torch.Tensor, h: torch.Tensor, token_set: list[int]
) -> torch.Tensor:
    """Compute Jacobian rows via loop over token_set."""
    logits_subset = logits[0, token_set]  # [|U|]
    D = h.shape[-1]
    J = torch.zeros(len(token_set), D, device=h.device, dtype=h.dtype)

    for i in range(len(token_set)):
        grad = torch.autograd.grad(
            outputs=logits_subset[i],
            inputs=h,
            retain_graph=(i < len(token_set) - 1),
        )[0]
        J[i] = grad[0, 0, :]

    return J


def softmax_jacobian(J: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Gauge-invariant Jacobian: J^(p) = diag(p) - pp^T @ J.

    J^(p)_i = p_i * (J_i - J_bar_p),  J_bar_p = sum_k p_k * J_k
    Removes the common-mode shift (gauge) from raw Jacobian.
    """
    J_bar = (p[:, None] * J).sum(dim=0, keepdim=True)  # [1, D]
    return p[:, None] * (J - J_bar)  # [K, D]


def uniform_centered(J: torch.Tensor) -> torch.Tensor:
    """Uniform centering: J - mean(J). Removes average shift."""
    return J - J.mean(dim=0, keepdim=True)


def _last_layer_jacobian(h: torch.Tensor, norm: torch.nn.Module) -> torch.Tensor:
    """Analytical J = d h_norm / d h at the final norm (no backward passes).

    Both Qwen3_5RMSNorm and nn.LayerNorm are handled. J is evaluated at the
    norm INPUT (h), never at h_norm — evaluating the formula at the norm
    output gives a matrix with ~0.87 relative error (no longer LayerNorm-like
    statistics). Qwen3.5's final norm is Qwen3_5RMSNorm:
      h_norm = (h * rsqrt(mean(h^2) + eps) * (1 + w))
      J = r * (diag(1+w) - xhat x^T / D),  r = rsqrt(mean(h^2)+eps), xhat = r*h
    LayerNorm: J = (I - 1/D - xhat xhat^T / D) / std, rows scaled by w.
    """
    x = h.reshape(1, -1).float()  # [1, D], the norm INPUT (row vector)
    D = x.shape[-1]
    eps = float(getattr(norm, "eps", None) or 1e-5)
    w = getattr(norm, "weight", None)
    dev = x.device
    if getattr(norm, "bias", None) is not None:  # LayerNorm family
        mu = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, unbiased=False, keepdim=True) + eps).sqrt()
        xhat = (x - mu) / std  # [1, D]
        J = (
            torch.eye(D, device=dev)
            - torch.ones(D, D, device=dev) / D
            - xhat.T @ xhat / D
        ) / std.squeeze(0)
        if w is not None:
            J = J * w.float().squeeze(0).unsqueeze(1)  # J = diag(w) @ J
        return J
    # RMSNorm family (Qwen3_5RMSNorm, Llama-style).
    r = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    xhat = x * r  # [D]
    # y_j = (1+w_j) r x_j; J = r * diag(1+w) @ (I - xhat xhat^T / D)
    # (derived: J_ij = g_j (r d_ij - r^3 x_i x_j / D) = g_j r (d_ij - xhat_i xhat_j / D))
    g = 1.0 + w.float().squeeze(0) if w is not None else None
    M = torch.eye(D, device=dev) - xhat.T @ xhat / D
    if g is None:
        return r.squeeze(0) * M
    return r.squeeze(0) * torch.diag(g) @ M


def compute_hidden_jacobian(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    create_graph: bool = False,
    verbose: bool = False,
    h: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Jacobian of normalized last hidden state w.r.t. hidden state at layer_idx.

    J_norm = d(h_norm) / d(h_{layer_idx}), shape [D, D].
    Independent of lm_head — computed once, reused across W updates.
    Then J_S = lm_head.weight[token_set] @ J_norm is differentiable through W.

    For last_layer: J_norm is LayerNorm Jacobian (analytical, no backward passes).

    Args:
        create_graph: if True, J_norm has grad_fn.
                      h is kept connected to model params (no detach). Full forward pass.
                      Costs ~D backward passes per sample.
        verbose: print progress during J_norm computation.
    """
    model.eval()
    n_layers = _n_layers(model)

    if create_graph:
        # Keep gradient graph alive: h is connected to model params (+ any hooks)
        S = input_ids.shape[1]
        # Build prefix cache FIRST — before any forward involving hooks.
        prefix_cache = _run_prefix_pass(model, input_ids, attention_mask)

        # Full forward to get h at layer_idx — connected to all model params + hooks
        with torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        assert outputs.hidden_states is not None  # requested
        h = outputs.hidden_states[layer_idx + 1][:, -1:, :]  # [1, 1, D], connected!
        model_dtype = next(model.parameters()).dtype
        assert h is not None  # narrowed: param was Tensor|None
        h = h.to(model_dtype)  # preserve grad graph
        if h.grad_fn is None:
            # Fully frozen graph: no trainable path reaches this layer
            # (e.g. linear_attention pair with no LoRA patch, or a
            # no_grad caller). Make h a leaf so the decoder pass below
            # still yields a differentiable h_norm; J_norm values are
            # identical to the connected case.
            h = h.detach().requires_grad_(True)

        if layer_idx == n_layers - 1:
            with torch.enable_grad():
                h_norm = _lm_module(model).norm(h[:, -1, :])  # [1, D]
        else:
            with torch.enable_grad():
                h_norm = _run_decoder_to_norm_compat(
                    model, prefix_cache, h, S, layer_idx
                )

        # [1, 1, D] — _compute_jacobian_rows_full indexes grad[0, 0, :].
        # enable_grad: autograd.grad needs grad mode ON even when the
        # caller is inside torch.no_grad().
        with torch.enable_grad():
            J_norm = _compute_jacobian_rows_full(
                h_norm, h, create_graph=True, verbose=verbose
            )
        return J_norm

    # Last layer: analytical norm Jacobian, no backward passes.
    if layer_idx == n_layers - 1:
        with torch.no_grad():
            if h is None:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
                assert outputs.hidden_states is not None  # requested
                h = outputs.hidden_states[layer_idx + 1][:, -1:, :]  # [1, 1, D]
            assert h is not None
            h = h.to(next(model.parameters()).dtype)
        return _last_layer_jacobian(h, _lm_module(model).norm)

    # General case: decode through remaining layers + norm
    S = input_ids.shape[1]
    prefix_cache = _run_prefix_pass(model, input_ids, attention_mask)
    if h is None:
        h = _extract_hidden_state(model, input_ids, attention_mask, layer_idx)
    model_dtype = next(model.parameters()).dtype
    h = h.to(model_dtype).detach().requires_grad_(True)
    # Re-enable grad: h_norm must carry a grad_fn over h even if the caller
    # is inside torch.no_grad() (e.g. eval loops in exp16r/exp17). The row
    # loop below (autograd.grad) ALSO needs grad mode ON — a no_grad caller
    # otherwise gets "element 0 ... does not have a grad_fn".
    with torch.enable_grad():
        h_norm = _run_decoder_to_norm_compat(model, prefix_cache, h, S, layer_idx)
        J_norm = _compute_jacobian_rows_full(
            h_norm, h, create_graph=create_graph, verbose=verbose
        )
    return J_norm


def _run_decoder_to_norm(
    text_model, prefix_cache, h: torch.Tensor, S: int, layer_idx: int
) -> torch.Tensor:
    """Decode h through remaining layers, return norm(x) without lm_head."""
    cache_clone = _deep_clone_cache(prefix_cache)
    pos = torch.full((4, 1, 1), S - 1, dtype=torch.long, device=h.device)
    text_position_ids = pos[0]
    rope_position_ids = pos[1:]
    position_embeddings = text_model.rotary_emb(h, rope_position_ids)

    x = h
    n_layers = text_model.config.num_hidden_layers
    start = layer_idx + 1
    causal_mask_mapping = _build_causal_masks(
        text_model, text_position_ids, cache_clone, x
    )
    with _clone_gdn_states_on_entry():
        for i in range(start, n_layers):
            layer_type = text_model.config.layer_types[i]
            attn_mask = causal_mask_mapping[layer_type]
            x = text_model.layers[i](
                x,
                position_embeddings=position_embeddings,
                attention_mask=attn_mask,
                position_ids=text_position_ids,
                past_key_values=cache_clone,
            )
    return text_model.norm(x[:, -1, :])


def _compute_jacobian_rows_full(
    output: torch.Tensor,
    h: torch.Tensor,
    create_graph: bool = False,
    verbose: bool = False,
) -> torch.Tensor:
    """Full [D, D] Jacobian via autograd.grad."""
    import time as _time

    D = h.shape[-1]
    out_flat = output.flatten()
    grads = []
    # create_graph=True: every row must retain the graph — the caller
    # backpropagates a final loss through J_norm afterwards (exp17 bridge
    # loss -> o_proj HVP). retain=(i<D-1) would free the (second-order)
    # graph after the last row, so the caller's first backward would fail
    # with "backward through the graph a second time". Memory is bounded
    # by the refresh schedule (one graph per si, freed at the next refresh
    # / pair end).
    start = _time.time()
    for i in range(D):
        grad = torch.autograd.grad(
            outputs=out_flat[i],
            inputs=h,
            retain_graph=(True if create_graph else i < D - 1),
            create_graph=create_graph,
        )[0]
        grads.append(grad[0, 0, :])
        if verbose and i % 128 == 0:
            elapsed = _time.time() - start
            rate = i / elapsed if elapsed > 0 else 999
            eta = elapsed * (D / i) - elapsed if i > 0 else 0.0
            print(f"        J_norm: {i}/{D} ({rate:.0f}/s, eta={eta:.0f}s)")
    J = torch.stack(grads)
    elapsed_total = _time.time() - start
    if verbose:
        print(f"        J_norm done: {D}/{D} in {elapsed_total:.1f}s")
    return J


def _lm_module(model):
    """The text model with .layers/.norm/.rotary_emb.

    Flat CausalLM: model.model IS the text model. Multimodal wrapper
    (Qwen3_5ForConditionalGeneration): model.model.language_model.
    """
    mm = model.model
    if hasattr(mm, "layers"):
        return mm
    return mm.language_model


def _n_layers(model) -> int:
    """num_hidden_layers for flat or wrapper configs."""
    cfg = model.config
    return int(
        cfg.num_hidden_layers
        if hasattr(cfg, "num_hidden_layers")
        else cfg.text_config.num_hidden_layers
    )


def _layer_types(model) -> list:
    """layer_types for flat or wrapper configs."""
    cfg = model.config
    return list(
        cfg.layer_types if hasattr(cfg, "layer_types") else cfg.text_config.layer_types
    )


def _run_decoder_to_norm_compat(
    model, prefix_cache, h: torch.Tensor, S: int, layer_idx: int
) -> torch.Tensor:
    """_run_decoder_to_norm over _lm_module (flat or nested model)."""
    return _run_decoder_to_norm(_lm_module(model), prefix_cache, h, S, layer_idx)


def h_norm_at_perturbed_h(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """h_norm after the prefix, with the l_S hidden state pinned to `h`.

    Finite-diff support (B2 spot check): call twice, h and h + eps*e_j,
    and difference the two h_norms. h shape [1, 1, D] (any device/dtype);
    the pinning is done on a copy — the input is not mutated.
    """
    S = input_ids.shape[1]
    prefix_cache = _run_prefix_pass(model, input_ids, attention_mask)
    with torch.no_grad():
        return _run_decoder_to_norm_compat(model, prefix_cache, h, S, layer_idx)


def _forward_h_norm(
    model,
    layer_idx,
    input_ids,
    attention_mask,
    h: torch.Tensor | None = None,
):
    """Forward once -> (h, h_norm), h graph-connected at layer l_S.

    Shared by compute_hidden_jacobian_rows and bridge_rows_backward (A4):
    the prefix pass runs before the model forward (hook-safe), h is the
    l_S hidden state (kept in graph so a caller that patched o_proj / lm_head
    gets HVP/bridge grads from the downstream backward), h_norm = RMSNorm(h)
    (last layer) or decoder-to-norm (otherwise).

    h (optional): reuse a pre-computed l_S hidden state instead of doing a
    fresh full forward (v23.1: exp17's KL forward already produced it, so
    the refresh reuses it and only runs the cheap decoder-tail-to-norm).
    """
    model.eval()
    n_layers = _n_layers(model)
    S = input_ids.shape[1]
    prefix_cache = _run_prefix_pass(model, input_ids, attention_mask)
    with torch.enable_grad():
        if h is None:
            # standalone: full forward to obtain h (original A4 behavior)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            assert outputs.hidden_states is not None  # requested
            h = outputs.hidden_states[layer_idx + 1][:, -1:, :]  # [1, 1, D]
        assert h is not None
        h = h.to(next(model.parameters()).dtype)
        if h.grad_fn is None:
            h = h.detach().requires_grad_(True)
        if layer_idx == n_layers - 1:
            h_norm = _lm_module(model).norm(h[:, -1, :])  # [1, D]
        else:
            h_norm = _run_decoder_to_norm_compat(
                model, prefix_cache, h, S, layer_idx
            )
    return h, h_norm


def bridge_rows_backward(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_set: list[int],
    p: torch.Tensor,
    W_rows: torch.Tensor,
    J_tgt_p: torch.Tensor,
    extra_loss: torch.Tensor | None = None,
    chunk_size: int = 8,
    verbose: bool = False,
    h: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    """v23 A4: chunked bridge backward (the v22 OOM fix).

    Forward once + J_bar once, then build each chunk of K rows and backprop
    its (1 - cos) bridge loss immediately (extra_loss — the KL — is
    backproped with the first chunk). Backproping per chunk frees that
    chunk's 2nd-order row-graphs at once, so peak = chunk_size + 1 (the
    shared J_bar) second-order graphs, not K. Per-chunk gradients are
    identical to a single backward on the total (1 - cos).mean() loss.

    Gradients reach W_rows (live lm_head + B_lh @ A_lh) and o_proj (HVP
    through h). Returns (J_S_p [K, D] detached, bridge_loss scalar).

    h (optional): reuse a pre-computed l_S hidden state (v23.1) so this
    does NOT redo the full forward — only the cheap decoder-tail-to-norm
    runs. exp17's KL forward already produced the same h.
    """
    import time as _time

    K = len(token_set)
    h, h_norm = _forward_h_norm(
        model, layer_idx, input_ids, attention_mask, h=h
    )

    def _row_grad(w: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            outputs=h_norm @ w,
            inputs=h,
            retain_graph=True,
            create_graph=True,
        )[0][0, 0, :]

    start = _time.time()
    J_bar = _row_grad(p @ W_rows)  # shared; freed by the last chunk's bwd
    # extra_loss (the KL) gets a SEPARATE backward before the chunk loop.
    # retain_graph=True keeps the shared h-graph alive for the chunks while
    # the KL's OWN forward graph (a separate forward in exp17) frees here.
    # We deliberately do NOT add it to chunk 0: in a shared-forward test/
    # wiring that would free the h-graph the chunks still need.
    if extra_loss is not None:
        extra_loss.backward(retain_graph=True)
    chunks: list[torch.Tensor] = []
    total_bridge = 0.0
    for c in range(0, K, chunk_size):
        hi = min(c + chunk_size, K)
        is_last = (c + chunk_size) >= K
        rows_c = torch.stack(
            [p[i] * (_row_grad(W_rows[i]) - J_bar) for i in range(c, hi)]
        )
        tgt_c = J_tgt_p[c:hi].float()
        cos_c = torch.nn.functional.cosine_similarity(rows_c.float(), tgt_c, dim=1)
        chunk_loss = (1.0 - cos_c).sum() / K
        # retain_graph on all-but-last keeps the ONE shared h forward graph
        # alive so the next chunk can still build its rows; the chunk's
        # 2nd-order row-graph is released by refcounting once we del it
        # (forward nodes stay, referenced by h / h_norm / J_bar).
        chunk_loss.backward(retain_graph=not is_last)
        total_bridge += chunk_loss.item()
        chunks.append(rows_c.detach())
        del rows_c, cos_c, chunk_loss, tgt_c
        if verbose:
            print(
                f"        bridge_rows: {hi}/{K} "
                f"({hi / max(_time.time() - start, 1e-6):.1f} chunks/s)"
            )
    return torch.cat(chunks), total_bridge


def compute_hidden_jacobian_rows(
    model,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_set: list[int],
    p: torch.Tensor,
    W_rows: torch.Tensor,
    create_graph: bool = False,
    chunk_size: int = 8,
    verbose: bool = False,
) -> torch.Tensor:
    """v23 A1: per-row J_S_p = p_i*(J_i - J_bar), no dense [D,D] J_norm.

    K-subset softmax gauge — the SAME gauge as the exp16 target (p_s =
    softmax(full_logits[token_set])) and the exp17 dense bridge
    (softmax_jacobian(W[token_set]@J_norm, p_L)):
      p = re-normalized dist over the K-subset [K], W_rows = W[token_set]
      [K, D]; J_bar = d(h_norm @ (p @ W_rows))/dh; row_i =
      p[i] * (d(h_norm @ W_rows[i])/dh - J_bar).
    h stays graph-connected (via _run_decoder_to_norm_compat, which
    routes through _lm_module so both flat CausalLM and the multimodal
    wrapper work), so a caller that patched the target layer's o_proj
    and/or lm_head gets its HVP/bridge gradients from the downstream
    loss backward. chunk_size (A4) bounds the 2nd-order graphs alive
    at once; create_graph=True retains them for a later loss backward,
    create_graph=False frees each chunk after the gradient (for cheap
    no-grad refreshes / evals).
    """
    model.eval()
    n_layers = _n_layers(model)
    S = input_ids.shape[1]

    h, h_norm = _forward_h_norm(model, layer_idx, input_ids, attention_mask)

    return _compute_jacobian_rows_matrixfree(
        h_norm,
        h,
        W_rows,
        p,
        token_set,
        create_graph=create_graph,
        chunk_size=chunk_size,
        verbose=verbose,
    )


def _compute_jacobian_rows_matrixfree(
    h_norm: torch.Tensor,
    h: torch.Tensor,
    W_rows: torch.Tensor,
    p: torch.Tensor,
    token_set: list[int],
    create_graph: bool = True,
    chunk_size: int = 8,
    verbose: bool = False,
) -> torch.Tensor:
    """J_S_p [K, D] rows, K-subset softmax gauge, no dense [D,D] J_norm.

    p [K] = re-normalized softmax over the K-subset (exp16/exp17 gauge),
    W_rows [K, D] = the selected rows. J_bar = d(h_norm @ (p @ W_rows))/dh;
    row_i = p[i]*(d(h_norm @ W_rows[i])/dh - J_bar). Identical to
    softmax_jacobian(W_rows @ J_norm, p) but with K backward passes
    instead of a retained [D,D] 2nd-order matrix (v22 OOM).
    Chunked (A4): only chunk_size row-graphs alive at once.
    """
    import time as _time

    K = len(token_set)

    def _row_grad(w: torch.Tensor) -> torch.Tensor:
        return torch.autograd.grad(
            outputs=h_norm @ w,
            inputs=h,
            retain_graph=True,
            create_graph=create_graph,
        )[0][0, 0, :]

    start = _time.time()
    J_bar = _row_grad(p @ W_rows)  # [D]; p@W_rows = effective selected row
    rows = []
    for c in range(0, K, chunk_size):
        for local, i in enumerate(range(c, min(c + chunk_size, K))):
            row = p[i] * (_row_grad(W_rows[i]) - J_bar)
            rows.append(row)
        if verbose:
            done = min(c + chunk_size, K)
            el = _time.time() - start
            print(f"        J_S_p: {done}/{K} ({done / max(el, 1e-6):.1f}/s)")
    out = torch.stack(rows)
    elapsed = _time.time() - start
    if verbose:
        print(f"        J_S_p done: {K}/{K} in {elapsed:.1f}s")
    return out

import hashlib
import os

import torch

_gramian_cache: dict[str, torch.Tensor] = {}
_GRAMIAN_DIR = os.path.join("results", ".gramians")


def jacobian_gap(
    J_L: torch.Tensor,
    J_S: torch.Tensor,
    E_L: torch.Tensor,
    E_S: torch.Tensor,
    probs_L: torch.Tensor,
    threshold: float = 1e-6,
) -> float:
    """Compute Jacobian bridge loss between large and small models.

    The bridge chain: J_L @ E_L^T @ E_S @ (E_S^T @ E_S)^{-1}.
    The inverse Gramian (E_S^T @ E_S)^{-1} corrects for non-orthogonal
    embeddings, so identity (J_L == J_S, E_L == E_S) yields loss ~0.
    """
    G_inv = _inverse_gramian(E_S)
    J_L_proj = _project_to_vocab(J_L, E_L)
    J_S_target = _project_to_small(J_L_proj, E_S, G_inv)
    mask = _valid_mask(J_S, J_S_target, threshold)

    if mask.sum() == 0:
        return 1.0

    return _weighted_cosine_loss(J_S, J_S_target, probs_L, mask)


def _project_to_vocab(J_L: torch.Tensor, E_L: torch.Tensor) -> torch.Tensor:
    """Project J_L [|U|, D_L] through E_L^T -> [|U|, V]."""
    return J_L.float() @ E_L.float().T


def _project_to_small(
    J_L_proj: torch.Tensor, E_S: torch.Tensor, G_inv: torch.Tensor
) -> torch.Tensor:
    """Project [|U|, V] through E_S @ G_inv -> [|U|, D_S].

    G_inv = (E_S^T @ E_S)^{-1} corrects non-orthogonal embeddings.
    """
    return J_L_proj @ E_S.float() @ G_inv


def _inverse_gramian(E: torch.Tensor) -> torch.Tensor:
    """Compute (E^T @ E)^{-1}, cached on disk per embedding hash."""
    key = _gramian_key(E)
    if key not in _gramian_cache:
        path = os.path.join(_GRAMIAN_DIR, f"{key}.pt")
        if os.path.exists(path):
            _gramian_cache[key] = torch.load(path, weights_only=True)
        else:
            _gramian_cache[key] = _compute_inverse_gramian(E)
            os.makedirs(_GRAMIAN_DIR, exist_ok=True)
            torch.save(_gramian_cache[key], path)
    return _gramian_cache[key].to(E.device)


def _gramian_key(E: torch.Tensor) -> str:
    """Stable hash of an embedding matrix for disk caching.

    Uses first/last rows as representative content — hashing the full
    248k×1024 matrix would be prohibitively expensive.
    """
    header = f"{tuple(E.shape)}|{str(E.device)}".encode()
    data = header + E[0].cpu().numpy().tobytes() + E[-1].cpu().numpy().tobytes()
    h = hashlib.sha256(data).hexdigest()[:16]
    return f"g_{h}"


def _compute_inverse_gramian(E: torch.Tensor) -> torch.Tensor:
    """Compute (E^T @ E)^{-1} via pseudoinverse for numerical stability."""
    G = E.float().T @ E.float()
    return torch.linalg.pinv(G)


def _valid_mask(
    J_S: torch.Tensor, J_target: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Mask rows where either norm < threshold."""
    n1 = J_S.float().norm(dim=1)
    n2 = J_target.float().norm(dim=1)
    return (n1 > threshold) & (n2 > threshold)


def _weighted_cosine_loss(
    J_S: torch.Tensor,
    J_target: torch.Tensor,
    probs: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Weighted cosine dissimilarity loss."""
    cos_sim = torch.nn.functional.cosine_similarity(
        J_S.float()[mask], J_target[mask], dim=1
    )
    w = probs[mask]
    w = w / (w.sum() + 1e-8)
    return (w * (1.0 - cos_sim)).sum().item()

# Deprecated alias: the "bridge" historically meant the 1-cos LOSS (lower=better).
# Terminology now: bridge/cos = alignment (higher=better); gap = 1-cos (lower=better).
jacobian_bridge = jacobian_gap

import torch


def compute_esg(
    jacobians: list[torch.Tensor],
    n_components: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ESG from a list of Jacobian matrices.

    Args:
        jacobians: list of N Jacobian matrices, each [|U|, D].
        n_components: number of top eigenvalues/vectors to return.

    Returns:
        eigenvalues: [n_components] sorted descending.
        eigenvectors: [n_components, |U|] corresponding eigenvectors.
    """
    G = _build_gramian(jacobians)
    eigenvalues, eigenvectors = _spectral_decomposition(G, n_components)
    return eigenvalues, eigenvectors


def _build_gramian(jacobians: list[torch.Tensor]) -> torch.Tensor:
    """G = (1/N) * sum_i (J_i @ J_i^T)."""
    U = jacobians[0].shape[0]
    device = jacobians[0].device
    G = torch.zeros(U, U, device=device, dtype=torch.float)

    for J in jacobians:
        G += J.float() @ J.float().T
    G /= len(jacobians)
    return G


def _spectral_decomposition(
    G: torch.Tensor, n_components: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigen decomposition, sorted descending."""
    eigenvalues, eigenvectors = torch.linalg.eigh(G)

    # Sort descending
    idx = eigenvalues.argsort(descending=True)
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components].T

    return eigenvalues, eigenvectors

from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def uniform_layer_mapping(n_layers_large: int, n_layers_small: int) -> dict[int, int]:
    """Uniform layer mapping from large to small model layers."""
    mapping = {}
    for i in range(n_layers_large):
        j = round(i * (n_layers_small - 1) / (n_layers_large - 1))
        mapping[i] = j
    return mapping


def get_layer_types(model: AutoModelForCausalLM) -> list[str]:
    """Get layer types from model config (full_attention / linear_attention)."""
    return list(model.config.layer_types)  # type: ignore[attr-defined]


_DOMAIN_DATASETS = {
    "wiki": ("allenai/c4", "en", "train", "text"),
    "math": ("openai/gsm8k", "main", "train", "question"),
    "code": ("neyralabs/apex-agents", None, "train", "prompt"),
    "chat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", "messages"),
}


def _load_dataset_for_domain(
    domain: str,
) -> tuple[str, str | None, str, str]:
    """Return (dataset_id, config, split, text_field) for a domain.

    Falls back to wiki (C4) if the target dataset is unavailable.
    """
    ds_id, cfg, split, field = _DOMAIN_DATASETS[domain]
    try:
        kw: dict[str, object] = {"split": split, "streaming": True}
        if cfg is not None:
            kw["name"] = cfg
        load_dataset(ds_id, **kw)  # type: ignore[call-overload]
        return ds_id, cfg, split, field
    except Exception:
        print(f"  Warning: domain {domain} ({ds_id}) unavailable, falling back to C4")
        return _DOMAIN_DATASETS["wiki"]


def _extract_text_from_sample(
    sample: dict,
    field: str,
) -> str:
    """Extract text from a dataset sample, handling different formats."""
    if field == "messages":
        # Chat format: list of {role, content}
        texts = [m.get("content", "") for m in sample.get("messages", [])]
        return " ".join(texts)
    val = sample.get(field, "")
    if isinstance(val, list):
        # e.g. code_contests solutions: list of strings
        return "\n".join(str(v) for v in val)
    if isinstance(val, dict):
        return str(val)
    return val


def load_calibration_data(
    model_name: str,
    max_length: int = 512,
    min_length: int = 256,
    n_samples: int = 4,
    domain: str = "wiki",
) -> list[dict[str, torch.Tensor]]:
    """Load and tokenize calibration data.

    domain: 'wiki' (C4), 'math' (GSM8K), 'code' (apex-agents), 'chat' (ultrachat).
    Falls back to wiki if the target dataset is unavailable.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds_id, cfg, split, field = _load_dataset_for_domain(domain)
    kw: dict[str, object] = {"split": split, "streaming": True}
    if cfg is not None:
        kw["name"] = cfg
    dataset = load_dataset(ds_id, **kw)  # type: ignore[call-overload]

    samples = []
    for raw in dataset:
        text = _extract_text_from_sample(raw, field)
        if not text:
            continue
        tokens = tokenizer(
            text, max_length=max_length, truncation=True, return_tensors="pt"
        )
        if tokens["input_ids"].shape[1] >= min_length:
            samples.append(tokens)
        if len(samples) >= n_samples:
            break
    return samples


def build_global_token_set(
    model: AutoModelForCausalLM,
    calibration_samples: list[dict[str, torch.Tensor]],
    K: int = 64,
) -> list[int]:
    """Build global token set from top-K logits across all samples."""
    token_counter: Counter[int] = Counter()
    model.eval()  # type: ignore[attr-defined]

    device = next(model.parameters()).device  # type: ignore[attr-defined]
    with torch.no_grad():
        for sample in calibration_samples:
            inputs = {
                "input_ids": sample["input_ids"].to(device),
                "attention_mask": sample["attention_mask"].to(device),
            }
            outputs = model(**inputs)  # type: ignore[operator]
            logits = outputs.logits[:, -1, :]
            top_idx = logits.topk(K, dim=-1).indices[0]
            token_counter.update(top_idx.tolist())

    U = [tid for tid, _ in token_counter.most_common(K)]
    return sorted(U)


def load_models(
    large_name: str, small_name: str, dtype: torch.dtype = torch.float16
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load large and small models with eager attention."""
    kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": "eager",
    }
    large = AutoModelForCausalLM.from_pretrained(large_name, **kwargs)
    large.eval()  # type: ignore[attr-defined]
    small = AutoModelForCausalLM.from_pretrained(small_name, **kwargs)
    small.eval()  # type: ignore[attr-defined]
    return large, small


def extract_embeddings(
    model: AutoModelForCausalLM,
) -> torch.Tensor:
    """Extract embedding matrix from model."""
    return model.model.embed_tokens.weight.detach()  # type: ignore[attr-defined]

import os

# Max depth under input_root to search (producer puts cache at depth 2:
# results/targets; 4 covers any reasonable layout).
_MAX_DEPTH = 4


def _has_pt(d: str) -> bool:
    try:
        return any(f.endswith(".pt") for f in os.listdir(d))
    except OSError:
        return False


def _walk(root: str, depth: int):
    """Yield dirs under root (listdir-based), depth-limited."""
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return
    for e in entries:
        p = os.path.join(root, e)
        if _has_pt(p):
            yield p
        if depth > 1:
            yield from _walk(p, depth - 1)


def find_targets_dir(preferred: str, input_root: str = "/kaggle/input") -> str:
    """Return the dir holding the target-cache *.pt files.

    Order: `preferred` (local path / runner-injected), then a listdir walk
    of every mount under `input_root`. Raises FileNotFoundError with the
    mount inventory when nothing is found (no silent fallback, no listdir crash).
    """
    if _has_pt(preferred):
        return preferred
    for hit in _walk(input_root, _MAX_DEPTH):
        return hit
    mounts = []
    try:
        for d in sorted(os.listdir(input_root)):
            try:
                mounts.append(f"{d}/:{os.listdir(os.path.join(input_root, d))[:4]}")
            except OSError:
                mounts.append(f"{d}/:<unreadable>")
    except OSError:
        mounts.append("<no input mount>")
    raise FileNotFoundError(
        f"No target cache .pt in {preferred} nor under {input_root}/ "
        f"(mounts: {mounts}). Run Stage 0 first — it must publish results/targets/*.pt."
    )



def _uniform_layer_mapping(n_L: int, n_S: int) -> dict[int, int]:
    """Uniform large->small layer mapping: l_L -> round(l_L*(n_S-1)/(n_L-1)).

    Inlined copy of prism.calibration.uniform_layer_mapping (identical formula)
    so this module stays standalone for kernel inlining.
    """
    return {i: round(i * (n_S - 1) / (n_L - 1)) for i in range(n_L)}


def get_layer_types(model) -> list[str]:
    """Per-layer attention type from model.config.layer_types.

    Qwen3.5 style hybrid: entries 'full_attention' or 'linear_attention'.
    `model.config` (post-AutoModel load) exposes `layer_types` at top level
    (Qwen3_5TextConfig). Returns lowercase list.
    """
    types = getattr(model.config, "layer_types", None)
    if types is None:
        # Fallback: nested text_config (defensive; Qwen3_5 exposes top-level).
        tc = getattr(model.config, "text_config", None)
        types = getattr(tc, "layer_types", None)
    if types is None:
        raise RuntimeError(
            "model.config has no layer_types (and no text_config.layer_types)"
        )
    return [str(t) for t in types]


def select_pairs(n_L: int, n_S: int, layer_types: list[str]) -> list[tuple[int, int]]:
    """Return (l_L, l_S) pairs: full-attn l_S + 0_0 + last, sorted by l_S."""
    if n_L < 1 or n_S < 1 or len(layer_types) != n_S:
        raise ValueError(
            f"invalid: n_L={n_L}, n_S={n_S}, layer_types={len(layer_types)}"
        )
    mapping = _uniform_layer_mapping(n_L, n_S)
    full_S = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    selected = {0, n_S - 1} | full_S
    pairs = [_nearest_preimage(mapping, j) for j in sorted(selected)]
    if (n_L - 1, n_S - 1) not in pairs:
        pairs.append((n_L - 1, n_S - 1))
    return pairs


def _nearest_preimage(mapping: dict[int, int], j: int) -> tuple[int, int]:
    """l_L = argmin |mapping[l] - j|, tie -> smaller l (the selected l_S is j)."""
    best_l, best_d = 0, float("inf")
    for li in range(len(mapping)):
        d = abs(mapping[li] - j)
        if d < best_d:
            best_l, best_d = li, d
    return (best_l, j)


def classify_pair(l_S: int, layer_types: list[str]) -> str:
    """'bridge' if small-model layer l_S has o_proj (full_attention), else 'kl_only'."""
    return (
        "bridge"
        if l_S < len(layer_types) and layer_types[l_S] == "full_attention"
        else "kl_only"
    )


# ---------- exp18_logit_kd ----------

import json
import os
import re
import contextlib
import time

import torch
# --- XLA / TPU support (no-op on GPU/CPU; active when device == "xla") ---
try:
    import torch_xla.core.xla_model as _xm
    _XLA = True
except Exception:
    _xm = None
    _XLA = False

@contextlib.contextmanager
def _step_cm(dev):
    """PJRT training step: mark_step() after the block commits the XLA graph
    on TPU; no-op on GPU/CPU. PJRT uses mark_step(), not old XRT-only xm.step()."""
    yield
    if _XLA and str(dev).startswith("xla"):
        _xm.mark_step()

import torch.nn.functional as F
from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

def main_18logitkd(
    small_model: str = "Qwen/Qwen3.5-0.8B",
    device: str = "auto",
    max_length: int = 64,
    min_length: int = 32,
    targets_dir: str = "results/targets",
    results_dir: str = "results",
):
    torch.manual_seed(42)
    start = time.time()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "xla":
        assert _XLA, "torch_xla not installed"
        device = _xm.xla_device()
        torch.set_num_threads(1)
    if device == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}")
    os.makedirs(results_dir, exist_ok=True)

    # -- Load small model -------------------------------------------------
    print(f"Loading small model: {small_model}")
    mS = AutoModelForCausalLM.from_pretrained(
        small_model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)  # type: ignore[arg-type]
    mS.eval()

    # -- Calibration data -------------------------------------------------
    n_samples = 2 if device == "cpu" else 4
    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        small_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )
    _orig_ids = [s["input_ids"].view(-1) for s in samples]

    # -- Load target cache ------------------------------------------------
    # FUSE-safe, layout-independent (prism.cache_io): preferred dir, else
    # listdir walk of /kaggle/input — no hardcoded slug/subpath.
    targets_dir = find_targets_dir(targets_dir)
    print(f"Using target cache at {targets_dir}")

    cache_files = sorted(
        f
        for f in os.listdir(targets_dir)
        if f.endswith(".pt") and not f.startswith(".")
    )
    if not cache_files:
        raise FileNotFoundError(
            f"No target cache .pt files found in {targets_dir}. Run Stage 0 first."
        )

    cache_by_pair: dict[str, list[dict]] = {}
    for cf in cache_files:
        data = torch.load(os.path.join(targets_dir, cf), weights_only=True)
        pair = data["pair"]
        m = re.search(r"_(\d+)\.pt$", cf)
        data["_si"] = int(m.group(1)) if m else -1
        cache_by_pair.setdefault(pair, []).append(data)
    for pair in cache_by_pair:
        cache_by_pair[pair].sort(key=lambda t: t["_si"])

    # Same order as Exp17 late→early
    all_pairs = sorted(cache_by_pair.keys(), key=lambda p: int(p.split("_")[1]))
    late_to_early = list(reversed(all_pairs))
    if device == "cpu":
        wanted = [23, 15, 7]
        late_to_early = [p for p in late_to_early if int(p.split("_")[1]) in wanted]
        if not late_to_early:
            late_to_early = all_pairs[:7]

    n_pairs = len(late_to_early)
    print(f"Pairs ({n_pairs}, late→early): {late_to_early}")

    steps = 100 if str(device).startswith(("cuda", "xla")) else 40
    r = 4
    lr = 1e-3

    per_layer_results = []

    # Cache original logits
    orig_logits_cache: list[torch.Tensor] = []
    with torch.no_grad():
        for s in samples[:n_samples]:
            ids = s["input_ids"].to(device)
            attn = s["attention_mask"].to(device)
            orig_logits_cache.append(
                mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
            )

    # -- Iterate pairs late→early -----------------------------------------
    for pair_key in late_to_early:
        targets = cache_by_pair[pair_key]
        l_S = int(pair_key.split("_")[1])
        token_set = targets[0]["token_set"]
        n_use = min(n_samples, len(targets))
        print(f"\n  Pair {pair_key} (l_S={l_S}, {n_use} targets)")

        # Freeze all params
        for p in mS.parameters():
            p.requires_grad = False

        # LoRA on lm_head only (same convention as Exp17 for fair comparison)
        # A_lh [r, D], B_lh [V, r], delta = B_lh @ A_lh [V, D]
        V, D = mS.lm_head.weight.shape
        A_lh = torch.nn.Parameter(torch.randn(r, D, device=device) * 0.02)
        B_lh = torch.nn.Parameter(torch.zeros(r, V, device=device).T)  # [V, r]

        trainable = [A_lh, B_lh]
        for p in trainable:
            p.requires_grad = True

        optimizer = torch.optim.Adam(trainable, lr=lr)

        for step in range(steps):
            with _step_cm(device):
                optimizer.zero_grad()
                total_loss_val = 0.0
                nan_hit = False

                for si in range(n_use):
                    s = samples[si]
                    tgt = targets[si]
                    ids = s["input_ids"].to(device)
                    attn = s["attention_mask"].to(device)

                    # Logit-KD: KL(p_ed ‖ p_L) on token_set subset.
                    # Target pairing: skip bridge-of-wrong-sample; KL leash below
                    # (eval) uses only the small model.
                    if tgt.get("_si") != si:
                        continue

                    # W_eff inside the sample loop: fresh graph per backward.
                    # (W_eff shared across samples would die on the 2nd
                    # backward() — no retain_graph on Kaggle 16GB.)
                    W_eff = mS.lm_head.weight + B_lh @ A_lh  # [V, D]
                    lm_b = mS.lm_head.bias if mS.lm_head.bias is not None else 0
                    # KD target: fp32 p_L_full restricted to token_set, renormalized.
                    # (tgt["p"] is the bf16 gauge weight for the Jacobian — rounded,
                    # and its mass over token_set is renormalized to 1, so it is not
                    # the true restricted distribution; p_L_full is the exact one.)
                    p_L_full = tgt["p_L_full"].to(device).float()
                    p_L_sub = p_L_full[
                        torch.as_tensor(token_set, dtype=torch.long, device=device)
                    ]
                    p_L_sub = p_L_sub / p_L_sub.sum()
                    out = mS.model(input_ids=ids, attention_mask=attn)
                    hidden = out.last_hidden_state[:, -1, :]  # [1, D]
                    logits_ed = hidden @ W_eff.T + lm_b  # [1, V]
                    p_ed_sub = torch.softmax(logits_ed[0, token_set].float(), dim=-1)
                    kd_loss = (
                        p_ed_sub * torch.log(p_ed_sub.clamp_min(1e-12) / (p_L_sub + 1e-12))
                    ).mean()
                    # NaN guard: one NaN step poisons all subsequent steps of the pair
                    if not torch.isfinite(kd_loss):
                        print(f"  [pair] sample {si}: NaN KD loss, stopping pair")
                        nan_hit = True
                        break
                    kd_loss.backward()
                    total_loss_val += kd_loss.item()

                if nan_hit:
                    break

                optimizer.step()

            if (step + 1) % 10 == 0 or step == 0:
                print(f"    step {step + 1}/{steps}: loss={total_loss_val:.4f}")

        # -- Apply LoRA permanently (same as Exp17: W += B@A, no in-place
        # .data swap — the swap breaks the autograd graph of W_eff).
        delta_lm = B_lh @ A_lh  # [V, D]
        with torch.no_grad():
            mS.lm_head.weight.data += delta_lm

        # Evaluate bridge (even though we trained with KD!)
        bridge_vals = []
        with torch.no_grad():
            for si in range(n_use):
                s = samples[si]
                tgt = targets[si]
                if tgt.get("_si") != si:
                    continue
                ids = s["input_ids"].to(device)
                attn = s["attention_mask"].to(device)
                J_tgt_p = tgt["J_tgt_p"].to(device)
                J_S_ed = compute_jacobian_for_tokens(mS, l_S, ids, attn, token_set)
                p_L = tgt["p"].to(device)
                J_S_ed_p = softmax_jacobian(J_S_ed, p_L)
                cos = (
                    torch.nn.functional.cosine_similarity(
                        J_S_ed_p.float(), J_tgt_p.float(), dim=1
                    )
                    .mean()
                    .item()
                )
                bridge_vals.append(cos)

        # -- Cache original logits for the LAST layer ---------------------
        # Computed after every pair; a per-pair snapshot taken before the
        # pair would be a layer mismatch: KL(p_ed(layer l) || p_orig(l_prev)).
        kl_pairs = []
        with torch.no_grad():
            for si in range(n_use):
                if targets[si].get("input_ids") is None:
                    continue  # pre-v2 cache: no pairing field
                tids = targets[si]["input_ids"].view(-1)
                match = next(
                    (
                        si2
                        for si2 in range(n_samples)
                        if samples[si2]["input_ids"].shape[0] == tids.shape[0]
                        and bool((samples[si2]["input_ids"].view(-1) == tids).all())
                    ),
                    None,
                )
                if match is not None:
                    kl_pairs.append((si, match))
        kl_sum = 0.0
        with torch.no_grad():
            for si, si2 in kl_pairs:
                s = samples[si2]
                ids = s["input_ids"].to(device)
                attn_s = s["attention_mask"].to(device)
                kl = F.kl_div(
                    F.log_softmax(orig_logits_cache[si2].float(), dim=-1),
                    F.log_softmax(
                        mS(input_ids=ids, attention_mask=attn_s)
                        .logits[:, -1, :]
                        .float(),
                        dim=-1,
                    ),
                    log_target=True,  # both args are log-probs: NaN without it
                    reduction="batchmean",
                )
                kl_sum += kl.item()
        kl_avg = kl_sum / max(1, len(kl_pairs))

        layer_result = {
            "pair": pair_key,
            "bridge_mean": sum(bridge_vals) / len(bridge_vals) if bridge_vals else 0,
            "KL_ed_orig": kl_avg,
        }
        per_layer_results.append(layer_result)
        print(f"    bridge={layer_result['bridge_mean']:.4f}, KL={kl_avg:.4f}")

        # Update orig logits for next layer
        with torch.no_grad():
            for si, s in enumerate(samples[:n_samples]):
                ids = s["input_ids"].to(device)
                attn_s = s["attention_mask"].to(device)
                orig_logits_cache[si] = mS(input_ids=ids, attention_mask=attn_s).logits[
                    :, -1, :
                ]

        del A_lh, B_lh

    # -- Compare with Exp17 if available ----------------------------------
    # Per-pair, aligned on pair_key — the v1 last-vs-first difference assumed
    # both experiments walked the same pair list, which the shallow-pair
    # fallback in 17/exp16r broke (and which stage0 v2 eliminates).
    exp17_path = os.path.join(results_dir, "exp17_lora_distill.json")
    comparison = None
    if os.path.exists(exp17_path):
        with open(exp17_path) as f:
            exp17_data = json.load(f)
        per_layer_17 = exp17_data.get("per_layer") or exp17_data.get(
            "per_order", {}
        ).get("late_to_early", {}).get("per_layer", [])
        if per_layer_17 and per_layer_results:
            p17 = {d["pair"]: d for d in per_layer_17 if "pair" in d}
            shared = [p for p in late_to_early if p in p17]
            if shared:
                # Cumulative sum over shared pairs: sum(kl_i - kl_{i-1}) = kl_last - kl_first
                bridge_17 = sum(float(p17[p]["bridge_mean"]) for p in shared) - float(
                    p17[shared[0]]["bridge_mean"]
                )
                kl_17 = float(p17[shared[-1]]["KL_ed_orig"]) - float(
                    p17[shared[0]]["KL_ed_orig"]
                )
                k18 = {d["pair"]: d for d in per_layer_results}
                bridge_18 = sum(float(k18[p]["bridge_mean"]) for p in shared) - float(
                    k18[shared[0]]["bridge_mean"]
                )
                kl_18 = float(k18[shared[-1]]["KL_ed_orig"]) - float(
                    k18[shared[0]]["KL_ed_orig"]
                )
                ratio_17 = bridge_17 / kl_17 if abs(kl_17) > 1e-8 else float("inf")
                ratio_18 = bridge_18 / kl_18 if abs(kl_18) > 1e-8 else float("inf")
                comparison = {
                    "shared_pairs": shared,
                    "delta_bridge_17": bridge_17,
                    "delta_KL_17": kl_17,
                    "delta_bridge_18": bridge_18,
                    "delta_KL_18": kl_18,
                    "ratio_17_bridge_per_KL": ratio_17,
                    "ratio_18_bridge_per_KL": ratio_18,
                }
                print("\nComparison Exp17 vs Exp18 (shared pairs):")
                print(
                    f"  Exp17 (bridge): Δbridge={bridge_17:.4f}, ΔKL={kl_17:.4f}, ratio={ratio_17:.4f}"
                )
                print(
                    f"  Exp18 (KD):     Δbridge={bridge_18:.4f}, ΔKL={kl_18:.4f}, ratio={ratio_18:.4f}"
                )
            else:
                print("\nExp17 JSON found but no shared pair keys — comparison skipped")

    # -- Save --------------------------------------------------------------
    result = {
        "description": "Control experiment for Exp17: same LoRA configuration (r=4 on lm_head) but loss is KL divergence between edited-small logits and large-model logits on token_set, instead of bridge loss. Purpose: compare delta-bridge per unit delta-KL between bridge-guided (Exp17) and logit-KD (Exp18) to test whether bridge guidance is more efficient at aligning sensitivity.",
        "per_layer": per_layer_results,
        "pairs": late_to_early,
        "r": r,
        "steps": steps,
        "comparison_with_exp17": comparison,
        "device": device,
        "elapsed_seconds": time.time() - start,
    }
    path = os.path.join(results_dir, "exp18_logit_kd.json")
    result["params"] = {
        "small_model": small_model,
        "device": device,
        "max_length": max_length,
        "min_length": min_length,
        "targets_dir": targets_dir,
    }
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {path}")
    print(f"Total time: {time.time() - start:.0f}s")


# ---------- run exp18_logit_kd ----------
gc.collect()
sig = inspect.signature(main_18logitkd)
params = list(sig.parameters)
kwargs = {}
if 'small_model' in params: kwargs['small_model'] = 'Qwen/Qwen3.5-0.8B'
if 'large_model' in params: kwargs['large_model'] = 'Qwen/Qwen3.5-4B'
if 'model' in params: kwargs['model'] = 'Qwen/Qwen3.5-0.8B'
if 'device' in params: kwargs['device'] = 'xla'
if 'results_dir' in params: kwargs['results_dir'] = RES
if 'targets_dir' in params: kwargs['targets_dir'] = os.path.join(RES, 'targets')
if 'small_model' in kwargs and 'model' in kwargs: del kwargs['model']
t0 = time.time()
try:
    main_18logitkd(**kwargs)
    print(f'[OK] exp18_logit_kd {time.time()-t0:.0f}s')
except Exception as e:
    print(f'[FAIL] exp18_logit_kd: {e}')
    traceback.print_exc()
print("\nResults:")
if os.path.isdir(RES):
    for fn in sorted(os.listdir(RES)):
        print(f'  {fn} {os.path.getsize(os.path.join(RES, fn)):,} bytes')
