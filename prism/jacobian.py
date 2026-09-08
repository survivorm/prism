"""Jacobian computation for decoder layers."""

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
