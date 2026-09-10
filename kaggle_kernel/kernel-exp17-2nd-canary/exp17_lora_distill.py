from __future__ import annotations
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-deps'] + ['transformers==5.15.1', 'safetensors==0.8.0'])
import transformers as _tf
_v = _tf.__version__
print(f'  [pip] transformers={_v}')
if _v != '5.15.1':
    print(f'[FAIL] transformers pinned to 5.15.1 but got {_v}; aborting before model load')
    sys.exit(0)

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


# ---------- exp17_lora_distill ----------

import json
import os
import re
import contextlib
import time
from collections.abc import Callable

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
    yield
    if _XLA and str(dev).startswith("xla"):
        _xm.mark_step()

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

def _layer_order(pairs: list[str], device: str) -> dict[str, list[str]]:
    late_to_early = sorted(pairs, key=lambda p: int(p.split("_")[1]), reverse=True)
    if device == "cpu":
        wanted = [23, 15, 7]
        subset = [p for p in late_to_early if int(p.split("_")[1]) in wanted]
        if not subset:
            subset = late_to_early[:7]
        return {"late_to_early": subset}
    return {
        "late_to_early": late_to_early,
        "early_to_late": list(reversed(late_to_early)),
    }

def main_17loradistill(
    small_model: str = "Qwen/Qwen3.5-0.8B",
    device: str = "auto",
    max_length: int = 64,
    min_length: int = 32,
    targets_dir: str = "results/targets",
    results_dir: str = "results",
    steps: int = 0,
    fd_check: int = 0,
    pairs: str = "",
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

    # -- Calibration data -------------------------------------------------
    n_samples = 2 if device == "cpu" else 4
    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        small_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )

    # -- Load target cache ------------------------------------------------
    # FUSE-safe, layout-independent (prism.cache_io): preferred dir, else
    # listdir walk of /kaggle/input — no hardcoded slug/subpath.
    targets_dir = find_targets_dir(targets_dir)
    print(f"Using target cache at {targets_dir}")

    # -- Load target cache ------------------------------------------------
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
        # Tag with the sample index stored in the filename ({pair}_{si}.pt)
        m = re.search(r"_(\d+)\.pt$", cf)
        data["_si"] = int(m.group(1)) if m else -1
        cache_by_pair.setdefault(pair, []).append(data)
    # Sort each pair's targets by sample index (filename sort breaks at si>=10)
    for pair in cache_by_pair:
        cache_by_pair[pair].sort(key=lambda t: t["_si"])

    all_pairs = sorted(cache_by_pair.keys(), key=lambda p: int(p.split("_")[1]))
    if pairs:
        allowed = {s.strip() for s in pairs.split(",") if s.strip()}
        missing = allowed - set(all_pairs)
        if missing:
            raise ValueError(f"--pairs {sorted(missing)} not in target cache")
        all_pairs = [p for p in all_pairs if p in allowed]
    orders = _layer_order(all_pairs, device)

    # C1: v23 = max 300 steps per pair (early-stop may end sooner). Local
    # CPU stays at 40 (smoke) unless --steps is given.
    if steps <= 0:
        steps = 500 if str(device).startswith(("cuda", "xla")) else 40
    r = 4
    lr = 3e-3
    kl_weight = 0.1

    per_order_results = {}

    for order_name, pair_seq in orders.items():
        print(f"\n{'=' * 60}")
        print(f"Order: {order_name} ({len(pair_seq)} pairs)")
        print(f"{'=' * 60}")

        # Load fresh model for this order
        mS = AutoModelForCausalLM.from_pretrained(
            small_model,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(device)  # type: ignore[arg-type]
        mS.eval()
        layer_types = get_layer_types(mS)

        # Cache original logits (updated after each layer)
        orig_logits_cache: list[torch.Tensor] = []
        with torch.no_grad():
            for s in samples[:n_samples]:
                ids = s["input_ids"].to(device)
                attn = s["attention_mask"].to(device)
                orig_logits_cache.append(
                    mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
                )

        per_layer_results: list[dict] = []

        for pair_key in pair_seq:
            targets = cache_by_pair[pair_key]
            # D1: pair targets to samples by EXACT input_ids (v2-cache has
            # input_ids), not filename order. stage0 and exp17 load the same
            # calibration samples (same loader/seed/lengths), so all n_samples
            # should match; on mismatch that sample's bridge is skipped (warn).
            t_by_ids = {tuple(t["input_ids"].reshape(-1).tolist()): t for t in targets}
            sample_targets: list[dict | None] = [
                t_by_ids.get(tuple(s["input_ids"].reshape(-1).tolist()))
                for s in samples[:n_samples]
            ]
            n_matched = sum(t is not None for t in sample_targets)
            if n_matched < n_samples:
                print(
                    f"    [warn] pairing: {n_matched}/{n_samples} samples matched "
                    f"by input_ids in {pair_key} — unmatched bridges skipped"
                )

            l_S = int(pair_key.split("_")[1])
            token_set = targets[0]["token_set"]
            n_use = min(n_samples, len(targets))
            print(
                f"\n  Pair {pair_key} (l_S={l_S}, {n_use} targets, "
                f"matched {n_matched}/{n_samples} by input_ids)"
            )
            layer_start = time.time()
            # Mode guard (v23): full-attn student layer has o_proj -> bridge
            # (o_proj LoRA + J refresh). GDN linear-attn has no o_proj ->
            # kl_only: the J refresh would build a 2nd-order graph that
            # reaches no trainable o_proj = pure waste. classify_pair defines
            # exactly this split; A_attn (below) is its runtime proxy.
            pair_mode = classify_pair(l_S, layer_types)

            # Freeze all model params
            for p in mS.parameters():
                p.requires_grad = False

            # LoRA on lm_head: W_eff = W + B_lh @ A_lh, B: [V,r], A: [r,D]
            V, D = mS.lm_head.weight.shape
            A_lh = torch.nn.Parameter(torch.randn(r, D, device=device) * 0.02)
            B_lh = torch.nn.Parameter(torch.zeros(r, V, device=device).T)  # [V, r]

            # LoRA on target layer's attention o_proj (full-attention layers
            # only; GDN linear-attention layers have out_proj, no o_proj).
            # Qwen3.5's o_proj is NON-square [out, in] = [1024, 2048], so the
            # LoRA delta is applied to o_proj's INPUT x [*, in] by patching
            # the Linear's forward: delta = B @ (A @ x). Dimensionally exact
            # for any in/out and immune to positional-vs-kwargs callers.
            layer_mod = mS.model.layers[l_S]
            o_proj_mod = None
            if hasattr(layer_mod, "self_attn"):
                o_proj_mod = getattr(layer_mod.self_attn, "o_proj", None)

            A_attn: torch.nn.Parameter | None = None
            B_attn: torch.nn.Parameter | None = None
            hook_remove: Callable[..., None] | None = None
            if o_proj_mod is not None:
                out_dim, in_dim = o_proj_mod.weight.shape
                A_attn = torch.nn.Parameter(
                    torch.randn(r, in_dim, device=device) * 0.02
                )
                B_attn = torch.nn.Parameter(
                    torch.randn(out_dim, r, device=device) * 0.02
                )  # v23: not zeros (dL/dA_attn>0, probe AND check)

                # This is how A_attn/B_attn connect to the grad graph: with
                # create_graph=True the J's h is built through the patched
                # o_proj, so dL/dJ_norm contracted with dJ_norm/dh reaches
                # o_proj.
                _orig_fwd = o_proj_mod.forward

                def _patched_o_proj(
                    x: torch.Tensor,
                    _b: torch.Tensor = B_attn,
                    _a: torch.Tensor = A_attn,
                    _f: Callable = _orig_fwd,
                ) -> torch.Tensor:
                    delta = (_b @ _a @ x.transpose(-1, -2)).transpose(-1, -2)
                    return _f(x) + delta

                o_proj_mod.forward = _patched_o_proj

                def _restore(
                    _m: torch.nn.Module = o_proj_mod, _f: Callable = _orig_fwd
                ) -> None:
                    _m.forward = _f

                hook_remove = _restore

            trainable = [A_lh, B_lh]
            if A_attn is not None:
                trainable += [A_attn, B_attn]  # type: ignore[list-item]
            for p in trainable:
                p.requires_grad = True

            optimizer = torch.optim.Adam(trainable, lr=lr)

            # -- v23 probe: matrix-free refresh path (same as in-loop) -----
            # k is FIXED (v22's budget-derived k removed — rows path memory
            # is per-chunk, the budget formula no longer applies).
            # v23.2: k 50->100 — refresh is 4 (n_samples) 2nd-order HVPs;
            # halving the rate halves the dominant wall-clock cost, the KL
            # leash + cheap dense lm_head bridge keep flowing between.
            # Probe: matrix-free rows, create_graph=True through the patched
            # o_proj, DIRECT cosine against J_tgt_p (no W matmul: W is live
            # in-loop, frozen throwaway here). p = re-normalized K-subset
            # gauge from the target cache (A5: bf16 p cast to fp32).
            # FATAL on OOM / dead graph (v23: no degrade — a degraded run is
            # a lie). AND-check: v23 B_attn init makes dL/dA_attn > 0 too.
            k = 100
            J_norms: dict[int, torch.Tensor] = {}
            first_s = samples[0]
            ids_0 = first_s["input_ids"].to(device)
            attn_0 = first_s["attention_mask"].to(device)
            if pair_mode == "kl_only":
                # GDN student layer: no o_proj, no LoRA there. KL-only leash
                # trains the lm_head LoRA; the J bridge / 2nd-order refresh
                # is skipped (it would reach nothing). Not a degrade — this
                # pair was never bridge-eligible (see classify_pair).
                print(f"    kl_only: no o_proj at l_S={l_S} (GDN) — J bridge skipped")
            elif A_attn is not None:
                p_0 = None
                for t in sample_targets:
                    if t is not None and "p_L_full" in t:
                        pf = t["p_L_full"].to(device).float()
                        p_0 = pf[token_set] / pf[token_set].sum()
                        break
                if p_0 is None:
                    print(f"    [FATAL] probe: no target sample on {pair_key}")
                    os._exit(3)
                W_probe = mS.lm_head.weight[token_set].detach()
                try:
                    probe_rows = compute_hidden_jacobian_rows(
                        mS,
                        l_S,
                        ids_0,
                        attn_0,
                        token_set,
                        p_0,
                        W_probe,
                        create_graph=True,
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    print(f"    [FATAL] probe OOM on {pair_key}: {e}")
                    os._exit(3)
                g_tgt_0 = (
                    next(t["J_tgt_p"] for t in sample_targets if t is not None)
                    .to(device)
                    .float()
                )
                probe_loss = torch.nn.functional.cosine_similarity(
                    probe_rows.float(), g_tgt_0, dim=1
                ).mean()
                probe_loss.backward()
                gA = (
                    A_attn.grad.abs().sum().item()
                    if A_attn.grad is not None
                    else 0.0
                )
                gB = (
                    B_attn.grad.abs().sum().item()
                    if B_attn is not None and B_attn.grad is not None
                    else 0.0
                )
                ok = gA > 0 and gB > 0
                print(
                    f"    [smoke_j] dL/dA_attn={gA:.2e}, dL/dB_attn={gB:.2e} "
                    f"-> {'OK' if ok else 'graph dead'}"
                )
                if not ok:
                    print(
                        "    [FATAL] create_graph graph does not reach o_proj"
                    )
                    os._exit(3)
                del probe_rows
                # B2 finite-diff spot check (plan B2): scalar s_0 =
                # h_norm @ W_rows[0]; FD derivative at h_0 vs analytic
                # matrix-free row element J_S_p[0, 0]. Rel err < 1e-2.
                # (The plan's formula is a scalar — element (0,0) is the
                # dimensionally consistent reading of "J_S_p[0] vs that FD".)
                if fd_check > 0:
                    with torch.no_grad():
                        dty = mS.lm_head.weight.dtype
                        h_0 = _extract_hidden_state(
                            mS, ids_0, attn_0, l_S
                        ).to(dty)
                        W0 = mS.lm_head.weight[token_set[0]].to(dty)
                        base = h_norm_at_perturbed_h(
                            mS, l_S, ids_0, attn_0, h_0
                        )
                        eps = 1e-4
                        h_p = h_0.clone()
                        h_p[0, 0, 0] += eps
                        pert = h_norm_at_perturbed_h(
                            mS, l_S, ids_0, attn_0, h_p
                        )
                        fd_scalar = ((pert - base) @ W0).mean().item() / eps
                    # analytic = same raw (pre-gauge) element: J_S[0,0]
                    # = sum_k W0[k] * J_norm[k,0] (dense J column, no gauge).
                    # (The matrix-free rows path's output is gauge-adjusted
                    # — softmax_jacobian — so it can't be compared 1:1 with
                    # this scalar FD; both use the same J_norm underneath.)
                    J_norm_d = compute_hidden_jacobian(mS, l_S, ids_0, attn_0)
                    analytic = (W0 @ J_norm_d[:, 0]).item()
                    rel = abs(fd_scalar - analytic) / max(1e-12, abs(analytic))
                    ok_fd = rel < 1e-2
                    print(
                        f"    [fd_check] fd={fd_scalar:.4e} analytic="
                        f"{analytic:.4e} rel={rel:.2e} -> "
                        f"{'OK' if ok_fd else 'MISMATCH'}"
                    )
                    if not ok_fd:
                        print("    [FATAL] finite-diff spot check failed")
                        os._exit(3)
            print(
                f"    J refresh: 2nd-order (o_proj HVP) every k={k} steps; "
                f"1st-order (lm_head, dense no-HVP, h-reuse) every step (v23.2b)"
            )

            # -- C2 early-stop (v23) -------------------------------------
            # No-grad bridge eval every 50 steps (fresh dense J per sample,
            # current effective weights — cheap: no create_graph graph). Stop
            # after 3 consecutive eval points with < 0.5% relative
            # improvement. Baseline at step 0 (fresh LoRA).
            N_EVAL, P_FLAT, THRESH = 50, 3, 0.005
            eval_history: list[tuple[int, float]] = []
            flat_streak = 0
            stopped_at = steps

            def _eval_bridge() -> float:
                vals = []
                with torch.no_grad():
                    for si in range(n_use):
                        tgt = sample_targets[si]
                        if tgt is None:
                            continue
                        s = samples[si]
                        ids = s["input_ids"].to(device)
                        attn_s = s["attention_mask"].to(device)
                        W_e = mS.lm_head.weight + B_lh @ A_lh
                        J_n = compute_hidden_jacobian(mS, l_S, ids, attn_s)
                        J_S_e = W_e[token_set] @ J_n
                        J_S_e_p = softmax_jacobian(
                            J_S_e, tgt["p"].to(device).float()
                        )
                        cos = torch.nn.functional.cosine_similarity(
                            J_S_e_p.float(),
                            tgt["J_tgt_p"].to(device).float(),
                            dim=1,
                        ).mean().item()
                        vals.append(1.0 - cos)
                return sum(vals) / len(vals) if vals else 0.0

            def _eval_point(step_no: int) -> None:
                nonlocal flat_streak, stopped_at
                cur = _eval_bridge()
                eval_history.append((step_no, cur))
                prev = eval_history[-2][1] if len(eval_history) >= 2 else None
                if prev is not None and prev > 0:
                    if (prev - cur) / prev < THRESH:
                        flat_streak += 1
                    else:
                        flat_streak = 0
                print(f"    [eval] step={step_no} bridge_loss={cur:.4f}")

            _eval_point(0)

            # C3: checkpoints at 60/120/180/240/300 (cumulative LoRA +
            # optimizer state + metrics). C4: the last one is also the
            # delivered lora_{pair}.pt.
            ckpt_dir = os.path.join(results_dir, "exp17")
            os.makedirs(ckpt_dir, exist_ok=True)
            CKPT_STEPS = (60, 120, 180, 240, 300)
            last_peak_vram = 0.0

            def _save_ckpt(step_no: int, final: bool) -> None:
                payload = {
                    "pair": pair_key,
                    "l_S": l_S,
                    "order": order_name,
                    "r": r,
                    "step": step_no,
                    "A_lh": A_lh.detach().cpu(),
                    "B_lh": B_lh.detach().cpu(),
                    "A_attn": (A_attn.detach().cpu() if A_attn is not None else None),
                    "B_attn": (B_attn.detach().cpu() if B_attn is not None else None),
                    "eval_cos": 1.0 - eval_history[-1][1] if eval_history else 0.0,
                    "optimizer": optimizer.state_dict(),
                    "peak_vram_gb": last_peak_vram,
                }
                name = (
                    f"lora_{pair_key}.pt" if final else f"ckpt_{pair_key}_{step_no}.pt"
                )
                torch.save(payload, os.path.join(ckpt_dir, name))
                print(f"    [ckpt] saved {name}")

            for step in range(steps):
                with _step_cm(device):
                    step_start = time.time()
                    optimizer.zero_grad()
                    total_bridge = 0.0
                    total_kl = 0.0
                    n_bridge = 0

                    for si in range(n_use):
                        s = samples[si]
                        tgt: dict | None = sample_targets[si]

                        # Effective lm_head weight: W + B_lh @ A_lh. Rebuilt PER
                        # SAMPLE — each sample runs its own backward() (docs #16),
                        # so a W_eff shared across si would be freed by si=0's
                        # backward and si=1's backward would hit "second backward"
                        # (RuntimeError). Fresh graph per backward = no double-free.
                        W_eff = mS.lm_head.weight + B_lh @ A_lh  # [V, D]

                        # KL leash — full forward (includes o_proj LoRA via hook).
                        # v23.1: output_hidden_states=True so the l_S hidden state
                        # is captured here and REUSED by the one-time fill and the
                        # refresh (matrix-free rows + dense re-fill) instead of
                        # each redoing a full forward of the same deterministic
                        # hidden state (3 full forwards -> 1 per refresh sample).
                        ids_s = s["input_ids"].to(device)
                        attn_s = s["attention_mask"].to(device)
                        # use_cache=False matches the J-path forwards so the
                        # reused h_s is bit-identical to what refresh / re-fill do.
                        out = mS.model(
                            input_ids=ids_s,
                            attention_mask=attn_s,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        hidden = out.last_hidden_state[:, -1, :]  # [1, D]
                        h_s = (
                            out.hidden_states[l_S + 1][:, -1:, :]
                            if A_attn is not None
                            and out.hidden_states is not None
                            else None
                        )  # [1, 1, D] l_S hidden state (graph-connected)
                        lm_b = mS.lm_head.bias if mS.lm_head.bias is not None else 0
                        logits_ed = hidden @ W_eff.T + lm_b  # [1, V]
                        p_ed = torch.softmax(logits_ed.float(), dim=-1)
                        with torch.no_grad():
                            p_orig = torch.softmax(orig_logits_cache[si].float(), dim=-1)
                        kl_loss = (
                            p_ed * torch.log(p_ed.clamp_min(1e-12) / (p_orig + 1e-12))
                        ).mean()

                        loss = kl_weight * kl_loss
                        total_kl += kl_loss.item()

                        already_bwd = False  # True when the refresh branch
                        # already did the (KL + bridge) backward inside
                        # bridge_rows_backward -> skip the per-sample loss.backward().
                        # Bridge — only for input_ids-matched targets (D1) AND
                        # only when the student layer is full-attn (has o_proj,
                        # A_attn set). GDN/kl_only pairs never build the held
                        # dense J or the 2nd-order refresh (no o_proj to train).
                        if tgt is not None and A_attn is not None:
                            J_tgt_p = tgt["J_tgt_p"].to(device)
                            p_L = tgt["p"].to(device).float()  # v23 A5: cache p is bf16
                            refresh_step = False
                            J_S_p: torch.Tensor | None = None
                            if si not in J_norms:
                                # one-time fill: dense J (create_graph=False = a
                                # 4MB [D,D] fp32 matrix, no 2nd-order graph — cheap;
                                # v22's OOM was the create_graph refresh, not this).
                                # v23.1: reuses the KL forward's h (no full forward).
                                J_norms[si] = compute_hidden_jacobian(
                                    mS, l_S, ids_s, attn_s, h=h_s
                                )
                                refresh_step = False
                            elif step % k == 0:
                                # refresh: A4 chunked bridge backward. bridge_
                                # rows_backward forwards once (KL is the extra_loss,
                                # backproped with chunk 0) then backprops each chunk
                                # of K rows (C=8) of (1-cos).mean() immediately ->
                                # peak = C+1 2nd-order row-graphs, not K (the v22
                                # OOM was the full K=256 retained create_graph graph).
                                # Gradients reach lm_head (live W_eff rows) AND
                                # o_proj (HVP through h) this step. The returned
                                # J_S_p is detached (graph consumed) — re-fill the
                                # held dense J_norm for stale steps (v22 freshness).
                                # held is already detached from prior stale steps
                                # (no live graph) — drop the old one, new fill below.
                                del J_norms[si]
                                p_refresh = tgt["p_L_full"].to(device).float()
                                p_refresh = (
                                    p_refresh[token_set] / p_refresh[token_set].sum()
                                )
                                try:
                                    rows, br = bridge_rows_backward(
                                        mS,
                                        l_S,
                                        ids_s,
                                        attn_s,
                                        token_set,
                                        p_refresh,
                                        W_eff[token_set],
                                        J_tgt_p,
                                        extra_loss=kl_weight * kl_loss,
                                        verbose=True,
                                        h=h_s,
                                    )
                                except (
                                    torch.cuda.OutOfMemoryError,
                                    RuntimeError,
                                ) as e:
                                    print(
                                        f"    [FATAL] v23 refresh OOM {pair_key} "
                                        f"step={step} si={si}: {e}"
                                    )
                                    os._exit(3)
                                J_S_p = rows
                                bridge_loss = torch.tensor(br)
                                total_bridge += br
                                n_bridge += 1
                                # new dense fill (no 2nd-order graph) replaces held;
                                # v23.1: reuses the KL forward's h (no full re-forward)
                                J_norms[si] = compute_hidden_jacobian(
                                    mS, l_S, ids_s, attn_s, h=h_s
                                )
                                refresh_step = True
                                already_bwd = True  # skip the loss.backward()
                                # below; the per-sample optimizer.step() still runs.
                                if str(device).startswith(("cuda", "xla")):
                                    # A2: VRAM telemetry after the (chunked)
                                    # refresh — the peak-relevant moment.
                                    peak = torch.cuda.max_memory_allocated() / 2**30
                                    cur = torch.cuda.memory_allocated() / 2**30
                                    last_peak_vram = max(last_peak_vram, peak)
                                    print(
                                        f"    [vram] pair={pair_key} step={step} "
                                        f"si={si} k={k} peak={peak:.1f}G "
                                        f"cur={cur:.1f}G"
                                    )
                            else:
                                # v23.2b DUAL CADENCE: recompute the 1st-order
                                # (no-HVP) J_norm EVERY step, reusing the KL
                                # forward's h (cheap decoder-tail + matmul, no
                                # 2nd-order graph). So the lm_head bridge runs on a
                                # FRESH hidden every step (no stale-J staleness),
                                # while the expensive 2nd-order (HVP, o_proj) stays
                                # at k below.
                                refresh_step = False
                                J_norms[si] = compute_hidden_jacobian(
                                    mS, l_S, ids_s, attn_s, h=h_s
                                )
                            if refresh_step and J_S_p is not None:
                                # J_S_p already detached by bridge_rows_backward
                                # (graph consumed by the per-chunk backward). Its
                                # (1-cos) + KL backward ran there; only recompute
                                # the metric scalar here for logging.
                                cos_sim = torch.nn.functional.cosine_similarity(
                                    J_S_p.float(), J_tgt_p.float(), dim=1
                                )
                                bridge_loss = (1.0 - cos_sim).mean()
                            elif J_norms.get(si) is not None:
                                # stale: held dense J_norm @ live W_eff rows ->
                                # lm_head bridge gradient only (o_proj is refresh-only
                                # by design; the dense path's h has no create_graph
                                # graph). Same as v22 stale behavior.
                                J_S = W_eff[token_set] @ J_norms[si]
                                J_S_p = softmax_jacobian(J_S, p_L)
                                cos_sim = torch.nn.functional.cosine_similarity(
                                    J_S_p.float(), J_tgt_p.float(), dim=1
                                )
                                bridge_loss = (1.0 - cos_sim).mean()
                                loss = loss + bridge_loss
                                total_bridge += bridge_loss.item()
                                n_bridge += 1

                        if not already_bwd:
                            loss.backward()

                        # This backward consumed the grad graph. Any live
                        # create_graph J for this si is now freed; a stale-step
                        # backward through it would be a "second backward"
                        # (RuntimeError). Detach it in place so its value still
                        # drives the lm_head bridge until the next refresh
                        # rebuilds a fresh graph (the o_proj HVP is refresh-only).
                        J_cur = J_norms.get(si)
                        if J_cur is not None and J_cur.grad_fn is not None:
                            J_norms[si] = J_cur.detach()
                    optimizer.step()

                step_time = time.time() - step_start
                eta = (steps - step - 1) * step_time
                if (step + 1) % 5 == 0 or step == 0:
                    print(
                        f"    step {step + 1}/{steps}: "
                        f"bridge={total_bridge / max(n_bridge, 1):.4f}, "
                        f"kl={total_kl:.4f}, step={step_time:.0f}s, eta={eta:.0f}s"
                    )

                if (step + 1) in CKPT_STEPS:
                    _save_ckpt(step + 1, final=False)

                if (step + 1) % N_EVAL == 0:
                    _eval_point(step + 1)
                    if flat_streak >= P_FLAT:
                        stopped_at = step + 1
                        print(
                            f"    [early-stop] {pair_key} at step {step + 1} "
                            "(3 consecutive evals < 0.5% improvement)"
                        )
                        break

            # -- Freeze lora deltas into weights ---------------------------
            delta_lm_final = B_lh @ A_lh  # [V, D]
            with torch.no_grad():
                mS.lm_head.weight.data += delta_lm_final
            if o_proj_mod is not None and A_attn is not None and B_attn is not None:
                # B[out, r] @ A[r, in] = [out, in] == o_proj.weight shape.
                o_proj_mod.weight.data += B_attn @ A_attn

            # Cleanup BEFORE eval: the delta is now frozen in the weights;
            # leaving the live patch on would double it in the eval forward.
            # (A_lh/B_lh/A_attn/B_attn kept alive — _save_ckpt below reads
            # them; the next pair iteration's rebinding releases them.)
            J_norms.clear()
            if hook_remove is not None:
                hook_remove()

            # C4: delivered per-pair artifact (A/B matrices + optimizer +
            # metrics), saved AFTER the freeze so the file holds the final
            # frozen deltas and the eval below runs without the patch.
            _save_ckpt(stopped_at, final=True)

            # -- Evaluate bridge (no grad, create_graph=False) -------------
            bridge_vals = []
            with torch.no_grad():
                W_final = mS.lm_head.weight  # frozen
                for si in range(n_use):
                    s = samples[si]
                    tgt = sample_targets[si]
                    if tgt is None:
                        continue
                    ids = s["input_ids"].to(device)
                    attn_s = s["attention_mask"].to(device)
                    J_tgt_p = tgt["J_tgt_p"].to(device)
                    p_L = tgt["p"].to(device)
                    J_norm_eval = compute_hidden_jacobian(mS, l_S, ids, attn_s)
                    J_S_ed = W_final[token_set] @ J_norm_eval
                    J_S_ed_p = softmax_jacobian(J_S_ed, p_L)
                    cos = (
                        torch.nn.functional.cosine_similarity(
                            J_S_ed_p.float(), J_tgt_p.float(), dim=1
                        )
                        .mean()
                        .item()
                    )
                    bridge_vals.append(cos)

            # KL to original
            kl_sum = 0.0
            with torch.no_grad():
                for si in range(min(n_samples, len(orig_logits_cache))):
                    s = samples[si]
                    ids = s["input_ids"].to(device)
                    attn_s = s["attention_mask"].to(device)
                    logits_ed = mS(input_ids=ids, attention_mask=attn_s).logits[
                        :, -1, :
                    ]
                    p_ed = torch.softmax(logits_ed.float(), dim=-1)
                    p_orig = torch.softmax(orig_logits_cache[si].float(), dim=-1)
                    kl = (
                        (p_ed * torch.log(p_ed.clamp_min(1e-12) / (p_orig + 1e-12)))
                        .mean()
                        .item()
                    )
                    kl_sum += kl
            kl_avg = kl_sum / min(n_samples, len(orig_logits_cache))

            layer_result = {
                "pair": pair_key,
                "bridge_mean": sum(bridge_vals) / len(bridge_vals)
                if bridge_vals
                else 0,
                "KL_ed_orig": kl_avg,
                "k_refresh": k,
                "stopped_at": stopped_at,
                "eval_history": eval_history,
            }
            layer_time = time.time() - layer_start
            print(f"    Layer {pair_key} done in {layer_time:.0f}s")
            print(f"    bridge={layer_result['bridge_mean']:.4f}, KL={kl_avg:.4f}")
            per_layer_results.append(layer_result)

            # Update orig logits for next layer (cumulative edits)
            with torch.no_grad():
                for si, s in enumerate(samples[:n_samples]):
                    ids = s["input_ids"].to(device)
                    attn_s = s["attention_mask"].to(device)
                    orig_logits_cache[si] = mS(
                        input_ids=ids, attention_mask=attn_s
                    ).logits[:, -1, :]

        per_order_results[order_name] = {
            "pairs": pair_seq,
            "per_layer": per_layer_results,
        }

    # -- Save --------------------------------------------------------------
    result = {
        "description": (
            "v23: per-layer LoRA r=4 on lm_head + attention o_proj of l_S, bridge loss on "
            "matrix-free per-sample J_S_p rows (no dense [D,D] graph: refresh steps build "
            "rows with W_rows = (W+B_lh@A_lh)[token_set] live, create_graph=True, chunked "
            "backward; one backward per row reaches lm_head AND o_proj — HVP, no D^3). "
            "Fill/stale steps use the cheap dense J_norm so the lm_head bridge gradient "
            "keeps flowing; o_proj is refresh-only. k=100 fixed. J refresh OOM -> FATAL "
            "exit 3 (no degrade). Max 300 steps/pair (CPU smoke 40). Targets matched by "
            "exact input_ids; mismatch -> bridge skipped (KL unaffected). After each pair "
            "both LoRA deltas are frozen into weights. Two orders (GPU), late-to-early "
            "only (CPU)."
        ),
        "per_order": per_order_results,
        "r": r,
        "steps": steps,
        "kl_weight": kl_weight,
        "device": device,
        "elapsed_seconds": time.time() - start,
    }
    path = os.path.join(results_dir, "exp17_lora_distill.json")
    result["params"] = {
        "small_model": small_model,
        "device": device,
        "max_length": max_length,
        "min_length": min_length,
        "targets_dir": targets_dir,
    }
    try:
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
    except OSError as e:
        print(f"Failed to write {path}: {e}")
        raise

    print(f"\nSaved to {path}")
    print(f"Total time: {time.time() - start:.0f}s")


# ---------- run exp17_lora_distill ----------
gc.collect()
sig = inspect.signature(main_17loradistill)
params = list(sig.parameters)
kwargs = {}
if 'small_model' in params: kwargs['small_model'] = 'Qwen/Qwen3.5-0.8B'
if 'large_model' in params: kwargs['large_model'] = 'Qwen/Qwen3.5-4B'
if 'model' in params: kwargs['model'] = 'Qwen/Qwen3.5-0.8B'
if 'device' in params: kwargs['device'] = 'xla'
if 'results_dir' in params: kwargs['results_dir'] = RES
if 'targets_dir' in params: kwargs['targets_dir'] = os.path.join(RES, 'targets')
if 'small_model' in kwargs and 'model' in kwargs: del kwargs['model']
kwargs['targets_dir']='/kaggle/input/datasets/voroninmichael/prism-exp16-targets'
kwargs['pairs']='15_11'
kwargs['steps']=100
t0 = time.time()
try:
    main_17loradistill(**kwargs)
    print(f'[OK] exp17_lora_distill {time.time()-t0:.0f}s')
except Exception as e:
    print(f'[FAIL] exp17_lora_distill: {e}')
    traceback.print_exc()
print("\nResults:")
if os.path.isdir(RES):
    for fn in sorted(os.listdir(RES)):
        print(f'  {fn} {os.path.getsize(os.path.join(RES, fn)):,} bytes')
