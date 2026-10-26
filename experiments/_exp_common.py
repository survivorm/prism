"""Shared plumbing for the new-function experiment series
(exp_T0/eps/T2/T1/delta) — new_functions_4.md §10.

Model/sample loading, token-set construction, the five EditSpecs, h/p
caching, and gauge-free J reconstruction from the cache. Metric math
lives in the exp_*.py files (each ≤ 100 lines, §10 rule); the sole
exception is ``row_cos`` (the per-position bridge scalar) — a mean-of-
row-cosines helper kept here so the exp files stay under the 100-line
limit.

Cost note: J is reconstructed on the fly from the h/p cache
(`compute_jacobian_for_tokens(h_override=...)`) — no J on disk, any r
without recompute (interventions design decision #4).
"""
import os
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM

from prism import (
    ExperimentSpec,
    apply_edit_and_cache,
    compute_esg,
    compute_jacobian_for_tokens,
    load_calibration_data,
    make_base,
    make_random_matched,
    make_random_ortho,
    make_readout_pull,
    make_tail_only,
    softmax_jacobian,
    top_r_svd,
)

POS_TAIL = 8  # last positions used for per-doc J lists (small-S safe)


def load_model(model_name: str, device: str = "auto") -> torch.nn.Module:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    m.eval()
    return m


def load_samples(model_name: str, n_docs: int, S: int, domain: str = "wiki") -> list[dict]:
    data = load_calibration_data(model_name, max_length=S, min_length=S,
                                 n_samples=n_docs, domain=domain)
    if len(data) < n_docs:
        raise RuntimeError(f"only {len(data)}/{n_docs} samples of exactly S={S}")
    return data[:n_docs]


def token_set_from(model, samples: list[dict], K: int) -> list[int]:
    """Top-K tokens by final-logit frequency across samples (exp27 U_K rule)."""
    counts: dict[int, int] = {}
    with torch.no_grad():
        for s in samples:
            out = model(input_ids=s["input_ids"].to(model.device),
                        attention_mask=s["attention_mask"].to(model.device))
            for t in torch.softmax(out.logits[0, -1].float(), dim=-1).topk(K).indices.tolist():
                counts[t] = counts.get(t, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:K]]


def gauge_free_J(model, L: int, ids, mask, ts: list[int],
                 h: torch.Tensor | None, p: torch.Tensor) -> torch.Tensor:
    """Gauge-free J^(p) [K, D] at the last position; h (optional) = [D]."""
    J = compute_jacobian_for_tokens(model, L, ids, mask, ts, h_override=h)
    return softmax_jacobian(J.float(), p)


def top_r_basis(model, samples, ts: list[int], L: int, r: int, n_pos: int = 8) -> torch.Tensor:
    """[K, r] orthonormal token-space basis from base ESG at layer L (doc0,
    last n_pos positions, gauge-free J^(p))."""
    s0 = samples[0]
    ids, mask = s0["input_ids"].to(model.device), s0["attention_mask"].to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    S = ids.shape[1]
    js = []
    for t in range(S - n_pos, S):
        h = out.hidden_states[L + 1][0, t, :].float().cpu()
        p = torch.softmax(out.logits[0, t].float()[ts], dim=-1).cpu()
        js.append(gauge_free_J(model, L, ids, mask, ts, h, p))
    _, evec = compute_esg(js, n_components=r)
    return evec.T.contiguous()  # [K, r], orthonormal columns


def build_edits(model, samples, ts: list[int], L0: int, L1: int,
                n_steps: int, seed: int, basis: torch.Tensor) -> list:
    """The five edits of exp_T0: base, readout_pull (target = gauge-free J at
    L1 — the layer-pull protocol), random_ortho, random_matched, tail_only."""
    s0 = samples[0]
    ids, mask = s0["input_ids"].to(model.device), s0["attention_mask"].to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    S = ids.shape[1]
    h = out.hidden_states[L1 + 1][0, S - 1, :].float().cpu()
    p = torch.softmax(out.logits[0, S - 1].float()[ts], dim=-1).cpu()
    target = gauge_free_J(model, L1, ids, mask, ts, h, p)  # [K, D]
    rp = make_readout_pull(model, ids, mask, L0, ts, target,
                           n_steps=n_steps, seed=seed, top_r_basis=basis)
    ref = rp.delta_W
    return [
        make_base(ts),
        rp,
        make_random_ortho(ref, basis, ts, seed=seed),
        make_random_matched(ref, basis, ts, seed=seed + 1),
        make_tail_only(ref, basis, ts),
    ]


def cache_edits(model, samples, edits: list, layers: list[int], cache_root: str) -> dict:
    """apply_edit_and_cache for every edit; returns {edit.name: h/p tensors}."""
    out: dict[str, dict] = {}
    for e in edits:
        apply_edit_and_cache(model, e, samples, layers, cache_root)
        hp = {}
        for d, s in enumerate(samples):
            S = s["input_ids"].shape[1]
            for L in layers:
                h = torch.load(os.path.join(cache_root, e.name, f"L{L:02d}",
                                            f"doc{d:04d}_h.pt"),
                               map_location="cpu")["h"]
                p = torch.load(os.path.join(cache_root, e.name, f"L{L:02d}",
                                            f"doc{d:04d}_p.pt"),
                               map_location="cpu")["p"]
                hp[(L, d)] = (h, p)
        out[e.name] = hp
    return out


def doc_js(model, L: int, s: dict, ts: list[int], h: torch.Tensor,
           p: torch.Tensor, n_pos: int) -> list[torch.Tensor]:
    """Per-position gauge-free J for the last n_pos positions of one doc."""
    ids, mask = s["input_ids"].to(model.device), s["attention_mask"].to(model.device)
    S = s["input_ids"].shape[1]
    js = []
    for t in range(S - n_pos, S):
        js.append(gauge_free_J(model, L, ids, mask, ts, h[t], p[t]))
    return js


def row_cos(A: torch.Tensor, B: torch.Tensor) -> float:
    """Mean over rows of cos — the per-position bridge scalar."""
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-12)
    B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return float((A * B).sum(1).mean())


def make_spec(experiment: str, model_name: str, ts: list[int], layers, r: int,
              n_docs: int, S: int, **kw) -> "ExperimentSpec":
    """Frozen ExperimentSpec with a make_base reference edit (delta_W=None).

    The reference edit carries only the token_set; run() rebuilds the real
    edits. ``kw`` forwards remaining spec fields (n_steps, B_perm, k_sigma,
    epsilon_ratio, n_theta, alpha, seed, cache_root, output_root, ...).
    """
    ts = list(ts)
    return ExperimentSpec(
        experiment=experiment, edit=make_base(ts), layers=tuple(int(l) for l in layers),
        token_set=tuple(ts), n_docs=n_docs, S=S, r=r, model_name=model_name, **kw,
    )


@contextmanager
def lm_head_rows_set(model, ts: list[int], w: torch.Tensor):
    """In-place set/restore lm_head.weight[token_set] = w (T2 interpolation).

    Qwen3.5 ties embeddings, but lm_head.weight is a resolvable attribute;
    we mutate only the K token rows under no_grad and restore on exit.
    """
    weight = model.lm_head.weight
    idx = torch.tensor(list(ts), device=weight.device)
    saved = weight[idx].clone()
    with torch.no_grad():
        weight[idx] = w
    try:
        yield
    finally:
        with torch.no_grad():
            weight[idx] = saved
