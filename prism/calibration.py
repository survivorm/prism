"""Calibration: data loading, tokenization, global token set, layer mapping."""

import warnings
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
    cfg = model.config
    return list(
        cfg.layer_types if hasattr(cfg, "layer_types") else cfg.text_config.layer_types
    )


_DOMAIN_DATASETS = {
    "wiki": ("allenai/c4", "en", "train", "text"),
    "math": ("nvidia/OpenMathInstruct-2", None, "train", "problem"),
    "code": ("neyralabs/apex-agents", None, "train", "prompt"),
    "chat": ("HuggingFaceH4/ultrachat_200k", None, "train_sft", "messages"),
}


def _load_dataset_for_domain(
    domain: str,
) -> tuple[str, str | None, str, str]:
    """Return (dataset_id, config, split, text_field) for a domain.

    Falls back to wiki (C4) if the target dataset is unavailable, with a
    RuntimeWarning: a fallback run silently poisons cross-domain comparisons
    (E-13/E-27 T_domain), so the substitution must be visible in the log.
    """
    ds_id, cfg, split, field = _DOMAIN_DATASETS[domain]
    try:
        kw: dict[str, object] = {"split": split, "streaming": True}
        if cfg is not None:
            kw["name"] = cfg
        ds = load_dataset(ds_id, **kw)  # type: ignore[call-overload]
        next(iter(ds))  # force the first fetch: streaming is lazy, a broken/
        # gated dataset would otherwise pass the check and fail inside
        # load_calibration_data (loud crash instead of the intended fallback).
        return ds_id, cfg, split, field
    except Exception:
        warnings.warn(
            f"domain {domain} ({ds_id}) unavailable — falling back to wiki/C4. "
            f"Domain labels in this run are INVALID for cross-domain "
            f"comparisons (E-13/E-27).",
            RuntimeWarning,
            stacklevel=2,
        )
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
    max_length: int = 64,
    min_length: int = 64,
    n_samples: int = 4,
    domain: str = "wiki",
) -> list[dict[str, torch.Tensor]]:
    """Load and tokenize calibration data.

    Defaults match the tested protocol (paper §12/C.6: S=64, filter >= 64,
    N=4); experiments pass explicit lengths. domain: 'wiki' (C4), 'math'
    (OpenMathInstruct-2), 'code' (apex-agents), 'chat' (ultrachat). Falls back
    to wiki if the target dataset is unavailable.
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
    large_name: str, small_name: str, dtype: torch.dtype = torch.float32
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load large and small models with eager attention.

    dtype defaults to float32: the tested protocols run on CPU, where fp16
    is ~10x slower (paper §12/C.9). Pass torch.float16 explicitly for GPU.
    Thread count (torch.set_num_threads(4) in §12/C.9) is set by the caller's
    setup, not here.
    """
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
    """Extract embedding matrix from model.

    The bridge projects through the AN-embedding matrix W (§3/§5). For the
    tied-weight checkpoints in scope, W = E (embed_tokens). If a checkpoint
    is loaded WITHOUT tied weights, W = lm_head.weight ≠ E and the projection
    would silently use the wrong matrix. We therefore warn ONCE per model if
    lm_head.weight is not the same storage as embed_tokens.weight — a loud,
    non-fatal signal that the W = E assumption does not hold for this model.
    """
    emb = model.model.embed_tokens.weight.detach()  # type: ignore[attr-defined]
    _warn_if_not_tied(model, emb)
    return emb


def _warn_if_not_tied(model: AutoModelForCausalLM, emb: torch.Tensor) -> None:
    lm_head = getattr(getattr(model, "lm_head", None), "weight", None)
    if lm_head is None:
        return
    if not lm_head.data_ptr() == emb.data_ptr():
        # data_ptr identity covers tied weights; fall back to value equality
        # for the (rare) case of a tied-but-copied view.
        if not torch.equal(lm_head.detach(), emb):
            warnings.warn(
                f"extract_embeddings: lm_head.weight is NOT tied to "
                f"embed_tokens.weight (model={type(model).__name__}). The "
                f"bridge projects through E assuming W = E (§3); with untied "
                f"weights the an-embedding matrix is lm_head.weight and the "
                f"gap/bridge numbers are silently wrong. Extract lm_head "
                f"explicitly if this is intended.",
                UserWarning,
                stacklevel=2,
            )
