"""Calibration: data loading, tokenization, global token set, layer mapping."""

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
    "math": ("nvidia/OpenMathInstruct-2", None, "train", "problem"),
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

    domain: 'wiki' (C4), 'math' (OpenMathInstruct-2), 'code' (apex-agents), 'chat' (ultrachat).
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
