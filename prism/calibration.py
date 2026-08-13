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


def load_calibration_data(
    model_name: str, max_length: int = 512, min_length: int = 256, n_samples: int = 4
) -> list[dict[str, torch.Tensor]]:
    """Load and tokenize calibration data from C4 dataset."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    samples = []
    for sample in dataset:
        tokens = tokenizer(
            sample["text"], max_length=max_length, truncation=True, return_tensors="pt"
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
