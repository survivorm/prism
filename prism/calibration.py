"""Calibration: data loading, tokenization, global token set, layer mapping."""

import hashlib
import json
import os
import re
import warnings
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- закрепление пула (PLAN «Запрос 11»: все результаты воспроизводимы) -------------------
# Порядок streaming-потока на Hub ничем не закреплён, поэтому «первые подходящие строки»
# меняются от прогона к прогону: E-0.1 при идентичных params дал другой top-token набор
# (pk_hash 5c12c174… -> 30d82ade…) и сдвинул C_b(sigma_d) до x31. Пин = сырые тексты принятых
# строк + отпечаток; при наличии пина поток не читается вообще, значит прогон воспроизводим
# и офлайн. Тексты (не тензоры): input_ids по-прежнему производит токенизатор.
PIN_DIR_ENV = "PRISM_CALIB_PIN_DIR"
PIN_WRITE_ENV = "PRISM_CALIB_PIN_WRITE"      # =1: разрешить записать новый пин
PIN_REQUIRE_ENV = "PRISM_CALIB_REQUIRE_PIN"  # =1: отсутствие пина = громкий отказ
LAST_POOL_FPS: dict[str, str] = {}           # domain -> отпечаток последнего загруженного пула


def pool_fingerprint(texts: list[str]) -> str:
    """Отпечаток пула по сырым текстам (от токенизатора не зависит)."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _default_pin_dir() -> Path | None:
    """Каталог пинов по умолчанию, или None если искать негде.

    В flattening-Kaggle-ядре модулей нет на диске и `__file__` не определён (упавший exp3:
    `NameError: name '__file__' is not defined` прямо в `_pin_path`) — значит путь репозитория
    строится только когда модуль реально лежит в репозитории; иначе пин ищется по `PIN_DIR_ENV`,
    а без него — поток (или громкий отказ при `PIN_REQUIRE_ENV`).
    """
    try:
        here = Path(__file__).resolve()
    except NameError:
        return None
    return here.parent.parent / "data" / "calib_pins"


def _pin_path(domain, model_name, n_samples, min_length, max_length,
              save_offset_mapping) -> Path | None:
    root = os.environ.get(PIN_DIR_ENV)
    base = Path(root) if root else _default_pin_dir()
    if base is None:
        return None
    slug = re.sub(r"[^0-9A-Za-z.]+", "_", model_name)
    mode = "offset" if save_offset_mapping else "plain"
    return base / f"{domain}__{slug}__n{n_samples}__min{min_length}__max{max_length}__{mode}.json"


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
    save_offset_mapping: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Load and tokenize calibration data.

    Defaults match the tested protocol (paper §12/C.6: S=64, filter >= 64,
    N=4); experiments pass explicit lengths. domain: 'wiki' (C4), 'math'
    (OpenMathInstruct-2), 'code' (apex-agents), 'chat' (ultrachat). Falls back
    to wiki if the target dataset is unavailable. save_offset_mapping: adds
    `offset_mapping` to each sample (E-31 sentence boundaries).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pin = _pin_path(domain, model_name, n_samples, min_length, max_length, save_offset_mapping)
    texts: list[str] | None = None
    ds_id = cfg = split = field = None
    if pin is not None and pin.exists():
        meta = json.loads(pin.read_text(encoding="utf-8"))
        texts = list(meta["texts"])
        print(f"[calib] {domain}: пин {pin.name}, строк={len(texts)}, "
              f"pool_fp={meta['fp']}, поток не читается", flush=True)
    elif os.environ.get(PIN_REQUIRE_ENV) == "1":
        raise RuntimeError(
            f"нет пина пула {pin or f'каталог пинов недоступен — задайте {PIN_DIR_ENV}'} при "
            f"{PIN_REQUIRE_ENV}=1: прогон на потоке с Hub недопустим, "
            f"иначе числа не воспроизводимы. Сгенерируйте пин с {PIN_WRITE_ENV}=1 один раз и "
            f"закоммитьте его."
        )
    else:
        ds_id, cfg, split, field = _load_dataset_for_domain(domain)
        kw: dict[str, object] = {"split": split, "streaming": True}
        if cfg is not None:
            kw["name"] = cfg
        dataset = load_dataset(ds_id, **kw)  # type: ignore[call-overload]
    samples = []
    # NOTE (audit4 Q9): the offset-mapping path tokenizes with
    # add_special_tokens=False, the plain path with the tokenizer default (True).
    # Deliberate, not a bug: offset_mapping must stay aligned 1:1 with the text
    # span (special tokens carry no span), and the two paths feed different
    # consumers. Consequence to keep in mind: a token set / calibration built
    # through the offset path is NOT the same sequence as the mainline E-16
    # calibration of the same text — do not mix their token sets.
    accepted: list[str] = []
    # Из пина идут уже извлечённые тексты (извлечение = _extract_text_from_sample на момент
    # записи пина); с потока идут сырые строки.
    rows = ((t, None) for t in texts) if texts is not None else ((None, raw) for raw in dataset)
    for pinned_text, raw in rows:
        text = pinned_text if pinned_text is not None else _extract_text_from_sample(raw, field)
        if not text:
            continue
        if save_offset_mapping:
            # Токенизатор вызывается на ОДНОЙ строке, поэтому input_ids — плоский список id, а
            # offset_mapping — список пар; индексация `[0]` здесь была багом (len(int) падал на
            # первом же вызове, найденном E-45b).
            toks = tokenizer(
                text, max_length=max_length, truncation=True,
                add_special_tokens=False, return_offsets_mapping=True,
            )
            if len(toks["input_ids"]) >= min_length:
                accepted.append(text)
                samples.append({
                    "input_ids": toks["input_ids"],
                    "offset_mapping": torch.tensor(
                        toks["offset_mapping"], dtype=torch.int32
                    ),
                })
        else:
            tokens = tokenizer(
                text, max_length=max_length, truncation=True,
                return_tensors="pt",
            )
            if tokens["input_ids"].shape[1] >= min_length:
                accepted.append(text)
                samples.append(tokens)
        if len(samples) >= n_samples:
            break

    if texts is None:
        texts = accepted
        fp = pool_fingerprint(texts)
        print(f"[calib] {domain}: поток с Hub, строк={len(texts)}, pool_fp={fp}", flush=True)
        if os.environ.get(PIN_WRITE_ENV) == "1":
            if pin is None:
                raise RuntimeError(
                    f"{PIN_WRITE_ENV}=1, но каталога пинов нет: {PIN_DIR_ENV} не задан, а "
                    "`__file__` недоступен (flattening-ядро). Задайте PRISM_CALIB_PIN_DIR "
                    "на путь, который попадает в вывод ядра."
                )
            pin.parent.mkdir(parents=True, exist_ok=True)
            pin.write_text(
                json.dumps(
                    {
                        "protocol": {"domain": domain, "model": model_name, "n_samples": n_samples,
                                     "min_length": min_length, "max_length": max_length,
                                     "offset_mapping": bool(save_offset_mapping)},
                        "source": {"dataset": ds_id, "config": cfg, "split": split, "field": field},
                        "fp": fp,
                        "texts": texts,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            print(f"[calib] {domain}: пин записан -> {pin}", flush=True)
    else:
        fp = pool_fingerprint(texts)
        if len(samples) < n_samples:
            raise RuntimeError(
                f"пин {pin.name} дал {len(samples)} строк вместо {n_samples}: токенизатор или "
                f"фильтр длины изменились относительно момента записи пина (pool_fp={fp}). "
                f"Молча читать другой пул нельзя — числа стали бы невоспроизводимыми."
            )
    LAST_POOL_FPS[domain] = fp
    return samples


def build_global_token_set(
    model: AutoModelForCausalLM,
    calibration_samples: list[dict[str, torch.Tensor]],
    K: int = 64,
) -> list[int]:
    """Frequency token set: the K tokens that most often appear in the per-sample
    top-K of the LAST position (`logits[:, -1]`) across the calibration pool.

    One basis for every position, so a stored p-row is SUB-stochastic (~half of the
    exp27b positions carry < 0.5 of the softmax mass) — that basis is what all
    published Jacobian numbers sit on. The positional convention (each position gets
    its own top-K, so p is a real truncation) is produced by
    `experiments/exp27b_dense.py: column_mode="positional"` (cache exp27c), not here;
    "union top-K at fixed K" is not definable, since the union of per-position top-K
    exceeds K (audit4 Q10 verdict round). Never mix numbers from the two bases:
    the basis belongs in the ChannelTag slice.
    """
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
