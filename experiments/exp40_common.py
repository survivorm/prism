"""Общее для E-40.x (docs/e40_todo.md §1.5): шапка артефактов, хеши, дамп, sentinel.

Ноль физики — только обвязка. Каждый exp40_x_*.py строит header через
`header(...)`, пишет через `dump_json(...)` (prism.json_utils: inf → null + флаг)
и вешает channel_tags из `prism.tag.make_tag` с дефолтами фазы 1.
"""
from __future__ import annotations

import hashlib
import os
import time

import torch

from prism import json_utils
from prism.tag import make_tag

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

CORE_VERSION = "v1.0.0"
#: kaggle-копия exp27b с полем P128_RAW (репозиторная копия беднее — E-37 знает про это).
RAW_CACHE_DIR = "/data/prism_kernels/exp27b/results"
SENTINEL_SUM = 0.5
LAYERS = [6, 12, 18, 22, 23]
DOMAINS = ["wiki", "math", "code", "chat"]
N_PER_DOMAIN = 8
S = 128


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def header(exp: str, tz: str, inputs: dict[str, str],
           channel_tags: list, extra: dict | None = None) -> dict:
    rep = {
        "experiment": exp,
        "tz": tz,
        "core_version": CORE_VERSION,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "observational": True,
        "new_forwards": 0,
        "inputs": {k: {"path": v, "sha256_16": sha16(v)} for k, v in inputs.items()},
        "channel_tags": [t._asdict() for t in channel_tags],
    }
    if extra:
        rep.update(extra)
    return rep


def dump_json(rep: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json_utils.dump(rep, f, indent=2, ensure_ascii=False)


def tag(**kw):
    """make_tag с дефолтами фазы 1 (natural, truncated_U128, cache, measured)."""
    kw.setdefault("protocol", "natural")
    kw.setdefault("p_mode", "truncated_U128")
    kw.setdefault("source", "cache")
    kw.setdefault("status", "measured")
    return make_tag(**kw)


def sentinel_mask(layer: int, n_docs: int = 32, s: int = S) -> torch.Tensor:
    """[n_docs, s] bool: P128_RAW.sum(-1) < 0.5 (порог E-34 зафиксирован ТЗ §1.2).

    rows берём clone'ами и stack'ом — прямое присваирование строк из torch.load
    dict в preallocated tensor тихо портило содержимое (урок раунда 5 E-39-suppl).
    """
    ck = torch.load(os.path.join(RAW_CACHE_DIR, f"exp27b_jcache_L{layer}.pt"),
                    map_location="cpu", weights_only=False)
    rows = [ck["P128_RAW"][(dom, si, t)].float().clone()
            for dom in DOMAINS for si in range(N_PER_DOMAIN) for t in range(s)]
    Pr = torch.stack(rows).reshape(n_docs, s, -1)
    return Pr.sum(-1) < SENTINEL_SUM


def doc_ids() -> list[str]:
    return [f"{dom}#{si}" for dom in DOMAINS for si in range(N_PER_DOMAIN)]
