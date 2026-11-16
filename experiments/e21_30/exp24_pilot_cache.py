"""Exp 24 pilot cache (docs/Exp24-16.md): 64-node Jacobian cache for the
Exp 24.0 pipeline-power study.

Nodes: (x, q) with q in {last=S, mid=S/2}, S=64 fixed. 32 samples
(8 x {wiki, math, code, chat}), 0.8B, layers {6, 22}.
J(x,q) = gauge-free softmax jacobian of the last-position logits of the
PREFIX x[:q] (causality: state before token q), token set = global top-64
of the student's final-logit top-K (common to both models, D3).

Cache layout (TZ 1.10):
    results/jcache/v1/{model_id}/{git_commit}/{token_set_id}/
        layer{L}_pos{last|mid}/J_{s:03d}.pt   [K, D]
        layer{L}_pos{last|mid}/p_{s:03d}.pt   [K]  (gauge p on U)
        layer{L}_pos{last|mid}/h_{s:03d}.pt   [D]  (layer-l hidden at q)
        layer{L}_KG.pt / KF.pt / KV.pt        (report kernels, TZ 1.3)
        manifest.json
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch

from prism.calibration import load_calibration_data
from prism.jacobian import compute_jacobian_for_tokens, softmax_jacobian

SEED = 42
MODEL_ID = "qwen35-08b"


def _token_set_id(token_set: list[int]) -> str:
    raw = ",".join(str(t) for t in sorted(token_set)).encode()
    return hashlib.sha1(raw).hexdigest()[:8]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd="."
        ).stdout.strip()
        return out[:8] or "nogit"
    except Exception:
        return "nogit"
