"""Shared exp27b dense-cache loader for the E-38 series (docs/E38.md).

The dense cache lives at ``kaggle_kernel/results/exp27b_jcache_L{l}.pt``
(6 files, ~2.1 GB each). Per-layer file layout (verified):

    JR[(dom, si, t)]  torch.Tensor [128, 1024]   raw J at position t
    P128[(dom, si, t)] torch.Tensor [128]        normalized p128 row (sum=1)
    P128_RAW[...]      raw sub-stochastic p128 row
    H[(dom, si, t)]   torch.Tensor [1024]        hidden state at position t
    RANKS[(dom, si, t)] tuple
    U128              list[128]                  token set (same for all)
    S=128, D=1024, domains=[wiki, math, code, chat], 8 samples/domain

Note the key order differs from exp27_jcache.pt (v11: (dom, si, l, S, POS)):
here the layer is the FILE, the key is (dom, si, t) with t = position.

N=32 docs total (4 domains x 8). Gauge-free J for metrics:
    J_gaugefree = p128[:, None] * (J - J.T @ p128[None, :])
(same construction as experiments/exp27_6_isolation.py).

Deterministic: no seeds needed in the loader itself; experiments own seeds.
"""
from __future__ import annotations

import os

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JCACHE_DIR = os.path.join(REPO, "kaggle_kernel", "results")
LAYERS = (6, 12, 18, 19, 22, 23)
DOMAINS = ("wiki", "math", "code", "chat")
N_PER_DOMAIN = 8
N_DOCS = 32
S = 128
D = 1024
K = 128


def jcache_path(layer: int) -> str:
    return os.path.join(JCACHE_DIR, f"exp27b_jcache_L{layer}.pt")


def load_layer(layer: int) -> dict:
    """Load one layer file; return dict with stacked arrays.

    Returns dict with keys:
        J   torch.Tensor [N_DOCS, S, K, D]  raw J
        P   torch.Tensor [N_DOCS, S, K]     normalized p128
        H   torch.Tensor [N_DOCS, S, D]     hidden states
        U128 list[int]                      token set
    Docs are ordered (domain-major): (wiki,0..7), (math,0..7), (code,0..7),
    (chat,0..7).
    """
    ck = torch.load(jcache_path(layer), map_location="cpu", weights_only=False)
    if ck["S"] != S or ck["D"] != D:
        raise ValueError(f"layer {layer}: unexpected S={ck['S']} D={ck['D']}")
    n = len(DOMAINS) * N_PER_DOMAIN
    J = torch.empty(n, S, K, D, dtype=torch.float32)
    P = torch.empty(n, S, K, dtype=torch.float32)
    H = torch.empty(n, S, D, dtype=torch.float32)
    i = 0
    for dom in DOMAINS:
        for si in range(N_PER_DOMAIN):
            for t in range(S):
                key = (dom, si, t)
                J[i, t] = ck["JR"][key].float()
                P[i, t] = ck["P128"][key].float()
                H[i, t] = ck["H"][key].float()
            i += 1
    return {"J": J, "P": P, "H": H, "U128": list(ck["U128"])}


def gauge_free_J(J: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Gauge-free J: (..., K, D) and P (..., K) -> (..., K, D).

    J_gf = p[:, None] * (J - J^T @ p)  — same as exp27_6_isolation.
    """
    c = torch.einsum("...k,...kd->...d", P, J)  # J^T p, (... , D)
    return P[..., :, None] * (J - c[..., None, :])
