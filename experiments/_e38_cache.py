"""Shared exp27b dense-cache loader for the E-38 series (docs/E38.md).

The dense cache lives at ``kaggle_kernel/results/exp27b_jcache_L{l}.pt``
(6 files, ~2.1 GB each) locally. On Kaggle, the producer kernel
``prism-exp27b-dense-profile``'s /kaggle/working is mounted read-only at
``/kaggle/input/prism-exp27b-dense-profile/results/`` (kernel_sources), so
``jcache_path`` falls back to that mount when the local file is absent.

Per-layer file layout (verified):

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

# REPO/JCACHE_DIR are meaningless when inlined into a Kaggle notebook cell
# (no __file__); fall back to cwd — the candidate list in jcache_path() is
# what actually resolves the file there.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # inlined into a notebook cell
    _HERE = os.getcwd()
REPO = os.path.abspath(os.path.join(_HERE, ".."))
JCACHE_DIR = os.path.join(REPO, "kaggle_kernel", "results")
# Producer kernel slug (kaggle_kernel/push_all.py, exp27b): its outputs mount
# at /kaggle/input/<slug>/ on a kernel that lists it in kernel_sources.
_E27B_KAGGLE_SLUG = "prism-exp27b-dense-profile"
LAYERS = (6, 12, 18, 19, 22, 23)
DOMAINS = ("wiki", "math", "code", "chat")
N_PER_DOMAIN = 8
N_DOCS = 32
S = 128
D = 1024
K = 128


def jcache_path(layer: int) -> str:
    """Resolve the per-layer cache file.

    Local: ``kaggle_kernel/results/exp27b_jcache_L{l}.pt``.
    Kaggle: the local path does not exist, so fall back to the producer
    kernel's outputs mount. Returns the canonical local name if neither
    exists (torch.load then fails loudly with a clear path).
    """
    local = os.path.join(JCACHE_DIR, f"exp27b_jcache_L{layer}.pt")
    if os.path.exists(local):
        return local
    kaggle = os.path.join(
        "/kaggle/input", _E27B_KAGGLE_SLUG, "results", f"exp27b_jcache_L{layer}.pt"
    )
    if os.path.exists(kaggle):
        return kaggle
    return local
    return os.path.join(JCACHE_DIR, f"exp27b_jcache_L{layer}.pt")


def load_layer_H(layer: int) -> torch.Tensor:
    """Load ONLY the hidden-state field H from one layer file → [N, S, D].

    For experiments that use H alone (e.g. E-38a): avoids stacking the
    2.1 GB JR tensor (peak ~4.2 GB → ~2.1 GB, and the per-position J/P
    copies are skipped). The file itself is still ~2.1 GB on disk — this
    is a RAM/time saver, not a disk one (docs/E38_fixes2.md §4).
    """
    ck = torch.load(jcache_path(layer), map_location="cpu", weights_only=False)
    if ck["S"] != S or ck["D"] != D:
        raise ValueError(f"layer {layer}: unexpected S={ck['S']} D={ck['D']}")
    n = len(DOMAINS) * N_PER_DOMAIN
    H = torch.empty(n, S, D, dtype=torch.float32)
    i = 0
    for dom in DOMAINS:
        for si in range(N_PER_DOMAIN):
            for t in range(S):
                H[i, t] = ck["H"][(dom, si, t)].float()
            i += 1
    return H


def load_layer(layer: int, want: tuple = ("J", "P", "H")) -> dict:
    """Load one layer file; return dict with stacked arrays.

    ``want`` selects which fields to stack (docs/E38_fixes2.md §6 —
    experiments should only pay for the fields they use): any subset of
    ("J", "P", "H"), default = all three. Skipping H (used by E-38b)
    drops the peak from ~4.2 GB to ~2.1 GB per layer.

    Returns dict with keys (only for requested fields):
        J   torch.Tensor [N_DOCS, S, K, D]  raw J
        P   torch.Tensor [N_DOCS, S, K]     normalized p128
        H   torch.Tensor [N_DOCS, S, D]     hidden states
        U128 list[int]                      token set
    Docs are ordered (domain-major): (wiki,0..7), (math,0..7), (code,0..7),
    (chat,0..7).

    Peak memory is driven by `want`: pass only the fields a consumer uses
    (e.g. want=("P",) for position audits) to stay in the ~2 GB class.
    """
    bad_want = set(want) - {"J", "P", "H"}
    if bad_want:
        raise ValueError(f"load_layer: unknown want fields {sorted(bad_want)}")
    # weights_only=False is deliberate: the cache is an internal artifact we
    # produced ourselves (tensors + tuples + a list of ints), not an
    # untrusted model state dict. Re-evaluate if the cache provenance changes.
    ck = torch.load(jcache_path(layer), map_location="cpu", weights_only=False)
    if ck["S"] != S or ck["D"] != D:
        raise ValueError(f"layer {layer}: unexpected S={ck['S']} D={ck['D']}")
    n = len(DOMAINS) * N_PER_DOMAIN
    J = torch.empty(n, S, K, D, dtype=torch.float32) if "J" in want else None
    P = torch.empty(n, S, K, dtype=torch.float32) if "P" in want else None
    H = torch.empty(n, S, D, dtype=torch.float32) if "H" in want else None
    i = 0
    for dom in DOMAINS:
        for si in range(N_PER_DOMAIN):
            for t in range(S):
                key = (dom, si, t)
                if J is not None:
                    J[i, t] = ck["JR"][key].float()
                if P is not None:
                    P[i, t] = ck["P128"][key].float()
                if H is not None:
                    H[i, t] = ck["H"][key].float()
            i += 1
    out = {}
    if J is not None:
        out["J"] = J
    if P is not None:
        # Guard: gauge_free_J below assumes P is normalized (sum=1). If
        # someone swaps in P128_RAW (sub-stochastic), metrics would
        # silently be "half gauge-free". Fail loudly instead.
        psum = P.sum(dim=-1)
        if not torch.allclose(psum, torch.ones_like(psum), atol=1e-3):
            bad = (psum - 1.0).abs().max().item()
            raise ValueError(f"layer {layer}: P is not normalized (max |sum-1| = {bad:.3e})")
        out["P"] = P
    if H is not None:
        out["H"] = H
    out["U128"] = list(ck["U128"])
    return out


def gauge_free_J(J: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Gauge-free J: (..., K, D) and P (..., K) -> (..., K, D).

    J_gf = p[:, None] * (J - J^T @ p)  — same as exp27_6_isolation.

    Requires P normalized (sum=1 per row): with sub-stochastic P128_RAW the
    result is a silent "half gauge-free" object with different semantics.
    """
    c = torch.einsum("...k,...kd->...d", P, J)  # J^T p, (... , D)
    return P[..., :, None] * (J - c[..., None, :])
