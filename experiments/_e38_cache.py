"""Shared exp27b dense-cache loader for the E-38 series (docs/E38.md).

The dense cache lives at ``kaggle_kernel/results/exp27b_jcache_L{l}.pt``
(6 files, ~2.1 GB each) locally. On Kaggle the producer kernel's
kernel-source mount is NOT attached (its outputs total 13 GB, above the
mount limit), so ``jcache_path`` resolves from the two Kaggle datasets
``prism-exp27b-jcache-jca`` (L6/12/18) and ``prism-exp27b-jcache-jcb``
(L19/22/23), mounted at
``/kaggle/input/datasets/voroninmichael/<dataset>/`` (dataset_sources,
same split exp32r31-tail-recompute used).

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
# Producer kernel slug (kaggle_kernel/push_all.py, exp27b): its kernel-source
# mount does NOT attach on 38d — the producer's outputs are 13 GB, above
# the kernel-source mount limit (observed 2026-10-25: /kaggle/input only
# contained notebooks + self). The cache is therefore ALSO published as two
# Kaggle datasets (same split as exp32r31-tail-recompute uses):
#   jca: L6, L12, L18   /   jcb: L19, L22, L23
# mounted at /kaggle/input/datasets/voroninmichael/prism-exp27b-jcache-{jca,jcb}/.
#: Which column basis to read. "exp27b" = the published GLOBAL FREQUENCY basis U128
#: (one basis for all positions; sub-stochastic p rows, ~half with mass < 0.5).
#: "exp27c" = POSITIONAL top-K (each position its own K columns, p a real softmax
#: truncation) — audit4 Q10 verdict round. Set with `PRISM_JCACHE=exp27c`, so a
#: recompute run is one env var and cannot silently mix bases: the file carries
#: `column_mode` and the loaders refuse a mismatch.
CACHE_ID = os.environ.get("PRISM_JCACHE", "exp27b").strip()
_COLUMN_MODE = {"exp27b": "freq", "exp27c": "positional"}
if CACHE_ID not in _COLUMN_MODE:
    raise ValueError(f"PRISM_JCACHE={CACHE_ID!r} not in exp27b|exp27c")
# Where a freshly downloaded (positional) cache lands; /data holds the big copies.
_EXTRA_DIRS = (JCACHE_DIR, os.path.join("/data/prism_kernels", CACHE_ID, "results"),
               os.path.join("/data/stage", CACHE_ID))
_E27B_KAGGLE_USERNAME = "voroninmichael"
_E27B_KAGGLE_SLUG = "prism-exp27b-dense-profile"
_E27B_DATASETS = ("prism-exp27b-jcache-jca", "prism-exp27b-jcache-jcb")
LAYERS = (6, 12, 18, 19, 22, 23)
# GDN (linear-attention) layers per model.config.layer_types (Qwen3.5-0.8B,
# 24 layers: GDN at {0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22},
# full attention at {3,7,11,15,19,23}). Among LAYERS: GDN = 6/12/18/22,
# full = 19/23. The old hardcoded (19, 22) in exp38a was INVERTED on 4/6
# layers (docs/E38_fixes.md §1.5, P1): a constant here, because loading
# the model just for the label would be disproportionate.
GDN_LAYERS = frozenset({0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
                        20, 21, 22})
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
    canonical = os.path.join(JCACHE_DIR, f"{CACHE_ID}_jcache_L{layer}.pt")
    for d in _EXTRA_DIRS:
        local = os.path.join(d, f"{CACHE_ID}_jcache_L{layer}.pt")
        if os.path.exists(local):
            return local
    # Kaggle: dataset mounts first (reliable), then the kernel-source mount
    # (only attaches if producer outputs are under the mount limit).
    # Dataset mount layout varies by Kaggle version: observed
    # /kaggle/input/<dataset>/ (2026-10-25, v6) and
    # /kaggle/input/datasets/<user>/<dataset>/ (exp32r31 run) — probe both.
    candidates = []
    for ds in _E27B_DATASETS:
        candidates += [
            os.path.join(
                "/kaggle/input", ds, f"{CACHE_ID}_jcache_L{layer}.pt"
            ),
            os.path.join(
                "/kaggle/input/datasets", _E27B_KAGGLE_USERNAME, ds,
                f"{CACHE_ID}_jcache_L{layer}.pt",
            ),
        ]
    candidates += [
        os.path.join(
            "/kaggle/input", f"{_E27B_KAGGLE_USERNAME}-{_E27B_KAGGLE_SLUG}",
            "results", f"{CACHE_ID}_jcache_L{layer}.pt",
        ),
        os.path.join(
            "/kaggle/input", _E27B_KAGGLE_SLUG, "results",
            f"{CACHE_ID}_jcache_L{layer}.pt",
        ),
    ]
    for mount in candidates:
        if os.path.exists(mount):
            return mount
    # Diagnostic: on Kaggle a miss means the producer mount layout changed
    # or the source wasn't attached — say so in the log loudly.
    if os.path.isdir("/kaggle/input"):
        try:
            lines = []
            for e in sorted(os.listdir("/kaggle/input")):
                p = os.path.join("/kaggle/input", e)
                if os.path.isdir(p):
                    try:
                        sub = sorted(os.listdir(p))[:8]
                    except OSError:
                        sub = ["<unreadable>"]
                    lines.append(f"{e}/ = {sub}")
                else:
                    lines.append(e)
        except OSError:
            lines = ["<unreadable>"]
        print(f"[e38cache] JCACHE MISS for L{layer}; /kaggle/input = {lines}",
              flush=True)
    return canonical          # canonical repo path, so torch.load reports something meaningful


# Где лежит файл кэша, в котором ЕСТЬ поле P128_RAW (raw-масса выбранных столбцов).
# exp27b: в файлах под kaggle_kernel/results его нет, оно дописано E-37 в копию (рядом
# лежат *.pre_raw.bak). exp27c: поле родное => пустой кортеж, берём jcache_path.
RAW_CACHE_DIRS = {"exp27b": ("/data/prism_kernels/exp27b/results", "/data/stage/jc_a"),
                  "exp27c": ()}


def raw_jcache_path(layer: int) -> str:
    """Путь к файлу текущего cache_id с полем P128_RAW (для sentinel-маски E-34/E-37)."""
    for d in RAW_CACHE_DIRS.get(CACHE_ID, ()):
        cand = os.path.join(d, f"{CACHE_ID}_jcache_L{layer}.pt")
        if os.path.exists(cand):
            return cand
    return jcache_path(layer)


def _check_basis(ck: dict, layer: int) -> None:
    """A positional-basis number must never be compared with a frequency-basis one:
    the columns of J are different objects, not a different slice of the same thing."""
    got = ck.get("column_mode", "freq")          # exp27b files predate the field
    if got != _COLUMN_MODE[CACHE_ID]:
        raise ValueError(
            f"layer {layer}: cache says column_mode={got!r} but PRISM_JCACHE="
            f"{CACHE_ID!r} expects {_COLUMN_MODE[CACHE_ID]!r} — mixing bases")


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
    _check_basis(ck, layer)
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
        U   torch.Tensor [N_DOCS, S, K]     column token ids PER POSITION
                                    (want=("U",), exp27c only — the freq cache has
                                     no per-position basis, only U128)
        U128 list[int]                      token set (freq: the basis itself;
                                     positional: the union over positions, provenance)
    Docs are ordered (domain-major): (wiki,0..7), (math,0..7), (code,0..7),
    (chat,0..7).

    Peak memory is driven by `want`: pass only the fields a consumer uses
    (e.g. want=("P",) for position audits) to stay in the ~2 GB class.
    """
    bad_want = set(want) - {"J", "P", "H", "U"}
    if bad_want:
        raise ValueError(f"load_layer: unknown want fields {sorted(bad_want)}")
    # weights_only=False is deliberate: the cache is an internal artifact we
    # produced ourselves (tensors + tuples + a list of ints), not an
    # untrusted model state dict. Re-evaluate if the cache provenance changes.
    ck = torch.load(jcache_path(layer), map_location="cpu", weights_only=False)
    if ck["S"] != S or ck["D"] != D:
        raise ValueError(f"layer {layer}: unexpected S={ck['S']} D={ck['D']}")
    _check_basis(ck, layer)
    if "U" in want and "UTOP" not in ck:
        raise ValueError(
            f"layer {layer}: this cache has no per-position basis (freq cache); "
            "want=('U',) needs the positional exp27c cache")
    n = len(DOMAINS) * N_PER_DOMAIN
    J = torch.empty(n, S, K, D, dtype=torch.float32) if "J" in want else None
    P = torch.empty(n, S, K, dtype=torch.float32) if "P" in want else None
    H = torch.empty(n, S, D, dtype=torch.float32) if "H" in want else None
    U = torch.empty(n, S, K, dtype=torch.long) if "U" in want else None
    i = 0
    for dom in DOMAINS:
        for si in range(N_PER_DOMAIN):
            if U is not None:
                U[i] = ck["UTOP"][(dom, si)].reshape(S, K).long()
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
    if U is not None:
        out["U"] = U
    out["U128"] = list(ck["U128"])
    out["column_mode"] = ck.get("column_mode", "freq")
    return out


def gauge_free_J(J: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Gauge-free J: (..., K, D) and P (..., K) -> (..., K, D).

    J_gf = p[:, None] * (J - J^T @ p)  — same as exp27_6_isolation.

    Requires P normalized (sum=1 per row): with sub-stochastic P128_RAW the
    result is a silent "half gauge-free" object with different semantics.
    """
    c = torch.einsum("...k,...kd->...d", P, J)  # J^T p, (... , D)
    return P[..., :, None] * (J - c[..., None, :])
