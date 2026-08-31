"""Shared target-cache resolution (Kaggle kernel_sources mount).

Kaggle FUSE mount: os.path.isdir() lies on /kaggle/input (returns False for
existing dirs) but os.listdir() works. So every check here is listdir-based,
and the walk is layout-independent: it finds a dir containing *.pt at any
depth under any mounted slug — no hardcoded producer layout assumed.
"""

import os

# Max depth under input_root to search (producer puts cache at depth 2:
# results/targets; 4 covers any reasonable layout).
_MAX_DEPTH = 4


def _has_pt(d: str) -> bool:
    try:
        return any(f.endswith(".pt") for f in os.listdir(d))
    except OSError:
        return False


def _walk(root: str, depth: int):
    """Yield dirs under root (listdir-based), depth-limited."""
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return
    for e in entries:
        p = os.path.join(root, e)
        if _has_pt(p):
            yield p
        if depth > 1:
            yield from _walk(p, depth - 1)


def find_targets_dir(preferred: str, input_root: str = "/kaggle/input") -> str:
    """Return the dir holding the target-cache *.pt files.

    Order: `preferred` (local path / runner-injected), then a listdir walk
    of every mount under `input_root`. Raises FileNotFoundError with the
    mount inventory when nothing is found (no silent fallback, no listdir crash).
    """
    if _has_pt(preferred):
        return preferred
    for hit in _walk(input_root, _MAX_DEPTH):
        return hit
    mounts = []
    try:
        for d in sorted(os.listdir(input_root)):
            try:
                mounts.append(f"{d}/:{os.listdir(os.path.join(input_root, d))[:4]}")
            except OSError:
                mounts.append(f"{d}/:<unreadable>")
    except OSError:
        mounts.append("<no input mount>")
    raise FileNotFoundError(
        f"No target cache .pt in {preferred} nor under {input_root}/ "
        f"(mounts: {mounts}). Run Stage 0 first — it must publish results/targets/*.pt."
    )
