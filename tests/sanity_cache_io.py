"""sanity: prism.cache_io.find_targets_dir — layout-independent, FUSE-safe (listdir-only) target-cache resolution."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prism.cache_io import find_targets_dir  # noqa: E402


def mk(d, files):
    os.makedirs(d, exist_ok=True)
    for f in files:
        with open(os.path.join(d, f), "w"):
            pass


def main():
    tmp = tempfile.mkdtemp()

    # 1. preferred dir wins (local dev path / injected RES/targets)
    pref = os.path.join(tmp, "results", "targets")
    mk(pref, ["3_2_0.pt"])
    got = find_targets_dir(pref, input_root=os.path.join(tmp, "input"))
    assert got == pref, f"preferred: {got}"

    # 2. walk finds .pt at an arbitrary depth under an arbitrary slug
    inp = os.path.join(tmp, "input")
    d2 = os.path.join(inp, "some-slug", "results", "targets")
    mk(d2, ["9_7_0.pt"])
    got = find_targets_dir(os.path.join(tmp, "none"), input_root=inp)
    assert got == d2, f"walk: {got}"

    # 3. .pt buried 3 levels deep, different slug
    inp3 = os.path.join(tmp, "input3")
    d3 = os.path.join(inp3, "another-mount", "a", "b", "targets")
    mk(d3, ["1_0.pt", "s.json"])
    got = find_targets_dir(os.path.join(tmp, "none3"), input_root=inp3)
    assert got == d3, f"deep walk: {got}"

    # 4. json-only mount is not mistaken for the target cache
    inp4 = os.path.join(tmp, "input4")
    mk(os.path.join(inp4, "json-only", "results"), ["s.json"])
    try:
        find_targets_dir(os.path.join(tmp, "none4"), input_root=inp4)
        raise AssertionError("json-only dir returned")
    except FileNotFoundError as e:
        assert "No target cache" in str(e)

    # 5. nothing found -> clear error naming the mounts, not a listdir crash
    try:
        find_targets_dir(
            os.path.join(tmp, "none5"), input_root=os.path.join(tmp, "empty-input")
        )
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as e:
        assert "No target cache" in str(e) and "empty-input" in str(e)

    print("PASS cache_io: preferred / walk / deep / json-only / clear-error")


if __name__ == "__main__":
    main()
