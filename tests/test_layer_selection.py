"""RED: layer re-selection rule (docs/exp17_change.md 0.1/0.2).

Rule (user, 2026-08-25):
  l_S = full_attention layers of small model  +  l_S=0 (0_0)  +  last layer.
  l_L = nearest preimage of l_S under uniform_layer_mapping (tie -> smaller l_L).
  Iteration is PER SELECTED l_S (mapping is many-to-one; per-l_L iteration dupes).
  Classify: l_S full_attention -> o_proj bridge; else (GDN incl. last) -> KL-only.
"""

from prism.layer_selection import classify_pair, select_pairs


def _layer_types(n_S: int, full: set[int]) -> list[str]:
    return ["full_attention" if i in full else "linear_attention" for i in range(n_S)]


def test_3mod4_gives_exactly_seven_pairs():
    lt = _layer_types(24, {3, 7, 11, 15, 19, 23})
    pairs = select_pairs(32, 24, lt)
    assert pairs == [(0, 0), (4, 3), (9, 7), (15, 11), (20, 15), (25, 19), (31, 23)]


def test_per_lS_inversion_no_dupes():
    # naive per-l_L iteration would emit l_S=7 twice (l_L=9 AND l_L=10 -> 7)
    lt = _layer_types(24, {3, 7, 11, 15, 19, 23})
    pairs = select_pairs(32, 24, lt)
    l_S_list = [j for _, j in pairs]
    assert len(l_S_list) == len(set(l_S_list))
    # nearest preimage: l_L = argmin |i*(n_S-1)/(n_L-1) - j|
    for l_L, j in pairs:

        def m(i: int) -> int:
            return round(i * (24 - 1) / (32 - 1))

        preimages = [i for i in range(32) if m(i) == j]
        assert l_L == min(preimages)


def test_last_layer_added_when_not_selected():
    # last (23) not full-attn and not 0 -> still force-included
    lt = _layer_types(24, {3, 7, 11, 15, 19})
    pairs = select_pairs(32, 24, lt)
    assert (31, 23) in pairs
    assert (0, 0) in pairs


def test_0_0_always_selected_even_if_last_is_full():
    # last (23) full-attn -> already in set; 0_0 must still be there, no dup
    lt = _layer_types(24, {3, 7, 11, 15, 19, 23})
    pairs = select_pairs(32, 24, lt)
    assert pairs.count((0, 0)) == 1
    assert pairs.count((31, 23)) == 1


def test_classification_full_vs_gdn():
    lt = _layer_types(24, {3, 7, 11, 15, 19, 23})
    assert classify_pair(3, lt) == "bridge"
    assert classify_pair(23, lt) == "bridge"  # last full-attn -> bridge (point 6)
    assert classify_pair(0, lt) == "kl_only"  # 0_0 GDN -> KL-only
    # last layer GDN -> pair exists but is KL-only (point 6 IF-branch)
    lt_gdn_last = _layer_types(24, {3, 7, 11, 15, 19})
    assert classify_pair(23, lt_gdn_last) == "kl_only"
