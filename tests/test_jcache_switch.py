"""Gate on the readout-basis switch (`PRISM_JCACHE`) in `experiments/_e38_cache.py`.

audit4 Q10: numbers from the global-frequency cache (exp27b) and from the
positional cache (exp27c) are different objects — different columns of J, not a
different slice of the same thing. The recompute of every dependent experiment is
supposed to be ONE env var, which is only safe while the reader (a) takes the file
name and the P128_RAW location from the cache id, (b) returns the per-position
basis for the positional cache, and (c) refuses a file whose `column_mode`
disagrees with the requested basis. Synthetic caches only — no real file touched.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                "experiments"))
import _e38_cache as ec  # noqa: E402

DOMS = ("wiki", "math")
NS, SS, DD, KK = 2, 3, 5, 4


def _shrink(monkeypatch, tmp_path, cache_id, raw_dirs=None):
    """Make the loader's geometry tiny and point every candidate dir at tmp_path."""
    monkeypatch.setattr(ec, "CACHE_ID", cache_id)
    monkeypatch.setattr(ec, "DOMAINS", DOMS)
    monkeypatch.setattr(ec, "N_PER_DOMAIN", NS)
    monkeypatch.setattr(ec, "S", SS)
    monkeypatch.setattr(ec, "D", DD)
    monkeypatch.setattr(ec, "K", KK)
    monkeypatch.setattr(ec, "JCACHE_DIR", str(tmp_path))
    monkeypatch.setattr(ec, "_EXTRA_DIRS", (str(tmp_path),))
    monkeypatch.setattr(ec, "RAW_CACHE_DIRS",
                        {cache_id: tuple(raw_dirs or ())})


def _write_cache(tmp_path, column_mode, *, with_utop=True, layer=6):
    g = torch.Generator().manual_seed(0)
    jr, p, h, utop = {}, {}, {}, {}
    for dom in DOMS:
        for si in range(NS):
            ut = torch.randint(0, 97, (SS, KK), generator=g)
            if with_utop:
                utop[(dom, si)] = ut
            for t in range(SS):
                jr[(dom, si, t)] = torch.randn(KK, DD, generator=g)
                p[(dom, si, t)] = torch.softmax(torch.randn(KK, generator=g), dim=-1)
                h[(dom, si, t)] = torch.randn(DD, generator=g)
    ck = {"JR": jr, "P128": p, "H": h, "P128_RAW": dict(p),
          "RANKS": {(d, s, t): (1.0, 2.0, 3.0) for d in DOMS
                    for s in range(NS) for t in range(SS)},
          "U128": list(range(KK)), "column_mode": column_mode, "pool_fp": "fp",
          "layers": [layer], "D": DD, "n_samples": NS, "domains": list(DOMS), "S": SS}
    if with_utop:
        ck["UTOP"] = utop        # частотный кэш поля не имеет вообще (не пустой dict)
    path = os.path.join(str(tmp_path), f"{ec.CACHE_ID}_jcache_L{layer}.pt")
    torch.save(ck, path)
    return ck, utop


def test_positional_cache_returns_the_per_position_basis(monkeypatch, tmp_path):
    _shrink(monkeypatch, tmp_path, "exp27c")
    ck, utop = _write_cache(tmp_path, "positional")
    out = ec.load_layer(6, want=("J", "P", "U"))
    assert out["column_mode"] == "positional"
    assert out["J"].shape == (len(DOMS) * NS, SS, KK, DD)
    assert out["U"].shape == (len(DOMS) * NS, SS, KK)
    # U[i] must be exactly the stored per-position basis, in domain-major doc order
    i = 0
    for dom in DOMS:
        for si in range(NS):
            assert torch.equal(out["U"][i], utop[(dom, si)])
            i += 1
    del ck


def test_frequency_cache_refuses_to_pretend_it_has_a_basis(monkeypatch, tmp_path):
    """On exp27b `want=("U",)` must fail loud: a single global U128 is NOT a basis."""
    _shrink(monkeypatch, tmp_path, "exp27b")
    _write_cache(tmp_path, "freq", with_utop=False)
    with pytest.raises(ValueError, match="no per-position basis"):
        ec.load_layer(6, want=("U",))
    assert ec.load_layer(6, want=("P",))["column_mode"] == "freq"


def test_mixing_bases_fails_loud(monkeypatch, tmp_path):
    """A positional run pointed at a freq file (or vice versa) must not load."""
    _shrink(monkeypatch, tmp_path, "exp27c")
    _write_cache(tmp_path, "freq")                   # wrong basis for exp27c
    with pytest.raises(ValueError, match="mixing bases"):
        ec.load_layer(6, want=("P",))


def test_files_without_the_field_are_read_as_frequency(monkeypatch, tmp_path):
    """exp27b files predate `column_mode`; absence means freq, not a crash."""
    _shrink(monkeypatch, tmp_path, "exp27b")
    ck, _ = _write_cache(tmp_path, "freq")
    ck.pop("column_mode")
    torch.save(ck, os.path.join(str(tmp_path), "exp27b_jcache_L6.pt"))
    assert ec.load_layer(6, want=("P",))["column_mode"] == "freq"


def test_raw_p128_location_follows_the_cache_id(monkeypatch, tmp_path):
    """No raw-carrying dir configured -> the resolved cache file itself, and never
    a file of the other cache id."""
    _shrink(monkeypatch, tmp_path, "exp27c")
    _write_cache(tmp_path, "positional")
    assert ec.raw_jcache_path(6) == ec.jcache_path(6)
    assert "exp27c_jcache_L6" in ec.raw_jcache_path(6)

    other = tmp_path / "rawdir"
    other.mkdir()
    stale = other / "exp27b_jcache_L6.pt"            # an exp27b file must never serve exp27c
    torch.save({"column_mode": "freq"}, stale)
    monkeypatch.setattr(ec, "RAW_CACHE_DIRS", {"exp27c": (str(other),)})
    assert ec.raw_jcache_path(6) == ec.jcache_path(6)
