"""ChannelTag: mandatory passport for every E-40.x result (ТЗ E40_math_core §II.1, M1).

Every artifact number must say HOW it was produced before it says HOW MUCH.
Three fields are ``identity-relevant``: two results may only be compared when
they agree on them.

- ``p_mode`` — absolute values of ``tr P``, ``iso``, ``O1`` change when p
  changes (renormalized P128 vs raw P128_RAW moved L6 O1/iso 0.511 → 0.322);
  ``g_uni`` does not depend on p at all (``p_algebra.g_uni``). Unknown mode is
  a stop condition, not a legal tag value (``other`` removed, патч v2 §1.6).
- ``p_slice`` — ``p_t`` (per-position) vs ``p_0`` (fixP) are DIFFERENT operators
  ``P(p)``; triple numbers without ``p_slice`` are not comparable. ``g_uni``
  needs no ``p_slice`` (p-free in both branches: raw has no p; gf ≡ 0).
- ``norm_type`` — norm vs energy differ by a square (E-33r's «98–99 %» was a
  norm labelled as energy; paper §6 item 19).
- ``slice`` — identities hold inside a slice only; ``g_uni(J^gf) ≡ 0`` is true
  exclusively on the gf slice (ТЗ §I.2 slice-locality rule).
- ``aggregation`` — median of a ratio ≠ ratio of sums; the identity
  ``frac_K = 1 − 2c + s²`` closes only under ``ratio_of_means`` (E6, ТЗ §I.5).
  Default ``ratio_of_means``.

Usage::

    from prism.tag import ChannelTag, make_tag, require_triple_tag
    t = make_tag(protocol="natural", slice="raw", window="full", side="right",
                 p_mode="truncated_U128", p_slice="p_t", layer=23, rank=16,
                 norm_type="energy", source="cache", status="measured")
    # aggregation defaults to "ratio_of_means"; any real/null ratio must set it.
    require_triple_tag(t)   # ValueError if a triple number would be reported
                            # without slice/aggregation/p_slice (патч v2 §1.6.5)

An artifact without a ``ChannelTag`` is invalid (ТЗ §II.1 rules).
"""

from __future__ import annotations

from collections import namedtuple

__all__ = ["ChannelTag", "make_tag", "IDENTITY_FIELDS", "TRIPLE_EXTRA_FIELDS",
           "require_triple_tag", "TAG_DOMAINS"]

#: Allowed values per field (fail-loud membership, trust-boundary style).
TAG_DOMAINS: dict[str, tuple[str, ...]] = {
    "protocol": ("natural", "deltaW"),
    "slice": ("raw", "gf", "fixP"),
    "window": ("top", "tail", "full"),
    "side": ("left", "right", "mixed"),
    "p_mode": ("truncated_U128", "full_softmax"),
    "p_slice": ("p_t", "p_0"),
    # The vocabulary the aggregator actually implements — the two lists must agree
    # exactly. `energy_weighted` used to be listed here and was an ALIAS of
    # `ratio_of_means` (energy-weighted mean of per-position ratios = ratio of sums),
    # i.e. a fourth name for a third thing: removed, and `per_position`, which really
    # is a different number and was implemented but untaggable, is in (audit4 Q3).
    "aggregation": ("ratio_of_means", "mean", "median", "per_position"),
    "norm_type": ("norm", "energy", "angle"),
    "source": ("cache", "recomputed", "audit-only"),
    "status": ("measured", "inferred", "not_testable"),
}

#: Fields two results must agree on before numbers are comparable (ТЗ §II.1).
IDENTITY_FIELDS = ("p_mode", "norm_type", "slice", "aggregation")
#: Extra fields mandatory when reporting a triple number (c, s, cos, frac_K);
#: g_uni does NOT need them (p-free), патч v2 §1.6 rules 4–5.
TRIPLE_EXTRA_FIELDS = ("p_slice",)

ChannelTag = namedtuple(
    "ChannelTag",
    ["protocol", "slice", "window", "side", "p_mode", "p_slice", "aggregation",
     "layer", "rank", "norm_type", "source", "status"],
)


def require_triple_tag(tag: "ChannelTag") -> "ChannelTag":
    """Fail loud if ``tag`` may not back a triple/real-null number: slice,
    aggregation and p_slice must all be set (патч v2 §1.6.5)."""
    missing = [f for f in ("slice", "aggregation", "p_slice")
               if getattr(tag, f) is None]
    if missing:
        raise ValueError(
            f"ChannelTag: triple/real-null reporting requires {missing} "
            f"(p_t vs p_0 are different operators P(p); median ≠ ratio; "
            f"identities are slice-local)")
    return tag


def make_tag(**kwargs) -> ChannelTag:
    """Build a ChannelTag, validating every enumerated field and requiring the
    identity-relevant ones. ``aggregation`` defaults to ``ratio_of_means``
    (ТЗ §II.1: mandatory for any ratio, this is the identity-closing default).
    ``layer``/``rank`` must be ints or None."""
    kwargs.setdefault("aggregation", "ratio_of_means")
    missing = [f for f in IDENTITY_FIELDS if f not in kwargs or kwargs[f] is None]
    if missing:
        raise ValueError(
            f"ChannelTag: identity-relevant fields missing: {missing} "
            f"(they are what makes two numbers comparable, ТЗ §II.1)"
        )
    for field, allowed in TAG_DOMAINS.items():
        val = kwargs.get(field)
        if val is None:
            continue
        if val not in allowed:
            raise ValueError(
                f"ChannelTag.{field}={val!r} not in {allowed}")
    for field in ("layer", "rank"):
        val = kwargs.get(field)
        if val is not None and not isinstance(val, (int,)):
            raise ValueError(f"ChannelTag.{field}={val!r} must be int or None")
    return ChannelTag(**{f: kwargs.get(f) for f in ChannelTag._fields})
