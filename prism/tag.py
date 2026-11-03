"""ChannelTag: mandatory passport for every E-40.x result (ТЗ E40_math_core §II.1, M1).

Every artifact number must say HOW it was produced before it says HOW MUCH.
Three fields are ``identity-relevant``: two results may only be compared when
they agree on them.

- ``p_provenance`` — absolute values of ``tr P``, ``iso``, ``O1`` change when p
  changes (renormalized P128 vs raw P128_RAW moved L6 O1/iso 0.511 → 0.322);
  ``g_uni`` does not depend on p at all (``p_algebra.g_uni``).
- ``norm_type`` — norm vs energy differ by a square (E-33r's «98–99 %» was a
  norm labelled as energy; paper §6 item 19).
- ``slice`` — identities hold inside a slice only; ``g_uni(J^gf) ≡ 0`` is true
  exclusively on the gf slice (ТЗ §I.2 slice-locality rule).
- ``aggregation`` — median of a ratio ≠ ratio of sums; the identity
  ``frac_K = 1 − 2c + s²`` closes only under ``ratio_of_means`` (E6, ТЗ §I.5).
  Default ``ratio_of_means``.

Usage::

    from prism.tag import ChannelTag, make_tag
    t = make_tag(protocol="natural", slice="raw", window="full", side="right",
                 p_provenance="truncated_U128", layer=23, rank=16,
                 norm_type="energy", source="cache", status="measured")
    # aggregation defaults to "ratio_of_means"; any real/null ratio must set it.

An artifact without a ``ChannelTag`` is invalid (ТЗ §II.1 rules).
"""

from __future__ import annotations

from collections import namedtuple

__all__ = ["ChannelTag", "make_tag", "IDENTITY_FIELDS", "TAG_DOMAINS"]

#: Allowed values per field (fail-loud membership, trust-boundary style).
TAG_DOMAINS: dict[str, tuple[str, ...]] = {
    "protocol": ("natural", "deltaW"),
    "slice": ("raw", "gf", "fixP"),
    "window": ("top", "tail", "full"),
    "side": ("left", "right", "mixed"),
    "p_provenance": ("truncated_U128", "full_softmax", "other"),
    "aggregation": ("ratio_of_means", "mean", "median", "energy_weighted"),
    "norm_type": ("norm", "energy", "angle"),
    "source": ("cache", "recomputed", "audit-only"),
    "status": ("measured", "inferred", "not_testable"),
}

#: Fields two results must agree on before numbers are comparable (ТЗ §II.1).
IDENTITY_FIELDS = ("p_provenance", "norm_type", "slice", "aggregation")

ChannelTag = namedtuple(
    "ChannelTag",
    ["protocol", "slice", "window", "side", "p_provenance", "aggregation",
     "layer", "rank", "norm_type", "source", "status"],
)


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
