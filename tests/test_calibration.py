"""Tests for calibration module."""

from prism.calibration import uniform_layer_mapping


def test_uniform_layer_mapping_basic():
    """36 layers -> 26 layers maps correctly."""
    mapping = uniform_layer_mapping(36, 26)
    assert len(mapping) == 36
    assert mapping[0] == 0
    assert mapping[35] == 25


def test_uniform_layer_mapping_identity():
    """Same number of layers maps 1:1."""
    mapping = uniform_layer_mapping(10, 10)
    for i in range(10):
        assert mapping[i] == i


def test_uniform_layer_mapping_monotonic():
    """Mapping is monotonically non-decreasing."""
    mapping = uniform_layer_mapping(36, 26)
    values = list(mapping.values())
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1]


def test_uniform_layer_mapping_all_layers_covered():
    """Every small model layer appears in mapping."""
    mapping = uniform_layer_mapping(36, 26)
    small_layers = set(mapping.values())
    assert small_layers == set(range(26))


def test_uniform_layer_mapping_qwen35_small():
    """Qwen3.5-0.8B has 26 layers."""
    mapping = uniform_layer_mapping(26, 26)
    assert mapping[0] == 0
    assert mapping[25] == 25
