"""PRISM: Jacobian-based analysis of transformer models.

Public API. Import from ``prism`` rather than submodules when possible.
Backward-compatible: ``from prism.jacobian import ...`` continues to work.
"""

from .bridge import jacobian_gap
from .esg import (
    compute_esg,
    compute_esg_per_sample,
    cross_positional_gramian,
)
from .jacobian import (
    compute_jacobian_for_tokens,
    compute_hidden_jacobian,
    compute_hidden_jacobian_rows,
    softmax_jacobian,
    uniform_centered,
    bridge_rows_backward,
    h_norm_at_perturbed_h,
)
from .calibration import (
    load_calibration_data,
    extract_embeddings,
    uniform_layer_mapping,
    get_layer_types,
    build_global_token_set,
)
from .diagnostics import (
    run_layer_diagnostics,
    run_identity_test,
    run_heatmap,
)
from .layer_selection import select_pairs, classify_pair
from .cache_io import find_targets_dir

__all__ = [
    "jacobian_gap",
    "compute_esg",
    "compute_esg_per_sample",
    "cross_positional_gramian",
    "compute_jacobian_for_tokens",
    "compute_hidden_jacobian",
    "compute_hidden_jacobian_rows",
    "softmax_jacobian",
    "uniform_centered",
    "bridge_rows_backward",
    "h_norm_at_perturbed_h",
    "load_calibration_data",
    "extract_embeddings",
    "uniform_layer_mapping",
    "get_layer_types",
    "build_global_token_set",
    "run_layer_diagnostics",
    "run_identity_test",
    "run_heatmap",
    "select_pairs",
    "classify_pair",
    "find_targets_dir",
]
