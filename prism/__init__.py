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
# NOTE (audit docs/changed_review.md, 2026-07-18): prism.calibration imports
# `datasets` at module level, so `import prism` REQUIRES `datasets` to be
# installed (pre-v6.0 __init__ was docstring-only and had no such dependency).
# Deliberately NOT wrapped in try/except and NOT made a lazy import:
# `load_calibration_data` is a core entry point of the public API, and a
# silent absence would surface as a confusing NameError deep in a
# long-running experiment. If `datasets` is missing, `import prism` fails
# with a clear ModuleNotFoundError at install/import time. To use the rest
# of the package without it: `pip install datasets` (it is small).

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
