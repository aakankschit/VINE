"""
Mixture VI subpackage.

Bundles the :class:`MixtureVI` estimator together with its supporting
machinery: GMM presets (``presets``), GMM fitting helpers (``fitting``),
and deprecated aliases (``_legacy``). The mixture VI model is the only
estimator with a substantial supporting surface, so it gets its own
subpackage.
"""

from ._legacy import GaussianMixtureVI, _make_mixture_vi_from_old_config
from .estimator import MixtureVI
from .fitting import (
    GMMSelectionResult,
    fit_gmm_cv,
    fit_sklearn_gmm,
    residuals_from_dataset,
    select_gmm_components,
)
from .presets import (
    DEFAULT_PRESET,
    DatasetPreset,
    GMMParams,
    get_preset,
    list_presets,
    register_preset,
)

__all__ = [
    "MixtureVI",
    "GaussianMixtureVI",
    "GMMParams",
    "DatasetPreset",
    "DEFAULT_PRESET",
    "get_preset",
    "list_presets",
    "register_preset",
    "fit_sklearn_gmm",
    "fit_gmm_cv",
    "select_gmm_components",
    "residuals_from_dataset",
    "GMMSelectionResult",
]
