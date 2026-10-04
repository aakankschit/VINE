"""
Compatibility shim for the pre-restructure import path.

The GMM fitting helpers now live in
:mod:`maple_fep.models.probabilistic.mixture.fitting`. Importing directly
from this module emits a :class:`DeprecationWarning`; prefer
``from maple_fep.utils import fit_sklearn_gmm, ...`` (which remains
canonical) or the new location.
"""

import warnings as _warnings

from ..models.probabilistic.mixture.fitting import (
    GMMSelectionResult,
    fit_gmm_cv,
    fit_sklearn_gmm,
    residuals_from_dataset,
    select_gmm_components,
)

_warnings.warn(
    "maple_fep.utils.sklearn_gmm_fitting is deprecated; import from "
    "maple_fep.utils (top-level) or "
    "maple_fep.models.probabilistic.mixture.fitting instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "fit_sklearn_gmm",
    "fit_gmm_cv",
    "select_gmm_components",
    "residuals_from_dataset",
    "GMMSelectionResult",
]
