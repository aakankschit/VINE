"""
Compatibility shim for the pre-restructure import path.

GMM presets now live in :mod:`maple_fep.models.probabilistic.mixture.presets`.
Importing directly from this module emits a :class:`DeprecationWarning`;
prefer ``from maple_fep.models import GMMParams, get_preset, ...`` (which
remains canonical) or the new location.
"""

import warnings as _warnings

from .probabilistic.mixture.presets import (
    DEFAULT_PRESET,
    DatasetPreset,
    GMMParams,
    _PRESETS,
    get_preset,
    list_presets,
    register_preset,
)

_warnings.warn(
    "maple_fep.models.presets is deprecated; import from "
    "maple_fep.models (top-level) or "
    "maple_fep.models.probabilistic.mixture.presets instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "GMMParams",
    "DatasetPreset",
    "DEFAULT_PRESET",
    "get_preset",
    "list_presets",
    "register_preset",
]
