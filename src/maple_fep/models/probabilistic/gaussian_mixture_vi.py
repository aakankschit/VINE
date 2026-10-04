"""
Compatibility shim for the pre-restructure import path.

The mixture VI estimator now lives in
:mod:`maple_fep.models.probabilistic.mixture`. This module re-exports
the public symbols for any code that still imports from the old path.
"""

from .mixture._legacy import GaussianMixtureVI, _make_mixture_vi_from_old_config
from .mixture.estimator import MixtureVI

__all__ = ["MixtureVI", "GaussianMixtureVI", "_make_mixture_vi_from_old_config"]
