"""
MAPLE Models Package

This package provides estimators for FEP analysis, organized into
probabilistic (variational inference) and deterministic (closed-form) methods.
"""

from .base import BaseEstimator
from .graph_data import GraphData
from .probabilistic.mixture.presets import (
    GMMParams,
    DatasetPreset,
    get_preset,
    list_presets,
    register_preset,
)
from .config import (
    BaseEstimatorConfig,
    VariationalEstimatorConfig,
    MixtureVIConfig,
    GaussianMixtureVIConfig,
    CycleClosureCorrectionConfig,
    SpectralCorrectionConfig,
    PriorType,
    GuideType,
    ErrorDistributionType,
    create_config,
)
from .probabilistic import VariationalEstimator, MixtureVI, GaussianMixtureVI
from .deterministic import CycleClosureCorrection, SpectralCorrection

__all__ = [
    # Base
    "BaseEstimator",
    "GraphData",
    # Presets
    "GMMParams",
    "DatasetPreset",
    "get_preset",
    "list_presets",
    "register_preset",
    # Estimators
    "VariationalEstimator",
    "MixtureVI",
    "GaussianMixtureVI",
    "CycleClosureCorrection",
    "SpectralCorrection",
    # Configs
    "BaseEstimatorConfig",
    "VariationalEstimatorConfig",
    "MixtureVIConfig",
    "GaussianMixtureVIConfig",
    "CycleClosureCorrectionConfig",
    "SpectralCorrectionConfig",
    # Enums
    "PriorType",
    "GuideType",
    "ErrorDistributionType",
    # Factory
    "create_config",
]
