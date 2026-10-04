"""
VINE Utilities Package

This package provides utility functions for the VINE package including
performance tracking, model comparison, data persistence, and GMM fitting.
"""

from .parameter_sweep import (ParameterSweep,
                              create_comprehensive_parameter_study,
                              create_prior_sweep_experiment,
                              create_gmvi_prior_sweep_experiment)
from .performance_tracker import (ModelRun, PerformanceTracker,
                                  compare_model_runs, load_performance_history)
from ..models.probabilistic.mixture.fitting import (
    fit_sklearn_gmm,
    fit_gmm_cv,
    select_gmm_components,
    residuals_from_dataset,
    GMMSelectionResult,
)

__all__ = [
    "PerformanceTracker",
    "ModelRun",
    "load_performance_history",
    "compare_model_runs",
    "ParameterSweep",
    "create_prior_sweep_experiment",
    "create_gmvi_prior_sweep_experiment",
    "create_comprehensive_parameter_study",
    "fit_sklearn_gmm",
    "fit_gmm_cv",
    "select_gmm_components",
    "residuals_from_dataset",
    "GMMSelectionResult",
]
