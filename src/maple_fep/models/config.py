"""
Configuration classes for MAPLE estimators.

This module provides configuration classes for different estimator types in MAPLE,
including the VariationalEstimator and MixtureVI, with proper validation
using Pydantic.
"""

from abc import ABC
from enum import Enum
from typing import List, Optional, Union
import warnings

from pydantic import BaseModel, Field, field_validator

import numpy as np

from .probabilistic.mixture.presets import DEFAULT_PRESET, GMMParams, get_preset


class PriorType(str, Enum):
    """Available prior distribution types for node values"""

    NORMAL = "normal"
    GAMMA = "gamma"
    UNIFORM = "uniform"
    STUDENT_T = "student_t"
    LAPLACE = "laplace"


class GuideType(str, Enum):
    """Available guide types for variational inference"""

    AUTO_DELTA = "auto_delta"
    AUTO_NORMAL = "auto_normal"


class ErrorDistributionType(str, Enum):
    """Available error distribution types for cycle errors"""

    NORMAL = "normal"
    SKEWED_NORMAL = "skewed_normal"


class BaseEstimatorConfig(BaseModel, ABC):
    """Base configuration class for all MAPLE probabilistic estimators"""

    learning_rate: float = Field(
        default=0.001, ge=1e-6, le=1.0, description="Learning rate for optimization"
    )

    num_steps: int = Field(
        default=5000, ge=10, le=100000, description="Number of optimization steps"
    )

    model_type: str = Field(
        description="Type of estimator (VariationalEstimator or GaussianMixtureVI)"
    )


class VariationalEstimatorConfig(BaseEstimatorConfig):
    """Configuration specific to VariationalEstimator"""

    model_type: str = Field(default="VariationalEstimator", frozen=True)

    prior_type: PriorType = Field(
        default=PriorType.NORMAL,
        description="Type of prior distribution for node values",
    )

    prior_parameters: List[float] = Field(
        default=[0.0, 1.0], description="Parameters for the prior distribution"
    )

    error_std: float = Field(
        default=1.0, gt=0.0, description="Standard deviation of cycle errors"
    )

    error_distribution: ErrorDistributionType = Field(
        default=ErrorDistributionType.NORMAL,
        description="Type of distribution for cycle errors",
    )

    error_skew: float = Field(
        default=0.0,
        description="Skewness parameter for skewed normal error distribution",
    )

    guide_type: GuideType = Field(
        default=GuideType.AUTO_DELTA,
        description="Type of guide for variational inference",
    )

    patience: int = Field(
        default=100,
        ge=1,
        description="Early stopping patience: stop training if no improvement after this many steps",
    )

    random_seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. If None, no seed is set.",
    )

    use_edge_weights: bool = Field(
        default=False,
        description="If True and edge uncertainties are available, use per-edge sigma_ij "
        "for heteroscedastic likelihood instead of global error_std.",
    )

    @field_validator("prior_parameters")
    @classmethod
    def validate_prior_parameters(cls, v, info):
        """Validate prior parameters based on prior type"""
        prior_type = info.data.get("prior_type", PriorType.NORMAL)

        if prior_type == PriorType.NORMAL:
            if len(v) != 2:
                raise ValueError(
                    "Normal prior requires exactly 2 parameters: [mean, std]"
                )
            if v[1] <= 0:
                raise ValueError("Standard deviation must be positive")

        elif prior_type == PriorType.GAMMA:
            if len(v) != 2:
                raise ValueError(
                    "Gamma prior requires exactly 2 parameters: [alpha, beta]"
                )
            if v[0] <= 0 or v[1] <= 0:
                raise ValueError("Gamma parameters must be positive")

        elif prior_type == PriorType.UNIFORM:
            if len(v) != 2:
                raise ValueError(
                    "Uniform prior requires exactly 2 parameters: [lower, upper]"
                )
            if v[0] >= v[1]:
                raise ValueError(
                    "Uniform prior: lower bound must be less than upper bound"
                )

        elif prior_type == PriorType.STUDENT_T:
            if len(v) != 2:
                raise ValueError(
                    "Student-T prior requires exactly 2 parameters: [df, scale]"
                )
            if v[0] <= 0 or v[1] <= 0:
                raise ValueError("Student-T parameters must be positive")

        elif prior_type == PriorType.LAPLACE:
            if len(v) != 2:
                raise ValueError(
                    "Laplace prior requires exactly 2 parameters: [loc, scale]"
                )
            if v[1] <= 0:
                raise ValueError("Laplace scale parameter must be positive")

        return v


def _default_likelihood_gmm() -> GMMParams:
    return get_preset(DEFAULT_PRESET).likelihood


def _default_prior_gmm() -> GMMParams:
    return get_preset(DEFAULT_PRESET).prior


class MixtureVIConfig(BaseEstimatorConfig):
    """Configuration for the unified MixtureVI model.

    Both likelihood and prior are Gaussian Mixture Models. By default the
    parameters come from the OpenFE preset (residuals fit on the OpenFE
    benchmark). Users with their own experimental data can fit custom GMMs
    via :meth:`from_user_data`.

    Four construction paths:

    >>> MixtureVIConfig()                       # OpenFE preset (default)
    >>> MixtureVIConfig.from_preset("cdk8")     # Named system-specific preset
    >>> MixtureVIConfig.from_dataset(dataset)   # In-sample fit; K chosen from data
    >>> MixtureVIConfig.from_user_data(         # Custom fit on user residuals
    ...     edge_residuals, node_residuals,
    ...     K_likelihood=3, K_prior=2,
    ... )

    ``from_dataset`` / ``from_user_data`` select the number of mixture
    components from the data (cross-validated log-likelihood) when the
    corresponding ``K`` is left as ``None``.
    """

    model_type: str = Field(default="MixtureVI", frozen=True)

    likelihood_gmm: GMMParams = Field(
        default_factory=_default_likelihood_gmm,
        description="GMM parameters for the likelihood. Defaults to the OpenFE preset.",
    )

    prior_gmm: GMMParams = Field(
        default_factory=_default_prior_gmm,
        description="GMM parameters for the prior. Defaults to the OpenFE preset.",
    )

    # --- Training hyperparameters -------------------------------------------
    kl_weight: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight for KL divergence term in ELBO",
    )

    n_epochs: int = Field(
        default=1000,
        ge=10,
        description="Maximum number of training epochs",
    )

    n_samples: int = Field(
        default=100,
        ge=1,
        description="Number of Monte Carlo samples for ELBO estimation",
    )

    patience: int = Field(
        default=50,
        ge=1,
        description="Early stopping patience",
    )

    learning_rate: float = Field(
        default=0.01,
        ge=1e-6,
        le=1.0,
        description="Learning rate for Adam optimizer",
    )

    random_seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. If None, no seed is set.",
    )

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "MixtureVIConfig":
        """Create a config from a named dataset preset.

        Parameters
        ----------
        name : str
            Preset name (e.g. ``"openfe"``, ``"cdk8"``).
        **overrides
            Any non-GMM config fields to override (e.g. ``learning_rate``,
            ``n_epochs``).

        Returns
        -------
        MixtureVIConfig
        """
        preset = get_preset(name)
        return cls(
            likelihood_gmm=preset.likelihood,
            prior_gmm=preset.prior,
            **overrides,
        )

    @classmethod
    def from_user_data(
        cls,
        edge_residuals: "np.ndarray",
        node_residuals: "np.ndarray",
        K_likelihood: Optional[int] = None,
        K_prior: Optional[int] = None,
        random_state: Optional[int] = None,
        *,
        k_values: Optional[List[int]] = None,
        n_splits: int = 5,
        **overrides,
    ) -> "MixtureVIConfig":
        """Fit GMMs from the user's own experimental residuals.

        Use this only when you have ground-truth experimental data — typically
        from a prior FEP map on the same protein system whose residuals can be
        used to correct a new map.

        Parameters
        ----------
        edge_residuals : np.ndarray
            1-D array of (predicted_DDG − experimental_DDG) per edge.
        node_residuals : np.ndarray
            1-D array of zero-centered (Pred. DG − Exp. DG) per node.
        K_likelihood : int, optional
            Number of GMM components for the likelihood (≥ 1). If ``None``
            (default), the number of components is chosen from the data by
            cross-validated log-likelihood (:func:`fit_gmm_cv`).
        K_prior : int, optional
            Number of GMM components for the prior (≥ 1). If ``None`` (default),
            chosen from the data the same way as ``K_likelihood``.
        random_state : int, optional
            Seed forwarded to sklearn's GaussianMixture for reproducibility.
        k_values : list of int, optional
            Candidate component counts to evaluate when a ``K`` is ``None``.
            Defaults to :func:`fit_gmm_cv`'s own default (``[1, …, 10]``).
        n_splits : int
            Number of cross-validation folds (``>= 2``) used during
            data-driven selection (only used when a ``K`` is ``None``).
        **overrides
            Any other MixtureVIConfig fields (e.g. ``learning_rate``). Passing
            ``random_seed`` here is accepted as an alias for ``random_state``.

        Returns
        -------
        MixtureVIConfig

        Notes
        -----
        The selected component counts are reflected in the returned config:
        inspect ``config.likelihood_gmm.n_components`` /
        ``config.prior_gmm.n_components``.
        """
        # Local import to avoid a hard dependency on sklearn at module load.
        from .probabilistic.mixture.fitting import fit_gmm_cv, fit_sklearn_gmm

        # ``random_seed`` is the config-level field name; accept it as an alias
        # for ``random_state`` so callers can bundle it into a shared overrides
        # dict without triggering a duplicate-keyword clash below. An explicit
        # ``random_state`` argument takes precedence.
        if "random_seed" in overrides:
            seed_alias = overrides.pop("random_seed")
            if random_state is None:
                random_state = seed_alias

        def _fit(data, k):
            if k is None:
                params, _ = fit_gmm_cv(
                    data,
                    k_values=k_values,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                return params
            return fit_sklearn_gmm(data, n_components=k, random_state=random_state)

        likelihood = _fit(edge_residuals, K_likelihood)
        prior = _fit(node_residuals, K_prior)
        return cls(
            likelihood_gmm=likelihood,
            prior_gmm=prior,
            random_seed=random_state,
            **overrides,
        )

    @classmethod
    def from_dataset(
        cls,
        dataset,
        K_likelihood: Optional[int] = None,
        K_prior: Optional[int] = None,
        random_state: Optional[int] = None,
        *,
        k_values: Optional[List[int]] = None,
        n_splits: int = 5,
        **overrides,
    ) -> "MixtureVIConfig":
        """Fit GMMs in-sample from a dataset's own experimental residuals.

        The most direct "data-driven" entry point: it reads the dataset's
        predicted and experimental values, forms the edge residuals
        (predicted − experimental ΔΔG) and the mean-centered node ΔG
        distribution, and fits the likelihood and prior GMMs. Leave a ``K`` as
        ``None`` to let the number of mixture components be chosen from the data
        by cross-validated log-likelihood — i.e. let the data tell you how many
        mixtures it needs.

        Requires experimental ground truth in the dataset; use only for
        in-sample benchmark analysis or when correcting a follow-up map for a
        system already characterized experimentally. With no ground truth, use
        the default :class:`MixtureVIConfig` (OpenFE preset) instead.

        Parameters
        ----------
        dataset : BaseDataset
            Dataset exposing ``dataset_edges`` / ``dataset_nodes`` with both
            predicted and experimental columns (see
            :func:`~maple_fep.models.probabilistic.mixture.fitting.residuals_from_dataset`).
        K_likelihood, K_prior : int, optional
            Component counts; ``None`` (default) selects from the data.
        random_state : int, optional
            Seed for reproducible fits.
        k_values : list of int, optional
            Candidate component counts for data-driven selection.
        n_splits : int
            Number of cross-validation folds (``>= 2``) for data-driven selection.
        **overrides
            Any other MixtureVIConfig fields.

        Returns
        -------
        MixtureVIConfig
        """
        from .probabilistic.mixture.fitting import residuals_from_dataset

        edge_residuals, node_residuals = residuals_from_dataset(dataset)
        return cls.from_user_data(
            edge_residuals,
            node_residuals,
            K_likelihood=K_likelihood,
            K_prior=K_prior,
            random_state=random_state,
            k_values=k_values,
            n_splits=n_splits,
            **overrides,
        )


class GaussianMixtureVIConfig(BaseEstimatorConfig):
    """Configuration specific to GaussianMixtureVI.

    .. deprecated::
        Use :class:`MixtureVIConfig` instead. This class will be removed in a
        future release.
    """

    model_type: str = Field(default="GaussianMixtureVI", frozen=True)

    prior_std: float = Field(
        default=5.0,
        gt=0.0,
        description="Prior standard deviation for node values (sigma_0)",
    )

    normal_std: float = Field(
        default=1.0, gt=0.0, description="Standard deviation for normal edges (sigma_1)"
    )

    outlier_std: float = Field(
        default=3.0,
        gt=0.0,
        description="Standard deviation for outlier edges (sigma_2)",
    )

    outlier_prob: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Global probability of an edge being an outlier (pi)",
    )

    kl_weight: float = Field(
        default=0.1, ge=0.0, description="Weight for KL divergence term in ELBO"
    )

    n_epochs: int = Field(
        default=1000, ge=10, description="Maximum number of training epochs"
    )

    n_samples: int = Field(
        default=100,
        ge=1,
        description="Number of Monte Carlo samples for ELBO estimation",
    )

    patience: int = Field(default=50, ge=1, description="Early stopping patience")

    # Override base class fields to match GMVI conventions
    learning_rate: float = Field(
        default=0.01, ge=1e-6, le=1.0, description="Learning rate for ADAM optimizer"
    )

    random_seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility. If None, no seed is set.",
    )

    @field_validator("outlier_prob")
    @classmethod
    def validate_outlier_prob(cls, v):
        """Ensure outlier probability is valid"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Outlier probability must be between 0 and 1")
        return v

    @field_validator("normal_std", "outlier_std")
    @classmethod
    def validate_std_relationship(cls, v, info):
        """Optionally validate relationship between normal and outlier stds"""
        return v

    def __init__(self, **data):
        warnings.warn(
            "GaussianMixtureVIConfig is deprecated. Use MixtureVIConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class CycleClosureCorrectionConfig(BaseModel):
    """Configuration specific to CycleClosureCorrection (Weighted Cycle Closure)"""

    model_type: str = Field(default="CycleClosureCorrection", frozen=True)

    tolerance: float = Field(
        default=1e-6,
        gt=0.0,
        description="Convergence tolerance for cycle closure iteration",
    )

    max_iterations: int = Field(
        default=1000, ge=1, description="Maximum number of cycle closure iterations"
    )

    use_weights: bool = Field(
        default=True,
        description="Whether to use edge weights (inverse variance) in correction",
    )

    max_cycle_length: int = Field(
        default=8,
        ge=3,
        description="Maximum cycle length to detect (larger = more cycles but slower)",
    )

    max_cycles: int = Field(
        default=100000,
        ge=1,
        description="Maximum number of cycles to find before stopping",
    )

    cycle_detection_timeout: float = Field(
        default=300.0,
        gt=0.0,
        description="Timeout in seconds for cycle detection (default: 5 minutes)",
    )

    @field_validator("tolerance")
    @classmethod
    def validate_tolerance(cls, v):
        """Ensure tolerance is reasonable"""
        if v <= 0:
            raise ValueError("Tolerance must be positive")
        if v > 1.0:
            raise ValueError("Tolerance too large (should be << 1)")
        return v

    @field_validator("max_cycle_length")
    @classmethod
    def validate_max_cycle_length(cls, v):
        """Warn if cycle length is too large"""
        if v > 12:
            warnings.warn(
                f"max_cycle_length={v} may be very slow for dense graphs. "
                "Consider using smaller values (6-8) for initial testing.",
                UserWarning,
            )
        return v


class SpectralCorrectionConfig(BaseModel):
    """Configuration specific to SpectralCorrection (Weighted Spectral Free-energy Correction)"""

    model_type: str = Field(default="SpectralCorrection", frozen=True)

    use_weights: bool = Field(
        default=True,
        description="If True (WSFC), use per-edge 1/sigma^2 weights derived from "
        "DeltaDeltaG Error. If False (SFC), use uniform weights (W=I).",
    )


def create_config(
    model_type: str = "VariationalEstimator", **kwargs
) -> Union[BaseEstimatorConfig, CycleClosureCorrectionConfig, SpectralCorrectionConfig]:
    """
    Factory function to create appropriate config based on model type.

    Parameters
    ----------
    model_type : str
        Type of estimator ("VariationalEstimator", "GaussianMixtureVI",
        "CycleClosureCorrection", or "SpectralCorrection")
    **kwargs
        Configuration parameters for the specific estimator

    Returns
    -------
    BaseEstimatorConfig or CycleClosureCorrectionConfig or SpectralCorrectionConfig
        Appropriate configuration object for the estimator type

    Examples
    --------
    >>> config = create_config("VariationalEstimator", learning_rate=0.001)
    >>> gmvi_config = create_config("GaussianMixtureVI", prior_std=5.0, outlier_prob=0.3)
    >>> wcc_config = create_config("CycleClosureCorrection", tolerance=1e-6, max_iterations=1000)
    >>> sfc_config = create_config("SpectralCorrection", use_weights=False)
    """
    normalized = model_type.upper().replace("_", "")
    if normalized in ("VARIATIONALESTIMATOR",):
        return VariationalEstimatorConfig(**kwargs)
    elif normalized in ("MIXTUREVI",):
        return MixtureVIConfig(**kwargs)
    elif normalized in ("GAUSSIANMIXTUREVI",):
        return GaussianMixtureVIConfig(**kwargs)
    elif normalized in ("CYCLECLOSURECORRECTION",):
        return CycleClosureCorrectionConfig(**kwargs)
    elif normalized in ("SPECTRALCORRECTION",):
        return SpectralCorrectionConfig(**kwargs)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Use 'VariationalEstimator', 'MixtureVI', 'GaussianMixtureVI', "
            "'CycleClosureCorrection', or 'SpectralCorrection'"
        )
