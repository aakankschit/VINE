"""
Pre-fitted GMM parameter presets for MixtureVI models.

This module provides dataclasses for GMM parameters and a registry of
pre-fitted presets from 7 benchmark FEP datasets. Users can also register
their own presets via ``register_preset()``.
"""

from dataclasses import dataclass, field
from typing import Dict, List


DEFAULT_PRESET = "openfe"


@dataclass
class GMMParams:
    """Parameters for a Gaussian Mixture Model.

    Parameters
    ----------
    means : List[float]
        Component means.
    variances : List[float]
        Component variances (σ²).
    weights : List[float]
        Component mixture weights (must sum to 1).
    """

    means: List[float]
    variances: List[float]
    weights: List[float]

    @property
    def n_components(self) -> int:
        """Number of mixture components."""
        return len(self.weights)

    def __post_init__(self):
        if not (len(self.means) == len(self.variances) == len(self.weights)):
            raise ValueError("means, variances, and weights must have the same length")


@dataclass
class DatasetPreset:
    """Bundles likelihood and prior GMM parameters for a dataset.

    Parameters
    ----------
    likelihood : GMMParams
        GMM parameters for the likelihood (error distribution).
    prior : GMMParams
        GMM parameters for the prior over node values.
    description : str
        Human-readable description of the dataset/preset.
    """

    likelihood: GMMParams
    prior: GMMParams
    description: str = ""


# ---------------------------------------------------------------------------
# Pre-fitted presets from MAPLE PR #1
# Likelihood params: MAPLE/src/maple/models/gaussian_markov_model_k_components.py
# Prior params: MAPLE/src/maple/models/gaussian_markov_model_gmm_both.py
# The OpenFE prior is the 3-component fit selected by k-fold cross-validation
# (see examples/fit_prior_gmm_openfe.ipynb), giving a VI(3,3) default model.
# ---------------------------------------------------------------------------

_PRESETS: Dict[str, DatasetPreset] = {
    "cdk8": DatasetPreset(
        likelihood=GMMParams(
            means=[-0.31578508, 2.23286293, -1.73513061],
            variances=[0.51183711, 0.38417805, 1.05073186],
            weights=[0.55242534, 0.21459543, 0.23297923],
        ),
        prior=GMMParams(
            means=[1.23704712, -0.88626907],
            variances=[0.68021314, 0.13787759],
            weights=[0.37844816, 0.62155184],
        ),
        description="CDK8 kinase benchmark dataset",
    ),
    "pfkfb3": DatasetPreset(
        likelihood=GMMParams(
            means=[-1.26681975, 0.42778768],
            variances=[0.88841351, 0.76856977],
            weights=[0.38589559, 0.61410441],
        ),
        prior=GMMParams(
            means=[-0.88935733, 1.03113603],
            variances=[0.33073825, 0.12993999],
            weights=[0.60167406, 0.39832594],
        ),
        description="PFKFB3 benchmark dataset",
    ),
    "tnks2": DatasetPreset(
        likelihood=GMMParams(
            means=[0.40202366, -2.25745204, 4.47],
            variances=[4.78160472e-01, 3.65668035e-02, 1.00000000e-06],
            weights=[0.87611446, 0.10305221, 0.02083333],
        ),
        prior=GMMParams(
            means=[-0.10657029, 0.53795106, -1.36587801],
            variances=[0.01044193, 0.55862123, 0.12852122],
            weights=[0.20815929, 0.48859241, 0.3032483],
        ),
        description="TNKS2 benchmark dataset",
    ),
    "cmet": DatasetPreset(
        likelihood=GMMParams(
            means=[1.16888523, -1.5114739, -0.24258218, -3.49069373],
            variances=[0.43612511, 0.24525734, 0.27272746, 0.04840062],
            weights=[0.29439822, 0.2202832, 0.441022, 0.04429658],
        ),
        prior=GMMParams(
            means=[-1.3259003, 1.51796586],
            variances=[1.04570346, 0.74677821],
            weights=[0.63619721, 0.36380279],
        ),
        description="CMET kinase benchmark dataset",
    ),
    "eg5": DatasetPreset(
        likelihood=GMMParams(
            means=[0.19925739, 0.22381401],
            variances=[3.51637547, 0.34405746],
            weights=[0.49988745, 0.50011255],
        ),
        prior=GMMParams(
            means=[-0.76612736, 1.21667198, 0.00939509],
            variances=[0.07792754, 0.54444327, 0.06793348],
            weights=[0.43564832, 0.27491748, 0.2894342],
        ),
        description="EG5 kinesin benchmark dataset",
    ),
    "syk": DatasetPreset(
        likelihood=GMMParams(
            means=[-0.01346477, 1.22145863, -1.08900642],
            variances=[0.59478066, 0.54845211, 1.2752598],
            weights=[0.40913483, 0.37144681, 0.21941836],
        ),
        prior=GMMParams(
            means=[-0.3695271, 0.50808779],
            variances=[0.48460983, 0.53944246],
            weights=[0.51647154, 0.48352846],
        ),
        description="SYK kinase benchmark dataset",
    ),
    "openfe": DatasetPreset(
        likelihood=GMMParams(
            means=[-0.11469188, -2.60908851, 2.15308002],
            variances=[1.34434458, 3.96742592, 4.90420322],
            weights=[0.6405804, 0.15375759, 0.20566201],
        ),
        prior=GMMParams(
            means=[-0.10361182, -1.88772899, 2.05033672],
            variances=[0.60573501, 0.72757168, 0.79523139],
            weights=[0.41412735, 0.29413621, 0.29173644],
        ),
        description="OpenFE benchmark dataset (recommended default)",
    ),
}


def get_preset(name: str) -> DatasetPreset:
    """Look up a pre-fitted dataset preset by name.

    Parameters
    ----------
    name : str
        Preset name (case-insensitive).

    Returns
    -------
    DatasetPreset

    Raises
    ------
    KeyError
        If the preset name is not found.
    """
    key = name.lower()
    if key not in _PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available presets: {list_presets()}")
    return _PRESETS[key]


def list_presets() -> List[str]:
    """Return the names of all registered presets."""
    return list(_PRESETS.keys())


def register_preset(name: str, preset: DatasetPreset) -> None:
    """Register a custom dataset preset.

    Parameters
    ----------
    name : str
        Name for the preset (will be lowercased).
    preset : DatasetPreset
        The preset to register.
    """
    _PRESETS[name.lower()] = preset
