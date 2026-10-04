"""
Deprecated aliases for the pre-restructure GMVI API.

``GaussianMixtureVI`` and ``GaussianMixtureVIConfig`` are kept here so existing
user code keeps working. They emit ``DeprecationWarning`` on construction and
will be removed in a future major release.

New code should use :class:`vine_fep.models.MixtureVI` and
:class:`vine_fep.models.MixtureVIConfig` directly.
"""

import warnings

from ....dataset import BaseDataset
from ...config import GaussianMixtureVIConfig, MixtureVIConfig
from .estimator import MixtureVI


def _make_mixture_vi_from_old_config(
    config: GaussianMixtureVIConfig, dataset: BaseDataset
) -> MixtureVI:
    """Bridge: build a :class:`MixtureVI` from the deprecated config.

    The legacy 2-component scalar fields (``prior_std``, ``normal_std``,
    ``outlier_std``, ``outlier_prob``) are ignored under the new
    GMM-everywhere design; the bridge config falls through to the OpenFE
    preset defaults on :class:`MixtureVIConfig`.
    """
    new_config = MixtureVIConfig(
        kl_weight=config.kl_weight,
        n_epochs=config.n_epochs,
        n_samples=config.n_samples,
        patience=config.patience,
        learning_rate=config.learning_rate,
        random_seed=config.random_seed,
    )
    return MixtureVI(config=new_config, dataset=dataset)


class GaussianMixtureVI(MixtureVI):
    """Deprecated alias for :class:`MixtureVI`.

    Accepts either ``MixtureVIConfig`` or the legacy
    ``GaussianMixtureVIConfig``.
    """

    def __init__(self, config, dataset: BaseDataset):
        if isinstance(config, GaussianMixtureVIConfig):
            warnings.warn(
                "GaussianMixtureVI is deprecated. Use MixtureVI with "
                "MixtureVIConfig instead. Legacy scalar fields "
                "(prior_std, normal_std, outlier_std, outlier_prob) are "
                "ignored under the new GMM-only design.",
                DeprecationWarning,
                stacklevel=2,
            )
            new_config = MixtureVIConfig(
                kl_weight=config.kl_weight,
                n_epochs=config.n_epochs,
                n_samples=config.n_samples,
                patience=config.patience,
                learning_rate=config.learning_rate,
                random_seed=config.random_seed,
            )
            super().__init__(config=new_config, dataset=dataset)
        else:
            super().__init__(config=config, dataset=dataset)
