"""
Unit tests for the MixtureVI (unified Mixture Variational Inference) model.

Both the likelihood and the prior are K-component Gaussian Mixture Models.
There are three construction paths for ``MixtureVIConfig``:

- ``MixtureVIConfig()`` — defaults to the OpenFE preset (K_lik=3, K_prior=2).
- ``MixtureVIConfig.from_preset(name)`` — load a named system-specific preset.
- ``MixtureVIConfig.from_user_data(edge_residuals, node_residuals, K_lik, K_prior)``
  — fit GMMs on the user's own residuals (only valid when they have
  experimental data).

Also tests the deprecated ``GaussianMixtureVI*`` aliases (warn but still work)
and the presets API.
"""

import warnings

import numpy as np
import pytest

from vine_fep.dataset import SyntheticFEPDataset
from vine_fep.models import (
    GMMParams,
    GaussianMixtureVI,
    GaussianMixtureVIConfig,
    MixtureVI,
    MixtureVIConfig,
    get_preset,
    list_presets,
    register_preset,
)


# ======================================================================
# MixtureVIConfig — construction paths
# ======================================================================


class TestMixtureVIConfigDefaults:
    """Bare ``MixtureVIConfig()`` should equal the OpenFE preset."""

    def test_default_uses_openfe_preset(self):
        config = MixtureVIConfig()
        openfe = get_preset("openfe")
        assert config.likelihood_gmm.n_components == openfe.likelihood.n_components
        assert config.prior_gmm.n_components == openfe.prior.n_components
        np.testing.assert_allclose(config.likelihood_gmm.means, openfe.likelihood.means)
        np.testing.assert_allclose(config.prior_gmm.means, openfe.prior.means)

    def test_default_training_hyperparams(self):
        config = MixtureVIConfig()
        assert config.kl_weight >= 0
        assert config.learning_rate > 0
        assert config.n_epochs > 0
        assert config.n_samples > 0
        assert config.patience > 0
        assert config.random_seed is None
        assert config.model_type == "MixtureVI"

    def test_model_type_immutable(self):
        config = MixtureVIConfig()
        with pytest.raises((ValueError, AttributeError)):
            config.model_type = "OTHER"


class TestMixtureVIConfigFromPreset:
    """All seven shipped presets should load with non-trivial GMMs."""

    @pytest.mark.parametrize(
        "name", ["openfe", "cdk8", "pfkfb3", "tnks2", "cmet", "eg5", "syk"]
    )
    def test_each_preset_loads(self, name):
        config = MixtureVIConfig.from_preset(name)
        assert config.likelihood_gmm.n_components >= 1
        assert config.prior_gmm.n_components >= 1

    def test_overrides_apply(self):
        config = MixtureVIConfig.from_preset("openfe", kl_weight=1.0, n_epochs=500)
        assert config.kl_weight == 1.0
        assert config.n_epochs == 500
        assert config.likelihood_gmm.n_components == 3  # preset value preserved

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError):
            MixtureVIConfig.from_preset("nonexistent")


class TestMixtureVIConfigFromUserData:
    """Custom-fit path for users with their own experimental residuals."""

    @staticmethod
    def _residuals(rng, K_lik=3, K_prior=2):
        # Bimodal-ish edges, centred-near-zero nodes
        edge_resid = np.concatenate(
            [
                rng.normal(loc=-1.0, scale=0.5, size=40),
                rng.normal(loc=1.0, scale=0.5, size=40),
            ]
        )
        node_resid = rng.normal(loc=0.0, scale=1.0, size=30)
        node_resid -= node_resid.mean()
        return edge_resid, node_resid

    def test_returns_requested_components(self):
        rng = np.random.default_rng(0)
        edge, node = self._residuals(rng)
        config = MixtureVIConfig.from_user_data(
            edge, node, K_likelihood=3, K_prior=2, random_state=42
        )
        assert config.likelihood_gmm.n_components == 3
        assert config.prior_gmm.n_components == 2

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(0)
        edge, node = self._residuals(rng)
        c1 = MixtureVIConfig.from_user_data(edge, node, 3, 2, random_state=7)
        c2 = MixtureVIConfig.from_user_data(edge, node, 3, 2, random_state=7)
        np.testing.assert_allclose(c1.likelihood_gmm.means, c2.likelihood_gmm.means)
        np.testing.assert_allclose(c1.prior_gmm.means, c2.prior_gmm.means)

    def test_overrides_apply(self):
        rng = np.random.default_rng(0)
        edge, node = self._residuals(rng)
        config = MixtureVIConfig.from_user_data(
            edge, node, 2, 2, random_state=1, kl_weight=0.5, n_epochs=200
        )
        assert config.kl_weight == 0.5
        assert config.n_epochs == 200


class TestMixtureVIConfigExplicitGMMs:
    """Users may also pass GMMParams directly."""

    def test_construct_with_explicit_gmms(self):
        gmm = GMMParams(means=[0.0, 1.0], variances=[0.5, 0.5], weights=[0.5, 0.5])
        config = MixtureVIConfig(likelihood_gmm=gmm, prior_gmm=gmm)
        assert config.likelihood_gmm.n_components == 2
        assert config.prior_gmm.n_components == 2


# ======================================================================
# Presets API
# ======================================================================


class TestPresetsAPI:
    def test_list_presets(self):
        presets = list_presets()
        assert set(presets) >= {
            "openfe",
            "cdk8",
            "pfkfb3",
            "tnks2",
            "cmet",
            "eg5",
            "syk",
        }

    def test_get_preset(self):
        preset = get_preset("openfe")
        assert preset.likelihood.n_components == 3
        assert preset.prior.n_components == 3

    def test_get_preset_case_insensitive(self):
        assert get_preset("OpenFE").likelihood.n_components == 3

    def test_get_preset_unknown(self):
        with pytest.raises(KeyError):
            get_preset("nonexistent")

    def test_register_preset(self):
        from vine_fep.models import DatasetPreset

        gmm = GMMParams(means=[0.0], variances=[1.0], weights=[1.0])
        register_preset(
            "custom_test", DatasetPreset(likelihood=gmm, prior=gmm, description="test")
        )
        assert "custom_test" in list_presets()
        retrieved = get_preset("custom_test")
        assert retrieved.likelihood.n_components == 1
        assert retrieved.prior.n_components == 1


# ======================================================================
# MixtureVI initialization
# ======================================================================


class TestMixtureVIInitialization:
    def test_default_uses_openfe(self, mock_dataset):
        model = MixtureVI(MixtureVIConfig(), mock_dataset)
        assert model.n_lik_components == 3
        assert model.n_prior_components == 3

    def test_init_from_preset(self, mock_dataset):
        config = MixtureVIConfig.from_preset("openfe")
        model = MixtureVI(config, mock_dataset)
        assert model.gmm_means is not None
        assert model.prior_means is not None

    def test_estimates_none_before_training(self, mock_dataset):
        model = MixtureVI(MixtureVIConfig(), mock_dataset)
        assert model.node_estimates is None
        assert model.edge_estimates is None
        assert model.node_uncertainties is None
        assert model.edge_uncertainties is None


# ======================================================================
# Training
# ======================================================================


class TestMixtureVITraining:
    def test_fit_default_preset(self, mock_dataset):
        config = MixtureVIConfig(n_epochs=50, n_samples=20)
        model = MixtureVI(config, mock_dataset)
        model.fit()
        result = model.get_results()
        assert "node_estimates" in result
        assert "edge_estimates" in result
        assert "node_uncertainties" in result
        assert "edge_uncertainties" in result

    def test_loss_decreases(self, mock_dataset):
        config = MixtureVIConfig(n_epochs=100, n_samples=20)
        model = MixtureVI(config, mock_dataset)
        model.fit()
        assert model.loss_history[-1] < model.loss_history[0]

    def test_uncertainties_positive(self, mock_dataset):
        config = MixtureVIConfig(n_epochs=50, n_samples=20)
        model = MixtureVI(config, mock_dataset)
        model.fit()
        for val in model.node_uncertainties.values():
            assert val > 0

    def test_loss_history_finite(self, mock_dataset):
        config = MixtureVIConfig(n_epochs=50, n_samples=20)
        model = MixtureVI(config, mock_dataset)
        model.fit()
        result = model.get_results()
        for loss in result["losses"]:
            assert np.isfinite(loss)

    def test_reproducibility_with_seed(self, mock_dataset):
        config = MixtureVIConfig(n_epochs=50, n_samples=20, random_seed=42)
        m1 = MixtureVI(config, mock_dataset)
        m1.fit()
        m2 = MixtureVI(config, mock_dataset)
        m2.fit()
        for k in m1.node_estimates:
            assert abs(m1.node_estimates[k] - m2.node_estimates[k]) < 1e-4

    def test_suffix_matches_components(self, mock_dataset):
        # Default = OpenFE preset = K_prior=3, K_lik=3
        config = MixtureVIConfig(n_epochs=50, n_samples=20)
        model = MixtureVI(config, mock_dataset)
        model.fit()
        assert model._suffix == "VI(3,3)"


# ======================================================================
# Results API
# ======================================================================


class TestMixtureVIResults:
    def test_get_results_before_training_raises(self, mock_dataset):
        model = MixtureVI(MixtureVIConfig(), mock_dataset)
        with pytest.raises(ValueError):
            model.get_results()


# ======================================================================
# Component responsibilities (replaces 2-component outlier probs)
# ======================================================================


class TestComponentResponsibilities:
    def test_shape_and_normalization(self, mock_dataset):
        config = MixtureVIConfig(n_epochs=50, n_samples=20)
        model = MixtureVI(config, mock_dataset)
        model.fit()
        resp = model.compute_edge_component_responsibilities()
        assert resp.shape == (model.graph_data.num_edges, model.n_lik_components)
        # Each row should sum to ~1
        np.testing.assert_allclose(resp.sum(axis=1), 1.0, atol=1e-5)
        # Each entry in [0, 1]
        assert (resp >= 0).all() and (resp <= 1).all()

    def test_before_training_raises(self, mock_dataset):
        model = MixtureVI(MixtureVIConfig(), mock_dataset)
        with pytest.raises(ValueError):
            model.compute_edge_component_responsibilities()


# ======================================================================
# Dataset integration
# ======================================================================


class TestMixtureVIDatasetIntegration:
    def test_add_predictions_default(self, mock_dataset):
        config = MixtureVIConfig(n_epochs=50, n_samples=20)
        model = MixtureVI(config, mock_dataset)
        model.fit()
        model.add_predictions_to_dataset()

        nodes_df = mock_dataset.dataset_nodes
        assert "VI(3,3)" in nodes_df.columns
        assert "VI(3,3)_uncertainty" in nodes_df.columns

        edges_df = mock_dataset.dataset_edges
        assert "VI(3,3)" in edges_df.columns
        assert "VI(3,3)_uncertainty" in edges_df.columns

    def test_predictions_before_train_raises(self, mock_dataset):
        model = MixtureVI(MixtureVIConfig(), mock_dataset)
        with pytest.raises(ValueError):
            model.add_predictions_to_dataset()


# ======================================================================
# Synthetic dataset end-to-end
# ======================================================================


class TestMixtureVIWithSyntheticDataset:
    def test_default_workflow(self):
        dataset = SyntheticFEPDataset(add_noise=True)
        config = MixtureVIConfig(n_epochs=50, n_samples=20, random_seed=123)
        model = MixtureVI(config, dataset)
        model.fit()

        results = model.get_results()
        assert "final_loss" in results
        assert "node_estimates" in results
        assert "node_uncertainties" in results

        resp = model.compute_edge_component_responsibilities()
        assert resp.shape[0] > 0

        model.add_predictions_to_dataset()
        assert "VI(3,3)" in dataset.dataset_nodes.columns

    def test_preset_workflow(self):
        dataset = SyntheticFEPDataset(add_noise=True)
        config = MixtureVIConfig.from_preset("cdk8", n_epochs=50, n_samples=20)
        model = MixtureVI(config, dataset)
        model.fit()
        model.add_predictions_to_dataset()
        # cdk8 preset is K_lik=3, K_prior=2
        assert "VI(2,3)" in dataset.dataset_nodes.columns


# ======================================================================
# Deprecation warnings
# ======================================================================


class TestDeprecationWarnings:
    def test_gaussian_mixture_vi_config_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GaussianMixtureVIConfig()
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_gaussian_mixture_vi_class_warns(self, mock_dataset):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = GaussianMixtureVIConfig()
            GaussianMixtureVI(config=config, dataset=mock_dataset)
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_deprecated_class_still_runs(self, mock_dataset):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            config = GaussianMixtureVIConfig(n_epochs=50, n_samples=20)
            model = GaussianMixtureVI(config=config, dataset=mock_dataset)
            model.fit()
            assert model.node_estimates is not None
            # Bridge config now lands in OpenFE preset → suffix VI(3,3)
            assert model._suffix == "VI(3,3)"
