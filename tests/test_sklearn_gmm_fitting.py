"""
Tests for the sklearn-based GMM fitting utility.

Tests cover fit_sklearn_gmm, select_gmm_components, fit_gmm_cv,
and GMMSelectionResult.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from vine_fep.models import GMMParams
from vine_fep.models.config import MixtureVIConfig
from vine_fep.utils import residuals_from_dataset
from vine_fep.utils import (
    GMMSelectionResult,
    fit_gmm_cv,
    fit_sklearn_gmm,
    select_gmm_components,
)


# ---------------------------------------------------------------------------
# Shared synthetic data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data():
    """1-D data from a single Gaussian."""
    rng = np.random.RandomState(42)
    return rng.normal(0, 1, size=200)


@pytest.fixture
def mixture_data():
    """Well-separated 3-component mixture for K-recovery tests."""
    rng = np.random.RandomState(42)
    c1 = rng.normal(-6, 0.5, size=200)
    c2 = rng.normal(0, 0.5, size=200)
    c3 = rng.normal(6, 0.5, size=200)
    return np.concatenate([c1, c2, c3])


# ---------------------------------------------------------------------------
# TestFitSklearnGMM
# ---------------------------------------------------------------------------


class TestFitSklearnGMM:
    """Tests for fit_sklearn_gmm()."""

    def test_returns_gmm_params(self, simple_data):
        result = fit_sklearn_gmm(simple_data, n_components=2, random_state=42)
        assert isinstance(result, GMMParams)

    def test_correct_n_components(self, simple_data):
        for k in [1, 2, 3]:
            result = fit_sklearn_gmm(simple_data, n_components=k, random_state=42)
            assert result.n_components == k

    def test_weights_sum_to_one(self, simple_data):
        result = fit_sklearn_gmm(simple_data, n_components=3, random_state=42)
        assert abs(sum(result.weights) - 1.0) < 1e-6

    def test_variances_positive(self, simple_data):
        result = fit_sklearn_gmm(simple_data, n_components=3, random_state=42)
        assert all(v > 0 for v in result.variances)

    def test_single_component(self, simple_data):
        result = fit_sklearn_gmm(simple_data, n_components=1, random_state=42)
        assert result.n_components == 1
        assert isinstance(result.means, list)
        assert len(result.means) == 1

    def test_reproducibility_with_seed(self, simple_data):
        r1 = fit_sklearn_gmm(simple_data, n_components=2, random_state=42)
        r2 = fit_sklearn_gmm(simple_data, n_components=2, random_state=42)
        np.testing.assert_allclose(r1.means, r2.means)
        np.testing.assert_allclose(r1.variances, r2.variances)
        np.testing.assert_allclose(r1.weights, r2.weights)

    def test_plugs_into_mixture_vi_config(self, simple_data):
        params = fit_sklearn_gmm(simple_data, n_components=2, random_state=42)
        config = MixtureVIConfig(likelihood_gmm=params)
        assert config.likelihood_gmm is params

        config2 = MixtureVIConfig(prior_gmm=params)
        assert config2.prior_gmm is params

    def test_1d_array_accepted(self, simple_data):
        result = fit_sklearn_gmm(simple_data, n_components=2, random_state=42)
        assert isinstance(result, GMMParams)

    def test_2d_column_array_accepted(self, simple_data):
        data_2d = simple_data.reshape(-1, 1)
        result = fit_sklearn_gmm(data_2d, n_components=2, random_state=42)
        assert isinstance(result, GMMParams)

    def test_rejects_empty_array(self):
        with pytest.raises(ValueError, match="empty"):
            fit_sklearn_gmm(np.array([]), n_components=2)

    def test_rejects_nan_input(self):
        data = np.array([1.0, 2.0, np.nan, 4.0])
        with pytest.raises(ValueError, match="NaN"):
            fit_sklearn_gmm(data, n_components=2)

    def test_rejects_inf_input(self):
        data = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.raises(ValueError, match="infinite"):
            fit_sklearn_gmm(data, n_components=2)

    def test_rejects_zero_components(self, simple_data):
        with pytest.raises(ValueError, match="n_components must be >= 1"):
            fit_sklearn_gmm(simple_data, n_components=0)

    def test_rejects_k_exceeding_samples(self):
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="exceeds"):
            fit_sklearn_gmm(data, n_components=10)


# ---------------------------------------------------------------------------
# TestSelectGMMComponents
# ---------------------------------------------------------------------------


class TestSelectGMMComponents:
    """Tests for select_gmm_components()."""

    def test_returns_selection_result(self, simple_data):
        result = select_gmm_components(simple_data, k_values=[2, 3], random_state=42)
        assert isinstance(result, GMMSelectionResult)

    def test_best_k_in_k_values(self, simple_data):
        k_values = [2, 3, 4]
        result = select_gmm_components(simple_data, k_values=k_values, random_state=42)
        assert result.best_k in k_values

    def test_cv_scores_length_matches_k_values(self, simple_data):
        k_values = [2, 3, 4, 5]
        result = select_gmm_components(simple_data, k_values=k_values, random_state=42)
        assert len(result.cv_scores) == len(k_values)
        assert result.k_values == k_values

    def test_default_k_values(self, simple_data):
        result = select_gmm_components(simple_data, random_state=42)
        assert result.k_values == list(range(1, 11))

    def test_custom_k_values(self, simple_data):
        k_values = [2, 5, 10]
        result = select_gmm_components(simple_data, k_values=k_values, random_state=42)
        assert result.k_values == k_values

    def test_single_k_value(self, simple_data):
        result = select_gmm_components(simple_data, k_values=[3], random_state=42)
        assert result.best_k == 3

    def test_recovers_true_k_on_well_separated_data(self, mixture_data):
        result = select_gmm_components(
            mixture_data, k_values=[2, 3, 4, 5], random_state=42
        )
        assert result.best_k == 3

    def test_reproducibility(self, simple_data):
        r1 = select_gmm_components(simple_data, k_values=[2, 3], random_state=42)
        r2 = select_gmm_components(simple_data, k_values=[2, 3], random_state=42)
        assert r1.best_k == r2.best_k
        np.testing.assert_allclose(r1.cv_scores, r2.cv_scores)

    def test_n_splits_greater_than_one(self, simple_data):
        result = select_gmm_components(
            simple_data, k_values=[2, 3], n_splits=3, random_state=42
        )
        assert isinstance(result, GMMSelectionResult)
        assert result.best_k in [2, 3]

    def test_best_score_matches_best_k(self, simple_data):
        result = select_gmm_components(simple_data, k_values=[2, 3, 4], random_state=42)
        best_idx = result.k_values.index(result.best_k)
        assert result.best_score == result.cv_scores[best_idx]

    def test_warns_on_k_exceeding_samples(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            select_gmm_components(data, k_values=[2, 10], random_state=42)
            dep_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert any("exceeds" in str(x.message) for x in dep_warnings)


# ---------------------------------------------------------------------------
# TestFitGMMCV
# ---------------------------------------------------------------------------


class TestFitGMMCV:
    """Tests for fit_gmm_cv() convenience function."""

    def test_returns_tuple(self, simple_data):
        result = fit_gmm_cv(simple_data, k_values=[2, 3], random_state=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        params, selection = result
        assert isinstance(params, GMMParams)
        assert isinstance(selection, GMMSelectionResult)

    def test_gmm_params_match_best_k(self, simple_data):
        params, selection = fit_gmm_cv(simple_data, k_values=[2, 3, 4], random_state=42)
        assert params.n_components == selection.best_k

    def test_end_to_end_with_mixture_vi_config(self, simple_data):
        params, _ = fit_gmm_cv(simple_data, k_values=[2, 3], random_state=42)
        config = MixtureVIConfig(likelihood_gmm=params, prior_gmm=params)
        assert config.likelihood_gmm.n_components == params.n_components
        assert config.prior_gmm.n_components == params.n_components

    def test_recovers_true_k(self, mixture_data):
        params, selection = fit_gmm_cv(
            mixture_data, k_values=[2, 3, 4, 5], random_state=42
        )
        assert selection.best_k == 3
        assert params.n_components == 3


# ---------------------------------------------------------------------------
# TestGMMSelectionResult
# ---------------------------------------------------------------------------


class TestGMMSelectionResult:
    """Tests for the GMMSelectionResult dataclass."""

    def test_dataclass_fields(self):
        r = GMMSelectionResult(
            best_k=3,
            k_values=[2, 3, 4],
            cv_scores=[-2.0, -1.8, -1.9],
            best_score=-1.8,
        )
        assert r.best_k == 3
        assert r.k_values == [2, 3, 4]
        assert r.cv_scores == [-2.0, -1.8, -1.9]
        assert r.best_score == -1.8


# ---------------------------------------------------------------------------
# Toy dataset for residual-extraction / from_dataset tests
# ---------------------------------------------------------------------------


def _toy_dataset(seed=0, with_exp=True):
    """A minimal dataset-like object with predicted + experimental columns.

    Edge residuals (FEP − Exp) are bimodal by construction; node ΔG values
    form two well-separated clusters.
    """
    rng = np.random.RandomState(seed)
    n_edges = 80
    exp_edges = rng.normal(0.0, 1.0, n_edges)
    err = np.concatenate(
        [
            rng.normal(-1.0, 0.3, n_edges // 2),
            rng.normal(1.5, 0.3, n_edges - n_edges // 2),
        ]
    )
    edges = pd.DataFrame({"DeltaDeltaG": exp_edges + err})
    if with_exp:
        edges["Experimental DeltaDeltaG"] = exp_edges

    node_dg = np.concatenate(
        [
            rng.normal(-3.0, 0.5, 20),
            rng.normal(3.0, 0.5, 20),
        ]
    )
    nodes = pd.DataFrame(
        {"Name": [f"L{i}" for i in range(len(node_dg))], "Exp. DeltaG": node_dg}
    )
    return SimpleNamespace(dataset_edges=edges, dataset_nodes=nodes), err, node_dg


# ---------------------------------------------------------------------------
# TestResidualsFromDataset
# ---------------------------------------------------------------------------


class TestResidualsFromDataset:
    """Tests for residuals_from_dataset()."""

    def test_returns_two_arrays(self):
        ds, _, _ = _toy_dataset()
        edge_resid, node_resid = residuals_from_dataset(ds)
        assert isinstance(edge_resid, np.ndarray)
        assert isinstance(node_resid, np.ndarray)
        assert edge_resid.ndim == 1 and node_resid.ndim == 1

    def test_edge_residual_is_pred_minus_exp(self):
        ds, err, _ = _toy_dataset()
        edge_resid, _ = residuals_from_dataset(ds)
        # FEP − Exp == the injected error, by construction.
        np.testing.assert_allclose(np.sort(edge_resid), np.sort(err))

    def test_nodes_are_mean_centered(self):
        ds, _, _ = _toy_dataset()
        _, node_resid = residuals_from_dataset(ds)
        assert abs(float(node_resid.mean())) < 1e-9

    def test_nodes_uncentered_when_disabled(self):
        ds, _, node_dg = _toy_dataset()
        _, node_resid = residuals_from_dataset(ds, center_nodes=False)
        np.testing.assert_allclose(np.sort(node_resid), np.sort(node_dg))

    def test_drops_nonfinite(self):
        ds, _, _ = _toy_dataset()
        ds.dataset_edges.loc[0, "DeltaDeltaG"] = np.nan
        ds.dataset_nodes.loc[0, "Exp. DeltaG"] = np.inf
        edge_resid, node_resid = residuals_from_dataset(ds)
        assert np.all(np.isfinite(edge_resid))
        assert np.all(np.isfinite(node_resid))
        assert len(edge_resid) == len(ds.dataset_edges) - 1
        assert len(node_resid) == len(ds.dataset_nodes) - 1

    def test_raises_without_experimental_edge_column(self):
        ds, _, _ = _toy_dataset(with_exp=False)
        with pytest.raises(ValueError, match="experimental"):
            residuals_from_dataset(ds)

    def test_raises_without_node_dataframe(self):
        ds, _, _ = _toy_dataset()
        ds.dataset_nodes = None
        with pytest.raises(ValueError, match="dataset_edges` and `dataset_nodes"):
            residuals_from_dataset(ds)


# ---------------------------------------------------------------------------
# TestMixtureVIConfigDataDrivenK
# ---------------------------------------------------------------------------


class TestMixtureVIConfigDataDrivenK:
    """from_user_data / from_dataset with K selected from the data (K=None)."""

    def test_from_user_data_autoselects_when_k_none(self):
        rng = np.random.RandomState(1)
        edge = np.concatenate(
            [rng.normal(-6, 0.5, 200), rng.normal(0, 0.5, 200), rng.normal(6, 0.5, 200)]
        )
        node = rng.normal(0, 1, 100)
        config = MixtureVIConfig.from_user_data(
            edge,
            node,
            K_likelihood=None,
            K_prior=2,
            k_values=[2, 3, 4],
            random_state=42,
        )
        # Well-separated 3-cluster edges -> CV should pick 3 components.
        assert config.likelihood_gmm.n_components == 3
        assert config.prior_gmm.n_components == 2

    def test_from_user_data_autoselect_reproducible(self):
        rng = np.random.RandomState(2)
        edge = rng.normal(0, 1, 120)
        node = rng.normal(0, 1, 60)
        c1 = MixtureVIConfig.from_user_data(edge, node, k_values=[2, 3], random_state=7)
        c2 = MixtureVIConfig.from_user_data(edge, node, k_values=[2, 3], random_state=7)
        assert c1.likelihood_gmm.n_components == c2.likelihood_gmm.n_components
        np.testing.assert_allclose(c1.likelihood_gmm.means, c2.likelihood_gmm.means)

    def test_from_dataset_fixed_k(self):
        ds, _, _ = _toy_dataset()
        config = MixtureVIConfig.from_dataset(
            ds, K_likelihood=3, K_prior=1, random_state=42
        )
        assert config.likelihood_gmm.n_components == 3
        assert config.prior_gmm.n_components == 1
        assert config.model_type == "MixtureVI"

    def test_from_dataset_autoselect_k(self):
        ds, _, _ = _toy_dataset()
        config = MixtureVIConfig.from_dataset(
            ds,
            K_likelihood=None,
            K_prior=None,
            k_values=[2, 3, 4],
            random_state=42,
        )
        assert config.likelihood_gmm.n_components in (2, 3, 4)
        assert config.prior_gmm.n_components in (2, 3, 4)

    def test_from_dataset_passes_overrides(self):
        ds, _, _ = _toy_dataset()
        config = MixtureVIConfig.from_dataset(
            ds,
            K_likelihood=2,
            K_prior=1,
            random_state=0,
            kl_weight=1.0,
            n_epochs=250,
        )
        assert config.kl_weight == 1.0
        assert config.n_epochs == 250
