"""
Unified Mixture Variational Inference Model (PyTorch).

This module provides ``MixtureVI``, a pure-PyTorch ``nn.Module`` that
implements full-rank Gaussian Markov Variational Inference for FEP networks.
Both the likelihood and the prior are K-component Gaussian Mixture Models;
the GMM parameters come from a frozen reference preset (OpenFE by default)
or from a user-supplied custom fit (see :meth:`MixtureVIConfig.from_user_data`).

KL divergence is computed by sampling.

The deprecated ``GaussianMixtureVI`` name is kept as an alias.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ....dataset import BaseDataset
from ...config import MixtureVIConfig
from ...graph_data import GraphData
from ...base import BaseEstimator


class MixtureVI(nn.Module, BaseEstimator):
    """Unified Mixture VI estimator for FEP networks.

    Parameters
    ----------
    config : MixtureVIConfig
        Model configuration.
    dataset : BaseDataset
        Dataset containing the FEP data.
    """

    def __init__(self, config: MixtureVIConfig, dataset: BaseDataset):
        nn.Module.__init__(self)
        self.config = config
        self.dataset = dataset
        self.graph_data: Optional[GraphData] = None
        self.node_estimates: Optional[Dict[str, float]] = None
        self.edge_estimates: Optional[Dict[Tuple[str, str], float]] = None
        self.node_uncertainties: Optional[Dict[str, float]] = None
        self.edge_uncertainties: Optional[Dict[Tuple[str, str], float]] = None

        # GMM likelihood params (always present)
        gmm = config.likelihood_gmm
        self.gmm_means = torch.tensor(gmm.means, dtype=torch.float32)
        self.gmm_variances = torch.tensor(gmm.variances, dtype=torch.float32)
        self.gmm_stds = torch.sqrt(self.gmm_variances)
        self.gmm_weights = torch.tensor(gmm.weights, dtype=torch.float32)
        self.n_lik_components = gmm.n_components

        # GMM prior params (always present)
        prior = config.prior_gmm
        self.prior_means = torch.tensor(prior.means, dtype=torch.float32)
        self.prior_vars = torch.tensor(prior.variances, dtype=torch.float32)
        self.prior_weights = torch.tensor(prior.weights, dtype=torch.float32)
        self.n_prior_components = prior.n_components

        # Variational parameters (initialized in initialize_parameters)
        self.node_means = None
        self.node_cholesky = None

        # Training history
        self.loss_history: List[float] = []
        self.elbo_history: List[float] = []
        self.expected_log_likelihood_history: List[float] = []
        self.kl_div_history: List[float] = []

    # ------------------------------------------------------------------
    # Graph data extraction (0-indexed, no Pyro 1-indexing)
    # ------------------------------------------------------------------

    def _extract_graph_data(self) -> GraphData:
        edges_df = getattr(self.dataset, "dataset_edges", None)
        if edges_df is None:
            raise ValueError("Dataset must have 'dataset_edges' attribute")

        source_nodes = []
        target_nodes = []
        edge_values = []
        node_to_idx: Dict[str, int] = {}
        idx_to_node: Dict[int, str] = {}
        idx = 0

        for _, row in edges_df.iterrows():
            if "Source" in edges_df.columns and "Destination" in edges_df.columns:
                ligand1, ligand2 = row["Source"], row["Destination"]
            elif "Ligand1" in edges_df.columns and "Ligand2" in edges_df.columns:
                ligand1, ligand2 = row["Ligand1"], row["Ligand2"]
            else:
                raise ValueError(
                    "Edge data must have either 'Source'/'Destination' "
                    "or 'Ligand1'/'Ligand2' columns"
                )

            if "DeltaDeltaG" in edges_df.columns:
                fep_value = float(row["DeltaDeltaG"])
            elif "DeltaG" in edges_df.columns:
                fep_value = float(row["DeltaG"])
            elif "FEP" in edges_df.columns:
                fep_value = float(row["FEP"])
            else:
                raise ValueError(
                    "Edge data must have 'DeltaDeltaG', 'DeltaG', or 'FEP' column"
                )

            if ligand1 not in node_to_idx:
                node_to_idx[ligand1] = idx
                idx_to_node[idx] = ligand1
                idx += 1
            if ligand2 not in node_to_idx:
                node_to_idx[ligand2] = idx
                idx_to_node[idx] = ligand2
                idx += 1

            source_nodes.append(node_to_idx[ligand1])
            target_nodes.append(node_to_idx[ligand2])
            edge_values.append(fep_value)

        self.graph_data = GraphData(
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            edge_values=edge_values,
            num_nodes=len(node_to_idx),
            num_edges=len(edge_values),
            node_to_idx=node_to_idx,
            idx_to_node=idx_to_node,
        )
        return self.graph_data

    # ------------------------------------------------------------------
    # Parameter initialization
    # ------------------------------------------------------------------

    def initialize_parameters(self) -> None:
        """Initialize variational parameters (mean vector + Cholesky factor)."""
        n = self.graph_data.num_nodes
        self.node_means = nn.Parameter(torch.zeros(n))

        diag_vals = torch.ones(n) * 0.1
        off_diag_vals = torch.ones(n * (n - 1) // 2) * 0.01

        L = torch.zeros(n, n)
        torch.diagonal(L)[:] = diag_vals
        k = 0
        for i in range(n):
            for j in range(i):
                L[i, j] = off_diag_vals[k]
                k += 1

        self.node_cholesky = nn.Parameter(L)
        self.register_buffer("tril_mask", torch.tril(torch.ones(n, n)))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _get_cholesky(self) -> torch.Tensor:
        """Return the constrained lower-triangular Cholesky factor."""
        L = self.node_cholesky * self.tril_mask
        diag = torch.diagonal(L, 0)
        return L - torch.diag(diag) + torch.diag(F.softplus(diag) + 1e-4)

    def sample_nodes(self, n_samples: Optional[int] = None) -> torch.Tensor:
        """Sample from the variational distribution q(z)."""
        if n_samples is None:
            n_samples = self.config.n_samples
        eps = torch.randn(n_samples, self.graph_data.num_nodes)
        L = self._get_cholesky()
        return self.node_means.unsqueeze(0) + eps @ L.T

    # ------------------------------------------------------------------
    # Edge predictions
    # ------------------------------------------------------------------

    def compute_edge_predictions(self, node_samples: torch.Tensor) -> torch.Tensor:
        """Compute z_j - z_i for all edges."""
        src = torch.tensor(self.graph_data.source_nodes, dtype=torch.long)
        dst = torch.tensor(self.graph_data.target_nodes, dtype=torch.long)
        return node_samples[:, dst] - node_samples[:, src]

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def compute_likelihood(self, node_samples: torch.Tensor) -> torch.Tensor:
        """Compute log p(x|z) under the K-component GMM likelihood."""
        edge_preds = self.compute_edge_predictions(node_samples)
        obs = torch.tensor(self.graph_data.edge_values, dtype=torch.float32)
        residuals = obs.unsqueeze(0) - edge_preds  # (S, E)

        log_components = []
        for k in range(self.n_lik_components):
            shifted = residuals - self.gmm_means[k]
            log_ck = -0.5 * (shifted / self.gmm_stds[k]) ** 2 - torch.log(
                self.gmm_stds[k] * torch.sqrt(2 * torch.tensor(torch.pi))
            )
            log_components.append(torch.log(self.gmm_weights[k]) + log_ck)
        return torch.logsumexp(torch.stack(log_components), dim=0)

    # ------------------------------------------------------------------
    # KL divergence
    # ------------------------------------------------------------------

    def compute_kl_divergence(
        self, node_samples: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute KL[q || p] by sampling (prior is a K-component GMM)."""
        assert (
            node_samples is not None
        ), "MixtureVI requires Monte Carlo samples for the KL term"
        log_q = self._q_logpdf(node_samples)
        log_p = self._p_logpdf(node_samples)
        return (log_q - log_p).mean()

    def _q_logpdf(self, x: torch.Tensor) -> torch.Tensor:
        """log q(x) for the full-rank Gaussian variational distribution."""
        L = self._get_cholesky()
        q = torch.distributions.MultivariateNormal(loc=self.node_means, scale_tril=L)
        return q.log_prob(x)

    def _p_logpdf(self, x: torch.Tensor) -> torch.Tensor:
        """log p(x) for the GMM prior.

        Each node value is an independent draw from the same 1-D K-component
        GMM prior, so ``p(x) = ∏_i Σ_k π_k N(x_i | μ_k, σ_k²)`` — the mixture
        is taken per node, then the log-densities are summed over nodes.
        """
        K = self.prior_means.shape[0]
        diff = x.unsqueeze(2) - self.prior_means.view(1, 1, K)  # (S, N, K)
        log_norm = -0.5 * torch.log(2 * torch.pi * self.prior_vars)  # (K,)
        log_N = log_norm.view(1, 1, K) - 0.5 * diff**2 / self.prior_vars.view(
            1, 1, K
        )  # (S, N, K)
        log_pis = torch.log(self.prior_weights)  # (K,)
        log_p_node = torch.logsumexp(log_N + log_pis.view(1, 1, K), dim=2)  # (S, N)
        return log_p_node.sum(dim=1)  # (S,)

    # ------------------------------------------------------------------
    # ELBO
    # ------------------------------------------------------------------

    def compute_elbo(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ELBO = E[log p(x|z)] - kl_weight * KL."""
        node_samples = self.sample_nodes()
        log_lik = self.compute_likelihood(node_samples)
        expected_ll = torch.mean(torch.sum(log_lik, dim=1))

        kl = self.compute_kl_divergence(node_samples)
        elbo = expected_ll - self.config.kl_weight * kl
        return elbo, expected_ll, kl

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Fit the model by maximizing the ELBO.

        Results are accessible via ``node_estimates``, ``edge_estimates``,
        ``get_results()``, and ``add_predictions_to_dataset()`` after this call.
        """
        # Extract graph
        self._extract_graph_data()

        # Seed
        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        # Init variational params
        self.initialize_parameters()

        # Reset histories
        self.loss_history = []
        self.elbo_history = []
        self.expected_log_likelihood_history = []
        self.kl_div_history = []

        optimizer = optim.Adam(
            [self.node_means, self.node_cholesky],
            lr=self.config.learning_rate,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=20
        )

        best_elbo = float("-inf")
        patience_counter = 0

        for epoch in range(self.config.n_epochs):
            optimizer.zero_grad()
            elbo, exp_ll, kl = self.compute_elbo()
            loss = -elbo
            loss.backward()
            optimizer.step()
            scheduler.step(elbo)

            self.loss_history.append(loss.item())
            self.elbo_history.append(elbo.item())
            self.expected_log_likelihood_history.append(exp_ll.item())
            self.kl_div_history.append(kl.item())

            if elbo > best_elbo:
                best_elbo = elbo
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.patience:
                break

        # Extract posterior
        self._compute_posterior_estimates()

    # ------------------------------------------------------------------
    # Posterior extraction
    # ------------------------------------------------------------------

    def _compute_posterior_estimates(self) -> None:
        """Draw 1000 samples and store node/edge estimates + uncertainties."""
        with torch.no_grad():
            samples = self.sample_nodes(n_samples=1000)
        means = samples.mean(dim=0).numpy()
        stds = samples.std(dim=0).numpy()

        self.node_estimates = {
            self.graph_data.idx_to_node[i]: float(means[i])
            for i in range(self.graph_data.num_nodes)
        }
        self.node_uncertainties = {
            self.graph_data.idx_to_node[i]: float(stds[i])
            for i in range(self.graph_data.num_nodes)
        }

        self.edge_estimates = {}
        self.edge_uncertainties = {}
        edges_df = getattr(self.dataset, "dataset_edges", None)
        if edges_df is not None:
            for _, row in edges_df.iterrows():
                if "Source" in edges_df.columns and "Destination" in edges_df.columns:
                    l1, l2 = row["Source"], row["Destination"]
                elif "Ligand1" in edges_df.columns and "Ligand2" in edges_df.columns:
                    l1, l2 = row["Ligand1"], row["Ligand2"]
                else:
                    continue
                if l1 in self.node_estimates and l2 in self.node_estimates:
                    key = (l1, l2)
                    self.edge_estimates[key] = (
                        self.node_estimates[l2] - self.node_estimates[l1]
                    )
                    self.edge_uncertainties[key] = float(
                        np.sqrt(
                            self.node_uncertainties[l1] ** 2
                            + self.node_uncertainties[l2] ** 2
                        )
                    )

    # ------------------------------------------------------------------
    # Results API (BaseEstimator interface)
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[str, Any]:
        if self.node_estimates is None:
            raise ValueError("Model must be fitted before getting results")

        results = {
            "node_estimates": self.node_estimates,
            "edge_estimates": self.edge_estimates,
            "config": self.config.model_dump(),
            "num_nodes": self.graph_data.num_nodes if self.graph_data else 0,
            "num_edges": self.graph_data.num_edges if self.graph_data else 0,
        }
        if self.loss_history:
            results["losses"] = self.loss_history
            results["final_loss"] = self.loss_history[-1]
        if self.node_uncertainties is not None:
            results["node_uncertainties"] = self.node_uncertainties
        if self.edge_uncertainties is not None:
            results["edge_uncertainties"] = self.edge_uncertainties
        return results

    # ------------------------------------------------------------------
    # Column suffix logic
    # ------------------------------------------------------------------

    @property
    def _suffix(self) -> str:
        """Column suffix: VI(K_prior, K_lik)."""
        return f"VI({self.n_prior_components},{self.n_lik_components})"

    # ------------------------------------------------------------------
    # Dataset integration
    # ------------------------------------------------------------------

    def add_predictions_to_dataset(self) -> None:
        if self.node_estimates is None:
            raise ValueError(
                "Model must be fitted before adding predictions to dataset"
            )

        suffix = self._suffix

        nodes_df = getattr(self.dataset, "dataset_nodes", None)
        if nodes_df is not None:
            node_preds = []
            node_uncs = []
            for ligand in nodes_df["Name"]:
                if ligand in self.node_estimates:
                    node_preds.append(self.node_estimates[ligand])
                    node_uncs.append(
                        self.node_uncertainties.get(ligand, np.nan)
                        if self.node_uncertainties
                        else np.nan
                    )
                else:
                    node_preds.append(np.nan)
                    node_uncs.append(np.nan)
            nodes_df[suffix] = node_preds
            if self.node_uncertainties is not None:
                nodes_df[f"{suffix}_uncertainty"] = node_uncs

        edges_df = getattr(self.dataset, "dataset_edges", None)
        if edges_df is not None:
            edge_preds = []
            edge_uncs = []
            for _, row in edges_df.iterrows():
                if "Source" in edges_df.columns and "Destination" in edges_df.columns:
                    key = (row["Source"], row["Destination"])
                elif "Ligand1" in edges_df.columns and "Ligand2" in edges_df.columns:
                    key = (row["Ligand1"], row["Ligand2"])
                else:
                    raise ValueError(
                        "Edge data must have 'Source'/'Destination' "
                        "or 'Ligand1'/'Ligand2' columns"
                    )
                if key in self.edge_estimates:
                    edge_preds.append(self.edge_estimates[key])
                    edge_uncs.append(
                        self.edge_uncertainties.get(key, np.nan)
                        if self.edge_uncertainties
                        else np.nan
                    )
                else:
                    edge_preds.append(np.nan)
                    edge_uncs.append(np.nan)
            edges_df[suffix] = edge_preds
            if self.edge_uncertainties is not None:
                edges_df[f"{suffix}_uncertainty"] = edge_uncs

        if hasattr(self.dataset, "estimators"):
            if suffix not in self.dataset.estimators:
                self.dataset.estimators.append(suffix)

    # ------------------------------------------------------------------
    # Component posteriors
    # ------------------------------------------------------------------

    def compute_edge_component_responsibilities(self) -> "np.ndarray":
        """Posterior P(component k | edge e) under the K-component likelihood.

        Returns an array of shape ``(num_edges, n_lik_components)``. The
        component with the largest variance is typically the "outlier" mode;
        you can identify it via ``np.argmax(self.gmm_variances.numpy())``.
        """
        if self.node_estimates is None or self.graph_data is None:
            raise ValueError(
                "Model must be fitted before computing component responsibilities"
            )

        with torch.no_grad():
            node_samples = self.sample_nodes(n_samples=1000)
            edge_preds = self.compute_edge_predictions(node_samples)
            obs = torch.tensor(self.graph_data.edge_values, dtype=torch.float32)
            residuals = obs.unsqueeze(0) - edge_preds  # (S, E)

            log_pis = torch.log(self.gmm_weights)  # (K,)
            shifted = residuals.unsqueeze(-1) - self.gmm_means  # (S, E, K)
            log_norm = -torch.log(
                self.gmm_stds * torch.sqrt(2 * torch.tensor(torch.pi))
            )  # (K,)
            log_c = log_norm - 0.5 * (shifted / self.gmm_stds) ** 2  # (S, E, K)
            log_unnorm = log_pis + log_c  # (S, E, K)
            log_resp = log_unnorm - torch.logsumexp(
                log_unnorm, dim=-1, keepdim=True
            )  # (S, E, K)

        return torch.exp(log_resp).mean(dim=0).numpy()  # (E, K)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_training_history(
        self,
        filename: Optional[str] = None,
        figsize: tuple = (14, 5),
        show_plot: bool = True,
    ):
        """Plot training loss and ELBO curves."""
        import matplotlib.pyplot as plt

        if not self.loss_history:
            raise RuntimeError("No training history. Call .fit() first.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        epochs = list(range(len(self.loss_history)))

        ax1.plot(epochs, self.loss_history, color="red", linewidth=2, label="Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(epochs, self.elbo_history, color="blue", linewidth=2, label="ELBO")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("ELBO")
        ax2.set_title("Evidence Lower Bound")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.suptitle(
            f"MixtureVI {self._suffix} Training History",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        if filename is not None:
            if not filename.endswith(".pdf"):
                filename = filename.rsplit(".", 1)[0] + ".pdf"
            plt.savefig(filename, format="pdf", dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        return fig
