"""
Gaussian Mixture Model fitting with cross-validation.

Provides functions for fitting GMMs to 1-D residual data (node or edge)
and selecting the optimal number of components via held-out log-likelihood.
The returned :class:`~vine_fep.models.GMMParams` objects plug
directly into :class:`~vine_fep.models.config.MixtureVIConfig`.

Functions
---------
fit_sklearn_gmm
    Fit a GMM with a fixed K on all data.
select_gmm_components
    Select the best K via cross-validated log-likelihood.
fit_gmm_cv
    Convenience: select K then fit on all data in one call.
residuals_from_dataset
    Extract in-sample (edge, node) residuals from a dataset for GMM fitting.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import warnings

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold

from .presets import GMMParams


@dataclass
class GMMSelectionResult:
    """Results from cross-validated GMM component selection.

    Attributes
    ----------
    best_k : int
        Best number of components selected by cross-validation.
    k_values : List[int]
        All K values that were evaluated.
    cv_scores : List[float]
        Mean held-out log-likelihood for each K (same order as k_values).
    best_score : float
        The best (highest) mean held-out log-likelihood.
    """

    best_k: int
    k_values: List[int] = field(default_factory=list)
    cv_scores: List[float] = field(default_factory=list)
    best_score: float = float("-inf")


def _validate_data(data: np.ndarray) -> np.ndarray:
    """Validate and flatten input data to 1-D."""
    data = np.asarray(data, dtype=np.float64).flatten()
    if data.size == 0:
        raise ValueError("Input data is empty")
    if np.any(np.isnan(data)):
        raise ValueError("Input data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError("Input data contains infinite values")
    return data


def _extract_gmm_params(gmm: GaussianMixture) -> GMMParams:
    """Extract GMMParams from a fitted sklearn GaussianMixture."""
    means = np.atleast_1d(gmm.means_.squeeze())
    variances = np.atleast_1d(gmm.covariances_.squeeze())
    weights = np.atleast_1d(gmm.weights_)
    return GMMParams(
        means=means.tolist(),
        variances=variances.tolist(),
        weights=weights.tolist(),
    )


def fit_sklearn_gmm(
    data: np.ndarray,
    n_components: int,
    random_state: Optional[int] = None,
    max_iter: int = 1000,
    n_init: int = 5,
) -> GMMParams:
    """Fit a Gaussian Mixture Model to 1-D data using sklearn EM.

    Fits on the entire dataset and returns parameters that can be used
    directly as ``likelihood_gmm`` or ``prior_gmm`` in
    :class:`~vine_fep.models.config.MixtureVIConfig`.

    Parameters
    ----------
    data : np.ndarray
        1-D array of observations (e.g. zero-centered node residuals
        or edge residuals).
    n_components : int
        Number of mixture components.
    random_state : int, optional
        Random seed for reproducibility.
    max_iter : int
        Maximum EM iterations.
    n_init : int
        Number of EM initialisations (best is kept).

    Returns
    -------
    GMMParams
        Fitted means, variances, and weights.
    """
    data = _validate_data(data)
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if n_components > len(data):
        raise ValueError(
            f"n_components ({n_components}) exceeds number of samples ({len(data)})"
        )

    data_2d = data.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )
    gmm.fit(data_2d)
    return _extract_gmm_params(gmm)


def select_gmm_components(
    data: np.ndarray,
    k_values: Optional[List[int]] = None,
    n_splits: int = 5,
    random_state: Optional[int] = None,
    max_iter: int = 1000,
    n_init: int = 5,
) -> GMMSelectionResult:
    """Select the best number of GMM components via k-fold cross-validation.

    For each K in ``k_values``, runs :class:`sklearn.model_selection.KFold`
    (shuffled): fits a :class:`sklearn.mixture.GaussianMixture` on each training
    fold and scores the mean per-sample log-likelihood on the held-out fold.
    The score for K is the mean across folds; the K with the highest mean
    held-out log-likelihood is returned.

    Parameters
    ----------
    data : np.ndarray
        1-D array of observations.
    k_values : list of int, optional
        Component counts to evaluate. Defaults to ``[1, 2, …, 10]``.
    n_splits : int
        Number of cross-validation folds (``>= 2``). Defaults to 5.
    random_state : int, optional
        Random seed for reproducibility (used for the fold shuffle and each
        GaussianMixture fit).
    max_iter : int
        Maximum EM iterations per fit.
    n_init : int
        Number of EM initialisations per fit.

    Returns
    -------
    GMMSelectionResult
        Contains ``best_k``, all evaluated ``k_values``, and their
        cross-validated scores.
    """
    data = _validate_data(data)

    if k_values is None:
        k_values = list(range(1, 11))

    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for k-fold cross-validation")

    X = data.reshape(-1, 1)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores: List[float] = []

    for k in k_values:
        if k > len(data):
            warnings.warn(
                f"k={k} exceeds number of samples ({len(data)}), skipping",
                UserWarning,
                stacklevel=2,
            )
            cv_scores.append(float("-inf"))
            continue

        fold_scores: List[float] = []
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    random_state=random_state,
                    max_iter=max_iter,
                    n_init=n_init,
                )
                gmm.fit(X[train_idx])
                fold_scores.append(gmm.score(X[test_idx]))
            except Exception:
                warnings.warn(
                    f"GaussianMixture fitting failed for k={k}, fold={fold_idx}",
                    UserWarning,
                    stacklevel=2,
                )

        if fold_scores:
            cv_scores.append(float(np.mean(fold_scores)))
        else:
            cv_scores.append(float("-inf"))

    best_idx = int(np.argmax(cv_scores))
    return GMMSelectionResult(
        best_k=k_values[best_idx],
        k_values=list(k_values),
        cv_scores=cv_scores,
        best_score=cv_scores[best_idx],
    )


def fit_gmm_cv(
    data: np.ndarray,
    k_values: Optional[List[int]] = None,
    n_splits: int = 5,
    random_state: Optional[int] = None,
    max_iter: int = 1000,
    n_init: int = 5,
) -> Tuple[GMMParams, GMMSelectionResult]:
    """Select the best K via k-fold cross-validation, then fit on all data.

    Convenience function that combines :func:`select_gmm_components` and
    :func:`fit_sklearn_gmm`.  First determines the optimal number of
    components, then refits on the full dataset with that K.

    Parameters
    ----------
    data : np.ndarray
        1-D array of observations.
    k_values : list of int, optional
        Component counts to evaluate. Defaults to ``[1, 2, …, 10]``.
    n_splits : int
        Number of cross-validation folds (``>= 2``). Defaults to 5.
    random_state : int, optional
        Random seed for reproducibility.
    max_iter : int
        Maximum EM iterations.
    n_init : int
        Number of EM initialisations.

    Returns
    -------
    tuple of (GMMParams, GMMSelectionResult)
        The fitted ``GMMParams`` (on all data with best K) and the
        selection diagnostics.
    """
    selection = select_gmm_components(
        data,
        k_values=k_values,
        n_splits=n_splits,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )

    params = fit_sklearn_gmm(
        data,
        n_components=selection.best_k,
        random_state=random_state,
        max_iter=max_iter,
        n_init=n_init,
    )

    return params, selection


# ---------------------------------------------------------------------------
# Dataset-level convenience: turn a dataset into GMM-ready residuals
# ---------------------------------------------------------------------------

# Column names searched (in order) when pulling residuals out of a dataset.
_EDGE_PRED_COLS = ("DeltaDeltaG", "FEP")
_EDGE_EXP_COLS = ("Experimental DeltaDeltaG", "Exp.")
_NODE_EXP_COLS = ("Exp. DeltaG",)


def _first_present(df, candidates: Tuple[str, ...]) -> Optional[str]:
    """Return the first candidate column name present in ``df``, else None."""
    columns = getattr(df, "columns", [])
    for name in candidates:
        if name in columns:
            return name
    return None


def residuals_from_dataset(
    dataset, center_nodes: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract ``(edge_residuals, node_residuals)`` for in-sample GMM fitting.

    This is the convenience that turns a dataset directly into the two 1-D
    arrays a :class:`~vine_fep.models.config.MixtureVIConfig` mixture needs,
    so callers never have to remember which columns hold the predicted and
    experimental values:

    - ``edge_residuals`` = predicted ΔΔG − experimental ΔΔG, one per edge
      (the likelihood input — the FEP edge-error distribution).
    - ``node_residuals`` = experimental ΔG, mean-centered (the prior input —
      the distribution of absolute binding free energies; centering handles
      the gauge freedom).

    Non-finite (NaN/inf) entries are dropped independently from each array.

    Parameters
    ----------
    dataset : BaseDataset
        Any dataset exposing ``dataset_edges`` and ``dataset_nodes`` DataFrames.
        **Experimental ground truth is required** — both an experimental edge
        column (``Experimental DeltaDeltaG`` / ``Exp.``) and an experimental
        node column (``Exp. DeltaG``) must be present.
    center_nodes : bool
        If True (default) mean-center the node values before returning.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        ``(edge_residuals, node_residuals)``.

    Raises
    ------
    ValueError
        If the dataset lacks the edge/node DataFrames or the required
        predicted/experimental columns, or if no finite residuals remain.
    """
    edges = getattr(dataset, "dataset_edges", None)
    nodes = getattr(dataset, "dataset_nodes", None)
    if edges is None or nodes is None:
        raise ValueError(
            "dataset must expose `dataset_edges` and `dataset_nodes` DataFrames"
        )

    pred_col = _first_present(edges, _EDGE_PRED_COLS)
    exp_col = _first_present(edges, _EDGE_EXP_COLS)
    if pred_col is None or exp_col is None:
        raise ValueError(
            "Edge data needs a predicted column "
            f"({' / '.join(_EDGE_PRED_COLS)}) and an experimental column "
            f"({' / '.join(_EDGE_EXP_COLS)}) to compute residuals. In-sample "
            "fitting requires experimental ground truth — use the OpenFE "
            "preset (default MixtureVIConfig()) when none is available."
        )

    edge_resid = np.asarray(edges[pred_col], dtype=np.float64) - np.asarray(
        edges[exp_col], dtype=np.float64
    )
    edge_resid = edge_resid[np.isfinite(edge_resid)]

    node_col = _first_present(nodes, _NODE_EXP_COLS)
    if node_col is None:
        raise ValueError(
            "Node data needs an experimental column "
            f"({' / '.join(_NODE_EXP_COLS)}) to fit the prior."
        )
    node_resid = np.asarray(nodes[node_col], dtype=np.float64)
    node_resid = node_resid[np.isfinite(node_resid)]
    if center_nodes and node_resid.size:
        node_resid = node_resid - node_resid.mean()

    if edge_resid.size == 0 or node_resid.size == 0:
        raise ValueError("No finite residuals remain after dropping NaN/inf values.")
    return edge_resid, node_resid
