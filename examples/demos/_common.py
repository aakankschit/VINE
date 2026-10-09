"""Shared helpers for the VINE manuscript demos.

Every demo notebook in ``examples/demos/`` imports from this module so that
styling, model hyperparameters, metric computation and figure saving are
defined in exactly one place. The settings below are the ones used to make
the manuscript figures; change them here only if you deliberately want
results that differ from the paper.
"""

from __future__ import annotations

import contextlib
import io
import random
import re
import warnings
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Matplotlib backend: when a demo runs as a plain script (``python 0X_*.py``)
# there is no display, and some vine_fep plotting helpers call ``plt.show()``.
# Switching to Agg makes those calls no-ops. Inside ``marimo edit`` marimo
# provides its own backend, so leave it alone there.
# ---------------------------------------------------------------------------
try:
    import marimo as _mo

    _IN_MARIMO = _mo.running_in_notebook()
except Exception:  # noqa: BLE001
    _IN_MARIMO = False
if not _IN_MARIMO:
    matplotlib.use("Agg")
warnings.filterwarnings("ignore", message=".*non-interactive.*cannot be shown")
warnings.filterwarnings("ignore", message=".*FigureCanvasAgg is non-interactive.*")

import matplotlib.pyplot as plt  # noqa: E402
import pyro  # noqa: E402

from vine_fep.dataset import FEPBenchmarkDataset, FEPDataset  # noqa: E402
from vine_fep.graph_analysis import bootstrap_statistic  # noqa: E402
from vine_fep.models import (  # noqa: E402
    GMMParams,
    GuideType,
    MixtureVI,
    MixtureVIConfig,
    PriorType,
    VariationalEstimator,
    VariationalEstimatorConfig,
)

# ---------------------------------------------------------------------------
# Experiment constants (manuscript settings)
# ---------------------------------------------------------------------------
DATASET_NAMES: List[str] = ["cdk8", "cmet", "eg5", "syk", "tnks2", "pfkfb3"]
SAMPLING_TIME = "5ns"

# Synthetic-outlier protocol: add NOISE_LEVEL kcal/mol to N_OUTLIER_EDGES
# randomly chosen edges per dataset.
NOISE_SEED = 19
NOISE_LEVEL = 20.0
N_OUTLIER_EDGES = 4

NBOOTSTRAP = 1000
DPI = 300

NODE_MODELS: List[str] = [
    "MLE",
    "MAP",
    "VI(1,1)",
    "VI(1,2)",
    "VI(1,3)",
    "VI(2,3)",
    "VI(3,3)",
]
# Edge comparisons also include CCC (cycle-closure-corrected ΔΔG shipped with
# the benchmark edge data).
EDGE_MODELS: List[str] = NODE_MODELS + ["CCC"]

MODEL_COLORS: Dict[str, str] = {
    "MLE": "#9467bd",
    "MAP": "#1f77b4",
    "VI(1,1)": "#2ca02c",
    "VI(1,2)": "#17becf",
    "VI(1,3)": "#bcbd22",
    "VI(2,3)": "#ff7f0e",
    "VI(3,3)": "#e377c2",
    "CCC": "#d62728",
}
DATASET_COLORS_PALETTE = "deep"

# Display name for the proprietary-partner map. The data files keep their
# original ``treeline_dataset_*.csv`` names; everything user-visible says
# "Industry".
INDUSTRY_LABEL = "Industry"


# ---------------------------------------------------------------------------
# Paths and figure output
# ---------------------------------------------------------------------------
def setup_style() -> None:
    """Force a light theme so figures render the same in any notebook theme."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.titlecolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )


def repo_root(notebook_dir: Path) -> Path:
    """Return the repository root given the ``examples/demos`` directory."""
    return Path(notebook_dir).resolve().parents[1]


def data_dir(notebook_dir: Path) -> Path:
    """Return ``examples/data`` given the ``examples/demos`` directory."""
    return Path(notebook_dir).resolve().parent / "data"


def figure_dir(notebook_dir: Path, demo_name: str) -> Path:
    """Create and return ``<repo>/figures/<demo_name>/``."""
    out = repo_root(notebook_dir) / "figures" / demo_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_figure(fig, out_dir: Path, stem: str, dpi: int = DPI) -> List[Path]:
    """Save ``fig`` as ``<stem>.pdf`` and ``<stem>.png`` in ``out_dir``."""
    paths = []
    for ext in ("pdf", "png"):
        path = Path(out_dir) / f"{stem}.{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    print(f"  saved {Path(out_dir).name}/{stem}.{{pdf,png}}")
    return paths


def slug(label: str) -> str:
    """Filesystem-safe version of a model label, e.g. ``VI(1,2)`` -> ``VI_1_2``."""
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_benchmark(
    names: Sequence[str] = DATASET_NAMES, sampling_time: str = SAMPLING_TIME
) -> Dict[str, FEPDataset]:
    """Load the Schindler benchmark maps (auto-downloaded and cached)."""
    benchmark = FEPBenchmarkDataset()
    out = {}
    for name in names:
        edges, nodes = benchmark.get_dataset(name, sampling_time)
        out[name] = FEPDataset(dataset_edges=edges, dataset_nodes=nodes)
    return out


def load_noisy_benchmark(
    names: Sequence[str] = DATASET_NAMES,
    sampling_time: str = SAMPLING_TIME,
    seed: int = NOISE_SEED,
    noise_level: float = NOISE_LEVEL,
    n_outlier_edges: int = N_OUTLIER_EDGES,
) -> Dict[str, FEPDataset]:
    """Load the benchmark maps and add synthetic outlier edges.

    ``noise_level`` kcal/mol is added to ``n_outlier_edges`` randomly chosen
    edges of each map; the edges are flagged in a ``noise_added`` column. The
    edge draw always runs over the full six-dataset list in the same order, so
    the injected outliers in a given map do not depend on which subset of
    ``names`` is requested.
    """
    benchmark = FEPBenchmarkDataset()
    random.seed(seed)
    np.random.seed(seed)
    out = {}
    order = list(DATASET_NAMES) + [n for n in names if n not in DATASET_NAMES]
    for name in order:
        edges, nodes = benchmark.get_dataset(name, sampling_time)
        edges = edges.copy()
        edges["noise_added"] = False
        idxs = np.random.choice(edges.index, size=n_outlier_edges, replace=False)
        for i in idxs:
            edges.loc[i, "DeltaDeltaG"] += noise_level
            edges.loc[i, "noise_added"] = True
        if name in names:
            out[name] = FEPDataset(dataset_edges=edges, dataset_nodes=nodes)
    return {n: out[n] for n in names}


def load_industry(data_path: Path) -> FEPDataset:
    """Load the Industry map from ``examples/data/treeline_dataset_*.csv``."""
    data_path = Path(data_path)
    return FEPDataset(
        dataset_edges=pd.read_csv(data_path / "treeline_dataset_edges.csv"),
        dataset_nodes=pd.read_csv(data_path / "treeline_dataset_nodes.csv"),
    )


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
Factory = Callable[[FEPDataset], object]


def _scale(full: int, quick: bool, quick_value: int) -> int:
    return quick_value if quick else full


def benchmark_factories(quick: bool = False) -> List[Tuple[str, Factory]]:
    """The seven estimators used for the Schindler benchmark figures.

    - MLE: uniform prior wide enough never to clip the optimum, AutoDelta.
    - MAP: N(0, 1) prior, AutoDelta.
    - VI(1,1): N(0, 1) prior, AutoNormal (mean-field VI).
    - VI(1,2): fixed normal+outlier likelihood (σ = 1 / 3, p_outlier = 0.2),
      single-Gaussian prior. This is the manuscript's original "GMVI".
    - VI(1,3), VI(2,3), VI(3,3): 3-component likelihood with a 1-, 2- or
      3-component prior, both fit in-sample on the map's own residuals.

    ``quick=True`` shortens training for a fast smoke run; the figures will
    then not match the manuscript exactly.
    """
    mix_kw = dict(
        kl_weight=0.1,  # 1.0 over-regularizes every mixture model
        n_epochs=_scale(5000, quick, 1000),
        n_samples=_scale(700, quick, 200),
        patience=300,
        learning_rate=0.001,
        random_seed=42,
    )

    def make_mle(dataset):
        cfg = VariationalEstimatorConfig(
            prior_type=PriorType.UNIFORM,
            prior_parameters=[-100.0, 100.0],
            guide_type=GuideType.AUTO_DELTA,
            num_steps=_scale(10000, quick, 3000),
            patience=300,
            learning_rate=0.001,
            error_std=1.0,
            random_seed=42,
        )
        return VariationalEstimator(config=cfg, dataset=dataset)

    def make_vi(dataset):
        cfg = VariationalEstimatorConfig(
            prior_type=PriorType.NORMAL,
            prior_parameters=[0.0, 1.0],
            guide_type=GuideType.AUTO_NORMAL,
            num_steps=_scale(100000, quick, 3000),
            patience=700,
            learning_rate=0.001,
            error_std=1.0,
            random_seed=42,
        )
        return VariationalEstimator(config=cfg, dataset=dataset)

    def make_vi_1_2(dataset):
        cfg = MixtureVIConfig(
            likelihood_gmm=GMMParams(
                means=[0.0, 0.0], variances=[1.0, 9.0], weights=[0.8, 0.2]
            ),
            prior_gmm=GMMParams(means=[0.0], variances=[1.0], weights=[1.0]),
            **mix_kw,
        )
        return MixtureVI(config=cfg, dataset=dataset)

    def make_fit(k_prior):
        def _make(dataset):
            cfg = MixtureVIConfig.from_dataset(
                dataset, K_likelihood=3, K_prior=k_prior, **mix_kw
            )
            return MixtureVI(config=cfg, dataset=dataset)

        return _make

    return [
        ("MLE", make_mle),
        ("MAP", map_factory(1.0, quick=quick)),
        ("VI(1,1)", make_vi),
        ("VI(1,2)", make_vi_1_2),
        ("VI(1,3)", make_fit(1)),
        ("VI(2,3)", make_fit(2)),
        ("VI(3,3)", make_fit(3)),
    ]


def map_factory(prior_std: float = 1.0, quick: bool = False) -> Factory:
    """MAP estimator with a N(0, prior_std²) prior (used by the σ₀ sweep too)."""

    def make_map(dataset):
        cfg = VariationalEstimatorConfig(
            prior_type=PriorType.NORMAL,
            prior_parameters=[0.0, float(prior_std)],
            guide_type=GuideType.AUTO_DELTA,
            num_steps=_scale(100000, quick, 3000),
            patience=300,
            learning_rate=0.001,
            error_std=1.0,
            random_seed=42,
        )
        return VariationalEstimator(config=cfg, dataset=dataset)

    return make_map


def industry_factories(quick: bool = False) -> List[Tuple[str, Factory]]:
    """The seven estimators with the Industry-map hyperparameters.

    These follow the original Industry analysis: longer training, and VI(1,2)
    uses a wider outlier component (σ = 5, p_outlier = 0.28) because this map
    contains edges that are tens of kcal/mol off.
    """
    mix_kw = dict(
        kl_weight=0.1,
        n_epochs=_scale(10000, quick, 1500),
        n_samples=_scale(1000, quick, 200),
        patience=500,
        learning_rate=0.001,
        random_seed=42,
    )

    def _ve(prior_type, params, guide, steps):
        def _make(dataset):
            cfg = VariationalEstimatorConfig(
                prior_type=prior_type,
                prior_parameters=params,
                guide_type=guide,
                num_steps=steps,
                learning_rate=0.001,
                error_std=1.0,
                random_seed=42,
            )
            return VariationalEstimator(config=cfg, dataset=dataset)

        return _make

    def make_vi_1_2(dataset):
        cfg = MixtureVIConfig(
            likelihood_gmm=GMMParams(
                means=[0.0, 0.0], variances=[1.0, 25.0], weights=[0.72, 0.28]
            ),
            prior_gmm=GMMParams(means=[0.0], variances=[1.0], weights=[1.0]),
            **mix_kw,
        )
        return MixtureVI(config=cfg, dataset=dataset)

    def make_fit(k_prior):
        def _make(dataset):
            cfg = MixtureVIConfig.from_dataset(
                dataset, K_likelihood=3, K_prior=k_prior, **mix_kw
            )
            return MixtureVI(config=cfg, dataset=dataset)

        return _make

    return [
        (
            "MLE",
            _ve(
                PriorType.UNIFORM,
                [-100.0, 100.0],
                GuideType.AUTO_DELTA,
                _scale(40000, quick, 5000),
            ),
        ),
        (
            "MAP",
            _ve(
                PriorType.NORMAL,
                [0.0, 1.0],
                GuideType.AUTO_DELTA,
                _scale(30000, quick, 5000),
            ),
        ),
        (
            "VI(1,1)",
            _ve(
                PriorType.NORMAL,
                [0.0, 1.0],
                GuideType.AUTO_NORMAL,
                _scale(30000, quick, 5000),
            ),
        ),
        ("VI(1,2)", make_vi_1_2),
        ("VI(1,3)", make_fit(1)),
        ("VI(2,3)", make_fit(2)),
        ("VI(3,3)", make_fit(3)),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# The AUTO_NORMAL VariationalEstimator writes a column named "VI"; the demos
# display and score it as "VI(1,1)". MixtureVI variants already write their
# own "VI(K_prior,K_lik)" label.
_RAW_COLUMN = {"VI(1,1)": "VI"}


def _reconcile_label(df: pd.DataFrame, label: str) -> None:
    raw = _RAW_COLUMN.get(label)
    if raw is None:
        return
    for src, dst in ((raw, label), (f"{raw}_uncertainty", f"{label}_uncertainty")):
        if src in df.columns:
            if dst in df.columns:
                del df[dst]
            df.rename(columns={src: dst}, inplace=True)


def train_model(
    dataset: FEPDataset, label: str, factory: Factory, verbose: bool = False
):
    """Fit one estimator from scratch and write its predictions into ``dataset``.

    The Pyro parameter store is cleared before every fit so no state leaks
    between models. Per-step training logs are suppressed unless ``verbose``.
    """
    pyro.clear_param_store()
    model = factory(dataset)
    sink = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(
        io.StringIO()
    )
    with sink:
        model.fit()
        model.add_predictions_to_dataset()
    _reconcile_label(dataset.dataset_nodes, label)
    _reconcile_label(dataset.dataset_edges, label)
    return model


def train_all(
    datasets: Dict[str, FEPDataset],
    factories: Iterable[Tuple[str, Factory]],
    tag: str = "",
) -> List[Tuple[str, str]]:
    """Train every factory on every dataset (in place). Returns a log."""
    factories = list(factories)
    log = []
    for ds_name, dataset in datasets.items():
        for label, factory in factories:
            train_model(dataset, label, factory)
            log.append((ds_name, label))
            print(f"  [{tag}] {ds_name:8s} {label:8s} trained")
    return log


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def centered_bootstrap(
    exp: np.ndarray, pred: np.ndarray, metric: str, nbootstrap: int = NBOOTSTRAP
) -> dict:
    """Bootstrap ``metric`` on mean-centered experimental vs predicted values."""
    exp = np.asarray(exp, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(exp) | np.isnan(pred))
    exp_c = exp[mask] - np.mean(exp[mask])
    pred_c = pred[mask] - np.mean(pred[mask])
    return bootstrap_statistic(
        exp_c,
        pred_c,
        statistic=metric,
        nbootstrap=nbootstrap,
        include_true_uncertainty=False,
        include_pred_uncertainty=False,
    )


def compute_metrics(
    datasets: Dict[str, FEPDataset],
    models: Sequence[str],
    level: str = "node",
    metrics: Sequence[str] = ("RMSE", "R2", "KTAU"),
    nbootstrap: int = NBOOTSTRAP,
) -> Dict[str, Dict[str, Dict[str, dict]]]:
    """Bootstrap metrics for every (dataset, model) pair.

    ``level="node"`` scores ΔG against ``Exp. DeltaG``; ``level="edge"``
    scores ΔΔG against ``Experimental DeltaDeltaG``. Both sides are
    mean-centered before scoring.
    """
    results: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for ds_name, dataset in datasets.items():
        results[ds_name] = {}
        if level == "edge":
            df = dataset.dataset_edges
            exp = df["Experimental DeltaDeltaG"].values
        else:
            df = dataset.dataset_nodes
            exp = df["Exp. DeltaG"].values
        exp_c = exp - np.nanmean(exp)
        for model in models:
            if model not in df.columns:
                continue
            pred = df[model].values
            mask = ~(np.isnan(exp_c) | np.isnan(pred))
            if mask.sum() < 3:
                continue
            pred_c = pred[mask] - np.nanmean(pred[mask])
            results[ds_name][model] = {}
            for metric in metrics:
                r = bootstrap_statistic(
                    exp_c[mask],
                    pred_c,
                    statistic=metric,
                    nbootstrap=nbootstrap,
                    include_true_uncertainty=False,
                    include_pred_uncertainty=False,
                )
                results[ds_name][model][metric] = {
                    "mean": r["mean"],
                    "ci_low": r["low"],
                    "ci_high": r["high"],
                }
    return results


def metrics_table(results: dict, metric: str = "RMSE") -> pd.DataFrame:
    """Flatten ``compute_metrics`` output into a dataset × model table of means."""
    rows = {}
    for ds_name, per_model in results.items():
        rows[ds_name] = {
            m: round(v[metric]["mean"], 3) for m, v in per_model.items() if metric in v
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
_YLABELS = {
    "RMSE": "RMSE (kcal/mol)",
    "R2": "R²",
    "KTAU": "Kendall's τ",
    "rho": "Pearson ρ",
}


def grouped_bar_plot(
    results: dict,
    models: Sequence[str],
    metric: str,
    title: str,
    ylim: Optional[Tuple[float, float]] = None,
):
    """All-dataset grouped bar chart with 95% bootstrap CIs (one bar per model)."""
    datasets_list = sorted(results.keys())
    fig, ax = plt.subplots(figsize=(12, 6))
    n_models = len(models)
    bar_width = 0.8 / n_models
    x_pos = np.arange(len(datasets_list))
    for m_idx, model in enumerate(models):
        means, err_lo, err_hi = [], [], []
        for ds in datasets_list:
            cell = results.get(ds, {}).get(model)
            if cell is None or metric not in cell:
                means.append(0.0)
                err_lo.append(0.0)
                err_hi.append(0.0)
            else:
                means.append(cell[metric]["mean"])
                err_lo.append(cell[metric]["mean"] - cell[metric]["ci_low"])
                err_hi.append(cell[metric]["ci_high"] - cell[metric]["mean"])
        offset = (m_idx - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(
            x_pos + offset,
            means,
            bar_width,
            label=model,
            color=MODEL_COLORS.get(model, "#888888"),
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.errorbar(
            x_pos + offset,
            means,
            yerr=[err_lo, err_hi],
            fmt="none",
            ecolor="black",
            capsize=3,
            alpha=0.7,
        )
        for bar, mean in zip(bars, means):
            if abs(mean) > 1e-6:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.02,
                    f"{mean:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    rotation=90,
                )
    ax.set_xlabel("Dataset", fontsize=20)
    ax.set_ylabel(_YLABELS.get(metric, metric), fontsize=20)
    ax.set_title(title, fontsize=22, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.upper() for d in datasets_list], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=13, title="Model", title_fontsize=14, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    fig.tight_layout()
    return fig


def metric_panel(
    results: dict, models: Sequence[str], ds_name: str, suptitle: str
):
    """Two-panel (RMSE | R²) bar chart for one dataset, all methods.

    Y-limits cover the full 95% CI extent and always include zero, so no
    whisker is clipped.
    """
    cell = results.get(ds_name, {})
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, metric, ylabel in (
        (axes[0], "RMSE", "RMSE (kcal/mol)"),
        (axes[1], "R2", "R²"),
    ):
        xs, means, lo, hi, cols = [], [], [], [], []
        for i, m in enumerate(models):
            mc = cell.get(m, {}).get(metric)
            if mc is None:
                continue
            xs.append(i)
            means.append(mc["mean"])
            lo.append(mc["mean"] - mc["ci_low"])
            hi.append(mc["ci_high"] - mc["mean"])
            cols.append(MODEL_COLORS.get(m, "#888888"))
        ax.bar(xs, means, color=cols, edgecolor="black", alpha=0.9)
        if xs:
            ax.errorbar(
                xs,
                means,
                yerr=[lo, hi],
                fmt="none",
                ecolor="black",
                capsize=5,
                linewidth=1.5,
            )
            bot = min(0.0, min(v - e for v, e in zip(means, lo)))
            top = max(v + e for v, e in zip(means, hi))
            span = (top - bot) or 1.0
            ax.set_ylim(bot - 0.08 * span, top + 0.12 * span)
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylabel(ylabel, fontsize=22)
        ax.set_title("R²" if metric == "R2" else metric, fontsize=22, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(suptitle, fontsize=24, fontweight="bold")
    fig.tight_layout()
    return fig


def inference_effects_figure(dataset: FEPDataset, model: str, target_name: str):
    """Quiver plot of how ``model`` shifts the raw FEP edges toward experiment.

    Wraps ``GraphCycleAnalysis.visualize_inference_effects`` and returns the
    figure without saving it (the caller saves with :func:`save_figure`).
    """
    from vine_fep.graph_analysis import GraphCycleAnalysis, GraphSetup

    gca = GraphCycleAnalysis(GraphSetup(dataset))
    with contextlib.redirect_stdout(io.StringIO()):
        fig = gca.visualize_inference_effects(
            target_name=target_name,
            baseline_attribute="DeltaDeltaG",
            comparison_attribute=model,
        )
    return fig
