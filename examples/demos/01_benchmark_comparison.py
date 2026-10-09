"""Demo 01: seven estimators on the six Schindler benchmark maps.

Run with ``marimo edit examples/demos/01_benchmark_comparison.py`` or
``python examples/demos/01_benchmark_comparison.py``.
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 01 · Benchmark comparison: seven estimators, six FEP maps

    This demo fits seven estimators to the six Schindler benchmark maps
    (CDK8, CMET, EG5, SYK, TNKS2, PFKFB3; 5 ns sampling) and scores the
    corrected node ΔG and edge ΔΔG against experiment with 95% bootstrap
    confidence intervals.

    | Label | Estimator | What it is |
    |---|---|---|
    | MLE | `VariationalEstimator` | wide uniform prior, AutoDelta guide |
    | MAP | `VariationalEstimator` | N(0, 1) prior, AutoDelta guide |
    | VI(1,1) | `VariationalEstimator` | N(0, 1) prior, mean-field AutoNormal guide |
    | VI(1,2) | `MixtureVI` | fixed normal + outlier likelihood (σ = 1 / 3, p = 0.2), Gaussian prior |
    | VI(1,3) | `MixtureVI` | 3-component likelihood, 1-component prior, fit in-sample |
    | VI(2,3) | `MixtureVI` | 3-component likelihood, 2-component prior, fit in-sample |
    | VI(3,3) | `MixtureVI` | 3-component likelihood, 3-component prior, fit in-sample |

    VI(*m*,*k*) means an *m*-component prior and a *k*-component likelihood.
    Edge plots also show CCC, the cycle-closure-corrected ΔΔG that ships with
    the benchmark data.

    **Manuscript figures reproduced**

    - `all_dataset_RMSE_node_comparison.pdf`: node RMSE, all datasets
    - `all_datasets_R2_node_comparison.pdf`: node R², all datasets

    **Supplementary outputs:** node Kendall's τ; edge RMSE, R² and Kendall's τ
    (with CCC); the raw TNKS2 edge-error distribution; a per-dataset panel.

    **Runtime:** about 7 minutes for the full run on a laptop CPU
    (42 model fits). Set `QUICK = True` below for a ~1 minute smoke run
    (2 datasets, shortened training, 200 bootstrap samples); its numbers will
    not match the paper.

    Figures are written to `figures/01_benchmark_comparison/` at the
    repository root, as PDF and PNG.
    """)
    return


@app.cell
def _(mo):
    import sys
    from pathlib import Path

    try:
        NB_DIR = Path(mo.notebook_dir() or Path(__file__).parent).resolve()
    except NameError:
        NB_DIR = Path.cwd()
    if not (NB_DIR / "_common.py").exists():
        NB_DIR = Path(__file__).parent.resolve()
    if str(NB_DIR) not in sys.path:
        sys.path.insert(0, str(NB_DIR))

    import _common as vc

    vc.setup_style()
    return NB_DIR, vc


@app.cell
def _(NB_DIR, vc):
    import os

    # Set QUICK = True for a fast smoke run (results will not match the paper).
    # From the shell, VINE_DEMOS_QUICK=1 does the same without editing the file.
    QUICK = False
    QUICK = QUICK or os.environ.get("VINE_DEMOS_QUICK") == "1"

    DATASETS = ["cdk8", "eg5"] if QUICK else vc.DATASET_NAMES
    NBOOT = 200 if QUICK else vc.NBOOTSTRAP
    FIG_DIR = vc.figure_dir(NB_DIR, "01_benchmark_comparison")
    print(f"QUICK={QUICK}  datasets={DATASETS}  figures -> {FIG_DIR}")
    return DATASETS, FIG_DIR, NBOOT, QUICK


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load the benchmark maps

    `FEPBenchmarkDataset` downloads the 5 ns Schindler data on first use and
    caches it under `~/.vine_cache`.
    """)
    return


@app.cell
def _(DATASETS, vc):
    datasets = vc.load_benchmark(DATASETS)
    for _name, _ds in datasets.items():
        print(
            f"  {_name:7s} {len(_ds.dataset_nodes):3d} ligands, "
            f"{len(_ds.dataset_edges):3d} edges"
        )
    return (datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit all seven estimators

    Each model is fit from scratch on each map; its predictions are written
    into the dataset as a new column named after the model label.
    """)
    return


@app.cell
def _(QUICK, datasets, vc):
    train_log = vc.train_all(datasets, vc.benchmark_factories(quick=QUICK), tag="clean")
    print(f"Trained {len(train_log)} models.")
    return (train_log,)


@app.cell
def _(NBOOT, datasets, train_log, vc):
    _ = train_log  # run after training (datasets are filled in place)
    metrics_node = vc.compute_metrics(datasets, vc.NODE_MODELS, "node", nbootstrap=NBOOT)
    metrics_edge = vc.compute_metrics(datasets, vc.EDGE_MODELS, "edge", nbootstrap=NBOOT)
    return metrics_edge, metrics_node


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Node RMSE and R² (manuscript figures)
    """)
    return


@app.cell
def _(FIG_DIR, metrics_node, vc):
    fig_node_rmse = vc.grouped_bar_plot(
        metrics_node, vc.NODE_MODELS, "RMSE", "Node RMSE (kcal/mol) — All benchmark datasets"
    )
    vc.save_figure(fig_node_rmse, FIG_DIR, "all_dataset_RMSE_node_comparison")
    fig_node_rmse
    return


@app.cell
def _(FIG_DIR, metrics_node, vc):
    fig_node_r2 = vc.grouped_bar_plot(
        metrics_node, vc.NODE_MODELS, "R2", "Node R² — All benchmark datasets", ylim=(0, 1.15)
    )
    vc.save_figure(fig_node_r2, FIG_DIR, "all_datasets_R2_node_comparison")
    fig_node_r2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supplementary: node Kendall's τ and edge-level metrics

    The edge panels add CCC. Edge metrics score the corrected ΔΔG of every
    edge against the experimental ΔΔG.
    """)
    return


@app.cell
def _(FIG_DIR, metrics_node, vc):
    fig_node_ktau = vc.grouped_bar_plot(
        metrics_node,
        vc.NODE_MODELS,
        "KTAU",
        "Node Kendall's τ — All benchmark datasets",
        ylim=(-0.2, 1.15),
    )
    vc.save_figure(fig_node_ktau, FIG_DIR, "all_datasets_KTau_node_comparison")
    fig_node_ktau
    return


@app.cell
def _(FIG_DIR, metrics_edge, vc):
    fig_edge_rmse = vc.grouped_bar_plot(
        metrics_edge, vc.EDGE_MODELS, "RMSE", "Edge RMSE (kcal/mol) — All benchmark datasets"
    )
    vc.save_figure(fig_edge_rmse, FIG_DIR, "all_dataset_RMSE_edge_comparison")
    fig_edge_rmse
    return


@app.cell
def _(FIG_DIR, metrics_edge, vc):
    fig_edge_r2 = vc.grouped_bar_plot(
        metrics_edge, vc.EDGE_MODELS, "R2", "Edge R² — All benchmark datasets", ylim=(0, 1.15)
    )
    vc.save_figure(fig_edge_r2, FIG_DIR, "all_datasets_R2_edge_comparison")
    fig_edge_r2
    return


@app.cell
def _(FIG_DIR, metrics_edge, vc):
    fig_edge_ktau = vc.grouped_bar_plot(
        metrics_edge,
        vc.EDGE_MODELS,
        "KTAU",
        "Edge Kendall's τ — All benchmark datasets",
        ylim=(-0.2, 1.15),
    )
    vc.save_figure(fig_edge_ktau, FIG_DIR, "all_datasets_KTau_edge_comparison")
    fig_edge_ktau
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supplementary: raw edge-error distribution (TNKS2)

    The uncorrected FEP edge error (`DeltaDeltaG` − experiment) with a fitted
    normal distribution. This is the error the estimators above correct.
    """)
    return


@app.cell
def _(FIG_DIR, datasets, vc):
    from vine_fep.graph_analysis import plot_error_distribution

    _name = "tnks2" if "tnks2" in datasets else next(iter(datasets))
    fig_err, _stats = plot_error_distribution(
        datasets[_name],
        predicted_column="DeltaDeltaG",
        experimental_column="Experimental DeltaDeltaG",
        data_type="edges",
    )
    vc.save_figure(fig_err, FIG_DIR, f"{_name}_edge_error_plot")
    fig_err
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interactive: all methods on one dataset

    Pick a dataset to see node RMSE and R² for every method. The selected
    panel is also saved as `per_dataset_<name>_metrics`.
    """)
    return


@app.cell
def _(DATASETS, mo):
    dataset_selector = mo.ui.dropdown(options=DATASETS, value=DATASETS[0], label="Dataset")
    dataset_selector
    return (dataset_selector,)


@app.cell
def _(FIG_DIR, dataset_selector, metrics_node, vc):
    _name = dataset_selector.value
    fig_panel = vc.metric_panel(
        metrics_node,
        vc.NODE_MODELS,
        _name,
        f"{_name.upper()} dataset — node ΔG (all methods)",
    )
    vc.save_figure(fig_panel, FIG_DIR, f"per_dataset_{_name}_metrics")
    fig_panel
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary tables (bootstrap means)
    """)
    return


@app.cell
def _(metrics_edge, metrics_node, mo, vc):
    _node = vc.metrics_table(metrics_node, "RMSE")
    _edge = vc.metrics_table(metrics_edge, "RMSE")
    print("Node RMSE (kcal/mol):\n", _node.to_string())
    print("\nEdge RMSE (kcal/mol):\n", _edge.to_string())
    mo.vstack([mo.md("**Node RMSE**"), _node, mo.md("**Edge RMSE**"), _edge])
    return


if __name__ == "__main__":
    app.run()
