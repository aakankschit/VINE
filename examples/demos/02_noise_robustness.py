"""Demo 02: robustness to synthetic outlier edges.

Run with ``marimo edit examples/demos/02_noise_robustness.py`` or
``python examples/demos/02_noise_robustness.py``.
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
    # 02 · Noise robustness: synthetic outlier edges

    Real FEP maps sometimes contain a few edges that are badly wrong. This
    demo stress-tests the estimators by adding **+20 kcal/mol to 4 randomly
    chosen edges** in each of the six Schindler benchmark maps (seed 19), then
    fitting the same seven estimators as demo 01 (MLE, MAP, VI(1,1), VI(1,2),
    VI(1,3), VI(2,3), VI(3,3)).

    Gaussian-likelihood estimators (MLE, MAP, VI(1,1)) spread each outlier's
    error over its neighbours. The mixture likelihoods in `MixtureVI` can
    assign an outlier edge to a wide component instead, so the node estimates
    are much less affected.

    **Manuscript figures reproduced**

    - `all_dataset_noisy_RMSE_node_comparison.pdf`: node RMSE with outliers
    - `pfkfb3_noisy_MLE_inference_effects.pdf`: how MLE moves the PFKFB3 edges
    - `pfkfb3_noisy_GMVI_inference_effects.pdf`: the same for VI(1,2)
      (the manuscript's "GMVI"; the file keeps the manuscript name)

    **Supplementary outputs:** node R² and Kendall's τ; edge RMSE, R² and
    Kendall's τ (with CCC).

    **Runtime:** about 9 minutes for the full run on a laptop CPU
    (42 model fits). `QUICK = True` runs CDK8 and PFKFB3 only with shortened
    training in about a minute; its numbers will not match the paper.

    Figures are written to `figures/02_noise_robustness/` at the repository
    root, as PDF and PNG.
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

    DATASETS = ["cdk8", "pfkfb3"] if QUICK else vc.DATASET_NAMES
    NBOOT = 200 if QUICK else vc.NBOOTSTRAP
    FIG_DIR = vc.figure_dir(NB_DIR, "02_noise_robustness")
    print(f"QUICK={QUICK}  datasets={DATASETS}  figures -> {FIG_DIR}")
    return DATASETS, FIG_DIR, NBOOT, QUICK


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build the noisy maps

    `vc.load_noisy_benchmark` loads each map and adds the outliers. The
    corrupted edges are flagged in a `noise_added` column. The random draw
    always walks the six maps in the same order, so a map gets the same
    outlier edges whether or not `QUICK` is set.
    """)
    return


@app.cell
def _(DATASETS, vc):
    datasets_noisy = vc.load_noisy_benchmark(DATASETS)
    for _name, _ds in datasets_noisy.items():
        _flagged = _ds.dataset_edges.index[_ds.dataset_edges["noise_added"]].tolist()
        print(f"  {_name:7s} outlier edge rows: {_flagged}")
    return (datasets_noisy,)


@app.cell
def _(QUICK, datasets_noisy, vc):
    train_log = vc.train_all(
        datasets_noisy, vc.benchmark_factories(quick=QUICK), tag="noisy"
    )
    print(f"Trained {len(train_log)} models.")
    return (train_log,)


@app.cell
def _(NBOOT, datasets_noisy, train_log, vc):
    _ = train_log  # run after training (datasets are filled in place)
    metrics_node = vc.compute_metrics(
        datasets_noisy, vc.NODE_MODELS, "node", nbootstrap=NBOOT
    )
    metrics_edge = vc.compute_metrics(
        datasets_noisy, vc.EDGE_MODELS, "edge", nbootstrap=NBOOT
    )
    return metrics_edge, metrics_node


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Node RMSE with outlier edges (manuscript figure)
    """)
    return


@app.cell
def _(FIG_DIR, metrics_node, vc):
    fig_noisy_rmse = vc.grouped_bar_plot(
        metrics_node,
        vc.NODE_MODELS,
        "RMSE",
        "Node RMSE — datasets with synthetic outlier edges",
    )
    vc.save_figure(fig_noisy_rmse, FIG_DIR, "all_dataset_noisy_RMSE_node_comparison")
    fig_noisy_rmse
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How MLE and VI(1,2) move the PFKFB3 edges (manuscript figures)

    Circles are the raw FEP ΔΔG, squares the corrected values, and arrows the
    shift (arrow shade = edge leverage). MLE leaves the four +20 kcal/mol edges
    far from experiment; VI(1,2) pulls them back toward the diagonal.
    """)
    return


@app.cell
def _(FIG_DIR, datasets_noisy, train_log, vc):
    _ = train_log
    _name = "pfkfb3" if "pfkfb3" in datasets_noisy else next(iter(datasets_noisy))
    fig_ie_mle = vc.inference_effects_figure(
        datasets_noisy[_name], "MLE", f"{_name} (noisy)"
    )
    vc.save_figure(fig_ie_mle, FIG_DIR, f"{_name}_noisy_MLE_inference_effects")
    fig_ie_mle
    return


@app.cell
def _(FIG_DIR, datasets_noisy, train_log, vc):
    _ = train_log
    _name = "pfkfb3" if "pfkfb3" in datasets_noisy else next(iter(datasets_noisy))
    fig_ie_vi12 = vc.inference_effects_figure(
        datasets_noisy[_name], "VI(1,2)", f"{_name} (noisy)"
    )
    # File name kept from the manuscript, where VI(1,2) was called "GMVI".
    vc.save_figure(fig_ie_vi12, FIG_DIR, f"{_name}_noisy_GMVI_inference_effects")
    fig_ie_vi12
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supplementary: other node and edge metrics
    """)
    return


@app.cell
def _(FIG_DIR, metrics_node, vc):
    fig_noisy_r2 = vc.grouped_bar_plot(
        metrics_node,
        vc.NODE_MODELS,
        "R2",
        "Node R² — datasets with synthetic outlier edges",
        ylim=(0, 1.15),
    )
    vc.save_figure(fig_noisy_r2, FIG_DIR, "all_datasets_noisy_R2_node_comparison")
    fig_noisy_r2
    return


@app.cell
def _(FIG_DIR, metrics_node, vc):
    fig_noisy_ktau = vc.grouped_bar_plot(
        metrics_node,
        vc.NODE_MODELS,
        "KTAU",
        "Node Kendall's τ — datasets with synthetic outlier edges",
        ylim=(-0.2, 1.15),
    )
    vc.save_figure(fig_noisy_ktau, FIG_DIR, "all_datasets_noisy_KTau_node_comparison")
    fig_noisy_ktau
    return


@app.cell
def _(FIG_DIR, metrics_edge, vc):
    fig_noisy_edge_rmse = vc.grouped_bar_plot(
        metrics_edge,
        vc.EDGE_MODELS,
        "RMSE",
        "Edge RMSE — datasets with synthetic outlier edges",
    )
    vc.save_figure(fig_noisy_edge_rmse, FIG_DIR, "all_dataset_noisy_RMSE_edge_comparison")
    fig_noisy_edge_rmse
    return


@app.cell
def _(FIG_DIR, metrics_edge, vc):
    fig_noisy_edge_r2 = vc.grouped_bar_plot(
        metrics_edge,
        vc.EDGE_MODELS,
        "R2",
        "Edge R² — datasets with synthetic outlier edges",
        ylim=(0, 1.15),
    )
    vc.save_figure(fig_noisy_edge_r2, FIG_DIR, "all_datasets_noisy_R2_edge_comparison")
    fig_noisy_edge_r2
    return


@app.cell
def _(FIG_DIR, metrics_edge, vc):
    fig_noisy_edge_ktau = vc.grouped_bar_plot(
        metrics_edge,
        vc.EDGE_MODELS,
        "KTAU",
        "Edge Kendall's τ — datasets with synthetic outlier edges",
        ylim=(-0.2, 1.15),
    )
    vc.save_figure(fig_noisy_edge_ktau, FIG_DIR, "all_datasets_noisy_KTau_edge_comparison")
    fig_noisy_edge_ktau
    return


@app.cell
def _(metrics_node, mo, vc):
    _table = vc.metrics_table(metrics_node, "RMSE")
    print("Node RMSE with outliers (kcal/mol):\n", _table.to_string())
    mo.vstack([mo.md("**Node RMSE with outlier edges**"), _table])
    return


if __name__ == "__main__":
    app.run()
