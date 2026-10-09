"""Demo 05: correlation plots for CDK8 (MAP, VI(1,2), MLE).

Run with ``marimo edit examples/demos/05_cdk8_correlation.py`` or
``python examples/demos/05_cdk8_correlation.py``.
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
    # 05 · CDK8 correlation plots

    A closer look at a single benchmark map. This demo fits MLE, MAP and
    VI(1,2) (the manuscript's original "GMVI") to the CDK8 map with the same
    settings as demo 01, then draws predicted-vs-experimental parity plots
    with bootstrapped RMSE, R², Kendall's τ and MUE for each model, for node
    ΔG and for edge ΔΔG. The edge plot also shows CCC.

    It is also a compact template for scoring your own map: swap in your
    `FEPDataset` and the list of models.

    **Figures produced:** `cdk8_node_correlation` and `cdk8_edge_correlation`.
    These correspond to the CDK8 correlation panels of the draft outline
    (`Outline.tex`); they are not in the current main text.

    **Runtime:** about 30 seconds.

    Figures are written to `figures/05_cdk8_correlation/` at the repository
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
    DATASET = "cdk8"
    MODELS = ["MAP", "VI(1,2)", "MLE"]
    FIG_DIR = vc.figure_dir(NB_DIR, "05_cdk8_correlation")
    dataset = vc.load_benchmark([DATASET])[DATASET]
    _factories = dict(vc.benchmark_factories())
    for _label in MODELS:
        vc.train_model(dataset, _label, _factories[_label])
        print(f"  {DATASET} {_label:8s} trained")
    return DATASET, FIG_DIR, MODELS, dataset


@app.cell
def _(DATASET, FIG_DIR, MODELS, dataset, vc):
    from vine_fep.graph_analysis import plot_model_comparison_correlation

    fig_node, node_stats = plot_model_comparison_correlation(
        dataset,
        models=MODELS,
        data_type="nodes",
        model_colors=vc.MODEL_COLORS,
        nbootstrap=vc.NBOOTSTRAP,
    )
    vc.save_figure(fig_node, FIG_DIR, f"{DATASET}_node_correlation")
    fig_node
    return node_stats, plot_model_comparison_correlation


@app.cell
def _(DATASET, FIG_DIR, MODELS, dataset, plot_model_comparison_correlation, vc):
    fig_edge, edge_stats = plot_model_comparison_correlation(
        dataset,
        models=MODELS + ["CCC"],
        data_type="edges",
        model_colors=vc.MODEL_COLORS,
        nbootstrap=vc.NBOOTSTRAP,
    )
    vc.save_figure(fig_edge, FIG_DIR, f"{DATASET}_edge_correlation")
    fig_edge
    return (edge_stats,)


@app.cell
def _(edge_stats, mo, node_stats):
    print("Node statistics:\n", node_stats.to_string())
    print("\nEdge statistics:\n", edge_stats.to_string())
    mo.vstack([mo.md("**Node statistics**"), node_stats, mo.md("**Edge statistics**"), edge_stats])
    return


if __name__ == "__main__":
    app.run()
