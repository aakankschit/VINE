"""Demo 04: an industry FEP map with large outlier edges.

Run with ``marimo edit examples/demos/04_industry_case_study.py`` or
``python examples/demos/04_industry_case_study.py``.
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
    # 04 · Industry case study: a real map with large outlier edges

    The Industry dataset is a 12-ligand, 20-edge FEP map from an industrial
    collaborator. It is a subgraph of the CMET benchmark that was recomputed
    with a modified protocol, and a few of its edges are off by tens of
    kcal/mol. Unlike demo 02 the outliers here are real, not injected.

    The demo plots the raw edge-error distribution, fits the seven estimators
    (MLE, MAP, VI(1,1), VI(1,2), VI(1,3), VI(2,3), VI(3,3)) with the longer
    training used for this map, and compares node and edge metrics. For
    VI(1,2) the outlier component is wider (σ = 5, p = 0.28) than in the
    benchmark demos.

    **Manuscript figures reproduced**

    | Output here | Manuscript file |
    |---|---|
    | `industry_edge_error_plot` | `treeline_edge_error_plot.pdf` |
    | `industry_node_metrics` | `treeline_barplots.pdf` |

    **Supplementary outputs:** `industry_edge_metrics` (edge ΔΔG RMSE and R²)
    and one inference-effect plot per method
    (`industry_inference_effects_<method>`).

    **Runtime:** under a minute for the full run on a laptop CPU.
    `QUICK = True` shortens training and uses 200 bootstrap samples (about 15 seconds); its numbers will not match the paper.

    Data: `examples/data/treeline_dataset_{edges,nodes}.csv`. Figures are
    written to `figures/04_industry_case_study/` at the repository root, as
    PDF and PNG.
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

    NBOOT = 200 if QUICK else vc.NBOOTSTRAP
    FIG_DIR = vc.figure_dir(NB_DIR, "04_industry_case_study")
    print(f"QUICK={QUICK}  figures -> {FIG_DIR}")
    return FIG_DIR, NBOOT, QUICK


@app.cell
def _(NB_DIR, vc):
    industry = vc.load_industry(vc.data_dir(NB_DIR))
    print(
        f"{vc.INDUSTRY_LABEL} dataset: {len(industry.dataset_nodes)} ligands, "
        f"{len(industry.dataset_edges)} edges"
    )
    return (industry,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Raw edge errors (manuscript figure)

    Most edges sit near zero error, but a handful are tens of kcal/mol off,
    giving two clearly separate populations. A single fitted normal (red
    dashed) describes neither.
    """)
    return


@app.cell
def _(FIG_DIR, industry, vc):
    from vine_fep.graph_analysis import plot_error_distribution

    fig_err, _stats = plot_error_distribution(
        industry,
        predicted_column="DeltaDeltaG",
        experimental_column="Experimental DeltaDeltaG",
        data_type="edges",
    )
    vc.save_figure(fig_err, FIG_DIR, "industry_edge_error_plot")
    fig_err
    return


@app.cell
def _(QUICK, industry, vc):
    train_log = vc.train_all(
        {vc.INDUSTRY_LABEL: industry}, vc.industry_factories(quick=QUICK), tag="industry"
    )
    return (train_log,)


@app.cell
def _(NBOOT, industry, train_log, vc):
    _ = train_log
    _d = {vc.INDUSTRY_LABEL: industry}
    metrics_node = vc.compute_metrics(_d, vc.NODE_MODELS, "node", nbootstrap=NBOOT)
    metrics_edge = vc.compute_metrics(_d, vc.NODE_MODELS, "edge", nbootstrap=NBOOT)
    return metrics_edge, metrics_node


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Node metrics (manuscript figure)

    The Gaussian-likelihood point estimates (MLE, MAP) inherit the outlier
    edges' error; every VI model brings node RMSE down to about 1 kcal/mol.
    """)
    return


@app.cell
def _(FIG_DIR, metrics_node, vc):
    fig_node = vc.metric_panel(
        metrics_node,
        vc.NODE_MODELS,
        vc.INDUSTRY_LABEL,
        f"{vc.INDUSTRY_LABEL} dataset — node ΔG (all methods)",
    )
    vc.save_figure(fig_node, FIG_DIR, "industry_node_metrics")
    fig_node
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supplementary: edge metrics
    """)
    return


@app.cell
def _(FIG_DIR, metrics_edge, vc):
    fig_edge = vc.metric_panel(
        metrics_edge,
        vc.NODE_MODELS,
        vc.INDUSTRY_LABEL,
        f"{vc.INDUSTRY_LABEL} dataset — edge ΔΔG (all methods)",
    )
    vc.save_figure(fig_edge, FIG_DIR, "industry_edge_metrics")
    fig_edge
    return


@app.cell
def _(metrics_edge, metrics_node, mo, vc):
    import pandas as _pd

    _t = _pd.concat(
        [
            vc.metrics_table(metrics_node, "RMSE").rename(index=lambda s: "node RMSE"),
            vc.metrics_table(metrics_edge, "RMSE").rename(index=lambda s: "edge RMSE"),
        ]
    )
    print(_t.to_string())
    mo.ui.table(_t.reset_index(names="metric"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supplementary: how each method moves the edges

    One quiver plot per method: circles are the raw FEP ΔΔG, squares the
    corrected values, arrows the shift (shade = edge leverage). All seven are
    saved; pick one below to view it.
    """)
    return


@app.cell
def _(FIG_DIR, industry, train_log, vc):
    _ = train_log
    import matplotlib.pyplot as plt

    effect_figs = {}
    for _m in vc.NODE_MODELS:
        _fig = vc.inference_effects_figure(
            industry, _m, f"{vc.INDUSTRY_LABEL} dataset"
        )
        vc.save_figure(_fig, FIG_DIR, f"industry_inference_effects_{vc.slug(_m)}")
        effect_figs[_m] = _fig
        plt.close(_fig)
    return (effect_figs,)


@app.cell
def _(mo, vc):
    method_selector = mo.ui.dropdown(
        options=vc.NODE_MODELS, value="VI(1,2)", label="Method"
    )
    method_selector
    return (method_selector,)


@app.cell
def _(effect_figs, method_selector):
    effect_figs[method_selector.value]
    return


if __name__ == "__main__":
    app.run()
