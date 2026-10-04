# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "vine-fep",
#     "marimo",
#     "matplotlib",
#     "pandas",
#     "numpy",
#     "pyro-ppl",
#     "torch",
#     "seaborn",
#     "scipy",
# ]
# ///

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
    # VINE-FEP: Complete Model Comparison Tutorial

    **VINE** provides multiple methods for correcting thermodynamic inconsistencies in Free Energy Perturbation (FEP) graphs. This tutorial demonstrates **all available correction models** on a single benchmark dataset and compares their performance.

    ## What you will learn

    1. Loading and exploring a benchmark FEP dataset
    2. Visualizing the perturbation graph and raw data quality
    3. Applying **six correction methods**:
       - **MLE** — Maximum Likelihood Estimation (uniform prior)
       - **MAP** — Maximum A Posteriori (Normal prior regularization)
       - **VI** — Variational Inference (full posterior with uncertainties)
       - **GMVI** — Gaussian Mixture VI (outlier-robust inference)
       - **WCC** — Weighted Cycle Closure (iterative deterministic correction)
       - **WSFC** — Weighted Spectral Free-energy Correction (graph Laplacian)
    4. Comparing model performance with statistical metrics and visualizations

    ## Background

    FEP calculations predict relative binding free energy differences (ΔΔG) between ligand pairs. These relative measurements form a **graph** where nodes are ligands and edges are perturbations. Due to finite sampling, the ΔΔG values are noisy and thermodynamically inconsistent (cycle closure errors ≠ 0). VINE corrects these inconsistencies to produce improved estimates of absolute binding free energies (ΔG).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Setup and Imports

    We import all necessary VINE modules. The package is organized into three main subpackages:

    - `vine_fep.dataset` — Data loading and management
    - `vine_fep.models` — Probabilistic and deterministic estimators
    - `vine_fep.graph_analysis` — Plotting, statistics, and graph utilities
    """)
    return


@app.cell
def _():
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pyro

    # Dataset
    from vine_fep.dataset import FEPDataset, FEPBenchmarkDataset

    # Models — Probabilistic
    from vine_fep.models import (
        VariationalEstimator,
        VariationalEstimatorConfig,
        GaussianMixtureVI,
        GaussianMixtureVIConfig,
        PriorType,
        GuideType,
    )

    # Models — Deterministic
    from vine_fep.models import (
        CycleClosureCorrection,
        CycleClosureCorrectionConfig,
        SpectralCorrection,
        SpectralCorrectionConfig,
    )

    # Graph analysis and plotting
    from vine_fep.graph_analysis import (
        plot_dataset_DGs,
        plot_dataset_DDGs,
        plot_model_comparison_bars,
        plot_model_comparison_correlation,
        plot_error_distribution,
        compute_simple_statistics,
        GraphSetup,
        GraphCycleAnalysis,
    )

    pyro.clear_param_store()
    print("All imports successful!")
    return (
        CycleClosureCorrection,
        CycleClosureCorrectionConfig,
        FEPDataset,
        GaussianMixtureVI,
        GaussianMixtureVIConfig,
        GraphCycleAnalysis,
        GraphSetup,
        GuideType,
        PriorType,
        SpectralCorrection,
        SpectralCorrectionConfig,
        VariationalEstimator,
        VariationalEstimatorConfig,
        np,
        pd,
        plot_error_distribution,
        plot_model_comparison_bars,
        plot_model_comparison_correlation,
        plt,
        pyro,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Load Benchmark Data

    VINE ships with 8 public FEP benchmark datasets from Schindler *et al.* Each can be loaded at two sampling times (5 ns or 20 ns).

    We use the **CDK8 (Cyclin-dependent kinase 8)** dataset at **5 ns** sampling — a moderately sized perturbation graph that illustrates model differences clearly.
    """)
    return


@app.cell
def _(FEPDataset, mo):
    dataset = FEPDataset(dataset_name="cdk8", sampling_time="5ns")

    _n_nodes = len(dataset.dataset_nodes)
    _n_edges = len(dataset.dataset_edges)

    mo.md(f"""
    **Dataset loaded:** CDK8 (5 ns)

    | Property | Value |
    |----------|-------|
    | Ligands (nodes) | {_n_nodes} |
    | Perturbations (edges) | {_n_edges} |
    | Avg. connectivity | {2 * _n_edges / _n_nodes:.1f} edges/node |
    """)
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2a. Inspect the raw data

    The dataset is organized into two DataFrames:

    - **`dataset_nodes`** — one row per ligand with experimental ΔG
    - **`dataset_edges`** — one row per perturbation with computed ΔΔG and its uncertainty
    """)
    return


@app.cell
def _(dataset, mo):
    mo.ui.table(dataset.dataset_nodes, label="Node Data (Ligands)")
    return


@app.cell
def _(dataset, mo):
    mo.ui.table(
        dataset.dataset_edges[
            ["Source", "Destination", "DeltaDeltaG", "DeltaDeltaG Error", "Experimental DeltaDeltaG"]
        ],
        label="Edge Data (Perturbations)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2b. Visualize the perturbation graph

    The FEP perturbation network shows which ligands are connected by alchemical transformations. Edge labels display the raw computed ΔΔG values.
    """)
    return


@app.cell
def _(GraphSetup, dataset, plt):
    graph_setup = GraphSetup(dataset)
    fig_graph = graph_setup.visualize_graph(
        title="CDK8 Perturbation Graph",
        figsize=(10, 8),
        edge_label_attr="DeltaDeltaG",
    )
    plt.gcf()
    return (graph_setup,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2c. Raw error distribution

    Before applying any corrections, let us examine the distribution of errors between the raw FEP ΔΔG predictions and the experimental values.
    """)
    return


@app.cell
def _(dataset, plot_error_distribution, plt):
    _fig_err, _stats_err = plot_error_distribution(
        dataset,
        predicted_column="DeltaDeltaG",
        data_type="edges",
        title="Raw FEP Error Distribution (before correction)",
        figsize=(10, 6),
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Apply Correction Models

    We now apply all six correction methods. Each model reads graph data from the dataset, fits its parameters, and writes predictions back to the dataset DataFrames.

    VINE's **dataset-centric design** means all predictions accumulate as new columns on the same `dataset` object, making comparison straightforward.

    ### Probabilistic Models

    These models formulate the correction as a Bayesian inference problem, estimating node values (absolute ΔG) that best explain the observed edge measurements (relative ΔΔG).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3a. MLE — Maximum Likelihood Estimation

    Uses a **uniform (flat) prior** with a point-estimate guide (`AutoDelta`). This is equivalent to ordinary least squares — no regularization, just fitting the data.

    $$\hat{z} = \arg\min_z \sum_{(i,j) \in E} \left( (z_j - z_i) - \Delta\Delta G_{ij}^{\mathrm{FEP}} \right)^2$$
    """)
    return


@app.cell
def _(
    GuideType,
    PriorType,
    VariationalEstimator,
    VariationalEstimatorConfig,
    dataset,
    pyro,
):
    pyro.clear_param_store()

    mle_config = VariationalEstimatorConfig(
        learning_rate=0.01,
        num_steps=2000,
        prior_type=PriorType.UNIFORM,
        prior_parameters=[-100.0, 100.0],
        guide_type=GuideType.AUTO_DELTA,
        patience=100,
        random_seed=42,
    )

    mle_model = VariationalEstimator(config=mle_config, dataset=dataset)
    mle_model.fit()
    mle_model.add_predictions_to_dataset()
    print("MLE complete.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3b. MAP — Maximum A Posteriori

    Adds a **Normal prior** centered at zero, which regularizes the node estimates and prevents overfitting to noisy edges.

    $$\hat{z} = \arg\max_z \left[ \sum_{(i,j)} \log p(\Delta\Delta G_{ij} \mid z) + \sum_k \log \mathcal{N}(z_k \mid 0, \sigma_0^2) \right]$$
    """)
    return


@app.cell
def _(
    GuideType,
    PriorType,
    VariationalEstimator,
    VariationalEstimatorConfig,
    dataset,
    pyro,
):
    pyro.clear_param_store()

    map_config = VariationalEstimatorConfig(
        learning_rate=0.01,
        num_steps=2000,
        prior_type=PriorType.NORMAL,
        prior_parameters=[0.0, 10.0],
        guide_type=GuideType.AUTO_DELTA,
        patience=100,
        random_seed=42,
    )

    map_model = VariationalEstimator(config=map_config, dataset=dataset)
    map_model.fit()
    map_model.add_predictions_to_dataset()
    print("MAP complete.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3c. VI — Variational Inference

    Uses `AutoNormal` guide to approximate the **full posterior distribution** over node values, providing uncertainty estimates alongside point predictions.

    $$q(z) = \prod_k \mathcal{N}(z_k \mid \mu_k, \sigma_k^2) \approx p(z \mid \text{data})$$
    """)
    return


@app.cell
def _(
    GuideType,
    PriorType,
    VariationalEstimator,
    VariationalEstimatorConfig,
    dataset,
    pyro,
):
    pyro.clear_param_store()

    vi_config = VariationalEstimatorConfig(
        learning_rate=0.01,
        num_steps=3000,
        prior_type=PriorType.NORMAL,
        prior_parameters=[0.0, 10.0],
        guide_type=GuideType.AUTO_NORMAL,
        patience=100,
        random_seed=42,
    )

    vi_model = VariationalEstimator(config=vi_config, dataset=dataset)
    vi_model.fit()
    vi_model.add_predictions_to_dataset()
    print("VI complete.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3d. GMVI — Gaussian Mixture Variational Inference

    Uses a **mixture likelihood** with normal and outlier components. This makes inference robust to problematic edges and can identify which perturbations are likely outliers.

    $$p(\Delta\Delta G_{ij} \mid z) = (1 - \pi)\,\mathcal{N}(\cdot \mid z_j - z_i,\, \sigma_1^2) + \pi\,\mathcal{N}(\cdot \mid z_j - z_i,\, \sigma_2^2)$$
    """)
    return


@app.cell
def _(GaussianMixtureVI, GaussianMixtureVIConfig, dataset, pyro):
    pyro.clear_param_store()

    gmvi_config = GaussianMixtureVIConfig(
        prior_std=5.0,
        normal_std=1.0,
        outlier_std=3.0,
        outlier_prob=0.2,
        n_epochs=1000,
        n_samples=100,
        learning_rate=0.01,
        patience=50,
        random_seed=42,
    )

    gmvi_model = GaussianMixtureVI(dataset=dataset, config=gmvi_config)
    gmvi_model.fit()
    gmvi_model.add_predictions_to_dataset()
    print("GMVI complete.")
    return (gmvi_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Deterministic Models

    These methods use closed-form linear algebra solutions rather than iterative optimization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3e. WCC — Weighted Cycle Closure

    Detects all cycles in the perturbation graph and iteratively redistributes cycle closure errors across edges, weighted by inverse variance (1/σ²). Node values are then computed via BFS from a reference node.
    """)
    return


@app.cell
def _(CycleClosureCorrection, CycleClosureCorrectionConfig, dataset):
    wcc_config = CycleClosureCorrectionConfig(
        tolerance=1e-6,
        max_iterations=1000,
        use_weights=True,
        max_cycle_length=8,
    )

    wcc_model = CycleClosureCorrection(config=wcc_config, dataset=dataset)
    wcc_model.fit()
    wcc_model.add_predictions_to_dataset()

    _results = wcc_model.get_results()
    print(f"WCC complete. Cycles found: {_results.get('num_cycles', 'N/A')}, "
          f"Converged: {_results.get('converged', 'N/A')}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 3f. WSFC — Weighted Spectral Free-energy Correction

    Solves for optimal node values via the **pseudoinverse of the weighted graph Laplacian**:

    $$z^* = L_W^+ B^T W \, \Delta\Delta G$$

    where $B$ is the incidence matrix and $W = \mathrm{diag}(1/\sigma_{ij}^2)$.
    """)
    return


@app.cell
def _(SpectralCorrection, SpectralCorrectionConfig, dataset):
    wsfc_config = SpectralCorrectionConfig(use_weights=True)

    wsfc_model = SpectralCorrection(config=wsfc_config, dataset=dataset)
    wsfc_model.fit()
    wsfc_model.add_predictions_to_dataset()
    print("WSFC complete.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 4. Compare Model Predictions

    All six models have now written their predictions to the dataset. Let us verify the available columns and inspect the results side by side.
    """)
    return


@app.cell
def _(dataset, mo):
    _estimators = dataset.get_estimators()

    mo.md(f"""
    **Models applied:** {', '.join(str(e) for e in _estimators)}

    ### Node-level predictions (ΔG)
    """)
    return


@app.cell
def _(dataset, mo):
    _node_cols = ["Name", "Exp. DeltaG"]
    _model_cols = ["MLE", "MAP", "VI", "GMVI", "WCC", "WSFC"]
    _available = [c for c in _model_cols if c in dataset.dataset_nodes.columns]
    _display_df = dataset.dataset_nodes[_node_cols + _available].copy()
    for _c in _available:
        _display_df[_c] = _display_df[_c].round(3)
    mo.ui.table(_display_df, label="Node Predictions (ΔG in kcal/mol)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 5. Performance Visualization

    ### 5a. Correlation plots — Node-level (ΔG)

    Side-by-side scatter plots of experimental vs. predicted absolute free energies for each model. All values are mean-centered to handle gauge freedom. Error bands show ±0.5 and ±1.0 kcal/mol zones.
    """)
    return


@app.cell
def _(dataset, plot_model_comparison_correlation, plt):
    _models = [m for m in ["MLE", "MAP", "VI", "GMVI", "WCC", "WSFC"]
               if m in dataset.dataset_nodes.columns]

    fig_corr_nodes, stats_corr_nodes = plot_model_comparison_correlation(
        dataset,
        models=_models,
        data_type="nodes",
        performance_metrics=["RMSE", "MUE", "R2", "rho"],
        grid_layout=(2, 3),
        target_name="CDK8",
        nbootstrap=500,
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5b. Correlation plots — Edge-level (ΔΔG)

    The same comparison at the edge level — how well do the corrected relative free energies match experiment?
    """)
    return


@app.cell
def _(dataset, plot_model_comparison_correlation, plt):
    _models_e = [m for m in ["MLE", "MAP", "VI", "GMVI", "WCC"]
                 if m in dataset.dataset_edges.columns]

    fig_corr_edges, stats_corr_edges = plot_model_comparison_correlation(
        dataset,
        models=_models_e,
        data_type="edges",
        performance_metrics=["RMSE", "MUE", "R2", "rho"],
        grid_layout=(2, 3),
        target_name="CDK8",
        nbootstrap=500,
    )
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5c. Bar chart comparison

    Bootstrap confidence intervals on RMSE, MUE, R², and Pearson ρ across all models — a compact summary of which methods perform best.
    """)
    return


@app.cell
def _(dataset, plot_model_comparison_bars, plt):
    _models_bar = [m for m in ["MLE", "MAP", "VI", "GMVI", "WCC", "WSFC"]
                   if m in dataset.dataset_nodes.columns]

    fig_bars, stats_bars = plot_model_comparison_bars(
        dataset,
        models=_models_bar,
        data_type="nodes",
        performance_metrics=["RMSE", "MUE", "R2", "rho"],
        grid_layout=(2, 2),
        target_name="CDK8",
        nbootstrap=500,
    )
    plt.gcf()
    return (stats_bars,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5d. Statistics summary table

    The numerical statistics behind the bar chart, including 95% bootstrap confidence intervals.
    """)
    return


@app.cell
def _(mo, stats_bars):
    _output = mo.md("_No statistics available._")
    if stats_bars is not None:
        _display = stats_bars.copy()
        for _col in ["Value", "Mean", "CI_Lower", "CI_Upper"]:
            if _col in _display.columns:
                _display[_col] = _display[_col].round(4)
        _output = mo.ui.table(_display, label="Performance Statistics (Node-level)")
    _output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 6. Outlier Detection with GMVI

    A unique advantage of the Gaussian Mixture VI model is its ability to compute **posterior outlier probabilities** for each edge. Edges with high outlier probability are likely to have problematic ΔΔG values due to insufficient sampling or poor overlap.
    """)
    return


@app.cell
def _(dataset, gmvi_model, mo, np):
    outlier_probs = gmvi_model.compute_edge_outlier_probabilities()

    _edges = dataset.dataset_edges.copy()
    _edges["Outlier_Probability"] = outlier_probs

    _outlier_df = _edges[
        ["Source", "Destination", "DeltaDeltaG", "DeltaDeltaG Error", "Outlier_Probability"]
    ].copy()
    _outlier_df["Outlier_Probability"] = _outlier_df["Outlier_Probability"].round(4)
    _outlier_df = _outlier_df.sort_values("Outlier_Probability", ascending=False)

    _n_flagged = int(np.sum(np.array(outlier_probs) > 0.5))

    mo.md(f"""
    **{_n_flagged} edges flagged as likely outliers** (posterior probability > 0.5):
    """)
    return


@app.cell
def _(dataset, gmvi_model, mo):
    _probs = gmvi_model.compute_edge_outlier_probabilities()
    _edges_display = dataset.dataset_edges[
        ["Source", "Destination", "DeltaDeltaG", "DeltaDeltaG Error"]
    ].copy()
    _edges_display["Outlier_Prob"] = [round(p, 4) for p in _probs]
    _edges_display = _edges_display.sort_values("Outlier_Prob", ascending=False)
    mo.ui.table(_edges_display, label="Edge Outlier Probabilities (sorted)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 7. Graph Cycle Analysis

    Understanding cycle closure errors reveals where the FEP graph is thermodynamically inconsistent. VINE provides tools to analyze cycles and visualize how each model corrects them.

    ### 7a. Cycle closure errors before and after correction
    """)
    return


@app.cell
def _(GraphCycleAnalysis, graph_setup, mo, np):
    cycle_analysis = GraphCycleAnalysis(graph_setup)
    cycle_analysis.find_cycles(method="basis")

    _raw_attrs = ["DeltaDeltaG"]
    _corrected_attrs = [a for a in ["MAP", "VI", "GMVI", "WCC"]
                        if a in graph_setup.get_edge_dataframe().columns]
    _all_attrs = _raw_attrs + _corrected_attrs

    cycle_errors_df = cycle_analysis.compute_cycle_errors(
        attributes=_all_attrs,
        method="basis",
    )

    _summary_rows = []
    for attr in _all_attrs:
        _col = f"{attr}_cycle_error"
        if _col in cycle_errors_df.columns:
            _vals = cycle_errors_df[_col].dropna().values
            _summary_rows.append({
                "Method": attr,
                "Mean |error|": f"{np.mean(np.abs(_vals)):.3f}",
                "Max |error|": f"{np.max(np.abs(_vals)):.3f}",
                "Std": f"{np.std(_vals):.3f}",
            })

    mo.md(f"""
    **Cycle basis analysis** found **{len(cycle_errors_df)} independent cycles**.

    Cycle closure errors should approach zero after correction — smaller values indicate better thermodynamic consistency.
    """)
    return cycle_analysis, cycle_errors_df


@app.cell
def _(cycle_errors_df, mo, np, pd):
    _all_attrs_ce = ["DeltaDeltaG", "MAP", "VI", "GMVI", "WCC"]
    _rows = []
    for _attr in _all_attrs_ce:
        _col = f"{_attr}_cycle_error"
        if _col in cycle_errors_df.columns:
            _vals = cycle_errors_df[_col].dropna().values
            _rows.append({
                "Method": _attr if _attr != "DeltaDeltaG" else "Raw FEP",
                "Mean |error| (kcal/mol)": round(float(np.mean(np.abs(_vals))), 3),
                "Max |error| (kcal/mol)": round(float(np.max(np.abs(_vals))), 3),
                "Std (kcal/mol)": round(float(np.std(_vals)), 3),
            })
    _ce_summary = pd.DataFrame(_rows)
    mo.ui.table(_ce_summary, label="Cycle Closure Error Summary")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7b. Edge leverages

    Leverage quantifies how much influence each edge has on the inferred node values. High-leverage edges disproportionately affect results and deserve extra scrutiny.
    """)
    return


@app.cell
def _(cycle_analysis, mo):
    leverages_df = cycle_analysis.compute_leverages(store_in_graph=True)
    _lev_display = leverages_df.round(4)
    mo.ui.table(_lev_display, label="Edge Leverages")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 8. Key Takeaways

    | Model | Type | Strengths | Limitations |
    |-------|------|-----------|-------------|
    | **MLE** | Probabilistic | Unbiased; no hyperparameters | No regularization; sensitive to noise |
    | **MAP** | Probabilistic | Regularized; robust | Requires prior choice; point estimate only |
    | **VI** | Probabilistic | Full uncertainties; posterior approximation | Slower; mean-field assumption |
    | **GMVI** | Probabilistic | Outlier detection; robust to bad edges | More hyperparameters; slower |
    | **WCC** | Deterministic | Intuitive cycle-based correction; fast | Depends on cycle structure |
    | **WSFC** | Deterministic | Closed-form; fast; uncertainty from Laplacian | Linear; assumes Gaussian errors |

    **General observations:**

    - Regularized methods (MAP, VI, GMVI) tend to outperform MLE on noisy data
    - GMVI is especially valuable when outlier edges are present
    - Deterministic methods (WCC, WSFC) are fast alternatives that often perform comparably
    - All corrections improve over raw FEP ΔΔG values — the choice depends on whether you need uncertainties, outlier detection, or speed
    """)
    return


if __name__ == "__main__":
    app.run()
