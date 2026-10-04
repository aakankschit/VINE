"""Demo 03: sweep the MAP prior width σ₀ and compare against MLE.

Run with ``marimo edit examples/demos/03_prior_sweep.py`` or
``python examples/demos/03_prior_sweep.py``.
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
    # 03 · MAP prior sweep: how much shrinkage helps?

    MAP places a Gaussian prior N(0, σ₀²) on the mean-centered node ΔG. A
    narrow prior shrinks the estimates toward zero; a wide one approaches MLE.
    This demo refits MAP on each of the six Schindler benchmark maps for
    σ₀ = 0.5, 1.0, …, 5.0 kcal/mol and compares the bootstrapped node metrics
    with an MLE baseline fit to the same map.

    **Manuscript figures reproduced**

    - `prior_sweep_RMSE_metrics.pdf`: node RMSE vs σ₀ (dashed = MLE)
    - `prior_sweep_relative_improvement_5ns_RMSE.pdf`: % RMSE improvement over MLE
    - `prior_sweep_R2_metrics.pdf`: node R² vs σ₀ (dashed = MLE)
    - `prior_sweep_R2_relative_improvement_5ns.pdf`: % R² improvement over MLE

    **Supplementary output:** `prior_sweep_correlation_metrics` (R² and
    Pearson ρ side by side).

    **Runtime:** about 15 minutes for the full run on a laptop CPU
    (6 MLE fits + 60 MAP fits, 1000 bootstrap samples each).
    `QUICK = True` sweeps 5 σ₀ values on 2 maps with shortened training in
    under a minute; its numbers will not match the paper.

    Figures are written to `figures/03_prior_sweep/` at the repository root,
    as PDF and PNG.
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

    import numpy as np

    # Set QUICK = True for a fast smoke run (results will not match the paper).
    # From the shell, VINE_DEMOS_QUICK=1 does the same without editing the file.
    QUICK = False
    QUICK = QUICK or os.environ.get("VINE_DEMOS_QUICK") == "1"

    DATASETS = ["cdk8", "tnks2"] if QUICK else vc.DATASET_NAMES
    PRIOR_STDS = (
        np.array([0.5, 1.0, 2.0, 3.0, 5.0]) if QUICK else np.arange(0.5, 5.5, 0.5)
    )
    NBOOT = 200 if QUICK else vc.NBOOTSTRAP
    FIG_DIR = vc.figure_dir(NB_DIR, "03_prior_sweep")
    print(f"QUICK={QUICK}  datasets={DATASETS}  σ₀={PRIOR_STDS.tolist()}")
    print(f"figures -> {FIG_DIR}")
    return DATASETS, FIG_DIR, NBOOT, PRIOR_STDS, QUICK, np


@app.cell
def _(DATASETS, vc):
    datasets = vc.load_benchmark(DATASETS)
    return (datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## MLE baseline

    One MLE fit per map (wide uniform prior, AutoDelta guide), scored with
    the same bootstrap as the sweep.
    """)
    return


@app.cell
def _(NBOOT, QUICK, datasets, vc):
    _make_mle = dict(vc.benchmark_factories(quick=QUICK))["MLE"]
    mle_baselines = {}
    for _name, _ds in datasets.items():
        vc.train_model(_ds, "MLE", _make_mle)
        _nodes = _ds.dataset_nodes
        mle_baselines[_name] = {
            _m: vc.centered_bootstrap(
                _nodes["Exp. DeltaG"].values, _nodes["MLE"].values, _m, NBOOT
            )["mean"]
            for _m in ("RMSE", "R2", "rho")
        }
        print(
            f"  {_name:7s} MLE RMSE = {mle_baselines[_name]['RMSE']:.3f}, "
            f"R² = {mle_baselines[_name]['R2']:.3f}"
        )
    return (mle_baselines,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sweep σ₀

    MAP is refit for every (map, σ₀) pair. Each fit overwrites the `MAP`
    column, so the prediction is copied out immediately.
    """)
    return


@app.cell
def _(NBOOT, PRIOR_STDS, QUICK, datasets, vc):
    import pandas as pd

    _rows = []
    for _name, _ds in datasets.items():
        for _sigma in PRIOR_STDS:
            vc.train_model(_ds, "MAP", vc.map_factory(float(_sigma), quick=QUICK))
            _nodes = _ds.dataset_nodes
            _exp = _nodes["Exp. DeltaG"].values
            _pred = _nodes["MAP"].values.copy()
            for _m in ("RMSE", "R2", "rho"):
                _r = vc.centered_bootstrap(_exp, _pred, _m, NBOOT)
                _rows.append(
                    dict(
                        dataset=_name,
                        prior_std=float(_sigma),
                        metric=_m,
                        mean=_r["mean"],
                        ci_low=_r["low"],
                        ci_high=_r["high"],
                    )
                )
        print(f"  {_name:7s} swept {len(PRIOR_STDS)} values of σ₀")
    sweep_df = pd.DataFrame(_rows)
    return (sweep_df,)


@app.cell
def _(DATASETS, vc):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # One fixed colour per benchmark map, consistent across all sweep figures.
    _palette = sns.color_palette(vc.DATASET_COLORS_PALETTE, len(vc.DATASET_NAMES))
    DS_COLORS = dict(zip(vc.DATASET_NAMES, _palette))

    def sweep_lines(ax, sweep_df, metric, mle_baselines, fontsize=16):
        """Plot metric vs σ₀ per map, with each map's MLE value dashed."""
        for _name in DATASETS:
            _sub = sweep_df[
                (sweep_df["dataset"] == _name) & (sweep_df["metric"] == metric)
            ].sort_values("prior_std")
            ax.plot(
                _sub["prior_std"],
                _sub["mean"],
                marker="o",
                label=_name,
                color=DS_COLORS[_name],
                linewidth=2,
                markersize=6,
            )
            ax.axhline(
                mle_baselines[_name][metric],
                color=DS_COLORS[_name],
                linestyle="--",
                alpha=0.5,
                linewidth=1.5,
            )
        ax.set_xlabel("Prior Standard Deviation (σ₀)", fontsize=fontsize)
        ax.grid(True, alpha=0.3)

    return DS_COLORS, plt, sweep_lines


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Absolute metrics vs σ₀ (manuscript figures)
    """)
    return


@app.cell
def _(DATASETS, FIG_DIR, mle_baselines, plt, sweep_df, sweep_lines, vc):
    fig_rmse, _ax = plt.subplots(figsize=(10, 6))
    sweep_lines(_ax, sweep_df, "RMSE", mle_baselines)
    _ax.set_ylabel("RMSE", fontsize=16)
    _ax.set_title("RMSE vs Prior Std", fontsize=18, fontweight="bold")
    _ax.tick_params(axis="both", labelsize=14)
    _h, _l = _ax.get_legend_handles_labels()
    fig_rmse.legend(
        _h,
        _l,
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(DATASETS),
        fontsize=14,
        title_fontsize=16,
    )
    fig_rmse.suptitle(
        "Error Metrics - Effect of Prior Standard Deviation\n"
        "(Dashed lines = MLE baseline)",
        fontsize=18,
        fontweight="bold",
    )
    fig_rmse.tight_layout()
    fig_rmse.subplots_adjust(bottom=0.18)
    vc.save_figure(fig_rmse, FIG_DIR, "prior_sweep_RMSE_metrics")
    fig_rmse
    return


@app.cell
def _(DATASETS, FIG_DIR, mle_baselines, plt, sweep_df, sweep_lines, vc):
    fig_r2, _ax = plt.subplots(figsize=(10, 6))
    sweep_lines(_ax, sweep_df, "R2", mle_baselines)
    _ax.set_ylabel("R²", fontsize=16)
    _ax.set_title("R² vs Prior Std", fontsize=18, fontweight="bold")
    _ax.tick_params(axis="both", labelsize=14)
    _h, _l = _ax.get_legend_handles_labels()
    fig_r2.legend(
        _h,
        _l,
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(DATASETS),
        fontsize=14,
        title_fontsize=16,
    )
    fig_r2.suptitle(
        "Correlation Metrics - Effect of Prior Standard Deviation\n"
        "(Dashed lines = MLE baseline)",
        fontsize=18,
        fontweight="bold",
    )
    fig_r2.tight_layout()
    fig_r2.subplots_adjust(bottom=0.18)
    vc.save_figure(fig_r2, FIG_DIR, "prior_sweep_R2_metrics")
    fig_r2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Improvement over MLE (manuscript figures)

    Positive values mean MAP beats MLE. For RMSE the improvement is
    (MLE − MAP) / MLE; for R² it is (MAP − MLE) / MLE.
    """)
    return


@app.cell
def _(DATASETS, DS_COLORS, FIG_DIR, mle_baselines, plt, sweep_df, vc):
    fig_rel_rmse, _ax = plt.subplots(figsize=(10, 6))
    for _name in DATASETS:
        _sub = sweep_df[
            (sweep_df["dataset"] == _name) & (sweep_df["metric"] == "RMSE")
        ].sort_values("prior_std")
        _base = mle_baselines[_name]["RMSE"]
        _ax.plot(
            _sub["prior_std"],
            (_base - _sub["mean"]) / _base * 100.0,
            marker="o",
            label=_name,
            color=DS_COLORS[_name],
            linewidth=2,
            markersize=6,
        )
    _ax.axhline(0, color="black", linestyle="--", linewidth=1, label="MLE baseline")
    _ax.set_xlabel("Prior Standard Deviation (σ₀)", fontsize=16)
    _ax.set_ylabel("RMSE Improvement over MLE (%)", fontsize=16)
    _ax.set_title("MAP Performance Relative to MLE", fontsize=18, fontweight="bold")
    _ax.tick_params(axis="both", labelsize=14)
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(0, 5.5)
    _ax.legend(title="Dataset", loc="best", fontsize=12)
    fig_rel_rmse.tight_layout()
    vc.save_figure(fig_rel_rmse, FIG_DIR, "prior_sweep_relative_improvement_5ns_RMSE")
    fig_rel_rmse
    return


@app.cell
def _(DATASETS, DS_COLORS, FIG_DIR, mle_baselines, plt, sweep_df, vc):
    fig_rel_r2, _ax = plt.subplots(figsize=(10, 6))
    for _name in DATASETS:
        _sub = sweep_df[
            (sweep_df["dataset"] == _name) & (sweep_df["metric"] == "R2")
        ].sort_values("prior_std")
        _base = mle_baselines[_name]["R2"]
        _ax.plot(
            _sub["prior_std"],
            (_sub["mean"] - _base) / _base * 100.0,
            marker="o",
            label=_name,
            color=DS_COLORS[_name],
            linewidth=2,
            markersize=6,
        )
    _ax.axhline(0, color="black", linestyle="--", linewidth=1, label="MLE baseline")
    _ax.set_xlabel("Prior Standard Deviation (σ₀)", fontsize=16)
    _ax.set_ylabel("R² Improvement over MLE (%)", fontsize=16)
    _ax.set_title("MAP R² Performance Relative to MLE", fontsize=18, fontweight="bold")
    _ax.tick_params(axis="both", labelsize=14)
    _ax.grid(True, alpha=0.3)
    _ax.set_xlim(0, 5.5)
    _ax.legend(title="Dataset", loc="best", fontsize=12)
    fig_rel_r2.tight_layout()
    vc.save_figure(fig_rel_r2, FIG_DIR, "prior_sweep_R2_relative_improvement_5ns")
    fig_rel_r2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Supplementary: R² and Pearson ρ side by side
    """)
    return


@app.cell
def _(DATASETS, FIG_DIR, mle_baselines, plt, sweep_df, sweep_lines, vc):
    fig_corr, _axes = plt.subplots(1, 2, figsize=(12, 5))
    for _ax, (_m, _lbl) in zip(_axes, [("R2", "R²"), ("rho", "Pearson ρ")]):
        sweep_lines(_ax, sweep_df, _m, mle_baselines, fontsize=12)
        _ax.set_ylabel(_lbl, fontsize=12)
        _ax.set_title(f"{_lbl} vs Prior Std", fontsize=14, fontweight="bold")
    _h, _l = _axes[0].get_legend_handles_labels()
    fig_corr.legend(
        _h,
        _l,
        title="Dataset",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(DATASETS),
        fontsize=12,
    )
    fig_corr.suptitle(
        "Correlation Metrics - Effect of Prior Standard Deviation\n"
        "(Dashed lines = MLE baseline)",
        fontsize=14,
        fontweight="bold",
    )
    fig_corr.tight_layout()
    fig_corr.subplots_adjust(bottom=0.18)
    vc.save_figure(fig_corr, FIG_DIR, "prior_sweep_correlation_metrics")
    fig_corr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Best σ₀ per map
    """)
    return


@app.cell
def _(mle_baselines, mo, sweep_df):
    _rows = []
    for _name, _g in sweep_df.groupby("dataset", sort=False):
        _rmse = _g[_g["metric"] == "RMSE"]
        _best = _rmse.loc[_rmse["mean"].idxmin()]
        _mle = mle_baselines[_name]["RMSE"]
        _rows.append(
            {
                "dataset": _name,
                "best σ₀ (RMSE)": _best["prior_std"],
                "MAP RMSE": round(_best["mean"], 3),
                "MLE RMSE": round(_mle, 3),
                "improvement %": round((_mle - _best["mean"]) / _mle * 100, 1),
            }
        )
    import pandas as _pd

    best_df = _pd.DataFrame(_rows)
    print(best_df.to_string(index=False))
    mo.ui.table(best_df)
    return


if __name__ == "__main__":
    app.run()
