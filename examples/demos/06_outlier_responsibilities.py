"""Demo 06: which edges does VI(2,3) treat as outliers?

Run with ``marimo edit examples/demos/06_outlier_responsibilities.py`` or
``python examples/demos/06_outlier_responsibilities.py``.
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
    # 06 · Outlier responsibilities under VI(2,3)

    A mixture likelihood does more than down-weight bad edges: after fitting,
    each edge has a posterior probability ("responsibility") of belonging to
    each likelihood component. The probability of the outlier component is a
    per-edge outlier score.

    Which component is "the outlier component"? Here it is the one whose
    mean is farthest from zero. Picking the widest component instead (as an
    earlier version of this analysis did) fails on these maps: fit in-sample,
    the GMM places a *narrow* component right at the +20 kcal/mol shift.

    This demo fits VI(2,3) (3-component likelihood, 2-component prior, both
    fit in-sample) to the six benchmark maps with synthetic outliers from
    demo 02 (+20 kcal/mol on 4 edges per map, seed 19) and checks whether the
    injected edges are the ones flagged. It uses
    `MixtureVI.compute_edge_component_responsibilities()`.

    **Figure produced:** `vi23_outlier_responsibilities`. This is a
    diagnostic, not a figure in the current manuscript.

    **Runtime:** about 10 seconds (shorter training than demo 02: 2000 epochs,
    100 Monte Carlo samples).

    Figures are written to `figures/06_outlier_responsibilities/` at the
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
    FIG_DIR = vc.figure_dir(NB_DIR, "06_outlier_responsibilities")
    datasets_noisy = vc.load_noisy_benchmark(vc.DATASET_NAMES)
    return FIG_DIR, datasets_noisy


@app.cell
def _(datasets_noisy, vc):
    import numpy as np
    import pandas as pd
    from vine_fep.models import MixtureVI, MixtureVIConfig

    _rows = []
    for _name, _ds in datasets_noisy.items():
        _cfg = MixtureVIConfig.from_dataset(
            _ds,
            K_likelihood=3,
            K_prior=2,
            kl_weight=0.1,
            n_epochs=2000,
            n_samples=100,
            patience=200,
            learning_rate=0.001,
            random_seed=42,
        )
        _model = vc.train_model(_ds, "VI(2,3)", lambda d, c=_cfg: MixtureVI(config=c, dataset=d))
        _resp = _model.compute_edge_component_responsibilities()
        # The outlier component is the one centred farthest from zero error.
        # (The widest component is not a safe choice: fit in-sample on these
        # maps, the GMM puts a *narrow* component at the +20 kcal/mol shift.)
        _outlier = int(np.argmax(np.abs(_model.gmm_means.numpy())))
        _flags = _ds.dataset_edges["noise_added"].values
        for _i, _p in enumerate(_resp[:, _outlier]):
            _rows.append(
                dict(
                    dataset=_name,
                    edge_idx=_i,
                    p_outlier=float(_p),
                    is_true_outlier=bool(_flags[_i]),
                )
            )
        print(f"  {_name:7s} fitted")
    outlier_df = pd.DataFrame(_rows)
    return (outlier_df,)


@app.cell
def _(mo, outlier_df):
    _flag = outlier_df["p_outlier"] > 0.5
    _true = outlier_df["is_true_outlier"]
    summary = (
        outlier_df.assign(hit=_flag & _true, false_pos=_flag & ~_true)
        .groupby("dataset", sort=False)
        .agg(
            true_outliers=("is_true_outlier", "sum"),
            flagged_correctly=("hit", "sum"),
            false_positives=("false_pos", "sum"),
        )
    )
    print(summary.to_string())
    mo.ui.table(summary.reset_index())
    return


@app.cell
def _(FIG_DIR, outlier_df, vc):
    import matplotlib.pyplot as plt

    fig_resp, _axes = plt.subplots(2, 3, figsize=(15, 8))
    for _ax, _name in zip(_axes.flatten(), vc.DATASET_NAMES):
        _sub = outlier_df[outlier_df["dataset"] == _name]
        _ax.scatter(
            _sub["edge_idx"],
            _sub["p_outlier"],
            c=["red" if t else "steelblue" for t in _sub["is_true_outlier"]],
            s=30,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.3,
        )
        _ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
        _ax.set_title(_name.upper(), fontsize=13, fontweight="bold")
        _ax.set_xlabel("Edge index", fontsize=11)
        _ax.set_ylabel("P(outlier)", fontsize=11)
        _ax.set_ylim(-0.05, 1.05)
    fig_resp.suptitle(
        "Per-edge outlier responsibility under VI(2,3) — red = true injected outlier",
        fontsize=15,
        fontweight="bold",
    )
    fig_resp.tight_layout()
    vc.save_figure(fig_resp, FIG_DIR, "vi23_outlier_responsibilities")
    fig_resp
    return


if __name__ == "__main__":
    app.run()
