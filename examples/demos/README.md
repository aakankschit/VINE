# VINE manuscript demos

These marimo notebooks reproduce the figures of the VINE manuscript using only
the public `vine_fep` API. Each notebook also runs as a plain Python script.

```bash
pip install -e ".[dev]"   # from the repository root; marimo is required
marimo edit examples/demos/01_benchmark_comparison.py   # interactive
python examples/demos/01_benchmark_comparison.py        # headless
```

**Output:** each demo writes its figures as PDF and PNG to
`figures/<demo_name>/` at the repository root (created on first run;
`figures/` is git-ignored).

**Data:** the Schindler benchmark maps are downloaded on first use and cached
in `~/.vine_cache`. The Industry map is read from
`examples/data/treeline_dataset_{edges,nodes}.csv`.

## Demos

| Demo | What it shows | Manuscript figures produced | Full run | `QUICK` |
|---|---|---|---|---|
| `01_benchmark_comparison.py` | 7 estimators on the 6 benchmark maps; node and edge RMSE, R², Kendall's τ | `all_dataset_RMSE_node_comparison`, `all_datasets_R2_node_comparison` | ~7 min | ~40 s |
| `02_noise_robustness.py` | Same estimators after adding +20 kcal/mol to 4 edges per map; PFKFB3 inference effects | `all_dataset_noisy_RMSE_node_comparison`, `pfkfb3_noisy_MLE_inference_effects`, `pfkfb3_noisy_GMVI_inference_effects` | ~9 min | ~40 s |
| `03_prior_sweep.py` | MAP prior width σ₀ = 0.5…5.0 vs an MLE baseline | `prior_sweep_RMSE_metrics`, `prior_sweep_relative_improvement_5ns_RMSE`, `prior_sweep_R2_metrics`, `prior_sweep_R2_relative_improvement_5ns` | ~15 min | ~50 s |
| `04_industry_case_study.py` | A real industry map with large outlier edges: error distribution, node/edge metrics, per-method inference effects | `industry_edge_error_plot` (= `treeline_edge_error_plot`), `industry_node_metrics` (= `treeline_barplots`) | ~45 s | ~15 s |
| `05_cdk8_correlation.py` | Parity plots for MAP, VI(1,2) and MLE on CDK8, node and edge | none in the main text (CDK8 panels of the draft outline) | ~30 s | n/a |
| `06_outlier_responsibilities.py` | Per-edge outlier probabilities from VI(2,3) on the noisy maps | none (diagnostic) | ~10 s | n/a |

Every demo also writes supplementary figures; the first cell of each
notebook lists them. Runtimes are wall-clock on an Apple-silicon laptop CPU.

### QUICK mode

Demos 01–04 have a `QUICK = False` flag in their configuration cell. Set it to
`True` (or run with `VINE_DEMOS_QUICK=1 python ...`) for a fast smoke run with
fewer datasets, shorter training and 200 bootstrap samples. QUICK results do
**not** match the manuscript; leave the flag off to reproduce the paper.

## Model labels

| Label | Estimator | Configuration |
|---|---|---|
| MLE | `VariationalEstimator` | uniform prior [−100, 100], AutoDelta |
| MAP | `VariationalEstimator` | N(0, 1) prior, AutoDelta |
| VI(1,1) | `VariationalEstimator` | N(0, 1) prior, AutoNormal |
| VI(1,2) | `MixtureVI` | fixed 2-component likelihood (σ = 1 / 3, p = 0.2), 1-component prior; the manuscript's original "GMVI" |
| VI(1,3) / VI(2,3) / VI(3,3) | `MixtureVI` | 3-component likelihood with a 1-, 2- or 3-component prior, fit in-sample via `MixtureVIConfig.from_dataset` |

All hyperparameters and seeds live in `_common.py`, which every demo imports.
Edit them there only if you want results that differ from the paper.
