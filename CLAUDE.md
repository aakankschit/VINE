# CLAUDE.md - MAPLE-FEP

Bayesian inference for Free Energy Perturbation (FEP) calculations. Corrects thermodynamic inconsistencies and detects outliers in FEP perturbation graphs.

## Codebase Map

```
src/maple_fep/
├── __init__.py              # Package root (v0.1.1)
├── models/                  # Estimators for FEP analysis
│   ├── base.py              # BaseEstimator ABC (fit, get_results, add_predictions_to_dataset)
│   ├── config.py            # Pydantic configs + enums + create_config factory
│   │                        # VariationalEstimatorConfig, MixtureVIConfig,
│   │                        # GaussianMixtureVIConfig (deprecated),
│   │                        # CycleClosureCorrectionConfig, SpectralCorrectionConfig
│   │                        # Enums: PriorType, GuideType, ErrorDistributionType
│   ├── graph_data.py        # Shared GraphData dataclass
│   ├── presets.py           # SHIM (deprecated direct import) → probabilistic/mixture/presets
│   ├── probabilistic/
│   │   ├── variational_estimator.py  # VariationalEstimator - MAP/VI via Pyro SVI
│   │   ├── gaussian_mixture_vi.py    # SHIM → mixture/{estimator,_legacy}
│   │   └── mixture/                  # Mixture VI subpackage
│   │       ├── estimator.py          # MixtureVI - K-component GMM likelihood + prior
│   │       ├── presets.py            # GMMParams, DatasetPreset, _PRESETS, get_preset, ...
│   │       ├── fitting.py            # fit_sklearn_gmm, select_gmm_components, fit_gmm_cv
│   │       └── _legacy.py            # GaussianMixtureVI alias (DeprecationWarning)
│   └── deterministic/
│       ├── cycle_closure.py          # CycleClosureCorrection - weighted cycle closure
│       └── spectral_correction.py    # SpectralCorrection - graph Laplacian (SFC/WSFC)
├── dataset/
│   ├── base_dataset.py      # BaseDataset (ABC)
│   ├── dataset.py           # FEPDataset - load from CSV/DataFrames; Exp.* columns optional
│   ├── FEP_benchmark_dataset.py  # FEPBenchmarkDataset - public benchmarks
│   └── synthetic_dataset.py # SyntheticFEPDataset - test data generation
├── graph_analysis/
│   ├── performance_stats.py # calculate_rmse, calculate_mae, bootstrap_statistic
│   ├── plotting_performance.py  # plot_dataset_DGs, plot_model_comparison_*
│   ├── graph_setup.py       # GraphSetup - network construction
│   └── graph_cycle_analysis.py  # GraphCycleAnalysis - cycle detection
└── utils/
    ├── parameter_sweep.py       # ParameterSweep, create_*_sweep_experiment
    │                            # (create_gmvi_prior_sweep_experiment is deprecated)
    ├── performance_tracker.py   # PerformanceTracker, ModelRun
    └── sklearn_gmm_fitting.py   # SHIM (deprecated direct import) → probabilistic/mixture/fitting

tests/
├── conftest.py              # Shared fixtures: mock_dataset, sample_*_data
├── test_node_model.py       # VariationalEstimator unit tests
├── test_gmvi_model.py       # MixtureVI unit tests (incl. deprecated alias coverage)
├── test_wsfc_model.py       # SpectralCorrection unit tests
├── test_datasets.py         # Dataset loading tests
├── test_integration.py      # End-to-end workflow tests
├── test_sklearn_gmm_fitting.py  # GMM fitting helper tests
└── test_performance_stats.py

examples/
├── MAPLE_demo.ipynb         # Main demo notebook
├── benchmark_all_datasets.py
├── fit_prior_gmm.ipynb      # Custom-fit GMM tutorial (use only with experimental data)
├── gen_openfe_inference.py  # MixtureVI usage demo (3 modes)
└── parameter_optimization_example.py
```

## MixtureVI usage modes

`MixtureVIConfig` exposes exactly three construction paths:

```python
MixtureVIConfig()                                           # OpenFE preset (default)
MixtureVIConfig.from_preset("cdk8")                         # Named system preset
MixtureVIConfig.from_user_data(edge_resid, node_resid,      # Custom fit
                               K_likelihood=3, K_prior=2)
```

Custom fits **require** user-supplied residuals (predicted − experimental). Use only when ground truth is available — typically when correcting a follow-up FEP map for a system already characterized experimentally. Otherwise the OpenFE preset is the recommended default.

## Quick Reference

| Task | Location |
|------|----------|
| Add new inference method | `src/maple_fep/models/` + config in `config.py`, inherit `BaseEstimator` |
| Add new prior type | Add to `PriorType` enum, implement in model class |
| Add a new dataset preset | `src/maple_fep/models/probabilistic/mixture/presets.py` (extend `_PRESETS`) |
| Test fixtures | `tests/conftest.py` - MockDataset, sample_*_data |
| Benchmark datasets | `FEPBenchmarkDataset` auto-downloads to `~/.maple_cache` |

## Commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Tests
pytest tests/ -v
pytest tests/ --cov=src/maple_fep --cov-report=html
pytest tests/test_integration.py -v  # Integration only

# Format/Lint
black src/ tests/
ruff check src/ tests/

# Docs
pip install -e ".[docs]"
cd docs && make html
```

## Local Norms

- **Python**: 3.10+ required
- **Style**: Black (88 chars), ruff for linting
- **Type hints**: Required on all public functions
- **Configs**: Use Pydantic `BaseModel` subclasses with `Field` validators
- **Tests**: Pytest; use fixtures from `conftest.py`; call `pyro.clear_param_store()` between model runs
- **Docstrings**: Google style; use Unicode for math (σ₀, ΔΔG)
- **Units**: kcal/mol throughout; mean-center node values to handle gauge freedom
- **Graph data format**: `{"N": int, "M": int, "src": Tensor, "dst": Tensor, "FEP": Tensor}`
- **CI**: GitHub Actions on `main`/`develop`; tests run on Python 3.10-3.12
- **Delete test scripts**: Remove any ad-hoc test scripts after use
- **Canonical imports**: prefer `from maple_fep.models import ...` and `from maple_fep.utils import ...`. The deeper paths (`maple_fep.models.presets`, `maple_fep.utils.sklearn_gmm_fitting`) are deprecation shims and emit `DeprecationWarning`.

## Self-Correction

1. **If the code map is stale**: Update the "Codebase Map" section to reflect current file structure and exports.

2. **If the user provides a correction**: Add it to "Local Norms" (or create a new clearly labeled section) so future sessions inherit it.
