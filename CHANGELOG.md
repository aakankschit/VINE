# Changelog

All notable changes to VINE (formerly MAPLE) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-04

### Changed
- **Renamed the project from MAPLE to VINE.** The PyPI package is now
  `vine-fep` (`pip install vine-fep`) and the import name is `vine_fep`.
  `import maple_fep` still works through a deprecated alias that returns the
  same module objects and emits a `DeprecationWarning`; it will be removed in
  0.3.0.
- The benchmark download cache defaults to `~/.vine_cache`. An existing
  `~/.maple_cache` is reused if `~/.vine_cache` does not exist yet.
- Gaussian-mixture VI was refactored into the `MixtureVI` estimator
  (`vine_fep.models.probabilistic.mixture`) with three construction modes:
  `MixtureVIConfig()` (OpenFE preset), `MixtureVIConfig.from_preset(name)`,
  and `MixtureVIConfig.from_user_data(edge_resid, node_resid, ...)`.

### Added
- Dataset presets: `GMMParams`, `DatasetPreset`, `get_preset`,
  `list_presets`, `register_preset` (exported from `vine_fep.models`).
- GMM fitting helpers: `fit_sklearn_gmm`, `fit_gmm_cv`,
  `select_gmm_components`, `residuals_from_dataset`, `GMMSelectionResult`
  (exported from `vine_fep.utils`).
- Demo notebooks reproducing the manuscript figures (`examples/demos/`).

### Deprecated
- `GaussianMixtureVI` / `GaussianMixtureVIConfig` (use `MixtureVI` /
  `MixtureVIConfig`) and the `vine_fep.models.probabilistic.gaussian_mixture_vi`
  module path. Removal planned for 0.3.0.

### Fixed
- CI coverage pointed at a non-existent `src/maple` path.
- `MANIFEST.in` and the API docs referenced the pre-rename `maple` module paths.

## [0.1.0] - 2025-02-01

### Added
- Initial release of MAPLE package
- MAP (Maximum A Posteriori) inference for FEP graphs
- VI (Variational Inference) with AutoNormal guide
- GMVI (Gaussian Mixture Variational Inference) for outlier detection
- Weighted Cycle Closure (WCC) baseline method
- FEP benchmark dataset loading utilities
- Synthetic dataset generation for testing
- Bootstrap statistics for uncertainty quantification
- Comprehensive documentation and examples

### Changed
- Separated package from paper repository

### Fixed
- N/A (initial release)
