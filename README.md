<p align="center">
  <img src="https://raw.githubusercontent.com/aakankschit/VINE/main/docs/VINE_logo.png" alt="VINE Logo" width="400"/>
</p>

<h1 align="center">VINE</h1>

<p align="center">
  <strong>A framework to estimate absolute free energies from relative free energies using Bayesian inference</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/vine-fep/"><img src="https://img.shields.io/pypi/v/vine-fep.svg" alt="PyPI version"></a>
  <a href="https://github.com/aakankschit/VINE/actions/workflows/ci.yml"><img src="https://github.com/aakankschit/VINE/actions/workflows/ci.yml/badge.svg" alt="CI/CD Pipeline"></a>
  <a href="https://vine-fep.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/vine-fep/badge/?version=latest" alt="Documentation Status"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

A Python package for Bayesian analysis of Free Energy Perturbation (FEP) calculations in computational drug discovery. VINE provides probabilistic and deterministic inference methods to correct thermodynamic inconsistencies and detect outliers in FEP perturbation graphs.

## Installation

```bash
# From PyPI
pip install vine-fep

# From source (development)
git clone https://github.com/aakankschit/VINE.git
cd VINE
pip install -e ".[dev]"
```

## Quick Start

```python
from vine_fep.models import GaussianMixtureVI, GaussianMixtureVIConfig
from vine_fep.dataset import FEPDataset

# Load your FEP data
dataset = FEPDataset("fep_edges.csv")

# Configure the GMVI model (Gaussian Mixture Variational Inference)
config = GaussianMixtureVIConfig(
    prior_std=5.0,       # Prior uncertainty on node values
    normal_std=1.0,      # Expected error for normal edges
    outlier_std=3.0,     # Expected error for outlier edges
    outlier_prob=0.2,    # Prior probability of an edge being an outlier
    learning_rate=0.01,
    n_epochs=1000
)

# Train and get results
model = GaussianMixtureVI(config=config, dataset=dataset)
model.fit()

# Get node estimates (absolute free energies)
results = model.get_results()
node_estimates = results["node_estimates"]

# Identify problematic edges
outlier_probs = model.compute_edge_outlier_probabilities()
```

## Key Features

- **Probabilistic Inference Methods**
  - **MAP**: Maximum A Posteriori estimation for regularized point estimates
  - **MLE**: Maximum Likelihood Estimation (unregularized)
  - **VI**: Variational Inference with full uncertainty quantification
  - **GMVI**: Gaussian Mixture VI for automatic outlier detection

- **Deterministic Graph Methods**
  - **WCC**: Weighted Cycle Closure for thermodynamic consistency enforcement
  - **WSFC/SFC**: Weighted Spectral Free-energy Correction via graph Laplacian pseudoinverse

- **Outlier Detection**
  - Probabilistic identification of problematic FEP edges
  - Mixture models for robust estimation

- **Uncertainty Quantification**
  - Full posterior distributions with confidence intervals
  - Bootstrap statistics for performance metrics
  - Laplacian-based uncertainties for spectral methods

## Methods at a Glance

| Method | Class | Type | Key Idea | Outputs |
|--------|-------|------|----------|---------|
| MAP | `VariationalEstimator` | Probabilistic | Regularized least squares via Bayesian prior | Point estimates |
| MLE | `VariationalEstimator` | Probabilistic | Ordinary least squares (uniform prior) | Point estimates |
| VI | `VariationalEstimator` | Probabilistic | Variational posterior approximation | Estimates + uncertainties |
| GMVI | `GaussianMixtureVI` | Probabilistic | Mixture likelihood for outlier robustness | Estimates + uncertainties + outlier scores |
| WCC | `CycleClosureCorrection` | Deterministic | Iterative cycle closure error correction | Corrected estimates + uncertainties |
| WSFC | `SpectralCorrection` | Deterministic | Graph Laplacian pseudoinverse: $z^* = L_W^+ B^T W x$ | Estimates + uncertainties |
| SFC | `SpectralCorrection` | Deterministic | Unweighted spectral correction (equivalent to MLE) | Estimates + uncertainties |

## Core Modules

| Module | Description |
|--------|-------------|
| `vine_fep.models.probabilistic` | Bayesian estimators: `VariationalEstimator` (MAP/VI/MLE), `GaussianMixtureVI` |
| `vine_fep.models.deterministic` | Graph-based methods: `CycleClosureCorrection` (WCC), `SpectralCorrection` (WSFC/SFC) |
| `vine_fep.models.config` | Pydantic configuration classes for all models |
| `vine_fep.dataset` | Dataset loading (`FEPDataset`), benchmarks (`FEPBenchmarkDataset`), synthetic data |
| `vine_fep.graph_analysis` | Performance statistics, visualization, graph construction, cycle analysis |
| `vine_fep.utils` | Parameter optimization (`ParameterSweep`), performance tracking |

## Documentation

- **[User Guide](https://vine-fep.readthedocs.io/)**: Installation, tutorials, examples
- **[API Reference](https://vine-fep.readthedocs.io/api/)**: Detailed module documentation
- **[Examples](examples/)**: Jupyter notebooks and scripts

Build documentation locally:
```bash
pip install -e ".[docs]"
cd docs && make html
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=src/vine_fep --cov-report=html

# Format code
black src/ tests/
ruff check src/ tests/
```

## References

VINE implements methods from the following publications:

- **WCC**: Li, Y.; Liu, R.; Liu, J.; Luo, H.; Wu, C.; Li, Z. An Open Source Graph-Based Weighted Cycle Closure Method for Relative Binding Free Energy Calculations. *J. Chem. Inf. Model.* **2023**, *63*, 561--570. [DOI: 10.1021/acs.jcim.2c01076](https://doi.org/10.1021/acs.jcim.2c01076)

- **SFC/WSFC**: Liu, R.; Lai, Y.; Yao, Y.; Huang, W.; Zhong, Y.; Luo, H.-B.; Li, Z. State Function-Based Correction: A Simple and Efficient Free-Energy Correction Algorithm for Large-Scale Relative Binding Free-Energy Calculations. *J. Phys. Chem. Lett.* **2025**, *16*, 5763--5768. [DOI: 10.1021/acs.jpclett.5c01119](https://doi.org/10.1021/acs.jpclett.5c01119)

## Citation

If you use VINE in your research, please cite:

```bibtex
@software{vine_fep_2025,
  title={VINE: A framework to estimate absolute free energies from relative free energies using Bayesian inference},
  author={Nandkeolyar, Aakankschit},
  year={2025},
  url={https://github.com/aakankschit/VINE},
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
