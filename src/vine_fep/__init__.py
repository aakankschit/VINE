"""
VINE: a framework to estimate absolute free energies from relative free
energies using Bayesian inference.

A Python package for analyzing Free Energy Perturbation (FEP) data using
probabilistic node models and Bayesian inference.
"""

__version__ = "0.2.0"
__author__ = "Aakankschit Nandkeolyar"
__email__ = "anandkeo@uci.edu"
__description__ = (
    "VINE: estimate absolute free energies from relative free energies "
    "(FEP) using Bayesian inference"
)

from . import dataset, graph_analysis, models, utils

__all__ = ["models", "dataset", "graph_analysis", "utils", "__version__"]
