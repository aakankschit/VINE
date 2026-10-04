Interactive Tutorials
=====================

These interactive tutorials demonstrate VINE's capabilities using real benchmark datasets.
The tutorials are built with `Marimo <https://marimo.io/>`_, a modern reactive Python notebook.

.. note::

   These tutorials are fully interactive. You can modify the code and see results update in real-time.
   To run the tutorials locally, install Marimo with ``pip install marimo`` and run:

   .. code-block:: bash

      marimo edit docs/tutorials/getting_started.py


Getting Started Tutorial
------------------------

This tutorial walks you through the basics of VINE:

- Loading benchmark FEP datasets
- Understanding the data structure (nodes and edges)
- Training a VariationalEstimator with Bayesian inference
- Analyzing model results and performance metrics
- Visualizing predictions vs experimental values

.. raw:: html

   <div style="margin: 2em 0;">
   <a href="getting_started.html" class="btn btn-primary" style="padding: 10px 20px; background-color: #2980b9; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">
   View Interactive Tutorial
   </a>
   </div>

The tutorial uses the **CDK8 (Cyclin-dependent kinase 8)** benchmark dataset to demonstrate:

1. **Data Loading**: How to use ``FEPDataset`` to load benchmark data
2. **Model Configuration**: Setting up ``VariationalEstimatorConfig`` with appropriate priors
3. **Training**: Running MAP inference with the ``VariationalEstimator``
4. **Analysis**: Calculating RMSE, MAE, and correlation metrics
5. **Visualization**: Creating correlation plots

Complete Model Comparison Tutorial
------------------------------------

This tutorial applies **all six correction methods** to a single benchmark dataset and compares their performance:

- **MLE** — Maximum Likelihood Estimation (uniform prior, no regularization)
- **MAP** — Maximum A Posteriori (Normal prior regularization)
- **VI** — Variational Inference (full posterior with uncertainties)
- **GMVI** — Gaussian Mixture VI (outlier-robust inference)
- **WCC** — Weighted Cycle Closure (iterative deterministic correction)
- **WSFC** — Weighted Spectral Free-energy Correction (graph Laplacian)

.. raw:: html

   <div style="margin: 2em 0;">
   <a href="model_comparison.html" class="btn btn-primary" style="padding: 10px 20px; background-color: #2980b9; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">
   View Model Comparison Tutorial
   </a>
   </div>

The tutorial covers:

1. **Data exploration**: Loading CDK8 benchmark data and visualizing the perturbation graph
2. **Model fitting**: Configuring and running all six correction methods
3. **Performance comparison**: Correlation plots, bar charts, and bootstrap statistics
4. **Outlier detection**: Using GMVI to identify problematic FEP edges
5. **Graph analysis**: Cycle closure errors and edge leverages


Source Files
------------

The Marimo notebook source files are available in the repository:

- `getting_started.py <https://github.com/aakankschit/vine-fep/blob/main/docs/tutorials/getting_started.py>`_ - Getting Started Tutorial
- `model_comparison.py <https://github.com/aakankschit/vine-fep/blob/main/docs/tutorials/model_comparison.py>`_ - Complete Model Comparison Tutorial


Running Tutorials Locally
-------------------------

To run tutorials interactively:

1. Install VINE with notebook dependencies:

   .. code-block:: bash

      pip install vine-fep[notebook]

2. Navigate to the tutorials directory:

   .. code-block:: bash

      cd docs/tutorials

3. Start Marimo:

   .. code-block:: bash

      marimo edit model_comparison.py

This will open an interactive notebook in your browser where you can modify code and see results update reactively.
