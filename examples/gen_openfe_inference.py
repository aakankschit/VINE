"""
MixtureVI demo: three usage paths.

Demonstrates how end users invoke ``MixtureVI`` after the package
restructure:

1. **Default** — uses the OpenFE preset (recommended for users without
   experimental ground truth).
2. **Named system preset** — e.g. ``"cdk8"`` for users on a specific
   protein system.
3. **Custom-fit** — only valid when the user has their own experimental
   data; fits GMMs on user residuals.

Plus a short demonstration of the underlying ``fit_sklearn_gmm`` helper.

Usage::

    python gen_openfe_inference.py
"""

from time import time

import numpy as np

from maple_fep.dataset import SyntheticFEPDataset
from maple_fep.models import MixtureVI, MixtureVIConfig, get_preset, list_presets
from maple_fep.utils import fit_sklearn_gmm


def run_default(dataset):
    """OpenFE preset (no arguments)."""
    print("\n" + "=" * 60)
    print("Mode 1: Default — OpenFE preset (no experimental data needed)")
    print("=" * 60)

    config = MixtureVIConfig(
        kl_weight=1.0,
        n_epochs=500,
        n_samples=100,
        patience=100,
        random_seed=42,
    )
    model = MixtureVI(config=config, dataset=dataset)
    start = time()
    model.fit()
    print(f"Training time: {time() - start:.1f}s")

    results = model.get_results()
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Number of nodes: {results['num_nodes']}")
    print(f"Number of edges: {results['num_edges']}")
    print(f"Column suffix: {model._suffix}")  # VI(3,3) for OpenFE preset
    model.add_predictions_to_dataset()
    return model


def run_named_preset(dataset, name="cdk8"):
    """Use a named system-specific preset."""
    print("\n" + "=" * 60)
    print(f"Mode 2: Named system preset — {name!r}")
    print("=" * 60)

    config = MixtureVIConfig.from_preset(
        name,
        kl_weight=1.0,
        n_epochs=500,
        n_samples=100,
        patience=100,
        random_seed=42,
    )
    model = MixtureVI(config=config, dataset=dataset)
    start = time()
    model.fit()
    print(f"Training time: {time() - start:.1f}s")
    print(f"Column suffix: {model._suffix}")
    model.add_predictions_to_dataset()
    return model


def run_custom_fit(dataset):
    """Custom fit on user-supplied residuals.

    Only valid when the user has experimental ground truth — typically
    from a prior FEP map on the same protein system.
    """
    print("\n" + "=" * 60)
    print("Mode 3: Custom fit — requires user-supplied residuals")
    print("=" * 60)

    # In a real run these would come from a prior dataset where
    # experimental DDG values were known. Here we synthesise them.
    rng = np.random.default_rng(0)
    edge_residuals = np.concatenate([
        rng.normal(-1.0, 0.5, 50),
        rng.normal(1.0, 0.5, 50),
    ])
    node_residuals = rng.normal(0.0, 1.0, 30)
    node_residuals -= node_residuals.mean()

    config = MixtureVIConfig.from_user_data(
        edge_residuals,
        node_residuals,
        K_likelihood=3,
        K_prior=2,
        random_state=42,
        kl_weight=1.0,
        n_epochs=500,
        n_samples=100,
        patience=100,
    )
    model = MixtureVI(config=config, dataset=dataset)
    start = time()
    model.fit()
    print(f"Training time: {time() - start:.1f}s")
    print(f"Column suffix: {model._suffix}")
    model.add_predictions_to_dataset()
    return model


def demo_gmm_fitting():
    """Show the underlying ``fit_sklearn_gmm`` helper."""
    print("\n" + "=" * 60)
    print("Demo: Underlying GMM fitting with fit_sklearn_gmm()")
    print("=" * 60)

    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(-1.0, 0.5, 200),
        np.random.normal(2.0, 1.0, 100),
    ])

    params = fit_sklearn_gmm(data, n_components=2, random_state=42)
    print(f"Fitted {params.n_components} components:")
    for k in range(params.n_components):
        print(f"  Component {k}: mean={params.means[k]:.3f}, "
              f"var={params.variances[k]:.3f}, weight={params.weights[k]:.3f}")
    return params


def main():
    print("Available presets:", list_presets())

    dataset = SyntheticFEPDataset(add_noise=True)
    print(f"\nSynthetic dataset: {dataset.cycle_data['N']} nodes, "
          f"{dataset.cycle_data['M']} edges")

    run_default(dataset)
    run_named_preset(dataset, name="cdk8")
    run_custom_fit(dataset)
    demo_gmm_fitting()

    print("\n" + "=" * 60)
    print("Summary of estimators added to dataset:")
    print("=" * 60)
    print(f"Estimators: {dataset.estimators}")
    print(f"\nNode columns: {list(dataset.dataset_nodes.columns)}")
    print(f"Edge columns: {list(dataset.dataset_edges.columns)}")


if __name__ == "__main__":
    main()
