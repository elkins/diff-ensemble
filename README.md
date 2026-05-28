# 🧬 DiffEnsemble: Differentiable IDP Ensemble Prediction

[![PyPI version](https://img.shields.io/pypi/v/diff-ensemble.svg)](https://pypi.org/project/diff-ensemble/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/diff-ensemble.svg)](https://pypi.org/project/diff-ensemble/)
[![Tests](https://github.com/elkins/diff-ensemble/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/diff-ensemble/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/Accelerated_by-JAX-blue.svg)](https://github.com/google/jax)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DiffEnsemble is a JAX-powered framework for predicting structural ensembles of Intrinsically Disordered Proteins (IDPs) using a Variational Autoencoder (VAE) coupled with differentiable biophysical observables.

---

### 🧪 For Structural Biologists
*   **Ensemble Averaging:** Automatically calculates ensemble-averaged SAXS profiles and NMR observables.
*   **Disorder Recovery:** Specifically designed for proteins that don't have a single "fixed" structure, providing a statistical view of the conformational landscape.

### 🤖 For Machine Learning Geeks
*   **VAE-Physics Integration:** A latent-space generative model where the reconstruction loss is a combination of latent KLD and physical observables (SAXS/NMR).
*   **Differentiable Torsions:** Maps latent vectors to 3D coordinates via a differentiable NeRF (Natural Extension Reference Frame) implementation.

---

## 🚀 Key Features

*   **JAX-Accelerated VAE:** High-performance training of generative models for IDPs.
*   **Debye-Based SAXS Prediction:** Differentiable back-calculation of SAXS profiles from structural ensembles.
*   **Latent Space Exploration:** Sample new conformations from the learned disordered landscape.

## 📦 Installation

```bash
pip install diff-ensemble
```

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
