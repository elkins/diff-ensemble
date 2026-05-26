# 🧬 DiffEnsemble

**DiffEnsemble** is a JAX-based tool for predicting structural ensembles of Intrinsically Disordered Proteins (IDPs) using Variational Autoencoders (VAEs) and differentiable biophysics.

## 🌟 Features

- **Generative Ensembles**: Predicts a cloud of conformations (e.g., 100 structures) rather than a single static structure.
- **Differentiable Biophysics**: Seamlessly integrates with `diff-biophys` to calculate ensemble-averaged SAXS and NMR observables.
- **Physics-Informed Training**: Optimize protein ensembles directly against experimental spectra via gradient descent.

## 🚀 Installation

```bash
pip install -e .
```

## 🛠️ Usage

```python
import jax.numpy as jnp
from diff_ensemble.model import EnsembleVAE

# Initialize model
model = EnsembleVAE(latent_dim=32)
# ... more usage examples coming soon ...
```

## 📖 Documentation

For full documentation, visit [DiffEnsemble Documentation](https://elkins.github.io/DiffEnsemble/).
