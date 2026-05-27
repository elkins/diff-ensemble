# 🧬 DiffEnsemble

[![Tests](https://github.com/elkins/diff-ensemble/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/diff-ensemble/actions/workflows/test.yml)
[![Docs](https://github.com/elkins/diff-ensemble/actions/workflows/docs.yml/badge.svg)](https://elkins.github.io/diff-ensemble/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/backend-JAX-9cf.svg)](https://github.com/google/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/type%20checked-mypy-blue)](https://mypy-lang.org/)

**DiffEnsemble** is a high-performance, differentiable framework for predicting structural ensembles of Intrinsically Disordered Proteins (IDPs). By combining generative deep learning with hardware-accelerated biophysics, it bridges the gap between protein sequence and solution-state experimental data.

---

## 🚀 For Machine Learning Engineers

### Generative Architecture
DiffEnsemble utilizes a **Variational Autoencoder (VAE)** implemented in **Flax** to learn the conformational manifold of flexible proteins.
- **Encoder**: Maps sequence-derived features to a latent Gaussian distribution $(\mu, \sigma)$.
- **Latent Space**: Enables efficient sampling of diverse structural states.
- **Decoder**: Predicts a set of $(\phi, \psi)$ backbone torsions that define the protein's fold.

### End-to-End Differentiability
Built entirely on **JAX**, the entire pipeline—from the VAE weights to the final biophysical observable—is auto-differentiable.
- **Biophysical Loss**: We compute the gradient of the error between predicted ensemble-averaged spectra and experimental data (SAXS/NMR) to update model weights directly.
- **Vectorized Sampling**: Leveraging JAX's `vmap`, we generate and process ensembles of 100+ structures in parallel on GPUs/TPUs.

---

## 🧪 For Structural Biologists

### The "Ensemble" Concept
Unlike AlphaFold, which predicts a single static structure, IDPs exist as a "cloud" of interconverting conformations. DiffEnsemble predicts this **structural ensemble**, which is essential for understanding proteins that do not have a stable fold.

### Differentiable Physics Engine
We use the **NeRF (Natural Extension Reference Frame)** algorithm to convert predicted torsions into 3D Cartesian coordinates. These coordinates are then passed to **DiffBiophys** kernels to calculate:
- **SAXS**: Small-Angle X-ray Scattering profiles via the Debye formula.
- **NMR**: Residual Dipolar Couplings (RDCs) and Chemical Shifts.

### Scientific Validation
DiffEnsemble is rigorously validated against peer-reviewed standards:
- **Sic1 Benchmark**: Recapitulates the dimensions of the Sic1 IDP as determined by the **Forman-Kay Group** (*JACS 2020*).
- **CASP16 T1200**: Benchmarked against the **Montelione Group's** SpA domain-linker-domain challenge using RDC Q-factors.
- **Polymer Physics**: Obeys Flory's scaling laws ($R_g \propto N^{0.588}$) for random coils in a good solvent.

---

## 🛠️ Quick Start

```python
import jax
import jax.numpy as jnp
from diff_ensemble.model import EnsembleVAE

# Initialize model (90 residues, 32 latent dims, 100 models)
model = EnsembleVAE(seq_len=90, latent_dim=32, ensemble_size=100)
rng = jax.random.PRNGKey(0)

# Generate a structural ensemble from sequence features
batch_x = jnp.ones((1, 90, 4)) # Example features
torsions, mean, logvar = model.apply({"params": params}, batch_x, rng)
coords = model.generate_coordinates(torsions) # Shape: (100, 270, 3)

# Save the cloud to a multi-model PDB for visualization
from diff_ensemble.io import save_ensemble_to_pdb
save_ensemble_to_pdb(coords, "ensemble_cloud.pdb")
```

## 📚 References

1. **Kingma & Welling (2013)**: *Auto-Encoding Variational Bayes*.
2. **Gomes et al. (2020)**: *Conformational Ensembles of an IDP Consistent with NMR, SAXS, and smFRET* (Forman-Kay Lab).
3. **McBride et al. (2025)**: *Predicting Pose Distribution of Protein Domains* (Montelione Lab).
4. **Parsons et al. (2005)**: *Practical conversion from torsion space to Cartesian space for in silico protein synthesis*. J. Comput. Chem. 26(10), 1063–1068.

## 🛠 Software Architecture

The project is structured for modularity and high-performance execution:

*   **`diff_ensemble/model.py`**: The Flax-based VAE architecture (Encoder/Decoder).
*   **`diff_ensemble/observables.py`**: Forward biophysical kernels and multi-objective loss functions.
*   **`diff_ensemble/train.py`**: The training orchestration and optimization loop using Optax.
*   **`diff_ensemble/io.py`**: PDB trajectory export and multi-model stack management.
*   **`diff_ensemble/ensemble.py`**: High-level API for population-weighted averaging.

---

## 🤝 Contributing & Support

We welcome contributions from both the Machine Learning and Structural Biology communities!
*   **Bugs/Features:** Please open an issue on the GitHub repository.
*   **Questions:** Visit our [Documentation](https://elkins.github.io/diff-ensemble/) or reach out via GitHub Discussions.

---

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🔗 Related Projects

DiffEnsemble depends on and integrates with:

- [diff-biophys](https://github.com/elkins/diff-biophys) — Differentiable JAX kernels for SAXS/NMR (core dependency)
- [synth-pdb](https://github.com/elkins/synth-pdb) — Synthetic structure generation for training data
- [synth-nmr](https://github.com/elkins/synth-nmr) — NMR observables for experimental targets
- [synth-saxs](https://github.com/elkins/synth-saxs) — SAXS profile simulation
- [TorsionTuner](https://github.com/elkins/TorsionTuner) — Single-structure refinement counterpart

---

## 📖 Citation

```bibtex
@software{diff_ensemble,
  author  = {Elkins, George},
  title   = {DiffEnsemble: Differentiable structural ensemble prediction for IDPs},
  year    = {2024},
  url     = {https://github.com/elkins/diff-ensemble},
  version = {0.1.0}
}
```
