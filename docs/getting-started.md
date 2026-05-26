# Getting Started

## Installation

```bash
pip install diff-ensemble
```

For GPU/TPU acceleration install the appropriate JAX backend **first**:

```bash
# CUDA 12 (recommended for NVIDIA GPUs)
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install diff-ensemble
```

---

## Quick Start

```python
import jax
import jax.numpy as jnp
from diff_ensemble import EnsembleVAE, save_ensemble_to_pdb

# 1. Build the model (90 residues, 32 latent dims, 100-member ensemble)
model = EnsembleVAE(seq_len=90, latent_dim=32, ensemble_size=100)
rng   = jax.random.PRNGKey(0)

# 2. Initialise parameters (one-time cost)
params = model.init(rng, jnp.ones((1, 90, 4)), rng)["params"]

# 3. Generate a structural ensemble from sequence features
sequence_features = jnp.ones((1, 90, 4))   # replace with real PSSM/one-hot
torsions, mean, logvar = model.apply({"params": params}, sequence_features, rng)

# 4. Convert torsions to 3D coordinates (N–Cα–C backbone)
coords = model.generate_coordinates(torsions)  # (100, 270, 3)

# 5. Export to a multi-model PDB for visualisation in PyMOL / VMD
save_ensemble_to_pdb(coords, "my_ensemble.pdb")
```

---

## Using the High-Level Predictor

:class:`~diff_ensemble.EnsemblePredictor` wraps trained parameters and
provides convenience methods for Rg, end-to-end distance, and
population-weighted observable averaging:

```python
from diff_ensemble import EnsemblePredictor

predictor = EnsemblePredictor(model, params)

# Generate coordinates
coords = predictor.predict(sequence_features, rng)

# Structural statistics
avg_rg, rg_per_model = predictor.compute_rg(coords)
print(f"Ensemble Rg = {avg_rg:.1f} Å")

avg_ree, _ = predictor.compute_end_to_end_distance(coords)
print(f"End-to-end distance = {avg_ree:.1f} Å")
```

---

## Training from Experimental Data

```python
from diff_ensemble.train import create_train_state, train_step
import jax.numpy as jnp
import numpy as np

# Load your experimental SAXS profile
data      = np.loadtxt("my_protein_saxs.dat", comments="#")
q_values  = jnp.array(data[:, 0])
exp_saxs  = jnp.array(data[:, 1])

# Dummy form factors (replace with real atomic scattering factors)
n_atoms      = 3 * 90   # N, Cα, C for 90 residues
form_factors = jnp.ones((n_atoms, len(q_values)))

# Set up training
state = create_train_state(model, rng, learning_rate=1e-4, input_shape=(1, 90, 4))

for step in range(1000):
    state, loss, bio, kl = train_step(
        state, sequence_features, exp_saxs, q_values, form_factors, beta=0.1
    )
    if step % 100 == 0:
        print(f"Step {step:4d} | total={loss:.4f} | SAXS={bio:.4f} | KL={kl:.4f}")
```
