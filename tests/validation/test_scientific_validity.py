"""Scientific validation tests for DiffEnsemble.

These tests exercise physical self-consistency of the generative model
(Flory scaling, Sic1 Rg) using the reference SAXS data shipped with the
package.  They do **not** require a trained model — they validate that the
random-initialised model produces physically reasonable geometry.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from diff_ensemble.model import EnsembleVAE

# Path to reference experimental data
_DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_sic1_saxs() -> tuple:
    """Load Sic1 SAXS data from the bundled .dat file.

    Returns:
        ``(q_values, intensities)`` as NumPy arrays.
    """
    dat_path = _DATA_DIR / "sic1_saxs.dat"
    data = np.loadtxt(dat_path, comments="#")
    return data[:, 0], data[:, 1]


def _compute_rg(coords: jnp.ndarray) -> float:
    """Return the ensemble-averaged Rg (Å) for ``(M, N, 3)`` coords."""
    center = jnp.mean(coords, axis=1, keepdims=True)
    sq_dist = jnp.sum((coords - center) ** 2, axis=-1)
    rg_per_model = jnp.sqrt(jnp.mean(sq_dist, axis=1))
    return float(jnp.mean(rg_per_model))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_polymer_scaling():
    """Validate Flory's scaling law: Rg ∝ N^ν with ν ≈ 0.588.

    A randomly initialised VAE should still produce backbone geometries that
    fall within the physical scaling range 0.35 < ν < 0.75.
    """
    seq_lengths = [20, 50, 100, 150]
    latent_dim = 16
    ensemble_size = 50
    rng = jax.random.PRNGKey(42)

    rgs = []
    for N in seq_lengths:
        model = EnsembleVAE(seq_len=N, latent_dim=latent_dim, ensemble_size=ensemble_size)
        params = model.init(rng, jnp.ones((1, N, 4)), rng)["params"]  # type: ignore[index]
        torsions, _, _ = model.apply({"params": params}, jnp.ones((1, N, 4)), rng)  # type: ignore[misc]
        coords = model.generate_coordinates(torsions)
        rgs.append(_compute_rg(coords))

    log_N = np.log(seq_lengths)
    log_Rg = np.log(rgs)
    slope, _ = np.polyfit(log_N, log_Rg, 1)

    print(f"Calculated Flory Exponent: {slope:.3f}")
    assert 0.35 < slope < 0.75, f"Flory exponent {slope:.3f} outside physical range [0.35, 0.75]"


def test_sic1_validation():
    """Validate DiffEnsemble against the published Sic1 Rg (~30.5 Å).

    Reference: Gomes et al., JACS 2020.
    Sic1 is 90 residues; experimental Rg ≈ 30.5 Å from SAXS.

    Also verifies that the bundled reference SAXS data is loadable and has the
    expected q-range coverage.
    """
    # --- Load and inspect the reference SAXS data -------------------------
    q_values, intensities = _load_sic1_saxs()
    assert len(q_values) > 0, "sic1_saxs.dat must be non-empty"
    assert q_values[0] < 0.05, "Data should start in the Guinier region (q < 0.05 Å⁻¹)"
    assert q_values[-1] >= 0.15, "Data should extend into the intermediate q range"
    # The profile should be monotonically decreasing (for a disordered protein)
    assert np.all(np.diff(intensities) < 0), "SAXS profile should be monotonically decreasing"

    # --- Generate ensemble from a randomly initialised model ---------------
    seq_len = 90
    model = EnsembleVAE(seq_len=seq_len, latent_dim=32, ensemble_size=100)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, seq_len, 4)), rng)["params"]  # type: ignore[index]
    torsions, _, _ = model.apply({"params": params}, jnp.ones((1, seq_len, 4)), rng)  # type: ignore[misc]
    coords = model.generate_coordinates(torsions)

    avg_rg = _compute_rg(coords)
    print(f"Predicted Sic1 Rg: {avg_rg:.2f} Å  (experimental: ~30.5 Å)")

    # A random coil of 90 residues has Rg between ~20 and 40 Å;
    # a trained model should converge to ~30.5 Å.
    assert 20.0 < avg_rg < 40.0, f"Predicted Rg {avg_rg:.1f} Å is outside physical range [20, 40]"


def test_ped_parity_placeholder():
    """Placeholder for PED database parity test.

    In a full validation, this would download PED coordinates for a benchmark
    IDP and compare DiffEnsemble observables against the deposited ensemble.
    Tracked at https://github.com/elkins/diff-ensemble/issues.
    """
    pytest.skip("PED parity test not yet implemented — see GitHub issue tracker.")
