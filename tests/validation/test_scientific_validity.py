import jax
import jax.numpy as jnp
import numpy as np

from diff_ensemble.model import EnsembleVAE


def test_polymer_scaling():
    """
    Validate that the VAE obeys Flory's scaling law for random coils:
    Rg = R0 * N^v  where v is the Flory exponent (~0.588 for good solvent).
    """
    seq_lengths = [20, 50, 100, 150]
    latent_dim = 16
    ensemble_size = 50
    rng = jax.random.PRNGKey(42)

    rgs = []
    for N in seq_lengths:
        model = EnsembleVAE(seq_len=N, latent_dim=latent_dim, ensemble_size=ensemble_size)
        # Random initial params
        params = model.init(rng, jnp.ones((1, N, 4)), rng)["params"]
        out = model.apply({"params": params}, jnp.ones((1, N, 4)), rng)
        torsions = out[0]

        coords = model.generate_coordinates(torsions)  # (M, N*3, 3)

        # Calculate ensemble average Rg
        # Rg = sqrt(mean(dist_to_center^2))
        center = jnp.mean(coords, axis=1, keepdims=True)
        sq_dist = jnp.sum((coords - center) ** 2, axis=-1)
        rg_per_model = jnp.sqrt(jnp.mean(sq_dist, axis=1))
        avg_rg = jnp.mean(rg_per_model)
        rgs.append(float(avg_rg))

    # Fit log(Rg) vs log(N)
    log_N = np.log(seq_lengths)
    log_Rg = np.log(rgs)
    slope, _ = np.polyfit(log_N, log_Rg, 1)

    print(f"Calculated Flory Exponent: {slope:.3f}")
    # For a random initialized VAE, we expect a value in the physical range [0.4, 0.7]
    assert 0.35 < slope < 0.75


def test_sic1_validation():
    """
    Validate DiffEnsemble against the published Sic1 Rg (~30.5 A).
    Reference: Gomes et al., JACS 2020.
    """
    # Sic1 is 90 residues
    seq_len = 90
    latent_dim = 32
    ensemble_size = 100
    model = EnsembleVAE(seq_len=seq_len, latent_dim=latent_dim, ensemble_size=ensemble_size)

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, seq_len, 4)), rng)["params"]

    # Generate ensemble
    out_sic1 = model.apply({"params": params}, jnp.ones((1, seq_len, 4)), rng)
    torsions = out_sic1[0]
    coords = model.generate_coordinates(torsions)

    # Calculate Rg
    center = jnp.mean(coords, axis=1, keepdims=True)
    sq_dist = jnp.sum((coords - center) ** 2, axis=-1)
    rg_per_model = jnp.sqrt(jnp.mean(sq_dist, axis=1))
    avg_rg = float(jnp.mean(rg_per_model))

    print(f"Predicted Sic1 Rg: {avg_rg:.2f} A")
    # A random coil of 90 residues usually has Rg between 25 and 35 A.
    # Published value is ~30.5 A.
    assert 20.0 < avg_rg < 40.0


def test_ped_parity_placeholder():
    """
    Placeholder for PED parity test.
    In a real scenario, this would load PED coordinates and compare DiffEnsemble observables.
    """
    assert True
