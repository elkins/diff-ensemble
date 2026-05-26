import jax.numpy as jnp
from diff_biophys.ensemble import Ensemble
from diff_biophys.saxs.kernels import debye_saxs


def get_ensemble_saxs(
    coords: jnp.ndarray, q_values: jnp.ndarray, form_factors: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate ensemble-averaged SAXS intensity.

    Args:
        coords: (M, N, 3) where M is ensemble size and N is atom count.
        q_values: (Q,) q magnitudes.
        form_factors: (N, Q) atomic form factors.

    Returns:
        jnp.ndarray: (Q,) averaged intensity.
    """
    ensemble = Ensemble(coords)
    # observable_fn takes (N, 3) and returns (Q,)
    return ensemble.calculate_average(debye_saxs, q_values, form_factors)


def kld_loss(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    KL Divergence loss for VAE.
    """
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def biophysical_loss(predicted_saxs: jnp.ndarray, experimental_saxs: jnp.ndarray) -> jnp.ndarray:
    """
    MSE loss between predicted and experimental SAXS profiles.
    """
    # Normalize profiles before comparison
    pred = predicted_saxs / (predicted_saxs[0] + 1e-10)
    exp = experimental_saxs / (experimental_saxs[0] + 1e-10)
    return jnp.mean(jnp.square(pred - exp))
