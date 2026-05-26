"""Biophysical forward kernels and loss functions for DiffEnsemble.

All functions operate on JAX arrays and are end-to-end differentiable,
enabling gradient-based optimisation through experimental observables.
"""

import jax.numpy as jnp
from diff_biophys.ensemble import Ensemble
from diff_biophys.saxs.kernels import debye_saxs

# ---------------------------------------------------------------------------
# SAXS observable
# ---------------------------------------------------------------------------


def get_ensemble_saxs(
    coords: jnp.ndarray,
    q_values: jnp.ndarray,
    form_factors: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate ensemble-averaged SAXS intensity via the Debye formula.

    Args:
        coords: ``(M, N, 3)`` coordinates where *M* is ensemble size and *N*
            is atom count.
        q_values: ``(Q,)`` scattering vector magnitudes in Å⁻¹.
        form_factors: ``(N, Q)`` atomic form factors.

    Returns:
        ``(Q,)`` ensemble-averaged intensity.
    """
    ensemble = Ensemble(coords)
    # observable_fn signature: (N, 3) → (Q,)
    return ensemble.calculate_average(debye_saxs, q_values, form_factors)


# ---------------------------------------------------------------------------
# NMR observables (stubs — require diff_biophys.nmr kernels)
# ---------------------------------------------------------------------------


def get_ensemble_rdc(
    coords: jnp.ndarray,
    bond_vectors: jnp.ndarray,
    alignment_tensor: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate ensemble-averaged Residual Dipolar Couplings (RDCs).

    .. note::
        This function requires the ``diff_biophys.nmr.rdc`` kernel, which is
        planned for a future release.  Calling it will raise
        :exc:`NotImplementedError` until the kernel is available.

    Args:
        coords: ``(M, N, 3)`` ensemble coordinates.
        bond_vectors: ``(K, 2)`` atom-index pairs defining each NH (or other)
            bond for which RDCs are measured.
        alignment_tensor: ``(3, 3)`` Saupe order tensor *A* for the alignment
            medium used in the experiment.

    Returns:
        ``(K,)`` ensemble-averaged RDC values in Hz.

    Raises:
        NotImplementedError: Until the ``diff_biophys.nmr`` module is released.
    """
    raise NotImplementedError(
        "RDC calculation requires the diff_biophys.nmr.rdc kernel, "
        "which is not yet available.  Track progress at "
        "https://github.com/elkins/diff-ensemble/issues."
    )


def get_ensemble_chemical_shifts(
    coords: jnp.ndarray,
    sequence: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate ensemble-averaged backbone chemical shifts.

    .. note::
        This function requires the ``diff_biophys.nmr.shifts`` kernel, which is
        planned for a future release.  Calling it will raise
        :exc:`NotImplementedError` until the kernel is available.

    Args:
        coords: ``(M, N, 3)`` ensemble coordinates.
        sequence: ``(seq_len,)`` integer-encoded amino-acid sequence
            (0 = ALA, …, 19 = VAL).

    Returns:
        ``(seq_len, 6)`` ensemble-averaged chemical shifts for backbone nuclei
        (Hα, Hβ, C, Cα, Cβ, N) in ppm.

    Raises:
        NotImplementedError: Until the ``diff_biophys.nmr`` module is released.
    """
    raise NotImplementedError(
        "Chemical shift calculation requires the diff_biophys.nmr.shifts kernel, "
        "which is not yet available.  Track progress at "
        "https://github.com/elkins/diff-ensemble/issues."
    )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def kld_loss(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """KL divergence between the encoder posterior and a unit Gaussian prior.

    Uses the closed-form expression for Gaussian KL:
    ``-0.5 * Σ (1 + log σ² − μ² − σ²)``.

    Args:
        mean: ``(latent_dim,)`` posterior mean.
        logvar: ``(latent_dim,)`` posterior log-variance.

    Returns:
        Scalar KL divergence (non-negative).
    """
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


def biophysical_loss(
    predicted_saxs: jnp.ndarray,
    experimental_saxs: jnp.ndarray,
) -> jnp.ndarray:
    """Normalised MSE between predicted and experimental SAXS profiles.

    Both profiles are normalised to their first data point (I(q=0) = 1) before
    comparison, making the loss invariant to absolute intensity scale.

    Args:
        predicted_saxs: ``(Q,)`` predicted SAXS intensities.
        experimental_saxs: ``(Q,)`` experimental SAXS intensities.

    Returns:
        Scalar mean-squared error after normalisation.
    """
    pred = predicted_saxs / (predicted_saxs[0] + 1e-10)
    exp = experimental_saxs / (experimental_saxs[0] + 1e-10)
    return jnp.mean(jnp.square(pred - exp))
