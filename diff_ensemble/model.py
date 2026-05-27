from typing import Any, cast

import flax.linen as nn
import jax
import jax.numpy as jnp
from diff_biophys.geometry.nerf import chain_nerf

# ---------------------------------------------------------------------------
# Bond geometry constants (Å and radians) for an idealised backbone
# ---------------------------------------------------------------------------
_B_N_CA: float = 1.458  # N–Cα bond length
_B_CA_C: float = 1.525  # Cα–C bond length
_B_C_N: float = 1.329  # C–N bond length (peptide)

_A_C_N_CA: float = 2.124  # 121.7°
_A_N_CA_C: float = 1.941  # 111.2°
_A_CA_C_N: float = 2.030  # 116.3°

_OMEGA: float = jnp.pi  # Trans peptide bond

# The first three atoms in init_coords are placed manually; NeRF builds from
# atom 4 onward, so we drop the first triplet from bonds/angles/dihedrals.
_INIT_COORDS = jnp.array([[0.0, 0.0, 0.0], [1.458, 0.0, 0.0], [2.0, 1.2, 0.0]])


def build_backbone_coords(torsions: jnp.ndarray) -> jnp.ndarray:
    """Convert backbone torsions (φ, ψ) to N–Cα–C Cartesian coordinates.

    Args:
        torsions: ``(ensemble_size, seq_len, 2)`` array where the last axis is
            ``[phi, psi]`` in radians.

    Returns:
        ``(ensemble_size, seq_len * 3, 3)`` Cartesian coordinates for all
        backbone heavy atoms (N, Cα, C) in each model of the ensemble.
    """
    ensemble_size, n_res, _ = torsions.shape

    # Pre-compute the *static* geometry arrays once, outside vmap.
    # Pattern per residue: (C→N bond, N→Cα bond, Cα→C bond)
    bond_pattern = jnp.array([_B_C_N, _B_N_CA, _B_CA_C] * n_res)  # (3*n_res,)
    angle_pattern = jnp.array([_A_CA_C_N, _A_C_N_CA, _A_N_CA_C] * n_res)  # (3*n_res,)
    omega_vec = jnp.full((n_res,), _OMEGA)  # (n_res,)

    # Drop first triplet — those atoms are provided via init_coords.
    bonds = bond_pattern[3:]  # (3*n_res - 3,)
    angles = angle_pattern[3:]  # (3*n_res - 3,)

    def _build_one(torsion_pair: jnp.ndarray) -> jnp.ndarray:
        """Build one structure from ``(n_res, 2)`` torsions."""
        phi = torsion_pair[:, 0]  # (n_res,)
        psi = torsion_pair[:, 1]  # (n_res,)

        # Interleave: for each residue the dihedral order is (ω, φ, ψ).
        # Stack columns then flatten to (3*n_res,) and drop the first triplet.
        dihedrals_full = jnp.stack([omega_vec, phi, psi], axis=1).reshape(-1)
        dihedrals = dihedrals_full[3:]  # (3*n_res - 3,)

        return cast(jnp.ndarray, chain_nerf(_INIT_COORDS, bonds, angles, dihedrals))

    return cast(jnp.ndarray, jax.vmap(_build_one)(torsions))  # (ensemble_size, 3*n_res, 3)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder(nn.Module):
    """Maps sequence features to latent distribution parameters (μ, log σ²)."""

    latent_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: ``(batch_size, seq_len, features)`` sequence feature tensor.

        Returns:
            Tuple of ``(mean, logvar)`` each shaped ``(batch_size, latent_dim)``.
        """
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))

        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        mean = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mean, logvar


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """Maps latent samples to protein backbone torsions (φ, ψ)."""

    seq_len: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            z: ``(ensemble_size, latent_dim)`` latent samples.

        Returns:
            ``(ensemble_size, seq_len, 2)`` torsion angles in radians.
        """
        ensemble_size = z.shape[0]

        x = nn.Dense(self.hidden_dim)(z)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)

        # Output φ and ψ for each residue; tanh squashes to (−π, π).
        torsions = nn.Dense(self.seq_len * 2)(x)
        torsions = jnp.tanh(torsions) * jnp.pi

        return torsions.reshape((ensemble_size, self.seq_len, 2))


# ---------------------------------------------------------------------------
# EnsembleVAE
# ---------------------------------------------------------------------------


class EnsembleVAE(nn.Module):
    """Variational Autoencoder for generating protein structural ensembles.

    The model encodes sequence features into a latent Gaussian distribution,
    draws ``ensemble_size`` samples, and decodes each sample into a set of
    backbone torsion angles.  Coordinates are obtained by calling
    :func:`build_backbone_coords`.

    Args:
        seq_len: Number of residues in the protein.
        latent_dim: Dimensionality of the latent space.
        ensemble_size: Number of conformations to sample per forward pass.
        hidden_dim: Width of the hidden layers in the encoder and decoder.
    """

    seq_len: int
    latent_dim: int
    ensemble_size: int = 100
    hidden_dim: int = 256

    def setup(self) -> None:
        self.encoder = Encoder(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.decoder = Decoder(seq_len=self.seq_len, hidden_dim=self.hidden_dim)

    def __call__(self, x: jnp.ndarray, rng: Any) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass: encode → reparameterise → decode.

        Args:
            x: ``(1, seq_len, features)`` sequence features.  Batch size must
                be 1; EnsembleVAE generates diversity through the ensemble
                dimension, not the batch dimension.
            rng: JAX PRNG key used for latent sampling.

        Returns:
            Tuple of ``(torsions, mean, logvar)`` where

            * ``torsions``: ``(ensemble_size, seq_len, 2)``
            * ``mean``:     ``(latent_dim,)``
            * ``logvar``:   ``(latent_dim,)``
        """
        mean_batch, logvar_batch = self.encoder(x)  # (batch_size, latent_dim)

        # EnsembleVAE processes one sequence at a time; squeeze the batch dim
        # so broadcasting with the ensemble axis is unambiguous.
        mean = mean_batch[0]  # (latent_dim,)
        logvar = logvar_batch[0]  # (latent_dim,)

        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, (self.ensemble_size, self.latent_dim))
        z = mean + eps * std  # (ensemble_size, latent_dim) ✓

        torsions = self.decoder(z)  # (ensemble_size, seq_len, 2)
        return torsions, mean, logvar

    def generate_coordinates(self, torsions: jnp.ndarray) -> jnp.ndarray:
        """Convert backbone torsions to N–Cα–C Cartesian coordinates.

        Delegates to the module-level :func:`build_backbone_coords` function,
        which is safe to call inside or outside a JIT-compiled context.

        Args:
            torsions: ``(ensemble_size, seq_len, 2)`` torsion angles.

        Returns:
            ``(ensemble_size, seq_len * 3, 3)`` Cartesian coordinates.
        """
        return build_backbone_coords(torsions)
