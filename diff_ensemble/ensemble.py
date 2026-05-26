"""High-level API for ensemble prediction and population-weighted averaging.

:class:`EnsemblePredictor` wraps a trained :class:`~diff_ensemble.model.EnsembleVAE`
and provides a convenient interface for:

* Generating structural ensembles from sequence features.
* Computing population-weighted ensemble averages of arbitrary observables.
* Computing standard structural statistics (Rg, end-to-end distance).
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from diff_ensemble.model import EnsembleVAE, build_backbone_coords


class EnsemblePredictor:
    """High-level wrapper for a trained :class:`EnsembleVAE`.

    Args:
        model: An :class:`~diff_ensemble.model.EnsembleVAE` instance.
        params: Trained Flax parameter dict (from ``model.init`` or a
            checkpoint).

    Example::

        import jax
        from diff_ensemble import EnsembleVAE, EnsemblePredictor

        model = EnsembleVAE(seq_len=90, latent_dim=32, ensemble_size=100)
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.ones((1, 90, 4)), rng)["params"]

        predictor = EnsemblePredictor(model, params)
        coords = predictor.predict(jnp.ones((1, 90, 4)), rng)
        print(predictor.compute_rg(coords))
    """

    def __init__(self, model: EnsembleVAE, params: Dict[str, Any]) -> None:
        self.model = model
        self.params = params

    # ------------------------------------------------------------------
    # Ensemble generation
    # ------------------------------------------------------------------

    def predict(
        self,
        sequence_features: jnp.ndarray,
        rng: Any,
        n_samples: Optional[int] = None,
    ) -> jnp.ndarray:
        """Generate a structural ensemble from sequence features.

        Args:
            sequence_features: ``(1, seq_len, features)`` input tensor.
            rng: JAX PRNG key.
            n_samples: If provided, overrides the model's default
                ``ensemble_size`` by creating a temporary model variant.
                When ``None`` the model's ``ensemble_size`` is used.

        Returns:
            ``(ensemble_size, seq_len * 3, 3)`` Cartesian coordinates for all
            backbone atoms (N, Cα, C) in each generated conformation.
        """
        if n_samples is not None and n_samples != self.model.ensemble_size:
            # Create a lightweight variant with the requested ensemble size.
            tmp_model = EnsembleVAE(
                seq_len=self.model.seq_len,
                latent_dim=self.model.latent_dim,
                ensemble_size=n_samples,
                hidden_dim=self.model.hidden_dim,
            )
            out = tmp_model.apply({"params": self.params}, sequence_features, rng)
        else:
            out = self.model.apply({"params": self.params}, sequence_features, rng)

        torsions, _, _ = out  # type: ignore[misc]  # Flax apply return is untyped
        return build_backbone_coords(torsions)

    # ------------------------------------------------------------------
    # Population-weighted averaging
    # ------------------------------------------------------------------

    def compute_population_average(
        self,
        observable_fn: Callable[..., jnp.ndarray],
        coords: jnp.ndarray,
        weights: Optional[jnp.ndarray] = None,
        *args: Any,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Compute a population-weighted ensemble average of an observable.

        Args:
            observable_fn: A function with signature
                ``(coords_single: (N, 3), *args, **kwargs) -> (...)`` that
                computes the observable for a single conformation.
            coords: ``(M, N, 3)`` ensemble coordinates.
            weights: ``(M,)`` population weights (must sum to 1).  Defaults to
                uniform weights (``1/M`` each).
            *args: Extra positional arguments forwarded to ``observable_fn``.
            **kwargs: Extra keyword arguments forwarded to ``observable_fn``.

        Returns:
            Observable averaged over the ensemble.
        """
        ensemble_size = coords.shape[0]

        if weights is None:
            weights = jnp.ones(ensemble_size) / ensemble_size

        # vmap over the ensemble dimension.
        per_model_obs = jax.vmap(lambda c: observable_fn(c, *args, **kwargs))(coords)
        # Weighted average: weights shape (M,) broadcast over observable dims.
        return jnp.average(per_model_obs, axis=0, weights=weights)

    # ------------------------------------------------------------------
    # Structural statistics
    # ------------------------------------------------------------------

    def compute_rg(
        self,
        coords: jnp.ndarray,
        weights: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the ensemble-averaged radius of gyration.

        Args:
            coords: ``(M, N, 3)`` ensemble coordinates.
            weights: ``(M,)`` population weights.  Defaults to uniform.

        Returns:
            ``(avg_rg, rg_per_model)`` — the population-weighted mean Rg and
            the per-model Rg array (both in Ångströms).
        """
        ensemble_size = coords.shape[0]
        if weights is None:
            weights = jnp.ones(ensemble_size) / ensemble_size

        center = jnp.mean(coords, axis=1, keepdims=True)  # (M, 1, 3)
        sq_dist = jnp.sum((coords - center) ** 2, axis=-1)  # (M, N)
        rg_per_model = jnp.sqrt(jnp.mean(sq_dist, axis=1))  # (M,)
        avg_rg = jnp.average(rg_per_model, weights=weights)

        return avg_rg, rg_per_model

    def compute_end_to_end_distance(
        self,
        coords: jnp.ndarray,
        weights: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute the ensemble-averaged end-to-end distance.

        Measures the distance between the N-terminal N atom (index 0) and the
        C-terminal C atom (last atom) of each backbone model.

        Args:
            coords: ``(M, N, 3)`` ensemble coordinates.
            weights: ``(M,)`` population weights.  Defaults to uniform.

        Returns:
            ``(avg_ree, ree_per_model)`` — weighted mean R_ee and per-model
            values (both in Ångströms).
        """
        ensemble_size = coords.shape[0]
        if weights is None:
            weights = jnp.ones(ensemble_size) / ensemble_size

        ree_per_model = jnp.linalg.norm(coords[:, -1, :] - coords[:, 0, :], axis=-1)
        avg_ree = jnp.average(ree_per_model, weights=weights)

        return avg_ree, ree_per_model
