from typing import Any, cast

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from diff_ensemble.model import EnsembleVAE, build_backbone_coords
from diff_ensemble.observables import biophysical_loss, get_ensemble_saxs, kld_loss


class TrainState(train_state.TrainState):
    """Extended Flax TrainState that carries the PRNG key for sampling."""

    key: Any


def create_train_state(
    model: EnsembleVAE,
    rng: Any,
    learning_rate: float,
    input_shape: tuple[int, ...],
) -> TrainState:
    """Initialise model parameters and return an initial :class:`TrainState`.

        Args:
            model: An :class:`~diff_ensemble.model.EnsembleVAE` instance.
            rng: JAX PRNG key.
            learning_rate: Adam learning rate.
            input_shape: Shape of a single input batch, e.g. ``(1, seq_len, 4)``.

    from typing import Any, cast
    ...
        Returns:
            Initialised :class:`TrainState`.
    """
    init_key, step_key = jax.random.split(rng)
    params = model.init(init_key, jnp.ones(input_shape), init_key)["params"]
    tx = optax.adam(learning_rate)
    return cast(
        TrainState,
        TrainState.create(apply_fn=model.apply, params=params, tx=tx, key=step_key),
    )


@jax.jit
def train_step(
    state: TrainState,
    batch_x: jnp.ndarray,
    exp_saxs: jnp.ndarray,
    q_values: jnp.ndarray,
    form_factors: jnp.ndarray,
    beta: float = 0.1,
) -> tuple[TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform a single gradient update step.

    The PRNG key stored in ``state`` is consumed for latent sampling, and the
    state is returned with a freshly split key ready for the next step.

    Args:
        state: Current :class:`TrainState`.
        batch_x: ``(1, seq_len, features)`` sequence feature tensor.
        exp_saxs: ``(Q,)`` experimental SAXS intensities.
        q_values: ``(Q,)`` scattering vector magnitudes.
        form_factors: ``(N_atoms, Q)`` atomic form factors.
        beta: KL-divergence weight in the ELBO loss.

    Returns:
        ``(new_state, total_loss, bio_loss, kl_loss)``
    """
    # Split *before* use so each step consumes a unique key.
    step_key, next_key = jax.random.split(state.key)

    def loss_fn(params: Any) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
        torsions, mean, logvar = state.apply_fn({"params": params}, batch_x, step_key)

        # Use the module-level function directly — no need to re-instantiate the
        # model inside the loss closure, keeping gradient flow clean and explicit.
        coords = build_backbone_coords(torsions)

        pred_saxs = get_ensemble_saxs(coords, q_values, form_factors)

        bio_loss = biophysical_loss(pred_saxs, exp_saxs)
        kl_loss = kld_loss(mean, logvar)

        total_loss = bio_loss + beta * kl_loss
        return total_loss, (bio_loss, kl_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (bio_loss, kl_loss)), grads = grad_fn(state.params)

    state = state.apply_gradients(grads=grads)
    state = state.replace(key=next_key)

    return state, loss, bio_loss, kl_loss
