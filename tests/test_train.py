import jax
import jax.numpy as jnp

from diff_ensemble.model import EnsembleVAE
from diff_ensemble.train import create_train_state, train_step


def test_initialization():
    seq_len = 10
    latent_dim = 16
    ensemble_size = 5
    model = EnsembleVAE(seq_len=seq_len, latent_dim=latent_dim, ensemble_size=ensemble_size)

    rng = jax.random.PRNGKey(0)
    input_shape = (1, seq_len, 4)  # 4 features per residue

    state = create_train_state(model, rng, learning_rate=1e-3, input_shape=input_shape)
    assert state.params is not None


def test_gradient_flow():
    seq_len = 10
    latent_dim = 16
    ensemble_size = 5
    model = EnsembleVAE(seq_len=seq_len, latent_dim=latent_dim, ensemble_size=ensemble_size)

    rng = jax.random.PRNGKey(0)
    batch_x = jnp.ones((1, seq_len, 4))

    # Dummy experimental data
    num_atoms = 3 * seq_len
    q_values = jnp.linspace(0.01, 0.5, 20)
    exp_saxs = jnp.ones(20)
    form_factors = jnp.ones((num_atoms, 20))

    state = create_train_state(model, rng, learning_rate=1e-3, input_shape=batch_x.shape)

    state, loss, bio_loss, kl_loss = train_step(state, batch_x, exp_saxs, q_values, form_factors)

    assert loss > 0
    assert bool(jnp.isfinite(loss))


class TestTrainStepRng:
    def _make_state(self):
        seq_len, latent_dim, ensemble_size = 10, 16, 5
        model = EnsembleVAE(seq_len=seq_len, latent_dim=latent_dim, ensemble_size=ensemble_size)
        rng = jax.random.PRNGKey(0)
        batch_x = jnp.ones((1, seq_len, 4))
        q_values = jnp.linspace(0.01, 0.5, 20)
        exp_saxs = jnp.ones(20)
        form_factors = jnp.ones((3 * seq_len, 20))
        state = create_train_state(model, rng, learning_rate=1e-3, input_shape=batch_x.shape)
        return state, batch_x, exp_saxs, q_values, form_factors

    def test_key_changes_after_each_step(self):
        """state.key must be different after every train_step call."""
        state, batch_x, exp_saxs, q, ff = self._make_state()
        key_before = state.key

        state1, _, _, _ = train_step(state, batch_x, exp_saxs, q, ff)
        assert not bool(jnp.array_equal(state1.key, key_before))

        state2, _, _, _ = train_step(state1, batch_x, exp_saxs, q, ff)
        assert not bool(jnp.array_equal(state2.key, state1.key))

    def test_two_steps_produce_different_losses(self):
        """Because each step samples different z, consecutive losses should differ."""
        state, batch_x, exp_saxs, q, ff = self._make_state()
        state1, loss1, _, _ = train_step(state, batch_x, exp_saxs, q, ff)
        _, loss2, _, _ = train_step(state1, batch_x, exp_saxs, q, ff)
        assert not bool(jnp.isclose(loss1, loss2))

    def test_beta_weighting_affects_total_loss(self):
        """Increasing beta should increase total loss if kl_loss > 0."""
        state, batch_x, exp_saxs, q, ff = self._make_state()

        _, loss_small, bio_small, kl_small = train_step(state, batch_x, exp_saxs, q, ff, beta=0.1)
        _, loss_large, bio_large, kl_large = train_step(state, batch_x, exp_saxs, q, ff, beta=10.0)

        assert bool(jnp.isclose(bio_small, bio_large))
        assert bool(jnp.isclose(kl_small, kl_large))

        if kl_small > 0:
            assert loss_large > loss_small
        else:
            assert bool(jnp.isclose(loss_large, loss_small))
