import jax
import jax.numpy as jnp

from diff_ensemble.model import Decoder, Encoder, EnsembleVAE, build_backbone_coords
from diff_ensemble.train import create_train_state, train_step

# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------


def test_initialization():
    seq_len = 10
    latent_dim = 16
    ensemble_size = 5
    model = EnsembleVAE(seq_len=seq_len, latent_dim=latent_dim, ensemble_size=ensemble_size)

    rng = jax.random.PRNGKey(0)
    input_shape = (1, seq_len, 4)  # 4 features per residue

    state = create_train_state(model, rng, learning_rate=1e-3, input_shape=input_shape)
    assert state.params is not None
    print("Initialization successful.")


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
    form_factors = jnp.ones((num_atoms, 20))  # Match new backbone atom count (N, Ca, C)

    state = create_train_state(model, rng, learning_rate=1e-3, input_shape=batch_x.shape)

    state, loss, bio_loss, kl_loss = train_step(state, batch_x, exp_saxs, q_values, form_factors)

    assert loss > 0
    print(f"Loss: {loss}, Bio Loss: {bio_loss}, KL Loss: {kl_loss}")
    print("Gradient flow test passed.")


# ---------------------------------------------------------------------------
# build_backbone_coords — shape contract and coordinate correctness
# ---------------------------------------------------------------------------


class TestBuildBackboneCoords:
    def test_output_shape(self):
        """Output must be (ensemble_size, seq_len * 3, 3)."""
        ensemble_size, seq_len = 7, 12
        torsions = jnp.zeros((ensemble_size, seq_len, 2))
        coords = build_backbone_coords(torsions)
        assert coords.shape == (ensemble_size, seq_len * 3, 3)

    def test_first_atom_at_origin(self):
        """The first atom (N of residue 1) is always placed at the origin."""
        torsions = jnp.zeros((3, 5, 2))
        coords = build_backbone_coords(torsions)
        # First atom of every model should be (0, 0, 0)
        assert bool(jnp.allclose(coords[:, 0, :], jnp.zeros((3, 3)), atol=1e-5))

    def test_coordinates_are_finite(self):
        """Random torsions should never produce NaN or Inf coordinates."""
        rng = jax.random.PRNGKey(7)
        torsions = jax.random.uniform(rng, (10, 20, 2), minval=-jnp.pi, maxval=jnp.pi)
        coords = build_backbone_coords(torsions)
        assert bool(jnp.all(jnp.isfinite(coords)))

    def test_different_torsions_give_different_coords(self):
        """Two distinct torsion sets must produce distinct coordinate sets."""
        torsions_a = jnp.zeros((1, 5, 2))
        torsions_b = jnp.ones((1, 5, 2)) * 0.5
        coords_a = build_backbone_coords(torsions_a)
        coords_b = build_backbone_coords(torsions_b)
        assert not bool(jnp.allclose(coords_a, coords_b))


# ---------------------------------------------------------------------------
# Encoder and Decoder — in isolation
# ---------------------------------------------------------------------------


class TestEncoder:
    def test_output_shapes(self):
        """Encoder must return two tensors both shaped (batch, latent_dim)."""
        latent_dim, hidden_dim = 16, 64
        seq_len, batch_size, features = 10, 2, 4

        encoder = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        rng = jax.random.PRNGKey(0)
        x = jnp.ones((batch_size, seq_len, features))
        params = encoder.init(rng, x)["params"]  # type: ignore[index]
        mean, logvar = encoder.apply({"params": params}, x)  # type: ignore[misc]

        assert mean.shape == (batch_size, latent_dim)  # type: ignore[union-attr]
        assert logvar.shape == (batch_size, latent_dim)  # type: ignore[union-attr]

    def test_mean_and_logvar_are_different(self):
        """Mean and log-variance heads should produce different values."""
        encoder = Encoder(latent_dim=8, hidden_dim=32)
        rng = jax.random.PRNGKey(1)
        x = jnp.ones((1, 5, 4))
        params = encoder.init(rng, x)["params"]  # type: ignore[index]
        mean, logvar = encoder.apply({"params": params}, x)  # type: ignore[misc]
        assert not bool(jnp.allclose(mean, logvar))  # type: ignore[arg-type]


class TestDecoder:
    def test_output_shape(self):
        """Decoder must return (ensemble_size, seq_len, 2) torsion tensor."""
        seq_len, latent_dim, ensemble_size = 10, 16, 8
        decoder = Decoder(seq_len=seq_len, hidden_dim=64)
        rng = jax.random.PRNGKey(0)
        z = jnp.ones((ensemble_size, latent_dim))
        params = decoder.init(rng, z)["params"]  # type: ignore[index]
        torsions = decoder.apply({"params": params}, z)  # type: ignore[misc]
        assert torsions.shape == (ensemble_size, seq_len, 2)  # type: ignore[union-attr]

    def test_torsions_bounded_in_pi(self):
        """tanh * π activation keeps all torsions in the closed interval [−π, π].

        Note: float32 tanh saturates to exactly ±1.0 for large inputs, so
        torsions may reach exactly ±π.  The correct invariant is ≤, not <.
        """
        seq_len, latent_dim, ensemble_size = 10, 16, 20
        decoder = Decoder(seq_len=seq_len, hidden_dim=64)
        rng = jax.random.PRNGKey(2)
        z = jax.random.normal(rng, (ensemble_size, latent_dim)) * 10  # large inputs
        params = decoder.init(rng, z)["params"]  # type: ignore[index]
        torsions = decoder.apply({"params": params}, z)  # type: ignore[misc]
        assert bool(jnp.all(torsions >= -jnp.pi))  # type: ignore[operator]
        assert bool(jnp.all(torsions <= jnp.pi))  # type: ignore[operator]


# ---------------------------------------------------------------------------
# train_step — RNG key advances between steps
# ---------------------------------------------------------------------------


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
        assert not bool(jnp.array_equal(state1.key, key_before)), "Key should change after step 1"

        state2, _, _, _ = train_step(state1, batch_x, exp_saxs, q, ff)
        assert not bool(jnp.array_equal(state2.key, state1.key)), "Key should change after step 2"

    def test_two_steps_produce_different_losses(self):
        """Because each step samples different z, consecutive losses should differ."""
        state, batch_x, exp_saxs, q, ff = self._make_state()
        state1, loss1, _, _ = train_step(state, batch_x, exp_saxs, q, ff)
        _, loss2, _, _ = train_step(state1, batch_x, exp_saxs, q, ff)
        # Losses from different z samples are extremely unlikely to be identical
        assert not bool(jnp.isclose(loss1, loss2)), (
            f"Step 1 loss {loss1} and step 2 loss {loss2} should differ"
        )


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_initialization()
    test_gradient_flow()
