"""Tests for the EnsemblePredictor high-level API (ensemble.py)."""

import jax
import jax.numpy as jnp
import pytest

from diff_ensemble.ensemble import EnsemblePredictor
from diff_ensemble.model import EnsembleVAE

# ---------------------------------------------------------------------------
# Shared constants and fixture
# ---------------------------------------------------------------------------

SEQ_LEN = 10
LATENT_DIM = 8
ENSEMBLE_SIZE = 5
FEATURES = 4


@pytest.fixture(scope="module")
def predictor_setup():
    """Create a model, params, and predictor once for the whole module."""
    model = EnsembleVAE(seq_len=SEQ_LEN, latent_dim=LATENT_DIM, ensemble_size=ENSEMBLE_SIZE)
    init_rng, sample_rng = jax.random.split(jax.random.PRNGKey(42))
    params = model.init(init_rng, jnp.ones((1, SEQ_LEN, FEATURES)), init_rng)["params"]
    predictor = EnsemblePredictor(model, params)
    features = jnp.ones((1, SEQ_LEN, FEATURES))
    return predictor, features, sample_rng


# ---------------------------------------------------------------------------
# EnsemblePredictor.predict
# ---------------------------------------------------------------------------


class TestPredict:
    def test_default_output_shape(self, predictor_setup):
        """predict() should return (ensemble_size, seq_len*3, 3) coordinates."""
        predictor, features, rng = predictor_setup
        coords = predictor.predict(features, rng)
        assert coords.shape == (ENSEMBLE_SIZE, SEQ_LEN * 3, 3)

    def test_n_samples_override_shape(self, predictor_setup):
        """n_samples != ensemble_size should produce the requested number of models."""
        predictor, features, rng = predictor_setup
        n = 12
        coords = predictor.predict(features, rng, n_samples=n)
        assert coords.shape == (n, SEQ_LEN * 3, 3)

    def test_n_samples_equal_to_default_takes_normal_path(self, predictor_setup):
        """n_samples == ensemble_size should still return the correct shape."""
        predictor, features, rng = predictor_setup
        coords = predictor.predict(features, rng, n_samples=ENSEMBLE_SIZE)
        assert coords.shape == (ENSEMBLE_SIZE, SEQ_LEN * 3, 3)

    def test_coordinates_are_finite(self, predictor_setup):
        """All output coordinates should be finite — no NaN or Inf."""
        predictor, features, rng = predictor_setup
        coords = predictor.predict(features, rng)
        assert bool(jnp.all(jnp.isfinite(coords)))

    def test_different_rng_produces_different_ensemble(self, predictor_setup):
        """Two different PRNG keys should yield different conformations."""
        predictor, features, _ = predictor_setup
        rng_a, rng_b = jax.random.split(jax.random.PRNGKey(99))
        coords_a = predictor.predict(features, rng_a)
        coords_b = predictor.predict(features, rng_b)
        assert not bool(jnp.allclose(coords_a, coords_b))


# ---------------------------------------------------------------------------
# EnsemblePredictor.compute_rg
# ---------------------------------------------------------------------------


class TestComputeRg:
    def test_positive_for_real_ensemble(self, predictor_setup):
        """Rg should be positive for any non-trivial ensemble."""
        predictor, features, rng = predictor_setup
        coords = predictor.predict(features, rng)
        avg_rg, rg_per_model = predictor.compute_rg(coords)
        assert float(avg_rg) > 0.0
        assert rg_per_model.shape == (ENSEMBLE_SIZE,)
        assert bool(jnp.all(rg_per_model > 0))

    def test_zero_for_coincident_atoms(self, predictor_setup):
        """All atoms at the same position → Rg = 0."""
        predictor, _, _ = predictor_setup
        coords = jnp.zeros((3, 9, 3))  # 3 models, 9 atoms, all at origin
        avg_rg, _ = predictor.compute_rg(coords)
        assert float(avg_rg) == pytest.approx(0.0, abs=1e-6)

    def test_uniform_weights_equal_default(self, predictor_setup):
        """Explicit uniform weights should produce the same result as no weights."""
        predictor, features, rng = predictor_setup
        coords = predictor.predict(features, rng)
        uniform = jnp.ones(ENSEMBLE_SIZE) / ENSEMBLE_SIZE
        avg_default, _ = predictor.compute_rg(coords)
        avg_uniform, _ = predictor.compute_rg(coords, weights=uniform)
        assert float(avg_default) == pytest.approx(float(avg_uniform), rel=1e-5)

    def test_single_model_weight_selects_that_model(self, predictor_setup):
        """Putting all weight on one model should return that model's Rg."""
        predictor, features, rng = predictor_setup
        coords = predictor.predict(features, rng)
        _, rg_per_model = predictor.compute_rg(coords)
        target = 2
        weights = jnp.zeros(ENSEMBLE_SIZE).at[target].set(1.0)
        avg_rg, _ = predictor.compute_rg(coords, weights=weights)
        assert float(avg_rg) == pytest.approx(float(rg_per_model[target]), rel=1e-5)


# ---------------------------------------------------------------------------
# EnsemblePredictor.compute_end_to_end_distance
# ---------------------------------------------------------------------------


class TestComputeEndToEndDistance:
    def test_non_negative_for_real_ensemble(self, predictor_setup):
        """R_ee must be non-negative for any ensemble."""
        predictor, features, rng = predictor_setup
        coords = predictor.predict(features, rng)
        avg_ree, ree_per_model = predictor.compute_end_to_end_distance(coords)
        assert float(avg_ree) >= 0.0
        assert ree_per_model.shape == (ENSEMBLE_SIZE,)
        assert bool(jnp.all(ree_per_model >= 0))

    def test_zero_when_first_and_last_atom_coincide(self, predictor_setup):
        """If first and last atom are at the same position, R_ee = 0."""
        predictor, _, _ = predictor_setup
        coords = jnp.ones((3, 1, 3)) * 5.0  # 3 models, 1 atom → first == last
        avg_ree, _ = predictor.compute_end_to_end_distance(coords)
        assert float(avg_ree) == pytest.approx(0.0, abs=1e-6)

    def test_known_distance_3_4_5_triangle(self, predictor_setup):
        """First atom at origin, last at (3, 4, 0) → R_ee = 5 Å (3-4-5 triangle)."""
        predictor, _, _ = predictor_setup
        n_atoms = 9
        coords = jnp.zeros((1, n_atoms, 3)).at[0, -1, :].set(jnp.array([3.0, 4.0, 0.0]))
        avg_ree, _ = predictor.compute_end_to_end_distance(coords)
        assert float(avg_ree) == pytest.approx(5.0, rel=1e-5)


# ---------------------------------------------------------------------------
# EnsemblePredictor.compute_population_average
# ---------------------------------------------------------------------------


class TestComputePopulationAverage:
    def test_uniform_weights_equal_unweighted_mean(self, predictor_setup):
        """Default (uniform) weighting should match a direct vmap mean."""
        predictor, features, rng = predictor_setup
        coords = predictor.predict(features, rng)

        def scalar_mean(c: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(c)

        pop_avg = predictor.compute_population_average(scalar_mean, coords)
        direct_mean = float(jnp.mean(jax.vmap(scalar_mean)(coords)))
        assert float(pop_avg) == pytest.approx(direct_mean, rel=1e-5)

    def test_all_weight_on_one_model(self, predictor_setup):
        """Weight of 1.0 on a single model should return that model's observable."""
        predictor, features, rng = predictor_setup
        coords = predictor.predict(features, rng)
        target_idx = 3
        weights = jnp.zeros(ENSEMBLE_SIZE).at[target_idx].set(1.0)

        def sum_coords(c: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(c)

        pop_avg = predictor.compute_population_average(sum_coords, coords, weights=weights)
        model_obs = sum_coords(coords[target_idx])
        assert float(pop_avg) == pytest.approx(float(model_obs), rel=1e-5)

    def test_vector_observable_returns_correct_shape(self, predictor_setup):
        """An observable returning a vector should produce a vector average."""
        predictor, features, rng = predictor_setup
        coords = predictor.predict(features, rng)

        def per_atom_norm(c: jnp.ndarray) -> jnp.ndarray:
            # Per-atom distance from origin — shape (n_atoms,) per model
            return jnp.linalg.norm(c, axis=-1)

        pop_avg = predictor.compute_population_average(per_atom_norm, coords)
        assert pop_avg.shape == (SEQ_LEN * 3,)
        assert bool(jnp.all(pop_avg >= 0))
