import jax.numpy as jnp
import pytest

from diff_ensemble.observables import (
    biophysical_loss,
    get_ensemble_chemical_shifts,
    get_ensemble_rdc,
    kld_loss,
)

# ---------------------------------------------------------------------------
# kld_loss (existing tests preserved)
# ---------------------------------------------------------------------------


class TestKldLoss:
    def test_zero_at_unit_gaussian(self):
        """KLD should be 0 when the posterior is a unit Gaussian."""
        latent_dim = 8
        mean = jnp.zeros(latent_dim)
        logvar = jnp.zeros(latent_dim)  # log(1) = 0
        loss = kld_loss(mean, logvar)
        assert float(loss) == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_non_unit(self):
        """KLD must be non-negative for any non-unit Gaussian."""
        mean = jnp.array([1.0, -2.0, 0.5])
        logvar = jnp.array([0.5, -0.5, 1.0])
        loss = kld_loss(mean, logvar)
        assert float(loss) >= 0.0

    def test_increases_with_mean_offset(self):
        """Larger mean offset → larger KLD."""
        logvar = jnp.zeros(4)
        loss_small = kld_loss(jnp.ones(4) * 0.1, logvar)
        loss_large = kld_loss(jnp.ones(4) * 5.0, logvar)
        assert float(loss_large) > float(loss_small)


# ---------------------------------------------------------------------------
# biophysical_loss (existing tests preserved, zero-denom guard added)
# ---------------------------------------------------------------------------


class TestBiophysicalLoss:
    def test_zero_for_identical_profiles(self):
        """Loss must be zero when predicted == experimental."""
        profile = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2])
        loss = biophysical_loss(profile, profile)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_scale_invariance(self):
        """Loss should be identical after rescaling either profile."""
        pred = jnp.array([2.0, 1.6, 1.2, 0.8, 0.4])
        exp = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2])
        # pred is exactly 2× exp — both normalise to the same curve
        loss = biophysical_loss(pred, exp)
        assert float(loss) == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different_shapes(self):
        """Loss must be positive when profiles differ in shape."""
        pred = jnp.array([1.0, 0.9, 0.8, 0.7, 0.6])
        exp = jnp.array([1.0, 0.5, 0.2, 0.1, 0.05])
        loss = biophysical_loss(pred, exp)
        assert float(loss) > 0.0

    def test_near_zero_first_element_does_not_produce_nan(self):
        """The 1e-10 guard in the normalisation denominator must prevent NaN/Inf."""
        pred = jnp.array([1e-15, 0.5, 0.3, 0.1])
        exp = jnp.array([1.0, 0.8, 0.5, 0.2])
        loss = biophysical_loss(pred, exp)
        assert bool(jnp.isfinite(loss)), "Loss must remain finite with near-zero first element"


# ---------------------------------------------------------------------------
# NMR observable stubs — must raise NotImplementedError
# ---------------------------------------------------------------------------


class TestNmrStubs:
    def test_get_ensemble_rdc_raises_not_implemented(self):
        """get_ensemble_rdc must raise NotImplementedError until kernels land."""
        coords = jnp.zeros((5, 10, 3))
        bond_vectors = jnp.zeros((5, 2))
        alignment_tensor = jnp.eye(3)
        with pytest.raises(NotImplementedError, match="diff_biophys.nmr"):
            get_ensemble_rdc(coords, bond_vectors, alignment_tensor)

    def test_get_ensemble_chemical_shifts_raises_not_implemented(self):
        """get_ensemble_chemical_shifts must raise NotImplementedError until kernels land."""
        coords = jnp.zeros((5, 10, 3))
        sequence = jnp.zeros(5, dtype=jnp.int32)
        with pytest.raises(NotImplementedError, match="diff_biophys.nmr"):
            get_ensemble_chemical_shifts(coords, sequence)

    def test_rdc_error_message_mentions_issue_tracker(self):
        """The error message should point users to the GitHub issue tracker."""
        coords = jnp.zeros((1, 3, 3))
        bond_vectors = jnp.zeros((1, 2))
        alignment_tensor = jnp.eye(3)
        with pytest.raises(NotImplementedError, match="github.com/elkins"):
            get_ensemble_rdc(coords, bond_vectors, alignment_tensor)
