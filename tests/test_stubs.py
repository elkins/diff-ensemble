import jax.numpy as jnp
import pytest

from diff_ensemble.observables import get_ensemble_chemical_shifts, get_ensemble_rdc


def test_rdc_stub_raises_error():
    coords = jnp.zeros((5, 10, 3))
    bond_vectors = jnp.zeros((10, 2))
    alignment_tensor = jnp.zeros((3, 3))
    with pytest.raises(NotImplementedError, match="RDC calculation requires"):
        get_ensemble_rdc(coords, bond_vectors, alignment_tensor)


def test_chemical_shifts_stub_raises_error():
    coords = jnp.zeros((5, 10, 3))
    sequence = jnp.zeros((10,), dtype=jnp.int32)
    with pytest.raises(NotImplementedError, match="Chemical shift calculation requires"):
        get_ensemble_chemical_shifts(coords, sequence)
