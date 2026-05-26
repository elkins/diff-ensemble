"""Package import smoke tests.

Verifies that every name declared in ``diff_ensemble.__all__`` is importable
and callable.  A broken re-export in ``__init__.py`` would otherwise only
surface at runtime, not during CI.
"""

import diff_ensemble


def test_all_exports_importable():
    """Every name in __all__ must resolve to a callable object."""
    missing = []
    not_callable = []

    for name in diff_ensemble.__all__:
        obj = getattr(diff_ensemble, name, None)
        if obj is None:
            missing.append(name)
        elif not callable(obj):
            not_callable.append(name)

    assert not missing, f"Names in __all__ not found on the module: {missing}"
    assert not not_callable, f"Names in __all__ that are not callable: {not_callable}"


def test_top_level_imports():
    """Explicit star-import style — catches namespace pollution or shadowing."""
    from diff_ensemble import (  # noqa: F401
        EnsemblePredictor,
        EnsembleVAE,
        biophysical_loss,
        build_backbone_coords,
        get_ensemble_chemical_shifts,
        get_ensemble_rdc,
        get_ensemble_saxs,
        kld_loss,
        save_ensemble_to_pdb,
    )


def test_submodule_imports_still_work():
    """Internal submodule imports must not break after __init__ is populated."""
    from diff_ensemble.ensemble import EnsemblePredictor  # noqa: F401
    from diff_ensemble.io import save_ensemble_to_pdb  # noqa: F401
    from diff_ensemble.model import EnsembleVAE, build_backbone_coords  # noqa: F401
    from diff_ensemble.observables import biophysical_loss, kld_loss  # noqa: F401
    from diff_ensemble.train import create_train_state, train_step  # noqa: F401
