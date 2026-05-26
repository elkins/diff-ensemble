"""DiffEnsemble — Differentiable VAE-based structural ensemble prediction.

Public API
----------
>>> from diff_ensemble import EnsembleVAE, build_backbone_coords
>>> from diff_ensemble import save_ensemble_to_pdb
>>> from diff_ensemble import EnsemblePredictor
>>> from diff_ensemble import get_ensemble_saxs, get_ensemble_rdc
>>> from diff_ensemble import kld_loss, biophysical_loss
"""

from diff_ensemble.ensemble import EnsemblePredictor
from diff_ensemble.io import save_ensemble_to_pdb
from diff_ensemble.model import EnsembleVAE, build_backbone_coords
from diff_ensemble.observables import (
    biophysical_loss,
    get_ensemble_chemical_shifts,
    get_ensemble_rdc,
    get_ensemble_saxs,
    kld_loss,
)

__all__ = [
    # Model
    "EnsembleVAE",
    "build_backbone_coords",
    # High-level API
    "EnsemblePredictor",
    # I/O
    "save_ensemble_to_pdb",
    # Observables
    "get_ensemble_saxs",
    "get_ensemble_rdc",
    "get_ensemble_chemical_shifts",
    # Losses
    "kld_loss",
    "biophysical_loss",
]
