import os

import biotite.structure.io.pdb as pdb
import numpy as np

from diff_ensemble.io import save_ensemble_to_pdb


def test_save_ensemble_to_pdb(tmp_path):
    """
    Test the PDB ensemble export utility.
    """
    ensemble_size = 3
    n_res = 5
    n_atoms = n_res * 3  # N-Ca-C

    # Create dummy coordinates
    coords = np.random.rand(ensemble_size, n_atoms, 3).astype(np.float32)

    file_path = os.path.join(tmp_path, "test_ensemble.pdb")
    res_names = ["GLY", "ALA", "SER", "VAL", "LEU"]

    # Act
    save_ensemble_to_pdb(coords, file_path, res_names=res_names)

    # Assert
    assert os.path.exists(file_path)

    # Read back and check
    pdb_file = pdb.PDBFile.read(file_path)
    stack = pdb_file.get_structure()

    assert stack.box is None  # PDB doesn't necessarily have a box
    assert stack.coord.shape == (ensemble_size, n_atoms, 3)
    assert stack.res_name[0] == "GLY"
    assert stack.res_name[3] == "ALA"  # Second residue, atoms are 0,1,2 (GLY) then 3,4,5 (ALA)
    assert np.allclose(stack.coord, coords, atol=1e-3)
