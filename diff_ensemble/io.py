import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import numpy as np


def save_ensemble_to_pdb(coords: np.ndarray, file_path: str, res_names: list[str] | None = None):
    """
    Save a structural ensemble to a multi-model PDB file.

    Args:
        coords: (M, N, 3) where M is ensemble size and N is atom count.
        file_path: Path to the output PDB file.
        res_names: List of residue names (length n_res). Defaults to ALA.
    """
    ensemble_size, n_atoms, _ = coords.shape
    n_res = n_atoms // 3  # Assuming N-Ca-C backbone

    if res_names is None:
        res_names = ["ALA"] * n_res

    # Create a template structure for one model
    # Atoms: N, CA, C for each residue
    atom_names = ["N", "CA", "C"] * n_res
    res_indices = []
    for i in range(n_res):
        res_indices.extend([i + 1] * 3)

    # Build biotite AtomArrayStack
    stack = struc.AtomArrayStack(ensemble_size, n_atoms)
    stack.coord = coords

    # Fill in metadata for the first model (metadata is shared in the stack)
    stack.chain_id = np.array(["A"] * n_atoms)
    stack.res_id = np.array(res_indices)
    stack.res_name = np.array([res_names[i - 1] for i in res_indices])
    stack.atom_name = np.array(atom_names)
    stack.element = np.array([name[0] for name in atom_names])  # N, C, C

    # Save to file
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(stack)
    pdb_file.write(file_path)
    print(f"Ensemble saved to {file_path}")
