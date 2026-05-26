import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable
from diff_biophys.geometry.nerf import chain_nerf

class Encoder(nn.Module):
    """
    Encoder that maps sequence features to latent distribution parameters.
    """
    latent_dim: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x is (batch_size, seq_len, features)
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1))
        
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        mean = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mean, logvar

class Decoder(nn.Module):
    """
    Decoder that maps latent samples to protein torsions (phi, psi).
    """
    seq_len: int
    hidden_dim: int = 256

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        # z is (ensemble_size, latent_dim)
        ensemble_size = z.shape[0]
        
        x = nn.Dense(self.hidden_dim)(z)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        # Output phi and psi for each residue (seq_len * 2)
        torsions = nn.Dense(self.seq_len * 2)(x)
        torsions = jnp.tanh(torsions) * jnp.pi
        
        return torsions.reshape((ensemble_size, self.seq_len, 2))

class EnsembleVAE(nn.Module):
    """
    Variational Autoencoder for generating protein structural ensembles.
    """
    seq_len: int
    latent_dim: int
    ensemble_size: int = 100
    hidden_dim: int = 256

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.decoder = Decoder(seq_len=self.seq_len, hidden_dim=self.hidden_dim)

    def __call__(self, x: jnp.ndarray, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, logvar = self.encoder(x)
        
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, (self.ensemble_size, self.latent_dim))
        z = mean + eps * std
        
        torsions = self.decoder(z)
        return torsions, mean, logvar

    def generate_coordinates(self, torsions: jnp.ndarray) -> jnp.ndarray:
        """
        Convert backbone torsions (phi, psi) to N-Ca-C Cartesian coordinates.
        """
        ensemble_size = torsions.shape[0]
        n_res = torsions.shape[1]
        
        # Bond lengths and angles (A and rad)
        # Simplified: N-Ca, Ca-C, C-N
        b_n_ca = 1.458
        b_ca_c = 1.525
        b_c_n = 1.329
        
        a_c_n_ca = 2.124 # 121.7 deg
        a_n_ca_c = 1.941 # 111.2 deg
        a_ca_c_n = 2.030 # 116.3 deg
        
        omega = jnp.pi # Trans peptide bonds
        
        def build_one_structure(torsion_pair):
            # torsion_pair: (n_res, 2)
            phi = torsion_pair[:, 0]
            psi = torsion_pair[:, 1]
            
            # For N-Ca-C backbone, we have 3 atoms per residue
            # Total atoms = 3 * n_res
            
            # Placeholder: interleaved bond lengths/angles/dihedrals
            # This is a simplified NeRF build loop for a backbone.
            # In a real tool, we'd use a more robust backbone builder.
            
            all_bonds = []
            all_angles = []
            all_dihedrals = []
            
            for i in range(n_res):
                # N atom (relative to previous C)
                all_bonds.append(b_c_n)
                all_angles.append(a_ca_c_n)
                all_dihedrals.append(omega)
                
                # Ca atom (relative to N)
                all_bonds.append(b_n_ca)
                all_angles.append(a_c_n_ca)
                all_dihedrals.append(phi[i])
                
                # C atom (relative to Ca)
                all_bonds.append(b_ca_c)
                all_angles.append(a_n_ca_c)
                all_dihedrals.append(psi[i])
            
            b = jnp.array(all_bonds[3:])
            a = jnp.array(all_angles[3:])
            d = jnp.array(all_dihedrals[3:])
            
            init_coords = jnp.array([
                [0.0, 0.0, 0.0],
                [1.458, 0.0, 0.0],
                [2.0, 1.2, 0.0]
            ])
            
            return chain_nerf(init_coords, b, a, d)

        v_build = jax.vmap(build_one_structure)
        return v_build(torsions)
