# Theory

## The Ensemble Problem for Disordered Proteins

Intrinsically Disordered Proteins (IDPs) do not adopt a single stable
three-dimensional structure.  Instead, they interconvert among an astronomically
large number of conformations in solution — a *structural ensemble*.  Classical
structure-determination tools (X-ray crystallography, single-particle cryo-EM)
are designed for ordered proteins and cannot capture this conformational
heterogeneity.  Solution-state techniques such as **SAXS**, **NMR**, and
**smFRET** are sensitive to ensemble-averaged properties, but converting those
measurements back into atomic coordinates requires solving an
underdetermined inverse problem.

DiffEnsemble addresses this by learning a probabilistic generative model that
maps sequence features directly to an ensemble of conformations, and training
it end-to-end against experimental observables.

---

## Variational Autoencoder

DiffEnsemble uses a **Variational Autoencoder (VAE)** (Kingma & Welling, 2013)
as its generative backbone.

### Encoder

The encoder $q_\phi(\mathbf{z} \mid \mathbf{x})$ maps sequence features
$\mathbf{x} \in \mathbb{R}^{L \times F}$ to the parameters of a Gaussian
distribution over a low-dimensional latent space $\mathbf{z} \in \mathbb{R}^d$:

$$
q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}\!\left(\boldsymbol{\mu}_\phi(\mathbf{x}),\, \text{diag}\!\left(\boldsymbol{\sigma}^2_\phi(\mathbf{x})\right)\right)
$$

### Reparameterisation Trick

To allow gradient flow through the stochastic sampling step, we write:

$$
\mathbf{z}^{(k)} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}^{(k)}, \quad \boldsymbol{\varepsilon}^{(k)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Drawing $K = $ `ensemble_size` samples gives the structural ensemble directly.

### Decoder

The decoder $p_\theta(\boldsymbol{\tau} \mid \mathbf{z})$ maps each latent
sample to a vector of backbone torsion angles:

$$
\boldsymbol{\tau}^{(k)} = (\phi_1^{(k)}, \psi_1^{(k)}, \ldots, \phi_L^{(k)}, \psi_L^{(k)}) \in (-\pi, \pi)^{2L}
$$

---

## Differentiable Physics Engine

### NeRF: Torsions → Cartesian Coordinates

The **Natural Extension Reference Frame (NeRF)** algorithm (Parsons et al., 2005)
converts internal coordinates (bond lengths, bond angles, dihedral angles) into
Cartesian coordinates in $O(N)$ time and — crucially — is fully differentiable
with respect to all dihedral angles.  DiffEnsemble uses the implementation
provided by `diff_biophys.geometry.nerf.chain_nerf`.

For each residue $i$ we place three backbone heavy atoms (N, Cα, C) using
fixed idealised bond lengths and angles, and the predicted $(\phi_i, \psi_i)$
torsions.  Peptide bonds are assumed to be **trans** ($\omega = \pi$).

### SAXS: Debye Formula

The small-angle X-ray scattering intensity of a single conformation is
calculated via the **Debye formula**:

$$
I(q) = \sum_{i=1}^{N} \sum_{j=1}^{N} f_i(q)\, f_j(q)\, \frac{\sin(q\, r_{ij})}{q\, r_{ij}}
$$

where $f_i(q)$ are atomic form factors and $r_{ij}$ is the distance between
atoms $i$ and $j$.  The ensemble-averaged profile is:

$$
\langle I(q) \rangle = \frac{1}{K} \sum_{k=1}^{K} I^{(k)}(q)
$$

### NMR Observables *(planned)*

Support for Residual Dipolar Couplings (RDCs) and backbone chemical shifts
via the `diff_biophys.nmr` kernels is planned for a future release.

---

## Training Objective

DiffEnsemble minimises an **Evidence Lower BOund (ELBO)** that combines a
biophysical reconstruction term with a KL regulariser:

$$
\mathcal{L} = \underbrace{\text{MSE}\!\left(\langle I(q) \rangle_{\text{pred}},\, I(q)_{\text{exp}}\right)}_{\text{biophysical term}} + \beta \underbrace{D_{\text{KL}}\!\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, \mathcal{N}(\mathbf{0},\mathbf{I})\right)}_{\text{KL regulariser}}
$$

The $\beta$ coefficient (default 0.1) controls the trade-off between fitting
the data and maintaining a well-regularised latent space (analogous to
$\beta$-VAE).

---

## Scientific Validation

| Benchmark | Metric | Status |
|-----------|--------|--------|
| Flory scaling ($R_g \propto N^{0.588}$) | Exponent ν ∈ [0.35, 0.75] | ✅ Implemented |
| Sic1 Rg (Gomes et al., JACS 2020) | $R_g \approx 30.5$ Å | ✅ Implemented |
| CASP16 T1200 SpA RDC Q-factor | Q < 0.3 | 🔄 Pending NMR kernels |
| PED database parity | RMSD of observables | 🔄 Planned |

---

## References

1. Kingma & Welling (2013). *Auto-Encoding Variational Bayes*.
   [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)
2. Gomes et al. (2020). *Conformational Ensembles of an IDP Consistent with
   NMR, SAXS, and smFRET* (Forman-Kay Lab). *JACS*.
3. McBride et al. (2025). *Predicting Pose Distribution of Protein Domains*
   (Montelione Lab).
4. Parsons et al. (2005). *Practical conversion from torsion space to Cartesian
   space for in silico protein synthesis*. *J. Comput. Chem.* 26(10), 1063–1068.
