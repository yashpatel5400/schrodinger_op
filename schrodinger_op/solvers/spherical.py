import numpy as np

import constants
import potentials
from dataset import GRF_spherical

def split_step_solver_spherical(V_grid, psi0, sph_transformer, T, num_steps):
    """
    Split-step solver for time-dependent Schrodinger eqn on sphere,
    using precomputed SphericalHarmonicsTransform 'sph_transformer'.
    
    psi0 : (Ntheta, Nphi) array
    V_grid : (Ntheta, Nphi)
    sph_transformer : SphericalHarmonicsTransform instance
    T, num_steps : float, int
    """
    dt = T / num_steps
    psi = psi0.copy()
    half_phase = np.exp(-0.5j * dt/constants.hbar * V_grid)

    # precompute kinetic factors
    Kfac = sph_transformer.spec_array_cplx()
    prefactor = (constants.hbar/(2.0*constants.m))*dt
    for ell in range(sph_transformer.lmax+1):
        for m in range(-ell, ell+1):
            Kfac[sph_transformer.zidx(ell, m)] = np.exp(-1.0j * prefactor * ell*(ell+1))
    
    for step in range(num_steps):
        psi *= half_phase
        flm = sph_transformer.analys_cplx(psi)
        flm *= Kfac
        psi = sph_transformer.synth_cplx(flm)
        psi *= half_phase
    return psi


def test_spherical_grf():
    """
    Example usage of GRF_spherical to get a random wavefunction on a sphere,
    then evolve it with split_step_solver_spherical.
    """
    import numpy as np
    
    # define a spherical grid
    N_theta = 32
    N_phi   = 64
    
    # define parameters for the GRF
    alpha, beta, gamma = 1.0, 1.0, 4.0
    Lmax = 20
    psi0 = GRF_spherical(alpha, beta, gamma, Lmax, N_theta, N_phi)
    
    # (Optional) normalize
    norm_psi0 = np.linalg.norm(psi0)
    if norm_psi0>1e-14:
        psi0 /= norm_psi0
    
    # Build a potential, e.g. a dipole or a constant Coulomb on the sphere
    V_sphere = potentials.uniform_sphere(N_theta, N_phi, k=1.0, e=1.0)

    # Evolve in time with your spherical solver
    T = 0.1
    num_steps = 50
    psi_final = split_step_solver_spherical(
        V_sphere,
        psi0,
        Lmax,
        T,
        num_steps
    )


if __name__ == "__main__":
    test_spherical_grf()