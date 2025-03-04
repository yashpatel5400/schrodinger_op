import numpy as np

import constants
import potentials
from dataset import GRF_spherical

import utils

def split_step_solver_spherical(V_grid, psi0, Lmax, T, num_steps):
    """
    Split-step solver for the time-dependent Schrodinger eqn on the unit sphere,
    using vectorized spherical harmonic transforms.
    """
    # Build a grid of angles from shape of V_grid
    Ntheta, Nphi = V_grid.shape
    theta_vals = np.linspace(0, np.pi, Ntheta)
    phi_vals   = np.linspace(0, 2*np.pi, Nphi, endpoint=False)

    dt = T / num_steps
    psi = psi0.copy()

    # half-step potential factor
    half_phase = np.exp(-0.5j * dt/constants.hbar * V_grid)

    # precompute kinetic factors
    # kinetic factor = exp(-i [hbar/(2m)] * ell(ell+1) * dt )
    kinetic_factors = np.zeros(Lmax+1, dtype=np.complex128)
    prefactor = (constants.hbar/(2.0*constants.m))*dt
    for ell in range(Lmax+1):
        kinetic_factors[ell] = np.exp(-1.0j * prefactor * ell*(ell+1))

    for _step in range(num_steps):
        # half-step potential
        psi *= half_phase

        # full-step kinetic in spherical harmonic domain
        flm = utils.sph_forward(psi, Lmax, theta_vals, phi_vals)

        # multiply each row by the corresponding factor
        for ell in range(Lmax+1):
            flm[ell,:] *= kinetic_factors[ell]

        psi = utils.sph_inverse(flm, Lmax, theta_vals, phi_vals)

        # half-step potential
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