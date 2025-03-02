import numpy as np
from scipy.special import sph_harm

import constants

def sph_forward(psi, Lmax, theta_vals, phi_vals):
    """
    Naive forward spherical-harmonic transform:
    psi -> f_{ell,m}, for ell=0..Lmax, m=-ell..ell.
    
    psi : (Ntheta, Nphi) complex array
    Lmax : int
    theta_vals : (Ntheta,) in [0, pi]
    phi_vals   : (Nphi,)   in [0, 2pi)
    
    Returns
    -------
    flm : 2D complex array of shape (Lmax+1, 2*Lmax+1)
          flm[ell, m+ell] = coefficient for (ell,m).
    """
    Ntheta = len(theta_vals)
    Nphi = len(phi_vals)
    dtheta = (theta_vals[-1] - theta_vals[0])/(Ntheta-1) if Ntheta>1 else 0.0
    dphi = (phi_vals[-1] - phi_vals[0])/(Nphi) if Nphi>1 else 0.0
    
    # We'll store f_{ell,m} in flm[ell, m+ell].
    flm = np.zeros((Lmax+1, 2*Lmax+1), dtype=np.complex128)
    
    # We'll do a double sum (i,j) over the grid.
    for i, th in enumerate(theta_vals):
        sin_th = np.sin(th)
        for j, ph in enumerate(phi_vals):
            val = psi[i,j]  # wavefunction at (th, ph)
            for ell in range(Lmax+1):
                for m_ in range(-ell, ell+1):
                    # Evaluate Y_l^m(ph, th). 
                    # Note sph_harm(m, ell, phi, theta).
                    Y_lm = sph_harm(m_, ell, ph, th)
                    # accumulate into flm
                    # integral includes sin(th).
                    flm[ell, m_+ell] += val * np.conjugate(Y_lm) * sin_th
    # Multiply by the small area element
    # dtheta * dphi is the "area" of each grid cell
    # (We assume uniform spacing in phi, and in theta).
    area_elem = dtheta*dphi
    flm *= area_elem
    return flm


def sph_inverse(flm, Lmax, theta_vals, phi_vals):
    """
    Naive inverse spherical-harmonic transform:
    f_{ell,m} -> psi(theta,phi).
    
    flm : (Lmax+1, 2Lmax+1) complex array
    Lmax : int
    theta_vals, phi_vals as above
    
    Returns
    -------
    psi : (Ntheta, Nphi) complex array
    """
    Ntheta = len(theta_vals)
    Nphi = len(phi_vals)
    psi = np.zeros((Ntheta, Nphi), dtype=np.complex128)
    for i, th in enumerate(theta_vals):
        for j, ph in enumerate(phi_vals):
            accum = 0.0+0.0j
            for ell in range(Lmax+1):
                for m_ in range(-ell, ell+1):
                    Y_lm = sph_harm(m_, ell, ph, th)
                    accum += flm[ell, m_+ell]*Y_lm
            psi[i,j] = accum
    return psi


def split_step_solver_spherical(V_grid, psi0, Lmax, N_theta, N_phi, T, num_steps):
    """
    Split-step solver for the time-dependent Schrodinger eqn on the unit sphere,
    using naive spherical harmonic transforms from SciPy.

    Parameters
    ----------
    psi0 : (Ntheta, Nphi) complex array, initial wavefunction
    V_grid : (Ntheta, Nphi) real or complex array, time-INDEPENDENT potential
    theta_vals, phi_vals : 1D arrays for the grid of angles
    Lmax : int
        max spherical harmonic degree
    num_steps : int

    Returns
    -------
    psi : (Ntheta, Nphi) complex array after time T=num_steps*dt
    """
    # Build a grid of angles
    theta_vals   = np.linspace(0, np.pi, N_theta)
    phi_vals     = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
    
    dt = T / num_steps
    psi = psi0.copy()
    # half-step potential factor
    half_phase = np.exp(-0.5j * dt/constants.hbar * V_grid)  # shape(Ntheta,Nphi)

    # precompute kinetic factors
    # kinetic factor = exp(-i [constants.hbar/(2m)] * ell(ell+1) * dt )
    kinetic_factors = np.zeros(Lmax+1, dtype=np.complex128)
    prefactor = (constants.hbar/(2.0*constants.m))*dt
    for ell in range(Lmax+1):
        kinetic_factors[ell] = np.exp(-1.0j * prefactor * ell*(ell+1))

    for _step in range(num_steps):
        # half-step potential
        psi *= half_phase

        # full-step kinetic in spherical harmonic domain
        flm = sph_forward(psi, Lmax, theta_vals, phi_vals)
        for ell in range(Lmax+1):
            for m_ in range(-ell, ell+1):
                flm[ell, m_+ell] *= kinetic_factors[ell]
        psi = sph_inverse(flm, Lmax, theta_vals, phi_vals)

        # half-step potential
        psi *= half_phase

    return psi