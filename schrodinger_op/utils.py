import numpy as np
from scipy.special import sph_harm

def sph_forward(psi, Lmax, theta_vals, phi_vals):
    """
    Vectorized forward spherical-harmonic transform:
      psi(θ,φ) -> f_{ell,m}, for ell=0..Lmax, m=-ell..ell.

    psi : (Ntheta, Nphi) complex array
    Lmax : int
    theta_vals : (Ntheta,) in [0, π]
    phi_vals   : (Nphi,)   in [0, 2π)

    Returns
    -------
    flm : 2D complex array of shape (Lmax+1, 2*Lmax+1)
          flm[ell, m+ell] = ∫ psi(θ,φ)*Yₗᵐ*(θ,φ)*sinθ dθ dφ
    """
    # Grid sizes
    Ntheta = len(theta_vals)
    Nphi   = len(phi_vals)

    # spacing for uniform integration
    dtheta = (theta_vals[-1] - theta_vals[0])/(Ntheta-1) if Ntheta>1 else 0.0
    dphi   = (phi_vals[-1] - phi_vals[0])/(Nphi) if Nphi>1 else 0.0

    # Build a 2D mesh for θ, φ
    # shape => (Ntheta, Nphi)
    TH, PH = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    # sin(TH)
    sin_TH = np.sin(TH)

    # The integral factor for each grid cell
    area_elem = dtheta*dphi

    # We'll store f_{ell,m} in flm[ell, m+ell].
    flm = np.zeros((Lmax+1, 2*Lmax+1), dtype=np.complex128)

    # Instead of looping over i,j, we do a smaller loop over (ell,m).
    # Then we evaluate Y_lm on the entire grid and do a single sum.
    for ell in range(Lmax+1):
        for m_ in range(-ell, ell+1):
            # Evaluate Y_l^m on entire grid (Ntheta,Nphi)
            # sph_harm(m, ell, φ, θ) => shape(Ntheta,Nphi)
            Y_lm_all = sph_harm(m_, ell, PH, TH)
            # integrand = psi * conj(Y_lm) * sin_TH
            integrand = psi * np.conjugate(Y_lm_all) * sin_TH
            # sum over all points
            flm[ell, m_+ell] = np.sum(integrand)*area_elem

    return flm


def sph_inverse(flm, Lmax, theta_vals, phi_vals):
    """
    Vectorized inverse spherical-harmonic transform:
      f_{ell,m} -> psi(θ,φ).

    flm : (Lmax+1, 2Lmax+1) complex array
    Lmax : int
    theta_vals, phi_vals as above

    Returns
    -------
    psi : (Ntheta, Nphi) complex array
    """
    Ntheta = len(theta_vals)
    Nphi   = len(phi_vals)
    # Build (θ, φ) mesh
    TH, PH = np.meshgrid(theta_vals, phi_vals, indexing='ij')

    # We'll accumulate in a single array
    psi = np.zeros((Ntheta, Nphi), dtype=np.complex128)

    # For each (ell,m), we just add flm[ell,m+ell]* Y_lm(θ,φ).
    for ell in range(Lmax+1):
        for m_ in range(-ell, ell+1):
            Y_lm_all = sph_harm(m_, ell, PH, TH)
            psi += flm[ell, m_+ell]* Y_lm_all

    return psi