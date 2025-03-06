import numpy as np
from scipy.special import sph_harm

class SphericalHarmonicsTransform:
    """
    Precomputes spherical harmonics Y_l^m on a fixed (N_theta x N_phi) grid
    for ell=0..Lmax, m=-ell..ell. Then can do 'forward' and 'inverse'
    transforms to/from spherical-harmonic space in a vectorized manner.
    """
    def __init__(self, Lmax, N_theta, N_phi):
        """
        Lmax: int
            maximum spherical harmonic degree
        N_theta, N_phi: int
            number of grid points in theta and phi dimension
        """
        self.Lmax    = Lmax
        self.N_theta = N_theta
        self.N_phi   = N_phi
        
        # 1) Create the uniform angle arrays
        self.theta_vals = np.linspace(0, np.pi, N_theta)
        self.phi_vals   = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
        
        # 2) precompute mesh TH, PH of shape (N_theta, N_phi)
        self.TH, self.PH = np.meshgrid(self.theta_vals, self.phi_vals, indexing='ij')
        
        # 3) precompute sin(TH) and area element for forward transform
        self.sin_TH = np.sin(self.TH)
        dtheta = (self.theta_vals[-1] - self.theta_vals[0])/(N_theta-1) if N_theta>1 else 0.0
        dphi   = (self.phi_vals[-1] - self.phi_vals[0])/(N_phi) if N_phi>1 else 0.0
        self.area_elem = dtheta*dphi
        
        # 4) Precompute all spherical harmonics Y_l^m(TH,PH).
        # We'll store them in a 4D array: shape (Lmax+1, 2Lmax+1, N_theta, N_phi).
        # Y_all[ell, m+ell] => shape (N_theta, N_phi).
        self.Y_all = np.zeros((Lmax+1, 2*Lmax+1, N_theta, N_phi), dtype=np.complex128)
        
        for ell in range(Lmax+1):
            for m_ in range(-ell, ell+1):
                # Evaluate Y_l^m on entire grid
                # sph_harm(m, ell, phi, theta) => shape (N_theta,N_phi)
                # but we have PH,TH => shape(N_theta,N_phi)
                # By default: sph_harm(m, ell, phi, theta).
                Y_lm = sph_harm(m_, ell, self.PH, self.TH)
                self.Y_all[ell, m_+ell] = Y_lm

        # 5) Also store the conjugates for forward transform
        # or we can conj on the fly. Let's do it once to speed up.
        self.Y_all_conj = np.conjugate(self.Y_all)


    def forward(self, psi):
        """
        Vectorized forward transform:
        psi(θ,φ) -> flm[ell, m+ell].
        
        psi : (N_theta, N_phi) array
        Returns
        -------
        flm : (Lmax+1, 2Lmax+1) complex array
        """
        # We want flm[ell,m+ell] = ∫ psi * conj(Y_l^m) * sinθ dθ dφ.
        # We'll do it in a single big operation:
        
        # 1) Expand psi => shape(1,1,N_theta,N_phi) for broadcasting
        #    same for sin_TH => shape(1,1,N_theta,N_phi)
        psi_4d = psi[np.newaxis, np.newaxis, ...]
        sin_4d = self.sin_TH[np.newaxis, np.newaxis, ...]
        
        # 2) integrand = psi * Y_all_conj * sinTH
        # shape => (Lmax+1, 2Lmax+1, N_theta, N_phi)
        integrand = psi_4d * self.Y_all_conj * sin_4d
        
        # 3) sum over axes -2, -1 => we get shape (Lmax+1, 2Lmax+1)
        flm = self.area_elem * np.sum(integrand, axis=(-2,-1))
        return flm


    def inverse(self, flm):
        """
        Vectorized inverse transform:
        flm[ell,m+ell] -> psi(θ,φ).
        
        flm : (Lmax+1, 2Lmax+1) array
        Returns
        -------
        psi : (N_theta, N_phi) complex array
        """
        # We want: psi(θ,φ) = sum_{ell,m} flm[ell,m+ell]* Y_all[ell,m+ell].
        
        # 1) expand flm => shape (Lmax+1, 2Lmax+1, 1, 1)
        flm_4d = flm[..., np.newaxis, np.newaxis]  # shape(Lmax+1,2Lmax+1,1,1)

        # 2) multiply by Y_all => shape(Lmax+1,2Lmax+1,N_theta,N_phi)
        # 3) sum over (ell,m) => axis=(0,1)
        # => shape (N_theta,N_phi)
        psi = np.sum( flm_4d * self.Y_all, axis=(0,1) )
        return psi