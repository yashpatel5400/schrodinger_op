import numpy as np

import utils

class LinearEstimatorSpherical:
    def __init__(self, solver, N, K):
        """
        solver: function(psi0) -> psi_T 
            a PDE solver that given an initial wavefunction psi0 on a sphere 
            returns the wavefunction at time T. 
            We expect this is `split_step_solver_spherical_scipy(...)`
            specialized with certain parameters.

        N: int
           number of discrete steps for theta or phi (depends on your approach).
           We'll assume you have a spherical grid of shape (Ntheta, Nphi).
           For simplicity here, we'll let N represent Ntheta. 
           Then maybe Nphi=2N or something.

        K: int
           maximum spherical harmonic degree (analogous to Lmax)
        """
        # build dictionary
        self.dictionary_phi, self.dictionary_psi = self.generate_dictionary_data_sph(solver, N, K)

    def generate_dictionary_data_sph(self, solver, N, Lmax):
        """
        Build a dictionary of time-evolved wavefunctions in spherical harmonic space:
          for (ell,m) in 0..Lmax, -ell..ell:
            1) define a 'delta' in (ell,m) space
            2) inverse transform => phi_{ell,m} in real space
            3) evolve => psi_{ell,m}^T
            4) store

        Returns
        -------
        dictionary_phi : 2D array [ell, m+ell], each is shape (Ntheta, Nphi) complex
        dictionary_psi : 2D array [ell, m+ell], each is shape (Ntheta, Nphi) complex
        """
        # We'll assume you have consistent (theta_vals, phi_vals) for the solver 
        # or solver has them baked in. For a simpler approach, let's suppose 
        # the solver or a global config has them.
        dictionary_phi = [[None]*(2*Lmax+1) for _ in range(Lmax+1)]
        dictionary_psi = [[None]*(2*Lmax+1) for _ in range(Lmax+1)]
        
        # We'll define a shape for flm: (Lmax+1, 2Lmax+1)
        # Then "delta" in flm means flm[ell, m+ell] = 1
        for ell in range(Lmax+1):
            for m_ in range(-ell, ell+1):
                # build a zero array
                flm_delta = np.zeros((Lmax+1, 2*Lmax+1), dtype=np.complex128)
                flm_delta[ell, m_+ell] = 1.0  # "delta" at that (ell,m)
                
                # inverse transform => phi_{ell,m}
                # this yields shape (Ntheta, Nphi)
                phi_lm = utils.sph_inverse(flm_delta, Lmax)  
                
                # evolve with PDE solver
                psi_lmT = solver(phi_lm)  

                dictionary_phi[ell][m_+ell] = phi_lm
                dictionary_psi[ell][m_+ell] = psi_lmT
        
        return dictionary_phi, dictionary_psi

    def compute_estimate(self, u, K=None):
        """
        The "linear estimate" in spherical harmonic space:

            estimate(u) = sum_{ell=0..K} sum_{m=-ell..ell} c_{ell,m} * dictionary_psi[ell][m+ell],

        where c_{ell,m} = the spherical harmonic coefficient of u.

        We do a forward transform on u => c_{ell,m}. If K is not None, 
        limit ell up to K. Then sum the dictionary solutions.
        """
        # forward transform => c_{ell,m}
        Lmax = len(self.dictionary_psi)-1  # we stored up to 'K' in the constructor
        flm = utils.sph_forward(u, Lmax)
        
        # build the estimate
        Ntheta, Nphi = u.shape
        est = np.zeros((Ntheta, Nphi), dtype=np.complex128)

        # if user passes a smaller K, we only sum up to that. 
        # or we sum up to Lmax if K=None or K > Lmax.
        if K is None or K > Lmax:
            K = Lmax

        for ell in range(K+1):
            for m_ in range(-ell, ell+1):
                # skip if dictionary_psi is None for that mode
                psi_lmT = self.dictionary_psi[ell][m_+ell]
                if psi_lmT is None:
                    continue
                c_lm = flm[ell, m_+ell]
                est += c_lm * psi_lmT
        
        return est
