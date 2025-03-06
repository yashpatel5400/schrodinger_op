import numpy as np

class LinearEstimatorSpherical:
    def __init__(self, solver, sph_transform, K, cached_dictionary_phi=None, cached_dictionary_psi=None):
        """
        solver: function(psi0) -> psi_T 
            a PDE solver that given an initial wavefunction psi0 on a sphere 
            returns the wavefunction at time T. 
            We expect this is `split_step_solver_spherical_scipy(...)`
            specialized with certain parameters.

        N: int
           number of discrete steps for theta or phi (depends on your approach).
           We'll assume you have a spherical grid of shape (N_theta, N_phi).
           For simplicity here, we'll let N represent N_theta. 
           Then maybe N_phi=2N or something.

        K: int
           maximum spherical harmonic degree (analogous to Lmax)
        """
        # build dictionary
        self.sph_transform = sph_transform
        if cached_dictionary_phi is None or cached_dictionary_psi is None:
            self.dictionary_phi, self.dictionary_psi = self.generate_dictionary_data_sph(solver, K)
        else:
            self.dictionary_phi = cached_dictionary_phi
            self.dictionary_psi = cached_dictionary_psi

    def generate_dictionary_data_sph(self, solver, K):
        """
        Build a dictionary of time-evolved wavefunctions in spherical harmonic space:
          for (ell,m) in 0..Lmax, -ell..ell:
            1) define a 'delta' in (ell,m) space
            2) inverse transform => phi_{ell,m} in real space
            3) evolve => psi_{ell,m}^T
            4) store

        Returns
        -------
        dictionary_phi : 2D array [ell, m+ell], each is shape (N_theta, N_phi) complex
        dictionary_psi : 2D array [ell, m+ell], each is shape (N_theta, N_phi) complex
        """
        # We'll assume you have consistent (theta_vals, phi_vals) for the solver 
        # or solver has them baked in. For a simpler approach, let's suppose 
        # the solver or a global config has them.
        Lmax = self.sph_transform.Lmax
        dictionary_phi = [[None]*(2*Lmax+1) for _ in range(Lmax+1)]
        dictionary_psi = [[None]*(2*Lmax+1) for _ in range(Lmax+1)]
        
        # We'll define a shape for flm: (Lmax+1, 2Lmax+1)
        # Then "delta" in flm means flm[ell, m+ell] = 1
        for ell in range(K+1):
            for m_ in range(-ell, ell+1):
                print(f"Computing {(ell, m_)}")
                # build a zero array
                flm_delta = np.zeros((Lmax+1, 2*Lmax+1), dtype=np.complex128)
                flm_delta[ell, m_+ell] = 1.0  # "delta" at that (ell,m)
                
                phi_lm = self.sph_transform.inverse(flm_delta)  
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
        N_theta, N_phi = u.shape
        flm = self.sph_transform.forward(u)
        
        # build the estimate
        est = np.zeros((N_theta, N_phi), dtype=np.complex128)

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
