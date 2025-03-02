import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

import constants
from potentials import free_particle_potential, barrier_potential, harmonic_oscillator_potential
from dataset import random_low_order_state
from solvers.time_dep import solver
from solvers.spherical import sph_inverse, sph_forward


class LinearEstimator:
    def __init__(self, solver, N, K):
        self.dictionary_phi, self.dictionary_psi = self.generate_dictionary_data_fft(solver, N, K)

    def generate_dictionary_data_fft(
        self, solver, N, K=None
    ):
        """
        Build a dictionary of time-evolved wavefunctions (psi_k^T) 
        for frequency indices kx,ky in [0..N-1].
        
        Steps:
        1) For each (kx, ky) in the allowed range (if K is not None, 
            we only keep those with effective |kx|,|ky|<=K).
        2) Construct freq_delta with a 1.0 at (kx,ky) and 0 otherwise.
        3) Inverse FFT => real-space plane wave phi_k.
        4) Evolve phi_k with 'split_step_solver' => psi_k^T.
        5) Store psi_k^T in dictionary_psi[kx,ky].
        
        Returns
        -------
        dictionary_psi : 2D array (shape = (N,N)) of real-space wavefunctions 
                        (each is shape (N,N) complex), or None if out of range.
        """
        dictionary_psi = [[None]*N for _ in range(N)]  # 2D array of wavefunctions or None
        dictionary_phi = [[None]*N for _ in range(N)]
        
        def freq_to_physical_index(k):
            """
            In standard NumPy FFT, the 'frequency' index k goes 0..N-1.
            The 'effective wavenumber' is k if k <= N//2, else k-N (i.e. negative freq).
            This helper returns that 'wrapped' integer for checking if |k_eff|<=K.
            """
            k_eff = k if k <= N//2 else k - N
            return k_eff
        
        for kx in range(N):
            for ky in range(N):
                # If we only want to keep a band -K..K in 'wrapped' sense:
                if K is not None:
                    kx_eff = freq_to_physical_index(kx)
                    ky_eff = freq_to_physical_index(ky)
                    if abs(kx_eff) > K or abs(ky_eff) > K:
                        continue

                # 1) freq_delta, shape (N,N)
                freq_delta = np.zeros((N, N), dtype=np.complex128)
                freq_delta[kx, ky] = 1.0
                
                # 2) inverse FFT => phi_k in real space
                phi_k = np.fft.ifft2(freq_delta)  # shape (N,N)
                
                # 3) evolve with the split-step solver
                psi_kT = solver(phi_k)
                
                # 4) store
                dictionary_phi[kx][ky] = phi_k
                dictionary_psi[kx][ky] = psi_kT
        
        return dictionary_phi, dictionary_psi


    def compute_estimate(self, u, K=None):
        """
        Compute the linear estimate:

            estimate(u) = sum_{kx,ky in band} U[kx,ky] * dictionary_psi[kx][ky],
        
        where U = fft2(u). If K is not None, only sum over 
        freq indices with 'wrapped' |kx|,|ky| <= K.

        Returns
        -------
        est : (N,N) complex array, the estimated time-evolved wavefunction
        """
        N = u.shape[0]
        U = np.fft.fft2(u)  # shape (N,N)
        
        est = np.zeros((N, N), dtype=np.complex128)
        
        def freq_to_physical_index(k):
            return k if k <= N//2 else k - N
        
        for kx in range(N):
            for ky in range(N):
                if self.dictionary_psi[kx][ky] is None:
                    # either out-of-band or not computed
                    continue
                if K is not None:
                    kx_eff = freq_to_physical_index(kx)
                    ky_eff = freq_to_physical_index(ky)
                    if abs(kx_eff) > K or abs(ky_eff) > K:
                        continue
                
                c = U[kx, ky]  # the coefficient in freq domain
                est += c * self.dictionary_psi[kx][ky]
        
        return est

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
                phi_lm = sph_inverse(flm_delta, Lmax)  
                
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
        flm = sph_forward(u, Lmax)
        
        # build the estimate
        Ntheta, Nphi = u.shape
        est = np.zeros((Ntheta, Nphi), dtype=np.complex128)

        # if user passes a smaller K, we only sum up to that. 
        # or we sum up to Lmax if K=None or K > Lmax.
        if K is None or K> Lmax:
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


def main():
    # Domain
    N = 64
    L = 2.0 * np.pi
    dx = L / N
    T = 0.1
    num_steps = 50
    K = 16  # keep freq in [-16..16], i.e. 'wrapped' indices

    # Potential
    omega = 2.0  # pick some frequency
    V_grid = harmonic_oscillator_potential(N, L, omega, m=constants.m)

    estimator = LinearEstimator(V_grid, N, dx, T, num_steps, K)
    
    # Suppose we have a new wavefunction:
    num_test_samples = 10
    test_samples = [random_low_order_state(N, K=K) for _ in range(num_test_samples)]
    for u_test in test_samples:
        u_est = estimator.compute_estimate(u_test)
        u_true = split_step_solver_2d(V_grid, u_test, N, dx, T, num_steps)
        err = np.linalg.norm(u_est - u_true)/(np.linalg.norm(u_true)+1e-14)
        print(f"Relative L2 error = {err:.2e}")


if __name__ == "__main__":
    main()