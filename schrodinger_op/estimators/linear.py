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
from solvers.time_dep import split_step_solver_2d


class LinearEstimator:
    def __init__(self, V_grid, N, dx, T, num_steps, K):
        self.dictionary_phi, self.dictionary_psi = self.generate_dictionary_data_fft(V_grid, N, dx, T, num_steps, K)

    def generate_dictionary_data_fft(
        self, V_grid, N, dx, T, num_steps, K=None
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
                psi_kT = split_step_solver_2d(V_grid, phi_k, N, dx, T, num_steps)
                
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


def random_low_order_state(N, K=16):
    """
    Construct a random wavefunction whose fft2 is nonzero only in 
    the band -K..K for both axes. Return it in real space.
    """
    freq_array = np.zeros((N,N), dtype=np.complex128)
    
    def wrap_index(k):
        # convert from [-K..K] to 'fft index' in [0..N-1]
        return k % N
    
    for kx_eff in range(-K, K+1):
        for ky_eff in range(-K, K+1):
            # pick random complex amplitude
            amp_real = np.random.randn()
            amp_imag = np.random.randn()
            c = amp_real + 1j*amp_imag
            # place it in freq_array
            kx = wrap_index(kx_eff)
            ky = wrap_index(ky_eff)
            freq_array[kx, ky] = c
    
    # go to real space
    psi0 = np.fft.ifft2(freq_array)
    
    # optionally normalize
    norm_psi = np.linalg.norm(psi0)
    if norm_psi > 1e-14:
        psi0 /= norm_psi
    
    return psi0


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