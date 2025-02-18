import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

import constants
from solver import split_step_solver_2d

omega = 1.0

# Potential for the 2D "harmonic oscillator"
def potential_2d(x, y):
    r2 = (x-0.5)**2 + (y-0.5)**2
    return 0.5 * constants.m * omega**2 * r2


# Dictionary data
def generate_dictionary_data(V_grid, N, dx, T_total, num_steps, K=16):
    data = []
    Xg, Yg = np.meshgrid(range(N), range(N), indexing='ij')
    for kx_ in range(-K, K+1):
        for ky_ in range(-K, K+1):
            phase = 2.0*np.pi*(kx_*Xg/N + ky_*Yg/N)
            phi_k = np.exp(1.0j*phase)
            psi_kT = split_step_solver_2d(V_grid, phi_k, N, dx, T_total, num_steps)
            data.append((phi_k, psi_kT))
    return data


def generate_test_set(V_grid, N, dx, T_total, num_steps, num_samples=100, K=16):
    """
    Generate test set of wavefunctions by random linear combinations:
        psi0(x) = sum_{|kx|,|ky|<=K} c_{kx,ky} * exp(2πi * (kx*x + ky*y)/N)
    Then evolve with the oracle solver.
    
    Returns:
    --------
    test_inputs: (num_samples, N, N) complex
    test_outputs: (num_samples, N, N) complex
    """
    Xgrid, Ygrid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    test_inputs = []
    test_outputs = []
    
    for _ in range(num_samples):
        # Initialize wavefunction to zero
        psi0 = np.zeros((N,N), dtype=np.complex128)
        
        for kx in range(-K, K+1):
            for ky in range(-K, K+1):
                # random coefficients c_{kx,ky}, can be complex
                c_real = np.random.randn()
                c_imag = np.random.randn()
                c = c_real + 1j*c_imag
                
                phase = 2.0*np.pi*(kx*Xgrid/N + ky*Ygrid/N)
                plane_wave = np.exp(1.0j * phase)
                psi0 += c * plane_wave
        
        # Optionally normalize
        norm_psi0 = np.linalg.norm(psi0)
        if norm_psi0 > 1e-14:
            psi0 /= norm_psi0
        
        # Evolve with our oracle (split-step or whichever solver you use)
        psiT = split_step_solver_2d(V_grid, psi0, N, dx, T_total, num_steps)
        
        test_inputs.append(psi0)
        test_outputs.append(psiT)
    
    return np.array(test_inputs), np.array(test_outputs)


def main():
    N = 64
    L = 2.0*np.pi
    dx = L/N
    x_vec = np.arange(N)*dx
    X, Y = np.meshgrid(x_vec, x_vec, indexing='ij')

    T_total = 1.0
    num_steps = 100
    dt = T_total / num_steps

    V_grid = potential_2d(X, Y)
    dictionary_data = generate_dictionary_data(V_grid, N, dx, T_total, num_steps, K=16)
    test_inputs, test_outputs = generate_test_set(V_grid, N, dx, T_total, num_steps, num_samples=100)

    # Build training set from dictionary
    phi_list = []
    psiT_list = []
    for (phi_k, psi_kT) in dictionary_data:
        phi_list.append(np.stack([np.real(phi_k), np.imag(phi_k)], axis=0))
        psiT_list.append(np.stack([np.real(psi_kT), np.imag(psi_kT)], axis=0))
    phi_arr = np.array(phi_list, dtype=np.float32)
    psiT_arr = np.array(psiT_list, dtype=np.float32)

    with open("dictionary_data.pkl", "wb") as f:
        pickle.dump(dictionary_data, f)

    with open("train.pkl", "wb") as f:
        pickle.dump((phi_arr, psiT_arr), f)

    with open("test.pkl", "wb") as f:
        pickle.dump((test_inputs, test_outputs), f)


if __name__ == "__main__":
    main()