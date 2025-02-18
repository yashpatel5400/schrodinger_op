import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

import constants

def split_step_solver_2d(V_grid, psi0, N, dx, T, num_steps):
    dt = T/num_steps
    psi = psi0.astype(np.complex128)
    
    V_half = np.exp(-0.5j*dt/constants.ℏ*V_grid)
    # Kinetic factor in Fourier space
    k_vec = 2.0*np.pi*np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(k_vec, k_vec, indexing='ij')
    k2 = kx**2 + ky**2
    Kfac = np.exp(-1.0j*dt/(2*constants.m)*(constants.ℏ*k2))
    
    for _ in range(num_steps):
        psi *= V_half
        psi_k = np.fft.fft2(psi)
        psi_k *= Kfac
        psi = np.fft.ifft2(psi_k)
        psi *= V_half
    return psi


# Example check: free particle
def free_particle_exact(psi0, N, dx, T):
    psi0_k = np.fft.fft2(psi0)
    k_vec = 2.0*np.pi*np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(k_vec, k_vec, indexing='ij')
    k2 = kx**2 + ky**2
    phase = np.exp(-1.0j*constants.ℏ*k2/(2*constants.m)*T)
    psiT_k = psi0_k * phase
    return np.fft.ifft2(psiT_k)


def test_spectral_solver_free_particle():
    """
    Validates the split_step_solver_2d against the exact free-particle solution
    for three test initial wavefunctions:
      1) A single plane wave (kx=1, ky=2).
      2) A sum of two plane waves (kx=3, ky=1) and (kx=-2, ky=4).
      3) A localized Gaussian wave packet.
    Prints the relative L2 error for each test.
    """
    
    # Choose final time and number of steps
    T = 0.1
    N = 64
    L = 2.0*np.pi
    dx = L/N
    num_steps = 50

    V_grid = np.zeros((N, N))  # Force free-particle potential (V=0)
    
    # 1) Single plane wave
    #    psi0(x,y) = exp(i*(kx*x + ky*y) * 2π/N)
    kx1, ky1 = 1, 2
    Xgrid, Ygrid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    phase = 2.0*np.pi*(kx1*Xgrid/N + ky1*Ygrid/N)
    psi0_1 = np.exp(1.0j*phase)
    
    # Exact solution and Oracle solution
    psi1_exact = free_particle_exact(psi0_1, N, dx, T)
    psi1_oracle = split_step_solver_2d(V_grid, psi0_1, N, dx, T, num_steps)
    
    # Relative L2 error
    err1_num = np.linalg.norm(psi1_oracle - psi1_exact)
    err1_den = np.linalg.norm(psi1_exact) + 1e-14
    rel_err1 = err1_num / err1_den

    # 2) Sum of two plane waves
    kx2a, ky2a = 3, 1
    kx2b, ky2b = -2, 4
    phase_a = 2.0*np.pi*(kx2a*Xgrid/N + ky2a*Ygrid/N)
    phase_b = 2.0*np.pi*(kx2b*Xgrid/N + ky2b*Ygrid/N)
    psi0_2 = np.exp(1.0j*phase_a) + 0.5*np.exp(1.0j*phase_b)  # some arbitrary scaling
    
    psi2_exact = free_particle_exact(psi0_2, N, dx, T)
    psi2_oracle = split_step_solver_2d(V_grid, psi0_2, N, dx, T, num_steps)
    err2_num = np.linalg.norm(psi2_oracle - psi2_exact)
    err2_den = np.linalg.norm(psi2_exact) + 1e-14
    rel_err2 = err2_num / err2_den
    
    # 3) Localized Gaussian wave packet
    #    psi0(x,y) = exp(-alpha * ((x - x0)^2 + (y - y0)^2)) * plane wave factor
    #    For simplicity, let x0 = y0 = N/2, alpha>0
    alpha = 0.02
    x0 = y0 = N // 2
    gauss = np.exp(-alpha*((Xgrid - x0)**2 + (Ygrid - y0)**2))
    # Optionally embed a plane wave factor with wave vector (kx=2, ky=-1):
    phase_c = 2.0*np.pi*(2*Xgrid/N + -1*Ygrid/N)
    psi0_3 = gauss * np.exp(1.0j*phase_c)
    
    psi3_exact = free_particle_exact(psi0_3, N, dx, T)
    psi3_oracle = split_step_solver_2d(V_grid, psi0_3, N, dx, T, num_steps)
    err3_num = np.linalg.norm(psi3_oracle - psi3_exact)
    err3_den = np.linalg.norm(psi3_exact) + 1e-14
    rel_err3 = err3_num / err3_den
    
    # Print results
    print("Free-Particle Validation Tests:")
    print(f"  1) Single plane wave (kx={kx1}, ky={ky1}), Rel. L2 Error = {rel_err1:.2e}")
    print(f"  2) Sum of plane waves (kx={kx2a},{ky2a}) & (kx={kx2b},{ky2b}), Rel. L2 Error = {rel_err2:.2e}")
    print(f"  3) Gaussian wave packet, Rel. L2 Error = {rel_err3:.2e}")


if __name__ == "__main__":
    test_spectral_solver_free_particle()