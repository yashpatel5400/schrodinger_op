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
from solvers.time_indep import compute_eigenpairs 


def time_indep_evolution(V_grid, psi0, N, dx, T, 
                         hbar=1.0, m=1.0, num_eigs=None):
    """
    Evolve psi0 -> psi(T) by expanding in the eigenbasis of
    H = -(hbar^2/(2m)) Lap + V_grid, then applying time evolution 
    exp(-i E_n T / hbar) to each component.

    Parameters
    ----------
    V_grid : (N, N) array
        The potential on the NxN grid (here V=0 for free particle, but can be anything).
    psi0 : (N, N) complex array
        Initial wavefunction at t=0
    N : int
        Number of grid points in each dimension
    dx : float
        Grid spacing (L/N)
    T : float
        Time at which we want the wavefunction
    hbar : float, optional
        Planck constant / 2π
    m : float, optional
        Particle mass
    num_eigs : int or None
        Number of eigenfunctions to compute/use. If None, we use N*N (the full basis).
        For large N, that might be too big. For demonstration, you can use a subset 
        if you know psi0 only has significant overlap with some subset of modes.

    Returns
    -------
    psi_T : (N, N) complex array
        The wavefunction at time T
    """
    # If num_eigs is None, default to the full dimension N^2 (but that might be big!)
    if num_eigs is None:
        num_eigs = N*N
    
    # 1) Compute eigenpairs
    E_vals, E_funcs = compute_eigenpairs(V_grid, N, dx, 
                                         num_eigs=num_eigs,
                                         hbar=hbar, m=m)
    # 2) Flatten psi0
    psi0_flat = psi0.ravel()
    
    # We'll accumulate psi(T) in a flattened array
    psiT_flat = np.zeros(N*N, dtype=np.complex128)
    
    # 3) For each eigenpair, compute the projection coefficient
    for i in range(num_eigs):
        # e_funcs[i] shape = (N, N)
        phi_n = E_funcs[i]
        phi_n_flat = phi_n.ravel()
        
        # (Optional) normalize phi_n
        norm_phi_n = np.linalg.norm(phi_n_flat)
        if norm_phi_n < 1e-14:
            continue
        phi_n_flat /= norm_phi_n
        
        # Coefficient A_n = <phi_n, psi0>
        # We'll use the discrete complex conjugate dot product
        A_n = np.vdot(phi_n_flat, psi0_flat)  # sum(phi_n^*(x) * psi0(x))
        
        # 4) multiply by the time evolution factor
        phase = np.exp(-1j * E_vals[i] * T / hbar)
        contrib = A_n * phase
        
        # 5) Add to psi(T)
        psiT_flat += contrib * phi_n_flat
    
    # Reshape back to (N, N)
    psi_T = psiT_flat.reshape((N, N))
    return psi_T


def split_step_solver_2d(V_grid, psi0, N, dx, T, num_steps):
    dt = T/num_steps
    psi = psi0.astype(np.complex128)
    
    V_half = np.exp(-0.5j*dt/constants.hbar*V_grid)
    # Kinetic factor in Fourier space
    k_vec = 2.0*np.pi*np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(k_vec, k_vec, indexing='ij')
    k2 = kx**2 + ky**2
    Kfac = np.exp(-1.0j*dt/(2*constants.m)*(constants.hbar*k2))
    
    for _ in range(num_steps):
        psi *= V_half
        psi_k = np.fft.fft2(psi)
        psi_k *= Kfac
        psi = np.fft.ifft2(psi_k)
        psi *= V_half
    return psi


def split_step_solver_2d_time_varying(V_fn, psi0, N, dx, T, num_steps):
    # used to solve if potential is time-varying (V_fn is now function that maps t -> field)
    dt = T / num_steps
    psi = psi0.astype(np.complex128).copy()

    k_vec = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
    kx, ky = np.meshgrid(k_vec, k_vec, indexing='ij')
    k2 = kx**2 + ky**2
    Kfac = np.exp(-1.0j * dt/(2.0*constants.m)*(constants.hbar*k2))

    for n in range(num_steps):
        t_n = n * dt
        t_np1 = (n+1)*dt

        V_n = V_fn(t_n)
        V_np1 = V_fn(t_np1)

        # half-step at t_n
        V_half_n = np.exp(-0.5j * dt/constants.hbar * V_n)
        psi *= V_half_n

        # full-step kinetic
        psi_k = np.fft.fft2(psi)
        psi_k *= Kfac
        psi = np.fft.ifft2(psi_k)

        # half-step at t_{n+1}
        V_half_np1 = np.exp(-0.5j * dt/constants.hbar * V_np1)
        psi *= V_half_np1

    return psi


def solver(V, psi0, N, dx, T, num_steps):
    if isinstance(V, np.ndarray):
        return split_step_solver_2d(V, psi0, N, dx, T, num_steps)
    return split_step_solver_2d_time_varying(V, psi0, N, dx, T, num_steps)


def test_free_particle():
    """
    Compare the time evolution of split_step_solver_2d vs.
    time_indep_evolution for three test initial wavefunctions:
      1) Single plane wave
      2) Sum of plane waves
      3) Localized Gaussian wave packet
    
    For a free particle (V=0).
    """
    # Choose final time and number of steps
    T = 0.1
    N = 64
    L = 2.0*np.pi
    dx = L/N
    num_steps = 50

    V_grid = free_particle_potential(N)

    # build coordinate mesh
    Xgrid, Ygrid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    
    # 1) Single plane wave
    kx1, ky1 = 1, 2
    phase = 2.0*np.pi*(kx1*Xgrid/N + ky1*Ygrid/N)
    psi0_1 = np.exp(1.0j*phase)
    
    # Evolve with split-step
    psi1_oracle = split_step_solver_2d(V_grid, psi0_1, N, dx, T, num_steps)
    
    # Evolve with time-indep approach
    psi1_eig = time_indep_evolution(V_grid, psi0_1, N, dx, T, hbar=1.0, m=1.0, num_eigs=256)
    
    # Compare
    err_num = np.linalg.norm(psi1_oracle - psi1_eig)
    err_den = np.linalg.norm(psi1_eig) + 1e-14
    rel_err1 = err_num / err_den
    print(f"[Case 1] Single plane wave => relative L2 error = {rel_err1:.2e}")

    # 2) Sum of two plane waves
    kx2a, ky2a = 3, 1
    kx2b, ky2b = -2, 4
    phase_a = 2.0*np.pi*(kx2a*Xgrid/N + ky2a*Ygrid/N)
    phase_b = 2.0*np.pi*(kx2b*Xgrid/N + ky2b*Ygrid/N)
    psi0_2 = np.exp(1.0j*phase_a) + 0.5*np.exp(1.0j*phase_b)
    
    psi2_oracle = split_step_solver_2d(V_grid, psi0_2, N, dx, T, num_steps)
    psi2_eig = time_indep_evolution(V_grid, psi0_2, N, dx, T, num_eigs=256)
    err2_num = np.linalg.norm(psi2_oracle - psi2_eig)
    err2_den = np.linalg.norm(psi2_eig) + 1e-14
    rel_err2 = err2_num / err2_den
    print(f"[Case 2] Sum of plane waves => relative L2 error = {rel_err2:.2e}")

    # 3) Localized Gaussian wave packet
    alpha = 0.02
    x0 = y0 = N // 2
    gauss = np.exp(-alpha*((Xgrid - x0)**2 + (Ygrid - y0)**2))
    phase_c = 2.0*np.pi*(2*Xgrid/N + -1*Ygrid/N)
    psi0_3 = gauss * np.exp(1.0j*phase_c)
    
    psi3_oracle = split_step_solver_2d(V_grid, psi0_3, N, dx, T, num_steps)
    psi3_eig = time_indep_evolution(V_grid, psi0_3, N, dx, T, num_eigs=1024)
    err3_num = np.linalg.norm(psi3_oracle - psi3_eig)
    err3_den = np.linalg.norm(psi3_eig) + 1e-14
    rel_err3 = err3_num / err3_den
    print(f"[Case 3] Gaussian wave packet => relative L2 error = {rel_err3:.2e}")


def test_harmonic_oscillator():
    """
    Compare the two methods for a harmonic oscillator potential.
    We'll pick an initial wavefunction, e.g. a Gaussian near center,
    and evolve it to time T.
    """
    N = 64
    L = 2*np.pi
    dx = L/N
    T = 0.1
    num_steps = 50
    
    omega = 2.0  # pick some frequency
    V_grid = harmonic_oscillator_potential(N, L, omega, m=constants.m)

    # initial wavefunction: Gaussian near center
    x0 = y0 = N//2
    Xgrid, Ygrid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    gauss = np.exp(-0.05*((Xgrid - x0)**2 + (Ygrid - y0)**2))
    # Possibly add a plane wave factor
    psi0 = gauss.astype(np.complex128)
    
    psi_td = split_step_solver_2d(V_grid, psi0, N, dx, T, num_steps)
    psi_eig = time_indep_evolution(V_grid, psi0, N, dx, T,
                                   hbar=constants.hbar, m=constants.m, 
                                   num_eigs=512)
    rel_err = (np.linalg.norm(psi_td - psi_eig)
               /(np.linalg.norm(psi_eig)+1e-14))
    print(f"[Harmonic Osc] Relative L2 error = {rel_err:.2e}")


def test_barrier():
    """
    Compare time evolution for a barrier potential
    """
    N = 64
    L = 2*np.pi
    dx = L/N
    T = 0.2
    num_steps = 100
    
    V_grid = barrier_potential(N, L, barrier_height=50.0, slit_width=0.2)
    
    # initial wavefunction: plane wave from left side plus some localization
    Xgrid, Ygrid = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    x0 = 10
    alpha = 0.02
    gauss = np.exp(-alpha*((Xgrid - x0)**2 + (Ygrid - N/2)**2))
    # traveling wave in +x direction:
    kx = 2
    phase = 2*np.pi*(kx*Xgrid/N)
    psi0 = gauss*np.exp(1j*phase)
    
    psi_td = split_step_solver_2d(V_grid, psi0, N, dx, T, num_steps)
    psi_eig = time_indep_evolution(V_grid, psi0, N, dx, T,
                                   hbar=constants.hbar, m=constants.m,
                                   num_eigs=512)
    rel_err = (np.linalg.norm(psi_td - psi_eig)
               /(np.linalg.norm(psi_eig)+1e-14))
    print(f"[Barrier Potential] Relative L2 error = {rel_err:.2e}")


if __name__ == "__main__":
    test_free_particle()
    test_harmonic_oscillator()
    test_barrier()