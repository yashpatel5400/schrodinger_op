import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import constants


def build_2d_laplacian_periodic(N, dx):
    """
    Build a sparse matrix for the 2D Laplacian with periodic BC on NxN grid,
    using second-order finite differences.
    
    Returns a sparse matrix L of shape (N^2, N^2) so that
    L * psi_vector approximates nabla^2(psi) in 2D.
    """
    # 1D Laplacian for periodic BC in each dimension
    # We can create a 1D periodic Laplacian and then do a Kronecker product for 2D.
    
    # Off-diagonals in 1D for periodic BC
    diag0 = np.full(N, -2.0)
    diag1 = np.ones(N)
    
    # For periodic BC, the first and last also connect
    # We'll build as a banded matrix and then convert to sp.diags
    # or we can do a simpler approach with indexing
    # E.g.:
    row = []
    col = []
    data = []
    for i in range(N):
        # main diag
        row.append(i)
        col.append(i)
        data.append(-2.0)
        
        # i -> i+1 mod N
        row.append(i)
        col.append((i+1)%N)
        data.append(1.0)
        
        # i -> i-1 mod N
        row.append(i)
        col.append((i-1)%N)
        data.append(1.0)
    L1D = sp.coo_matrix((data,(row,col)), shape=(N,N), dtype=float)
    
    # Combine for 2D: Lap2D = kron(I, L1D) + kron(L1D, I)
    I = sp.eye(N, format='coo', dtype=float)
    Lx = sp.kron(I, L1D, format='coo')
    Ly = sp.kron(L1D, I, format='coo')
    L2D = Lx + Ly
    
    # We'll scale by (1/dx^2) * sign
    # Typically, discrete Laplacian is (L2D / dx^2).
    L2D = L2D / (dx*dx)
    
    return L2D


def build_hamiltonian(V_grid, N, dx, hbar=1.0, m=1.0):
    """
    Builds the Hamiltonian matrix H = -(hbar^2/(2m))*Lap + V(x).
    V_grid is NxN, Lap is NxN with periodic BC.
    Returns a sparse matrix H of shape (N^2, N^2).
    """
    # Laplacian
    L2D = build_2d_laplacian_periodic(N, dx)  # shape (N^2, N^2)
    
    # potential
    # Flatten V_grid into (N^2,) and build diagonal
    V_diag = sp.diags(V_grid.ravel(), 0, format='coo')
    
    kinetic_coeff = -(hbar**2)/(2*m)
    
    H = kinetic_coeff * L2D + V_diag
    return H


# 1) Build solver for TISE
def compute_eigenpairs(V_grid, N, dx, num_eigs=10, hbar=1.0, m=1.0):
    """
    Builds the Hamiltonian, diagonalizes it, and returns the lowest num_eigs
    eigenvalues, eigenfunctions.
    
    Returns:
      E_vals: array of length num_eigs
      E_funcs: list of length num_eigs, each an NxN array (the eigenfunction)
    """
    H = build_hamiltonian(V_grid, N, dx, hbar, m)
    
    # Convert to dense if N^2 is not too large, or use sparse eigensolver
    # For N=64 => 4096x4096. This might still be feasible with enough memory, 
    # but better is to use a sparse solver (e.g. spla.eigsh). 
    # We'll do a symmetric solver for real potentials => H is Hermitian
    # But we must convert to e.g. csc_matrix for 'eigsh':
    H_csc = H.tocsc()
    
    # use scipy.sparse.linalg.eigsh for the lowest `num_eigs` eigenvalues
    E_vals, E_vecs = spla.eigsh(H_csc, k=num_eigs, which='SM')
    
    # Sort them (eigsh might not return sorted)
    idx_sort = np.argsort(E_vals)
    E_vals = E_vals[idx_sort]
    E_vecs = E_vecs[:, idx_sort]  # shape (N^2, num_eigs)
    
    # Reshape each eigenvector -> NxN
    E_funcs = []
    for i in range(num_eigs):
        vec_i = E_vecs[:, i]  # shape (N^2,)
        psi_i = vec_i.reshape((N, N))
        E_funcs.append(psi_i)
    
    return E_vals, E_funcs


# 2) For free particle on T^2, we know plane-wave eigenfunctions, energies:
#    E_{kx,ky} = (hbar^2 / (2m)) * ((2πkx/L)^2 + (2πky/L)^2).
#    psi_{kx,ky}(x,y) = (1/L)*exp(i 2π(kx x + ky y)/L).
def free_particle_analytic_eigenfunction(kx, ky, X, Y, L=2.0*np.pi):
    """
    Returns the plane-wave eigenfunction for a free particle on [0,L]^2 with
    periodic boundary conditions and quantum numbers (kx, ky) in Z.
    
    The function returned is:
       psi_{kx,ky}(x,y) = (1/L) * exp(i * 2π * (kx x + ky y)/L)
    but with domain typically L=2π => x, y in [0, 2π].
    
    Parameters
    ----------
    kx, ky : int
        The integer "wavenumbers"
    X, Y   : 2D arrays of shape (N, N)
        The mesh of x,y coordinates
    L      : float
        The size of the periodic domain (default 2π).
    
    Returns
    -------
    psi : 2D complex128 array of shape (N, N)
    """
    # plane-wave factor:
    phase = 2.0*np.pi*((kx*X)/L + (ky*Y)/L)
    psi = (1.0 / L) * np.exp(1.0j * phase)
    return psi


def compare_eigenfunction(psi_num, psi_ana):
    """
    Compare two 2D wavefunctions (psi_num, psi_ana) up to a global complex factor.
    Returns:
      alpha  : complex scalar that best aligns them (psi_num ~ alpha * psi_ana)
      relerr : relative L2 difference 
               = ||psi_num - alpha psi_ana|| / ||psi_num||
    """
    # Flatten:
    num_flat = psi_num.ravel()
    ana_flat = psi_ana.ravel()
    
    # Inner products (treat them as complex):
    # <num, ana> = sum( num_flat * conj(ana_flat) )
    numerator = np.vdot(ana_flat, num_flat)  # or num_flat.dot(np.conjugate(ana_flat))
    denominator = np.vdot(ana_flat, ana_flat)
    
    if abs(denominator) < 1e-14:
        # degenerate or zero function, skip
        alpha = 0.0
    else:
        alpha = numerator / denominator
    
    diff = num_flat - alpha*ana_flat
    norm_num = np.linalg.norm(num_flat)
    if norm_num < 1e-14:
        relerr = np.linalg.norm(diff)
    else:
        relerr = np.linalg.norm(diff)/norm_num
    return alpha, relerr


def best_linear_combination(psi_num_list, psi_target):
    """
    Given a list of degenerate numeric eigenfunctions (psi_num_list),
    find the coefficients c_i (complex) that best fit the target function psi_target.
    
    Returns c: complex array of shape (len(psi_num_list),)
    such that sum_i c_i * psi_num_list[i] best matches psi_target in L^2 sense.
    Also returns the relative L2 error.
    """
    # Flatten everything:
    Nf = len(psi_num_list)
    shape = psi_target.shape
    A = []
    for psi_n in psi_num_list:
        A.append(psi_n.ravel())
    A = np.array(A)  # shape (Nf, N^2)
    
    b = psi_target.ravel()  # shape (N^2,)
    
    # We'll do a least-squares solve A^H c ~ b
    # but we treat them as complex, so let's build a normal equations approach:
    # c = (A.H A)^{-1} A.H b
    # We'll do "vdot" carefully:
    A_HA = np.zeros((Nf, Nf), dtype=np.complex128)
    A_Hb = np.zeros(Nf, dtype=np.complex128)
    for i in range(Nf):
        for j in range(Nf):
            A_HA[i,j] = np.vdot(A[i], A[j])
        A_Hb[i] = np.vdot(A[i], b)
    
    c = np.linalg.solve(A_HA, A_Hb)
    
    # compute the best-fit function
    fit = np.zeros_like(b, dtype=np.complex128)
    for i in range(Nf):
        fit += c[i]*A[i]
    
    diff = fit - b
    err = np.linalg.norm(diff) / np.linalg.norm(b)
    
    return c, err


def check_numerical_vs_analytic_free_particle(k, rel_e_funcs, N, L=2*np.pi):
    """
    Given a list of numeric eigenfunctions (e_funcs) from compute_eigenpairs
    and a list of (kx, ky) "candidate" wave numbers, try to see if each e_funcs[i]
    matches plane-wave_{kx,ky} up to a global phase factor.
    
    e_funcs: list of NxN arrays (possibly real)
    k_candidates: list of (kx, ky) pairs of integers
    N: grid size
    L: domain size
    """
    dx = L/N
    # Build mesh:
    xvals = np.arange(N)*dx
    X, Y = np.meshgrid(xvals, xvals, indexing='ij')
    
    psi_target = free_particle_analytic_eigenfunction(k[0], k[1], X, Y, L=L)
    c, rel_err = best_linear_combination(rel_e_funcs, psi_target)
    print("Best linear combination => rel_err =", rel_err)


if __name__ == "__main__":
    N = 64
    L = 2.0*np.pi
    num_eigs = 5
    V_grid = np.zeros((N,N))  # free particle
    E_vals, E_funcs = compute_eigenpairs(V_grid, N, L/N, num_eigs=num_eigs, hbar=constants.hbar, m=constants.m)

    k_eigfunc_idx_pairs = [
        ([0, 0], [0, 1]),
        ([1, 0], [1, 5]),
        ([0, 1], [1, 5]),
        ([-1, 0], [1, 5]),
        ([0, -1], [1, 5]),
    ]
    for k, eigfunc_idx in k_eigfunc_idx_pairs:
        check_numerical_vs_analytic_free_particle(np.array(k), E_funcs[eigfunc_idx[0]:eigfunc_idx[1]], N)