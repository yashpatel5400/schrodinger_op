import numpy as np
from scipy.fft import ifftn


def get_mesh(N, L, d=2):
    dx = L/N # NOTE: uniform gridding over domain (may wish to make non-uniform later)
    vals = [np.arange(N)*dx for _ in range(d)]
    return np.meshgrid(*vals, indexing='ij')


def free_particle_potential(N):
    return np.zeros((N, N))


def barrier_potential(N, L, barrier_height=20.0, slit_width=0.4):
    """
    Place a vertical barrier at x=L/2 except for a small 'slit' in y.
    """
    X, Y = get_mesh(N, L)
    Vgrid = np.zeros((N,N))
    # barrier region ~ x ~ L/2 => i0 = N//2
    i0 = N//2
    # We allow a window in y centered at ymid = L/2
    # slit_width is in physical units
    j_low = int((0.5 - slit_width/2.0)*N)
    j_high= int((0.5 + slit_width/2.0)*N)
    
    for j in range(N):
        if j<j_low or j>j_high:
            Vgrid[i0,j] = barrier_height
    return Vgrid


def harmonic_oscillator_potential(N, L, omega=1.0, m=1.0):
    """
    Build 2D harmonic oscillator potential on [0,L]^2,
    centered at (L/2, L/2) for a grid NxN.
    
    V(x,y) = 0.5*m*omega^2 * ((x - L/2)^2 + (y - L/2)^2).
    """
    X, Y = get_mesh(N, L)
    Xc = X - L/2
    Yc = Y - L/2
    return 0.5*m*(omega**2)*(Xc**2 + Yc**2)


def random_potential(N, alpha, beta, gamma):
    # Random variables in KL expansion
    xi = np.random.randn(N, N)
    K1, K2 = np.meshgrid(np.arange(N), np.arange(N))

    # Define the (square root of) eigenvalues of the covariance operator
    coef = alpha**(1/2) *(4*np.pi**2 * (K1**2 + K2**2) + beta)**(-gamma / 2)
        
    # Construct the KL coefficients
    L = N * coef * xi

    #to make sure that the random field is mean 0
    L[0, 0] = 0
    return ifftn(L, norm='forward')


def paul_trap(N, L, t, U0=1.0, V0=1.0, omega=1.0, r0=1.0):
    X, Y = get_mesh(N, L)
    factor = (U0 + V0 * np.cos(omega*t)) / (r0**2)
    return factor * (X**2 + Y**2)


def get_spherical_mesh(N_theta, N_phi):
    theta_vals = np.linspace(0, np.pi, N_theta)
    phi_vals   = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    return np.meshgrid(theta_vals, phi_vals, indexing='ij')


def uniform_sphere(N_theta, N_phi, k=1.0, e=1.0):
    """
    Radially-symmetric Coulomb potential on the unit sphere => constant = -k e^2.
    """
    V_grid = np.full((N_theta, N_phi), fill_value=(-k*e**2), dtype=np.float64)
    return V_grid


def dipole_potential_sphere(N_theta, N_phi, V0=1.0):
    """
    Build a grid V(theta,phi) = V0 cos(theta)
    on an (N_theta x N_phi) mesh of angles.
    """
    TH, PH = get_spherical_mesh(N_theta, N_phi)
    V_grid = V0 * np.cos(TH)  # shape (N_theta, N_phi)
    return V_grid

def shaken_lattice(N, L, t, V0=4.0, k=4*np.pi, A=0.08, omega=15.0):
    X, Y = get_mesh(N, L)
    phase = k * (X - A * np.sin(omega * t))
    return V0 * np.cos(phase) + V0 * np.cos(k * Y)


def gaussian_pulse(N, L, t, V0=100.0, x0=0.0, y0=0.0,
                   sigma_x=1.2, sigma_y=1.2, sigma_t=1.0,
                   tcenters=(0.0,)):
    X, Y = get_mesh(N, L)
    spatial = np.exp(-((X - x0)**2)/(2.0 * sigma_x**2)
                     -((Y - y0)**2)/(2.0 * sigma_y**2))
    # allow multiple pulses
    if np.isscalar(tcenters):
        tcenters = (tcenters,)
    time_envelope = sum(np.exp(-((t - t0)**2)/(2.0 * sigma_t**2)) for t0 in tcenters)
    return V0 * spatial * time_envelope