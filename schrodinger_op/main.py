import argparse
import os
import pickle
from neuralop.models import FNO2d
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
import shtns

import constants
import potentials
import solvers.time_dep
import solvers.spherical
from dataset import construct_dataset, GRF, GRF_spherical

from estimators.fno import train_fno
from estimators.sfno import train_sfno
from estimators.deeponet import train_onet
from estimators.linear import LinearEstimator
from estimators.linear_spherical import LinearEstimatorSpherical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_estimator(estimator, test_samples):
    est_errors = []
    for psi0, psi_true in test_samples:
        # True PDE solution
        if isinstance(estimator, LinearEstimator) or isinstance(estimator, LinearEstimatorSpherical):
            # Linear estimator solution
            psi_est = estimator.compute_estimate(psi0)
        else:
            estimator.eval()
            
            # FNO solution: we must feed 2ch input => 2ch output
            # Build input with shape (1,2,N,N)
            psi0_real = psi0.real
            psi0_imag = psi0.imag
            psi0_2ch = np.stack([psi0_real, psi0_imag], axis=0)  # (2,N,N)
            inp_torch = torch.from_numpy(psi0_2ch).unsqueeze(0).float().to(device)  # (1,2,N,N)
            
            pred_torch = estimator(inp_torch)  # (1,2,N,N)
            psi_fno_2ch = pred_torch[0].detach().cpu().numpy()  # shape (2,N,N)
            # Reconstruct complex wavefunction
            psi_est = psi_fno_2ch[0] + 1j*psi_fno_2ch[1]

        # Now compute L2 errors vs psi_true
        err_num = np.linalg.norm(psi_est - psi_true)
        err_den = np.linalg.norm(psi_true) + 1e-14
        rel_err = err_num / err_den
        est_errors.append(rel_err)
    return np.array(est_errors)


def main(potential, estimator_types):
    spherical_coords = potential in ["coulomb", "dipole"] # only Coulomb and dipole potentials use spherical (for now)
    
    # --- Spherical coordinates constants --- #
    Lmax    = 20   # modes used for projections to/from spherical coordinates (implicitly taken to be N/2 for FFT Euclidean case)
    N_theta = 32   # spatial discretizations (theta/phi)
    N_phi   = 64
    K_sph   = 10   # support of modes for linear estimator

    # sph_transform = SphericalHarmonicsTransform(Lmax, N_theta, N_phi)
    sph_transform = shtns.sht(Lmax)
    sph_transform.set_grid(N_theta, N_phi)

    # --- Euclidean coordinates constants --- #
    N = 64          # spatial resolution
    L = 2 * np.pi   # spatial domain size
    dx = L/N        # spatial discretization
    K_euc = 16      # support of modes for train/test data (over [-K, K]^d)

    # --- Temporal discretization constants --- #
    T = 0.1         # total time evolution
    num_steps = 50  # temporal resolution (for numerical PDE solver)
    
    # ----- Generate potentials and train/test initial conditions ----- #
    Vs = {
        "free": potentials.free_particle_potential(N),
        "harmonic_oscillator": potentials.harmonic_oscillator_potential(N, L, omega=2.0, m=constants.m),
        "barrier": potentials.barrier_potential(N, L, barrier_height=50.0, slit_width=0.2),
        "random": potentials.random_potential(N, alpha=1, beta=1, gamma=4),
        "paul_trap": lambda t : potentials.paul_trap(N, L, t, U0=10.0, V0=15.0, omega=3.0, r0=2.0),
        "coulomb": potentials.uniform_sphere(N_theta, N_phi),
        "dipole": potentials.dipole_potential_sphere(N_theta, N_phi),
    }
    V = Vs[potential]

    if spherical_coords:
        solver = lambda psi0 : solvers.spherical.split_step_solver_spherical(V, psi0, sph_transform, T, num_steps)
    else:
        solver = lambda psi0 : solvers.time_dep.solver(V, psi0, N, dx, T, num_steps)
        
    
    num_train = (Lmax + 1) ** 2 if spherical_coords else (2 * K_euc + 1) ** 2 # (2K+1)^2 to match lin est. sample count
    num_test  = 50

    # np.random.seed(42)
    train_fn = os.path.join(constants.data_dir, potential, "train.npy")
    test_fn  = os.path.join(constants.data_dir, potential, "test.npy")

    if os.path.exists(train_fn) and os.path.exists(test_fn):
        train_samples, test_samples = np.load(train_fn), np.load(test_fn)
    else:
        train_samples, test_samples = [], []
        for sample_idx in range(num_train + num_test):
            print(f"Computing sample: {sample_idx}...")
            if spherical_coords:
                psi0 = GRF_spherical(1, 1, 6, sph_transform)
            else:
                psi0 = GRF(1, 1, 4, N)

            psiT = solver(psi0)
            if sample_idx < num_train:
                train_samples.append((psi0, psiT))
            else:
                test_samples.append((psi0, psiT))
        train_samples, test_samples = np.array(train_samples), np.array(test_samples)

        os.makedirs(os.path.join(constants.data_dir, potential), exist_ok=True)
        np.save(train_fn, train_samples)
        np.save(test_fn, test_samples)
    
    train_loader = construct_dataset(train_samples, batch_size=4)

    # ----- Compute errors for different estimators ----- #
    for estimator_type in estimator_types:
        print(f"Building {estimator_type} estimator...")
        
        torch_cache_fn = os.path.join(constants.models_dir, potential, f"{estimator_type}.pt")
        lin_cache_fn   = os.path.join(constants.models_dir, potential, f"{estimator_type}.pkl")

        os.makedirs(os.path.join(constants.models_dir, potential), exist_ok=True)
        if estimator_type == "linear":
            if spherical_coords:
                if os.path.exists(lin_cache_fn):
                    print(f"Loading cached estimator from: {lin_cache_fn}...")
                    with open(lin_cache_fn, "rb") as f:
                        (cached_dictionary_phi, cached_dictionary_psi) = pickle.load(f)
                    estimator = LinearEstimatorSpherical(solver, sph_transform, K_sph, cached_dictionary_phi, cached_dictionary_psi)

                else:
                    estimator = LinearEstimatorSpherical(solver, sph_transform, K_sph)
                    with open(lin_cache_fn, "wb") as f:
                        pickle.dump((estimator.dictionary_phi, estimator.dictionary_psi), f)
            else:
                estimator = LinearEstimator(solver, N, K_euc)
        elif estimator_type == "fno":
            if spherical_coords:
                estimator = train_sfno(train_loader, N_theta, N_phi, num_epochs=20)
                torch.save(estimator.state_dict(), torch_cache_fn)
            else:
                estimator = train_fno(train_loader, N, K=K_euc, num_epochs=20)
                torch.save(estimator.state_dict(), torch_cache_fn)
        elif estimator_type == "onet":
            estimator = train_onet(train_loader, N, num_epochs=20)
            torch.save(estimator.state_dict(), torch_cache_fn)

        os.makedirs(os.path.join(constants.results_dir, potential), exist_ok=True)
        df = pd.DataFrame.from_dict({estimator_type : test_estimator(estimator, test_samples)})
        df.to_csv(os.path.join(constants.results_dir, potential, f"{estimator_type}.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential")
    parser.add_argument("--estimator", default="all", help="Estimator to fit/evaluate. One of: linear, fno, onet, or all")
    args = parser.parse_args()

    if args.estimator == "all":
        estimator_types = ["linear", "fno", "onet"]
    else:
        estimator_types = [args.estimator]
    main(args.potential, estimator_types)