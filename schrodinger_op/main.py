import argparse
import os
import pickle
from neuralop.models import FNO2d
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

import constants
import potentials
from dataset import random_low_order_state, construct_dataset, GRF
from estimators.fno import train_fno
from estimators.deeponet import train_onet
from estimators.linear import LinearEstimator
import solvers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_estimator(estimator, test_samples):
    est_errors = []
    for psi0, psi_true in test_samples:
        # True PDE solution
        if isinstance(estimator, LinearEstimator):
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
    N = 64          # spatial resolution
    L = 2 * np.pi   # spatial domain size
    dx = L/N        # spatial discretization
    T = 0.1         # total time evolution
    num_steps = 50  # temporal resolution (for numerical PDE solver)
    K = 16          # support of modes for train/test data (over [-K, K]^d)
    
    # ----- Generate potentials and train/test initial conditions ----- #
    V_grids = {
        "free": potentials.free_particle_potential(N),
        "harmonic_oscillator": potentials.harmonic_oscillator_potential(N, L, omega=2.0, m=constants.m),
        "barrier": potentials.barrier_potential(N, L, barrier_height=50.0, slit_width=0.2),
        "random": potentials.random_potential(N, alpha=1, beta=1, gamma=4),
    }
    V_grid = V_grids[potential]

    num_train = (2 * K + 1) ** 2 # (2K+1)^2 to match lin est. sample count
    num_test  = 50

    np.random.seed(42)
    train_samples, test_samples = [], []
    for sample_idx in range(num_train + num_test):
        psi0 = GRF(1, 1, 4, N) # random_low_order_state(N, K=K)
        psiT = solvers.time_dep.split_step_solver_2d(V_grid, psi0, N, dx, T, num_steps)
        if sample_idx < num_train:
            train_samples.append((psi0, psiT))
        else:
            test_samples.append((psi0, psiT))
    train_samples, test_samples = np.array(train_samples), np.array(test_samples)
    train_loader = construct_dataset(train_samples, batch_size=4)

    # ----- Compute errors for different estimators ----- #
    for estimator_type in estimator_types:
        print(f"Building {estimator_type} estimator...")
        
        os.makedirs(os.path.join(constants.models_dir, potential), exist_ok=True)
        if estimator_type == "linear":
            estimator = LinearEstimator(V_grid, N, dx, T, num_steps, K)
        elif estimator_type == "fno":
            estimator = train_fno(train_loader, N, K=K, num_epochs=20)
            torch.save(estimator.state_dict(), os.path.join(constants.models_dir, potential, f"{estimator_type}.pt"))
        elif estimator_type == "onet":
            estimator = train_onet(train_loader, N, num_epochs=20)
            torch.save(estimator.state_dict(), os.path.join(constants.models_dir, potential, f"{estimator_type}.pt"))

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