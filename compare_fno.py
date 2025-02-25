import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import ttest_rel

from torch.utils.data import Dataset
from neuralop.models import FNO2d
import torch.nn as nn

import constants
from estimator import LinearEstimator, random_low_order_state
from potentials import free_particle_potential, harmonic_oscillator_potential, barrier_potential
from time_dep import split_step_solver_2d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DictionaryComplexDataset(Dataset):
    """
    A PyTorch Dataset that yields (phi_2chan, psi_2chan) from dictionary data,
    where each is shape (2, N, N):
      channel 0 => real part
      channel 1 => imaginary part
    """
    def __init__(self, dictionary_phi, dictionary_psi):
        self.samples = []
        N = len(dictionary_phi)  # assuming NxN
        for kx in range(N):
            for ky in range(N):
                phi_k = dictionary_phi[kx][ky]  # (N,N) complex or None
                psi_k = dictionary_psi[kx][ky]  # (N,N) complex or None
                if phi_k is not None and psi_k is not None:
                    # Convert to 2-channel
                    # shape => (2, N, N)
                    phi_2ch = np.stack([phi_k.real, phi_k.imag], axis=0)
                    psi_2ch = np.stack([psi_k.real, psi_k.imag], axis=0)
                    
                    self.samples.append((phi_2ch, psi_2ch))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        phi_2ch, psi_2ch = self.samples[idx]
        # Convert to float tensors
        phi_t = torch.from_numpy(phi_2ch).float()
        psi_t = torch.from_numpy(psi_2ch).float()
        return phi_t, psi_t


def build_fno_model_2chan(N, K):
    """
    Build a 2D FNO that maps (batch_size, 2, N, N) -> (batch_size, 2, N, N).
    """
    model = FNO2d(
        in_channels=2,
        out_channels=2,
        resolution=(N, N),
        n_modes_height=K,
        n_modes_width=K,
        hidden_channels=32
    )
    return model


def train_fno(dictionary_phi, dictionary_psi, N, K=16, num_epochs=50, batch_size=4):
    # 1) Build the complex dataset
    dataset = DictionaryComplexDataset(dictionary_phi, dictionary_psi)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2) Build FNO with 2->2 channels
    model = build_fno_model_2chan(N, K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # 3) Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for (phi_in, psi_out) in train_loader:
            phi_in = phi_in.to(device)   # shape (B,2,N,N)
            psi_out = psi_out.to(device) # shape (B,2,N,N)
            
            optimizer.zero_grad()
            pred = model(phi_in)   # -> (B,2,N,N)
            loss = loss_fn(pred, psi_out)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * phi_in.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train MSE Loss = {epoch_loss:.4e}")
    return model

def compare_estimators(estimator, model, 
                       V_grid, N, dx, T, num_steps, K=16):
    # 4) Compare on random test samples
    #    We'll create a set of test wavefunctions and see how the linear estimator
    #    and the FNO do.
    num_test = 50
    test_samples = [random_low_order_state(N, K=K) for _ in range(num_test)]
    
    lin_est_errors = []
    fno_errors = []
    
    model.eval()
    with torch.no_grad():
        for psi0 in test_samples:
            # True PDE solution
            psi_true = split_step_solver_2d(V_grid, psi0, N, dx, T, num_steps)
            
            # Linear estimator solution
            psi_lin = estimator.compute_estimate(psi0)
            
            # FNO solution: we must feed 2ch input => 2ch output
            # Build input with shape (1,2,N,N)
            psi0_real = psi0.real
            psi0_imag = psi0.imag
            psi0_2ch = np.stack([psi0_real, psi0_imag], axis=0)  # (2,N,N)
            inp_torch = torch.from_numpy(psi0_2ch).unsqueeze(0).float().to(device)  # (1,2,N,N)
            
            pred_torch = model(inp_torch)  # (1,2,N,N)
            psi_fno_2ch = pred_torch[0].cpu().numpy()  # shape (2,N,N)
            # Reconstruct complex wavefunction
            psi_fno = psi_fno_2ch[0] + 1j*psi_fno_2ch[1]
            
            # Now compute L2 errors vs psi_true
            err_lin_num = np.linalg.norm(psi_lin - psi_true)
            err_lin_den = np.linalg.norm(psi_true) + 1e-14
            rel_lin_err = err_lin_num / err_lin_den
            lin_est_errors.append(rel_lin_err)
            
            err_fno_num = np.linalg.norm(psi_fno - psi_true)
            err_fno_den = np.linalg.norm(psi_true) + 1e-14
            rel_fno_err = err_fno_num / err_fno_den
            fno_errors.append(rel_fno_err)
    
    # 5) Paired t-test
    lin_est_errors = np.array(lin_est_errors)
    fno_errors = np.array(fno_errors)
    t_stat, p_val = ttest_rel(lin_est_errors, fno_errors)
    print(f"\nPaired t-test: t={t_stat:.4f}, p={p_val:.4e}")
    print(f"Mean(Linear)={lin_est_errors.mean():.4e},  Mean(FNO)={fno_errors.mean():.4e}")


def main(potential):
    N = 64
    L = 2*np.pi
    dx = L/N
    T = 0.1
    num_steps = 50
    K = 16
    
    # Potential
    V_grids = {
        "free": free_particle_potential(N),
        "harmonic_oscillator": harmonic_oscillator_potential(N, L, omega=2.0, m=constants.m),
        "barrier": barrier_potential(N, L, barrier_height=50.0, slit_width=0.2),
    }
    V_grid = V_grids[potential]
    
    # Build the linear estimator
    print("Building estimator...")
    estimator = LinearEstimator(V_grid, N, dx, T, num_steps, K)
    
    # Train & compare
    model = train_fno(
        estimator.dictionary_phi, 
        estimator.dictionary_psi,
        N, K=K, num_epochs=50, batch_size=4
    )
    compare_estimators(estimator, model, V_grid, N, dx, T, num_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--potential")
    args = parser.parse_args()
    main(args.potential)