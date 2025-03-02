from scipy.fft import ifftn
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class DictionaryComplexDataset(Dataset):
    """
    A PyTorch Dataset that yields (phi_2chan, psi_2chan) from dictionary data,
    where each is shape (2, N, N):
      channel 0 => real part
      channel 1 => imaginary part
    """
    def __init__(self, train_samples):
        self.samples = []
        N = len(train_samples)
        for (psi0, psiT) in train_samples:
            psi0_2ch = np.stack([psi0.real, psi0.imag], axis=0)
            psiT_2ch = np.stack([psiT.real, psiT.imag], axis=0)
                    
            self.samples.append((psi0_2ch, psiT_2ch))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        psi0_2ch, psiT_2ch = self.samples[idx]
        # Convert to float tensors
        psi0 = torch.from_numpy(psi0_2ch).float()
        psiT = torch.from_numpy(psiT_2ch).float()
        return psi0, psiT


def construct_dataset(samples, batch_size):
    dataset = DictionaryComplexDataset(samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


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

def GRF(alpha, beta, gamma, N):
    # Random variables in KL expansion
    xi = np.random.randn(N, N)

    K1, K2 = np.meshgrid(np.arange(N), np.arange(N))

    # Define the (square root of) eigenvalues of the covariance operator
    coef = alpha**(1/2) *(4*np.pi**2 * (K1**2 + K2**2) + beta)**(-gamma / 2)
    #coef = alpha**(1/2) *((K1**2 + K2**2) + beta)**(-gamma / 2)

    # Construct the KL coefficients
    L = N * coef * xi

    #to make sure that the random field is mean 0
    L[0, 0] = 0

    return ifftn(L, norm='forward')