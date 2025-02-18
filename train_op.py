import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# FNO
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, nx_modes, ny_modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nx_modes = nx_modes
        self.ny_modes = ny_modes
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels, nx_modes, ny_modes, 2)*0.01)
        
    def forward(self, x):
        # x_ft: (B, in_channels, Nx, Ny_half)
        x_ft = torch.fft.rfft2(x, norm='ortho')
        B, C_in, Nx, Ny_half = x_ft.shape
        
        # Prepare output in Fourier space: (B, C_out, Nx, Ny_half)
        out_ft = torch.zeros(B, self.out_channels, Nx, Ny_half, 
                            dtype=x_ft.dtype, device=x_ft.device)
        
        # Get w as a complex weight: shape (C_in, C_out, nx_modes, ny_modes)
        w_complex = self.weight[...,0] + 1j * self.weight[...,1]
        # w_complex: (C_in, C_out, nx_modes, ny_modes)

        nx = min(self.nx_modes, Nx)
        ny = min(self.ny_modes, Ny_half)

        # Accumulate over in_channels -> out_channels
        for cin in range(C_in):
            for cout in range(self.out_channels):
                out_ft[:, cout, :nx, :ny] += (
                    x_ft[:, cin, :nx, :ny] * w_complex[cin, cout, :nx, :ny]
                )

        # iFFT back to real space
        x_out = torch.fft.irfft2(out_ft, s=(Nx, 2*(Ny_half-1)), norm='ortho')
        return x_out


class SimpleFNO2d(nn.Module):
    def __init__(self, modes=16, width=32):
        super().__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Linear(2, width)  # lift 2->width
        self.convs = nn.ModuleList([SpectralConv2d(width,width,modes,modes) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(4)])
        self.fc1 = nn.Linear(width,2)

    def forward(self, x):
        # x (batch, 2, N, N)
        b, c, nx, ny = x.shape
        x = x.permute(0,2,3,1)  # (b,nx,ny,2)
        x = self.fc0(x)         # (b,nx,ny,width)
        x = x.permute(0,3,1,2)  # (b,width,nx,ny)
        for conv, w_ in zip(self.convs, self.ws):
            y = conv(x)
            x = w_(x)
            x = x + y
            x = nn.functional.gelu(x)
        x = x.permute(0,2,3,1)
        x = self.fc1(x)
        x = x.permute(0,3,1,2)
        return x


def l2_error(pred, true):
    pred_c = pred[:,0,...]+1j*pred[:,1,...]
    true_c = true[:,0,...]+1j*true[:,1,...]
    errs = []
    for i in range(pred.shape[0]):
        err = np.linalg.norm((pred_c[i]-true_c[i]).cpu().numpy())
        denom = np.linalg.norm(true_c[i].cpu().numpy())+1e-14
        errs.append(err/denom)
    return np.mean(errs)


def train_models(phi_arr, psiT_arr, test_inputs, test_outputs, dictionary_data):
    # Define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fno_model = SimpleFNO2d(modes=16, width=32).to(device)
    optimizer = optim.Adam(fno_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # Set up datasets
    train_dataset = TensorDataset(
        torch.from_numpy(phi_arr),
        torch.from_numpy(psiT_arr),
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_inp_torch = torch.from_numpy(
        np.stack([np.real(test_inputs), np.imag(test_inputs)], axis=1)
    ).float().to(device)
    test_out_torch = torch.from_numpy(
        np.stack([np.real(test_outputs), np.imag(test_outputs)], axis=1)
    ).float().to(device)

    # Train model
    train_losses, test_errors = [], []
    epochs = 20
    for ep in range(epochs):
        fno_model.train()
        run_loss = 0
        for bx, by in train_loader:
            print("s")
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = fno_model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()*bx.size(0)
        train_loss = run_loss/len(train_loader.dataset)
        fno_model.eval()
        with torch.no_grad():
            # Evaluate on 100 test samples
            pred_list = []
            for i in range(test_inp_torch.shape[0]):
                inp_i = test_inp_torch[i:i+1]
                out_i = fno_model(inp_i)
                pred_list.append(out_i)
            pred_test = torch.cat(pred_list, dim=0)
            test_err = l2_error(pred_test, test_out_torch)
        train_losses.append(train_loss)
        test_errors.append(test_err)
        print(f"Epoch {ep+1}, train_loss={train_loss:.4e}, test_err={test_err:.4e}")

    # Linear estimator
    dictionary_phi = np.array([d[0] for d in dictionary_data])  # shape (#k, N, N)
    dictionary_psi = np.array([d[1] for d in dictionary_data])  # shape (#k, N, N)
    def linear_estimator(u):
        conj_phi = np.conjugate(dictionary_phi)
        coefs = np.tensordot(conj_phi, u, axes=([1,2],[0,1]))
        pred = np.tensordot(coefs, dictionary_psi, axes=(0,0))
        return pred

    # Evaluate on test set
    fno_model.eval()
    fno_errors, lin_errors = [], []
    with torch.no_grad():
        for i in range(test_inp_torch.shape[0]):
            # FNO
            inp_i = test_inp_torch[i:i+1]
            out_i = fno_model(inp_i)
            err_i = l2_error(out_i, test_out_torch[i:i+1])
            fno_errors.append(err_i)

    for i in range(len(test_inputs)):
        u0 = test_inputs[i]
        uT = test_outputs[i]
        uT_est = linear_estimator(u0)
        err = np.linalg.norm(uT_est - uT)
        denom = np.linalg.norm(uT)
        lin_errors.append(err/(denom+1e-14))

    fno_errors = np.array(fno_errors)
    lin_errors = np.array(lin_errors)

    # Paired t-test
    t_stat, p_val = ttest_rel(fno_errors, lin_errors)
    print(f"FNO mean error={np.mean(fno_errors):.4e}, Linear mean error={np.mean(lin_errors):.4e}")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_val:.4e}")

    # Plot
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_errors, label='Test Error')
    plt.legend()
    plt.savefig("errors.png")


if __name__ == "__main__":
    with open("train.pkl", "rb") as f:
        (phi_arr, psiT_arr) = pickle.load(f)

    with open("test.pkl", "rb") as f:
        (test_inputs, test_outputs) = pickle.load(f)

    with open("dictionary_data.pkl", "rb") as f:
        dictionary_data = pickle.load(f)

    train_models(phi_arr, psiT_arr, test_inputs, test_outputs, dictionary_data)