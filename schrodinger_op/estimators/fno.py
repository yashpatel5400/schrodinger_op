import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from neuralop.models import FNO2d
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def train_fno(train_loader, N, K=16, num_epochs=50):
    model = build_fno_model_2chan(N, K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
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