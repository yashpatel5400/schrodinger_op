import torch
import numpy as np
from torch_harmonics.examples.sfno import SphericalFourierNeuralOperatorNet as SFNO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_sfno_spherical(N_theta, N_phi, num_layers=4, scale_factor=3, embed_dim=16, 
                         big_skip=True, pos_embed="lat", use_mlp=False, 
                         normalization_layer="none"):
    model = SFNO(img_size=(N_theta, N_phi), 
                 in_chans=2,
                 out_chans=2,
                 grid="equiangular",
                 num_layers=num_layers,
                 scale_factor=scale_factor,
                 embed_dim=embed_dim,
                 big_skip=big_skip,
                 pos_embed=pos_embed,
                 use_mlp=use_mlp,
                 normalization_layer=normalization_layer)
    return model


def l2loss_sphere(prd, tar, N_theta, N_phi):
    """
    L2 loss over the sphere for prd, tar of shape (batch, channels, N_theta, N_phi).

    We do:
      diff = (prd - tar)**2
      integrand = diff * sin(theta)  (with uniform dtheta, dphi)
      sum over theta, phi
      average over batch

    Parameters
    ----------
    prd, tar : (batch, channels, N_theta, N_phi) Torch tensors
    N_theta, N_phi : int
    device : torch device
    Returns
    -------
    loss : scalar
    """

    # 1) Build or cache theta, phi Tensors (or do it once outside the function).
    #    Suppose we do it here for clarity. 
    #    We'll do an equiangular grid:
    theta_vals = torch.linspace(0, np.pi, N_theta, device=device)
    phi_vals   = torch.linspace(0, 2*np.pi, N_phi, device=device)
    
    # spacing
    dtheta = (theta_vals[-1] - theta_vals[0])/(N_theta-1) if N_theta>1 else 0.0
    dphi   = (phi_vals[-1] - phi_vals[0])/(N_phi)   if N_phi>1 else 0.0
    area_elem = dtheta*dphi  # each theta-phi cell area factor (besides sin(theta))

    # We'll create a 2D mesh of shape (N_theta,N_phi), 
    # but we only need sin(theta) for each row.
    # sin_theta shape => (N_theta,1)
    sin_theta = torch.sin(theta_vals).unsqueeze(1)  # shape(N_theta,1)
    
    # 2) diff = (prd - tar)**2 => shape(batch,chan,N_theta,N_phi)
    diff = (prd - tar)**2
    
    # 3) multiply by sin(theta). We'll broadcast sin_theta aphig the phi dimension.
    #    shape => (1,1,N_theta,N_phi) * (batch,chan,N_theta,N_phi) => (batch,chan,N_theta,N_phi)
    diff_sin = diff * sin_theta.unsqueeze(0).unsqueeze(0)
    
    # 4) sum over theta & phi => shape (batch,chan)
    #    .sum(dim=-1).sum(dim=-1) => sum over last 2 dims
    diff_int = diff_sin.sum(dim=-1).sum(dim=-1)* area_elem  # shape(batch,chan)

    # 5) sum over channels if you want, or keep them separate. We'll sum them.
    diff_int = diff_int.sum(dim=-1)  # shape (batch,)
    
    # 6) average over batch
    loss = diff_int.mean()
    return loss


def train_sfno(train_loader, N_theta, N_phi, num_epochs=20):
    """
    train_loader: yields (psi_in, psi_out) of shape (batch, 2, N_theta, N_phi) for real+imag
    N_theta, N_phi: int, the spherical grid size
    device: torch device
    """
    # 1) Build the SFNO model
    model = build_sfno_spherical(N_theta, N_phi).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        count = 0
        for (psi_in, psi_out) in train_loader:
            # shape => (batch, channels, N_theta, N_phi)
            psi_in  = psi_in.to(device)
            psi_out = psi_out.to(device)
            
            optimizer.zero_grad()
            pred = model(psi_in)   # shape => (batch, channels, N_theta, N_phi)
            loss = l2loss_sphere(pred, psi_out, N_theta, N_phi)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * psi_in.size(0)
            count += psi_in.size(0)
        epoch_loss = running_loss / count
        print(f"Epoch {epoch+1}/{num_epochs}, SFNO sphere L2 Loss = {epoch_loss:.4e}")

    return model