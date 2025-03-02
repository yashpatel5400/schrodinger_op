import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepONet2D(nn.Module):
    """
    A simple 2D DeepONet that maps a 2-channel NxN input wavefunction 
    (real+imag) to a 2-channel NxN output wavefunction.

    Implementation outline (very simplified):
      - 'branch_net': consumes the entire input wavefunction 
         (flattened real+imag => dimension = 2*N*N) => embedding B in R^branch_dim
      - 'trunk_net': for each coordinate (x,y), produce an embedding T(x,y) in R^branch_dim
         Then final output at (x,y) is: sum_j [ B[j] * T_j(x,y) ] 
         in R^2 (the 2 output channels). We store the trunk net to produce 
         2 * branch_dim channels => the first half for real, the second half for imag.

    For NxN output, we gather all coordinate points in a single pass to produce NxN output.
    """
    def __init__(self, N, branch_dim=64, hidden_branch=128, hidden_trunk=128):
        """
        N : int 
            input wavefunction resolution NxN
        branch_dim : int
            dimension of the latent embedding from the branch net
        hidden_branch, hidden_trunk : int
            hidden layer sizes in MLPs for branch/trunk

        We'll store a grid of coordinates x,y in [-1,1], or [0,1], etc. 
        for the trunk net queries.
        """
        super().__init__()
        self.N = N
        self.branch_dim = branch_dim
        
        # For simplicity, define coordinate mesh in [-1,1] x [-1,1] 
        # or [0,1], up to your PDE's domain conventions.
        # We'll do a uniform mesh for trunk queries:
        xs = np.linspace(-1,1,N)
        ys = np.linspace(-1,1,N)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        coords = np.stack([X, Y], axis=-1)  # shape (N, N, 2)
        coords_flat = coords.reshape(-1, 2) # shape (N^2, 2)
        self.register_buffer("coords_flat", torch.from_numpy(coords_flat).float())  
          # So we can do trunk forward in a single pass

        # 1) Branch net: MLP mapping 2*N*N -> branch_dim
        #    We'll flatten the wavefunction (2ch*N*N) into a single vector 
        #    and feed it to MLP.
        self.branch_net = nn.Sequential(
            nn.Linear(2*N*N, hidden_branch),
            nn.ReLU(),
            nn.Linear(hidden_branch, hidden_branch),
            nn.ReLU(),
            nn.Linear(hidden_branch, branch_dim)
        )

        # 2) Trunk net: MLP mapping (x,y) -> 2*branch_dim
        #    Because we want to produce 'branch_dim' channels for real part 
        #    and 'branch_dim' for imag part => total 2*branch_dim. 
        #    We'll do a simple MLP with 'hidden_trunk' size.
        self.trunk_net = nn.Sequential(
            nn.Linear(2, hidden_trunk),
            nn.ReLU(),
            nn.Linear(hidden_trunk, hidden_trunk),
            nn.ReLU(),
            nn.Linear(hidden_trunk, 2*branch_dim)
        )

    def forward(self, inp):
        """
        inp: (batch_size, 2, N, N) 
             the wavefunction's real+imag channels.
        Returns:
             (batch_size, 2, N, N) wavefunction's real+imag.
        """
        B = inp.size(0)
        # Flatten wavefunction to (B, 2*N*N)
        inp_flat = inp.view(B, -1)  # shape (B, 2*N*N)
        branch_out = self.branch_net(inp_flat)  # shape (B, branch_dim)

        # trunk: we'll evaluate trunk_net at each coordinate => shape (N^2, 2*branch_dim)
        trunk_out = self.trunk_net(self.coords_flat)  # shape (N^2, 2*branch_dim)

        # We'll produce output for each batch item. 
        # The standard DeepONet approach: 
        #   output(x,y) = sum_{j=1..branch_dim} [ B[j] * trunk_j(x,y) ] for real part,
        #                 sum_{j=1..branch_dim} [ B[j] * trunk_{j+branch_dim}(x,y) ] for imag part.
        # We'll do this for each sample in the batch. 
        # So trunk_out is shape (N^2, 2*branch_dim). 
        # branch_out is shape (B, branch_dim). 
        # We'll do an outer product approach.

        # reshape trunk_out => (N^2, 2, branch_dim)
        trunk_out_resh = trunk_out.view(self.N*self.N, 2, self.branch_dim)  
          # The first dimension is real/imag index, the second is branch_dim
          # Actually we define trunk_out_resh[i,0,:] = real trunk, trunk_out_resh[i,1,:] = imag trunk

        # We'll do: 
        #   real_out[i,batch] = sum_j [ branch_out[batch,j]* trunk_out_resh[i,0,j] ]
        #   imag_out[i,batch] = sum_j [ branch_out[batch,j]* trunk_out_resh[i,1,j] ]
        # We can use batch matrix multiplication or just do a for-loop. 
        # For simplicity, we do a direct approach:

        # trunk_out_resh => shape (N^2, 2, branch_dim)
        # branch_out => shape (B, branch_dim)
        # We'll do real_out => shape (B, N^2), imag_out => shape (B, N^2)
        # real_out[b,i] = (branch_out[b,:]) dot ( trunk_out_resh[i,0,:] )
        # imag_out[b,i] = (branch_out[b,:]) dot ( trunk_out_resh[i,1,:] )

        # We'll do this with a small gemm:
        # real_part = trunk_out_resh[:,0,:]  shape (N^2, branch_dim)
        # imag_part = trunk_out_resh[:,1,:]  shape (N^2, branch_dim)

        real_part = trunk_out_resh[:, 0, :]  # shape (N^2, branch_dim)
        imag_part = trunk_out_resh[:, 1, :]  # shape (N^2, branch_dim)

        # We'll do real_out = ( branch_out @ real_part^T ), shape (B, N^2)
        # so real_part^T is shape (branch_dim, N^2)
        real_part_t = real_part.transpose(0,1)  # shape (branch_dim, N^2)
        real_out = torch.matmul(branch_out, real_part_t)  # (B, N^2)

        imag_part_t = imag_part.transpose(0,1)  # shape (branch_dim, N^2)
        imag_out = torch.matmul(branch_out, imag_part_t)  # (B, N^2)

        # Now we reshape => (B, 2, N, N)
        # channel0= real, channel1= imag
        out = torch.stack([real_out, imag_out], dim=1)  # shape (B,2,N^2)
        out = out.view(B, 2, self.N, self.N)            # shape (B,2,N,N)

        return out


def build_onet_model(N, branch_dim=64, hidden_branch=128, hidden_trunk=128):
    model = DeepONet2D(N, branch_dim=branch_dim, 
                       hidden_branch=hidden_branch, 
                       hidden_trunk=hidden_trunk)
    return model


def train_onet(train_loader, N, num_epochs=20):
    """
    train_samples: list of (psi0, psiT) pairs in complex NxN 
    N: domain resolution
    """
    # Build model
    model = build_onet_model(N).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for (psi0_in, psiT_out) in train_loader:
            psi0_in = psi0_in.to(device)   # shape (B,2,N,N)
            psiT_out = psiT_out.to(device) # shape (B,2,N,N)

            optimizer.zero_grad()
            pred = model(psi0_in)   # -> (B,2,N,N)
            loss = loss_fn(pred, psiT_out)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * psi0_in.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"DeepONet Epoch {epoch+1}/{num_epochs}, Train MSE Loss = {epoch_loss:.4e}")
    return model
