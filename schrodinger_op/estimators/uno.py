# estimators/uno.py
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNO(nn.Module):
    """
    U-Net Neural Operator:
      in:  (B, 2, N, N)
      out: (B, 2, N, N)
    """
    def __init__(self, in_ch=2, base=48, depth=4, out_ch=2):
        super().__init__()
        C = base

        # Encoder
        self.enc1 = conv_block(in_ch, C)
        self.enc2 = conv_block(C,   2*C)
        self.enc3 = conv_block(2*C, 4*C)
        self.enc4 = conv_block(4*C, 8*C)

        self.down = nn.MaxPool2d(2)

        # Bottleneck
        self.bott = conv_block(8*C, 8*C)

        # Decoder
        self.up4 = nn.ConvTranspose2d(8*C, 4*C, 2, stride=2)
        self.dec4 = conv_block(8*C, 4*C)

        self.up3 = nn.ConvTranspose2d(4*C, 2*C, 2, stride=2)
        self.dec3 = conv_block(4*C, 2*C)

        self.up2 = nn.ConvTranspose2d(2*C, C, 2, stride=2)
        self.dec2 = conv_block(2*C, C)

        self.head = nn.Conv2d(C, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # (B, C,   N,   N)
        e2 = self.enc2(self.down(e1))   # (B, 2C, N/2, N/2)
        e3 = self.enc3(self.down(e2))   # (B, 4C, N/4, N/4)
        e4 = self.enc4(self.down(e3))   # (B, 8C, N/8, N/8)

        # Bottleneck
        b  = self.bott(e4)

        # Decoder
        d4 = self.up4(b)                 # (B, 4C, N/4, N/4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)                # (B, 2C, N/2, N/2)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)                # (B, C, N, N)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        y  = self.head(d2)               # (B, 2, N, N)
        return y


def train_uno(train_loader, num_epochs=50, lr=1e-3, base=48, depth=4, weight_decay=0.0):
    model = UNO(in_ch=2, base=base, depth=depth, out_ch=2).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        for xin, y in train_loader:
            xin = xin.to(device)  # (B,2,N,N)
            y   = y.to(device)
            opt.zero_grad()
            pred = model(xin)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            running += loss.item() * xin.size(0)
        print(f"[UNO] Epoch {epoch+1}/{num_epochs}  MSE={running/len(train_loader.dataset):.4e}")
    return model
