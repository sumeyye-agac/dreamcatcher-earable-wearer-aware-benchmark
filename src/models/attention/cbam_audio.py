import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, F, T]
        b, c, f, t = x.shape
        avg = x.mean(dim=(2, 3))  # [B, C]
        mx = x.amax(dim=(2, 3))  # [B, C]
        w = self.mlp(avg) + self.mlp(mx)
        w = self.sigmoid(w).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, F, T]
        avg = x.mean(dim=1, keepdim=True)  # [B, 1, F, T]
        mx, _ = x.max(dim=1, keepdim=True)  # [B, 1, F, T]
        a = torch.cat([avg, mx], dim=1)  # [B, 2, F, T]
        w = self.sigmoid(self.conv(a))  # [B, 1, F, T]
        return x * w


class CBAM(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        sa_kernel: int = 7,
        *,
        use_ca: bool = True,
        use_sa: bool = True,
    ):
        super().__init__()
        self.use_ca = bool(use_ca)
        self.use_sa = bool(use_sa)
        self.ca = ChannelAttention(channels, reduction=reduction) if self.use_ca else nn.Identity()
        self.sa = SpatialAttention(kernel_size=sa_kernel) if self.use_sa else nn.Identity()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
