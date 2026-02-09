import torch.nn as nn

from src.models.attention.cbam_audio import CBAM


class TinyCNN_CBAM(nn.Module):
    """TinyCNN with CBAM attention (~30K params)

    Architecture: 1→16→32(+CBAM)→64 channels
    Adds CBAM attention after the second conv block.
    """

    def __init__(
        self,
        n_classes: int,
        cbam_reduction: int = 8,
        cbam_sa_kernel: int = 7,
        *,
        use_ca: bool = True,
        use_sa: bool = True,
    ):
        super().__init__()

        # Block 1: 1 → 16
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Block 2: 16 → 32
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # CBAM Attention on 32 channels
        self.cbam = CBAM(32, cbam_reduction, cbam_sa_kernel, use_ca=use_ca, use_sa=use_sa)

        # Block 3: 32 → 64
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.cbam(x)  # Apply attention
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
