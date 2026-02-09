import torch.nn as nn

from src.models.attention.cbam_audio import CBAM


class ExtremeTinyCNN_CBAM(nn.Module):
    """ExtremeTinyCNN with CBAM attention (~8K params)

    Architecture: 1→8→16(+CBAM)→32 channels
    Adds CBAM attention after the second conv block.
    Uses smaller reduction ratio (4) for the smaller model size.
    """

    def __init__(
        self,
        n_classes: int,
        cbam_reduction: int = 4,  # Smaller reduction for smaller model
        cbam_sa_kernel: int = 7,
        *,
        use_ca: bool = True,
        use_sa: bool = True,
    ):
        super().__init__()

        # Block 1: 1 → 8
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Block 2: 8 → 16
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # CBAM Attention on 16 channels
        self.cbam = CBAM(16, cbam_reduction, cbam_sa_kernel, use_ca=use_ca, use_sa=use_sa)

        # Block 3: 16 → 32
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.cbam(x)  # Apply attention
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
