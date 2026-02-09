import torch.nn as nn


class ExtremeTinyCNN(nn.Module):
    """
    Extremely compressed CNN for knowledge distillation.
    ~6K parameters (4x smaller than TinyCNN).

    Architecture: 8→16→32 channels
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)
