import torch.nn as nn


class TinyCNN(nn.Module):
    """
    Tiny CNN for log-mel inputs [B, 1, n_mels, time]
    Simple CNN-only architecture without RNN.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
        )
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        return self.classifier(x)
