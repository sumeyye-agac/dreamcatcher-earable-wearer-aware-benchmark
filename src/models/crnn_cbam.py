import torch
import torch.nn as nn
from src.models.attention.cbam_audio import CBAM


class CRNN_CBAM(nn.Module):
    def __init__(
        self,
        n_classes: int,
        rnn_hidden: int = 64,
        rnn_layers: int = 1,
        cbam_reduction: int = 8,
        cbam_sa_kernel: int = 7,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.cbam = CBAM(32, cbam_reduction, cbam_sa_kernel)

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(
            input_size=64,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(rnn_hidden, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.cbam(x)
        x = self.block3(x)

        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        out, _ = self.rnn(x)
        return self.classifier(out[:, -1])
