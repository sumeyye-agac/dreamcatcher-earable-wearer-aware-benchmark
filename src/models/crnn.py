import torch.nn as nn


class CRNN(nn.Module):
    """
    CRNN for log-mel inputs [B, 1, n_mels, time]
    """

    def __init__(self, n_classes: int, rnn_hidden: int = 64, rnn_layers: int = 1):
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
        )

        self.rnn = nn.GRU(
            input_size=64,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.classifier = nn.Linear(rnn_hidden, n_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = x.mean(dim=2)
        x = x.transpose(1, 2)
        out, _ = self.rnn(x)
        return self.classifier(out[:, -1])
