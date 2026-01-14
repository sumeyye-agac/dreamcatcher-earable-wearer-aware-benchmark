import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class Wav2Vec2Teacher(nn.Module):
    def __init__(self, n_classes: int, model_name: str = "facebook/wav2vec2-base"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        for p in self.encoder.parameters():
            p.requires_grad = False

        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes),
        )

    @torch.no_grad()
    def forward(self, audio_16k: torch.Tensor) -> torch.Tensor:
        device = audio_16k.device
        audio_np = audio_16k.detach().cpu().numpy()
        inputs = self.processor(audio_np, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        outputs = self.encoder(input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.head(pooled)
