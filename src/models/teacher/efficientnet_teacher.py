"""
EfficientNet teacher for knowledge distillation.
Lightweight and efficient vision model for spectrograms.
"""
import numpy as np
import torch
import torch.nn as nn
from transformers import EfficientNetModel, EfficientNetImageProcessor


class EfficientNetTeacher(nn.Module):
    """
    EfficientNet-based teacher model for audio classification via spectrograms.

    Uses a pre-trained EfficientNet encoder (frozen) with a trainable classification head.
    Input: log-mel spectrogram [batch, n_mels, time] -> converted to 3-channel image
    Output: logits [batch, n_classes]

    EfficientNet-b0 is very lightweight (~5M params) compared to ViT (~86M).
    """

    def __init__(self, n_classes: int, model_name: str = "google/efficientnet-b0"):
        super().__init__()

        # Load pre-trained EfficientNet encoder
        self.processor = EfficientNetImageProcessor.from_pretrained(model_name)
        self.encoder = EfficientNetModel.from_pretrained(model_name)

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Trainable classification head
        # EfficientNet-b0 hidden size is 1280
        hidden = self.encoder.config.hidden_dim
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def preprocess_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Convert log-mel spectrogram to RGB image format for EfficientNet.

        Args:
            spec: [batch, n_mels, time] log-mel spectrogram

        Returns:
            images: [batch, 3, 224, 224] RGB images
        """
        batch_size = spec.shape[0]
        device = spec.device

        # Convert to numpy for processor
        spec_np = spec.detach().cpu().numpy()

        # Normalize to [0, 255] range
        spec_min = spec_np.min(axis=(1, 2), keepdims=True)
        spec_max = spec_np.max(axis=(1, 2), keepdims=True)
        spec_norm = (spec_np - spec_min) / (spec_max - spec_min + 1e-8)
        spec_norm = (spec_norm * 255).astype('uint8')

        # Convert to 3-channel by replicating
        spec_rgb = np.stack([spec_norm] * 3, axis=1)

        # Process each image in batch
        images = []
        for i in range(batch_size):
            # Convert [3, n_mels, time] to PIL-compatible format
            img = spec_rgb[i].transpose(1, 2, 0)  # [n_mels, time, 3]

            # Use EfficientNet processor to resize to 224x224 and normalize
            processed = self.processor(images=img, return_tensors="pt")
            images.append(processed['pixel_values'][0])

        # Stack batch
        images = torch.stack(images).to(device)
        return images

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EfficientNet teacher.

        Args:
            spec: [batch, n_mels, time] log-mel spectrogram

        Returns:
            logits: [batch, n_classes]
        """
        # Preprocess spectrogram to RGB images
        images = self.preprocess_spectrogram(spec)

        # Forward through EfficientNet encoder
        outputs = self.encoder(pixel_values=images)

        # Use pooled output
        pooled = outputs.pooler_output  # [batch, hidden_dim]

        # Classification head
        logits = self.head(pooled)

        return logits

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str, device: str = "cpu"):
        """
        Load a trained teacher from checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: Device to load model on

        Returns:
            Loaded EfficientNetTeacher model
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(
            n_classes=checkpoint['n_classes'],
            model_name=checkpoint['model_name']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
