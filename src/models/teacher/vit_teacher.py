"""
ViT (Vision Transformer) teacher for knowledge distillation.
Treats log-mel spectrograms as images.
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel


class ViTTeacher(nn.Module):
    """
    ViT-based teacher model for audio classification via spectrograms.

    Uses a pre-trained ViT encoder (frozen) with a trainable classification head.
    Input: log-mel spectrogram [batch, n_mels, time] -> converted to 3-channel image
    Output: logits [batch, n_classes]
    """

    def __init__(self, n_classes: int, model_name: str = "google/vit-base-patch16-224"):
        super().__init__()

        # Load pre-trained ViT encoder
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.encoder = ViTModel.from_pretrained(model_name)

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Trainable classification head
        hidden = self.encoder.config.hidden_size  # 768 for ViT-base
        self.head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        )

    def preprocess_spectrogram(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Convert log-mel spectrogram to RGB image format for ViT.

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
        spec_norm = (spec_norm * 255).astype("uint8")

        # Convert to 3-channel by replicating
        # [batch, n_mels, time] -> [batch, 3, n_mels, time]
        spec_rgb = np.stack([spec_norm] * 3, axis=1)

        # Transpose to [batch, 3, height, width] if needed
        # Our spec is already [batch, n_mels, time], replicate to [batch, 3, n_mels, time]
        # ViT expects [batch, 3, H, W]

        # Process each image in batch
        images = []
        for i in range(batch_size):
            # Convert [3, n_mels, time] to PIL-compatible format
            img = spec_rgb[i].transpose(1, 2, 0)  # [n_mels, time, 3]

            # Use ViT processor to resize to 224x224 and normalize
            processed = self.processor(images=img, return_tensors="pt")
            images.append(processed["pixel_values"][0])

        # Stack batch
        images = torch.stack(images).to(device)
        return images

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT teacher.

        Args:
            spec: [batch, n_mels, time] log-mel spectrogram

        Returns:
            logits: [batch, n_classes]
        """
        # Preprocess spectrogram to RGB images
        images = self.preprocess_spectrogram(spec)

        # Forward through ViT encoder
        outputs = self.encoder(pixel_values=images)

        # Use CLS token representation
        pooled = outputs.last_hidden_state[:, 0]  # [batch, hidden_size]

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
            Loaded ViTTeacher model
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = cls(n_classes=checkpoint["n_classes"], model_name=checkpoint["model_name"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model
