"""Teacher models (training only)."""

from .vit_teacher import ViTTeacher
from .efficientnet_teacher import EfficientNetTeacher

__all__ = ["ViTTeacher", "EfficientNetTeacher"]
