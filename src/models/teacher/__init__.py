"""Teacher models (training only)."""

from .efficientnet_teacher import EfficientNetTeacher
from .vit_teacher import ViTTeacher

__all__ = ["ViTTeacher", "EfficientNetTeacher"]
