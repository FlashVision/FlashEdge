"""Mobile-optimized model architectures for edge deployment."""

from flashedge.models.architectures.mobilenet import MobileNetV3Small, MobileNetV3Large
from flashedge.models.architectures.efficientnet import EfficientNetLite
from flashedge.models.architectures.tiny_vit import TinyViT

__all__ = [
    "MobileNetV3Small",
    "MobileNetV3Large",
    "EfficientNetLite",
    "TinyViT",
]
