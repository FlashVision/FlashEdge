"""Model optimization techniques for edge deployment."""

from flashedge.optimization.quantization import INT8Quantizer, DynamicQuantizer, QATTrainer
from flashedge.optimization.pruning import StructuredPruner, UnstructuredPruner
from flashedge.optimization.distillation import KnowledgeDistiller
from flashedge.optimization.nas import MobileNAS

__all__ = [
    "INT8Quantizer",
    "DynamicQuantizer",
    "QATTrainer",
    "StructuredPruner",
    "UnstructuredPruner",
    "KnowledgeDistiller",
    "MobileNAS",
]
