"""FlashEdge — Edge AI & TinyML toolkit for on-device inference, model compression, and mobile deployment."""

__version__ = "1.0.0"

from flashedge.models.flashedge_model import FlashEdge
from flashedge.engine.exporter import Exporter
from flashedge.solutions.edge_deployer import EdgeDeployer
from flashedge.solutions.model_optimizer import ModelOptimizer
from flashedge.solutions.device_profiler import DeviceProfiler
from flashedge.analytics.benchmark import Benchmark

__all__ = [
    "FlashEdge",
    "Exporter",
    "EdgeDeployer",
    "ModelOptimizer",
    "DeviceProfiler",
    "Benchmark",
    "__version__",
]
