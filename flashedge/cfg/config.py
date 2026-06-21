"""Dataclass-based configuration for FlashEdge pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExportConfig:
    """Configuration for model export."""

    format: str = "onnx"
    output: str = "exported/model.onnx"
    input_shape: List[int] = field(default_factory=lambda: [1, 3, 224, 224])
    opset_version: int = 17
    simplify: bool = True
    optimize: bool = True
    fp16: bool = False
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    method: str = "ptq"
    dtype: str = "int8"
    calibration_samples: int = 500
    per_channel: bool = True
    symmetric: bool = True
    calibration_data: str = ""


@dataclass
class PruningConfig:
    """Configuration for pruning."""

    method: str = "structured"
    sparsity: float = 0.3
    criterion: str = "l1_norm"
    iterative: bool = True
    iterations: int = 3
    finetune_epochs: int = 5


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    temperature: float = 4.0
    alpha: float = 0.7
    teacher_path: str = ""
    student_path: str = ""
    loss_type: str = "kl_div"
    epochs: int = 50


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""

    strategy: str = "evolutionary"
    population_size: int = 50
    generations: int = 30
    mutation_prob: float = 0.1
    target_latency_ms: float = 10.0
    max_flops: float = 300e6
    max_params: float = 3e6


@dataclass
class ProfilingConfig:
    """Configuration for profiling."""

    enabled: bool = True
    device: str = "cpu"
    warmup_runs: int = 10
    benchmark_runs: int = 100


@dataclass
class ModelConfig:
    """Configuration for the model."""

    path: str = ""
    architecture: str = "mobilenetv3_small"
    num_classes: int = 1000


@dataclass
class Config:
    """Main FlashEdge configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            raw: Dict[str, Any] = yaml.safe_load(f) or {}

        config = cls()

        section_map = {
            "model": (ModelConfig, "model"),
            "export": (ExportConfig, "export"),
            "quantization": (QuantizationConfig, "quantization"),
            "pruning": (PruningConfig, "pruning"),
            "distillation": (DistillationConfig, "distillation"),
            "nas": (NASConfig, "nas"),
            "profiling": (ProfilingConfig, "profiling"),
        }

        for key, (dc_cls, attr) in section_map.items():
            if key in raw:
                setattr(config, attr, dc_cls(**{k: v for k, v in raw[key].items() if k in dc_cls.__dataclass_fields__}))

        return config

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def get_config(path: Optional[str | Path] = None) -> Config:
    if path is None:
        return Config()
    return Config.from_yaml(path)
