"""Power consumption estimation for edge and mobile devices.

Estimates power draw based on model FLOPs, memory access patterns, and
device-specific TDP (Thermal Design Power) characteristics.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from flashedge.profiling.flops import FLOPsCounter
from flashedge.profiling.latency import LatencyProfiler


DEVICE_PROFILES: Dict[str, Dict[str, float]] = {
    "mobile": {
        "tdp_watts": 5.0,
        "flops_per_watt": 2e9,
        "memory_watts_per_gb": 0.3,
        "idle_watts": 0.5,
    },
    "raspberry_pi": {
        "tdp_watts": 7.5,
        "flops_per_watt": 1e9,
        "memory_watts_per_gb": 0.2,
        "idle_watts": 2.0,
    },
    "jetson_nano": {
        "tdp_watts": 10.0,
        "flops_per_watt": 5e9,
        "memory_watts_per_gb": 0.5,
        "idle_watts": 1.5,
    },
    "intel_nuc": {
        "tdp_watts": 28.0,
        "flops_per_watt": 10e9,
        "memory_watts_per_gb": 1.0,
        "idle_watts": 5.0,
    },
    "desktop_cpu": {
        "tdp_watts": 65.0,
        "flops_per_watt": 20e9,
        "memory_watts_per_gb": 2.0,
        "idle_watts": 15.0,
    },
}


class PowerEstimator:
    """Estimate power consumption for inference on edge devices.

    Uses a model-based estimation combining compute FLOPs, memory bandwidth,
    and device-specific power characteristics.

    Args:
        device_type: Target device type (mobile, raspberry_pi, jetson_nano, intel_nuc, desktop_cpu).
        tdp_watts: Override TDP in watts (uses device profile if None).
        custom_profile: Custom device power profile.
    """

    def __init__(
        self,
        device_type: str = "mobile",
        tdp_watts: Optional[float] = None,
        custom_profile: Optional[Dict[str, float]] = None,
    ) -> None:
        if custom_profile is not None:
            self.profile = custom_profile
        elif device_type in DEVICE_PROFILES:
            self.profile = DEVICE_PROFILES[device_type].copy()
        else:
            raise ValueError(
                f"Unknown device_type '{device_type}'. "
                f"Available: {list(DEVICE_PROFILES.keys())}"
            )

        if tdp_watts is not None:
            self.profile["tdp_watts"] = tdp_watts

        self.device_type = device_type

    def estimate(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        latency_ms: Optional[float] = None,
    ) -> Dict[str, float]:
        """Estimate power consumption for a model.

        Args:
            model: PyTorch model to analyze.
            input_shape: Shape of the input tensor.
            latency_ms: Measured latency in ms. If None, profiles automatically.

        Returns:
            Dictionary with power estimation results.
        """
        flops_counter = FLOPsCounter()
        flops_result = flops_counter.count(model, input_shape)
        total_flops = flops_result["flops"]

        if latency_ms is None:
            profiler = LatencyProfiler(device="cpu")
            latency_result = profiler.profile(model, input_shape, warmup=5, runs=20)
            latency_ms = latency_result["mean_ms"]

        latency_s = latency_ms / 1000.0

        compute_watts = min(
            total_flops / max(latency_s, 1e-6) / self.profile["flops_per_watt"],
            self.profile["tdp_watts"],
        )

        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        memory_gb = param_bytes / (1024**3)
        memory_watts = memory_gb * self.profile["memory_watts_per_gb"]

        total_watts = self.profile["idle_watts"] + compute_watts + memory_watts
        total_watts = min(total_watts, self.profile["tdp_watts"])

        energy_per_inference_mj = total_watts * latency_s * 1000

        inferences_per_kwh = (1000 * 3600) / max(energy_per_inference_mj / 1000, 1e-10)

        return {
            "estimated_watts": total_watts,
            "compute_watts": compute_watts,
            "memory_watts": memory_watts,
            "idle_watts": self.profile["idle_watts"],
            "energy_per_inference_mj": energy_per_inference_mj,
            "inferences_per_kwh": inferences_per_kwh,
            "latency_ms": latency_ms,
            "total_flops": total_flops,
            "tdp_watts": self.profile["tdp_watts"],
            "device_type": self.device_type,
        }

    @staticmethod
    def list_device_profiles() -> Dict[str, Dict[str, float]]:
        """List all available device power profiles."""
        return DEVICE_PROFILES.copy()
