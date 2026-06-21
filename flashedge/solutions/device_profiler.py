"""Device profiler — comprehensive model analysis for a target device."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch.nn as nn


class DeviceProfiler:
    """Profile a model's performance characteristics for a target device.

    Combines latency, memory, FLOPs, and power profiling into a single report.

    Args:
        device: Target device for profiling (cpu, cuda).
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def profile(
        self,
        model_or_path: Union[str, nn.Module],
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        warmup: int = 10,
        runs: int = 100,
    ) -> Dict[str, Any]:
        """Run comprehensive profiling on a model.

        Args:
            model_or_path: PyTorch model or path to ONNX model.
            input_shape: Shape of the input tensor.
            warmup: Warmup iterations for latency profiling.
            runs: Benchmark iterations.

        Returns:
            Dictionary with comprehensive profiling results.
        """
        from flashedge.profiling.latency import LatencyProfiler

        latency_profiler = LatencyProfiler(device=self.device)
        latency = latency_profiler.profile(model_or_path, input_shape, warmup, runs)

        result: Dict[str, Any] = {"latency": latency}

        if isinstance(model_or_path, nn.Module):
            from flashedge.profiling.memory import MemoryProfiler
            from flashedge.profiling.flops import FLOPsCounter

            mem_profiler = MemoryProfiler(device=self.device)
            memory = mem_profiler.profile(model_or_path, input_shape)
            result["memory"] = memory

            flops_counter = FLOPsCounter()
            flops = flops_counter.count(model_or_path, input_shape)
            result["flops"] = {
                "flops": flops["flops"],
                "macs": flops["macs"],
                "flops_str": flops["flops_str"],
                "macs_str": flops["macs_str"],
            }

        elif isinstance(model_or_path, str):
            path = Path(model_or_path)
            if path.exists():
                from flashedge.profiling.memory import MemoryProfiler

                mem_profiler = MemoryProfiler()
                file_info = mem_profiler.profile_file(model_or_path)
                result["file_info"] = file_info

        return result

    def full_report(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        device_type: str = "mobile",
        runs: int = 100,
    ) -> Dict[str, Any]:
        """Generate a full device profiling report including power estimation.

        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.
            device_type: Target device type for power estimation.
            runs: Benchmark iterations.

        Returns:
            Comprehensive profiling report.
        """
        result = self.profile(model, input_shape, runs=runs)

        from flashedge.profiling.power import PowerEstimator

        power_estimator = PowerEstimator(device_type=device_type)
        power = power_estimator.estimate(model, input_shape, latency_ms=result["latency"]["mean_ms"])
        result["power"] = power

        return result

    def print_report(self, report: Dict[str, Any]) -> None:
        """Print a formatted profiling report."""
        print("\n" + "=" * 60)
        print("  FlashEdge Device Profiling Report")
        print("=" * 60)

        if "latency" in report:
            lat = report["latency"]
            print(f"\n  Latency:")
            print(f"    Mean:       {lat['mean_ms']:.3f} ms")
            print(f"    P50:        {lat.get('p50_ms', 0):.3f} ms")
            print(f"    P95:        {lat.get('p95_ms', 0):.3f} ms")
            print(f"    P99:        {lat.get('p99_ms', 0):.3f} ms")
            print(f"    Throughput: {lat['fps']:.1f} FPS")

        if "memory" in report:
            mem = report["memory"]
            print(f"\n  Memory:")
            print(f"    Parameters: {mem['num_parameters']:,}")
            print(f"    Model size: {mem['model_size_mb']:.2f} MB")
            print(f"    Peak memory:{mem['peak_memory_mb']:.2f} MB")

        if "flops" in report:
            fl = report["flops"]
            print(f"\n  Compute:")
            print(f"    FLOPs: {fl['flops_str']}")
            print(f"    MACs:  {fl['macs_str']}")

        if "power" in report:
            pw = report["power"]
            print(f"\n  Power ({pw.get('device_type', 'unknown')}):")
            print(f"    Estimated:  {pw['estimated_watts']:.3f} W")
            print(f"    Per inference: {pw['energy_per_inference_mj']:.3f} mJ")
            print(f"    Inferences/kWh: {pw['inferences_per_kwh']:.0f}")

        if "file_info" in report:
            fi = report["file_info"]
            print(f"\n  File:")
            print(f"    Size: {fi['file_size_mb']:.2f} MB")

        print("\n" + "=" * 60)
