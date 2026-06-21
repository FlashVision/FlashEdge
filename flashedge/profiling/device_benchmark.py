"""Standardized per-device benchmarking for edge deployment targets.

Estimates inference latency, power consumption, and deployment feasibility
across common edge devices by combining actual host benchmarks with hardware
spec extrapolation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


DEVICE_PROFILES: Dict[str, Dict[str, Any]] = {
    "raspberry_pi_4": {
        "name": "Raspberry Pi 4 (4GB)",
        "cpu": "ARM Cortex-A72",
        "cores": 4,
        "freq_ghz": 1.5,
        "ram_gb": 4,
        "npu": None,
        "gpu": "VideoCore VI",
        "tdp_watts": 6.0,
        "compute_gflops": 13.5,
        "memory_bandwidth_gbps": 4.0,
    },
    "raspberry_pi_5": {
        "name": "Raspberry Pi 5 (8GB)",
        "cpu": "ARM Cortex-A76",
        "cores": 4,
        "freq_ghz": 2.4,
        "ram_gb": 8,
        "npu": None,
        "gpu": "VideoCore VII",
        "tdp_watts": 12.0,
        "compute_gflops": 32.0,
        "memory_bandwidth_gbps": 8.5,
    },
    "jetson_nano": {
        "name": "NVIDIA Jetson Nano",
        "cpu": "ARM Cortex-A57",
        "cores": 4,
        "freq_ghz": 1.43,
        "ram_gb": 4,
        "npu": None,
        "gpu": "Maxwell 128-core",
        "gpu_gflops": 472,
        "tdp_watts": 10.0,
        "compute_gflops": 472,
        "memory_bandwidth_gbps": 25.6,
    },
    "jetson_orin_nano": {
        "name": "NVIDIA Jetson Orin Nano",
        "cpu": "ARM Cortex-A78AE",
        "cores": 6,
        "freq_ghz": 1.5,
        "ram_gb": 8,
        "npu": "Ampere GPU",
        "gpu": "Ampere 1024-core",
        "gpu_gflops": 40000,
        "tdp_watts": 15.0,
        "compute_gflops": 40000,
        "memory_bandwidth_gbps": 68,
    },
    "mobile_cpu_mid": {
        "name": "Mobile CPU (Mid-range, e.g. Snapdragon 7 Gen 1)",
        "cpu": "Kryo 4 Gold+Silver",
        "cores": 8,
        "freq_ghz": 2.4,
        "ram_gb": 6,
        "npu": "Hexagon DSP",
        "gpu": "Adreno 644",
        "tdp_watts": 5.0,
        "compute_gflops": 24,
        "memory_bandwidth_gbps": 17,
    },
    "mobile_cpu_high": {
        "name": "Mobile CPU (High-end, e.g. Snapdragon 8 Gen 3)",
        "cpu": "Kryo Prime+Gold+Silver",
        "cores": 8,
        "freq_ghz": 3.3,
        "ram_gb": 12,
        "npu": "Hexagon NPU",
        "gpu": "Adreno 750",
        "tdp_watts": 8.0,
        "compute_gflops": 73,
        "memory_bandwidth_gbps": 51.2,
    },
    "mobile_gpu_mid": {
        "name": "Mobile GPU (Mid-range, e.g. Mali-G710)",
        "cpu": "ARM Cortex",
        "cores": 8,
        "freq_ghz": 2.0,
        "ram_gb": 8,
        "npu": None,
        "gpu": "Mali-G710 MP10",
        "gpu_gflops": 1197,
        "tdp_watts": 6.0,
        "compute_gflops": 1197,
        "memory_bandwidth_gbps": 25.6,
    },
    "coral_edge_tpu": {
        "name": "Google Coral Edge TPU",
        "cpu": "ARM Cortex-A53",
        "cores": 4,
        "freq_ghz": 1.5,
        "ram_gb": 1,
        "npu": "Edge TPU",
        "gpu": None,
        "tdp_watts": 2.0,
        "compute_gflops": 4000,
        "memory_bandwidth_gbps": 4,
    },
}


class DeviceBenchmark:
    """Standardized benchmarking across common edge deployment targets.

    Combines actual host latency measurements with hardware spec extrapolation
    to estimate performance on target devices without requiring physical access.

    Args:
        device_profile: Key into DEVICE_PROFILES or a custom dict with hardware specs.
    """

    def __init__(self, device_profile: Union[str, Dict[str, Any]] = "raspberry_pi_4") -> None:
        if isinstance(device_profile, str):
            if device_profile not in DEVICE_PROFILES:
                raise ValueError(
                    f"Unknown device profile '{device_profile}'. Available: {list(DEVICE_PROFILES.keys())}"
                )
            self.profile = DEVICE_PROFILES[device_profile].copy()
            self.device_name = device_profile
        else:
            self.profile = device_profile.copy()
            self.device_name = device_profile.get("name", "custom_device")

    def benchmark(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        warmup: int = 10,
        runs: int = 100,
    ) -> Dict[str, Any]:
        """Run actual latency benchmark on host and extrapolate to target device.

        Args:
            model: PyTorch model to benchmark.
            input_shape: Shape of the input tensor.
            warmup: Number of warmup iterations.
            runs: Number of benchmark iterations.

        Returns:
            Dictionary with host latency stats and estimated device latency.
        """
        import numpy as np

        model = model.cpu().eval()
        dummy = torch.randn(*input_shape)

        print(f"  Benchmarking on host ({runs} runs, {warmup} warmup)...")
        with torch.no_grad():
            for _ in range(warmup):
                model(dummy)

            latencies: List[float] = []
            for i in range(runs):
                start = time.perf_counter()
                model(dummy)
                elapsed = (time.perf_counter() - start) * 1000.0
                latencies.append(elapsed)

        arr = np.array(latencies)
        host_stats = {
            "mean_ms": float(arr.mean()),
            "std_ms": float(arr.std()),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
        }

        host_gflops = self._estimate_host_gflops()
        target_gflops = float(self.profile["compute_gflops"])
        compute_ratio = host_gflops / target_gflops if target_gflops > 0 else 1.0

        estimated_device_ms = host_stats["mean_ms"] * compute_ratio
        print(f"  Host mean latency: {host_stats['mean_ms']:.2f} ms")
        print(f"  Estimated device latency: {estimated_device_ms:.2f} ms (ratio: {compute_ratio:.2f}x)")

        return {
            "host_stats": host_stats,
            "estimated_device_latency_ms": estimated_device_ms,
            "estimated_device_fps": 1000.0 / estimated_device_ms if estimated_device_ms > 0 else 0.0,
            "host_gflops": host_gflops,
            "target_gflops": target_gflops,
            "compute_ratio": compute_ratio,
            "device": self.device_name,
        }

    def estimate_latency(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Estimate latency on the target device without running the model on it.

        Uses FLOPs count and device compute/memory specs to estimate the latency
        bottleneck (compute-bound vs memory-bound).

        Args:
            model: PyTorch model to analyze.
            input_shape: Shape of the input tensor.

        Returns:
            Dictionary with estimated latency, bottleneck type, FPS, and utilization.
        """
        total_flops = self._count_flops(model, input_shape)
        model_size_bytes = self._get_model_size(model) * 1024 * 1024

        target_gflops = float(self.profile["compute_gflops"])
        memory_bw_gbps = float(self.profile["memory_bandwidth_gbps"])

        compute_time_ms = (total_flops / (target_gflops * 1e9)) * 1000.0 if target_gflops > 0 else float("inf")

        memory_time_ms = (model_size_bytes / (memory_bw_gbps * 1e9)) * 1000.0 if memory_bw_gbps > 0 else 0.0

        estimated_latency_ms = max(compute_time_ms, memory_time_ms)
        bottleneck = "compute" if compute_time_ms >= memory_time_ms else "memory"

        utilization = min(compute_time_ms / estimated_latency_ms, 1.0) if estimated_latency_ms > 0 else 0.0

        print(f"  Estimated latency on {self.profile.get('name', self.device_name)}: {estimated_latency_ms:.2f} ms")
        print(f"  Bottleneck: {bottleneck} (compute={compute_time_ms:.2f}ms, memory={memory_time_ms:.2f}ms)")

        return {
            "estimated_latency_ms": estimated_latency_ms,
            "compute_time_ms": compute_time_ms,
            "memory_time_ms": memory_time_ms,
            "bottleneck": bottleneck,
            "fps": 1000.0 / estimated_latency_ms if estimated_latency_ms > 0 else 0.0,
            "utilization": utilization,
            "total_flops": total_flops,
            "model_size_mb": model_size_bytes / (1024 * 1024),
            "device": self.device_name,
        }

    def estimate_power(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Estimate power consumption based on device TDP and model utilization.

        Args:
            model: PyTorch model to analyze.
            input_shape: Shape of the input tensor.

        Returns:
            Dictionary with power watts, energy per inference, and battery life estimate.
        """
        latency_info = self.estimate_latency(model, input_shape)
        tdp_watts = float(self.profile["tdp_watts"])
        utilization = latency_info["utilization"]

        idle_fraction = 0.3
        power_watts = tdp_watts * (idle_fraction + (1.0 - idle_fraction) * utilization)

        latency_s = latency_info["estimated_latency_ms"] / 1000.0
        energy_per_inference_mj = power_watts * latency_s * 1000.0

        battery_capacity_wh = 5.0 * 3.7  # 5000mAh at 3.7V nominal
        continuous_hours = battery_capacity_wh / power_watts if power_watts > 0 else float("inf")

        print(f"  Estimated power: {power_watts:.2f} W (TDP: {tdp_watts:.1f} W)")
        print(f"  Energy per inference: {energy_per_inference_mj:.3f} mJ")
        print(f"  Battery life (5000mAh): {continuous_hours:.1f} hours continuous")

        return {
            "power_watts": power_watts,
            "energy_per_inference_mj": energy_per_inference_mj,
            "battery_life_hours": continuous_hours,
            "tdp_watts": tdp_watts,
            "utilization": utilization,
            "device": self.device_name,
        }

    def check_deployment_feasibility(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Check whether the model can feasibly be deployed on the target device.

        Evaluates memory fit and real-time performance (30 FPS = 33ms threshold).

        Args:
            model: PyTorch model to evaluate.
            input_shape: Shape of the input tensor.

        Returns:
            Dictionary with feasibility flags, sizing info, and recommendations.
        """
        model_size_mb = self._get_model_size(model)
        available_ram_mb = float(self.profile["ram_gb"]) * 1024

        runtime_overhead_factor = 2.5
        required_ram_mb = model_size_mb * runtime_overhead_factor
        fits_in_memory = required_ram_mb < (available_ram_mb * 0.7)

        latency_info = self.estimate_latency(model, input_shape)
        realtime_threshold_ms = 33.33
        meets_realtime = latency_info["estimated_latency_ms"] <= realtime_threshold_ms

        recommendations: List[str] = []
        if not fits_in_memory:
            recommendations.append(
                f"Model requires ~{required_ram_mb:.0f} MB but device has {available_ram_mb:.0f} MB. "
                "Consider quantization (INT8) or pruning to reduce size."
            )
        if not meets_realtime:
            fps = latency_info["fps"]
            recommendations.append(
                f"Estimated {fps:.1f} FPS (need 30 FPS). "
                "Consider model distillation, channel pruning, or a more powerful device."
            )
        if latency_info["bottleneck"] == "memory":
            recommendations.append(
                "Model is memory-bandwidth limited. Quantization (INT8/INT4) will help more than FLOPs reduction."
            )
        if model_size_mb > 50:
            recommendations.append(f"Model is {model_size_mb:.1f} MB. For edge, target <20 MB with quantization.")
        if not recommendations:
            recommendations.append("Model appears suitable for deployment on this device.")

        print(f"  Deployment feasibility on {self.profile.get('name', self.device_name)}:")
        print(
            f"  Memory: {'OK' if fits_in_memory else 'FAIL'} ({model_size_mb:.1f} MB model, {available_ram_mb:.0f} MB device)"
        )
        print(f"  Realtime: {'OK' if meets_realtime else 'FAIL'} ({latency_info['fps']:.1f} FPS estimated)")

        return {
            "fits_in_memory": fits_in_memory,
            "meets_realtime": meets_realtime,
            "model_size_mb": model_size_mb,
            "required_ram_mb": required_ram_mb,
            "available_ram_mb": available_ram_mb,
            "estimated_fps": latency_info["fps"],
            "estimated_latency_ms": latency_info["estimated_latency_ms"],
            "bottleneck": latency_info["bottleneck"],
            "recommendations": recommendations,
            "device": self.device_name,
        }

    def compare_devices(
        self,
        model: nn.Module,
        device_names: Optional[List[str]] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> List[Dict[str, Any]]:
        """Compare estimated performance across multiple target devices.

        Args:
            model: PyTorch model to evaluate.
            device_names: List of device profile keys to compare. If None, uses all.
            input_shape: Shape of the input tensor.

        Returns:
            List of per-device result dicts sorted by estimated latency (fastest first).
        """
        if device_names is None:
            device_names = list(DEVICE_PROFILES.keys())

        total_flops = self._count_flops(model, input_shape)
        model_size_mb = self._get_model_size(model)
        model_size_bytes = model_size_mb * 1024 * 1024

        print(
            f"  Comparing {len(device_names)} devices (model: {total_flops / 1e6:.1f} MFLOPs, {model_size_mb:.1f} MB)"
        )

        results: List[Dict[str, Any]] = []
        for name in device_names:
            if name not in DEVICE_PROFILES:
                continue
            profile = DEVICE_PROFILES[name]
            target_gflops = float(profile["compute_gflops"])
            memory_bw_gbps = float(profile["memory_bandwidth_gbps"])

            compute_ms = (total_flops / (target_gflops * 1e9)) * 1000.0 if target_gflops > 0 else float("inf")
            memory_ms = (model_size_bytes / (memory_bw_gbps * 1e9)) * 1000.0 if memory_bw_gbps > 0 else 0.0
            latency_ms = max(compute_ms, memory_ms)

            tdp = float(profile["tdp_watts"])
            utilization = min(compute_ms / latency_ms, 1.0) if latency_ms > 0 else 0.0
            power_watts = tdp * (0.3 + 0.7 * utilization)

            available_ram_mb = float(profile["ram_gb"]) * 1024
            feasible = (model_size_mb * 2.5 < available_ram_mb * 0.7) and (latency_ms <= 33.33)

            results.append(
                {
                    "device_name": name,
                    "device_label": profile["name"],
                    "latency_ms": latency_ms,
                    "fps": 1000.0 / latency_ms if latency_ms > 0 else 0.0,
                    "power_watts": power_watts,
                    "feasible": feasible,
                    "bottleneck": "compute" if compute_ms >= memory_ms else "memory",
                }
            )

        results.sort(key=lambda x: x["latency_ms"])

        for r in results:
            status = "OK" if r["feasible"] else "--"
            print(
                f"  [{status}] {r['device_label']:<45} {r['latency_ms']:>8.2f} ms  {r['fps']:>7.1f} FPS  {r['power_watts']:.1f}W"
            )

        return results

    def full_report(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> Dict[str, Any]:
        """Generate a comprehensive benchmark report for the target device.

        Args:
            model: PyTorch model to analyze.
            input_shape: Shape of the input tensor.

        Returns:
            Dictionary with latency, power, feasibility, and formatted summary.
        """
        print(f"  Generating full report for {self.profile.get('name', self.device_name)}...")

        latency = self.estimate_latency(model, input_shape)
        power = self.estimate_power(model, input_shape)
        feasibility = self.check_deployment_feasibility(model, input_shape)

        summary_lines = [
            f"Device Benchmark Report: {self.profile.get('name', self.device_name)}",
            "=" * 60,
            f"CPU: {self.profile.get('cpu', 'N/A')} @ {self.profile.get('freq_ghz', 0)} GHz ({self.profile.get('cores', 0)} cores)",
            f"GPU: {self.profile.get('gpu', 'N/A')}",
            f"NPU: {self.profile.get('npu', 'None')}",
            f"RAM: {self.profile.get('ram_gb', 0)} GB | TDP: {self.profile.get('tdp_watts', 0)} W",
            "-" * 60,
            f"Model Size: {feasibility['model_size_mb']:.1f} MB",
            f"Total FLOPs: {latency['total_flops'] / 1e6:.1f} MFLOPs",
            f"Estimated Latency: {latency['estimated_latency_ms']:.2f} ms",
            f"Estimated FPS: {latency['fps']:.1f}",
            f"Bottleneck: {latency['bottleneck']}",
            f"Power Draw: {power['power_watts']:.2f} W",
            f"Energy/Inference: {power['energy_per_inference_mj']:.3f} mJ",
            f"Battery Life (5000mAh): {power['battery_life_hours']:.1f} h",
            "-" * 60,
            f"Fits in Memory: {'Yes' if feasibility['fits_in_memory'] else 'No'}",
            f"Meets 30 FPS: {'Yes' if feasibility['meets_realtime'] else 'No'}",
            "-" * 60,
            "Recommendations:",
        ]
        for rec in feasibility["recommendations"]:
            summary_lines.append(f"  - {rec}")

        summary = "\n".join(summary_lines)

        return {
            "device": self.device_name,
            "device_profile": self.profile,
            "latency": latency,
            "power": power,
            "feasibility": feasibility,
            "summary": summary,
        }

    def _count_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Count total FLOPs using forward hooks on compute-heavy layers.

        Args:
            model: PyTorch model.
            input_shape: Input tensor shape.

        Returns:
            Total FLOPs as integer.
        """
        model = model.cpu().eval()
        total_flops = [0]
        hooks: List[Any] = []

        def _conv2d_hook(module: nn.Conv2d, inp: Any, out: Any) -> None:
            output = out if isinstance(out, torch.Tensor) else out[0]
            batch, cout, hout, wout = output.shape
            cin = module.in_channels
            kh, kw = module.kernel_size
            groups = module.groups
            macs = (cin // groups) * kh * kw * cout * hout * wout
            flops = 2 * macs
            if module.bias is not None:
                flops += cout * hout * wout
            total_flops[0] += flops

        def _linear_hook(module: nn.Linear, inp: Any, out: Any) -> None:
            output = out if isinstance(out, torch.Tensor) else out[0]
            batch_size = output.shape[0]
            macs = module.in_features * module.out_features * batch_size
            flops = 2 * macs
            if module.bias is not None:
                flops += module.out_features * batch_size
            total_flops[0] += flops

        def _bn_hook(module: nn.Module, inp: Any, out: Any) -> None:
            output = out if isinstance(out, torch.Tensor) else out[0]
            total_flops[0] += 4 * output.numel()

        for _, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(_conv2d_hook))
            elif isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(_linear_hook))
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                hooks.append(module.register_forward_hook(_bn_hook))

        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            model(dummy)

        for h in hooks:
            h.remove()

        return total_flops[0]

    def _get_model_size(self, model: nn.Module) -> float:
        """Calculate model size in megabytes.

        Args:
            model: PyTorch model.

        Returns:
            Model size in MB.
        """
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_bytes + buffer_bytes) / (1024 * 1024)

    def _estimate_host_gflops(self) -> float:
        """Estimate host CPU compute capability via a timed matrix multiplication.

        Returns:
            Estimated GFLOPS of the host machine.
        """
        n = 512
        a = torch.randn(n, n)
        b = torch.randn(n, n)

        torch.mm(a, b)

        iterations = 5
        start = time.perf_counter()
        for _ in range(iterations):
            torch.mm(a, b)
        elapsed = time.perf_counter() - start

        flops_per_matmul = 2 * n * n * n
        total_flops = flops_per_matmul * iterations
        gflops = (total_flops / elapsed) / 1e9

        return gflops
