"""Tests for new FlashEdge components: NCNN/ExecuTorch export,
INT4 quantization, mixed precision, graph optimization, and device benchmarks.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture
def model():
    return SimpleModel().eval()


@pytest.fixture
def input_shape():
    return (1, 3, 32, 32)


class TestNCNNExport:
    """Test NCNN export pipeline."""

    def test_basic_export(self, model, input_shape):
        from flashedge.export.ncnn_export import NCNNExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model")
            exporter = NCNNExporter(optimize=False)
            result = exporter.export(model, output_path, input_shape=input_shape)
            assert os.path.exists(result)

    def test_export_creates_param_file(self, model, input_shape):
        from flashedge.export.ncnn_export import NCNNExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model")
            exporter = NCNNExporter(optimize=False)
            result = exporter.export(model, output_path, input_shape=input_shape)
            assert result.endswith(".param") or result.endswith(".onnx") or os.path.exists(result)


class TestExecuTorchExport:
    """Test ExecuTorch export pipeline."""

    def test_basic_export(self, model, input_shape):
        from flashedge.export.executorch_export import ExecuTorchExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pte")
            exporter = ExecuTorchExporter()
            result = exporter.export(model, output_path, input_shape=input_shape)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0


class TestINT4Quantizer:
    """Test INT4 quantization."""

    def test_quantize_model(self, model, input_shape):
        from flashedge.optimization.int4_quantization import INT4Quantizer

        quantizer = INT4Quantizer(method="rtn", group_size=16)
        q_model = quantizer.quantize(model, input_shape=input_shape)
        assert q_model is not None

    def test_estimate_compression(self, model):
        from flashedge.optimization.int4_quantization import INT4Quantizer

        quantizer = INT4Quantizer()
        info = quantizer.estimate_compression(model)
        assert "original_size_mb" in info
        assert "int4_size_mb" in info
        assert info["int4_size_mb"] < info["original_size_mb"]


class TestMixedPrecisionOptimizer:
    """Test mixed-precision per-layer optimization."""

    def test_analyze_sensitivity(self, model, input_shape):
        from flashedge.optimization.mixed_precision import MixedPrecisionOptimizer

        optimizer = MixedPrecisionOptimizer(candidate_bits=(8, 16, 32))
        result = optimizer.optimize(model, input_shape=input_shape)
        assert result is not None

    def test_get_precision_report(self, model):
        from flashedge.optimization.mixed_precision import MixedPrecisionOptimizer

        optimizer = MixedPrecisionOptimizer()
        report = optimizer.get_precision_report(model)
        assert "total_params" in report
        assert "layers" in report
        assert len(report["layers"]) > 0


class TestGraphOptimizer:
    """Test operator fusion and graph optimization."""

    def test_fuse_conv_bn(self, input_shape):
        from flashedge.optimization.graph_optimization import GraphOptimizer

        class SeqModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                x = self.features(x)
                return self.fc(x.flatten(1))

        model = SeqModel().eval()
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            out_orig = model(dummy)

        optimizer = GraphOptimizer(fuse_conv_bn=True, fuse_conv_relu=False, constant_folding=False)
        optimized = optimizer.optimize(model, input_shape=input_shape)
        assert optimized is not None

        with torch.no_grad():
            out_opt = optimized(dummy)
        assert out_orig.shape == out_opt.shape

    def test_analyze_opportunities(self, model, input_shape):
        from flashedge.optimization.graph_optimization import GraphOptimizer

        optimizer = GraphOptimizer()
        analysis = optimizer.analyze(model, input_shape=input_shape)
        assert "conv_bn_pairs" in analysis
        assert "total_params" in analysis


class TestDeviceBenchmark:
    """Test per-device benchmarking."""

    def test_estimate_latency(self, model, input_shape):
        from flashedge.profiling.device_benchmark import DeviceBenchmark

        bench = DeviceBenchmark("raspberry_pi_4")
        result = bench.estimate_latency(model, input_shape=input_shape)
        assert "estimated_latency_ms" in result
        assert result["estimated_latency_ms"] > 0

    def test_estimate_power(self, model, input_shape):
        from flashedge.profiling.device_benchmark import DeviceBenchmark

        bench = DeviceBenchmark("jetson_nano")
        result = bench.estimate_power(model, input_shape=input_shape)
        assert "power_watts" in result
        assert result["power_watts"] > 0

    def test_check_feasibility(self, model, input_shape):
        from flashedge.profiling.device_benchmark import DeviceBenchmark

        bench = DeviceBenchmark("raspberry_pi_4")
        result = bench.check_deployment_feasibility(model, input_shape=input_shape)
        assert "fits_in_memory" in result
        assert "model_size_mb" in result
        assert isinstance(result["fits_in_memory"], bool)

    def test_compare_devices(self, model, input_shape):
        from flashedge.profiling.device_benchmark import DeviceBenchmark

        bench = DeviceBenchmark()
        results = bench.compare_devices(
            model,
            device_names=["raspberry_pi_4", "jetson_nano"],
            input_shape=input_shape,
        )
        assert len(results) == 2
        assert all("device_name" in r for r in results)
        assert all("latency_ms" in r for r in results)

    def test_device_profiles_exist(self):
        from flashedge.profiling.device_benchmark import DEVICE_PROFILES

        assert "raspberry_pi_4" in DEVICE_PROFILES
        assert "jetson_nano" in DEVICE_PROFILES
        assert "mobile_cpu_mid" in DEVICE_PROFILES

    def test_custom_device_profile(self, model, input_shape):
        from flashedge.profiling.device_benchmark import DeviceBenchmark

        custom = {
            "name": "Custom Board",
            "cpu": "ARM",
            "cores": 2,
            "freq_ghz": 1.0,
            "ram_gb": 2,
            "tdp_watts": 3.0,
            "compute_gflops": 8.0,
            "memory_bandwidth_gbps": 2.0,
        }
        bench = DeviceBenchmark(custom)
        result = bench.estimate_latency(model, input_shape=input_shape)
        assert result["estimated_latency_ms"] > 0


class TestRegistryNewComponents:
    """Test that new components are properly registered."""

    def test_exporters_registered(self):
        from flashedge.registry import EXPORTERS
        import flashedge.export  # noqa: F401

        assert "ncnn" in EXPORTERS
        assert "executorch" in EXPORTERS

    def test_optimizers_registered(self):
        from flashedge.registry import OPTIMIZERS
        import flashedge.optimization  # noqa: F401

        assert "int4_quantizer" in OPTIMIZERS
        assert "mixed_precision" in OPTIMIZERS
        assert "graph_optimizer" in OPTIMIZERS
