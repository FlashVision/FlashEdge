"""Comprehensive tests for FlashEdge: exporters, optimization, profiling, CLI."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn


class SimpleConvModel(nn.Module):
    """Minimal model for testing all FlashEdge functionality."""

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
    return SimpleConvModel().eval()


@pytest.fixture
def input_shape():
    return (1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Export: ONNX
# ---------------------------------------------------------------------------

onnxscript = pytest.importorskip("onnxscript")


class TestONNXExport:
    def test_export_basic(self, model, input_shape):
        from flashedge.export.onnx_export import OnnxExporter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.onnx")
            exporter = OnnxExporter(opset_version=13, simplify=False, optimize=False)
            result = exporter.export(model, path, input_shape=input_shape)
            assert os.path.exists(result)
            assert os.path.getsize(result) > 0

    def test_export_simplify(self, model, input_shape):
        from flashedge.export.onnx_export import OnnxExporter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.onnx")
            exporter = OnnxExporter(opset_version=13, simplify=True, optimize=True)
            result = exporter.export(model, path, input_shape=input_shape)
            assert os.path.exists(result)

    def test_validate(self, model, input_shape):
        from flashedge.export.onnx_export import OnnxExporter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.onnx")
            exporter = OnnxExporter(opset_version=13, simplify=False, optimize=False)
            exporter.export(model, path, input_shape=input_shape)
            result = exporter.validate(path, model=model, input_shape=input_shape, atol=1e-4)
            assert result["valid_schema"]
            assert result["inference_ok"]

    def test_model_info(self, model, input_shape):
        from flashedge.export.onnx_export import OnnxExporter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.onnx")
            exporter = OnnxExporter(opset_version=13, simplify=False, optimize=False)
            exporter.export(model, path, input_shape=input_shape)
            info = exporter.get_model_info(path)
            assert info["file_size_bytes"] > 0
            assert info["num_nodes"] > 0


# ---------------------------------------------------------------------------
# Export: TorchScript
# ---------------------------------------------------------------------------


class TestTorchScriptExport:
    def test_trace(self, model, input_shape):
        from flashedge.export.torchscript import TorchScriptExporter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pt")
            exporter = TorchScriptExporter(method="trace")
            result = exporter.export(model, path, input_shape=input_shape)
            assert os.path.exists(result)
            loaded = torch.jit.load(result)
            out = loaded(torch.randn(*input_shape))
            assert out.shape == (1, 10)

    def test_validate(self, model, input_shape):
        from flashedge.export.torchscript import TorchScriptExporter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pt")
            exporter = TorchScriptExporter(method="trace")
            exporter.export(model, path, input_shape=input_shape)
            result = exporter.validate(path, model=model, input_shape=input_shape)
            assert result["load_ok"]
            assert result["inference_ok"]


# ---------------------------------------------------------------------------
# Export: Unified Exporter
# ---------------------------------------------------------------------------


class TestUnifiedExporter:
    def test_export_onnx(self, model, input_shape):
        from flashedge.engine.exporter import Exporter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.onnx")
            exporter = Exporter(model)
            result = exporter.export(path, format="onnx", input_shape=input_shape)
            assert os.path.exists(result)

    def test_export_torchscript(self, model, input_shape):
        from flashedge.engine.exporter import Exporter

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pt")
            exporter = Exporter(model)
            result = exporter.export(path, format="torchscript", input_shape=input_shape)
            assert os.path.exists(result)

    def test_invalid_format_raises(self, model):
        from flashedge.engine.exporter import Exporter

        exporter = Exporter(model)
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export("model.xyz", format="invalid_format")


# ---------------------------------------------------------------------------
# ONNX Runtime Inference
# ---------------------------------------------------------------------------


class TestONNXRuntimeInference:
    def test_inference(self, model, input_shape):
        from flashedge.export.onnx_export import OnnxExporter
        from flashedge.runtime.onnx_runtime import OnnxInferenceSession

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.onnx")
            exporter = OnnxExporter(opset_version=13, simplify=False, optimize=False)
            exporter.export(model, path, input_shape=input_shape)

            runtime = OnnxInferenceSession(path)
            dummy = torch.randn(*input_shape).numpy()
            outputs = runtime.predict(dummy)
            assert outputs[0].shape == (1, 10)


# ---------------------------------------------------------------------------
# Optimization: INT8 Quantization
# ---------------------------------------------------------------------------


class TestINT8Quantization:
    def test_dynamic_quantization(self, model):
        from flashedge.optimization.quantization import DynamicQuantizer

        quantizer = DynamicQuantizer()
        quantized = quantizer.quantize(model)
        dummy = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = quantized(dummy)
        assert out.shape == (1, 10)

    def test_static_quantization(self, model, input_shape):
        from flashedge.optimization.quantization import INT8Quantizer

        quantizer = INT8Quantizer(calibration_samples=10, backend="fbgemm")
        quantized = quantizer.quantize(model, input_shape=input_shape)
        assert quantized is not None


# ---------------------------------------------------------------------------
# Optimization: Pruning
# ---------------------------------------------------------------------------


class TestStructuredPruning:
    def test_basic_pruning(self, model):
        from flashedge.optimization.pruning import StructuredPruner

        pruner = StructuredPruner(sparsity=0.3, iterative=False)
        pruned = pruner.prune(model)
        dummy = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = pruned(dummy)
        assert out.shape == (1, 10)

    def test_iterative_pruning(self, model):
        from flashedge.optimization.pruning import StructuredPruner

        pruner = StructuredPruner(sparsity=0.3, iterative=True, iterations=2)
        pruned = pruner.prune(model)
        assert pruned is not None

    def test_importance_scores(self, model):
        from flashedge.optimization.pruning import StructuredPruner

        pruner = StructuredPruner(sparsity=0.3)
        scores = pruner.get_importance_scores(model)
        assert len(scores) > 0
        assert all(isinstance(v, float) for v in scores.values())

    def test_invalid_sparsity(self):
        from flashedge.optimization.pruning import StructuredPruner

        with pytest.raises(ValueError, match="sparsity"):
            StructuredPruner(sparsity=1.5)


class TestUnstructuredPruning:
    def test_magnitude_pruning(self, model):
        from flashedge.optimization.pruning import UnstructuredPruner

        pruner = UnstructuredPruner(sparsity=0.5, method="magnitude")
        pruned = pruner.prune(model)
        dummy = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out = pruned(dummy)
        assert out.shape == (1, 10)

    def test_global_pruning(self, model):
        from flashedge.optimization.pruning import UnstructuredPruner

        pruner = UnstructuredPruner(sparsity=0.4, global_pruning=True)
        pruned = pruner.prune(model)
        assert pruned is not None


# ---------------------------------------------------------------------------
# Optimization: Graph Optimization (operator fusion)
# ---------------------------------------------------------------------------


class TestGraphOptimization:
    def test_conv_bn_fusion(self, model, input_shape):
        from flashedge.optimization.graph_optimization import GraphOptimizer

        optimizer = GraphOptimizer(
            fuse_conv_bn=True, fuse_conv_relu=False, constant_folding=False, dead_code_elimination=False
        )
        optimized = optimizer.optimize(model, input_shape=input_shape)
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            out = optimized(dummy)
        assert out.shape == (1, 10)

    def test_conv_relu_fusion(self, model, input_shape):
        from flashedge.optimization.graph_optimization import GraphOptimizer

        optimizer = GraphOptimizer(
            fuse_conv_bn=True, fuse_conv_relu=True, constant_folding=False, dead_code_elimination=False
        )
        optimized = optimizer.optimize(model, input_shape=input_shape)
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            out = optimized(dummy)
        assert out.shape == (1, 10)

    def test_constant_folding(self, model, input_shape):
        from flashedge.optimization.graph_optimization import GraphOptimizer

        optimizer = GraphOptimizer(
            fuse_conv_bn=False, fuse_conv_relu=False, constant_folding=True, dead_code_elimination=False
        )
        optimized = optimizer.optimize(model, input_shape=input_shape)
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            out = optimized(dummy)
        assert out.shape == (1, 10)

    def test_full_optimization(self, model, input_shape):
        from flashedge.optimization.graph_optimization import GraphOptimizer

        optimizer = GraphOptimizer()
        optimized = optimizer.optimize(model, input_shape=input_shape)
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            out = optimized(dummy)
        assert out.shape == (1, 10)

    def test_analyze(self, model, input_shape):
        from flashedge.optimization.graph_optimization import GraphOptimizer

        optimizer = GraphOptimizer()
        analysis = optimizer.analyze(model, input_shape=input_shape)
        assert "conv_bn_pairs" in analysis
        assert "conv_relu_pairs" in analysis
        assert analysis["total_params"] > 0


# ---------------------------------------------------------------------------
# Profiling: Latency
# ---------------------------------------------------------------------------


class TestLatencyProfiler:
    def test_profile(self, model, input_shape):
        from flashedge.profiling.latency import LatencyProfiler

        profiler = LatencyProfiler(device="cpu")
        result = profiler.profile(model, input_shape, warmup=2, runs=5)
        assert result["mean_ms"] > 0
        assert result["fps"] > 0
        assert result["p50_ms"] > 0

    def test_per_layer(self, model, input_shape):
        from flashedge.profiling.latency import LatencyProfiler

        profiler = LatencyProfiler(device="cpu")
        result = profiler.profile_per_layer(model, input_shape, runs=3)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Profiling: Memory
# ---------------------------------------------------------------------------


class TestMemoryProfiler:
    def test_profile(self, model, input_shape):
        from flashedge.profiling.memory import MemoryProfiler

        profiler = MemoryProfiler()
        result = profiler.profile(model, input_shape)
        assert result["num_parameters"] > 0
        assert result["model_size_mb"] > 0


# ---------------------------------------------------------------------------
# Profiling: FLOPs
# ---------------------------------------------------------------------------


class TestFLOPsCounter:
    def test_count(self, model, input_shape):
        from flashedge.profiling.flops import FLOPsCounter

        counter = FLOPsCounter()
        result = counter.count(model, input_shape)
        assert result["flops"] > 0
        assert result["macs"] > 0
        assert "flops_str" in result

    def test_summary(self, model, input_shape):
        from flashedge.profiling.flops import FLOPsCounter

        counter = FLOPsCounter()
        summary = counter.summary(model, input_shape)
        assert "FLOPs" in summary


# ---------------------------------------------------------------------------
# Profiling: Power Estimator
# ---------------------------------------------------------------------------


class TestPowerEstimator:
    def test_estimate(self, model, input_shape):
        from flashedge.profiling.power import PowerEstimator

        estimator = PowerEstimator(device_type="mobile")
        result = estimator.estimate(model, input_shape)
        assert result["estimated_watts"] > 0
        assert result["energy_per_inference_mj"] > 0

    def test_list_profiles(self):
        from flashedge.profiling.power import PowerEstimator

        profiles = PowerEstimator.list_device_profiles()
        assert "mobile" in profiles
        assert "raspberry_pi" in profiles

    def test_invalid_device_raises(self):
        from flashedge.profiling.power import PowerEstimator

        with pytest.raises(ValueError, match="Unknown device_type"):
            PowerEstimator(device_type="nonexistent")


# ---------------------------------------------------------------------------
# Profiling: Device Benchmarks
# ---------------------------------------------------------------------------


class TestDeviceBenchmark:
    def test_estimate_latency(self, model, input_shape):
        from flashedge.profiling.device_benchmark import DeviceBenchmark

        bench = DeviceBenchmark("raspberry_pi_4")
        result = bench.estimate_latency(model, input_shape)
        assert result["estimated_latency_ms"] > 0
        assert result["bottleneck"] in ("compute", "memory")
        assert result["fps"] > 0

    def test_check_feasibility(self, model, input_shape):
        from flashedge.profiling.device_benchmark import DeviceBenchmark

        bench = DeviceBenchmark("jetson_orin_nano")
        result = bench.check_deployment_feasibility(model, input_shape)
        assert "fits_in_memory" in result
        assert "meets_realtime" in result
        assert "recommendations" in result

    def test_invalid_profile_raises(self):
        from flashedge.profiling.device_benchmark import DeviceBenchmark

        with pytest.raises(ValueError, match="Unknown device profile"):
            DeviceBenchmark("nonexistent_device")

    def test_compare_devices(self, model, input_shape):
        from flashedge.profiling.device_benchmark import DeviceBenchmark

        bench = DeviceBenchmark("raspberry_pi_4")
        results = bench.compare_devices(model, device_names=["raspberry_pi_4", "jetson_nano"], input_shape=input_shape)
        assert len(results) == 2
        assert results[0]["latency_ms"] <= results[1]["latency_ms"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestFlashEdgeCLI:
    def test_build_parser(self):
        from flashedge.cli import build_parser

        parser = build_parser()
        assert parser is not None

    def test_version_cmd(self, capsys):
        import argparse

        from flashedge.cli import cmd_version

        cmd_version(argparse.Namespace())
        captured = capsys.readouterr()
        assert "FlashEdge" in captured.out

    def test_check_cmd(self, capsys):
        import argparse

        from flashedge.cli import cmd_check

        cmd_check(argparse.Namespace())
        captured = capsys.readouterr()
        assert "PyTorch" in captured.out


# ---------------------------------------------------------------------------
# Integration: model → export → optimize → profile
# ---------------------------------------------------------------------------


class TestIntegrationEdge:
    def test_export_then_profile(self, model, input_shape):
        from flashedge.export.onnx_export import OnnxExporter
        from flashedge.profiling.flops import FLOPsCounter
        from flashedge.profiling.memory import MemoryProfiler

        mem = MemoryProfiler()
        mem_result = mem.profile(model, input_shape)
        assert mem_result["num_parameters"] > 0

        counter = FLOPsCounter()
        flops_result = counter.count(model, input_shape)
        assert flops_result["flops"] > 0

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.onnx")
            exporter = OnnxExporter(opset_version=13, simplify=False, optimize=False)
            exporter.export(model, path, input_shape=input_shape)
            assert os.path.getsize(path) > 0

    def test_optimize_then_export(self, model, input_shape):
        from flashedge.optimization.pruning import UnstructuredPruner

        pruner = UnstructuredPruner(sparsity=0.3)
        pruned = pruner.prune(model)

        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            out = pruned(dummy)
        assert out.shape == (1, 10)
