"""Tests for model export functionality."""

from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.nn as nn

pytest.importorskip("onnxscript")


class SimpleModel(nn.Module):
    """Minimal model for testing exports."""

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


class TestOnnxExport:
    """Test ONNX export pipeline."""

    def test_basic_export(self, model, input_shape):
        from flashedge.export.onnx_export import OnnxExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")
            exporter = OnnxExporter(opset_version=13, simplify=False, optimize=False)
            result = exporter.export(model, output_path, input_shape=input_shape)

            assert os.path.exists(result)
            assert os.path.getsize(result) > 0

    def test_export_with_simplification(self, model, input_shape):
        from flashedge.export.onnx_export import OnnxExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model_sim.onnx")
            exporter = OnnxExporter(opset_version=13, simplify=True, optimize=True)
            result = exporter.export(model, output_path, input_shape=input_shape)
            assert os.path.exists(result)

    def test_export_validation(self, model, input_shape):
        from flashedge.export.onnx_export import OnnxExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")
            exporter = OnnxExporter(opset_version=13, simplify=False, optimize=False)
            exporter.export(model, output_path, input_shape=input_shape)

            result = exporter.validate(output_path, model=model, input_shape=input_shape, atol=1e-4)
            assert result["valid_schema"]
            assert result["inference_ok"]
            assert result["outputs_match"]

    def test_get_model_info(self, model, input_shape):
        from flashedge.export.onnx_export import OnnxExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")
            exporter = OnnxExporter(opset_version=13, simplify=False, optimize=False)
            exporter.export(model, output_path, input_shape=input_shape)

            info = exporter.get_model_info(output_path)
            assert info["file_size_bytes"] > 0
            assert info["num_nodes"] > 0
            assert len(info["inputs"]) == 1
            assert len(info["outputs"]) == 1


class TestTorchScriptExport:
    """Test TorchScript export pipeline."""

    def test_trace_export(self, model, input_shape):
        from flashedge.export.torchscript import TorchScriptExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")
            exporter = TorchScriptExporter(method="trace")
            result = exporter.export(model, output_path, input_shape=input_shape)

            assert os.path.exists(result)

            loaded = torch.jit.load(result)
            dummy = torch.randn(*input_shape)
            with torch.no_grad():
                output = loaded(dummy)
            assert output.shape == (1, 10)

    def test_validation(self, model, input_shape):
        from flashedge.export.torchscript import TorchScriptExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")
            exporter = TorchScriptExporter(method="trace")
            exporter.export(model, output_path, input_shape=input_shape)

            result = exporter.validate(output_path, model=model, input_shape=input_shape)
            assert result["load_ok"]
            assert result["inference_ok"]
            assert result["outputs_match"]


class TestExporter:
    """Test the unified Exporter class."""

    def test_export_onnx(self, model, input_shape):
        from flashedge.engine.exporter import Exporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.onnx")
            exporter = Exporter(model)
            result = exporter.export(output_path, format="onnx", input_shape=input_shape)
            assert os.path.exists(result)

    def test_export_torchscript(self, model, input_shape):
        from flashedge.engine.exporter import Exporter

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "model.pt")
            exporter = Exporter(model)
            result = exporter.export(output_path, format="torchscript", input_shape=input_shape)
            assert os.path.exists(result)

    def test_invalid_format(self, model):
        from flashedge.engine.exporter import Exporter

        exporter = Exporter(model)
        with pytest.raises(ValueError, match="Unsupported format"):
            exporter.export("model.xyz", format="invalid")
