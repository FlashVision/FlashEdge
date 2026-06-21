"""Tests for the registry system."""

from __future__ import annotations

import pytest

from flashedge.registry import Registry, MODELS, EXPORTERS, RUNTIMES, OPTIMIZERS


class TestRegistry:
    def test_register_and_build(self):
        reg = Registry("test")

        @reg.register("MyClass")
        class MyClass:
            def __init__(self, value=1):
                self.value = value

        assert "MyClass" in reg
        assert len(reg) == 1

        instance = reg.build("MyClass", value=42)
        assert instance.value == 42

    def test_register_duplicate(self):
        reg = Registry("test")

        @reg.register("Dup")
        class Dup:
            pass

        with pytest.raises(KeyError, match="already registered"):

            @reg.register("Dup")
            class Dup2:
                pass

    def test_build_unknown(self):
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.build("Unknown")

    def test_get(self):
        reg = Registry("test")

        @reg.register("Foo")
        class Foo:
            pass

        assert reg.get("Foo") is Foo
        assert reg.get("Bar") is None

    def test_list(self):
        reg = Registry("test")

        @reg.register("B")
        class B:
            pass

        @reg.register("A")
        class A:
            pass

        assert reg.list() == ["A", "B"]

    def test_repr(self):
        reg = Registry("my_registry")
        assert "my_registry" in repr(reg)

    def test_global_registries_exist(self):
        assert MODELS.name == "models"
        assert EXPORTERS.name == "exporters"
        assert RUNTIMES.name == "runtimes"
        assert OPTIMIZERS.name == "optimizers"


class TestModelsRegistry:
    def test_mobilenet_registered(self):

        assert "mobilenetv3_small" in MODELS
        model = MODELS.build("mobilenetv3_small", num_classes=10)
        assert model is not None

        import torch

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 10)

    def test_efficientnet_registered(self):

        assert "efficientnet_lite" in MODELS

    def test_tiny_vit_registered(self):

        assert "tiny_vit" in MODELS


class TestExportersRegistry:
    def test_exporters_registered(self):

        assert "onnx" in EXPORTERS
        assert "tflite" in EXPORTERS
        assert "coreml" in EXPORTERS
        assert "openvino" in EXPORTERS
        assert "torchscript" in EXPORTERS


class TestOptimizersRegistry:
    def test_optimizers_registered(self):

        assert "int8_quantizer" in OPTIMIZERS
        assert "dynamic_quantizer" in OPTIMIZERS
        assert "structured_pruner" in OPTIMIZERS
        assert "unstructured_pruner" in OPTIMIZERS
        assert "knowledge_distiller" in OPTIMIZERS
        assert "mobile_nas" in OPTIMIZERS
