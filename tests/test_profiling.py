"""Tests for profiling functionality."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture
def model():
    return SimpleModel().eval()


@pytest.fixture
def input_shape():
    return (1, 3, 32, 32)


class TestLatencyProfiler:
    def test_profile_pytorch(self, model, input_shape):
        from flashedge.profiling.latency import LatencyProfiler

        profiler = LatencyProfiler(device="cpu")
        result = profiler.profile(model, input_shape, warmup=2, runs=5)

        assert "mean_ms" in result
        assert "std_ms" in result
        assert "p50_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert "fps" in result
        assert result["mean_ms"] > 0
        assert result["fps"] > 0

    def test_profile_per_layer(self, model, input_shape):
        from flashedge.profiling.latency import LatencyProfiler

        profiler = LatencyProfiler(device="cpu")
        result = profiler.profile_per_layer(model, input_shape, runs=3)

        assert len(result) > 0
        assert all("layer" in r for r in result)
        assert all("mean_ms" in r for r in result)
        assert all(r["mean_ms"] >= 0 for r in result)


class TestMemoryProfiler:
    def test_profile(self, model, input_shape):
        from flashedge.profiling.memory import MemoryProfiler

        profiler = MemoryProfiler()
        result = profiler.profile(model, input_shape)

        assert result["num_parameters"] > 0
        assert result["model_size_mb"] > 0
        assert result["peak_memory_mb"] >= 0
        assert result["num_trainable"] > 0


class TestFLOPsCounter:
    def test_count(self, model, input_shape):
        from flashedge.profiling.flops import FLOPsCounter

        counter = FLOPsCounter()
        result = counter.count(model, input_shape)

        assert result["flops"] > 0
        assert result["macs"] > 0
        assert "flops_str" in result
        assert "macs_str" in result
        assert len(result["per_layer"]) > 0

    def test_summary(self, model, input_shape):
        from flashedge.profiling.flops import FLOPsCounter

        counter = FLOPsCounter()
        summary = counter.summary(model, input_shape)

        assert "FLOPs Summary" in summary
        assert "Total FLOPs" in summary


class TestPowerEstimator:
    def test_estimate(self, model, input_shape):
        from flashedge.profiling.power import PowerEstimator

        estimator = PowerEstimator(device_type="mobile")
        result = estimator.estimate(model, input_shape)

        assert result["estimated_watts"] > 0
        assert result["energy_per_inference_mj"] > 0
        assert result["inferences_per_kwh"] > 0
        assert result["device_type"] == "mobile"

    def test_list_profiles(self):
        from flashedge.profiling.power import PowerEstimator

        profiles = PowerEstimator.list_device_profiles()
        assert "mobile" in profiles
        assert "raspberry_pi" in profiles
        assert "jetson_nano" in profiles

    def test_invalid_device(self):
        from flashedge.profiling.power import PowerEstimator

        with pytest.raises(ValueError, match="Unknown device_type"):
            PowerEstimator(device_type="nonexistent")
