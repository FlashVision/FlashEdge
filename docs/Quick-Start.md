# Quick Start

This guide walks you through exporting, optimizing, and profiling a model for edge deployment.

## 1. Export to ONNX

```python
import torch
from flashedge.export import OnnxExporter

model = torch.hub.load("pytorch/vision", "mobilenet_v3_small", pretrained=True)
model.eval()

exporter = OnnxExporter(opset_version=17, simplify=True)
exporter.export(model, "mobilenet_v3_small.onnx", input_shape=(1, 3, 224, 224))
```

## 2. Quantize to INT8

```python
from flashedge.optimization import INT8Quantizer

quantizer = INT8Quantizer(per_channel=True)
quantizer.quantize_onnx("mobilenet_v3_small.onnx", "mobilenet_v3_small_int8.onnx")
```

## 3. Profile Latency

```python
from flashedge.profiling import LatencyProfiler

profiler = LatencyProfiler(device="cpu")
report = profiler.profile("mobilenet_v3_small_int8.onnx", input_shape=(1, 3, 224, 224))
print(f"Mean: {report['mean_ms']:.2f} ms | P99: {report['p99_ms']:.2f} ms")
```

## 4. Benchmark Comparison

```python
from flashedge.analytics import Benchmark

bench = Benchmark()
results = bench.compare(
    models={
        "fp32": "mobilenet_v3_small.onnx",
        "int8": "mobilenet_v3_small_int8.onnx",
    },
    input_shape=(1, 3, 224, 224),
)
bench.print_report(results)
```

## CLI Workflow

```bash
# Export
flashedge export --config configs/flashedge_onnx_mobile.yaml

# Profile
flashedge profile --model mobilenet_v3_small.onnx --device cpu --runs 100

# Benchmark
flashedge benchmark --model mobilenet_v3_small_int8.onnx --device cpu
```
