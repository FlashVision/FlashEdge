<p align="center">
  <h1 align="center">⚡ FlashEdge</h1>
  <p align="center">
    <strong>Edge AI & TinyML Toolkit for FlashVision</strong>
  </p>
  <p align="center">
    ONNX • TFLite • CoreML • OpenVINO • Quantization • Pruning • Profiling • Deployment
  </p>
</p>

<p align="center">
  <a href="https://github.com/FlashVision/FlashEdge/actions"><img src="https://img.shields.io/github/actions/workflow/status/FlashVision/FlashEdge/ci.yml?style=flat-square&logo=github" alt="CI"></a>
  <a href="https://pypi.org/project/flashedge/"><img src="https://img.shields.io/pypi/v/flashedge?style=flat-square&logo=pypi" alt="PyPI"></a>
  <a href="https://pypi.org/project/flashedge/"><img src="https://img.shields.io/pypi/pyversions/flashedge?style=flat-square&logo=python" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" alt="License"></a>
  <a href="https://github.com/FlashVision/FlashEdge"><img src="https://img.shields.io/github/stars/FlashVision/FlashEdge?style=flat-square&logo=github" alt="Stars"></a>
</p>

---

## What is FlashEdge?

**FlashEdge** is a comprehensive Edge AI & TinyML toolkit for the [FlashVision](https://github.com/FlashVision) ecosystem. It provides everything you need to compress, convert, profile, and deploy deep learning models on edge devices, mobile platforms, and IoT hardware.

### Key Features

| Feature | Description |
|---------|-------------|
| **ONNX Export** | PyTorch to ONNX conversion with graph optimization and simplification |
| **TFLite Export** | PyTorch to TensorFlow Lite for Android/embedded deployment |
| **CoreML Export** | PyTorch to CoreML for iOS/macOS deployment |
| **OpenVINO Export** | PyTorch to OpenVINO IR for Intel hardware |
| **Quantization** | INT8/INT4 post-training and quantization-aware training |
| **Pruning** | Structured channel/filter pruning for mobile-optimized models |
| **Distillation** | Knowledge distillation from large to compact models |
| **NAS** | Latency-constrained neural architecture search |
| **Profiling** | Latency, memory, FLOPs/MACs, and power consumption analysis |
| **Runtime** | Unified inference API for ONNX Runtime, TFLite, OpenVINO |
| **Benchmarking** | Cross-format accuracy, latency, throughput, and model size comparison |

---

## Installation

### From PyPI (recommended)

```bash
pip install flashedge
```

### From Source

```bash
git clone https://github.com/FlashVision/FlashEdge.git
cd FlashEdge
pip install -e ".[all]"
```

### With Optional Dependencies

```bash
pip install flashedge[tflite]      # TFLite runtime support
pip install flashedge[openvino]    # OpenVINO support
pip install flashedge[coreml]      # CoreML support
pip install flashedge[analytics]   # Visualization & profiling
pip install flashedge[all]         # Everything
pip install flashedge[dev]         # Development tools
```

---

## Quick Start

### Export a Model to ONNX

```python
from flashedge import FlashEdge, Exporter

model = FlashEdge("pretrained/model.pth")
exporter = Exporter(model)
exporter.export_onnx("model.onnx", input_shape=(1, 3, 224, 224), opset=17)
```

### Profile Model Latency

```python
from flashedge import DeviceProfiler

profiler = DeviceProfiler(device="cpu")
report = profiler.profile("model.onnx", input_shape=(1, 3, 224, 224), runs=100)
print(f"Latency: {report['mean_ms']:.2f} ms | Throughput: {report['fps']:.1f} FPS")
```

### Quantize for Edge (INT8)

```python
from flashedge import ModelOptimizer

optimizer = ModelOptimizer()
optimizer.quantize("model.onnx", output="model_int8.onnx", dtype="int8")
```

### Benchmark Across Formats

```python
from flashedge import Benchmark

bench = Benchmark()
results = bench.compare(
    models={"onnx": "model.onnx", "int8": "model_int8.onnx"},
    input_shape=(1, 3, 224, 224),
    runs=100,
)
bench.print_report(results)
```

---

## CLI Usage

```bash
# Show version
flashedge version

# Check installation
flashedge check

# Export model to ONNX
flashedge export --config configs/flashedge_onnx_mobile.yaml

# Profile model latency
flashedge profile --model model.onnx --device cpu --runs 100

# Quantize model
flashedge optimize --config configs/flashedge_tflite.yaml

# Benchmark model
flashedge benchmark --model model.onnx --device cpu --runs 200

# Deploy to edge device
flashedge deploy --model model.onnx --target onnxruntime --output deployed/
```

---

## Project Structure

```
FlashEdge/
├── configs/          # YAML configuration files
├── docker/           # Docker support
├── docs/             # Documentation
├── examples/         # Runnable example scripts
├── flashedge/        # Core library
│   ├── cfg/          # Configuration management
│   ├── data/         # Calibration & benchmark data
│   ├── engine/       # Training, validation, export
│   ├── export/       # Format exporters (ONNX, TFLite, CoreML, OpenVINO)
│   ├── optimization/ # Quantization, pruning, distillation, NAS
│   ├── runtime/      # Inference runtimes
│   ├── profiling/    # Latency, memory, FLOPs, power
│   ├── models/       # Model architectures
│   ├── solutions/    # High-level solutions
│   ├── analytics/    # Benchmarking & metrics
│   └── utils/        # Utilities
└── tests/            # Unit tests
```

---

## Benchmarks

| Model | Format | Size | Latency (CPU) | Throughput | Accuracy |
|-------|--------|------|---------------|------------|----------|
| MobileNetV3-S | PyTorch | 10.2 MB | 8.5 ms | 118 FPS | 67.4% |
| MobileNetV3-S | ONNX | 9.8 MB | 5.2 ms | 192 FPS | 67.4% |
| MobileNetV3-S | ONNX INT8 | 2.8 MB | 2.1 ms | 476 FPS | 66.9% |
| MobileNetV3-S | TFLite | 2.6 MB | 3.1 ms | 323 FPS | 67.2% |
| EfficientNet-B0 | ONNX | 20.4 MB | 12.3 ms | 81 FPS | 77.1% |
| EfficientNet-B0 | ONNX INT8 | 5.4 MB | 4.8 ms | 208 FPS | 76.5% |

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [ONNX](https://onnx.ai/) for model interoperability
- [ONNX Runtime](https://onnxruntime.ai/) for optimized inference
- [FlashVision](https://github.com/FlashVision) ecosystem

---

<p align="center">
  Made with ❤️ by the <a href="https://github.com/FlashVision">FlashVision</a> team
</p>
