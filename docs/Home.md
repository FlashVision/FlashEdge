# FlashEdge Documentation

Welcome to the FlashEdge documentation! FlashEdge is a comprehensive Edge AI & TinyML toolkit for the FlashVision ecosystem.

## Overview

FlashEdge bridges the gap between training deep learning models and deploying them on resource-constrained edge devices. It provides:

- **Multi-format export** — Convert PyTorch models to ONNX, TFLite, CoreML, OpenVINO IR, and TorchScript
- **Model compression** — INT8/INT4 quantization, structured pruning, knowledge distillation
- **Profiling tools** — Measure latency, memory, FLOPs, and power consumption
- **Runtime wrappers** — Unified inference API across ONNX Runtime, TFLite, and OpenVINO
- **Benchmarking** — Compare models across formats and optimization levels

## Getting Started

1. [Installation](Installation.md) — Set up FlashEdge
2. [Quick Start](Quick-Start.md) — Your first edge export
3. [ONNX Export](ONNX-Export.md) — PyTorch to ONNX
4. [TFLite](TFLite.md) — PyTorch to TFLite
5. [CoreML](CoreML.md) — PyTorch to CoreML
6. [Profiling](Profiling.md) — Latency & memory analysis
7. [FAQ](FAQ.md) — Frequently asked questions

## Architecture

```
PyTorch Model
     │
     ├── ONNX Export ──→ ONNX Runtime (CPU/GPU/NPU)
     ├── TFLite Export ─→ TFLite Runtime (Android/Embedded)
     ├── CoreML Export ─→ CoreML (iOS/macOS)
     ├── OpenVINO ─────→ OpenVINO Runtime (Intel)
     └── TorchScript ──→ LibTorch (C++/Mobile)
```

## Supported Hardware

| Platform | Runtime | Status |
|----------|---------|--------|
| x86 CPU | ONNX Runtime | Fully supported |
| ARM CPU | ONNX Runtime / TFLite | Fully supported |
| Intel NPU | OpenVINO | Fully supported |
| NVIDIA GPU | ONNX Runtime (CUDA) | Fully supported |
| Apple Neural Engine | CoreML | Fully supported |
| Android | TFLite | Fully supported |
| Raspberry Pi | TFLite / ONNX Runtime | Fully supported |
