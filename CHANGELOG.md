# Changelog

All notable changes to FlashEdge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2025-06-21

### Added
- Initial release of FlashEdge
- **Export**: ONNX, TFLite, CoreML, OpenVINO IR, TorchScript export pipelines
- **Optimization**: INT8/INT4 quantization, structured pruning, knowledge distillation, mobile-aware NAS
- **Runtime**: ONNX Runtime, TFLite interpreter, OpenVINO inference wrappers
- **Profiling**: Latency (CPU/GPU/NPU), memory footprint, FLOPs/MACs, power estimation
- **Models**: MobileNetV3, EfficientNet-Lite, TinyViT architectures
- **Engine**: Distillation trainer, validator, predictor, exporter
- **Solutions**: EdgeDeployer, ModelOptimizer, DeviceProfiler
- **Analytics**: Benchmarking with latency, throughput, model size, accuracy metrics
- **CLI**: version, settings, check, export, profile, optimize, benchmark, deploy commands
- Docker support with GPU-enabled Dockerfile
- GitHub Actions CI pipeline
- Documentation and examples

---

## [Unreleased]

### Planned
- TensorRT export backend
- On-device benchmark agents for Android/iOS
- NNAPI delegate support
- Federated edge training
- Model zoo with pre-optimized edge checkpoints
