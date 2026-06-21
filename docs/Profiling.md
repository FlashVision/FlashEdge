# Profiling

FlashEdge provides comprehensive profiling tools for understanding model performance on edge hardware.

## Latency Profiling

```python
from flashedge.profiling import LatencyProfiler

profiler = LatencyProfiler(device="cpu")
report = profiler.profile("model.onnx", input_shape=(1, 3, 224, 224), warmup=10, runs=100)

print(f"Mean: {report['mean_ms']:.2f} ms")
print(f"Std:  {report['std_ms']:.2f} ms")
print(f"P50:  {report['p50_ms']:.2f} ms")
print(f"P95:  {report['p95_ms']:.2f} ms")
print(f"P99:  {report['p99_ms']:.2f} ms")
print(f"FPS:  {report['fps']:.1f}")
```

## Memory Profiling

```python
from flashedge.profiling import MemoryProfiler

profiler = MemoryProfiler()
report = profiler.profile(model, input_shape=(1, 3, 224, 224))

print(f"Model size: {report['model_size_mb']:.2f} MB")
print(f"Peak memory: {report['peak_memory_mb']:.2f} MB")
print(f"Parameters: {report['num_parameters']:,}")
```

## FLOPs Counting

```python
from flashedge.profiling import FLOPsCounter

counter = FLOPsCounter()
report = counter.count(model, input_shape=(1, 3, 224, 224))

print(f"FLOPs: {report['flops'] / 1e6:.2f} M")
print(f"MACs:  {report['macs'] / 1e6:.2f} M")
```

## Power Estimation

```python
from flashedge.profiling import PowerEstimator

estimator = PowerEstimator(device_type="mobile", tdp_watts=5.0)
report = estimator.estimate(model, input_shape=(1, 3, 224, 224))

print(f"Estimated power: {report['estimated_watts']:.3f} W")
print(f"Energy per inference: {report['energy_per_inference_mj']:.3f} mJ")
```

## CLI Usage

```bash
flashedge profile --model model.onnx --device cpu --runs 100
```
