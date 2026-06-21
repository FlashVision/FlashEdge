"""Example: Profile a model's latency, memory, FLOPs, and power consumption.

Usage:
    python examples/profile_model.py
"""

from flashedge.models.architectures.mobilenet import MobileNetV3Small
from flashedge.profiling.latency import LatencyProfiler
from flashedge.profiling.memory import MemoryProfiler
from flashedge.profiling.flops import FLOPsCounter
from flashedge.profiling.power import PowerEstimator

model = MobileNetV3Small(num_classes=1000)
input_shape = (1, 3, 224, 224)

print("=" * 60)
print("  MobileNetV3-Small Profiling")
print("=" * 60)

print("\n1. Latency Profiling (CPU)")
lat_profiler = LatencyProfiler(device="cpu")
latency = lat_profiler.profile(model, input_shape, warmup=10, runs=50)
print(f"   Mean: {latency['mean_ms']:.3f} ms | P95: {latency['p95_ms']:.3f} ms | FPS: {latency['fps']:.1f}")

print("\n2. Memory Profiling")
mem_profiler = MemoryProfiler()
memory = mem_profiler.profile(model, input_shape)
print(f"   Parameters:  {memory['num_parameters']:,}")
print(f"   Model size:  {memory['model_size_mb']:.2f} MB")
print(f"   Peak memory: {memory['peak_memory_mb']:.2f} MB")

print("\n3. FLOPs Counting")
flops_counter = FLOPsCounter()
flops = flops_counter.count(model, input_shape)
print(f"   FLOPs: {flops['flops_str']}")
print(f"   MACs:  {flops['macs_str']}")

print("\n4. Power Estimation (Mobile)")
power_est = PowerEstimator(device_type="mobile")
power = power_est.estimate(model, input_shape, latency_ms=latency["mean_ms"])
print(f"   Estimated power:      {power['estimated_watts']:.3f} W")
print(f"   Energy/inference:     {power['energy_per_inference_mj']:.3f} mJ")
print(f"   Inferences per kWh:   {power['inferences_per_kwh']:.0f}")

print("\n5. Per-Layer Latency Breakdown (top 5)")
per_layer = lat_profiler.profile_per_layer(model, input_shape, runs=5)
for i, layer in enumerate(per_layer[:5]):
    print(f"   {layer['layer']:<40} {layer['mean_ms']:.3f} ms ({layer['percentage']:.1f}%)")

print("\n" + "=" * 60)
