"""Example: Benchmark and compare models across formats.

Usage:
    python examples/benchmark_latency.py

Note: Run export_onnx.py first to generate the ONNX model files.
"""

import os
from pathlib import Path

from flashedge.models.architectures.mobilenet import MobileNetV3Small
from flashedge.export.onnx_export import OnnxExporter
from flashedge.optimization.quantization import INT8Quantizer
from flashedge.analytics.benchmark import Benchmark

os.makedirs("exported", exist_ok=True)

model = MobileNetV3Small(num_classes=1000)
input_shape = (1, 3, 224, 224)

onnx_path = "exported/mobilenetv3_small.onnx"
int8_path = "exported/mobilenetv3_small_int8.onnx"

if not Path(onnx_path).exists():
    print("Exporting ONNX model...")
    exporter = OnnxExporter(opset_version=17, simplify=True)
    exporter.export(model, onnx_path, input_shape=input_shape)

if not Path(int8_path).exists():
    print("Quantizing to INT8...")
    quantizer = INT8Quantizer()
    quantizer.quantize_onnx(onnx_path, int8_path)

print("\nRunning benchmark...")
bench = Benchmark()
results = bench.compare(
    models={
        "FP32 ONNX": onnx_path,
        "INT8 ONNX": int8_path,
    },
    input_shape=input_shape,
    warmup=10,
    runs=50,
)

bench.print_report(results)

csv_path = bench.to_csv(results, "exported/benchmark_results.csv")
print(f"\nResults saved to: {csv_path}")
