"""Example: Export a model to TFLite format.

Usage:
    python examples/export_tflite.py

Note: Requires tensorflow or tflite-runtime for full conversion.
      Without TensorFlow, uses a fallback binary format.
"""

from flashedge.models.architectures.mobilenet import MobileNetV3Small
from flashedge.export.tflite_export import TFLiteExporter

model = MobileNetV3Small(num_classes=10)
model.eval()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

exporter = TFLiteExporter(
    quantize=False,
    dtype="float32",
    optimize_for_size=True,
)

output_path = exporter.export(
    model,
    "exported/mobilenetv3_small.tflite",
    input_shape=(1, 3, 224, 224),
)

print(f"\nExported TFLite model: {output_path}")

print("\nTo run with INT8 quantization:")
print("  exporter = TFLiteExporter(quantize=True, dtype='int8')")
