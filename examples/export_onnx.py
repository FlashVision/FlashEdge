"""Example: Export a MobileNetV3-Small model to ONNX format.

Usage:
    python examples/export_onnx.py
"""

from flashedge.models.architectures.mobilenet import MobileNetV3Small
from flashedge.export.onnx_export import OnnxExporter

model = MobileNetV3Small(num_classes=1000)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

exporter = OnnxExporter(
    opset_version=17,
    simplify=True,
    optimize=True,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)

output_path = exporter.export(
    model,
    "exported/mobilenetv3_small.onnx",
    input_shape=(1, 3, 224, 224),
)

info = exporter.get_model_info(output_path)
print(f"\nONNX Model Info:")
print(f"  Nodes:     {info['num_nodes']}")
print(f"  Size:      {info['file_size_mb']:.2f} MB")
print(f"  Opset:     {info['opset_version']}")
print(f"  Inputs:    {info['inputs']}")
print(f"  Outputs:   {info['outputs']}")

result = exporter.validate(output_path, model=model, input_shape=(1, 3, 224, 224))
print(f"\nValidation:")
print(f"  Schema valid: {result['valid_schema']}")
print(f"  Inference OK: {result['inference_ok']}")
print(f"  Max diff:     {result.get('max_output_diff', 'N/A')}")
