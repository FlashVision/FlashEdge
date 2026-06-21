"""Example: Deploy a model to an edge runtime.

Usage:
    python examples/deploy_edge.py
"""

from flashedge.models.architectures.mobilenet import MobileNetV3Small
from flashedge.solutions.edge_deployer import EdgeDeployer
from flashedge.solutions.device_profiler import DeviceProfiler
from flashedge.utils.io import save_model

model = MobileNetV3Small(num_classes=1000)

checkpoint_path = "exported/mobilenetv3_small_checkpoint.pth"
save_model(model, checkpoint_path, metadata={"architecture": "mobilenetv3_small", "num_classes": 1000})
print(f"Saved checkpoint: {checkpoint_path}")

deployer = EdgeDeployer()
result = deployer.deploy(
    checkpoint_path,
    target="onnxruntime",
    output_dir="deployed/onnxruntime",
    input_shape=(1, 3, 224, 224),
    quantize=False,
    validate=True,
)

print(f"\nDeployment result:")
print(f"  Target:    {result['target']}")
print(f"  Model:     {result['model_path']}")
print(f"  Manifest:  {result['manifest']}")

print("\n--- Device Profiling ---")
profiler = DeviceProfiler(device="cpu")
report = profiler.full_report(model, input_shape=(1, 3, 224, 224), device_type="mobile")
profiler.print_report(report)
