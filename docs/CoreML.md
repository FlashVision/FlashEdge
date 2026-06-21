# CoreML Export

FlashEdge supports exporting PyTorch models to CoreML format for iOS, macOS, watchOS, and tvOS deployment.

## Basic Export

```python
from flashedge.export import CoreMLExporter

exporter = CoreMLExporter(compute_units="ALL", minimum_deployment_target="iOS16")
exporter.export(model, "model.mlpackage", input_shape=(1, 3, 224, 224))
```

## Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compute_units` | "ALL" | CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ALL |
| `minimum_deployment_target` | "iOS16" | Minimum iOS/macOS version |
| `convert_to` | "mlprogram" | mlprogram or neuralnetwork |
| `classifier` | False | Export as classifier model |

## Compute Units

- `CPU_ONLY` — CPU only
- `CPU_AND_GPU` — CPU and GPU
- `CPU_AND_NE` — CPU and Neural Engine
- `ALL` — All available compute units (recommended)

## With Classification Labels

```python
exporter = CoreMLExporter(classifier=True)
exporter.export(
    model, "classifier.mlpackage",
    input_shape=(1, 3, 224, 224),
    class_labels=["cat", "dog", "bird"],
)
```
