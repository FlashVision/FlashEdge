# ONNX Export

FlashEdge provides comprehensive ONNX export with graph optimization and validation.

## Basic Export

```python
from flashedge.export import OnnxExporter

exporter = OnnxExporter(opset_version=17, simplify=True)
exporter.export(model, "model.onnx", input_shape=(1, 3, 224, 224))
```

## Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opset_version` | 17 | ONNX opset version |
| `simplify` | True | Run onnxsim graph simplification |
| `optimize` | True | Apply ONNX graph optimizations |
| `dynamic_axes` | None | Dynamic batch/sequence dimensions |
| `fp16` | False | Convert weights to FP16 |

## Dynamic Axes

```python
exporter.export(
    model, "model.onnx",
    input_shape=(1, 3, 224, 224),
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
```

## FP16 Export

```python
exporter = OnnxExporter(opset_version=17, fp16=True)
exporter.export(model, "model_fp16.onnx", input_shape=(1, 3, 224, 224))
```

## Validation

```python
exporter.validate("model.onnx", input_shape=(1, 3, 224, 224))
```

## ONNX Runtime Inference

```python
from flashedge.runtime import OnnxInferenceSession

session = OnnxInferenceSession("model.onnx", providers=["CPUExecutionProvider"])
output = session.predict(input_tensor)
```
