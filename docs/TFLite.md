# TFLite Export

FlashEdge supports exporting PyTorch models to TensorFlow Lite format for mobile and embedded deployment.

## Export Pipeline

The TFLite export works through a PyTorch → ONNX → TensorFlow → TFLite pipeline.

```python
from flashedge.export import TFLiteExporter

exporter = TFLiteExporter(quantize=True, dtype="int8")
exporter.export(model, "model.tflite", input_shape=(1, 3, 224, 224))
```

## Quantized Export

```python
exporter = TFLiteExporter(
    quantize=True,
    dtype="int8",
    representative_dataset="data/calibration/",
    num_calibration_samples=200,
)
exporter.export(model, "model_int8.tflite", input_shape=(1, 3, 224, 224))
```

## TFLite Runtime Inference

```python
from flashedge.runtime import TFLiteInferenceSession

session = TFLiteInferenceSession("model.tflite")
output = session.predict(input_tensor)
```

## Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `quantize` | False | Enable quantization |
| `dtype` | "float32" | Target dtype (float32, float16, int8) |
| `optimize_for_size` | True | Optimize for model size |
| `target_ops` | TFLITE_BUILTINS | Target operator sets |
