# FAQ

## General

### What is FlashEdge?
FlashEdge is an Edge AI & TinyML toolkit for converting, optimizing, and deploying deep learning models on resource-constrained devices.

### What formats does FlashEdge support?
ONNX, TFLite, CoreML, OpenVINO IR, and TorchScript.

### Do I need a GPU?
No. FlashEdge works on CPU-only machines. A GPU accelerates export and profiling but is not required.

## Export

### Why is my ONNX export failing?
Common causes:
1. Unsupported operations — check the opset version
2. Dynamic control flow — use TorchScript tracing instead of scripting
3. In-place operations — refactor to out-of-place equivalents

### How do I reduce ONNX model size?
- Use FP16 export: `OnnxExporter(fp16=True)`
- Apply INT8 quantization after export
- Prune the model before export

## Optimization

### What is the best quantization method?
- For maximum compatibility: dynamic quantization
- For best accuracy: post-training quantization with calibration data
- For training flexibility: quantization-aware training

### How much accuracy do I lose with INT8?
Typically less than 1% accuracy drop for well-calibrated models. Use representative calibration data for best results.

## Deployment

### Can I run FlashEdge models on Raspberry Pi?
Yes. Export to TFLite or ONNX and use the respective runtime on ARM.

### Can I run models on Android?
Yes. Export to TFLite and use the TFLite Android runtime.

### Can I run models on iOS?
Yes. Export to CoreML and integrate with Xcode.
