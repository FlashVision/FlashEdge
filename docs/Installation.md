# Installation

## Requirements

- Python 3.8+
- PyTorch 2.0+

## From PyPI

```bash
pip install flashedge
```

## From Source

```bash
git clone https://github.com/FlashVision/FlashEdge.git
cd FlashEdge
pip install -e ".[all]"
```

## Optional Dependencies

```bash
# TFLite runtime for on-device inference
pip install flashedge[tflite]

# OpenVINO for Intel hardware
pip install flashedge[openvino]

# CoreML for Apple devices
pip install flashedge[coreml]

# Visualization and analytics
pip install flashedge[analytics]

# Everything
pip install flashedge[all]

# Development tools
pip install flashedge[dev]
```

## Verify Installation

```bash
flashedge version
flashedge check
```

## Docker

```bash
docker build -t flashedge -f docker/Dockerfile .
docker run -it flashedge version
```
