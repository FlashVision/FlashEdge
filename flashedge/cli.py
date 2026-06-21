"""FlashEdge Command-Line Interface.

Usage:
    flashedge version
    flashedge settings
    flashedge check
    flashedge export --config <path>
    flashedge profile --model <path> --device <dev>
    flashedge optimize --config <path>
    flashedge benchmark --model <path> --device <dev>
    flashedge deploy --model <path> --target <runtime> --output <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BANNER = r"""
  _____ _           _     _____    _
 |  ___| | __ _ ___| |__ | ____|__| | __ _  ___
 | |_  | |/ _` / __| '_ \|  _| / _` |/ _` |/ _ \
 |  _| | | (_| \__ \ | | | |__| (_| | (_| |  __/
 |_|   |_|\__,_|___/_| |_|_____\__,_|\__, |\___|
                                      |___/
"""


def cmd_version(args: argparse.Namespace) -> None:
    from flashedge import __version__

    print(f"FlashEdge v{__version__}")


def cmd_settings(args: argparse.Namespace) -> None:
    import torch

    from flashedge import __version__

    print(BANNER)
    print(f"  FlashEdge:   v{__version__}")
    print(f"  PyTorch:     v{torch.__version__}")
    print(f"  CUDA:        {'v' + torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
    print(f"  Python:      {sys.version.split()[0]}")
    print(f"  Platform:    {sys.platform}")
    print()


def cmd_check(args: argparse.Namespace) -> None:
    import importlib

    print("FlashEdge System Check")
    print("=" * 40)

    checks = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("tqdm", "tqdm"),
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNXRuntime"),
    ]

    optional = [
        ("tflite_runtime", "TFLite Runtime"),
        ("openvino", "OpenVINO"),
        ("coremltools", "CoreMLTools"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
    ]

    all_ok = True
    for module, name in checks:
        try:
            mod = importlib.import_module(module)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {name:15s} v{ver}")
        except ImportError:
            print(f"  [FAIL] {name:15s} NOT INSTALLED")
            all_ok = False

    print()
    print("Optional Dependencies:")
    for module, name in optional:
        try:
            mod = importlib.import_module(module)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {name:15s} v{ver}")
        except ImportError:
            print(f"  [--] {name:15s} not installed")

    print()
    if all_ok:
        print("All required dependencies OK!")
    else:
        print("Some dependencies are missing. Run: pip install flashedge[all]")


def cmd_export(args: argparse.Namespace) -> None:
    import yaml

    print(f"Exporting model with config: {args.config}")
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    export_cfg = config.get("export", {})
    model_cfg = config.get("model", {})
    fmt = export_cfg.get("format", "onnx")
    model_path = model_cfg.get("path")
    output = export_cfg.get("output", f"exported/model.{fmt}")
    input_shape = export_cfg.get("input_shape", [1, 3, 224, 224])

    print(f"  Format:      {fmt}")
    print(f"  Model:       {model_path}")
    print(f"  Output:      {output}")
    print(f"  Input shape: {input_shape}")

    import torch

    if model_path and Path(model_path).exists():
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(model, dict) and "model" in model:
            model = model["model"]
    else:
        print(f"  Warning: Model file not found at '{model_path}', creating dummy for demonstration")
        from flashedge.models.architectures.mobilenet import MobileNetV3Small

        model = MobileNetV3Small(num_classes=model_cfg.get("num_classes", 1000))

    model.eval()

    if fmt == "onnx":
        from flashedge.export.onnx_export import OnnxExporter

        exporter = OnnxExporter(
            opset_version=export_cfg.get("opset_version", 17),
            simplify=export_cfg.get("simplify", True),
            optimize=export_cfg.get("optimize", True),
            fp16=export_cfg.get("fp16", False),
        )
        exporter.export(model, output, input_shape=tuple(input_shape))
    elif fmt == "tflite":
        from flashedge.export.tflite_export import TFLiteExporter

        exporter = TFLiteExporter(
            quantize=export_cfg.get("quantize", False),
            dtype=export_cfg.get("quantization_dtype", "float32"),
        )
        exporter.export(model, output, input_shape=tuple(input_shape))
    elif fmt == "coreml":
        from flashedge.export.coreml_export import CoreMLExporter

        exporter = CoreMLExporter(
            compute_units=export_cfg.get("compute_units", "ALL"),
            minimum_deployment_target=export_cfg.get("minimum_deployment_target", "iOS16"),
        )
        exporter.export(model, output, input_shape=tuple(input_shape))
    elif fmt == "openvino":
        from flashedge.export.openvino_export import OpenVINOExporter

        exporter = OpenVINOExporter(
            compress_to_fp16=export_cfg.get("compress_to_fp16", True),
        )
        exporter.export(model, export_cfg.get("output_dir", output), input_shape=tuple(input_shape))
    else:
        print(f"  Error: Unsupported format '{fmt}'")
        sys.exit(1)

    print("  Export complete!")


def cmd_profile(args: argparse.Namespace) -> None:
    print(f"Profiling model: {args.model}")
    print(f"  Device: {args.device}")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    from flashedge.profiling.latency import LatencyProfiler

    profiler = LatencyProfiler(device=args.device)
    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    report = profiler.profile(str(model_path), input_shape=input_shape, warmup=args.warmup, runs=args.runs)

    print(f"\nLatency Report ({args.runs} runs):")
    print(f"  Mean:       {report['mean_ms']:.3f} ms")
    print(f"  Std:        {report['std_ms']:.3f} ms")
    print(f"  P50:        {report['p50_ms']:.3f} ms")
    print(f"  P95:        {report['p95_ms']:.3f} ms")
    print(f"  P99:        {report['p99_ms']:.3f} ms")
    print(f"  Throughput: {report['fps']:.1f} FPS")


def cmd_optimize(args: argparse.Namespace) -> None:
    import yaml

    print(f"Optimizing model with config: {args.config}")
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    opt_cfg = config.get("optimization", {})
    model_cfg = config.get("model", {})

    if opt_cfg.get("quantize"):
        print("  Running quantization...")
        from flashedge.optimization.quantization import INT8Quantizer

        quantizer = INT8Quantizer(per_channel=True)
        model_path = model_cfg.get("path", "")
        output = opt_cfg.get("output", "optimized_model.onnx")
        if model_path and Path(model_path).exists():
            quantizer.quantize_onnx(model_path, output)
            print(f"  Saved quantized model to: {output}")
        else:
            print(f"  Warning: Model not found at '{model_path}'")

    if opt_cfg.get("prune"):
        print("  Running pruning...")
        print("  Note: Use Python API for pruning — requires a live PyTorch model.")

    print("  Optimization complete!")


def cmd_benchmark(args: argparse.Namespace) -> None:
    print(f"Benchmarking model: {args.model}")
    print(f"  Device: {args.device}")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    from flashedge.profiling.latency import LatencyProfiler

    input_shape = tuple(int(x) for x in args.input_shape.split(","))
    profiler = LatencyProfiler(device=args.device)
    report = profiler.profile(str(model_path), input_shape=input_shape, warmup=args.warmup, runs=args.runs)

    file_size_mb = model_path.stat().st_size / (1024 * 1024)

    print(f"\nBenchmark Results ({args.runs} runs):")
    print(f"  Model size:  {file_size_mb:.2f} MB")
    print(f"  Mean latency: {report['mean_ms']:.3f} ms")
    print(f"  Std latency:  {report['std_ms']:.3f} ms")
    print(f"  Throughput:   {report['fps']:.1f} FPS")


def cmd_deploy(args: argparse.Namespace) -> None:
    print(f"Deploying model: {args.model}")
    print(f"  Target runtime: {args.target}")
    print(f"  Output dir:     {args.output}")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    from flashedge.solutions.edge_deployer import EdgeDeployer

    deployer = EdgeDeployer()
    deployer.deploy(str(model_path), target=args.target, output_dir=args.output)
    print("  Deployment complete!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flashedge",
        description="FlashEdge — Edge AI & TinyML toolkit for FlashVision",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("version", help="Show FlashEdge version")
    subparsers.add_parser("settings", help="Display settings and environment")
    subparsers.add_parser("check", help="Check installation and dependencies")

    p_export = subparsers.add_parser("export", help="Export a model to edge format")
    p_export.add_argument("--config", type=str, required=True, help="Path to export config YAML")

    p_profile = subparsers.add_parser("profile", help="Profile model latency")
    p_profile.add_argument("--model", type=str, required=True, help="Path to model file")
    p_profile.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p_profile.add_argument("--input-shape", type=str, default="1,3,224,224", help="Input shape (comma-separated)")
    p_profile.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p_profile.add_argument("--runs", type=int, default=100, help="Benchmark iterations")

    p_optimize = subparsers.add_parser("optimize", help="Optimize a model (quantize, prune)")
    p_optimize.add_argument("--config", type=str, required=True, help="Path to optimization config YAML")

    p_bench = subparsers.add_parser("benchmark", help="Benchmark a model")
    p_bench.add_argument("--model", type=str, required=True, help="Path to model file")
    p_bench.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p_bench.add_argument("--input-shape", type=str, default="1,3,224,224", help="Input shape (comma-separated)")
    p_bench.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    p_bench.add_argument("--runs", type=int, default=100, help="Benchmark iterations")

    p_deploy = subparsers.add_parser("deploy", help="Deploy model to edge device")
    p_deploy.add_argument("--model", type=str, required=True, help="Path to model file")
    p_deploy.add_argument(
        "--target", type=str, default="onnxruntime", choices=["onnxruntime", "tflite", "openvino", "coreml"]
    )
    p_deploy.add_argument("--output", type=str, default="deployed/", help="Output directory")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "export": cmd_export,
        "profile": cmd_profile,
        "optimize": cmd_optimize,
        "benchmark": cmd_benchmark,
        "deploy": cmd_deploy,
    }

    if args.command is None:
        print(BANNER)
        parser.print_help()
        return

    handler = commands.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
