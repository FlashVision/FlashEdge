"""Benchmarking framework for comparing models across formats and optimizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple


class Benchmark:
    """Benchmark and compare models across formats, quantization levels, and devices.

    Provides methods to run latency benchmarks, compare model sizes,
    and generate formatted comparison reports.
    """

    def compare(
        self,
        models: Dict[str, str],
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        warmup: int = 10,
        runs: int = 100,
        device: str = "cpu",
    ) -> Dict[str, Dict[str, Any]]:
        """Compare multiple models on latency and size.

        Args:
            models: Dictionary mapping names to model file paths.
            input_shape: Shape of the input tensor.
            warmup: Warmup iterations.
            runs: Benchmark iterations.
            device: Target device.

        Returns:
            Dictionary mapping model names to benchmark results.
        """
        from flashedge.profiling.latency import LatencyProfiler

        profiler = LatencyProfiler(device=device)
        results: Dict[str, Dict[str, Any]] = {}

        for name, model_path in models.items():
            path = Path(model_path)
            if not path.exists():
                print(f"  Warning: {model_path} not found, skipping")
                continue

            latency = profiler.profile(model_path, input_shape, warmup, runs)
            file_size_mb = path.stat().st_size / (1024 * 1024)

            results[name] = {
                "file_path": model_path,
                "file_size_mb": file_size_mb,
                **latency,
            }

        return results

    def print_report(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print a formatted benchmark comparison report.

        Args:
            results: Results from the compare() method.
        """
        if not results:
            print("  No results to display.")
            return

        print("\n" + "=" * 90)
        print("  FlashEdge Benchmark Report")
        print("=" * 90)
        print(f"  {'Model':<20} {'Size (MB)':<12} {'Mean (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)':<12} {'FPS':<10}")
        print("-" * 90)

        for name, metrics in results.items():
            print(
                f"  {name:<20} {metrics['file_size_mb']:<12.2f} {metrics['mean_ms']:<12.3f} "
                f"{metrics.get('p95_ms', 0):<12.3f} {metrics.get('p99_ms', 0):<12.3f} "
                f"{metrics['fps']:<10.1f}"
            )

        print("=" * 90)

        if len(results) >= 2:
            names = list(results.keys())
            baseline = results[names[0]]
            for name in names[1:]:
                other = results[name]
                size_ratio = other["file_size_mb"] / baseline["file_size_mb"]
                speed_ratio = baseline["mean_ms"] / other["mean_ms"]
                print(f"\n  {name} vs {names[0]}:")
                print(
                    f"    Size:    {size_ratio:.2f}x ({'-' if size_ratio < 1 else '+'}{abs(1 - size_ratio) * 100:.1f}%)"
                )
                print(f"    Speed:   {speed_ratio:.2f}x {'faster' if speed_ratio > 1 else 'slower'}")

    def to_csv(self, results: Dict[str, Dict[str, Any]], output_path: str) -> str:
        """Export benchmark results to CSV.

        Args:
            results: Results from the compare() method.
            output_path: Path to save the CSV file.

        Returns:
            Path to the saved CSV file.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        headers = ["model", "file_size_mb", "mean_ms", "std_ms", "p50_ms", "p95_ms", "p99_ms", "fps"]
        lines = [",".join(headers)]

        for name, metrics in results.items():
            values = [
                name,
                f"{metrics.get('file_size_mb', 0):.2f}",
                f"{metrics.get('mean_ms', 0):.3f}",
                f"{metrics.get('std_ms', 0):.3f}",
                f"{metrics.get('p50_ms', 0):.3f}",
                f"{metrics.get('p95_ms', 0):.3f}",
                f"{metrics.get('p99_ms', 0):.3f}",
                f"{metrics.get('fps', 0):.1f}",
            ]
            lines.append(",".join(values))

        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

        return str(path)
