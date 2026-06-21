"""Visualization utilities for benchmark results and profiling data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def plot_latency_comparison(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Latency Comparison",
) -> Optional[str]:
    """Plot latency comparison bar chart across models.

    Args:
        results: Benchmark results from Benchmark.compare().
        output_path: Path to save the plot. If None, displays interactively.
        title: Plot title.

    Returns:
        Path to the saved plot, or None if displayed interactively.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Visualization requires matplotlib: pip install flashedge[analytics]")
        return None

    names = list(results.keys())
    means = [results[n]["mean_ms"] for n in names]
    stds = [results[n].get("std_ms", 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color="#4C78A8", edgecolor="white")

    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{mean:.2f}", ha="center", fontsize=10)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def plot_size_comparison(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    title: str = "Model Size Comparison",
) -> Optional[str]:
    """Plot model size comparison bar chart.

    Args:
        results: Benchmark results with file_size_mb field.
        output_path: Path to save the plot.
        title: Plot title.

    Returns:
        Path to the saved plot.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Visualization requires matplotlib: pip install flashedge[analytics]")
        return None

    names = list(results.keys())
    sizes = [results[n].get("file_size_mb", 0) for n in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, sizes, color="#E45756", edgecolor="white")

    ax.set_xlabel("Size (MB)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2, f"{size:.2f} MB", va="center", fontsize=10)

    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return output_path
    else:
        plt.show()
        return None
