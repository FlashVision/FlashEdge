"""I/O utility functions for model saving, loading, and downloading."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.request import urlretrieve

import torch
import torch.nn as nn


def save_model(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a model checkpoint with optional metadata.

    Args:
        model: PyTorch model to save.
        path: Path to save the checkpoint.
        metadata: Optional metadata dictionary.

    Returns:
        Path to the saved checkpoint.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model,
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }

    torch.save(checkpoint, str(output))
    return str(output)


def load_model(
    path: str,
    device: str = "cpu",
) -> nn.Module:
    """Load a model from a checkpoint file.

    Args:
        path: Path to the checkpoint file.
        device: Device to load the model onto.

    Returns:
        Loaded PyTorch model.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device).eval()

    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                data = checkpoint[key]
                if isinstance(data, nn.Module):
                    return data.to(device).eval()

    raise ValueError(f"Cannot extract model from checkpoint at {path}")


def download_url(
    url: str,
    output_dir: str = "downloads",
    filename: Optional[str] = None,
    checksum: Optional[str] = None,
) -> str:
    """Download a file from a URL.

    Args:
        url: URL to download.
        output_dir: Directory to save the file.
        filename: Optional filename (derived from URL if None).
        checksum: Optional SHA256 checksum for verification.

    Returns:
        Path to the downloaded file.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1].split("?")[0]

    filepath = output / filename

    if filepath.exists():
        if checksum and _verify_checksum(str(filepath), checksum):
            return str(filepath)
        elif checksum is None:
            return str(filepath)

    print(f"  Downloading: {url}")
    urlretrieve(url, str(filepath))

    if checksum and not _verify_checksum(str(filepath), checksum):
        filepath.unlink()
        raise ValueError(f"Checksum verification failed for {filename}")

    return str(filepath)


def _verify_checksum(filepath: str, expected: str) -> bool:
    """Verify SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected
