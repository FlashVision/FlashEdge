"""INT8/INT4 quantization for edge deployment.

Supports post-training quantization (PTQ), dynamic quantization, and
quantization-aware training (QAT) using PyTorch's quantization API.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from flashedge.registry import OPTIMIZERS


@OPTIMIZERS.register("int8_quantizer")
class INT8Quantizer:
    """Post-Training INT8 quantization using PyTorch's static quantization.

    Args:
        per_channel: Use per-channel quantization for better accuracy.
        symmetric: Use symmetric quantization.
        calibration_samples: Number of calibration samples to process.
        backend: Quantization backend (fbgemm for x86, qnnpack for ARM).
    """

    def __init__(
        self,
        per_channel: bool = True,
        symmetric: bool = True,
        calibration_samples: int = 500,
        backend: str = "fbgemm",
    ) -> None:
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.calibration_samples = calibration_samples
        self.backend = backend

    def quantize(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> nn.Module:
        """Quantize a PyTorch model to INT8 using static quantization.

        Args:
            model: The PyTorch model to quantize.
            calibration_loader: DataLoader providing calibration samples.
            input_shape: Shape of the input tensor (used for synthetic calibration).

        Returns:
            Quantized PyTorch model.
        """
        torch.backends.quantized.engine = self.backend

        model = copy.deepcopy(model).cpu().eval()

        if self.per_channel:
            qconfig = torch.ao.quantization.get_default_qconfig(self.backend)
        else:
            observer = (
                torch.ao.quantization.MinMaxObserver
                if self.symmetric
                else torch.ao.quantization.MovingAverageMinMaxObserver
            )
            qconfig = torch.ao.quantization.QConfig(
                activation=observer.with_args(dtype=torch.quint8),
                weight=torch.ao.quantization.default_weight_observer,
            )

        model.qconfig = qconfig
        torch.ao.quantization.prepare(model, inplace=True)

        with torch.no_grad():
            if calibration_loader is not None:
                count = 0
                for batch in calibration_loader:
                    if isinstance(batch, (tuple, list)):
                        batch = batch[0]
                    model(batch.cpu())
                    count += len(batch)
                    if count >= self.calibration_samples:
                        break
            else:
                for _ in range(min(self.calibration_samples, 100)):
                    dummy = torch.randn(*input_shape)
                    model(dummy)

        torch.ao.quantization.convert(model, inplace=True)

        return model

    def quantize_onnx(
        self,
        onnx_path: str,
        output_path: str,
        calibration_loader: Optional[Any] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> str:
        """Quantize an ONNX model to INT8 using ONNX Runtime quantization.

        Args:
            onnx_path: Path to the input ONNX model.
            output_path: Path to save the quantized ONNX model.
            calibration_loader: Optional calibration data loader.
            input_shape: Shape for synthetic calibration data.

        Returns:
            Path to the quantized model.
        """
        from onnxruntime.quantization import quantize_static, quantize_dynamic, QuantType

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if calibration_loader is not None:
            reader = _OnnxCalibrationReader(calibration_loader, input_shape, self.calibration_samples)
            quantize_static(
                onnx_path,
                output_path,
                reader,
                quant_format=None,
                per_channel=self.per_channel,
                weight_type=QuantType.QInt8,
            )
        else:
            quantize_dynamic(
                onnx_path,
                output_path,
                weight_type=QuantType.QInt8,
            )

        size_before = Path(onnx_path).stat().st_size / (1024 * 1024)
        size_after = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  Quantized: {size_before:.2f} MB → {size_after:.2f} MB ({size_after / size_before:.1%})")

        return output_path


class _OnnxCalibrationReader:
    """Calibration data reader for ONNX Runtime static quantization."""

    def __init__(self, loader: Any, input_shape: Tuple[int, ...], max_samples: int) -> None:
        self.loader = loader
        self.input_shape = input_shape
        self.max_samples = max_samples
        self._iter = None
        self._count = 0

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._iter is None:
            self._iter = iter(self.loader)

        if self._count >= self.max_samples:
            return None

        try:
            batch = next(self._iter)
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            self._count += len(batch)
            return {"input": batch.numpy().astype(np.float32)}
        except StopIteration:
            return None


@OPTIMIZERS.register("dynamic_quantizer")
class DynamicQuantizer:
    """Dynamic quantization — quantizes weights statically, activations dynamically at runtime.

    Suitable for models with variable-length inputs (NLP, sequence models).
    """

    def __init__(self, dtype: torch.dtype = torch.qint8) -> None:
        self.dtype = dtype

    def quantize(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to a model.

        Args:
            model: The PyTorch model to quantize.

        Returns:
            Dynamically quantized model.
        """
        model = copy.deepcopy(model).cpu().eval()

        quantized = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU},
            dtype=self.dtype,
        )

        return quantized


@OPTIMIZERS.register("qat_trainer")
class QATTrainer:
    """Quantization-Aware Training (QAT) — inserts fake quantization during training.

    Args:
        backend: Quantization backend (fbgemm, qnnpack).
        epochs: Number of fine-tuning epochs.
        lr: Learning rate for QAT fine-tuning.
    """

    def __init__(
        self,
        backend: str = "fbgemm",
        epochs: int = 10,
        lr: float = 1e-4,
    ) -> None:
        self.backend = backend
        self.epochs = epochs
        self.lr = lr

    def train(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
    ) -> nn.Module:
        """Run quantization-aware training.

        Args:
            model: The model to fine-tune with QAT.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            criterion: Loss function (default: CrossEntropyLoss).

        Returns:
            Quantized model after QAT.
        """
        torch.backends.quantized.engine = self.backend
        model = copy.deepcopy(model).cpu().train()

        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(self.backend)
        torch.ao.quantization.prepare_qat(model, inplace=True)

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    images, targets = batch[0], batch[1]
                else:
                    images = batch
                    targets = torch.zeros(images.size(0), dtype=torch.long)

                optimizer.zero_grad()
                outputs = model(images.cpu())
                loss = criterion(outputs, targets.cpu())
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            print(f"  QAT Epoch {epoch + 1}/{self.epochs} — Loss: {avg_loss:.4f}")

        model.eval()
        torch.ao.quantization.convert(model, inplace=True)

        return model
