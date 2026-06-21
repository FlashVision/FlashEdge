"""INT4 weight quantization for aggressive edge model compression.

Implements GPTQ-style Hessian-guided 4-bit quantization with group-wise
scaling, as well as a simpler round-to-nearest (RTN) fallback.  Packed
representations store two 4-bit values per byte to halve weight memory.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashedge.registry import OPTIMIZERS

_INT4_MIN = -8
_INT4_MAX = 7
_UINT4_MAX = 15


@OPTIMIZERS.register("int4_quantizer")
class INT4Quantizer:
    """GPTQ-style 4-bit weight quantizer for edge deployment.

    Weights are quantised to 4-bit integers in groups of ``group_size``
    columns.  Each group stores its own floating-point scale (and optional
    zero-point), keeping accuracy loss manageable while achieving ~8× weight
    compression.

    Args:
        method: Quantisation algorithm — ``"gptq"`` (Hessian-guided) or
            ``"rtn"`` (simple round-to-nearest).
        group_size: Number of weight elements per quantisation group.
        symmetric: Use symmetric quantisation (zero-point fixed at 0).
        block_size: Internal block size for the GPTQ Hessian solve.
    """

    def __init__(
        self,
        method: str = "gptq",
        group_size: int = 128,
        symmetric: bool = True,
        block_size: int = 64,
    ) -> None:
        if method not in ("gptq", "rtn"):
            raise ValueError(f"method must be 'gptq' or 'rtn', got '{method}'")
        self.method = method
        self.group_size = group_size
        self.symmetric = symmetric
        self.block_size = block_size

    def quantize(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any] = None,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    ) -> nn.Module:
        """Quantize model weights to INT4.

        Args:
            model: The PyTorch model to quantize.
            calibration_loader: DataLoader for Hessian estimation (GPTQ).
                Ignored when ``method="rtn"``.
            input_shape: Shape for synthetic calibration data.

        Returns:
            A deep copy of the model with INT4-quantised weights
            (dequantised back to FP32 for standard inference).
        """
        model = copy.deepcopy(model).cpu().eval()
        hessians: Dict[str, torch.Tensor] = {}

        if self.method == "gptq":
            hessians = self._collect_hessians(model, calibration_loader, input_shape)

        quantized_count = 0
        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue

            weight = module.weight.data
            original_shape = weight.shape
            flat = weight.view(weight.shape[0], -1)

            original_cols = flat.shape[1]
            if self.method == "gptq" and name in hessians:
                packed, scales, zeros = self._quantize_weight_gptq(flat, hessians[name])
            else:
                packed, scales, zeros = self._quantize_weight_rtn(flat)

            dequantized = self._dequantize(packed, scales, zeros, original_cols=original_cols)
            module.weight.data = dequantized.view(original_shape)
            quantized_count += 1

        print(f"  INT4 quantization complete: {quantized_count} layers quantized (method={self.method})")
        return model

    # ------------------------------------------------------------------
    # Hessian collection
    # ------------------------------------------------------------------

    def _collect_hessians(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any],
        input_shape: Tuple[int, ...],
    ) -> Dict[str, torch.Tensor]:
        """Accumulate diagonal Hessian estimates from calibration data.

        Args:
            model: The model under calibration.
            calibration_loader: Optional data loader.
            input_shape: Shape for synthetic calibration.

        Returns:
            Mapping of layer name → diagonal Hessian tensor.
        """
        hessians: Dict[str, torch.Tensor] = {}
        hooks: List[Any] = []

        def _make_hook(layer_name: str):
            def hook_fn(module: nn.Module, inp: Any, out: Any) -> None:
                x = inp[0].detach()
                if x.dim() > 2:
                    x = x.view(x.shape[0], -1)
                if layer_name not in hessians:
                    hessians[layer_name] = torch.zeros(x.shape[1], device=x.device)
                hessians[layer_name] += (x ** 2).sum(dim=0)
            return hook_fn

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(_make_hook(name)))

        print("  Collecting Hessian estimates...")
        with torch.no_grad():
            if calibration_loader is not None:
                for i, batch in enumerate(calibration_loader):
                    if isinstance(batch, (tuple, list)):
                        batch = batch[0]
                    model(batch.cpu())
                    if i >= 31:
                        break
            else:
                for _ in range(32):
                    model(torch.randn(*input_shape))

        for h in hooks:
            h.remove()

        for key in hessians:
            hessians[key] = hessians[key].clamp(min=1e-8)

        return hessians

    # ------------------------------------------------------------------
    # Core quantization kernels
    # ------------------------------------------------------------------

    def _quantize_weight_gptq(
        self,
        weight: torch.Tensor,
        hessian: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPTQ-style Hessian-guided INT4 weight quantisation.

        Args:
            weight: 2-D weight matrix ``(out_features, in_features)``.
            hessian: Diagonal Hessian vector of length ``in_features``.

        Returns:
            Tuple of ``(packed_int4, scales, zeros)`` tensors.
        """
        rows, cols = weight.shape
        gs = min(self.group_size, cols)
        num_groups = math.ceil(cols / gs)

        scales = torch.zeros(rows, num_groups)
        zeros = torch.zeros(rows, num_groups)
        quantized = torch.zeros_like(weight, dtype=torch.int8)

        hessian_diag = hessian[:cols].float()
        if hessian_diag.numel() < cols:
            hessian_diag = torch.cat([hessian_diag, torch.ones(cols - hessian_diag.numel())])

        for g in range(num_groups):
            start = g * gs
            end = min(start + gs, cols)
            w_group = weight[:, start:end].float()
            h_group = hessian_diag[start:end]

            if self.symmetric:
                abs_max = w_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
                scale = abs_max / _INT4_MAX
                zero = torch.zeros(rows, 1)
            else:
                w_min = w_group.amin(dim=1, keepdim=True)
                w_max = w_group.amax(dim=1, keepdim=True)
                scale = ((w_max - w_min) / _UINT4_MAX).clamp(min=1e-8)
                zero = w_min

            h_weight = (h_group / h_group.sum()).unsqueeze(0)
            q = ((w_group - zero) / scale).round()

            if self.symmetric:
                q = q.clamp(_INT4_MIN, _INT4_MAX)
            else:
                q = q.clamp(0, _UINT4_MAX)

            w_hat = q * scale + zero
            error = (w_group - w_hat) * h_weight
            correction = (error / scale).round()
            q = q + correction
            if self.symmetric:
                q = q.clamp(_INT4_MIN, _INT4_MAX)
            else:
                q = q.clamp(0, _UINT4_MAX)

            quantized[:, start:end] = q.to(torch.int8)
            scales[:, g] = scale.squeeze(1)
            zeros[:, g] = zero.squeeze(1)

        packed = self._pack_int4(quantized)
        return packed, scales, zeros

    def _quantize_weight_rtn(
        self,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Simple round-to-nearest INT4 quantisation.

        Args:
            weight: 2-D weight matrix.

        Returns:
            Tuple of ``(packed_int4, scales, zeros)``.
        """
        rows, cols = weight.shape
        gs = min(self.group_size, cols)
        num_groups = math.ceil(cols / gs)

        scales = torch.zeros(rows, num_groups)
        zeros = torch.zeros(rows, num_groups)
        quantized = torch.zeros_like(weight, dtype=torch.int8)

        for g in range(num_groups):
            start = g * gs
            end = min(start + gs, cols)
            w_group = weight[:, start:end].float()

            if self.symmetric:
                abs_max = w_group.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
                scale = abs_max / _INT4_MAX
                zero = torch.zeros(rows, 1)
                q = (w_group / scale).round().clamp(_INT4_MIN, _INT4_MAX)
            else:
                w_min = w_group.amin(dim=1, keepdim=True)
                w_max = w_group.amax(dim=1, keepdim=True)
                scale = ((w_max - w_min) / _UINT4_MAX).clamp(min=1e-8)
                zero = w_min
                q = ((w_group - zero) / scale).round().clamp(0, _UINT4_MAX)

            quantized[:, start:end] = q.to(torch.int8)
            scales[:, g] = scale.squeeze(1)
            zeros[:, g] = zero.squeeze(1)

        packed = self._pack_int4(quantized)
        return packed, scales, zeros

    # ------------------------------------------------------------------
    # Packing / unpacking
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_int4(values: torch.Tensor) -> torch.Tensor:
        """Pack two INT4 values into a single byte.

        Args:
            values: Tensor of INT4 values (dtype int8, values in ``[-8, 15]``).

        Returns:
            Packed tensor with half the columns (each byte holds two values).
        """
        rows, cols = values.shape
        padded_cols = cols + (cols % 2)
        padded = torch.zeros(rows, padded_cols, dtype=torch.int8)
        padded[:, :cols] = values
        low = padded[:, 0::2] & 0x0F
        high = (padded[:, 1::2] & 0x0F) << 4
        return (low | high).to(torch.uint8)

    @staticmethod
    def _unpack_int4(packed: torch.Tensor, original_cols: int | None = None) -> torch.Tensor:
        """Unpack bytes back into individual INT4 values.

        Args:
            packed: Packed byte tensor.
            original_cols: Original number of columns (for trimming padding).

        Returns:
            Unpacked INT4 tensor.
        """
        low = (packed & 0x0F).to(torch.int8)
        high = ((packed >> 4) & 0x0F).to(torch.int8)
        interleaved = torch.stack([low, high], dim=-1).view(packed.shape[0], -1)
        if original_cols is not None:
            interleaved = interleaved[:, :original_cols]
        return interleaved

    def _dequantize(
        self,
        packed: torch.Tensor,
        scales: torch.Tensor,
        zeros: torch.Tensor,
        original_cols: int | None = None,
    ) -> torch.Tensor:
        """Dequantize packed INT4 weights back to floating-point.

        Args:
            packed: Packed INT4 weight tensor.
            scales: Per-group scale factors.
            zeros: Per-group zero-points.
            original_cols: Original number of columns before packing (for trimming padding).

        Returns:
            Dequantized FP32 weight tensor.
        """
        rows = packed.shape[0]
        num_groups = scales.shape[1]
        gs = self.group_size

        total_cols = packed.shape[1] * 2
        quantized = self._unpack_int4(packed, total_cols)
        out_cols = original_cols if original_cols is not None else total_cols

        result = torch.zeros(rows, out_cols, dtype=torch.float32)
        for g in range(num_groups):
            start = g * gs
            end = min(start + gs, out_cols)
            q = quantized[:, start:end].float()
            result[:, start:end] = q * scales[:, g:g + 1] + zeros[:, g:g + 1]

        return result

    # ------------------------------------------------------------------
    # ONNX quantization
    # ------------------------------------------------------------------

    def quantize_onnx(self, onnx_path: str, output_path: str) -> str:
        """Apply INT4 quantization to an ONNX model.

        Args:
            onnx_path: Path to the input ONNX model.
            output_path: Path to save the quantized ONNX model.

        Returns:
            Path to the quantized model.
        """
        import onnx
        from onnx import numpy_helper
        import numpy as np

        model = onnx.load(onnx_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        quantized_count = 0
        for initializer in model.graph.initializer:
            w = numpy_helper.to_array(initializer)
            if w.ndim < 2 or w.size < self.group_size:
                continue

            original_shape = w.shape
            flat = w.reshape(w.shape[0], -1)
            tensor = torch.from_numpy(flat.astype(np.float32))
            _, scales, zeros = self._quantize_weight_rtn(tensor)
            dequantized = self._dequantize(
                self._quantize_weight_rtn(tensor)[0], scales, zeros
            ).numpy().reshape(original_shape)

            new_init = numpy_helper.from_array(dequantized.astype(np.float32), name=initializer.name)
            initializer.CopyFrom(new_init)
            quantized_count += 1

        onnx.save(model, output_path)

        size_before = Path(onnx_path).stat().st_size / (1024 * 1024)
        size_after = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"  ONNX INT4: {quantized_count} tensors quantised, {size_before:.2f} → {size_after:.2f} MB")
        return output_path

    # ------------------------------------------------------------------
    # Compression estimate
    # ------------------------------------------------------------------

    def estimate_compression(self, model: nn.Module) -> Dict[str, Any]:
        """Estimate compression ratio achievable with INT4 quantisation.

        Args:
            model: The PyTorch model to analyse.

        Returns:
            Dictionary with original and estimated INT4 sizes plus
            compression ratio.
        """
        original_bytes = 0
        int4_bytes = 0
        layer_details: List[Dict[str, Any]] = []

        for name, module in model.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            w = module.weight.data
            numel = w.numel()
            orig = numel * w.element_size()
            num_groups = math.ceil(numel / self.group_size)
            packed = math.ceil(numel / 2)
            meta = num_groups * w.shape[0] * 4 * 2
            compressed = packed + meta

            original_bytes += orig
            int4_bytes += compressed
            layer_details.append({
                "name": name,
                "original_kb": orig / 1024,
                "int4_kb": compressed / 1024,
                "ratio": orig / max(compressed, 1),
            })

        return {
            "original_size_mb": original_bytes / (1024 * 1024),
            "int4_size_mb": int4_bytes / (1024 * 1024),
            "compression_ratio": original_bytes / max(int4_bytes, 1),
            "layers": layer_details,
        }
