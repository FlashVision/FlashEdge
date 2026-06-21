"""PyTorch to CoreML conversion for iOS/macOS deployment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from flashedge.registry import EXPORTERS


@EXPORTERS.register("coreml")
class CoreMLExporter:
    """Export PyTorch models to CoreML format for Apple platforms.

    Args:
        compute_units: CoreML compute units (CPU_ONLY, CPU_AND_GPU, CPU_AND_NE, ALL).
        minimum_deployment_target: Minimum iOS/macOS version (e.g., iOS16).
        convert_to: Target format (mlprogram or neuralnetwork).
        classifier: Whether to export as a classifier model.
    """

    def __init__(
        self,
        compute_units: str = "ALL",
        minimum_deployment_target: str = "iOS16",
        convert_to: str = "mlprogram",
        classifier: bool = False,
    ) -> None:
        self.compute_units = compute_units
        self.minimum_deployment_target = minimum_deployment_target
        self.convert_to = convert_to
        self.classifier = classifier

    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        class_labels: Optional[List[str]] = None,
    ) -> str:
        """Export a PyTorch model to CoreML format.

        Args:
            model: The PyTorch model to export.
            output_path: Path to save the CoreML model (.mlpackage or .mlmodel).
            input_shape: Shape of the input tensor.
            class_labels: Class labels for classifier models.

        Returns:
            Path to the exported CoreML model.
        """
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError("CoreML export requires coremltools: pip install flashedge[coreml]")

        model = model.cpu().eval()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(*input_shape)
        traced_model = torch.jit.trace(model, dummy_input)

        compute_unit_map = {
            "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
            "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
            "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
            "ALL": ct.ComputeUnit.ALL,
        }
        compute_unit = compute_unit_map.get(self.compute_units, ct.ComputeUnit.ALL)

        ct_input = ct.ImageType(
            name="input",
            shape=input_shape,
            scale=1.0 / 255.0,
            bias=[0, 0, 0],
        )

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct_input],
            convert_to=self.convert_to,
            compute_units=compute_unit,
            minimum_deployment_target=getattr(ct.target, self.minimum_deployment_target, None),
        )

        if self.classifier and class_labels:
            mlmodel = self._add_classifier_config(mlmodel, class_labels)

        mlmodel.save(str(output_path))

        print(f"  CoreML model saved: {output_path}")
        return str(output_path)

    def _add_classifier_config(self, mlmodel: Any, class_labels: List[str]) -> Any:
        """Add classifier configuration to a CoreML model."""
        try:

            spec = mlmodel.get_spec()
            classifier_config = spec.description.predictedFeatureName
            if not classifier_config:
                spec.description.predictedFeatureName = "classLabel"

            return mlmodel
        except Exception:
            return mlmodel

    def get_model_info(self, coreml_path: str) -> Dict[str, Any]:
        """Get metadata about a CoreML model."""
        try:
            import coremltools as ct

            model = ct.models.MLModel(coreml_path)
            spec = model.get_spec()

            return {
                "description": spec.description.metadata.shortDescription,
                "num_inputs": len(spec.description.input),
                "num_outputs": len(spec.description.output),
                "compute_units": str(model.compute_unit),
            }
        except ImportError:
            return {"error": "coremltools not installed"}
