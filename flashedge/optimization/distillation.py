"""Knowledge distillation from large teacher to compact student models."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashedge.registry import OPTIMIZERS


@OPTIMIZERS.register("knowledge_distiller")
class KnowledgeDistiller:
    """Knowledge distillation transferring soft logits from teacher to student.

    Args:
        temperature: Softmax temperature for softening probability distributions.
        alpha: Weight for distillation loss vs. task loss (0–1).
        loss_type: Loss function type — "kl_div", "mse", or "cosine".
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        loss_type: str = "kl_div",
    ) -> None:
        self.temperature = temperature
        self.alpha = alpha
        self.loss_type = loss_type

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        task_criterion: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute the combined distillation + task loss.

        Args:
            student_logits: Raw logits from the student model.
            teacher_logits: Raw logits from the teacher model.
            targets: Ground-truth labels.
            task_criterion: Loss function for the task (default: CrossEntropyLoss).

        Returns:
            Combined loss tensor.
        """
        if task_criterion is None:
            task_criterion = nn.CrossEntropyLoss()

        task_loss = task_criterion(student_logits, targets)
        distill_loss = self._distillation_loss(student_logits, teacher_logits)

        combined = self.alpha * distill_loss + (1.0 - self.alpha) * task_loss
        return combined

    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss between student and teacher logits."""
        T = self.temperature

        if self.loss_type == "kl_div":
            student_soft = F.log_softmax(student_logits / T, dim=-1)
            teacher_soft = F.softmax(teacher_logits / T, dim=-1)
            loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        elif self.loss_type == "mse":
            student_soft = student_logits / T
            teacher_soft = teacher_logits / T
            loss = F.mse_loss(student_soft, teacher_soft) * (T * T)

        elif self.loss_type == "cosine":
            loss = 1.0 - F.cosine_similarity(student_logits, teacher_logits, dim=-1).mean()

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss

    def distill(
        self,
        teacher: nn.Module,
        student: nn.Module,
        train_loader: Any,
        epochs: int = 50,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> nn.Module:
        """Run the full knowledge distillation training loop.

        Args:
            teacher: The teacher model (frozen).
            student: The student model (trainable).
            train_loader: DataLoader providing training samples.
            epochs: Number of training epochs.
            lr: Learning rate.
            device: Device to train on.

        Returns:
            Trained student model.
        """
        teacher = teacher.to(device).eval()
        student = student.to(device).train()

        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                if isinstance(batch, (tuple, list)):
                    images, targets = batch[0].to(device), batch[1].to(device)
                else:
                    images = batch.to(device)
                    targets = torch.zeros(images.size(0), dtype=torch.long, device=device)

                with torch.no_grad():
                    teacher_logits = teacher(images)

                student_logits = student(images)
                loss = self.compute_loss(student_logits, teacher_logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(num_batches, 1)
            print(f"  Distillation Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

        student.eval()
        return student
