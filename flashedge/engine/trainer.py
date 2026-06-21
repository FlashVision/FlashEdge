"""Distillation and fine-tuning trainer for edge model optimization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    """Trainer for fine-tuning and knowledge distillation for edge models.

    Supports standard fine-tuning and teacher-student distillation with
    cosine annealing, warmup, and automatic mixed precision.

    Args:
        epochs: Number of training epochs.
        lr: Learning rate.
        optimizer_name: Optimizer name (adam, sgd, adamw).
        weight_decay: Weight decay.
        warmup_epochs: Number of warmup epochs.
        device: Training device.
        use_amp: Whether to use automatic mixed precision.
    """

    def __init__(
        self,
        epochs: int = 50,
        lr: float = 1e-3,
        optimizer_name: str = "adam",
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        device: str = "cpu",
        use_amp: bool = False,
    ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()

    def train(
        self,
        model: nn.Module,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        criterion: Optional[nn.Module] = None,
        teacher: Optional[nn.Module] = None,
        distiller: Optional[Any] = None,
        save_dir: str = "checkpoints",
    ) -> Dict[str, Any]:
        """Run training loop.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            criterion: Loss function (default: CrossEntropyLoss).
            teacher: Optional teacher model for distillation.
            distiller: Optional KnowledgeDistiller instance.
            save_dir: Directory to save checkpoints.

        Returns:
            Dictionary with training history.
        """
        model = model.to(self.device).train()
        criterion = criterion or nn.CrossEntropyLoss()

        if teacher is not None:
            teacher = teacher.to(self.device).eval()

        optimizer = self._build_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        history: Dict[str, list] = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_loss = float("inf")

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            lr = self._get_lr(optimizer, epoch)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False)
            for batch in pbar:
                if isinstance(batch, (tuple, list)):
                    images, targets = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    images = batch.to(self.device)
                    targets = torch.zeros(images.size(0), dtype=torch.long, device=self.device)

                optimizer.zero_grad()

                if self.use_amp and scaler is not None:
                    with torch.amp.autocast("cuda"):
                        outputs = model(images)
                        if teacher is not None and distiller is not None:
                            with torch.no_grad():
                                teacher_out = teacher(images)
                            loss = distiller.compute_loss(outputs, teacher_out, targets, criterion)
                        else:
                            loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(images)
                    if teacher is not None and distiller is not None:
                        with torch.no_grad():
                            teacher_out = teacher(images)
                        loss = distiller.compute_loss(outputs, teacher_out, targets, criterion)
                    else:
                        loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.6f}")

            scheduler.step()

            avg_train_loss = total_loss / max(num_batches, 1)
            history["train_loss"].append(avg_train_loss)

            if val_loader is not None:
                val_loss, val_acc = self._validate(model, val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                print(
                    f"Epoch {epoch + 1}/{self.epochs} — "
                    f"Train loss: {avg_train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f}"
                )

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), save_path / "best.pth")
            else:
                print(f"Epoch {epoch + 1}/{self.epochs} — Train loss: {avg_train_loss:.4f}")

        torch.save(model.state_dict(), save_path / "last.pth")
        return history

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        params = [p for p in model.parameters() if p.requires_grad]
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)

    def _get_lr(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr * warmup_factor
        return optimizer.param_groups[0]["lr"]

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        val_loader: Any,
        criterion: nn.Module,
    ) -> tuple:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in val_loader:
            if isinstance(batch, (tuple, list)):
                images, targets = batch[0].to(self.device), batch[1].to(self.device)
            else:
                continue

            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        avg_loss = total_loss / max(len(val_loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy
