from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset

from ..evaluation.metrics import compute_metrics
from ..losses.business_spread import LossComponents
from ..losses.paper_mae import PaperMAE
from .checkpoint import save_checkpoint
from .early_stopping import EarlyStopping
from .reproducibility import seed_everything
from .scheduler import scheduled_learning_rate


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 0.0
    max_steps: int = 1200
    min_steps: int = 200
    eval_every: int = 25
    patience_checks: int = 8
    gradient_clip_norm: float = 1.0
    seed: int = 42
    schedule_total_steps: int = 1200
    nominal_lr_decay_steps: tuple[int, ...] = (300, 600, 900)
    lr_decay_gamma: float = 0.5


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in batch[0]:
        values = [x[key] for x in batch]
        out[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return out


class Trainer:
    """Deterministic daily-sample trainer with auditable training phases.

    The validation loader is used only for early stopping.  The scheduler uses
    ``schedule_total_steps`` rather than the shortened smoke-run length, and
    the model is restored to the best validation-MAE state before saving.
    """

    def __init__(self, model: torch.nn.Module, train_dataset: Dataset, val_dataset: Dataset, loss_fn: Callable[..., Any], output_dir: str | Path, config: TrainingConfig | None = None, device: str | torch.device = "cpu"):
        self.model, self.train_dataset, self.val_dataset, self.loss_fn = model, train_dataset, val_dataset, loss_fn
        self.output_dir, self.config, self.device = Path(output_dir), config or TrainingConfig(), torch.device(device)
        self.model.to(self.device)
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("trainer requires non-empty train and validation datasets")

    def _move(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval(); preds, targets, masks = [], [], []
        with torch.no_grad():
            for raw in loader:
                b = self._move(raw)
                preds.append(self.model(b["y_backcast"], b["x_backcast"], b["x_future"]).detach().cpu())
                targets.append(b["y_future"].cpu()); masks.append(b["score_mask"].cpu())
        self.model.train()
        return compute_metrics(torch.cat(preds), torch.cat(targets), torch.cat(masks))

    def fit(self) -> dict[str, Any]:
        seed_everything(self.config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        train_loader = DataLoader(self.train_dataset, batch_size=min(self.config.batch_size, len(self.train_dataset)), shuffle=True, collate_fn=_collate)
        val_loader = DataLoader(self.val_dataset, batch_size=min(self.config.batch_size, len(self.val_dataset)), shuffle=False, collate_fn=_collate)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        stopper = EarlyStopping(self.config.patience_checks)
        curve, grads = [], []
        best_state, best_step, best_val = copy.deepcopy(self.model.state_dict()), 0, float("inf")
        iterator = iter(train_loader); step = 0
        while step < self.config.max_steps:
            try: batch = next(iterator)
            except StopIteration: iterator = iter(train_loader); batch = next(iterator)
            step += 1; batch = self._move(batch)
            # Keep nominal 300/600/900 decay nodes fixed even for smoke or
            # convergence runs that intentionally stop before step 1200.
            lr = scheduled_learning_rate(self.config.learning_rate, step, self.config.schedule_total_steps, self.config.lr_decay_gamma, len(self.config.nominal_lr_decay_steps), self.config.nominal_lr_decay_steps)
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            pred = self.model(batch["y_backcast"], batch["x_backcast"], batch["x_future"])
            kwargs = {"pred": pred, "target": batch["y_future"], "mask": torch.ones_like(batch["y_future"]), "bridge_mask": batch["bridge_mask"], "score_mask": batch["score_mask"], "progress": step / self.config.max_steps}
            loss = self.loss_fn(**kwargs) if isinstance(self.loss_fn, PaperMAE) else self.loss_fn(**{k: kwargs[k] for k in ("pred", "target", "bridge_mask", "score_mask", "progress")})
            if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            pre = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            pre_value = float(pre.detach().cpu()); clipped = pre_value > self.config.gradient_clip_norm
            post = 0.0
            for p in self.model.parameters():
                if p.grad is not None: post += float(torch.sum(p.grad.detach() ** 2).cpu())
            post = post ** 0.5
            if not torch.isfinite(torch.tensor(post)): raise FloatingPointError(f"non-finite gradient at step {step}")
            optimizer.step()
            grads.append({"step": step, "lr": lr, "grad_norm_pre_clip": pre_value, "grad_norm_post_clip": post, "clipped": int(clipped)})
            if step % self.config.eval_every == 0 or step == self.config.min_steps:
                metrics = self._evaluate(val_loader)
                curve.append({"step": step, "train_loss": float(loss.detach().cpu()), "validation_mae": metrics["mae"], "direction_accuracy": metrics["direction_accuracy"], "positive_recall": metrics["positive_recall"], "negative_recall": metrics["negative_recall"], "balanced_accuracy": metrics["balanced_accuracy"]})
                if metrics["mae"] < best_val:
                    best_val, best_step, best_state = metrics["mae"], step, copy.deepcopy(self.model.state_dict())
                if step >= self.config.min_steps:
                    stopper.update(metrics["mae"])
                if step >= self.config.min_steps and stopper.should_stop: break
        self.model.load_state_dict(best_state)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in (("training_curve.csv", curve), ("gradient_stats.csv", grads)):
            if rows:
                with (self.output_dir / name).open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        save_checkpoint(self.output_dir / "checkpoint.pt", self.model, optimizer, best_step, {"best_validation_mae": best_val, "gradient_clip_fraction": sum(x["clipped"] for x in grads) / max(len(grads), 1)})
        clip_fraction = sum(x["clipped"] for x in grads) / max(len(grads), 1)
        return {"best_step": best_step, "best_validation_mae": best_val, "steps": step,
                "final_step": step, "early_stopped": step < self.config.max_steps,
                "gradient_clip_fraction": clip_fraction,
                "gradient_instability_warning": clip_fraction > 0.25,
                "training_curve": curve, "gradient_stats": grads}
