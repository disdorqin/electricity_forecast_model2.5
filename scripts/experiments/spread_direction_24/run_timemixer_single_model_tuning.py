"""Single-model TimeMixer tuning lab for causal hourly spread forecasting.

Experiment-only runner.  It deliberately keeps the existing safe proxy input
fixed while testing optimization health, capacity and direction-aware losses.
It never writes production ledgers/models or calls the fusion pipeline.

Stages supported by this runner:
- ``micro_overfit``: fit a small historical block and score the same block;
- ``walk_forward``: retrain for each target day using only earlier days.

The direction loss is computed against the *raw-spread zero threshold mapped
into each slot's StandardScaler space*.  Using sign(pred_scaled) directly is
incorrect because every output slot has its own fitted mean/scale.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.run_lear_spread_experiment import (  # noqa: E402
    _daily_map,
    prepare_source_cache,
)
from scripts.experiments.spread_direction_24.run_timemixer_structure_experiment import (  # noqa: E402
    PAST_BASE_COLS,
    _date_list,
    _device,
    _metrics,
    _prepare_feature_cache,
)
from TimeMixer.backbones import TimeMixerBackbone  # noqa: E402
from utils.resolution import HOURLY  # noqa: E402


LOSS_NAMES = ("mae", "huber", "directional", "multitask", "directional_multitask")
INPUT_MODES = ("proxy_mask", "pure_mask", "blended_proxy")
CHECKPOINT_METRICS = ("loss", "reg_balanced", "aux_balanced")


class DailyDataset(Dataset):
    def __init__(self, past: np.ndarray, future: np.ndarray, y: np.ndarray):
        self.past = torch.as_tensor(past, dtype=torch.float32)
        self.future = torch.as_tensor(future, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        return self.past[idx], self.future[idx], self.y[idx]


class DualHeadTimeMixer(nn.Module):
    """TimeMixer with a continuous spread head and an auxiliary direction head."""

    def __init__(self, past_dim: int, future_dim: int, args):
        super().__init__()
        self.backbone = TimeMixerBackbone(
            past_dim=past_dim,
            future_dim=future_dim,
            pred_len=HOURLY.slots_per_day,
            hidden_dim=args.hidden_dim,
            n_blocks=args.blocks,
            scales=args.scales,
            dropout=args.dropout,
        )
        z_dim = args.hidden_dim * (args.scales + 1)
        self.regression_head = self.backbone.head
        self.backbone.head = nn.Identity()
        self.direction_head = nn.Sequential(
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, args.hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, HOURLY.slots_per_day),
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor):
        z = self.backbone(past, future)
        return self.regression_head(z), self.direction_head(z)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _fit_scalers(past: np.ndarray, future: np.ndarray, y: np.ndarray, train_idx: np.ndarray):
    past_scaler = StandardScaler().fit(past[train_idx].reshape(-1, past.shape[-1]))
    future_scaler = StandardScaler().fit(future[train_idx].reshape(-1, future.shape[-1]))
    y_scaler = StandardScaler().fit(y[train_idx])
    return past_scaler, future_scaler, y_scaler


def _transform(past_scaler, future_scaler, y_scaler, past, future, y=None):
    past_t = past_scaler.transform(past.reshape(-1, past.shape[-1])).reshape(past.shape)
    future_t = future_scaler.transform(future.reshape(-1, future.shape[-1])).reshape(future.shape)
    y_t = y_scaler.transform(y) if y is not None else None
    return past_t, future_t, y_t


def _class_weights(y_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = (y_raw > 0).sum(axis=0).astype(float)
    neg = (y_raw < 0).sum(axis=0).astype(float)
    n = np.maximum(pos + neg, 1.0)
    pos_w = np.where(pos > 0, n / (2.0 * pos), 1.0)
    neg_w = np.where(neg > 0, n / (2.0 * neg), 1.0)
    return np.clip(pos_w, 0.25, 4.0), np.clip(neg_w, 0.25, 4.0)


def _direction_loss_from_regression(
    pred_scaled: torch.Tensor,
    y_scaled: torch.Tensor,
    zero_scaled: torch.Tensor,
    pos_w: torch.Tensor,
    neg_w: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    target_sign = torch.sign(y_scaled - zero_scaled)
    eligible = target_sign.ne(0)
    margin = (pred_scaled - zero_scaled) / max(float(temperature), 1e-4)
    loss = F.softplus(-target_sign * margin)
    weights = torch.where(target_sign > 0, pos_w, neg_w)
    weighted = loss * weights * eligible
    return weighted.sum() / eligible.sum().clamp_min(1)


def _aux_direction_loss(
    logits: torch.Tensor,
    y_scaled: torch.Tensor,
    zero_scaled: torch.Tensor,
    pos_w: torch.Tensor,
    neg_w: torch.Tensor,
) -> torch.Tensor:
    target_sign = torch.sign(y_scaled - zero_scaled)
    eligible = target_sign.ne(0)
    target = target_sign.gt(0).float()
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = torch.where(target_sign > 0, pos_w, neg_w)
    weighted = loss * weights * eligible
    return weighted.sum() / eligible.sum().clamp_min(1)


def _loss_parts(pred, yb, logits, *, loss_name, zero_scaled, pos_w, neg_w, args):
    if loss_name == "mae":
        regression = F.l1_loss(pred, yb)
    else:
        regression = F.smooth_l1_loss(pred, yb, beta=args.huber_beta)
    directional = torch.zeros((), device=pred.device)
    auxiliary = torch.zeros((), device=pred.device)
    if loss_name in {"directional", "directional_multitask"}:
        directional = _direction_loss_from_regression(
            pred, yb, zero_scaled, pos_w, neg_w, args.direction_temperature
        )
    if loss_name in {"multitask", "directional_multitask"}:
        if logits is None:
            raise RuntimeError("multitask loss requires direction logits")
        auxiliary = _aux_direction_loss(logits, yb, zero_scaled, pos_w, neg_w)
    total = regression + args.direction_weight * directional + args.aux_weight * auxiliary
    return total, regression, directional, auxiliary


def _build_model(past_dim: int, future_dim: int, args, device: torch.device) -> nn.Module:
    if args.loss in {"multitask", "directional_multitask"}:
        model = DualHeadTimeMixer(past_dim, future_dim, args)
    else:
        model = TimeMixerBackbone(
            past_dim=past_dim,
            future_dim=future_dim,
            pred_len=HOURLY.slots_per_day,
            hidden_dim=args.hidden_dim,
            n_blocks=args.blocks,
            scales=args.scales,
            dropout=args.dropout,
        )
    return model.to(device)


def _forward(model: nn.Module, xb: torch.Tensor, fb: torch.Tensor, loss_name: str):
    out = model(xb, fb)
    if loss_name in {"multitask", "directional_multitask"}:
        return out[0], out[1]
    return out, None


def _direction_batch_metrics(pred, logits, yb, zero_scaled) -> tuple[int, int, int, int]:
    true_sign = torch.sign(yb - zero_scaled)
    pred_sign = torch.sign(pred - zero_scaled)
    eligible = true_sign.ne(0)
    reg_correct = eligible & pred_sign.eq(true_sign)
    if logits is None:
        return int(reg_correct.sum()), int(eligible.sum()), 0, 0
    aux_sign = torch.where(logits >= 0, torch.ones_like(true_sign), -torch.ones_like(true_sign))
    aux_correct = eligible & aux_sign.eq(true_sign)
    return int(reg_correct.sum()), int(eligible.sum()), int(aux_correct.sum()), int(eligible.sum())


def _train_bundle(past: np.ndarray, future: np.ndarray, y: np.ndarray, args, device: torch.device, *, micro: bool = False) -> dict:
    n = len(y)
    if micro:
        train_idx = np.arange(n)
        valid_idx = np.arange(n)
    else:
        split = max(1, int(n * (1.0 - args.val_ratio)))
        train_idx = np.arange(split)
        valid_idx = np.arange(split, n) if split < n else np.arange(max(0, n - 1), n)
    ps, fs, ys = _fit_scalers(past, future, y, train_idx)
    past_t, future_t, y_t = _transform(ps, fs, ys, past, future, y)
    train_past = past_t[train_idx].copy()
    if args.proxy_dropout > 0:
        if args.input_mode != "proxy_mask":
            raise ValueError("--proxy-dropout is defined only for input_mode=proxy_mask")
        spread_input_idx = PAST_BASE_COLS.index("spread_input")
        zero_raw_scaled = float(-ps.mean_[spread_input_idx] / ps.scale_[spread_input_idx])
        rng = np.random.default_rng(args.seed + n)
        unavailable_view = train_past[:, 14:HOURLY.slots_per_day, spread_input_idx]
        drop = rng.random(unavailable_view.shape) < float(args.proxy_dropout)
        unavailable_view[drop] = zero_raw_scaled
    train_ds = DailyDataset(train_past, future_t[train_idx], y_t[train_idx])
    valid_ds = DailyDataset(past_t[valid_idx], future_t[valid_idx], y_t[valid_idx])
    loader = DataLoader(train_ds, batch_size=min(args.batch_size, len(train_ds)), shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=min(args.batch_size, len(valid_ds)), shuffle=False, num_workers=0)

    model = _build_model(past.shape[-1], future.shape[-1], args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    zero_scaled = torch.tensor(-ys.mean_ / ys.scale_, dtype=torch.float32, device=device).view(1, -1)
    pos_np, neg_np = _class_weights(y[train_idx])
    pos_w = torch.tensor(pos_np, dtype=torch.float32, device=device).view(1, -1)
    neg_w = torch.tensor(neg_np, dtype=torch.float32, device=device).view(1, -1)

    best_state = None
    best_valid = float("inf") if args.checkpoint_metric == "loss" else -float("inf")
    patience_left = args.patience
    history: list[dict] = []
    max_epochs = args.micro_epochs if micro else args.epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        sums = np.zeros(4, dtype=float)
        reg_hits = reg_total = aux_hits = aux_total = 0
        for xb, fb, yb in loader:
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred, logits = _forward(model, xb, fb, args.loss)
            total, reg, direction, aux = _loss_parts(
                pred, yb, logits, loss_name=args.loss, zero_scaled=zero_scaled,
                pos_w=pos_w, neg_w=neg_w, args=args,
            )
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            sums += np.asarray([float(total.detach()), float(reg.detach()), float(direction.detach()), float(aux.detach())]) * len(yb)
            rh, rt, ah, at = _direction_batch_metrics(pred.detach(), None if logits is None else logits.detach(), yb, zero_scaled)
            reg_hits += rh; reg_total += rt; aux_hits += ah; aux_total += at
        sums /= max(1, len(train_ds))

        model.eval()
        valid_total = 0.0
        val_reg_hits = val_reg_total = val_aux_hits = val_aux_total = 0
        val_reg_pos_hits = val_reg_pos_total = val_reg_neg_hits = val_reg_neg_total = 0
        val_aux_pos_hits = val_aux_pos_total = val_aux_neg_hits = val_aux_neg_total = 0
        with torch.no_grad():
            for xb, fb, yb in valid_loader:
                xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
                pred, logits = _forward(model, xb, fb, args.loss)
                total, _, _, _ = _loss_parts(
                    pred, yb, logits, loss_name=args.loss, zero_scaled=zero_scaled,
                    pos_w=pos_w, neg_w=neg_w, args=args,
                )
                valid_total += float(total) * len(yb)
                rh, rt, ah, at = _direction_batch_metrics(pred, logits, yb, zero_scaled)
                val_reg_hits += rh; val_reg_total += rt; val_aux_hits += ah; val_aux_total += at
                true_sign = torch.sign(yb - zero_scaled)
                reg_sign = torch.sign(pred - zero_scaled)
                pos = true_sign > 0
                neg = true_sign < 0
                val_reg_pos_total += int(pos.sum()); val_reg_neg_total += int(neg.sum())
                val_reg_pos_hits += int((pos & reg_sign.eq(true_sign)).sum())
                val_reg_neg_hits += int((neg & reg_sign.eq(true_sign)).sum())
                if logits is not None:
                    aux_sign = torch.where(logits >= 0, torch.ones_like(true_sign), -torch.ones_like(true_sign))
                    val_aux_pos_total += int(pos.sum()); val_aux_neg_total += int(neg.sum())
                    val_aux_pos_hits += int((pos & aux_sign.eq(true_sign)).sum())
                    val_aux_neg_hits += int((neg & aux_sign.eq(true_sign)).sum())
        valid_total /= max(1, len(valid_ds))
        reg_pos_acc = val_reg_pos_hits / max(1, val_reg_pos_total)
        reg_neg_acc = val_reg_neg_hits / max(1, val_reg_neg_total)
        reg_balanced = 0.5 * (reg_pos_acc + reg_neg_acc)
        aux_balanced = math.nan
        if val_aux_pos_total and val_aux_neg_total:
            aux_balanced = 0.5 * (
                val_aux_pos_hits / val_aux_pos_total + val_aux_neg_hits / val_aux_neg_total
            )
        history.append({
            "epoch": epoch,
            "train_loss": float(sums[0]),
            "train_regression_loss": float(sums[1]),
            "train_direction_loss": float(sums[2]),
            "train_aux_loss": float(sums[3]),
            "train_direction_accuracy": reg_hits / max(1, reg_total),
            "train_aux_direction_accuracy": aux_hits / max(1, aux_total) if aux_total else math.nan,
            "valid_loss": valid_total,
            "valid_direction_accuracy": val_reg_hits / max(1, val_reg_total),
            "valid_balanced_direction_accuracy": reg_balanced,
            "valid_aux_direction_accuracy": val_aux_hits / max(1, val_aux_total) if val_aux_total else math.nan,
            "valid_aux_balanced_direction_accuracy": aux_balanced,
        })

        if args.checkpoint_metric == "loss":
            checkpoint_value = valid_total
            improved = checkpoint_value < best_valid - args.min_delta
        elif args.checkpoint_metric == "reg_balanced":
            checkpoint_value = reg_balanced
            improved = checkpoint_value > best_valid + args.min_delta
        else:
            if logits is None or math.isnan(aux_balanced):
                raise ValueError("--checkpoint-metric aux_balanced requires a multitask loss")
            checkpoint_value = aux_balanced
            improved = checkpoint_value > best_valid + args.min_delta
        if improved:
            best_valid = checkpoint_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        elif not micro:
            patience_left -= 1
            if patience_left <= 0:
                break
        if micro and history[-1]["train_direction_accuracy"] >= args.micro_target_accuracy:
            break

    if best_state is not None and not micro:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "past_scaler": ps,
        "future_scaler": fs,
        "y_scaler": ys,
        "history": history,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "best_checkpoint_value": best_valid,
        "checkpoint_metric": args.checkpoint_metric,
    }


def _predict(bundle: dict, past: np.ndarray, future: np.ndarray, args, device: torch.device):
    ps, fs, ys = bundle["past_scaler"], bundle["future_scaler"], bundle["y_scaler"]
    past_t, future_t, _ = _transform(ps, fs, ys, past, future)
    ds = DailyDataset(past_t, future_t, np.zeros((len(past), HOURLY.slots_per_day), dtype=np.float32))
    loader = DataLoader(ds, batch_size=min(64, max(1, len(ds))), shuffle=False, num_workers=0)
    pred_out, logit_out = [], []
    bundle["model"].eval()
    with torch.no_grad():
        for xb, fb, _ in loader:
            pred, logits = _forward(bundle["model"], xb.to(device), fb.to(device), args.loss)
            pred_out.append(pred.cpu().numpy())
            if logits is not None:
                logit_out.append(logits.cpu().numpy())
    pred_raw = ys.inverse_transform(np.vstack(pred_out))
    logits = np.vstack(logit_out) if logit_out else None
    return pred_raw, logits


def _aux_metrics(true: np.ndarray, logits: np.ndarray | None) -> dict:
    if logits is None:
        return {}
    true_sign = np.sign(true)
    pred_sign = np.where(logits >= 0, 1, -1)
    eligible = true_sign != 0
    correct = eligible & (pred_sign == true_sign)
    pos, neg = true_sign > 0, true_sign < 0
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "aux_direction_accuracy": float(correct[eligible].mean()) if eligible.any() else math.nan,
        "aux_positive_accuracy": pos_acc,
        "aux_negative_accuracy": neg_acc,
        "aux_balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
    }


def _confident_wrong_rate(true: np.ndarray, pred: np.ndarray, threshold: float) -> float:
    true_sign, pred_sign = np.sign(true), np.sign(pred)
    eligible = (true_sign != 0) & (np.abs(pred) >= threshold)
    if not eligible.any():
        return math.nan
    return float((pred_sign[eligible] != true_sign[eligible]).mean())


def _frame_metrics(frame: pd.DataFrame, model_name: str, split: str, confident_threshold: float) -> dict:
    base = _metrics(frame, model_name, split)
    base["confident_wrong_sign_rate"] = _confident_wrong_rate(
        frame["y_true_spread"].to_numpy(float), frame["y_pred_spread"].to_numpy(float), confident_threshold
    )
    if "direction_logit" in frame.columns:
        logits = frame["direction_logit"].to_numpy(float).reshape(-1, HOURLY.slots_per_day)
        true = frame["y_true_spread"].to_numpy(float).reshape(-1, HOURLY.slots_per_day)
        base.update(_aux_metrics(true, logits))
    return base


def _build_ledger(days: list[str], y_true: np.ndarray, y_pred: np.ndarray, logits: np.ndarray | None, args, stage: str) -> pd.DataFrame:
    frames = []
    for i, day in enumerate(days):
        frame = pd.DataFrame({
            "target_day": day,
            "hour_business": np.arange(1, HOURLY.slots_per_day + 1),
            "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
            "y_true_spread": y_true[i],
            "y_pred_spread": y_pred[i],
            "model_name": f"tm_{args.loss}_h{args.hidden_dim}_b{args.blocks}",
            "stage": stage,
            "input_fill_scheme": args.input_mode,
        })
        if logits is not None:
            frame["direction_logit"] = logits[i]
            frame["direction_prob_positive"] = 1.0 / (1.0 + np.exp(-np.clip(logits[i], -50, 50)))
            frame["aux_predicted_direction"] = np.where(logits[i] >= 0, 1, -1)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _apply_input_mode(cache: dict[str, dict], input_mode: str, proxy_alpha: float) -> dict[str, dict]:
    """Materialize the requested causal representation without touching shared cache.

    ``proxy_mask`` is the historical baseline: the primary spread channel uses
    the safe rolling-median proxy after 14:00, while ``spread_visible_d1``
    explicitly marks those slots unavailable.

    ``pure_mask`` removes that proxy from the primary channel after 14:00 by
    setting it to the neutral raw-spread value 0. The availability channel is
    kept, and lag2/lag7/rolling-median remain separate proxy channels.

    ``blended_proxy`` scales only the unavailable primary-channel proxy by
    ``proxy_alpha``. It is a diagnostic bridge between the two endpoints:
    alpha=1 is the current proxy behaviour and alpha=0 is pure masking.
    """
    if input_mode not in INPUT_MODES:
        raise ValueError(f"unknown input_mode={input_mode}; allowed={INPUT_MODES}")
    if input_mode == "proxy_mask":
        return cache

    spread_input_idx = PAST_BASE_COLS.index("spread_input")
    visible_idx = PAST_BASE_COLS.index("spread_visible_d1")
    transformed: dict[str, dict] = {}
    for day, item in cache.items():
        copied = dict(item)
        past = np.asarray(item["past"], dtype=float).copy()
        unavailable = past[:, visible_idx] < 0.5
        if input_mode == "pure_mask":
            past[unavailable, spread_input_idx] = 0.0
        else:
            past[unavailable, spread_input_idx] *= float(proxy_alpha)
        copied["past"] = past
        transformed[day] = copied
    return transformed


def _prepare(args):
    out_root = Path(args.output_root)
    source = Path(args.data_path)
    _, raw, source_info = prepare_source_cache(source, out_root, Path(args.cache_root) if args.cache_root else None)
    raw["时刻"] = pd.to_datetime(raw["时刻"], errors="raise")
    raw["价差"] = pd.to_numeric(raw["价差"], errors="coerce")
    day_map = _daily_map(raw)
    dates = _date_list(day_map, args.start, args.end)
    if not dates:
        raise ValueError("no target dates in requested range")
    cache = _prepare_feature_cache(raw, dates, args.training_days)
    cache = _apply_input_mode(cache, args.input_mode, args.proxy_alpha)
    return out_root, raw, source_info, dates, cache


def run_micro_overfit(args) -> dict:
    out_root, _, source_info, dates, cache = _prepare(args)
    available = sorted(cache)
    candidates = [d for d in available if d < dates[0]]
    days = candidates[-args.micro_days:]
    if len(days) < args.micro_days:
        raise ValueError(f"need {args.micro_days} pre-target days, got {len(days)}")
    past = np.stack([cache[d]["past"] for d in days])
    future = np.stack([cache[d]["future"] for d in days])
    y = np.stack([cache[d]["y"] for d in days])
    device = _device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    started = time.perf_counter()
    bundle = _train_bundle(past, future, y, args, device, micro=True)
    pred, logits = _predict(bundle, past, future, args, device)
    ledger = _build_ledger(days, y, pred, logits, args, "micro_overfit")
    summary = _frame_metrics(ledger, ledger["model_name"].iloc[0], "micro_overfit", args.confident_threshold)
    summary.update({
        "parameter_count": bundle["parameter_count"],
        "epochs_ran": len(bundle["history"]),
        "final_train_loss": bundle["history"][-1]["train_loss"],
        "final_train_direction_accuracy": bundle["history"][-1]["train_direction_accuracy"],
    })
    if "train_aux_direction_accuracy" in bundle["history"][-1]:
        summary["final_train_aux_direction_accuracy"] = bundle["history"][-1]["train_aux_direction_accuracy"]
    _atomic_csv(out_root / "micro_overfit_summary.csv", pd.DataFrame([summary]))
    _atomic_json(out_root / "training_history.json", bundle["history"])
    manifest = {
        "pipeline": "timemixer_single_model_tuning",
        "mode": "micro_overfit",
        "status": "complete",
        "days": days,
        "loss": args.loss,
        "parameter_count": bundle["parameter_count"],
        "source": source_info,
        "runtime": {"device": str(device), "torch": torch.__version__, "elapsed_seconds": time.perf_counter() - started},
        "config": vars(args),
        "summary": summary,
    }
    _atomic_json(out_root / "manifest.json", manifest)
    return manifest


def run_walk_forward(args) -> dict:
    out_root, _, source_info, dates, cache = _prepare(args)
    available = sorted(cache)
    device = _device(args.device)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    started = time.perf_counter()
    ledger_parts: list[pd.DataFrame] = []
    histories: dict[str, list[dict]] = {}
    parameter_count = None
    for target_day in dates:
        train_days = [d for d in available if d < target_day][-args.training_days:]
        if len(train_days) < args.min_training_days:
            continue
        past = np.stack([cache[d]["past"] for d in train_days])
        future = np.stack([cache[d]["future"] for d in train_days])
        y = np.stack([cache[d]["y"] for d in train_days])
        bundle = _train_bundle(past, future, y, args, device, micro=False)
        parameter_count = bundle["parameter_count"]
        pred, logits = _predict(bundle, cache[target_day]["past"][None, ...], cache[target_day]["future"][None, ...], args, device)
        ledger_parts.append(_build_ledger([target_day], cache[target_day]["y"][None, ...], pred, logits, args, "walk_forward"))
        histories[target_day] = bundle["history"]

    if not ledger_parts:
        raise RuntimeError("no walk-forward predictions generated")
    ledger = pd.concat(ledger_parts, ignore_index=True)
    model_name = ledger["model_name"].iloc[0]
    overall = _frame_metrics(ledger, model_name, "requested_range", args.confident_threshold)
    daily_rows = []
    for day, group in ledger.groupby("target_day", sort=True):
        daily_rows.append(_frame_metrics(group, model_name, str(day), args.confident_threshold))
    period_rows = []
    for period, group in ledger.groupby("period", sort=False):
        period_rows.append(_frame_metrics(group, model_name, "requested_range", args.confident_threshold) | {"period": period})
    history_summary = []
    for day, hist in histories.items():
        best = min(hist, key=lambda r: r["valid_loss"])
        history_summary.append({
            "target_day": day,
            "epochs_ran": len(hist),
            "best_epoch": best["epoch"],
            "best_valid_loss": best["valid_loss"],
            "best_valid_direction_accuracy": best["valid_direction_accuracy"],
            "best_valid_aux_direction_accuracy": best.get("valid_aux_direction_accuracy", math.nan),
            "final_train_direction_accuracy": hist[-1]["train_direction_accuracy"],
        })
    history_df = pd.DataFrame(history_summary)
    overall.update({
        "parameter_count": int(parameter_count or 0),
        "mean_epochs_ran": float(history_df["epochs_ran"].mean()),
        "median_best_epoch": float(history_df["best_epoch"].median()),
        "mean_best_valid_direction_accuracy": float(history_df["best_valid_direction_accuracy"].mean()),
    })
    _atomic_parquet(out_root / "ledger" / "evaluation_ledger.parquet", ledger)
    _atomic_csv(out_root / "ledger" / "evaluation_ledger.csv", ledger)
    _atomic_csv(out_root / "summary" / "overall.csv", pd.DataFrame([overall]))
    _atomic_csv(out_root / "summary" / "daily.csv", pd.DataFrame(daily_rows))
    _atomic_csv(out_root / "summary" / "period.csv", pd.DataFrame(period_rows))
    _atomic_csv(out_root / "summary" / "training.csv", history_df)
    _atomic_json(out_root / "training_history.json", histories)
    manifest = {
        "pipeline": "timemixer_single_model_tuning",
        "mode": "walk_forward",
        "status": "complete",
        "start": dates[0], "end": dates[-1], "days": int(ledger["target_day"].nunique()),
        "loss": args.loss,
        "parameter_count": int(parameter_count or 0),
        "information_boundary": {"forecast_origin": "D-1 14:00", "source_max_ds": "<= cutoff"},
        "input_fill_scheme": args.input_mode,
        "source": source_info,
        "runtime": {"device": str(device), "torch": torch.__version__, "platform": platform.platform(), "elapsed_seconds": time.perf_counter() - started},
        "config": vars(args),
        "summary": overall,
    }
    _atomic_json(out_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["micro_overfit", "walk_forward"], required=True)
    p.add_argument("--data-path", default="data/24/canonical/shandong_pmos_hourly.csv")
    p.add_argument("--cache-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_shared_cache_opt_20260820")
    p.add_argument("--output-root", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--loss", choices=LOSS_NAMES, default="mae")
    p.add_argument("--input-mode", choices=INPUT_MODES, default="proxy_mask")
    p.add_argument("--proxy-alpha", type=float, default=1.0)
    p.add_argument("--proxy-dropout", type=float, default=0.0)
    p.add_argument("--checkpoint-metric", choices=CHECKPOINT_METRICS, default="loss")
    p.add_argument("--training-days", type=int, default=365)
    p.add_argument("--min-training-days", type=int, default=60)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--scales", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--huber-beta", type=float, default=0.5)
    p.add_argument("--direction-weight", type=float, default=0.5)
    p.add_argument("--aux-weight", type=float, default=0.5)
    p.add_argument("--direction-temperature", type=float, default=0.5)
    p.add_argument("--confident-threshold", type=float, default=50.0)
    p.add_argument("--micro-days", type=int, default=30)
    p.add_argument("--micro-epochs", type=int, default=300)
    p.add_argument("--micro-target-accuracy", type=float, default=0.98)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "micro_overfit":
        result = run_micro_overfit(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        run_walk_forward(args)


if __name__ == "__main__":
    main()
