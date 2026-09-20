"""Experiment-only TimeMixer structures for causal hourly spread forecasting.

This runner compares five single-model structures on the same 24-hour input:

* ``tm_unified24``: one 24-to-24 TimeMixer;
* ``tm_segmented_unweighted``: three independent 24-to-8 heads/models;
* ``tm_segmented_input_weighted``: the same three models with fixed temporal
  input gates (own block 1.2, neighbour 1.0, remote 0.8);
* ``tm_shared_heads_equal``: one shared TimeMixer encoder and three heads;
* ``tm_shared_heads_difficulty``: shared encoder/heads with fixed segment loss
  weights derived from the already audited 60-day segment difficulty.

It reuses the experiment FeatureStore only.  It never writes production
ledgers, model directories, or calls any delivery-stage learner/fuser.
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
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.run_lear_spread_experiment import (  # noqa: E402
    _safe_spread_features,
    _daily_map,
    prepare_source_cache,
)
from scripts.experiments.spread_direction_24.spread_metrics import smape_percent  # noqa: E402
from TimeMixer.backbones import TimeMixerBackbone  # noqa: E402
from utils.resolution import HOURLY  # noqa: E402


SEGMENTS = (("1_8", 0, 8), ("9_16", 8, 16), ("17_24", 16, 24))
PAST_BASE_COLS = (
    "spread_input",
    "spread_lag2",
    "spread_lag7",
    "spread_rolling_median",
    "spread_visible_d1",
    "spread_source_lag_days",
    "hour_sin",
    "hour_cos",
)
DEFAULT_DIFFICULTY_LOSS_WEIGHTS = np.asarray([0.95, 1.20, 0.85], dtype=np.float32)
MODEL_NAMES = (
    "tm_unified24",
    "tm_segmented_unweighted",
    "tm_segmented_input_weighted",
    "tm_shared_heads_equal",
    "tm_shared_heads_difficulty",
)


class DailyDataset(Dataset):
    def __init__(self, past: np.ndarray, future: np.ndarray, y: np.ndarray):
        self.past = torch.tensor(past, dtype=torch.float32)
        self.future = torch.tensor(future, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.past[idx], self.future[idx], self.y[idx]


class SharedHeadsTimeMixer(nn.Module):
    """One TimeMixer encoder with three independent 8-point prediction heads."""

    def __init__(self, past_dim: int, future_dim: int, hidden_dim: int, blocks: int, scales: int, dropout: float):
        super().__init__()
        self.encoder = TimeMixerBackbone(
            past_dim=past_dim,
            future_dim=future_dim,
            pred_len=HOURLY.slots_per_day,
            hidden_dim=hidden_dim,
            n_blocks=blocks,
            scales=scales,
            dropout=dropout,
        )
        z_dim = hidden_dim * (scales + 1)
        self.encoder.head = nn.Identity()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(z_dim),
                    nn.Linear(z_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, end - start),
                )
                for _, start, end in SEGMENTS
            ]
        )

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z = self.encoder(past, future)
        return tuple(head(z) for head in self.heads)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def _date_list(day_map: dict[str, pd.DataFrame], start: str, end: str) -> list[str]:
    return [d for d in sorted(day_map) if start <= d <= end]


def _prepare_feature_cache(raw: pd.DataFrame, dates: list[str], training_days: int) -> dict[str, dict]:
    day_map = _daily_map(raw)
    forecast_cols = [c for c in raw.columns if str(c).endswith("预测值")]
    history_by_period = {
        period: raw[raw["_business_period"].eq(period)].sort_values("时刻").copy()
        for period in range(1, HOURLY.slots_per_day + 1)
    }
    first = pd.Timestamp(dates[0]) - pd.Timedelta(days=training_days + 10)
    cache: dict[str, dict] = {}
    for day in sorted(day_map):
        if not (first.strftime("%Y-%m-%d") <= day <= dates[-1]):
            continue
        frame, source_max = _safe_spread_features(raw, day_map, day, forecast_cols, history_by_period)
        # The chosen post-cutoff strategy is the previously strongest safe
        # strategy: same-slot rolling median. D-2 same-slot is only fallback.
        rolling = pd.to_numeric(frame["spread_rolling_median"], errors="coerce")
        fallback = pd.to_numeric(frame["spread_lag2"], errors="coerce")
        safe = pd.to_numeric(frame["spread_safe_lag"], errors="coerce")
        frame["spread_input"] = pd.to_numeric(frame["spread_lag1_visible"], errors="coerce")
        after = frame["hour_business"].to_numpy(int) > 14
        frame.loc[after, "spread_input"] = rolling.loc[after].where(rolling.loc[after].notna(), fallback.loc[after])
        frame["spread_input"] = frame["spread_input"].where(frame["spread_input"].notna(), safe)
        if frame["spread_input"].isna().any():
            raise ValueError(f"{day}: safe rolling input has missing values")
        if pd.to_datetime(source_max, errors="coerce").notna().any():
            cutoff = pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
            if (pd.to_datetime(source_max, errors="coerce") > cutoff).any():
                raise RuntimeError(f"{day}: feature source exceeds cutoff")
        past = frame[list(PAST_BASE_COLS)].to_numpy(float)
        future_cols = [c for c in frame.columns if c.startswith("fcast::")]
        future = frame[future_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        y = frame["y_true_spread"].to_numpy(float)
        if past.shape[0] != HOURLY.slots_per_day or future.shape[0] != HOURLY.slots_per_day:
            raise ValueError(f"{day}: expected 24 rows")
        if not np.isfinite(past).all() or not np.isfinite(future).all() or not np.isfinite(y).all():
            raise ValueError(f"{day}: non-finite model input/label")
        cache[day] = {
            "past": past,
            "future": future,
            "y": y,
            "source_max": pd.to_datetime(source_max, errors="coerce").max(),
            "future_cols": future_cols,
        }
    return cache


def _slot_weights(segment_index: int) -> np.ndarray:
    values = np.full(HOURLY.slots_per_day, 0.8, dtype=np.float32)
    values[8:16] = 1.0
    values[16:24] = 0.8
    if segment_index == 0:
        values[:8], values[8:16], values[16:24] = 1.2, 1.0, 0.8
    elif segment_index == 1:
        values[:8], values[8:16], values[16:24] = 1.0, 1.2, 1.0
    else:
        values[:8], values[8:16], values[16:24] = 0.8, 1.0, 1.2
    return values


def _fit_scalers(past: np.ndarray, future: np.ndarray, y: np.ndarray, train_idx: np.ndarray):
    ps = StandardScaler().fit(past[train_idx].reshape(-1, past.shape[-1]))
    fs = StandardScaler().fit(future[train_idx].reshape(-1, future.shape[-1]))
    ys = StandardScaler().fit(y[train_idx])
    return ps, fs, ys


def _transform(ps, fs, ys, past, future, y=None):
    past_t = ps.transform(past.reshape(-1, past.shape[-1])).reshape(past.shape)
    future_t = fs.transform(future.reshape(-1, future.shape[-1])).reshape(future.shape)
    y_t = ys.transform(y) if y is not None else None
    return past_t, future_t, y_t


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _train_model(
    past: np.ndarray,
    future: np.ndarray,
    y: np.ndarray,
    *,
    kind: str,
    args,
    device: torch.device,
    input_weight: np.ndarray | None = None,
    loss_weights: np.ndarray | None = None,
) -> dict:
    n = len(y)
    split = max(1, int(n * (1.0 - args.val_ratio)))
    train_idx = np.arange(split)
    valid_idx = np.arange(split, n) if split < n else np.arange(max(0, n - 1), n)
    ps, fs, ys = _fit_scalers(past, future, y, train_idx)
    past_t, future_t, y_t = _transform(ps, fs, ys, past, future, y)
    if input_weight is not None:
        past_t = past_t * input_weight.reshape(1, -1, 1)
        future_t = future_t * input_weight.reshape(1, -1, 1)
    train_ds = DailyDataset(past_t[train_idx], future_t[train_idx], y_t[train_idx])
    valid_ds = DailyDataset(past_t[valid_idx], future_t[valid_idx], y_t[valid_idx])
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if kind == "unified":
        model = TimeMixerBackbone(
            past_dim=past.shape[-1], future_dim=future.shape[-1], pred_len=y.shape[1],
            hidden_dim=args.hidden_dim, n_blocks=args.blocks, scales=args.scales, dropout=args.dropout,
        ).to(device)
    elif kind == "shared_heads":
        model = SharedHeadsTimeMixer(
            past.shape[-1], future.shape[-1], args.hidden_dim, args.blocks, args.scales, args.dropout
        ).to(device)
    else:
        raise ValueError(kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_valid = float("inf")
    patience = args.patience
    weights = torch.tensor(loss_weights if loss_weights is not None else [1.0, 1.0, 1.0], dtype=torch.float32, device=device)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, fb, yb in loader:
            xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, fb)
            if kind == "unified":
                loss = torch.nn.functional.l1_loss(pred, yb)
            else:
                losses = [torch.nn.functional.l1_loss(p, yb[:, start:end]) for p, (_, start, end) in zip(pred, SEGMENTS)]
                loss = sum(weights[i] * losses[i] for i in range(3)) / weights.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(yb)
        train_loss /= max(1, len(train_ds))
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for xb, fb, yb in valid_loader:
                xb, fb, yb = xb.to(device), fb.to(device), yb.to(device)
                pred = model(xb, fb)
                if kind == "unified":
                    loss = torch.nn.functional.l1_loss(pred, yb)
                else:
                    losses = [torch.nn.functional.l1_loss(p, yb[:, start:end]) for p, (_, start, end) in zip(pred, SEGMENTS)]
                    loss = sum(weights[i] * losses[i] for i in range(3)) / weights.sum()
                valid_loss += float(loss) * len(yb)
        valid_loss /= max(1, len(valid_ds))
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_valid - 1e-6:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "past_scaler": ps, "future_scaler": fs, "y_scaler": ys, "history": history}


def _predict(bundle: dict, past: np.ndarray, future: np.ndarray, device: torch.device, input_weight: np.ndarray | None, shared: bool) -> np.ndarray:
    ps, fs, ys = bundle["past_scaler"], bundle["future_scaler"], bundle["y_scaler"]
    past_t, future_t, _ = _transform(ps, fs, ys, past, future)
    if input_weight is not None:
        past_t = past_t * input_weight.reshape(1, -1, 1)
        future_t = future_t * input_weight.reshape(1, -1, 1)
    pred_len = int(getattr(ys, "n_features_in_", 24))
    ds = DailyDataset(past_t, future_t, np.zeros((len(past), pred_len), dtype=np.float32))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    out = []
    bundle["model"].eval()
    with torch.no_grad():
        for xb, fb, _ in loader:
            pred = bundle["model"](xb.to(device), fb.to(device))
            if shared:
                pred = torch.cat(pred, dim=1)
            out.append(pred.detach().cpu().numpy())
    return ys.inverse_transform(np.vstack(out))


def _metrics(frame: pd.DataFrame, model_name: str, split: str) -> dict:
    true = frame["y_true_spread"].to_numpy(float)
    pred = frame["y_pred_spread"].to_numpy(float)
    ts, ps = np.sign(true), np.sign(pred)
    eligible = ts != 0
    correct = eligible & (ts == ps)
    pos, neg = ts > 0, ts < 0
    pos_acc = float(correct[pos].mean()) if pos.any() else math.nan
    neg_acc = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "model_name": model_name,
        "split": split,
        "days": int(frame["target_day"].nunique()),
        "n_slots": int(len(frame)),
        "n_positive_actual": int(pos.sum()),
        "n_negative_actual": int(neg.sum()),
        "direction_accuracy": float(correct[eligible].mean()),
        "positive_accuracy": pos_acc,
        "negative_accuracy": neg_acc,
        "balanced_direction_accuracy": float(np.nanmean([pos_acc, neg_acc])),
        "mae": float(np.mean(np.abs(pred - true))),
        "rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
        "spread_smape_pct": smape_percent(true, pred),
    }


def _split_name(index: int) -> str:
    if index < 30:
        return "development_30d"
    if index < 45:
        return "confirmation_15d"
    return "holdout_15d"


def run(args) -> dict:
    out_root = Path(args.output_root)
    source = Path(args.data_path)
    _, raw, source_info = prepare_source_cache(source, out_root, Path(args.cache_root) if args.cache_root else None)
    raw["时刻"] = pd.to_datetime(raw["时刻"], errors="raise")
    raw["价差"] = pd.to_numeric(raw["价差"], errors="coerce")
    day_map = _daily_map(raw)
    dates = _date_list(day_map, args.start, args.end)
    if len(dates) < 45:
        raise ValueError("structure experiment requires at least 45 target days")
    if len(dates) != 60:
        raise ValueError("this registered experiment expects exactly 60 target days")
    cache = _prepare_feature_cache(raw, dates, args.training_days)
    usable = [d for d in dates if d in cache]
    if usable != dates:
        raise ValueError(f"feature cache incomplete: missing={sorted(set(dates)-set(usable))[:5]}")
    device = _device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rows: list[pd.DataFrame] = []
    metrics: list[dict] = []
    histories: dict[str, list] = {}
    started = time.perf_counter()
    available_days = sorted(cache)

    active_models = tuple(x.strip() for x in args.models.split(",") if x.strip())
    unknown = sorted(set(active_models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"unknown models={unknown}; allowed={list(MODEL_NAMES)}")
    difficulty_weights = np.asarray(args.difficulty_weights, dtype=np.float32)
    if difficulty_weights.shape != (3,) or np.any(difficulty_weights <= 0):
        raise ValueError("--difficulty-weights must contain three positive values")

    for idx, target_day in enumerate(dates):
        earlier = [d for d in available_days if d < target_day]
        train_days = earlier[-args.training_days:]
        if len(train_days) < args.min_training_days:
            continue
        past_train = np.stack([cache[d]["past"] for d in train_days])
        future_train = np.stack([cache[d]["future"] for d in train_days])
        y_train = np.stack([cache[d]["y"] for d in train_days])
        target_past = cache[target_day]["past"][None, ...]
        target_future = cache[target_day]["future"][None, ...]
        target_y = cache[target_day]["y"]
        split = _split_name(idx)
        models_for_day: dict[str, tuple[dict, np.ndarray | None, bool]] = {}

        if "tm_unified24" in active_models:
            bundle = _train_model(past_train, future_train, y_train, kind="unified", args=args, device=device)
            models_for_day["tm_unified24"] = (bundle, None, False)
            histories.setdefault("tm_unified24", []).append(bundle["history"])

        for weighted, name in ((None, "tm_segmented_unweighted"), (_slot_weights(0), "tm_segmented_input_weighted")):
            if name not in active_models:
                continue
            stitched = np.zeros((1, 24), dtype=float)
            segment_hist = {}
            for seg_i, (seg_name, start, end) in enumerate(SEGMENTS):
                # Each segment model has the same full 24 input and only its
                # own 8 labels. The shared input gate is segment-specific.
                seg_y = y_train[:, start:end]
                seg_bundle = _train_model(
                    past_train, future_train, seg_y,
                    kind="unified", args=args, device=device,
                    input_weight=(_slot_weights(seg_i) if weighted is not None else None),
                )
                # The generic unified trainer predicts 24 points. We use the
                # segment slice as the supervised target through a local view
                # below; this branch is replaced after construction.
                pred = _predict(seg_bundle, target_past, target_future, device, (_slot_weights(seg_i) if weighted is not None else None), False)
                stitched[:, start:end] = pred
                segment_hist[seg_name] = seg_bundle["history"]
            models_for_day[name] = ({"pred_override": stitched}, None, False)
            histories.setdefault(name, []).append(segment_hist)

        for loss_weights, name in ((np.ones(3, dtype=np.float32), "tm_shared_heads_equal"), (difficulty_weights, "tm_shared_heads_difficulty")):
            if name not in active_models:
                continue
            bundle = _train_model(past_train, future_train, y_train, kind="shared_heads", args=args, device=device, loss_weights=loss_weights)
            models_for_day[name] = (bundle, None, True)
            histories.setdefault(name, []).append(bundle["history"])

        for model_name, (bundle, weight, shared) in models_for_day.items():
            if "pred_override" in bundle:
                pred = bundle["pred_override"][0]
            else:
                pred = _predict(bundle, target_past, target_future, device, weight, shared)[0]
            out = pd.DataFrame({
                "target_day": target_day,
                "hour_business": np.arange(1, 25),
                "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
                "y_true_spread": target_y,
                "y_pred_spread": pred,
                "model_name": model_name,
                "segment_model_id": model_name + "_" + pd.Series(["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8),
                "segment_training": True,
                "input_fill_scheme": "d1_p1_p14_plus_rolling_median_p15_p24_fallback_lag48",
                "training_days": len(train_days),
                "information_cutoff": pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14),
            })
            out["actual_direction"] = np.sign(out["y_true_spread"]).astype(int)
            out["predicted_direction"] = np.sign(out["y_pred_spread"]).astype(int)
            out["direction_eligible"] = out["actual_direction"] != 0
            out["direction_correct"] = out["direction_eligible"] & (out["actual_direction"] == out["predicted_direction"])
            rows.append(out)
            metrics.append(_metrics(out, model_name, split))

    ledger = pd.concat(rows, ignore_index=True)
    _atomic_parquet(out_root / "ledger" / "evaluation_ledger.parquet", ledger)
    _atomic_csv(out_root / "ledger" / "evaluation_ledger.csv", ledger)
    daily_metrics_df = pd.DataFrame(metrics)
    _atomic_csv(out_root / "summary" / "daily_metrics.csv", daily_metrics_df)
    split_metrics = []
    period_metrics = []
    for (model_name, split), group in ledger.assign(split=ledger["target_day"].map({d: _split_name(i) for i, d in enumerate(dates)})).groupby(["model_name", "split"], sort=False):
        split_metrics.append(_metrics(group, model_name, split))
        for period in ("1_8", "9_16", "17_24"):
            period_group = group[group["period"].eq(period)]
            period_metrics.append(_metrics(period_group, model_name, split) | {"period": period})
    metrics_df = pd.DataFrame(split_metrics)
    _atomic_csv(out_root / "summary" / "metrics_by_split.csv", metrics_df)
    _atomic_csv(out_root / "summary" / "period_metrics_by_split.csv", pd.DataFrame(period_metrics))
    summary = metrics_df[metrics_df["split"].eq("holdout_15d")].copy()
    if summary.empty:
        summary = metrics_df.copy()
    _atomic_csv(out_root / "summary" / "model_summary.csv", summary)
    _atomic_json(out_root / "training_history.json", histories)
    manifest = {
        "pipeline": "spread_direction_24_timemixer_structure_experiment",
        "status": "complete",
        "resolution": "hourly",
        "hourly_only": True,
        "models": list(active_models),
        "start": dates[0], "end": dates[-1], "days": len(dates),
        "training_months": 12,
        "training_days": args.training_days,
        "evaluation_splits": {"development": 30, "confirmation": 15, "holdout": 15},
        "input_fill_scheme": "d1_p1_p14_plus_rolling_median_p15_p24_fallback_lag48",
        "input_weights": {"own": 1.2, "adjacent": 1.0, "remote": 0.8},
        "difficulty_loss_weights": difficulty_weights.tolist(),
        "information_boundary": {"forecast_origin": "D-1 14:00", "target_day_actual_labels": "label only", "source_max_ds": "<= cutoff"},
        "source": source_info,
        "runtime": {"python": sys.version, "platform": platform.platform(), "device": str(device), "seed": args.seed},
        "model_config": vars(args),
        "rows": int(len(ledger)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(out_root / "range_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="data/24/canonical/shandong_pmos_hourly.csv")
    p.add_argument("--cache-root", default="outputs/experiments/01_spread_24/invalid_leakage/legacy_cutoff_leakage_spread_direction_24_shared_cache_opt_20260820")
    p.add_argument("--output-root", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--models", default=",".join(MODEL_NAMES))
    p.add_argument("--difficulty-weights", type=float, nargs=3, default=DEFAULT_DIFFICULTY_LOSS_WEIGHTS.tolist())
    p.add_argument("--training-days", type=int, default=365)
    p.add_argument("--min-training-days", type=int, default=60)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--scales", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
