"""Fast CPU XLinear-style daily adapter for causal hourly spread forecasting.

This is an experiment-only *daily adapter*, not the upstream trainer.  It keeps
XLinear's lightweight sigmoid-gating idea and explicit exogenous projection,
while adapting the I/O to this project's D-1 14:00 -> target-day 24-slot
business contract.  Future forecast-grid slots are flattened in chronological
order, so their hour alignment is preserved (no mean pooling).

Architecture references:
- AAAI 2026 XLinear: https://github.com/Zaiwen/XLinear
- Nixtla XLinear future-exogenous projection:
  https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/xlinear.py
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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.spread_direction_24.run_fast_linear_probe import (
    ProbeConfig,
    _build_vector,
)
from scripts.experiments.spread_direction_24.run_lear_spread_experiment import (
    SPREAD_COL,
    _daily_map,
    _date_list,
    _metrics,
    _safe_spread_features,
    prepare_source_cache,
)
from utils.resolution import HOURLY

CONFIGS = (
    ProbeConfig("x0_hist7_future", 7, False, False, 0.0),
    ProbeConfig("x1_hist7_context_proxy_future", 7, True, True, 0.0),
    ProbeConfig("x2_hist7_context_proxydrop25_future", 7, True, True, 0.25),
    ProbeConfig("x3_hist14_context_proxydrop25_future", 14, True, True, 0.25),
)


class GatingBlock(nn.Module):
    """XLinear sigmoid gating: x * sigmoid(MLP(x))."""
    def __init__(self, dim: int, hidden_ff: int, dropout: float = 0.0):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(dim, hidden_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_ff, dim), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight(x)


class XLinearDailyAdapter(nn.Module):
    """Small XLinear-style adapter with explicit ordered future-exog branch."""
    def __init__(self, history_dim: int, exog_dim: int, hidden: int = 64, ff: int = 128, dropout: float = 0.05):
        super().__init__()
        self.history_proj = nn.Linear(history_dim, hidden)
        self.global_token = nn.Parameter(torch.zeros(1, hidden))
        self.temporal_gate = GatingBlock(2 * hidden, ff, dropout)
        self.exog_proj = nn.Linear(exog_dim, hidden)
        self.exog_gate = GatingBlock(hidden, max(16, hidden // 2), dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(3 * hidden), nn.Dropout(dropout), nn.Linear(3 * hidden, HOURLY.slots_per_day)
        )

    def forward(self, history: torch.Tensor, exog: torch.Tensor) -> torch.Tensor:
        h = self.history_proj(history)
        g = self.global_token.expand(len(history), -1)
        temporal = self.temporal_gate(torch.cat([h, g], dim=-1))
        ex = self.exog_gate(self.exog_proj(exog))
        return self.head(torch.cat([temporal, ex], dim=-1))


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


def _dims(cfg: ProbeConfig, forecast_count: int) -> tuple[int, int]:
    history_dim = cfg.history_days * HOURLY.slots_per_day
    exog_dim = forecast_count * HOURLY.slots_per_day + 4
    if cfg.use_context:
        exog_dim += HOURLY.slots_per_day
    if cfg.use_proxy:
        exog_dim += 4 * HOURLY.slots_per_day
    return history_dim, exog_dim


def _scale_fit(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray):
    imp = SimpleImputer(strategy="median")
    Xtr = imp.fit_transform(X[train_idx])
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(y[train_idx])
    return imp, xs, ys


def _apply_proxy_dropout(x: torch.Tensor, proxy_indices: list[int], rate: float) -> torch.Tensor:
    if rate <= 0 or not proxy_indices:
        return x
    out = x.clone()
    cols = torch.as_tensor(proxy_indices, dtype=torch.long, device=x.device)
    mask = torch.rand((len(x), len(proxy_indices)), device=x.device) < rate
    block = out.index_select(1, cols)
    block = torch.where(mask, torch.zeros_like(block), block)
    out[:, cols] = block
    return out


def _train_predict(X: np.ndarray, y: np.ndarray, xt: np.ndarray, proxy_indices: list[int], cfg: ProbeConfig, args, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(X)
    split = max(30, int(n * 0.8))
    split = min(split, n - 1)
    tr = np.arange(split)
    va = np.arange(split, n)
    imp, xs, ys = _scale_fit(X, y, tr)
    Xs = xs.transform(imp.transform(X)).astype(np.float32)
    yt = ys.transform(y).astype(np.float32)
    xts = xs.transform(imp.transform(xt[None, :])).astype(np.float32)
    hist_dim, exog_dim = _dims(cfg, args.forecast_count)
    if hist_dim + exog_dim != Xs.shape[1]:
        raise ValueError(f"dimension mismatch {hist_dim}+{exog_dim}!={Xs.shape[1]}")
    # proxy_indices are in full-vector coordinates; exog branch starts at history_dim.
    proxy_exog_idx = [i - hist_dim for i in proxy_indices if i >= hist_dim]
    model = XLinearDailyAdapter(hist_dim, exog_dim, args.hidden, args.ff, args.dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    X_tensor = torch.from_numpy(Xs)
    y_tensor = torch.from_numpy(yt)
    best_state = None
    best_val = float("inf")
    patience = args.patience
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(tr))
        for start in range(0, len(tr), args.batch_size):
            idx = order[start:start + args.batch_size]
            xb = X_tensor[idx]
            yb = y_tensor[idx]
            h = xb[:, :hist_dim]
            ex = xb[:, hist_dim:]
            ex = _apply_proxy_dropout(ex, proxy_exog_idx, cfg.proxy_dropout)
            optimizer.zero_grad(set_to_none=True)
            pred = model(h, ex)
            loss = F.l1_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            xv = X_tensor[va]
            pv = model(xv[:, :hist_dim], xv[:, hist_dim:])
            val = float(F.l1_loss(pv, y_tensor[va]))
        if val < best_val - 1e-5:
            best_val = val
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = args.patience
        else:
            patience -= 1
            if patience <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xx = torch.from_numpy(xts)
        pred_scaled = model(xx[:, :hist_dim], xx[:, hist_dim:]).numpy()
    pred = ys.inverse_transform(pred_scaled)[0]
    return pred, int(sum(p.numel() for p in model.parameters())), best_epoch, epoch, best_val


def run(args) -> dict:
    out_root = Path(args.output_root)
    _, raw, source_info = prepare_source_cache(Path(args.data_path), out_root, Path(args.cache_root) if args.cache_root else None)
    raw["时刻"] = pd.to_datetime(raw["时刻"], errors="raise")
    raw[SPREAD_COL] = pd.to_numeric(raw[SPREAD_COL], errors="coerce")
    forecast_cols = [c for c in raw.columns if str(c).endswith("预测值")]
    args.forecast_count = len(forecast_cols)
    day_map = _daily_map(raw)
    dates = _date_list(day_map, args.start, args.end)
    history_by_period = {p: raw[raw["_business_period"].eq(p)].sort_values("时刻").copy() for p in range(1, 25)}
    cache_start = (pd.Timestamp(dates[0]) - pd.Timedelta(days=args.training_days + 50)).strftime("%Y-%m-%d")
    safe_cache = {
        day: _safe_spread_features(raw, day_map, day, forecast_cols, history_by_period)[0]
        for day in sorted(day_map) if cache_start <= day <= dates[-1]
    }
    active = {x.strip() for x in args.configs.split(",") if x.strip()} if args.configs else {c.name for c in CONFIGS}
    configs = [c for c in CONFIGS if c.name in active]
    if active - {c.name for c in CONFIGS}:
        raise ValueError(f"unknown configs={sorted(active - {c.name for c in CONFIGS})}")
    all_rows, audits = [], []
    started = time.perf_counter()
    for cfg in configs:
        vectors = {}
        for day in sorted(safe_cache):
            try:
                vectors[day] = _build_vector(day, cfg=cfg, day_map=day_map, safe_frame=safe_cache[day], forecast_cols=forecast_cols)
            except (KeyError, ValueError):
                continue
        for target_day in dates:
            train_days = [d for d in sorted(vectors) if d < target_day][-args.training_days:]
            if len(train_days) < args.min_training_days:
                continue
            X = np.stack([vectors[d][0] for d in train_days])
            y = np.stack([vectors[d][1] for d in train_days])
            xt, yt, proxy_idx, source_max = vectors[target_day]
            cutoff = pd.Timestamp(target_day) - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
            if source_max > cutoff:
                raise RuntimeError(f"{target_day}/{cfg.name}: source exceeds cutoff")
            t0 = time.perf_counter()
            pred, params, best_epoch, epochs_ran, best_val = _train_predict(
                X, y, xt, proxy_idx, cfg, args, args.seed + pd.Timestamp(target_day).dayofyear
            )
            frame = pd.DataFrame({
                "target_day": target_day,
                "hour_business": np.arange(1, 25),
                "period": ["1_8"] * 8 + ["9_16"] * 8 + ["17_24"] * 8,
                "y_true_spread": yt,
                "y_pred_spread": pred,
                "model_name": cfg.name,
                "information_cutoff": cutoff,
            })
            all_rows.append(frame)
            audits.append({
                "target_day": target_day, "model_name": cfg.name, "training_days": len(train_days),
                "feature_count": X.shape[1], "parameter_count": params, "best_epoch": best_epoch,
                "epochs_ran": epochs_ran, "best_valid_loss": best_val,
                "fit_seconds": time.perf_counter() - t0, "source_max_ds": source_max,
            })
    ledger = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame([_metrics(g, name) for name, g in ledger.groupby("model_name", sort=False)])
    summary = summary.sort_values(["balanced_direction_accuracy", "direction_accuracy"], ascending=[False, False])
    _atomic_parquet(out_root / "ledger" / "evaluation_ledger.parquet", ledger)
    _atomic_csv(out_root / "summary" / "model_summary.csv", summary)
    _atomic_csv(out_root / "summary" / "training_audit.csv", pd.DataFrame(audits))
    manifest = {
        "pipeline": "spread_direction_24_xlinear_daily_adapter",
        "status": "complete",
        "adapter_note": "XLinear-style daily business-contract adapter; not upstream NeuralForecast trainer",
        "upstream": ["https://github.com/Zaiwen/XLinear", "https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/xlinear.py"],
        "start": dates[0], "end": dates[-1], "days": len(dates),
        "training_days": args.training_days,
        "configs": [c.__dict__ for c in configs],
        "model_config": {k: v for k, v in vars(args).items() if k != "forecast_count"},
        "forecast_columns": forecast_cols,
        "information_boundary": {"forecast_origin": "D-1 14:00", "completed_history": "D-2 and earlier", "target_grid": "forecast only", "target_actual": "label only"},
        "future_order_preserved": True,
        "historical_preflight_override": "24-point freshness failed; experiment ends before source max and integrity/leakage checks passed",
        "source": source_info,
        "runtime": {"device": "cpu", "torch": torch.__version__, "platform": platform.platform(), "elapsed_seconds": time.perf_counter() - started},
        "top_models": summary.to_dict(orient="records"),
    }
    _atomic_json(out_root / "manifest.json", manifest)
    print(summary.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default="data/24/canonical/shandong_pmos_hourly.csv")
    p.add_argument("--cache-root", default="outputs/experiments/01_spread_24/legacy_root/spread_direction_24_shared_cache_opt_20260820")
    p.add_argument("--output-root", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--training-days", type=int, default=365)
    p.add_argument("--min-training-days", type=int, default=180)
    p.add_argument("--configs", default="")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
