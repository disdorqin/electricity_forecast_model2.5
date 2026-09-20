from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA = ROOT / "data/24/canonical/shandong_pmos_hourly.csv"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_hourly(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="gb18030")
    raw["时刻"] = pd.to_datetime(raw["时刻"], errors="raise")
    raw["日前电价"] = pd.to_numeric(raw["日前电价"], errors="coerce")
    raw["实时电价"] = pd.to_numeric(raw["实时电价"], errors="coerce")
    raw["spread"] = raw["实时电价"] - raw["日前电价"]
    raw = raw.sort_values("时刻").drop_duplicates("时刻", keep="last").reset_index(drop=True)
    return raw[["时刻", "spread"]]


@dataclass
class Scaler:
    mean: float
    std: float

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class Seq2Seq(nn.Module):
    """Compact LSTM encoder-decoder matching the paper's core recipe.

    Paper anchors preserved: lag=48, encoder-decoder LSTM, Adam, batch=64,
    dropout=0.2. Exact neuron counts are treated as a local reproduction choice.
    """

    def __init__(self, hidden: int = 64, encoder_layers: int = 2, decoder_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        enc_dropout = dropout if encoder_layers > 1 else 0.0
        dec_dropout = dropout if decoder_layers > 1 else 0.0
        self.encoder = nn.LSTM(1, hidden, num_layers=encoder_layers, batch_first=True, dropout=enc_dropout)
        self.decoder = nn.LSTM(1, hidden, num_layers=decoder_layers, batch_first=True, dropout=dec_dropout)
        self.bridge_h = nn.Linear(hidden, hidden)
        self.bridge_c = nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.decoder_layers = decoder_layers

    def _bridge_state(self, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Use the top encoder state and replicate to decoder depth.
        ht = torch.tanh(self.bridge_h(h[-1])).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        ct = torch.tanh(self.bridge_c(c[-1])).unsqueeze(0).repeat(self.decoder_layers, 1, 1)
        return ht, ct

    def forward(self, x: torch.Tensor, out_len: int, y_teacher: torch.Tensor | None = None, teacher_forcing: float = 0.0) -> torch.Tensor:
        _, (h, c) = self.encoder(x)
        h, c = self._bridge_state(h, c)
        prev = x[:, -1:, :]
        outs = []
        for t in range(out_len):
            z, (h, c) = self.decoder(prev, (h, c))
            pred = self.head(self.drop(z[:, -1:, :]))
            outs.append(pred)
            if self.training and y_teacher is not None and teacher_forcing > 0:
                use_truth = torch.rand(x.shape[0], 1, 1, device=x.device) < teacher_forcing
                truth = y_teacher[:, t : t + 1, :]
                prev = torch.where(use_truth, truth, pred)
            else:
                prev = pred
        return torch.cat(outs, dim=1)


def contiguous_hourly_map(frame: pd.DataFrame) -> dict[pd.Timestamp, float]:
    return {pd.Timestamp(r.时刻): float(r.spread) for r in frame.itertuples(index=False) if np.isfinite(r.spread)}


def build_next1(frame: pd.DataFrame, lag: int = 48) -> pd.DataFrame:
    vals = contiguous_hourly_map(frame)
    times = sorted(vals)
    rows = []
    for target in times:
        hist = [target - pd.Timedelta(hours=k) for k in range(lag, 0, -1)]
        if all(t in vals for t in hist):
            rows.append({"target_ts": target, "x": np.asarray([vals[t] for t in hist], np.float32), "y": np.asarray([vals[target]], np.float32)})
    return pd.DataFrame(rows)


def build_day24(frame: pd.DataFrame, lag: int = 48, future_len: int = 34) -> pd.DataFrame:
    vals = contiguous_hourly_map(frame)
    min_day = frame["时刻"].min().normalize() + pd.Timedelta(days=3)
    max_day = frame["时刻"].max().normalize() - pd.Timedelta(days=1)
    rows = []
    for day in pd.date_range(min_day, max_day, freq="D"):
        cutoff = day - pd.Timedelta(days=1) + pd.Timedelta(hours=14)
        hist = [cutoff - pd.Timedelta(hours=k) for k in range(lag - 1, -1, -1)]
        fut = [cutoff + pd.Timedelta(hours=k) for k in range(1, future_len + 1)]
        target24 = [day + pd.Timedelta(hours=h) for h in range(1, 24)] + [day + pd.Timedelta(days=1)]
        if all(t in vals for t in [*hist, *fut, *target24]):
            y_all = np.asarray([vals[t] for t in fut], np.float32)
            # Last 24 future steps are exactly business-day h1..h24.
            assert fut[-24:] == target24
            rows.append({
                "target_day": day.strftime("%Y-%m-%d"),
                "cutoff": cutoff,
                "x": np.asarray([vals[t] for t in hist], np.float32),
                "y": y_all,
            })
    return pd.DataFrame(rows)


def stack_arrays(frame: pd.DataFrame, scaler: Scaler) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(frame["x"].to_list())
    y = np.stack(frame["y"].to_list())
    x = scaler.transform(x)[..., None].astype(np.float32)
    y = scaler.transform(y)[..., None].astype(np.float32)
    return x, y


def direction_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    yt = np.sign(np.asarray(y, float).reshape(-1))
    yp = np.sign(np.asarray(pred, float).reshape(-1))
    mask = yt != 0
    pos = yt > 0
    neg = yt < 0
    correct = yt == yp
    pa = float(correct[pos].mean()) if pos.any() else math.nan
    na = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "n_nonzero": int(mask.sum()),
        "direction_accuracy": float(correct[mask].mean()) if mask.any() else math.nan,
        "positive_accuracy": pa,
        "negative_accuracy": na,
        "balanced_direction_accuracy": float(np.nanmean([pa, na])),
        "mae": float(np.mean(np.abs(np.asarray(pred) - np.asarray(y)))),
        "rmse": float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2))),
    }


def train_model(model: nn.Module, train_loader: DataLoader, val_x: torch.Tensor, val_y: torch.Tensor, out_len: int,
                epochs: int, patience: int, lr: float, teacher_forcing: float, device: torch.device) -> tuple[nn.Module, list[dict]]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best = math.inf
    best_state = None
    wait = 0
    hist = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, out_len=out_len, y_teacher=yb, teacher_forcing=teacher_forcing)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pv = model(val_x.to(device), out_len=out_len)
            vl = float(loss_fn(pv, val_y.to(device)).detach().cpu())
        tr = float(np.mean(losses))
        hist.append({"epoch": epoch, "train_loss": tr, "val_loss": vl})
        if vl + 1e-6 < best:
            best = vl
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch={epoch} train={tr:.5f} val={vl:.5f} best={best:.5f}")
        if wait >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def predict(model: nn.Module, x: np.ndarray, out_len: int, device: torch.device, batch: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    xt = torch.as_tensor(x, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(xt), batch):
            outs.append(model(xt[i:i+batch].to(device), out_len=out_len).cpu().numpy())
    return np.concatenate(outs, axis=0)[..., 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["next1", "day24"], required=True)
    ap.add_argument("--data", default=str(DEFAULT_DATA.relative_to(ROOT)))
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-start", required=True)
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--val-start", required=True)
    ap.add_argument("--val-end", required=True)
    ap.add_argument("--test-start", required=True)
    ap.add_argument("--test-end", required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--encoder-layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--teacher-forcing", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.perf_counter()
    set_seed(args.seed)
    outdir = ROOT / args.output
    outdir.mkdir(parents=True, exist_ok=True)
    raw = load_hourly(ROOT / args.data)
    samples = build_next1(raw) if args.mode == "next1" else build_day24(raw)
    key = "target_ts" if args.mode == "next1" else "target_day"
    dates = pd.to_datetime(samples[key])
    tr = samples[(dates >= pd.Timestamp(args.train_start)) & (dates <= pd.Timestamp(args.train_end))].copy()
    va = samples[(dates >= pd.Timestamp(args.val_start)) & (dates <= pd.Timestamp(args.val_end))].copy()
    te = samples[(dates >= pd.Timestamp(args.test_start)) & (dates <= pd.Timestamp(args.test_end))].copy()
    if min(len(tr), len(va), len(te)) == 0:
        raise ValueError(f"empty split train={len(tr)} val={len(va)} test={len(te)}")

    train_values = np.concatenate([np.concatenate(tr["x"].to_list()), np.concatenate(tr["y"].to_list())])
    scaler = Scaler(float(np.mean(train_values)), float(np.std(train_values) + 1e-6))
    xtr, ytr = stack_arrays(tr, scaler)
    xva, yva = stack_arrays(va, scaler)
    xte, yte = stack_arrays(te, scaler)
    out_len = ytr.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(hidden=args.hidden, encoder_layers=args.encoder_layers, decoder_layers=1, dropout=0.2).to(device)
    loader = DataLoader(ArrayDataset(xtr, ytr), batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    model, history = train_model(model, loader, torch.as_tensor(xva), torch.as_tensor(yva), out_len,
                                 args.epochs, args.patience, args.lr, args.teacher_forcing, device)
    pred_scaled = predict(model, xte, out_len, device)
    pred_all = scaler.inverse(pred_scaled)
    true_all = scaler.inverse(yte[..., 0])

    if args.mode == "day24":
        pred_eval = pred_all[:, -24:]
        true_eval = true_all[:, -24:]
        metrics = direction_metrics(true_eval, pred_eval)
        rows = []
        for i, day in enumerate(te["target_day"].astype(str)):
            for h in range(1, 25):
                rows.append({"target_day": day, "hour_business": h, "y_true_spread": float(true_eval[i, h-1]), "y_pred_spread": float(pred_eval[i, h-1])})
        ledger = pd.DataFrame(rows)
        ledger["period"] = np.where(ledger["hour_business"] <= 8, "1_8", np.where(ledger["hour_business"] <= 16, "9_16", "17_24"))
        period = []
        for p, g in ledger.groupby("period", sort=False):
            period.append({"period": p, **direction_metrics(g["y_true_spread"].to_numpy(), g["y_pred_spread"].to_numpy())})
        atomic_csv(outdir / "period_metrics.csv", pd.DataFrame(period))
        monthly = []
        ledger["month"] = ledger["target_day"].str.slice(0, 7)
        for m, g in ledger.groupby("month", sort=True):
            monthly.append({"month": m, **direction_metrics(g["y_true_spread"].to_numpy(), g["y_pred_spread"].to_numpy())})
        atomic_csv(outdir / "monthly_metrics.csv", pd.DataFrame(monthly))
    else:
        pred_eval = pred_all[:, 0]
        true_eval = true_all[:, 0]
        metrics = direction_metrics(true_eval, pred_eval)
        ledger = pd.DataFrame({"target_ts": pd.to_datetime(te["target_ts"]).astype(str), "y_true_spread": true_eval, "y_pred_spread": pred_eval})

    atomic_parquet(outdir / "predictions.parquet", ledger)
    atomic_csv(outdir / "training_history.csv", pd.DataFrame(history))
    torch.save(model.state_dict(), outdir / "model.pt")
    manifest = {
        "status": "complete",
        "experiment": "das_2022_seq2seq_shandong_reproduction",
        "paper_doi": "10.1109/ACCESS.2021.3133499",
        "mode": args.mode,
        "paper_anchors": {"lag_hours": 48, "batch_size": args.batch, "optimizer": "Adam", "dropout": 0.2, "paper_epochs": 2000},
        "local_architecture": {"hidden": args.hidden, "encoder_layers": args.encoder_layers, "decoder_layers": 1},
        "splits": {"train": [args.train_start, args.train_end], "validation": [args.val_start, args.val_end], "test": [args.test_start, args.test_end]},
        "samples": {"train": len(tr), "validation": len(va), "test": len(te)},
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
        "metrics": metrics,
        "production_chain_touched": False,
        "runtime_seconds": time.perf_counter() - t0,
    }
    atomic_json(outdir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.mode == "day24":
        print("\nPERIOD")
        print(pd.read_csv(outdir / "period_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
