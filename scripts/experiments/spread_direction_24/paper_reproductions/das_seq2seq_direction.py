from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from das_seq2seq_hourly import ROOT, DEFAULT_DATA, Scaler, atomic_csv, atomic_json, atomic_parquet, build_day24, load_hourly


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class DirectionDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


class Seq2SeqDirection(nn.Module):
    """Encoder-decoder direction classifier inspired by Das et al. band forecast.

    The encoder sees only the 48 legal spread observations ending at D-1 14:00.
    Decoder inputs are horizon embeddings, never future realized spread.
    """
    def __init__(self, hidden: int = 64, encoder_layers: int = 2, out_len: int = 34, dropout: float = 0.2):
        super().__init__()
        self.encoder = nn.LSTM(1, hidden, num_layers=encoder_layers, batch_first=True,
                               dropout=(dropout if encoder_layers > 1 else 0.0))
        self.horizon = nn.Embedding(out_len, 16)
        self.bridge_h = nn.Linear(hidden, hidden)
        self.bridge_c = nn.Linear(hidden, hidden)
        self.decoder = nn.LSTM(16, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.out_len = out_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, c) = self.encoder(x)
        h0 = torch.tanh(self.bridge_h(h[-1])).unsqueeze(0)
        c0 = torch.tanh(self.bridge_c(c[-1])).unsqueeze(0)
        idx = torch.arange(self.out_len, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        dec_in = self.horizon(idx)
        z, _ = self.decoder(dec_in, (h0, c0))
        return self.head(self.drop(z)).squeeze(-1)


def stack(frame: pd.DataFrame, scaler: Scaler) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(frame["x"].to_list())
    y = np.stack(frame["y"].to_list())
    x = scaler.transform(x)[..., None].astype(np.float32)
    y_cls = (y > 0).astype(np.float32)
    return x, y_cls


def metrics(y_spread: np.ndarray, prob: np.ndarray) -> dict:
    yt = np.sign(np.asarray(y_spread, float).reshape(-1))
    yp = np.where(np.asarray(prob).reshape(-1) >= 0.5, 1, -1)
    mask = yt != 0
    pos = yt > 0; neg = yt < 0; correct = yt == yp
    pa = float(correct[pos].mean()) if pos.any() else math.nan
    na = float(correct[neg].mean()) if neg.any() else math.nan
    return {
        "n_nonzero": int(mask.sum()),
        "direction_accuracy": float(correct[mask].mean()),
        "positive_accuracy": pa,
        "negative_accuracy": na,
        "balanced_direction_accuracy": float(np.nanmean([pa, na])),
    }


def train(model, loader, xval, yval, epochs, patience, lr, device):
    # Balanced BCE: positive weight from training labels.
    ycat = torch.cat([y for _, y in loader], dim=0)
    pos = float(ycat.sum()); neg = float(ycat.numel() - ycat.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = math.inf; best_state = None; wait = 0; hist = []
    for ep in range(1, epochs + 1):
        model.train(); losses=[]
        for xb, yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            logit=model(xb); loss=loss_fn(logit,yb); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): vl=float(loss_fn(model(xval.to(device)), yval.to(device)).detach().cpu())
        tr=float(np.mean(losses)); hist.append({"epoch":ep,"train_loss":tr,"val_loss":vl})
        if vl+1e-5 < best:
            best=vl; wait=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: wait += 1
        if ep==1 or ep%10==0: print(f"epoch={ep} train={tr:.5f} val={vl:.5f} best={best:.5f}")
        if wait>=patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return model,hist


def predict(model,x,device):
    model.eval(); out=[]; xt=torch.tensor(x,dtype=torch.float32)
    with torch.no_grad():
        for i in range(0,len(xt),256): out.append(torch.sigmoid(model(xt[i:i+256].to(device))).cpu().numpy())
    return np.concatenate(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default=str(DEFAULT_DATA.relative_to(ROOT)))
    ap.add_argument("--output",required=True)
    for n in ("train-start","train-end","val-start","val-end","test-start","test-end"): ap.add_argument("--"+n,required=True)
    ap.add_argument("--epochs",type=int,default=80); ap.add_argument("--patience",type=int,default=12)
    ap.add_argument("--hidden",type=int,default=64); ap.add_argument("--batch",type=int,default=64); ap.add_argument("--lr",type=float,default=1e-3); ap.add_argument("--seed",type=int,default=42)
    a=ap.parse_args(); t0=time.perf_counter(); set_seed(a.seed)
    raw=load_hourly(ROOT/a.data); s=build_day24(raw); dates=pd.to_datetime(s.target_day)
    tr=s[(dates>=a.train_start)&(dates<=a.train_end)].copy(); va=s[(dates>=a.val_start)&(dates<=a.val_end)].copy(); te=s[(dates>=a.test_start)&(dates<=a.test_end)].copy()
    vals=np.concatenate([np.concatenate(tr.x.to_list()),np.concatenate(tr.y.to_list())]); scaler=Scaler(float(vals.mean()),float(vals.std()+1e-6))
    xtr,ytr=stack(tr,scaler); xva,yva=stack(va,scaler); xte,yte=stack(te,scaler)
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=Seq2SeqDirection(hidden=a.hidden,out_len=ytr.shape[1]).to(dev)
    loader=DataLoader(DirectionDataset(xtr,ytr),batch_size=a.batch,shuffle=True,num_workers=0,pin_memory=(dev.type=="cuda"))
    model,hist=train(model,loader,torch.tensor(xva,dtype=torch.float32),torch.tensor(yva,dtype=torch.float32),a.epochs,a.patience,a.lr,dev)
    prob=predict(model,xte,dev); yspread=np.stack(te.y.to_list()); pe=prob[:,-24:]; ye=yspread[:,-24:]
    summary=metrics(ye,pe)
    rows=[]
    for i,day in enumerate(te.target_day.astype(str)):
        for h in range(1,25): rows.append({"target_day":day,"hour_business":h,"y_true_spread":float(ye[i,h-1]),"positive_prob":float(pe[i,h-1]),"pred_direction":1 if pe[i,h-1]>=.5 else -1})
    led=pd.DataFrame(rows); led["period"]=np.where(led.hour_business<=8,"1_8",np.where(led.hour_business<=16,"9_16","17_24")); led["month"]=led.target_day.str[:7]
    per=[]
    for p,g in led.groupby("period",sort=False): per.append({"period":p,**metrics(g.y_true_spread.to_numpy(),g.positive_prob.to_numpy())})
    mon=[]
    for m,g in led.groupby("month",sort=True): mon.append({"month":m,**metrics(g.y_true_spread.to_numpy(),g.positive_prob.to_numpy())})
    out=ROOT/a.output; atomic_parquet(out/"predictions.parquet",led); atomic_csv(out/"period_metrics.csv",pd.DataFrame(per)); atomic_csv(out/"monthly_metrics.csv",pd.DataFrame(mon)); atomic_csv(out/"training_history.csv",pd.DataFrame(hist)); torch.save(model.state_dict(),out/"model.pt")
    man={"status":"complete","experiment":"das_seq2seq_direct_direction_24","paper_inspiration":"Das et al. 2022 band/classification Seq2Seq","forecast_origin":"D-1 14:00","input":"48 observed hourly spreads ending at cutoff","decoder":"34 horizon embeddings; no future realized inputs","target":"34 future direction labels; evaluate last24 canonical business-day directions","splits":{"train":[a.train_start,a.train_end],"validation":[a.val_start,a.val_end],"test":[a.test_start,a.test_end]},"samples":{"train":len(tr),"validation":len(va),"test":len(te)},"device":str(dev),"epochs_ran":len(hist),"metrics":summary,"production_chain_touched":False,"runtime_seconds":time.perf_counter()-t0}; atomic_json(out/"manifest.json",man)
    print(json.dumps(man,ensure_ascii=False,indent=2)); print(pd.DataFrame(per).to_string(index=False))

if __name__=="__main__": main()
