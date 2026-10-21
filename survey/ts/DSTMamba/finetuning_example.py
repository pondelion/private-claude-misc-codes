"""Real chronological DataFrame-to-forecast fine-tuning example.

Required columns: series_id, timestamp, feature columns, target columns.
Every series is sorted by timestamp. Validation uses later labels only;
training uses earlier labels only. No library installation is performed.
"""
from __future__ import annotations

import argparse
import copy
import pandas as pd
import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.utils.data import Dataset, DataLoader
from ts_block import DSTMambaBlock


class WindowDataset(Dataset):
    """Return history [L,Cin] and future ground truth [H,Ctarget]."""

    def __init__(self, groups, lookback, horizon, split):
        self.groups = groups
        self.lookback = lookback
        self.horizon = horizon
        self.refs = []
        for group_id, (features, targets) in enumerate(groups):
            length = len(features)
            cut = int(length * 0.8)
            # start is the first future label index; histories may cross the split
            # boundary for validation, but labels must be strictly post-cut.
            lo = lookback if split == "train" else max(lookback, cut)
            hi = cut if split == "train" else length
            for start in range(lo, hi - horizon + 1):
                if split == "train" and start + horizon > cut:
                    continue
                self.refs.append((group_id, start))
        if not self.refs:
            raise ValueError(
                f"no {split} windows: reduce lookback/horizon or add rows")

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, index):
        group_id, start = self.refs[index]
        features, targets = self.groups[group_id]
        return (torch.from_numpy(features[start - self.lookback:start].copy()),
                torch.from_numpy(targets[start:start + self.horizon].copy()))

    @property
    def lookback(self):
        return self._lookback

    @lookback.setter
    def lookback(self, value):
        self._lookback = value

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value):
        self._horizon = value


class Forecaster(nn.Module):
    """Reusable feature block followed by a direct multi-horizon forecast head."""

    def __init__(self, in_dim, targets, horizon, lookback):
        super().__init__()
        hidden = 32
        self.block = DSTMambaBlock(in_dim=in_dim,
                                   out_dim=hidden,
                                   seq_len=lookback,
                                   d_model=32,
                                   node_dim=8,
                                   d_state=8,
                                   depth=1,
                                   backend="auto")
        self.head = nn.Linear(hidden, horizon * targets)
        self.horizon, self.targets = horizon, targets

    def forward(self, history):
        # [B,L,Cin] -> [B,L,32] -> final history representation [B,32]
        # -> [B,H,Ctarget]. The future never enters the model input.
        h = self.block(history)[:, -1]
        # [B,H*Ctarget] -> [B,H,Ctarget]; H is the prediction horizon.
        return rearrange(self.head(h),
                         "b (h c) -> b h c",
                         h=self.horizon,
                         c=self.targets)


def prepare_groups(df, feature_cols, target_cols):
    if not feature_cols or not target_cols:
        raise ValueError("feature_cols and target_cols must be nonempty")
    required = {"series_id", "timestamp", *feature_cols, *target_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df[["series_id", "timestamp"]].isna().any().any():
        raise ValueError("series_id/timestamp must not be null")
    groups = []
    for _, g in df.groupby("series_id", sort=False):
        g = g.sort_values("timestamp")
        if g["timestamp"].duplicated().any():
            raise ValueError("duplicate timestamp within series")
        features = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g[target_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(features).all() or not np.isfinite(targets).all():
            raise ValueError(
                "NaN/inf found; impute using training-only statistics")
        groups.append((features, targets))
    if not groups:
        raise ValueError("empty DataFrame")
    return groups


def fit(df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        lookback: int = 64,
        horizon: int = 8,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str | None = None):
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback/horizon must be positive")
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs/batch_size/lr must be positive")
    groups = prepare_groups(df, feature_cols, target_cols)
    # Fit global feature scaling on training rows only; validation statistics
    # never influence either the transform or the optimizer.
    train_features = np.concatenate([x[:int(len(x) * 0.8)] for x, _ in groups])
    mean = train_features.mean(0)
    std = np.maximum(train_features.std(0), 1e-6)
    groups = [((x - mean).astype(np.float32) / std, y) for x, y in groups]
    train = WindowDataset(groups, lookback, horizon, "train")
    valid = WindowDataset(groups, lookback, horizon, "valid")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Forecaster(len(feature_cols), len(target_cols), horizon,
                       lookback).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        for history, future in train_loader:
            history, future = history.to(device), future.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(history)
            loss = nn.functional.mse_loss(pred, future)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for history, future in valid_loader:
                pred = model(history.to(device))
                total += nn.functional.mse_loss(pred,
                                                future.to(device),
                                                reduction="sum").item()
                count += future.numel()
        val_mse = total / count
        print(f"epoch={epoch+1} val_mse={val_mse:.6g}")
        if val_mse < best_loss:
            best_loss, best_state = val_mse, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, {
        "feature_mean": mean,
        "feature_std": std,
        "val_mse": best_loss
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv",
                        help="CSV with series_id,timestamp,features,targets")
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--lookback", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    fit(pd.read_csv(args.csv),
        args.features,
        args.targets,
        lookback=args.lookback,
        horizon=args.horizon,
        epochs=args.epochs)
