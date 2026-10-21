"""Shared, task-neutral DataFrame training example for the six TS surveys.

Each model-specific finetuning_example.py supplies only its encoder builder.
This file contains a real chronological dataset, pooling/head, optimization,
validation, and a CSV CLI.  No pretrained library or package install is used.
"""
from __future__ import annotations

import argparse
import copy
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class WindowDataset(Dataset):
    """History [T,Cin], next-row label [K] or scalar, phase scalar."""

    def __init__(self, groups: list[tuple[np.ndarray, np.ndarray,
                                          np.ndarray | None]], seq_len: int,
                 split: str) -> None:
        self.groups = groups
        self.seq_len = seq_len
        self.refs: list[tuple[int, int]] = []
        if split not in {"train", "valid"}:
            raise ValueError("split must be train or valid")
        for group_id, (features, _, _) in enumerate(groups):
            cutoff = int(len(features) * 0.8)
            # label_index is strictly later than the last input time step.
            start = seq_len if split == "train" else max(seq_len, cutoff)
            stop = cutoff if split == "train" else len(features)
            self.refs.extend(
                (group_id, label_index) for label_index in range(start, stop))
        if not self.refs:
            raise ValueError(f"no {split} windows; add data or reduce seq_len")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int):
        group_id, label_index = self.refs[index]
        features, targets, phases = self.groups[group_id]
        first = label_index - self.seq_len
        history = torch.from_numpy(features[first:label_index].copy())
        target = torch.from_numpy(np.array(targets[label_index], copy=True))
        phase = 0 if phases is None else int(phases[first])
        return history, target, torch.tensor(phase, dtype=torch.long)


class PooledTaskModel(nn.Module):
    """Encoder [B,T,Cin]->[B,T,D], mean pool -> task head [B,K]."""

    def __init__(self,
                 encoder: nn.Module,
                 hidden_dim: int,
                 num_outputs: int,
                 needs_phase: bool = False) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_dim, num_outputs)
        self.needs_phase = needs_phase

    def forward(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if self.needs_phase:
            h = self.encoder(x, phase=phase)  # [B,T,D]
        else:
            h = self.encoder(x)  # [B,T,D]
        pooled = h.mean(dim=1)  # [B,D]
        return self.head(pooled)  # [B,K]


def prepare_groups(df: pd.DataFrame, feature_cols: list[str],
                   target_cols: list[str], task: str, phase_col: str | None):
    """Sort each series and preserve label alignment without future inputs."""
    if not feature_cols or not target_cols:
        raise ValueError("feature_cols and target_cols must be nonempty")
    if task == "classification" and len(target_cols) != 1:
        raise ValueError("classification requires exactly one label column")
    required = {"series_id", "timestamp", *feature_cols, *target_cols}
    if phase_col is not None:
        required.add(phase_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df[list(required)].isna().any().any():
        raise ValueError("missing values must be handled before training")
    groups = []
    for _, frame in df.groupby("series_id", sort=False):
        frame = frame.sort_values("timestamp")
        if frame["timestamp"].duplicated().any():
            raise ValueError("duplicate timestamps within a series")
        x = frame[feature_cols].to_numpy(dtype=np.float32)
        if task == "regression":
            y = frame[target_cols].to_numpy(dtype=np.float32)
        else:
            y = frame[target_cols[0]].to_numpy(dtype=np.int64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("infinite values are not supported")
        phase = (None if phase_col is None else frame[phase_col].to_numpy(
            dtype=np.int64))
        groups.append((x, y, phase))
    if not groups:
        raise ValueError("empty DataFrame")
    return groups


def fit(df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        encoder_builder: Callable[[int, int, int], nn.Module],
        seq_len: int = 64,
        hidden_dim: int = 32,
        task: str = "regression",
        num_classes: int | None = None,
        phase_col: str | None = None,
        needs_phase: bool = False,
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str | None = None) -> dict:
    """Train on early next-row labels and validate on later labels.

    Shapes: input [B,T,Cin]; encoder [B,T,D]; pooled [B,D]; prediction
    [B,Ctarget] (regression) or [B,K] (classification).  The training
    objective and head belong here, never inside the reusable encoder.
    """
    if task not in {"regression", "classification"}:
        raise ValueError("task must be regression or classification")
    if min(seq_len, hidden_dim, epochs, batch_size) < 1 or lr <= 0:
        raise ValueError(
            "positive seq_len, hidden_dim, epochs, batch_size, lr required")
    if task == "classification" and (num_classes is None or num_classes < 2):
        raise ValueError("classification needs num_classes >= 2")
    groups = prepare_groups(df, feature_cols, target_cols, task, phase_col)
    train_features = np.concatenate(
        [x[:int(len(x) * 0.8)] for x, _, _ in groups], axis=0)
    if len(train_features) == 0:
        raise ValueError("empty training partition")
    # No validation row contributes to input-scaling statistics.
    mean = train_features.mean(axis=0)
    std = np.maximum(train_features.std(axis=0), 1e-6)
    groups = [((x - mean).astype(np.float32) / std, y, phase)
              for x, y, phase in groups]
    train = WindowDataset(groups, seq_len, "train")
    valid = WindowDataset(groups, seq_len, "valid")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    outputs = len(target_cols) if task == "regression" else num_classes
    encoder = encoder_builder(len(feature_cols), hidden_dim, seq_len)
    model = PooledTaskModel(encoder,
                            hidden_dim,
                            outputs,
                            needs_phase=needs_phase).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
    best_loss, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        for history, target, phase in train_loader:
            history = history.to(device)  # [B,T,Cin]
            target = target.to(device)  # [B,Ctarget] or [B]
            phase = phase.to(device)  # [B]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history, phase)
            loss = criterion(prediction, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for history, target, phase in valid_loader:
                prediction = model(history.to(device), phase.to(device))
                target = target.to(device)
                if task == "regression":
                    total += nn.functional.mse_loss(prediction,
                                                    target,
                                                    reduction="sum").item()
                    count += target.numel()
                else:
                    total += nn.functional.cross_entropy(
                        prediction, target, reduction="sum").item()
                    count += len(target)
        valid_loss = total / count
        print(f"epoch={epoch + 1} valid_{task}_loss={valid_loss:.6g}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_valid_loss": best_loss,
        "feature_mean": mean,
        "feature_std": std,
        "seq_len": seq_len,
        "task": task
    }


def run_cli(builder: Callable[[int, int, int], nn.Module],
            needs_phase: bool = False) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--task",
                        choices=["regression", "classification"],
                        default="regression")
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--phase-col")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    fit(pd.read_csv(args.csv),
        args.features,
        args.targets,
        builder,
        seq_len=args.seq_len,
        task=args.task,
        num_classes=args.num_classes,
        phase_col=args.phase_col,
        needs_phase=needs_phase,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr)
