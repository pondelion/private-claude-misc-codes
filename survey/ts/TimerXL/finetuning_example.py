"""DataFrame window training with causal Timer-XL as an encoder.

The downstream pooling/head is illustrative; the official objective is
multivariate next-patch prediction and is described separately in README.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetuning_common import fit as fit_encoder, run_cli
from ts_block import TimerXLBlock


def build_encoder(in_dim: int, out_dim: int, seq_len: int) -> TimerXLBlock:
    if seq_len % 8:
        raise ValueError("Timer-XL requires seq_len divisible by patch_len=8")
    return TimerXLBlock(in_dim, out_dim, patch_len=8, d_model=64,
                        d_ff=128, heads=4, layers=2)


def fit(df: pd.DataFrame, feature_cols: list[str],
        target_cols: list[str], **kwargs) -> dict:
    """Train [B,T,Cin] -> pooled [B,D] -> next-row GT [B,K]."""
    return fit_encoder(df, feature_cols, target_cols, build_encoder,
                       needs_phase=False, **kwargs)


if __name__ == "__main__":
    run_cli(build_encoder, needs_phase=False)
