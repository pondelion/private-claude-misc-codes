"""Chronological DataFrame example with iTransformer as a reusable encoder.

The shared training loop adds an illustrative pooling/head and supports
regression or classification. It does not reuse the paper's forecast head.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetuning_common import fit as fit_encoder, run_cli
from ts_block import ITransformerBlock


def build_encoder(in_dim: int, out_dim: int,
                  seq_len: int) -> ITransformerBlock:
    return ITransformerBlock(in_dim, out_dim, seq_len, d_model=64,
                             d_ff=128, heads=4, layers=2)


def fit(df: pd.DataFrame, feature_cols: list[str],
        target_cols: list[str], **kwargs) -> dict:
    """Fit on [B,T,Cin] windows and next-row [B,K] targets."""
    return fit_encoder(df, feature_cols, target_cols, build_encoder,
                       needs_phase=False, **kwargs)


if __name__ == "__main__":
    run_cli(build_encoder, needs_phase=False)
