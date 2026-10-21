"""DataFrame example: WFTNet encoder -> mean pooling -> task head.

This model-specific adapter uses the shared chronological training loop in
../finetuning_common.py.  All task loss/head logic remains outside ts_block.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetuning_common import fit as fit_encoder, run_cli
from ts_block import WFTNetBlock


def build_encoder(in_dim: int, out_dim: int, seq_len: int) -> WFTNetBlock:
    """Construct the reusable [B,T,Cin] -> [B,T,Cout] encoder."""
    return WFTNetBlock(in_dim, out_dim, d_model=32, d_ff=64, layers=1,
                       top_k=2, seq_len=seq_len)


def fit(df: pd.DataFrame, feature_cols: list[str], target_cols: list[str],
        **kwargs) -> dict:
    """Fit a one-step regression/classification example from sorted windows."""
    return fit_encoder(df,
                       feature_cols,
                       target_cols,
                       build_encoder,
                       needs_phase=False,
                       **kwargs)


if __name__ == "__main__":
    run_cli(build_encoder, needs_phase=False)
