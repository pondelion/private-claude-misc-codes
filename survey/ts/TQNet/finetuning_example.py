"""Chronological DataFrame training with TQNet's measured window phase.

Provide ``phase_col`` containing each row's cycle index; the dataset passes
the phase of the FIRST history observation to TQNetBlock.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetuning_common import fit as fit_encoder, run_cli
from ts_block import TQNetBlock


def build_encoder(in_dim: int, out_dim: int, seq_len: int) -> TQNetBlock:
    if seq_len % 4:
        raise ValueError("TQNet's 4 attention heads require seq_len % 4 = 0")
    return TQNetBlock(in_dim, out_dim, seq_len, cycle_len=24, d_model=64,
                      heads=4)


def fit(df: pd.DataFrame, feature_cols: list[str],
        target_cols: list[str], phase_col: str, **kwargs) -> dict:
    """Fit next-row labels using [B,T,Cin] and first-row phase [B]."""
    return fit_encoder(df, feature_cols, target_cols, build_encoder,
                       phase_col=phase_col, needs_phase=True, **kwargs)


if __name__ == "__main__":
    run_cli(build_encoder, needs_phase=True)
