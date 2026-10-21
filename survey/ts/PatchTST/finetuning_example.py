"""DataFrame window training with PatchTST as a reusable TS encoder."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetuning_common import fit as fit_encoder, run_cli
from ts_block import PatchTSTBlock


def build_encoder(in_dim: int, out_dim: int, seq_len: int) -> PatchTSTBlock:
    return PatchTSTBlock(in_dim, out_dim, seq_len, patch_len=8, stride=4,
                         d_model=64, d_ff=128, heads=4, layers=2)


def fit(df: pd.DataFrame, feature_cols: list[str],
        target_cols: list[str], **kwargs) -> dict:
    """Train [B,T,Cin] -> pooled [B,D] -> next-row GT [B,K]."""
    return fit_encoder(df, feature_cols, target_cols, build_encoder,
                       needs_phase=False, **kwargs)


if __name__ == "__main__":
    run_cli(build_encoder, needs_phase=False)
