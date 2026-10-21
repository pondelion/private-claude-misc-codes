"""Compare raw input through the unchanged official TQNet pre-head output."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch

from ts_block import TQNetBlock


def main(repo: Path) -> None:
    path = repo / "models/TQNet.py"
    spec = importlib.util.spec_from_file_location("official_tqnet", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = SimpleNamespace(seq_len=16, pred_len=8, enc_in=3, cycle=12,
                             model_type="mlp", d_model=16, dropout=0.0,
                             use_revin=True)
    official = module.Model(config).eval()
    candidate = TQNetBlock(3, 5, 16, 12, d_model=16,
                           heads=4, attention_dropout=0.5).eval()
    with torch.no_grad():
        candidate.temporal_query.copy_(official.temporalQuery)
    candidate.channel_attention.load_state_dict(
        official.channelAggregator.state_dict())
    candidate.input_proj.load_state_dict(official.input_proj.state_dict())
    candidate.model.load_state_dict(official.model.state_dict())
    embedded, hidden = [], []
    h1 = official.input_proj.register_forward_hook(
        lambda _module, _inputs, value: embedded.append(value))
    h2 = official.model.register_forward_hook(
        lambda _module, _inputs, value: hidden.append(value))
    try:
        x = torch.randn(2, 16, 3)
        phase = torch.tensor([0, 5])
        official(x, phase)
    finally:
        h1.remove()
        h2.remove()
    torch.testing.assert_close(candidate.forward_features(x, phase),
                               embedded[0] + hidden[0], atol=1e-6,
                               rtol=1e-5)
    print("TQNet input-to-pre-head encoder parity PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True, type=Path)
    main(parser.parse_args().official_repo)
