"""Compare Timer-XL raw input through official OpenLTM encoder output.

Use the official OpenLTM checkout indicated by Timer-XL's README. The test
does not replace its attention, rotary projection, bias, mask, or TimerLayer.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from ts_block import TimerXLBlock


def main(repo: Path) -> None:
    sys.path.insert(0, str(repo))
    try:
        from models.timer_xl import Model
    finally:
        sys.path.pop(0)
    for covariate in (False, True):
        config = SimpleNamespace(
            input_token_len=8, output_token_len=8, output_attention=False,
            d_model=32, d_ff=64, n_heads=4, dropout=0.0,
            covariate=covariate, flash_attention=False,
            activation="gelu", e_layers=2, use_norm=True)
        official = Model(config).eval()
        candidate = TimerXLBlock(3, 5, patch_len=8, d_model=32,
                                 d_ff=64, heads=4, layers=2,
                                 covariate=covariate).eval()
        candidate.embedding.load_state_dict(official.embedding.state_dict())
        candidate.norm.load_state_dict(official.blocks.norm.state_dict())
        for own, ref in zip(candidate.layers, official.blocks.attn_layers):
            for name in ("conv1", "conv2", "norm1", "norm2"):
                getattr(own, name).load_state_dict(
                    getattr(ref, name).state_dict())
            inner = ref.attention
            for own_name, official_name in (
                    ("query", "query_projection"),
                    ("key", "key_projection"),
                    ("value", "value_projection"),
                    ("output", "out_projection")):
                getattr(own.attention, own_name).load_state_dict(
                    getattr(inner, official_name).state_dict())
            candidate_bias = own.attention.binary_bias
            reference_bias = inner.inner_attention.attn_bias.emb
            candidate_bias.load_state_dict(reference_bias.state_dict())
        captured = []
        hook = official.blocks.register_forward_hook(
            lambda _module, _inputs, result: captured.append(result[0]))
        try:
            sample = torch.randn(2, 32, 3)
            official(sample, None, None)
        finally:
            hook.remove()
        torch.testing.assert_close(candidate.forward_features(sample),
                                   captured[0], atol=1e-5, rtol=1e-5)
    print("Timer-XL input-to-encoder output parity PASS (both masks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True, type=Path)
    main(parser.parse_args().official_repo)
