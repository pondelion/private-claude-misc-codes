"""Compare raw input through official PatchTST backbone and H=T readout."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from ts_block import PatchTSTBlock


def main(repo: Path) -> None:
    sys.path.insert(0, str(repo))
    try:
        from layers.PatchTST_backbone import PatchTST_backbone
    finally:
        sys.path.pop(0)
    official = PatchTST_backbone(
        c_in=3, context_window=32, target_window=32, patch_len=8,
        stride=4, n_layers=2, d_model=16, n_heads=4, d_ff=32,
        norm="BatchNorm", attn_dropout=0.0, dropout=0.0,
        res_attention=True, pe="zeros", learn_pe=True,
        padding_patch="end", revin=True, affine=True).eval()
    ours = PatchTSTBlock(3, 3, 32, 8, 4, d_model=16, d_ff=32,
                         heads=4, layers=2).eval()
    with torch.no_grad():
        ours.affine_weight.copy_(official.revin_layer.affine_weight)
        ours.affine_bias.copy_(official.revin_layer.affine_bias)
        ours.position.copy_(official.backbone.W_pos)
        ours.output.weight.copy_(torch.eye(3))
        ours.output.bias.zero_()
    ours.embedding.load_state_dict(official.backbone.W_P.state_dict())
    ours.time_readout.load_state_dict(official.head.linear.state_dict())
    for own, ref in zip(ours.layers, official.backbone.encoder.layers):
        for own_name, ref_name in (("query", "W_Q"), ("key", "W_K"),
                                   ("value", "W_V"), ("output", "to_out")):
            getattr(own.attention, own_name).load_state_dict(
                getattr(ref.self_attn, ref_name).state_dict())
        for own_name, ref_name in (("norm_attn", "norm_attn"),
                                   ("norm_ffn", "norm_ffn"), ("ff", "ff")):
            target = getattr(own, own_name)
            source = getattr(ref, ref_name)
            if own_name.startswith("norm"):
                source = source[1]
            target.load_state_dict(source.state_dict())
    captured = []
    hook = official.backbone.register_forward_hook(
        lambda _module, _inputs, value: captured.append(value))
    try:
        sample = torch.randn(2, 32, 3)
        expected = official(sample.transpose(1, 2)).transpose(1, 2)
    finally:
        hook.remove()
    torch.testing.assert_close(ours.forward_features(sample), captured[0],
                               atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(ours(sample), expected,
                               atol=1e-5, rtol=1e-5)
    print("PatchTST input-to-backbone and same-length output parity PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    main(parser.parse_args().official_repo)
