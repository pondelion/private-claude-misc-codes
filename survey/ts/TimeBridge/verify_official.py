"""Compare raw input to the unchanged released TimeBridge encoder boundary."""
from __future__ import annotations

import argparse
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from ts_block import TimeBridgeBlock, period_norm


def load_source(path: Path, names: set[str], namespace: dict) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError(f"missing official definition in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
         namespace)


def copy_common(ours: nn.Module, official: nn.Module) -> None:
    for name in ("fc1", "fc2", "norm1", "norm2"):
        getattr(ours, name).load_state_dict(
            getattr(official, name).state_dict())
    for name in ("attention", "attention1", "attention2",
                 "conv1", "conv2"):
        if hasattr(ours, name):
            getattr(ours, name).load_state_dict(
                getattr(official, name).state_dict())


def main(repo: Path) -> None:
    env = {"torch": torch, "nn": nn, "F": F,
           "rearrange": rearrange, "math": math, "copy": copy,
           "sqrt": math.sqrt}
    load_source(repo / "layers/SelfAttention_Family.py",
                {"TSMixer", "ResAttention"}, env)
    load_source(repo / "layers/Transformer_EncDec.py",
                {"TSEncoder", "PeriodNorm", "IntAttention",
                 "PatchSampling", "CointAttention"}, env)
    load_source(repo / "layers/Embed.py", {"PatchEmbed"}, env)
    load_source(repo / "model/TimeBridge.py", {"Model"}, env)
    config = SimpleNamespace(revin=True, enc_in=3, period=8, seq_len=32,
                             pred_len=16, num_p=2, d_model=16, d_ff=32,
                             n_heads=4, ia_layers=1, pd_layers=1,
                             ca_layers=1, stable_len=4, attn_dropout=0.0,
                             dropout=0.0, activation="gelu")
    official = env["Model"](config).eval()
    ours = TimeBridgeBlock(3, 5, 32, period=8, num_p=2, d_model=16,
                           d_ff=32, heads=4, ia_layers=1, pd_layers=1,
                           ca_layers=1, stable_len=4).eval()
    ours.embedding.load_state_dict(official.embedding.proj[0].state_dict())
    for own, ref in zip(ours.layers, official.encoder.attn_layers):
        copy_common(own, ref)
    # Confirm the feature-normalization operation independently; it is used
    # in no_grad for Integrated Attention queries and keys.
    probe = torch.randn(6, 4, 16)
    torch.testing.assert_close(period_norm(probe, 4),
                               env["PeriodNorm"](probe, 4),
                               atol=1e-6, rtol=1e-6)
    captured = []
    hook = official.encoder.register_forward_hook(
        lambda _module, _inputs, result: captured.append(result[0]))
    try:
        sample = torch.randn(2, 32, 3)
        official(sample, None, None, None)
    finally:
        hook.remove()
    torch.testing.assert_close(ours.forward_features(sample), captured[0],
                               atol=1e-5, rtol=1e-5)
    print("TimeBridge input-to-encoder output parity PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True, type=Path)
    main(parser.parse_args().official_repo)
