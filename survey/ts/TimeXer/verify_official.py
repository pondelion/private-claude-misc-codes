"""Compare the released multivariate input-to-encoder path without its head."""
from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from ts_block import TimeXerBlock


def classes(path: Path, names: set[str], namespace: dict) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError(f"missing official class in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
         namespace)


def attention_weights(ours: nn.Module, reference: nn.Module) -> None:
    for own_name, ref_name in (("query", "query_projection"),
                               ("key", "key_projection"),
                               ("value", "value_projection"),
                               ("output", "out_projection")):
        getattr(ours, own_name).load_state_dict(
            getattr(reference, ref_name).state_dict())


def main(repo: Path) -> None:
    env = {"torch": torch, "nn": nn, "F": F,
           "np": np, "sqrt": math.sqrt, "math": math}
    classes(repo / "layers/Embed.py",
            {"PositionalEmbedding", "DataEmbedding_inverted"}, env)
    classes(repo / "layers/SelfAttention_Family.py",
            {"FullAttention", "AttentionLayer"}, env)
    classes(repo / "models/TimeXer.py",
            {"FlattenHead", "EnEmbedding", "Encoder", "EncoderLayer",
             "Model"}, env)
    config = SimpleNamespace(task_name="long_term_forecast", features="M",
                             seq_len=32, pred_len=8, use_norm=True,
                             patch_len=8, enc_in=3, d_model=16, d_ff=32,
                             n_heads=4, e_layers=2, dropout=0.0,
                             embed="fixed", freq="h", factor=5,
                             activation="gelu")
    official = env["Model"](config).eval()
    ours = TimeXerBlock(3, 5, 32, patch_len=8, d_model=16,
                         d_ff=32, heads=4, layers=2).eval()
    ours.patch_embed.load_state_dict(
        official.en_embedding.value_embedding.state_dict())
    ours.exogenous_embed.load_state_dict(
        official.ex_embedding.value_embedding.state_dict())
    ours.norm.load_state_dict(official.encoder.norm.state_dict())
    with torch.no_grad():
        ours.global_token.copy_(official.en_embedding.glb_token)
    for own, ref in zip(ours.layers, official.encoder.layers):
        attention_weights(own.self_attention, ref.self_attention)
        attention_weights(own.cross_attention, ref.cross_attention)
        for name in ("norm1", "norm2", "norm3", "conv1", "conv2"):
            getattr(own, name).load_state_dict(
                getattr(ref, name).state_dict())
    captured = []
    hook = official.encoder.register_forward_hook(
        lambda _module, _inputs, result: captured.append(result))
    try:
        sample = torch.randn(2, 32, 3)
        official.forecast_multi(sample, None, None, None)
    finally:
        hook.remove()
    expected = rearrange(captured[0], "(b c) n d -> b c d n", c=3)
    torch.testing.assert_close(ours.forward_features(sample), expected,
                               atol=1e-5, rtol=1e-5)
    print("TimeXer multivariate input-to-encoder parity PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    main(parser.parse_args().official_repo)
