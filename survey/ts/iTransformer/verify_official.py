"""Compare raw input through the unchanged official encoder boundary.

The official forecast projection is a task head and is not compared.
Run with --official-repo pointing at the iTransformer repository.
"""
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

from ts_block import ITransformerBlock


def load_definitions(path: Path, names: set[str], namespace: dict) -> None:
    """Execute official class bodies unchanged; skip unrelated imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError(f"missing official class in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
         namespace)


def main(repo: Path) -> None:
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np,
                 "sqrt": math.sqrt}
    load_definitions(repo / "layers/Embed.py", {"DataEmbedding_inverted"},
                     namespace)
    load_definitions(repo / "layers/SelfAttention_Family.py",
                     {"FullAttention", "AttentionLayer"}, namespace)
    load_definitions(repo / "layers/Transformer_EncDec.py",
                     {"EncoderLayer", "Encoder"}, namespace)
    load_definitions(repo / "model/iTransformer.py", {"Model"}, namespace)
    config = SimpleNamespace(seq_len=16, pred_len=8, output_attention=False,
                             use_norm=True, embed="fixed", freq="h",
                             dropout=0.0, class_strategy="projection",
                             factor=5, d_model=16, n_heads=4, d_ff=32,
                             activation="gelu", e_layers=2)
    official = namespace["Model"](config).eval()
    candidate = ITransformerBlock(3, 5, seq_len=16, d_model=16, d_ff=32,
                                  heads=4, layers=2, dropout=0).eval()
    candidate.embedding.load_state_dict(
        official.enc_embedding.value_embedding.state_dict())
    candidate.norm.load_state_dict(official.encoder.norm.state_dict())
    for own, ref in zip(candidate.layers, official.encoder.attn_layers):
        attn = ref.attention
        for name, source in (("query", attn.query_projection),
                             ("key", attn.key_projection),
                             ("value", attn.value_projection),
                             ("output", attn.out_projection)):
            getattr(own.attention, name).load_state_dict(source.state_dict())
        for name in ("conv1", "conv2", "norm1", "norm2"):
            getattr(own, name).load_state_dict(
                getattr(ref, name).state_dict())
    captured = []
    hook = official.encoder.register_forward_hook(
        lambda _module, _inputs, result: captured.append(result[0]))
    try:
        x = torch.randn(2, 16, 3)
        official(x, None, None, None)
    finally:
        hook.remove()
    torch.testing.assert_close(candidate.forward_features(x), captured[0],
                               atol=1e-6, rtol=1e-5)
    print("iTransformer input-to-encoder output parity PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True, type=Path)
    main(parser.parse_args().official_repo)
