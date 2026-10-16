"""LoopVLの画像forward、response CE、backwardを一巡する。実行: ``python main_flow.py``。"""

from __future__ import annotations

import torch

from loopvl_model import TinyLoopVL, response_cross_entropy


def main() -> None:
    torch.manual_seed(3)
    model = TinyLoopVL(vocab_size=80, width=48, vision_width=32, heads=4, layers_per_stack=1)
    batch, sequence = 2, 25
    images = torch.randn(batch, 3, 32, 32)                  # (B,3,H,W)
    ids = torch.randint(1, 80, (batch, sequence))           # (B,S)
    valid = torch.ones(batch, sequence, dtype=torch.bool)
    visual = torch.zeros_like(valid); visual[:, 2:18] = True # N=4*4 visual slots
    instruction = torch.zeros_like(valid); instruction[:, 18:21] = True
    response = torch.zeros_like(valid); response[:, 21:] = True
    types = (~response).long()                               # prefix=1, response=0
    output = model(images, ids, types, valid, visual, instruction)
    loss = response_cross_entropy(output.logits, ids, response)
    loss.backward()
    assert output.logits.shape == (batch, sequence, 80)
    assert len(output.recurrent.invocation_names) == 8       # L,L,L,Hを2 cycle
    assert model.projector.fc1.weight.grad is not None
    assert model.backbone.L.layers[0].attn.qkv.weight.grad is not None
    assert model.backbone.H.layers[0].attn.qkv.weight.grad is not None
    print({"loss": float(loss.detach()), "logits": tuple(output.logits.shape),
           "schedule": output.recurrent.invocation_names,
           "visual_update_rms": [round(float(x.detach()), 4) for x in output.recurrent.visual_update_rms]})


if __name__ == "__main__":
    main()
