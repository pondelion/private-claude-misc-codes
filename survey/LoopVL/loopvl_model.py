"""vision interfaceとrecurrent backboneを結ぶ理解用LoopVL。実行例: ``python main_flow.py``。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from attention import build_rope_positions, prefix_lm_attention
from recurrent_backbone import LoopVLBackbone, RecurrentOutput
from visual_interface import TinyFrozenVisionEncoder, VisionProjector


@dataclass
class LoopVLOutput:
    logits: torch.Tensor                 # (B,S,V)
    recurrent: RecurrentOutput


class TinyLoopVL(nn.Module):
    """画像を実際にpatch化し、visual slotへ挿入してresponse logitsを返す。"""

    def __init__(self, vocab_size: int, image_size: int = 32, patch_size: int = 8,
                 vision_width: int = 48, width: int = 64, heads: int = 4,
                 layers_per_stack: int = 2, h_cycles: int = 2, l_cycles: int = 3) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid_height = self.grid_width = image_size // patch_size
        self.visual_tokens = self.grid_height * self.grid_width
        self.vision = TinyFrozenVisionEncoder(vision_width, patch_size)
        self.projector = VisionProjector(vision_width, width)
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.backbone = LoopVLBackbone(width, heads, layers_per_stack, h_cycles, l_cycles)
        self.final_norm, self.lm_head = nn.RMSNorm(width), nn.Linear(width, vocab_size, bias=False)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                token_types: torch.Tensor, valid: torch.Tensor,
                visual_mask: torch.Tensor, instruction_mask: torch.Tensor) -> LoopVLOutput:
        """主なshape: image ``(B,3,H,W)``, ids/masks ``(B,S)``, logits ``(B,S,V)``。"""
        text = self.token_embedding(input_ids)                 # (B,S,D)
        with torch.no_grad():
            visual_features = self.vision(images)              # (B,N,Cv)
        visual = self.projector(visual_features)               # (B,N,D)
        if not torch.all(visual_mask.sum(1) == self.visual_tokens):
            raise ValueError("visual slot count must match patch count")
        embeddings = text.clone()
        embeddings[visual_mask] = visual.reshape(-1, visual.shape[-1])
        allowed = prefix_lm_attention(token_types, valid)       # (B,S,S)
        positions = build_rope_positions(token_types, visual_mask, self.grid_height, self.grid_width)
        recurrent = self.backbone(embeddings, allowed, positions, visual_mask, instruction_mask)
        return LoopVLOutput(self.lm_head(self.final_norm(recurrent.hidden)), recurrent)


def response_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor,
                           response_mask: torch.Tensor) -> torch.Tensor:
    """responseだけのteacher-forced next-token CE。

    ``logits:(B,S,V)``, ``input_ids/response_mask:(B,S)``。position ``i-1``の
    logitsでtoken ``i``を予測し、header/image/instruction/paddingはlossから除外する。
    """
    valid_targets = response_mask[:, 1:]
    if not valid_targets.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[:, :-1][valid_targets], input_ids[:, 1:][valid_targets])
