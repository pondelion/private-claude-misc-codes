"""画像patchをLoopVL hidden幅へ接続するvisual interface。実行例は``main_flow.py``。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class TinyFrozenVisionEncoder(nn.Module):
    """教材用の凍結patch encoder。``(B,3,H,W)->(B,N,Cv)``。

    公式のPenguin Vision Encoderそのものではないが、画像を実際にpatch特徴へ変換する。
    pretrained Penguinの能力は再現対象外で、後段のLoopVL処理を実行可能にするための小型代替である。
    """

    def __init__(self, vision_width: int = 64, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(3, vision_width, patch_size, stride=patch_size)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.patch_embed(images)                 # (B,Cv,Gh,Gw)
        return features.flatten(2).transpose(1, 2)          # (B,N=Gh*Gw,Cv)


class VisionProjector(nn.Module):
    """公式と同じLinear-GELU-Linear + parameter-free RMS calibration。

    ``features: (B,N,1024) -> embeddings: (B,N,1536)``が本モデルのshape。
    引数を小さくすれば教材版も同じ処理で動く。
    """

    def __init__(self, vision_width: int, model_width: int, target_rms: float = 1.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(vision_width, model_width)
        self.fc2 = nn.Linear(model_width, model_width)
        self.register_buffer("target_rms", torch.tensor(float(target_rms)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        projected = self.fc2(F.gelu(self.fc1(features)))
        normalized = F.rms_norm(projected, (projected.shape[-1],))
        return normalized * self.target_rms.to(projected.dtype)
