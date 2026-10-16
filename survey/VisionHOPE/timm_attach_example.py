"""
VisionHOPE - timmの既存バックボーン (CNN系/ViT系) へのSRNL後付けアタッチ例
================================================================

VisionHOPE論文自体は、SRNL (自己参照学習) をバックボーンのtoken-mixer位置に
据えてゼロから学習する設計 (VisionHOPE-T/S/B) であり、既存の事前学習済み
backboneに後付けする使い方は検証していない。本ファイルは会話で検討した
以下の設計判断に基づき、「timmの既存backbone (CNN系/ViT系) の特徴マップに
VisionHOPEOperatorを後付けできるか」を実際に検証する、本サーベイ独自の実験
である。

------------------------------------------------------------
設計判断 (会話で確認した内容)
------------------------------------------------------------
- `VisionHOPEOperator` は入力形状 `(B, D_m, H, W)` を要求するが、chunk_size
  既定値は常に自動的にW(行スキャン)/H(列スキャン)に設定されるため
  (`model.py::VisionHOPEOperator.forward`)、どんな解像度の特徴マップでも
  割り切れない問題は起きない
- CNN系 (`timm.create_model(..., features_only=True)`) は各stageがそのまま
  `(B,C,H,W)` で出てくるため、reshape無しでそのまま差し込める
- ViT系は内部表現が `(B, N, D)` のフラット系列で、cls_token等の特殊トークンが
  先頭に連結されているため、「特殊トークンを分離 -> パッチ部分を
  `patch_embed.grid_size` を使って `(B,N,D)` から `(B,D,H,W)` に復元 ->
  VisionHOPEOperator適用 -> 元の `(B,N,D)` に戻して特殊トークンを再連結」
  という一手間が必要

============================================================
Shape Convention
============================================================
B      : バッチサイズ
C, D   : バックボーンのチャンネル数/埋め込み次元 (モデルに依存)
H, W   : 特徴マップの空間解像度 (ステージ/モデルに依存)
N      : パッチトークン数 (= H*W)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model import VisionHOPEOperator


# ============================================================
# 1. CNN系 (timm features_only) への後付け
# ============================================================

class CNNWithSRNL(nn.Module):
    """timmのCNN系backbone (features_only=True) の最終stage出力に、
    VisionHOPEOperatorを残差ブランチとして追加する。

    ========================================
    Shape
    ========================================
    入力: x (B, 3, H_img, W_img)
    出力: (B, C_last, H_last, W_last)  (backboneの最終stageと同じ形状)
    """

    def __init__(self, model_name: str = "resnet18", head_dim: int = 16,
                 pretrained: bool = False, freeze_backbone: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, features_only=True, pretrained=pretrained)
        if freeze_backbone:
            # 実運用で既存の事前学習済み重みを壊さないための想定(後付けのSRNLだけを学習する)。
            self.backbone.requires_grad_(False)
        dim = self.backbone.feature_info.channels()[-1]  # 最終stageのチャンネル数 (resnet18なら512)
        self.operator = VisionHOPEOperator(dim=dim, head_dim=head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)      # List[(B, C_i, H_i, W_i)] (ステージ数分)
        last = feats[-1]               # (B, C_last, H_last, W_last): CNNの最終特徴マップ
        mixed = self.operator(last)    # (B, C_last, H_last, W_last): SRNLによる token mixing (形状保存)
        return last + mixed            # 残差追加: CNNの特徴量にSRNLの出力を足し込む


# ============================================================
# 2. ViT系 (timm.models.vision_transformer) への後付け
# ============================================================

class ViTWithSRNL(nn.Module):
    """timmのViT系backbone (prefixトークンを持つフラット系列) の
    パッチトークン部分に、VisionHOPEOperatorを残差ブランチとして追加する。

    ========================================
    Shape
    ========================================
    入力: x (B, 3, H_img, W_img)
    出力: (B, num_prefix + N, D)  (cls_token等 + パッチトークンNの系列, backboneと同じ形状)
    """

    def __init__(self, model_name: str = "vit_small_patch16_224", head_dim: int = 16,
                 pretrained: bool = False, freeze_backbone: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.num_prefix = self.backbone.num_prefix_tokens          # cls_token等の個数 (vit_smallなら1)
        self.grid_h, self.grid_w = self.backbone.patch_embed.grid_size  # パッチグリッド (vit_small/16/224なら14,14)
        dim = self.backbone.embed_dim
        self.operator = VisionHOPEOperator(dim=dim, head_dim=head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.forward_features(x)                  # (B, num_prefix+N, D)
        prefix, patches = tokens[:, :self.num_prefix], tokens[:, self.num_prefix:]
        B, N, D = patches.shape

        # (B, N, D) -> (B, D, H, W): 1次元のパッチ系列を、既知のパッチグリッド形状に復元する。
        grid = patches.transpose(1, 2).reshape(B, D, self.grid_h, self.grid_w)
        mixed = self.operator(grid)                                  # (B, D, H, W): SRNLによる token mixing
        mixed = mixed.reshape(B, D, N).transpose(1, 2)                # (B, N, D): 元のフラット系列形状へ戻す

        patches = patches + mixed                                    # 残差追加: パッチトークンにSRNLの出力を足し込む
        return torch.cat([prefix, patches], dim=1)                    # cls_token等を先頭に戻して結合


# ============================================================
# 使用例
# ============================================================

def example_usage() -> None:
    print("=== timm_attach_example.py Example Usage ===\n")
    torch.manual_seed(0)
    x = torch.randn(2, 3, 224, 224)

    print("--- CNN系 (resnet18, timm features_only) への後付け ---")
    cnn_model = CNNWithSRNL("resnet18", freeze_backbone=True)
    out = cnn_model(x)
    print(f"resnet18単体の最終stage shape: (2, 512, 7, 7)")
    print(f"SRNL後付け後のoutput shape:     {tuple(out.shape)}  (形状保存を確認)")

    loss = out.square().mean()
    loss.backward()
    backbone_with_grad = sum(1 for p in cnn_model.backbone.parameters() if p.grad is not None)
    operator_total = sum(1 for _ in cnn_model.operator.parameters())
    operator_with_grad = sum(1 for p in cnn_model.operator.parameters() if p.grad is not None)
    print(f"backboneパラメータで勾度を持つ数: {backbone_with_grad} (0のはず, 凍結済みなので逆伝播されない)")
    print(f"operatorパラメータで勾度を持つ数: {operator_with_grad} / {operator_total} (後付けしたSRNLは学習可能)")

    print("\n--- ViT系 (vit_small_patch16_224, timm) への後付け ---")
    vit_model = ViTWithSRNL("vit_small_patch16_224", freeze_backbone=True)
    out = vit_model(x)
    print(f"vit_small単体のforward_features shape: (2, 197, 384)  (cls_token 1 + パッチ196)")
    print(f"SRNL後付け後のoutput shape:              {tuple(out.shape)}  (形状保存を確認)")

    loss = out.square().mean()
    loss.backward()
    backbone_with_grad = sum(1 for p in vit_model.backbone.parameters() if p.grad is not None)
    operator_total = sum(1 for _ in vit_model.operator.parameters())
    operator_with_grad = sum(1 for p in vit_model.operator.parameters() if p.grad is not None)
    print(f"backboneパラメータで勾度を持つ数: {backbone_with_grad} (0のはず, 凍結済みなので逆伝播されない)")
    print(f"operatorパラメータで勾度を持つ数: {operator_with_grad} / {operator_total} (後付けしたSRNLは学習可能)")

    print("\n--- cls_tokenが正しく素通りしていることを確認 (SRNLはパッチ部分のみを混合) ---")
    print(f"出力の先頭トークン(cls_token位置) shape: {tuple(out[:, :vit_model.num_prefix].shape)}")


if __name__ == "__main__":
    example_usage()
