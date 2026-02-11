# Case 7: DPT-Style Decoder

## 概要

Dense Prediction Transformer (DPT) の論文に基づいたデコーダ実装です。
ViTエンコーダからの特徴をマルチスケールピラミッドに変換し、RefineNet風のFusion Blockで段階的に融合・アップサンプリングします。

**参考論文**: "Vision Transformers for Dense Prediction" (Ranftl et al., 2021)
- 論文: https://arxiv.org/abs/2103.13413
- 公式実装: https://github.com/intel-isl/DPT

## アーキテクチャ

```
ViT Encoder Features (all at 1/16 resolution)
    │
    ├─ Layer 1 (Early)  ─────┬─ Reassemble (4x upsample) ─► 1/4 res
    ├─ Layer 2              ─┼─ Reassemble (2x upsample) ─► 1/8 res
    ├─ Layer 3              ─┼─ Reassemble (no change)   ─► 1/16 res
    └─ Layer 4 (Deep)       ─┴─ Reassemble (2x downsample)─► 1/32 res
                                        │
    ┌───────────────────────────────────┘
    │
    ▼  Progressive Fusion (Deep → Shallow)
    ┌─────────────────────────────────────────────────┐
    │ FusionBlock4: layer_4_rn                        │
    │     ↓ (2x upsample)                             │
    │ FusionBlock3: path_4 + layer_3_rn               │
    │     ↓ (2x upsample)                             │
    │ FusionBlock2: path_3 + layer_2_rn               │
    │     ↓ (2x upsample)                             │
    │ FusionBlock1: path_2 + layer_1_rn               │
    │     ↓ (2x upsample)                             │
    └─────────────────────────────────────────────────┘
    │
    ▼  Final Upsampling
    Output (Full Resolution)
```

## 主要コンポーネント

### 1. Reassemble Module

ViT特徴をマルチスケール解像度に変換します。
- **Channel Projection**: 1x1 Convでチャネル数を統一（デフォルト256）
- **Spatial Resampling**: ConvTranspose2dまたはStrided Conv2dで解像度を調整

```python
# timmのViTはすでに(B, C, H, W)形式で返すため、
# 論文のREAD/CONCATENATEはスキップ
```

### 2. ResidualConvUnit (RCU)

RefineNetから採用された残差畳み込みユニットです。

```
Input
  │
  ├──────────────────────────┐
  ▼                          │
ReLU → Conv3x3 → (BN)        │
  ▼                          │
ReLU → Conv3x3 → (BN)        │
  ▼                          │
  + ◄────────────────────────┘
  │
Output
```

### 3. FeatureFusionBlock

DPT/RefineNet風の特徴融合ブロックです。

```
path (from deeper)    skip (from same level ViT)
      │                        │
      │                   ResConfUnit1
      │                        │
      └────────────► + ◄───────┘
                     │
               ResConfUnit2
                     │
               2x Upsample (bilinear)
                     │
                 Conv1x1
                     │
                  Output
```

## 2つのバリアント

### ViTDecoderDPT (フルバージョン)

論文に忠実な実装。マルチスケールピラミッドを構築してから融合します。

```python
decoder = ViTDecoderDPT(
    encoder_channels=[384, 384, 384, 384],  # 4段階
    decoder_channels=256,
    use_bn=True,
    final_upsampling=16
)
```

**特徴**:
- マルチスケール特徴ピラミッド (1/4, 1/8, 1/16, 1/32)
- より多くのパラメータ
- 元のDPT論文に近い動作

### ViTDecoderDPTSimplified (簡略版)

ViTの特性（全層同一解像度）を活かした効率的な実装。

```python
decoder = ViTDecoderDPTSimplified(
    encoder_channels=[384, 384, 384],  # 3段階でもOK
    decoder_channels=256,
    use_bn=True,
    final_upsampling=16
)
```

**特徴**:
- リサンプリングなしで直接融合
- 少ないパラメータ
- 高速な推論

## パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `encoder_channels` | エンコーダ出力チャネルリスト | 必須 |
| `decoder_channels` | デコーダ特徴チャネル数 | 256 |
| `use_bn` | BatchNormの使用 | True |
| `upsampling_mode` | 最終アップサンプリング方式 | 'bilinear' |
| `final_upsampling` | 最終アップサンプリング倍率 | 2 (DPT) / 8 (Simplified) |

### final_upsamplingについて

| 値 | 出力解像度 | 説明 |
|----|-----------|------|
| 2 (DPT default) | 1/2 | 論文通り。Output Headでフル解像度へ |
| 8 (Simplified default) | 1/2 | 1/16 → 1/2 |
| 16 | フル解像度 | メモリ使用量に注意 |

## 使用例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from case7_dpt_style.decoder import ViTDecoderDPT, ViTDecoderDPTSimplified
import timm

# ViTエンコーダ
encoder = timm.create_model(
    'vit_small_patch16_224',
    pretrained=True,
    features_only=True,
    out_indices=(3, 6, 9, 11)  # 4段階の特徴を取得
)

# DPTデコーダ（デフォルトで1/2解像度出力）
decoder = ViTDecoderDPT(
    encoder_channels=[384, 384, 384, 384],
    decoder_channels=256,
    final_upsampling=2  # デフォルト: 1/2解像度出力
)

# フォワードパス
x = torch.randn(2, 3, 512, 1280)
features = encoder(x)  # List of (B, 384, 32, 80) tensors
output = decoder(*features)  # (B, 256, 256, 640) ※1/2解像度

# === Output Head（論文通りの設計）===
# チャネル数を削減しながらアップサンプリング → メモリ効率が良い
class SegmentationHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=19):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')  # 1/2 → Full
        x = F.relu(self.conv2(x))
        return self.classifier(x)

seg_head = SegmentationHead(256, num_classes=19)
segmentation = seg_head(output)  # (B, 19, 512, 1280)
```

### メモリ効率についての注意

論文では**デコーダ出力は1/2解像度**で、フル解像度へのアップサンプリングはOutput Headで行います：

```
Decoder Output: (B, 256, 256, 640)  ← 1/2解像度
    ↓ Conv3x3 (256 → 128)
    ↓ 2x Upsample (bilinear)        ← ここでフル解像度に
    ↓ Conv3x3 (128 → 32)
    ↓ Conv1x1 (32 → num_classes)
Final Output: (B, num_classes, 512, 1280)
```

これにより**256chのフル解像度テンソルを保持しない**ため、メモリ効率が良くなります。

## 性能の目安

| バリアント | パラメータ数 | 速度 | 精度 |
|-----------|-------------|------|------|
| ViTDecoderDPT | ~4-5M | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| ViTDecoderDPTSimplified | ~3-4M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 特徴と利点

### 利点
- **グローバルな一貫性**: ViTのグローバルアテンションと相性が良い
- **マルチスケール融合**: 異なる抽象度の特徴を効果的に統合
- **残差接続**: 勾配の流れを改善し、学習を安定化
- **実績のある設計**: 深度推定やセグメンテーションで高い性能を実証

### 欠点
- ResidualConvUnitが多いため、case5 (Weighted Sum) より重い
- BatchNormを使う場合、バッチサイズが小さいと不安定になる可能性

## 推奨用途

- **深度推定**: 元々DPTが設計されたタスク
- **セマンティックセグメンテーション**: ADE20Kなどで高い性能
- **大規模データセット**: DPTは大量データで最大の効果を発揮

## チューニングポイント

1. **use_bn**: 回帰タスク（深度推定）ではFalse推奨、分類タスク（セグメンテーション）ではTrue推奨
2. **encoder_channels**: timmのViTモデルに応じて調整
   - vit_small: 384
   - vit_base: 768
   - vit_large: 1024
3. **final_upsampling**: patch_sizeに応じて調整（patch16なら16）
