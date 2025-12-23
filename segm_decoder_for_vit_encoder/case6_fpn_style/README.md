# Case 6: FPN-Style Decoder (実績のあるアプローチ)

## アーキテクチャ概要

### Version 1: Basic FPN-Style
```
Encoder Features (shallow → deep)
    ↓
Lateral Connections (1×1 conv)
    ↓
Top-Down Pathway:
  deepest → refinement → fused[n]
  fused[n] + feat[n-1] → refinement → fused[n-1]
  ...
    ↓
Concatenate all FPN levels
    ↓
Final Fusion & Refinement
    ↓
Final Upsampling
    ↓
Output
```

### Version 2: FPN-Style with Attention
```
... (same as V1 until concatenation)
    ↓
Attention-based Fusion
  (Learn spatial weights for each FPN level)
    ↓
Final Projection & Refinement
    ↓
Final Upsampling
    ↓
Output
```

## 特徴

### 利点
- ✅ FPNの実績ある設計思想
- ✅ 段階的refinementで安定
- ✅ マルチレベル特徴の活用
- ✅ 解釈性が高い

### 欠点
- ❌ ViTでは解像度が同じため、本来のFPNほどの効果は期待できない可能性
- ❌ 段階的処理のため並列化しづらい

## パラメータ

### Version 1 (Basic)
```python
ViTDecoderFPNStyle(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_refinement=True,              # Refinement blocks使用
    upsampling_mode='bilinear',
    final_upsampling=16,
)
```

### Version 2 (with Attention)
```python
ViTDecoderFPNStyleV2(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_attention=True,               # Attention-based fusion
    upsampling_mode='bilinear',
    final_upsampling=16,
)
```

## 使い方

```python
from decoder import ViTDecoderFPNStyle, ViTDecoderFPNStyleV2
import torch

# Version 1: Basic
decoder_v1 = ViTDecoderFPNStyle(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_refinement=True,
    final_upsampling=16
)

# Version 2: with Attention
decoder_v2 = ViTDecoderFPNStyleV2(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_attention=True,
    final_upsampling=16
)

# Forward pass
features = [
    torch.randn(2, 384, 32, 80),
    torch.randn(2, 384, 32, 80),
    torch.randn(2, 384, 32, 80),
]

output_v1 = decoder_v1(*features)
output_v2 = decoder_v2(*features)

print(output_v1.shape)  # (2, 256, 512, 1280)
print(output_v2.shape)  # (2, 256, 512, 1280)
```

## 2つのバージョンの比較

| Feature | V1 (Basic) | V2 (Attention) |
|---------|-----------|----------------|
| Fusion | Concatenation | Attention-weighted |
| Parameters | 中 | やや多い |
| Speed | やや速い | やや遅い |
| Flexibility | 低 | 高 |

## FPNとの違い

**従来のFPN** (CNN用):
- 異なる解像度の特徴を融合
- 上位層をアップサンプリングして下位層と足し合わせ

**ViT用FPN-Style**:
- 同じ解像度の特徴を融合
- アップサンプリング無しで足し合わせ
- 「抽象度の違い」を活用

## チューニングポイント

1. **use_refinement** (V1のみ):
   - True: 各段階で丁寧にrefine
   - False: シンプルに足し合わせるだけ

2. **use_attention** (V2のみ):
   - True: 適応的な重み付け
   - False: 単純なconcatenation

## 推奨ケース

### Version 1 推奨
- ✅ 実績のあるアーキテクチャを好む
- ✅ シンプルで安定した実装が必要
- ✅ FPNに慣れている

### Version 2 推奨
- ✅ より柔軟な特徴融合が必要
- ✅ V1では精度が不足
- ✅ 位置によって重要な層が異なる

## ベンチマーク (参考値)

### V1 (Basic)
| Metric | Value |
|--------|-------|
| Parameters | ~3M |
| Speed | ⭐⭐⭐ |
| Accuracy | ⭐⭐⭐ |
| Memory | ⭐⭐⭐ |

### V2 (Attention)
| Metric | Value |
|--------|-------|
| Parameters | ~4M |
| Speed | ⭐⭐⭐ |
| Accuracy | ⭐⭐⭐⭐ |
| Memory | ⭐⭐⭐ |

## Top-Down Pathwayの意義

ViTでは全層が同じ解像度だが、以下の違いがある:

1. **浅い層**: 低レベル特徴 (局所的なパターン)
2. **深い層**: 高レベル特徴 (抽象的な概念)

Top-down pathwayにより:
- 深い層の抽象的な情報を浅い層に伝播
- 浅い層の細かい情報で深い層を補完
- 段階的に統合することで安定した学習

## 可視化例

```python
# 各FPN levelの出力を保存して可視化可能
# (実装にhookを追加する必要がある)

import torch.nn as nn

class FPNWithVisualization(ViTDecoderFPNStyle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fpn_outputs = []

    def forward(self, *features):
        # ... (元の実装)
        # fpn_features を保存
        self.fpn_outputs = fpn_features
        # ... (残りの処理)
        return x

# 使用例
decoder = FPNWithVisualization(...)
output = decoder(*features)
for i, fpn_feat in enumerate(decoder.fpn_outputs):
    # Visualize fpn_feat
    pass
```

## 実装バリエーション

### 軽量版
```python
decoder = ViTDecoderFPNStyle(
    encoder_channels=[384, 384, 384],
    decoder_channels=128,       # ← 減らす
    use_refinement=False,       # ← 無効化
    final_upsampling=16
)
```

### 精度重視版
```python
decoder = ViTDecoderFPNStyleV2(
    encoder_channels=[384, 384, 384],
    decoder_channels=512,       # ← 増やす
    use_attention=True,         # ← Attention有効
    final_upsampling=16
)
```
