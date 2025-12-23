# Case 2: Multi-Scale Feature Pyramid Decoder (精度重視)

## アーキテクチャ概要

```
Input Features (all same spatial resolution)
    ↓
Each Feature → ASPP (multi-scale context extraction)
    ↓
Concatenate all processed features
    ↓
Channel Attention + Spatial Attention
    ↓
Refinement Convolutions
    ↓
Final Upsampling
    ↓
Output
```

## 特徴

### 利点
- ✅ マルチスケール情報を効果的に活用
- ✅ ASPP による広い受容野
- ✅ Channel/Spatial Attentionで重要特徴を選択
- ✅ セグメンテーションタスクと相性が良い
- ✅ 細かいパターンと大域的な文脈の両方を捉える

### 欠点
- ❌ 計算コストが高い
- ❌ パラメータ数が多い
- ❌ メモリ使用量が多い

## パラメータ

```python
ViTDecoderMultiScaleFPN(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    dilation_rates=(6, 12, 18),        # ASPP dilation rates
    use_channel_attention=True,
    use_spatial_attention=True,
    upsampling_mode='bilinear',
    final_upsampling=16,
)
```

## 使い方

```python
from decoder import ViTDecoderMultiScaleFPN
import torch

# Create decoder
decoder = ViTDecoderMultiScaleFPN(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    dilation_rates=(6, 12, 18),
    use_channel_attention=True,
    use_spatial_attention=True,
    final_upsampling=16
)

# Forward pass
features = [
    torch.randn(2, 384, 32, 80),
    torch.randn(2, 384, 32, 80),
    torch.randn(2, 384, 32, 80),
]

output = decoder(*features)
print(output.shape)  # (2, 256, 512, 1280)
```

## チューニングポイント

1. **dilation_rates**:
   - 小さい値: 細かいパターン重視
   - 大きい値: 大域的な文脈重視
   - 推奨: (3, 6, 12) ~ (12, 24, 36)

2. **use_channel_attention / use_spatial_attention**:
   - 両方True: 最高精度、遅い
   - 片方のみ: バランス
   - 両方False: 速いが精度低下

## 推奨ケース

- ✅ 精度が最優先
- ✅ マルチスケール情報が重要
- ✅ 心電図のような細かいパターン検出
- ✅ 計算リソースに余裕がある
- ✅ オフライン処理

## ベンチマーク (参考値)

| Metric | Value |
|--------|-------|
| Parameters | ~8-10M |
| Speed | ⭐⭐ |
| Accuracy | ⭐⭐⭐⭐⭐ |
| Memory | ⭐⭐ |

## 心電図セグメンテーションでの推奨理由

心電図タスクでは以下が重要:
- P波、QRS波、T波などの細かいパターン検出
- リード間の大域的な文脈理解
- ノイズに対するロバスト性

→ ASPPのマルチスケール処理とAttentionの組み合わせが最適
