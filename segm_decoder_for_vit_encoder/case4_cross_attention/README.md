# Case 4: Cross-Attention Decoder (精度最重視)

## アーキテクチャ概要

```
Encoder Features
    ↓
Project all to common embedding dimension
    ↓
Query: Deepest layer
Key/Value: Concatenated all layers
    ↓
Transformer Decoder Layers
  - Multi-Head Self-Attention
  - Multi-Head Cross-Attention
  - Feed-Forward Network
    ↓
Reshape to spatial format
    ↓
Final Upsampling
    ↓
Output
```

## 特徴

### 利点
- ✅ 最も表現力が高い
- ✅ 層間の相互作用を柔軟にモデル化
- ✅ 適応的な特徴選択
- ✅ Transformerの強力な表現力を活用

### 欠点
- ❌ 計算コストが非常に高い (O(N²))
- ❌ メモリ使用量が多い
- ❌ 学習が不安定になる可能性
- ❌ 推論速度が遅い

## パラメータ

```python
ViTDecoderCrossAttention(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    num_decoder_layers=4,             # Transformer decoder層数
    num_heads=8,                      # Attention head数
    mlp_ratio=4.0,                    # FFN拡張率
    dropout=0.1,
    upsampling_mode='bilinear',
    final_upsampling=16,
)
```

## 使い方

```python
from decoder import ViTDecoderCrossAttention
import torch

# Create decoder
decoder = ViTDecoderCrossAttention(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    num_decoder_layers=4,
    num_heads=8,
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

1. **num_decoder_layers**:
   - 増やす → 精度向上、計算コスト増大
   - 減らす → 軽量化、精度低下
   - 推奨: 2-6

2. **num_heads**:
   - 多い → 多様な attention patterns
   - 少ない → 軽量、表現力低下
   - 推奨: 4-8

3. **decoder_channels**:
   - num_headsで割り切れる値にする
   - 推奨: 256, 384, 512

## メモリ最適化

大きな画像サイズで使用する場合:

```python
# Gradient checkpointing (学習時)
# 実装例
for layer in decoder.decoder_layers:
    layer = torch.utils.checkpoint.checkpoint_wrapper(layer)

# より小さいdecoder_channels
decoder = ViTDecoderCrossAttention(
    encoder_channels=[384, 384, 384],
    decoder_channels=128,  # ← 減らす
    num_decoder_layers=2,  # ← 減らす
    num_heads=4,
    final_upsampling=16
)
```

## 推奨ケース

- ✅ 精度が絶対的に重要
- ✅ 計算リソースが豊富 (GPU)
- ✅ 研究・実験段階
- ✅ オフライン処理
- ✅ バッチサイズが小さい

## ベンチマーク (参考値)

| Metric | Value |
|--------|-------|
| Parameters | ~12-15M |
| Speed | ⭐ |
| Accuracy | ⭐⭐⭐⭐⭐ |
| Memory | ⭐ |

## 注意点

### メモリ使用量
- Attention の計算量: O((N_layers × H × W)²)
- 大きな画像では非常に大きなメモリを消費
- 512×1280の画像では特に注意

### 学習の安定性
- Warmup learning rateの使用を推奨
- Layer normalizationが重要
- Dropoutで正則化

### 推論の高速化
- 可能であればONNX exportやTensorRTで最適化
- Batch sizeを小さく保つ
