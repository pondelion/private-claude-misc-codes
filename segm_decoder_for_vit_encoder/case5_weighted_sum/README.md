# Case 5: Weighted Sum Decoder (超速度重視)

## アーキテクチャ概要

### Version 1: Simple Weighted Sum
```
Input Features
    ↓
Each Feature → 1x1 Conv (project to decoder_channels)
    ↓
Learnable Weights (softmax/sigmoid)
    ↓
Weighted Sum
    ↓
Optional Refinement Convolutions
    ↓
Final Upsampling
    ↓
Output
```

### Version 2: Spatial-Aware Weighted Sum
```
Input Features
    ↓
Each Feature → 1x1 Conv (project to decoder_channels)
    ↓
Concat → Weight Prediction Network → Spatial Weights (H×W)
    ↓
Spatial-wise Weighted Sum
    ↓
Optional Refinement Convolutions
    ↓
Final Upsampling
    ↓
Output
```

## 特徴

### 利点
- ✅ 最も高速
- ✅ 最少パラメータ数
- ✅ メモリ効率が最高
- ✅ 実装がシンプル
- ✅ エッジデバイスで動作可能

### 欠点
- ❌ 表現力は限定的
- ❌ 複雑なパターンの捉えづらさ
- ❌ マルチスケール情報の活用が弱い

## パラメータ

### Version 1 (Simple)
```python
ViTDecoderWeightedSum(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_refinement=True,              # Refinement conv使用
    num_refinement_blocks=2,
    use_softmax_weights=True,         # True: sum=1, False: independent
    upsampling_mode='bilinear',
    final_upsampling=16,
)
```

### Version 2 (Spatial-Aware)
```python
ViTDecoderWeightedSumV2(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_refinement=True,
    num_refinement_blocks=2,
    upsampling_mode='bilinear',
    final_upsampling=16,
)
```

## 使い方

```python
from decoder import ViTDecoderWeightedSum, ViTDecoderWeightedSumV2
import torch

# Version 1: Simple
decoder_v1 = ViTDecoderWeightedSum(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_refinement=True,
    use_softmax_weights=True,
    final_upsampling=16
)

# Version 2: Spatial-Aware
decoder_v2 = ViTDecoderWeightedSumV2(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_refinement=True,
    final_upsampling=16
)

# Forward pass
features = [
    torch.randn(2, 384, 32, 80),
    torch.randn(2, 384, 32, 80),
    torch.randn(2, 384, 32, 80),
]

output = decoder_v1(*features)
print(output.shape)  # (2, 256, 512, 1280)

# Check learned weights (V1 only)
weights = decoder_v1.get_layer_weights()
print(f"Layer weights: {weights}")
```

## 2つのバージョンの比較

| Feature | V1 (Simple) | V2 (Spatial-Aware) |
|---------|-------------|-------------------|
| Weights | Global (same for all positions) | Spatial (different per position) |
| Parameters | 最少 | やや多い |
| Speed | 最速 | やや遅い |
| Flexibility | 低 | 中 |

## チューニングポイント

1. **use_refinement**:
   - True: 精度向上、やや遅い
   - False: 最速、精度やや低下

2. **num_refinement_blocks**:
   - 0-1: 最軽量
   - 2-3: バランス
   - 4+: 精度重視

3. **use_softmax_weights** (V1のみ):
   - True: 重みの合計=1 (相対的な重要度)
   - False: 独立な重み (絶対的な重要度)

## 推奨ケース

### Version 1 推奨
- ✅ 速度が最優先
- ✅ エッジデバイスでの推論
- ✅ リアルタイム処理
- ✅ 軽量モデルが必要
- ✅ ベースライン実装

### Version 2 推奨
- ✅ 位置によって重要な層が異なる
- ✅ V1では精度が不足
- ✅ まだ軽量性が重要

## ベンチマーク (参考値)

### V1 (Simple)
| Metric | Value |
|--------|-------|
| Parameters | ~0.8M |
| Speed | ⭐⭐⭐⭐⭐ |
| Accuracy | ⭐⭐ |
| Memory | ⭐⭐⭐⭐⭐ |

### V2 (Spatial-Aware)
| Metric | Value |
|--------|-------|
| Parameters | ~1.2M |
| Speed | ⭐⭐⭐⭐ |
| Accuracy | ⭐⭐⭐ |
| Memory | ⭐⭐⭐⭐ |

## 実装バリエーション

### 超軽量版 (Refinement無し)
```python
decoder = ViTDecoderWeightedSum(
    encoder_channels=[384, 384, 384],
    decoder_channels=128,       # ← 減らす
    use_refinement=False,       # ← 無効化
    final_upsampling=16
)
```

### 精度重視版 (Refinement強化)
```python
decoder = ViTDecoderWeightedSum(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_refinement=True,
    num_refinement_blocks=4,    # ← 増やす
    final_upsampling=16
)
```

## 学習済み重みの可視化

```python
# V1のみ
weights = decoder.get_layer_weights()
print(f"Layer 0 weight: {weights[0]:.3f}")
print(f"Layer 1 weight: {weights[1]:.3f}")
print(f"Layer 2 weight: {weights[2]:.3f}")

# どの層が重要か分かる
# 例: [0.15, 0.25, 0.60] → 深い層が最も重要
```
