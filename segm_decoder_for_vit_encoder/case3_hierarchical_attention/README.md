# Case 3: Hierarchical Channel Attention Decoder (バランス型)

## アーキテクチャ概要

```
feat[0]
    ↓
Project to decoder_channels
    ↓
+ feat[1] → Channel Attention → Refinement → fused[0]
    ↓
+ feat[2] → Channel Attention → Refinement → fused[1]
    ↓
...
    ↓
Final Refinement
    ↓
Final Upsampling
    ↓
Output
```

## 特徴

### 利点
- ✅ 速度と精度のバランスが良い
- ✅ 段階的処理で学習が安定
- ✅ 解釈性が高い(各段階の貢献度を可視化可能)
- ✅ 実装がシンプルで理解しやすい

### 欠点
- ❌ 並列化しづらい(段階的処理のため)
- ❌ 最深層への依存が強い

## パラメータ

```python
ViTDecoderHierarchicalAttention(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_attention=True,               # Attention使用
    attention_reduction=16,            # Attention reduction ratio
    upsampling_mode='bilinear',
    final_upsampling=16,
)
```

## 使い方

```python
from decoder import ViTDecoderHierarchicalAttention
import torch

# Create decoder
decoder = ViTDecoderHierarchicalAttention(
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

output = decoder(*features)
print(output.shape)  # (2, 256, 512, 1280)
```

## チューニングポイント

1. **use_attention**:
   - True: 精度重視、やや遅い
   - False: 速度重視、精度やや低下

2. **attention_reduction**:
   - 小さい値(8): より表現力が高い、重い
   - 大きい値(32): 軽量、表現力やや低下
   - 推奨: 16

## 推奨ケース

- ✅ 速度と精度の両立が必要
- ✅ 安定した学習が重要
- ✅ 最初の実用実装として
- ✅ プロダクション環境
- ✅ 特徴の段階的統合が有効なタスク

## ベンチマーク (参考値)

| Metric | Value |
|--------|-------|
| Parameters | ~3-4M |
| Speed | ⭐⭐⭐ |
| Accuracy | ⭐⭐⭐⭐ |
| Memory | ⭐⭐⭐ |

## 可視化・解釈性

段階的な融合プロセスのため、各段階での特徴マップを可視化することで:
- どの層がどの程度貢献しているか
- 浅い層と深い層の情報統合の様子
- Attentionによる特徴選択の様子

を理解しやすい。

## バリエーション

```python
# Attention無し (高速版)
decoder = ViTDecoderHierarchicalAttention(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_attention=False,  # ← 無効化
    final_upsampling=16
)
```
