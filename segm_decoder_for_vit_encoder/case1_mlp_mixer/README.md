# Case 1: MLP Mixer Decoder (速度重視)

## アーキテクチャ概要

```
Input Features (all same spatial resolution)
    ↓
Concatenate along channel dimension
    ↓
Channel Reduction (1x1 conv)
    ↓
MLP Mixer Blocks (token mixing + channel mixing)
    ↓
Final Upsampling
    ↓
Output
```

## 特徴

### 利点
- ✅ 実装がシンプル
- ✅ 推論速度が速い
- ✅ メモリ効率が良い
- ✅ パラメータ数が少ない

### 欠点
- ❌ 精度は他の手法に劣る可能性
- ❌ 受容野の拡大が限定的
- ❌ マルチスケール情報の活用が弱い

## パラメータ

```python
ViTDecoderMLPMixer(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    num_mixer_blocks=4,              # MLP Mixerブロック数
    channel_expansion=4,              # Channel MLP拡張率
    token_expansion=0.5,              # Token MLP拡張率
    dropout=0.0,
    upsampling_mode='bilinear',
    final_upsampling=16,
)
```

## 使い方

```python
from decoder import ViTDecoderMLPMixer
import torch

# Create decoder
decoder = ViTDecoderMLPMixer(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    num_mixer_blocks=4,
    final_upsampling=16
)

# Forward pass
features = [
    torch.randn(2, 384, 32, 80),  # Layer 0
    torch.randn(2, 384, 32, 80),  # Layer 1
    torch.randn(2, 384, 32, 80),  # Layer 2
]

output = decoder(*features)
print(output.shape)  # (2, 256, 512, 1280)
```

## チューニングポイント

1. **num_mixer_blocks**:
   - 増やす → 精度向上、速度低下
   - 減らす → 速度向上、精度低下
   - 推奨: 3-6

2. **channel_expansion**:
   - MLP隠れ層の拡張率
   - 推奨: 2-4

3. **token_expansion**:
   - Token mixing MLPの拡張率
   - 推奨: 0.5-1.0

## 推奨ケース

- ✅ リアルタイム推論が必要
- ✅ メモリが限られている
- ✅ ベースライン実装として
- ✅ エッジデバイスでの推論

## ベンチマーク (参考値)

| Metric | Value |
|--------|-------|
| Parameters | ~1.5M |
| Speed | ⭐⭐⭐⭐ |
| Accuracy | ⭐⭐ |
| Memory | ⭐⭐⭐⭐ |
