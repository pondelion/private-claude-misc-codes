# DPT-Hybrid Implementation

## 概要

DPT-Hybridは、ResNet50をパッチ埋め込みバックボーンとして使用するハイブリッドアーキテクチャです。
ResNet中間層の特徴とTransformer層の特徴を組み合わせることで、より豊かなマルチスケール表現を実現します。

## DPT-Hybrid vs Pure ViT (DPT-Base/Large)

| 項目 | DPT-Hybrid | DPT-Base/Large |
|-----|-----------|----------------|
| パッチ埋め込み | ResNet50 | Linear Projection |
| 特徴解像度 | 異なる (1/4, 1/8, 1/16) | 同一 (1/16) |
| Skip connections | ResNet + Transformer | Transformer only |
| パラメータ効率 | 良い | やや大きい |
| 小規模データ | 有利 | 大規模データ向き |

## アーキテクチャ

```
Input Image: (B, 3, H, W)
    │
    ├─ ResNet50 Stem
    │       │
    │       ├─ Block1 → R0: (B, 256, H/4, W/4)  ───────────┐
    │       │                                               │
    │       ├─ Block2 → R1: (B, 512, H/8, W/8)  ────────┐   │
    │       │                                           │   │
    │       └─ Block3 → tokens: (B, 1024, H/16, W/16)   │   │
    │               │                                   │   │
    │               ↓ Flatten + Project                 │   │
    │               │                                   │   │
    └─────────── ViT Transformer Layers                 │   │
                    │                                   │   │
                    ├─ Layer 9  → (B, 768, H/16, W/16) ─┤   │
                    │                                   │   │
                    └─ Layer 12 → (B, 768, H/16, W/16) ─┤   │
                                                        │   │
    ┌───────────────────────────────────────────────────┘   │
    │                                                       │
    ▼ Decoder                                               │
    ┌─────────────────────────────────────────────────────────┐
    │ Reassemble (channel projection)                       │
    │   layer_rn[0]: (B, 256, H/4, W/4)   ← R0 ─────────────┘
    │   layer_rn[1]: (B, 256, H/8, W/8)   ← R1
    │   layer_rn[2]: (B, 256, H/16, W/16) ← Transformer
    │   layer_rn[3]: (B, 256, H/32, W/32) ← Transformer (downsampled)
    │                                                       │
    │ Progressive Fusion (deep → shallow)                   │
    │   fusion[3]: 1/32 → 1/16                             │
    │   fusion[2]: 1/16 + skip → 1/8                       │
    │   fusion[1]: 1/8 + skip → 1/4                        │
    │   fusion[0]: 1/4 + skip → 1/2                        │
    └─────────────────────────────────────────────────────────┘
    │
    ▼
Output: (B, 256, H/2, W/2)
```

## ファイル構成

```
hybrid/
├── encoder.py   # DPTHybridEncoder: timmからhybrid特徴を抽出
├── decoder.py   # ViTDecoderDPTHybrid: 異なる解像度の特徴を融合
└── README.md    # このファイル
```

## 使用例

### 方法1: timmのHybrid ViTモデルを使用

```python
import torch
from case7_dpt_style.hybrid.encoder import DPTHybridEncoder
from case7_dpt_style.hybrid.decoder import ViTDecoderDPTHybrid

# Encoder
encoder = DPTHybridEncoder(
    model_name='vit_base_r50_s16_384.orig_in21k',
    pretrained=True,
)

# Decoder
decoder = ViTDecoderDPTHybrid(
    encoder_channels=encoder.get_output_channels(),  # [256, 512, 768, 768]
    encoder_strides=encoder.get_output_strides(),    # [4, 8, 16, 16]
    decoder_channels=256,
)

# Forward
x = torch.randn(2, 3, 512, 1280)
features = encoder(x)
output = decoder(features)  # (B, 256, 256, 640)
```

### 方法2: 手動でResNet + ViTを組み合わせ

```python
import torch
import timm
from case7_dpt_style.hybrid.decoder import ViTDecoderDPTHybrid

# ResNet for intermediate features
resnet = timm.create_model('resnet50', features_only=True, out_indices=(1, 2))

# ViT for global features
vit = timm.create_model('vit_base_patch16_384', features_only=True, out_indices=(5, 11))

# Extract features manually
x = torch.randn(2, 3, 512, 1280)
resnet_feats = resnet(x)  # [R0, R1]

# For ViT, need to handle separately (resize input or use hybrid model)
# This is simplified - actual implementation needs more care
```

## Shape Flow 詳細

入力画像 512x1280 の場合：

| Stage | Feature | Shape | Resolution |
|-------|---------|-------|------------|
| Input | Image | (B, 3, 512, 1280) | Full |
| ResNet Block1 | R0 | (B, 256, 128, 320) | 1/4 |
| ResNet Block2 | R1 | (B, 512, 64, 160) | 1/8 |
| Transformer | Layer 9 | (B, 768, 32, 80) | 1/16 |
| Transformer | Layer 12 | (B, 768, 32, 80) | 1/16 |
| Reassemble | layer_rn[0] | (B, 256, 128, 320) | 1/4 |
| Reassemble | layer_rn[1] | (B, 256, 64, 160) | 1/8 |
| Reassemble | layer_rn[2] | (B, 256, 32, 80) | 1/16 |
| Reassemble | layer_rn[3] | (B, 256, 16, 40) | 1/32 |
| Fusion[3] | path | (B, 256, 32, 80) | 1/16 |
| Fusion[2] | path | (B, 256, 64, 160) | 1/8 |
| Fusion[1] | path | (B, 256, 128, 320) | 1/4 |
| Fusion[0] | path | (B, 256, 256, 640) | 1/2 |
| **Output** | | **(B, 256, 256, 640)** | **1/2** |

## Pure ViT版との違い

### Decoder側の違い

| 項目 | Pure ViT版 (ViTDecoderDPT) | Hybrid版 (ViTDecoderDPTHybrid) |
|-----|--------------------------|------------------------------|
| 入力特徴 | 全て同一解像度 | 異なる解像度 |
| Reassemble | Channel proj + Spatial resample | Channel proj only |
| 融合時のskip | Reassembleで調整済み | そのままの解像度 |

### Encoder側の違い

Pure ViT版では `timm.create_model(..., features_only=True)` で簡単に取得できるのに対し、
Hybrid版ではResNet中間層とTransformer層を別々に取得する必要があります。

## 注意事項

1. **timmのバージョン依存**: Hybrid ViTモデルの内部構造はtimmのバージョンによって異なる場合があります

2. **メモリ使用量**: 1/4解像度の特徴を保持するため、Pure ViT版よりメモリを多く使用します
   ```
   layer_rn[0] (1/4): ~42 MB/sample
   layer_rn[1] (1/8): ~10 MB/sample
   layer_rn[2] (1/16): ~2.6 MB/sample
   layer_rn[3] (1/32): ~0.65 MB/sample
   ```

3. **モデル名**: timmで利用可能なHybrid ViTモデル
   - `vit_base_r50_s16_384.orig_in21k`
   - `vit_base_r50_s16_224.orig_in21k`
   - `vit_small_r26_s32_224.augreg_in21k_ft_in1k`
   など

## 参考

- DPT論文: [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413)
- 公式実装: https://github.com/intel-isl/DPT
- timm Hybrid ViT: https://huggingface.co/docs/timm/models/vision-transformer-hybrid
