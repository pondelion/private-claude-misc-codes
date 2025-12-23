# Quick Start Guide

最速で始めるためのガイド。

## 1. どの実装を選ぶ?

### フローチャート

```
開始
  ↓
Q1: リアルタイム推論が必要?
  YES → Case 5 (Weighted Sum)
  NO  ↓
Q2: 精度が最優先?(計算コスト気にしない)
  YES → Case 2 (Multi-Scale FPN) or Case 4 (Cross-Attention)
  NO  ↓
Q3: バランスの良い実装が欲しい?
  YES → Case 3 (Hierarchical Attention) ★推奨★
  NO  ↓
Q4: 実績のある手法を使いたい?
  YES → Case 6 (FPN-Style)
  NO  ↓
  → Case 1 (MLP Mixer)
```

### 用途別推奨

| 用途 | 推奨案 | 理由 |
|------|--------|------|
| 最初の実装 | Case 3 | バランスが良く安定 |
| プロダクション | Case 3 or 5 | 速度と精度のバランス |
| 研究・実験 | Case 2 or 4 | 最高精度 |
| エッジデバイス | Case 5 | 最軽量・高速 |
| 心電図セグメンテーション | Case 2 | マルチスケール情報が重要 |

## 2. インストール

必要なパッケージ:

```bash
pip install torch torchvision
pip install timm  # ViT encoder用
```

## 3. 基本的な使い方

### Step 1: Encoderの準備

```python
import torch
import timm

# ViT encoder作成
encoder = timm.create_model(
    'vit_small_patch16_224',
    features_only=True,
    pretrained=True
)

# 出力チャネル確認
encoder_channels = encoder.feature_info.channels()
print(f"Encoder channels: {encoder_channels}")
# 例: [384, 384, 384]
```

### Step 2: Decoderの準備

```python
# Case 3 (推奨) の例
import sys
sys.path.append('/home/ym/codes/claude_tmp_output/case3_hierarchical_attention')
from decoder import ViTDecoderHierarchicalAttention

decoder = ViTDecoderHierarchicalAttention(
    encoder_channels=encoder_channels,
    decoder_channels=256,
    use_attention=True,
    final_upsampling=16  # ViTのpatch_size
)
```

### Step 3: Forward Pass

```python
# 入力画像
x = torch.randn(2, 3, 224, 224)

# Encoder
features = encoder(x)
print([f.shape for f in features])
# [(2, 384, 14, 14), (2, 384, 14, 14), (2, 384, 14, 14)]

# Decoder
output = decoder(*features)
print(output.shape)
# (2, 256, 224, 224)
```

### Step 4: Segmentation Head追加

```python
import torch.nn as nn

# Segmentation head
seg_head = nn.Conv2d(256, num_classes, kernel_size=1)

# Full pipeline
features = encoder(x)
dec_out = decoder(*features)
seg_output = seg_head(dec_out)

print(seg_output.shape)
# (2, num_classes, 224, 224)
```

## 4. 全実装の比較テーブル

| 案 | パラメータ | 速度 | 精度 | メモリ | 推奨用途 |
|---|-----------|------|------|--------|----------|
| 1. MLP Mixer | 1.5M | ★★★★ | ★★ | ★★★★ | ベースライン |
| 2. Multi-Scale FPN | 8-10M | ★★ | ★★★★★ | ★★ | 最高精度 |
| 3. Hierarchical Attention | 3-4M | ★★★ | ★★★★ | ★★★ | **バランス** |
| 4. Cross-Attention | 12-15M | ★ | ★★★★★ | ★ | 研究用 |
| 5. Weighted Sum | 0.8M | ★★★★★ | ★★ | ★★★★★ | 最軽量 |
| 6. FPN-Style | 3M | ★★★ | ★★★ | ★★★ | 実績重視 |

## 5. ベンチマーク実行

全実装を自動比較:

```bash
cd /home/ym/codes/claude_tmp_output
python benchmark.py
```

出力例:
```
Model                               Params          Time (ms)    Memory (MB)
--------------------------------------------------------------------------------
Case 1: MLP Mixer                   1,534,208       12.34        245.67
Case 2: Multi-Scale FPN             8,923,456       45.67        892.34
Case 3: Hierarchical Attention      3,456,789       23.45        456.78
...
```

## 6. SignalSegModelV7への統合

```python
# example_integration.py を参照
python /home/ym/codes/claude_tmp_output/example_integration.py
```

または:

```python
from example_integration import SignalSegModelViT

model = SignalSegModelViT(
    num_classes=1,
    encoder_model_name='vit_small_patch16_224',
    decoder_type='hierarchical_attention',  # ← ここで選択
    decoder_channels=256,
)

# 入力: (B, N_LEADS, C, H, W)
x = torch.randn(2, 16, 3, 512, 1280)
pred_mask, pred_y_coord = model(x)
```

## 7. よくある問題と解決策

### 問題1: メモリ不足

**解決策:**
```python
# 1. decoder_channelsを減らす
decoder = Decoder(..., decoder_channels=128)

# 2. Batch sizeを減らす
dataloader = DataLoader(..., batch_size=1)

# 3. 軽量な実装を使う
# Case 4 → Case 3 or Case 5
```

### 問題2: 速度が遅い

**解決策:**
```python
# 1. 軽量な実装を使う
# Case 2/4 → Case 5

# 2. 推論モードを使う
model.eval()
with torch.inference_mode():
    output = model(x)

# 3. Mixed precision
with torch.cuda.amp.autocast():
    output = model(x)
```

### 問題3: 精度が不足

**解決策:**
```python
# 1. より強力な実装を使う
# Case 5 → Case 3 → Case 2

# 2. decoder_channelsを増やす
decoder = Decoder(..., decoder_channels=512)

# 3. より多くのencoder層を使う
encoder = timm.create_model(
    'vit_small_patch16_224',
    features_only=True,
    out_indices=(0, 2, 4, 6, 8, 11)  # ← 増やす
)
```

## 8. 次のステップ

1. **各案の詳細を読む**: 各フォルダの `README.md` を参照
2. **ベンチマークを実行**: `benchmark.py` で比較
3. **カスタマイズ**: パラメータを調整して最適化
4. **統合**: `example_integration.py` を参考に実装

## 9. チートシート

### 速度優先
```python
from case5_weighted_sum.decoder import ViTDecoderWeightedSum
decoder = ViTDecoderWeightedSum(
    encoder_channels=[384]*3,
    decoder_channels=128,
    use_refinement=False
)
```

### 精度優先
```python
from case2_multiscale_fpn.decoder import ViTDecoderMultiScaleFPN
decoder = ViTDecoderMultiScaleFPN(
    encoder_channels=[384]*3,
    decoder_channels=512,
    use_channel_attention=True,
    use_spatial_attention=True
)
```

### バランス
```python
from case3_hierarchical_attention.decoder import ViTDecoderHierarchicalAttention
decoder = ViTDecoderHierarchicalAttention(
    encoder_channels=[384]*3,
    decoder_channels=256,
    use_attention=True
)
```

## 10. サポート

- 詳細なドキュメント: 各フォルダの `README.md`
- 実装例: `example_integration.py`
- ベンチマーク: `benchmark.py`
- 全体概要: ルートの `README.md`
