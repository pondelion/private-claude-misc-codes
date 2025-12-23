# ViT Decoder Implementations - Index

## 📁 ファイル構成

```
claude_tmp_output/
├── README.md                    # 全体概要と詳細比較
├── QUICKSTART.md                # クイックスタートガイド
├── INDEX.md                     # このファイル
├── benchmark.py                 # 全実装のベンチマーク
├── test_all_decoders.py         # 全実装のテスト
├── example_integration.py       # SignalSegModelV7への統合例
│
├── case1_mlp_mixer/
│   ├── README.md                # Case 1の詳細
│   └── decoder.py               # MLP Mixer実装
│
├── case2_multiscale_fpn/
│   ├── README.md                # Case 2の詳細
│   └── decoder.py               # Multi-Scale FPN実装
│
├── case3_hierarchical_attention/
│   ├── README.md                # Case 3の詳細
│   └── decoder.py               # Hierarchical Attention実装
│
├── case4_cross_attention/
│   ├── README.md                # Case 4の詳細
│   └── decoder.py               # Cross-Attention実装
│
├── case5_weighted_sum/
│   ├── README.md                # Case 5の詳細
│   └── decoder.py               # Weighted Sum実装 (V1 & V2)
│
└── case6_fpn_style/
    ├── README.md                # Case 6の詳細
    └── decoder.py               # FPN-Style実装 (V1 & V2)
```

## 🚀 使い始める

### 1. まず読む
- [QUICKSTART.md](QUICKSTART.md) - 最速で始めるためのガイド
- [README.md](README.md) - 全体概要と詳細比較

### 2. テストする
```bash
# 全実装が動作するか確認
python test_all_decoders.py

# ベンチマーク (パラメータ数、速度、メモリを比較)
python benchmark.py
```

### 3. 統合例を見る
```bash
# SignalSegModelV7への統合例
python example_integration.py
```

## 📊 実装一覧

| # | 名前 | フォルダ | 特徴 | 推奨用途 |
|---|------|----------|------|----------|
| 1 | MLP Mixer | [case1_mlp_mixer](case1_mlp_mixer/) | 速度重視 | ベースライン |
| 2 | Multi-Scale FPN | [case2_multiscale_fpn](case2_multiscale_fpn/) | 精度重視 | 最高精度が必要 |
| 3 | Hierarchical Attention | [case3_hierarchical_attention](case3_hierarchical_attention/) | **バランス** | **最初の実装** |
| 4 | Cross-Attention | [case4_cross_attention](case4_cross_attention/) | 精度最重視 | 研究用 |
| 5 | Weighted Sum | [case5_weighted_sum](case5_weighted_sum/) | 超速度重視 | エッジデバイス |
| 6 | FPN-Style | [case6_fpn_style](case6_fpn_style/) | 実績重視 | FPN愛用者 |

## 🎯 用途別推奨

### 初めて実装する
→ [Case 3: Hierarchical Attention](case3_hierarchical_attention/)
- 速度と精度のバランスが良い
- 実装が安定している

### 心電図セグメンテーション
→ [Case 2: Multi-Scale FPN](case2_multiscale_fpn/)
- 細かいパターン(P波、QRS、T波)と大域的文脈の両方が重要
- ASPPのマルチスケール処理が有効

### リアルタイム推論
→ [Case 5: Weighted Sum](case5_weighted_sum/)
- 最も高速
- 最少パラメータ数

### 最高精度が必要
→ [Case 2: Multi-Scale FPN](case2_multiscale_fpn/) or [Case 4: Cross-Attention](case4_cross_attention/)
- 計算コストは高いが最も表現力が高い

## 📖 各実装の詳細

各フォルダ内の `README.md` に詳細が記載されています:

1. [case1_mlp_mixer/README.md](case1_mlp_mixer/README.md)
2. [case2_multiscale_fpn/README.md](case2_multiscale_fpn/README.md)
3. [case3_hierarchical_attention/README.md](case3_hierarchical_attention/README.md)
4. [case4_cross_attention/README.md](case4_cross_attention/README.md)
5. [case5_weighted_sum/README.md](case5_weighted_sum/README.md)
6. [case6_fpn_style/README.md](case6_fpn_style/README.md)

## 🔧 カスタマイズ

各実装は以下のパラメータをサポート:

- `encoder_channels`: エンコーダーの出力チャネル数のリスト
- `decoder_channels`: デコーダーの出力チャネル数
- `final_upsampling`: 最終的なアップサンプリング倍率 (通常16)
- `upsampling_mode`: 'bilinear' or 'nearest'

詳細は各実装の `README.md` を参照。

## 📝 クイックリファレンス

### Case 1: MLP Mixer
```python
from case1_mlp_mixer.decoder import ViTDecoderMLPMixer
decoder = ViTDecoderMLPMixer(encoder_channels=[384]*3, decoder_channels=256)
```

### Case 2: Multi-Scale FPN
```python
from case2_multiscale_fpn.decoder import ViTDecoderMultiScaleFPN
decoder = ViTDecoderMultiScaleFPN(encoder_channels=[384]*3, decoder_channels=256)
```

### Case 3: Hierarchical Attention ⭐
```python
from case3_hierarchical_attention.decoder import ViTDecoderHierarchicalAttention
decoder = ViTDecoderHierarchicalAttention(encoder_channels=[384]*3, decoder_channels=256)
```

### Case 4: Cross-Attention
```python
from case4_cross_attention.decoder import ViTDecoderCrossAttention
decoder = ViTDecoderCrossAttention(encoder_channels=[384]*3, decoder_channels=256)
```

### Case 5: Weighted Sum
```python
from case5_weighted_sum.decoder import ViTDecoderWeightedSum
decoder = ViTDecoderWeightedSum(encoder_channels=[384]*3, decoder_channels=256)
```

### Case 6: FPN-Style
```python
from case6_fpn_style.decoder import ViTDecoderFPNStyle
decoder = ViTDecoderFPNStyle(encoder_channels=[384]*3, decoder_channels=256)
```

## 🎓 学習リソース

1. **基本を理解する**: [README.md](README.md)
2. **すぐに始める**: [QUICKSTART.md](QUICKSTART.md)
3. **詳細を学ぶ**: 各フォルダの `README.md`
4. **実装例を見る**: [example_integration.py](example_integration.py)
5. **性能を比較**: [benchmark.py](benchmark.py)

## 🐛 トラブルシューティング

### メモリ不足
- `decoder_channels` を減らす (256 → 128)
- より軽量な実装を使う (Case 4 → Case 5)
- Batch sizeを減らす

### 速度が遅い
- より高速な実装を使う (Case 2 → Case 5)
- `torch.inference_mode()` を使う
- Mixed precisionを使う

### 精度が不足
- より強力な実装を使う (Case 5 → Case 2)
- `decoder_channels` を増やす (256 → 512)
- より多くのエンコーダー層を使う

詳細は [QUICKSTART.md](QUICKSTART.md) の「よくある問題と解決策」を参照。

## 📊 比較表

| 実装 | Params | Speed | Accuracy | Memory | 推奨度 |
|------|--------|-------|----------|--------|--------|
| Case 1 | 1.5M | ★★★★ | ★★ | ★★★★ | ⭐⭐ |
| Case 2 | 8-10M | ★★ | ★★★★★ | ★★ | ⭐⭐⭐⭐ |
| Case 3 | 3-4M | ★★★ | ★★★★ | ★★★ | ⭐⭐⭐⭐⭐ |
| Case 4 | 12-15M | ★ | ★★★★★ | ★ | ⭐⭐⭐ |
| Case 5 | 0.8M | ★★★★★ | ★★ | ★★★★★ | ⭐⭐⭐ |
| Case 6 | 3M | ★★★ | ★★★ | ★★★ | ⭐⭐⭐ |

## 🤝 貢献

各実装は独立しているため、簡単にカスタマイズ・拡張できます。

## 📄 ライセンス

MIT License
