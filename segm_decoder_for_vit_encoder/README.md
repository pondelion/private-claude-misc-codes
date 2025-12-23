# ViT Decoder Implementations for Segmentation

ViT系モデル(空間解像度が一定)向けの6種類のDecoderアーキテクチャ実装。

## 概要

通常のCNN系エンコーダー(ResNet, EfficientNet等)では空間解像度が段階的に減少しますが、ViT系モデルでは全層で解像度が一定です。このため、従来のUNet-styleデコーダー(段階的アップサンプリング)は使用できません。

**ViTの特徴:**
- 全エンコーダー層で同じ空間解像度 (例: H/16 × W/16)
- 各層で異なるレベルの抽象度を持つ特徴
- チャネル数も全層で同じことが多い (例: 全層384ch)

## 実装案の比較

| 案 | フォルダ | 速度 | 精度 | メモリ | パラメータ数 | 特徴 |
|---|---|---|---|---|---|---|
| 1 | case1_mlp_mixer | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 少 | 全層concat + MLP Mixer |
| 2 | case2_multiscale_fpn | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 多 | ASPP + Attention |
| 3 | case3_hierarchical_attention | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 中 | 段階的融合 + Attention |
| 4 | case4_cross_attention | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ | 多 | Transformer Decoder |
| 5 | case5_weighted_sum | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | 最少 | 学習可能な加重平均 |
| 6 | case6_fpn_style | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 中 | FPN-style lateral connection |

## 各案の詳細

### Case 1: MLP Mixer (速度重視)

**アプローチ:**
```
全エンコーダー層 → concat → MLP Mixer blocks → 出力
```

**利点:**
- シンプルで実装が容易
- 推論速度が速い
- メモリ効率が良い

**欠点:**
- 精度は他の案に劣る可能性
- 受容野の拡大が限定的

**推奨ケース:**
- リアルタイム推論が必要
- メモリが限られている
- ベースライン実装として

---

### Case 2: Multi-Scale Feature Pyramid (精度重視)

**アプローチ:**
```
各層 → ASPP(multi-scale context) → concat → Attention → 出力
```

**利点:**
- マルチスケール情報を効果的に活用
- Channel/Spatial Attentionで重要特徴を選択
- セグメンテーションタスクと相性が良い

**欠点:**
- 計算コストが高い
- パラメータ数が多い

**推奨ケース:**
- 精度が最優先
- マルチスケール情報が重要(心電図の細かいパターン検出など)
- 計算リソースに余裕がある

---

### Case 3: Hierarchical Channel Attention (バランス型)

**アプローチ:**
```
feat[0] + feat[1] → attention → fused[0]
fused[0] + feat[2] → attention → fused[1]
...
```

**利点:**
- 速度と精度のバランスが良い
- 段階的処理で学習が安定
- 解釈性が高い(各段階の貢献度を可視化可能)

**欠点:**
- 並列化しづらい(段階的処理のため)

**推奨ケース:**
- 速度と精度の両立が必要
- 安定した学習が重要
- 最初の実用実装として

---

### Case 4: Cross-Attention (精度最重視)

**アプローチ:**
```
Query: 最深層
Key/Value: 全層concat
→ Transformer Decoder layers → 出力
```

**利点:**
- 最も表現力が高い
- 層間の相互作用を柔軟にモデル化
- 適応的な特徴選択

**欠点:**
- 計算コストが非常に高い(O(N²))
- メモリ使用量が多い
- 学習が不安定になる可能性

**推奨ケース:**
- 精度が絶対的に重要
- 計算リソースが豊富
- 研究・実験段階

---

### Case 5: Weighted Sum (超速度重視)

**アプローチ:**
```
各層 → 1x1 conv(次元統一) → 学習可能な重みで加重平均 → 出力
```

**2つのバリエーション:**
1. **Simple Version**: グローバルな重み(全位置で同じ)
2. **Spatial-Aware Version**: 位置ごとに異なる重み

**利点:**
- 最も高速
- 最少パラメータ数
- メモリ効率が最高

**欠点:**
- 表現力は限定的
- 複雑なパターンの捉えづらさ

**推奨ケース:**
- 速度が最優先
- エッジデバイスでの推論
- 軽量モデルが必要

---

### Case 6: FPN-Style (実績のあるアプローチ)

**アプローチ:**
```
深い層から浅い層へtop-down pathway
各層でlateral connectionとrefinement
全FPN層をfusion → 出力
```

**2つのバリエーション:**
1. **Basic Version**: Concatenation fusion
2. **V2 with Attention**: Attention-based fusion

**利点:**
- FPNの実績ある設計思想
- 段階的refinementで安定
- マルチレベル特徴の活用

**欠点:**
- ViTでは解像度が同じため、本来のFPNほどの効果は期待できない可能性

**推奨ケース:**
- 実績のあるアーキテクチャを好む
- マルチレベル特徴表現が重要
- 安定性重視

## 使用例

```python
import torch
import timm

# ViT encoder
encoder = timm.create_model('vit_small_patch16_dinov3.lvd1689m',
                            features_only=True,
                            pretrained=True)

# Example: Case 3 (Hierarchical Attention)
from case3_hierarchical_attention.decoder import ViTDecoderHierarchicalAttention

decoder = ViTDecoderHierarchicalAttention(
    encoder_channels=[384, 384, 384],
    decoder_channels=256,
    use_attention=True,
    final_upsampling=16
)

# Forward pass
x = torch.randn(2, 3, 512, 1280)  # (B, C, H, W)
features = encoder(x)
output = decoder(*features)

print(f"Output shape: {output.shape}")  # (2, 256, 512, 1280)
```

## 推奨フローチャート

```
1. まず試す
   ↓
   Case 5 (Weighted Sum) - 最軽量ベースライン
   ↓
   精度が不足?
   ↓
2. バランス型を試す
   ↓
   Case 3 (Hierarchical Attention) または Case 6 (FPN-Style)
   ↓
   まだ精度が不足?
   ↓
3. 精度重視
   ↓
   Case 2 (Multi-Scale FPN) または Case 4 (Cross-Attention)
```

## 心電図セグメンテーションでの推奨

**現在のタスク(SignalSegModelV7)を考慮:**

1. **第一候補: Case 2 (Multi-Scale FPN)**
   - 理由: 心電図の細かいパターン(P波、QRS、T波等)と大域的な文脈の両方が重要
   - ASPP による multi-scale context が有効

2. **第二候補: Case 3 (Hierarchical Attention)**
   - 理由: 速度と精度のバランス、実装の安定性
   - 段階的な特徴統合が心電図の階層的パターンに適合

3. **軽量版が必要な場合: Case 5 Spatial-Aware Version**
   - 位置ごとに異なる重みで、リード間の違いを捉える

## テスト方法

各フォルダ内のdecoder.pyを直接実行すると、動作確認とパラメータ数のカウントができます:

```bash
cd case1_mlp_mixer
python decoder.py

cd ../case2_multiscale_fpn
python decoder.py

# ... 他の案も同様
```

## カスタマイズポイント

全ての実装で以下のパラメータが調整可能:

- `decoder_channels`: デコーダーの出力チャネル数
- `final_upsampling`: 最終的なアップサンプリング倍率(通常16)
- `upsampling_mode`: 'bilinear' or 'nearest'

各案固有のパラメータは、各ファイルのdocstringを参照してください。

## 実装時の注意点

1. **ViTの特徴抽出設定**
   ```python
   # features_only=True を指定
   encoder = timm.create_model('vit_...', features_only=True)

   # 出力層の選択 (デフォルトは最後の3層)
   # より多くの層が欲しい場合:
   encoder = timm.create_model('vit_...',
                               features_only=True,
                               out_indices=(0, 2, 4, 6, 8, 11))
   ```

2. **次元の確認**
   - ViT の出力は (B, C, H/16, W/16) が一般的
   - final_upsampling=16 で元の解像度に戻す

3. **メモリ管理**
   - 大きな画像サイズ(512×1280など)の場合、Case 4は特にメモリを消費
   - gradient checkpointing の使用を検討

## ライセンス

各実装はMITライセンスの下で提供されます。
