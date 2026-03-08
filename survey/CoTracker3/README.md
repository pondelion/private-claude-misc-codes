# CoTracker3 Understanding

**CoTracker3 (Simpler and Better Point Tracking by Pseudo-Labelling Real Videos)** の簡潔な擬似コード実装とドキュメント

論文: [CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos](https://arxiv.org/abs/2410.11831)
公式実装: https://github.com/facebookresearch/co-tracker

---

## 目次

1. [概要](#概要)
2. [核心的アイデア](#核心的アイデア)
3. [アーキテクチャ](#アーキテクチャ)
4. [主要イノベーション](#主要イノベーション)
5. [処理フロー](#処理フロー)
6. [訓練](#訓練)
7. [推論](#推論)
8. [データセット](#データセット)
9. [実験結果](#実験結果)
10. [形状ガイド](#形状ガイド)
11. [FAQ](#faq)

---

## 概要

CoTracker3は**ビデオ中の任意の点を長期間にわたって追跡する (Point Tracking)** タスクのための手法です。先行手法の不要なコンポーネントを除去・簡略化し、さらに**Pseudo-Labelling (擬似ラベル)** による半教師あり学習で、1000倍少ないデータで既存SoTAを超える性能を実現しています。

### 従来のPoint Tracker vs CoTracker3

```
【従来手法 (TAPIR, BootsTAPIR, LocoTrack)】
1. CNN特徴抽出
2. グローバルマッチング (全フレーム間の対応付け)
3. ローカルリファインメント (近傍探索)
4. 各点を独立に追跡 → オクルージョンに弱い
→ 複雑なパイプライン、大量データ (15M動画) が必要

【CoTracker3】
1. CNN特徴抽出
2. 4D相関ボリューム計算
3. Transformerによる反復的リファインメント (複数点を共同追跡)
→ グローバルマッチング不要、シンプルで高速
→ 15k動画のPseudo-Labelで十分 (1000倍少ない)
```

### 主要な特徴

- **シンプルなアーキテクチャ**: グローバルマッチング除去、MLP相関処理、統合的な可視性更新
- **4D相関特徴**: LocoTrackの4D相関をさらにシンプルなMLPで処理
- **共同追跡 (Joint Tracking)**: 複数点のクロストラックアテンションでオクルージョンに頑健
- **Virtual Tracks**: 64個の仮想トークンで効率的なグローバル文脈集約
- **Pseudo-Labelling**: 複数の教師モデルのアンサンブルで実動画にラベル付け
- **Online/Offline両対応**: 同じアーキテクチャでストリーミング/バッチ処理が可能
- **25Mパラメータ**: CoTracker (45M) の約半分、LocoTrack (12M) より大きいが27%高速

### 性能比較 (TAP-Vid Mean δ_avg^vis)

| 手法 | 訓練データ | パラメータ | Mean δ_avg^vis |
|------|-----------|-----------|---------------|
| TAPIR | Kubric | 31M | 68.0 |
| CoTracker | Kubric | 45M | 73.1 |
| LocoTrack | Kubric | 12M | 75.1 |
| BootsTAPIR | Kubric + 15M実動画 | 78M | 75.0 |
| **CoTracker3 online** | **Kubric + 15k実動画** | **25M** | **76.1** |
| **CoTracker3 offline** | **Kubric + 15k実動画** | **25M** | **76.6** |

---

## 核心的アイデア

### 1. Pseudo-Labelling による半教師あり学習

Point Trackingでは実動画のアノテーションが極めて困難 (ピクセル精度が必要) なため、従来は合成データ (Kubric) のみで訓練していました。

CoTracker3のキーアイデアは、**合成データで訓練した複数のTrackerを教師として、実動画に擬似ラベルを付与し、それで学生モデルを訓練する**ことです。

```python
# 擬似ラベル生成パイプライン
teachers = [CoTracker3_online, CoTracker3_offline, CoTracker, TAPIR]
# 全て合成データ (Kubric) のみで訓練済み

for video in real_videos:
    # バッチごとにランダムに教師を選択
    teacher = random.choice(teachers)

    # SIFTで良質なクエリ点を8フレームからサンプリング
    query_points = sift_sampling(video, num_frames=8, num_points=384)

    # 教師モデルで擬似ラベル (トラック座標) を生成
    pseudo_tracks = teacher(video, query_points)  # (B, T, N, 2)

    # 学生モデルを擬似ラベルで訓練
    student_tracks = student(video, query_points)
    loss = huber_loss(student_tracks, pseudo_tracks)
```

**なぜ教師を超えられるのか？**

1. **データ多様性**: 実動画は合成データよりはるかに多様
2. **ドメインギャップ軽減**: 実動画分布での学習
3. **アンサンブル効果**: 複数教師のランダム選択でノイズ軽減
4. **相補性**: offline教師はオクルージョンに強い、online教師はクエリ点付近の精度が高い

### 2. アーキテクチャの簡略化

CoTracker3は先行手法から以下を除去・簡略化:

```
【除去されたコンポーネント】
✗ グローバルマッチングモジュール (TAPIR, BootsTAPIR, LocoTrack)
✗ アドホックな相関処理ネットワーク (LocoTrack)
✗ 可視性専用の別ネットワーク (CoTracker)
✗ トラック特徴の更新メカニズム (CoTracker)

【簡略化されたコンポーネント】
○ 4D相関 → シンプルなMLP (49×49 → 256)
○ Transformerトークン: 相関特徴 + Fourier変位埋め込みのみ
○ 可視性・信頼度: Transformerの出力ヘッドで統合的に更新
```

---

## アーキテクチャ

CoTracker3は3つの主要コンポーネントで構成されます:

```
┌─────────────────────────────────────────────────────────────────┐
│                         CoTracker3                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────────┐  ┌────────────────────┐ │
│  │ BasicEncoder │→ │ 4D Correlation  │→ │ EfficientUpdate    │ │
│  │ (CNN特徴)    │  │ (多スケール相関) │  │ Former             │ │
│  │ stride=4     │  │ 4レベル         │  │ (Time+Spaceアテン) │ │
│  └─────────────┘  └─────────────────┘  └────────────────────┘ │
│        ↓                  ↓                      ↓             │
│  fmaps: (B,T,       corr_embs:              delta_coords,      │
│   128,H/4,W/4)      (B,T,N,1024)           delta_vis,          │
│                                              delta_conf         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              M回反復 (iterative refinement)              │   │
│  │  coords^(m+1) = coords^(m) + delta_coords^(m+1)        │   │
│  │  vis^(m+1)    = vis^(m)    + delta_vis^(m+1)            │   │
│  │  conf^(m+1)   = conf^(m)   + delta_conf^(m+1)          │   │
│  │  → 毎回、更新された座標で相関を再計算                    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1. BasicEncoder (CNN特徴抽出)

**役割**: 各フレームからd=128次元の特徴マップを抽出

```python
# 入力
video: (B*T, 3, H, W)

# 多スケール特徴抽出
conv1: 7×7, stride=2, 64ch  →  (B*T, 64, H/2, W/2)
layer1: ResBlock×2, 64ch    →  a: (B*T, 64, H/2, W/2)
layer2: ResBlock×2, 96ch    →  b: (B*T, 96, H/4, W/4)
layer3: ResBlock×2, 128ch   →  c: (B*T, 128, H/8, W/8)
layer4: ResBlock×2, 128ch   →  d: (B*T, 128, H/16, W/16)

# 全スケールをH/4×W/4にリサイズして結合
a,b,c,d = bilinear_upsample(a,b,c,d, size=(H/4, W/4))
x = conv2(cat([a,b,c,d]))  →  (B*T, 256, H/4, W/4)
x = conv3(x)               →  (B*T, 128, H/4, W/4)

# L2正規化
fmaps = x / sqrt(max(sum(x^2, dim=-1), 1e-12))

# 出力
fmaps: (B, T, 128, H/4, W/4)
```

### 2. 4D Correlation Features

**役割**: クエリ点の近傍と各フレームの対応点近傍間の密な相関を計算

```python
# 4レベルのピラミッド構築
fmaps_pyramid = [fmaps]                        # Level 0: (B,T,128,H/4,W/4)
for i in range(3):
    fmaps = avg_pool2d(fmaps, 2)               # Level i+1: 解像度1/2
    fmaps_pyramid.append(fmaps)

# 各レベルで4D相関を計算
for level in range(4):
    # クエリフレームの点周辺から(2Δ+1)²個のサポート特徴を抽出 (Δ=3 → 7×7=49)
    track_feat_support = sample(fmaps[t_q], query_coords, radius=3)
    # shape: (B, N, 49, 128) → 49個のサポート点の128次元特徴

    # 各フレームの推定トラック位置周辺から近傍特徴を抽出
    corr_feat = sample(fmaps[t], track_coords, radius=3)
    # shape: (B, T, N, 7, 7, 128) → 7×7近傍の128次元特徴

    # 4D相関ボリューム: サポート点 × 近傍点 の全ペア内積
    corr_volume = einsum("btnhwc,bnijc->btnhwij", corr_feat, track_feat_support)
    # shape: (B, T, N, 7, 7, 7, 7) = (B, T, N, 49, 49)

    # MLPで次元削減
    corr_emb = MLP(corr_volume.flatten())  # (B*T*N, 49*49) → (B*T*N, 256)

# 全レベル結合
corr_embs = cat(corr_embs_all_levels)  # (B, T, N, 256×4) = (B, T, N, 1024)
```

### 3. EfficientUpdateFormer (反復的リファインメント)

**役割**: 相関特徴と変位情報をもとに、Transformerで座標・可視性・信頼度を反復更新

```python
# Transformer入力の構築 (全1110次元)
transformer_input = cat([
    vis,                    # (B, T, N, 1)    可視性
    confidence,             # (B, T, N, 1)    信頼度
    corr_embs,              # (B, T, N, 1024) 4レベル相関特徴
    rel_pos_emb,            # (B, T, N, 84)   相対変位のFourier Encoding
])  # → (B, T, N, 1110)

# 入力変換 + 時間埋め込み加算
x = Linear(1110 → 384) + time_embedding

# Transformer構造 (time_depth=3, space_depth=3)
for i in range(3):
    # 時間方向Self-Attention (各点の全フレーム間)
    x = TimeAttnBlock(x)       # (B*N, T, 384)

    # 空間方向: Virtual Tracksを介した間接的アテンション
    v = SelfAttn(virtual)            # (B*T, 64, 384) 仮想トークン自己注意
    v = CrossAttn(virtual ← x)      # 実トラック → 仮想トークン
    x = CrossAttn(x ← virtual)      # 仮想トークン → 実トラック

# 出力ヘッド
delta = Linear(384 → 4)  # [delta_x, delta_y, delta_vis, delta_conf]

# 座標・可視性・信頼度を更新
coords += delta[:, :, :, :2]
vis += delta[:, :, :, 2]
conf += delta[:, :, :, 3]
```

---

## 主要イノベーション

### 1. 4D相関のMLP簡略化

**問題**: LocoTrackの4D相関処理はアドホックな専用モジュールを使用し、複雑

**解決**: シンプルなMLP (2層) で十分

```python
# LocoTrack: 専用の4D畳み込みモジュール (複雑)
# CoTracker3: 単純なMLP
self.corr_mlp = Mlp(
    in_features=49 * 49,      # 7×7×7×7 = 2401次元の相関ボリュームをフラット化
    hidden_features=384,       # 隠れ層
    out_features=256,          # 出力256次元
)
# GELUアクティベーション、ドロップアウトなし
```

**効果**: LocoTrackより27%高速 (209µs/frame vs 290µs/frame)

### 2. グローバルマッチングの除去

**問題**: TAPIR, BootsTAPIR, LocoTrackは全フレーム間のグローバルマッチングステージを使用

**解決**: 4D相関 + Transformerの反復リファインメントで十分

```
【TAPIR/BootsTAPIR/LocoTrack】
全フレーム × 全ピクセルの相関マップ → 大域的位置推定
     ↓
ローカルリファインメント

【CoTracker3】
クエリ点近傍 × 推定位置近傍の局所的4D相関のみ
     ↓
Transformerで反復的に位置を更新 (6回)
→ グローバルマッチング不要で同等以上の精度
```

### 3. Virtual Tracks (仮想トラック)

**目的**: N個のトラック間の空間的アテンションはO(N²)で計算量が大きい → 64個の仮想トークンを介して効率化

```python
# 学習可能な仮想トラック
virtual_tracks = nn.Parameter(randn(1, 64, 1, 384))

# 空間アテンション (各時間ステップごと)
for each time_step:
    # Step 1: 仮想トークン同士の自己注意
    virtual = SelfAttention(virtual)            # (B*T, 64, 384)

    # Step 2: 実トラック → 仮想トークン (情報集約)
    virtual = CrossAttention(Q=virtual, KV=real_tracks)  # (B*T, 64, 384)

    # Step 3: 仮想トークン → 実トラック (情報配信)
    real_tracks = CrossAttention(Q=real_tracks, KV=virtual)  # (B*T, N, 384)

# 計算量: O(N×64) ≪ O(N²) for N >> 64
# 最終出力からは仮想トラックを除去
```

### 4. 相対変位のFourier Encoding

**目的**: 隣接フレーム間の相対的な動き情報をTransformerに入力

```python
# 前方・後方の相対変位を計算
rel_forward = coords[:, :-1] - coords[:, 1:]    # t→t+1方向
rel_backward = coords[:, 1:] - coords[:, :-1]   # t+1→t方向

# 正規化 (モデル解像度でスケール)
scale = tensor([W/stride, H/stride])  # [128, 96]
rel_forward = rel_forward / scale
rel_backward = rel_backward / scale

# Fourier Encoding (min_deg=0, max_deg=10)
# 入力: (B, T, N, 4) = [rel_fw_x, rel_fw_y, rel_bw_x, rel_bw_y]
# 各次元に対して: sin(x*2^i), cos(x*2^i) for i=0,...,9
# 出力: 4 + 4×10×2 = 84次元
rel_pos_emb = posenc(cat([rel_forward, rel_backward]), min_deg=0, max_deg=10)
# shape: (B, T, N, 84)
```

### 5. 信頼度 (Confidence) の推定

**目的**: 追跡精度の自己評価。予測点が真値から12ピクセル以内かどうかを分類

```python
# Confidence = "予測点が正しい位置にあるか" の確率
# Ground Truth:
gt_conf = (||predicted - gt|| < 12px).float()

# 学習:
loss_conf = BCE(sigmoid(confidence), gt_conf)

# 推論時:
# visibility × confidence を閾値処理して、
# 追跡に失敗した点をフィルタリング
final_visibility = sigmoid(vis) * sigmoid(conf) > threshold
```

---

## 処理フロー

### Offline推論フロー (全フレーム一括処理)

```
入力: video (B, T, 3, H, W), queries (B, N, 3) = [frame, x, y]

┌──────────────────────────────────────────┐
│ ステップ1: 前処理                        │
├──────────────────────────────────────────┤
│ video = 2 * (video / 255) - 1.0          │
│ queried_coords = queries[:, :, 1:3]      │
│ queried_coords /= stride  (=4)           │
└──────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────┐
│ ステップ2: CNN特徴抽出                   │
├──────────────────────────────────────────┤
│ fmaps = BasicEncoder(video)              │
│ shape: (B, T, 128, H/4, W/4)            │
│ → L2正規化                              │
└──────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────┐
│ ステップ3: 相関ピラミッド構築            │
├──────────────────────────────────────────┤
│ Level 0: (B, T, 128, H/4, W/4)          │
│ Level 1: avg_pool → (B, T, 128, H/8, W/8)   │
│ Level 2: avg_pool → (B, T, 128, H/16, W/16) │
│ Level 3: avg_pool → (B, T, 128, H/32, W/32) │
└──────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────┐
│ ステップ4: クエリ点のサポート特徴取得    │
├──────────────────────────────────────────┤
│ 各レベルで:                              │
│   track_feat_support = sample(           │
│     fmaps[t_q], query_coords,            │
│     radius=3  → 7×7=49サポート点         │
│   )                                      │
│ shape: (B, N, 49, 128) per level         │
└──────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────┐
│ ステップ5: 反復リファインメント (M=4~6回)│
├──────────────────────────────────────────┤
│ 初期化:                                  │
│   coords = query_coords.expand(B,T,N,2)  │
│   vis = zeros(B, T, N)                   │
│   conf = zeros(B, T, N)                  │
│                                          │
│ for m in range(M):                       │
│   coords = coords.detach()  # 勾配切断   │
│                                          │
│   # 4D相関計算 (4レベル)                 │
│   corr_embs = []                         │
│   for level in range(4):                 │
│     corr_feat = sample_neighborhood(     │
│       fmaps_pyramid[level], coords       │
│     )                                    │
│     corr_volume = einsum(corr_feat,      │
│                          track_support)  │
│     corr_emb = MLP(corr_volume)          │
│     corr_embs.append(corr_emb)           │
│   corr_embs = cat(corr_embs)  # 1024    │
│                                          │
│   # 相対変位Fourier Encoding             │
│   rel_pos = posenc(displacements)  # 84  │
│                                          │
│   # Transformer入力組み立て               │
│   input = cat([vis, conf,                │
│                corr_embs, rel_pos])      │
│   input += time_embedding                │
│                                          │
│   # Transformerで更新量予測              │
│   delta = UpdateFormer(input)  # 4次元   │
│                                          │
│   # 座標・可視性・信頼度を更新           │
│   coords += delta[:2]                    │
│   vis += delta[2]                        │
│   conf += delta[3]                       │
└──────────────────────────────────────────┘
            ↓
┌──────────────────────────────────────────┐
│ ステップ6: 後処理                        │
├──────────────────────────────────────────┤
│ coords *= stride  (特徴空間→画像空間)    │
│ vis = sigmoid(vis)                       │
│ conf = sigmoid(conf)                     │
└──────────────────────────────────────────┘
            ↓
出力: tracks (B, T, N, 2), visibility (B, T, N), confidence (B, T, N)
```

### Online推論フロー (スライディングウィンドウ)

```
入力: video_chunk (B, T_chunk, 3, H, W) をチャンクごとに受信

初期化:
  model.init_video_online_processing()
  → online_ind = 0
  → online_track_feat = [None] * 4
  → online_coords_predicted = None

ウィンドウ処理 (window_len=16, step=8):
┌──────────────────────────────────────────┐
│ Window 0: frames 0-15                    │
│   ├─ 特徴抽出 & 相関計算                 │
│   ├─ Transformer反復リファインメント      │
│   └─ 予測をキャッシュ                    │
├──────────────────────────────────────────┤
│ Window 1: frames 8-23                    │
│   ├─ 前ウィンドウの予測 (frames 8-15)    │
│   │   を初期値として引き継ぎ             │
│   ├─ 新フレーム (16-23) を処理           │
│   └─ 50%オーバーラップで滑らかに接続     │
├──────────────────────────────────────────┤
│ Window 2: frames 16-31                   │
│   └─ ...                                │
└──────────────────────────────────────────┘

特徴のキャッシュ:
  - クエリ点の特徴はウィンドウ間で累積的に更新
  - 新しいウィンドウでクエリフレームが含まれる場合のみ特徴を追加
  - sample_mask = (queried_frames >= left) & (queried_frames < right)
  - online_track_feat[level] += track_feat * sample_mask
```

---

## 訓練

### 2段階訓練パイプライン

```
Stage 1: 合成データ (Kubric) での事前訓練
  ├─ データ: TAP-Vid-Kubric (6000シーケンス, 512×512, 120フレーム)
  ├─ Online: T=64, window=16, 384クエリ点, 50k iterations
  ├─ Offline: T∈{30,...,60}, 512クエリ点, 50k iterations
  ├─ 損失: L_track + L_occl + L_conf (全損失)
  └─ GPU: 32× NVIDIA A100 80GB

Stage 2: 実動画での疑似ラベル学習
  ├─ データ: ~100k本のインターネット動画 (30秒/本)
  ├─ 教師: CoTracker3 online/offline, CoTracker, TAPIR (Kubric訓練済み)
  ├─ クエリ点: SIFTで8フレームから384点サンプリング
  ├─ 損失: L_track のみ (vis/confヘッドはフリーズ)
  ├─ 15k iterations, lr=5e-5
  └─ GPU: 同上
```

### 損失関数

```python
# 1. トラック損失 (Huber Loss, 指数的重み付け)
L_track = Σ_{m=1}^{M} γ^{M-m} × (1_occ/5 + 1_vis) × Huber(P^(m), P*)
# γ=0.8: 後の反復ほど重い重み
# オクルージョン中の点は重み1/5

# 2. 可視性損失 (Binary Cross Entropy)
L_occl = Σ_{m=1}^{M} γ^{M-m} × BCE(σ(V^(m)), V*)

# 3. 信頼度損失 (Binary Cross Entropy)
L_conf = Σ_{m=1}^{M} γ^{M-m} × BCE(σ(C^(m)), 1[||P^(m) - P*|| < 12])
# 予測が12ピクセル以内かどうかを分類

# Pseudo-Label学習時:
# L_trackのみ使用 (vis/confヘッドはフリーズ)
```

### 訓練設定

```python
# 共通設定
optimizer = AdamW(β1=0.9, β2=0.999, weight_decay=1e-5)
precision = bfloat16
gradient_clipping = True

# Stage 1 (Kubric)
lr = 5e-4
scheduler = linear_warmup(1000 steps) + cosine_decay
iterations = 50_000
batch_size = 1 (per GPU) × 32 GPUs

# Stage 2 (Real data)
lr = 5e-5
scheduler = cosine_decay (warmupなし)
iterations = 15_000
batch_size = 1 (per GPU) × 32 GPUs
# SIFTが十分な点を検出できない動画はスキップ
```

### Online vs Offline訓練の違い

| 項目 | Online | Offline |
|------|--------|---------|
| 入力長 | T=64 | T∈{30,...,60} |
| ウィンドウ長 | 16 | T (全フレーム) |
| 方向 | 前方のみ | 前方+後方 |
| 損失計算 | クエリフレーム以降のウィンドウのみ | 全フレーム |
| 時間埋め込み | 固定 (16次元) | 線形補間 (60次元→T次元) |
| クエリ点 | 動画の前半にバイアス | 時間的に均一 |

---

## 推論

### Predictor API

```python
# Offlineモード
predictor = CoTrackerPredictor(checkpoint="scaled_offline.pth")
tracks, visibility = predictor(
    video,                      # (B, T, 3, H, W) [0-255]
    queries=queries,            # (B, N, 3) [frame, x, y] or None
    grid_size=10,               # クエリなし時のグリッドサイズ
    grid_query_frame=0,         # グリッドの開始フレーム
    segm_mask=mask,             # (B, 1, H, W) マスク内のみ追跡
    backward_tracking=False,    # 後方追跡も行うか
)
# tracks: (B, T, N, 2), visibility: (B, T, N)

# Onlineモード
predictor = CoTrackerOnlinePredictor(checkpoint="scaled_online.pth")
# 最初のチャンク
predictor(video_chunk_0, is_first_step=True, queries=queries)

# 以降のチャンク
for chunk in video_chunks:
    tracks, visibility = predictor(chunk, is_first_step=False)
```

### 推論時の工夫

```python
# サポートポイントの追加 (TAP-Vid評価時)
# 1点のクエリ点に対して:
#   - 5×5のグローバルグリッド点 (25点)
#   - 8×8のローカルグリッド点 (64点)
# を追加して共同追跡 → クロストラックアテンションの効果を引き出す

# 可視性の閾値処理
final_visibility = sigmoid(vis) * sigmoid(conf)
# AJメトリクスの改善に寄与
```

---

## データセット

### 訓練データ

| データセット | 種類 | 規模 | 用途 |
|-------------|------|------|------|
| **TAP-Vid-Kubric** | 合成 | 6000シーケンス | Stage 1: 事前訓練 |
| **インターネット動画** | 実 | ~100k本 (30秒/本) | Stage 2: Pseudo-Label学習 |

### 評価データ

| データセット | 種類 | 規模 | 特徴 |
|-------------|------|------|------|
| **TAP-Vid-Kinetics** | 実 | 1144動画 | YouTube動画、複雑なカメラ動作 |
| **TAP-Vid-DAVIS** | 実 | 30動画 | セグメンテーション用動画 |
| **RGB-Stacking** | 合成 | - | ロボットのテクスチャレス物体 |
| **RoboTAP** | 実 | 265動画 | ロボット操作タスク |
| **Dynamic Replica** | 合成 | 20シーケンス | オクルージョン評価用 |

### データフォーマット

```python
# クエリ点
queries: (B, N, 3)
# queries[b, n] = [t_query, x_query, y_query]
# t_query: クエリフレームのインデックス (float → long)
# x_query, y_query: 画像座標 (ピクセル単位)

# トラック (出力)
tracks: (B, T, N, 2)
# tracks[b, t, n] = [x_t, y_t]  # フレームtでの点nの座標

# 可視性
visibility: (B, T, N)
# 0: オクルージョン / フレーム外, 1: 可視

# 信頼度
confidence: (B, T, N)
# 0: 追跡失敗の可能性大, 1: 追跡成功の確信
```

---

## 実験結果

### TAP-Vid ベンチマーク (Table 1)

| 手法 | 訓練データ | Kinetics AJ↑ | RGB-S AJ↑ | DAVIS AJ↑ | Mean δ_avg^vis↑ |
|------|-----------|-------------|----------|----------|---------------|
| TAPIR | Kub | 49.6 | 55.5 | 56.2 | 68.0 |
| CoTracker | Kub | 49.6 | 67.4 | 61.8 | 73.1 |
| LocoTrack | Kub | 52.9 | 69.7 | 62.9 | 75.1 |
| **CoTracker3 online** | Kub | 54.1 | 71.1 | 64.5 | 75.1 |
| **CoTracker3 offline** | Kub | 53.5 | **74.0** | 63.3 | 75.9 |
| BootsTAPIR | Kub+15M | 54.6 | 70.8 | 61.4 | 75.0 |
| **CoTracker3 online** | **Kub+15k** | **55.8** | 71.7 | 63.8 | **76.1** |
| **CoTracker3 offline** | **Kub+15k** | 54.7 | **74.3** | **64.4** | **76.6** |

### Dynamic Replica (オクルージョン追跡, Table 2)

| 手法 | δ_avg^vis↑ | δ_avg^occ↑ | パラメータ | 速度 (µs/frame/point) |
|------|-----------|-----------|-----------|---------------------|
| TAPIR | 66.1 | 27.2 | 31M | 293 |
| CoTracker | 68.9 | 37.6 | 45M | 472 |
| LocoTrack | 71.4 | 29.8 | 12M | 290 |
| **CoTracker3 online** | **72.9** | **41.0** | 25M | 405 |
| **CoTracker3 offline** | 69.8 | **41.8** | 25M | **209** |

### 主要な観察

1. **Kubricのみでも強力**: CoTracker3 onlineはKubricのみでもLocoTrackと同等 (δ_avg=75.1)
2. **1000倍少ないデータで超越**: 15k実動画 >> BootsTAPIR (15M実動画)
3. **オクルージョンに圧倒的**: δ_avg^occ=41.8 (CoTracker3 offline) vs 29.8 (LocoTrack)
4. **Cross-track attentionが鍵**: オクルージョン追跡で+5.1pt (Table 3)
5. **Offlineがオクルージョンに強い**: 全フレーム同時参照で軌跡補間が可能

---

## 形状ガイド

### 入力・中間・出力形状

| 段階 | 名称 | 形状 | 説明 |
|------|------|------|------|
| **入力** | video | `(B, T, 3, H, W)` | RGB動画 [0-255] |
| | queries | `(B, N, 3)` | [frame_idx, x, y] |
| **CNN特徴** | fmaps | `(B, T, 128, H/4, W/4)` | L2正規化済み特徴マップ |
| **ピラミッド** | fmaps_pyramid[0] | `(B, T, 128, H/4, W/4)` | Level 0 |
| | fmaps_pyramid[1] | `(B, T, 128, H/8, W/8)` | Level 1 |
| | fmaps_pyramid[2] | `(B, T, 128, H/16, W/16)` | Level 2 |
| | fmaps_pyramid[3] | `(B, T, 128, H/32, W/32)` | Level 3 |
| **クエリ特徴** | track_feat_support | `(B, 49, N, 128)` | 7×7サポート点特徴 (per level) |
| **相関** | corr_volume | `(B, T, N, 7, 7, 7, 7)` | 4D相関ボリューム (per level) |
| | corr_emb | `(B*T*N, 256)` | MLP圧縮後 (per level) |
| | corr_embs | `(B, T, N, 1024)` | 4レベル結合 |
| **変位埋め込み** | rel_pos_emb | `(B, T, N, 84)` | Fourier Encoding |
| **Transformer入力** | x | `(B, N, T, 1110)` | 全特徴結合 |
| **Transformer内部** | x_proj | `(B, N, T, 384)` | 線形射影後 |
| | virtual_tracks | `(1, 64, 1, 384)` | 学習可能仮想トラック |
| **Transformer出力** | delta | `(B, N, T, 4)` | [Δx, Δy, Δvis, Δconf] |
| **最終出力** | tracks | `(B, T, N, 2)` | 画像座標 (ピクセル単位) |
| | visibility | `(B, T, N)` | sigmoid後 [0, 1] |
| | confidence | `(B, T, N)` | sigmoid後 [0, 1] |

### 軸の意味

- **B**: バッチサイズ
- **T**: フレーム数 (Online: window_len=16, Offline: 動画全体)
- **N**: 追跡点数 (訓練: 384~512, 評価: 1+サポート点)
- **H, W**: 画像の高さ・幅 (モデル内部: 384×512)
- **128**: CNN特徴の次元数 (latent_dim)
- **384**: Transformer隠れ層次元 (hidden_size)
- **1110**: Transformer入力次元 (1+1+1024+84)
- **64**: Virtual Tracksの数

---

## FAQ

### Q1: なぜグローバルマッチングが不要なのか？

**A**: 4D相関特徴 + Transformerの反復リファインメントで代替可能。

4D相関は各フレームのtrack推定位置周辺から49個のサポート点との内積を計算するため、局所的ながら十分なマッチング情報を提供します。Transformerが反復的に位置を更新するたびに相関を再計算するため、段階的に正しい位置に収束します。グローバルマッチングの除去はCoTracker3を27%高速化しています。

### Q2: Virtual Tracksとは何か？

**A**: 64個の学習可能なパラメータで、実トラック間の情報交換を仲介します。

直接的なN点間のSelf-Attention (O(N²)) の代わりに、仮想トークンを介した間接的な情報交換 (O(N×64)) で効率化しています。全トラックのグローバル文脈を集約し、特にオクルージョン中の点の位置推定を助けます。推論時の最終出力からは除去されます。

### Q3: Online版とOffline版の違いは？

**A**: アーキテクチャは同一。訓練方法と推論時の処理が異なります。

| 項目 | Online | Offline |
|------|--------|---------|
| ウィンドウ長 | 16 | T (全フレーム) |
| 追跡方向 | 前方のみ | 双方向 |
| メモリ | 低い (一定) | 高い (Tに比例) |
| オクルージョン | やや弱い | 強い |
| ユースケース | リアルタイム/ストリーミング | バッチ処理 |
| 速度 | 405µs/frame/point | 209µs/frame/point |

### Q4: Pseudo-Labelで教師を超えられるのはなぜ？

**A**: 4つの理由があります。

1. **データ規模**: 合成データより桁違いに多い実動画から学習
2. **ドメインギャップ軽減**: 実動画の分布に直接適合
3. **アンサンブル効果**: ランダムに選択された複数教師でノイズが平均化
4. **相補性**: Offline教師はオクルージョンに強く、Online教師はクエリ点付近の精度が高い

実際、自己学習 (Self-training) だけでも+1.2ptの改善が見られ (Table 4)、実動画への適応自体が有効です。

### Q5: SIFTによるクエリ点サンプリングの理由は？

**A**: SIFTは「追跡しやすい点」を自然に選択するフィルタとして機能します。

記述子が生成できない曖昧な領域 (空、水面など) の点を除外し、訓練の安定性を向上させます。他のサンプリング手法 (SuperPoint, DISK, Uniform) との差は小さいですが、SIFTが全ベンチマークで安定して高い性能を示します (Table 6)。SIFTで十分な点が検出できない動画は訓練時にスキップされます。

### Q6: 信頼度 (Confidence) ヘッドのフリーズが重要な理由は？

**A**: Pseudo-Label学習時にvis/confの教師信号がないため、これらのヘッドをフリーズします。

フリーズしない場合、トラック座標の損失のみでvis/confヘッドが更新され、Kubricで学習した可視性・信頼度の予測能力が劣化します (catastrophic forgetting)。別の線形層でトラック座標用の出力を分離し、vis/confヘッドはKubric訓練時の重みを保持することで、AJ +0.8, OA +3.9の改善が得られます (Table 7)。

### Q7: スケーリング実験から分かることは？

**A**: 100本の実動画からでも改善が見られ、30k本程度で飽和します。

CoTracker3 online/offlineとLocoTrackは30k本程度で教師の性能を超え、改善が飽和します。一方、初期性能が低いCoTracker (v2) は100k本でもまだ改善が続きます。これは学生が教師を超えると、教師からの追加学習の価値が減少するためと考えられます。

### Q8: 反復回数Mの影響は？

**A**: 訓練時はM=4、推論時はM=4~6が一般的です。

各反復で座標が更新され、相関特徴が再計算されるため、徐々に正しい位置に収束します。損失は指数的重み γ^{M-m} (γ=0.8) で後の反復ほど重く重み付けされ、最終的な予測精度を優先します。

### Q9: CoTracker2との違いは？

**A**: CoTracker3は以下の点でCoTracker2から大きく変わっています。

| 項目 | CoTracker2 | CoTracker3 |
|------|-----------|-----------|
| 相関 | 2D相関 + CorrBlock | 4D相関 + MLP |
| トークン入力 | 特徴+フロー+track_mask+vis | 相関+変位Fourier+vis+conf |
| 可視性予測 | 別ネットワーク | Transformer出力ヘッド |
| track特徴更新 | 専用更新モジュール | なし (毎回再サンプリング) |
| パラメータ数 | 45M | 25M |
| Transformer深さ | 6+6 | 3+3 |
| 信頼度推定 | なし | あり |

### Q10: 推論時のサポートポイントは必要か？

**A**: TAP-Vid評価プロトコルでは1点ずつ評価するため、サポートポイントが重要です。

CoTracker3はクロストラックアテンションにより複数点を共同追跡して性能を発揮します。1点のみだとこの利点が失われるため、5×5グローバルグリッド + 8×8ローカルグリッドのサポートポイントを追加して文脈を提供します。

---

## ファイル構成

```
CoTracker3_understanding/
├── README.md                       # 本ファイル
├── main_flow.py                    # メインフロー (全体パイプライン)
├── feature_extraction.py           # BasicEncoder (CNN特徴抽出)
├── correlation_and_transformer.py  # 4D相関 + EfficientUpdateFormer
├── loss_computation.py             # 損失関数 (track, vis, conf)
└── pseudo_label_training.py        # Pseudo-Labelling訓練パイプライン
```

各ファイルの詳細は対応するファイルを参照してください。

---

## 参考文献

- 論文: [CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real Videos](https://arxiv.org/abs/2410.11831)
- 公式実装: https://github.com/facebookresearch/co-tracker
- CoTracker: [It is Better to Track Together](https://arxiv.org/abs/2307.12185)
- PIPs: [Particle Video Revisited](https://arxiv.org/abs/2204.04153)
- TAPIR: [Tracking Any Point with Per-frame Initialization and Temporal Refinement](https://arxiv.org/abs/2306.08637)
- LocoTrack: [Local All-Pair Correspondence for Point Tracking](https://arxiv.org/abs/2407.15420)
- BootsTAPIR: [Bootstrapped Training for Tracking-Any-Point](https://arxiv.org/abs/2402.00847)
- Kubric: [A Scalable Dataset Generator](https://arxiv.org/abs/2203.03570)

---

## まとめ

CoTracker3は以下を実現した、シンプルかつ高性能なPoint Trackerです:

1. **アーキテクチャの簡略化**: グローバルマッチング除去、MLP相関処理、統合出力ヘッドで、CoTrackerの半分のパラメータ
2. **効率的な共同追跡**: Virtual Tracksによるクロストラックアテンションで、特にオクルージョン追跡に優れる
3. **Pseudo-Labelling**: 複数教師のアンサンブルで実動画にラベル付け、1000倍少ないデータでSoTA超え
4. **柔軟性**: 同一アーキテクチャでOnline (リアルタイム) / Offline (高精度) の両モードに対応
5. **高速性**: LocoTrackより27%高速 (クロストラックアテンションがあるにもかかわらず)
