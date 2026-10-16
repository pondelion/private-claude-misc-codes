# VisionHOPE Understanding - 簡略化疑似コード集

**VisionHOPE: Visual Backbones as Self-Modifying Learning Systems**
の理解を目的とした簡略化疑似コード集です。

論文: [arXiv:2609.33325](https://arxiv.org/abs/2609.33325)
公式コード: ローカルクローン `gh/VisionHOPE` ([github.com/PSRben/VisionHOPE](https://huggingface.co/PSRben/VisionHOPE))

## 📋 目次

- [概要](#概要)
- [TTT / 自己参照学習 (HOPE) の仕組み (前提知識)](#ttt--自己参照学習-hope-の仕組み-前提知識)
- [通常のCNN/Transformer/SSMバックボーンとの差分 (新規性はどこにあるか)](#通常のcnntransformerssmバックボーンとの差分-新規性はどこにあるか)
- [公開物の実態 (重要)](#公開物の実態-重要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [VisionHOPEの主要イノベーション](#visionhopeの主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [学習データフォーマット](#学習データフォーマット)
- [FAQ](#faq)

---

## 概要

**VisionHOPEは何なのか:** ResNet (CNN)・Swin/ViT (Transformer)・Vision
Mamba (SSM) と同じ立ち位置の、**汎用の画像バックボーン (特徴抽出器)**
である。入力画像 (B,3,H,W) を渡すと階層的な特徴マップを出力するだけなので、
画像分類ヘッド・物体検出ヘッド (Mask R-CNN)・セマンティックセグメンテーション
ヘッド (UPerNet) など、既存のどのタスクヘッドにも差し替えて組み込める。

ただし、この論文の知的な核心は「新しいネットワーク構造の発明」ではない。
トークンの混ぜ方自体を推論中に書き換える「自己参照学習 (HOPE)」という
**汎用の学習方式**が、Nested Learning (NL, NeurIPS 2025) という別の論文で
すでに提案されており、VisionHOPEは**それを画像ドメインで安定して動くように
翻案した論文**である。ちょうどTTT (Test-Time Training) という汎用方式を
ViT³/Vision-TTTが画像向けに翻案したのと同じ構図にある。HOPE自体が何を
しているかは次節で具体的に説明し、VisionHOPE独自の新規性 (安定化・2次元化)
は[新規性セクション](#通常のcnntransformerssmバックボーンとの差分-新規性はどこにあるか)で扱う。

**どんな場合に使えるか:**
- 画像分類・物体検出・セマンティックセグメンテーションなど、既存のCNN/
  Transformer/SSM/TTT系バックボーンを使っている任意のパイプラインの「差し替え
  候補」として (論文はImageNet-1K分類・COCO検出/インスタンスセグメンテーション・
  ADE20Kセマンティックセグメンテーションの3タスクで評価している)
- パラメータ数を抑えつつ高い精度が欲しい場合に選択肢となる
  (VisionHOPE-B: 91.2M paramsでImageNet-1K Top-1 85.6%)

**キーコントリビューション (VisionHOPE論文が実際に付け加えた部分のみ):**
- **安定性保証付きステップサイズ制御 (中心的な新規性)**: ソフト注入キャップ +
  スペクトラルクランプにより、NL/HOPEの自己参照更新を画像バックボーンに
  適用した際の発散を防ぎ、メモリのフロベニウスノルムが発散しないこと
  (non-expansion) を数学的に証明
- **4方向スキャンへの翻案**: 元々1次元向けのSRNLを、画像を→←↓↑の4方向に
  シーケンス化し、各方向で独立したSRNLインスタンスを行/列単位のチャンクで
  走らせ、チャンネル方向に学習融合する形に適応
- **ImageNet-1K/COCO/ADE20Kで強い性能**: VisionHOPE-B (91.2M params) で
  ImageNet-1K Top-1 85.6%、COCO Box AP 50.5 (3x schedule)、ADE20K mIoU 51.8

(背景として使っている自己参照学習 (SRNL) 自体はNL/HOPE由来であり、
VisionHOPE独自のコントリビューションではない。詳細はFAQ Q4参照)

**タスク:**
- ImageNet-1K 画像分類
- COCO 物体検出・インスタンスセグメンテーション (Mask R-CNNのバックボーン)
- ADE20K セマンティックセグメンテーション (UPerNetのバックボーン)

**モデルサイズ:** VisionHOPE-T/S/B (26.6M / 52.9M / 91.2M params)。
バックボーンはタスク共通で、検出・セグメンテーションは標準的なヘッド
(Mask R-CNN / UPerNet) に差し替えるだけで使える。

---

## TTT / 自己参照学習 (HOPE) の仕組み (前提知識)

VisionHOPEの新規性を理解する前に、まず土台となる「TTT (Test-Time
Training)」と「自己参照 (Self-Referential)」が具体的に何をしているかを
知る必要がある。抽象的な説明ではなく、実際に何が計算されているかを
順番に追う。

### ステップ1: TTT (Test-Time Training) という名前の意味 — 何を「訓練」しているのか

通常のCNN/Transformer/SSMは、学習が終わると全ての重みが固定され、推論時は
その固定された重みで入力を順伝播するだけ。重み自体は推論中、一切変化しない。

TTT (Test-Time Training) はこの前提を破る。TTT系の手法 (DeltaNet, Mamba-2,
TTT-layerなど) は、ネットワークの内部に**もう1つ小さなモデルを持っていて、
その小さなモデルのパラメータを、推論の最中に本物の勾度降下で更新し続ける**。
この「内部モデル」は、行列M 1枚だけで表現される最も単純な線形モデル
`f(k) = M @ k` (keyを受け取ってvalueを出力するだけの写像) である。

画像の各パッチ(トークン)xから、このミニモデルにとっての「1個の訓練サンプル」
(k_t, v_t) が (別の固定された射影で) 生成される。パッチを1つ処理するごとに、
このミニモデルMに対して**訓練を1ステップ実行する**:

```
v̂_t = M_{t-1} @ k_t                   # 現在のMでの予測 (= このミニモデルの「推論」)
損失 L_t = ||v̂_t - v_t||^2             # このトークンに対する予測誤差
M_t = M_{t-1} - η * ∇_M L_t            # Mを勾度降下で1ステップ更新 (= このミニモデルの「訓練」)
    = M_{t-1} - η * outer(v̂_t - v_t, k_t)
```

この「推論」と「訓練」が、画像を処理している最中に、トークンごとに交互に
(実質同時に)起きている。これが"Test-Time Training" (推論時に訓練する)
という名前そのものの意味である。学習済みの固定パラメータを単に適用する
CNN/Transformerとは違い、M自身は画像ごとに・画像の中のパッチの順に、
その場で書き換わっていく一時的なパラメータである。

この「訓練されたばかりのM」は、クエリベクトルq_t (k_t・v_tとは別に、
やはりトークンxから生成される) を使って `y_t = M_t @ q_t` のように読み出され、
これがCNNの畳み込み出力やAttentionの出力と同じ位置づけの「このトークンの
混合結果」として、以降の層 (FFNなど) に渡される。つまりTTTは「Attentionの
代わりに、画像を見ながら成長する小さなモデルを都度訓練し、その出力を
token mixingの結果として使う」手法である。

**(補足) Attentionとの対比**: Attentionは、N個のトークン全部のkey/valueを
保持しておき、各トークンのqueryをN個全部のkeyと比較して(内積+softmax)
重み付き平均を取る方式で、計算量はトークン数Nに対してO(N^2)で増える。
TTTはN個分のkey/valueを保持する代わりに、全部の対応関係を固定サイズの
M 1枚に圧縮してしまうので、Mのサイズも計算量もNに依存しない。ただし
この効率性はTTTの**付随的な利点**であり、"Test-Time Training" という
名前が指しているのはあくまで上記の「推論中にパラメータMを訓練する」
という部分である。

### ステップ2: 「自己参照」が追加するもの (具体的に何をするか)

ステップ1のTTTでは、ミニモデルMの訓練データ (k_t, v_t) と学習率ηは、
現在のパッチx_tを**固定された線形層**に通すだけで計算される。学習が
終われば、この「k_t・v_t・ηの作り方自体」はそれ以降変化しない
(通常のTTT系はここで止まる)。

HOPEはこの「作り方」自体を担当する4つの行列(M_k, M_v, m_η, m_α)を導入し、
これらも**Mと全く同じ更新式で動かし続ける別のメモリ**にする。1パッチ
x_tを処理するたびに、次の2段階が起きる:

**① まず4つのメモリで、今回使うk_t・v_t・η_t・α_tを計算する:**
```
k_t = M_k @ x_t          # 「key」を生成する専用メモリ
v_t = M_v @ x_t          # 「value」を生成する専用メモリ
η_t = φ_η(m_η・x_t)       # 「学習率」を生成する専用メモリ
α_t = φ_α(m_α・x_t)       # 「保持率」を生成する専用メモリ
```

**② M_m・M_k・M_v・m_η・m_αの5つ全部を、①で作ったk_t・v_tを使って、
全く同じ1本の更新式で動かす:**
```
g     = M @ (k_t - v_t)                    # Mでk_tとv_tをそれぞれ写し、その差を取る (誤差)
M_new = M @ T_t  -  η_t * outer(g, k_t)     # T_tはα_t・η_t・k_tから決まる遷移行列
```
この同じ式に、M_m・M_k・M_v・m_η・m_αのそれぞれを「M」として代入するだけで
5つ全部が更新される(実装は[srnl.py](srnl.py)の`sr_dgd_update()`/`sr_dgd_update_vector()`)。
つまり**k_tを生成したM_k自身も、「k_tとv_tの差が小さくなる方向」に
動かされる**。M_v・m_η・m_αも同様に、自分が生成した値を使って自分自身が
更新される。

重要なのは、この誤差gに外部から与えられる「正解」が存在しないこと。
k_t (M_kの出力) とv_t (M_vの出力) という、**他のメモリが自分で生成した
値どうし**を比較しているだけ。これが"自己参照"(self-referential)の意味
そのもの ― どのメモリも、外部のラベルではなく他のメモリの出力を使って
自分自身を訓練している。「何を記憶するか(M_m)」だけでなく「どう学習するか
(k,v,ηの作り方)」自体が画像を処理している最中に変化していくので、
論文タイトルの"self-modifying"(自己修正)にもなる。

**アナロジー**: 1人の生徒 (M_m) が勉強していて、「今日の問題(key)」を出す
助手(M_k)と「今日の正解(value)」を用意する助手(M_v)がいる。普通のTTTでは
この2人の助手は固定のやり方で問題と正解を用意するだけだが、HOPEでは、
出された問題と正解がズレていたら、生徒だけでなく2人の助手自身も
(自分の問題/正解の出し方を) 修正する。

### ステップ3: この後の説明の読み方

以降で出てくる「M」は上の表のどれかの行列を指す。**HOPE自体 (この5メモリの
仕組み) はVisionHOPE論文が発明したものではなく、Nested Learning論文が
既に提案したもの。** VisionHOPEはこれを画像に安定して適用するための翻案を
行った論文であり、その翻案の内容 (何が新しいか) は次節で説明する。

---

## 通常のCNN/Transformer/SSMバックボーンとの差分 (新規性はどこにあるか)

### 変わらない部分

全体の骨格は見慣れた階層型バックボーン (ConvNeXt/Swin/PVT系) そのもの:
stride-4ステム → 4ステージ (各ステージ間でstride-2ダウンサンプル) →
GAP+線形分類ヘッド。各ブロックも「トークン混合 (mixer) + FFN」の2残差ブランチ
構造で、GRN付きFFN・LayerScale・DropPathなど最近の標準装備も共通。

### 変わる部分: トークン混合の仕組みそのもの

**① (背景, NL/HOPE由来) 固定された混合則 → 自己参照で学習する混合則**

前節で説明した通り、SRNL (5つのメモリが互いをdelta ruleで更新し合う仕組み)
自体はNested Learning (NL/HOPE) 由来であり、VisionHOPE固有の新規性ではない
(VisionHOPEの新規性は②③、特に②)。他のバックボーンと並べると違いが分かりやすい:

```
CNN (ConvNeXt系の mixer): y = conv(x; W)         W は画像ごとに不変
Attention:                y = softmax(QK^T/√d) V  Q,K,V の計算則は不変
SSM (Mamba):               h_t = A h_{t-1} + B x_t  A,B は入力依存だが"更新則"は固定
SRNL:                      M_t = M_{t-1}(αI - η k k^T) - η g k^T
                           → M自体が「その画像を見ている最中に」学習された勾度降下で更新される
```
(注: ここでの「CNN」は、ConvNeXt/Swin/PVTのようなpatchifyステムを持つ階層型
バックボーンでmixerとして使われるconvを指す。古典的なCNN (ResNet/VGG等) は
そもそもトークン列という概念を使わず画像を直接畳み込むため、この比較の対象外)

**② (VisionHOPEの新規性, 中心的な貢献) 発散しないことを数学的に保証した上で自己参照にする**

「メモリが自分自身を更新する」設計は素朴には不安定になりやすい (更新則が
更新対象に依存するため、増幅がフィードバックして暴走しうる)。VisionHOPEは
ソフト注入キャップ (自己参照による注入量を上限で抑える) とスペクトラルクランプ
(保持される遷移のスペクトルノルムを上限で抑える) を組み合わせ、
`||M_t||_F <= ||M_{t-1}||_F` (non-expansion) を各トークンで証明している
(論文 Proposition 1, Corollary 1)。これが無いと、自己参照学習は理論上も
実践上も不安定になりうる。

**③ (VisionHOPEの新規性) 空間への適用: 4方向チャンク分割**

前節のSRNLは元々1次元のシーケンス(テキストのトークン列)向けの再帰であり、
画像の2次元構造にそのまま使うことはできない。VisionHOPEは→←↓↑の4方向に
シーケンス化し、各方向のチャンク境界を行/列の境界に揃える (行優先スキャンは
チャンク長=W、列優先スキャンはチャンク長=H)。これにより「チャンク境界の
メモリからチャンク内の全トークン分のk,v,η,α,yを並列計算し、メモリの更新
だけをトークン順に逐次実行する」という効率的な計算ができる (論文Eq.14-16,
Sec 3.1.3)。

---

## 公開物の実態 (重要)

本サーベイを作る過程で、ローカルにクローンされた公式リポジトリ
(`gh/VisionHOPE`) を実際に調べ、以下を確認した。

| コンポーネント | 公開状況 | 確認した事実 |
|---|---|---|
| `models/{operator,block,hierarchical,layers,common,factory}.py` | **フルコード公開 (素のPyTorch)** | 4方向スキャン・残差ブロック・階層バックボーン・T/S/Bの各サイズ設定は全て通常のPyTorchで実装・公開されている |
| `models/srnl.py` | **CUDA拡張ラッパーのみ公開** | `SRNL` クラスは実際の再帰演算 (SR-DGD + 安定化ステップ制御) を全て `models/srnl_core/csrc/*.cu,*.cuh` のCUDAカーネルに委譲する。Python/PyTorchレベルでの再帰のソースコードは公開されていない (JITコンパイルされる生CUDAのみ) |
| `models/srnl_core/csrc/memory.cuh`, `chunk.cu` | **CUDAソースは公開 (C++/CUDA)** | 活性化関数 (`phi_eta`=スケール付きsoftplus, `phi_alpha`=バイアス付きsigmoid) とステップ制御の数値定数 (`STEP_SCALE=0.025`, `ALPHA_BIAS=2.197224577`, `INJECTION_MARGIN=0.999`, `SPECTRAL_MARGIN=0.999999`) はこの生CUDAソースから実際に読み取れる。本サーベイの `srnl.py` はこれらの値をそのまま採用した純PyTorch再実装である |
| `train/classification.py` | **フルコード公開** | 損失関数は timm 標準の `LabelSmoothingCrossEntropy`/`SoftTargetCrossEntropy` (mixup/cutmix時) で、VisionHOPE独自の新しい損失は無い |

**結論:** アーキテクチャの「組み立て方」(Operator/Block/階層構造) は公開コード通りに
素のPyTorchで再現できる。一方、論文の核心である SRNL の再帰演算そのものは
CUDA拡張としてのみ公開されており、PyTorchレベルの参照実装は存在しない。
本サーベイの `srnl.py` は、公開されている論文の数式とCUDAソースの定数・活性化関数を
直接の根拠として、CUDA拡張をビルドせずCPU/GPUで動く形に独立に再構成したものである
(実装時に実際にNaN勾度のバグを1件発見・修正した。詳細は `srnl.py` のコメント参照)。

---

## アーキテクチャ全体像

```
入力画像 (B, 3, H, W)
        ↓
┌─────────────────────────────────────────────────────────┐
│ Stem: stride-4 畳み込み (PatchEmbedding)                   │
└─────────────────────────────────────────────────────────┘
        ↓
┌─ Stage 1〜4 (各ステージ間に stride-2 Downsample) ──────────┐
│                                                           │
│  VisionHOPEBlock × depths[i]:                             │
│    x = x + CPE(x)                                         │
│    branch = Operator(DW(Proj_in(Norm1(x)))) + LePE        │
│    x = x + γ1 * Proj_out(branch)       <- 残差1 (Operator)│
│    x = x + γ2 * FFN(Norm2(x))          <- 残差2 (FFN)     │
│                                                           │
│  VisionHOPEOperator (論文Fig.2a):                          │
│    4方向シリアライズ(→←↓↑, 行/列チャンク) -> SRNL x4        │
│    -> 空間復元 -> チャンネル方向に方向融合 (λ_r)             │
│                                                           │
│  SRNL (5メモリ連動学習, srnl.py):                           │
│    境界メモリから k,v,η,α,y を並列生成 (チャンク内)           │
│    -> トークン順にSR-DGDで5メモリを逐次更新 (チャンク間で継承) │
└───────────────────────────────────────────────────────────┘
        ↓
Norm -> 1x1 Conv(projection_dim) -> BN -> SiLU -> GAP -> Linear
        ↓
出力 logits (B, num_classes)
```

---

## ファイル構成

### 1. [srnl.py](srnl.py)
**モデルブロック: SRNL (5メモリ自己参照学習) の純PyTorch参照実装**

- `phi_eta()` / `phi_alpha()`: 学習率・保持率メモリの出力を実際のステップ/保持率へ
  写す活性化関数 (公式CUDAソースの定数を使用)
- `stability_matched_step()`: ソフト注入キャップ + スペクトラルクランプ (論文Def.1)
- `srnl_chunkwise_recurrence()`: チャンク内並列生成 + トークン順逐次更新のSR-DGD再帰
- `SRNLReference`: 標準インターフェース `(B,N,dim) -> (B,N,dim)` を持つ `nn.Module`
- `example_usage()`: 実際に forward/backward を実行し、勾度が流れること、
  メモリのフロベニウスノルムが発散しないこと (non-expansion) を数値的に確認する

### 2. [model.py](model.py)
**メインフロー: 4方向Operator・残差Block・階層バックボーン**

- `VisionHOPEOperator`: 4方向スキャン+SRNL+空間復元+チャンネル融合 (論文Eq.22-24)
- `VisionHOPEBlock`: CPE・gated depthwise conv・Operator・LePE・GRN付きFFNの
  2残差ブランチ構造 (論文Eq.25)
- `HierarchicalVisionHOPE`: 4ステージの階層バックボーン + 分類ヘッド
- `MODEL_CONFIGS` / `create_model()`: 公式のTiny/Small/Baseサイズ設定
- `example_usage()`: カスタム極小構成と、公式`visionhope_tiny`設定 (層数のみ削減)
  の両方で実際にforward/backwardを実行する

### 3. [loss.py](loss.py)
**ロス: ラベル平滑化 / Mixup・CutMix用Soft-Target Cross-Entropy**

- `LabelSmoothingCrossEntropy`: timm標準のラベル平滑化CE
- `SoftTargetCrossEntropy`: Mixup/CutMix後のソフトラベル用CE
- `mixup_cutmix()`: 画像とラベルに実際にMixup/CutMixを適用するユーティリティ

### 4. [finetune_example.py](finetune_example.py)
**実際に動く画像分類ファインチューニングスクリプト (Trainer不使用)**

- `ImageFolderDataset`: pandas DataFrame (image_path, label) から画像を読み込む
  Dataset (実際のファイルI/O + torchvision変換)
- `load_model_for_finetuning()`: チェックポイントの部分読み込み (分類ヘッドの
  クラス数違いに対応) とステージ凍結をサポート
- `train()` / `evaluate()`: AdamW + Cosine decay の手動訓練ループ
- `main()`: 実行すると合成画像データセット (実際にディスクへPNG書き出し) で
  全パイプラインを実際に学習させるデモが走る (`python finetune_example.py`)

---

## VisionHOPEの主要イノベーション

以下では、まず背景となる自己参照学習の機構 (0) を説明したうえで、
**VisionHOPE論文が実際に新しく付け加えた部分 (①②)** を分けて示す。

### 0. 背景: Self-Referential DGD (5メモリの連動学習) — ※NL/HOPE由来、VisionHOPE独自の新規性ではない

**概要**: パッチ化された特徴マップに対するトークン混合則 (attention/
ConvNeXt系conv/SSM) は学習済みパラメータに従うのみで、推論中にその規則自体
が画像内容に応じて進化することはないという従来の限界に対し、
Nested Learning (NL/HOPE) が提案した機構。5つのメモリが互いを使って互いを
オンラインの勾度降下 (Delta Gradient Descent) で更新し合う。

```python
# srnl.py srnl_chunkwise_recurrence (抜粋)
v_hat = torch.einsum('bij,bj->bi', M, v_t)        # 自己生成ターゲット (他メモリのvを自分の状態で変換)
g = torch.einsum('bij,bj->bi', M, k_t) - v_hat     # オンライン勾度 (現在の応答との誤差)
M_new = torch.einsum('bij,bjk->bik', M, T_t) - eta_tilde * outer(g, k_t)  # SR-DGD更新
```

**VisionHOPEが変えた点**: 元のHOPEはメモリをMLPとして実装するが、VisionHOPEは
閉形式の更新式を導出できる**線形(行列)メモリ**を採用する (これも厳密には
VisionHOPE独自というより、NL論文内で「MLPメモリの特殊ケース」として既に
示されている式をそのまま使ったもの)。

**実装**: [srnl.py](srnl.py)

---

### ① VisionHOPE論文の新規性 (中心的な貢献): Stability-Matched Step-Size Control

**問題**: NLの無制約な自己参照更新を**そのまま画像バックボーンに適用すると**、
更新則が更新対象自身が生成した量に依存するフィードバック構造のため、発散しうる
(理論上のnon-expansion保証が無い)。これは元のNL/HOPE論文では扱われていない、
VisionHOPEが画像ドメインへの適用で新たに発見・解決した問題。

**解決**:
```python
# srnl.py stability_matched_step (抜粋): ソフト注入キャップ + スペクトラルクランプ
eta_bar = eta_inj * (1 - exp(-eta_raw / eta_inj))   # 注入量を (1-alpha) 以下に抑える
eta_spec = 2 * margin * alpha / ||k||^2              # 保持遷移のスペクトルノルムを alpha 以下に抑える
eta_tilde = min(eta_bar, eta_spec)
```

**効果**: 論文Corollary 1で `||M_t||_F <= ||M_{t-1}||_F` を証明。本サーベイでも
`srnl.py` の `example_usage()` で実測し、メモリノルムが発散しないことを確認した。
この安定化が無いままSRNLをVisionバックボーンに組み込むことはできない
(=この論文の存在理由そのものと言える部分)。

**実装**: [srnl.py](srnl.py)

---

### ② VisionHOPE論文の新規性: 4方向スキャン + 行/列アラインチャンク (2次元への翻案)

**問題**: NL/HOPEのSRNLは元々1次元のシーケンス (テキストのトークン列) 向けの
再帰であり、画像の2次元構造にそのまま使うことはできない。単純に1方向だけ
走らせると方向に偏ったバイアスが生じるうえ、チャンク境界が空間構造と噛み合わないと
並列化効率が落ちる。

**解決**: →←↓↑の4方向それぞれに独立したSRNLインスタンスを走らせ、行優先スキャン
のチャンク長をW、列優先スキャンのチャンク長をHに揃える (論文Eq.23)。

**実装**: [model.py](model.py) の `VisionHOPEOperator`

---

**最終的な効果 (①②+階層バックボーンへの組み込み)**: ImageNet-1K Top-1で
VisionHOPE-T (26.6M) が84.1%、VisionHOPE-B (91.2M) が85.6%。

---

## 処理フロー詳細

### 推論・学習フロー (model.py)

```python
from model import create_model

model = create_model("visionhope_small", num_classes=1000)
x = torch.randn(1, 3, 224, 224)
logits = model(x)  # (1, 1000)
```

### ファインチューニングフロー (finetune_example.py)

```python
import pandas as pd
from finetune_example import load_model_for_finetuning, train

df = pd.DataFrame({"image_path": [...], "label": [...]})
model = load_model_for_finetuning("visionhope_tiny", num_classes=10,
                                   checkpoint_path="weights/visionhope_tiny.pth")
train(model, train_df=df, num_classes=10, output_dir="./out")
```

---

## 学習データフォーマット

`finetune_example.py` が期待する pandas DataFrame のスキーマ:

| カラム | 型 | 説明 |
|---|---|---|
| `image_path` | str | 画像ファイルのパス |
| `label` | int | クラスID (0 ~ num_classes-1) |

論文の実際の学習設定 (Appendix "Training Protocols"): ImageNet-1Kを300エポック、
AdamW、Cosine decay、Mixup(0.8)/CutMix(1.0)、Label smoothing(0.1)、
EMA使用。COCO/ADE20KはImageNet事前学習済みバックボーンを初期値として
それぞれMask R-CNN (1x/3x schedule) とUPerNetで学習する。

---

## FAQ

### Q1: なぜ SRNL の再帰を公式のCUDA拡張のまま使わず再実装したのか?

**A**: 公式 `models/srnl.py` の `SRNL` クラスを実際に調べると、再帰演算本体を
`models/srnl_core/csrc/*.cu,*.cuh` の生CUDAカーネルに完全に委譲しており、
Python/PyTorchレベルでの計算ロジックが一切存在しない (`torch.utils.cpp_extension.load_inline`
でJITコンパイルされる)。CUDA拡張をビルドせずに動作確認できるPyTorch実装が無いと、
GPU環境無しでは論文のアルゴリズムを検証できない。そこで論文の数式 (Eq.1-21) と、
CUDAソースに実際に定義されている活性化関数・定数を直接の根拠として、
独立したPyTorch参照実装を作成した。

### Q2: 実装中にどんな問題に遭遇したか?

**A**: `stability_matched_step()` のスペクトラルクランプ項 `eta_spec = 2*margin*alpha/||k||^2`
で、キーのノルムがゼロに近い場合にゼロ除算を避けるための下限クランプを、公式CUDAの
定数 (`ETA2ALPHA_DENOM_EPS=1e-20`) と同じ値にすると、PyTorchの自動微分がこの除算を
素直に微分した結果、勾度がfloat32の表現範囲 (~3.4e38) を超えてinfになり、
`torch.minimum` の非選択側で `0 * inf = NaN` が伝播するバグを実際に発見した。
公式CUDAカーネルは手で導出した解析的backwardを使うためこの問題を回避できるが、
PyTorchの自動微分に頼る参照実装ではより緩やかな下限 (1e-6) に変更して解決した
(`srnl.py` のコメント参照)。

### Q3: SRNLはAttention/SSMと比べてどう違うのか?

**A**: 計算量はSSM (Mamba等の線形再帰) と同様にトークン数に線形。決定的な違いは
「メモリの更新則」自体が学習済みパラメータではなく、他の4つのメモリが生成する
量 (キー・値・ステップ・保持率) に応じて **オンラインに勾度降下で動的に決まる**
点。SSMのA,B行列は入力依存でも「どう更新するか」という式自体は固定されているが、
SRNLは複数のメモリが互いの出力を使って互いのSR-DGD更新式のパラメータを生成し合う
(論文 Sec 3.1.2 "Self-Referential Nested Learning")。ただしこの「自己参照」という
考え方自体はVisionHOPE論文のオリジナルではなく、Nested Learning (NL) が "HOPE"
として提案した汎用機構である (下記Q4参照)。

### Q4: VisionHOPEの「新規性」は結局どこにあるのか? SRNLを提案したのはこの論文なのか?

**A**: いいえ、違う。SRNL (自己参照ネスト学習) という学習機構自体は、
Nested Learning (NL, NeurIPS 2025) が "HOPE" という名前で既に提案した
**汎用のシーケンス混合方式** (元々は言語モデリング向けにMLPメモリとして実装) で、
VisionHOPE論文のオリジナルではない。論文中にも "In the original HOPE
architecture in NL, these memories are instantiated as MLPs... VisionHOPE
accordingly adopts this linear $L_2$ recurrence" と明記されている。

VisionHOPE論文が実際に新しく付け加えたのは次の3点である:
1. HOPEのMLPメモリを、閉形式の更新式を持つ**線形(行列)メモリ**に変更
2. **安定性保証付きステップサイズ制御** (本論文の中心的な新規性): NLの無制約な
   自己参照更新を画像バックボーンにそのまま適用すると発散することを発見し、
   これを防ぐソフト注入キャップ+スペクトラルクランプを導出・証明した。
   元のNL/HOPE論文にはこの問題への対処が無く、VisionHOPEが画像ドメインへの
   適用で初めて必要になった工学的な詰めである
3. 1次元の再帰を2次元画像に適用するための**4方向・行/列境界揃えチャンク化**

つまりこの論文は「ゼロから新しいトークン混合機構を発明した」わけではなく、
「既存の汎用学習方式 (TTT系列のさらに先にあるNL/HOPE) を、画像ドメインで
安定して動かすための翻案論文」という位置づけが正確である。これは
TTT (Test-Time Training) という汎用方式に対してViT³やVision-TTTが画像向けの
翻案を行った構図と同じである。

### Q5: チャンク内の並列生成フェーズと逐次更新フェーズの境界は?

**A**: 各チャンクの最初に固定されている境界メモリ (直前のチャンク終了時点の状態) 
から、そのチャンク内の **全トークン分の** k,v,η,α,y を一括 (並列) 計算する。
その後、5つのメモリの更新だけをトークン順に逐次実行する。読み出しy_tは境界メモリ
だけに依存するため (論文Eq.16)、チャンク内で再計算する必要がない。

### Q6: 実運用でこのサーベイのコードをそのまま使えるか?

**A**: `model.py` の `VisionHOPEOperator`/`VisionHOPEBlock`/`HierarchicalVisionHOPE`
は公式のアーキテクチャをそのまま再現しているので構造的には使える。`srnl.py` の
`SRNLReference` は数学的に論文と同じ計算を行うが、トークンごとの逐次python loopで
実装されているため、公式のCUDA拡張 (融合カーネル) に比べて大幅に遅い。実運用の
学習・大規模推論では公式リポジトリの `models/srnl.py` (CUDA拡張版) を使うべきで、
本サーベイのコードは「アルゴリズムを読んで理解する・小規模に動作確認する」ための
ものである。

---

## まとめ

VisionHOPE は、Nested Learning (NL) が提案した汎用の自己参照学習機構 "HOPE"
を画像ドメインに翻案した論文である。背景となる機構 (0) と、この論文自体が
新しく付け加えた部分 (①②) を分けて整理すると:

0. **(背景, NL/HOPE由来)** SRNL: 5つのメモリ (内容・キー・値・学習率・保持率) が、
   画像を処理している最中にオンラインのDelta Gradient Descentで互いを更新し合う。
   この機構自体はVisionHOPEのオリジナルではない
1. **(VisionHOPEの新規性①, 中心的な貢献)** Stability-Matched Step-Size Control:
   ソフト注入キャップ+スペクトラルクランプで、NLの無制約な自己参照更新を画像
   バックボーンに適用した際に生じる発散を防ぎ、数学的に保証する
2. **(VisionHOPEの新規性②)** 4方向チャンクスキャン: 元々1次元向けのSRNLを、
   画像の2次元構造に合わせて行/列境界を揃えた4方向の独立したインスタンスとして
   走らせ、チャンネル方向に学習融合する

これらを組み込んだ演算子を階層型バックボーンに実装し、ImageNet-1K/COCO/ADE20Kで
学習・評価した結果が、使い方としては「ResNet/Swinの代わりに使える画像バックボーン」
という完成品である。**この論文の知的な新規性の核心は0ではなく①(と②)であり、
「自己参照学習そのものの発明」ではなく「既存の汎用学習方式を画像ドメインで安定に
動かすための翻案」である点に注意。**

**公開状況**: アーキテクチャの組み立て (Operator/Block/階層構造) は素のPyTorchで
公開。SRNLの再帰演算そのものはCUDA拡張としてのみ公開され、PyTorch参照実装は
存在しない。本サーベイはそのギャップを論文の数式とCUDAソースの定数から
実際に動くコードとして埋めている (実装過程でNaN勾度のバグを1件発見・修正済み)。

---

## 参考文献

- 論文: [VisionHOPE: Visual Backbones as Self-Modifying Learning Systems](https://arxiv.org/abs/2609.33325)
- 公式リポジトリ: [github.com/PSRben/VisionHOPE](https://huggingface.co/PSRben/VisionHOPE)
- 基盤理論: Nested Learning: The Illusion of Deep Learning Architectures (NeurIPS 2025)
  — SRNL/HOPE (自己参照学習機構) そのものの出典。VisionHOPEはこれを画像ドメインに
  翻案した論文であり、SRNL/HOPE自体の提案者ではない (詳細はFAQ Q4)

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。
`model.py` のアーキテクチャ部分は公式実装とほぼ同一ですが、SRNLの再帰計算
(`srnl.py`) は公式のCUDA拡張のソースコードが非公開のため、論文の数式と
公開されているCUDAカーネルソースの定数から独立に再構成した参照実装であり、
実際の実装 (非公開のCUDAカーネル本体) とは異なる場合があります。
