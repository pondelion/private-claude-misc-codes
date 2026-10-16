# AuK Understanding - 簡略化疑似コード集

**AuK: An Open-Source Foundational Model for Speech Generation and Editing**
の理解を目的とした簡略化疑似コード集です。

論文: [arXiv:2609.08936](https://arxiv.org/abs/2609.08936)
公式コード: ローカルクローン `gh/AuK` ([github.com/Tencent-Hunyuan/AuK](https://huggingface.co/tencent/AuK))

## 📋 目次

- [概要](#概要)
- [通常のTTS/音声編集モデルとの差分 (新規性はどこにあるか)](#通常のttsx音声編集モデルとの差分-新規性はどこにあるか)
- [公開物の実態 (重要)](#公開物の実態-重要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [AuKの主要イノベーション](#aukの主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [学習データフォーマット](#学習データフォーマット)
- [FAQ](#faq)

---

## 概要

**AuK は何なのか:** TTS (Text-to-Speech) エンジンや音声編集ツール
(ピッチ/速度変更、ノイズ除去、話者分離など) として使われている既存の各種音声
モデルを、**1つの基盤モデルに統合したもの**である。通常、「読み上げさせたい」
「声の感情を変えたい」「ノイズを消したい」「BGMと声を分離したい」といった
音声タスクはそれぞれ専用のモデル/パイプラインを別々に使う必要がある。AuKは
これらをすべて「指示文 (自然言語) + (必要なら)参照音声」という1つの入力形式に
統一し、単一のモデルで16種類の音声生成・編集タスクを処理する。使い方の感覚は
画像編集における instruction-based image editing モデル (指示文で画像を編集する
モデル) の音声版、と捉えると分かりやすい。

**どんな場合に使えるか:** テキストからの音声合成 (Zero-shot/Instruct TTS)、
既存音声の内容編集・音響編集 (ピッチ/速度/音量)・パラ言語編集 (感情/音色の
変更)、音声強調・話者分離・音楽分離など、「音声に対して何らかの生成/変換/編集を
行いたい」場面全般が対象になる。専用モデルを複数組み合わせるパイプラインの
代わりに、1つのモデルと自然言語の指示だけで済ませたい場合に向いている。

**AuK の特徴:**
- **統一自然言語指示インターフェース**: Zero-shot TTS・Instruct TTS・コンテンツ編集・
  音響編集 (ピッチ/速度/音量)・パラ言語編集 (感情/音色/脱アクセント等)・
  音声強調・話者分離・音楽分離を、全て「指示文 + (任意の)参照音声」という
  単一の形式で扱う
- **MLLM層融合による意味条件**: 固定層の出力だけでなく、MLLM (Qwen2.5-Omni)
  の全層の隠れ状態を学習済み重みで加重平均することで、言語的・音響的・
  クロスモーダルな情報を多層から集約する
- **FLUX-style ハイブリッドDiT**: M個のdual-stream MMDiTブロック (意味条件と
  音響条件のジョイントアテンション) + N個のsingle-stream DiTブロックで
  潜在空間のフロー速度を予測する
- **Rectified Flow + 階層的条件ドロップアウト**: 1.5Bパラメータのバックボーンを
  生成・編集タスク混合で学習し、Classifier-Free Guidanceに対応する

**タスク:** Zero-shot/Instruct TTS、コンテンツ編集、音響編集、パラ言語編集、
音声強調・話者分離・音楽分離 (全16種類の指示駆動タスク)

**モデルサイズ:** 1.5B (バックボーン)。AuK (高品質) と AuK-Flash (4ステップ蒸留,
高速) の2バリアント。

---

## 通常のTTS/音声編集モデルとの差分 (新規性はどこにあるか)

### 変わらない部分

骨格は近年の潜在拡散/フローマッチング音声モデル (F5-TTS, Flux系) と同じ:
VAEで波形を潜在空間に圧縮し、DiT系Transformerでノイズ除去 (フロー速度予測)、
VAEデコーダで波形に戻す。RoPE・QK-RMSNorm・AdaLN変調・SwiGLU FFNといった
構成要素も最近の拡散Transformerの標準装備そのもの。

### 変わる部分: 条件の作り方とタスクの統一の仕方

**① 「どのタスクか」を専用モジュールで分岐せず、自然言語指示1つに統一**

従来の音声編集システムは、TTS用エンコーダ・編集用エンコーダ・強調用モジュール
など、タスクごとに専用の分岐やヘッドを持つことが多い。AuKは「指示テキスト +
(任意の)参照音声」という**1つの入力形式**で16種類のタスクを表現し、
タスクの違いは全てMLLMに渡す自然言語の指示文の違いとして吸収する
(モデル構造自体はタスクに依存しない)。

**② MLLMの「最後の層」ではなく「全ての層」を条件として使う**

```
通常のTTS:     c = MLLM(text).last_hidden_state        <- 最終層だけ使う
AuK:           c = sum_l  w_l * LayerNorm(MLLM(text).hidden_states[l])
                   w_l は softmax(学習済みロジット) で正規化
```
浅い層の表現 (音韻・局所的な音響特徴) と深い層の表現 (意味・話者の意図) を
両方条件として使えるようにする、ELMo的な層融合。この重み自体が学習対象
であり、タスクに応じて「どの深さの情報を重視するか」が学習時に決まる。

**③ 参照音声の有無で条件構造が変わる (dual-stream設計)**

参照音声がある場合: VAEエンコーダの参照潜在 + ノイズ付きターゲット潜在を
連結した「音響ストリーム」と、MLLM融合済みの「意味ストリーム」をdual-stream
MMDiTでジョイントアテンションさせる。参照音声が無い場合 (Instruct TTS等)は
音響ストリームにターゲット潜在だけが入る。同じモデルが両方のケースを
シームレスに扱う。

### 学習・推論の仕組み (Rectified Flowの速度場回帰)

アーキテクチャ上、生成も編集も「ノイズから目的の潜在への線形パス上の速度を
予測するタスク」として統一されている。

- 学習時: クリーン潜在 z1 (VAEエンコーダの出力) とガウス雑音 z0 から、
  ランダムな時刻 t で線形補間した z_t = (1-t)z0 + t*z1 をモデルに入力し、
  一定の目標速度 v_t = z1 - z0 (tに依存しない!) を回帰させる
  (`loss.py::build_flow_matching_batch`)
- 条件ドロップアウトで CFG (Classifier-Free Guidance) に対応: 確率0.3で
  音響条件をドロップ、さらに確率0.2で音響・テキスト両方を強制的にドロップ
  (unconditional) する階層的なルール (`loss.py::sample_condition_dropout`)
- 推論時: ランダムな雑音から始め、学習された速度場をODEソルバー (Euler法等)
  で積分して潜在を得て、VAEデコーダで波形に戻す

---

## 公開物の実態 (重要)

本サーベイを作る過程で、ローカルにクローンされた公式リポジトリ (`gh/AuK`) を
実際に調べ、以下を確認した。

| コンポーネント | 公開状況 | 確認した事実 |
|---|---|---|
| `src/auk/model/{modules,flux2_edit,cfm_edit}.py` | **フルコード公開 (素のPyTorch)** | AdaLN・SwiGLU FFN・MMDiT/DiTブロック・フローマッチングの学習ループは全て通常のPyTorchで実装・公開されている |
| `src/auk/model/vae/bigvgan_flow_vae.py` | **フルコード公開** | BigVGANベースのVAE (エンコーダ+正規化フロー+デコーダ) もコード自体は公開。ただし学習済み重みはHugging Face上の別リポジトリ (`tencent/AuK`) のチェックポイントとして配布されており、本サーベイでは重みのダウンロードは行っていない |
| MLLM (Qwen2.5-Omni-3B) | **公開済み外部モデルへの依存** | AuK自体のリポジトリにはMLLM本体のコードは無く、`huggingface.co/Qwen/Qwen2.5-Omni-3B` を別途ダウンロードして使う設計。学習中はfreeze (論文 Sec Unified Pre-Training) |
| `src/auk/train/train.py` | **フルコード公開** | 本物の学習ループ (DeepSpeed ZeRO-2, bf16, EMA等) が公開されている |

**結論:** アーキテクチャ・学習ループ・VAEは全て素のPyTorchコードとして公開されて
いる (VisionHOPEのようなCUDA拡張ブラックボックスは無い)。本サーベイが
「公式のフルサイズモデルをそのまま動かさない」理由は実装の非公開ではなく、
**モデルサイズ** (バックボーン1.5B + Qwen2.5-Omni-3B + VAE) にある。
そのため本サーベイの `model.py`/`loss.py` は公式のアーキテクチャ・損失関数を
ほぼそのまま移植した本物の実装だが、`finetune_example.py` 内の
VAEエンコーダ・テキストエンコーダは、サイズの都合で用意した凍結スタンドイン
である (学習対象であるTransformerバックボーンと層融合パラメータ自体は
本物の実装を使用し、実際に学習が進むことを確認済み)。

---

## アーキテクチャ全体像

```
指示テキスト + (任意の)参照音声
        ↓                      ↓
┌─────────────────┐   ┌─────────────────────┐
│ MLLM (Qwen2.5-   │   │ VAE Encoder          │
│ Omni, frozen)    │   │ (24kHz -> 50Hz, 64d) │
│ 全層の隠れ状態    │   │ 参照潜在 z_ref        │
└─────────────────┘   └─────────────────────┘
        ↓                      ↓
┌─────────────────┐            │
│ SemanticLayerFusion│          │
│ c_sem = Σ softmax(w_l)*LN(h_l)│
└─────────────────┘            │
        ↓                      ↓
   意味ストリーム s(0)      音響ストリーム a(0) = [z_ref ; z_t(ノイズ付きターゲット)]
        └──────────┬───────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ M個の dual-stream MMDiTBlock                   │
│  (意味/音響それぞれ独自QKVO, ジョイントアテンション) │
└─────────────────────────────────────────────┘
                    ↓ 連結
┌─────────────────────────────────────────────┐
│ N個の single-stream DiTBlock (自己注意)          │
└─────────────────────────────────────────────┘
                    ↓
          予測フロー速度 v̂_t (ターゲット位置のみ抽出)
                    ↓ (推論時はODE積分を繰り返す)
              VAE Decoder -> 出力波形
```

---

## ファイル構成

### 1. [model.py](model.py)
**モデルブロック/メインフロー: MMDiT/DiTハイブリッドバックボーン + MLLM層融合**

- `build_rope()` / `apply_rope()`: 標準RoPE (公式は `x_transformers` に依存するが
  本ファイルは自前実装)
- `AdaLayerNorm` / `AdaLayerNormFinal`: 時間条件付き6-way変調 (shift/scale/gate)
- `SwiGLUFeedForward`: ゲート付きFFN
- `MultiHeadAttentionCore`: 単一ストリーム/ジョイントアテンション両対応のコア
  (QK-RMSNorm + RoPE + SDPA)
- `MMDiTBlock` / `DiTBlock`: dual-stream / single-streamの残差ブロック
- `SemanticLayerFusion`: MLLM多層隠れ状態のsoftmax重み付き融合 (論文Eq.2)
- `Flux2EditBackbone`: 全体を統合したバックボーン (ゼロ初期化含む)
- `example_usage()`: 実際にforward/backwardを実行し、zero-init設計の結果
  勾度が最初は出力層に集中すること、数ステップ学習すると手前の層にも
  勾度が伝わり始めることを実測で確認する

### 2. [loss.py](loss.py)
**ロス: Rectified Flow + 階層的条件ドロップアウト**

- `sample_flow_time()`: 一様 / logit-normal のフロー時刻サンプリング
- `sample_condition_dropout()`: 階層的な条件ドロップアウト (CFG対応)
- `build_flow_matching_batch()`: z1 (クリーン潜在) から (z_t, 目標速度, t) を構築
- `masked_flow_matching_loss()`: パディング除外マスク付きMSE
- `euler_ode_sample()`: 推論時のODE積分 (公式はtorchdiffeqに依存、本ファイルは
  依存を増やさない固定ステップEuler法を自前実装)

### 3. [finetune_example.py](finetune_example.py)
**実際に動くFlow-Matchingファインチューニングスクリプト (Trainer不使用)**

- `ToyVAEEncoder` / `ToyTextEncoder`: 凍結スタンドイン (実運用ではBigVGAN-flow
  VAE / Qwen2.5-Omni に差し替え、学習対象は変わらない)
- `AuKDataset`: pandas DataFrame (instruction, target_audio_path,
  source_audio_path) から実際に音声ファイルを読み込むDataset
- `train()`: AdamW の手動訓練ループ。バックボーン+層融合のみを最適化
  (VAE/テキストエンコーダは凍結, 論文の設計通り)
- `main()`: 実行すると合成音声 (実際にディスクへWAV書き出し) で全パイプラインを
  実際に学習させるデモが走る (`python finetune_example.py`)

---

## AuKの主要イノベーション

### 1. **MLLM層融合による意味条件 (Semantic Layer Fusion)**
**問題**: 固定層 (多くは最終層) の出力だけでは、タスクによって必要な抽象度
(音韻的 vs 意味的) が異なる場合に最適でない。

**解決**:
```python
# model.py SemanticLayerFusion (抜粋)
stacked = torch.stack([F.layer_norm(h, [d_llm]) for h in hidden_states], dim=0)
weights = F.softmax(self.layer_weights, dim=0)
c_sem = (stacked * weights[:, None, None, None]).sum(dim=0) * self.layer_scale
```

**実装**: [model.py](model.py) の `SemanticLayerFusion`

---

### 2. **Dual-Stream MMDiT + Single-Stream DiT のFLUX-styleハイブリッド**
**問題**: 単純な単一ストリームでは、意味条件と音響条件を同じ重みで扱うため、
ストリームごとの統計の違いに適応しづらい。

**解決**: 最初のM層は意味/音響それぞれ独立のQKVO射影を持ちつつジョイント
アテンションで情報交換し (dual-stream)、残りのN層は連結した単一シーケンスを
通常の自己注意で処理する (single-stream)。

**実装**: [model.py](model.py) の `MMDiTBlock` / `DiTBlock` / `Flux2EditBackbone`

---

### 3. **Rectified Flow + 階層的条件ドロップアウト**
**問題**: 1つのモデルで「完全条件付き」「テキストのみ」「無条件」の3通りの
生成を可能にし、CFGで生成品質を上げたい。

**解決**:
```python
# loss.py sample_condition_dropout (抜粋)
drop_audio_cond = random() < 0.3
if random() < 0.2:
    return True, True  # 音響・テキスト両方ドロップ (unconditional)
return drop_audio_cond, False
```

**実装**: [loss.py](loss.py) の `sample_condition_dropout()`

---

## 処理フロー詳細

### 学習フロー (model.py + loss.py)

```python
from model import Flux2EditBackbone, SemanticLayerFusion
from loss import build_flow_matching_batch, masked_flow_matching_loss

fusion = SemanticLayerFusion(num_layers=L)
c_sem = fusion(hidden_states)                     # MLLM多層出力(凍結)から意味条件を融合

batch = build_flow_matching_batch(z1)              # z1: VAEエンコーダのクリーン潜在
v_pred = backbone(batch.z_t, c_sem, batch.time, ref=ref_latent)
loss = masked_flow_matching_loss(v_pred, batch.target_velocity)
loss.backward()
```

### ファインチューニングフロー (finetune_example.py)

```python
import pandas as pd
from finetune_example import load_model_for_finetuning, train

df = pd.DataFrame({
    "instruction": ["Say hello in a cheerful voice."],
    "target_audio_path": ["audio/target1.wav"],
    "source_audio_path": [None],
})
backbone, fusion, text_encoder, vae_encoder = load_model_for_finetuning()
train(backbone, fusion, text_encoder, vae_encoder, train_df=df, output_dir="./out")
```

---

## 学習データフォーマット

`finetune_example.py` が期待する pandas DataFrame のスキーマ:

| カラム | 型 | 説明 |
|---|---|---|
| `instruction` | str | タスク指示テキスト |
| `target_audio_path` | str | ターゲット音声ファイルのパス |
| `source_audio_path` | str or None | 参照/編集元音声のパス (無ければ指示のみタスク) |
| `duration` | float (任意) | 目安秒数 (公式はdynamic length-bucketed batchingに使用) |

公式の実際のJSONL形式 (README "Fine-tuning"より): 1行が `duration` と
`messages` (user: instruction+任意のaudio_url, assistant: audio_url) を持つ
会話形式。本サーベイはこれを1行1サンプルのフラットなDataFrameに単純化した。

論文の実際の学習設定 (Sec Unified Pre-Training): 30層 (10 MMDiT + 20 DiT),
隠れ次元1536, 24ヘッド×64次元, SwiGLU中間次元3072, 約1.5Bパラメータ。
256GPU, DeepSpeed ZeRO-2, bf16, AdamW (β=(0.9,0.95)), peak lr 1e-4
(2000stepでwarmup後constant), EMA decay 0.9999。

---

## FAQ

### Q1: なぜ公式のモデルをそのまま使わず再実装したのか?

**A**: AuKのアーキテクチャ自体は全て素のPyTorchで公開されており
(VisionHOPEのようなCUDA拡張の壁は無い)、再実装の技術的な必要は無い。
再実装した理由は単純に **モデルサイズ** で、バックボーン1.5B+MLLM
(Qwen2.5-Omni-3B)+VAEを合わせて数GBの重みをダウンロード・ロードする必要があり、
本サーベイの「アルゴリズムを読んで理解する」という目的には、構造を保った
小規模な実装で十分と判断した。`model.py`/`loss.py` のロジック自体は公式の
`modules.py`/`flux2_edit.py`/`cfm_edit.py` とほぼ同一である。

### Q2: `finetune_example.py` のVAE/テキストエンコーダはどの程度「本物」か?

**A**: `ToyVAEEncoder`/`ToyTextEncoder` 自体はサイズを縮小したスタンドインで、
実際の重みは持たない。しかし論文の設計通り、これらは学習中 **凍結** される
コンポーネントであり、実際に学習されるのは `Flux2EditBackbone` と
`SemanticLayerFusion` (層融合の重み) であり、こちらは公式と同一の本物の実装
である。実運用ではスタンドインを `BigVGANFlowVAE`/`Qwen2.5-Omni` に差し替える
だけで、学習ループ自体は変更不要なように設計している。

### Q3: zero-init設計のせいで学習初期に勾度が出力層にしか流れないのは問題ないのか?

**A**: 問題ない。これはDiT系モデルで広く使われる意図的な設計 (zero-init trick)
で、学習開始時にモデルの出力を正確に0にし (恒等写像からスタート)、学習が
進むにつれてAdaLNのゲートが0から離れ、徐々に手前の層にも勾度が伝わるように
なる。本サーベイの `model.py::example_usage()` で、1回目のforward/backwardでは
`proj_out` の重みだけが非ゼロ勾度を受け取り、5ステップ最適化した後には
手前のMMDiTブロックのAdaLN層にも勾度が伝わり始めることを実測で確認した。

### Q4: 推論時のCFG (cfg_strength) はどう使われるのか?

**A**: 公式 `cfm_edit.py::sample` では、条件付き (c_cond) と無条件 (c_uncond,
テキスト・音響両方0埋め) の2つのバッチを連結してバックボーンに通し、
`v_cond + (v_cond - v_uncond) * cfg_strength` で外挿する。本サーベイの
`loss.py::euler_ode_sample` は単一の速度場関数を仮定した単純化版であり、
CFG外挿のロジック自体は実装していない (学習時の条件ドロップアウトが
主題のため)。

### Q5: 実運用でこのサーベイのコードをそのまま使えるか?

**A**: `model.py` の `Flux2EditBackbone`/`SemanticLayerFusion` は構造的に公式と
同一なので、本物のMLLM出力・VAE潜在を渡せば動く。`loss.py` の関数群も
PyTorch標準APIのみに依存しており、そのまま使える。`finetune_example.py` の
`ToyVAEEncoder`/`ToyTextEncoder` は、実運用では公式の `BigVGANFlowVAE`
(`src/auk/model/vae/bigvgan_flow_vae.py`, 公開されている) と
`transformers.Qwen2_5OmniModel` に差し替える前提のスタンドインである。

---

## まとめ

AuK は以下の要素で「指示駆動の統一音声生成・編集」を実現する:

1. **MLLM層融合**: 固定層でなく全層の隠れ状態を学習済み重みで融合し、
   タスクに応じて必要な抽象度の意味条件を自動的に選び取る
2. **Dual-Stream + Single-Stream ハイブリッドDiT**: 意味/音響ストリームを
   独立に扱うジョイントアテンション層と、統合して扱う自己注意層を組み合わせる
3. **Rectified Flow + 階層的条件ドロップアウト**: 生成・編集を同じ速度場回帰
   タスクとして統一し、CFGに対応する

**公開状況**: アーキテクチャ・学習ループ・VAEコードは全て素のPyTorchで公開。
MLLM (Qwen2.5-Omni) は公開された別モデルへの依存。本サーベイはモデルサイズの
都合でMLLM/VAEを凍結スタンドインに置き換えつつ、学習対象であるバックボーンと
層融合は本物の実装で再現し、実際に学習が進むことを確認している。

---

## 参考文献

- 論文: [AuK Technical Report: An Open-Source Foundational Model for Speech Generation and Editing](https://arxiv.org/abs/2609.08936)
- 公式リポジトリ: [github.com/Tencent-Hunyuan/AuK](https://huggingface.co/tencent/AuK)
- 関連: Qwen2.5-Omni (MLLM semantic encoder), FLUX (dual/single-stream DiT設計),
  F5-TTS (RoPE+ConvPositionEmbedding設計), BigVGAN (VAEデコーダ), DAC (VAE損失)

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。
`model.py`/`loss.py` は公式実装とほぼ同一のロジックですが、`finetune_example.py`
内のVAE/テキストエンコーダはモデルサイズの都合による凍結スタンドインであり、
実際の実装 (Qwen2.5-Omni, BigVGAN-flow VAE) とは異なります。
