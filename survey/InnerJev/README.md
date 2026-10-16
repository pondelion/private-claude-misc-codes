# InnerJev Understanding - 簡略化疑似コード集

**General Decision Models: Benchmarking and Insights Beyond Jev**
の理解を目的とした簡略化疑似コード集です。

論文: `paper/main.tex` (ローカルクローン)
公式コード: ローカルクローン `gh/InnerJev` ([jylin001206/InnerJev-4B-Full](https://huggingface.co/jylin001206/InnerJev-4B-Full) 等)

## 📋 目次

- [概要](#概要)
- [決定モデルとLLMの違い、およびInnerJevの新規性](#決定モデルとllmの違いおよびinnerjevの新規性)
- [公開物の実態 (重要)](#公開物の実態-重要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [InnerJevの主要イノベーション](#innerjevの主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [学習データフォーマット](#学習データフォーマット)
- [FAQ](#faq)

---

## 概要

**InnerJev は何なのか (正確な立ち位置):** 「LLMの次トークン分布を、選択肢
ラベルだけに制限して読み出せば、生成無しに決定(多肢選択/Yes-No/スコア)が
得られる」という読み出し技術 (論文内の呼称: **First-Token Decision
Readout**) 自体は、実はInnerJev論文が発明したものではない。論文は
「学習済みのQwen3.8-27Bに、**何の追加学習もせず**この読み出しを適用するだけで、
既にJev (比較対象の非公開決定モデルAPI) に近い精度が出る」と明記している
("Even without training, this readout of Qwen3.8-27B comes within a few
points of Jev")。つまりこの読み出し自体は**LLMが元々持っている性質**であり、
どのLLMにもゼロショットで使える一般的な技術である。

**InnerJevが実際に新しく付け加えたのは、この読み出しを訓練で改善する手順
(Reasoning-to-Readout Self-Distillation) である**: 同じモデルにthinkingを
有効にして2回推論させ、その回答分布(思考の末に到達した、より正確な決定)を
教師として、thinkingを使わない1回の読み出しがそれに近づくよう訓練する。
外部の教師モデルも正解ラベルも使わない自己蒸留である。この訓練を
Qwen3.5-4B/Qwen3.8-27Bに適用して得られたモデルが **InnerJev-4B/InnerJev-27B**
であり、「InnerJev」という名前が指すのは読み出し技術そのものではなく、
**この自己蒸留訓練の手順、およびそれによって生まれた具体的なモデル**である。

**どんな場合に使えるか:** エージェントの行動選択・コンテンツの安全性判定・
感情/意図分類・満足度スコアリングなど、「限られた選択肢の中から1つを選ぶ、
あるいは順序尺度で評価する」という決定を大量に・低レイテンシで繰り返し行う
必要がある場面が対象になる。本家のJev (ホスト型APIのみで提供される非公開モデル)
のような専用の決定モデルAPIを使う代わりに、公開済みのLLM (Qwen3.5/3.8) を
自分の環境でそのまま高速な決定モデルとして動かしたい場合に向いている。

**キーコントリビューション (InnerJev論文が実際に付け加えた部分のみ):**
- **Reasoning-to-Readout Self-Distillation (中心的な新規性)**: 同じモデルに
  thinkingを有効にして2回推論させ、その回答分布 (思考後の到達点) を教師として、
  thinkingを使わない1回のforward passの読み出しに蒸留する (自己蒸留、外部の
  教師や正解ラベルは学習目的関数に使わない)
- **タイプ別複合損失**: choice/noul/score という3種類の決定フォーマットに対して、
  重み付きCross-Entropy + score専用の順序回帰項 + 元モデルへのKL正則化を
  組み合わせた単一の損失関数で学習する
- **LoRA/Full-rankの両対応**: 同じ損失・同じモジュール (言語層のattention/MLP
  projection) をLoRAまたはfull-rankで更新でき、埋め込み層・出力層・正規化層は
  凍結したままなので、学習後も読み出し自体の推論コストは変わらない

(背景として使っている「First-Token Decision Readout」(次トークン分布を
選択肢ラベルに制限して読み出す) 自体はInnerJev論文のオリジナルではなく、
学習済みLLMが元々持っている性質であり、無学習でも一定の精度が出る。
詳細はFAQ Q2参照)

**タスク:** choice (多肢選択), noul (2値判定/Yes確率), score (順序尺度での
期待値) の3種類の決定フォーマットを、1回のforward passで解く

**モデルサイズ:** Qwen3.5-4B / Qwen3.8-27B をベースにした InnerJev-4B / InnerJev-27B
(各LoRA版・Full-rank版の2バリアント)

---

## 決定モデルとLLMの違い、およびInnerJevの新規性

この論文は、まず「決定モデル (decision model, Jevが代表例)」というカテゴリを
「生成LLM」と対比して定義し (背景)、その上で「InnerJevという決定モデルの作り方」
を「Jevや他の決定モデル」と対比して提案する (新規性)、という2段構成になっている。
この2つの対比軸を混同しないよう、分けて説明する。

### 背景: 「決定モデル」というカテゴリ自体の、通常の生成LLMとの違い (InnerJev独自の新規性ではない)

論文は「decision model」を、*自然言語の文脈を、タスク固有の再学習無しに、
構造化された決定へ直接写す能力を持つモデル*と定義し、Jev (ホスト型APIのみで
提供される非公開モデル) をその代表例とする。この「直接写す」とは具体的には:

```
通常の生成LLM: プロンプト -> 複数トークン生成 -> 生成テキストをパースして決定を得る
決定モデル:    プロンプト + "The correct answer is (" -> 1回のforward pass
               -> 次トークン分布を選択肢ラベルだけに制限 -> それ自体が決定
```

デコードが不要なため、推論コストはバックボーンと入力長だけで決まり、生成
ステップ数に依存しない。論文はこれを「First-Token Decision Readout」と呼ぶが、
**これは決定モデルというカテゴリの定義的な特徴であり、InnerJev独自の新規性
ではない**。実際、論文は「学習済みLLM (Qwen3.8-27B) にこの読み出しを**何も
追加学習せず**適用するだけで、既にJevに近い精度が出る」ことを確認している
(`model.py::Readout`; 詳細はFAQ Q2)。つまり「任意の既存LLMは、追加学習無しで
既に(そこそこの)決定モデルである」というのがこの背景部分の主張であり、
InnerJevの新規性はここではなく次節にある。

### InnerJevの新規性: 「他の決定モデル (Jev・Intern-Decision-4B等)」との違い

InnerJevが具体的に新しく提案するのは、上記の「追加学習無しの読み出し」を
**自己蒸留で訓練して改善する手順**である。Jev自体は非公開APIで学習レシピも
不明、また論文は他にもJevを再現しようとした公開決定モデル (Intern-Decision-4B
等) の存在に言及しているが、InnerJevはそれらとは異なる具体的な手法として
以下を提案する:

**① (中心的な新規性) 「自分自身の思考後の答え」を教師にする自己蒸留**

```
通常の蒸留: 強い教師モデル (別モデル) の出力 -> 弱いモデルを学習
InnerJev:  同じモデルがthinkingありで2回推論した回答分布の平均 (教師)
           -> 同じモデルのthinkingなし・最初の答えトークン (生徒) を学習
```

教師と生徒は同じモデル・同じ読み出し・同じラベル空間であり、違いは
「推論チェーンを挟むかどうか」だけ。正解ラベルは学習目的関数に使われず
(評価とhard-Noul判定にのみ使用)、thinkingが答えを改善する分だけ生徒が
改善する。「推論の計算過程」ではなく「推論が到達した決定分布」だけを
1トークンに蒸留する点が、通常のCoT蒸留(思考過程自体を模倣させる手法)
との違いになる。

**② 回答フォーマットに応じて損失項を使い分ける**

```python
# loss.py loss_terms (抜粋)
ce = weight * -(teacher * logs).sum(-1)                    # 全フォーマット共通
ordinal = weight * 0.5 * is_score * expectation_diff**2 / (K-1)**2  # scoreのみ
kl = kappa * KL(baseline || p)                               # kappaはhard-Noulで0.2, それ以外0.05
```

choice/noul/scoreという質的に異なる3種類の決定を単一のヘッド・単一の損失で
扱うために、scoreには「期待値のずれ」を追加で罰する順序回帰項を、
全体には元モデルからの逸脱を抑えるKL項を加える。

**この新規性による、他の決定モデルとの実際の比較結果 (論文報告値)**: この
自己蒸留をQwen3.8-27Bに適用したInnerJev-27Bは、JEVal上でJev自体をわずかに
上回る精度を、Jevより大幅に低いレイテンシ(典型的な入力で約110ms)で達成する。
より小さいInnerJev-4Bも、同程度のレイテンシ帯で比較した場合、他の公開決定
モデル(Intern-Decision-4B)を上回る精度を報告している。

### 変わらない部分 (アーキテクチャ)

ベースはごく普通の事前学習済みcausal LM (Qwen系)。アーキテクチャに新しい層や
モジュールは一切追加されない。チャットテンプレートも標準のものを使う。

### 学習・推論の仕組み

- 推論時: チャットテンプレート (thinking無効) + `The correct answer is (` で終わる
  プロンプトを1回forwardし、最後の有効トークン位置の隠れ状態を255個のラベル
  トークンの埋め込みに内積して確率を得る (`model.py::Readout.forward`)
- 学習時: 同じモデルでthinkingを有効にして2回サンプリングした回答分布の平均を
  教師とし、thinkingを無効にした1回の読み出しがそれに一致するように
  複合損失で学習する (`loss.py::loss_terms`)。LoRA (r=64, alpha=128) または
  言語層のattention/MLP projectionのfull-rank更新のいずれかで、埋め込み・
  出力層・正規化層は凍結する

---

## 公開物の実態 (重要)

本サーベイを作る過程で、ローカルにクローンされた公式リポジトリ (`gh/InnerJev`)
を実際に調べ、以下を確認した。

| コンポーネント | 公開状況 | 確認した事実 |
|---|---|---|
| `innerjev/{learning,prompts,data,scheduling}.py` | **フルコード公開 (素のPyTorch)** | `Readout`クラス・`loss_terms`・プロンプト構築・バッチ計画は全て通常のPyTorchで実装・公開されている (本サーベイ内で全部合わせて669行) |
| `innerjev/train.py` | **フルコード公開** | DDP/FSDPによる実際の学習ループ (optimizer step・勾度クリップ・チェックポイント選択まで) が公開されている |
| `innerjev/infer.py` | **フルコード公開** | BF16/NF4/INT8での推論エンジン (`DecisionEngine`) が公開されている |
| ベースモデル (Qwen3.5-4B/Qwen3.8-27B) | **公開済み外部モデルへの依存** | InnerJev自体のリポジトリにはベースモデルのコードは無く、Hugging Face上の別リポジトリ (`jylin001206/InnerJev-4B-Full`等) からダウンロードする設計 |
| 学習データ (29,326/39,279問) | **公開データセット** | `jylin001206/InnerJev-4B-Training-Data`等としてHugging Face Datasetsで公開 |

**結論:** InnerJevは、本サーベイで調査した5つの研究の中で最もコードが
シンプルかつ完全に公開されている (VisionHOPEのようなCUDA拡張も、AuKのような
巨大なマルチモーダル条件付けパイプラインも無い)。実際、本サーベイでは
**実際にダウンロード済みの公式チェックポイント `jylin001206/InnerJev-4B-Full`
(Qwen3.5-4B, 約10GB BF16) をロードし、`model.py`の`Readout`/`option_labels`/
`build_prompt`をそのまま使って本物の決定を読み出すことができた**
(`hf_inference_example.py`)。そのため `model.py`/`loss.py` は公式実装をほぼ
そのまま移植した本物の実装であり、`finetune_example.py` だけがモデルサイズの
都合でバックボーンを小さなスタンドイン (`TinyCausalLM`) に差し替えている
(損失関数・Readout・LoRA構成のロジック自体は本物と同一)。

---

## アーキテクチャ全体像

```
入力: state + question + options (A,B,C,...のラベル付き)
        ↓
チャットテンプレート適用 (thinking無効) + "The correct answer is (" で終端
        ↓
┌─────────────────────────────────────────┐
│  既存の事前学習済み causal LM (無改造)       │
│  (Qwen3.5-4B / Qwen3.8-27B など)            │
└─────────────────────────────────────────┘
        ↓ 最後の有効トークン位置の隠れ状態 h
┌─────────────────────────────────────────┐
│  Readout: 255個の選択肢ラベルの出力埋め込み  │
│  だけに内積  logits = W_label^T h          │
└─────────────────────────────────────────┘
        ↓ softmax (有効な選択肢数だけに制限)
   choice: 最尤ラベル / noul: P(yes) / score: 期待レベル

学習時 (self-distillation):
  同モデル+thinking (2回サンプル) --read out--> 教師分布 q̄ (平均)
  同モデル (thinkingなし)          --read out--> 生徒分布 p_θ  ←損失で学習
  元モデル (凍結)                  --read out--> baseline p_0  (KL正則化の基準)
```

---

## ファイル構成

### 1. [model.py](model.py)
**モデルブロック/メインフロー: プロンプト構築 + Readout射影**

- `text()` / `option_labels()` / `build_body()` / `build_prompt()`: 公式
  `prompts.py` とほぼ同一のプロンプト構築ロジック (状態・質問・ラベル付き
  選択肢を並べ、`The correct answer is (` で終端する)
- `Readout`: 最後の有効トークン位置の隠れ状態から、255個の選択肢ラベル
  トークンだけのロジットを読み出すモジュール (公式 `learning.py::Readout`
  と同一ロジック)。LoRA時は凍結スナップショットの出力埋め込み、full-rank時は
  現在の出力埋め込みを使う
- `TinyTransformerStack` / `TinyCausalLM`: デモ用の極小causal LM (実運用では
  Qwen3.5/Qwen3.8に差し替える)
- `example_usage()`: 極小LLMで実際にforward/backwardを実行し、Readoutの
  出力shape・勾度が正しく伝わることを確認する

### 2. [loss.py](loss.py)
**ロス: 自己蒸留複合損失 (重み付きCE + 順序回帰項 + KL)**

- `loss_terms()`: 公式 `learning.py::loss_terms` と同一の複合損失。
  `(total, ce, ordinal, kl)` の4つのper-sample値を返す
- `example_usage()`: choice/noul/scoreの3タイプそれぞれの損失内訳を確認し、
  hard-Noul (`kappa=0.2`) とsoft-Noul (`kappa=0.05`) でKL項が正しく
  切り替わる (比率ちょうど4.0) ことを実測で確認する

### 3. [finetune_example.py](finetune_example.py)
**実際に動くLoRA自己蒸留ファインチューニングスクリプト (Trainer不使用)**

- `ToyTokenizer`: デモ用の極小トークナイザ (実運用では
  `AutoTokenizer.from_pretrained(Qwen3.5-...)` に差し替える)
- `DecisionFrameDataset`: pandas DataFrame (id, type, state, question,
  options, train_target, p0, gold_target) を1行=1サンプルとして返すDataset
- `build_lora_model()`: `model.py::TinyCausalLM` に `peft.LoraConfig`
  (r=64, alpha=128, 公式と同じ構成) を適用
- `train()`: AdamWの手動訓練ループ。`loss_terms()`で損失計算し、LoRAの
  trainableパラメータだけを更新する
- `example_usage()`: 合成データ (24問, choice/noul/score均等) を生成して
  実際に2エポック学習させ、学習後の予測確率と教師分布を比較表示する

### 4. [hf_inference_example.py](hf_inference_example.py)
**ローカルにダウンロード済みの本物の公式チェックポイントで実際に推論する例**

- `load_real_model()`: `jylin001206/InnerJev-4B-Full` (Qwen3.5-4Bベース,
  full-rank学習済み) をBF16でロードする (公式 `infer.py::DecisionEngine`
  と同一の手順)
- `example_usage()`: choice/noul/score各1問を実際にforwardし、本物の
  モデルが出力する確率・P(yes)・期待スコアを表示する

---

## InnerJevの主要イノベーション

以下では、まず背景となる読み出し機構 (0) を説明したうえで、**InnerJev論文が
実際に新しく付け加えた部分 (①②)** を分けて示す。

### 0. 背景: First-Token Decision Readout — ※InnerJev独自の新規性ではない
**概要**: LLMに決定を出させるには通常、答えを生成してテキストをパースする
必要があり、デコードのコストと曖昧なパースのリスクを伴う。この読み出しは、
生成を一切行わず、既存の出力層の該当255行を読み出すだけで、choice/noul/score
すべての決定フォーマットを1回のforward passで得る。

```python
# model.py Readout.forward (抜粋)
slot = hidden[torch.arange(len(hidden)), last_valid]        # 最後の有効トークン位置
weight = base.get_output_embeddings().weight.index_select(0, self.label_ids)
return F.linear(slot, weight).float()                        # (B, K=255)
```

**重要**: この読み出し自体は、学習済みLLMに**何も追加学習せず**適用するだけで
機能する(論文: "Even without training, this readout of Qwen3.8-27B comes
within a few points of Jev")。InnerJev論文のオリジナルの貢献ではなく、
LLMが元から持っている性質を取り出す方法にすぎない。

**実装**: [model.py](model.py) の `Readout`

---

### ① (InnerJevの新規性, 中心的な貢献) Reasoning-to-Readout Self-Distillation
**問題**: thinkingを有効にすると決定の質は大きく上がるが、推論チェーンの
生成コストがかかる。外部の教師モデルや正解ラベルを使わずにこの改善を
1回のforward passに内在化したい。

**解決**: 同じモデルにthinkingありで2回サンプリングさせ、その回答分布の
平均を教師として、thinkingなしの最初の答えトークンをその教師に近づける。
正解ラベルは学習目的関数には登場しない (自己蒸留)。これにより上記0の
「無学習でもある程度使える読み出し」が、「thinkingした場合と同等の精度を
1回のforward passで出せる読み出し」に改善される。

**実装**: 学習データの `train_target` (教師分布) が本サーベイの
`finetune_example.py`/`loss.py` が受け取る入力であり、教師分布そのものの
生成 (2回のthinkingサンプリング+平均) は公式リポジトリの外側 (データ準備段階)
で行われる

---

### ② (InnerJevの新規性) タイプ別複合損失 (順序回帰 + KL正則化)
**問題**: choice (カテゴリカル)・noul (2値)・score (順序尺度) という
性質の異なる3種類の決定を単一の損失で扱うと、score特有の「順序」情報が
通常のCross-Entropyだけでは活用されず、また自己蒸留によって元モデルの
挙動から過度に離れてしまうリスクがある。

**解決**:
```python
# loss.py loss_terms (抜粋)
ordinal = weight * 0.5 * is_score * ((E[p]-E[q])/(K-1))**2    # scoreのみ
kl = kappa * KL(baseline || p)                                  # kappa: hard-Noulは0.2, 他は0.05
```

**実装**: [loss.py](loss.py) の `loss_terms()`

---

## 処理フロー詳細

### 推論フロー (model.py)

```python
from model import option_labels, build_prompt, Readout

labels = option_labels(tokenizer)                 # 255個の単一トークンラベル
prompt_text = build_prompt(tokenizer, row, labels)  # "...The correct answer is ("
readout = Readout(base_model, label_ids, full_rank=True)
logits = readout(input_ids, attention_mask)         # (B, K=255)
probs = logits[:, :n_options].softmax(-1)
```

### ファインチューニングフロー (finetune_example.py)

```python
import pandas as pd
from finetune_example import train, _make_synthetic_dataframe

df = _make_synthetic_dataframe("train.jsonl", n_rows=24)
readout = train("train.jsonl", epochs=2, batch_size=4, lr=5e-4)
```

### 本物のチェックポイントでの推論フロー (hf_inference_example.py)

```python
from hf_inference_example import load_real_model
from model import option_labels, build_prompt, Readout

tokenizer, model = load_real_model()  # jylin001206/InnerJev-4B-Full をBF16ロード
labels = option_labels(tokenizer)
readout = Readout(model, label_ids, full_rank=True)
# ... build_prompt + readout(...) で実際の決定を取得
```

---

## 学習データフォーマット

`finetune_example.py` が期待する pandas DataFrame のスキーマ (公式JSONL
`innerjev/data.py::read_rows` と同一スキーマ):

| カラム | 型 | 説明 |
|---|---|---|
| `id` | str | 一意なID |
| `type` | str | `"choice"` / `"noul"` / `"score"` |
| `state` | str または dict | 状況説明 (構造化データの場合はJSON化して埋め込む) |
| `question` | str | 質問文 |
| `options` | List[{"key": str, "desc": str}] | 2〜255個の順序付き選択肢 |
| `train_target` | List[float] | 教師分布 (optionsと同じ長さ, 合計1) |
| `p0` | List[float] | 蒸留前ベースライン分布 (KL正則化の基準点) |
| `gold_target` | List[float] | 正解分布 (評価用, hard-Noul判定にのみ使用) |

公式の実際の学習設定 (README "Training"より): AdamW (β=(0.9,0.999),
ε=1e-8, weight decay 0), グローバルバッチ64, warmup 3%後cosine decay,
勾度クリップ1.0。LoRAはr=64/alpha=128/dropout0・lr=5e-5 (DDP)、full-rankは
lr=2e-6 (FSDP, FP32マスター重み)。学習入力はtruncationされない
(最大6,000トークン)。

---

## FAQ

### Q1: なぜ公式のモデルをそのまま使わず再実装したのか?

**A**: InnerJevのロジック自体 (`Readout`, `loss_terms`, プロンプト構築) は
全て45行/30行程度の非常にシンプルな素のPyTorchコードであり、再実装の技術的な
壁は無い。実際、本サーベイでは `jylin001206/InnerJev-4B-Full`
(ローカルにダウンロード済み) を実際にロードして、本サーベイの
`model.py::Readout`/`option_labels`/`build_prompt` をそのまま使って
本物の決定を取得できた (`hf_inference_example.py`、下記Q3参照)。
`model.py`/`loss.py` はそのため公式 `learning.py`/`prompts.py` とほぼ
1対1に対応するロジックである。`finetune_example.py` だけはモデルサイズの
都合 (バックボーンを4Bから数十パラメータ規模の `TinyCausalLM` に縮小) で
スタンドインを使っているが、これは学習ロジック自体の単純化ではなく、
単に「小さいモデルで学習ループが正しく動くこと」を素早く確認するためのもの。

### Q2: InnerJevの「新規性」は結局どこにあるのか? 読み出し自体を提案したのはInnerJevなのか?

**A**: いいえ、違う。「First-Token Decision Readout」(次トークン分布を
選択肢ラベルに制限して読み出す) 自体は、学習済みLLMが元から持っている性質を
取り出す方法であり、InnerJev論文は何も新しいモデル機構をここに追加していない。
論文中にも "Even without training, this readout of Qwen3.8-27B comes
within a few points of Jev" (学習無しでもこの読み出しだけでJevに近い精度が
出る) と明記されている。

InnerJev論文が実際に新しく付け加えたのは次の2点である:
1. **Reasoning-to-Readout Self-Distillation** (中心的な新規性): 同じモデルに
   thinkingを有効にして2回推論させ、その回答分布を教師として、thinkingなしの
   読み出しをそれに近づける自己蒸留訓練。これにより「無学習でもある程度
   使える読み出し」を「thinking相当の精度を1回のforward passで出せる読み出し」
   に改善する
2. **タイプ別複合損失**: choice/noul/scoreを単一の損失(重み付きCE+順序回帰項
   +KL正則化)で扱うための工学的な詰め

つまり「InnerJev-4B/27B」という名前が指すのは、読み出し技術そのものではなく、
**この自己蒸留訓練によって生まれた具体的なモデル**である。

### Q3: `hf_inference_example.py` で実際に何を確認したのか?

**A**: ローカルにキャッシュされていた公式チェックポイント
`jylin001206/InnerJev-4B-Full` (Qwen3.5-4Bベース, full-rank学習済み,
BF16で約10GB) を実際にGPUへロードし、本サーベイの `Readout`/`option_labels`/
`build_prompt` を使って、以下3つの実際の質問に対する決定を1回のforward
passずつで取得した:

- **choice** (コードレビューでの対応): `request_changes` に99.88%の確率を
  割り当てた (API破壊的変更へのdeprecationパス要求という、人間の良識的判断と
  一致)
- **noul** (エラーの原因判定): 「third-partyのネットワーク起因である」
  (`no`) に99.24%
- **score** (保守性への懸念度, 0〜4の5段階): 期待スコア3.14 (長大な関数+
  深い入れ子+テスト無し、という記述に対して「severe」に近い妥当な評価)

これにより、本サーベイの `Readout` 実装 (255ラベル語彙・最後の有効トークン
位置からの読み出し・softmax正規化) が、実際の公開済み学習済みモデルに対して
論文の記述通りに動作することを実測で確認した。

### Q4: LoRAとfull-rankで性能差はあるのか?

**A**: 論文の報告 (Section "InnerJev sits on the accuracy-latency frontier")
によれば、同じモデルサイズであればLoRA版とfull-rank版は1ポイント以内の
差に収まり、精度の向上は「何を学習するか (自分自身の思考後の分布)」に
起因し、「重みの更新方法 (LoRAかfull-rankか)」にはほとんど依存しないと
述べられている。本サーベイの `finetune_example.py` はLoRA
(r=64, alpha=128, 公式と同一構成) を使っているが、`build_lora_model()` の
`LoraConfig` を外せば同じ `train()` ループでfull-rank学習にもそのまま
対応できる (対象となる線形層を `requires_grad_(True)` にするだけでよい)。

### Q5: 自己蒸留の教師分布 (2回のthinkingサンプルの平均) 自体はこのサーベイのコードで作れるか?

**A**: 作れない (意図的に対象外)。本サーベイの `loss.py::loss_terms()` は
「教師分布 `train_target` が既に与えられている」ことを前提とした損失関数
であり、`train_target` 自体の生成 (同モデルにthinking有効で2回サンプリング
し、`Readout` で読み出した分布を平均する) は学習データ準備段階の処理で、
公式リポジトリにもこのサンプリング自体のスクリプトは含まれていない
(配布されるのは生成済みのJSONLデータセット)。本サーベイの
`finetune_example.py::_make_synthetic_dataframe()` では、この手順を
省略し、ランダムな合成分布を `train_target` として直接与えている。

### Q6: 実運用でこのサーベイのコードをそのまま使えるか?

**A**: `model.py` の `Readout`/`option_labels`/`build_prompt` は本物の
公開チェックポイントで動作確認済み (Q3) であり、`AutoTokenizer`/
`AutoModel` でロードした任意のcausal LMにそのまま使える。
`loss.py::loss_terms()` もPyTorch標準APIのみに依存しており、`train_target`/
`p0`/`gold_target` を用意できればそのまま使える。`finetune_example.py` の
`TinyCausalLM`/`ToyTokenizer` は、実運用では
`transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B")` と
対応する `AutoTokenizer` に差し替える前提のスタンドインである。

---

## まとめ

InnerJev は、「学習済みLLMが元から持っている First-Token Decision Readout」
という性質を、自己蒸留で実際に使える精度まで引き上げる論文である。背景となる
機構 (0) と、この論文自体が新しく付け加えた部分 (①②) を分けて整理すると:

0. **(背景, InnerJevのオリジナルではない)** First-Token Decision Readout:
   次トークン分布を255個の単一トークン選択肢ラベルに制限して読み出すだけで、
   生成無しに決定が得られる。学習済みLLMに何も追加学習せずとも一定の精度が
   出ることが論文中で確認されている
1. **(InnerJevの新規性①, 中心的な貢献)** Reasoning-to-Readout
   Self-Distillation: 同じモデルの「thinkingあり2回推論の平均」を教師として、
   「thinkingなし1回読み出し」の生徒を学習する自己蒸留。正解ラベルは使わない
2. **(InnerJevの新規性②)** タイプ別複合損失: choice/noul/scoreを単一の
   ヘッド・単一の損失 (重み付きCE + score専用順序回帰項 + KL正則化) で
   統一的に扱う

これらの訓練をQwen3.5-4B/Qwen3.8-27Bに適用した結果が、InnerJev-4B/27Bという
具体的なモデルである。**この論文の知的な新規性の核心は0ではなく①(と②)であり、
「読み出し技術そのものの発明」ではなく「読み出しを自己蒸留で改善する訓練手順」
である点に注意。**

**公開状況**: アーキテクチャ・学習ループ・推論エンジンのコードは全て
素のPyTorchで公開 (本サーベイが調査した中で最もシンプル・最も完全に公開
されている実装)。本サーベイは実際にダウンロード済みの公式チェックポイント
`jylin001206/InnerJev-4B-Full` をロードして `Readout` が本物の決定を正しく
読み出すことを確認済みであり (`hf_inference_example.py`)、
`finetune_example.py` のみモデルサイズの都合でバックボーンを縮小した
スタンドインを使っている。

---

## 参考文献

- 論文: General Decision Models: Benchmarking and Insights Beyond Jev
  (ローカルクローン `paper/`)
- 公式リポジトリ: ローカルクローン `gh/InnerJev`
- 公開モデル: [jylin001206/InnerJev-4B-Full](https://huggingface.co/jylin001206/InnerJev-4B-Full),
  [jylin001206/InnerJev-4B-LoRA](https://huggingface.co/jylin001206/InnerJev-4B-LoRA),
  [jylin001206/InnerJev-27B-Full](https://huggingface.co/jylin001206/InnerJev-27B-Full),
  [jylin001206/InnerJev-27B-LoRA](https://huggingface.co/jylin001206/InnerJev-27B-LoRA)
- 公開データセット: [jylin001206/InnerJev-4B-Training-Data](https://huggingface.co/datasets/jylin001206/InnerJev-4B-Training-Data),
  [jylin001206/InnerJev-27B-Training-Data](https://huggingface.co/datasets/jylin001206/InnerJev-27B-Training-Data)
- 関連: Qwen3.5/Qwen3.8 (ベースモデル), LoRA (Hu et al. 2022, 軽量ファインチューニング),
  Knowledge Distillation (Hinton et al. 2015, 自己蒸留の目的関数の基礎)

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。
`model.py`/`loss.py`/`hf_inference_example.py` は公式実装とほぼ同一のロジックで、
実際に公開済みチェックポイントで動作確認済みです。`finetune_example.py` 内の
`TinyCausalLM`/`ToyTokenizer` のみ、モデルサイズの都合による縮小スタンドインで、
実際の実装 (Qwen3.5-4B/Qwen3.8-27B) とは異なります。
