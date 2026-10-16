# OneStreamer Understanding - 簡略化疑似コード集

**OneStreamer: Unifying Perception, Memory, and Proactive Response in Streaming Video Interaction**
の理解を目的とした簡略化疑似コード集です。

論文: [arXiv:2610.01762](https://arxiv.org/abs/2610.01762)
公式コード: [github.com (MCG-NJU/OneStreamer, ローカルクローン `gh/OneStreamer`)](https://mcg-nju.github.io/OneStreamer/)
公開チェックポイント: [huggingface.co/MCG-NJU/OneStreamer-4B](https://huggingface.co/MCG-NJU/OneStreamer-4B)
公開データセット: [huggingface.co/datasets/MCG-NJU/OneStreamer-1M](https://huggingface.co/datasets/MCG-NJU/OneStreamer-1M)

## 📋 目次

- [概要](#概要)
- [Qwen3-VLの通常動画モードとの差分 (新規性はどこにあるか)](#qwen3-vlの通常動画モードとの差分-新規性はどこにあるか)
- [公開物の実態 (重要)](#公開物の実態-重要)
- [アーキテクチャ全体像](#アーキテクチャ全体像)
- [ファイル構成](#ファイル構成)
- [OneStreamerの主要イノベーション](#onestreamerの主要イノベーション)
- [処理フロー詳細](#処理フロー詳細)
- [学習データフォーマット](#学習データフォーマット)
- [FAQ](#faq)

---

## 概要

**OneStreamer は何なのか:** 一言で言えば、**「ライブ配信や監視カメラのように
途切れなく流れ続ける映像を見ながら、質問が来るのを待たずに自分から記録・要約・
応答するビデオチャットボット」**である。通常のVideo-LLM (動画質問応答モデル) は
「切り出された動画クリップ + 質問」をまとめて受け取り、1回の生成で回答する
という使い方しかできない。OneStreamerはこれとは違い、終わりの無い映像ストリーム
を1フレームずつ受け取り続けながら、「今は何も言わず観察を続ける」「今の出来事を
記録しておく (メモリ形成)」「今まさに応答すべきタイミングだ (能動応答)」という
3種類の判断を、都度自分で下しながら動き続けるモデルである。

**どんな場合に使えるか:** 配信の実況・見守りカメラの異常検知・工場ラインの
継続監督など、「質問される前からの出来事」について後から聞かれても答えられる
必要があるアプリケーション、あるいは「何かが起きたらモデル側から知らせてほしい」
という能動的なアラート/カウント用途が主な対象になる。具体的なタスクは下記の
3種類で、いずれも標準的なVideo-LLM (例: Qwen3-VLの動画モード) では「質問が来て
初めて動画を読む」という前提のため原理的に扱えない。

**OneStreamerの特徴:**
- **Query非依存の記憶形成**: 質問が来る前から時間整合的なキャプションを書き続け、
  後から来る質問に答えられる「再利用可能な事実メモリ」を構築する
- **知覚とメモリ形成を共有の生成過程で統一**: </Observe>/</Summary>/</Silence>/</Standby>/</Response>
  という5つの制御トークンだけで「記録する/観察を続ける/応答する」を1つの自己回帰生成として扱う
- **密な状態教師信号の問題を解決**: PSTL (Proactive State Transition Learning) により
  </Silence> の洪水に教師信号を支配されず、状態変化・持続の両方を学習する
- **8つのストリーミング動画理解ベンチマークで比較手法中最高性能** (4Bモデル)

**タスク:**
- Streaming Video Captioning / Memory Formation (PHCM)
- Proactive Video QA (質問に「いつ」答えるかも学習する)
- Proactive Interaction (継続説明・アラート・能動カウント等)

**ベースモデル:** Qwen3-VL-4B-Instruct (Vision Encoder + Projector は凍結、
LLM本体とProjectorのみ学習, 表3参照)

**性能** (論文 Table 1-2 より抜粋, 4Bモデル同士の比較):
| Model | OVOBench Overall | StreamingBench RT | ProactiveVQA Avg. | OVO-Timing F1 |
|-------|---:|---:|---:|---:|
| Qwen3-VL (base) | 58.8 | 81.8 | 34.3 | 29.4 |
| VideoChat3 | 58.5 | 81.9 | 37.6 | 33.6 |
| **OneStreamer-4B** | **72.1** | **86.9** | **48.7** | **41.6** |

---

## Qwen3-VLの通常動画モードとの差分 (新規性はどこにあるか)

### 変わらない部分

「クエリ＋直近Nフレームを入れて生成する」という**入出力の形自体はQwen3-VLの動画モードと
同じ**。Vision Encoder→Projector→LLMという経路、画像トークンの埋め込み方なども一切
変えていない (姉妹フォルダ [survey/Qwen3VL](../Qwen3VL/) の実装がそのまま使える)。

### 変わる部分は2段階ある

**① 制御トークンによる「プロアクティブ応答」のループ化 (PSTLが学習を支える)**

通常のQwen-VL動画チャットは「動画+質問 → 即答える」というreactiveな1発生成。
OneStreamerは**毎ステップ** `</Silence>`/`</Standby>`/`</Response>` のどれかを
出させる自己回帰ループに変えている。これはPHCMなしでも成立する変化で、論文の
ablationでいう「FIFO」設定 (Recent-N窓だけ、記憶保持なし) でもこの制御トークン
機構は使われている。つまり「いつ答えるか」を学習させる部分はPHCMより基礎的な差分。
これを学習可能にする工夫が **PSTL** (`</Silence>`過多による学習の偏りを防ぐ)。

**② PHCM = 長期記憶の作り方**

①の仕組みの上に、「ウィンドウから出たフレームの情報を捨てずにテキスト化して残す」
のがPHCM。論文タイトルの"Memory"に対応する部分。キモは:
- 記憶の**内容**: 生の視覚特徴ではなく、モデル自身が書いた `</Observe>`/`</Summary>` テキスト
- 記憶の**作り方**: 質問が来る前から (query-independentに) 書き続け、学習時も同じ
  キャプション生成を教師信号にしている (推論時だけの後付けプロンプトではない)。
  Appendix A.3のablationでは、推論時にプロンプトで同様の説明をさせても、
  PHCMとして学習済みのモデルほどの記憶効果 (歴史的QAの改善) は出ないことが
  示されている

まとめると、**「いつ応答するか」(①制御トークン+PSTL) と「何を長期記憶として残すか」(②PHCM)**
の2つが新規性であり、②が「普通のQwen-VL動画利用」からの見た目の差分として最も分かりやすい。

### 学習・推論の仕組み (特別な分類器ではなく通常のNTP)

アーキテクチャ上、状態を判定する専用の分類ヘッドは存在しない。制御トークンは
語彙に追加された5つの特殊トークンに過ぎず、**完全に通常のNext Token Prediction
に乗っかっているだけ**。

- **学習時のGTラベル**: データセットの `timeline` アノテーション (silence区間の
  `interval` / observe等の `at` ポイントイベント) から、各ステップで「次に出すべき
  制御トークン」を機械的に割り当てて作る (`timeline_to_steps()`)。これを
  「`<画像トークン>` の直後に制御トークンID、本文が続く場合はその本文トークン」という
  ただのトークン列に変換し (`OneStreamerCollator._build_conversation`)、通常の
  Cross-Entropy (shift NTP loss) で学習する
- **PSTLがやっているのはラベルの間引きだけ**: アーキテクチャやlossの形は変えず、
  「どの位置の制御トークンに教師信号 (label) を与えるか」を選別するのみ。選ばれなかった
  位置は `labels=-100` で ignore_index 扱いになる。本文 (キャプション/回答) のラベルは
  常にそのまま与えられる (`pstl_loss.py`)
- **推論時も特別な閾値判定ではなく通常の自己回帰デコード** (greedyまたはサンプリング)
  の結果そのもの。1フレーム分の画像トークンを追加して `model.generate()` を呼び、
  出てきたテキストの先頭付近に含まれる制御トークンを `find_control_token()` で検出し、
  それに応じて「何もしない(Silence)」「キャプションを履歴に積む(Observe/Summary)」
  「ユーザーに見せる(Response)」を分岐する。つまり「分類モデルが判断して生成モデルに
  指示する」のではなく、**生成モデル自身が次トークンとして出したものがそのまま
  行動の指示になる**、という一段構成

---

## 公開物の実態 (重要)

本サーベイを作る過程で、ローカルにクローンされた公式リポジトリ (`gh/OneStreamer`) と
HuggingFace 上の公開物を実際に調べ、以下を確認した。これは「何を再実装し、何を
参照に留めるか」の判断根拠になっている。

| 公開物 | 内容 | 確認した事実 |
|---|---|---|
| `gh/OneStreamer/Inference/inference.py` | 推論デモ (131行) | `StreamingSession` (Recent-N FIFOウィンドウ) のみ。学習コードは一切含まれない |
| `gh/OneStreamer/Eval/VLMEvalKit/.../qwen3_ovotiming/inference_fast.py` | 評価用の実推論実装 (424行) | FIFOウィンドウからフレームが出る際、**キャプションも一緒に削除**される実装 (`del messages[first_u:first_u+2]`)。つまり公開コードは論文の主要提案 PHCM (キャプション保持) を実装していない、Recent-N FIFOベースラインである |
| `huggingface.co/MCG-NJU/OneStreamer-4B` の `config.json` | モデル定義 | `architectures: ["Qwen3VLForConditionalGeneration"]` のみ。カスタム `modeling_*.py` は同梱されていない = 標準の `transformers.Qwen3VLForConditionalGeneration` をそのままロードできる |
| 同 `added_tokens.json` | 追加特殊トークン | `</Response>`:151669, `</Standby>`:151670, `</Silence>`:151671, `</Summary>`:151672, `</Observe>`:151673 が Qwen3-VL の既存特殊トークン (151643-151668) に続けて実際に登録されている |
| `huggingface.co/datasets/MCG-NJU/OneStreamer-1M` | 学習データ (JSONL) | `schema: onestreamer-public-meta-v1` の実レコードを確認。`timeline` 配列が `{"type":"silence","interval":[lo,hi]}` / `{"type":"observe","at":t,"text":...,"prompt_id":...}` 形式で構成されている (下記「学習データフォーマット」に実例を引用) |

**結論:** バックボーン (Vision Encoder + 36層LLM) 自体は公開されている標準アーキテクチャで
再実装の必要が無い (姉妹フォルダ [survey/Qwen3VL](../Qwen3VL/) を参照)。一方で
論文の核心である **PHCM のメモリ保持ロジック** と **PSTL の学習時ラベル構築** は
公開コードに含まれていないため、本サーベイでは論文本文・Appendixの記述に基づいて
これらを実際に動くコードとして再構築した (各ファイルのdocstringに「再構築である」旨を明記)。

---

## アーキテクチャ全体像

```
入力: ライブ動画ストリーム (時間順の clip 列) + (タスクによっては) ユーザー指示
        ↓
┌─────────────────────────────────────────────────────────┐
│ 1. Vision Encoder + Projector (Qwen3-VL, 凍結)            │
│    各 clip -> 視覚トークン列                               │
│    詳細: 姉妹フォルダ survey/Qwen3VL/vision_encoder.py 参照  │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Recent-N FIFO 視覚ウィンドウ (streaming_session.py)      │
│    直近 N フレームの視覚トークンだけを保持                    │
└─────────────────────────────────────────────────────────┘
        ↓ (ウィンドウを出たフレームの扱いが FIFO/PHCM の分岐点)
┌─────────────────────────────────────────────────────────┐
│ 3. テキスト履歴 = 対話 + PHCM キャプション記憶               │
│    </Observe> ローカル詳細キャプション / </Summary> イベント要約 │
│    フレームがウィンドウを出てもキャプションは残る (PHCM)      │
└─────────────────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────────────────┐
│ 4. LLM (Qwen3, causal) : 視覚トークン + テキスト履歴の混合列  │
│    制御トークンを含む因果シーケンスとして自己回帰生成          │
│    (control_tokens.py)                                   │
└─────────────────────────────────────────────────────────┘
        ↓
出力: </Silence> (継続観察) / </Observe>|</Summary> (記録)
      / </Standby> (証拠不十分) / </Response> + 本文 (応答)
```

学習時は上記と同じ因果シーケンスを教師強制 (teacher forcing) で与え、
PSTL (pstl_loss.py) で状態トークンへの教師信号を選択的に間引く
(main_flow.py が全体を統合)。

---

## ファイル構成

### 1. [control_tokens.py](control_tokens.py)
**モデルブロック: 制御トークン語彙とタスク別状態機械**

- `CONTROL_TOKENS`: 5つの制御トークン文字列 (`</Silence>`, `</Observe>`,
  `</Summary>`, `</Standby>`, `</Response>`)
- `TASK_SPECS`: タスクごとの状態語彙とPSTL出力アンカー集合 $\mathcal{A}_\tau$
  (論文 Appendix A.1 Table をそのまま構造化)
- `add_control_tokens()` / `resize_embeddings_for_control_tokens()`:
  実際の HuggingFace `tokenizer.add_special_tokens` / `model.resize_token_embeddings`
  API を呼ぶ関数 (新規埋め込み行は既存埋め込みの平均で初期化)

### 2. [pstl_loss.py](pstl_loss.py)
**ロス: Proactive State Transition Learning (論文 Sec 3.3, Appendix A.1)**

- `select_supervised_positions()`: 遷移グループ化 → クォータ $q$ 計算 (Eq.1) →
  一様無復元サンプリング (Eq.2) を実装
- `build_labels_with_pstl()`: 既存の全トークン教師ラベルに対し、選択されなかった
  制御トークンの位置だけ `-100` で上書きする (キャプション/応答本文には触れない)
- `compute_loss()` / `compute_loss_breakdown()`: 通常の shift+Cross-Entropy と、
  PSTL適用前後の比較分析

### 3. [streaming_session.py](streaming_session.py)
**キー実装: 推論時の Recent-N FIFO & PHCM キャプション記憶**

- `FIFOStreamingSession`: 公式 `inference_fast.py::StreamingSession` を再現した
  ベースライン (フレームがウィンドウを出ると対応するキャプションも失われる)
- `PHCMStreamingSession`: 論文 Appendix A.2 の記述に基づく再構築。
  フレームがウィンドウを出ても `</Observe>`/`</Summary>` のキャプションは
  テキスト履歴に永続的に残す
- `example_usage()`: 同一の合成フレーム列で両者を実際に走らせ、
  最終的に残るキャプション数の違いを数値で比較する

### 4. [main_flow.py](main_flow.py)
**メインフロー: 学習時フォワードパス全体 (バックボーン + PSTL ラベル統合)**

- `OneStreamerForTraining`: 任意のCausalLMバックボーンをラップし、
  バッチ内サンプルごとに `pstl_loss.build_labels_with_pstl` でラベルを
  調整してから `backbone(input_ids=..., labels=...)` を呼ぶ
- `example_usage()`: 合成PHCMシーケンスに対して実際に forward/backward を実行し、
  PSTL適用の有無で loss が変わること、勾配が流れることを確認する

### 5. [finetune_example.py](finetune_example.py)
**実際に動くファインチューニングスクリプト (Trainer不使用, PyTorch基本機能のみ)**

- `OneStreamerDataset` / `OneStreamerCollator`: pandas DataFrame
  (実データセット `OneStreamer-1M` と同じ `timeline`/`prompts` スキーマ) から
  因果視覚言語シーケンス + PSTL適用済みラベルを構築
- `timeline_to_steps()`: 区間注釈 (`silence`/`standby`) と点注釈
  (`observe`/`summary`/`response`) を離散フレームステップ列に変換
- `train()` / `evaluate()`: AdamW + Cosine decay + 勾配累積 + AMP の手動訓練ループ
  (ハイパーパラメータは論文 Appendix "Training Configuration" に準拠)
- `main()`: 実行すると合成動画 (cv2で生成) を使って全パイプラインを実際に
  1ステップ学習させるデモが走る (`python finetune_example.py`)

### 6. [hf_inference_example.py](hf_inference_example.py)
**公開HuggingFaceモデル (`MCG-NJU/OneStreamer-4B`) を使った実推論サンプル**

- `HFEngine`: `transformers.AutoProcessor` / `Qwen3VLForConditionalGeneration` を
  実際にロードして使う推論エンジン (公式 `_Qwen3VLInferenceFastEngine` と同一ロジック)
- `video_windows()` / `resize_frame_for_streaming()`: 公式 `inference.py` と同じ
  フレームサンプリング・固定トークン予算へのリサイズ
- `run()`: `--memory phcm`/`fifo` で `streaming_session.py` の
  `PHCMStreamingSession`/`FIFOStreamingSession` を切り替えて実行できる
  (公式デモが提供するのは実質 `fifo` のみ)
- **動作確認状況**: `AutoProcessor.from_pretrained("MCG-NJU/OneStreamer-4B")` の
  ロード、実際の制御トークンID (`</Silence>`:151671等) の一致、
  `apply_chat_template` による画像+テキストのトークン化 (`pixel_values`,
  `image_grid_thw` が正しい形状で得られること) はこの環境で実際に検証済み。
  言語モデル本体の重み (~8GB) はダウンロードしていないため、`model.generate()`
  を含む完全なエンドツーエンド実行は未検証 (GPU環境で実際のチェックポイントを
  指定して実行することで確認できる)

```bash
python hf_inference_example.py \
    --model-path MCG-NJU/OneStreamer-4B \
    --video path/to/video.mp4 \
    --question "How many times does the event: showing something to the camera happen?" \
    --memory phcm
```

---

## OneStreamerの主要イノベーション

### 1. **Proactive Hierarchical Caption Memory (PHCM)**
**問題**: Recent-N ウィンドウだけでは古い視覚情報を失う。全履歴を保持すると
計算量・メモリが爆発する (論文 Table "Answer-stage efficiency": Full history は
360秒で 25.18GB / 62,094 tokens / TTFT 4.56s)。

**解決**:
```python
# streaming_session.py の PHCMStreamingSession._trim_sliding_window (抜粋)
if control in (OBSERVE, SUMMARY):
    del self._messages[first_u]          # 画像だけ削除、キャプションは残す
else:
    del self._messages[first_u : first_u + 2]  # </Silence> は両方削除
```

**効果**: PHCM は Full History 比で平均レイテンシ 84.2%減, GPU memory 65.3%減,
peak入力長 90.4%減 (論文 Appendix A.3 "Efficiency of Online Memory Construction")。

**実装**: [streaming_session.py](streaming_session.py)

---

### 2. **Proactive State Transition Learning (PSTL)**
**問題**: ストリーミング系列は `</Silence>` が大半を占め、密な教師信号は
「待つこと」への過学習を招く。しかし状態変化のみ supervise すると
「状態持続」(例: 連続した `</Observe>` がそれぞれ別のローカル詳細を記録している場合)
の教師信号が失われる。

**解決**:
```python
# pstl_loss.py (抜粋): Eq.1, Eq.2 の実装
q = max(len(indices) for (x, y), indices in groups.items() if y in anchors)
for (x, y), indices in groups.items():
    k = min(len(indices), q)
    selected.update(rng.sample(indices, k))   # 一様無復元抽出
```

**効果**: 論文は OVOBench 系ablationで、状態トークンの **27.5%のみ** を
supervise しても密な教師信号より良い結果を報告している。

**実装**: [pstl_loss.py](pstl_loss.py)

---

### 3. **5つの制御トークンによる統一生成インターフェース**
**問題**: 「記録するかどうか」と「応答するかどうか」を別モジュール
(例: 外部の変化点検出器 + 別の応答モデル) で扱うと、学習信号が分断される。

**解決**: 全てを1つの自己回帰生成の出力トークンとして統一。
```python
# control_tokens.py
CONTROL_TOKENS = ("</Silence>", "</Observe>", "</Summary>", "</Standby>", "</Response>")
TASK_SPECS = {
    "phcm":                 TaskSpec(state_vocab=(...), anchors={"</Observe>", "</Summary>"}),
    "proactive_qa":         TaskSpec(state_vocab=(...), anchors={"</Response>"}),
    "proactive_interaction":TaskSpec(state_vocab=(...), anchors={"</Response>"}),
}
```

**実装**: [control_tokens.py](control_tokens.py)

---

## 処理フロー詳細

### 推論フロー (streaming_session.py)

```python
engine = ...  # transformers.Qwen3VLForConditionalGeneration をラップした推論エンジン
session = PHCMStreamingSession(
    engine=engine,
    system=SYSTEM_PROMPT,
    question="How many times does the event: showing something happen?",
    max_num_frames=16,   # Recent-16 (論文のOVOBench/StreamingBench設定)
)

for step, (frames, time_secs) in enumerate(video_stream):
    answer = session.step_frames(frames, time_secs, step)
    # answer: "</Silence>" | "</Observe> ..." | "</Summary> ..." | "</Response> ..."
    if "</Response>" in answer:
        print("proactive response:", answer.split("</Response>", 1)[-1])
```

### 学習フロー (main_flow.py + finetune_example.py)

```python
# 1タスクのタイムライン注釈 -> 離散ステップ
steps = timeline_to_steps(row["timeline"], fps=row["fps"])
# 2. 因果シーケンス構築 (OneStreamerCollator._build_conversation)
input_ids, labels_full, control_events, num_images = collator._build_conversation(
    steps, row["prompts"], row["task"]
)
# 3. PSTL でラベル調整
labels, stats = build_labels_with_pstl(labels_full, control_events, TASK_SPECS[row["task"]].anchors)
# 4. バックボーンへ forward (labels を渡すだけで内部でshift+CE lossまで計算される)
outputs = backbone(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
outputs.loss.backward()
```

**学習ハイパーパラメータ** (論文 Appendix "Training Configuration" Table より):
| パラメータ | 値 |
|---|---|
| Optimizer | AdamW ($\beta=(0.9,0.95)$, $\epsilon=10^{-8}$) |
| Peak / Min LR | $1\times10^{-5}$ / $1\times10^{-6}$ |
| LR schedule | Cosine decay + Warmup比 0.03 |
| Weight decay | 0 |
| Max grad norm | 1.0 |
| Global batch size | 128 (packed sequences) |
| Sequence長 | 131,072 tokens |
| 訓練対象モジュール | Projector + LLM (Vision Encoder は凍結) |
| Precision | BF16, FSDP, FlashAttention-2 |

---

## 学習データフォーマット

`huggingface.co/datasets/MCG-NJU/OneStreamer-1M` から実際に取得した
JSONLレコード (Flat captioning, 一部省略) をそのまま引用する:

```json
{
  "schema": "onestreamer-public-meta-v1",
  "source": "YouTube-Collected",
  "classification": {"category": "Proactive Caption Memory", "task": "Flat captioning"},
  "video_path": "Gemini_New_Annotation/videos_segment/.../1ncLcKV28Qw_segment_0001.mp4",
  "image_size": [960, 540],
  "video_ranges": [[0.0, 296.5]],
  "prompts": [{"id": "p1", "kind": "text", "at": 1.0,
               "text": "Continuously observe the video and caption each completed visual segment."}],
  "timeline": [
    {"type": "silence", "interval": [0.0, 45.0]},
    {"type": "observe", "at": 46.0, "text": " [0s-46s] The video begins with an intro ...", "prompt_id": "p1"},
    {"type": "silence", "interval": [46.0, 169.0]},
    {"type": "observe", "at": 170.0, "text": " [46s-170s] The match begins ...", "prompt_id": "p1"},
    "...",
    {"type": "silence", "interval": [296.0, 296.5]}
  ]
}
```

`finetune_example.py` の `OneStreamerDataset`/`OneStreamerCollator` はこの
`timeline`/`prompts` 構造を直接の入力として受け取る (`task` 列だけ追加で必要:
`classification.task` などから `phcm`/`proactive_qa`/`proactive_interaction`
にマッピングして用意する)。

区間イベント (`silence`/`standby`) は `fps` に応じて複数ステップへ展開され、
点イベント (`observe`/`summary`/`response`) はその時刻を終端とする1ステップに
変換される (`timeline_to_steps()`)。

---

## FAQ

### Q1: なぜ Qwen3-VL のバックボーン自体を実装しないのか？

**A**: 公開チェックポイントの `config.json` が `Qwen3VLForConditionalGeneration`
(transformers標準クラス) のみを指定しており、カスタム `modeling_*.py` が
一切同梱されていないことを実際に確認したため。つまり本物の重みを使う場合、
`transformers.Qwen3VLForConditionalGeneration.from_pretrained(...)` を呼ぶだけで
バックボーンは完成する。その完全な内部実装 (SigLIP系Vision Encoder, Interleaved
MRoPE, 36層LLM) は姉妹フォルダ [survey/Qwen3VL](../Qwen3VL/) で既に詳述している
ため、本サーベイは OneStreamer が実際に追加した部分 (制御トークン・PHCM・PSTL)
に集中している。

### Q2: 公式コードをそのまま動かせば PHCM の挙動を確認できるか？

**A**: できない。`gh/OneStreamer/Eval/VLMEvalKit/.../inference_fast.py` の
`StreamingSession._trim_sliding_window()` は、ウィンドウから押し出されるフレームの
画像メッセージと、その直後の assistant 応答メッセージ (キャプションを含み得る)
を `del messages[first_u:first_u+2]` で **まとめて削除**している。これは
論文がablationで比較する "FIFO" ベースラインの挙動そのものであり、
論文の主要提案である PHCM (キャプションをフレームより長く保持する) は
公開コードには実装されていない。[streaming_session.py](streaming_session.py) の
`PHCMStreamingSession` は論文 Appendix A.2 の記述から再構築したものである。

### Q3: PSTL は推論時にも使うのか？

**A**: 使わない。論文 Appendix A.1: "At inference, PSTL is no longer applied.
The model generates control tokens and text directly from its learned
conditional distributions." PSTL は学習時のラベル構築 (どの制御トークンに
教師信号を与えるか) だけに影響し、推論時は通常の自己回帰生成そのまま。

### Q4: PSTL のクォータ $q$ はどう決まる？

**A**: 1つの学習シーケンス内で、出力アンカー (例: PHCMなら`</Observe>`/`</Summary>`)
へ向かう遷移グループの最大サイズ。
```python
q = max_X max_{a in anchors} n_{X->a}
```
この $q$ がシーケンス内の **全ての** 遷移グループ (アンカーでないものも含む) の
サンプリング上限として共有される (論文: "the quota is computed once per training
sequence from all source-to-anchor transitions and then shared across groups")。

### Q5: `</Observe>` と `</Summary>` の違いは？

**A**: 時間粒度が異なる。`</Observe>` は短い区間の直接観察 (物体・動作・状態変化)
によるローカル詳細キャプション、`</Summary>` は完了したイベント/セグメント全体の
セマンティックサマリー (より疎・より粗い粒度)。論文 Appendix "Effectiveness of
Hierarchical Caption Memory" では両方を組み合わせることで OVOBench Overall が
最高の72.1になることを報告している (個別では71.7 / 71.2)。

### Q6: `</Standby>` と `</Silence>` の違いは？

**A**: どちらも「まだ応答しない」を意味するが、`</Standby>` は「関連する証拠が
出現しつつあるが、まだ回答するには不十分」(Proactive QA専用)、`</Silence>` は
「関連イベントが始まっていない、または入力が質問と無関係」。PHCMタスクには
`</Standby>` は存在せず、Proactive Interactionタスクにも存在しない
(control_tokens.TASK_SPECS 参照)。

### Q7: 1つの学習シーケンスにおけるフルビデオ要約の位置づけは？

**A**: PHCM (streaming caption) データでは、ストリーム終了後に
「全体を要約してください」という指示を追加し、動画レベルのキャプションを
`</Response>` の応答として与える (論文 Sec 4.1)。これは `</Observe>`/`</Summary>`
のような永続メモリレコードではなく、指示条件付きの一度限りの出力である。
`finetune_example.py` の `OneStreamerCollator._build_conversation` は
`pending_prompts` を最終ステップ直前に挿入することでこれを再現している。

### Q8: データ合成パイプラインで応答タイミングはどう決まる？

**A**: 論文 Sec 4.2 / Appendix A.4 (Fine-grained response-time calibration):
粗い証拠区間の中で1秒刻みの候補時刻を評価し、推論系QAは参照回答の条件付き
尤度 (逆パープレキシティ)、知覚系タスクは視覚-テキスト類似度でスコアリングし、
メディアンスムージング後にピークから0.04以内の最も早い時刻を採用する。
この再調整により、平均13.72秒応答時刻が早まり、86.62%のレコードがより早い
応答時刻を得たと報告されている。本サーベイはこの合成パイプライン自体は
実装対象外とし (データ合成はGemini/Seed/Qwenへの大規模API呼び出しを要し、
再現にはモデル権限とコストが必要なため)、合成済みデータの **消費側** である
学習シーケンス構築 (`timeline_to_steps`) に焦点を当てている。

### Q9: 実運用でこのサーベイのコードをそのまま使えるか？

**A**: `control_tokens.py` と `pstl_loss.py` は本物の `transformers` API
(`tokenizer.add_special_tokens`, `model.resize_token_embeddings`,
`torch.nn.functional.cross_entropy`) だけに依存しており、そのまま使える。
`streaming_session.py` も `engine` を本物の推論ラッパーに差し替えれば動く。
`main_flow.py` と `finetune_example.py` 内の `_TinyMultimodalBackboneForDemo` /
`_ToyTokenizer` / `_ToyCausalLM` はネットワーク接続なしで動作確認するための
最小限のスタンドインであり、実運用では `transformers.AutoTokenizer` /
`transformers.Qwen3VLForConditionalGeneration` に差し替える
(メソッド名・契約は同一に揃えてある)。

---

## まとめ

OneStreamer は以下の3つの要素でストリーミング動画理解を統一的に扱う:

1. **5つの制御トークン**: `</Silence>`/`</Observe>`/`</Summary>`/`</Standby>`/`</Response>`
   による「記録/観察継続/応答」の統一生成インターフェース
2. **PHCM**: Recent-N視覚ウィンドウ + 永続キャプション記憶による効率的な長期記憶
3. **PSTL**: 遷移ベースのクォータ方式による状態トークン教師信号の間引き

**ベースモデル**: Qwen3-VL-4B-Instruct (標準 `transformers` クラスのまま拡張)

**公開状況**: 推論デモ・評価コードは公開済みだが、PHCMのメモリ保持ロジックと
学習コード (PSTL含む) は非公開。本サーベイはそのギャップを論文記述から
実際に動くコードとして埋めている。

---

## 参考文献

- 論文: [OneStreamer: Unifying Perception, Memory, and Proactive Response in Streaming Video Interaction](https://arxiv.org/abs/2610.01762)
- 公式リポジトリ: ローカルクローン `gh/OneStreamer` (Inference/ と Eval/ のみ公開)
- 公開チェックポイント: [huggingface.co/MCG-NJU/OneStreamer-4B](https://huggingface.co/MCG-NJU/OneStreamer-4B)
- 公開データセット: [huggingface.co/datasets/MCG-NJU/OneStreamer-1M](https://huggingface.co/datasets/MCG-NJU/OneStreamer-1M)
- ベースモデル: [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) (姉妹フォルダ [survey/Qwen3VL](../Qwen3VL/) 参照)
- 評価基盤: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)

---

**Note**: このドキュメント群は理解を目的とした簡略化疑似コードです。
バックボーン (Qwen3-VL) は再実装せず標準APIの利用を前提とし、PHCM/PSTLなど
公開コードに含まれない部分は論文記述からの再構築であることを各ファイルに
明記しています。実際の実装 (非公開の学習コード) とは異なる場合があります。
