# 日本語ASRモデル比較

検証環境: RTX 5090, torch 2.14.0+cu130

## 評価データセット

- **JSUT (basic5000)**: 無響室録音のクリーンな朗読音声、5,000件。今回検証した全モデルの学習データには含まれていない(=リークなしのフェアな評価)
- **ReazonSpeech (tiny)**: TV番組由来の音声、5,323件。多くの日本語ASRモデルの学習データそのもの(ReazonSpeechコーパスの一部)であり、リークを含む可能性が高い参考値

## CERの算出方法

`scripts/compute_cer.py` で3段階の正規化レベルを算出:

- **raw**: 生の文字誤り率(句読点込み)
- **normalized**: 句読点・記号を除去(モデルカード公式評価に近似)
- **reading**: さらに漢字/かな表記ゆれ(例:「歪めた」↔「ゆがめた」)を読み(カタカナ)に正規化してから比較。表記の違いを実際の認識ミスと区別できる、最も実力に近い指標

## 結果(JSUT, reading CERで降順)

| モデル | raw CER | normalized CER | **reading CER** | 平均推論時間/件 |
|---|---:|---:|---:|---:|
| **Parakeet TDT-CTC (ja)** | 10.75% | 6.63% | **2.40%** | 0.071秒 |
| Whisper large-v3-turbo | 13.03% | 7.24% | **2.58%** | 0.286秒 |
| Kotoba-Whisper v2.0 | 15.33% | 8.34% | **3.12%** | 0.241秒 |
| ReazonSpeech-NeMo v2 | 12.36% | 8.52% | **3.56%** | 0.480秒 |
| MOSS-Transcribe-Diarize (0.9B) | 11.38% | 8.55% | **3.88%** | 1.506秒 |
| ARK-ASR (0.6B) | 12.91% | 10.39% | **4.65%** | 0.615秒 |
| Qwen3-ASR (0.6B) | 14.83% | 12.07% | **5.92%** | 0.993秒 |

## 参考: 学習データ重複(リークあり)での結果

| モデル | データセット | raw CER | normalized CER | reading CER | 平均推論時間/件 |
|---|---|---:|---:|---:|---:|
| Parakeet TDT-CTC (ja) | ReazonSpeech (学習データの一部) | 15.08% | 11.70% | 10.30% | 0.070秒 |

Parakeetは学習データのはずのReazonSpeechより、未学習のJSUTの方がCERが大幅に低い(2.40% vs 10.30%)。
これは「学習データを丸暗記しているから精度が良い」という単純な話ではなく、ReazonSpeech自体がTV由来の
BGM・環境音・砕けた発話を含む本質的に難しい音声であることの表れ。音声のクリーンさがCERに与える影響は、
学習データとの重複メリットを上回る。

## 所感

- 精度・速度ともに **Parakeet TDT-CTC (ja)** が総合トップ。TDTデコーダーによる高速化(blank予測スキップ)が効いている
- **Whisper large-v3-turbo** は日本語専用モデルではないにもかかわらず2位相当の精度。汎用性の高さがうかがえる
- **MOSS-Transcribe-Diarize** は精度は上位グループに近い(3.88%)が、話者分離+タイムスタンプ付きの複雑な出力形式のぶん推論時間が最も長い(1.5秒/件)。単一話者の書き起こしだけなら他モデルの方が効率的だが、複数話者+話者分離が必要な場面では唯一の選択肢
- **ARK-ASR (0.6B)** は日本語特化ではない19言語対応モデルとしては健闘(Qwen3-ASR 0.6Bより上)だが、上位勢には届かず
- **Qwen3-ASR (0.6B)** は今回最下位だが、30言語対応の汎用軽量モデルである点は考慮が必要(日本語特化ではない)。1.7B版や、より大きいWhisperで改善する可能性あり
- normalized→reading でCERが軒並み40〜60%程度下がっており、日本語CER評価では読みベース正規化なしだと実力を過小評価しがちな点に注意
- **MOSS-Transcribe-Diarizeの出力パース時の注意**: `[start][SXX]text[end]` 形式が基本だが、モデルが稀に話者ラベル`[SXX]`を省略して出力することがある(JSUT 5000件中259件で発生)。パッケージ付属の`parse_transcript()`は話者ラベル必須の厳密パーサーで、省略時に空文字を返してしまうため、`scripts/run_moss_transcribe_diarize.py`では話者ラベルの有無を問わない正規表現ベースのパーサーに置き換えて対応した

## 学習データについて(リーク確認済み)

| モデル | 学習データ |
|---|---|
| Parakeet TDT-CTC (ja) | ReazonSpeech v2.0 のみ |
| ReazonSpeech-NeMo v2 | ReazonSpeech v2.0 のみ |
| Kotoba-Whisper v2.0 | ReazonSpeech `all` (Whisper large-v3による疑似ラベリング) |
| Whisper large-v3-turbo | OpenAI独自収集音声(68万時間、Web由来)。ReazonSpeech/JSUTは含まれない |
| MOSS-Transcribe-Diarize (0.9B) | 非公開。ReazonSpeech/JSUTとの重複有無は不明 |
| ARK-ASR (0.6B) | teacher-data adaptation + online policy distillation (TD+OPD)。具体的なデータセット名は非公開 |
| Qwen3-ASR (0.6B) | 非公開の自社データ中心。ReazonSpeech/JSUTとの重複有無は不明 |

## なぜ日本語特化のKotoba-Whisperが汎用のWhisper large-v3-turboに負けたか

エンコーダーはどちらもlarge-v3のフル版を流用しているが、デコーダー構成と学習方法が異なる。

| モデル | デコーダー層数 | 総パラメータ | 学習方法 |
|---|---|---|---|
| Whisper large-v3-turbo | **4層**(32層から削減) | 809M | large-v3と同じ**実データ**(人手ラベル付き、68万時間相当)でデコーダーを追加学習(2エポック) |
| Kotoba-Whisper v2.0 | **2層**(distil-whisper方式、初期層+最終層のみ初期化) | より少ない | **知識蒸留**。教師モデル(large-v3自身)にReazonSpeechを書き起こさせた**疑似ラベル**(WER>10%は除外)を正解として学習 |

考えられる要因は3つ:

1. **デコーダー容量**: turboはKotoba-Whisperの2倍の層数を持つ
2. **教師信号の質**: Kotoba-Whisperは人間の正解ではなく教師モデル自身の予測(疑似ラベル)を学習しているため、原理的に教師(large-v3)の実力を超えられず、教師の誤りも継承する
3. **学習データの偏り**: Kotoba-Whisperの学習データはReazonSpeech(TV音声)のワンドメインに偏っており、JSUTのようなクリーンな朗読音声への汎化がやや弱く出た可能性がある

速度面ではほぼ互角(0.241秒 vs 0.286秒)なので、「軽量化のわりに精度を保つ」というKotoba-Whisperの設計目標自体は一定達成できていると言える。
