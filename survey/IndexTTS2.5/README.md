# IndexTTS 2.5 — アーキテクチャ・学習・推論のサーベイ

## 1. 概要

IndexTTS 2.5 は、**本文から離散semantic token列を自己回帰生成し、参照音声の話者・感情を反映した音声へ変換するzero-shot TTS**である。
全体は `Text-to-Semantic (T2S) → Semantic-to-Mel (S2M) → vocoder` の3段。
各段階は異なる表現を予測する独立したモデルで構成される。

semantic tokenは、発音・言語内容・prosodyを圧縮した25Hzの離散列。
S2Mはその列とreference Mel・speaker vectorを条件として、連続値のMelをFlow Matchingで生成する。
最後にBigVGANがMelから波形を合成する。
話者参照と感情参照を別々に指定できることが、表現制御の重要な設計である。

IndexTTS 2からの主な変更は次の4点。

| 変更 | 変更内容 | 目的 |
| --- | --- | --- |
| semantic codec 50→25Hz | T2Sが生成する時刻方向token数を半減 | AR生成・学習の時間とメモリを削減 |
| S2MのU-DiT→Zipformer | Mel生成のbackboneを変更 | 連続Melの生成を高速化 |
| 多言語conditioning | boundary／token別language／instructionの3案を比較 | 共通文字の言語混同、誤発音を減らす |
| GRPO post-training | ASRとspeaker similarityのrewardでT2Sを更新 | 発音精度と話者保持を改善 |

中国語・英語・日本語・スペイン語・アラビア語を扱う。
新規性は「TransformerでTTSをすること」ではなく、既存の3段構成を保ちながら、
semantic時間圧縮、多言語識別、S2M高速化、post-trainingを組み合わせた点にある。

### 対象資料と論文・公開コードの構成差

論文は **IndexTTS 2.5 Technical Report**。
論文の説明は、提供された `main.tex` の本文に基づく。
公式コードの参照revisionは `d9e41aac89fd00b3d71497fddb287b7f24613712`。

**論文はZipformer版S2Mを報告するが、このrevisionにZipformer本体はない。**
`infer_v2_5.py → S2Melネットワーク → CFM → DiT` が公開コードで確認できる経路である。
したがって本資料は、Zipformer版の効果を論文から説明し、Pythonでは公開DiT版を実装する。
本資料のPython実装は公開DiT版を対象とし、未公開のZipformer版は実装対象に含めない。
また、`infer_v2_5.py` が実際にimportするT2Sは `gpt/model_v2.py`。
公開推論との対応を確認する際は、ファイル名ではなく、このimport先の実装を参照する。

## 2. 全体アーキテクチャ

### 2.1 軸の意味

| 記号 | 意味 |
| --- | --- |
| B | batchの発話数 |
| S16 | 16kHz参照音声のsample数 |
| L | 本文のtext token数。文字数とは異なる |
| T50 | W2v-BERT特徴の約50Hz frame数 |
| U | codec圧縮後の25Hz semantic token数 |
| D | T2Sの隠れ状態の次元数。モデル設定値 |
| Q | FVQ検索空間の次元数。公開codec既定8 |
| R | 参照Melのframe数 |
| M | 生成対象Melのframe数 |
| F | Mel bin数。公開S2Mの構成例では80 |
| C | S2Mに渡すsemantic条件の特徴次元数 |

### 2.2 推論時の3段階の処理

以下の図では、概要で説明した **①T2S、②S2M、③vocoder** をそれぞれ枠で示す。
参照音声からの特徴抽出は、3段階へ条件を供給する前処理である。
semantic codecのdecoderとlength regulatorは、T2S出力をMel生成に適した表現へ
変換するため、ここでは②S2Mの処理範囲に含める。

```text
読み上げる本文                             話者・感情の参照音声
      │ 正規化・G2P・tokenizer                      │ 特徴抽出
      ▼                                           ▼
本文ID [B,L]・言語ID [B]                 話者/感情embedding・参照特徴・参照Mel
      │                                           │
      │                  ┌────────────────────────┘
      ▼                  ▼ 話者・感情embedding
┌────────────────────────────────────────────────────────────────────┐
│ ① Text-to-Semantic（T2S）：本文からsemantic tokenを生成              │
│                                                                    │
│ 話者・感情条件 + 本文embedding + 生成済みsemantic tokenのembedding  │
│   → causal GPT-2 → semantic token分類head → 次のsemantic IDを生成 │
│                                                                    │
│ 出力：semantic IDs [B,U]（25Hz、各IDは0～8191）                     │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│ ② Semantic-to-Mel（S2M）：semantic tokenからMelを生成                │
│                                                                    │
│ IDs [B,U]                                                          │
│   → FVQ lookup + semantic codec decoder                            │
│   → 復元semantic特徴 [B,2U,1024]（約50Hz）                         │
│   → length regulator → 生成区間の条件 [B,M,C]                      │
│                                   │                                │
│ 参照W2v-BERT特徴 → length regulator → 参照区間の条件 [B,R,C]        │
│                                   │ 時間軸で連結                   │
│                                   ▼                                │
│                     semantic条件 [B,R+M,C]                          │
│                                   │                                │
│ 参照Mel [B,F,R] ──────────────────┤                                │
│ 話者embedding [B,192] ────────────┤                                │
│ 初期noise [B,F,R+M] ──────────────┤                                │
│                                   ▼                                │
│ Conditional Flow Matching：velocity推定とEuler更新を反復           │
│ velocity推定器＝公開コードはDiT、論文で報告する構成はZipformer       │
│                                                                    │
│ 出力：生成区間のMel [B,F,M]（参照区間は出力から除外）              │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│ ③ vocoder：Melから波形を生成                                       │
│                                                                    │
│ Mel [B,F,M] → BigVGAN → waveform [B,1,Sout]                          │
│                                                                    │
│ 出力：22,050Hzのmono音声（公開推論の保存形式）                     │
└────────────────────────────────────────────────────────────────────┘
```

`Sout`は生成波形のsample数。参照音声を除いた、本文に対応する波形を出力する。
①のTransformerは次の離散IDを分類し、②のモデルは連続値のMelを生成し、
③のBigVGANはそのMelを波形へ変換する。それぞれ予測対象が異なる。

### 2.3 参照音声から抽出する特徴と入力先

| 参照音声 | 抽出処理 | 抽出結果 | 入力先と用途 |
| --- | --- | --- | --- |
| 話者参照 | 16kHz音声 → 80bin fbank → CAMPPlus | 話者embedding `[B,192]` | ①T2Sの条件embeddingと、②S2Mのvelocity推定器への話者条件に使用 |
| 感情参照 | W2v-BERT → Conformer → Perceiver（1 query） | 感情embedding `[B,1024]` | ①T2Sの条件embeddingに使用し、semantic tokenの生成を条件付ける |
| 話者参照 | W2v-BERT hidden[17] → 固定統計による正規化 | 参照semantic特徴 `[B,T50,1024]` | ②S2Mのlength regulatorで参照Mel長Rへ整列し、参照区間のsemantic条件を作る |
| 話者参照 | STFT・Mel filterbank | 参照Mel `[B,F,R]` | ②S2Mに観測済みの音響的文脈として与える |

T2Sでは話者embeddingを`Linear(192,D)`で射影し、感情embeddingを
`Linear(1024,D) → Linear(D,D)`で射影して加算する。詳細は4.1節に示す。
感情条件はT2Sへ入力され、生成されたsemantic列を通じて後段へ伝わる。
S2Mには感情embeddingを直接入力しない。

公開推論では、参照側のsemantic条件は正規化済みW2v-BERT特徴から直接作る。
一方、生成側はT2Sが出力したIDをcodec decoderで復元してから条件を作る。
両者は最終的に同じ条件の特徴次元数Cへ変換されるが、その前の処理経路は異なる。
参照Melは指定された参照音声から計算し、生成対象の正解音声は推論入力に使わない。

Melのhop長・正規化方法・vocoder構成は、対応するcheckpointの設定に従う。
Mel bin数や長さの比率が同じでも、前処理の定義が異なればvocoderへの入力は互換にならない。

## 3. semantic codec：50Hz特徴から25Hzの離散ID列への変換

### 3.1 semantic tokenの抽出処理

```text
16kHz waveform [B,S16]
  │ SeamlessM4TFeatureExtractor
  ▼
acoustic input features [B,Tin,Fin]
  │ W2v-BERT 2.0
  │ hidden_states[17]を取得（公開get_emb）
  ▼
semantic features [B,T50,1024]
  │ (feature - stored_mean) / sqrt(stored_variance)
  ▼
normalized features [B,T50,1024]
  │ transpose → Conv1d(k=3,stride=2,padding=1) → GELU
  ▼
downsampled [B,1024,U], U≈T50/2
  │ Vocos/ConvNeXt feature encoder
  ▼
hidden [B,1024,U]
  │ 1×1 Conv: 1024→Q=8
  ▼
search vectors [B,8,U]
  │ 各時刻でL2-normalized nearest-neighbor search
  │ codebook [8192,8]
  ▼
IDs [B,U], 各値0..8191 ────────────────── T2Sの教師
```

これは8192種類のsemantic symbolを持つ単一codebookである。
25Hzの離散列では、約10秒の発話を約250個のsemantic tokenで表現する。
text tokenizerのglobal IDとsemantic codebook内の0..8191は別空間。
T2Sはsemantic ID専用のembeddingと分類headを持ち、本文tokenとは別の語彙として扱う。

FVQは低次元空間で距離を測る。
公開実装では検索用vectorとcodebookをL2正規化するが、
選択したcodebookのlookup値そのものを正規化してdecoderへ渡すわけではない。
この区別を [models/codec/model.py](models/codec/model.py) でも保持した。

### 3.2 codec decoderによるsemantic特徴の復元

```text
IDs [B,U]
  → Embedding [B,U,8]
  → 1×1 Conv projection [B,1024,U]
  → Vocos feature decoder [B,U,1024]
  → nearest時間拡張×2 + 学習可能Conv(k=3)
  → reconstructed semantic features [B,2U,1024]
```

codecの出力はW2v-BERTに対応する特徴であり、Melでも波形でもない。
復元特徴をlength regulatorへ渡し、その後のCFMとvocoderが音響波形を作る。
公開codecの`forward`は奇数T50を末尾1frameだけ除き、偶数長で再構成lossを計算する。
推論時の`encode`はstride Convのceil側の長さになる。

### 3.3 ConvNeXt blockの内部

```text
features [B,W,U] ────────────────────────────────────┐
  │ depthwise Conv1d(k=7,pad=3,groups=W)              │
  ▼                                                  │
[B,W,U] → transpose [B,U,W]                           │
  │ LayerNorm(eps=1e-6)                              │
  │ Linear(W,I) → GELU → Linear(I,W)                 │
  │ channel-wise learned layer scale                │
  ▼                                                  │
[B,U,W] → transpose [B,W,U] → residual add ◀──────────┘
```

WはVocos内部特徴の次元数、IはFFNの中間特徴次元数。
時間方向はdepthwise Conv、特徴方向はLinearが担当する。
codec encoder/decoderは別パラメータで、重みを共有しない。

## 4. T2S：話者・感情・言語による条件付け

**T2Sは、本文を条件に「次の音声semantic token」を予測する自己回帰タスク。**

- **入力**：本文token列・話者／感情／言語条件・これまでの音声semantic token列。
- **予測対象**：次の音声semantic token。本文の続きを予測するタスクではない。
- **学習**：正解音声をcodecで離散化したID列を教師にして、次token予測のCEを計算する。
  学習時の音声履歴には正解token、推論時には生成済みtokenを使う。

### 4.1 話者・感情embeddingの統合

```text
speaker [B,192] ─ Linear(192,D) ───────────────┐
                                             │ add
emotion [B,1024] → Linear(1024,D) → Linear(D,D)┘
                                             ▼
                                      c [B,1,D]
                                             │ concat_time
                                    zero slot [B,2,D]
                                             ▼
                                      cond [B,3,D]
```

論文は一般形 `[c,p,BT,text,BA,semantic]` を記す。
しかし公開CAMPPlus経路では、旧duration embeddingに相当する2slotへゼロを入れている。
この2slotにはduration predictorの出力は入力されない。
公開推論にはS2Mの`duration_factor`もあるが、それは別の長さ制御である。

### 4.2 text embeddingとlanguage

公開`prepare_gpt_inputs`のCAMPPlus経路では、
`text_embedding + text_position_embedding + language_embedding` を作る。
language vectorは各text位置へ加算される。

論文の「Token-Level Concatenation」という名称を、
公開コードのfeature軸に対する結合演算とは区別する必要がある。
本版の [models/t2s/model.py](models/t2s/model.py) は確認できた加算経路を使う。
sampleごとのlanguage IDを受け取り、token別language識別frontendは実装していない。
code-switchingを厳密に扱うなら、token別ID列に拡張し、教師ラベルも用意する必要がある。

### 4.3 autoregressive decoderと位置埋込

```text
sequence position:   condition       text segment           audio segment
inputs:              c  0  0 | BT text0 ... textL-1 ET | BA q0 ... qU-1
source of embedding: continuous   text embedding            audio embedding
position IDs:                       0  1 ... L+1            0  1 ... U
                      └───────────────────┬───────────────────────────┘
                                          ▼
                          GPT-2 causal decoder × N layers
                                          ▼
                                  hidden [B,S,D]
                                  ├─ text head → [B,L+1,Vtext]
                                  └─ audio head → [B,U+1,8194]
audio GT:                                                q0 q1 ... qU-1 EA
```

BA/EAはaudio開始・終了token。
codebookを8192語彙とする場合、audio headは開始・終了込み8194分類。
BAの出力で最初のq0を予測し、qU-1の出力でEAを予測する。
学習時はGT q0を入力して次のq1を予測する。各位置の予測対象は入力tokenの1つ先に配置する。

textとaudioはそれぞれ専用の学習可能な絶対位置埋込を持つ。
公開実装では位置情報の重複加算を避けるため、GPT-2内蔵のposition embeddingを無効化する。
ローカルT2S実装でも、公式と同じくGPT内蔵位置埋め込みを使わず、本文・音声の位置埋め込みを別に加算する。

### 4.4 GPT-2 decoder layerの構造

```text
hidden [B,S,D] ───────────────────────────────┐
  → LayerNorm                               │
  → Q,K,V projections [B,H,S,D/H]            │
  → causal Self-Attention (過去keyのみ)        │
  → output projection [B,S,D]                │
  → residual add ◀──────────────────────────┘
       │                                    ┐
       → LayerNorm                          │
       → Linear(D,4D) → GELU → Linear(4D,D) │
       → residual add ◀─────────────────────┘
```

Hはattention head数。公開T2SはGPT-2系であり、
VoxCPM2のMiniCPM/RMSNorm/SwiGLU/FSQと同じblockではない。
本版は公式同様の`GPT2Model(config)`を利用し、重みをランダム初期化する。

## 5. 多言語戦略の比較と新規性

| 戦略 | 入力の変化 | 効果 | 制約 |
| --- | --- | --- | --- |
| Boundary-aware | `<ZH> text </ZH>`等で区間を囲む | utteranceの言語を明示 | 長文では境界から遠い位置の制御が弱まり得る |
| Token-level | 各text tokenにlanguage categoryを付与 | 日中の漢字など同形文字を区別 | token別language ID付与frontendが必要 |
| Instruction-guided | instruction + EOP + text | 自然言語指示で目的言語を指定 | prompt設計依存、code-switch時の冗長性 |

論文の同条件100K step比較では、token-levelがZH/EN/ESのWERと全言語のSSで良好。
日本語WERではinstruction-guidedが良い。

| 比較 | Boundary | Token-level | Instruction |
| --- | ---: | ---: | ---: |
| SeedTTS ZH WER (%) | 1.602 | 1.119 | 1.613 |
| SeedTTS EN WER (%) | 3.577 | 3.253 | 3.314 |
| Index JA WER (%) | 10.444 | 10.498 | 9.226 |
| Index ES WER (%) | 5.832 | 5.200 | 5.399 |

この比較では、対応言語数に加え、同じ文字に対する言語ごとの発音を区別する方法を検討している。
一方、code-switching全般が解決済みという結果ではない。

## 6. S2M：semantic特徴を条件とするMel生成

### 6.1 semantic特徴とMelの時間軸の整列

```text
codec-decoded semantic [B,T50,1024]
  → Linear(1024,C)
  → nearest interpolation to Mel length M
  → Conv(k=3) → GroupNorm → Mish （複数段）
  → final Conv(k=1)
  → aligned semantic condition [B,M,C]
```

ここでの補間は、25/50Hzの条件と高頻度Melを整列する公式length regulatorの処理。
補間だけで音響情報を完成させるのではなく、後段CFMが連続Melを生成する。
batch入力では公式regulatorと同じく入力時間軸を最大Mel長へ補間し、その後に発話ごとの有効長maskを適用する。
元の独自実装にあった発話ごとの切り出し・補間は除去した。

公開推論のtarget Mel長は、codec復元後のframe数に対して
`int(T50 * 1.72 * duration_factor)` で求める。
25HzのID数へ直接1.72を掛けるのではなく、decode後の約50Hz長が対象。
推論時はこの規則で生成長を決め、生成対象の正解Mel長は使用しない。

### 6.2 公開DiT経路の入力融合

```text
noisy Mel    [B,F,R+M] ─── transpose ── [B,R+M,F] ─────────────┐
prompt Mel   [B,F,R+M] ─── transpose ── [B,R+M,F] ─────────────┤
semantic     [B,R+M,C] ─── Linear(C,Dm) ─ [B,R+M,Dm] ─────────┤ concat
speaker      [B,192] ───── 時間方向broadcast [B,R+M,192] ───────┘
                                                                │
                     Linear(2F+Dm+192,Dm)                        ▼
                                                     hidden [B,R+M,Dm]
time t [B] → sinusoidal(1000*t) → MLP → [B,Dm] ── adaptive norm条件
```

prompt Mel channelは参照区間だけclean Mel、target区間はゼロ。
noisy Melの参照区間はゼロに保つ。参照Melと生成中のMelは別の入力channelで扱う。

### 6.3 時刻依存Transformer

```text
hidden [B,R+M,Dm]
  → AdaptiveRMSNorm(time): scale(t)*RMSNorm(x)+shift(t)
  → full Self-Attention + RoPE + residual
  → AdaptiveRMSNorm(time)
  → SwiGLU FFN + residual
  → repeat blocks（前半のhiddenを後半へU-shaped skip）
  → final AdaptiveRMSNorm
  → concat(hidden, original noisy Mel) → long-skip Linear
  → MLP(Dm→Dm→F)
  → velocity [B,F,R+M]
```

図はcontinuous semantic、speaker channel concat、time adaptive norm、MLP outputの経路を示す。
本実装はローカルの`SemanticToMelNetwork`を使い、time token、style token、WaveNet output等も公式configの選択を保持する。
Zipformerのblock数・特徴次元数・downsampling scheduleは論文本文と手元コードから確定できない。
そのため、ここでは内部構造を確認できる公開DiT版を図示する。

## 7. 学習目的：入力・予測対象・正解データ・損失関数

### 7.1 codecの再構成学習

入力とGTは同じ正規化済みW2v-BERT特徴。
encoder→量子化→decoderで復元し、離散codebookを介して元の特徴を再構成できるよう学習する。

```text
f [B,T50,1024]
  → codec → f_hat [B,T50,1024]
L_reconstruction = mean((f_hat-f)^2)

e [B,Q,U] = projected encoder features
q [B,Q,U] = selected codebook vectors
L_commit   = 0.15 * mean((e-stopgrad(q))^2)
L_codebook =        mean((q-stopgrad(e))^2)
L_codec = L_reconstruction + L_commit + L_codebook
```

量子化のargmaxは微分できないため、encoderへの勾配計算にstraight-through estimatorを使う。
T2Sを学習する段階ではcodecを固定し、同じ音声に対するIDの意味を安定させる。

### 7.2 T2Sの次token予測

入力は本文IDs＋speaker/emotion条件＋正解semantic列の1token前まで。
予測は次のsemantic tokenのlogits、GTは固定codecから作ったIDとEOS。

```text
audio input:  BA q0 q1 ... q(U-1)
audio GT:     q0 q1 q2 ... EA
logits:       [B,U+1,8194]
L_audio = CE(logits, GT), padding位置は除外
```

本版の学習例は音声semantic tokenのCEのみを計算する。
本文tokenのNTP損失は実装しておらず、`text_loss_weight`は0のみ受け付ける。
公開`UnifiedVoice.forward`は`return_latent=True`でhiddenを返す経路であり、
この経路の戻り値は隠れ状態であり、分類headが出力するlogitsとは異なる。

### 7.3 S2MのFlow Matching損失

入力はノイズ付きGT Mel、時刻t、semantic条件、参照prefix Mel、speaker。
予測はMelそのものではなくvelocity。
GTはGT Melとサンプルしたnoiseから解析的に作るvelocity。

```text
m1    = GT Mel [B,F,R+M]
eps   = N(0,I) [B,F,R+M]
t     = Uniform(0,1) [B]
sigma = 1e-6

x_t   = (1-(1-sigma)*t)*eps + t*m1
v_GT  = m1 - (1-sigma)*eps
v_hat = DiT(x_t, reference_Mel, semantic, speaker, t)

valid target = prompt_length <= frame < mel_length
L_CFM = 各発話のvalid target上のMSEを求め、batch平均
```

参照部分はloss対象外。paddingをGTの無音と誤認して学習しない。
学習時に長いODE chainを逆伝播する必要はなく、1回サンプルしたtでvelocityを学ぶ。
条件dropoutを使い、推論でunconditional fieldも計算できるようにする。

### 7.4 GRPO

1本文に対して4候補を生成し、凍結ASRによるWERとspeaker similarityでrewardを付ける。
candidate間で相対的に良い出力の生成確率を上げる。
この最適化の対象はT2Sであり、ASRの認識結果をsemantic token GTに置換する処理ではない。
[training/grpo.py](training/grpo.py) は実rewardからGRPO objectiveを計算する。
clip幅・KL係数など非公開の細目は利用者設定で、論文の正確な値を再現したとはしない。

## 8. 推論：T2Sの逐次生成とS2Mの反復生成

まずT2Sがsemantic ID列を最後まで生成し、その後S2MがMelを生成する。
2つの反復処理はこの順で実行され、T2Sの各token生成時にS2Mを呼ぶ構成ではない。

```text
① T2S：次のsemantic IDを自己回帰生成
  q0 ~ P(q|conditions,text,BA)
  q1 ~ P(q|conditions,text,BA,q0)
  ... EOSまで
        │ semantic ID列が完成してからS2Mへ渡す
        ▼
② S2M：ODE更新を反復し、全Mel時間位置を同時に生成
  x0 = noise
  for t=0 ... 1:
      v_cond   = DiT(x,t,semantic,reference,speaker)
      v_uncond = DiT(x,t,0,0,0)
      v = (1+cfg)*v_cond - cfg*v_uncond
      x = x + dt*v
  target Mel → BigVGAN
```

公開推論は25 CFM step、`inference_cfg_rate=0.7`の経路を持つ。
これは標準CFGのscaleなら1.7に対応する。
VoxCPM2とはCFG係数の定義が異なるため、係数値を比較する際には合成式も確認する必要がある。
T2S生成は公式inference_speechを使用し、HFのKV cache・sampling・beam search・penaltyを適用する。
CUDA graphや別serving engineの速度最適化は、この学習用pipelineには追加していない。

## 9. データと論文結果

音声分離・VAD・ASR・話者分離・同一話者segmentのmerge・品質filterを行う。
merge後の発話は最大25秒。
G2Pは曖昧な読みを無条件に置換せず、英語CMU辞書や日本語の文脈依存読みに注意する。

論文の学習量には内部不整合がある。
本文は約100K時間と書く一方、列挙する30K ZH＋25K EN＋42K JA＋22.9K ES＋20K ARは139.9K時間。
本資料では本文の総量と内訳の合計を併記し、原文の不整合を明示する。
emotionalデータは135時間で、情動付き教師は主にZH/EN。
未学習言語のemotion transferを別評価している。

| 条件 | 論文の代表結果 |
| --- | --- |
| T2S RTF：IndexTTS2→2.5 | 0.232→0.119 |
| S2M RTF：U-DiT→Zipformer | 0.081→0.017 |
| 上記速度測定hardware | NVIDIA A10 / Xeon Platinum 8350C |
| Zipformer対U-DiT preference | 56%対40%、同率4% |
| CV3 emotion ZH accuracy | 0.713（比較CosyVoice3 0.467） |
| CV3 emotion EN accuracy | 0.673（比較CosyVoice3 0.567） |

RTFは生成時間/音声時間で小さいほど速い。
S2Mだけの0.017をTTS全体のRTFと混同しない。
論文のZipformer測定値は、本ディレクトリのDiT実装の測定値ではない。

## 10. ファイル構成とscratch学習の境界

### 10.1 モデル本体と公式実装の対応

モデルの基準は参照revisionの**公式コード**である。T2S、semantic codec、
DiT-S2Melのネットワークは、このディレクトリ内に実装している。
学習・推論経路から公式`indextts`のモデルクラスをimportしない。
各接続APIの`core`属性がローカル実装の本体を保持する。
外部の学習済みW2v-BERTなどは固定特徴抽出器として読み込む。

| 提供API | ローカル実装 | 主な処理 |
| --- | --- | --- |
| `TextToSemantic` | `models/t2s/` | GPT-2、条件射影、本文・音声位置埋め込み、KV cache、beam sampling |
| `SemanticCodec` | `models/codec/` | ConvNeXt/Vocos、FVQ、時間圧縮と特徴復元 |
| `S2Mel` | `models/s2mel/` | length regulator、DiT、CFM loss、Euler積分、CFG |

公式との数値比較はまだ実行していない。以下の比較用スクリプトと、
実行せずに確認したソース構造を区別する。

### 10.2 公式構成と小型variantの設定

`config_*_example.json`の既定は、公式config.yamlから該当section全体を読む。
語彙・dropout・条件付け・mask等を独自の既定値で置換しない。

```json
{
  "official_config_path": "checkpoints/IndexTTS-2.5/config.yaml"
}
```

小型化は`config_t2s_small.json`等の明示的variantで選択する。
例えば次は公式T2Sと同じ演算で、特徴次元数・層数・head数だけを変更する。

```json
{
  "official_config_path": "checkpoints/IndexTTS-2.5/config.yaml",
  "variant": {"layers": 8, "model_dim": 512, "heads": 8}
}
```

変更可能な容量fieldは[models/config.py](models/config.py)で限定する。
小型化による精度変化と、公式とのロジックの差分を混同しない。
保存checkpointには解決済みの全設定を格納するので、再構築時に元のYAMLを必要としない。

### 10.3 同一入力・重みによる比較

[verification/verify_official.py](verification/verify_official.py)は、参照側にも**公式クラスの実体**を別に構築し、
`strict=True`で同じ重みをコピーして、codecの各出力・T2S logits・regulator・
DiT velocity・CFM Melを比較する。自作の参照式やmock依存への置換は使用しない。
T2S logitsの参照には公式`GPT2InferenceModel.forward`を使用する。
同じsampling設定でも乱数によって生成列は変わるため、logitsも比較対象に含める。

**この比較はまだ実行していない。** 計算資源を使用しない指示に従い、今回は
ソースの読み取り・修正・構文の静的確認のみを行った。数値一致を実測済みとは記載しない。
後で実行できる時点で、利用者が明示的に`--run`を指定する。

```bash
# 実行例。今回は未実行。既定では両側を同じ小型variantで比較する。
python -m verification.verify_official --run --model-dir checkpoints/IndexTTS-2.5 --report outputs/official_comparison.json
# 公式容量の比較は上記へ--full-sizeを追加する（より多くの資源が必要）。
```

### 10.4 ディレクトリと入口

```text
IndexTTS2.5/
├── README.md                 # 論文・公式実装のsurvey
├── main_flow.py              # 本文・参照音声から合成までのAPI
├── training_example.py       # 学習の入口
├── prepare_assets.py         # 固定重み・辞書の準備
├── prepare_data.py           # 音声と本文から学習用cacheを作成
├── pretrained_inference.py   # 公開重みによる合成の入口
├── configs/                  # 公式容量・小型variantの設定JSON
├── models/
│   ├── t2s/                  # GPT-2・条件付け・自己回帰生成
│   ├── codec/                # semantic codec・FVQ
│   ├── s2mel/                # regulator・DiT・flow matching
│   ├── emotion/              # Conformer・Perceiver
│   ├── conditioning.py       # 固定特徴抽出器との接続
│   └── config.py             # 完全設定と容量variantの解決
├── pretrained/
│   ├── campplus/             # 話者特徴ネットワーク
│   └── bigvgan/              # Melから波形へのvocoder
├── data_processing/          # tokenizer・正規化・Mel・資産読み込み
├── training/grpo.py          # GRPO loss
└── verification/
    └── verify_official.py    # 公式コードとの比較専用（未実行）
```

ルートのスクリプトをこのディレクトリを作業ディレクトリとして実行する。
ネットワーク定義・前処理・固定重み用ネットワークを役割別に分離した。
公式コードをimportするのは、利用者が明示的に実行する比較専用スクリプトの
参照側だけである。

scratch学習できるのはここで定義したcodec・T2S backbone・DiT-S2M。
W2v-BERT、CAMPPlus、emotion frontend、BigVGANには固定の公開重みを使用する。
W2v-BERTは外部の学習済みモデルを読み込む。残りのネットワーク定義は
`pretrained/`と`models/emotion/`に置いている。
これらを小さなLinearやランダム波形に置換して全システム再現を装ってはいない。
emotion cacheは公式emotion Conformer＋Perceiverで得る1024次元vector。
T2S本体を学習してもcache元のemotion frontendは更新されない。
frontendも学習する場合は実モジュールを組み込み、cache境界を外す必要がある。

## 11. DataFrameから学習する

### 生音声と書き起こしからの準備

#### 固定する学習済みモデルの準備

音声と本文以外に追加の教師ラベルは必要ないが、特徴抽出器とvocoderの
学習済み重みは必要になる。各重みの由来と、学習中の扱いは次のとおり。

| 構成要素 | 重みの配布元・保存先 | 使用する処理 | この学習例での扱い |
| --- | --- | --- | --- |
| W2v-BERT 2.0 | `facebook/w2v-bert-2.0` → `hf_cache/w2v-bert-2.0` | semantic特徴抽出 | 固定した外部学習済みモデル |
| CAMPPlus | `funasr/campplus`の`campplus_cn_common.bin` → `hf_cache` | 話者embedding抽出 | 固定した外部学習済みモデル |
| BigVGAN | `nvidia/bigvgan_v2_22khz_80band_256x` → `hf_cache/bigvgan` | 合成Melから波形生成 | 固定した外部学習済みモデル。codec/T2S/S2Melのoptimizer対象外 |
| 感情Conformer/Perceiver | `IndexTeam/IndexTTS-2.5`のGPT checkpoint | 感情embedding抽出 | IndexTTS自身の学習済み部分を固定して再利用 |
| 正規化統計・tokenizer・構成 | 同じIndexTTS-2.5配布資産 | 特徴正規化・本文ID化・構築 | 学習と推論で同じものを使用 |

感情embeddingの抽出に、別の外部感情分類器や感情ラベルを使っているわけではない。
前処理では固定特徴抽出器とvocoderをローカルの資産読み込み処理で構築する。
感情encoderの重みを抽出するためGPT checkpointも必要だが、生成GPTは構築しない。

必要な依存ライブラリを用意した後、次の手順で資産を取得・検査する。
`--download`は利用者が実際に取得する場合だけ指定する。本作業では取得していない。

```bash
python prepare_assets.py --model-dir checkpoints/IndexTTS-2.5 --download
python prepare_assets.py --model-dir checkpoints/IndexTTS-2.5
```

再現実験では`--index-revision`、`--w2v-revision`、`--campplus-revision`、
`--bigvgan-revision`で使用するcommitを固定する。
`--hash-manifest outputs/asset_hashes.json`を指定すると資産のSHA256を保存できる。
スクリプトの既定動作はローカル検査だけで、モデルを構築せず不足pathを報告する。

`training_example.py`からもraw表を直接指定できる。
これにより、資産検査→音声前処理→指定stageの学習が順に実行される。

```bash
python training_example.py --raw-metadata data/raw.parquet --stage codec \
  --model-dir checkpoints/IndexTTS-2.5 --cache-dir data/codec_cache \
  --language JA --config configs/config_codec_example.json --output checkpoints/codec.pt
python training_example.py --raw-metadata data/raw.parquet --stage t2s \
  --model-dir checkpoints/IndexTTS-2.5 --cache-dir data/t2s_cache \
  --language JA --codec-checkpoint checkpoints/codec.pt --output checkpoints/t2s.pt
```

S2Melも`--stage s2mel`で同様に学習する。前処理済みデータの再利用には
既存の`--metadata`入口を使う。以下は前処理を独立して実行する場合の説明である。

中間メタデータは**Parquet（pyarrow、ZSTD圧縮）**で保存する。
embedding・latent・IDなどの数値配列は**Tensor単体の`.pt`**へ保存し、
dtype/shapeを保持する。CSVへ配列を文字列化して格納しない。
元の音声path・本文表はCSV/Parquetの両方から読める。`pyarrow`が必要だが、
自動インストールやCSVへのfallbackは行わない。

**利用者が用意する表の必須列は`audio_path`と`text`だけ。**
`speaker_id`と感情などの追加メタデータは任意で、存在しなくてよい。
下表の`*_path`列は[prepare_data.py](prepare_data.py)が生成する内部cacheであり、
最初から揃っていることを前提にしない。

```csv
audio_path,text
audio/0001.wav,こんにちは。
audio/0002.wav,今日は晴れています。
```

音声pathはこのCSVの親基準。各行は音声全体と書き起こしが対応する1発話とする。
前処理はmono化・resampleを行うが、音声だけを勝手にcropしない。
長さ上限（既定30秒）を超える行は、音声と本文を対応させて分割してから渡す。
言語は表に追加せず、単一言語datasetなら`--language JA`などで指定できる。
多言語datasetは言語別に前処理し、`preparation.json`の共通language mapを保って統合する。

```text
audio_path + text（必須）
    ├─16kHz mono → W2v-BERT hidden[17]・固定統計 → features[T50,1024]
    │                      └─学習済みscratch codec → codes[U] / decoded[2U,1024]
    ├─16kHz mono → 80bin fbank → CAMPPlus → speaker[192]
    ├─W2v-BERT特徴 → 公式感情Conformer + Perceiver → emotion[1024]
    ├─公式Mel設定のsample rateへresample → mel[F,M]
    └─本文cleaning・言語別正規化・日本語G2P・tokenizer → text_ids[L]
```

`T50`は約50Hz frame数、`U`は約25Hz code数、`F`はMel bin数、`M`はMel長、
`L`は本文token数。感情embeddingは**音声から計算する表現**で、感情ラベルを要求しない。
話者IDなしでも各発話からCAMPPlusの声質特徴を抽出できる。
この前処理では同じ発話の音声からglobalな声質・感情条件を抽出する。
別発話referenceや感情指示との対応を学習するデータ作りとは区別する。

実行順は次のとおり。ローカルのIndexTTS2.5資産を固定特徴抽出に使う。
対象datasetに追加ラベルを用意する必要はない。
前処理では生成GPT・S2Melを構築せず、音声生成も行わない。
必要資産はスクリプト冒頭に列挙し、自動download/compileは無効にしている。

```bash
# 1. raw音声からcodec学習用の正規化semantic特徴を抽出
python prepare_data.py --metadata data/raw.csv --language JA --stage codec --model-dir checkpoints/IndexTTS-2.5 --output data/codec_cache
# 2. scratch codecを学習（音声→IDの定義を先に固定する）
python training_example.py --stage codec --metadata data/codec_cache/codec.parquet --config configs/config_codec_example.json --output checkpoints/codec.pt
# 3. その学習済みcodecでGT ID/復元特徴を作り、他の特徴も音声から抽出
python prepare_data.py --metadata data/raw.csv --language JA --stage all --model-dir checkpoints/IndexTTS-2.5 --codec-checkpoint checkpoints/codec.pt --output data/tts_cache
# 4. 実tokenizer語彙と実Mel bin数を反映した生成configで各段を学習
python training_example.py --stage t2s --metadata data/tts_cache/t2s.parquet --config data/tts_cache/config_t2s.json --output checkpoints/t2s.pt
python training_example.py --stage s2mel --metadata data/tts_cache/s2mel.parquet --config data/tts_cache/config_s2mel.json --output checkpoints/s2mel.pt
```

生成Parquetはメタデータファイルの親基準の相対pathを持ち、学習CLIの`load_metadata()`が解決する。
Python APIでも`load_metadata("data/tts_cache/t2s.parquet")`を`train()`へ渡せる。
生成configは公式gpt/s2mel section全体を保存し、既定で小型化しない。
容量を下げる場合だけ、後述のvariantを明示する。
前処理後にcodecを更新した場合はIDと復元特徴を再抽出する。
`s2mel.parquet`には生成時の表現に合わせてcodec復元特徴を保存する。
既存outputは上書きしない。自動train/dev分割はせず、必要なら**raw表を先に分割**する。
これらのモデル実行は、利用者の指示に従い本作業では行っていない。

### 前処理後の学習入力

各pathは利用者のdataset内の相対pathでもよく、この資料にマシン固有pathは不要。
`.pt`には下表のtensorだけを保存する。

| stage | 必須DataFrame列 | tensor内容 |
| --- | --- | --- |
| codec | `features_path` | 正規化済み `[T50,1024]` |
| t2s | `text_ids_path` | 本文のみ `[L]` int64 |
| t2s | `codes_path` | 固定codecで抽出した `[U]` int64 |
| t2s | `speaker_path` | CAMPPlus `[192]` float |
| t2s | `emotion_path` | emotion frontend `[1024]` float |
| t2s | `language_id` | 使用language辞書と一致する整数 |
| s2mel | `features_path` | codec復元を学習条件に使う場合はその `[T50,1024]` |
| s2mel | `mel_path` | 対応する正解Mel `[F,M]` |
| s2mel | `speaker_path` | speaker条件 `[192]` |

T2Sの開始/終了tokenはtrainerが付けるので、本文cacheへ二重に追加しない。
codec学習後に固定してcodesを作り、そのcodesでT2Sを学習する。
S2MではGT Mel長が学習時に既知なのでlength regulatorの教師長に使える。
推論では本文から生成したcodesと長さ規則を使うため、train/inferenceの長さ情報は異なる。

```python
import json
from training_example import load_metadata, train

metadata = load_metadata("data/tts_cache/t2s.parquet")
with open("data/tts_cache/config_t2s.json", encoding="utf-8") as handle:
    config = json.load(handle)
model = train(metadata, "t2s", config, "checkpoints/t2s.pt")
```

ここでは公式の全設定からモデルを構築する。容量を変更する場合もモデルの演算構造は変更しない。
容量、語彙、audio special ID、feature正規化を後で変えるとcheckpoint互換性も変わる。
各stageのcheckpointにはconfigとoptimizer状態を保存する。

CLI用の3つのconfig例は、公式config.yamlから該当section全体を読み込む。
言語IDは公式のlang_to_tokenを使用し、独自の0～5への再割り当ては行わない。

```bash
python training_example.py --stage codec --metadata data/codec.parquet --config configs/config_codec_example.json
python training_example.py --stage t2s --metadata data/t2s.parquet --config configs/config_t2s_example.json
python training_example.py --stage s2mel --metadata data/s2mel.parquet --config configs/config_s2mel_example.json
```

これらは利用者が後で実行するための例で、この資料の作成中には実行していない。
Python 3.10以降、PyTorch、einops、Transformers、pandasを使う。
raw音声のfrontend/vocoderには公式repository側の依存も必要。
ここではライブラリの追加インストールも行っていない。

### 11.1 scratch学習したモデルで音声を合成する

[main_flow.py](main_flow.py)は**本文と話者参照音声を入力**として受け取る。
話者embedding、感情embedding、本文ID、参照Melを利用者が作る必要はない。
内部では[data_processing/frontend.py](data_processing/frontend.py)を呼び、学習データの準備と同じ方法で計算する。
感情参照音声は任意で、省略時は話者参照音声から感情embeddingも抽出する。
ゼロvectorや固定の中立ラベルを代入する処理ではない。

```text
本文 ── 正規化・G2P・tokenizer ── 本文ID
話者参照音声 ── W2v-BERT ─────── 参照semantic特徴
             ├─ CAMPPlus ─────── 話者embedding
             └─ Mel計算 ──────── 参照Mel
感情参照音声（任意）
  └─ 指定あり：別音声のW2v-BERT特徴 ─┐
     指定なし：話者参照特徴を再利用 ─┴─ Conformer/Perceiver ─ 感情embedding
                                      │
                  本文・話者・感情条件 ── T2S → codec/S2M → vocoder
```

`--model-dir`はデータ準備に使ったものと同じ公式資産を指定する。
ここから固定抽出器とvocoderを読み、codec・T2S・S2Melは自分で学習した
checkpointを読み込む。したがって、本文と参照の入力形式は公式推論と同様でも、
合成に使用する生成モデルの重みはscratch学習したものである。

```bash
python main_flow.py --model-dir checkpoints/IndexTTS-2.5 \
  --codec-checkpoint checkpoints/codec.pt \
  --t2s-checkpoint checkpoints/t2s.pt \
  --s2mel-checkpoint checkpoints/s2mel.pt \
  --text 'こんにちは。' --reference data/speaker.wav \
  --language JA --output outputs/scratch.wav
```

別の感情参照を使用する場合だけ`--emotion-reference data/emotion.wav`を追加する。
本文の言語は、前処理と共通のZH/EN/JA/ES/ARから指定する。
Pythonでは`IndexTTSPipeline.from_checkpoints(...)`で構築し、
`result = pipeline("こんにちは。", "data/speaker.wav", language="JA")`で呼び出せる。
`result`は生成ID、Mel、波形のTensorを保持する辞書である。
条件を事前計算する研究用途には`generate_from_features()`を用意しているが、
通常の`forward()`では感情embeddingを外部入力として要求しない。

### 11.2 公式pretrainedモデルで推論する

[pretrained_inference.py](pretrained_inference.py)は、本ディレクトリのモデル実装に
公式`IndexTeam/IndexTTS-2.5`の公開重みを読み込み、WAVを保存する。
`--model-dir`に事前準備済みのモデルディレクトリを指定する。

```bash
python pretrained_inference.py \
  --model-dir checkpoints/IndexTTS-2.5 \
  --reference data/speaker.wav --language JA \
  --text 'こんにちは。音声合成のサンプルです。' \
  --output outputs/index.wav
```

別の感情参照は`--emotion-reference data/emotion.wav`で指定する。
省略すると話者参照音声から感情特徴も抽出する。
GPUは`--device cuda:0`、既存出力の置換は`--overwrite`で指定する。
このスクリプトは依存ライブラリをインストールしない。今回も実行していない。

## 12. 利点と制約

- 25Hz化はAR step数を減らすが、codecの表現能力とのtrade-offを持つ。
- semantic tokenだけでは細かな音響を完全保存できず、referenceとS2Mが重要。
- 話者とemotionを別参照にしても数学的な完全独立を保証するわけではない。
- 低WER、speaker similarity、emotion similarity、MOSは別の評価軸。
- G2Pやlanguage指定は発音の制御を助けるが、誤認識・誤タグは逆効果になる。
- 全体が単一のcausal waveform generatorではなく、S2Mは全Mel時間位置を参照する。
- 外部componentのsample rate・正規化・Mel仕様はモデル本体と同じくらい重要。

## 13. 確認状態

**現時点の提供物は、論文と同等の精度を再現する完全な学習recipeではない。**
固定済み外部モデルの再利用自体はモデル構成の一部であるが、それだけで
以下の差分が解消されるわけではない。

| 項目 | 現在の提供範囲 | 公式精度の再現に対する制約 |
| --- | --- | --- |
| モデル容量 | 公式configが既定。小型化は明示variantのみ | 容量を変える実験では、公式容量とは区別して評価する |
| モデル本体 | ローカルT2S・codec・S2Melをrandom構築 | 公式の演算と構成を基準に実装。出力の実測比較は未実行 |
| S2M | 公式公開コードのDiT版 | 公式コードを基準とする。論文の別backboneへの置換は行わない |
| 学習loop | 単一processの基本AdamW loop | 今回の厳密一致の対象外 |
| データ | 利用者の音声/text表を前処理 | 学習データ量・品質が異なるため精度は別途評価が必要 |
| 感情frontend/vocoder | 公式学習済み重みを固定して使用 | 学習用前処理と推論で同じ重みを使用する |
| GRPO | 実測rewardを受け取るloss部品 | 今回の厳密一致の対象外。全更新loopは未実装 |

これらを実装済み・性能再現済みとは扱わない。未公開情報についても、推測した
学習設定を公式recipeとして提示しない。

利用者の指定により、モデルの実行・学習・推論・benchmarkは実施していない。
実装時にはTeXと公開ソースを読み、shape・教師shift・mask・CFM時刻方向を静的に照合した。
公式実装との出力の数値比較は未実施である。

## 14. 一次資料

- [IndexTTS 2.5 Technical Report](https://arxiv.org/abs/2601.03888)
- [公式repository](https://github.com/index-tts/index-tts)
- [公開T2S](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/gpt/model_v2.py)
- [公開codec](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/codec/models.py)
- [公開CFM](https://github.com/index-tts/index-tts/blob/d9e41aac89fd00b3d71497fddb287b7f24613712/indextts/s2mel/modules/flow_matching.py)

TeX中のコメントアウトされた旧結果と`deleted`の改稿箇所を、現行本文の結果に混ぜない。
コード・モデルの利用条件は公式repositoryとcheckpointそれぞれのlicenseを確認する。
