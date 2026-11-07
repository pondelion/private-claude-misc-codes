# Fish Audio S2 — Dual-ARと音声codecのアーキテクチャサーベイ

## 1. 概要と対象バージョン

Fish Audio S2は、**音声の時刻方向とcodebook深さ方向を別の自己回帰Transformerで生成するTTS**。
大きなSlow ARが発話内容と大域的prosodyを計画し、小さなFast ARが同時刻の細かな音響情報を補う。
生成するのはMelや連続latentではなく、音声codecの離散codebook IDs。
codec decoderがそのIDsから44.1kHz波形を生成する。
合成処理は **①Slow ARによる次frameのsemantic ID生成、②Fast ARによる
同frameの残りのcodebook ID生成、③codec decoderによる波形復元**に分かれる。

本資料の対象は、提供されたTeXの **Fish Audio S2 Technical Report** とS2-Proの公開経路。
同じrepositoryには後発のDrama 3 / `fish_omni`も存在するが、
それらのMoE・hybrid attention・audio towerは、本資料のS2の説明対象には含めない。
参照revisionは `0f8a12e8a651062fdee9c33a894bdccd3a8513f1`。

論文のモデルはQwen3-4BをSlow AR、4層TransformerをFast ARとして使う。
codecは10codebookのRVQで、最初の1つがsemantic、残り9つがacoustic residual。
textとreferenceを文脈として入れ、自然言語の局所指示・複数話者・複数turnを扱う。

### 1.1 既存手法との構造・学習方法の差分

| 比較対象 | 差分 | 効果と代償 |
| --- | --- | --- |
| 全codebookを時刻方向にflattenするAR | 時刻方向Slowと深さ方向Fastを分離 | 大きなLMの系列長を約1/10へ。Fastの逐次生成は残る |
| semantic-only TTS | 全10bookの情報を次時刻Slow入力へ融合 | acoustic historyを継承。embedding計算とFastが追加 |
| 非causal codec | causal CNN＋window attentionでcodecを構成 | streaming可能。context/state管理が必要 |
| utterance全体のstyle指定のみ | 単語・句位置に自然言語指示を挿入 | 局所prosodyを制御。対応したannotationが必要 |
| pretrainingのみ | SFT＋group rewardによるpost-training | 発音抜け・話者drift等を改善。reward品質に依存 |

Dual-ARそのものはFish-Speech系列からの設計であり、S2で初めてARを導入したわけではない。
S2の技術的焦点は、Qwen3ベースの拡張、低rateのsemantic/acoustic codec、
自然言語制御を学ぶ大規模annotation、段階的学習とRLの組合せにある。

## 2. 表記と全体図

| 記号 | 意味 |
| --- | --- |
| B | batchの発話数 |
| L | text token数 |
| T | 音声codecの時刻frame数 |
| S | text/reference/target混在列の総時刻位置数 |
| N=10 | 各音声frameのcodebook数 |
| Ksem=4096 | semantic codebook内のID数 |
| Kac=1024 | 提供codec YAMLのacoustic codebook内のID数 |
| Ds / Df | Slow AR / Fast ARの隠れ状態の次元数 |
| Na | 学習batch内のloss対象audio frame総数 |
| Swav | 44.1kHz waveformのsample数 |

### 2.1 推論全体

図中の①～③は概要の3段階に対応する。参照音声のencodeとMCFは①への入力を
構築する処理である。①と②をframeごとに繰り返し、そのID列を③へ渡す。

```text
reference waveform [B,Swav_ref] / 44.1kHz
  │ audio codec encoder + RVQ
  ▼
reference IDs [B,10,Tref] ── system prompt内のreference segmentへ
                                            │
本文 + speaker tags + 局所自然言語instruction │ tokenizer
                                            ▼
                      mixed context main IDs [B,S]
                      aligned codebook IDs [B,S,10]
                                            │
                         Multi-Codebook Fusion (MCF)
                                            ▼
                                  embedding [B,S,Ds]
                                            │
                    ┌───────────────────────┴──────────────────┐
                    │ ① Slow AR: Qwen3-4B型 causal Transformer │
                    │ 時刻tまでの文脈から次frame t+1を計画      │
                    └───────────────────────┬──────────────────┘
                                            ├─ LM head → q0(t+1)をsample
                                            │             semantic ID
                                            └─ hidden h(t+1) [B,Ds]
                                                           │ projection
                                                           ▼
                    ┌────────────────────────────────────────────────┐
                    │ ② Fast AR: 4-layer causal Transformer           │
                    │ h + q0 → q1 → q2 → ... → q9                    │
                    │ このframe内だけcodebook深さ方向にAR             │
                    └───────────────────────┬────────────────────────┘
                                            ▼
                               full frame [q0,...,q9] [B,10]
                                            ├─ MCF → 次のSlow入力へ
                                            └─ codec decodeの入力へ
                                                      │
                   accumulated/streaming IDs [B,10,T]  │
                                                      ▼
                   ┌─────────────────────────────────────────────┐
                   │ ③ codec decoder：波形復元                   │
                   │ RVQ lookup → Transformer → upsampling       │
                   │ → 波形decoder → waveform [B,1,Swav]         │
                   └─────────────────────────────────────────────┘
                                44.1kHz mono音声
```

semanticとacousticは10本の独立した音声波形ではない。
**同じ時刻の1つの連続latentを残差的に近似する10個の離散ID**である。
各IDをlookupした特徴を加算して同時刻のlatentを復元し、その時系列から波形を生成する。

### 2.2 IndexTTS2.5・VoxCPM2との構成比較

```text
IndexTTS2.5: text → AR semantic IDs → Mel Flow Matching → BigVGAN
VoxCPM2:     text → patch AR states → local continuous-latent Flow → AudioVAE
Fish S2:     text → Slow semantic AR + Fast codebook AR → codec waveform decoder
```

Fish S2の合成本体にdiffusion denoising loopはない。
連続latentの生成をMSE velocity回帰で学ぶVoxCPM2と異なり、Dual-ARの主lossはCE。

## 3. audio tokenizer：波形から約21.53Hz・10codebookのID列への変換

### 3.1 encoder側

```text
waveform [B,1,Swav] / 44,100samples/sec
  │ causal DAC encoder
  │ strided rates [2,4,8,8]（積512）
  ▼
acoustic features [B,1024,T512]  ≈86.13frames/sec
  │ additional ConvNeXt downsampling [2,2]（積4）
  ▼
low-rate features [B,1024,T]     44,100/2048 ≈21.53frames/sec
  │ causal sliding-window Transformer
  ▼
features ready for quantization [B,1024,T]
  │ semantic VQ (1book,4096 IDs)
  ├─ semantic vector z_sem [B,1024,T]
  ├─ semantic IDs q0 [B,1,T]
  │
  │ residual = features - z_sem
  │ residual VQ (9books,各1024 IDs)
  ├─ residual vector z_res [B,1024,T]
  └─ acoustic IDs q1..q9 [B,9,T]

codes = concat_books(q0, q1..q9) [B,10,T]
z_quantized = z_sem + z_res      [B,1024,T]
```

約21Hzという丸め表現の正確なrateは44100/2048。
10秒なら約215frame、全bookをflattenすると約2150 IDs。
Slow ARでは215位置の音声表現として扱い、Fastが各位置のbookを補う。

### 3.2 RVQの「残差」

```text
input latent z
  ├─ VQ0 → nearest code e0 ──────────────────────────┐
  └─ residual r1 = z-e0                              │
       ├─ VQ1 → e1 ─────────────────────────────────┤ sum
       └─ r2 = r1-e1                                │
            ├─ VQ2 → e2 ────────────────────────────┤
            ...                                     │
            └─ VQ9 → e9 ────────────────────────────┘
                                                     ▼
                                  z_hat = e0 + e1 + ... + e9
```

各bookのID=42でも、book0とbook1では異なるembeddingを指す。
整数IDを足して1つのIDにするのではなく、lookupした連続vectorを足す。
codebook_dim=8のfactorized quantizationと1024次元のcodec latentも別の次元。

### 3.3 semantic distillation

第1bookだけからw2v-BERT 2.0の第16層activationを回帰するauxiliary headを学習する。
これによりq0へ言語・発音情報が残りやすくなる。
第1codebookに言語内容を保持させるため、semantic特徴への蒸留損失を明示的に導入している。

ただし、手元の公開`rvq.py`はsemantic predictor呼び出しがコメントアウトされている。
codec本体の公開形状と、全private training recipeが公開されているかは別問題。
本版adapterでcodecをランダム初期化するだけでは、semantic distillationを学習したことにはならない。
教師w2v-BERT、head、データ、lossの具体設定を用意する必要がある。

### 3.4 decoder側

```text
IDs [B,10,T]
  → per-book lookup / factorized output projections
  → sum semantic + residual vectors [B,1024,T]
  → causal window Transformer
  → learned upsampling ×4 [B,1024,T512]
  → waveform decoder upsampling ×512
  → waveform [B,1,Swav]
```

論文はEVA-GAN型decoderへの変更を説明する。
一方、提供`modded_dac_vq.yaml`には`overwrite_decoder`指定がなく、
公開`DAC`は既定の`Decoder`を構築する。
論文の名称だけを根拠に、YAMLで作ったdecoderをEVA-GAN完全再現と呼ばない。
使うcheckpoint・設定の対応を確認することが必要である。

## 4. Slow ARとMCF：全codebookの特徴を利用した次frame予測

### 4.1 tokenizer IDとcodec ID

```text
codec semantic ID q0 ∈ {0,...,4095}
  │ global tokenizerのsemantic専用bandへoffset
  ▼
main token ID = semantic_begin + q0

逆変換: q0 = main_token_id - semantic_begin
```

main tokenにはtext・境界・EOSも入る。
main tokenがsemantic bandにあるときだけ、対応位置のcodebook列をMCFへ加える。
`semantic_begin`の実値はcheckpoint/tokenizer設定を使う。
本版の設定例にある32000を、公式全モデル共通の固定IDだと思ってはいけない。

### 4.2 MCFの構造

```text
main_id(t) ─────── text/LM embedding ── eLM [B,Ds] ───────────┐
q0(t) ─── codebook embedding E0 ─────── e0 [B,Ds] ────────────┤
q1(t) ─── codebook embedding E1 ─────── e1 [B,Ds] ────────────┤ sum
...                                                         │
q9(t) ─── codebook embedding E9 ─────── e9 [B,Ds] ────────────┘
                                                               ▼
                                             input to next Slow step [B,Ds]
```

数式は `x = eLM(main_id) + Σ_k E_k(qk)`。
q0はeLMとE0という**独立した2種類のembedding**を通る。
誤って二重計上したのではなく、論文が明記する設計である。
全bookに同じEを使うのではなく、Slow側はbookごとの表を持つ。
実装では1つの大きな表に`qk+k*K`のoffsetで格納できる。

公開互換経路では`scale_codebook_embeddings=True`ならaudio位置を`sqrt(N+1)`で割る。
N=10なので11個の寄与に対するscale。
論文のMCF式は単純和までで、このscaleを数式へ書いていない。
本版の`scale_mcf`で明示する。

text位置ではcodebook和をゼロにし、通常のtext embeddingだけを使う。
codebook ID=0は正常な音声symbolなので、それ自体をpadding判定に使わない。

### 4.3 Slow Transformer block

```text
hidden [B,S,Ds] ──────────────────────────────────────────┐
  → RMSNorm                                              │
  → QKV projections                                      │
     Q [B,H,S,d] / K,V [B,G,S,d]                         │
  → per-head Q/K RMSNorm（configurationで指定）             │
  → RoPE                                                 │
  → causal GQA: 同時刻までのpast context                    │
  → output projection [B,S,Ds]                            │
  → residual add ◀───────────────────────────────────────┘
       │                                                 ┐
       → RMSNorm                                         │
       → SiLU(Wgate*x) * Wup*x [B,S,I]                   │
       → Wdown [B,S,Ds]                                  │
       → residual add ◀──────────────────────────────────┘
```

最終norm後のhiddenからtext vocabulary全体のlogitsを作る。
音声生成中はsemantic bandとEOSだけをsample可能に制限する。
自然言語instructionに対応する通常text tokenを音声frameとしてsampleするわけではない。

Fastへのhiddenは公開`norm_fastlayer_input`に応じてnorm前後が切り替わる。
本版は設定として保持し、logitsを作ったhiddenとFast条件の関係を明示する。

## 5. Fast AR：4層Transformerによるcodebook方向の自己回帰生成

### 5.1 1frameの入力と出力

```text
depth position     0        1        2        3       ...      9
input             W*h     Emb(q0)  Emb(q1)  Emb(q2)   ...    Emb(q8)
                   │        │        │        │                │
                   └────────┴────────┴────────┴────────────────┘
                          Fast causal Transformer ×4
                   │        │        │        │                │
predicted GT       q0       q1       q2       q3      ...       q9
inference use      discard   use      use      use              use
pretrain loss      auxiliary CE + 各acoustic CE
SFT loss           weight0=0、残りbookを重み付け
```

q0はSlowで既に決まるので、推論ではFast位置0の分類結果を採用しない。
それでも位置0のhiddenはprefixとしてFastの後続位置へ影響する。
q0 embeddingを位置1へ置き、その出力でq1を予測する。
位置kでGT qkを入力してqk自身を予測する構造ではない。

### 5.2 共有と独立

- SlowとFastは独立のTransformer重み。
- Fastの4層もそれぞれ独立で、1層を4回loop共有するわけではない。
- Fastはcodebook深さごとに同じembedding tableを使う。
- 深さの区別はRoPE位置で与える。
- SlowのMCF表はbook別で、Fast共有表とは別パラメータ。

### 5.3 時間方向とcodebook方向の自己回帰処理

```text
                     audio time t  ───────────────────────────►
                   frame0      frame1       frame2      ...
Slow computation     Slow0 ───── Slow1 ────── Slow2
                     │          │           │
book depth k         ▼          ▼           ▼
  q0               Slow q0    Slow q0     Slow q0
  q1               Fast q1    Fast q1     Fast q1
  q2               Fast q2    Fast q2     Fast q2
  ...                 │          │           │
  q9               Fast q9    Fast q9     Fast q9
                     │          │           │
                     └─MCF─────►└─MCF───────►
```

Fastのattention cacheはframeが変わるとリセットする。
Fastが前frameのq9へ直接attentionし続ける構造ではない。
時間historyはMCF→Slow→hslowを通じて伝わる。
学習時は全audio frameをNaというbatch軸へ並べ、Fast処理を並列化できる。

## 6. 学習目的：入力・予測対象・正解データ・損失関数

### 6.1 Audio tokenizerの学習

入力は44.1kHz waveform、GTはその波形とsemantic teacher。
encoder→RVQ→decoderで再構成し、VQのcodebook/commitment lossと音響lossを最適化する。
論文はMPD、multi-resolution discriminator、multi-scale STFT discriminatorのGAN frameworkを記す。
446M codecを約100万step学習したと報告する。
各discriminator構成・重み・private semantic headを含む完全recipeは手元にない。

[audio_codec.py](audio_codec.py) の`reconstruct()`は公式実体を接続して勾配を保持する。
同ファイルのSTFT lossはscratch実験で使える実装部品だが、
それだけで論文のGAN＋distillation学習を再現したとは扱わない。

### 6.2 Slow ARの教師shift

```text
input main:   ... assistant_header | sem(q0_t0) | sem(q0_t1) | sem(q0_t2)
input books:  ...       zero       |  books_t0  |  books_t1  |  books_t2
output GT:              q0_t0     |    q0_t1   |    q0_t2   | EOS
```

各Slow状態は過去の全codebookをMCF経由で受け取り、次時刻のsemanticを予測する。
本版では`main_ids[:,:-1]`と`main_ids[:,1:]`を組にする。
reference/systemは入力に残し、loss maskだけ0にする。
参照音声を入力から消すとcloning条件も失われるため、maskの役割を取り違えない。

```text
L_slow = masked_mean(CE(slow_logits[B,S-1,V], next_main_ids[B,S-1]))
```

論文はtoken別λtも書くが具体的な全値を公開していない。
本版はsupervised tokenを一様に扱う学習例。
本文・system・reference・targetそれぞれの損失対象位置は、データ構築時のmaskで指定する。

### 6.3 Fast ARのGT

Slowの「次frameを予測するhidden」と、その次frameのGT10bookを対応させる。
現在入力frameのcodesをそのままFast GTにすると、Slowとの時刻が1つずれる。

```text
hidden = Slow(input until frame t-1)
target_codes = GT frame t, [Na,10]
fast_input = concat(project(hidden), Emb(target_codes[:,:-1]))
fast_logits [Na,10,K]
L_fast = mean_audio_frames( sum_k(wk * CEk) / sum_k(wk) )
L_total = L_slow + L_fast （本版の既定係数）
```

pretrainingではq0もauxiliary supervisionに含める。
SFTではq0の重みを0にし、残りを減衰重みで扱う。
具体的な減衰率は論文に固定値がないため、`codebook_weights`で利用者が指定する。
学習codeは重みを0にしたbookのCE計算も実際に行い、寄与だけ重みで除く。

論文Fast loss式は分子にk=0を含めつつ分母がk=1..N-1と書かれている。
本版は意図が明確な「有効book重み和による平均」を採用し、その規約をコードに明示する。
この正規化は論文式と異なるため、実装上の差分として区別する。

### 6.4 Dual-AR学習と波形再構成損失の関係

Dual-ARのCE学習中は、固定codecから作ったID列がGT。
各stepでcodec decode→波形MSEを必ず計算する構造ではない。
一方、RLでは合成音声をASR/quality/speakerモデルへ渡しrewardを作る。
codecの事前学習、Dual-ARのCE学習、RLの音声評価は別stage。

## 7. data pipelineと4段階学習

### 7.1 制御条件の学習に必要なアノテーション

```text
raw recording
  → source separation + VAD
  → speech quality filter
  → rich-transcription ASR
       transcript / speaker turns / local vocal instructions
  → text/audio pairs for pretraining/SFT
```

quality modelはw2v-BERT 2.0＋MLPで、SNR・話者一貫性・録音品質・明瞭性を判定する。
rich-transcription ASRはQwen3-Omni-30B-A3Bをfine-tuneしたもの。
`<|speaker:0|>`や`[angry]`、`[inhale]`、`[emphasis]`等をtext列へ入れる。
角括弧の自然言語指示をすべて専用の単一tokenとして登録する必要はない。
instructionの語列と局所音響との対応をデータから学ぶ。

### 7.2 training stages

| stage | 対象 | 役割 |
| --- | --- | --- |
| audio tokenizer | codec | 波形再構成、semantic-distilled第1book |
| pretraining | Slow/Fast | 1000万時間超、約80言語・方言、text/audio alignment |
| SFT | Slow/Fast | 高品質・表現・細かいcontrolを強化 |
| RL post-training | Slow/Fast | intelligibility、quality、speaker保持を改善 |

pretrainingは8,192→16,384 contextへ段階的に拡張する。
new semantic/control embeddingは既存text embeddingの平均・共分散に基づく分布で初期化する。
scratchモデルはtext embedding自体もrandomなので、この「既存LLMへの追加初期化」とは条件が違う。
本版trainerはscratchを明示し、Qwen3の言語能力まで既に獲得済みとは扱わない。

system側にreferenceを置いてlossをmaskすることは、reference prefix cacheにも有利。
user textに毎回違うreference構造を混ぜるより、同じ声の繰り返しrequestでcacheを再利用しやすい。

## 8. RL：SlowとFastへ同じ音声rewardを返す

1promptからG候補を生成し、候補iに対して次のrewardを測る。

```text
R_i = lambda_STT  * ASR/content/instruction reward
    + lambda_Pref * acoustic-quality reward
    + lambda_SIM  * speaker-similarity reward

A_i = R_i - mean_group(R)
```

標準偏差で割らないDr.GRPO型の中心化。
reward varianceの小さなgroupだけgradientを過大にするbiasを避ける狙いがある。
Slow/Fastの両方へ同じcandidate advantageを渡す。

```text
L_policy = -A * log_probability + beta * KL_to_reference
L_RL = L_slow_RL + gamma * L_fast_RL
```

[post_training.py](post_training.py) はこのobjectiveを実装する。
ASR、quality、speaker rewardは外部の実測値を入力し、ランダムrewardは作らない。
論文はMLPへrsLoRA（r=16, alpha=64）を適用し、reference LoRAをCPUへ退避してswapする。
非同期scoringとwaveform cacheもthroughputの工夫であり、モデルのattention構造とは分けて考える。

## 9. 推論と計算量

```text
for each audio time frame t:
    Slow(prefix) → semantic logits + hslow
    sample q0 from semantic band (or EOS)
    if EOS: finish

    Fast prefill position0 with projected hslow
    feed Emb(q0) at depth1 → sample q1
    feed Emb(q1) at depth2 → sample q2
    ... → q9

    aggregate all 10books through MCF
    append one audio position to Slow history
    codec.decode may emit a waveform chunk
```

大きなSlowが音声1frameあたり1回、小さなFastが約9回のsampling。
10book分を大きなLMに逐次処理させるより軽いが、Fastの計算がゼロになるわけではない。
Fastのcontext長は10前後に固定され、長文のquadratic attentionを担わない。

本版`generate()`は処理理解のためprefixを再計算する。
公式servingのpaged KV cache、continuous batching、CUDA graph、RadixAttentionは含まない。
SGLangの性能値をこのPython実装の性能値として読まない。

論文のH200 production engineではRTF=0.195、TTFA約100ms、
高並列時3000超acoustic tokens/sec、voice prefix cache hit平均86.4%を報告する。
TTFAはcache/co-schedulingを含む条件付き測定で、cold start一般の保証ではない。
codecとLMを同GPUにMPSで同居させることも報告される。

## 10. 論文の評価

| 評価 | Fish S2の報告 | 読み方 |
| --- | --- | --- |
| Seed ZH error | 0.54% | protocolの中国語ASR評価 |
| Seed EN WER | 0.99% | 発音正確性の評価 |
| Seed ZH-hard | 5.99% | 常に全比較モデル最良ではない |
| Long EN WER | 4.38% | 独自の長文切り詰め条件を含む |
| Long ZH CER | 5.95% | EN WERと同じ単位ではない |
| Instruction EN TAR | 0.881 | 指示発現率、自然さと別指標 |
| Instruction ZH TAR | 0.984 | 局所tagのactivation評価 |
| EmergentTTS overall win | 81.88% | LLM judge、強いprompt rewrite設定 |

自然さ・WER・speaker SIM・instruction activationは違う尺度。
LLM judgeの勝率を、人間MOSの絶対値へ読み替えない。
論文はLong-TTS-Evalをcontext長の都合で文境界で切り、長さも変化させている。
無制限長で一度も話者driftしないという実証ではない。
低資源言語のいくつかはMiniMax/ElevenLabsのWERが良いとも記す。

## 11. ファイル構成と提供する処理

| ファイル | 内容 |
| --- | --- |
| [dual_ar.py](dual_ar.py) | MCF、Slow/Fast Transformer、shift済みCE、実AR生成 |
| [data.py](data.py) | mixed text/audio packing、reference loss mask、DataFrame Dataset |
| [audio_codec.py](audio_codec.py) | 公式codec実体の構築・ID encode/decode・勾配付き再構成 |
| [main_flow.py](main_flow.py) | mixed prefix→Dual-AR→44.1kHz waveform |
| [training_example.py](training_example.py) | 基本PyTorchのscratch training loop |
| [prepare_data.py](prepare_data.py) | 音声pathと本文から実codec IDと公式prompt、学習Parquet/configを生成 |
| [pretrained_inference.py](pretrained_inference.py) | 公式S2-Pro pretrained Dual-ARとcodecによる参照付き合成・WAV保存 |
| [post_training.py](post_training.py) | 実測rewardを使うgrouped policy loss |

codecは巨大で独立したcausal CNN/Transformer/RVQ subsystemのため公式本体を使う。
実波形を作らないmock decoderは置いていない。
本版`AudioCodec.reconstruct()`は公開encoder→quantizerの`.z`→decoderを直接呼ぶ。
手元の`DAC.forward()`は`encode()`の戻す離散IDsを連続latent用decoderへ渡す経路のため、
その便利wrapperを無条件に利用しない。モデル層そのものは公式実体である。

codecを追加学習する場合は`reconstruct()`の戻り値のcommitment/codebook lossを使える。
論文の全GAN/distillation recipeの欠落を、STFT lossだけで埋めたことにはしていない。

## 12. DataFrameからscratch学習

### 生音声と書き起こしからの準備

中間メタデータは**Parquet（pyarrow、ZSTD圧縮）**で保存する。
embedding・latent・IDなどの数値配列は**Tensor単体の`.pt`**へ保存し、
dtype/shapeを保持する。CSVへ配列を文字列化して格納しない。
元の音声path・本文表はCSV/Parquetの両方から読める。`pyarrow`が必要だが、
自動インストールやCSVへのfallbackは行わない。

**最初に必要な表の列は`audio_path`と`text`だけ。** 話者ID、感情ラベル、
codec ID、prompt IDが揃っていることは前提にしない。
[prepare_data.py](prepare_data.py)が実音声からcodec IDを抽出し、本文から
公式Conversation書式のsystem/user/assistant voice prefixを作る。
下表のpath列はその生成物である。

```csv
audio_path,text
audio/0001.wav,こんにちは。
audio/0002.wav,今日は晴れています。
```

音声pathはメタデータファイルの親基準。音声全体と書き起こしが対応する発話単位にする。
前処理は公式codecにmono化/resampleを任せ、IDs `[N,T]`を保存する。
`N`=codebook数、`T`=約21.53Hz frame数。本文promptはIDs `[L]`で、
`L`=system/user/assistant headerを含むtoken数。GT音声はprefixに混ぜない。
書き起こしに存在しない感情指示や話者tagを勝手に追加しない。

```bash
python prepare_data.py --metadata data/raw.csv --model-dir checkpoints/s2-pro --output data/cache --device cuda:0
python training_example.py --metadata data/cache/train.parquet --config data/cache/config_scratch.json --output checkpoints/fish.pt
```

公式fish-speech packageと依存、ローカルのS2-Pro config/tokenizer/codec重みが
必要。LM本体はロードしない。`--repo`で未installの公式repoも指定できる。
生成configには実tokenizerのsemantic band/EOS/語彙数と実codecのbook数を反映し、
小型のSlow/Fast特徴次元数を明示する。公開4Bの性能再現用構成ではない。
この実装のoffset式はsemantic IDの連続配置を前提とし、非連続の配置は入力エラーとする。

speaker_idも感情カテゴリも通常CE学習の追加教師として必要ない。
この最小表からはreferenceなしの通常TTS用データを作る。
speaker conditioningやemotion instructionを学ぶ別の対応データができたとは扱わない。
元表に任意メタデータがあれば保持するが、未知ラベルを正解として補完しない。
長さ上限は既定30秒で、超過行は音声だけをcropせず拒否する。
出力pathはメタデータファイルの親基準で、学習CLI/`load_metadata()`が解決する。
既存outputの上書き、自動download、追加install、作成時のモデル実行は行っていない。

### 前処理後の学習入力

Dual-ARの学習前に固定codecでIDをcacheする。
codecを同時に変更すると同じIDが指す音響が変わり、LM教師も不安定になる。

| 列 | 内容 |
| --- | --- |
| `prefix_ids_path` | tokenizer済みconditioning text IDs `[L]` |
| `target_codes_path` | target codec IDs `[10,T]` int64 |
| `reference_codes_path` | 任意reference `[10,Tref]` |
| `after_reference_ids_path` | referenceあり時の後続区切り・user本文・assistant header |

referenceなしではprefixがassistant音声生成開始位置まで含む。
referenceありではprefixをsystem reference直前までとし、reference後の残りを別列で渡す。
role tokenやspeaker tagは使用tokenizerの実IDを使い、適当な文字数をID列に代用しない。

```python
import json
from dual_ar import config_from_dict
from training_example import load_metadata, train

with open("data/cache/config_scratch.json", encoding="utf-8") as handle:
    config = config_from_dict(json.load(handle))
model = train(load_metadata("data/cache/train.parquet"), config, "checkpoints/fish.pt")

# SFT型の重み例（減衰率0.8は利用者設定。論文の確定値ではない）
weights = [0.0] + [0.8 ** k for k in range(9)]
# train(..., codebook_weights=weights)で同じ実装へ適用できる。
```

本版は論文の4B重みを自動downloadしない。
scratchで4B構成を作るには、実checkpointのconfigを参照して容量を指定する。
語彙band、codebook_size、norm/QK norm、attention headごとの特徴次元数、RoPE仕様を揃える必要がある。
公式のweight名と本版のweight名は同じではなく、checkpointを無変換でloadできるとはしない。

提供codecのacoustic codebookは1024語彙だが、LMの共通Fast分類headには、それより大きい語彙数を設定できる。
本版`acoustic_vocab_size=1024`は生成時の非semantic bookをその有効範囲に制限するoption。
実codecが1024語彙の場合、生成時にsampling対象を有効ID範囲へ制限する必要がある。
codec設定に合わせてこのoptionを使うか、統一語彙の実checkpoint構成を指定する。

CLIでは同梱 [config_example.json](config_example.json) を使える。

```bash
python training_example.py --metadata data/train.parquet --config config_example.json
```

Dual-AR学習にはPython 3.10以降、PyTorch、einops、pandasを使う。
codec本体の構築には公式fish-speech package、Hydra、OmegaConf等の公式依存が必要。
`build_official_codec(yaml_path)`はYAMLの実moduleをrandom構築し、checkpointを自動取得しない。
学習済みcodecを使う場合は対応state dictを明示的に読み込む。
作成時には上記コマンドやmodelを実行しておらず、追加インストールも行っていない。

### 12.1 公式pretrainedモデルで推論する

[pretrained_inference.py](pretrained_inference.py) は公式 `init_model()`、`generate_long()`、
`load_codec_model()`、`encode_audio()`、`decode_to_audio()` を呼ぶ。
公式 `fishaudio/s2-pro` のモデル一式と `codec.pth` を事前に配置する。
ローカル推論なのでAPIサーバーは不要。scratch用のDualARは読み込まない。

```bash
python pretrained_inference.py --model-dir checkpoints/s2-pro \
  --reference data/speaker.wav --reference-text '参照音声に実際に含まれる本文。' \
  --text '<|speaker:0|>こんにちは。[whisper]これは小さな声です。' \
  --output outputs/fish.wav
```

`--reference-text`は参照音声の正確なtranscriptを指定する。
参照音声は公式codecで実際にID化し、公式conversation packingへ渡す。
長い本文では複数の生成chunkを時間軸で連結し、全chunkを実codecでdecodeする。
semanticのglobal token IDやbook数をsample側で決め打ちせず、公式loaderから読む。

packageがinstallされていない場合は `--repo vendor/fish-speech` を追加する。
codecが別directoryなら `--codec checkpoints/s2-pro/codec.pth` を明示できる。
`--precision auto` はCUDAならBF16対応を確認してBF16/FP16、その他はFP32を選ぶ。
`--temperature 0.7 --top-p 0.7 --top-k 30` は参照revisionのS2-Pro既定sampling値。
必要な依存は公式fish-speechとsoundfile。script内でinstall/downloadは行わない。
Drama 3のcheckpointはこのS2-Proサンプルでは拒否する。
既存出力は `--overwrite` 指定時だけ置換する。
この推論サンプルも今回実行していない。

## 13. 実用上の制約

- flatten型よりSlow contextは短いが、text・reference・長いtargetのKVメモリは必要。
- referenceを長くし過ぎればprefillとcontext budgetを使う。
- Fastの誤差はMCF経由で後続Slowにも伝わるため、teacher-forcing CEだけで生成安定性は決まらない。
- 初期q0を誤ると、Fastだけで本文の欠落を完全修正することは難しい。
- codecのsemantic情報保持や音響再構成の品質が低い場合、LMの改善による生成品質の向上にも制約が生じる。
- causal codecのchunk出力はstate/cacheを維持する必要があり、毎chunkゼロ状態decodeとは異なる。
- textに局所指示を書けることと、未知指示をすべて正確に実現できることは同義ではない。

## 14. 確認状態と一次資料

ユーザー指定に従い、import、compile、学習、推論、benchmarkは実行していない。
TeXと公開ソースの読解により、2方向AR・MCF・教師shift・mask・ID変換を静的に照合した。
動作確認済みやバグ不存在という保証はしていない。
論文図の画像/PDFは参照せず、TeX本文の記述からASCII図を構成した。

- [Fish Audio S2 Technical Report](https://arxiv.org/abs/2603.08823)
- [公式repository](https://github.com/fishaudio/fish-speech)
- [S2 Dual-AR公開実装](https://github.com/fishaudio/fish-speech/blob/0f8a12e8a651062fdee9c33a894bdccd3a8513f1/fish_speech/models/text2semantic/llama.py)
- [生成時のSlow/Fast接続](https://github.com/fishaudio/fish-speech/blob/0f8a12e8a651062fdee9c33a894bdccd3a8513f1/fish_speech/models/text2semantic/inference.py)
- [公開codec config](https://github.com/fishaudio/fish-speech/blob/0f8a12e8a651062fdee9c33a894bdccd3a8513f1/fish_speech/configs/modded_dac_vq.yaml)

TeXのarchitecture/training/inference/data-pipeline/evaluationを参照した。
最新repositoryのDrama 3記事をS2の論文として引用しない。
コードとS2-Pro weightsでlicense条件が異なるので、それぞれを確認する。
