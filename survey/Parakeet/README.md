# Parakeet-TDT-0.6B-v3 論文・公式実装サーベイ

## 1. この資料の範囲

本資料は NVIDIA の論文 **“Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST”** と NeMo Speech の公式実装を対象に、Parakeet-TDT-0.6B-v3 の設計、データ、学習、損失、推論、評価を整理したものである。

論文は Canary-1B-v2 と Parakeet-TDT-0.6B-v3 の共同レポートであり、データ分析や学習戦略の多くはCanaryを中心に説明されている。一方、Parakeetについて論文が直接明記しているのは主に次である。

- 600MパラメータのASR専用モデルで、25の欧州言語を扱う
- 24層FastConformer encoderとTDT decoderを使う
- Canary用データのうちASR subset約66万時間で学習する
- multilingual CTC checkpointから初期化し、128基のA100で15万step事前学習する
- corpusと言語をそれぞれ温度係数0.5でbalanced samplingする
- NeMo ASR Set 3.0の高品質データ7,500時間を使い、4基のA100で5,000step fine-tuneする
- English Open ASR Leaderboardで平均WER 6.32%、RTFx 3332.74を報告する

以下では、論文に明記された事実、公式NeMo実装から確認できる仕組み、教材用再実装を区別して記述する。論文はPDF画像ではなく、ローカルのTeXソースだけを参照した。

## 2. 論文が解こうとしている問題

大規模音声モデルでは、精度、対応言語数、推論速度の間に強いトレードオフがある。WhisperやSeamlessM4Tのような汎用Transformer encoder-decoderは大量の弱教師データから高い汎化性能を得るが、自己回帰decoderを含む大規模モデルは推論コストが高い。また、Web規模データには次の問題がある。

- 疑似ラベル由来の誤りやhallucination
- 英語と高資源言語への偏り
- 特定言語でVoxPopuliの議会音声が95%以上を占めるdomain偏重
- ASR、X→English翻訳、English→X翻訳のデータ量の偏り
- corpus品質と規模のトレードオフ

論文全体は、これらをFastConformerによる効率化、大規模データcuration、階層的sampling、高品質subsetによるfine-tuningで解決しようとしている。CanaryはASRとASTを扱う汎用モデル、ParakeetはASRに絞ってさらに小型・高速にしたモデルという位置づけである。

Parakeetの重要な点は、単純にモデルを小さくしたことではない。音声向けencoderとTDTを組み合わせ、大量のASRデータだけに容量を集中させることで、0.6B規模でも大規模multimodal modelに近い精度と、桁違いのthroughputを両立している。

## 3. データ

### 3.1 論文全体のデータ構成

論文の全学習データは約173.8万時間で、内訳は次のとおりである。

| Corpus | ASR | X→En | En→X | 合計時間 |
|---|---:|---:|---:|---:|
| Granary | 638,229.73 | 358,284.40 | 0 | 996,514.13 |
| NeMo ASR Set 3.0 | 19,843.10 | 10,165.01 | 197,020.66 | 227,028.77 |
| Supplementary | 0 | 0 | 477,739.85 | 477,739.85 |
| 小計 | 658,072.83 | 368,449.41 | 674,760.51 | 1,701,282.75 |
| non-speech | - | - | - | 36,581.15 |
| 総計 | - | - | - | 1,737,863.91 |

GranaryはMOSEL、YTC、YODASなどを含み、多くがpseudo-labelである。これを、人手ラベルを含むNeMo ASR Set 3.0で補う。後者は全体の約13%にすぎないが、AMI、FLEURS、Common Voice、MLS、Golosなどを含み、最終fine-tuningでも重要になる。

### 3.2 Parakeetが使うsubset

Parakeet-TDT-0.6B-v3はASTデータを使わず、上記のASR部分、約658,000時間を25言語で学習する。英語だけで約285,000時間、約40%を占める。非英語でも言語間の差が大きく、例えば論文AppendixではGranary ASRがスペイン語45,812.67時間に対してウクライナ語932.67時間である。

低資源言語ではVoxPopuliへの依存が強い。これは言語coverageを維持するには必要だが、EU議会という狭いregister、約30秒中心のsegment長、会話音声との差というdomain biasを生む。論文のfine-tuningは、単に高品質データを追加するだけでなく、この偏りを緩和する役割を持つ。

### 3.3 non-speechデータ

論文では36,581時間のnon-speech音声に空文字列targetを割り当てる。目的は、Whisper由来のpseudo-labelに含まれ得るhallucinationを抑え、「発話がなければ何も生成しない」ことを学習させること。論文はCanary向けに言語対も無作為割当すると述べるが、Parakeet固有のnon-speech混入量は明記していない。

### 3.4 階層的balanced sampling

samplingは、まず言語内でcorpusをbalanceし、その後に言語をbalanceする。

言語 `l` 内のcorpus `c` の時間を `n(c)`、言語合計を `N_l` とすると、

```text
w_c = (n(c) / N_l)^alpha
p_c = w_c / sum_c w_c
```

次に言語時間 `n(l)` と全体時間 `N_total` から、

```text
w_l = (n(l) / N_total)^beta
p_l = w_l / sum_l w_l
p(c,l) = p_l * p_c
```

とする。Parakeet事前学習では `alpha=0.5`, `beta=0.5`。指数1なら元のデータ比率、0に近づくほど均等samplingになるため、0.5は高資源データを完全には捨てず、低資源corpus/languageを増幅する妥協点である。Lhotse dynamic bucketingと組み合わせ、論文全体の設定では各batchに少なくとも14の言語対が入ったと報告される。

## 4. モデルアーキテクチャ

### 4.1 全体像とshape

`B`: batch size、`S`: waveform sample数、`Tm`: 10 ms単位のMel frame数、`Te`: encoder frame数、`F=128`: Mel bin数、`D`: encoder hidden、`U`: target token数、`H`: prediction hidden、`V`: blankを含まない語彙数、`K`: duration候補数とする。

```text
waveform                         (B, S), 16 kHz
  └─ STFT + 128-bin log-Mel
log-Mel                         (B, F=128, Tm)
  └─ per-feature normalization + SpecAugment
  └─ 3段のstride-2 depthwise separable subsampling
subsampled sequence             (B, Te, D), Te ≈ ceil(Tm/8)
  └─ 24 FastConformer blocks
encoder output                  (B, Te, D)

teacher-forcing token IDs       (B, U)
  └─ blank/BOS prepend + embedding + LSTM prediction network
prediction states               (B, U+1, H)

encoder × prediction joint lattice
token logits                    (B, Te, U+1, V+1)
duration logits                 (B, Te, U+1, K)
```

10 msのMel frameを8倍subsampleするので、encoder 1 stepはおよそ80 msに対応する。論文がParakeetについて明記するのは24層FastConformerとTDT decoderまでであり、hidden sizeなどcheckpoint固有の全hyperparameterは論文本文に載っていない。公式repositoryの汎用Hybrid TDT-CTC YAMLは17層、`D=512`、8 headsであり、0.6B-v3そのものの構成値と混同してはいけない。

### 4.2 FastConformer encoder

通常のConformer blockは、長距離依存を扱うself-attentionと、局所音響patternを扱うconvolutionを組み合わせる。FastConformerは主に次の変更で計算を削減する。

1. 入口で8倍subsamplingし、attentionに入る系列を短縮する
2. subsamplingとConformer convolutionにdepthwise separable convolutionを使う
3. convolution kernelを典型的な31から9へ縮小する
4. channel数を抑える

系列長を1/8にするとfull attentionの`T²`部分は理論上約1/64になる。論文は従来Conformerに対し2–3倍の推論高速化を説明する。blockはMacaron形式で、概念的には次の順序で処理する。

```text
x1 = x  + 0.5 * FFN(LN(x))          (B,Te,D)
x2 = x1 + MHSA(LN(x1), mask)        (B,Te,D)
x3 = x2 + ConvModule(x2)            (B,Te,D)
y  = LN(x3 + 0.5 * FFN(LN(x3)))     (B,Te,D)
```

convolution moduleはpointwise convolution、GLU、depthwise temporal convolution、normalization、SiLU、pointwise projectionからなる。本資料の [fastconformer.py](fastconformer.py) はこの経路を実装する。公式版のrelative positional attention、cache-aware streaming、distributed/export対応は多数の基盤fileを横断するため、教材版ではPyTorch MHAを使用している。

### 4.3 TDT prediction networkとjoint

TDTはCTCのようにencoder出力だけから各frameを独立分類するのではなく、RNN-Tと同様に既出token列も条件にする。

- encoder: 音声文脈 `h_t ∈ R^D`
- prediction network: 過去token列の状態 `g_u ∈ R^H`
- joint network: `h_t` と `g_u` を射影・加算し、tokenとdurationを予測

```text
z[t,u] = tanh(W_enc h[t] + W_pred g[u])
token_logits[t,u]    = W_token z[t,u]       (V+1 classes)
duration_logits[t,u] = W_duration z[t,u]    (K classes)
```

NeMoでは単一joint出力の末尾に`K`個のextra outputを追加し、token部分とduration部分を別々にnormalizationする。公式汎用configのduration集合は `[0,1,2,3,4]`。本資料の [transducer.py](transducer.py) はteacher forcingと1-step推論の両方を実装する。

## 5. 学習時に何を入力し、何を予測し、何を正解とするか

### 5.1 1サンプルに必要な教師データ

ASR fine-tuningでデータセットが持つのは、基本的に次の2つだけである。

```text
audio: 音声waveform
text:  その音声の正しい書き起こし
```

例えば、音声が「今日は晴れです」、書き起こしも`今日は晴れです`なら、tokenizerでtextを
subword ID列へ変換する。

```text
waveform x                          (S,)
GT text y                           "今日は晴れです"
GT token IDs                        [y_1, y_2, ..., y_U]
```

重要なのは、データセットに次の教師ラベルは不要という点である。

- 各tokenが音声の何秒目に対応するか
- 各encoder frameでblank/tokenのどちらを出すか
- TDTが予測するdurationの正解値
- CTCのframe単位class

学習データから直接与えるGTは、原則として書き起こしtoken列`y_1...y_U`だけである。tokenと音声frameの
対応は潜在変数として扱い、TDT lossとCTC lossが「GT transcriptへ到達する全alignment」を足し合わせる。

### 5.2 mini-batchの入力

長さの異なる音声とtoken列をpaddingして、次のbatchを作る。

```text
waveform          x                 (B,S_max)
waveform length   x_len             (B,)
GT token IDs      y                 (B,U_max)
GT token length   y_len             (B,)
```

`x_len`と`y_len`があるため、padding部分はlossに入らない。waveformはlog-MelとSpecAugmentを経て
FastConformerへ入り、音声条件表現を作る。

```text
x (B,S_max)
  -> log-Mel (B,128,Tm)
  -> SpecAugment (B,128,Tm)            # train時のみ
  -> FastConformer
h (B,Te,D), h_len (B,)
```

SpecAugmentは入力音響特徴だけを変え、GT transcriptは変えない。周波数帯や時間区間を隠しても同じ文字列を
出すように学習させることで、noiseや局所的な欠損への頑健性を促す。

### 5.3 TDT branchは何を予測するか

TDT branchではGT token列を1 token右へずらした系列をprediction networkへ入れる。これは通常の
autoregressive teacher forcingと同じ考え方で、位置`u`では正解prefix`y_{<u}`だけを条件にして、
次の正解token`y_u`を予測する。

```text
prediction入力    [BOS, y_1, y_2, ..., y_{U-1}]    (B,U)
予測対象           [y_1, y_2, ..., y_U]              (B,U)
```

実装上は全targetにBOSを付けて`U+1`個のprediction stateを作る。

```text
g = PredictionNetwork([BOS] + y)       (B,U+1,H)
```

FastConformer出力`h_t`とprefix状態`g_u`の全組合せをjoint networkへ入れ、各格子点`(t,u)`で
2種類のlogitsを予測する。

```text
token logits       z_token             (B,Te,U+1,V+1)
duration logits    z_duration          (B,Te,U+1,K)
```

token側は`V`個のsubwordとblankのどれを出すか、duration側は例えば`[0,1,2,3,4]`のどれだけ
encoder時間を進めるかを表す。ただし、あるGT tokenにduration 2を割り当てる、といった直接教師はない。

GT transcriptと一致する経路だけをTDT lattice上に残す。状態`(t,u)`からは、

```text
次のGT token y_u を出す:
    (t,u) -> (t+d,u+1)
    score = log p(y_u | h_t, g_u) + log p(d | h_t, g_u)

blankを出す:
    (t,u) -> (t+d,u), d>0
    score = log p(blank | h_t, g_u) + log p(d | h_t, g_u)
```

という遷移を作る。誤ったtokenを出す経路はGT transcriptを作れないため、正解経路の確率和には入らない。
一方、どの時刻でGT tokenを出すか、どのdurationを選ぶかには複数の正解候補がある。それらを
forward algorithmの`logsumexp`ですべて足し合わせる。

```text
P_TDT(y | x) = sum_{a in A_TDT(x,y)} P(a | x)
L_TDT        = -log P_TDT(y | x)
```

`A_TDT(x,y)`は、最終的に音声をすべて消費し、GT token列を順番どおり生成するtoken/duration alignment集合。
つまりTDT lossは、正解token列を高確率にするだけでなく、音声内容に応じて有用なdurationを自動的に
獲得する。durationの明示GTを作らなくてよいことがTDT objectiveの重要な性質である。

### 5.4 小さなTDT alignment例

encoder長`Te=3`、GT tokenが`[A,B]`、duration候補が`[0,1,2]`だとする。GTへ到達できる経路には、
例えば次がある。

```text
経路1:
(0,0) -- A,d=0 --> (0,1)
(0,1) -- blank,d=1 --> (1,1)
(1,1) -- B,d=1 --> (2,2)
(2,2) -- final blank,d=1 --> (3,2)

経路2:
(0,0) -- blank,d=1 --> (1,0)
(1,0) -- A,d=1 --> (2,1)
(2,1) -- B,d=0 --> (2,2)
(2,2) -- final blank,d=1 --> (3,2)
```

どちらも音声frameを3まで消費し、token列`[A,B]`を生成するので正解alignmentである。TDT lossは
どちらか1経路をhard targetにせず、両者を含む全経路の確率を加算する。学習が進むと、音響的に妥当で
効率のよいtoken位置・durationへ確率が集中する。

### 5.5 CTC branchは何を予測するか

CTC branchはprediction networkを使わず、各encoder frameから直接token/blank分布を予測する。

```text
h (B,Te,D)
  -> Linear + log_softmax
CTC log probabilities             (B,Te,V+1)
```

各frameのargmax系列が`[blank, A, A, blank, B]`なら、連続重複をまとめてblankを除くCTC collapse後は
`[A,B]`になる。学習時はargmaxを取らず、collapse後にGT token列`y`となる全frame alignmentを
forward-backwardで周辺化する。

```text
P_CTC(y | x) = sum_{pi: collapse(pi)=y} product_t p(pi_t | h_t)
L_CTC        = -log P_CTC(y | x)
```

したがってCTCにもframe単位GTは不要で、TDT branchと同じtranscriptを教師として使える。TDTとの違いは、
CTCのframe予測が過去tokenに条件付かず、時間順の単調alignmentに限定される点である。

### 5.6 2つのlossをどう統合するか

Hybrid modelでは1つのFastConformer出力をTDTとCTCの両方へ渡し、同じGT transcriptから2 lossを計算する。

```text
                    ┌-> Prediction + Joint -> L_TDT
waveform -> Encoder-|
                    └-> CTC Head ----------> L_CTC

L_total = (1 - w_ctc) * L_TDT + w_ctc * L_CTC
```

汎用公式configの`w_ctc=0.3`なら、

```text
L_total = 0.7 * L_TDT + 0.3 * L_CTC
```

となる。backpropagationでは、TDT decoder/jointは`L_TDT`から、CTC headは`L_CTC`から更新され、共有encoderは
両lossの加重和から更新される。TDTを実運用decoderとして使うmodelをCTC lossだけでfine-tuneすると、
encoderだけがCTC向けに変化して既存TDT decoderとの整合が崩れる可能性がある。そのため本資料の
fine-tuning例はencoder、prediction network、joint、CTC headをすべて更新する。

### 5.7 学習1 stepの疑似コード

```python
# 入力と唯一の明示GT
waveform, waveform_len = batch["audio"]       # (B,S), (B,)
targets, target_len = batch["token_ids"]      # (B,U), (B,)

# 共有音声encoder
mel, mel_len = preprocessor(waveform, waveform_len)  # (B,128,Tm)
encoded, enc_len = encoder(mel, mel_len)              # (B,D,Te) in NeMo

# TDT: GT prefixをteacher forcingし、token/duration alignmentを周辺化
pred = prediction_network(targets, target_len)        # (B,H,U+1)
joint = joint_network(encoded, pred)                  # (B,Te,U+1,V+1+K)
loss_tdt = tdt_loss(joint, targets, enc_len, target_len)

# CTC: 同じencoderと同じGT transcriptを使い、frame alignmentを周辺化
ctc_log_probs = ctc_head(encoded)                     # (B,Te,V+1)
loss_ctc = ctc_loss(ctc_log_probs, targets, enc_len, target_len)

# 共有objective
loss = 0.7 * loss_tdt + 0.3 * loss_ctc
loss.backward()
optimizer.step()
```

## 6. TDT lossの実装詳細

### 6.1 RNN-TからTDTへの拡張

transducer latticeの状態 `(t,u)` は、音声frameを`t`個、正解tokenを`u`個消費した状態を表す。通常RNN-Tの遷移は次の2種類である。

```text
label: (t,u) -> (t,u+1)
blank: (t,u) -> (t+1,u)
```

TDTは各遷移でduration `d` も予測する。

```text
label + duration d: (t,u) -> (t+d,u+1), d >= 0
blank + duration d: (t,u) -> (t+d,u),   d > 0
```

これによりblankを1 frameずつ繰り返さず、複数encoder frameを一度にskipできる。TDT lossは正解token列を生成するすべてのtoken/duration alignmentの確率をforward-backwardで周辺化した負対数尤度である。受理には最後のblank遷移が必要になる。

[losses.py](losses.py) はこのforward DPをautograd可能なPyTorchで実装する。入力shapeはtoken logits `(B,Te,U+1,V+1)`、duration logits `(B,Te,U+1,K)`、target `(B,U)`、出力はscalar。

### 6.2 公式lossとの境界

公式NeMoは同じlattice計算をNumba CUDA kernelで高速化し、さらに次を持つ。

- `sigma=0.02`: logit under-normalization
- `omega=0.1`: 一定確率で通常RNN-T lossを混ぜるsampling
- `fastemit_lambda`: token emissionを早める正則化。汎用config既定値は0
- `clamp`: joint gradient clipping。汎用config既定値は無効

教材版TDT DPはlatticeの理解と小型検証を目的とし、`sigma`と`omega`は実装していない。0.6Bモデルを実際に学習する [finetuning_example.py](finetuning_example.py) ではcheckpoint内の公式CUDA lossを呼ぶ。

### 6.3 Hybrid TDT-CTC

手元の日本語checkpoint `nvidia/parakeet-tdt_ctc-0.6b-ja` は共有encoderにTDTとCTCの2 headを持つ。公式 `EncDecHybridRNNTCTCModel.training_step` の目的関数は、

```text
L_total = (1 - w_ctc) * L_tdt + w_ctc * L_ctc
```

であり、汎用Hybrid configでは`w_ctc=0.3`。CTCは単調alignmentを強く促し、TDTは履歴依存のtoken予測とduration予測を学習する。CTCだけでfine-tuneすると共有encoderが変化する一方、TDT prediction/jointが追従せず、実際の`transcribe()`で使うTDT経路が悪化し得る。そのためfine-tune例は両lossと両headを更新する。

## 7. 学習レシピ

### 7.1 Parakeet-TDT-0.6B-v3事前学習

論文に記載されたParakeet固有の手順は次のとおり。

1. Granary ASR subsetでpretrainされたmultilingual CTC checkpointから初期化
2. 約66万時間、25言語のASRデータを使用
3. corpus/language samplingは`alpha=0.5`, `beta=0.5`
4. 128基のNVIDIA A100で150,000 step学習
5. 24層FastConformer + TDT decoderを使用

Canaryで使われたinverse-square-root schedule、AdamW、duration bucketing、OOMptimizerという基盤は同じ論文内で説明されるが、Parakeet固有のlearning rate、warmup、global batch sizeは本文に明記されない。したがってCanaryの値をParakeetの確定値として転記すべきではない。

### 7.2 高品質fine-tuning

Parakeetの最終stageは、NeMo ASR Set 3.0の高品質ASRデータ7,500時間を使用し、4基のA100で5,000 step実施される。Canary側の詳細な実験では、VoxPopuli偏重を抑えるため、高品質subsetへ急に切り替えるよりもsampling weightをcosine scheduleで徐々に一様分布へ移す方法が安定した。ただし、論文はParakeetの5,000 stepで同じdynamic scheduleを使ったとは明記していない。

### 7.3 DataFrameからのfine-tune実装

[finetuning_example.py](finetuning_example.py) は`pandas.DataFrame`の次のschemaを受ける。

| column | dtype | 意味 |
|---|---|---|
| `audio` | str | 音声file path |
| `text` | str | transcript |

NeMo Trainer、Lightning、Hydraは使わず、`Dataset`、`DataLoader`、`AdamW`、`backward()`、gradient clipping、cosine schedulerを直接使用する。NeMoに依存するのはpretrained checkpointと、そのpreprocessor、encoder、decoder、joint、TDT loss、CTC head/lossだけである。

```python
import pandas as pd
from finetuning_example import train

train_df = pd.DataFrame({
    "audio": ["/data/a.wav", "/data/b.wav"],
    "text": ["こんにちは", "音声認識"],
})
model = train(train_df, epochs=1, batch_size=1, output="parakeet_ft.pt")
```

内部shapeは次のとおり。

```text
waveform          (B,S)
preprocessor      (B,128,Tm)
encoder           (B,D,Te)       # NeMo APIはchannel-first
targets           (B,U)
decoder           (B,H,U+1)
joint lattice     (B,Te,U+1,V+1+K)  # 非fused時の概念shape
CTC log-probs     (B,Te,V+1)
loss              scalar
```

joint latticeは非常に大きいため、checkpointの`joint.fuse_loss_wer=True`なら、公式実装と同じくsub-batch単位でjointとlossを融合計算する。追加installは行わず、TDT lossはCUDAを前提とする。

## 8. 推論

### 8.1 TDT greedy decoding

[decoding.py](decoding.py) は、各状態 `(t,u)` でtokenとdurationのargmaxを取り、次のように更新する。

```text
if token == blank:
    t += max(duration, 1)
else:
    output.append(token)
    prediction_state = LSTM(token, prediction_state)
    timestamp.append(t)
    t += duration
```

duration 0のtokenを同じframeで複数生成できるため、無限loopを防ぐ`max_symbols_per_step`が必要。TDTが高速なのは、音声の無出力区間をdurationでまとめてskipできるためである。

### 8.2 beam search

beam searchは`(token列, t, prediction state, score)`を複数保持し、tokenとdurationの組を展開する。NeMoは通常beamに加えてTDT向けmodified ALSDとbatched実装、CUDA graph対応を持つ。本資料では重要な状態遷移を明確にするためgreedyを実装し、production beamは公式`tdt_beam_decoding.py`および`tdt_malsd_batched_computer.py`を参照対象とする。

### 8.3 CTC decodingとtimestamp

CTC greedyはframeごとのargmaxから連続重複を畳み、blankを除く。正解transcriptまたは別ASR出力が既知なら、CTC log probability上でViterbi alignmentを行いtoken/word/segment timestampを得られる。

論文のNFAは600Mのmultilingual Parakeet CTC modelを用い、CanaryのASR/AST出力をalignする。このCTC modelはCanaryと同じtokenizer、同じASR subsetで128基のA100を使い250,000 step学習される。ASRは音声と文字が単調対応するのでword timestampが成立しやすい。一方ASTは語順変化を含み、CTC model自体も翻訳を学んでいないため、論文は翻訳結果にはword単位ではなくsegment単位timestampを推奨する。また、評価言語が主に欧州言語であるため、中国語など語順差の大きな言語への一般化は未検証とされる。

## 9. 評価結果

### 9.1 English Open ASR Leaderboard

WERはEnglish/multilingual用のHugging Face Open ASR Leaderboard normalizerで正規化される。

| Model | Parameters | RTFx | Avg WER |
|---|---:|---:|---:|
| Whisper-large-v3 | 1.55B | 145.51 | 7.44 |
| Voxtral-Mini-3B-2507 | 3B | 109.86 | 7.05 |
| Phi-4-multimodal-instruct | 5.6B | 62.12 | 6.14 |
| **Parakeet-TDT-0.6B-v3** | **0.6B** | **3332.74** | **6.32** |
| Canary-1B-v2 | 1B | 749.00 | 7.15 |

Parakeetは最小クラスでありながらPhi-4との差が0.18 WER pointで、throughputは約54倍。論文の中心的な主張は絶対WERだけでなく、このaccuracy/throughput比にある。ただしRTFxはhardware、batching、decoder設定に依存するため、別環境の測定値と無条件に比較できない。

dataset別WERはAMI 11.39、Earnings22 11.19、GigaSpeech 9.57、LibriSpeech clean 1.92、LibriSpeech other 3.59、SPGI 3.98、TED-LIUM 2.80、VoxPopuli 6.09。

### 9.2 multilingual ASR

24言語評価の平均WERは9.7%。内訳はFLEURS 11.52、CoVoST2 9.78、MLS 7.83で、Whisper-large-v3の9.9%をわずかに上回る。全比較model共通の6言語（en/fr/de/it/pt/es）では平均5.3%、内訳は4.37/4.79/6.74で、SeamlessM4T-v2-largeの5.3%と同等、Whisper-large-v3の5.8%より良い。

一方、Appendixの言語別結果には大きなばらつきがある。例えばCoVoST2ではスペイン語3.41、イタリア語6.05に対し、エストニア語38.36、ロシア語31.80である。平均値だけを見ると、低資源言語やdomain mismatchの弱点を見落とす。このばらつきは、論文が詳述するVoxPopuli偏重と整合的である。

### 9.3 noise robustness

| SNR (dB) | 100 | 50 | 25 | 10 | 5 | 0 | -5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Parakeet WER | 1.92 | 1.92 | 1.96 | 2.15 | 2.62 | 4.82 | 12.21 |
| Canary WER | 2.18 | 2.16 | 2.01 | 2.29 | 2.80 | 5.08 | 19.38 |

Parakeetは全SNRで安定し、特に-5 dBではCanaryより約37% relativeに低いWERとなる。論文はYouTubeやVoxPopuliなど自然にnoiseを含む大規模学習データがrobustnessに寄与した可能性を示しているが、因果を切り分けるablationは提示していない。

## 10. 論文から読み取れる強みと限界

### 強み

- 0.6Bで大規模modelに近いWERと非常に高いRTFxを両立
- 8倍subsamplingとTDT duration skipがencoder・decoder双方の計算を削減
- 25言語を単一ASR modelで扱う
- 高品質fine-tuningと階層samplingでWeb-scale pseudo-labelの偏りを緩和
- severe noiseでもCanaryより低いWER
- CTC系modelを別途使うことでforced alignmentにも展開可能

### 限界・論文で未確定な点

- 共同reportのため、Parakeet固有のoptimizer、learning rate、batch size、tokenizer詳細が不足
- 25言語は欧州言語中心で、世界規模のlanguage coverageではない
- 低資源言語の一部はVoxPopuliへ強く依存し、言語別WERの差も大きい
- RTFxの測定条件を揃えない外部比較には注意が必要
- TDTはCUDA向け専用loss/decoder実装への依存が強い
- timestamp節の主対象はCanary + auxiliary Parakeet CTCであり、TDT token timestamp自体の定量評価ではない
- AST timestampの品質評価は主にmanualで、語順差の大きい言語は未検証

## 11. 公式実装との対応

| 概念 | 公式実装 |
|---|---|
| FastConformer | `nemo/collections/asr/modules/conformer_encoder.py` |
| 8x subsampling | `nemo/collections/asr/parts/submodules/subsampling.py` |
| prediction network / joint | `nemo/collections/asr/modules/rnnt.py` |
| Hybrid model | `nemo/collections/asr/models/hybrid_rnnt_ctc_models.py` |
| TDT loss wrapper | `nemo/collections/asr/parts/numba/rnnt_loss/rnnt_pytorch.py` |
| TDT CUDA DP | `.../rnnt_loss/utils/cuda_utils/gpu_rnnt_kernel.py` |
| TDT greedy | `nemo/collections/asr/parts/submodules/rnnt_greedy_decoding.py` |
| TDT beam | `nemo/collections/asr/parts/submodules/tdt_beam_decoding.py` |
| mALSD | `nemo/collections/asr/parts/submodules/tdt_malsd_batched_computer.py` |
| Hybrid TDT-CTC config | `examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_tdt_ctc_bpe.yaml` |

## 12. このディレクトリのコード

- [audio_frontend.py](audio_frontend.py): STFT、128-bin log-Mel、正規化、SpecAugment
- [fastconformer.py](fastconformer.py): 8倍subsamplingとFastConformer block
- [transducer.py](transducer.py): prediction network、TDT joint、CTC head
- [losses.py](losses.py): TDT forward DP、CTC、Hybrid loss
- [decoding.py](decoding.py): TDT/CTC greedy decoding
- [main_flow.py](main_flow.py): 全部品を接続しbackwardまで行う小型model
- [finetuning_example.py](finetuning_example.py): DataFrameとpretrained NeMo modelを使う実fine-tune

`main_flow.py`は表示だけのmockではなく、waveformから両lossを計算し、encoderまでbackpropした後、同じmodelのTDT decodeを実行する。

```bash
cd survey/Parakeet
python main_flow.py
python finetuning_example.py \
  --train train.csv --valid valid.csv --epochs 3 --batch-size 4
```

## 13. 参照した一次資料

一次資料:

- [Canary-1B-v2 & Parakeet-TDT-0.6B-v3 論文](https://arxiv.org/abs/2509.14128)
- [NVIDIA NeMo Speech公式実装](https://github.com/NVIDIA-NeMo/NeMo)
- [Parakeet-TDT-0.6B-v3 model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [Parakeet TDT-CTC 0.6B Japanese model card](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja)
