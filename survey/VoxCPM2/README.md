# VoxCPM2 — 連続latent・階層AR・局所Flow Matchingのサーベイ

## 1. 概要

VoxCPM2は、**連続音声latentを4frameずつまとめ、次のpatchを自己回帰的に生成するTTS**。
1patchの具体的な連続値はLocal Diffusion TransformerによるFlow Matchingで生成する。
離散speech tokenを次token分類するIndexTTSやFish S2とは、音声の表現とlossが異なる。

構成は、音声をlatentへ変換するAudioVAE encoderと、
**①LocEnc・TSLM・FSQ・RALMによる生成条件の計算、②LocDiTによる次patchの生成、
③AudioVAE decoderによる波形復元**からなる。
AudioVAE encoderは学習音声や参照音声を前処理するために使用する。
本文だけから合成する場合、新たな生成対象の波形をencoderへ入力する必要はない。
AudioVAEは16kHz音声を64次元・25Hzのlatentへ圧縮する。
4frameを1patchにするため、発話全体を処理するLMの音声位置数は毎秒6.25個となる。
waveform decoderは同じlatentから48kHz音声を作る。

論文は30言語・9中国語方言、自然言語によるvoice design、reference cloning、
style制御付きcloning、continuationを一つのbackboneで扱う。
backboneは約2B、学習データは200万時間超。

### 1.1 「tokenizer-free」の意味

ここでのtokenizer-freeは、外部音声codecの離散IDをLMの分類対象にせず、
連続latentを生成対象にする設計を指す。本文のtokenizerとAudioVAEは使用する。
また、内部の隠れ状態にはFSQによる量子化を適用する。
FSQはsemantic stateを制約する内部ボトルネックであり、
音声を8192語彙などへ割り当ててCEで分類するspeech tokenizerとは役割が異なる。

### 1.2 VoxCPM系列からの差分

| 項目 | VoxCPM | VoxCPM1.5 | VoxCPM2 |
| --- | --- | --- | --- |
| backbone規模（論文表） | 約0.6B | 約0.8B | 約2B |
| TSLM | 24層・1024次元 | 24層・1024次元 | 28層・2048次元 |
| LocEnc | 4層・1024次元 | 8層・1024次元 | 12層・1024次元 |
| RALM | 6層 | 8層 | 8層・2048次元 |
| LocDiT | 4層・1024次元 | 8層・1024次元 | 12層・1024次元 |
| patch内frame数P | 2 | 4 | 4 |
| LM側音声rate | 12.5Hz | 6.25Hz | 6.25Hz |
| FSQ latentの特徴次元数 | 256 | 256 | 512 |
| 入力→出力sample rate | 16→16kHz | 44.1→44.1kHz | 16→48kHz |
| RALM前融合 | 加算 | 加算 | concat＋projection |
| LocDiT条件 | semantic/residual/timeの加算 | 同系統 | 3つの独立prefix token |
| 任意reference prefix | continuation中心 | continuation中心 | transcript不要の独立参照segment |

モデル規模の拡大に加え、**意味情報と音響情報を加算する前に個別の特徴として保持し、
RALMとLocDiTへの入力方法を変更した**点が主な構造上の差分である。
TSLMにはRoPE、RALMにはNoPEを使う設計も長文安定性を狙う変更である。

## 2. 表記と全体アーキテクチャ

| 記号 | 意味 |
| --- | --- |
| B | batch内の発話数 |
| S16 / S48 | 16kHz入力／48kHz出力のsample数 |
| Tf | 25HzのAudioVAE frame数 |
| P=4 | 1patch内のVAE frame数 |
| N | 音声patch数。Tf/P、約6.25Hz |
| L | text tokenizerによる本文token数 |
| S | reference・text・targetを含むLM総位置数 |
| C=64 | 1音声frameの連続latentの特徴次元数 |
| De | Local Encoder出力の特徴次元数。論文1024 |
| Dl | TSLM/RALMの隠れ状態の次元数。論文2048 |
| Dd | Local DiTの隠れ状態の次元数。論文1024 |
| Q | FSQの量子化空間の次元数。論文512 |

### 2.1 学習時の全体図

まず、生成時の主要な処理を3つの枠に分ける。
学習時にはAudioVAE encoderで得た正解patchを①の過去文脈と②の損失計算に使用する。

```text
本文ID・参照latent・既に生成したpatch
                  │
                  ▼
┌──────────────────────────────────────────────────────────────────┐
│ ① 生成条件の計算：LocEnc → TSLM → FSQ → 特徴融合 → RALM          │
│    音声patchをLM位置へ集約し、次patchのsemantic/residual条件を得る │
└────────────────────────────────┬─────────────────────────────────┘
                                 │ semantic/residual条件
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ② patch生成：LocDiT / Flow Matching                              │
│    直前patchと①の条件を使用し、noiseから次patch [B,4,64]を生成    │
└────────────────────────────────┬─────────────────────────────────┘
                                 ├─ ①の音声文脈に追加して次patchへ
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│ ③ 波形復元：AudioVAE decoder                                    │
│    生成latent [B,64,4N] → 48kHz mono波形 [B,1,S48]               │
└──────────────────────────────────────────────────────────────────┘
```

`B`は発話数、`N`は生成patch数、`S48`は48kHz波形のsample数。
①と②はpatchごとに繰り返す。③はまとめて、または対応するstreaming実装で実行する。
以下はこの構成を学習時のGT・損失関数と対応付けた詳細図である。

```text
GT waveform [B,S16] at16kHz
  │ AudioVAE V2 encoder（通常backbone学習前に用意・固定）
  ▼
latent [B,64,Tf] at25Hz
  │ group 4 consecutive frames
  ▼
patches z [B,N,4,64] ────────────────────────────────────────┐
  │ LocEnc: patch内full attention                            │ GT
  ▼                                                         │
patch embeddings E [B,N,De]                                  │
  │ Linear(De,Dl)                                            │
  ▼                                                         │
E_lm [B,N,Dl]                                               │
  │ mix with text/reference positions                       │
  ▼                                                         │
combined input [B,S,Dl]                                      │
  │                                                         │
  ├─ TSLM causal Transformer 28L                            │
  │       ▼                                                 │
  │   semantic hidden H [B,S,Dl]                             │
  │       │ text位置:そのまま / audio位置:FSQ(Q=512)          │
  │       ▼                                                 │
  │   semantic skeleton Hq [B,S,Dl] ──────────────┐           │
  │       │                                      │           │
  │       └─ concat(Hq, E_lm*audio_mask)          │           │
  │                    [B,S,2Dl]                 │           │
  │                    Linear(2Dl,Dl)            │           │
  │                        ▼                     │           │
  │                  RALM causal 8L / NoPE       │           │
  │                        ▼                     │           │
  │                  residual Hres [B,S,Dl]      │           │
  │                        │                     │           │
  └────────────────────────┼─────────────────────┘           │
                           │ shift right by ONE LM position  │
                           ▼                                 │
             sem/res conditions [B,S,Dl]                     │
               │ separate Linear(Dl,Dd)                      │
               ▼                                              │
             mu [B,S,2Dd]                                    │
               │ flatten B*S as independent patch batch      │
               ▼                                              │
previous GT patch ────► LocDiT / Flow Matching ◄───────────────┘
[B*S,64,4]               │ noise/timeでGT patchを汚す
                         ▼
                 velocity [B*S,64,4]
                         │ masked MSE vs analytical velocity GT
                         ▼
                       L_flow

shifted semantic [B,S,Dl] → Linear→SiLU→Linear(2) → stop logits[B,S,2]
                                                     │ masked CE
                                                     ▼
                                                   L_stop
total = L_flow + L_stop
```

LocEncは過去全patchを一度にattentionするencoderではない。
`B*N`をbatch軸へ畳み、patchごとの5token（CLS＋4frame）へfull attentionする。
TSLM/RALMはpatchをまたぐ長期依存を扱い、LocDiTは1patch内の音響特徴を生成する。

### 2.2 推論時の1patch生成処理

```text
text/reference + generated patches z0 ... z(i-1)
                   │ LocEnc + TSLM + FSQ + RALM
                   ▼
             h_sem_i [B,Dl], h_res_i [B,Dl]
                   │ 別々にprojection
                   ▼
             LocDiT conditions [B,2,Dd]
                   │
noise x1 [B,64,4] ──┼─ previous clean patch [B,64,4]
                   ▼
       LocDiTのODE更新を反復し、次の1patchを生成
                   ▼
       generated continuous patch zi [B,4,64]
                   ├─ 次のLocEnc入力へ
                   ├─ stop predictorが終了を判定
                   └─ AudioVAE decoder → 48kHzの約160ms分
```

ARはpatch方向、Flow Matchingはpatch内の連続値方向。
「6.25Hzだから6.25回のTransformerだけで1秒生成」ではなく、
各patch生成に複数回の小さなLocDiT評価が必要である。

## 3. AudioVAE V2：16kHzから48kHzへ

### 3.1 非対称codec

```text
16kHz waveform [B,1,S16]
  → causal CNN encoder
       stride 2 → 5 → 8 → 8  （積640）
  → posterior mean latent [B,64,S16/640] = [B,64,Tf]
                           25frames/sec
  → group P=4
  → [B,N,4,64]             6.25patches/sec

predicted latent [B,64,Tf]
  → causal CNN decoder
       upsample 8 → 6 → 5 → 2 → 2 → 2 （積1920）
  → waveform [B,1,Tf*1920] at48kHz
```

`16000/640=25`、`48000/1920=25`なので、同じlatent時間軸を保ったまま出力帯域を増やせる。
16kHz入力から失われた高周波の真値を数学的に復元しているわけではない。
48kHz decoderがデータ分布から高帯域の音響を生成するimplicit super-resolutionである。

公式`AudioVAE.encode()`はposteriorの`mu`を返す。
backboneのGT cacheを作るとき、毎回posterior noiseをサンプルする処理とは異なる。
本版 [audio_vae.py](audio_vae.py) は実AudioVAE本体を注入し、encode/decodeを接続する。

### 3.2 AudioVAEの再構成品質と生成品質の関係

latentを完璧に予測しても、codecのreconstruction errorは残る。
論文のVCTKではAudioVAE V2のMelD-48k=1.335、MelD-16k=0.813、
STOI-16k=0.907、PESQ-16k=3.906。
44kHzを直接入力するVoxCPM1.5のMelD-48k=1.139より良いとは報告していない。
V2の価値は高帯域品質だけでなく、16kHz入力との整合性・低いlatent rateとの両立にある。

## 4. Local Encoder：4frameからpatch embeddingへの変換

```text
z [B,N,P=4,C=64]
  → Linear(64,De)                       [B,N,4,De]
  → prepend learned CLS per patch      [B,N,5,De]
  → rearrange                          [B*N,5,De]
  → MiniCPM Transformer ×12, full attention
  → take CLS position                  [B*N,De]
  → rearrange                          [B,N,De]
  → enc_to_lm Linear(De,Dl)             [B,N,Dl]
```

CLSはpatchごとに別パラメータを持つのではなく、同じ学習vectorを各patchへ展開する。
patch間でLocal Encoderの重みも共有する。
各frameの特徴を保持したままTransformerへ入力し、learned tokenの出力でpatchを表現する。
CLSは4frameすべてへattentionして情報をまとめる。

## 5. TSLM・FSQ・RALM

### 5.1 text/audioの二つのtrack

各LM位置にtext IDとaudio patch格納領域があり、modality maskが片方を選ぶ。

```text
text_ids:   [text0 text1 ... AUDIO_START  0    0    ... AUDIO_END]
audio:      [ zero  zero ...    zero     z0   z1   ...    zero ]
audio_mask: [  0     0  ...       0       1    1   ...      0  ]

combined = text_embedding * text_mask + LocEnc(audio) * audio_mask
```

audio maskの0/1は音声値が0かどうかではなく位置の種別。
audio codebook内IDの有無を判定しているわけでもない。
referenceも音声位置ならLocEncで埋め込むが、loss maskは0にする。

### 5.2 MiniCPM block

```text
hidden [B,S,Dl] ─────────────────────────────────────┐
  → RMSNorm                                         │
  → GQA Q:[B,H,S,d], K/V:[B,G,S,d]                   │
  → RoPE（TSLM）/ NoPE（RALM）                       │
  → causal attention                                │
  → Linear(H*d,Dl)                                  │
  → multiply scale_depth/sqrt(number_of_layers)      │
  → residual add ◀──────────────────────────────────┘
       │                                            ┐
       → RMSNorm                                    │
       → SiLU(Wgate*x) * (Wup*x)    [B,S,I]          │
       → Wdown                    [B,S,Dl]          │
       → 同じmuP residual scale                     │
       → residual add ◀─────────────────────────────┘
```

Hはquery head数、GはKV head数、dはattention headごとの特徴次元数、IはFFNの中間特徴次元数。
`Dl/H`とdが常に同じとは限らないためhead_dimを設定として持つ。
muPのresidual scaleを除くと、同じ層数・隠れ状態の次元数でも公式と別の計算になる。
本版 [transformer.py](transformer.py) はLongRoPEのfactorと特徴次元数を明示指定できる。
checkpointのshort/long factorを無視して通常RoPEで置換しない。

### 5.3 FSQの処理と論文／コード差

```text
TSLM hidden [B,S,Dl]
  → Linear(Dl,Q=512)
  → tanh                               [-1,1]
  → round(scale*x)/scale
  → straight-through during training
  → Linear(Q,Dl)
  → semi-discrete semantic state [B,S,Dl]
```

論文は「9 levels」と記す。
一方、公開`ScalarQuantizationLayer`は`scale=9`で `round(9*x)/9`。
この式の値域は整数-9..9を9で割った**最大19水準**である。
「scale=9なので9水準」とそのまま説明してはいけない。
本版は公開コードの式を採用し、`fsq_scale`を明示している。
19^512の巨大な分類headを作るわけではなく、512軸を個別に量子化して再射影する。

text位置はFSQを通さず、audio位置だけに適用する。
この処理により、本文位置の意味表現にはFSQによる情報量の制限を適用しない。

### 5.4 RALM前の融合

```text
semantic state Hq [B,S,Dl] ────────────────┐
                                         ├─ concat [B,S,2Dl]
LocEnc E_lm * audio_mask [B,S,Dl] ────────┘
                                                │ Linear(2Dl,Dl)
                                                ▼
                                      RALM causal / NoPE
                                                ▼
                                      residual state [B,S,Dl]
```

semantic skeletonと観測音響historyを別々の入力として渡す。
旧版の単純加算と異なり、projectionが両側の寄与を学べる。
RALMは「TSLMと同じ重みをもう一度使うloop」ではなく独立したTransformer stack。

## 6. LocDiT：入力token列とattentionの範囲

### 6.1 1patchの11token

```text
prefix conditions                         clean context       denoising target
┌────────────┬────────────┬─────────────┬───────────────────┬───────────────────┐
│ sem token  │ res token  │ time token  │ previous patch    │ current noisy     │
│ [N,1,Dd]   │ [N,1,Dd]   │ [N,1,Dd]    │ 4×[N,1,Dd]        │ 4×[N,1,Dd]        │
└────────────┴────────────┴─────────────┴───────────────────┴───────────────────┘
                               2 + 1 + 4 + 4 = 11 positions
                                            │
                      LocDiT 12 layers / full attention
                                            │
                            take last 4 positions only
                                            ▼
                             Linear(Dd,64) → v[N,64,4]
```

Nはこの図では処理するpatch総数（学習時B*S）、音声patch数と混同しないため実装ではcountも使う。
semantic/residualを足して1tokenにまとめず、独立したprefixとして保持する。
previous clean patchとcurrent noisy patchも別projection。
time embeddingはsinusoidal(1000*t)→MLP。
optional mean-flow用のdelta-time embeddingも加算する。

LocDiTは各noisy位置を同時に見てよい。
同じpatch内の未来frameを見られることと、未生成の次patch GTを見てよいことは異なる。
後者はteacher forcingのshiftで防ぐ。

## 7. 学習目的：入力位置・予測対象・損失関数

### 7.1 予測対象に対するLM状態の1位置shift

```text
input position      ... AUDIO_START | z0       | z1       | z2(last)
TSLM/RALM output    ...     h_a      | h_0      | h_1      | h_2
predict target             z0       | z1       | z2       | (no loss)
stop GT                    0        | 0        | 1        | (no loss)
previous patch             zero     | z0       | z1       |
```

学習batchではGT patchを同位置に格納したまま、LM hiddenを1位置右へshiftする。
`z1`のCFMを学習するときに使うのは`z0`までを見たhidden。
GT `z1`をLocEncへ入れたhiddenから`z1`を予測すると自己回答漏洩になる。
本版 [model.py](model.py) はこのshiftを省略していない。

### 7.2 Flow Matching

入力はnoisy target patch、時刻、直前のclean patch、2種類のLM条件。
出力は連続latentのvelocity。GTはGT latentとnoiseから作る。

```text
z_GT  [N,64,4] = clean target patch
eps   [N,64,4] ~ N(0,I)
t     [N] ~ sigmoid(N(-0.4,1))     # 公式既定log-norm

x_t   = (1-t)*z_GT + t*eps
v_GT  = eps - z_GT
v_hat = LocDiT(x_t, h_sem, h_res, previous_patch, t)

frame_error [N,4] = mean_latent_dim((v_hat-v_GT)^2)
L_flow = sum(frame_error * target_mask) / sum(target_mask)
```

**t=0がclean、t=1がnoise。**
IndexTTSのCFMとは時刻の向きが逆。
推論で1→0へ進むため、更新式は`x = x - dt*v`となる。
CEで64次元latentの数値を分類しているわけではない。

### 7.3 CFGの学習

学習時には10%の確率でsemantic/residualのLM条件をゼロに置き換える。
previous acoustic patchは残す。
unconditionalとは「何も条件を一切見ない」のではなく、LM条件を外したfieldである。

推論では `v_uncond + cfg*(v_cond-v_uncond)` を用いる。
論文の既定cfg=2.0、実用範囲1.5〜3.0。
強くすれば必ず良いわけではなく、過度なguidanceは局所音響を不安定にし得る。

### 7.4 stop loss

shift後semantic hidden→Linear→SiLU→biasなし2-class Linear。
GTは各target patchのcontinue=0 / 最後のtarget patchでstop=1。
reference・text・paddingはmaskしてCEへ寄与させない。

```text
stop_logits [B,S,2]
L_stop = masked_mean(CE(stop_logits, stop_labels))
L_total = L_flow + L_stop
```

stopは最後のpatchを生成した後に参照する。
最後のpatchを生成する前にstopしてしまうと、学習と推論の終了位置がずれる。
公開trainerの既定重みはdiff=1、stop=1。

### 7.5 optional mean-flow

公開`UnifiedCFM`には`mean_mode`とJVPによる平均velocity学習もある。
通常CFMではr=t、mean-flowでは一部sampleでr<tとし、
`v_target = v - (t-r)*d(model_output)/dt`を使う。
これはVoxCPM2本文の基本CFM objectiveと区別する。
本版にもoptional経路を実装し、既定は`mean_mode=False`。
headやlossを呼ばずにpassする省略ではない。

## 8. unified sequenceと5つの利用形態

```text
Basic TTS:
  text | AUDIO_START → target patches

Voice design:
  (voice/style description) text | AUDIO_START → target patches

Reference cloning:
  REF_START ref-patches REF_END | text | AUDIO_START → target patches

Controllable cloning:
  REF_START ref-patches REF_END | (style) text | AUDIO_START → target patches

Continuation:
  prompt-text + target-text | AUDIO_START | observed-prompt-patches → new patches
```

referenceは同一話者の独立したclipで、transcriptを必須としない。
continuationは本文と対応する観測音声の先頭部分。
referenceをcontinuationとして扱うと、時間的につながらない音声を直前patch条件へ入れてしまう。
本版generationはprefix末尾がaudioならcontinuation、text boundaryならprevious patchをゼロにする。

公開special IDsはAUDIO_START=101、AUDIO_END=102、REF_START=103、REF_END=104。
これは使用text tokenizerとの契約なので、別tokenizerを作るならID配置も一貫して変更する。
「0埋めaudio trackにtext token 0があるからtext位置」といった値依存判定は使わない。

## 9. 推論solverとstreaming

```text
for each generated patch i:
    prepare semantic/residual state from observed history
    x = Gaussian noise [B,64,4]
    times = descending schedule 1→0
    apply sway to schedule
    for each local solver step:
        compute conditional / LM-unconditional velocity
        optionally apply CFG-Zero* scaling
        x -= dt * guided_velocity
    emit patch [B,4,64]
    AudioVAE decoder may emit waveform chunk
    evaluate stop, then update acoustic history
```

swayはstepを高noise側へ再配置するtime schedule。
CFG-Zero*は初期stepをゼロfieldにし、unconditional fieldを最適scaleで補正する。
どちらも学習可能な追加重みではない。
step数1でCFG-Zero*を有効にすると、その唯一のstepがゼロになるので、実用では複数stepを使う。

本版は学習・処理理解を優先してTSLM/RALMのKV cacheを実装せず、prefixを再計算する。
音声値を省略しているのではないが、公式servingの速度を再現する実装ではない。
decoderの状態を保持するstreaming処理は、公式の`streaming_decode()`が提供する。
全latentを一度にdecodeする本版adapterと、stateful chunk APIは区別する。

## 10. データ・学習curriculum

基礎データは音源分離、VAD、ASR alignment、品質filterを通した200万時間超。
ZH/ENが多数で、残り28言語は概ね1K〜50K時間と報告される。
voice/style annotationは年齢・gender・accent・声質・場面、感情・速度・pitch・energy等を自然言語化する。

3段階でobjectiveは共通、データ配分と長さを変える。

| 段階 | 主な内容 | 長さ |
| --- | --- | --- |
| multilingual pretraining | transcription/audio、basic/continuation | 最大60秒、4096位置 |
| controllable joint pretraining | plain TTSを残してdescription/referenceを徐々に増やす | 最大3分、8192位置 |
| high-quality SFT | 高品質・表現豊かな実録音、言語balanced sampling | 2秒〜5分、8192位置 |

referenceはspeaker embedding cosine>0.7等で同一話者を採掘し、直前segmentを除く。
styleと本文内容の相関を減らすため、同じstyleで異なる本文を合成したデータをstage2へ追加する。
最後のSFTはnative recording中心にして、自己生成artifactの蓄積を抑える。

## 11. 論文の結果と制約

| 評価 | VoxCPM2の報告 |
| --- | --- |
| Seed EN WER / SIM | 1.84% / 75.3% |
| Seed ZH CER / SIM | 0.97% / 79.5% |
| Seed ZH-hard CER / SIM | 8.13% / 75.3% |
| 内部30言語平均WER | 1.68% |
| RTX4090 PyTorch RTF | 0.30、約8GB |
| RTX4090 Nano-vLLM RTF | 0.13、約8GB |

全benchmarkで最低WERという論文ではない。
例えばSeedのFish S2の0.99% EN WER、0.54% ZH errorはこの表のVoxCPM2より低い。
VoxCPM2の特徴は連続latent、話者類似度、多様なcontrol、低いLM token rateの組合せ。
内部30言語評価はpublic Seed/CV3とデータが違うので、その数字を直接比較しない。

48kHz出力は25Hz latentの時間解像度が上がったことを意味しない。
4frame/160msのpatch局所生成には、極端に細かいprosody制御とのtrade-offがある。
speaker reference・style instructionの分離も、完全な因果的独立性を保証しない。

## 12. 実装ファイル

| ファイル | 主な内容 |
| --- | --- |
| [transformer.py](transformer.py) | MiniCPMのGQA/RMSNorm/SwiGLU/LongRoPE/muP |
| [local_modules.py](local_modules.py) | patch CLS encoder、FSQ、3prefix LocDiT |
| [flow_matching.py](flow_matching.py) | log-normal time、CFM/optional mean-flow、CFG/sway/Zero* |
| [model.py](model.py) | TSLM/RALM融合、教師shift、stop loss、実AR生成 |
| [data.py](data.py) | reference対応packer、DataFrame Dataset、mask付きcollate |
| [audio_vae.py](audio_vae.py) | 公式AudioVAE本体のencode/decode adapter |
| [main_flow.py](main_flow.py) | 生成prefixから48kHz波形までの接続 |
| [training_example.py](training_example.py) | 基本PyTorch学習loop、optimizer、scheduler、checkpoint |
| [prepare_data.py](prepare_data.py) | 音声pathと本文から公式VAE latentと本文ID、学習Parquet/configを生成 |
| [pretrained_inference.py](pretrained_inference.py) | 公式VoxCPM2 pretrainedモデルによるTTS・cloning・continuationとWAV保存 |

AudioVAEは独立した約数百行のcausal/streaming subsystemなので、公式本体を参照する。
小さなConvTranspose decoderを同等品として置いていない。
backboneは本ディレクトリで定義したmoduleを実際に使い、ランダム初期化で学習可能。
既存checkpointとの自動weight-key変換やbit一致確認を行ったとは主張しない。

## 13. DataFrame学習例

### 生音声と書き起こしからの準備

中間メタデータは**Parquet（pyarrow、ZSTD圧縮）**で保存する。
embedding・latent・IDなどの数値配列は**Tensor単体の`.pt`**へ保存し、
dtype/shapeを保持する。CSVへ配列を文字列化して格納しない。
元の音声path・本文表はCSV/Parquetの両方から読める。`pyarrow`が必要だが、
自動インストールやCSVへのfallbackは行わない。

必要な表の列は`audio_path`と`text`だけ。話者IDと感情ラベルは任意で、
存在しないまま[prepare_data.py](prepare_data.py)を使える。
各行は対応する1発話の音声と全文とし、音声の相対pathはメタデータファイルの親基準。

```csv
audio_path,text
audio/0001.wav,こんにちは。
audio/0002.wav,今日は晴れています。
```

前処理は音声をmono/16kHzへ変換し、固定した公式AudioVAE V2で
latent `[N,4,64]`を抽出する。`N`=patch数、4=patch内frame数、64=特徴次元数。
本文は公式のmulti-character mask付きtokenizerでIDs `[L]`へ変える。
`L`=本文token数。追加教師としてF0や感情カテゴリを要求しない。
**後述の`text_ids_path`/`latent_path`はこの処理が生成する列であり、利用者の前提データではない。**

```bash
python prepare_data.py --metadata data/raw.csv --model-dir checkpoints/VoxCPM2 --output data/cache --device cuda:0
python training_example.py --metadata data/cache/train.parquet --config data/cache/config_scratch.json --output checkpoints/voxcpm2.pt
```

公式packageまたは`--repo`で指定したrepoとその依存、ローカルのtokenizer/config/
VAE重みが必要。LM重みはロードしない。VAEは固定し、scratch学習対象はbackbone。
生成configには実語彙と境界IDを反映し、LM特徴次元数/層数は小型実験用の値を明示する。
公式容量を再現する設定ではない。長い音声は無断で切らず、既定30秒を超えたら拒否する。
末尾はpatch境界へzero paddingし、実16kHz sample数をCSVへ記録する。

この最小表からはreferenceなしの通常TTS用データを作る。
同一話者の別発話referenceを選ぶには、その対応関係が必要であり、
この前処理では話者の同一性を推定せず、感情指示も本文へ追加しない。
したがって、これだけでinstruction/style制御学習まで実施したとは扱わない。
出力Parquetのpathはメタデータファイルの親基準で学習CLIが解決する。Pythonからも
`load_metadata()`で読める。自動download、追加install、作成時のモデル実行はしていない。

| 列 | 保存内容 |
| --- | --- |
| `text_ids_path` | tokenizer済み本文 `[L]` int64。境界IDはpackerが追加 |
| `latent_path` | target AudioVAE mean latent `[N,4,64]` float |
| `reference_path` | 任意の同一話者reference latent `[Nr,4,64]` float |

raw waveformをbackbone学習中に毎回VAE encodeせず、固定AudioVAEでcacheを作る。
同じAudioVAEとtext tokenizerを推論にも使う。
referenceが別話者ならモデル構造が正しくてもcloning教師が矛盾する。

```python
import json
from model import config_from_dict
from training_example import load_metadata, train

# 小型で同じ計算構造を学習する設定例。論文2Bの容量とは異なる。
with open("data/cache/config_scratch.json", encoding="utf-8") as handle:
    config = config_from_dict(json.load(handle))
model = train(load_metadata("data/cache/train.parquet"), config, "checkpoints/vox.pt")
```

公開modelの実configへ合わせる場合はhead数、kv_channels、RoPE factor、
scale_emb、scale_depth、tokenizer語彙、FSQ scaleも指定する。
層数と隠れ状態の次元数だけを合わせてもすべて同一configurationにはならない。

trainerはreference/text/targetのmaskを別々に保持し、paddingはlossへ加えない。
参照音声は条件として使用し、音声生成lossの対象から除外する。
stop labelはtarget最後のpatchへ付け、text末尾やAUDIO_ENDへ付けない。

同梱 [config_example.json](config_example.json) は上の小型構成に対応する。

```bash
python training_example.py --metadata data/train.parquet --config config_example.json
```

backbone学習にはPython 3.10以降、PyTorch、einops、pandasを使う。
optional mean-flowは`torch.func.jvp`とmath SDPA backendを必要とする。
AudioVAE本体のrandom構築は、公式packageをimport可能にして次のように行える。

```python
from voxcpm.modules.audiovae import AudioVAEV2, AudioVAEConfigV2
from audio_vae import AudioVAEAdapter

vae = AudioVAEV2(AudioVAEConfigV2())  # random weights。downloadしない
adapter = AudioVAEAdapter(vae)
```

random VAEの出力を、そのまま学習済みの良い音声latentと扱ってはいけない。
実際のbackbone学習には、事前学習を完了した同一VAEを固定して使う。
これらのコマンド・構築例は作成時に実行しておらず、追加インストールもしていない。

### 13.1 公式pretrainedモデルで推論する

[pretrained_inference.py](pretrained_inference.py) は公式 `voxcpm.VoxCPM.from_pretrained()` を使う。
公式 `openbmb/VoxCPM2` のモデル一式を事前にローカルへ配置する。
`local_files_only=True`、denoiser無効、compile/warmup無効でロードする。
scratch用の `model.py` は読み込まない。

```bash
# 通常TTS。本文先頭に(声やstyleの説明)を加えればvoice designも同じAPIで使える。
python pretrained_inference.py --model-dir checkpoints/VoxCPM2 \
  --text 'こんにちは。' --output outputs/vox.wav

# transcript不要のreference cloning。
python pretrained_inference.py --model-dir checkpoints/VoxCPM2 \
  --reference data/speaker.wav \
  --text '(穏やかな口調で)こんにちは。' --output outputs/vox_clone.wav

# continuation。prompt-textはprompt音声に実際に含まれる本文を指定する。
python pretrained_inference.py --model-dir checkpoints/VoxCPM2 \
  --prompt-audio data/prompt.wav --prompt-text '参照音声に含まれる本文。' \
  --text 'ここから新しく生成する本文です。' --output outputs/vox_continue.wav
```

referenceとcontinuationは併用可能。
公式packageがinstallされていない場合は `--repo vendor/VoxCPM` を追加する。
`--cfg 2.0 --steps 10 --seed 42` でguidance、local flow step数、seedを指定できる。
必要な依存は公式VoxCPMとsoundfile。モデル出力の実sample rateを取得してWAV保存する。
既存出力は `--overwrite` 指定時だけ置換する。
本サンプルの実行・モデル取得・追加インストールは今回行っていない。

## 14. 確認状態と一次資料

ユーザー指定により、import、compile、学習、推論、速度測定を含む動作確認は行っていない。
TeXとソースの静的読解に基づく実装であり、実行済み・バグ不存在とは記さない。
論文は `main.tex` のテキストだけを参照した。

- [VoxCPM2 Technical Report](https://arxiv.org/abs/2606.06928)
- [公式repository](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM2本体](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/model/voxcpm2.py)
- [公開CFM](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/modules/locdit/unified_cfm.py)
- [公開Local DiT V2](https://github.com/OpenBMB/VoxCPM/blob/cce58a5b59303c9bd63d12f23afe6a49a4a80c59/src/voxcpm/modules/locdit/local_dit_v2.py)

参照revisionは `cce58a5b59303c9bd63d12f23afe6a49a4a80c59`。
本文の層数・特徴次元数・objectiveと、公開クラスのdefault値／optional機能は区別した。
コードとモデルは公式licenseの条件も確認する。
