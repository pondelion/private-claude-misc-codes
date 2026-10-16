# dLLM / Diffusion Language Model 論文・公式実装サーベイ

## 1. このsurveyの対象

ここでいう **dLLM** には2つの意味がある。

1. **Diffusion Language Model (DLM)**: token列を左から1個ずつではなく、MASKだらけの列から反復的に復元する生成モデルの総称
2. **dLLM framework**: MDLM、BD3-LM、LLaDA、Dream、EditFlow、Fast-dLLM、A2D、評価などを共通interfaceで扱う公式library

論文は新しい単一model architectureの提案というより、DLM研究を`Trainer`、`Sampler`、`Evaluation`へ整理し、
既存DLMの学習・変換・推論を再現可能にするtechnical reportである。本surveyはframework APIの列挙ではなく、
その中心にある **何を壊し、何を予測し、何をGTとして、どうlossにするか** を理解することを主眼にする。

## 2. まず全体像

token列を `x0=(x0^1,...,x0^L) in V^L`、特殊tokenを`[MASK]=m`、noise時刻を`t in [0,1]`とする。

```text
clean x0:   私 は 猫 が 好き EOS
                  forward corruption q(xt|x0), t=0.5
noisy xt:   私 [M] 猫 [M] [M] EOS
                  bidirectional Transformer
prediction:       は      が  好き       <- 各MASK位置のclean token分類
                  iterative reverse process
sample:      私 は 猫 が 好き EOS
```

画像diffusionのように連続値へGaussian noiseを加えるのではなく、MDLMは離散tokenを吸収状態`[MASK]`へ
置換する。学習時は通常のdiffusionと同様に、各sampleについて全時刻範囲`t in [0,1]`から時刻を1つだけ
ランダムに選ぶ。その`t`が指定するmask率でclean列`x0`を1回だけ破壊して`xt`を作り、modelには
`xt`からMASK位置のclean token `x0`を予測させる。後述する純粋なabsorbing-mask型MDLMでは`t`自体を
modelへ入力する必要はない。1 training step内で
`x1 -> ... -> x0`というreverse process全体を実行・backpropagateするわけではない。多数のsampleとstepで異なる
`t`を引くことにより、modelは結果的に全noise levelのdenoisingを学習する。

推論時は通常のdiffusionと同様に、強く破壊された状態（生成領域がほぼ、または全てMASK）から開始し、modelを
複数回呼んで徐々にcleanなtoken列へ戻す。ただしGaussian diffusionで数値noiseを少しずつ除く代わりに、MDLMでは
各反復でMASK位置のtoken候補とconfidenceを予測し、選ばれた位置をMASKから通常tokenへ確定していく。実装によっては
低confidence位置を再びMASKへ戻す。したがって概念は逐次的なreverse denoisingだが、状態遷移は連続値の更新ではなく
離散tokenのunmask/remaskである。

## 3. AR、BERT MLM、MDLM、BD3-LMの違い

| 方法 | 学習時に見える情報 | 予測対象 | 生成方法 | 主な性質 |
|---|---|---|---|---|
| Autoregressive LM | 左側tokenのみ | 次token | 左から1 tokenずつ | likelihood明瞭、KV cache容易 |
| BERT MLM | 左右のvisible token | 選ばれたMASK | 通常は生成modelではない | 固定mask率が一般的 |
| MDLM | 左右のvisible/復元token | MASK位置のclean token | 全出力領域を反復復元 | 任意順・並列commit可能 |
| BD3-LM | 過去block + 現在block内の左右 | 現在blockのMASK | blockは左から、内部は反復復元 | 可変長・KV cacheとの折衷 |

MDLMは単なる「BERTを何度も呼ぶ」ではない。連続時刻`t`でmask率を変え、その時刻に対応した重みで
denoising lossを積分することにより、全mask率にわたる生成reverse processを学ぶ。一方、実用上は
BERTやAR checkpointをこのobjectiveで追加学習してDLM化できることも論文の重要なrecipeである。

## 4. MDLMのforward process

linear scheduleでは各tokenを独立に確率`t`でMASKする。

```math
q(x_t^i | x_0^i)
= (1-t) I[x_t^i=x_0^i] + t I[x_t^i=m].
```

一般scheduleでは`alpha(t)`をcleanなまま残る確率とし、mask確率は`1-alpha(t)`になる。公式実装は
sampleごとに`t_b ~ Uniform(epsilon,1)`を1つ引き、教師対象位置だけをBernoulli maskする。
prompt、system message、paddingなど`labels=-100`の位置は壊さない。

```text
input_ids x0       : (B,L)    clean prompt + response
eligible           : (B,L)    response/GT位置だけTrue
t                  : (B,)     sampleごとのnoise時刻
p_mask             : (B,L)    1-alpha(t_b)を列方向へbroadcast
masked             : (B,L)    Bernoulli(p_mask) AND eligible
noisy input xt     : (B,L)    masked位置だけMASK id
```

[noise_schedule.py](noise_schedule.py) がlinear/cosine schedule、[mdlm.py](mdlm.py) がこのcorruptionを実装する。

### 4.1 なぜnoise時刻`t`をmodelへ入力しなくてもよいのか

公式MDLM実装では`t`をtime embeddingや特殊tokenとしてmodelへ渡していない。`t`は`x0`から`xt`を作るときの
mask率と、lossの重みを決めるためにだけ使う。modelが直接学ぶ分布は`p_theta(x0 | xt)`である。

これは純粋なMASK corruptionでは、壊れた位置が`[MASK]`として明示され、maskされる確率も元tokenの値に
依存しないためである。ある`xt`と整合するclean列`x0`について、linear scheduleのforward確率に含まれる
次の部分はすべての候補`x0`に共通する。

```math
t^{N_mask} (1-t)^{N_visible}
```

したがってBayes則でposteriorを正規化すると、この`t`依存項は分子・分母で打ち消される。

```math
p(x0 | xt, t) = p(x0 | xt)
```

直感的には、`[MASK]`が「この位置は完全に情報を失った」、通常tokenが「この位置は壊れていない」と既に示すため、
clean tokenを予測するdenoiserへnoise強度を別入力する必要がない。Gaussian diffusionでは観測値だけからnoise強度を
一意に判別できないため、通常は`t` conditioningが必要になる。random replacementなど、破壊された位置が入力から
判別できない別のdiscrete corruptionでは、同様に`t`を省略できるとは限らない。

### 4.2 それなら単なるMasked Language Modelingではないか

1回の学習処理だけを見れば、MDLMは実際に「様々なmask率で行うMasked Language Modeling」である。

```text
tをsample -> tに応じてMASK -> MASK位置の元tokenをcross entropyで予測
```

diffusion modelとしての違いは、主に生成過程と学習するnoise levelの範囲にある。通常のBERT MLMは固定に近いmask率で
欠損補完能力を学ぶが、MDLMはcleanに近い状態から全MASKに近い状態までのdenoisingを学び、推論時には全MASK状態から
modelを反復適用して文章を生成する。

```text
ほぼ全MASKのx_T -> 一部をunmask -> 残りをunmask -> ... -> cleanなx_0
```

したがってMDLMは、学習lossの形だけなら可変mask率MLM、生成modelとして見ればMASKによるforward corruptionと
反復reverse denoisingを持つdiscrete diffusionである。confidence順にtokenを確定する実装は、Gaussian diffusionの
厳密な数値solverというより、diffusion objectiveで学習した反復mask-predict生成に近い。

## 5. 学習objective: 入力・予測・GT・loss

### 5.1 何を入力し、何を予測するか

instruction tuningの1 sampleを`prompt P`と正解response`Y`とする。

```text
clean:       [BOS P SEP | y1 y2 y3 EOS]
loss対象:     False...  | True........
noisy input: [BOS P SEP | M  y2 M  EOS]  <- modelへの入力
GT:                       y1    y3        <- clean x0の同じ位置
logits:      (B,L,V), MASKになったresponse位置だけCEへ入れる
```

modelはnoise自体や「MASKか否か」を予測するのではなく、各MASK位置に元々あった **clean vocabulary token**
`x0^i`を分類する。別のdiffusion-step headや回帰lossはない。visible response tokenはcontextとして使うが、
その試行のCEには入らない。promptは常に条件として見えるがcorruptionにもCEにも入らない。

### 5.2 なぜ1/tを掛けるか

linear scheduleの論文objectiveは次である。

```math
L_MDLM = E_{t,x0} [ (1/t) sum_{i in M_t} -log p_theta(x0^i | x_t) ].
```

位置`i`がlossに現れる確率は`t`なので、indicatorを`1/t`倍すると期待値が1になる。低noise時刻ではMASKが
少ないため、そこに大きな重みを与えて時刻間の寄与を補正する。一般の`alpha(t)`では公式実装は

```math
w(t) = -alpha'(t) / (1-alpha(t))
```

を使用する。linearなら`alpha=1-t`なので`w(t)=1/t`である。実装上は`t=0`の発散を避けるため
`epsilon=1e-3`からsampleし、分母にも微小値を加える。

公式`loss_norm_type="token"`に合わせ、加重CE和は「その回にMASKされた数」ではなく全eligible token数で割る。
これによりmask率の違うbatchでもMonte Carlo estimatorの意味が変わらない。[mdlm.py](mdlm.py) の
`mdlm_loss`がshapeを含めて対応する。

### 5.3 AR checkpointをMDLMへ変換するright shift

通常AR modelのposition `i`のlogitはtoken `i+1`を予測するよう学習済みだが、MDLMはposition `i`で
token `i`を復元する。公式A2D recipeは入力にBOSをprependし、logitsを1位置右へずらす
`right_shift_logits=True`を使う。これによりarchitectureを改造せずnext-token能力を再利用できる。
scratchのbidirectional modelには不要である。

## 6. model architecture

DLMを成立させる本質は特定の新規layerではなく、同じTransformerを **非causal attention** で使い、
MASK位置ごとにvocabulary logitsを出す点にある。

```text
token ids       (B,L)
 -> token + position embedding
hidden          (B,L,D)
 -> N x [bidirectional self-attention + MLP]
hidden          (B,L,D)
 -> vocabulary head
logits          (B,L,V)
```

[model.py](model.py) はPyTorchだけの小型実装である。これはLLaDA等の数十億parameterを複製するものではないが、
全位置logits、bidirectional attention、任意のBD3 maskというobjectiveに必要な処理を省略せず実装している。

## 7. BD3-LM: block diffusion

MDLMは長さ`L`全体を双方向に扱うため、ARのように確定prefixのKV cacheを素直に再利用しにくく、生成長も
事前確保しがちである。BD3-LMは列をサイズ`S`のblock `B1,...,BK`へ分ける。

```math
p_theta(x) = product_k p_theta(x^{B_k} | x^{<B_k}).
```

block間はautoregressive、現在block内だけdiffusionにする。学習時の公式実装はnoisy列`x_t`とclean列`x_0`
を連結する。

```text
concat input: [x_t^B1 x_t^B2 ... | x_0^B1 x_0^B2 ...]  shape (B,2L)
position ids: [0..L-1             | 0..L-1]             shape (B,2L)
logits:                                                   (B,2L,V)
lossに使う:   noisy halfのみ                              (B,L,V)
```

attentionの許可関係は次の通り。

| query | 読めるkey | 理由 |
|---|---|---|
| noisy block k | 同じnoisy block k全体 | block内denoising |
| noisy block k | clean block `< k` | 確定済みhistoryで条件付け |
| noisy block k | clean block `>= k` | **禁止**。GT leakになる |
| clean block k | clean block `<= k` | block-causalなclean表現 |
| clean stream | noisy stream | 禁止 |

[bd3lm.py](bd3lm.py) はこの`(2L,2L)` boolean maskを公式の3項
`block diagonal + offset block causal + block causal`どおり構築する。推論時は完成した過去blockを固定し、
次のblockをMASKで追加して内部だけ反復復元するため、過去blockのKV cacheが可能になる。

## 8. 推論: MASKから文章を作る

### 8.1 MDLM sampling

1. promptの後ろへ`G`個のMASKを置く: canvas `(B,P+G)`
2. 全canvasをbidirectional forwardしlogits `(B,P+G,V)`を得る
3. 各MASK位置のcandidate tokenをargmaxまたはGumbel-maxで選ぶ
4. candidate確率をconfidenceとし、予定数の上位位置をcommitする
5. 未確定位置をMASKのまま残し、全て確定するまで反復する

左から順とは限らず、容易なtokenから確定できる。一方、confidenceは正しさの保証ではなく、誤って早く
commitしたtokenを固定する場合もある。公式samplerの`remasking="random"`はconfidenceの代わりにランダム順を
使う。temperature、step数、block size、commit schedule、CFG、suppress tokenに結果が強く依存する。

### 8.2 BD3-LM sampling

BD3-LMは次blockだけをMASKで初期化し、そのblockを複数stepで完成させてから次へ進む。block size 1なら
ほぼAR、出力全長ならMDLMに近い。blockを大きくすると並列性が上がる一方、1 forwardの計算量と事前確保が増える。
[sampling.py](sampling.py) に両samplerと全中間canvas`histories`を実装した。

### 8.3 EOSと生成長

MDLMは最初に生成領域を確保するため「いつ止めるか」がARほど自明ではない。公式SFT recipeではpadding部分を
EOSでlabelし、余分なMASKからEOSを予測するよう学ぶ。decode後は最初のEOS以降をtrimする。BD3-LMはblockを
逐次追加できるため、EOSを含むblockの生成後に停止できる。

## 9. dLLM frameworkの構成

### Trainer

- `MDLMTrainer`: mask生成、schedule weight、masked CE、SFT、pretraining
- `BD3LMTrainer`: 同じlossに二重系列とblock attentionを追加
- `EditFlowTrainer`: mask置換だけでなく挿入・置換・削除を扱う別系統
- Hugging Face Trainer / Accelerate / PEFT上に薄く実装され、FSDPやDeepSpeedへ拡張可能

### Sampler

`Sampler(model).sample()`でmodel固有実装とdecode algorithmを分離する。vanilla MDLM/BD3-LMに加え、
Fast-dLLM、LLaDA、Dream等のsamplerを差し替えられる。中間履歴の可視化は、生成順序が固定されないDLMの
debugに重要である。

### Evaluation

DLMはdecode hyperparameter感度が高いため、model weightだけを比較しても公平にならない。論文は統一evaluation
harnessと設定記録を重視し、LLaDA/Dreamで単一parameter変更でも性能が大きく落ちうることを示す。

## 10. 主要recipeと論文から得られる知見

### Open-weight DLMのSFT

LLaDAやDreamを通常の会話DataFrameからfinetuneできる。attention maskを安易にpadding maskとして渡すと、
EOS paddingを生成対象として扱うSFT設計と衝突するため、公式にはcollator wrapperが用意されている。

### BERT-to-chat

pretrained BERTは既に双方向MASK復元能力を持つ。可変mask率、生成用特殊token、chat SFTを加えることで、
大規模な継続pretrainingなしでもgenerative DLMへ変換できることをrecipeとして示す。

### Tiny-A2D: AR-to-Diffusion

Qwen3-0.6Bをarchitecture変更なしにMDLM/BD3-LMへSFT変換する。報告例ではGSM8KがMDLM 29.3、BD3-LM
46.3、HumanEvalが30.5/46.3で、block化が全taskで常に優位ではないが、ARの知識をDLMへ移せることを示す。
ただし元のAR baseを多くの指標で超える結果ではなく、「変換可能」と「品質を維持」は区別すべきである。

### Fast-dLLM

DLMは反復forwardの回数がbottleneckになる。Fast-dLLMはconfidence-aware parallel updateとblock単位の近似KV
cacheで高速化する。ただしcurrent block内はtokenが毎step変わるため、厳密なKV再利用ではなく近似を含む。

## 11. DataFrame finetune example

[finetuning_example.py](finetuning_example.py) は`prompt`, `response`列を持つDataFrameから次を実行する。

```text
DataFrame row
 -> [BOS, prompt, SEP, response, EOS] token化
 -> response/EOSだけeligible=True
 -> random tとMASK corruption
 -> bidirectional Transformer forward
 -> scheduler-weighted clean-token CE
 -> backward, gradient clipping, AdamW update
 -> 学習modelを実際のMDLM samplerで復号
```

外部DLM frameworkやmock modelには依存しない。文字tokenizerはsampleを単独実行可能にするための選択であり、
実データではsubword tokenizerに置換する。短いtoy dataの生成品質は保証しないが、forward/backward/update/decode
は全て実処理である。

## 12. 実装上の落とし穴

- `alpha(t)`をmask確率と誤読しない。公式ではclean保持確率でありmask率は`1-alpha(t)`。
- lossを全tokenへ掛けない。原則としてその試行で実際にMASKされた教師対象だけ。
- 分母を毎回のMASK数に変えると`1/t` estimatorの意味が変わる。
- promptまでcorrupt/教師化するとconditional generationのobjectiveではなくなる。
- bidirectional attentionでclean GTを入力したまま同位置CEを取ると答えが丸見え。GT位置は必ずMASKにする。
- BD3 trainingでcurrent/future clean blockがnoisy queryから見えると重大なlabel leakage。
- AR変換時のBOS/logit shiftを忘れるとpositionが1つずれる。
- sampling step数を増やせば常に改善するとは限らず、学習時noise分布との整合も必要。
- 全MASKを一度にargmaxして終了するのはiterative diffusionではない。
- DLMの「並列生成」は1 forwardで全文完成という意味ではない。複数位置を同時commitする複数step生成である。

## 13. このディレクトリの構成

- [noise_schedule.py](noise_schedule.py): `alpha(t)`、mask率、loss重み、reverse確率
- [model.py](model.py): 任意attention mask対応の小型bidirectional Transformer
- [mdlm.py](mdlm.py): forward corruption、right shift、MDLM loss
- [bd3lm.py](bd3lm.py): 二重系列BD3 attentionとtraining loss
- [sampling.py](sampling.py): confidence-guided MDLM/BD3-LM sampling
- [main_flow.py](main_flow.py): 両objectiveのbackwardと両samplerを一巡
- [finetuning_example.py](finetuning_example.py): DataFrameを用いた基本PyTorch finetune

```bash
cd survey/dLLM
python noise_schedule.py
python model.py
python mdlm.py
python bd3lm.py
python main_flow.py
python finetuning_example.py
```

## 14. 再現範囲

教材コードはMDLM/BD3-LMのcorruption、attention、objective、optimizer update、iterative samplingを再現する。
大規模model architecture、distributed training、Flash/Flex Attention、CFG、Fast-dLLM cache、benchmark harnessを
短いsurvey codeへ複製してはいない。これらは理解対象のcore algorithmとは独立で、checkpoint依存・GPU kernel依存が
大きいためである。READMEでは公式実装上の役割を説明し、core処理は省略せずPyTorchで動かしている。

## 15. 一次資料

- [dLLM technical report](https://arxiv.org/abs/2602.22661)
- [公式GitHub](https://github.com/ZHZisZZ/dllm)
- [公式model collection](https://huggingface.co/dllm-hub)
- [MDLM: Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [Block Diffusion](https://arxiv.org/abs/2503.09573)
