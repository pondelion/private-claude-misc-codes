# LoopVL 論文・公式実装サーベイ

## 1. LoopVLは何を提案したか

LoopVLは、同じTransformer parameterを繰り返し適用して実効depthを増やすrecurrent Transformerを、
画像と言語を同時に扱うVision-Language Modelへ拡張したモデルである。約1B parameterのlanguage backboneは
異なるparameterを持つ16-layerのL moduleと16-layerのH moduleだけを保持する。一方、標準H2L3 scheduleでは

```text
L -> L -> L -> H -> L -> L -> L -> H
```

と8回moduleを呼ぶため、1 forwardで`8 * 16 = 128` Transformer layer applicationsを実行する。同じL stack、
同じH stackを再利用するのでparameter数を8 stack分には増やさず、計算depthだけを増やす。

通常のunified-token VLMでも、visual tokenはTransformer layerを通るたびに更新される。したがって、visual tokenを
複数layerで更新すること自体はLoopVLの新規性ではない。LoopVLの差分は、異なるparameterを持つlayerを深く積む代わりに、
同じL/H parameter stackを更新済みstateへ何度も適用する点にある。vision encoderは最初に画像patch特徴を1回だけ作り、
後のcycleでは同じphysical layerが、既に変化したvisual-language stateを再処理する。

## 2. Module-LoopとModel-Loop

LoopVLは2階層のrecurrent stateを持つ。

- `z_L`: fine-grainedなlocal state。H stateを条件として短い間隔で何度も更新する
- `z_H`: high-level state。複数回のL更新を統合し、次cycleの条件を作る

H2L3の`H=2`はmodel cycle数、`L=3`は各cycle内のL invocation数を表す。

```text
初期化:
  z_H = input embeddings                         (B,S,D)
  z_L = 0                                        (B,S,D)

model cycle 1:
  z_L = L(z_L + z_H)    # 同じL parameterで3回
  z_L = L(z_L + z_H)
  z_L = L(z_L + z_H)
  z_H = H(z_H + z_L)

model cycle 2:
  z_L = L(z_L + z_H)    # z_Lもzeroへ戻さず保持
  z_L = L(z_L + z_H)
  z_L = L(z_L + z_H)
  z_H = H(z_H + z_L)

出力:
  logits = LMHead(z_H)                           (B,S,V)
```

ここで`L(...)`と`H(...)`は1 layerではなく、それぞれ16 Transformer layersからなるmodule全体である。Lの3回は
parameterを共有し、2つのmodel cycle間でも共有する。LとH同士は別parameterである。

Module-Loopは比較的安定したH条件の下でLを反復更新する内側loop、Model-LoopはHまで更新した状態でもう一度
L/H計算を行う外側loopである。[recurrent_backbone.py](recurrent_backbone.py)がこの状態遷移を実装する。

## 3. 入力からlogitsまでのshape

`B`: batch、`N`: visual token数、`S`: visual/textを結合した系列長、`D=1536`、`V`: 語彙数とする。

```text
image
  -> Penguin Vision Encoder（frozen）
patch features                                  (B,N,1024)
  -> Linear(1024,1536) -> GELU -> Linear(1536,1536)
  -> parameter-free RMS norm + scale calibration
projected visual embeddings                    (B,N,1536)

[header | reserved visual slots | instruction | response]
token IDs                                       (B,S)
  -> token embedding                            (B,S,1536)
  -> reserved visual slotsをprojected featuresで置換
unified multimodal embeddings                   (B,S,1536)
  -> H2L3 recurrence: 128 layer applications
final H state                                   (B,S,1536)
  -> shared LM head
vocabulary logits                               (B,S,V)
```

公式のdynamic resolutionでは1画像当たり256--1024 visual tokensを使い、aspect ratioに応じてgrid height/widthが
変わる。Penguinは各patchにつき1024-dimensional featureを返す。vision encoderは全multimodal stageでfreezeされ、
visual interfaceとlanguage backboneを学習する。

[visual_interface.py](visual_interface.py)は公式projectorの処理を実装する。教材版のvision encoderは画像を実際に
patch化する小型frozen Convであり、Penguinのpretrained能力をmockするものではない。Penguin本体はcheckpointと
外部実装に依存するため複製せず、LoopVL固有のprojector以降を検証可能にしている。

## 4. 通常のVLMとの差分: shared moduleによるvisual stateの再処理

visual tokenをlayerごとに更新すること自体はLoopVL固有ではない。visual tokenをlanguage Transformerの系列へ挿入する
通常のVLMでも、各layerのattentionとFFNによりvisual hidden stateは順次更新される。

```text
通常の32-layer VLM:
v0 -> Layer1(v0) -> Layer2(v1) -> ... -> Layer32(v31)
      ^^^^^^^^^^^    ^^^^^^^^^^^
      layerごとに異なるparameter

LoopVL H2L3:
v0 -> L(v0) -> L(v1) -> L(v2) -> H(v3)
   -> L(v4) -> L(v5) -> L(v6) -> H(v7)
      ^^^ 同じ16-layer L stackを6回共有
                                  ^^^ 同じ16-layer H stackを2回共有
```

したがってLoopVLの新規性と分析上のポイントは次である。

- 32 unique layers分のparameterしか持たないが、L/H stackを再利用して128 layer applicationsを実行する
- L/Hのpersistent stateを保持し、更新済みvisual-language stateを次のinvocationへ渡す
- model cycleの開始時に、初期projected visual embeddingをgate付きanchorとして再注入する
- 同じphysical layerのcycle 1とcycle 2を比較できるため、parameter差ではなく入力stateの変化に伴うattention再配分を観測できる

最後の点がVisual Aha Momentの分析に重要である。通常のVLMでLayer 16とLayer 32のattentionが異なっても、両layerは
parameterが異なるため、変化がhidden stateとparameterのどちらに由来するかを分けにくい。LoopVLでは同じL/H layerを
cycle間で揃えて比較できるので、「同一変換が、異なるrecurrent stateを受け取ったときにvisual evidenceの読み方をどう
変えるか」を観測できる。

なお、vision encoderをloopごとに再実行するわけではない。最初に得たpatch featuresをprojectして系列へ挿入し、その後は
language backbone内のvisual hidden stateを更新する。この点も、多くの通常VLMと共通する処理であり、それ自体を
LoopVLの新規性とはみなさない。

## 5. Visual anchorとquery-conditioned gate

反復更新だけでは初期画像情報が薄れる可能性があるため、各model cycleの開始時に、projector直後のvisual embeddingを
固定anchorとしてH stateのvisual位置へ足す。

visual token `i`、cycle `c`について概念的には次である。

```text
instruction_state = mean(H stateのinstruction positions)       (B,D)
gate_i = 2 * sigmoid(MLP(visual_state_i, instruction_state))    (B,N), 0<gate<2
alpha_c = 0.5 * sigmoid(raw_alpha_c)                            scalar, 0<alpha<0.5
reinjection_i = alpha_c * gate_i * initial_visual_anchor_i      (B,N,D)
H_visual_i <- H_visual_i + reinjection_i
```

gateは質問に応じてpatchごとの再注入量を変える。出力範囲が`(0,2)`なので初期値1がneutralであり、公式実装はgate
最終linearをzero初期化する。`alpha_c`はcycleごとのlearnable scaleである。anchorは現在のvisual stateそのものではなく、
最初のprojected visual embeddingを保持した値である。

論文はanchor/gateを除いた別学習modelでもcross-cycle attention shiftが現れたと報告している。したがってVisual Aha
Momentをanchor機構だけの効果とは主張していない。

## 6. Attention mask: PrefixLM

LoopVLは通常の完全causal LMではない。header、visual token、instructionを1つのconditioning prefixとし、その中は
bidirectionalにする。responseだけがautoregressiveである。

| query位置 | 読めるkey |
|---|---|
| header/image/instruction prefix | prefix全体 |
| response token `y_t` | prefix全体とresponse `y_<=t` |
| prefix | responseは読めない |

```text
sequence: [header | image | instruction | y1 | y2 | y3]

prefix query:   [可 可 可 | 不 不 不]
y2 query:       [可 可 可 | 可 可 不]
```

これによりvisual token自身もinstructionを双方向に参照して、質問に応じたstateへ更新できる。一方でresponseの未来tokenを
見せないため、generation時と同じ条件でnext-token predictionを学習できる。[attention.py](attention.py)がbool maskを
実装する。

## 7. Hybrid 2D RoPE

visual tokenは画像gridのrow/columnを保持する必要がある。LoopVLは各tokenへ2軸座標を与える。

```text
visual patch at row r, column c: position = (r,c)
text token at sequence position p: position = (p,p)
```

attention head dimensionの前半にrow由来の角度、後半にcolumn由来の角度を使う。textでは両座標が同じなので通常の
1D RoPEと等価になる。これにより同じattention layerで画像の2D配置とtext順序を扱う。[attention.py](attention.py)の
`build_rope_positions`と`apply_rope`が実装する。

## 8. 学習objective: 何を入力し、何をGTとして、何を予測するか

LoopVLのSFT lossは特殊なrecurrent lossではなく、response tokenだけに掛ける通常のautoregressive CEである。

1 sampleは画像`I`、instruction`Q`、正解response`Y=(y1,...,yT)`を持つ。

```text
model入力:
  [header | visual embeddings | instruction | y1 | y2 | ... | yT]

予測とGT:
  position y1の直前のlogits -> GT y1
  position y2の直前のlogits -> GT y2
  ...
  position yTの直前のlogits -> GT yT

loss対象:
  response tokenだけ

loss対象外:
  header、visual slots、instruction、padding
```

数式では、conditioning prefixを`x=(I,Q)`とすると、

```math
p_theta(Y|x) = product_t p_theta(y_t | x,y_<t)
```

```math
L_SFT = - sum_{batch i} sum_t log p_theta(y_{i,t}|x_i,y_{i,<t})
        / sum_i T_i.
```

GTは正解response token列だけである。「各loopで正しいvisual state」「正しいattention領域」「途中のL/H state」などの
教師は与えない。最終H stateから出したresponse CEをrecurrent graphへbackpropagateすることで、共有L/H module、
visual gate、anchor scale、projectorをend-to-endに学習する。[loopvl_model.py](loopvl_model.py)の
`response_cross_entropy`がこのshiftとmaskを実装する。

## 9. Selective backpropagation

全8 invocationは常にforward実行するが、公式学習は全てをbackpropagation graphへ保持しない。H2L3ではwarmup中に
active graphを次のように広げる。

```text
最初:   最後の2 invocation [L6,H2]
後半:   最後の5 invocation [H1,L4,L5,L6,H2]
対象外: 最初の3 invocation [L1,L2,L3]
```

対象外invocationもstateを作るためforwardには必要だが、その区間へのgradient historyを保存しない。parameterは共有なので、
後半invocationで使われた同じL/H parameterには勾配が入る。目的は128-layer相当のactivation memoryを抑えることであり、
「初期loopを実行しない」「初期loopだけ別parameterでfreezeする」という意味ではない。

本ディレクトリの小型modelは全invocationを通常backpropagationする。短い教材modelではmemory節約が不要であり、detachを
入れると「最終lossが全state遷移へどう伝わるか」の確認を妨げるためである。公式規模を再現する場合はearly invocationsを
no-grad実行し、active区間の直前でL/H stateをdetachしてgradient checkpointingと組み合わせる。

## 10. 学習pipeline

1. Language pretraining
   - HRM-Text frameworkを基礎に独自recurrent LMを75B text tokensで学習
2. Stage 1: Visual Language Alignment
   - LLaVA-559K、0.33B tokens
   - Penguinとlanguage backboneをfreeze、visual interfaceだけ更新
   - 最大長2048、visual tokens 256--1024
3. Stage 2: Multimodal Mid-Training
   - OneVision-1.5 Mid-Training-85M、49B tokens
   - Penguinはfreeze、language backboneとvisual interfaceを更新
   - 最大長3072
4. Supervised Fine-Tuning
   - 16.81B tokens
   - OneVision Instruct/Spatial、LLaVA、Vision-R1-Cold
5. Visual RL
   - Vision-R1-RL、GRPO
   - answer correctness reward
   - generation上限を2048から4096へ増やすprogressive schedule

Stage 1/2の主要optimizerはAdamW `(beta1,beta2)=(0.9,0.95)`、gradient clip 1.0、3% warmup。Stage 1 projector
LRは`1e-3`、Stage 2はlanguage `2e-5`、projector `1e-4`。vision encoderはfreezeする。

## 11. 推論

推論は通常のautoregressive generationである。画像・instruction prefixを処理して次token logitsを出し、tokenを1個追加して
再びforwardする。公開runtimeはgreedy decodingを使う。標準scheduleは学習時と同じH2L3であり、新しいresponse tokenを
生成する各forwardでもrecurrent backboneを実行する。

論文のschedule sweepでは、同じH2L3-trained checkpointに対してcycle数・L回数を推論時だけ変更すると性能が大幅に変わる。
例えばMMStarはH2L3で63.47だが、H1L3で0.07、H3L3で58.20。単純にloop数を増やすほど良いわけではなく、training時の
recurrenceとinference時のrecurrenceを一致させることが重要である。

## 12. Visual Aha Momentと診断指標

論文は、cycle境界を越えたときに同じshared layerのvisual attention分布が大きく変わる現象をVisual Aha Momentと呼ぶ。
これは「正答の理由を発見した」という因果的保証ではなく、cross-cycleでvisual evidence allocationが顕著に変化したという
観測名である。

- normalized spatial entropy: 小さいほど少数patchへ集中
- Gini concentration: 大きいほどattentionが不均等・集中
- 80% coverage: attention質量80%に必要なpatch割合。小さいほど集中
- visual hidden update RMS: invocation前後でvisual stateがどれだけ変化したか
- endpoint cosine similarity: 異なるcycleの同一patch stateの方向類似度
- logit-lens KL: intermediate output分布が最終分布へどれだけ近いか

[diagnostics.py](diagnostics.py)に最初の3指標を実装した。attention massと正規化後の空間分布は別物であり、画像への
総attentionが減っても、残ったattentionが少数patchへ集中することはある。また集中先がtask-relevantとは限らない。

## 13. Parameter共有のメリットとコスト

### 13.1 主なメリットはparameter効率

LoopVLは約1B規模、32 unique Transformer layers分のparameterだけを保存しながら、同じL/H stackを再利用して
128 layer applicationsを実行する。したがって、128個の独立layerを保持する場合よりcheckpointとmodel-weight memoryを
小さくできる。LoopVLの狙いは「少ないparameterを一度だけ使って高速に答える」ことではなく、「少ないparameterへ
追加計算を繰り返し割り当て、深い処理を行う」ことである。

### 13.2 同じparameter規模の通常VLMより精度は上がった

論文は、LoopVLと同じhidden size 1536、同じ32 unique layers、同じ0.14T training tokensを持つ非recurrent
Transformer-VL 1Bを比較している。

| Benchmark | Transformer-VL 1B | LoopVL |
|---|---:|---:|
| MMStar | 55.33 | 63.47 |
| RealWorldQA | 55.29 | 70.98 |
| VMCBench | 55.10 | 70.90 |
| AI2D | 60.01 | 75.49 |
| ChartQA | 51.12 | 74.52 |

この比較ではLoopVLが全掲載benchmarkで上回るため、同程度のstored parameter数に対し、shared moduleを反復して
executed depthを32から128へ増やす構成が精度向上に結び付いている。ただし計算量は同じではない。

```text
Transformer-VL 1B training FLOPs: 0.92e21
LoopVL training FLOPs:            2.47e21
```

LoopVLは約2.7倍の学習計算を使っている。したがって「parameter共有だけで無償に精度が上がる」のではなく、
**parameter数を増やす代わりに反復計算を増やして精度を得る**設計と解釈すべきである。

### 13.3 4B dense modelとの比較

4B Deep/Wide baselineのtraining FLOPsはそれぞれ`2.89e21`、`2.99e21`で、LoopVLの`2.47e21`に比較的近い。
この比較ではLoopVLがMMStar、RealWorldQA、MathVisionなどで上回る一方、VMCBench、AI2D、ChartQAなどでは
4B baselineが上回る。妥当な結論は「1B規模のstored parameterで4B dense modelと競争力のある性能を得た」であり、
全taskで大型dense modelを上回ったということではない。

### 13.4 推論高速化ではない

LoopVLは通常1B baselineの32 layer applicationsに対して128 layer applicationsを逐次実行するため、同じhardware・
実装条件なら推論計算も重くなる。論文はparameter効率とaccuracyを利点としており、通常の32-layer VLMに対する
推論高速化を主張していない。

```text
利点:
  - checkpointとweight memoryを小さく保てる
  - 同じparameter規模の通常VLMより高い精度を得た
  - 追加computeをrecurrent refinementとして利用できる

コスト:
  - forward当たりのlayer実行回数とFLOPsが増える
  - 推論latencyの短縮を目的とした構成ではない
  - trainingと異なるloop scheduleへ変更すると精度が大きく低下する
```

### 13.5 明示的なデメリット

1. **推論latencyが大きい。** 同じ1B/32-layer規模の通常VLMが32 layer applicationsで済むのに対し、標準
   LoopVLは128回実行する。parameterを共有してもmatrix multiplication自体は毎回必要であり、4倍の実効depthを
   無料にはできない。特にautoregressive generationではtoken生成ごとにこのrecurrent計算が必要になる。

2. **学習計算量も増える。** 論文の1B baselineは`0.92e21 FLOPs`、LoopVLは`2.47e21 FLOPs`である。
   weight数が少ないことと、学習が安価であることは別である。

3. **activation memoryと学習実装が複雑になる。** 128-layer相当の全activationを保持するとmemoryが大きいため、
   selective backpropagation、gradient checkpointing、active graphのwarmupが必要になる。全forwardを実行しながら
   一部invocationだけgradient graphへ入れるため、通常のstacked Transformerよりdebugが難しい。

4. **recurrence scheduleに対して脆い。** H2L3で学習したcheckpointはH2L3推論で最も良く、同じ総layer実行数でも
   H4L1などへ並べ替えると大幅に性能が落ちる。推論時にloop数を自由に増減できる汎用的なtest-time scalingには
   なっていない。

5. **追加loopを増やしても単調に改善しない。** MMStarはH2L3の63.47に対し、より深いH3L3で58.20、H4L3で
   24.00まで低下する。計算を増やせば必ず状態が改善されるわけではなく、training distributionから外れた反復は
   hidden stateや出力を悪化させ得る。

6. **stored parameter容量は増えない。** parameter共有はmemory効率には有利だが、各depthへ別々の変換を割り当てる
   dense modelほどparameter容量を増やせない。同程度のcomputeを使う4B dense modelがLoopVLを上回るtaskもあり、
   recurrenceが幅・独立depthの完全な代替になるわけではない。

7. **Visual Aha Momentは精度改善の因果説明ではない。** 同じlayerのattention変化を比較しやすいことは分析上の利点だが、
   attention集中やpatch順位の変化だけでは、その領域が最終回答を改善したとは証明できない。

## 14. DataFrame fine-tune example

[finetuning_example.py](finetuning_example.py)は次のDataFrameを受け取る。

```text
image          instruction        response
image_001.png  この画像の色は?     赤
image_002.png  物体はいくつ?       3
```

処理は画像load、patch encode、visual slot置換、PrefixLM mask、H2L3 recurrence、response-only CE、backward、AdamW update
まで実動する。文字tokenizerと小型frozen patch encoderは単独動作のための教材部品であり、実checkpoint fine-tuneでは公式
tokenizer/Penguinへ置換する。printだけで適用を装う処理やdummy lossは使用していない。

## 15. 実装上の注意

- H2L3のL3を「3 layers」と誤読しない。16-layer L stackを3回呼ぶ
- L/Hは互いにparameter共有しない。同じ種類のinvocation間だけ共有する
- cycle 2開始時にL stateをzeroへ戻さない
- visual encoderをloopごとに再実行しない
- anchorには更新後stateではなく初期projected visual embeddingを使う
- prefixから未来responseを見せない。GT leakageになる
- response CEをvisual/header/instructionへ掛けない
- textに2D spatial coordinateを割り当てず、`(p,p)`で1D RoPE互換にする
- inference loopを増やせば改善するとは限らない。学習scheduleとの一致が重要
- attention集中を正答根拠やcausal explanationと断定しない

## 16. ファイル構成

- [attention.py](attention.py): PrefixLM mask、visual 2D/text 1D-compatible RoPE
- [visual_interface.py](visual_interface.py): frozen patch encoderとRMS-calibrated projector
- [recurrent_backbone.py](recurrent_backbone.py): L/H module、nested recurrence、visual gate/anchor
- [loopvl_model.py](loopvl_model.py): visual slot置換、LM head、response-only loss
- [diagnostics.py](diagnostics.py): entropy、Gini、coverage
- [main_flow.py](main_flow.py): forward/backwardとshape検証
- [finetuning_example.py](finetuning_example.py): DataFrameからの実学習loop

```bash
cd survey/LoopVL
python attention.py
python diagnostics.py
python main_flow.py
python finetuning_example.py
```

## 17. 一次資料

- [論文](https://arxiv.org/abs/2609.38426)
- [公式GitHub](https://github.com/Tier-Flow/LoopVL)
- [公式checkpoint](https://huggingface.co/TierFlow/LoopVL)
