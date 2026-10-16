# GroundAnything 論文・公式実装サーベイ

## 1. 概要

GroundAnythingは、物体検出、referring grounding、pointing、OCR、document layout、GUI操作、
visual promptingを、画像とqueryから構造化token列を生成する1つの4B modelへ統合する。
最大の特徴は、座標列を左から1 tokenずつ生成するGroundAnything-VLMを、block内のtokenを
双方向に並列復元するdiffusion language model（DLM）へ変換する点である。

論文の主張は「boxは本質的に左から順に決める必要がない」という観察にある。画像内の複数物体、
label、座標は同じ視覚証拠で同時に制約されるため、現在tokenより右側の部分復元結果も参照できる
masked diffusionがgroundingに適する。block間だけcausalにすることで、可変長出力とKV cacheも維持する。

## 2. GroundAnything-VLMとの違い

| | GroundAnything-VLM | GroundAnything |
|---|---|---|
| 生成 | 通常のautoregressive | blockwise masked diffusion |
| block内attention | causal | bidirectional |
| block間 | causal | causal |
| 語彙 | text + 1000座標 + `</c>` | 左記 + diffusion MASK |
| 学習 | causal CE + causal GRPO | AR/MDM混合変換 + causal GRPO |
| main評価 | AR | entropy-guided pure diffusion |

両modelはMoonViT-V2、projector、Qwen3-4B decoder、座標語彙、出力headを共有する設計だが、
公開checkpointは別である。GroundAnythingのcausal/diffusion modeは同一weightを共有する。

## 3. 入出力protocol

入力は画像`I`とtask query`Q`。出力`Y=(y_1,...,y_L)`はtext、構造marker、座標tokenの混合列。
boxは`xyxy`の4 token、pointは`xy`の2 tokenで、座標は画像幅・高さに対して0--999へ量子化する。

```text
<|object_ref_start|>person<|object_ref_end|>
<|box_start|><10><20><30><40>,<50><60><70><80><|box_end|>
```

上の改行は表示用で、実際のresponseは1本のtoken列である。同じidentifierに属する複数instanceは
`,`で区切り、異なるidentifierのentryは`, `で区切る。OCRでは認識文字列がidentifierになる。
query対象が画像に存在しない場合はpayloadに通常textの`None`を出す。これは座標tokenでもpaddingでもない。

### 3.1 0--999の座標tokenをpixel座標へ戻す方法

モデルの語彙には、座標用の離散tokenが1000個、`<0>`, `<1>`, ..., `<999>`として割り当てられている。
モデルは連続座標を回帰するのではなく、x座標またはy座標としてこの1000 tokenのどれか1つを予測する。

例えばx位置として`<100>`が生成された場合は`q_x=100`、y位置として`<200>`が生成された場合は
`q_y=200`と読む。`q_x`, `q_y`は式で使う整数変数であり、その名前のtokenが登録されているわけではない。
幅`W`、高さ`H` pixelの画像では、これらを次の式でpixel座標へ戻す。

```text
x_pixel = q_x / 999 * (W - 1)
y_pixel = q_y / 999 * (H - 1)
```

`<0>`が左端または上端、`<999>`が右端または下端に対応する。`W-1`, `H-1`なのはpixel indexが
`0..W-1`, `0..H-1`だからである。整数pixelが必要なら最後にroundする。

box payloadは`<q_x1><q_y1><q_x2><q_y2>`なので、例えば1920x1080画像に対する
`<100><200><500><650>`は次のpixel boxになる。

```text
x1 = 100 / 999 * 1919 = 192.09
y1 = 200 / 999 * 1079 = 216.02
x2 = 500 / 999 * 1919 = 960.46
y2 = 650 / 999 * 1079 = 702.05

整数pixelへ丸める場合: (192, 216, 960, 702) in xyxy
```

逆に学習データのpixel座標を座標tokenへ変換する式は次である。

```text
q_x = round(999 * x_pixel / (W - 1))
q_y = round(999 * y_pixel / (H - 1))
```

量子化により複数のpixel位置が同じtokenへ写るため、完全な可逆変換ではない。なお、論文が別途記載する
`floor((999*v+500)/1000)`は、既に`0..1000` gridで表現された既存annotation値`v`をGroundAnythingの
`0..999` gridへ移す式であり、pixel座標を直接変換する式ではない。

1000個の座標をそれぞれatomic tokenにすることで、数字文字列を複数の通常tokenへ分割せず、boxを必ず4 token、
pointを必ず2 tokenで表せる。ablationではtextual coordinateよりARで1.43 percentage point、DLMで
5.68 percentage point高い。serialize・座標変換の実装は[spatial_protocol.py](spatial_protocol.py)に示す。

## 4. Architectureとshape

```text
image (B,3,H,W)
 -> MoonViT-V2: patch=14, 27 layers, width=1024
patch features (B,Np,1024), Np<=4096
 -> 2x2 spatial aggregation: 4*1024=4096
 -> MLP 4096->4096->2560 + RMSNorm
visual tokens (B,Nv,2560), Nv<=1024

text prompt/response ids (B,T)
 -> embedding (B,T,2560)
 -> visual placeholderをvisual tokensで置換
 -> Qwen3 decoder: 36 layers, width=2560, FFN=9728,
    32 query heads / 8 KV heads / head dim=128
 -> vocabulary logits (B,T,152671)
```

GroundAnything-VLMの語彙152,670はbase 151,669 + 座標1000 + category delimiter。
GroundAnythingはMASKを1つ足す。input embeddingとoutput headはuntied。MASKの各rowは4つの既存特殊token
embeddingのFP32平均で初期化する。全multimodal parameterは約4.844Bで、「4B」はlanguage backboneを指す。

[model_blocks.py](model_blocks.py) はvisual embedding挿入と、同じdecoder weightをcausal/bidirectional
maskで使い分ける核心を小型PyTorch modelとして実装する。MoonViT/Qwen3全層の複製ではない。

## 5. 学習時の入力・予測・GT・loss

### 5.1 Grounding SFT

1 sampleは画像、query、構造化GT responseを持つ。画像tokenとuser promptは入力contextでありloss対象外、
assistant responseのidentifier、marker、座標、EOSがすべてGT tokenとなる。

```text
input:  [visual tokens V | prompt P | response Y]
GT:                                  y1 y2 ... yL EOS
loss:  assistant responseだけのshifted causal cross entropy
```

座標用の別regression headやL1/GIoU lossは使わない。`<0>`--`<999>`もlabelやmarkerと同じtoken CEで学ぶ。

### 5.2 AR-to-Diffusion conversion

clean sequenceを`x=[V,P,Y]`とし、教師対象response `Y`を連続32-token blockへ分ける。block境界は
box tupleやentry境界と無関係で、最後のblockは短くてよく、semantic paddingはしない。

各block `b`でnoise level `t_b ~ Uniform(0,1)`をsampleする。各非EOS response位置で
`m1~Bernoulli(t_b)`, `m2=1-m1`とし、相補的な2 noisy viewを作る。したがって通常tokenは必ず
片方のviewだけでMASKになり、EOSは両方でMASKになる。画像、prompt、header、非教師turnは壊さない。

```text
clean:  P | A B C D | E F EOS
view1:  P | M B M D | E M M
view2:  P | A M C M | M F M
             block0     block1
```

公式joint inputは`[view1; view2; clean]`で、visual encodingを1度だけ共有する。noisy response queryは
clean image/prompt、厳密に過去のclean response block、同じnoisy block全体を読める。現在・未来の
clean target、他方のnoisy viewは読めない。clean streamは通常causalでnoisy viewを読めない。

### 5.3 Objective

MDMは2 viewでMASKされた位置だけを元のclean tokenへ戻すCE、ARはclean response全tokenのnext-token CE。

```text
L_MDM = - sum(view a, i in masked_a) log p(y_i | noisy view a, visible context)
         / (masked token総数)

L_AR  = - sum(i in valid response) log p(y_i | V,P,y_<i)
         / (valid response token数)

L = 0.5 * L_MDM + 0.5 * L_AR
```

両lossともhidden position `i-1`がtarget `i`を予測するshiftを保持する。inverse-noise weightingはしない。
prompt、visual position、padding等は除外する。[block_diffusion.py](block_diffusion.py)は相補corruption、
block attention mask、両lossを実装する。

AR lossを残す理由は、変換前の順序生成能力を維持し、同一weightでcausal rollout/verificationも可能にするため。
MDM lossはmaskされた座標や構造tokenを、同一blockの左右contextと画像から復元する能力を与える。

## 6. 全学習pipeline

1. Connector alignment: vision/languageをfreezeしprojectorだけをcaption dataで学習
2. Joint multimodal pretraining: 全moduleをtext + image-textで学習
3. General image/video instruction training: 32K context
4. Coordinate alignment/SFT: 8 spatial task群、8K context、全module更新
5. GroundAnythingのみAR-to-Diffusion conversion
   - Phase I: vision freeze、projector/language更新
   - Phase II: 全module更新
   - 256 GPUs、global batch 8192、clean length 4096、block 32
6. causal GRPO reinforcement post-training

conversionはAdamW `(0.9,0.95)`、weight decay 0.1、gradient clip 1.0、cosine + 3% warmup。
language/projector LRは`2e-5`、Phase II vision LRは`2e-6`。

## 7. GRPO

diffusion modelだがRL rolloutとlikelihood評価はcausal modeで行う。各promptから`G=8` responseを
temperature 0.7でsampleし、boxではset completenessとstrict localization、OCRではtext-geometry、
pointingでは位置・個数・label・formatをreward化する。

componentをgroup内標準化して加重和を作り、もう一度group標準化してadvantage `A_i`を得る。
token probability ratioを`r=exp(logp_current-logp_old)`としてPPO型clip lossを使う。epsilonは0.2、
reference-policy KLはなく、vision encoderをfreezeしてprojector/languageを更新する。
[rewards.py](rewards.py)にIoU matching、advantage、clipped lossを示す。

## 8. Entropy-guided diffusion inference

画像/queryをcausal prefillした後、response blockをMASKで初期化する。block内をbidirectional forwardし、
MASK位置`j`のunmodified distribution entropy
`H_j=-sum_v p_j(v)log p_j(v)`を計算する。`H_j<=tau`を全commitし、該当がなければ最小entropy位置を
1つcommitする。確定tokenは再maskしないので必ず進行する。

blockが完成してもbidirectional hidden stateは右側tokenを見ておりcausal KV cacheに使えない。そのため
完成blockをcausal forwardでもう一度流し、authoritative KVを構築する。D回denoiseしたblockはこのpassを
含めD+1 NFEを使う。これはAR verificationではなく、完成tokenを拒否・修正しない。
[decoding.py](decoding.py)がframework非依存版を実装する。

optional self-speculationではdiffusion draftをcausal branchで検証し、先頭から一致する最長prefixだけを
acceptする。最初の不一致で打ち切り、後方の偶然一致は捨てる。main benchmarkは公平なpure DLM比較のため
self-speculationではなくentropy-guided decodingを使用する。

## 9. データengine

public datasetと複数teacherをfield単位で融合する。annotation全体を丸ごと採否判定せず、text、geometry、
semantic evidenceを個別に検証する。queryが全instance列挙を要求する場合はfield validityだけでなくcoverageも
passする必要がある。空predictionだけでは「対象なし」の証明にならない。uncertain fieldは別teacherや
local observationへ回し、accepted labelで統合expertを反復改善する。

## 10. 評価と主要結果

30 benchmark、11 capability、44 baselineを評価。box/OCRはF1mIoU、pointingはF1@Point、spatial/GUIは
task固有accuracy。model結果は5 seed×2回の10 run平均。

- GroundAnything-VLM: overall 72.42%、同規模で新SOTA、GPT-6 Astra 71.35%とも競合
- GroundAnything entropy-guided: overall 61.75%、従来同規模AR SOTAを超え、VLMの2倍超の速度
- COCO self-speculative: VLM比4.51倍、F1mIoU低下0.74 point
- SGLang + CUDA Graph + selective FP8: native PyTorch比3.45--4.06倍
- self-speculation最終stack: 3.73 ms/token

accuracyだけならAR版が高い。一方DLMの価値はdense/long outputで並列性を得つつ、同規模の従来AR SOTAを
超える点にある。bidirectional DLMはDense200のunmatched valid proposalをcausal MTPの24.89%から14.79%へ
減らした。2 refinement passまではaccept効率が改善するが、それ以上は追加計算が上回る。

## 11. 限界と注意点

- pure DLMはGroundAnything-VLMよりoverall accuracyが低い
- structured grammar違反、重複、逆転boxなどは依然あり、decode policy/parserが重要
- block boundaryはsemantic tupleを尊重しないため、mask/shift実装のoff-by-oneが重大
- entropyはcommit reliabilityであり、sample方法や正しさそのものとは同一でない
- self-speculativeの精度はcausal verificationを含むためpure diffusion結果と分ける必要がある
- 4.844B full modelの再学習・serveには大規模GPUとcustom SGLang環境が必要
- 教材modelはconversion algorithmを再現するが、MoonViT/Qwen3 pretrained能力は再現しない

## 12. ファイル構成

- [spatial_protocol.py](spatial_protocol.py): 座標量子化とresponse grammar
- [model_blocks.py](model_blocks.py): shared causal/bidirectional decoder
- [block_diffusion.py](block_diffusion.py): complementary corruption、attention、AR/MDM loss
- [decoding.py](decoding.py): entropy-guided parallel decoding
- [rewards.py](rewards.py): IoU rewardとGRPO loss
- [main_flow.py](main_flow.py): joint objectiveのbackwardとdecode
- [finetuning_example.py](finetuning_example.py): DataFrame + 素のPyTorch学習loop

```bash
cd survey/GroundAnything
python main_flow.py
python finetuning_example.py
```

## 13. 一次資料

- [論文](https://arxiv.org/abs/2609.39600)
- [公式GitHub](https://github.com/groundingpi/GroundAnything)
- [GroundAnything checkpoint](https://huggingface.co/GroundingPI/GroundAnything)
- [GroundAnything-VLM checkpoint](https://huggingface.co/GroundingPI/GroundAnything-VLM)
