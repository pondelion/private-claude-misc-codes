# PatchTST — patch化とchannel independence

## 概要

PatchTST（ICLR 2023）は、時系列を一点ずつではなく**部分系列patch**に分け、そのpatchをTransformerのtokenにする長期予測モデル。もう一つの柱は **channel independence**：`C` 本の各変数を `[B·C,N,D]` として同じTransformerに通す。パラメータは共有するが、**encoder内で別変数の値を見ない**。原著は未来 `H` 点を予測する。ここでは[ts_block.py](ts_block.py)を任意の下流headと組み合わせられる `[B,T,Cin]→[B,T,Cout]` encoderとして提供する。

## 何が新しいか／他手法との差

| 設計 | 通常の点token Transformer | PatchTST |
| --- | --- | --- |
| token単位 | 時刻1点 `[C]` | 1変数の連続`P`点 |
| token数 | `T` | `N=floor((T-P)/S)+2`（end pad使用時） |
| attention | 多くは時刻間、変数は埋込で融合 | 各変数**内**のpatch間。変数間は独立 |
| 系列数の扱い | 入力LinearがCに依存 | backbone重みは全変数共有 |

`P` 点を一つに埋めることで局所形状を保ちつつattention token数を約 `T/S` に減らす。attention mapは各変数あたり `N²`。一方、channel independenceは変数間相関を**モデル化しない**という明確な代償を持つ。`[B,T,C]`を受け取れることだけで「多変量融合モデル」とは呼べない。このsurvey版では最後の `Linear(Cin,Cout)` を常に置くため**out projection時に初めて系列間を融合**する。原著backbone内に融合を足したと誤読しないこと。

## モデルアーキテクチャと形状

`B`: batch、`T`: lookback、`C`: 変数数、`P`: patch長、`S`: stride、`N`: patch数、`D`: hidden、`H`: 未来長。

```text
原著 X:[B,T,C] → transpose [B,C,T]
  → RevIN（各 b,c の時間平均/分散、optional affine） [B,C,T]
  → 右端に最後の値を S 点複製                 [B,C,T+S]
  → unfold(window=P,stride=S)             [B,C,N,P]
  → shared Linear(P→D) + 位置埋込          [B,C,N,D]
  → merge batch×channel                   [B·C,N,D]
  → TSTEncoderLayer × L                   [B·C,N,D]
       Q,K,V  [B·C,heads,N,D/heads]
       scores [B·C,heads,N,N]              ← 同一変数のpatch間だけ
       residual attention logits（層間）→ softmax → value
       residual → BatchNorm/LayerNorm → FFN → residual → norm
  → unmerge                              [B,C,D,N] ← encoder境界
  → flatten D·N → Linear(D·N→H)           [B,C,H]
  → RevIN denorm                          [B,H,C] ← 原著予測

汎用block: [B,C,D,N] → Linear(D·N→T) → denorm [B,T,C]
                    → Linear(Cin→Cout) → [B,T,Cout]
```

`N=floor((T-P)/S)+2` は公式 `padding_patch='end'` の場合。末尾に `S` 点を複製しない設定では `+1`。窓長 `T` はpatch数 `N` とflatten readoutの重みに入るので、本blockは構築時固定。`padding_patch='end'` は元論文の主設定に対応する。端にpatchが余ることと時刻paddingは別概念。

## attention と正規化を読む

PatchTSTの標準 `res_attention=True` では、第 `l` 層のsoftmax直前に**前層のraw score**を足す。`A_l=softmax(scale(Q_lK_lᵀ)+score_{l-1})`。既にsoftmax済みのattention重みを足すのではない。本版もlogit `scores` を次層へ渡す。既定正規化は `BatchNorm1d(D)` を `[B·C,D,N]` へ掛ける。小batchや変数間分布差が大きい状況ではBatchNorm統計に注意。LayerNormも選択できるが、公式既定の検証設定とは異なる。

RevINは各窓・各変数の `mean:[B,1,C]`, `std:[B,1,C]` を計算し、patch化前に `(X−mean)/std` とする。affineありならさらに変数別scale/bias。公式予測ではhead後に逆変換する。本blockの `forward_features` は正規化後の特徴だけを返し、`forward` は同長readoutの後に逆変換してから出力channel射影を掛ける。したがって `Cout≠Cin` でも統計の軸を取り違えない。

## 原著の二つの学習目的

**教師あり長期予測**：履歴 `X:[B,T,C]` のみをencoderへ入れ、未来実測 `Y:[B,H,C]` をGTにする。`D·N→H` のflatten headが各変数の未来H点を出し、RevINを戻した `Ŷ` と `Y` の平均MSE。未来GTはencoder入力に含まれない。

**自己教師あり事前学習**：非重複patchを作り、選んだpatchをzero maskする。masked patchの潜在表現に `Linear(D→P)` の復元headを付け、**maskしたpatchの原値** `[B,C,M,P]` をGTとしてMSEを掛ける。未来値の予測ではない。事前学習後に復元headを外し、予測head等を付け替える。supervised側ではoverlap patchが可能だが、pretrain側は隣patchからmasked内容が漏れないよう非重複を選ぶ。両objectiveを同時に説明せず区別する。

本 [finetuning_example.py](finetuning_example.py) は別の一般的な組込例。過去 `T` 点を `[B,T,Cout]` へ変換し、一例のpoolingと外部headで次行の回帰/分類GTにMSE/CEを掛ける。どのpooling/headを使うかはタスク次第。本blockが自動で原著のmask事前学習を行うわけではない。

## 公開実装との直接照合

[verify_official.py](verify_official.py) は公開 `PatchTST_backbone` を**そのままimport**して実行し、`backbone` の `[B,C,D,N]` 出力をhookで取得。`T=32,C=3,P=8,S=4,D=16,L=2`、RevIN・end padding・BatchNorm・層間residual attentionを含め、本blockの `forward_features` と一致した。さらに公式forecast headを `H=T` に設定し、対応する重みをコピーして、汎用blockの最後のchannel Linearを恒等にした条件では、入力から同長出力まで一致した。これは **H=Tという照合条件** であり、本blockの役割が未来予測に変わるわけではない。

他の `pe`、pre-norm、個別channel head、mask等の設定は未比較。公式重みを任意設定の本blockへそのまま読み込めるとは主張しない。コードにはpass/mockを使わず、使うencoder処理は実装している。

## `Cin≈1000` の場合

各変数のpatch系列を独立処理するのでattention演算・activationは概ね `O(B·Cin·N²·D)`、`Cin²` のattention表は作らない。**1000変数でも二乗爆発しない**という利点がある。ただし `[B·Cin,N,D]` のactivation量とBatchNorm計算は系列数に線形増加し、GPUメモリは小さくない。出力 `Linear(Cin,Cout)` の重みは `Cin·Cout`。入力チャネル同士の高次相互作用が重要なら、この最後の線形融合だけでは不足する可能性がある。逆に巨大なchannel self-attentionを足すと二乗コストを取り戻してしまう。

## 論文のablationをどう読むか

論文の予測実験では、patch化とchannel independenceの両方を使った設定が、片方だけより強い。たとえば予測長96のWeatherでMSEは両方あり `0.152`、channel independenceのみ `0.164`、patchのみ `0.168`、両方なし `0.177`。同じ条件のElectricityでは順に `0.130 / 0.136 / 0.196 / 0.205`。この比較が示すのは**原著の予測設定での相補性**であり、任意の下流回帰で常にその順位になる証拠ではない。特に、本blockでは最後のchannel projectionを追加するため、完全なchannel-independent予測器とは異なる。

patch化が有利な条件を式で見ると、点tokenなら1系列のattention scoreは `[T,T]`、本設定なら `[N,N]`。ただしpatch embed `[P]→[D]` がP点の区別を一旦D次元へ圧縮するため、非常に短い局所イベントの位置精度を重視するタスクでは `P,S` を検証する必要がある。`S<P` の重複patchでは同じ観測が複数tokenに入り、masked-patch事前学習の隣tokenリークが起こり得るため、原著の事前学習設定と教師あり設定のpatch規則を混同しない。

## 汎用encoderとしての境界

`forward_features` の `[B,C,D,N]` は系列ごとの**patch表現**であり、時刻ごとに直接対応する `[B,T,Cout]` ではない。本blockの同長readout `Linear(D·N,T)` は各系列の全patch表現を見て時刻軸Tへ戻す。従って出力時刻 `t` の特徴には入力窓の `t` より後の情報も含まれ得る。**窓全体を観測した後の系列単位タスク**には使えるが、各時点で未来を見ず予測するオンラインcausal encoderとしては使えない。次時点予測のDataFrame例では、入力窓末尾までの観測だけで次行GTを予測するのでこの問題はない。

またchannel fusionは `Linear(Cin,Cout)` 一層なので、全入力系列が共同で形成する非線形相互作用はencoder内部では学べない。下流で必要なら、外部にchannel attentionやMLPを置く選択肢があるが、その部分はPatchTSTの原著encoderではないと明示して比較する。

## 使用例と制約

```python
from ts_block import PatchTSTBlock

block = PatchTSTBlock(12, 32, seq_len=64, patch_len=8, stride=4)
h = block(x)                    # x:[B,64,12] -> [B,64,32]
patches = block.forward_features(x)  # [B,12,D,N]
```

同長`h`をpooling→regression/classification head等へ渡せる。系列の定義と列順は学習・推論で固定する。未来の `x` を窓へ混ぜない。`python ts_block.py` でforward/backward、`python verify_official.py --official-repo <PatchTST-supervised-repo>` で公式照合。DataFrame例：`python finetuning_example.py --csv data.csv --features f1 f2 --targets y --seq-len 64`。

## patch数とattentionの具体例

`T=32,P=8,S=4`、公式supervised側のend paddingを使う場合、末尾値を4点複製して長さ36にする。patch開始位置は `0,4,8,12,16,20,24,28` で `N=8`。各patch `[8]` を `[D]` に埋める。`C=3` ならattention入力は `[B·3,8,D]` となり、1系列につき`[8,8]`のscoreを作る。**`[24,24]`の全系列attentionではない。** 元の32時刻を8patchに縮めても、patchの重複により局所情報が完全に独立になるわけでもない。

```text
時刻: 0       8       16      24      32
      [patch0 ]
          [patch1 ]
              [patch2 ]
                  ...
                              [patch7 + 末尾複製]
変数 c ごとにこのpatch列を同じbackboneへ投入
```

strideを小さくするとpatch数Nが増えて局所の重なりも増す。注意するのは、attentionの二乗コストはTではなくNで決まり、`S`を半分にするとNは概ね2倍、score要素は概ね4倍になること。長窓で軽いという主張も、P・S・Cの組み合わせ次第である。

## 原著の二つの経路と本blockの境界

```text
supervised forecasting:
  X[B,T,C] → RevIN → overlapping patch → shared encoder[B,C,D,N]
           → flatten[D·N] → forecast Linear(D·N,H)
           → RevIN inverse → Yhat[B,H,C] → future YとのMSE

self-supervised masked patch:
  X[B,T,C] → non-overlap patch → 一部のpatchを0に置換
           → shared encoder → masked位置の[D] tokenをLinear(D,P)
           → マスク前の実測patchとのMSE（非mask位置は対象外）

本surveyの下流encoder:
  X[B,T,C] → supervised側と同じpatch/backbone[B,C,D,N]
           → Linear(D·N,T) → RevIN inverse → Linear(Cin,Cout)
           → 特徴h[B,T,Cout] → 任意の下流head
```

三つの経路を混同すると、**事前学習のmask復元head**を未来予測headと思ったり、同長adapterを原著学習の一部と思ったりする。事前学習では元のpatchをGTにしており、未来H点はGTではない。汎用blockはsupervised版backboneを使い、mask復元の学習ロジックは内蔵しない。

## 公開コードと本実装の対応

| 処理 | 公開実装 | 本survey | 判定 |
| --- | --- | --- | --- |
| 入力正規化 | `RevIN` | `PatchTSTBlock` 内 | encoder入力で照合 |
| end paddingとpatch | `PatchTST_backbone` | 同名設定のpatch化 | 照合 |
| QKV/残差attention | `PatchTST_layers` | `ResidualPatchAttention` | encoder末尾で照合 |
| FFN・正規化 | `TSTEncoderLayer` | `PatchEncoderLayer` | encoder末尾で照合 |
| horizon予測 | flatten head `D·N→H` | 原著目的としてREADMEで説明 | 汎用encoderの範囲外 |
| 同長・系列間融合 | 原著backbone外 | `D·N→T` と `Cin→Cout` | 意図的なadapter |

原著と同じtoken表現 `[B,C,D,N]` を検証境界にする理由は、これより後が**タスクの出力長H**に依存するから。任意の回帰/分類headを差し込む場合、このencoder境界までの一致こそ重要である。最後のchannel Linearで系列を混ぜても、encoder内部のattentionが系列間独立という事実は変わらない。

## 利用時の確認事項

1. 1行が同じ時刻のC変数を表すよう列を整列する。`C`の並びは学習・推論で固定する。
2. `seq_len=T`、patch長P、stride Sからpatch数Nが決まり、readoutの重み幅`D·N`も固定される。
3. 欠損はRevINへNaNとして渡さず、データ側で補完と欠損indicator等を検討する。
4. 下流のGTが入力窓より未来にある場合、splitは窓の時間範囲を考慮してリークを避ける。
5. 特定時刻tの因果的特徴が必要なら、窓末尾をtに合わせる。本block内にcausal attention maskはない。

## DataFrame学習例のラベル位置

`r`行目を教師とする窓は`r-T`行目から`r-1`行目までの特徴列だけを使う。元論文のGTは未来H点だが、サンプルのGTは**次の1行**である。

```text
DataFrame（1つのseries_id、timestamp昇順）
  行 r-T ... r-1 : 入力 features → [T,Cin]
  行 r           : target列だけを読む → 回帰[K] / 分類整数
  batch          : x[B,T,Cin] → PatchTSTBlock → h[B,T,Cout]
  一例のtask head: mean_time(h)[B,Cout] → Linear → 予測[B,K]
  損失          : 回帰MSE / 分類CrossEntropy
```

訓練と検証を時系列順のlabel indexで分け、入力標準化統計は訓練側のみで計算する。窓自体は検証開始前の過去行を参照し得るが、未来の行を入力しない。`series_id`が複数なら各系列内で窓を作り、系列の境界を跨がない。分類ラベルは`0,...,K-1`の整数として扱う。mean poolingは単なる組込例で、タスクに合えば最後のpatch、attention pooling、時刻別head等に変えられる。

## 設定値・計算量の読み方

| 設定 | 変えると起きること | 公式一致との関係 |
| --- | --- | --- |
| `patch_len=P` | 1tokenが見る連続点数が変わる | patch埋込の重み幅が変わる |
| `stride=S` | Nとpatchの重なりが変わる | 位置埋込・flatten headの幅が変わる |
| `d_model=D` | 表現幅とFFN計算が変わる | encoder重みを別形状で学習し直す |
| `res_attention` | 前層scoreを次層に足すか変わる | 公開照合は有効設定 |
| `revin` | 窓別の分布調整が変わる | 公開照合は有効設定 |

1系列のattention scoreは`[heads,N,N]`だが、実際には`B·C`系列分を同時処理する。`C≈1000`で二乗attentionがないことは、1000系列分のactivationが無料という意味ではない。比較実験ではCだけでなくB、T、P、S、Dを揃えて測る。

## channel independenceの利点と限界を分解

| 観点 | 原著encoderで起きること | 本blockで起きること |
| --- | --- | --- |
| 重み共有 | 全C系列に同じpatch embed/Transformer | 同じ |
| 変数ID | 明示的なID埋込なし | 最後のLinearは入力列位置ごとに重みを持つ |
| 早期融合 | なし | なし |
| late fusion | 原著backboneにはなし | `Linear(Cin,Cout)`で一度行う |
| 対象列数の変更 | backboneは共有できる | 最終Linearの形状は変更が必要 |

たとえば2列の温度・出来高を入れても、backbone内部では温度patchが出来高patchをattentionで参照しない。最後のLinearで温度側と出来高側の同長出力を混ぜられるが、それは**それぞれが独立に窓を処理した後**。変数間の状態依存の相互作用が必要なら、後段に非線形な融合を設計するか、最初から融合する別encoderと比較する。独立処理は欠点だけでなく、C増加時にattentionの二乗コストが出ない利点でもある。

## 予測headと同長adapterの重みshape

```text
公式forecast: [B,C,D,N] → flatten [B,C,D·N]
              → Linear(D·N,H) → [B,H,C]

本block:     [B,C,D,N] → flatten [B,C,D·N]
              → Linear(D·N,T) → [B,T,C]
              → Linear(Cin,Cout) → [B,T,Cout]
```

`H=T`の照合設定では二つの最初のLinearのshapeが一致するため、対応重みを移して同長出力の比較ができる。しかし、普段のH≠T予測ではその重みはそのまま移せない。**shapeが同じ特殊設定での一致**と、任意の予測headを移植できることは異なる。原著の事前学習head`D→P`はさらに別の重みshapeである。

## 一次資料

原論文：[A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)（ICLR 2023）。公式実装：[yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)。照合箇所：論文 Model Structure / Representation Learning、公開 `PatchTST_supervised/layers/PatchTST_backbone.py`、`PatchTST_layers.py`、`RevIN.py`。
