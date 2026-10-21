# iTransformer — 変数を token にする Transformer

## 概要

iTransformer（ICLR 2024）は、通常の時系列 Transformer の「時刻を token、同時刻の全変数を feature」とする軸を反転し、**1 変数の履歴全体を 1 token** にする多変量予測モデルである。入力 `X:[B,T,C]` なら token 数は時刻長 `T` ではなく変数数 `C`。各変数の履歴 `[T]` を共有 Linear で `[D]` に埋め、変数間 self-attention で融合する。原著の出力は未来 `H` 点 `[B,H,C]`、本 survey の [ts_block.py](ts_block.py) は後段タスクに挿せる同長特徴 `[B,T,Cout]` を返す。

この「反転」は単に tensor を転置するだけではない。attention の対象が時刻間から**系列間**へ、FFN が扱う情報が同時刻の変数ベクトルから**1 変数の時間的パターン**へ変わる。特徴 `h[b,t]` をオンライン t 時点の予測に使うなら、入力窓には t 以前しか含めないこと。encoder 内は非causalである。

## 既存法との差分・新規性

| 方法 | attention token | 系列間融合 | 時間方向の扱い |
| --- | --- | --- | --- |
| 通常の temporal Transformer | 時刻 `t` | 各時刻の入力射影から暗黙に混合 | attention が `T×T` |
| PatchTST | 各変数の patch | backbone では独立、重み共有 | 各変数内の patch attention |
| **iTransformer** | **各変数の履歴全体** | **変数 token 間の `C×C` attention** | 各 token の共有 Linear/FFN |

変数 token に時刻の順序を追加の位置埋込として与えない。履歴の `t` 番目が embedding の `t` 番目の入力重みに入るため、Linear 自体が時間位置を区別する。一方、変数 ID embedding もない。`C` の増減・順序変更に構造的に対応できるが、変数固有の意味が必要なら外部で与える必要がある。

## アーキテクチャと形状

`B`: batch、`T`: lookback、`C`: 変数数、`D`: hidden、`H`: 未来長、`L`: encoder 層数。

```text
原著の入力 X                                        [B,T,C]
  ├─ optional instance norm（各 b,c の T 軸）     [B,T,C]
  └─ transpose                                    [B,C,T]
       └─ shared Linear(T→D) + dropout            [B,C,D]
            └─ EncoderLayer × L                   [B,C,D]
                 ├─ Q,K,V=[B,heads,C,D/heads]
                 ├─ softmax(QKᵀ/√d)=[B,heads,C,C]  ← C 変数を融合
                 ├─ attention residual → LayerNorm
                 └─ 1×1 Conv FFN → residual → LayerNorm
            └─ final LayerNorm                    [B,C,D] ← encoder 境界
                 └─ 公式 forecast Linear(D→H)     [B,C,H]
                      └─ transpose + denorm        [B,H,C]

本ブロックの差分（encoder 境界以降のみ）:
  [B,C,D] → Linear(D→T) → [B,T,C]
          → Linear(Cin→Cout) → [B,T,Cout]
```

`forward_features(x)` は公式encoder相当の `[B,C,D]` を返す。`forward(x)` は汎用同長 adapter を加えた `[B,T,Cout]` を返す。`Cin=Cout` でも最後の channel Linear は省略せず、attention に加えて late fusion できる。`T` は embedding の入力幅なので構築時固定。`C` は公式encoder内なら可変だが、本ブロックは出力射影と入出力検査により `Cin` 固定である。

## 1 層の計算を詳しく

`q,k,v = Linear(h)` を head に分割し、`A[b,h,c1,c2]=softmax_c2(q·k/√(D/heads))` とする。変数 `c1` の更新は全 `c2` の value の重み付き和。ここでの `C×C` は系列間attentionであり、時間 `T×T` attentionではない。続く `Conv1d(kernel=1)` 2 層は `[B,D,C]` として動き、各変数tokenの**特徴軸 D**を変換する。LayerNormも各 `[D]` token内に掛かる。実装の残差と正規化は post-norm (`norm1(x+attn)`, `norm2(x+ffn)`)。

原著コードでは `DataEmbedding_inverted.value_embedding=Linear(T,D)`、`EncoderLayer` の attention は非mask `FullAttention`、encoder末尾は LayerNorm。追加の時刻 marker があると marker の系列も token に連結できる。本汎用blockは `x` だけの契約を優先し marker を使わない。従って、marker 有りの公式設定との同一性は主張しない。

## 学習 objective：入力・予測・GT

原著のforecastでは履歴 `X:[B,T,C]` が入力、未来実測 `Y:[B,H,C]` が GT。encoder token `[B,C,D]` を共有 `Linear(D,H)` に通し、`Ŷ:[B,H,C]` と `Y` の要素平均 MSE を最小化する。GT未来値はencoder入力やattentionに入らない。instance normを使う場合、入力窓から求めた各変数の平均・標準偏差で正規化し、forecast head の後に同じ統計で戻してから MSE を掛ける。任意の `forward_features` token に別の教師があるわけではない。

本 [finetuning_example.py](finetuning_example.py) は**別の汎用タスク例**。DataFrameの過去 `T` 行 `[B,T,Cin]` をblockへ通し、例として時間mean pooling `[B,Cout]` と外部headを使い、次行の回帰実測 `[B,K]` または分類ラベル `[B]` に MSE/CE を掛ける。pooling/head は差し替え可能で、原著の未来 `H` 点forecast objectiveとは混同しない。

## 実装と公式照合の境界

[verify_official.py](verify_official.py) は公式 `DataEmbedding_inverted`、`FullAttention`、`AttentionLayer`、`EncoderLayer`、`Encoder`、`Model` の**クラス定義を変更せず**読み込む。無関係な外部attentionライブラリの import を避けるため AST で対象クラスだけ抽出するが、対象の forward 本体は変更しない。対応重みをコピーし、公式 `Model.forward` の `encoder` 出力を hook で取得して、本 `forward_features` と同一入力で比較した。`T=16,C=3,D=16,L=2` の設定で数値一致。比較は**予測headの前**であり、後段の `Linear(D→T)` と `Linear(Cin→Cout)` は汎用化 adapter なので公式一致の判定対象外。

同じ重み・入力・設定で一致したという意味であり、任意の `T,C,D`、markerあり設定、公式checkpoint読み込みまで自動保証するものではない。特に `d_model % heads == 0` が必要。

## 1000 系列入力と計算量

系列数 `C≈1000` では各層の attention score が `[B,heads,C,C]` で、演算は概ね `O(B·C²·D)`、score のメモリは `O(B·heads·C²)`。例として `B=16,heads=4,C=1000,FP32` なら score だけで約256 MB、逆伝播中はさらに増える。入力 `Linear(T,D)` は `C` に線形だが、attention が支配的になり得る。入力の早期圧縮で `C` を減らせるが、その場合「元変数を各 token として直接 attention」する原著の解釈は変わる。精度と速度を測って選ぶ。

## ablationから分かること・分からないこと

論文は「変数間をattention、各変数内の時間パターンをFFN」という役割分担を検証している。論文表の全予測長平均MSEでは、ElectricityのiTransformer `0.178` に対しattentionを両軸に使う変更は `0.193`、TrafficではiTransformer `0.428` に対し同変更は `0.913`。一方、WeatherではFFNのみの変更 `0.248` がiTransformer `0.258` よりよく、**すべてのデータで元構成が最良**とは言えない。いずれも論文の予測headを含む結果であり、この汎用encoderを別タスクに挿した場合の優劣とは区別する。

論文が示すattention可視化は、ある変数tokenが別変数tokenを参照できることを直感的に示す。ただしattention重みが大きいだけで、その変数が予測の因果的説明になったとは言えない。変数定義・スケール・欠損率が違う実データで使う場合は、attention図だけでなく除外実験や予測精度を確認する。

## 可変変数数と本blockの固定契約

原著は、埋込が各変数の長さTの系列に共有適用され、Transformer本体も変数数Cに依存しない点から、変数を部分サンプリングして学習する運用を論じる。これは**原著encoder内部**の性質。本blockの入出力契約は明示的に `in_dim=Cin` と `out_dim=Cout` を持ち、最後の `Linear(Cin,Cout)` はCに依存する。従ってこの完成blockを同じ重みのまま異なる列数へそのまま入力することはできない。列を固定して学習・運用する通常の表形式タスク向けにこのadapterを選んだ。可変Cが必要なら、`forward_features` のtoken集合へ別途可変長readoutを設計する必要があり、それは本版と異なる仕様になる。

## 使用例と注意

```python
from ts_block import ITransformerBlock

block = ITransformerBlock(in_dim=12, out_dim=32, seq_len=64)
h = block(x)                     # x:[B,64,12] -> h:[B,64,32]
tokens = block.forward_features(x)  # [B,12,D]
```

`h` に対する pooling・regression/classification head は用途に応じて選ぶ。系列の並びがサンプル間で変わると、末尾の `Linear(Cin,Cout)` の意味が変わる。各窓でチャンネル定義を固定する。原著のtoken処理には系列IDを区別する埋込がない点にも注意。

検証例：`python ts_block.py`、`python verify_official.py --official-repo <iTransformer-repo>`。DataFrame例は `python finetuning_example.py --csv data.csv --features f1 f2 --targets y --seq-len 64`。学習例は `../finetuning_common.py` の chronological split、DataLoader、AdamW、MSE/CE を利用し、NeMo等は不要。

## 通常Transformerと何を「反転」したか

`T=96,C=8,D=64` の場合、通常の時間tokenモデルでは96個のtokenがそれぞれ8変数の値を含む。一方、iTransformerでは8個のtokenがそれぞれ96時刻の値を含む。

```text
通常:       X[B,96,8] → 96 token [B,96,64] → score[B,Hh,96,96]
iTransformer: X[B,96,8] →  8 token [B, 8,64] → score[B,Hh, 8, 8]
                            ↑ 各tokenの入力は1変数の96点
```

Tが長くCが小さいならattention token数は大幅に減る。逆に`C=1000,T=96`ではiTransformerは1000 tokenとなり、単純な時間token Transformerよりattention表が大きい。このモデルを「常に長系列だから高速」と理解しない。計算量比較にはTとCの両方が必要。

## encoderの一層を追う

```text
h_l:[B,C,D]
 → Q,K,V各[B,Hh,C,d]（d=D/Hh）
 → score QKᵀ/√d:[B,Hh,C,C]
 → softmaxをkey系列軸に掛ける
 → valueの加重和:[B,C,D]
 → attention出力投影＋残差＋LayerNorm:[B,C,D]
 → 1×1 Conv(D→Dff)→活性化→1×1 Conv(Dff→D)
 → 残差＋LayerNorm:[B,C,D]
```

各tokenは1系列の窓全体から作られているので、attentionは「同時刻の変数AとBの相関」だけを見ているのではない。**窓全体を要約した表現同士**の関係を学ぶ。Conv1dは系列軸Cに沿って幅1で適用され、C個のtokenを直接混ぜない。C間を混ぜる主要演算はattentionである。

## 原著の予測と本blockの下流利用を分離

```text
共通: 過去窓 X[B,T,C] → 可逆な窓正規化 → Linear(T,D)
      → 変数間Transformer → Z[B,C,D]

原著: Z → Linear(D,H) → 逆正規化 → 未来予測[B,H,C]
      → 実測未来Y[B,H,C]とMSE

本版: Z → Linear(D,T) → Linear(Cin,Cout) → h[B,T,Cout]
      → 外部の任意task head → 下流GTに応じたloss
```

「同長」のTは出力特徴の軸であって、原著の予測horizon Hではない。`forward_features`のZは窓内の時間軸を1tokenへ圧縮しているため、本版の`h[:,t]`は一般に窓内全時刻に依存する。時刻tでのcausal特徴ではない。次行GTを予測するなら、入力窓をGTより前だけで作る。

## 公式処理と本実装の対応

| 段階 | 公式 | 本survey |
| --- | --- | --- |
| 窓正規化 | `forecast`内で変数別mean/std | `ITransformerBlock.forward_features` |
| 変数token化 | `DataEmbedding_inverted` | `Linear(T,D)`とdropout |
| 変数間attention | `FullAttention` + `AttentionLayer` | `VariateAttention` |
| FFN・残差・norm | `EncoderLayer` | `InvertedEncoderLayer` |
| 複数層と最終norm | `Encoder` | block内の層反復・norm |
| 未来Hへの予測 | `projection` | 汎用encoderの範囲外 |

元の論文で系列数が変えられるという主張は、表の**token化からencoderまで**の話。本版の同長幅変換まで含めて可変Cだと主張しない。時刻markerを入力する公式変種、変数ID埋込の追加、線形readoutの変更は、同じ名前でもこの照合設定とは異なるモデルになる。

## DataFrameの1サンプルと下流loss

入力DataFrameは少なくとも`series_id,timestamp,feature列,target列`を持つ。1系列内でr行目を教師にした場合の計算は次のとおり。

```text
x = features[r-T:r]                     [T,Cin]
z = ITransformerBlock.forward_features  [Cin,D]
h = ITransformerBlock(x)                 [T,Cout]
p = mean_time(h)                         [Cout]  ← poolingの一例
ŷ = task_head(p)                         [K]
y = targets[r]                           [K] / 整数class
loss = MSE(ŷ,y) / CrossEntropy(ŷ,y)     scalar
```

窓はr行より前で終わるため、GTをencoderへ与えない。各seriesのlabel位置を時間順80/20に分割し、入力標準化統計は学習側のみで計算する。これは原著の`[B,H,C]`予測MSEとは違い、任意のK次元ターゲットへ適用する**使用例**。平均poolingが一般に最良とは主張せず、時刻別headやattention poolingなどに置換できる。

## 設定を変えると何が変わるか

| 設定 | 役割 | 固定されるもの |
| --- | --- | --- |
| `seq_len=T` | 1変数tokenの入力幅 | 埋込`Linear(T,D)`と同長readout |
| `in_dim=C` | attentionのtoken数 | 本blockの最後のchannel projection |
| `d_model=D` | 変数tokenの特徴幅 | QKV、FFN、forecast/同長readout |
| `heads=Hh` | attentionの分割数 | `D % Hh = 0`が必要 |
| `layers=L` | 系列間情報交換の反復 | 計算・activationが概ねL倍 |

原著の変数token化はCに依存する学習パラメータを持たないが、本blockの`Cin→Cout`は持つ。だからCの変更に関する性質を、encoderと完成blockで分けて述べる必要がある。C=1000で早期入力圧縮を加えるのは計算上の選択肢だが、その時は1000個の元変数token同士のattentionではなく、圧縮後の潜在token間attentionになる。

## 「系列間融合」の意味を具体化

変数`c1`のattention出力は、各headで全変数`c2`のvalueの重み付き和。重みは入力窓によって変化する。よって原著encoder内で既に融合しており、本版の最後の`Cin→Cout`は**初回の融合ではない**。一方、時刻T方向については`Linear(T,D)`で各系列を埋めるので、T個の時刻token間attentionを行っているわけではない。

```text
系列aの履歴[T] ─→ token a[D] ─┐
系列bの履歴[T] ─→ token b[D] ─┼→ 変数間self-attention[C×C]
系列cの履歴[T] ─→ token c[D] ─┘       ↓
                             各系列の更新token[D]
```

これは相関行列を固定的に計算する処理とも異なる。query/keyは学習重みで作られ、入力窓に応じてscoreが変わる。ただしattention重みが大きい系列が統計学的な原因変数とは限らない。下流で解釈性が必要なら、系列除去や時期別安定性の検証を別途行う。

## 単変量入力と大規模多変量入力

`C=1`では系列間attentionのsoftmaxは1要素で選択の余地がなく、iTransformerの核となる変数間相互作用がない。`C`が増えると関係を表現できるが、scoreはC²。C=1000で最初に`Linear(1000,64)`へ圧縮してからblockへ渡す構成は計算上現実的な場合があるものの、tokenは元の1000変数ではなく64個の**混合特徴**を表す。論文の変数tokenの意味と違うため、精度・解釈を同じものとして扱わない。

本版の`[B,T,Cout]`出力では、同じCoutでもTを変えたモデル間でreadoutの重み幅が違う。評価時だけ窓長を変えることはできない。長窓の効果を調べるなら、各Tごとに別構成・別学習として比較する。

## 導入時に確認する点

- 入力の各列が同じ意味・同じ順序で全サンプルに存在するか。
- 一つの窓のT点をすべて観測した後で使うタスクか。窓内部の時点tへの逐次利用なら情報リークしない窓の切り方が必要。
- `T,C,D,Hh`の組み合わせで、`[B,Hh,C,C]`のscoreを保持できるか。
- 時刻markerを使う公式設定と、本版のmarker無しencoderを混同していないか。
- 出力`h:[B,T,Cout]`のTが未来horizonではなく同長特徴の軸だと理解しているか。

## 一次資料

原論文：[iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)（ICLR 2024）。公式実装：[thuml/iTransformer](https://github.com/thuml/iTransformer)。照合箇所：論文の Structure Overview / Inverted Transformer Components、公開 `model/iTransformer.py`、`layers/Embed.py`、`layers/Transformer_EncDec.py`、`layers/SelfAttention_Family.py`。
