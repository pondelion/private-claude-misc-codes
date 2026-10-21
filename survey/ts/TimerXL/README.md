# Timer-XL — 多変量の次patch予測を行うcausal Transformer

## 概要

Timer-XL（ICLR 2025）は、単変量系列の次patch予測を行うTimerを、多変量・外生変数付き時系列まで拡張したdecoder-only Transformer。1変数の `P` 点を1 tokenにし、`C` 変数×`N` 個のpatch token間で**時間的因果性を保ちながら系列間も見る** `TimeAttention` を提案する。原著の事前学習・予測では、各tokenが**次のpatch** `[P]` を予測する。本surveyの[ts_block.py](ts_block.py)はencoder表現を通常の下流タスクに挿せる `[B,T,Cin]→[B,T,Cout]` に戻すadapterを付ける。

指定されたTimer-XLの公式ディレクトリには論文とREADME/quickstartのみで、モデル実装のPythonファイルはなかった。公式READMEが指す[OpenLTMの `models/timer_xl.py`](https://github.com/thuml/OpenLTM/blob/main/models/timer_xl.py) と関連layersを参照した。今回照合した公開ソースはOpenLTM commit `0b30050`。公式コードを見ずに推測した実装として扱わない。

## 何が従来と違うか

| モデル | token | 系列間情報 | 時間方向の因果性 |
| --- | --- | --- | --- |
| PatchTST | 各系列のpatch | encoderでは独立 | 非causal encoder |
| iTransformer | 各系列の履歴全体 | 変数token attention | 非causal encoder |
| Timer | 単変量のpatch | 対象外 | causal、次patch予測 |
| **Timer-XL** | **変数×時刻patch** | **同時点までの他変数patchへattention** | **未来patchはmask** |

新規性は、単純な「全 `C·N` tokenをcausal 1D列としてattention」ではない点。変数Aの第 `i` patchを予測する状態は、**変数Bの第 `i` patchも参照可能**だが、どの変数の第 `i+1` patchも参照できない。変数間の同時点依存を保ちつつ、次patch教師へのリークを防ぐ。系列間の同一/異なる別を表す学習可能なhead別biasとpartial RoPEも使う。

## アーキテクチャとshape

`B`: batch、`T`: lookback、`P`: token長、`N=T/P`、`C`: 変数数、`D`: hidden、`Hh`: attention head数。

```text
入力 X:[B,T,C]（TはPで割り切る）
  → optional instance norm（各b,cのT軸） [B,T,C]
  → 変数ごと非重複patch                 [B,C,N,P]
  → shared Linear(P→D)                 [B,C,N,D]
  → flatten(C,N)                      [B,C·N,D]

  TimerLayer × L:
    Q,K,V 線形射影                    [B,Hh,C·N,D/Hh]
    Q,Kのhead幅前半にpartial RoPE
    scores                            [B,Hh,C·N,C·N]
    ＋ learnable same/different-variate bias
    ＋ 時間causal mask（未来patchを -∞）
    softmax→weighted V→output Linear  [B,C·N,D]
    attention residual→LayerNorm
    1×1 Conv FFN→residual→LayerNorm   [B,C·N,D]
  final LayerNorm                     [B,C·N,D] ← encoder境界

原著: Linear(D→P_out) → [B,C,N,P_out] → 次patch予測
本block: Linear(D→P) → [B,T,C] → Linear(Cin→Cout) [B,T,Cout]
```

本blockの `forward_features(x)` は原著 `TimerBlock` 末尾 `[B,C·N,D]` を返す。`forward(x)` の `D→P` と `Cin→Cout` は汎用同長表現用adapterで、**次patch予測headの学習objectiveではない**。`Cin=Cout` でもchannel出力射影は常にlearnable。公式encoderではattention時点で系列間融合済み。

## TimeAttentionを正確に読む

多変量モードの許可行列は `allowed[(c_q,i),(c_k,j)] = (j≤i)`。すべての変数対で同じ時刻方向の下三角maskを使う。query `(c_q,i)` は他変数の同時刻 `i` までを見る。attention biasは `c_q==c_k` と `c_q≠c_k` の2値を各headで学ぶ。これは変数IDごとのembeddingではなく、**同じ系列かどうか**の関係を符号化するため、チャンネルを同時に並べ替えても相互作用の種類は保たれる。

公式 `TimeAttention` のnon-flash経路では、Q/Kの**各head幅の前半**にRoPEを掛ける。公開実装はtokenを `[C,N]` から `[C·N]` にflattenした**通しindex**をRoPE位置に使う。論文の時間位置・変数順序不変の説明とコードの実際は区別すべきで、本版は公開コードのindex規則に合わせた。また公開non-flash実装は `(QKᵀ＋bias)/√d` の順に計算する。本版もその順序にした。数学的に似て見える `(QKᵀ)/√d＋bias` では同じ重みで一致しない。

`covariate=True` は別mask。各変数内の過去を見せ、最後の目的変数には外生変数の全patchを見せる公開ルールを使う。最後の変数をターゲットとして配置する契約が必要。一般の多変量入力には通常 `covariate=False` を使う。本blockは公開non-flash経路を実装し、`flash_attention=True` による別kernelの挙動まで同一とは主張しない。

## 学習objective：何を入力し、何を当てるか

各変数 `c` の観測系列を長さPのpatch `x[c,i]:[P]` に分け、時刻 `i` までの全変数patchをcontextとして**次の `x[c,i+1]:[P]`** を出す。モデルのtoken出力 `ŷ[c,i+1]:[P]` と実測 `x[c,i+1]` を対応させ、全変数・全有効token・P点にわたりMSEを平均する。最後の観測patchに対する「次patch」のGTがなければそこは学習lossの対象にできない。原著の予測は同じheadをautoregressiveに繰り返し、生成したpatchを次のcontextに加える。**各時刻を1値ずつ生成するのではなくP点単位**。

本block `forward(x)` は同じ長さTへ写すため、原著の「1 patch先をGTにする」学習を暗黙に行わない。[finetuning_example.py](finetuning_example.py) は独立の実用例で、DataFrameの過去窓 `X:[B,T,Cin]` から同長特徴 `h:[B,T,Cout]` を得て、一例のpooling/headで次行の回帰値やクラスlogitsを予測し、対応GTにMSE/CEを掛ける。原著の自己教師ありnext-patch lossとの違いを保つ。

## 公式処理との照合

[verify_official.py](verify_official.py) は公式OpenLTMの `Model`、`TimerBlock`、`TimerLayer`、`TimeAttention` を**そのままimport**し、`Model.forward` 中の `blocks` 出力をhookで取る。同じ入力・対応重みで本 `forward_features` と比較した。`T=32,P=8,C=3,D=32,L=2`、通常の多変量maskと `covariate=True` の両方で数値一致。公式attention・RoPE・bias・maskは差し替えていない。比較対象は**encoder境界**であり、汎用adapterや公式次patchheadの出力一致と読み替えない。

この検証は公開commitのnon-flash実装、`use_norm=True`、上記設定についてのもの。他のパッチ長・実GPU kernel・任意checkpointの完全互換を自動保証しない。照合を再実行するにはOpenLTMの公式checkoutが必要。

## 大規模入力・計算量・因果性の注意

token数は `Ltoken=C·N`。通常の密attentionは `O(B·(C·N)²·D)`、scoreメモリは `O(B·Hh·(C·N)²)`。`C=1000,N=8` なら一つのheadで6400万scoreを持ち、FP32で約256MB、batch/head/backwardでさらに増える。**1000変数をそのまま入れる汎用高速blockとは言えない**。patchを長くしてNを減らす、系列を選択する、chunk化する等は可能だが、精度・因果性・attention範囲のトレードオフを伴う。効率は実寸法で測る。

attention maskはpatch未来を遮るが、`use_norm=True` の窓統計は入力窓**全T点**から計算する。`forward_features(x)` の早いpatch tokenを「その時点だけで計算したオンライン表現」として読むと、正規化経由の将来情報を含み得る。窓全体が観測済みの下流推論には問題ない。オンライン逐次処理なら各時点までのprefixで統計を再計算するか `use_norm=False` 等を検討する。

## 論文の効率主張と本実装を分ける

論文は多変量を独立処理する方式と比べ、`C·N` token間attentionで系列間依存を直接表現できる点を重視する。反面、dense attentionの計算量は `O(C²N²)` で、channel-independentな `O(CN²)` より重い。原著はFlashAttention等を用いた長文脈学習を論じるが、本blockの公開non-flash経路照合は**score行列 `[B,heads,CN,CN]` を明示的に作る**。原著の高速kernelを使ったときのpeak memoryを本blockへそのまま当てはめてはいけない。1000系列・長い窓では、この制約が採用可否を決める。

論文のattention可視化には同系列の時間依存と異系列の同時点依存が現れる。ただしattention重みの図は、予測に必要な因果関係を証明しない。実際の下流データで系列間相互作用が役立つかは、系列を独立処理するbaselineとの比較で確認する。追加系列が多いほど常に良いわけではなく、ノイズ系列がattention候補を増やす可能性もある。

## maskを具体例で読む

`C=2,N=3`、flatten順が `(c0,t0),(c0,t1),(c0,t2),(c1,t0),(c1,t1),(c1,t2)` とする。query `(c0,t1)` はkey `(c0,t0),(c0,t1),(c1,t0),(c1,t1)` を参照でき、`t2` の両系列は参照できない。**flattenされた1次元列での通常の下三角maskとは違う**：それだと `(c0,t1)` から後ろに並ぶ `(c1,t0)` も誤って隠す。この差がTimeAttentionの実装上の要所。

一方、`covariate=True` では最後の変数をターゲット扱いするため、変数の並び順が単なる表示順でなく入力契約になる。一般の回帰encoderとして全系列を対等に扱うなら `covariate=False` を選ぶ。`forward_features` の `[B,C·N,D]` はpatch単位の内部状態であり、同長adapter後の `[B,T,Cout]` とも原著の次patch予測値とも別物。

## 使用例

```python
from ts_block import TimerXLBlock

block = TimerXLBlock(in_dim=12, out_dim=32, patch_len=8)
h = block(x)                       # x:[B,64,12] -> h:[B,64,32]
tokens = block.forward_features(x) # [B,12*8,D]
```

`T % patch_len == 0` が必要。`python ts_block.py` は実blockのforward/backward、`python verify_official.py --official-repo <OpenLTM-repo>` は公式encoder照合。DataFrame例：`python finetuning_example.py --csv data.csv --features f1 f2 --targets y --seq-len 64`。

## 原著の系列化を具体的に追う

`T=24,P=8,C=2` なら1系列は3 patch、全体で6 tokenとなる。

```text
入力 [B,24,2]
  変数0: x[0:8,0], x[8:16,0], x[16:24,0]  → token 0,1,2
  変数1: x[0:8,1], x[8:16,1], x[16:24,1]  → token 3,4,5
各patch [8] → 共有 Linear(8,D) → 6個の [D] token
```

各patch内の8点はtoken embedding時に同時に入る。したがって「patch時刻 `i` の次patchを予測する」とは、現在patchの8点までを読んで次の8点をまとめて出す意味である。1点ずつのautoregressive予測とは計算の単位が違う。`N=3` はpatch個数、`C·N=6` はattention token個数であり、Cをembeddingの特徴次元に潰しているわけではない。

## 論文モデルと汎用blockを並べて見る

```text
                    共通する公式encoder
 [B,T,C] ──patch── [B,CN,D] ──TimeAttention×L── [B,CN,D]
                                                     │
               原著の次patch学習・予測             │      本surveyの下流encoder
                  ↓                                  │          ↓
             Linear(D,P)                     Linear(D,P) + channel Linear
                  ↓                                  ↓
           未来patch [B,C,N,P]                  特徴 [B,T,Cout]
                  ↓                                  ↓
           次patchの実測とのMSE              任意のpooling/headと下流GT
```

左枝では出力tokenの時刻を1 patch進めてGTと比較する。右枝では**同じ入力窓のT位置へ戻す**。同じ `D→P` 幅であっても教師信号と用途が異なる。blockの`forward_features`と公式のencoder境界を一致確認し、右枝のadapterを「公式のforecast headと等価」とは扱わない。

## TimeAttentionの情報経路を式で確認

query tokenを `(c,i)`、key tokenを `(c',j)` とすると、通常の多変量maskは `M[(c,i),(c',j)] = 0 (j≤i), -∞ (j>i)`。許可される集合は **変数c'に依存せず時刻jだけで決まる**。同じ時刻の他系列を使えるのは、各時点で全系列が観測済みという原著の前提に合う。異なる時刻解像度・遅延報告の外生系列をそのまま入れると、この「同時点で既知」が崩れる可能性がある。

学習可能なbiasは「同一変数ならa、別変数ならb」というheadごとの2値。`C` 個の系列それぞれへ固有のbiasを付ける仕組みではない。よって入力列の意味を学習したい場合は、値の分布や別の埋込から区別する必要がある。コード上のRoPEはflatten後の位置indexを使うので、論文の概念的な時間・変数因子分解と**完全な順序不変性**を同一視しない。

## 実装対応と確認範囲

| 論文上の部品 | 公式OpenLTMでの境界 | 本survey |
| --- | --- | --- |
| patch埋込 | `Model` のpatch投影 | `TimerXLBlock.forward_features` のpatch化・投影 |
| 時間/系列attention | `TimeAttention` | `TimeAttention` |
| attention＋FFNの反復 | `TimerLayer` / `TimerBlock` | `TimerLayer` と反復 |
| 次patch予測 | 公式forecast head | 汎用blockには含めない |
| 同長特徴化 | 原著にはない | `TimerXLBlock.forward` のadapter |

照合は両maskのencoder末尾に対して実施した。高速kernel経路、別形式の外生入力、任意の事前学習済みcheckpointの読み込みはこの比較からは導けない。再利用時は`T%P=0`を守り、`covariate=True`なら最後の列をターゲットに配置する。入力を全窓正規化する設定では、maskだけを見て各tokenがオンラインcausalだと判断しない。

## DataFrame学習例での入力・GTを固定する

学習例は原著のnext-patch事前学習ではなく、任意の次行タスクを示す。`series_id`ごとに`timestamp`昇順へ並べ、label indexを`r`としたとき次を作る。

```text
入力窓: features[r-T:r]       [T,Cin]   ← 最後はr-1行
位相:   不要
教師:   targets[r]            [K]または整数クラス
batch:  history[B,T,Cin] → block[B,T,Cout]
        → 例として時間mean[B,Cout] → head[B,K]
loss:   回帰ならMSE、分類ならCrossEntropy
```

各seriesの早い80%を学習label、後ろ20%を検証labelとし、入力標準化の平均・分散は学習側からだけ求める。検証窓の履歴が学習期間に跨るのは、検証labelより過去の観測として使うため。系列単位で完全隔離した汎化評価が欲しければ、split方法を変更する必要がある。`h.mean(dim=1)`は一例であり、poolingもheadもblockに固定されない。

## 設定値を変えたときの意味

| 値 | 大きくすると | 注意 |
| --- | --- | --- |
| patch長P | token数Nが減り、attentionが軽くなる | P点を一括予測・埋込するので時間解像度が粗くなる |
| 層数L | attentionとFFNの反復が増える | score/activationと遅延が増える |
| hidden幅D | patch表現・QKVの容量が増える | FFNとQKVの計算・パラメータが増える |
| 変数数C | 同時点の他系列を多く読める | dense scoreは概ねC²で増える |

本blockで原著との処理一致を保つなら、patchの並べ方、mask、bias、RoPE位置、正規化順序を一緒に確認する。単に`nn.TransformerEncoder`へ置換して同じshapeを返してもTimeAttentionと等価ではない。

## attention maskの誤実装を見分ける小例

次の表はquery `(c0,t1)` がどのkeyを読めるかを示す。変数順にflattenする公式配置を前提にする。

| key | 単純なflatten下三角 | Timer-XLの多変量mask |
| --- | :---: | :---: |
| `(c0,t0)` | ○ | ○ |
| `(c0,t1)` | ○ | ○ |
| `(c0,t2)` | × | × |
| `(c1,t0)` | × | ○ |
| `(c1,t1)` | × | ○ |
| `(c1,t2)` | × | × |

単純な下三角では、**後ろに並ぶ変数の過去・同時点**まで隠してしまう。これではtokenのshapeが同じでもTimer-XLの多変量処理を再現しない。逆にmaskを外すと未来patchが見え、next-patch GTが漏れる。公式照合を両maskで行ったのはこの違いを確認するため。

## 性能上の主張を読み違えない

原著の狙いは、長い履歴のpatch化と変数間相互作用を組み合わせた統一予測。`C·N` tokenへ全組合せattentionする以上、Cが大きい状況では計算が軽いとは限らない。FlashAttentionによるメモリ削減はkernelに依存し、PyTorchでscoreを明示する本blockのメモリとは区別する。また、maskが因果的でも、全窓統計を使う正規化や**本blockの同長readout**は時刻t以後の入力に依存し得る。原著の「next-patch生成の因果性」と、下流で`h[:,t]`をリアルタイム特徴と見なせることは別。

## 再利用時の確認項目

- 出力が未来のGTに依存しないよう、入力窓の終点を予測時点より前へ置く。
- `covariate=False`なら全系列を同格に扱い、`True`なら最後の系列をターゲットとして並べる。
- 入力Tがpatch長Pの倍数であることを確認する。短い端数を無断で捨てない。
- `C·N` tokenのscoreメモリをBとhead数込みで見積もる。
- 中間tokenを逐次オンライン特徴に使うなら、全窓正規化による未来窓内情報を確認する。

## 一次資料

原論文：[Timer-XL: Long-Context Transformers for Unified Time Series Forecasting](https://arxiv.org/abs/2410.04803)（ICLR 2025）。公式実装：[thuml/Timer-XL](https://github.com/thuml/Timer-XL) が案内する [OpenLTM](https://github.com/thuml/OpenLTM) の `models/timer_xl.py`、`layers/SelfAttention_Family.py`、`layers/Attn_Bias.py`、`layers/Attn_Projection.py`、`utils/masking.py`。論文の参照箇所：Approach / Multivariate Next Token Prediction / TimeAttention / Implementation Details。
