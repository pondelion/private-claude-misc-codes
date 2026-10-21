# TimeBridge — 短期の各変数内変動から長期の系列間関係へ

## 概要

TimeBridge（ICML 2025）は、時系列の短期変動をそのまま変数間でattentionすると、非定常な衝撃による見かけの相関を拾いやすい、という問題意識から出発する。最初に**各変数内**でpatch時系列のattentionを行い、そのquery/keyだけを平滑化する。次にpatch数を減らして長い区間の情報を集約し、最後に**同じ縮約patch位置の変数間**でattentionする。短期の「integrated attention」と長期の「cointegrated attention」を段階的につなぐことがTimeBridgeの主眼。原著は `[B,T,C]→[B,H,C]` の未来予測。本surveyの[ts_block.py](ts_block.py)は同じencoderを同長 `[B,T,Cout]` の汎用ブロックへ接続する。

## 既存法との差分と技術上の新規性

PatchTSTは変数内patch attentionに強いが、backbone内の変数間相互作用を設けない。iTransformerは変数間attentionに強いが、各変数の全履歴を1tokenへ圧縮してから相互作用する。TimeBridgeは**patch内の短期情報を学ぶ段階**と**長期集約後に変数間を融合する段階**を分ける。原著の「cointegration」は統計的概念を動機にした命名であり、モデルが厳密な単位根検定やJohansen検定をforward内で実行するという意味ではない。実際の実装は平滑化・attention・patch downsamplingの神経ネットワークである。

## アーキテクチャと各処理のshape

`B`: batch、`T`: 履歴長、`C`: 入力変数数、`P`: patch長、`N=T/P`、`M`: 縮約後patch数、`D`: hidden、`H`: 原著予測長。公開コードは時刻markerが無ければ4列のゼロmarkerを加えるので、encoder内の変数数は `C+4`。図の例は `T=32,P=8,N=4,M=2`。

```text
history X:[B,T,C], marker:[B,T,4]（未指定ならゼロ）
  → 各変数の窓平均/標準偏差で入力正規化 [B,T,C]
  → markerを列方向連結 [B,T,C+4]
  → 非重複patch化＋shared Linear(P→D) [B,C+4,N,D]

  IntegratedAttention × ia_layers   [B,C+4,N,D]
    変数ごとに [B·(C+4),N,D]
    Q,K = PeriodNorm(x)（D方向の移動平均を引く、no_grad）
    V = 生のx
    attention scores [B·(C+4),heads,N,N]
    residual→LayerNorm→FFN→residual→LayerNorm

  PatchSampling × pd_layers         [B,C+4,M,D]
    Conv1d(N→M)でquery候補、global patch meanを連結
    Conv1d(M+1→M)＋skip → Q [B·(C+4),M,D]
    元patchのK,Vは [B·(C+4),N,D]
    cross attention [B·(C+4),heads,M,N]
    residual→LayerNorm→FFN→LayerNorm

  CointAttention × ca_layers        [B,C+4,M,D]
    同じpatch位置の全C+4変数を融合
    公式既定 axial=True: C+4を正方形R×Rへpadし、行attention→列attention
    直接full attention設定なら score [B·M,heads,C+4,C+4]
    residual→LayerNorm→FFN→LayerNorm

  encoder出力 [B,C+4,M,D]（原著はdataのC列だけ残す）
  原著 flatten(M·D)→Linear(→H)→denorm [B,H,C]
  汎用block flatten(M·D)→Linear(→T)→Linear(Cin→Cout) [B,T,Cout]
```

`forward_features(x,markers)` はmarker token込みで `[B,C+4,M,D]` を返す。`forward(x,markers)` はdata変数の `C` tokenだけをreadoutし、最後のchannel射影で常に系列を融合する。encoder内でも **CointAttentionで out projection 前に融合済み**。`pd_layers=0` なら `M=N`、複数downsampling層では最初だけ `N→M`、以後は `M→M` の公式動作を保持する。

## Integrated/cointegrated attentionを数式と実装で区別

Integrated側では `Q=K=PeriodNorm(x)`、`V=x`。`PeriodNorm` はtoken系列の時間軸 `N` ではなく、各patch tokenの**特徴軸 D**に長さ `stable_len` の移動平均を計算し、元値から引く。論文の「patchごとのstationary化」を読んで時間軸移動平均だと誤実装しやすい。公開 `PeriodNorm` のコードに合わせた。また `with torch.no_grad()` 内でquery/key用の平滑化を計算するため、この経路そのものには勾配が通らない。ただし `V=x`、residual、FFNを経由して入力への勾配は残る。

Cointegrated側では平滑化したQ/Kを使わず、縮約後の非定常表現を変数方向でattentionする。公式の既定はaxial経路で、`R=ceil(sqrt(C+4))` として0 padし、行attention、転置した列attention＋skip、最後にpad変数を除く。これは単一の全 `C×C` attentionとは**同じ計算ではない**。本blockは両方を実装するが、公式照合は既定axialで行った。

## 学習objective：入力、予測、GT

原著の入力は履歴 `X:[B,T,C]` と任意の時刻marker `[B,T,4]`。encoder `[B,C+4,M,D]` のうち**data C列**だけをflattenし、`Linear(M·D→H)` で未来 `Ŷ:[B,H,C]` を出す。入力窓の各変数統計でdenormした後、未来実測 `Y:[B,H,C]` とのMSEを最小化する。未来GTはencoderのvalueやmarkerには入れない。marker未指定の公式forwardはゼロを使う。

[finetuning_example.py](finetuning_example.py) は原著予測headを使わない別の組込例。DataFrameの履歴を `h:[B,T,Cout]` にし、一例のpooling/headから次行の回帰値または分類logitsを出して MSE/CEを掛ける。headやpoolingは下流タスク次第で交換できる。論文の未来全区間MSEと取り違えないこと。

## 公式との一致をどう確認したか

[verify_official.py](verify_official.py) は公開 `TSEncoder`、`PeriodNorm`、`IntAttention`、`PatchSampling`、`CointAttention`、`PatchEmbed`、`Model` の**公式定義を無変更で抽出**し、実際の `Model.forward` を動かす。外部importが不要な対象定義だけASTで読むが、forward内容を差し替えない。`PeriodNorm` を単体でも比較し、`T=32,C=3,P=8,M=2,D=16` で公式encoder末尾 `[B,C+4,M,D]` と本 `forward_features` が同じ入力・対応重みで一致した。未来headは汎用encoderの比較対象外で、同長readoutは独自adapterと明示する。

公開forecastは時刻markerを4列のゼロに補う場合がある。本比較もmarker無し（ゼロ）設定。異なる実markerを渡す場合は同じ値を双方に与える必要がある。各patchのサイズ、`stable_len`、axial/fullの設定が一致しなければ一致判定にはならない。

## 1000系列でのコストと制約

Integrated側は系列数 `C` に線形で、`O(B·C·N²·D)` 程度。縮約後のfull cointegrationは `O(B·M·C²·D)`、公式既定axialなら各軸長 `R≈sqrt(C)` を使うので概算 `O(B·M·C·sqrt(C)·D)` のattention演算になる。ただし実際のactivation・pad・head数・kernel実装でpeakは変わる。`C≈1000` ではfull attentionよりaxialが有利でも、patch×系列のactivationと中間コピーが重い。入力Linearで早期融合・圧縮する手もあるが、原著の各変数を別tokenとしてcointegrationする意味は変わるため、実験で比較する。

`T` は `period` で割り切れる必要がある。本blockは `T` と `C` を構築時固定する。markerをゼロで省略して学習したモデルへ、推論時だけ意味あるmarkerを与えるのは分布変更。列順とmarkerの定義を揃える。

## 論文のablation：順序と正規化は交換できない

TimeBridgeの核は二種類のattentionを単に並べることではない。Integrated側は各変数のpatch間を時間方向に扱い、Cointegrated側は時間方向を短くしたうえで系列間を扱う。論文のablationでは **Integrated→Cointegrated** の順序が、逆順・どちらか一方より良い。したがって本blockで二つのencoder段を入れ替えると、shapeが通っても公式のモデル処理ではなくなる。

正規化のablationでは、ElectricityのMSEが「Integrated側は正規化、Cointegrated側は原系列を保つ」原設定で `0.149`、両側を原系列にすると `0.153`、順序を逆にすると `0.155`、両側を正規化すると `0.153`。この差は巨大ではないが、**系列間の長期的水準差まで消すとcointegrationの手掛かりを失い得る**という設計理由に合う。Trafficでは原設定 `0.360` に対し逆設定 `0.381`。これらは原著の予測タスクの値で、別タスクや本blockの同長readoutの精度を保証しない。

## `forward_features`の意味と下流の注意

出力 `[B,C+4,M,D]` の先頭C行がデータ系列、追加4行は時刻marker用。`M` はpatchを安定長へ縮約した時間token数で、**元のT時刻に1対1対応しない**。同長readoutはこの圧縮表現をT点へ投影し、最後のchannel projectionで系列を再融合する。原著encoder内でもCointegrated側で系列間融合済みなので、この最終Linearが初めての融合ではない。

`forward_features` や `forward` は入力窓全体に依存する。したがって入力窓中の時刻 `t` の出力を、時刻 `t` 時点で利用可能なオンラインcausal特徴と解釈しない。系列単位の将来回帰・分類なら、観測済みの窓全体だけを入力することでリークを避ける。markerを使う場合は未来GTから得た特徴ではなく、当該入力窓で既知のカレンダー値に限定する。

## 使用例

```python
from ts_block import TimeBridgeBlock

block = TimeBridgeBlock(12, 32, seq_len=64, period=8, num_p=4)
h = block(x)                       # [B,64,12] -> [B,64,32]
tokens = block.forward_features(x) # [B,16,4,D]（data12＋marker4）
```

`python ts_block.py` はforward/backward、`python verify_official.py --official-repo <TimeBridge-repo>` は公式encoder境界を検証。DataFrame例：`python finetuning_example.py --csv data.csv --features f1 f2 --targets y --seq-len 64`。

## 2種類のattentionを図で分ける

`T=32,P=8,C=3` でmarker4列を付けると、patch埋込後は `[B,7,4,D]`。ここで `7=C+4`、`4=N`。二つのattentionは次元が明確に違う。

```text
Integrated：各系列 token を固定して「patch軸」を読む
  7系列それぞれ: [4,D] → attention score [heads,4,4]
  結果: [B,7,4,D]
  ↓ PatchSampling で4patch→M=2patch
  結果: [B,7,2,D]

Cointegrated：各縮約patch位置を固定して「系列軸」を読む
  2位置それぞれ: [7,D] → axial系列間attention
  結果: [B,7,2,D]
```

最初から系列を混ぜない理由は、短い局所変動と長い系列間関係を分ける設計だから。PatchSamplingを飛ばせばCoint側がより多くのpatch位置で動くが、原著既定の「安定化した少数token間で融合」という計算・帰納バイアスは変わる。marker4系列もpatch化されてencoderへ入るので、**Cだけ**でattentionの系列数を見積もると誤る。

## axial attentionの近似を追う

系列数`V=C+4`を`R=ceil(sqrt(V))`として`R²`へ0 padし、`[R,R]`の格子とみなす。各行のR系列間attentionと、格子を転置した各列のR系列間attentionを順に掛ける。scoreはfullの`[V,V]`一枚ではなく、各軸の`[R,R]`を複数作る。これにより系列方向のコストを抑えるが、**full attentionと厳密に同じ重み付けにはならない**。`V=1004`なら`R=32`、格子は1024位置で20個のpad位置を含む。padと実系列の区別、並び順、行→列の順序を変更すると公式の結果は変わる。

## 原著・汎用block・下流head

```text
共通: 履歴[B,T,C] + marker[B,T,4]
   → patch[B,C+4,N,D]
   → Integrated → PatchSampling → Cointegrated
   → encoder[B,C+4,M,D]

原著: dataのC系列のみ選ぶ → flatten[M·D] → Linear(M·D,H)
   → 逆正規化 → 未来予測[B,H,C] → 未来GTとのMSE

本版: dataのC系列のみ選ぶ → flatten[M·D] → Linear(M·D,T)
   → channel Linear(Cin,Cout) → 同長特徴[B,T,Cout]
   → 別のpooling/head/下流GT
```

時刻markerはencoder内で4個の追加系列tokenとして働くが、**出力として予測する対象ではない**。公式forecastでも本blockでもreadout前にmarker列を除く。これを残して`C+4→Cout`にすると、公式のencoderを使っていてもreadoutの意味が変わる。

## 公式実装との対応

| 処理 | 公開コード | 本survey |
| --- | --- | --- |
| patch化 | `PatchEmbed` | `TimeBridgeBlock`内のpatch投影 |
| 各系列内の時間attention | `IntAttention` + `PeriodNorm` | `IntegratedAttention` |
| patch数を減らす | `PatchSampling` | `PatchSampling` |
| 縮約後の系列間attention | `CointAttention` | `CointegratedAttention` |
| encoder段の連結 | `TSEncoder` | blockのforward_features |
| 固定Hの未来予測 | 公式forecast head | 汎用blockには含めない |

公開`PeriodNorm`は特徴軸Dでの操作であり、図の`N=4`個のpatchを平滑化する操作ではない。さらにQ/K経路は`no_grad`、Vは元表現という仕様が、数値だけでなく勾配経路にも効く。見た目が似たLayerNormや時間移動平均へ置き換えると公式encoderではない。照合スクリプトはPeriodNorm単体とencoder末尾の双方を比較する。

## 導入前の判断

短期の各系列パターンと長期の系列間関係を分けて扱いたい場合に候補となる。ただし`period`が入力Tを割り切ること、固定Tのreadoutを使うこと、markerの4列を一貫した定義にすることが前提。未知のmarkerを訓練時と違う値で補うと分布が変わる。時刻ごとにリアルタイムで得る特徴を必要とする場合は、窓全体を見るこのencoderをそのまま各時刻へ貼り付けない。出力`[B,T,Cout]`は系列単位の窓から作られる同長特徴である。

## DataFrame学習例と原著objectiveの差

原著の教師は履歴窓より先の**H点×C変数**。一方、同梱の学習例は任意の回帰・分類を示すため、窓末尾の次の**1行**のターゲット列を使う。

```text
入力 history = feature列の r-T ... r-1行   [T,Cin]
時刻marker = 未指定なら4列のゼロ             [T,4]
encoder内部                               [Cin+4,M,D]
汎用blockの同長特徴                       [T,Cout]
例のmean pooling                          [Cout]
task head                                 [K]
GT = target列の r行                       [K] / 整数class
loss                                      MSE / CrossEntropy
```

`series_id`ごとに時刻を並べて窓を切り、異なる系列の境界を跨がない。label indexの早い80%をtrain、後ろ20%をvalidationとする。入力特徴の標準化はtrain期間だけで推定する。カレンダーmarkerを使う拡張では、対象時点に既知の時刻情報だけを使い、ラベル値や未来の測定値をmarkerに混ぜない。平均poolingはタスクheadの**一例**であり、TimeBridgeの論文objectiveではない。

## 設定値とshape制約を整理

| 設定 | 作用する軸 | 変更時の影響 |
| --- | --- | --- |
| `period=P` | 時刻T→patch N=T/P | patch投影幅とtoken数を変更 |
| `num_p=M` | PatchSampling後のpatch数 | cointegrated側の演算数・readout幅を変更 |
| `stable_len` | Q/K用の特徴軸D | `PeriodNorm`の平滑化範囲を変更 |
| `axial` | 系列軸C+4 | full系列attentionとの計算・結果を変更 |
| `d_model=D` | token特徴幅 | QKV・FFN・head数整除条件を変更 |

`stable_len`を「何patch前まで見るか」と読むのは誤りで、公開コードではtokenの特徴次元D上に作用する。`num_p`を小さくすると系列間attentionは軽くなる一方、時間方向の情報を少数tokenへより強く圧縮する。マーカー4列を追加するため、Cが少ないデータではその4tokenの相対的な比率も大きい。

## 小さいCでもmarkerとaxial padは無視できない

`C=3`でもencoder上はmarkerを含めて7系列。`R=ceil(sqrt(7))=3`なので9格子位置へpadし、2位置がpadになる。これは「C=3のfull attention」と同じではない。最終readoutでは元のC系列だけを取り出すが、encoder中ではmarker系列とpadした格子構造の影響を受ける。**marker無しならC系列だけ**とみなして自作すると公式処理からずれる。ゼロmarkerでも4系列分の埋込経路が存在する。

`num_p=M`が小さいと系列間attentionを行う時間位置は少なくなる。長窓Tの全patchでCointAttentionを行うモデルと比べ、計算を抑える代わりに高周波の局所情報はIntegrated段とPatchSampling段で要約される。短いイベントのタイミング自体を下流で当てたい場合は、圧縮率と同長readoutの影響を検証する。

## 正規化を入れ替えた場合の解釈

Integrated側のQ/KにだけPeriodNormを使い、Vには原信号を残す設計は、**attentionの参照先を決める手掛かり**と、**伝播する値**を分ける。Q/K/Vすべてを正規化すると伝播値の長期的な水準差も消える。逆に全て非正規化では短期の非定常ショックがattention重みへ強く入り得る。論文のablationはこれらの構成を比べており、READMEの数値はその予測設定での結果。原著の順序・演算を保存するため、携帯版で単純なLayerNormへ置換しない。

## 比較する際の最小条件

公式encoderと出力を比較するなら、同じ入力Xだけでなく、時刻markerの4列、patch period、縮約後token数M、Integrated/Cointの層数、axial設定、重みを合わせる。とくにmarker無しを「4列のゼロ追加」として処理するか「4列を削除」として処理するかは別物。`PeriodNorm`を特徴軸Dではなくpatch軸Nに実装してもshapeが合う場合があるが、数値も勾配経路も一致しない。対応する公式`Model.forward`からencoderをhookして照合する理由は、部品単位のshapeテストだけではこの種の誤りを検出できないからである。

encoder境界の`[B,C+4,M,D]`は、data系列とmarker系列を含む。最後の4系列を下流headの予測対象にしない点、`M`がTではなく縮約後patch数である点を確認する。

Integrated側のみを見ると系列間融合はまだ起きていない。Cointegrated側の出力までをencoderとして利用してはじめて、このモデルの主眼である長期的な系列間相互作用が含まれる。

## 一次資料

原論文：[TimeBridge](https://arxiv.org/abs/2410.04442)（ICML 2025）。公式実装：[TimeBridge](https://github.com/Hank0626/TimeBridge)。照合箇所：論文 `sections/3-method.tex` と Appendix、公開 `model/TimeBridge.py`、`layers/Transformer_EncDec.py`、`layers/SelfAttention_Family.py`、`layers/Embed.py`。
