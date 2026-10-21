# TimeXer — endogenous patch と exogenous variate token を接続する

## 概要

TimeXer（NeurIPS 2024）は、予測対象系列（endogenous）と、その予測に役立つ外部系列（exogenous）を**違う粒度でtoken化**するTransformer。対象系列は短いpatchのtokenにして時間内構造を学び、外部系列は履歴全体を1tokenにする。対象系列に追加したlearnable global tokenだけが外部系列tokenへcross-attentionし、その情報を次層の対象patchへ渡す。論文の元タスクは対象未来 `Y:[B,H,1]` または全変数を並列に対象にした `[B,H,C]` 予測。本surveyでは後者の公開 `features='M'` 経路を汎用同長encoder化した。

## 従来法との差分・新規性

すべてを時刻tokenにすれば、外部変数が欠損・低頻度・時刻ずれを持つ時に整列が難しく、系列数が大きいとtoken数が膨らむ。一方、すべてを変数tokenにすると、予測対象系列の短期変動をpatch間attentionで直接扱いにくい。TimeXerは **endogenous=patch＋global、exogenous=変数ごとのwhole-series token** と役割を分ける。global tokenが `patch↔外部変数` の橋になる。既存の自己attentionを新しい演算に置き換えたモデルではなく、**非対称token設計とself/cross-attentionの接続**が新規性。

公開実装の `features='M'` は、`C` 本すべてを順にendogenous対象にして、同じモデル重みで並列処理する。各対象のexogenous集合には全 `C` 変数（対象自身も含む）が入る。論文の単一対象・外部変数だけの説明と、公開マルチ変数forwardの実際を区別する。

## アーキテクチャ全体とshape

`B`: batch、`T`: 窓長、`C`: 変数数、`P`: patch長、`N=floor(T/P)`、`D`: hidden、`H`: 予測長。

```text
入力 X [B,T,C]
  └─ 各 b,c の時間軸instance norm [B,T,C]
       ├─ endogenous branch: 全C変数をそれぞれ対象として処理
       │    [B,C,T] → 非重複patch [B,C,N,P]
       │    → shared Linear(P→D)＋位置埋込 [B,C,N,D]
       │    → 各変数のlearnable global tokenを連結 [B,C,N+1,D]
       │    → [B·C,N+1,D]
       └─ exogenous branch:
            1変数の履歴全体をshared Linear(T→D) [B,C,D]
            各endogenous対象へ同じ外部token集合を渡す [B·C,C,D]

TimeXer EncoderLayer × L:
  1. 対象の N patch＋global でself-attention
       scores [B·C,heads,N+1,N+1]
       residual→norm1 [B·C,N+1,D]
  2. global 1個だけをquery、外部C tokenをkey/valueとしてcross-attention
       scores [B·C,heads,1,C]
       global residual→norm2、patch側はそのまま
  3. 結合 [B·C,N+1,D] → 1×1 Conv FFN→residual→norm3
  → final LayerNorm → [B,C,D,N+1] ← encoder境界

原著: flatten [D·(N+1)] → Linear(→H) → denorm [B,H,C]
汎用block: 同じflatten → Linear(→T) → Linear(Cin→Cout) [B,T,Cout]
```

cross-attentionのqueryは対象のglobal token **1個**。patch tokenが直接外部全変数を読むのではない。外部情報がpatch側へ移るのは、そのglobalとpatchが次の層のself-attentionで交流するとき。したがって `layers=1` と `layers=2` では情報伝播の深さが違う。`forward_features(x)` は公式encoder末尾 `[B,C,D,N+1]`、`forward(x)` は同長task adapterまで含む。

## 具体的な学習objective

原著の単一対象設定では、過去の対象 `x:[B,T,1]` と外部系列 `z:[B,T_ex,C_ex]` が入力。予測値 `ŷ:[B,H,1]` と**対象系列だけの未来実測** `y:[B,H,1]` のMSEを最小化し、外部系列未来の予測損失は掛けない。公開 `features='M'` 経路では、履歴の全 `C` 変数を対象として並列化し、`ŷ:[B,H,C]` と未来GT `[B,H,C]` をMSE比較する。どちらも未来の正解値はencoderへ入らない。

公式コードでは対象・外部の両branchが**正規化済みの同じ履歴**を見る。`EnEmbedding` は `T` のうち `floor(T/P)·P` 点だけを非重複patchに使い、端の余りはendogenous branchで使わない。一方、exogenous `Linear(T,D)` は履歴全体を見る。最終のdenormは各対象変数の入力統計による。汎用blockはforecast headではなく `[B,T,Cout]` の特徴を返すため、そこでdenormはしない。

[finetuning_example.py](finetuning_example.py) はDataFrameの過去 `T` 点→汎用encoder→一例のpooling/head→次行の回帰/分類GTにMSE/CEを掛ける。原著の未来 `H` 点予測とは別の下流利用例。pooling/headはタスクに合わせて差し替えられる。

## 公式コードとの入出力照合

[verify_official.py](verify_official.py) は公開 `models/TimeXer.py::Model`、`EnEmbedding`、`EncoderLayer` 等の**定義本体を無変更で読み込む**。無関係な外部ライブラリ import を避けるためASTで該当クラスだけ抽出。公式 `forecast_multi` 実行中の `encoder` 出力をhookし、同じ入力・対応重みで本blockの `forward_features` と比較した。`T=32,C=3,P=8,D=16,L=2` で一致。これは `features='M'`、時刻marker無し、予測head前の一致。別の `features='MS'` や独立の外部履歴長 `T_ex≠T`、markerあり設定まで保証しない。

モデル側にchannel index embeddingはないが、learnable global tokenは公式 `EnEmbedding.glb_token:[1,C,1,D]` で**変数index別**に存在する。そのためチャンネル順序と `C` は学習・推論で固定が必要。本blockも`Cin`固定である。

## 大きな系列数・速度の問題

公開 `features='M'` はendogenous対象が `C` 本、各対象が `C` 本の外部tokenを見る。cross-attentionだけでも `O(B·C²·D)`、self-attentionは `O(B·C·(N+1)²·D)`。実装上、外部token集合を対象ごとに展開するため `[B·C,C,D]` のactivationも必要。`C≈1000` ではこのまま使うのは相当に重い。`1000` 系列を扱うなら対象変数を絞る、外部候補を選択する、入力を潜在変数へ圧縮する等の設計変更が実務上必要になり得る。ただしそれは**公開の全変数対全変数TimeXerとは別のモデル**になる。予測精度の保証なしに「最適化」と呼ばない。

## 論文のablationが示す情報経路

論文の複数データセット平均MSEは元構成 `0.307`、外生系列を全期間1tokenではなくpatch tokenにする変更 `0.312`、global tokenを除く変更 `0.316`、専用cross-attentionを単なる外生表現の加算へ置く変更 `0.329`、外生tokenを単に連結する変更 `0.312`。つまり、**外生情報を何tokenにするか**だけでなく、global tokenを介してどの演算でendogenous側へ渡すかが結果に効く。ただし個別データでは入れ替わる例もあり、平均値だけを見て「常に全期間1tokenが優れる」とは言えない。

欠損値の実験では外生入力をzero/randomで置き換えた平均MSEが `0.330/0.333`、元構成 `0.307` より悪い。内生入力を同様に壊した場合は `1.125/1.150` とさらに悪い。これは**元の実験条件で内生系列が主信号、外生系列が補助信号**という解釈と整合するが、本blockがNaNを自動処理することを意味しない。欠損を含むDataFrameは事前に埋めるかmask機構を別途設計する。

## 原著の外生変数設定と本blockの範囲

原著は「予測したい内生系列」と「それを助ける外生系列」を明確に分ける。外生系列は別データソース・別サンプリング間隔でも構わず、論文の実験には時刻解像度の違う系列がある。本blockは汎用 `[B,T,C]` 契約のため**全列を同じTで受け、全列を内生候補にも外生候補にも使う**公式 `features='M'` 経路を実装した。別長の `X_ex:[B,T_ex,Cex]` を受ける二入力APIや、特定ターゲット1列だけを予測する軽量設定は含まない。公式encoder一致を確認したのはこの同長・多変量経路であり、論文の全適用範囲をカバーしたという主張ではない。

endogenousのpatch列 `[B·C,N,D]` は対象系列の局所的な時間形状を保持し、exogenousのglobal token `[B·C,C,D]` は各外部系列の窓全体を表す。cross-attentionは両者の**時間粒度の異なる情報**を結びつける。単に入力時点でC列をLinear混合してからpatch化すると、外生と内生を別粒度で扱う原著の構造は失われる。

## 組込例と制約

```python
from ts_block import TimeXerBlock

block = TimeXerBlock(in_dim=12, out_dim=32, seq_len=64,
                     patch_len=8, d_model=64)
h = block(x)                       # [B,64,12] -> [B,64,32]
tokens = block.forward_features(x) # [B,12,64,9]
```

`T` は外部series embedの入力幅なので固定。`N=floor(T/P)`、出力のflatten readout幅も固定。現blockは全入力列をendogenous候補として扱うため、「特定ターゲット1本＋外部変数多数」の軽量用途にはそのまま適さない。最後の `Cin→Cout` は常にlearnableで、`Cin=Cout`でも省略しない。

`python ts_block.py` でforward/backward、`python verify_official.py --official-repo <TimeXer-repo>` で公式encoder比較。DataFrame例：`python finetuning_example.py --csv data.csv --features f1 f2 --targets y --seq-len 64`。

## 1層と2層で外生情報の届き方が違う

`N=4,C=3,D=16` の1対象について、endogenous側は4 patch＋globalの計5 token、exogenous側は3変数の計3 tokenになる。

```text
層1開始: patch[4,16] + global[1,16]
  self-attention: 5 token間で情報交換 → patch/global双方が更新
  cross-attention: globalだけが外生3 tokenを読む → globalが再更新
  FFN: 各tokenの特徴を変換（新たなtoken間伝播はない）

層2開始: patch[4,16] + 外生情報を得たglobal[1,16]
  self-attention: patchがglobalを参照できる → 外生情報がpatchへ届く
```

このため、1層のencoder出力でもglobalには外生情報があるが、**cross-attention後の同じ層でpatchがそれを直接読むわけではない**。最終readoutはpatchとglobalをすべてflattenするので1層でも外生情報を出力に利用できる。ただしpatch token自体へ反映させるには後続層が必要。この情報経路は、全patchを外生tokenへ直接cross-attendさせる設計とは違う。

## 原著と本版の二つの図

```text
原著の単一対象:
  endogenous[B,T,1] ─→ patch + global ─┐
                                      ├→ encoder → Linear(D·(N+1),H)
  exogenous[B,T_ex,Cex] → 各系列1token ─┘             → 未来対象[B,H,1]

公式features='M' / 本surveyが照合した経路:
  全入力[B,T,C] → 対象をC本に展開 [B·C,T,1]
               → 各対象のpatch + global [B·C,N+1,D]
  同じ全入力 → 外生token集合 [B,C,D]
               → 各対象へ複製 [B·C,C,D]
  encoder → [B,C,D,N+1]
        ├→ 公式: 未来H点×C系列
        └→ 本版: 同長T点×Cout特徴
```

上図の二入力単一対象と、下図の全C対象経路は同じ説明にまとめてはならない。特に下図では対象自身も外生token集合に含まれ、`C`回のcross-attentionにそれぞれC個のkey/valueを与えるため、大Cで二乗コストが現れる。

## 公式部品とsurvey実装の対応

| 役割 | 公式実装 | 本survey |
| --- | --- | --- |
| 対象patch＋global | `EnEmbedding` | `TimeXerBlock` のpatch投影・global token |
| 外生変数token | `ExEmbedding` | 全履歴`T→D`投影 |
| self / cross / FFN | `EncoderLayer` | `TimeXerLayer` |
| attention kernel | `FullAttention` | `FullAttention` |
| 多層encoder | `Encoder` | block内の層反復 |
| 未来horizon | forecast head | 汎用blockの範囲外 |

`forward_features`までを公式から照合した理由は、以後の`H`へのflatten headがタスク依存だから。`T`を入力幅とする外生埋込、対象のpatch数、変数別global tokenを公式に合わせることが、汎用化してもencoderの同一性を保つ条件になる。入力列をすべて対象として使う必要がないタスクでは、より軽量な単一対象経路を別途設計する余地があるが、このblockの公式照合結果をそのまま流用できない。

## 実験設計での注意

外生列を入力する場合、予測時点で実際に入手できる過去値だけで窓を作る。たとえばマクロ統計の公表遅延があるのに、元データの対象日時だけでjoinすると、未公表の外生情報が入力へ混ざる。これはモデルのcross-attentionとは別のデータリーク問題。欠損補完時も未来を参照する補完をしない。短い局所パターンはPとNで、全履歴の要約は`T→D`で解像度が決まるため、両者を同じ時間粒度と見なさず設定を調整する。

## DataFrameからの下流学習を具体化

`series_id`別にtimestampを整列し、ラベル行rの直前T行を入力とする。学習サンプルは次の経路で、**公式forecastのH点未来GTとは別**。

```text
features[r-T:r]                  [T,Cin]（全列を対象/外生の双方へ）
TimeXerBlock                    [T,Cout]
一例のmean pooling              [Cout]
タスクhead                     [K]
targets[r]                     [K]の回帰値／整数クラス
MSEまたはCrossEntropy           scalar
```

この例では、C列のどれが「外生だけ」なのかをAPIで区別しない。公式の`features='M'`経路に合わせて全C列を候補として処理するためである。外生系列だけを増やして特定ターゲットを予測したいなら、単一対象経路を別モジュールとして実装する方が二乗計算を避けやすい。入力列に将来時刻の値を含めないこと、公開遅延のある変数を過去観測として誤って結合しないことが必須。

## 設定値の選び方とトレードオフ

| 設定 | 効果 | 注意 |
| --- | --- | --- |
| patch長P | 対象側のtoken数Nを減らす | 端の`T mod P`点は対象patchへ入らない |
| 層数L | globalとpatchの複数回の情報交換 | L=1ではcross後にpatchがglobalを再読込しない |
| hidden幅D | 対象・外生tokenの容量 | 全C対象への展開量も増える |
| 変数数C | 外生候補を増やす | 公開M経路は対象C×外生Cの組を処理 |

Pを長くすれば対象self-attentionの`(N+1)²`は減るが、外生C個へのcross-attentionの`C²`は消えない。系列1000本を入れる用途では、patch長だけを調整しても根本のボトルネックは残る。実測する場合はB,C,T,P,D,Lを記録して比較する。

## `C=1`と`C≫1`での意味の違い

`C=1`なら外生tokenも対象自身の1個だけで、cross-attentionのsoftmaxは長さ1。外部系列のどれを読むか選ぶ機構ではなくなる。Cが増えると各対象のglobalが他系列tokenを選択できるが、全対象C本×候補C本になる。つまり本blockの計算は「外生情報を1回だけ全体へ混ぜる」処理ではない。各対象ごとにqueryが違い、同じ候補集合へ問い合わせる。

| 経路 | token数 | Cを増やしたとき |
| --- | --- | --- |
| endogenous self-attention | 各対象`N+1` | C対象ぶん、概ね線形 |
| exogenous whole-series embed | C | 概ね線形 |
| global-to-exogenous cross | 各対象`1×C` | C対象ぶんで概ね二乗 |
| 最終channel projection | `Cin→Cout` | `Cin·Cout`の重み |

公開実装と同じ演算を維持するか、単一ターゲットだけに限定してコストを下げるかは用途次第。後者は実務上有用でも、この`features='M'`の公式照合済みblockと同じモデルとは呼ばない。

## 原著の学習で未来外生系列をどう扱うか

原著のendogenous forecastのGTは未来の対象系列。外生系列は**予測を助ける入力**として扱われ、その未来を同時に当てることは単一対象設定では要求されない。一方、全Cを対象とする公開M経路では各系列が出力対象にもなる。現在のDataFrame学習例は全C入力を同長特徴へ変換した後、利用者が指定するtarget列だけを外部headで予測する。この3つを同じ教師信号だと見なさない。

時間解像度の異なる外生情報を使う元の問題設定に対して、本blockの同一TのDataFrame入力は制限付き。もし低頻度外生系列を単純に前方補完して同一Tへ揃えたなら、補完の規則と公表可能時刻が性能・リークの両方に影響する。公式の二入力設定を忠実に使いたい場合、このblockを無理に拡張したと主張せず、別APIでT_exを保持して実装・検証する必要がある。

## `forward_features`を利用する場合

返る`[B,C,D,N+1]`の最終軸はpatch N個とglobal 1個。globalは外生情報をcross-attentionで受ける窓全体の集約tokenであり、patchの1つと同じ時刻位置ではない。単純にN+1個を時刻として平均すると、globalを1時刻と誤解した重み付けになる。時刻方向のpooling設計ではpatchとglobalを区別する。本版の`forward`は両者をflattenして学習可能な`D·(N+1)→T`でまとめるので、この混在をheadが学習できる。

## 導入前の確認項目

- C本すべてを対象にする必要があるか、特定の1本だけでよいか。
- 外生値は予測時点で実際に公表済みか。
- 入力TがPで割り切れず、対象patch側だけ端の点を落としていないか。
- Cが大きいとき、対象C×外生Cの展開を許容できるか。
- 変数順を学習時と一致させ、変数別global tokenの意味を保っているか。

## 一次資料

原論文：[TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables](https://arxiv.org/abs/2402.19072)（NeurIPS 2024）。公式実装：[thuml/TimeXer](https://github.com/thuml/TimeXer)。照合箇所：論文 TimeXer / Forecasting Loss、公開 `models/TimeXer.py` と `layers/Embed.py`。
