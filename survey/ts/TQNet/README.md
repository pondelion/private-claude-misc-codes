# TQNet — 周期位相で変化する query による系列間融合

## 概要

TQNet（ICML 2025）は、多変量時系列の**系列間相関**を、入力値から query を作る通常のself-attentionとは異なる方法で学ぶ予測器。周期 `W` に沿った学習可能な表 `θ:[W,C]` から、各窓の開始位相に応じて長さ `T` の query を切り出す。query が全変数の観測履歴を key/value として参照し、続く浅い MLP が時間方向の表現を変換する。原著の目的は `X:[B,T,C]→Y:[B,H,C]` の予測。本 survey の [ts_block.py](ts_block.py) は同じencoder骨格を汎用 `[B,T,Cin]→[B,T,Cout]` にした。

## 新規性と既存技術との差分

通常の変数間attentionは `Q=XWq,K=XWk,V=XWv` とするため、1サンプルの異常値やノイズが query と key の両方を動かす。TQNetは `Q=θ_phase Wq`、`K=XWk,V=XWv`。queryはデータセット全体で学ぶ周期テンプレートだが、key/valueは**そのサンプルの観測**なので、全サンプルで同じattention結果になるわけではない。CycleNetのような周期表の直接加減算でもない。周期位相で**相関を問い合わせる側**を変える点が核となる。

`W` は「予測したい未来長」でも「Transformerのhead数」でもなく、周期表の行数。時刻 `t` を `W` で割った余りが窓先頭の位相。安定周期の事前知識やACFから決める前提で、未知・不安定な周期では query の有効性が落ち得る。

## アーキテクチャ図と全shape

`B`: batch、`T`: history length、`C`: 入力系列数、`W`: 周期、`D`: hidden、`H`: 予測長。

```text
history X [B,T,C]        phase p [B]（各窓の最初の時刻 mod W）
  │                         │
  ├─ 各系列をT軸でinstance norm    θ[W,C]からθ[(p+t) mod W,c]
  │   [B,T,C]                      [B,T,C] → transpose [B,C,T]
  └─ transpose [B,C,T] ─────────────┐
             │ K,V                  │ Q
             └──── TQ-MHA ──────────┘
                    attention [B,heads,C,C]（系列間）
                    channel information [B,C,T]
        x + channel information [B,C,T]
             → shared Linear(T→D) [B,C,D]
             → Linear→GELU→Linear→GELU [B,C,D]
             → residual + input [B,C,D]  ← encoder 境界
             → 公式 Dropout→Linear(D→H) [B,C,H]
             → transpose + instance denorm [B,H,C]

本ブロックの差分:
  [B,C,D] → Dropout→Linear(D→T) → [B,T,C]
          → Linear(Cin→Cout) → [B,T,Cout]
```

PyTorch `nn.MultiheadAttention(embed_dim=T,num_heads=heads,batch_first=True)` を公式と同じ形で使う。`C` が token 数、**埋込幅は `T`**。そのため `T % heads == 0` が必須で、他モデルの `d_model % heads` と混同しない。`heads=4` が公式固定値。本blockは明示的に変更可能だが、公式照合は4で実施した。

## Temporal Queryの具体例

`W=4,T=8,p=3` なら query 用行インデックスは `3,0,1,2,3,0,1,2`。この8行それぞれに `C` 個の学習パラメータがあり、transposeして `Q_raw:[C,8]` になる。`T=8` は既定4 headで割り切れる。batch の各サンプルで開始位相 `p` が違えば query も変わる。`phase=None` の本blockは全窓を `p=0` とみなすだけで、位相を推定しない。実データで周期位相があるなら必ず `phase:[B]` を渡す。

attention の query はテンプレートから、key/value は正規化済み履歴から作る。`A[b,h,c_q,c_k]=softmax_{c_k}(QKᵀ/√(T/heads))`。`A` を value に掛けて `channel_information[b,c_q,:]` を得る。異なる変数を読めるので、**out projection前から系列間融合あり**。時間依存を直接表すのは主に `Linear(T,D)` と MLP であり、attention は変数軸 `C` に掛かる。

## 学習 objective：何を予測し、何にlossを掛けるか

原著では履歴 `X:[B,T,C]` とその開始位相 `p:[B]` を入力し、未来の全変数 `Y:[B,H,C]` をGTとする。`TQ-MHA→MLP` の隠れ表現 `[B,C,D]` を `Linear(D,H)` で未来系列へ変換し、窓の統計でdenormした `Ŷ:[B,H,C]` と実測 `Y` のMSEを最小化する。未来真値はqueryやkey/valueに入力されず、loss計算にだけ使う。`temporalQuery` 自体にもこのlossから勾配が流れる。

公式コードの instance norm は `torch.var` の既定（不偏分散）を使い、mean/varianceはdetachしない。iTransformerの同名に見える正規化とは差がある。本blockのencoder前半は公式順序に合わせた。原著のdenormは**forecast headの後**であり、本blockの同長特徴には適用しない。別タスクのfeatureに入力単位を強制的に戻すと意味が変わるためである。

[finetuning_example.py](finetuning_example.py) は別の下流例。過去窓から `[B,T,Cout]` を得て、一例としてpooling→外部head→次行の回帰または分類GTにMSE/CEを掛ける。`phase_col` はDataFrameに必要で、各窓**先頭行**の位相を渡す。未来の位相を混ぜない。pooling/headは固定の「標準」ではなく、用途に合わせて置換可能。

## 公式との照合・意図的な境界

[verify_official.py](verify_official.py) は公開 `models/TQNet.py::Model` をそのままimportして実行。公式 `input_proj` と `model` の出力をhookで取得し、その和（公式のhead直前）と `TQNetBlock.forward_features(x,phase)` を同じ入力・重みで比較した。`T=16,C=3,D=16,W=12`、非ゼロ位相を含む設定で数値一致。後続の `D→T` readoutと `Cin→Cout` は汎用化のためのadapterであり、公式の固定未来長headと同一とは主張しない。

`forward_features` の比較が通っても、任意の周期 `W` がデータに正しいとは保証しない。実行時には `seq_len`、`cycle_len`、チャンネル順序を訓練時と一致させる。`T` は attention のembed幅と入力Linear幅なので、構築した同じ重みで任意Tには変更できない。

## 多数系列での計算量・実用上の選択

系列数 `C≈1000` ではTQ-MHAのscoreは `[B,heads,C,C]`、計算概算 `O(B·C²·T)`。query表のパラメータは `W·C`、入力 `Linear(T,D)` は共有なのでCに線形だが、attentionの二乗項が重い。例えば `B=16,heads=4,C=1000,FP32` のscoreだけで約256 MB、訓練時はさらに必要。大規模では変数選択・分割・入力射影を検討できるが、射影後の潜在系列に対するTQは**元の1000系列それぞれを問い合わせる原著処理とは違う**。速度と精度の比較で採否を決める。

## 論文のablationと成立条件

論文のquery/key比較は設計意図を明確にする。raw履歴を両方に使うquery/keyは「その窓の値だけで相関を決める」設定、学習可能なTQをquery・raw履歴をkeyにする本方式は「**周期からの大域的な問い合わせ**と**現在の観測による選択**」を分離した設定、query/keyを両方学習可能にすると窓固有の変動を取り込むkeyがなくなる。報告された順位を読む際は、この情報経路の違いと一緒に見るべき。TQ moduleの除去、attentionの除去、単なるMLP化も論文で比較され、周期queryと系列間相互作用を別々に検証している。ただしこれらは原著のforecast headとデータでの結果であり、汎用blockの別タスク精度保証ではない。

著者が挙げる制約は**明瞭で安定した周期が必要**なこと。位相 `p` が誤ると `θ[(p+t) mod W,c]` の参照位置もずれ、学習時と異なるqueryになる。周期が時間とともにドリフトする系列、複数周期が競合する系列、系列間相関が弱いデータでは設計上の利点が縮む可能性がある。`phase=None` で動くことと周期条件付けが機能することは別。訓練窓と推論窓の位相計算規則を揃える。

## 下流encoderで解釈すべき点

`forward_features:[B,T,C]→[B,C,D]` は、**各系列についてT点の履歴全体をD次元へ要約したtoken**。この段階で時刻軸の長さTは消える。本blockの同長出力 `[B,T,Cout]` は`Linear(D,T)`で作り直した特徴であって、各出力時刻が対応する入力時刻だけに依存する特徴ではない。従って時刻ごとの厳密なcausal予測に流用しない。窓全体を観測済みとして系列単位の回帰・分類に使う際は、この全窓集約がむしろ目的に合う。

## 使い方

```python
from ts_block import TQNetBlock

block = TQNetBlock(in_dim=12, out_dim=32, seq_len=64,
                   cycle_len=24, heads=4)
h = block(x, phase=phase)  # x:[B,64,12], phase:[B] -> [B,64,32]
tokens = block.forward_features(x, phase)  # [B,12,D]
```

`python ts_block.py` はforward/backward、`python verify_official.py --official-repo <TQNet-repo>` は公式encoder一致を確認する。学習例：`python finetuning_example.py --csv data.csv --features f1 f2 --targets y --phase-col phase --seq-len 64`。`phase` はデータ側で `timestamp mod W` 等から用意する。未設定でCLIを動かすと全窓が位相0になり、原著の想定した周期条件付けにならない。

## サンプルごとのattentionを手で追う

`B=2,C=3,T=8,W=4` とし、窓先頭位相が `[1,3]` ならquery表を読む行は次のようになる。

```text
b=0: [1,2,3,0,1,2,3,0] → θから [T,C]=[8,3] を取得
b=1: [3,0,1,2,3,0,1,2] → θから [T,C]=[8,3] を取得
transpose: query候補 [B,C,T]=[2,3,8]
正規化した入力から key/value候補 [B,C,T]=[2,3,8]
MHA: attention score [B,heads,C,C]
```

ここでqueryの系列 `c_q` は**「未来の時刻」ではなく、出力を更新したい系列**を指す。keyの系列 `c_k` は参照する観測系列。softmaxは`c_k`軸で掛けるので、各query系列がどの入力系列を参照するかが決まる。位相はquery側の8成分を変えるが、key/valueの観測値を周期テンプレートで置換しない。したがって「固定queryだから予測が全サンプル同じ」という理解は誤り。

## 原著と汎用blockの分岐図

```text
       phase[B] ─→ 周期表 θ[W,C] ─→ Q[B,C,T]
                                         │
 X[B,T,C] ─→ 窓正規化 ─→ [B,C,T] ─→ K,V ─→ TQ-MHA
                                         │
                               residual + shared T→D + MLP
                                         ↓
                                 encoder [B,C,D]
                                  ├─ 原著: D→H、denorm、未来MSE
                                  └─ 本版: D→T、Cin→Cout、下流head
```

`[B,C,D]`は原著の予測器にも汎用blockにも共通する境界。ここで既にC変数間の情報交換を済ませている。最後の`Cin→Cout`を取り除いてもTQ-MHAがあるので、原著はchannel-independentモデルではない。逆に最後のchannel projectionの存在だけを根拠に原著の系列間attentionを説明してはいけない。

## 実装と論文の対応表

| 意味 | 公式実装の処理 | 本surveyでの扱い |
| --- | --- | --- |
| 周期query | `temporalQuery` から位相でgather | `forward_features(x,phase)`内でgather |
| 変数間相互作用 | `MultiheadAttention`、token数C | 同じMHAを利用 |
| 時間方向の要約 | 各変数へ`Linear(T,D)`とMLP | encoderとして保持 |
| 未来長Hへの変換 | `Linear(D,H)` | 説明のみ。汎用blockには入れない |
| 同長特徴 | 原著にはない | `Linear(D,T)` + channel projection |

重要なのは`embed_dim=T`がPyTorch MHAの仕様という点。一般のTransformerの感覚で`embed_dim=D`に変更すると、attentionの重みshapeも計算も公式と一致しない。位相`p`は各窓**開始時刻**のもの。窓末尾の位相を与えるとquery表はT点ぶん回転してしまう。

## 汎用利用の判断

周期が明瞭で、入力列の間に状態依存の相関があるならTQNetの周期queryを試す理由がある。周期が不明確なら、`W`を適当に置くだけではこの仮定を満たさない。Cが大きいと`C×C`のattentionが支配的になり、Tだけを短くしてもメモリ問題を解決できない。系列単位の下流タスクには窓全体を読んだ表現として使えるが、窓内各時点のオンライン推論には使えない。位相列の欠損やtimestampのタイムゾーン・日付境界がずれると、学習時のqueryと推論時のqueryが一致しない。

## 位相列を作るときの時刻の意味

周期が24時刻なら、各窓先頭の時刻を0～23へ写す。ただし周期W=24は「24行が同じ長さの周期」を意味する。時間間隔が不規則なDataFrameで単純に行番号mod24を使うと、物理時刻と周期表の対応が崩れる。timestampから位相を計算し、欠測行があるなら補完・再サンプリングの規則を明示する。週周期や営業日周期では休日やサマータイムの処理も同じ規則に揃える。

```text
例: 1時間ごとの時系列、W=24
  窓先頭が月曜03:00 → phase=3
  64点窓のquery index → 3,4,...,23,0,1,... を64点分
  教師は64点の末尾の次行。ただしphaseは窓先頭の3を使う。
```

位相を誤って**予測対象時刻**の値にするとqueryはT点ずれた行から始まり、`T mod W=0`でない限り同じ処理ではなくなる。周期をW行の表として持つこと自体は学習可能だが、「その表の何行目から読むか」はデータ処理側の責任。

## DataFrameからの独立した学習例

```text
series_id と timestamp で系列別・時刻順に整列
→ 履歴 features[r-T:r]                [T,Cin]
→ 窓先頭 phase[r-T]                    scalar integer
→ TQNetBlock(x,phase)                  [T,Cout]
→ 例として時間平均 + task Linear       [K]
→ 同じseriesのtargets[r]              [K] / class index
→ MSE / CrossEntropy
```

splitは各seriesのlabel indexを時間順80/20に分ける。標準化統計は訓練側のみから計算し、検証行の情報を先取りしない。原著のH点予測へ使いたい場合は、このサンプルの1行GT・headではなく、未来H点のターゲット窓と対応するforecast headを用意する。汎用blockが原著タスクを暗黙実装しているわけではない。

## パラメータと計算の所在

周期表`θ[W,C]`はCに線形、TQ-MHAのQKV射影は埋込幅Tに依存する。score行列はCの二乗、後段の共有`Linear(T,D)`は各系列に適用される。従ってCを1000に増やした場合、周期表よりattention scoreとそのbackwardが先に重くなりやすい。Cを減らす前処理は有効な設計変更になり得るが、その後のqueryは元系列ではなく**混合された潜在系列**の位相テンプレートを学ぶことになる。

## 3通りのquery/keyを対比

| queryの由来 | keyの由来 | 何に依存して系列間attentionを決めるか |
| --- | --- | --- |
| 現在窓X | 現在窓X | そのサンプルの観測値のみ |
| 周期表θ | 現在窓X | 共通の周期的query＋その窓の観測値（TQNet） |
| 学習表 | 学習表 | その窓固有の観測値をkeyに使わない |

TQNetの中間形は「季節パターンに沿って、今どの変数を参照すべきか」をデータから学ぶ意図に合う。ただし、季節性がない入力や周期の変動が大きい入力では、この追加バイアスは不利にもなり得る。論文のablationはそのモデル・データ・予測headでの比較であり、ここで提供する一般的な下流encoderの最良性を保証しない。

## 隠れtokenが何を保持するか

`TQ-MHA`直後は`[B,C,T]`。これは各系列が他系列の履歴の重み付き合成を得た状態で、時間幅Tが残っている。その後の`Linear(T,D)`で各系列の履歴全体をD次元へ圧縮する。従って`[B,C,D]`出力のC軸は**入力変数の軸**であり、D軸が将来H時刻でもない。`Linear(D,T)`の同長readoutはタスク用の特徴を再展開するだけで、公式の未来H点を予測しているわけではない。

`C=1`ならattention scoreは`[B,heads,1,1]`でsoftmaxは常に1。系列間を選択するTQの利点は消え、残るのは周期表からのquery射影と時間MLP等の処理。Cが少ない単変量タスクを主目的とするなら、より単純なbaselineと比較するのが妥当。

## 1サンプルの予測で何が変わるか

同じ履歴値Xでも、位相pが変われば参照する周期表の行がずれるためqueryが変わり、系列間attentionの重みが変わり得る。逆に位相pが同じでも、Xが変わればkey/valueが変わるため出力は変わり得る。つまりモデルは**周期位相だけの固定予測器でも、周期を無視した通常self-attentionでもない**。

```text
sample A: p=2, X=A → Q=θ[2:...], K/V=f(A) → attention(A,2)
sample B: p=2, X=B → Q=θ[2:...], K/V=f(B) → attention(B,2)
sample C: p=5, X=A → Q=θ[5:...], K/V=f(A) → attention(A,5)
```

ここで`f`はMHA内部のkey/value射影を含む略記。AとBの差が「現在の観測」、2と5の差が「周期条件」である。この二軸を分けて考えるとTemporal Queryの役割が分かる。

## 採用前に比較するbaseline

同じ入力窓・同じsplitで、周期queryを使わない系列間attentionと、各系列を独立に処理する時間MLPを比較する。前者との差は周期queryの寄与、後者との差は系列間fusingの寄与を見る助けになる。周期Wの候補は検証データだけで決め、テスト期間の周期性を見てから選ぶと評価が楽観的になる。これは原著ablationを手元の下流データへ移すための実験設計であり、READMEの既報MSEをそのまま再現する手順ではない。

原著の主張を適用できるのは、周期Wと各窓の位相がデータの観測周期に対応している場合。周期表を学習可能にしただけでは、その対応をモデルが自動推定するわけではない。

## 一次資料

原論文：[Temporal Query Network for Efficient Multivariate Time Series Forecasting](https://arxiv.org/abs/2505.12917)（ICML 2025）。公式実装：[ACAT-SCUT/TQNet](https://github.com/ACAT-SCUT/TQNet) の `models/TQNet.py`。照合箇所：論文 Methodology / Temporal Query / TQ-MHA / Appendix、公開forwardと学習driver。
