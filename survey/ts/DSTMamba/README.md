# DST-Mamba survey

対象: [*Decomposed Spatio-Temporal Mamba for Long-Term Traffic Prediction*](https://ojs.aaai.org/index.php/AAAI/article/view/33281), AAAI 2025。指定 PDF はテキストとして確認し、[公式コード](https://github.com/Anle-He/DST-Mamba) の `DSTMamba_Arch.py`、`MambaEnc.py`、`SeriesMix.py` と照合した。名前の ST は単なる「時間系列向け Mamba」ではなく、**交通センサの node 間空間依存と時間依存を分解する**意味である。

## このディレクトリのブロックの使い方

主用途は **`DSTMambaBlock` を系列 encoder として他のモデルへ組み込むこと**。`x:[B,T,Cin]→h:[B,T,Cout]` を返し、GT・loss・pooling・下流 head は持たない。`h.mean(dim=1):[B,Cout]` を回帰/分類 head に渡すか、時刻別 head を `h` に直接適用できる。ただし **T=`seq_len` と Cin=`in_dim` は構築時に固定**する。各入力 channel を node とみなして node 軸で双方向 scan するため、特徴順も固定する。異なる長さ・特徴集合を一つの重みで処理する encoder ではない。

以下の `H` 点の将来予測・MSE は **原著の交通予測タスク**。[finetuning_example.py](finetuning_example.py) も予測用 head を後付けする一例であり、この同長ブロックの必須出力・loss ではない。

## 概要

DST-Mamba は、**交通センサごとの過去 L 点から、全センサの将来 H 点を予測する**モデルである。入力の主信号 `X:[B,L,N]` は `N` 個の固定センサの時系列で、出力は `Ŷ:[B,H,N]`。変動成分をセンサ間の双方向 Mamba で、緩やかな trend を時間多尺度の線形経路で予測して足し合わせる。`B` は batch、`L` は履歴長、`N` はセンサ数、`H` は予測長。学習の GT は未来の実測 `Y:[B,H,N]` で、最終予測に対する MSE を使う。

このディレクトリの [ts_block.py](ts_block.py) は両経路を **`H=L` に同長化**し、必要ならセンサ軸 `N` を `Cout` へ写す。以下の図はまず原著/公開 forward の予測経路を描き、同長 adapter とは区別する。

## 新規性・既存手法との差分

| 比較対象 | DST-Mamba が変える点 | 意味・制約 |
| --- | --- | --- |
| 通常の時間軸 Mamba | 1時刻を token にせず、**1センサの L 点を1 token** にして N 軸を走査 | Mamba が直接扱うのはセンサ間依存。任意の特徴列を並べただけでは交通センサの前提にならない |
| 単一経路の時空間モデル | 移動平均で変動と trend を分け、前者を node Mamba、後者を多尺度線形 mixing に割り当てる | 高速変動と緩やかな時間変化に別の帰納バイアスを与える |
| 片方向の node scan | node 列を順・逆の二方向から走査 | 両側の node 文脈を使えるが、node 順序への依存は消えない |
| 全 node 共通の埋込 | node 固有の低ランク adapter を加える | 固定センサ集合を表せる一方、新しい node 数・順序にそのまま移植できない |

論文には隣接行列による graph aggregation の式があるが、公開 `forward` は明示的な隣接行列を受け取らない。図は **公開コードで実際に流れる経路**を示し、存在しない GNN 層は描かない。

## モデルアーキテクチャ全体像（公開 forward）

```text
交通センサ履歴 X [B,L,N] ──────────────── 将来実測 Y [B,H,N] = GT
          │                                      │
          ▼                                      │
RevIN: node 別平均/分散 [B,1,N]                  │
          │ 正規化 X_norm [B,L,N]                 │
          ├─────────────────────┐                │
          ▼                     ▼                │
  変動 / seasonal 経路       trend 経路           │
  移動平均を引く             各尺度で時間 downsample
  [B,L,N]                   [B,L_i,N]
      │                          │ 各尺度を個別に移動平均分解
      ▼                          ▼
  node ごと L→D_I 埋込      trend_i [B,N,L_i]
  [B,N,D_I]                    │ coarse→fine MLP mixing
      │ node 固有低ランク適応    │ 各尺度 Linear(L_i,H)→[B,H,N]
      ▼                          │ 尺度間で加算
  [B,N,D_I+D_A=D]               ▼
      │ forward Mamba(N軸)   trend予測 [B,H,N]
      │ + reverse Mamba(N軸)        │
      │ + FFN / LayerNorm          │ node別の学習重みを掛ける
      ▼                            │
  [B,N,D]                       │
      │ Linear(D,H)               │
      ▼                            │
  seasonal予測 [B,H,N] ──── + ─────┘
                        │ RevIN 逆変換
                        ▼
                   予測 Ŷ [B,H,N] ── MSE ── GT
```

図の `L_i` は downsample 後の各時間長、`D_I` は時系列埋込幅、`D_A` は node 適応幅。**Mamba が走査する長さは L ではなく N** で、時間情報は node token と trend 経路に入る。公開 runner の feature 次元 `[B,L,N,F]` や最終 singleton 軸 `[B,H,N,1]` は、図では主信号の処理を見やすくするため省いた。今回の同長ブロックでは図中の `H` を `L` に置き換える。

## 原著の入力・二分岐・出力

公式 runner の観測は概ね `[B,L,N,F]` で、モデル本体は最後の feature から `x:[B,L,N]` を取り出す。N は固定数の道路センサ/ノード、L は履歴長。出力は `[B,H,N,1]`、H は予測長。つまり多特徴入力を一般に全て消費する設計ではない。

| 経路 | 入力 → 出力 | 内容 |
| --- | --- | --- |
| RevIN | `[B,L,N] → [B,L,N]` | node ごとの履歴平均/分散で正規化し、最後に逆変換 |
| 移動平均分解 | `x → seasonal,trend:各[B,L,N]` | `seasonal=x-trend` |
| seasonal token | `[B,L,N] → [B,N,E]` | 各 node の **L 点を一つの token** へ線形埋込 |
| node adapter | `[B,N,E] → [B,N,D]` | node ごとに学習する低ランク適応を連結 |
| 双方向 Mamba | `[B,N,D] → [B,N,D]` | scan 軸は時間 L ではなく **node N** |
| seasonal 予測 | `[B,N,D] → [B,H,N]` | node token から H 点の予測 |
| trend 多尺度 | `[B,L,N] → [B,H,N]` | downsample、低→高解像度 mixing、各尺度線形投影・加算 |
| 統合 | `[B,H,N] → [B,H,N,1]` | seasonal + node ごとの学習重み×trend、RevIN 逆変換 |

双方向とは `Mamba(tokens)+flip(Mamba_reverse(flip(tokens)))`。node 順に任意性があるので逆向きを加えるが、**順序不変にはならない**。論文はグラフ由来情報も論じる一方、公開された上記 `forward` に隣接行列を渡す明示的 GNN はなく、node-specific adapter が主な空間表現。ここをグラフ畳み込みと誤記しない。

## 原著の学習 objective（交通予測）

履歴 `X:[B,L,N,1]` から将来 `Ŷ:[B,H,N,1]` を予測し、同じ時刻/node の実測 `Y` と比較する教師あり forecasting。論文 Eq. (7)–(8) は seasonal 予測と重み付き trend 予測を足した最終値に対する **MSE** `mean((Ŷ-Y)^2)` を学習目的とする。MAE は評価指標であり、seasonal/trend の GT が別々に与えられるのではない。本サンプルも最終値の MSE を採用する。

## 汎用ブロックで何を変えたか

[ts_block.py](ts_block.py) の `DSTMambaBlock(in_dim=N,out_dim=Cout,seq_len=L)` は `[B,L,N]→[B,L,Cout]`。RevIN、移動平均分解、node ごとの **固定長** temporal token、低ランク adapter、公式順序の双方向 node Mamba/Post-LN encoder、多尺度 trend の coarse-to-fine MLP、尺度別 `Linear(L_i,L)`、node 別 trend 重み、RevIN 逆変換を実装した。**原著の未来 H 点を出す head は H=L の同長化**であり、最後の `Linear(N,Cout)` は `N=Cout` でも適用する実用化のための追加。原著 checkpoint と互換ではない。各入力 channel は 1 node と解釈するため `in_dim` と node 順は固定。L も learned Linear の形状に入るため固定。

```python
from ts_block import DSTMambaBlock
block = DSTMambaBlock(in_dim=12, out_dim=32, seq_len=96, backend="auto")
h = block(sensor_history)  # [B,96,12] -> [B,96,32]
```

[finetuning_example.py](finetuning_example.py) は DataFrame の過去値のみから future labels を予測する実学習例。node とみなす特徴列の順序は全系列で固定する。

## 性能・限界

原著の利点は季節変動を node 間 Mamba、安定した trend を軽い多尺度線形経路で分担する点。双方向 scan は N 軸なので計算は node 数にも比例する。単一金融銘柄の無関係な数値特徴を node とみなす場合、交通網の前提は成立しない可能性がある。PyTorch fallback scan は長い N では遅く、公式 `mamba_ssm` kernel 使用時と速度を同一視しない。

## 論文の問題設定を具体的に見る

交通予測では `N` 個の固定センサについて、同時刻の速度/流量などが相互に影響する。入力 `X∈R^{B×L×N}` の L は過去時間、N は地理的に区別できる node。一般的な時間方向 Mamba は L を scan し、各 token に N 個の node 値を詰める。DST-Mamba はまず各 node の L 点を一つの token に埋め、**Mamba の系列長を N に置き換える**。これにより Mamba が直接モデル化するのは「時点間」ではなく「node 間」の依存になる。時間情報は token 埋込・FFN と trend 分岐に保持される。

論文が解こうとする二つの違う難しさは、交通系列の局所的変動に効く空間相関と、緩やかな長期 trend の双方を、巨大な時空間 attention でまとめて扱うコストである。seasonal を非線形 Mamba、trend を軽い線形多尺度系に分担するのが新規性の中心。「Mamba を時系列へそのまま挿した」ものではない。

### 1サンプルの forward を追う

たとえば `B=2,L=96,N=170,H=12,D=64` とする。

```text
入力 X                         [2,96,170]
moving average trend X_TR       [2,96,170]
seasonal X_SE = X-X_TR          [2,96,170]

seasonal:
  各 node の96点を埋込          [2,170,D_I]
  node-specific適応を連結       [2,170,D_I+D_A=64]
  forward Mamba(N軸)            [2,170,64]
  reverse Mamba(N軸)            [2,170,64]
  反転し直して和 + FFN          [2,170,64]
  将来12点へ投影                [2,12,170]

trend:
  多段の時間downsample          各[B,L_i,N]
  coarse→fineのtop-down mixing 各[B,L_i,N]
  各尺度の時間線形投影          各[B,12,N]
  尺度和                        [2,12,170]

予測 = seasonal + λ_node*trend [2,12,170]
RevIN逆変換、末尾feature軸     [2,12,170,1]
```

時間 L と node N を入れ替えているため、原著の Mamba 計算量を「履歴長 L に線形」とだけ説明するのは不正確。主たる selective scan は N に依存し、L は各 node token の線形埋込や trend の時間線形層に入る。

### trend の式と目的

論文 Eq. (3) は `X_TR=AvgPool(Pad(X))`、`X_SE=X-X_TR`。ここで seasonal はカレンダー上の厳密な季節成分ではなく、移動平均を引いた変動成分である。**公開 forward では正規化済みの元信号 `X` を各尺度へ downsample し、その各尺度を別々に移動平均分解して trend `X_TR^(i):[B,L_i,N]` を得る**。最初に求めた高解像度 trend を繰り返し downsample する処理ではない。粗い尺度から細かい尺度へ `X'_TR^(i)=X_TR^(i)+ScaleMix(X_TR^(i+1))` と加算する。次に各尺度を `L_i→H` の線形層で将来値へ写し、尺度間を足す。粗い trend が長期変化、細かい trend が局所的変動を支える構図。

公式 `SeriesMix.py` の `up_sampling` は MLP で隣接尺度間の長さを揃える。公式 `DSTMamba_Arch.py` の `linear_mappings[i]` は各尺度 `L_i→H` の固定長射影。したがって `L,H,ds_window,ds_layers` を後から自由に変えて同じ checkpoint を使う構成ではない。

### spatial token と graph の論文/コード差

論文 Eq. (5) は `E_I=S X_SE` と書き、S を事前定義 graph または DTW 距離から得た隣接行列とする。これが論文の **graph aggregation** の記述。一方、指定された公開 `DSTMamba_Arch.py` の `forward` では隣接行列 S を受け取らず、`DataEmbedding` が `[B,L,N]→[B,N,D_I]` の時間線形埋込を行う。続く node-specific adapter は `adapter:[N,D_I,r]` と `lora:[r,D_A]` を使い、各 node の埋込から `D_A` 次元を作る。**紙面の Eq. (5) を公開 forward がそのまま実装している、と断言できない。** 本汎用ブロックも公開 forward のように明示的隣接行列を要求しない。

適応部分は公開コードの順序で `M_n=ReLU(A_n @ W)`、`a_{b,n}=e_{b,n} @ M_n`。形状は `e:[B,N,D_I]`、`A_n:[D_I,r]`、`W:[r,D_A]`、`M:[N,D_I,D_A]`、`a:[B,N,D_A]`。ReLU は `e` を掛ける**前**にある。node ID に依存する A_n は固定センサ配置を学習できるが、未見 node の追加や順序変更にはそのまま適用できない。学習済み N=170 のモデルへ N=171 の特徴を渡すことはできない。

### 双方向 scan で何が改善し、何は改善しないか

論文 Eq. (6) は `H_FW=Mamba(E)`、`H_BW=Mamba_rev(flip_N(E))`、`H_SE=H_FW+flip_N(H_BW)`。通常の causal scan だけなら、先頭側の node は後方の node を参照できない。reverse 側を足すと全 node が両側文脈を持てる。これは **空間 node 軸の双方向化**なので、将来時間の観測を使う操作ではない。ただし node の任意 permutation に対して出力が完全に同じになるわけではなく、前後二方向という順序バイアスは残る。

### GT と MSE をさらに明確に

1 sample は履歴 `X[t-L+1:t,N]`、将来 `Y[t+1:t+H,N]` を持つ。モデルは seasonal/trend それぞれの予測を生成するが、教師は **将来の実測 Y 一種類だけ**。各 branch を別 GT に合わせるのではなく、`Ŷ=Ŷ_SE+λŶ_TR` の最終値を `Y` と同じ shape で比較する。論文 Eq. (8) は L2/MSE、公式実験設定は Adam、初期 LR `1e-3`、batch 32。MAE は評価指標。loss 計算前には padding/欠測値があれば mask の扱いが必要だが、紙面の式は dense Y を前提にする。

## 実験・ablation の読み方

論文の5データセットは Traffic、PEMS03/04/07/08。センサ数はそれぞれ 862、358、307、883、170 と大きく異なり、Mamba scan 軸 N の長さもかなり変わる。入力長は主要比較で 96。ablation は adaptive embedding、decomposition、bidirectional、trend branch、multi-scale mixing などを取り除く。論文は特に seasonal 側を外した低下が大きいと述べ、交通変動を跨 node で扱う仮説を支持する。一方、これは交通データ特有の評価であり、株価テクニカル特徴を node とみなす用途へそのまま一般化できない。

## コピペ用同長ブロックの対応関係

本 [ts_block.py](ts_block.py) は原著の learned 時間行列を勝手に adaptive pooling/interpolation へ置換しない。構築時に `seq_len=L` を決め、`Linear(L,D_I)`、各尺度の `Linear(L_i,L)`、coarse-to-fine MLP を実行する。原著との主な変更は **future horizon H を L にして同長出力を得る**こと、および必要なら N→Cout を写すことである。異なる L を同じ重みで処理できない制約は、公式処理を保つために受け入れる。

系列全体の回帰・分類なら `h:[B,L,Cout]` を時間 pooling してから任意の head を付ける。時刻別のタスクなら pooling せずに使う。**GT と loss はブロックに含まれない**。[finetuning_example.py](finetuning_example.py) は下流例として最終時間特徴から将来値を回帰するが、論文の二分岐 `[B,H,N]` head をそのまま訓練するものではない。入力の各列が本当に固定 node なら有用だが、列が「価格・RSI・出来高」のように異質な測定量なら、node 軸の inductive bias は必ずしも自然ではない。

## 公式コード対応・確認すべき実装上の罠

| 処理 | 公式ファイル | 本汎用モジュール |
| --- | --- | --- |
| moving-average 分解 | `SeriesDec.py` | `forward` 内の平均 pool |
| 時間→node 埋込 | `Embed.py` | 固定 `seq_len` の `embedding` |
| node-specific 低 rank 適応 | `DSTMamba_Arch.py::adapter/lora` | `adapter/low_rank` |
| 双方向 node Mamba/FFN | `MambaEnc.py` | `_BiNodeLayer`。公式同様の post-norm、scan 入力 residual なし |
| trend 多尺度混合 | `SeriesMix.py` | `up_sampling` の二層 MLP と尺度別 `linear_mappings` |
| 予測 head / RevIN | `DSTMamba_Arch.py` | H=L の `projector`、RevIN、任意 N→Cout 写像 |

公開コードは `decom_type=='STD'` の場合に `self.decom` を作るが、その後の `forward` は無条件に `self.decom(x)` を呼ぶ。したがって別の decom_type を渡すと成立しない。また `ds_type=='conv'` の branch は `self.ds_method` を参照しており、設定値と属性名がずれている。ここでの汎用版は移動平均分解と avg/max downsampling に絞った。原著 checkpoint の再現用として使う場合は、これらの実装差を必ず確認する。

現在の汎用版は `ds_type='avg'|'max'` を選べる。公式 `conv` 分岐は上記の参照属性不一致があり含めない。[verify_official.py](verify_official.py) は、公式モデルの `mamba_ssm.Mamba` だけを両側で同じ PyTorch mixer に差し替え、`L=H=16`、`ds_type='avg'`、同じ重み、末尾出力射影を恒等とした**入力から最終出力までの forward 一致**を確認する。これは fused Mamba kernel・公式 checkpoint・未選択設定の一致を意味しない。

## 一次資料

- [論文: Decomposed Spatio-Temporal Mamba for Long-Term Traffic Prediction](https://ojs.aaai.org/index.php/AAAI/article/view/33281)（提供 PDF はテキストとして参照）
- [公式 DST-Mamba 実装](https://github.com/Anle-He/DST-Mamba)

## 動作確認と DataFrame 学習の入口

このディレクトリで `python ts_block.py` を実行すると、実際の block に乱数 tensor を渡して出力 shape と backward を検証する。`python ../validate_blocks.py` は5モデル横断で、異なる T と数値特性を追加検証する。追加ライブラリを入れない PyTorch 実行では Mamba 系の fallback を使い、速度比較ではない。

CSV を用意する場合は `series_id,timestamp` と特徴列・target 列が必要。例えば次のコマンドは過去64点から次8点の実測 `target` を予測する。

```bash
python finetuning_example.py data.csv --features feature_a feature_b --targets target --lookback 64 --horizon 8 --epochs 10
```

既に DataFrame `df` があるなら `from finetuning_example import fit; model, info = fit(df, ["feature_a", "feature_b"], ["target"])`。返す `info` には train 行だけから求めた feature mean/std と最良の validation MSE が入る。推論時は同じ列順・同じ正規化統計で過去窓を作る。本文で述べた原著の benchmark 成績を再現するには、このサンプル以外に原著の入力長、head、split、正規化と optimizer 設定を揃える必要がある。

## 学習・推論の擬似コード（原著）

次は論文/公開実装の固定長 forecast 経路。汎用 `DSTMambaBlock` の同長 API とは区別する。

```text
history                           [B,L,N]
GT future                         [B,H,N]
mu,sigma = stats(history,time=L)  各[B,1,N]
x = (history-mu)/sigma             [B,L,N]
trend = MovingAverage(x)          [B,L,N]
season = x-trend                  [B,L,N]

node_token = TimeEmbed(season)     [B,N,D_I]
node_specific = Adapter(node_token) [B,N,D_A]
e = concat(node_token,node_specific) [B,N,D]
spatial = ForwardMamba(e)
        + flip(BackwardMamba(flip(e,N)),N) [B,N,D]
season_pred = Linear(D,H)(spatial).transpose  [B,H,N]

signal_scales = Downsample(x)      m本の[B,L_i,N]
trend_scales = MovingAverage(signal_scales) 各[B,L_i,N]
mixed_scales = TopDownMix(trend_scales)
trend_pred = sum_i Linear(L_i,H)(mixed_scales[i])
                                  [B,H,N]
pred_norm = season_pred + lambda_node*trend_pred
pred = pred_norm*sigma+mu          [B,H,N]
loss = mean((pred-GT future)^2)    scalar
```

`season_pred` と `trend_pred` に個別の ground truth はない。推論では同じ順で branch を足し、逆正規化して H 点を一括出力する。H は projector の重み形状に入るため、公開 checkpoint のまま任意 horizon へ変えられない。
