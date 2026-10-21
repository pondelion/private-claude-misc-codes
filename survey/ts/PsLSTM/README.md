# P-sLSTM survey

対象: *Unlocking the Power of LSTM for Long Term Time Series Forecasting*。公式実装は [P-sLSTM](https://github.com/Eleanorkong/P-sLSTM) の `models/P_sLSTM.py` と xLSTM/sLSTM 部分。ここでは **論文の固定予測長モデル** と、このディレクトリの **任意長・同長ブロック**を区別する。

## このディレクトリのブロックの使い方

主用途は **`PsLSTMBlock` を系列 encoder として他のモデルへ組み込むこと**。`x:[B,T,Cin]→h:[B,T,Cout]` を返すだけで、GT・loss・pooling・下流 head は持たない。例えば `h.mean(dim=1):[B,Cout]` を回帰/分類 head に渡すか、pooling せず各時刻の特徴を使う。T はこの同長ブロックでは入力ごとに変えられる。パッチ内部は非因果的なので、オンライン推定で時刻 t の特徴を使うなら、t より後の観測を含む窓を渡さない。

以下の論文説明に出てくる `H` 点の将来予測・MSE は **原著のタスク**であり、このブロックの必須処理ではない。[finetuning_example.py](finetuning_example.py) の将来予測も下流タスクの一例にすぎない。

## 概要

P-sLSTM は、**過去の多変量時系列を変数ごとに時間パッチへ分割し、そのパッチ列を sLSTM で読んで将来 H 点を一括予測する**長期予測モデルである。入力は過去 `X:[B,L,M]`、出力は将来 `Ŷ:[B,H,M]`。各変数には同じ encoder/head の重みを使うが、encoder 内では変数同士を混ぜない。学習では将来の実測 `Y:[B,H,M]` との MSE を最小化する。ここで `B` は batch、`L` は履歴長、`M` は変数数、`H` は予測長であり、パッチ数は `N`、パッチ幅は `P`、埋込幅は `E` とする。

このディレクトリの [ts_block.py](ts_block.py) は再利用のため **`[B,T,Cin]→[B,T,Cout]` の同長特徴ブロック**に変えたもの。以下の図はまず原著の予測モデルを示し、差分は図の後および「コピペ用ブロック」で説明する。

## 新規性・既存手法との差分

| 比較対象 | 時間の処理 | P-sLSTM の差分と意図 |
| --- | --- | --- |
| 通常の LSTM | 観測点を L 回順に処理 | P 点を 1 token に圧縮し、sLSTM の更新を N 回に減らす |
| patch なし sLSTM | 安定化 gate/state を使うが L 回更新 | 同じ sLSTM 系の記憶機構を **パッチ列**へ適用し、長い系列の逐次更新数を削減 |
| 一般的な多変量 encoder | 変数を同じ token 内で混合し得る | 変数を `B×M` 本の独立系列として処理し、backbone 重みを共有する |

したがって提案の中心は「sLSTM という新しい cell を発明したこと」ではなく、**既存 sLSTM と時間パッチ化・channel independence を長期予測用に組み合わせたこと**にある。パッチ化で逐次 step 数は減るが、パッチ内情報は線形埋込へ圧縮され、変数間の直接相互作用も失う。精度・速度の利点はこの交換条件の下で評価する。

## モデルアーキテクチャ全体像（原著の予測モデル）

```text
過去の実測 X [B,L,M]                         将来の実測 Y [B,H,M] = GT
       │                                             │
       ▼                                             │
変数を batch に移す [B·M,L]                         │
       │ 各変数を独立系列とし、encoder/head の重みは共有 │
       ▼                                             │
時間パッチ化 [B·M,N,P]  ── Linear(P,E) ──> [B·M,N,E]
       N = floor((L-P)/S)+1、S は stride             │
                                                     ▼
                           ┌ sLSTM block × depth ──────────────┐
                           │ causal Conv → head-wise gates      │
                           │ (h,c,n,m) をパッチ順に N 回更新     │
                           │ GroupNorm → gated FFN + residual  │
                           └──────────────┬─────────────────────┘
                                          [B·M,N,E]
                                               │ 全パッチの state を連結
                                               ▼
                                          [B·M,N·E]
                                               │ Linear(N·E,H)
                                               ▼
                                          [B·M,H]
                                               │ 変数軸 M を戻す
                                               ▼
                                        予測 Ŷ [B,H,M] ── MSE ── GT
```

図の `n,m` は sLSTM の正規化 state であり、変数数 `M` とは別物。原著 head は **全 N パッチの出力を使う固定長 `N·E→H`** で、最終 state だけを読む RNN ではない。今回の同長ブロックはこの head を使わず、`[B·M,N,E]→パッチ値→overlap-add→[B,T,Cin]→[B,T,Cout]` に置き換える。したがって原著と本ブロックの出力の意味を混同しない。

## 何を解決するか

通常の LSTM は長い依存を学べると期待される一方、実際には forget gate による短記憶化や長系列の不安定性がある。sLSTM は指数型 input gate、選択可能な forget gate、normalizer state、複数 head の recurrent mixing を導入する。参照した公開設定の forget gate は sigmoid である。しかし論文は「sLSTM 単独でも長期記憶は保証されない」と議論する。P-sLSTM の主張は、**時間を patch にまとめて recurrent step 数を減らし、各変数を同一 backbone で独立処理**すると長期予測に有利、というもの。チャネル間相互作用は主な設計に含まれない。

## 論文・公式コードの経路と形状

| 処理 | 入力 → 出力 | 意味 |
| --- | --- | --- |
| 入力 | `[B,L,M]` | B: batch, L: 観測長, M: 変数 |
| channel independence | `[B,L,M] → [B·M,L]` | 変数を batch に移し、backbone の重みは共有 |
| patching | `[B·M,L] → [B·M,N,P]` | P: patch size、S: stride、`N=floor((L-P)/S)+1` |
| embedding | `[B·M,N,P] → [B·M,N,E]` | 各 patch を E 次元 token 化 |
| sLSTM stack | `[B·M,N,E] → [B·M,N,E]` | patch 順の recurrent 更新 |
| 予測 head | `[B·M,N,E] → [B·M,N·E] → [B·M,H]` | H: horizon。patch 軸と特徴軸を flatten |
| 復元 | `[B·M,H] → [B,H,M]` | M 変数の将来値 |

xLSTM stack の設定では sLSTM block が全層に置かれ、mLSTM は使われない。patch 数は設定時の L/P/S で決まり、公式の flatten head は長さ固定である。

## sLSTM の中で何を更新するか

各 patch token `u_t∈R^E` と前 state `h,c,n,m∈R^E` から gate を作る。論文は forget gate の指数版も論じるが、**公開 xLSTM vanilla 実装の既定 `function="slstm"` は sigmoid forget gate** である。この実装に合わせると、生 logit を `ĩ_t,f̃_t` として `a_t=m_{t-1}+log(sigmoid(f̃_t))`、`m_t=max(a_t,ĩ_t)`、`i_t=exp(ĩ_t-m_t)`、`f_t=exp(a_t-m_t)`。次に `c_t=f_t c_{t-1}+i_t z_t`、`n_t=f_t n_{t-1}+i_t`、`h_t=o_t(c_t/n_t)`。m は指数の overflow を避ける log-domain stabilizer。input/forget gate の入力には causal convolution をかけ、複数 head は再帰重みを block diagonal 化して実現する。この順序を「普通の LSTM に patch を入れただけ」と同一視しない。

## 原著の学習 objective（長期予測）

教師あり long-term forecasting。入力は時刻 `t-L+1…t` の観測 `X∈R^{B×L×M}`、予測は次の `H` 点 `Ŷ∈R^{B×H×M}`、GT は実観測 `Y∈R^{B×H×M}`。実装上の基本損失は要素平均 MSE `mean((Ŷ-Y)^2)`。MAE は評価指標で、論文の「予測精度」と学習時 loss は区別する。patch 化は入力の表現変更であり、別の教師ラベルを作るわけではない。

## コピペ用ブロック

[ts_block.py](ts_block.py) の `PsLSTMBlock(in_dim,out_dim)` は `[B,T,Cin] → [B,T,Cout]`。channel-independent patch embedding → 安定化 sLSTM → patch 値に戻す → overlap-add/coverage 補正 → 最後に channel mixing。この overlap-add と channel mixing は **固定 horizon の原著 head からの汎用化**であり、論文の予測結果をそのまま再現する実装ではない。patch 内の値はまとめて入力されるので、同長出力は各 patch 内で非因果的。オンライン逐次予測では、履歴窓だけを与えて最後の表現から未来を予測する。

```python
from ts_block import PsLSTMBlock
block = PsLSTMBlock(in_dim=8, out_dim=32, patch_size=16, stride=8)
features = block(history)  # [B,L,8] -> [B,L,32]
```

[finetuning_example.py](finetuning_example.py) は DataFrame の `series_id,timestamp` と任意の特徴・target 列から、過去窓/将来 GT を生成し、実際に MSE 学習する。ここではブロック末尾の `[B,32]` を `[B,H,Ctarget]` に写す。正規化統計は各系列の前半 80% のみから計算し、validation の label は後半のみ。

## 実用上の判断

P を大きくすると recurrent step は減るが短周期の情報が patch 内に押し込まれる。論文も patch size に中間最適値があると報告する。CI は変数間の因果・相関を backbone で扱えないため、強いクロス変数依存があるデータでは最後の mixing だけで足りない可能性がある。PyTorch sLSTM の recurrent loop は CUDA fused kernel より遅い。GPU 長系列で速度重視なら xLSTM の最適化 backend を別途検討する。

## 論文をもう少し丁寧に読む

### なぜ sLSTM だけでは十分でないか

論文の論点は「指数 gate を導入すれば必ず長期記憶できる」ではない。forget gate が実効的に 1 未満なら過去状態の影響は繰り返し更新で幾何級数的に薄まり、逆に 1 を超える領域では cell state の急増と数値不安定性が起き得る。いずれも「長い L をそのまま L 回更新すればよい」という見方に反する。patching は L 個の観測を N 個の token に圧縮して **recurrent step 数そのもの**を減らし、各 step 内では P 個の局所観測を一括して見せる。たとえば `L=336,P=S=16` なら 336 step の代わりに 21 patch step となる。ただしこれは 336 点全ての逐次 state 更新を正確に保持するという意味ではなく、時間解像度の一部を patch 埋込へ委ねる設計上の交換である。

論文の短記憶性に関する議論は gate に条件を置いた解析であり、任意の重み・任意のデータに対する一般定理として「sLSTM は長期依存を学べない」と結論していない。また実験で patching が役立ったことと、この解析が全ての性能差の因果を証明したことは別である。

### 通常の LSTM・sLSTM・P-sLSTM の差

| 構成 | 更新単位 | memory | 変数の扱い | 予測 head |
| --- | --- | --- | --- | --- |
| LSTM baseline | 通常は時点 | `h,c`、sigmoid gates | 実装設定次第 | task 固有 |
| sLSTM baseline | 時点 | `h,c,n,m`、安定化 input gate | 実装設定次第 | task 固有 |
| 論文 P-sLSTM | **patch** | sLSTM の `h,c,n,m` | **channel independence** | 全 patch state を flatten して H 点を一括予測 |

P-sLSTM の `P` は patch size であり、parameter 数ではない。原著は「全 patch state を使う」head なので最終 state だけで将来を生成する RNN でもない。flatten head は `N·E` に比例して大きくなり、入力 L を変えると重み形状が変わる。これは実用上かなり重要な制約である。

### 1サンプルの学習を具体化する

例えば 3 変数の系列について、過去 `L=32` 点から次の `H=4` 点を予測する。入力 `X[0:32,0:3]` は `[32,3]`、GT `Y[32:36,0:3]` は `[4,3]`。batch 化後はそれぞれ `[B,32,3]` と `[B,4,3]`。P=8,S=8 のとき各変数 4 個の patch ができる。各変数を別 batch として同じ sLSTM に通すため、encoder 入力は `[3B,4,E]`、head は `[3B,4E]→[3B,4]`、reshape 後の `Ŷ` は `[B,4,3]`。loss は `B·4·3` 個の対応する予測・GT ペアの二乗誤差平均である。patch token 自体を正解 patch と照合する auxiliary loss はない。

原著 `exp/exp_main.py` は Adam と `nn.MSELoss()` を使い、validation loss による early stopping を行う。評価の MSE/MAE は、学習時の MSE と別の集計指標。標準スクリプトでは PEMS の `L=96,H∈{12,24,48,96}`、その他の主な実験では `L=336,H∈{96,192,336,720}`。これらの値は本サンプルのデフォルト `lookback=64,horizon=8` と異なり、ベンチマーク再現設定ではない。

### sLSTM state を手計算的に追う

各 head の各 channel で `c,n,m` を保持する。最初は `c=n=0,m=-∞`。最初の token では `m_1=ĩ_1`、正規化 input weight `i_1=1` なので、`c_1=tanh(z̃_1)`、`n_1=1`、`h_1=sigmoid(õ_1)·tanh(z̃_1)`。2 token 目から前 state と新しい情報の重みを `m` を使って相対的に正規化する。したがって `n` は単なる追加の隠れ特徴ではなく、`c` の読み出しを正規化するための分母である。

公開 xLSTM 実装の `slstm_forward_pointwise` は `logsigmoid(fraw)` を使う。この README の前段で触れた論文の「指数型 forget gate を許す」という一般説明と、**今回参照した公式設定が実際に使う sigmoid forget** を分ける必要がある。[ts_block.py](ts_block.py) は後者に合わせ、head-wise 入力/再帰投影、patch ごとの MultiHead GroupNorm、gated FFN、block-dependent forget bias、stack 末尾 LayerNorm を実装する。[検証スクリプト](verify_official.py) は公式 vanilla cell の FP32 出力・入力勾配に加え、対応する重みを移した**公式 xLSTM stack 全体の出力**との一致を確認する。ただし原著の固定 horizon 予測 head では各入力時刻の特徴を返せないため、本版では同長 overlap-add/readout に変更している。最終出力の一致は主張しない。

### patch size・stride の設計

`N=floor((L-P)/S)+1` は **末尾を切り捨てる**公式 `unfold` の式。L-P が S で割り切れなければ最後の数点は公式 model に入力されない。一方、汎用 [ts_block.py](ts_block.py) は T を任意長で扱い、右端を最後の観測値で pad して最終点まで覆い、同長に戻す。ここは明確な仕様差である。stride<P の overlap は情報を細かく残すが N と recurrent 計算量を増やす。stride=P は最も単純で、実装に overlap-add が不要な fast path を持つ。

### 論文の実験から言えること・言えないこと

論文は Weather、Electricity、Solar、ETTm1、PEMS03 を評価し、40設定中23件で1位、10件で2位と記載する。sLSTM baseline より90%、通常 LSTM より95%の設定で良いという主張もある。一方、PEMS03 では相対的に弱く、論文自身が高ノイズと denoising 不在を可能性として挙げる。patch size は大きいほど必ず良いのではなく、中間に良い範囲がある。これらは **原著の固定長 forecast model** の結果であり、同長汎用ブロック＋独自 head の精度保証ではない。

## このディレクトリで実際に使うとき

`PsLSTMBlock(8,32)` の出力は `[B,T,32]` の時刻別特徴。系列全体の回帰・分類なら時間 pooling で `[B,32]` にし、用途に応じた head を付ける。時刻別の回帰・分類なら `[B,T,32]` のまま head に渡す。**pooling・head・GT・loss はブロックの外側**で定義する。[finetuning_example.py](finetuning_example.py) は任意の下流例として最終時刻の特徴から将来 H 点を回帰するだけで、この方法をブロックの標準 objective としない。

原著の channel independence が欲しいなら、`channel_mix` を恒等に近い設定にする、あるいは混合前の各変数特徴を取り出す設計へ変更する。本汎用版は `Cout≠Cin` を満たすため最終 mixing を追加した。これは sLSTM backbone 内で変数を融合する変更ではなく、出力幅を選ぶための readout である。

## 公式実装との対応と未実装境界

| 論点 | 公式箇所 | 本ディレクトリ |
| --- | --- | --- |
| patch + CI + flatten forecast head | `models/P_sLSTM.py` | patch + CI は `ts_block.py`、head は同長 overlap-add へ変更 |
| sLSTM cell/gates | `models/xlstm/blocks/slstm/{layer.py,cell.py,src/vanilla/slstm.py}` | `SLSTMCell` と `SLSTMLayer` |
| xLSTM block stack | `models/xlstm/xlstm_block_stack.py` | 複数 `SLSTMLayer` を積む。公式の全設定・初期化は含めない |
| forecast MSE / validation | `exp/exp_main.py` | `finetuning_example.py` の AdamW/MSE/時系列 validation |

公式の CUDA/xLSTM 最適化 kernel、原著 checkpoint 読み込み、データセット固有前処理はこの PyTorch + `einops` の copy/paste ブロックに含めない。これらは外部実装・重みに依存し、`(B,T,Cin)→(B,T,Cout)` の独立モジュールという目的の外にあるためである。論文数値の再現には、公式 code と同じ split、scaling、P/S、head、学習設定が必要。

## 一次資料

- [論文: Unlocking the Power of LSTM for Long Term Time Series Forecasting](https://arxiv.org/abs/2408.10006)（本調査では提供 TeX を読解）
- [公式 P-sLSTM 実装](https://github.com/Eleanorkong/P-sLSTM)

## 動作確認と DataFrame 学習の入口

このディレクトリで `python ts_block.py` を実行すると、実際の block に乱数 tensor を渡して出力 shape と backward を検証する。`python ../validate_blocks.py` は5モデル横断で、異なる T と数値特性を追加検証する。追加ライブラリを入れない PyTorch 実行では Mamba 系の fallback を使い、速度比較ではない。

CSV を用意する場合は `series_id,timestamp` と特徴列・target 列が必要。例えば次のコマンドは過去64点から次8点の実測 `target` を予測する。

```bash
python finetuning_example.py data.csv --features feature_a feature_b --targets target --lookback 64 --horizon 8 --epochs 10
```

既に DataFrame `df` があるなら `from finetuning_example import fit; model, info = fit(df, ["feature_a", "feature_b"], ["target"])`。返す `info` には train 行だけから求めた feature mean/std と最良の validation MSE が入る。推論時は同じ列順・同じ正規化統計で過去窓を作る。本文で述べた原著の benchmark 成績を再現するには、このサンプル以外に原著の入力長、head、split、正規化と optimizer 設定を揃える必要がある。

## 原著の学習・推論の擬似コード

次は公式の**長期予測モデル**を shape で示す擬似コードであり、`PsLSTMBlock.forward` の処理や必須 loss ではない。

```text
入力:
  history = X[:, t-L+1:t+1, :]                  [B,L,M]
  future_gt = X[:, t+1:t+H+1, :]                [B,H,M]

forward:
  per_channel = history.transpose(time,channel) [B,M,L]
  patches = unfold(rearrange(per_channel, 'b m l -> (b m) l'), P,S)
                                                 [B*M,N,P]
  tokens = Linear(P,E)(patches)                   [B*M,N,E]
  states = sLSTM_stack(tokens)                    [B*M,N,E]
  forecast = Linear(N*E,H)(rearrange(states, 'bm n e -> bm (n e)'))
                                                 [B*M,H]
  pred = rearrange(forecast, '(b m) h -> b h m', b=B, m=M)
                                                 [B,H,M]
  loss = mean((pred-future_gt)^2)                 scalar

inference:
  同じ history を入れ pred を1回で得る。GT future は入力しない。
  新しい実測が来たら history を更新して再度 forward する。
```

H 点は一括出力なので「最初の予測を次の入力へ足し H 回繰り返す」recursive forecast ではない。backbone が見るのは過去 L 点のみだが、patch token は patch 内 P 点をまとめて見るため、同長特徴をオンラインで時点ごとに使うときには patch 内の右側点に注意する。
