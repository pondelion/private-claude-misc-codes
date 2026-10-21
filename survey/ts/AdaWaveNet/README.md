# AdaWaveNet — 学習可能な lifting 小波で非定常な系列を多尺度処理

## 概要

AdaWaveNet は非定常時系列の予測・欠損補完・超解像を対象に、固定 wavelet filter の代わりに**学習可能な lifting scheme** を用いる。原著の全体は移動平均で trend と seasonal を分け、seasonal を多段 lifting で粗い近似と複数の detail に分け、最粗近似にはチャンネル間 self-attention を適用し、逆 lifting で系列を戻す。trend 側はチャンネルを K-means で群分けし、群別の時間 Linear で予測する。

このディレクトリでは原著の `L→H` task heads をそのまま持ち込まず、[ts_block.py](ts_block.py) を `x:[B,T,Cin]→h:[B,T,Cout]` の encoder として使う。pooling/回帰/分類 head は外付けである。

## 新規性と既存方式との差

固定 DWT の filter は全サンプル・全タスクに同じ basis を使う。AdaWaveNet は even/odd の相互作用を depthwise Conv1d の `U` と `P` で学習するため、分解係数自体をデータに適合できる。さらに最粗尺度だけにチャンネル間 attention を置き、長い生系列上の full attention より時間長を短くする。単純に wavelet 分解するだけでなく、trend を別系列として予測し、チャンネル群ごとに異なる変換を持つ点が原著の特徴。ただし学習可能な U/P は固定 Haar のような直交性・周波数帯域境界を保証しないし、K-means の結果はデータ構成に依存する。

## モデルアーキテクチャ全体像（原著の予測モデル）

次の図は**論文の forecast 設定と公開 `models/AdaWaveNet.py` の実行経路**を区別して示す。`H` は未来長、`J` は lifting 段数、`C` は変数数。学習用の GT は右端で初めて使う。入力へ future を連結しているわけではない。

```text
過去 X [B,L,C] ──────────────────────────────────────────┐
      │                                                 │
      └── moving_avg(time) ──> trend M [B,L,C]          │
               │                  │                      │
               │                  └── RevIN(norm)        │
               │                        [B,L,C]          │
               │                        │ per-channel K-means label [C]
               │                        ▼                 │
               │                    cluster-wise Linear(L,H)
               │                        [B,H,C]          │
               │                    RevIN(denorm)        │
               │                        │ trend_hat       │
               ▼                        │                 │
      seasonal S=X-M [B,L,C]          │                 │
               │ instance normalization                  │
               ▼                                          │
           [B,C,L]                                       │
               │  AdaWave level 1: split even/odd       │
               ├────────── detail d1 [B,C,L/2] ─────┐ │
               ▼                                      │ │
           coarse c1 [B,C,L/2]                         │ │
               │  AdaWave level 2                     │ │
               ├────────── detail d2 [B,C,L/4] ───┐ │ │
               ▼                                   │ │ │
           coarse c2 [B,C,L/4]                      │ │ │
               │  ... J levels                         │ │ │
               ▼                                      │ │ │
           coarse cJ [B,C,L/2^J]                     │ │ │
               │ channel-token embedding +           │ │ │
               │ Transformer Encoder + projection    │ │ │
               ▼                                      │ │ │
           predicted coarse [B,C,H/2^J]              │ │ │
               │  InvAdaWave J with dJ + coarse skip ┤ │ │
               ▼                                      │ │ │
           [B,C,H/2^(J-1)]  ← detail d(J-1) ─────────┘ │ │
               │  ... InvAdaWave down to level 1        │ │
               ▼                                        │ │
           seasonal_hat [B,H,C] ← d1 ──────────────────┘ │
               │ inverse instance normalization          │
               └────────────── + trend_hat ──────────────┘
                              ▼
                         prediction [B,H,C]
                              │ compare with future GT [B,H,C]
                              ▼
                           MSE scalar
```

図の `L/2^j` は偶数で割り切れる構成の直感的表記。公開コードの level 別 `linear_levels`, `coef_linear_levels`, `coef_dec_levels` は coarse と detail をそれぞれ線形射影して inverse path へ加える。したがって「detail をそのまま置いて逆 wavelet するだけ」ではない。Trend branch の cluster label は全チャンネルの波形を K-means した結果で、各 batch sample に個別の cluster を割り当てるわけではない。`self.clusters` は初回 forward 後に buffer として保持される。

## 原著の forward と学習

`B`: batch、`L`: 観測長、`H`: 予測長、`C`: チャンネル数、`J`: lifting 段数。典型的に `L_j=L/2^j`（公開コードは偶数・長さ整合を前提とする）。

```text
history X [B,L,C]
  → moving average trend M [B,L,C], seasonal S=X-M [B,L,C]
  ├─ seasonal: normalize → [B,C,L]
  │  → J× adaptive lifting: c_j,d_j each [B,C,L_j]
  │  → coarsest c_J [B,C,L_J]
  │  → channel-token embedding/Transformer/low-rank projection
  │  → predicted coarse coefficients [B,C,H_J]
  │  → inverse lifting with transformed details → [B,C,H]
  │  → seasonal denorm [B,H,C]
  └─ trend: RevIN + channel K-means + cluster-wise Linear(L→H)
      → trend denorm [B,H,C]
  → sum seasonal + trend [B,H,C]
  → GT future [B,H,C] と予測 MSE [scalar]
```

欠損補完・超解像では GT/マスクと出力長が変わる。したがってこの forecast 図を全タスク共通の教師信号だと解釈してはいけない。論文は `AdaWave` 係数に detail/approximation 正則化を記述するが、公開 `AdpWaveletBlock.forward` では `x=c` に代入した後 `dist(c.mean(), x.mean())` を計算するため、その approximation 項はそのまま読むと恒等的にゼロ。また forward で得た `r` は後段予測 path で loss に足されていない。論文意図と公開コードの実際を区別する必要がある。

## lifting の式と正逆の注意

論文本文の予測→更新の説明は標準形 `d=o-P(e), c=e+U(d)`。しかし公開 `LiftingScheme` は `modified=True` がデフォルトで、**実際の forward** は `c=e+U(o), d=o-P(c)`。その厳密な逆は `o=d+P(c), e=c-U(o)` だが、公開 `InverseLiftingScheme` は**独立した重み**で `e=c-U(d), o=d+P(e)` を計算する。本 block の主経路もこの公開 decoder と同じ式・独立重みを使う。`AdaptiveLifting.inverse()` は数学的な往復確認用であり、主 block の decoder には使わない。

## 本 block の処理と形状

```text
x [B,T,Cin] → trend (moving average) / seasonal [B,Cin,T]
seasonal → instance norm [B,Cin,T]
  → per-level learned lifting: (c_j,d_j) [B,Cin,ceil(T_(j-1)/2)]
  → 各 level の c_j,d_j を時間軸で別々に instance norm
  → coarsest c_J [B,Cin,T_J]
  → Linear(T_J→D) + dropout + channel-token Transformer + Linear(D→T_J)
      [B,Cin,T_J]
  → reverse levels: coarse skip + 独立した2本の detail Linear + 公式 inverse lifting
      [B,Cin,T]
  → inverse norm + learned trend Linear(T→T) [B,Cin,T]
  → Linear(Cin→Cout) → h [B,T,Cout]
```

原著の cluster 別 trend Linear を同長 `T→T` で残し、各入力変数の cluster ID `[Cin]` を `forward(..., clusters=...)` で受ける。省略時は全変数を cluster 0 とする。逆 lifting は encode 側と独立した U/P を持ち、detail も公式同様に独立した2本の Linear を加算する。奇数 T は末尾複製と中間 crop による汎用 block 用拡張で、公式との出力照合は T が分解段数で割り切れる設定で行った。時間方向の各 Linear があるため `seq_len` は構築時に固定。channel-token attention は `Cin` に対して二乗計算になるので、多数の銘柄/センサを一つの channel 軸に置く場合は速度を確認する。

## 使い方・学習

```python
from ts_block import AdaWaveNetBlock

block = AdaWaveNetBlock(in_dim=8, out_dim=32, seq_len=96,
                        levels=2, d_model=64, heads=4)
h = block(x)                 # [B,96,8] → [B,96,32]
prediction = head(h.mean(1)) # Linear(32,K) → [B,K]
```

[finetuning_example.py](finetuning_example.py) は DataFrame の各系列を時刻順に分け、過去窓から直後の行の label を回帰または分類する。GT は block へ渡さず、mean pooling の後の head に対して MSE/CE を計算する。元論文の forecast・imputation・super-resolution の共通再現実験ではない。

## AdaWave の 1 段を具体的に追う

入力を `z:[B,C,T]` とする。偶数・奇数位置を分けると `e=z[...,0::2]`, `o=z[...,1::2]` で各 `[B,C,T/2]`。公式既定の `modified=True` では、チャンネルごとの depthwise convolution、GELU、`LayerNorm([C,T/2])` からなる `U/P` を使う。

```text
o [B,C,T/2] ── U ─┐
                   +── e → c [B,C,T/2]  低解像度の近似
c [B,C,T/2] ── P ─┐
                   −── o → d [B,C,T/2]  詳細係数
```

`d` は単なる高周波 FFT bin ではない。learnable filter が「粗い構造で説明できない odd の成分」を作るので、固定 Haar の detail と解釈を完全同一視できない。`c,d` が分かり同一の U/P を使う数学的な逆では `o=d+P(c)` を先に求め、次に `e=c-U(o)` とする。しかし**公式 decoder と本 block の主経路は、その逆を使わず、独立パラメータの `e=c-U(d), o=d+P(e)` を使う**。

## タスク別の入力・教師信号を混同しない

| 原著タスク | モデル入力 | 出力と GT | 注意 |
| --- | --- | --- | --- |
| 長期予測 | 過去観測 `[B,L,C]` | 未来予測/実測 `[B,H,C]` | train は未来 MSE。論文では多くの設定で `L=H` を採用 |
| ランダム補完 | 欠損を模した入力 `[B,L,C]` と mask | 復元系列/元観測 `[B,L,C]` | どの要素で損失を平均するかは学習 driver の実装を確認すべき |
| contiguous gap 補完 | 連続時刻を欠損させた入力 | 元系列との復元誤差 | ランダム欠損より長い文脈推定が必要 |
| 超解像 | 間引いた低解像度 `[B,L_low,C]` | 高解像度 GT `[B,L_high,C]` | 欠損点だけでなく時間解像度自体を増やす |

README 内の `MSE(future,GT)` は **forecast 経路だけ**の説明である。公開 `imputation()` は `mask` 引数を受けるが、`moving_avg_imputation` は値が 0 かどうかで欠損を判定しており、真の観測値 0 と欠損 0 が同居するデータでは曖昧。`torch.ones(...).cuda()` を関数内で生成しているため CPU-only や任意 device へそのまま移植できない。この survey の同長 encoder はそれを隠して補完 API と称さず、補完したい場合は外側で mask を明示的に扱う。

## 原著の ablation と読み方

TeX の ablation 表では、Weather 予測の MSE は full 0.233 に対して channel-wise attention を除くと 0.256、adaptive wavelet block を除くと 0.255。Traffic 予測は full 0.415、attention 除去 0.519、wavelet 除去 0.473。一方 PTB-XL extended imputation では full 0.069、grouped linear 除去 0.080、wavelet 除去 0.082。**どの部品が効くかはデータ・タスクに依存する。** とくに本 block は grouped trend prediction を置換したので、これらの full-model ablation 差を本 block の下流改善量として流用できない。

論文の効率図は iTransformer と同程度の計算・メモリ範囲、DLinear より重いことを示す。`T/2^J` へ短縮した attention でもトークン軸は C（変数数）なので、C が数百〜千に増えると channel attention の二乗項が現れる。固定 wavelet に対する可塑性は計算無料ではない。

## 公式コードとの対応表

| 論文概念 | 公開コード | このディレクトリ | 一致範囲 |
| --- | --- | --- | --- |
| moving average trend/season | `series_decomp` | `_moving_average` | 端複製による同長平滑化 |
| learnable lifting | `LiftingScheme.P/U` | `AdaptiveLifting.predict/update` | modified forward は係数・入力勾配を公式と数値照合。十分な長さでは reflect pad、短い band だけ replicate pad |
| level ごとの係数 norm | `AdpWaveletBlock.norm_x/norm_d` | `_instance_normalize` | affine なし、時間軸の biased 分散。長さ1も扱えるよう明示式で計算 |
| coarse channel attention | `DataEmbedding_inverted` + `Encoder` | `coarse_embedding` + `channel_encoder` | 公式の全 Encoder と出力照合済み |
| inverse synthesis | `InverseLiftingScheme` | `_PublishedInverseLifting` | 独立 U/P と式が一致 |
| trend prediction | `KMeans` + `ClusteredLinear(L,H)` | 外部 cluster ID + `trend_mixer(T,T)` | cluster 別 Linear の演算が一致 |
| task projection | `projection(...→H)` | `output(Cin→Cout)` | 出力の意味が異なる |

公式の対応重みをコピーし、`T=H=16`・1 cluster・追加出力射影を恒等にしたとき、**入力 `[B,T,Cin]` から最終 `[B,T,Cin]` まで公式 `forecast` の forward 出力と一致**した。比較には独立 inverse、channel encoder、trend branch を含む。K-means による cluster ID 推定そのものは block の外で、同じ ID を両者へ与えた。

## 同長 block の学習・推論を擬似コードで見る

```text
train:  DataFrame の series_id ごとに時刻 sort
        先頭 80% だけで feature μ/σ を推定
        x = label 時刻直前 T 点の feature             [B,T,Cin]
        h = AdaWaveNetBlock(x)                        [B,T,Cout]
        pooled = mean_time(h)                         [B,Cout]
        y_hat = external_head(pooled)                 [B,K]
        y = label 時刻の実測回帰値 / クラス           [B,K] / [B]
        loss = MSE(y_hat,y) / CE(y_hat,y)             scalar
        loss.backward(); AdamW.step()
valid:  後半 label のみで同じ forward、勾配更新なし
infer:  過去 T 点を同じ train 統計で標準化して block→head
```

このデータ分割でも、同一 timestamp の複数銘柄間の漏洩、将来にしか分からない特徴の混入、欠損補間に全期間統計を使う漏洩は自動では防げない。実データ側で列の availability と group split を設計する。

## 各軸・設定値の意味と導入時の確認表

| 引数 | 何を決めるか | 大きくしたときの主な代償 |
| --- | --- | --- |
| `seq_len=T` | time Linear と lifting の LayerNorm shape | 固定 T の重みを再学習/再構築する必要 |
| `levels=J` | coarse 長 `ceil(...ceil(T/2)/2...)` | detail branch と U/P が増え、最粗の時刻情報が減る |
| `lifting_kernel` | 各 U/P depthwise Conv の局所受容野 | 畳み込み計算と端 padding への感度が増える |
| `d_model` | channel token の attention 幅 | channel attention/FFN のパラメータとメモリ増 |
| `heads` | coarse channel attention の分割数 | `d_model%heads=0` が必要 |
| `trend_kernel` | trend を見積もる移動平均の幅 | 季節側へ残す速い変動との境界が変わる |

データの各列を `Cin` に置くと channel-wise attention は列同士を相互参照する。一方、複数の全く異なる資産や装置を同一 `Cin` に束ねる場合は「列順が固定」「同じ列が全サンプルで同じ意味」という前提が必要。特徴列が変動するデータセットにはそのまま適用できない。`T` が短いと J 段の coarse がほぼ 1 点となり、attention が列間相関を見る以外に時間構造を細かく扱えなくなる。`J` の選択はサンプル時間長に対して検討する。

本 encoder の output `[B,T,Cout]` は全履歴窓を見た特徴なので、窓内各時刻 `h[:,t]` が後続の同窓サンプルを参照しない causal 表現ではない。系列単位の予測なら末尾まで観測済みの履歴を入力にして使えるが、オンラインの時刻 t の分類には t より未来の観測を同じ窓に入れない。これは channel attention だけでなく moving average と time Linear でも重要。

`AdaptiveLifting` の逆変換検証と予測精度検証は別物である。前者は U/P を共有した正逆の代数だけを確認し、attention や trend mixer の有用性は示さない。実務導入時は `levels`, `trend_kernel`, `heads` を変更した ablation と、Haar/WPMixer のような固定 filter baseline を同じ分割で比較する。

## 一次資料と再現境界

原論文: [AdaWaveNet: Adaptive Wavelet Network for Time Series Analysis](https://arxiv.org/abs/2405.11124)。公式実装: [comp-well-org/AdaWaveNet](https://github.com/comp-well-org/AdaWaveNet)。照合箇所: 論文 `main.tex` の Adaptive Wavelet Network/Experiments/Discussion、公開 `models/AdaWaveNet.py`, `layers/LiftingScheme.py`。本 survey は論文の表の精度再現や公式 checkpoint parity を主張しない。確認できたのは本 block の形状・勾配・lifting 可逆性まで。
