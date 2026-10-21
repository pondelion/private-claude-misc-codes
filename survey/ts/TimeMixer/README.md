# TimeMixer — 季節性とトレンドを異なる方向に尺度間 mixing

## 概要

TimeMixer は長期・短期予測用の多尺度 MLP モデルで、入力を段階的に downsample し、各尺度で season/trend に分解する。その後、**season は細尺度から粗尺度へ、trend は粗尺度から細尺度へ**別々に mixing する *Past-Decomposable-Mixing*（PDM）を重ねる。原著の最終予測は各尺度から別々の未来予測を作り足し合わせる *Future-Multipredictor-Mixing*（FMM）が担当する。

ここでの目的は任意の回帰・分類へ差せる encoder なので、[ts_block.py](ts_block.py) は PDM を実装し、出力を最細尺度の `[B,T,Cout]` に揃える。未来 horizon と FMM は block に入れない。**PDM だけの block の精度を、原著 PDM+FMM の論文精度と同一視しないこと。**

## 新規性・他方式との差

通常の単一尺度 MLP は時間解像度を固定する。TimeMixer は fine の微細な変動と coarse の大域 trend を別尺度で扱い、さらに同じ方向へ一律に混ぜるのではなく season と trend で情報伝搬の向きを変える。論文の ablation では FMM 除去、分解除去、両方同方向の mixing がいずれも予測精度を落とすと報告する。これは予測の原著設定での証拠であり、この task-neutral block の精度保証ではない。時間 MLP の重みは尺度ごとの系列長に依存するので、任意長 T を同一重みで使えるモデルではない。

## モデルアーキテクチャ全体像（原著の予測モデル）

次の図では `w=2,M=2` の例を具体化する。`P` は過去長、`F` は未来長、`D` は embedding 幅。各尺度の系列長は `P, floor(P/2), floor(P/4)` であり、変数数 C とは別。**PDM は encoder、FMM は forecast decoder/head** という分業が本論文の大枠。

```text
過去 X [B,P,C] ──────────┬──────────────────────────────────────┐
                       │ avg pool w=2                         │
                       ▼                                      │
               X1 [B,P/2,C]                                  │
                       │ avg pool w=2                         │
                       ▼                                      │
               X2 [B,P/4,C]                                  │
                       │                                      │
        ┌──────────────┼──────────────────────┐               │
        ▼              ▼                      ▼               │
     Embed X0       Embed X1               Embed X2           │
   [B,P,D]        [B,P/2,D]            [B,P/4,D]             │
        │              │                      │               │
        └──────────────┴──────────┬───────────┘               │
                                  ▼                           │
                    PDM block × e_layers                      │
          各尺度 i: moving-average decomposition              │
               seasonal S_i [B,P_i,D] + trend R_i [B,P_i,D]  │
                    │                        │                 │
          season bottom-up              trend top-down         │
       S0 ─Linear_time(P,P/2)→ S1       R2 ─Linear(P/4,P/2)→ R1
       S1 ─Linear_time(P/2,P/4)→ S2    R1 ─Linear(P/2,P)→ R0
                    │                        │                 │
                    └─────── merge per scale ┘                 │
                                  │                           │
                E0' [B,P,D], E1' [B,P/2,D], E2' [B,P/4,D]    │
                     │                │            │          │
                     ▼                ▼            ▼          │
                  FMM head 0       FMM head 1   FMM head 2    │
                  time P→F        time P/2→F  time P/4→F      │
                     │                │            │          │
                  Ŷ0 [B,F,C]      Ŷ1 [B,F,C]   Ŷ2 [B,F,C]    │
                     └────────────────┴────────────┘          │
                                  ▼ sum                       │
                            Ŷ [B,F,C] ── MSE ── future GT [B,F,C]
```

PDM 内の `S0→S1→S2` は単に先頭尺度からの出力をコピーするのではなく、各 coarse scale の既存 `S_i` に time MLP の結果を**加算**する。trend 側は逆向きに各 fine scale の `R_i` へ加算する。FMM はどれか一尺度を選ぶ attention ではなく、**各尺度から F 点の予測を独立に作って足し合わせる**。そのため同長 encoder が最細尺度のみを返す場合、FMM が与える複数予測の補完効果は保持されない。

## 原著のアーキテクチャと objective

`B`: batch、`P`: 履歴長、`F`: 未来長、`C`: 入力変数、`D`: embedding 幅、`M`: downsample 段数、`T_m=floor(P/w^m)`。`w` は縮小率。

```text
history X [B,P,C]
  → downsample ladder X_m [B,T_m,C], m=0..M
  → per-scale embedding E_m [B,T_m,D]
  → L× PDM:
       moving-average decomposition: S_m,T_m' 各 [B,T_m,D]
       season bottom-up: S_(m+1) += MLP_time(T_m→T_(m+1))(S_m)
       trend top-down:  T'_m += MLP_time(T_(m+1)→T_m)(T'_(m+1))
       merge → E'_m [B,T_m,D]
  → FMM: 各尺度 E'_m から Linear_time(T_m→F) + channel projection
       各予測 ŷ_m [B,F,C] → sum_m ŷ_m [B,F,C]
  → GT future [B,F,C] と MSE 等の forecast loss [scalar]
```

`T_m` は時間長、`T'_m` は trend と記号が紛らわしいため上では prime を付けた。FMM は細尺度の細部と粗尺度の長期傾向の予測をアンサンブルする役割で、PDM と同じものではない。元実装には時刻特徴 embedding、scale 別 norm、`channel_independence` の分岐、複数 task path がある。分類 path は最細尺度 embedding を活性化・flatten して専用 head に渡す。

## 本 encoder の内部

```text
x [B,seq_len,Cin]
  → avg/max pooling ladder [B,T_m,Cin]
  → 尺度別 Normalize [B,T_m,Cin]
  → CI=0 なら moving-average 分解の seasonal を選択 [B,T_m,Cin]
  → circular Conv1d embedding [B,T_m,D]
  → e_layers 個の PDM: [B,T_m,D] 各尺度
  → 最細尺度を選択 [B,seq_len,D]
  → Linear(D→Cout) → h [B,seq_len,Cout]
  → 任意 pooling/head は外側
```

`channel_independence=False` では `Conv1d(Cin→D)` が入力変数を混ぜる。`True` では各変数を `[B*Cin,T_m,1]` として共有 `Conv1d(1→D)` に通し、最細尺度を `[B,T,Cin*D]` にまとめてから出力 Linear に渡す。各 PDM は moving average で分解し、季節性の下向き MLP と trend の上向き MLP を別重みで適用。原著公開コードの `channel_independence=0` path では `cross_layer` を分解後に適用し `season+trend` を返すが、論文の概念式の residual `x+FF(...)` をそのまま実装しているわけではない。ここは**公開コードの分岐**に従った。

`seq_len` と `down_sampling_window` を変えると尺度ごとの `Linear(T_m→T_(m±1))` の重み形状が変わるため再構築が必要。`down_sampling_method` は avg/max。公式の尺度別 Normalize と CI=0 の `pre_enc` seasonal 経路を block 内に保持する。公式の conv downsample と時刻 marker 経路は含めない。FMM は未来予測 head であり、同長特徴を返す本 block の外に置く。

追加オプション（既定値はいずれも公式 forecast 経路と同一で、公式 parity は既定値で確認）:

- `input_decomp`（既定 `True`）: CI=0 で埋め込み前に季節成分 `x−MA(x)` だけを渡す公式 `pre_enc` の挙動。窓内で一定の水準は `use_norm=False` でも完全に消える（数値確認済み）。`False` にすると入力をそのまま埋め込み、`use_norm=False` と組み合わせると公式 classification 経路と同じ入力になる。
- `length_policy`（既定 `"strict"`）: 尺度ごとの pooling は切り捨てのため、`seq_len` が `down_sampling_window**down_sampling_layers` の倍数でないと粗い尺度から**直近**の時刻が落ちる（公式も同じ挙動で、公式の `seq_len=96` では問題にならない）。`"strict"` はそのような `seq_len` を構築時に `ValueError` で拒否する。`"crop_oldest"` は任意の `seq_len` を受け付け、各 pooling の前に**最も古い**余りを落とす。最細尺度と出力は全時刻を保つ。
- 埋め込み `Conv1d` は公式 `TokenEmbedding` と同じ `kaiming_normal_(mode="fan_in", nonlinearity="leaky_relu")` で初期化する（重みを公式からコピーする parity 確認では初期化の差は検出されないため、明示的に合わせた）。

## 使い方・DataFrame 例

```python
from ts_block import TimeMixerBlock

block = TimeMixerBlock(in_dim=8, out_dim=64, seq_len=96,
                       d_model=64, d_ff=128, e_layers=2)
h = block(x)                 # [B,96,8] → [B,96,64]
prediction = head(h.mean(1)) # Linear(64,K) → [B,K]
```

[finetuning_example.py](finetuning_example.py) は各系列の DataFrame を時刻順 80/20 に分け、train 行のみで特徴標準化し、`history:[B,T,Cin]` から**直後 1 行の**回帰値/クラスを外部 head で学習する。GT は encoder の中に入らない。原著の未来 F 点アンサンブル学習とは別タスク。

## downsampling と系列分解を具体化する

`seq_len=96`, `window=2`, `down_sampling_layers=2` なら各尺度の時間長は `96,48,24`。原著の基本設定は avg pooling を使い、fine から coarse へマクロな変動を集める。各尺度を D 次元へ埋め込んだ後、移動平均 `trend_i=MA_k(E_i)` と `season_i=E_i-trend_i` に分ける。k 点移動平均の端では最初/最後の値を複製するので、分解後も `[B,T_i,D]` の長さを保つ。

尺度 i=0 の seasonal を粗い i=1 へ送るとき、`[B,D,96]→Linear(96,48)→GELU→Linear(48,48)` の**時間軸 MLP**を使う。これは単なる平均 pooling ではなく、96 個の時間位置の結合を学習する。trend の逆方向は `[B,D,24]→Linear(24,48)→GELU→Linear(48,48)` など別パラメータ。向きが逆である理由は、細尺度の周期細部を粗尺度に注入し、粗尺度で安定したマクロ trend を細尺度へ伝えるため。二方向の Linear を共有してはいけない。

`channel_independence=0` の公開 PDM は分解後の季節性・trend 双方に `cross_layer(D→d_ff→D)` をかける。`channel_independence=1` の場合は前段で `[B,T,C]→[B*C,T,1]` へ変形して各変数を独立に埋め込み、PDM の `out_cross_layer` の残差を使う。本版もこの二つの PDM 分岐を実装した。`True` の最終 readout では `[B*C,T,D]→[B,T,C*D]→[B,T,Cout]` として、**最後の射影だけ**で変数を融合する。原著の CI=1 forecast projection は各変数へ戻すため、本版の `C*D→Cout` は task-neutral な追加である。

## 原著の訓練・推論をもう一段具体的に

学習 sample は、直近 P 時刻の観測 `X:[B,P,C]` と、その後 F 時刻の実測 `Y:[B,F,C]` の対。forecast では全未来点を一括で出すため、推論時に F 回 autoregressive loop を回さない。scale 別 predictor は PDM 出力から同時に計算され、最終 forecast は和。train で `MSE(Ŷ,Y)` を計算する。原著には長期/短期予測以外に classification、anomaly detection、imputation の forward path もあるが、head と正解の意味はそれぞれ異なる。PDM が「全タスクで同じ MSE」を最適化するわけではない。

公開 `forecast()` は `__multi_scale_process_inputs`、scale ごとの `Normalize`、CI=0 なら `pre_enc` の季節性分解、`DataEmbedding_wo_pos`、PDM stack、`future_multi_mixing`、scale 予測和、scale0 の逆正規化という経路を通る。本 block は**forecast の元入力から PDM 最細尺度出力まで**を残す。`x_mark=None`・avg downsample では、公式の対応経路と出力が一致する。未来予測 FMM 以降だけを外す。

## 論文 ablation をどう読むか

TeX の Model Analysis は FMM を除くと長期/短期の予測精度が下がると報告し、season/trend を同じ方向へ混ぜる設定や分解無しで直接尺度を混ぜる設定も悪化すると示す。よって原著の新規性は「avg pooling を何回か行った」だけではなく、**何を分け、どちら向きに混ぜ、最後にどの尺度を予測へ利用するか**にある。逆に本版の下流タスクで FMM を含まないことは原著の完全実装ではなく、良好な予測成績を保証する根拠にもならない。

PDM の時間 Linear は `T_i×T_(i+1)` 程度の重みを持ち、最細尺度の長さが増えるほどパラメータ・演算が増える。self-attention の T² 行列は構築しないが、時間 MLP に T 依存の二乗項が残る。「fully MLP-based だから全ての長系列で線形計算」という意味ではない。原著の効率比較もその特定の P/F/C・設定での実測として読む。

## コード対応と制約を一覧化

| 役割 | 公開 `models/TimeMixer.py` | 本版 | 相違 |
| --- | --- | --- | --- |
| 多尺度生成 | `__multi_scale_process_inputs` | `avg_pool1d` / `max_pool1d` | conv downsample は未搭載 |
| 入力埋込 | `DataEmbedding_wo_pos` | circular `Conv1d(Cin,D)` | 時刻マーカー埋込は別 |
| 移動平均 | `series_decomp` | `moving_average` | 端複製と長さを保つ |
| 季節性 mixing | `MultiScaleSeasonMixing` | `season_down` | 時間 MLP の細→粗 |
| trend mixing | `MultiScaleTrendMixing` | `trend_up` | 時間 MLP の粗→細 |
| 未来予測 | `future_multi_mixing` | 実装しない | 任意 head を外付け |
| 同長出力 | 分類 path は flatten head | 最細尺度 `Linear(D,Cout)` | 各時刻表現を返す |

`moving_avg` は奇数に限定した。偶数 kernel では元の公開 moving-average 実装も端 pad の定義に注意が必要で、ここでは長さ不変を保証する設定に絞った。デフォルトは `down_sampling_layers=2` だが、T が小さく最粗尺度が 0 点になる設定は拒否する。元の horizon head/checkpoint は本版と重み shape が違うので流用できない。

## 再利用のための学習擬似コード

~~~text
training example:
  x = 過去 T 点の feature [B,T,Cin]
  multi-scale PDM h = TimeMixerBlock(x) [B,T,Cout]
  pooled = mean_time(h) [B,Cout]
  prediction = task_head(pooled) [B,K]
  GT = 次行の回帰 target [B,K] または class [B]
  loss = MSE / CrossEntropy; backward; AdamW update
validation:
  時刻順の後半の label だけを使い、window history は過去のみ
inference:
  train と同じ T・feature 順・scaler で encoder/head を呼ぶ
~~~

原著の forecast と異なり、ここでの pooling は簡潔な下流例にすぎない。特徴の直近時刻が重要なら `h[:, -1]`、複数の時刻に意味があるなら attention pooling、時刻別 label なら pooling しない head を検討する。未来の観測や後から確定する特徴を入力窓へ混ぜないことが最優先。

## 固定長制約と下流 head の選び方

TimeMixer の `season_down[i]` は入力長 `T_i`、出力長 `T_(i+1)` を持つ learned Linear、`trend_up[i]` は逆向きの learned Linear。よってモデル構築時の `seq_len` を変えると重みの shape 自体が変わる。例えば `T=96,w=2,M=2` と `T=128,w=2,M=2` の両方を、同一 `TimeMixerBlock` へ無加工で通すことはできない。可変長データは前処理で共通窓長へ crop/pad/resample するか、別々の block を作る必要がある。pad した時刻も learned Linear に入るため、pad 値の影響を考慮する。

mean pooling は時刻位置を潰した系列単位の例で、未来予測専用の公式 FMM とは違う。回帰 target が窓末尾に特に依存するなら `h[:, -1]` を使う選択もある。チャンネル独立設定では最後の `C*D→Cout` が初めて入力変数間を混ぜるため、全体を通じた豊かな cross-variate interaction を期待するなら `channel_independence=False` も比較する。逆に変数間関係が不安定なデータでは `True` が正則化として働く可能性があるが、精度改善は検証で判断する。

## 論文式・公開コード・本版の三者を整理

論文 PDM の概念式は `X^l = X^(l-1) + FeedForward(S-Mix+T-Mix)` と residual 付き。しかし公開 `PastDecomposableMixing.forward` の `channel_independence=0` 経路では、分解後に `cross_layer` を当てて mixed season+trend をそのまま返し、同じ形の外側 residual を加えない。`channel_independence=1` 経路では `ori+out_cross_layer(...)` がある。本版は公開経路のこの差を保持した。論文式の図だけから「どちらも同じ residual」と実装すると、重みを合わせても数値が変わる。

また公開 forecast は入力時の scale 別 `Normalize` と `pre_enc` を使うが、classification では尺度ごとに直接 embedding して PDM へ渡す。本版は既定では forecast 側の入力経路（scale 別 Normalize と `pre_enc`）に従い、`use_norm=False, input_decomp=False` で classification 側と同じ入力経路になる。いずれの場合も予測 path の trend 残差 head、future temporal feature、FMM は持たない。したがって下流回帰に使うときに「forecast で使った TimeMixer と完全同一」と呼ばない。保持したのは**多尺度の PDM 設計**である。

## 処理別の形状チェック

| 処理 | channel-dependent (`CI=False`) | channel-independent (`CI=True`) |
| --- | --- | --- |
| 入力 scale i | `[B,T_i,Cin]` | `[B,T_i,Cin]` |
| embedding 直前 | `[B,Cin,T_i]` | `[B*Cin,1,T_i]` |
| circular token embedding 後 | `[B,T_i,D]` | `[B*Cin,T_i,D]` |
| PDM の season/trend | 各 `[B,T_i,D]` | 各 `[B*Cin,T_i,D]` |
| 最細尺度の readout 直前 | `[B,T,D]` | `[B,T,Cin*D]` |
| encoder 出力 | `[B,T,Cout]` | `[B,T,Cout]` |

この表で `B*Cin` は変数を batch 次元へ畳んだもの。注意機構の head 数や空間 node 数ではない。PDM の時間混合は各隠れ feature に共通の時間 Linear を作用させるため、少なくとも非因果的な全窓演算となる。時刻 t の特徴を online 判定へ使うなら、入力窓に t 以降を含めない。

## 実装のパラメータをどう調整するか

`down_sampling_layers=M` を増やすと coarse scale を多く作れるが、最粗 `T_M` が数点になると時間 MLP が持つ時系列情報は限られる。`down_sampling_window=w` を大きくすると一段で急速に短くなり、seasonal の短い変動が粗尺度へ伝わる前に pooling で失われ得る。`moving_avg=k` は season/trend の分割境界であり、系列の支配的周期やノイズ幅との関係を見て選ぶ。`e_layers` は PDM を何回反復するかで、増やすほど尺度間で情報交換を繰り返すが、計算・過学習の可能性も増える。

TimeMixer の「season」は暦上の曜日周期のように明示指定するラベルではなく、移動平均を引いた残差の呼び名。よって k の選び方次第で短い trend も season 側へ入り得る。CycleNet のような既知位相 `phase` や learnable `Q:[W,C]` を持つわけでもない。これらを「同じ周期モデル」とまとめると、必要な入力条件と帰納バイアスを見誤る。

`down_sampling_method="max"` は大きなピークを残しやすいが、負値主体の系列では最大値が平常値を代表するとは限らない。原著の基本説明は平均 downsampling。精度を求める実用設定では avg/max を validation で比較し、特徴列の正規化・欠損処理を一定に保つ。`conv` downsampling は本携帯版には無いため、原著の該当設定結果との直接比較を行わない。

出力を最細尺度に限定した理由は `[B,T,Cout]` の時刻対応を保持するためである。粗尺度を直接 upsample して連結する追加 readout も考えられるが、その補間方法・重み付けは原著 PDM の定義ではない。本版は PDM 内の season/trend 情報交換を通じて粗尺度の情報を最細尺度へ伝え、その結果だけを出力する。FMM と異なる点をここでも明示しておく。

## 原著 forecast の train/inference 擬似コード

~~~text
train sample: history [B,P,C], future GT [B,F,C]
for m=0..M: downsample history → [B,T_m,C]
scale 別 normalize + embedding → [B,T_m,D]
for l=1..L: PDM の season 細→粗 / trend 粗→細 mixing
for m=0..M: Predictor_m([B,T_m,D]) → [B,F,C]
sum_m predictions → final [B,F,C]
MSE(final,future GT) scalar → backward
infer: history だけを入力し同じ F 点を一括で得る
adapter: PDM 後の最細尺度を [B,P,Cout] へ投影して返す
~~~

ここで `T_m` は `floor(P/w^m)` であり、downsample の window と moving-average の kernel は別パラメータ。推論の F 点は同時に出るため、時刻を一歩ずつ生成する autoregressive decoder ではない。

## 一次資料・検証範囲

原論文: [TimeMixer (ICLR 2024)](https://arxiv.org/abs/2405.14616)。公式実装: [kwuking/TimeMixer](https://github.com/kwuking/TimeMixer)。照合箇所: 論文 `iclr2024_conference.tex`、公開 `models/TimeMixer.py`, `layers/Autoformer_EncDec.py`, `layers/Embed.py`。`x_mark=None`・avg downsample の公式 forecast encoder を実行し、尺度別 Normalize→CI=0 の seasonal 前処理→value embedding→PDM 最細尺度出力まで CI=0/1 の両方で数値一致を確認した。
