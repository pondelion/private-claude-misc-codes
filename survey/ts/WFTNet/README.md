# WFTNet — FFT の大域周期と wavelet の局所周期を融合

## 概要

WFTNet は *Wavelet-Fourier Transform Network*。FFT で窓全体に強い周期を見つけて 2D 周期マップへ並べる大域 branch と、連続小波変換（CWT）で時刻ごとの尺度応答を作る局所 branch を並列に処理する。両方を Inception 型の 2D 畳み込みで変換し、周期性の強さに応じて融合する構想である。原著は長期予測 `history:[B,L,C]→future:[B,H,C]` を学習する。

この survey の目的は原著の予測 decoder を固定で持ち込むことではなく、[ts_block.py](ts_block.py) を `x:[B,T,Cin]→h:[B,T,Cout]` の特徴 encoder として使うこと。下流の pooling/head/loss は利用者側で決める。

## 差分と新規性

TimesNet の FFT 周期 2D map は窓全体の周期を抽出するが、周期や局所周波数が窓内で変わる現象の表現は弱い。WFTNet は CWT の `尺度×時刻` map を追加して、時間局在した周期性を表す。固定周波数だけの Fourier 法とも、wavelet だけの法とも異なるのは、両 branch の予測特徴を融合する点。ただし 2D Inception と 8 尺度 CWT は単純な 1D conv より重くなり、FFT top-k はバッチ全体の平均スペクトルで選ぶため、サンプルごとの卓越周期が大きく異なる batch では共有周期が不適切になり得る。

## モデルアーキテクチャ全体像（原著の予測経路）

予測器全体の時間長を `T_e=L+H`、WFTBlock の特徴幅を `D` とする。図の二つの「2D」は同じ意味ではない。FFT 側は**周期ごとに折りたたんだ行列**、CWT 側は**scale×time 係数**である。

```text
過去 X [B,L,C] ───────────────────────────────────────────┐
    │ window mean/std, normalize                           │
    ▼                                                       │
X_norm [B,L,C] ── DataEmbedding ── [B,L,D]                  │
    │ learned time Linear(L,T_e)                            │
    ▼                                                       │
E^0 [B,T_e,D]                                             future GT Y [B,H,C]
    │                                                       │
    ├──── Fourier/global branch ─────────────────────┐     │
    │  rFFT(time) [B,F,D]                            │     │
    │    ├─ batch/channel 平均振幅で top-k bin を選択│     │
    │    ├─ bin f_k → period p_k=floor(T_e/f_k)       │     │
    │    └─ 各 sample の振幅 → softmax weights [B,K] │     │
    │  for k=1..K:                                    │     │
    │    zero-pad time → [B,D,ceil(T_e/p_k),p_k]       │     │
    │    InceptionConv2d(D→d_ff) → GELU              │     │
    │    InceptionConv2d(d_ff→D) → crop [B,T_e,D]     │     │
    │  amplitude 加重和 → F_global [B,T_e,D]          │     │
    │                                                 │     │
    └──── Wavelet/local branch ────────────────────────┤     │
       CWT(8 Morlet scales) [B,D,8,T_e]               │     │
       same InceptionConv2d ×2 [B,D,8,T_e]            │     │
       depthwise Conv2d kernel=(8,1) [B,D,1,T_e]      │     │
       transpose/squeeze → F_local [B,T_e,D]           │     │
                                                      ▼     │
        mix = α^10·F_global + (1-α^10)·F_local [B,T_e,D]   │
                  + E^0 residual → LayerNorm [B,T_e,D]      │
                    │ e_layers 回 WFTBlock を積む            │
                    ▼                                       │
               Linear(D→C) [B,T_e,C]                       │
                    │ inverse normalize                     │
                    ▼                                       │
               final H points Ŷ [B,H,C] ── MSE ────────────┘
```

ここで `F=floor(T_e/2)+1`。大域 branch の `K` 個の 2D map は**異なる period 幅**を持つため、同一の Conv2d を K 回適用した後、元の時間長に戻してから重み付き加算する。局所 branch も同じ `period_conv` モジュールを再利用しているため、両 branch の 2D 抽出器は重み共有である。最後の strip convolution は各 feature map に独立な depthwise 8×1 で、尺度を 1 に潰す。

## 原著の forward/objective と形状

`B`: batch、`T_e=L+H`: 時間拡張後の長さ、`D`: embedding 幅、`K`: 上位周波数数。`p_k=floor(T_e/f_k)` は**公式コード**の FFT bin `f_k` から得る period。論文の「周波数の逆数」は概念表現であり実際の整数化はこの床関数。

```text
history [B,L,C] → instance norm → value embedding [B,L,D]
  → learned Linear(L→L+H) [B,T_e,D]
  ├─ rFFT time [B,floor(T_e/2)+1,D]
  │    → global top-K bins f_k, per-sample amplitude weights [B,K]
  │    → each 2D period grid [B,D,ceil(T_e/p_k),p_k]
  │    → 2× Inception conv → restore time [B,T_e,D]
  │    → softmax amplitude-weighted sum [B,T_e,D]
  └─ CWT 8 scales [B,D,8,T_e]
       → same 2× Inception conv → depthwise (8,1) conv
       → local feature [B,T_e,D]
  → blend + residual → LayerNorm (stacked e_layers) [B,T_e,D]
  → Linear(D→C), inverse norm, take final H points [B,H,C]
  → future GT [B,H,C] と forecast loss [scalar]
```

Wavelet branch の 8 尺度は公式 `2**linspace(-1, wavelet_scale, 8)`。Inception は 1,3,... の奇数 kernel Conv2d の平均で、wavelet branch の strip conv `(8,1)` が尺度軸だけを畳み込む。`top_k=0` では FFT branch が無い。

## PWC の重要な論文・公開コード差分

論文の Periodicity-Weighted Coefficient は各チャンネルの低周波エネルギーについて `max(power)/sum(power)` を求めて C 平均する**入力依存の `α_b`**。融合重みは `α_b^n`。一方、公開 `models/Wavelet.py` は `configs.period_coeff**10` という**固定 scalar**を使用し、論文式の `α_b` を forward で計算しない。この差は結果の解釈に関わるため明記する。本 block の `pwc_mode="official_fixed"` は公開実装通りの固定値、`pwc_mode="paper_adaptive"` は**WFTBlock 入力の D 次元特徴**から、最初の `pwc_bins` 個の非 DC bin で論文型のエネルギー比を計算する別モード。論文が記す元系列の C 変数で計算する場合と数値同一とは限らない。

## encoder 化した実装

```text
x [B,T,Cin] → window norm → circular Conv1d(Cin→D) + position [B,T,D]
  → WFTBlock(s) (FFT period map + ptwt Morlet CWT + fusion) [B,T,D]
  → LayerNorm [B,T,D] → Linear(D→Cout) [B,T,Cout]
  ＋ raw input x → Linear(Cin→Cout) [B,T,Cout]
  → 任意の head は外側
```

構築時に `seq_len=T` を指定するのは位置 embedding と入力長の整合のため。**予測専用の `T→T+H` 拡張と末尾 H 点の切り出しは、この encoder には入れない。** 各時刻の D 次元特徴をそのまま `Cout` へ射影するため、`D→Cin→Cout` の中間ボトルネックもない。正規化で失われる平均・スケールを下流で利用できるよう、元入力からの学習可能な skip を加える。`Cout<Cin` の場合など、出力次元の制約による情報損失まで防ぐ保証はない。CWT は `ptwt.cwt(..., "morl")` を使う。公式指定の `ptwt==0.1.6` と導入版 `1.0.1` では既定精度が異なるため、後者には `precision=10` を指定する。

## 使い方・学習

```python
from ts_block import WFTNetBlock

block = WFTNetBlock(in_dim=8, out_dim=32, d_model=32,
                    top_k=2, seq_len=T, pwc_mode="official_fixed")
h = block(x)                    # [B,T,8] → [B,T,32]
score = head(h.mean(dim=1))     # head Linear(32,K) → [B,K]
```

[finetuning_example.py](finetuning_example.py) は DataFrame の過去窓と直後 1 行の regression/classification label でこの block と pooling head を学習する。これは一般の encoder 組込例であり、原著の長期予測実験とは異なる。WFTNet は 2D map/CWT の計算が重いので大規模実用前に実データ形状で latency と必要精度を測る。

## FFT 周期 branch を数値例で理解する

`T_e=96` の rFFT は bin index `0..48` を返す。0 は DC（平均）で、公式 `FFT_for_Period` はその重みを 0 にして選択対象から外す。仮に `f=4` が上位 bin なら `p=floor(96/4)=24`。入力の時間軸を 24 点ごとの 4 行に折りたたむと `[B,D,4,24]`。最後の軸は周期内の位置、行軸は周期の繰り返し回数。2D Conv は同じ周期内の変化だけでなく、隣接周期間の変化にも作用できる。`f=7` なら `p=13` で 96 は 13 で割れないため、最後にゼロを 8 点足して `[B,D,8,13]` にし、Conv 後に元の 96 時刻へ crop する。

周波数の**選択**は batch/channel を平均したスペクトルで一度行うが、選択された K 本を**どう足すか**は sample ごとに `softmax(|FFT|)` から計算する。したがって全 sample で同じ Fourier 特徴になるわけではない。ただしバッチ内に互いに無関係な周期が混在すると、トップ K の選択自体が希薄になる。

## CWT branch は FFT と何が違うか

FFT の各 bin は窓全体の周期強度であり、窓の前半だけ強い周期があったかを直接示さない。CWT は尺度 `s` ごとに wavelet を各時刻にずらして相関を計算し、`coeff[b,d,s,t]` を得る。`s` が小さいほど短い変動、大きいほど長い変動へ反応しやすい。ただし尺度を単純に FFT bin の番号と同一視してはいけない。Morlet の中心周波数、離散 sampling、境界 extension で実際の対応が変わる。

本版の [ts_block.py](ts_block.py) は公式と同じ `ptwt.cwt` を呼ぶ。入力 `[B,D,T]` から、8 尺度の係数 `[8,B,D,T]` を取得して `[B,D,8,T]` に並べ替える。公式指定の旧版は内部精度 10、導入版 1.0.1 の既定は 12 なので、後者では `precision=10` を明示した。これにより旧版と同じ CWT 係数を得ることを数値確認した。実行には PyTorch・`einops` に加え `ptwt` と NumPy が必要である。

## PWC: 論文の構想と released forward の区別

論文での sample `b` の重みは、おおむね `α_b = (1/C) Σ_c max_{f≤m}|FFT[b,f,c]|² / Σ_{f≤m}|FFT[b,f,c]|²` という、支配的な単一周期へエネルギーが集中している度合い。`α_b≈1` なら強い大域周期、低いなら局所 branch の比率を増やす。α を 10 乗するので、例えば α=0.8 でも `α^10≈0.107` と Fourier branch はかなり抑えられる。固定 `period_coeff=0.8` を使う公開 forward は各 sample に応じてこの比率を変えない。本版の `paper_adaptive` は paper の意図を隠れ特徴 D 上で実装した別モードで、**公開 checkpoint の完全再現用ではない**。

さらに `top_k=0` なら Fourier branch は存在せず wavelet-only になる。公開 `Wavelet.forward` はその場合だけ内部の未来長 `projection(seq_len+pred_len→pred_len)` を通る。本版では `pred_len=None` としてこの**予測専用射影を使わず**、入力時刻 T 個の特徴を保持する。

## 原著の検証から何が言えるか

論文 TeX の branch ablation では ECL の `H=96` で WFTNet MSE 0.164、Fourier-only 0.168、Wavelet-only 0.196。電力需要の強い周期では Fourier 側が有利だったと読める。しかし ETTh2 の `H=720` では WFTNet 0.430、Fourier-only 0.434、Wavelet-only 0.421 で、**常に融合が最良というわけではない**。論文本文の「consistent」な表現より表を優先して解釈すべき。各数値は原著 forecast 設定であり、ここでの分類/回帰 encoder 性能ではない。

## 実装の対応と推論コスト

| 論文上の部品 | 公開コード | 本 block | 境界 |
| --- | --- | --- | --- |
| FFT top-k period | `FFT_for_Period` | `WFTBlock.forward` | 同じ bin→`T//index` 規則。動的 top-k は勾配対象外 |
| periodic 2D feature | `period_conv` | `period_conv` | odd kernel 群と平均を保持。kernel 数は設定 |
| CWT local map | `ptwt.cwt` | `MorletCWT` | 同じ 8 尺度・旧版と同じ計算精度 10 で係数一致 |
| scale compression | depthwise `(8,1)` | `scale_conv` | 同じ軸と kernel shape |
| blend | fixed `period_coeff**10` | `official_fixed`/`paper_adaptive` | 二つの定義を切り分ける |
| 同長出力 | `T→T+H` と末尾 H 点 | T 個の WFT 特徴を保持して `D→Cout`、元入力 skip を加算 | 予測用拡張・切り出しは除外した汎用 encoder adapter |

FFT は概ね `O(B D T log T)`、K 回の 2D 畳み込みは period map を合計 `K*T` 要素で処理する。CWT は 8 尺度の畳み込みで、その計算量は最大 wavelet filter 長に依存する。尺度を大きくすると計算量も増える。単に「FFT は速いから WFTNet 全体も速い」とは言えず、実際の `B,T,D,K,scale` でベンチマークする。

## 下流学習・推論の擬似コード

~~~text
train:  x = 過去 T 点の特徴 [B,T,Cin]
        h = WFTNetBlock(x) [B,T,Cout]
        pooled = mean_time(h) [B,Cout]
        y_hat = task_head(pooled) [B,K]
        GT = 次行の回帰値 [B,K] / class [B]
        loss = MSE / CrossEntropy → backward → optimizer.step()
valid:  時刻分割の後半 label のみで loss 計測
infer:  過去 T 点だけで同じ encoder/head を呼ぶ
~~~

原著の decoder に過去の正解をコピーする teacher forcing は無い。窓内の future GT を入力に入れていないことが漏洩防止の最初の確認点となる。このサンプルは未来 1 行の head だが、系列全体の分類なら別 label、各時刻の anomaly score なら pooling を外した head を定義する。

## 軸を取り違えないための形状表

| Tensor | 形状 | 軸の意味と変換 |
| --- | --- | --- |
| 埋込系列 | `[B,T,D]` | T は観測窓の時刻、D は隠れ特徴 |
| FFT 係数 | `[B,F,D]` | F は非負周波数 bin、`F=floor(T/2)+1` |
| top-k weights | `[B,K]` | K 本の選択 bin に対するサンプル別 softmax |
| period grid k | `[B,D,R_k,p_k]` | `p_k=floor(T/f_k)`、`R_k=ceil(T/p_k)`。最後は pad 可 |
| 期間 Conv の出力 | `[B,D,R_k,p_k]` | padding 付き 2D 畳み込みは grid shape を維持 |
| CWT map | `[B,D,8,T]` | 第 3 軸が wavelet 尺度で、期間 grid の行軸とは別 |
| scale conv 出力 | `[B,D,1,T]` | 8 尺度を一つへ縮約。時間長 T は維持 |
| 融合表現 | `[B,T,D]` | 残差後に LayerNorm、最後だけ `D→Cout` |

`period_coeff` と `pwc_bins` は**別物**。前者は公開コードの固定 blending hyperparameter、後者は本版 `paper_adaptive` で何個の FFT bin をエネルギー比に含めるかの設定。`top_k` は 2D period branch で何本の周期を抽出するか。これらをすべて「周期数」と呼ぶと設定が混同される。

## 実用での使い分けと限界

窓全体にほぼ一定の周期があり同じ時間帯で強い場合、FFT branch の `top_k` を少数残すことに意味がある。周期が窓の途中で変化したり間欠的に現れるなら CWT branch が補完し得る。どちらの特徴が必要かは downstream の target で変わるので、`top_k=0`（wavelet-only）、`period_coeff=1` 相当（Fourier を強める）、adaptive blend を同じ分割で比較する。下流評価のために validation から `period_coeff` を探索することはできるが、test 結果で選ぶと漏洩する。

この block の 2D Conv は grid の後続行や後続列を見る非因果処理。`h[b,t]` をオンライン t 時点の判断に使うなら、入力 window は t までで切り、未来の `x[t+1:]` を含めない。`batch` に違う長さの系列を padding した場合、padding も FFT/CWT に反映されるため、単に attention mask を渡すだけでは解決しない。系列ごとに同長 window を作るか、pad の影響を明示的に評価する。

学習時の CWT は固定 wavelet による `ptwt.cwt` で、学習されるのは後段 Inception/scale convolution の重み。`pwc_mode="paper_adaptive"` はスペクトルのエネルギーから比率を**forward ごとに算出**するが、FFT top-k の離散 index 選択そのものは微分可能な連続パラメータではない。FFT が算出されることと、全ての周期選択が end-to-end で滑らかに学習されることは別。

## 学習 driver まで追った objective

公式 `exp/exp_main.py` は `nn.MSELoss()` を選び、`Model.forward` が返した最後の H 点と `batch_y` の未来 H 点を比較する。DataLoader は `label_len` を使って decoder 用系列を用意するが、公開 `models/Wavelet.py` の `forecast` は `x_dec` の値を予測計算に使わない。したがって「訓練時に未来の真値を decoder へ teacher forcing している」という説明は誤り。`x_mark_enc` は embedding の時刻特徴として使われる可能性があるので、値と時刻 marker も分けて読む。

公開 `Model.forecast` は入力窓で平均・分散を算出し、埋込前に正規化する。WFTBlock に入る前に time Linear で L を L+H に引き伸ばすため、FFT top-k/CWT が見るのは**生入力窓 X ではなく、すでに learned time extension された D 次元特徴**。この順序は本 block の `x:[B,T,Cin]` の FFT が埋込後に計算されることと共通だが、未来長への extension は省かれている。学習 loss は最終の `C` 変数の未来値にのみ掛かり、FFT 係数や CWT map に別の GT は無い。

`period_coeff` の適切な値は `top_k=0` では無関係（global branch が存在しない）。`top_k>0` の場合でも公開 script は dataset ごとに `top_k`, `wavelet_scale`, `period_coeff` を設定する。これらを変えずに異なる系列へ流用して論文の精度を期待してはいけない。下流用途ではまず同じ validation 分割で branch ablation を行うべきである。

公開 scripts はデータセットに応じて `top_k=0` と 5 の両方を使い、wavelet scale も変えている。たとえば Weather 用 script には `top_k=0` の設定があり、原著の「FFT と wavelet の融合」をどの公開実験でも同じ強さで使ったとは言えない。モデル名だけから両 branch が必ず動くと判断せず、再現したい設定の script と forward 分岐を合わせて読む。ここでの `WFTNetBlock` の既定 `top_k=3` は汎用利用の開始点であり、特定 dataset の公式最適値ではない。

この block の `Cout` を大きくするのは最終 Linear の幅を増やすだけで、FFT/CWT/Inception の内部幅 D は `d_model` で決まる。容量を増やす必要があるなら `d_model`, `d_ff`, `layers`, `num_kernels` を区別して調整する。`layers` を増やすと FFT と CWT を毎層計算するため、単純な線形 head の幅変更よりレイテンシ増が大きい。

## 原著 forecast の学習・推論を一行ずつ

~~~text
train input: history X [B,L,C], future GT Y [B,H,C]
normalize X with its own μ,σ [B,1,C]
embed X, then learned time extension L→L+H [B,L+H,D]
for each WFTBlock: FFT top-k maps + CWT maps → Inception → blend
project D→C and inverse normalize [B,L+H,C]
prediction = final H points [B,H,C]
loss = mean((prediction - Y)^2) scalar; backpropagate
infer input: history X only; same steps without Y/loss
infer output: final H points [B,H,C]
~~~

未来 GT は train の loss にのみ登場し、FFT top-k の入力や CWT map を作る入力ではない。時刻 marker が利用可能なら embedding に渡すが、値の未来観測とは別物。この区別を保つと、「学習時に未来の正解で 2D map を作る」という誤解を避けられる。

## 一次資料・再現境界

原論文: [WFTNet](https://arxiv.org/abs/2309.11319)。公式実装: [Hank0626/WFTNet](https://github.com/Hank0626/WFTNet)。照合箇所: 論文 `Template.tex`、公開 `models/Wavelet.py`, `layers/Conv_Blocks.py`, `layers/Embed.py`。CWT の precision 設定は公式指定の `ptwt==0.1.6` に合わせる。汎用 encoder の出力は未来予測用の時間拡張・切り出しを除いたため、公式 `Model.forward` の最終出力との数値一致対象ではない。
