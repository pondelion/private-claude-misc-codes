# FITS — 複素周波数補間を系列 encoder として使う

## 概要

FITS (*Frequency Interpolation Time Series analysis baseline*) は、時系列を時間領域で大きな Transformer に通す代わりに、低周波の複素スペクトルを **1 層の複素 Linear** で変換するモデルである。原著は主に予測と再構成・異常検知を扱う。FFT の複素数が振幅と位相を表すため、実数の大規模ネットワークを用いずに周波数間の結合と位相移動を学習できる。「10k パラメータ」は論文の代表設定での規模であり、全設定で不変の保証ではない。

原著の核は入力長 `L` から出力長 `L+H` への *周波数補間* である。一方、ここで再利用したいのは予測 head ではなく `[B,T,Cin] -> [B,T,Cout]` の encoder。したがって [ts_block.py](ts_block.py) では長さ比 `η=1` として同長の複素周波数 mixing を行い、GT や horizon を一切持たない。これは FITS の核を利用した encoder 化であって、原著の予測モデルそのものではない。

## 新規性と既存手法との差分

| 比較対象 | FITS との差 |
| --- | --- |
| 時間領域の線形予測 | rFFT 後の**複素**係数を線形写像し、振幅と位相を同時に変える。 |
| Fourier 成分をそのまま選択・平均 | `cut_freq` 個の低周波を互いに混合する学習可能な複素行列を置く。 |
| 長系列 Transformer | self-attention を用いず、入力の低周波数で主な学習パラメータ数が決まる。ただし FFT と全長スペクトルのメモリは必要。 |

高周波を捨てることは計算削減とノイズ除去の仮定であり、スパイクや短時間の異常が重要なタスクでは情報損失にもなる。`residual=True` は入力特徴を残す実用上のオプションだが、原著の経路とは異なる。

## モデルアーキテクチャ全体像（原著の `L→L+H`）

FITS の「補間」は時間領域の中間点を単純に線形補間することではない。観測窓の**複素 FFT 係数** `K` 個を、長くした系列に対応する `floor(ηK)` 個の複素係数へ learned Linear で写す。`η=(L+H)/L` として `L+H` 点を一度に再構成し、そこから最後の H 点を forecast とみなす。

```text
履歴 X [B,L,C] ────────────────────────────────────────────┐
    │                                                      │
    ├─ sample・channel ごとの mean μ [B,1,C]              │
    └─ sample・channel ごとの std  σ [B,1,C]              │
    │                                                      │
    ▼ normalize                                            │
Z=(X-μ)/σ [B,L,C]                                          │
    │ rFFT along L (real→complex)                           │
    ▼                                                      │
S [B,floor(L/2)+1,C] complex                               │
    │ low-pass: bin 0..K-1 だけ残す                         │
    ▼                                                      │
S_low [B,K,C] complex                                      │
    │ transpose frequency as last axis [B,C,K]              │
    │ shared complex Linear(K→floor(ηK))                    │
    ▼                                                      │
S_mix [B,floor(ηK),C] complex                              │
    │ 高周波 bin をゼロ詰め                                 │
    ▼                                                      │
S_full [B,floor((L+H)/2)+1,C] complex                      │
    │ irFFT to length L+H / energy correction ×η           │
    ▼                                                      │
Z_hat [B,L+H,C] ── inverse norm with μ,σ                   │
    ▼                                                      │
output [B,L+H,C] ──┬─ first L: backcast [B,L,C] （生成する）
                  └─ final H: forecast [B,H,C]
                        │
                        └─ official train: MSE with future GT [B,H,C]
paper: concat(X,future GT) [B,L+H,C] を全長教師にする案も説明
```

論文は backcast と forecast の両方へ教師を与えられる利点を説明する。しかし提供された公開 `exp/exp_main.py` の train/validation は**最後の H 点だけ**を slice して実測 future との MSE を計算する。これは論文の説明とコードの重要な差。FITS の model forward 自体は GT を受け取らず、`[B,L+H,C]` を生成する。`nn.Linear` の複素重みは保持した K bin 間を密に結合する。

## 原著のモデルと学習 objective

記号: `B` batch、`L` 観測長、`H` 予測 horizon、`C` 変数数、`F=floor(L/2)+1` rFFT bin 数、`K=cut_freq`。

```text
history x                        [B,L,C]
  └─ 入力窓ごとの平均 μ, 標準偏差 σ   各 [B,1,C]
z=(x-μ)/σ                        [B,L,C]
rFFT(z, time)                    [B,F,C] complex
LPF: low = spectrum[:K]          [B,K,C] complex
complex Linear(K -> floor(ηK))   [B,floor(ηK),C] complex
zero-pad to output spectrum      [B,floor((L+H)/2)+1,C]
irFFT + η 倍の長さ補正            [B,L+H,C]
逆正規化                          [B,L+H,C]
論文上の全長教師候補: concat(history,future) [B,L+H,C]
公開 train の GT: future の実測   [B,H,C]
公開 train loss: MSE(output[:,-H:],GT) [scalar]
公開 valid/test: future 区間のみ  [B,H,C]
```

長さ比 `η=(L+H)/L` で周波数領域の出力 bin 数も伸ばす。`nn.Linear(K,floor(ηK), dtype=complex)` は各チャンネルで重みを共有する設定が基本で、公開コードにはチャンネル別重みの `individual=True` もある。原著の再構成タスクでは、元系列を downsample した観測を入力し、元系列を GT として再構成損失を置く。`FITS.py` の forward は予測と低周波由来成分の tuple を返すが、実際の loss は実験側の設定で決まる。

## 本実装のアーキテクチャと形状

```text
x [B,T,Cin] → instance normalize [B,T,Cin]
             → rFFT [B,floor(T/2)+1,Cin]
             → first K bins [B,K,Cin]
             → shared / per-channel complex Linear(K→K) [B,K,Cin]
             → high-bin zero fill → irFFT(n=T) [B,T,Cin]
             → inverse normalize [B,T,Cin]
             → channel projection [B,T,Cout] = h
下流だけ: h.mean(time) [B,Cout] → 回帰/分類 head [B,出力数]
```

`K <= T/2+1`、**偶数 T** が必要。公式の `irfft` は `n` を指定せず、奇数 T の元長には戻らない。`[B,T,Cout]` 契約と公式出力を同時に守れないため、本 block は奇数 T を明示的に拒否する。分散推定は公式と同じ不偏分散を既定にした。`output` は encoder 用の `Cin→Cout` 写像で、`Cin=Cout` の初期値は単位行列。原著の `η>1` 周波数補間は未来長を作る処理であり、同長 block では `η=1` とする。`residual=False` では捨てた高周波を復元できない。

## 使い方・学習例

```python
from ts_block import FITSBlock

block = FITSBlock(in_dim=8, out_dim=32, cut_freq=16)
h = block(x)                   # x:[B,T,8], T>=30 → h:[B,T,32]
pooled = h.mean(dim=1)         # [B,32]
logits_or_values = head(pooled)  # head: Linear(32,K) → [B,K]
```

`head` と MSE/CE は用途側で定義する。[finetuning_example.py](finetuning_example.py) は DataFrame の `series_id,timestamp` と特徴列・目的列から「直後 1 行の値/クラス」を予測する**例**である。時刻順 80/20 分割、train 側のみで特徴を標準化し、過去 `[B,T,Cin]`、GT `[B,K]` または `[B]`、平均 pooling した特徴に head を学習させる。原著の horizon 予測を再現するスクリプトではない。

## 照合と限界

原著 TeX `methods.tex`, `experiment.tex` と公式 `models/FITS.py` の処理順を照合した。偶数 T、`η=1`、同じ複素重み、追加射影が恒等の場合、低周波切り出しから最終復元まで公式 forward と一致する。任意の `residual=True` だけは原著にない追加設定。周期が窓長内で安定する系列に向くが、非周期的突発値や低周波以外が主情報なら別 block も比較する。

## なぜ複素 Linear が重要か

実 FFT 係数 `S[f,c]=a+ib` は絶対値 `sqrt(a²+b²)` と角度 `atan2(b,a)` を持つ。実数の magnitude だけを調整すると振幅は変えられても、周期波形がどの時刻でピークを迎えるかという位相を表現しにくい。複素重み `w=u+iv` と掛けると `(u+iv)(a+ib)` により振幅の倍率と位相回転を同時に学習できる。FITS の `Linear(K,K')` はこの複素積を K bin から K' bin へ和として集める。原著はこの周波数補間を、長くなった系列により周波数解像度が変わることへ対応する操作として位置付ける。

例: `L=96,H=24` なら `η=120/96=1.25`。`K=16` の入力低周波から、複素 Linear は `floor(1.25*16)=20` bin を出す。これを長さ 120 の rFFT 格納長 `floor(120/2)+1=61` bin までゼロ埋めして、irFFT で 120 点へ戻す。`K=16` は**16 個の learned 周波数 token**ではなく先頭 16 個の FFT bin。周波数番号と絶対 Hz の対応は sampling interval に依存するので、データ取得間隔が変われば同じ K の意味も変わる。

`torch.fft.rfft` の DC bin `f=0` は平均に対応する。窓平均を引くとこの成分は理想的には 0 になり、複素 Linear が大きな DC に容量を使うのを避ける。Linear には bias があるため、変換後の DC は厳密に 0 に固定されるわけではない。最終的に元の μ,σ を戻すので、モデルが生スケールの違いを一切失うわけでもない。

## LPF の効果と失敗条件

`S[:,K:]=0` は周波数を低い方から K 個残す hard low-pass filter。雑音が主に高周波で有益な信号が低周波なら、学習する複素行列は K×K' と小さくなり、入力周波数の多くを無視してもよい。しかし急峻なジャンプ、局所スパイク、短時間の異常、価格の急変が GT に重要なら、LPF がまさにその情報を捨てる。LPF は「ノイズを自動判定する」適応機構ではない。カットオフ K を上げれば保持情報は増えるが、複素 Linear のパラメータ・演算も増える。個別チャンネル行列を使う `individual=True` ではさらに C 倍となる。

本 block の `residual=True` は正規化済み入力を irFFT 後へ足して高周波情報を別経路で保持する。これは**原著ではない追加**であり、LPF の効果とトレードオフになる。実用では `residual=False/True` を下流 validation で比較し、事前に優越を決めつけない。

## 予測・再構成・同長 encoder の objective を分ける

| 使用形態 | 入力 | モデル出力 | GT/loss |
| --- | --- | --- | --- |
| 原著 forecast の構想 | `[B,L,C]` | `[B,L+H,C]` | 全長 backcast+forecast も利用可能と論文が説明 |
| 公開 forecast train | `[B,L,C]` | `[B,L+H,C]` から末尾 H | **未来の実測 `[B,H,C]` だけ**との MSE |
| 原著 anomaly/reconstruction | 元窓を N 分の 1 へ等間隔 downsample | 元窓長へ upsample/reconstruct | 元窓との再構成誤差。閾値超過を anomaly とする |
| 本 survey encoder | `[B,T,Cin]` | 特徴 `[B,T,Cout]` | **block は GT/loss を決めない**。外部 head に任せる |

異常検知の論文実験では window 200 を 4 倍 downsample して再構成し、点ごとの誤差から閾値を選ぶ。報告値は SMD/SWaT で高い F1、SMAP/MSL では低めであり、周波数仮定が合わない信号にも万能という結論ではない。閾値を validation の F1 最大化で選ぶ手法は、ラベルがない真の教師なし配備とは区別する必要がある。

## コードとの対応と検証範囲

| 処理 | 公開 `models/FITS.py` | 本 `FITSBlock` | 差 |
| --- | --- | --- | --- |
| RIN | mean/var を時刻方向で計算 | 同じ式、`variance_unbiased` 切替 | 既定では短窓のため biased var |
| LPF | `rfft` の `K:` をゼロ、先頭 K を取る | 先頭 K のみを線形へ渡す | 保持 bin は同じ |
| 周波数 Linear | complex `K→floor(ηK)` | complex `K→K` | 同長 encoder では η=1 |
| rFFT 出力まで zero-pad | `[B,F_out,C]` | `[B,F_in,C]` | 目標長が違う |
| inverse FFT | `irfft` + length ratio | 偶数 T の `irfft(n=T)` | 公式と同じ T 点を返す。奇数 T は拒否 |
| 最終出力 | 予測系列と低周波成分の tuple | feature tensor のみ | task-neutral API |

`T=16,η=1,Cin=Cout`、同じ複素重み、出力射影恒等で、公開 `Model.forward` の第一出力とこの block の出力が一致した。shared/individual の両方を確認した。奇数 T は公式出力長と同長 API が両立しないため入力時に拒否する。

## 汎用 task head の学習・推論例

~~~text
train:  x = 過去 T 点 [B,T,Cin]
        h = FITSBlock(x) [B,T,Cout]
        pooled = mean_time(h) [B,Cout]
        y_hat = task_head(pooled) [B,K]
        GT = 次行の回帰値 [B,K] / class index [B]
        loss = MSE / CrossEntropy → backward → optimizer.step()
valid:  後半時刻の label のみを評価、scaler は先頭 80% で fit
infer:  同じ feature 順・scaler、かつ FFT bin が K 個以上ある長さ T
~~~

`T` が変わっても `K<=floor(T/2)+1` なら同一重みを使えるが、FFT bin の実時間上の周期は窓長と sampling 間隔で変わる。「形状が動く」と「同じ物理周期を学習している」は別問題。バッチ内に系列ごと異なる sampling interval があるなら、事前に resample するなど入力定義を揃える。

## 実装前に決めるパラメータと軸の意味

| 値 | 原著 forecast | 本 block | 決め方・注意 |
| --- | --- | --- | --- |
| `L`/`T` | 履歴長 L | 入力窓長 T | rFFT 周波数 bin の分解能を決める |
| `H` | 未来点数 | 無い | 下流 head が別に定義する |
| `K=cut_freq` | 残す低周波 bin 数 | 同じ | `K<=floor(T/2)+1` 必須 |
| `η` | `(L+H)/L` | 1 | 本版は時間長を変えない |
| `C`/`Cin` | 観測変数数 | 入力 feature 数 | FFT は各変数の時刻方向に独立に計算 |
| `Cout` | 通常 C | 出力 feature 幅 | 本版最後のチャンネル Linear で指定 |

周波数の Linear がチャンネル間を混ぜると誤解しやすい。`individual=False` なら同じ `K×K` の複素行列を**各入力チャンネルへ独立に適用**するだけで、チャンネル間の相関は最後の `Cin→Cout` 射影まで直接混ざらない。`individual=True` は C 個の別々の行列を持つが、それでも各行列は自分のチャンネルの周波数だけを見る。異なる特徴間の時間遅れ関係を中間層で表現したいなら、出力 Linear 一層だけで十分かを評価する。

## 実務上の性能と数値上の注意

複素 Linear の重みは PyTorch の complex dtype なので、実数 Conv/Linear と同じ AMP・量子化・高速化カーネルが常に使えるわけではない。パラメータ数だけから wall-clock の速さを推測せず、実際の device/shape で測る。入力が全部ほぼ定数なら `σ≈sqrt(eps)` と小さくなるが、正規化後の値は有限で、逆正規化で μ を戻す。本版は公式と同じ不偏分散を既定にし、T は偶数かつ T>1 に限定する。

`irfft` は real 系列を戻すが、周波数 0 と偶数長の Nyquist bin には実数制約がある。complex Linear でこれらの位置に虚部が生じても、逆 transform は real signal として解釈できる成分のみを使う。したがって「全複素出力係数が独立に時間信号へ反映される」わけではない。奇数 T では公式 `irfft` が T−1 点を返すため、本版はその入力を拒否する。

## FITS を encoder として比較するときの ablation

同じ下流 head・同じ時刻分割で `cut_freq` を数段階変え、高周波を切ることが有利かを確認する。`K=1` は正規化後の DC しか残らず、RIN が mean を引くため、実質的な時間変動をほとんど表現できない。`K` を Nyquist まで増やせば high-frequency 情報は入力として残るが、学習する複素行列は大きくなり、FITS の軽量性が薄れる。`residual=True` は低周波 branch とは別に元信号の情報を保持するので、スパイク検出のようなタスクでは重要な比較対象になる。ただし residual path が強すぎると FFT branch を使わずに下流 head が解く可能性もあり、学習後に `freq_upsampler` の変化量や branch ablation を見る。

出力 `h:[B,T,Cout]` は各時刻に対応しているが、周波数変換と全 K bin の dense mixing は**窓全体を参照する非因果処理**。`h[:,t]` が時刻 t までの観測だけに基づくとは限らない。オンライン判定なら入力窓を t で終える。価格系列などで翌日の情報を含む窓から今日の label を評価するのは、block の問題ではなくデータ切り出しの漏洩である。

公式原著はおよそ 10k パラメータという訴求をしているが、実際の複素 Linear のパラメータ数は shared でおおむね `2*K*K'` 個の実数相当の weight と `2*K'` 個の bias、individual ならそれが C 倍。ここに本版の `Cin*Cout` の実数 projection も加わる。**10k はモデル名に付く定数ではなく設定から計算される値**である。

この survey の同長複素 Linear は `K→K` なので、原著が強調する**周波数解像度を上げる補間**そのものは行わない。使っているのは周波数の低域選択・複素 mixing・時間領域への再合成という backbone 部分である。もし研究目的が原著 FITS の予測性能再現なら、`η>1`、未来長に対応する zero-pad、未来だけの公式 train loss と evaluation を用いる別モデルが必要。本 block を `FITS` と呼んでいても task adapter と原著予測器の差は消えない。

## 原著 forward と公式 loss の擬似コード

~~~text
train input: history X [B,L,C], future Y [B,H,C]
μ,σ = window statistics(X) [B,1,C]
low = rFFT((X-μ)/σ)[:, :K] [B,K,C] complex
mixed = complex Linear(K→floor(ηK))(low) complex
full = zero-pad mixed to floor((L+H)/2)+1 bins
generated = η*irFFT(full) [B,L+H,C]
generated = generated*σ+μ [B,L+H,C]
official loss = MSE(generated[:, -H:], Y) scalar
infer: X のみから generated[:, -H:] [B,H,C] を得る
adapter: η=1 として同長変換し [B,T,Cout] を返す
~~~

`irFFT` の係数だけを GT と比べるのではない。最終的に時間領域へ戻した値へ MSE を掛け、勾配は irFFT と複素 Linear を通って戻る。公開 forecast driver の `dec_inp` は他モデルと共通の API 用に作られるが、FITS の `Model.forward(x)` は履歴 x だけを受け取る。

## 一次資料

原論文: [FITS: Modeling Time Series with 10k Parameters (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/701251e1db4a2e4dd2ef23f5265d5936-Abstract-Conference.html)。公式実装: [VEWOXIC/FITS](https://github.com/VEWOXIC/FITS)。照合対象: 論文 `methods.tex`, `experiment.tex`, `appendix.tex` と公開 `models/FITS.py`。`η=1,T=16` で公式複素 Linear をコピーし、shared/individual 両経路の forward 出力一致を確認した。**原著の `η>1` 補間は同長 block には存在しない。** 奇数 T は公式出力長が T−1 となるため受け付けない。
