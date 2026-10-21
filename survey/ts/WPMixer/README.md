# WPMixer — 小波帯域ごとに独立して patch/mixing する encoder

## 概要

WPMixer (*Wavelet Patch Mixer*) は、長期予測に対して「wavelet 分解で異なる時間分解能の係数を分離し、**各帯域を別々の patch mixer** で処理してから逆変換する」モデル。主張の中心は wavelet だけでも MLP だけでもなく、近傍の局所情報を patch にまとめ、patch 間の広域相互作用と patch embedding 内の特徴相互作用を帯域別に学習する構成にある。論文は予測精度・計算量を比較し、学習は Smooth L1 loss を使用する。

本ディレクトリの [ts_block.py](ts_block.py) は、予測 `L→H` に固定された原著の出力 head を **帯域別の同長 `L_i→L_i` head** に変え、`[B,T,Cin]→[B,T,Cout]` の汎用 encoder として使う。任意の回帰・分類 head は外側に置く。

## 新規性と既存法との差

TimeMixer は downsample された系列の trend/season を尺度間で mixing する。一方 WPMixer は DWT の近似・詳細係数を明示的に分離し、**尺度を跨ぐ mixer ではなく**各係数列専用の branch を持つ。PatchTST の patching を wavelet 係数それぞれに適用するが、Transformer attention ではなく token/embedding の二種類の MLP を用いる。これにより、急変を含む高周波 detail を移動平均の trend/season 分解で均してしまう問題を避ける設計である。ただし小波の種類・分解段数・patch 長はデータ依存であり、自動最適化されない。

## モデルアーキテクチャ全体像（原著の `L→H`）

`DWT` は `m` 回の二分割で `A_m`（最粗近似）と各尺度の detail を出す。公式 `pytorch_wavelets` の返却順は **`A_m,D_1,...,D_m`（detail は細かい尺度から）**で、branch の重みもこの順に対応する。重要なのは各係数系列に**別の ResolutionBranch**があり、同じパラメータを全帯域へ機械的に共有しないこと。図では 2 段に具体化した。

```text
過去 X [B,L,C] ── global RevIN norm ── [B,C,L]
                                 │
                                 ▼
                          level-1 DWT
                       ┌──────┴──────┐
                    A1 [B,C,L1]   D1 [B,C,L1] ────────┐
                       │                              │
                    level-2 DWT                        │
                   ┌───┴────┐                         │
               A2 [B,C,L2] D2 [B,C,L2]               │
                   │           │                       │
                   ▼           ▼                       ▼
              Branch A2     Branch D2             Branch D1
                [B,C,L2]     [B,C,L2]             [B,C,L1]
                   │           │                       │
             per-band RevIN（各 branch に独立の統計）     │
                   │           │                       │
             end-repeat pad + unfold [B,C,N_i,P]       │
                   │ BN2d + Linear(P,d)                │
                   ▼                                   │
              Mixer 1 [B,C,N_i,d]                      │
                   │                                   │
              + Mixer 2(residual) [B,C,N_i,d]          │
                   │ BN2d → flatten(N_i*d)             │
                   │ Linear(N_i*d,H_i)                  │
                   ▼                                   ▼
              A2_hat [B,C,H2]  D2_hat [B,C,H2]   D1_hat [B,C,H1]
                   └─────────┬─────────────┬───────────┘
                             ▼             │
                        inverse DWT level 2│
                             │             │
                        reconstructed A1   │
                             └──────┬──────┘
                                    ▼
                             inverse DWT level 1
                                    │ [B,C,H]
                                    ▼ global inverse RevIN
                             prediction [B,H,C]
                                    │ compare with future GT [B,H,C]
                                    ▼ SmoothL1 loss [scalar]
```

各 branch の Mixer 内部はさらに二つの軸がある。`[B,C,N_i,d]` の `N_i` は patch 数、`d` は patch embedding 幅。token MLP は **N_i 軸**、embedding MLP は **d 軸**を処理し、チャンネル C を直接 MLP で混ぜるわけではない。`H_i` は未来 H 点を wavelet 分解したときの i 番目の係数長で、すべて H と同じではない。

## 原著の予測 objective

`B`: batch、`L`: 履歴長、`H`: 予測長、`C`: チャンネル数、`m`: 分解段数。`L_j` は j 段目係数長。原著では分解した**未来長 H**の dummy 系列から各 branch の予測係数長 `H_j` を定める。

```text
history X [B,L,C] → global RevIN [B,L,C]
  → DWT(m levels) = {A_m,D_1,...,D_m}（branch への入力順）
       A_m,D_m [B,C,L_m], D_1 [B,C,L_1]
  → branch i を帯域ごとに別パラメータで処理
       per-band RevIN [B,C,L_i]
       end-repeat pad + unfold [B,C,N_i,P]
       BN + Linear(P→d) [B,C,N_i,d]
       Mixer 1 → Mixer 2+residual → BN [B,C,N_i,d]
       flatten(N_i*d) + Linear(N_i*d→H_i) [B,C,H_i]
       per-band inverse RevIN [B,C,H_i]
  → inverse DWT [B,H,C] → global inverse RevIN [B,H,C]
  → GT future [B,H,C] と SmoothL1Loss → scalar
```

1 個の Mixer の内部は `BN → [B,d,C,N_i] → token MLP(N_i→t_f N_i→N_i) → [B,C,N_i,d] → BN → residual embedding MLP(d→d_f d→d)`。token MLP は patch index 方向を、embedding MLP は各 patch の隠れ次元を混ぜる。チャンネル方向を直接跨ぐ MLP ではない。公式実装では同一解像度内の 2 個の Mixer は独立パラメータ。

## 本ブロックの形状図

```text
x [B,T,Cin] → global instance norm → [B,Cin,T]
  → Haar DWT = A_m [B,Cin,L_m] + D_1...D_m [B,Cin,L_j]
  → 各 band 専用 ResolutionBranch: [B,Cin,L_j]→[B,Cin,L_j]
  → Haar inverse DWT [B,Cin,T]
  → inverse norm [B,T,Cin]
  → Linear(Cin→Cout) → h [B,T,Cout]
下流: h.mean(time) [B,Cout] → 任意の regression/classification head
```

`L_j=ceil(L_(j-1)/2)`。奇数長の DWT は公式既定の `zero` 境界に合わせて末尾へ 0 を補う。逆変換では中間 low が detail より長い場合だけ切り、最後に復元系列の**末尾 T 点**を取る。`N_i=floor((max(L_i,P)+S-P)/S)+1`（`P`: patch_len、`S`: stride）。各 band は**独立の learnable affine を持つ RevIN**で正規化し、head の後に affine と統計を逆変換する。head `N_i*d→L_i` は **特徴抽出用の同長変換**であって、未来 GT を block に入力しない。

## 実装差分・注意

公式コードの wavelet backend は `pytorch_wavelets` で、論文は Daubechies・Symlet・Coiflet・Biorthogonal も探索する。本実装は外部 wavelet package に依存せず自動微分できる **Haar/db1 のみ**を実装し、それ以外は黙って別 wavelet に置換せず例外を出す。Haar の係数式は `A=(even+odd)/√2`, `D=(even-odd)/√2`、逆式は `even=(A+D)/√2`, `odd=(A-D)/√2`。Haar/db1 については公式同梱版・公開版 `pytorch_wavelets` の係数と境界処理が偶数・奇数長で一致することを確認した。原著の `H_i` head を `L_i` に変更し、`Cin→Cout` 射影を加えたため、原著 checkpoint を直接読み込むモデルではない。BN は小 batch では統計が不安定になり得るので、検証時は必ず `eval()` にする。

## 使い方と DataFrame 学習

```python
from ts_block import WPMixerBlock

block = WPMixerBlock(in_dim=12, out_dim=64, seq_len=96,
                     levels=2, patch_len=8, patch_stride=4)
h = block(x)              # [B,96,12] → [B,96,64]
prediction = head(h.mean(dim=1))  # head Linear(64,K) → [B,K]
```

[finetuning_example.py](finetuning_example.py) は DataFrame の `series_id,timestamp` でソート・時系列分割し、次行の regression/classification GT を使って pooling head と encoder を一緒に学習する。これは原著の horizon 予測 objective ではなく、オフ・ザ・シェルフ encoder の組込例。特徴の標準化は train 行だけで fit する。

## なぜ wavelet と patch を併用するのか

長い原系列をいきなり大きな MLP で混ぜると、短い突発変動と長期の緩やかな変化が同じ時間解像度に重なる。DWT は連続的な low/high filter と間引きにより、`A_m` に低周波の大局構造、`D_j` に各段の局所変化を分ける。ここで得た係数をそのまま出力せず、各係数列を patch に切る。patch は局所の相関を `P` 点ごとにまとめ、`N_i` 個の patch を token MLP で混ぜることで、係数列全体の長距離相互作用を実現する。最後に逆 DWT を用いるので、各帯域の変換結果は元の時間軸へ戻る。

例として `T=16,m=2` の Haar なら、`A_1,D_1` は各 8 点、さらに `A_2,D_2` は各 4 点。branch への入力順は `A_2,D_1,D_2` であり、`A_1` を別 branch として加えると情報が重複する。`P=4,S=2` なら末尾を 2 点複製して `A_2` の patch は 2 個、`D_1` の patch は 4 個となる。実際の公式 wavelet backend は wavelet filter 長や境界モードで係数長が変わるため、論文も dummy 変換で `L_i,H_i` を取得する。Haar 携帯版は自分の padding 規則からそれを計算する。

## Mixer のオペレーションを一段ずつ

入力 patch `P_i:[B,C,N_i,P]` に BatchNorm2d(C) をかけ、共有 `Linear(P,d)` で `E_i:[B,C,N_i,d]` にする。この Linear は各チャンネル・各 patch に同じパラメータを使うが、A/D 帯域ごとには別 branch なので重みは別。Mixer の token path は `[B,C,N_i,d]→[B,d,C,N_i]` に並べ替え、最後の patch 軸を `N_i→t_f N_i→N_i` と変換する。次に `[B,C,N_i,d]` に戻し BatchNorm2d を通す。embedding path は最後の `d` 軸だけを `d→d_f d→d` と変換して residual を足す。

これを 2 個繰り返すが、公開 `ResolutionBranch` の residual は**2 番目 Mixer の前後**にある: `z=Mixer1(E)`, `out=BN(z+Mixer2(z))`。各 Mixer の embedding MLP にも内部 residual がある。一方 token MLP の手前に `x+...` を入れる構造ではない。ここを単に「MLP-Mixer 2 層」と要約すると skip の位置が失われる。

公式 `ResolutionBranch` の class 定義を読み出し、同じ重み・非自明な RevIN affine で本実装と照合した。band 入力 `[B,C,L_i]` からの出力が一致する。さらに、実際の `pytorch_wavelets` を使う公式 `WPMixerCore` と本版の `T=H=32` の同長経路で、RevIN→全 branch→逆変換までの出力一致を確認した。Haar/db1 の DWT/iDWT 係数も公開版と公式同梱版で直接照合済みである。

## 予測 head と同長 head の差を数式で

原著 branch i の head は `Linear(N_i*d,H_i)`、ここで `H_i` は未来 H 点の wavelet 係数長。したがって原著の出力は GT future と直接比べられる `[B,H,C]`。同長 encoder で同じ head をそのまま使うと出力 T が保持されず、回帰/classification encoder に差し込めない。本版は `Linear(N_i*d,L_i)` に変更して、入力 band と同じ係数長を返す。逆 DWT の結果は `[B,T,Cin]`、その後 `Cin→Cout` へ写す。**H=0 と形式的に呼ぶより、band ごとの readout 長を `H_i` から `L_i` に替えた**と理解するのが正確。

## 論文の学習・実験を読む

教師は次 H 時刻の実測値 `Y:[B,H,C]`、入力は過去 L 時刻 `X:[B,L,C]`。論文の主実験は `SmoothL1(Ŷ,Y)` を使い、MSE/MAE は評価指標。公式 `exp/exp_main.py` の criterion は `args.loss` で `mse` と `smoothL1` を選べるので、**公開コードの全実行が SmoothL1 固定という意味ではない**。同じ「MSE の表」でも学習 loss と評価 metric を分けて読む。

論文の計算比較では ETTh1・統一設定・`d=16` で WPMixer の GFLOPs が TimeMixer の 1/10 未満と報告する。これは特定の予測実験の構成であり、`C`, `P`, `S`, `d`, wavelet level が変わっても常に 1/10 とは言えない。ablation では分解段数 `m=1..5` の最適値が dataset と horizon に依存し、SmoothL1 と MSE の比較も別途行う。既定 `levels=2` を万能値と考えず、実際の系列長と目的に合わせて評価する。

## 本 block と公式実装の対応

| 機能 | 公式コード | 携帯版 | 違い |
| --- | --- | --- | --- |
| global RevIN | `WPMixerCore.revin` | `input_affine_weight/bias` + window statistics | affine を保持。mean/std は勾配から分離 |
| DWT/IDWT | `pytorch_wavelets` backend | `haar_decompose/haar_reconstruct_official` | Haar/db1 のみ。zero 境界・逆変換の切り詰め順と係数値を直接照合済み |
| per-band RevIN | `ResolutionBranch.revin` | `ResolutionBranch` 内の mean/std | 公開版と同じ係数単位の統計。affine は元実装で既定 off |
| patching | `do_patching` | `unfold` + end-repeat pad | 短い係数列 `L_i<P` も扱うため追加 pad |
| patch/embedding mixer | `TokenMixer`, `Mixer` | `_Mixer` | 軸・Linear・BN・skip の順序を保持 |
| future band head | `Linear(N_i*d,H_i)` | `Linear(N_i*d,L_i)` | **目的が異なる** |
| output channel | 原著の forecast 変数 C | `Linear(Cin,Cout)` | 下流 feature への射影 |

公式 DWT/IDWT の Haar/db1 係数と、実ライブラリを使う公式 core 同長経路の出力を照合済み。公式 `ResolutionBranch` の出力・入力勾配も一致した。別途、偶数・奇数 T の block output shape/勾配、Haar DWT→IDWT の往復も確認した。wavelet family を変えると filterbank と境界長が変わるため、後から `wavelet="db5"` と指定して同じ重みを使うことはできない。

## 下流でどう使うか・失敗しやすい点

~~~text
train: x = 過去 T 点 [B,T,Cin]
       h = WPMixerBlock(x) [B,T,Cout]
       pooled = mean_time(h) [B,Cout]
       y_hat = external_head(pooled) [B,K]
       GT = 次行の回帰 target [B,K] / class [B]
       MSE または CE → backward → optimizer.step()
infer: 同じ長さ T の履歴だけを標準化し encoder/head へ渡す
~~~

`seq_len` は各 branch の patch 数と head 入力次元を決めるので固定。データ窓が短く `T<2^levels` なら生成時に例外。`BatchNorm2d` は batch と patch 方向の統計を共有するため小 batch や分布変化で揺れる。`eval()` で running stats を使うこと、train/valid を時刻順に分けること、原著の予測 MSE と downstream の回帰/分類 metric を混同しないことが必要。

## 帯域長・patch 数とパラメータ数の見積もり

`N_i` が増えると token MLP の最初の Linear は概ね `N_i×(t_f N_i)`、二番目も `(t_f N_i)×N_i` の重みを持つ。branch ごとに二つ Mixer があるため、同じ d でも高解像度の D1 branch が最も重くなることが多い。さらに branch head の重みは `L_i×(N_i*d)`。したがって patch 長 P を短くするほど N_i が増え、計算もパラメータも増えやすい。wavelet により粗い A_m/D_m は短くなるが、D1 は元長の約半分なので、分解段数だけを増やしても高解像度 branch の負担は消えない。

| 記号 | 意味 | この block の例 `T=16,m=2,P=4,S=2` |
| --- | --- | --- |
| `L_1,L_2` | DWT 各段の係数長 | 8,4 |
| `A_2,D_1,D_2` | branch への入力順 | 長さ 4,8,4 |
| `N_i` | 末尾 pad 後の patch 数 | 2,2,4 |
| `d` | 一 patch の埋込幅 | 構築時の `d_model` |
| `Cout` | 下流 block の出力特徴幅 | 最後の `Linear(Cin,Cout)` |

`patch_len` が低解像度 band より長い場合、原著の `unfold` は空になる可能性がある。本版はその band を最後値で P 点まで補ってから S 点足し、少なくとも 2 個の patch が作れるようにした。これは短い window で実際に使うための例外処理であり、公式と同じ padding ではない。P を band 長に対して極端に大きくする設定は、補った値への依存が強くなるので避けたい。

## 周波数情報をどう復元するか

Haar の逆演算は `A` と `D` が対応する一対の係数であるとき、`even=(A+D)/√2`, `odd=(A-D)/√2` として二つの時刻値を作る。本版では各 branch が A/D を変換した後にこの式を適用する。元入力を単純に正確復元するのが目的ではなく、**係数空間で学習した表現を時間軸へ写し直す**ために逆 DWT を使う。branch が恒等関数なら正逆は一致するが、学習済み branch が変換した後の output はもちろん元入力と同じではない。検証スクリプトの可逆性テストは filterbank の正逆だけを確認し、モデル全体が恒等になることを主張しない。

出力 `h[b,t,:]` の意味は「入力の時刻 t に対応する多尺度変換特徴」。しかし token mixer は全 patch position を線形混合するため、`h[b,t]` には窓全体の情報が入り得る。これは回帰・分類用の履歴 encoder としては有用でも、逐次オンライン推定で未来の観測を混ぜてはいけない場合は window の切り方を制限する必要がある。

## `ts_block.py` を別プロジェクトへ移す際の最小契約

必要な Python 依存は PyTorch と `einops`。`pytorch_wavelets` は本版 Haar backend には不要。ただし原著で検証した db5 等の wavelet family を使いたい場合は、別 backend とそれに対応する係数長・境界処理を追加し、学習済み重みもその設定で学習し直す必要がある。単に Haar の filter 係数だけ db5 に置き換えても、長さや inverse 側を含めて整合しない。

`WPMixerBlock(in_dim,out_dim,seq_len,...)` の `Cin` は毎 batch 同じ順序である。BN2d(Cin) は**チャンネル index ごと**に running mean/var と affine を持つため、列を並び替えるとモデルの意味が壊れる。wavelet 操作も各チャンネル別にかかり、異なるチャンネル間の明示的な相互作用は最後の `Linear(Cin,Cout)` が主。相関が重要なデータでは、別の channel mixer を外側に足すか、他モデルと比較する。

ブロックを使った推論では `model.eval()` を呼び、train で学習した global/branch BN の running stats を使う。DataFrame 側の z-score scaler は BN・RevIN と役割が別で、系列間の入力スケールを揃える前処理。train+validation 全行で scaler を fit すると、いくら RevIN が窓内統計を使っても将来の分布情報が漏れる。shared finetuning 例は train partition の行だけから scaler を算出する。

最後に、波形を再構成する同長 feature block は下流 target の形式を決めない。予測したいなら `[B,T,Cout]` から horizon head、分類なら pooling と CE head、点ごとの異常度なら時刻別 head を別に設計する。原著の SmoothL1 を block 内に固定していないのは「学習を省略した」ためではなく、タスクごとの GT が異なるためである。

原著が帯域ごとに別 branch を持つのは、異なる周波数範囲の係数を一つの共通 patch mixer に混ぜてしまわないため。各 branch の独立性は計算量とパラメータを増やす反面、D1 の急変を A_m の滑らかな成分と区別して扱える。すべての帯域に同じ重みを共有させる変更は効率改善の別設計であって、原著の再現ではない。

## 原著 forecast の train/inference を短く再掲

~~~text
train: X=過去 L 点 [B,L,C], Y=未来 H 点 [B,H,C]
       global RevIN(X) [B,L,C] → DWT(m) で m+1 band
       各 band: branch 固有の RevIN→patch→2 Mixer→未来係数 head
       inverse DWT→inverse global RevIN = Ŷ [B,H,C]
       SmoothL1(Ŷ,Y) scalar （公式 criterion は設定で変更可能）
       backward→optimizer update
infer: 同じ過去 X だけを渡し Ŷ を出す。Y は入力しない
adapter: 同長 band head + external downstream head は別経路
~~~

この再掲は、原著では `H_i` が未来 H から決まるのに対し、本 block では `L_i` が入力 T から決まる、という違いを一目で確認するため。二つの学習 objective を同じものとして説明しない。

## 一次資料・再現境界

原論文: [WPMixer (AAAI 2025)](https://arxiv.org/abs/2412.17176)。公式実装: [Secure-and-Intelligent-Systems-Lab/WPMixer](https://github.com/Secure-and-Intelligent-Systems-Lab/WPMixer)。照合箇所: 論文 `Full_paper.tex` の Model Architecture/Experiments、公開 `models/wavelet_patch_mixer.py`, `models/decomposition.py`, `exp/exp_main.py`。
