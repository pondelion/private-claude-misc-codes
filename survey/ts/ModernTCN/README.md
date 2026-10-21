# ModernTCN survey

対象: [*ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis*](https://openreview.net/forum?id=vpJMJerXHU), ICLR 2024 と [公式実装](https://github.com/luodhhh/ModernTCN)。PDF はテキストとして参照し、実装の細部は `ModernTCN-Long-term-forecasting/models/ModernTCN.py` と照合した。目的は attention を使わず、長い有効受容野と変数間相互作用を持つ純畳み込み backbone を作り、予測・補完・分類・異常検知へ共有すること。

## このディレクトリのブロックの使い方

主用途は **`ModernTCNBlock` を系列 encoder として他のモデルへ組み込むこと**。`x:[B,T,Cin]→h:[B,T,Cout]` のみを返し、GT・loss・pooling・下流 head を持たない。`h.mean(dim=1):[B,Cout]` を系列単位の回帰/分類 head に渡すか、pooling せず時刻別 head に使える。T は入力ごとに変えられる。畳み込みは非因果なので、時刻 t の特徴 `h_t` は入力窓内の t より後を参照し得る。

以下の固定 H 予測 head と MSE は **原著の一タスク**であり、同長ブロックの契約ではない。[finetuning_example.py](finetuning_example.py) の将来予測も任意の下流学習例である。

## 概要

ModernTCN は、**時間パッチを大きな depthwise 畳み込みで処理し、変数内・変数間の情報を二段の FFN で混ぜる**時系列 backbone である。過去系列 `X:[B,L,M]` を変数ごとにパッチ埋込し、`[B,M,D,P]` の特徴を複数 stage で更新する。`B` は batch、`L` は入力時間長、`M` は変数数、`D` は変数ごとの潜在幅、`P` は patch 数。出力形式と教師信号はタスク head によって異なり、長期予測では将来 `Ŷ:[B,H,M]` と実測 `Y:[B,H,M]` の MSE を使う。

このディレクトリの [ts_block.py](ts_block.py) は予測専用 head を **`[B,T,Cin]→[B,T,Cout]` の同長特徴 readout** に変更したもの。図は先に原著の backbone と予測 head を示し、その後に adapter の違いを説明する。

## 新規性・既存手法との差分

| 比較対象 | ModernTCN の差分 | 意味・限界 |
| --- | --- | --- |
| 小 kernel / dilation 中心の従来 TCN | **大 kernel の depthwise 畳み込み**で時間方向を広く見る | attention なしで受容野を広げるが、実効受容野と実測速度は設定依存 |
| 一つの全 channel FFN | 変数内 D 次元を混ぜる FFN1 と、同じ潜在座標で変数 M を混ぜる FFN2 を分離 | 時間混合・変数内・変数間の役割を明確に分ける |
| 学習時と推論時に同じ二枝 Conv | large/small Conv+BN を学習し、推論時に一つの大 kernel Conv へ畳み込む | 推論の枝数を減らす。融合そのものが予測精度を上げるわけではない |
| 単一解像度 Conv | patch stem と stage 間 downsample で粗い時間解像度へ進む | 長い文脈を圧縮するが、細かな時刻情報は失い得る |

提案は「Conv1d を重ねただけ」ではない。特に **FFN2 に入る前の `[B,M,D,P]→[B,D,M,P]` 軸交換**が変数間混合を正しく実現する条件で、ここを省くと group Conv の意味が変わる。

## モデルアーキテクチャ全体像（長期予測設定）

```text
履歴 X [B,L,M]
  │ 任意の RevIN（有効設定のみ）
  ▼
変数を独立 stream に分ける [B·M,1,L]
  │ patch Conv1d(kernel=patch_size, stride=patch_stride)
  ▼
patch token [B,M,D_1,P_1]
  │
  ├─ Stage 1: ModernTCN Block × n_1 ────────────────┐
  │      入力 [B,M,D_1,P_1]                         │
  │        ├─ large DWConv+BN ─┐                    │
  │        └─ small DWConv+BN ─┴─ 加算 [B,M,D_1,P_1]│
  │        ├─ FFN1: M groups、各変数内の D_1 を混合 │
  │        ├─ 軸交換 [B,D_1,M,P_1]                  │
  │        ├─ FFN2: D_1 groups、変数 M を混合       │
  │        └─ 元の軸へ戻して residual 加算          │
  │             └─────────────────────────────────────┘
  │ BN → stride Conv による時間 downsample / 幅変更
  ▼
Stage 2 … Stage S: [B,M,D_s,P_s]、同じ Block を各 stage で反復
  │
  ├─ 原著の予測 head: 変数ごと [D_s,P_s] を連結 → Linear(D_s·P_s,H)
  │                     → 予測 Ŷ [B,H,M] ── MSE ── GT Y [B,H,M]
  └─ 今回の同長 adapter: D_s を readout → 時間補間 → [B,T,Cout]
```

`P_s` は stage s の patch 数で、予測 horizon `H` とは別。図の二枝 large/small Conv は **学習時**の形で、`eval()` 後の kernel 融合では同等の一枝 Conv にできる。公開 long-term forecasting file に存在する multi-scale fusion 用層は、この図の対象 `forward_feature` では呼ばれないため描いていない。今回の同長 readout・時間補間は原著の固定 H 予測 head ではない。

## 原著 backbone の経路

| 処理 | 入力 → 出力 | 意味 |
| --- | --- | --- |
| 観測 | `[B,L,M]` | L: 時間、M: 変数 |
| RevIN (選択) | `[B,L,M] → [B,L,M]` | sample/変数別統計で正規化 |
| patch stem | 各変数 `[B·M,1,L] → [B·M,D,P]` | Conv1d(kernel=patch, stride=stride) |
| stage block | `[B,M,D,P] → [B,M,D,P]` | 後述の DWConv+2 FFN |
| stage 間 | 時間 patch 数 `P` を段階的に縮小し、`D` を変更 | 深い stage ほど粗い表現 |
| task head | 予測なら `[B,H,M]`、補完なら `[B,L,M]` 等 | task ごとに別 |

最重要 block はまず **reparameterizable large-kernel depthwise convolution**（学習時の large Conv+BN と small Conv+BN を加算）。次に変数 M を group とする pointwise FFN で各変数内の D 特徴を混ぜる。軸を `[B,D,M,P]` に入れ替えて D を group とする第2 FFN で **同じ潜在座標の変数間**を混ぜる。最後に residual 加算する。単に「大きい畳み込みを積んだだけ」ではなく、時間依存と変数間依存を分けた設計。inference では BN を畳み込みへ fold し、大/小 kernel を一つの depthwise kernel に合成できる。これは演算グラフを短くするが精度向上の仕組みではなく、eval 時の関数を保つ推論最適化。

論文の ablation でも、この二段 FFN を一つの未分解 FFN にすると長期予測の平均 MSE が悪化する。例として ETTh1 は 0.404→0.425、ILI は 1.440→1.813（小さいほどよい）。ConvFFN2 を除いた補完実験でも ETTm1 の MSE は 0.020→0.038。したがって変数間混合は単なる実装詳細ではなく、提案の中核である。

## 学習の入力・GT・loss

ModernTCN は複数タスク用 backbone なので単一 objective はない。long-term forecasting なら `X:[B,L,M]` → `Ŷ:[B,H,M]`、GT は将来実測 `Y:[B,H,M]`、典型的に MSE。補完なら隠した既知点の再構成、分類なら系列 label と CE、異常検知なら再構成誤差という具合に head と loss が変わる。これらを同じ loss とみなさない。

## 実用同長ブロック

[ts_block.py](ts_block.py) の `ModernTCNBlock(in_dim,out_dim)` は `[B,T,Cin]→[B,T,Cout]`。公式の patch stem、large/small 再パラメータ化 DWConv、二段 grouped FFN を保ち、task 固有の head を時間補間＋出力線形に置換した。既定は軽い単一 stage だが、`stage_dims=(32,64,128,256), stage_depths=(2,2,2,2)` のようにすれば公式に沿う **stage 間 BN→stride Conv の多段 backbone** を構築できる。原著の固定 horizon head と RevIN は同長 adapter へ含めないので、論文モデル完全再現ではない。推論前は `model.eval(); model.merge_kernels()` で畳み込み統合できる。学習再開時は非統合モデルを使う。

```python
from ts_block import ModernTCNBlock
block = ModernTCNBlock(8, 32, d_model=32, patch_size=8, patch_stride=4)
h = block(history)  # [B,T,8] -> [B,T,32]
block.eval()
block.merge_kernels()  # 学習済みモデルで推論する直前のみ
```

[finetuning_example.py](finetuning_example.py) は下流タスクの一例として、DataFrame から過去窓と未来 GT を生成し、block の最終表現から future MSE を学習する。汎用 block 自体の objective ではない。

## 速度と落とし穴

large-kernel DWConv は全特徴別に時間混合でき、attention の `T²` 項がない。ただし大 kernel・多数変数では畳み込みの実測速度を確認する。small branch の BN 統合は eval 統計が確定してからのみ正確。非因果畳み込みなので同長出力 `h_t` は `x_{>t}` を参照し得る。将来を含む系列を入力して online t の予測に使うと漏洩する。履歴窓全体から未来 head を出す運用は問題ない。

## 通常の TCN から何を変えたか

従来の TCN は比較的小さい kernel と dilation の積み重ねで時間受容野を広げることが多い。ModernTCN は computer vision の modern ConvNet で有効だった大 kernel depthwise convolution と reparameterization を時系列へ持ち込み、さらに **変数軸 M が画像の channel と同じ意味ではない**ことを踏まえ、FFN を変数内と変数間に分解する。単なる「Transformer を Conv1d に置き換えた」モデルでも、attention を使った hybrid でもない。

原著は長期予測、短期予測、補完、分類、異常検知の5タスクを評価する。共通なのは backbone の時系列表現であり、task head・GT・loss はそれぞれ異なる。論文の task-general という表現は「一つの同じ出力 head で全タスクを解く」という意味ではない。

### patch stem の意味

`X:[B,L,M]` を `[B·M,1,L]` にし、`Conv1d(1,D,kernel=P,stride=S)` で `[B·M,D,K]` を作る。元系列の各変数がそれぞれ D 次元の patch 表現となる。元の M を復元すると `[B,M,D,K]`。この段階では変数間を混ぜず、全変数が stem weight を共有する。P が大きいと一つの patch が長くなり、K は減る。S を小さくすると patch が overlap し、K と計算量が増える。

公式 long-term forecasting モデルは stem の後に複数 stage を置き、stage 間では `Conv1d(D_i,D_{i+1},kernel=r,stride=r)` で patch 軸を縮小する。最終 stage の `[B,M,D_last,K_last]` を flatten head に渡し、`[B,M,H]` を予測する。backbone 出力の K と `target_window=H` は別であり、元系列長 L と予測長 H が同じ必要はない。

### Block の軸操作を一つずつ追う

例として `B=2,M=7,D=64,K=24,expansion=2` とする。

```text
入力                                        [2,7,64,24]
M,Dを畳み depthwise Conv1d                   [2,448,24]
BN後、Mをgroupとした ConvFFN1:
  group数7、group内64→128→64                [2,448,24]
reshape/transposeして Dをgroupとした ConvFFN2:
  [B,D,M,K]を[B,D*M,K]                       [2,448,24]
  group数64、group内7→14→7                  [2,448,24]
元の[B,M,D,K]へ戻し residual 加算            [2,7,64,24]
```

最初の depthwise Conv は `M·D` 個の channel を各々独立に時間方向で混ぜる。ConvFFN1 の group 数 M は、同じ変数の D 特徴だけを 1x1 Conv で混ぜる。ConvFFN2 の前に `[B,M,D,K]→[B,D,M,K]` と並べ替えるのが重要で、これを忘れて単に group 数 D の Conv1d を置くと、実際には意図した M 変数同士が同じ group に入らない。公開実装の二回の reshape/permute はこのためにある。各 FFN の演算順は公式 `Conv1d→Dropout→GELU→Conv1d→Dropout`。前版ブロックでは最初の Dropout を GELU 後に置き、二番目の Dropout を省いていたため修正した。

### large/small kernel reparameterization

学習時は `Conv_large→BN_large` と `Conv_small→BN_small` を足す。例えば large=15、small=3 なら small kernel の両端に 6 個ずつゼロを pad して large と同じ重み形状へ揃えられる。BN を eval 統計で畳み込むと、各 branch が等価な `Conv(weight_fused,bias_fused)` となる。二つの重み・bias を足せば inference で一回の Conv1d に置換できる。

`weight_fused = weight_conv · γ/√(running_var+eps)`、`bias_fused = β+(bias_conv-running_mean)·γ/√(running_var+eps)`。これは **eval 時の BN running statistics を前提**にするため、学習中に merge して optimizer step を続ける使い方は誤り。本 [ts_block.py](ts_block.py) の `merge_kernels()` は eval でしか呼べず、融合前後の値を [検証スクリプト](../validate_blocks.py) が比較する。

### 公式 long-term code の multi-scale についての注意

公開 `ModernTCN.py` は `lat_layer/smooth_layer/up_sample_conv` を生成し `use_multi_scale` option を持つ。しかし、指定の long-term forecasting 版 `forward_feature` は各 stage を順に通して **最終 stage の x だけを返し**、それらの fusion layer を呼んでいない。実験 shell script の多くは `--use_multi_scale False` を指定する。したがって「公開 forward が必ず FPN 型の多尺度 upsampling/fusion を実行する」と書くのは誤り。公式 repository 内には複数タスクの別実装があり、タスク別に挙動を確認する必要がある。

同じ long-term file に calendar/time-feature 用 `te_patch` も定義されるが、示した `forward_feature(x,te)` 内では `te` が参照されない。時刻特徴を渡したからモデルが必ずそれを利用する、とは断言できない。本汎用版はこの未使用経路を含めない。

## 何を入力して、何を正解にするか

### 長期予測

過去 `X:[B,L,M]` のみを入力し、head が `Ŷ:[B,H,M]` を出す。GT は同じ M 変数の未来実測 `Y:[B,H,M]`。一般的な `nn.MSELoss(Ŷ,Y)` は B·H·M 要素の二乗誤差平均。RevIN を使う設定では入力 window のみから統計を計算し、予測値を逆変換して GT と同じ単位で比較する。分解 option が有効なら `X=trend+residual` を別々の ModernTCN に通し、予測を足す。これは全設定で必須な骨格ではない。

### 他タスク

補完では観測を意図的に mask した入力から隠した値を予測し、通常は mask 対象位置を GT と比較する。異常検知では再構成値と観測のずれを異常 score にする。分類では系列全体から class logits を出し、class ID との CE を取る。これらは同じ `[B,L,M]→[B,H,M]` head ではなく、それぞれの task-specific head と loss を要する。本 `ModernTCNBlock` は head 前の時間特徴を同長で返す汎用化。

## 論文の検証で重要な結果

論文の block ablation では、ConvFFN1 と ConvFFN2 の両方を残す構成が、片方だけ・未分解 FFN・変数軸を捨てる構成より良い。上記 ETTh1/ILI の数字はその一例。補完で ConvFFN2 を外すと ETTm1 の MSE が 0.020→0.038、ETTh1 が 0.050→0.089 と悪化する。異常検知でも MSL F1 が 84.92→72.13 と報告される。これは cross-variable 経路を削ると別タスクでも性能が落ちるという証拠であり、単にパラメータが増えた効果だけを完全に切り分けたものではない。

大 kernel による有効受容野（ERF）の可視化も論文の主張を支える。ただし theoretical receptive field と empirical ERF は同じではない。深い stage と大 kernel により理論上遠い時点を参照できても、学習された勾配の影響範囲はデータ・重みに依存する。金融・交通など別分布で「必ず長期依存を拾う」とは言えない。

## off-the-shelf ブロックでの変更と推奨設定

本 `ModernTCNBlock` は既定で単一解像度の patch stem + 指定 depth の Block を持ち、`stage_dims/stage_depths` を渡すと複数 stage と stage 間 downsampling を使う。原著の固定 H flatten head、RevIN、任意の trend/residual 二重モデルは同長 adapter に含めない。代わりに最終 `[B,M,D,K]→[B,K,M]→時間補間して[B,T,M]→Linear(M,Cout)` とし、任意 T で同長特徴を返す。原著の予測 head は未来 `H` 点を返すだけで入力の各時刻に対応する特徴を返せないため、この置換が必要である。ただし補間は細かい時間変化を平滑化し得る。高周波の時点対応が重要なら patch size/stride を小さくする。

[verify_official.py](verify_official.py) は公式 long-term forecasting 実装の `Block` と `ModernTCNUnit` の出力・入力勾配を比較する。さらに stem、2 stage、stage 間 downsampling、両 stage の Block まで、対応する重みで**backbone 最終出力が一致**することを確認する。原著の固定 horizon head を同長の補間・readout に置き換えた部分は一致しない。別途 `merge_kernels()` 前後の出力も検査する。

複数系列を batch 化する際は M と特徴順を固定する。`ConvFFN2` は M 軸を混ぜるため、列順を入れ替えてそのまま推論することはできない。padding された可変長系列でも Conv/BN は padding の値を見てしまうので、長さ mask による特別な処理を追加する。BN は小 batch や分布変動の大きいストリーミングデータでは統計が不安定になり得るため、訓練/推論分布を揃える。

## 公式実装との対応

| 概念 | 公開コード | 本ディレクトリ |
| --- | --- | --- |
| patch stem / stage downsample | `models/ModernTCN.py::ModernTCN` | `ModernTCNBlock.stem/downsamples/stages`。既定一段、複数段選択可 |
| large/small DWConv と BN fuse | `ReparamLargeKernelConv` | `ReparamDepthwise`、`merge_kernel` |
| 変数内・変数間 FFN | `Block` の `ffn1/ffn2` | `ModernTCNUnit.ffn_var/ffn_cross` |
| 予測/補完等の task head | task 別 `ModernTCN.py` | 同長 readout と出力 Linear。未来予測 head は fine-tune script |
| 論文 forecasting objective | task runner の MSE | `finetuning_example.py::fit` の MSE |

公開 `ReparamLargeKernelConv.PaddingTwoEdge1d` には `torch.cat(...,dims=-1)` のような実行不能な引数名が見られる。本版は `F.pad` と device/dtype 一致を用いて BN 融合を実装し、数値一致をテストした。原著の学習済み checkpoint を直接読み込むための key/shape 互換は提供しない。

## 一次資料

- [論文: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis](https://openreview.net/forum?id=vpJMJerXHU)（提供 PDF のテキストと公式実装を照合）
- [公式 ModernTCN 実装](https://github.com/luodhhh/ModernTCN)

## 動作確認と DataFrame 学習の入口

このディレクトリで `python ts_block.py` を実行すると、実際の block に乱数 tensor を渡して出力 shape と backward を検証する。`python ../validate_blocks.py` は5モデル横断で、異なる T と数値特性を追加検証する。追加ライブラリを入れない PyTorch 実行では Mamba 系の fallback を使い、速度比較ではない。

CSV を用意する場合は `series_id,timestamp` と特徴列・target 列が必要。例えば次のコマンドは過去64点から次8点の実測 `target` を予測する。

```bash
python finetuning_example.py data.csv --features feature_a feature_b --targets target --lookback 64 --horizon 8 --epochs 10
```

既に DataFrame `df` があるなら `from finetuning_example import fit; model, info = fit(df, ["feature_a", "feature_b"], ["target"])`。返す `info` には train 行だけから求めた feature mean/std と最良の validation MSE が入る。推論時は同じ列順・同じ正規化統計で過去窓を作る。本文で述べた原著の benchmark 成績を再現するには、このサンプル以外に原著の入力長、head、split、正規化と optimizer 設定を揃える必要がある。

## 原著の学習・推論の擬似コード（長期予測設定）

以下は公開 long-term forecasting モデルの経路であり、`ModernTCNBlock.forward` の API ではない。同長ブロックは `[B,T,Cin]→[B,T,Cout]` の特徴を返した時点で終了し、pooling・head・GT・loss は下流タスク側が決める。

```text
history                               [B,L,M]
GT future                             [B,H,M]
x = optional_RevIN_norm(history)      [B,L,M]
x = transpose_to_variables(x)         [B,M,L]
patches = shared_patch_stem(x)        [B,M,D_0,K_0]
for stage i:
    patches = optional_downsample(patches) [B,M,D_i,K_i]
    for block in stage:
        patches = DWConv_large_small(patches)
               -> ConvFFN1(groups=M)
               -> ConvFFN2(groups=D_i)
               -> residual add
flatten_head(patches)                 [B,M,H]
pred = transpose + optional_RevIN_denorm [B,H,M]
loss = mean((pred-GT future)^2)       scalar
```

推論時は future GT を使わず同じ history を forward する。H 点は直接一括生成され、自己回帰的に前予測を次へ入力しない。reparameterization 後の DWConv は eval 演算を一つへまとめるが、stage 数・出力値の意味・予測 head は変えない。
