# WaveTS survey

対象: *Multi-Order Wavelet Derivative Transform for Deep Time Series Forecasting* / WaveTS（指定 TeX `maincontent.tex` と [公式コード](https://github.com/zhouziyu02/WaveTS) の `models/WaveTS.py`）。目的は Fourier の大域周波数だけで見落としやすい局所的な急変を、wavelet の時間-周波数分解と「微分次数」別の枝で扱うこと。

## まず、このディレクトリのブロックは何をするか

**`WaveTSBlock` の契約は `x:[B,T,Cin] → h:[B,T,Cout]` だけ。** `forward` は未来予測、backcast、GT、loss を扱わない。各時刻に対応する特徴 `h` を返し、その後の pooling や task head は呼び出し側が自由に決める。`T` は時間、`Cin` は入力特徴数、`Cout` は出力特徴数。例えば `[B,48,16]→[B,48,64]` の encoder として使える。

```text
実際の WaveTSBlock.forward（タスク非依存）
x [B,T,Cin]
  → 入力窓の統計で正規化                         [B,T,Cin]
  → Haar 分解・order 別の帯域変換/FRU/逆変換    各枝 [B,T,Cin]
  → order 枝の時間方向結合・同長射影             [B,T,Cin]
  → 逆正規化・出力特徴射影                       h [B,T,Cout]

ここで終了。未来ラベルも loss も `WaveTSBlock` 内にはない。
```

ただし **`T=seq_len` はモデル構築時に固定**する。FRU と枝融合に時間長依存の学習行列があるため、同じインスタンスへ別の T を入れることはできない。異なる入力長を使う場合は、長さごとに別インスタンスを構築する。これは入出力の「同長」とは別の制約であり、任意 T に一つの重みを使い回せるという意味ではない。

## 概要

WaveTS は、**過去の時系列を wavelet の低周波・各尺度の高周波に分け、複数の「微分次数」枝で将来までの系列を生成する**予測モデルである。入力 `X:[B,L,C]` に対して、過去の再構成（backcast）と未来予測（forecast）を連結した `Ŷ:[B,L+H,C]` を返す。`B` は batch、`L` は履歴長、`H` は予測長、`C` は変数数。公開コードの既定学習では GT `concat(X,Yfuture):[B,L+H,C]` との全要素 MSE を取るため、**未来だけでなく過去の再構成も学習対象**である。

このディレクトリの [ts_block.py](ts_block.py) は将来 `H` を作る原著 head ではなく、`H=0` とした **同長 `[B,T,Cin]→[B,T,Cout]` ブロック**。以下の図は原著の `L→L+H` 経路を示す。

## 新規性・既存手法との差分

| 比較対象 | WaveTS の違い | 何に効かせたいか／注意点 |
| --- | --- | --- |
| Fourier/DFT 系予測 | wavelet で時間位置と尺度の両方を保持 | 局所的な急変・異なる周期幅を区別する |
| 通常の DWT + 予測 head | order ごとに wavelet 係数を再スケールし、**帯域別 FRU を別パラメータで学習** | 低周波と詳細帯域を別々に `Lband→(L+H)band` へ伸ばす |
| 単一 order | 複数 order 枝をそれぞれ逆変換し、時間軸連結後に学習射影で融合 | order ごとの表現を選択・統合する。単純平均ではない |

重要なのは、公開コードの WDT が「元信号を n 回有限差分してから DWT」する処理ではないこと。**DWT 係数に order・尺度依存の係数を掛け、FRU の後で割って iDWT** する。FRU が恒等ならこの往復は原理的に元へ戻るため、wavelet 変換単体が未来を作るわけではない。論文の微分 wavelet 基底の理論説明と、公開実装の数値処理は区別する。

## モデルアーキテクチャ全体像（原著の `L→L+H`）

```text
過去 X [B,L,C] ── 時間 L の平均/分散で正規化 ──> X_norm [B,C,L]
                                                      │
                         ┌──────── order n=1,…,O の独立枝 ────────┐
                         │                                         │
                         │ DWT K 段                               │
                         │   LL_K [B,C,L_K]                       │
                         │   LH_k [B,C,L_k] (k=1,…,K)             │
                         │        │ order n と尺度 k で再スケール  │
                         │        ▼                                │
                         │ 帯域別 FRU = 独立 Linear               │
                         │   LL_K: L_K→L'_K                       │
                         │   LH_k: L_k→L'_k                        │
                         │   L'_k は L+H に対応する係数長          │
                         │        │ 係数スケールを戻す             │
                         │        ▼                                │
                         │ iDWT ──> 枝出力 [B,L+H,C]              │
                         └─────────────────────────────────────────┘
                                      │ O 枝を時間方向に連結
                                      ▼
                             [B,O·(L+H),C]
                                      │ Linear(O·(L+H),L+H)
                                      ▼
                             [B,L+H,C] ── 入力統計で逆正規化
                                      ▼
                      Ŷ [B,L+H,C] = [backcast | forecast]
                                      │
                      GT [B,L+H,C] = [過去 X | 未来 Y]
                                      ▼
                         全区間 MSE（公開 train の既定）
```

`L_k` は入力の第 k 帯域長、`L'_k` は将来 `H` を含む出力長に対応する係数長であり、wavelet level `K` と order 数 `O` は別物。FRU の長さ変換が未来を生成する主要な学習処理。今回の同長ブロックでは `L'_k=L_k`、枝出力を `[B,L,C]` にしてから `[B,L,Cout]` へ写すため、原著の未来予測そのものではない。

## 原著の入出力と objective

入力は過去 `X:[B,L,C]`、出力は **backcast と forecast を連結した** `Ŷ:[B,L+H,C]`。GT は `concat(X,Yfuture):[B,L+H,C]`。公開 training code の既定 `ft=False` では、この連結 GT と出力の **全要素 MSE** `mean((Ŷ-concat(X,Yfuture))²)` を学習する（`ft=True` は未来区間のみ）。つまり実際の loss は `L/(L+H)·MSE_past + H/(L+H)·MSE_future`。過去と未来の両方が教師対象である。推論で使うのは通常 `Ŷ[:,L:]`。正規化は各サンプル・各変数について時間 L 軸で行い、出力後に逆変換する。**論文 Eq. には `MSE_past + MSE_future = MSE_all` とも書かれているが、これは数学的には一般に成り立たない**。公開コードの挙動を再現するなら全要素 MSE に合わせる。

| 処理 | 入力 → 出力 | 軸 |
| --- | --- | --- |
| 入力正規化 | `[B,L,C] → [B,L,C]` | L 方向平均/分散 |
| 各 order n の WDT | `[B,C,L] → LL:[B,C,ceil(L/2^K)], LH_k:[B,C,ceil(L/2^k)]` | K: wavelet level |
| FRU | 各 `LL/LH` の長さを `L` 対応から `L+H` 対応へ線形変換 | 各 C で同じ線形重み |
| iWDT | `LL,LH → [B,L+H,C]` | wavelet 逆変換 |
| N-order 融合 | N 枝を時間連結後に射影 → `[B,L+H,C]` | コードでは `projection_time` を使用 |

## WDT をどう理解するか

論文は wavelet 基底の n 次微分と係数の尺度補正を導入する。公開コードで実際に行う処理は、標準 DWT の detail 第 k 尺度へ `(-1)^n 2^(n k)`、LL へ最深尺度の同様の係数を掛け、逆処理ではそれを割って IDWT すること。**コードは微分済み wavelet basis を数値的に構築していない**。したがって理論上の「微分係数」と公開コードの「DWT 係数の再スケーリング」は厳密には区別する。また変換と逆変換だけなら恒等写像になり、学習可能な FRU を挟むことが本質である。公開コードでは `projection_variate` を定義するが `forward` では未使用で、実際の融合は時間方向の射影。

## 汎用同長ブロック

[ts_block.py](ts_block.py) の `WaveTSBlock(in_dim,out_dim,seq_len=L)` は PyTorch と `einops` で Haar/db1 wavelet を正逆変換し、order ごとの係数再スケール→**帯域別 `Linear(L_j,L_j)` の FRU**→逆変換→**order を時間軸連結して `Linear(O·L,L)`**→channel 出力写像と進む。`[B,L,Cin]→[B,L,Cout]`。公式の FRU を同長化するため `H=0` と置いた形で、学習行列の長さは `L` に依存する。**任意長を一つの checkpoint で扱う設計ではない。** 奇数 L では各 level の右端を `reflect` で延長して trim する。公式 `pytorch_wavelets` の Haar/db1 係数と偶数・奇数長で一致する。

```python
from ts_block import WaveTSBlock
block = WaveTSBlock(8, 32, seq_len=96, levels=3, orders=(1, 2))
h = block(history)  # [B,96,8] -> [B,96,32]
```

[finetuning_example.py](finetuning_example.py) は同長特徴ブロックに backcast/forecast の二つの head を足し、train では公開コードの `ft=False` と同様に過去 L 点＋未来 H 点を連結した全要素 MSE、validation では未来 H 点のみの MSE を使う。原著と同じ「入力全変数の再構成」にするには `target_cols=feature_cols` とする。異なる target subset も学習可能だが、そのときは原著と異なる task である。**FRU と branch fusion の演算種類・順序は揃えたが、公式の `L→L+H` 予測 head は同長 `L→L` に置換**した。

## 実用上の注意

Haar は高速で可逆だが、非 Haar wavelet を使う公式設定と同一の結果にはならない。order の指数スケールは大きい値で数値範囲を広げるため、高 order/level は避け、AMP 使用時は勾配と損失を確認する。FRU の時間方向 `Linear` は非因果的なので未来観測を与えず、履歴窓のみに適用する。

## Wavelet と WDT を式から理解する

### DWT だけなら何が得られるか

1次元の Haar DWT では隣接二点 `(x_{2j},x_{2j+1})` を `LL_j=(x_{2j}+x_{2j+1})/√2`、`LH_j=(x_{2j}-x_{2j+1})/√2` に分ける。LL は局所平均、LH は局所差分に対応する。LL にもう一度同じ操作を適用すると、さらに長い尺度の平均と差分が得られる。K 段なら `LL_K` と `LH_1,…,LH_K` があり、すべて揃えば逆 DWT で元信号を復元できる。たとえば T=16,K=3 なら形状は `LL_3:[B,C,2]`、`LH_1:[B,C,8]`、`LH_2:[B,C,4]`、`LH_3:[B,C,2]`。

通常の DWT は局所化されるが、原著は変化点への感度をさらに強めるため derivative-order を導入する。公開コードでは `order=n` の枝で `LH_k` に `(-1)^n 2^{nk}`、`LL_K` に `(-1)^n 2^{nK}` を掛ける。学習済み FRU を通した後、iWDT で同じ係数を割って逆変換する。もし FRU を完全な恒等写像とすれば枝全体はほぼ入力の再構成に戻る。異なる order が異なる表現を学べるのは、**同じ元係数でも異なるスケールで FRU に入り、order ごとに別 parameter を持つ**ためである。単なる可逆変換そのものが将来値を作るわけではない。

理論節は微分 wavelet 基底 `ψ^(n)` を使った定義と可逆性を論じるが、指定された公開 `WaveletDerivativeTransform.forward` は `DWT1DForward` の結果への上記 rescale である。理論と実際の計算手順を同じ文で混ぜない。とくに「元系列を有限差分で n 回微分してから DWT に通す」と理解すると、公開実装とは異なる。

### なぜ LL と LH を別 FRU に通すか

LL は長期の平均的な変化、LH は各尺度の局所変動を担う。論文と公開実装の FRU はそれぞれ別の `nn.Linear` を持ち、係数系列の長さを過去窓 L に対応する長さから `L+H` に対応する長さへ拡張する。例えば L=16,H=8,K=2 なら、`LL_2` の長さ 4 を 6 に、`LH_2` の長さ 4 を 6 に、`LH_1` の長さ 8 を 12 に写す。それらを iWDT に入れて 24 点を再構成し、前 16 点を backcast、後ろ 8 点を forecast と読む。**係数の長さを拡張する線形層が、未来部分を作る主要な学習部品**である。

公開 `FRU` は各変数で同じ線形層を共有する。入力を `[B,C,Lband]→[B·C,Lband]` と flatten して `Linear(Lband,Lband_out)` を適用し、`[B,C,Lband_out]` に戻す。band 間では重みを共有しない。これは単に DWT を使うだけのモデルより、周波数帯ごとの変換を学ぶ柔軟性がある一方、変数間の相互作用は主要な FRU 内にない。

### order 枝をどう融合するか

原著の N 枝はそれぞれ `[B,L+H,C]` に戻る。`torch.cat(order_representations,dim=1)` は **時間軸**連結で `[B,N(L+H),C]`。その後 `transpose` して `Linear(N(L+H),L+H)` を時間軸へ適用する。これは order をただ平均する方式ではなく、出力時刻ごとに全 order・全再構成時刻を混ぜられる大きな線形写像である。公開 class の `projection_variate` は定義されているが `forward` で使用されないため、実際の branch 融合に「学習された変数方向 projection」も入る、と説明すると誤りになる。

## 学習目的を数値例で確認する

`B=1,L=4,H=2,C=1` とする。モデル出力は `Ŷ=[ŷ_1,ŷ_2,ŷ_3,ŷ_4,ŷ_5,ŷ_6]`、GT は `[x_1,x_2,x_3,x_4,y_5,y_6]`。公開 default training の `nn.MSELoss` は6つの二乗誤差の平均。したがって過去4点が全体 loss の `4/6`、未来2点が `2/6` に対応する。未来 MSE を過去 MSE と同重みで足すなら別の objective になる。validation/test は公開 `exp_main_F.py` で未来 `H` 点だけを切り出して MSE を計算する。**train loss と validation loss が同じ時間範囲ではない**ことも再現時の重要点。

正規化は入力 window の `mean:[B,1,C]` と `std:[B,1,C]` を使い、`X_norm=(X-mean)/std`。モデルの `L+H` 点出力にも同じ mean/std を適用して元スケールへ戻す。未来の平均・分散を先に計算して使うわけではない。公開コードでは mean を detach し、分散を `unbiased=False` で計算する。

## 論文の実験と解釈

論文は長期・短期予測で DFT に置換した FourierTS、標準 DWT に置換した WaveletTS との ablation を行う。著者は多尺度局所化と order 別係数変換の組合せが性能に寄与すると解釈する。ただし DFT/DWT 置換は WDT だけでなく表現・枝構成にも差を生むため、観測された性能差を純粋に「微分」という一要素だけの因果効果とみなすべきではない。論文は全体として平均 5.1% の MSE 改善を報告するが、比較対象・データセット・予測長に依存する集計であり、汎用同長ブロックの期待改善率ではない。

効率表は ETTm1、lookback 360、horizon 96、batch 64 で WaveTS の peak memory 380 MB、学習 2.61 s/epoch、推論 0.015 s/batch を報告する。これは論文の実験環境とフル予測モデルに対する数値。PyTorch-only Haar ブロックへそのまま移植できる速度保証ではない。論文 Appendix の明示的限界は、評価が等間隔サンプリングの benchmark に限られ、不規則間隔や event-driven 系列での頑健性が未検証という点である。

## 今回の汎用ブロックを正しく使う

原著 FRU の `Linear(Lband,Lband_out)` は L/H 固定だから、任意 T の単一 checkpoint と両立しない。本 [ts_block.py](ts_block.py) は `seq_len=L` を構築時に固定し、`Lband_out=Lband` として公式同様の帯域別 Linear を使う。DWT→order 別 rescale→FRU→iDWT→時間連結・射影の順序を保つが、**公式の未来 H 点まで拡張する head は含まない**。同長出力から将来を予測する head は fine-tune 例で別に置く。

T が奇数のときは各 Haar 分解 level の右端へ一つ内側の値を `reflect` で補い、iDWT のたびに元の level 長へ trim する。これにより T が 2^K で割り切れなくても `[B,T,C]` を返す。長さ 16、17、31、32 で公式 `pytorch_wavelets` の Haar/db1 係数と逆変換結果を直接比較した。大量欠損、非等間隔系列、極端に短い T での wavelet 解釈は別問題である。

## 公式実装との対応

| 要素 | 公開コード | 今回のコード |
| --- | --- | --- |
| wavelet decomposition | `models/WaveTS.py::WaveletDerivativeTransform`、`pytorch_wavelets` | `_haar_forward`、純 PyTorch |
| 係数 rescale / inverse | `WaveletDerivativeTransform/InverseWaveletDerivativeTransform` | order ごとの factor と `_haar_inverse` |
| FRU | `FRU`、固定係数長 Linear | `FRU`、`Linear(L_j,L_j)` に同長化 |
| branch fusion | 時間連結 + `projection_time` | 同じ時間連結 + `Linear(O·L,L)` |
| normalization と forecast | `Model.forward` | instance norm/denorm は実装。未来予測 head は別 |
| 全区間 MSE | `exp/exp_main_F.py::train(ft=False)` | 二 head の出力を連結して全区間 MSE。validation は未来のみ |

公開 checkpoint とパラメータ互換ではない。`H=0` の同長化で FRU/time projection の行列幅が変わり、wavelet package の basis・boundary も限定したため。完全再現には公式の `L→L+H`、同じ wavelet package/basis/mode、原著 head と全区間 objective が必要。

[verify_official.py](verify_official.py) は公式 `FRU` の出力・入力勾配、Haar/db1 の DWT/iDWT 係数、そして実際の `pytorch_wavelets` を使った公式 `Model.forward` の同長経路を比較し、いずれも一致を確認する。非 Haar basis は本 block の対象外である。

## 一次資料

- [論文: Multi-Order Wavelet Derivative Transform for Deep Time Series Forecasting](https://arxiv.org/abs/2505.11781)（本調査では提供 TeX を読解）
- [公式 WaveTS 実装](https://github.com/zhouziyu02/WaveTS)

## 動作確認と DataFrame 学習の入口

このディレクトリで `python ts_block.py` を実行すると、実際の block に乱数 tensor を渡して出力 shape と backward を検証する。`python ../validate_blocks.py` は5モデル横断で、異なる T と数値特性を追加検証する。追加ライブラリを入れない PyTorch 実行では Mamba 系の fallback を使い、速度比較ではない。

CSV を用意する場合は `series_id,timestamp` と特徴列・target 列が必要。例えば次のコマンドは過去64点から次8点の実測 `target` を予測する。

```bash
python finetuning_example.py data.csv --features feature_a feature_b --targets target --lookback 64 --horizon 8 --epochs 10
```

既に DataFrame `df` があるなら `from finetuning_example import fit; model, info = fit(df, ["feature_a", "feature_b"], ["target"])`。返す `info` には train 行だけから求めた feature mean/std と最良の validation MSE が入る。推論時は同じ列順・同じ正規化統計で過去窓を作る。本文で述べた原著の benchmark 成績を再現するには、このサンプル以外に原著の入力長、head、split、正規化と optimizer 設定を揃える必要がある。

## 汎用的な pooling → 回帰・分類 head の例

`WaveTSBlock` を時系列 encoder として使い、時間方向の平均 pooling と線形 head を後付けする例。pooling や head は別のものに差し替えられる。

```python
import torch
from torch import nn
from ts_block import WaveTSBlock


class SequenceModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int,
                 num_outputs: int, seq_len: int):
        super().__init__()
        self.encoder = WaveTSBlock(
            in_dim=in_dim,
            out_dim=hidden_dim,
            seq_len=seq_len,
            levels=2,
            orders=(1, 2),
        )
        self.head = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)            # [B,T,in_dim] → [B,T,hidden_dim]
        pooled = h.mean(dim=1)         # [B,T,hidden_dim] → [B,hidden_dim]
        return self.head(pooled)       # [B,num_outputs]
```

例えば `num_outputs=1` なら系列単位の回帰値、`num_outputs=K` なら K クラスの logits を返せる。回帰の MSE、分類の cross-entropy など、**GT と loss はタスク側で選ぶ**。各時刻を予測するタスクなら pooling を使わず `h:[B,T,hidden_dim]` に時刻別 head を適用すればよい。入力特徴はそれぞれ独立に Haar/FRU に通り、変数間混合は最後の学習可能な `Linear(Cin,Cout)` で行う。この射影は `Cin=Cout` でも省略しない。高次元入力では実測メモリ・速度を確認する。

[finetuning_example.py](finetuning_example.py) の backcast/forecast と MSE は、原著の予測学習を理解するための**一つの downstream 例**にすぎない。`WaveTSBlock` に二つの head やその objective が内蔵されているわけではない。原著の `L→L+H` FRU と全区間 MSE は上の「モデルアーキテクチャ全体像」「原著の入出力と objective」に残したが、汎用 encoder の使用時には不要である。
